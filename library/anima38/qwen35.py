# Qwen3.5-4B hybrid text backbone (Gated DeltaNet + gated attention) used by Anima-3.8B v1.1.
#
# Port of forge_sam3_extension/sam3ext/anima38/{qwen35.py,layers.py} (forge-anima-3.8B, MIT,
# (c) 2026 GumGum10 contributors; see LICENSE-forge-anima-3.8B.txt in this directory).
#
# The math is kept line for line. The only structural change: Forge builds the model with
# ``operations=ForgeOperations`` + ``manual_cast_enabled=True``; here the ``Cast*`` layers below do the
# same thing without Forge. They keep the checkpoint's storage dtype (qwen35_4b.safetensors stores every
# Linear weight as float8_e4m3fn and everything else as bfloat16) and cast each weight to the input's
# dtype/device at use. The embedding keeps its dtype (Forge: ``skip_weight_dtype=True``), so the hidden
# states run in bfloat16 like Forge unless ``compute_dtype`` asks for something else.

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Qwen3.5 layers whose hidden states feed Semantic Connector v2 (Forge runtime.py LAYER_INDICES).
LAYER_INDICES = (7, 15, 23, 31)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


# region manual-cast layers (Forge weights_manual_cast equivalent)


class CastLinear(nn.Linear):
    """nn.Linear that keeps its stored dtype (e.g. float8_e4m3fn) and casts weight/bias to the input's
    dtype and device at every call."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)


class CastConv1d(nn.Conv1d):
    """nn.Conv1d with the same manual cast as CastLinear."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return self._conv_forward(x, weight, bias)


class CastEmbedding(nn.Embedding):
    """Embedding lookup that keeps the table's dtype (Forge ``skip_weight_dtype=True``). If the table lives
    on another device than the ids, the lookup runs where the table is and only the rows are moved."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        out = F.embedding(
            x.to(weight.device), weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
        )
        return out.to(x.device)


# endregion

# region layers (sam3ext/anima38/layers.py)


class RMSNorm(nn.Module):
    """Direct-scale RMSNorm used by Qwen3.5's gated delta block."""

    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        y = y * self.weight.to(device=y.device, dtype=torch.float32)
        return y.to(dtype=x.dtype)


class Qwen35RMSNorm(nn.Module):
    """Qwen3.5 RMSNorm whose checkpoint weights are deltas around one."""

    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        y = y * (1.0 + self.weight.to(device=y.device, dtype=torch.float32))
        return y.to(dtype=x.dtype)


def _delta_rule_loop(query, key, value, g_exp, beta):
    """Reference Gated DeltaNet recurrence, token by token in fp32 -> [B, H, L, V] (not yet transposed back).

    Inputs are [B, H, L, *] contiguous fp32; ``query`` is already scaled and ``g_exp`` is ``g.exp()``.
    """
    batch, heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    state = torch.zeros(batch, heads, k_dim, v_dim, device=query.device, dtype=torch.float32)
    outputs = []

    for i in range(seq_len):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g_exp[:, :, i].unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))

    return torch.stack(outputs, dim=2)


class _ScanGraphCache:
    """CUDA-graph replay of ``_delta_rule_loop`` for inference (TE caching, sampling prompts).

    The loop launches about ten tiny kernels per token, so a few hundred tokens are bound by launch overhead,
    not math. A graph captured for one input shape replays the very same kernels in the same order with a
    single launch, so the output is bit-identical to the eager loop. All Gated DeltaNet layers share one
    graph per shape (same head/dim layout), and the caption variants of an image share their token length.
    The first call for a shape runs eagerly (it also loads the kernels, which must not happen during
    capture); the next call captures. A capture error switches the cache off for this process and falls
    back to the eager loop. ``ANIMA38_QWEN35_CUDA_GRAPH=0`` disables it.
    """

    def __init__(self, max_graphs: int = 8):
        # strip(): cmd's `set VAR=0 && ...` leaves a trailing space in the value
        self.enabled = os.environ.get("ANIMA38_QWEN35_CUDA_GRAPH", "1").strip().lower() not in ("0", "off", "false", "no")
        self.capture_streams: Dict[torch.device, torch.cuda.Stream] = {}  # one side stream per device
        self.max_graphs = max_graphs
        self.graphs: "OrderedDict[tuple, tuple]" = OrderedDict()  # key -> (graph, static inputs, static output)
        self.seen: set = set()

    def usable(self, tensor: torch.Tensor) -> bool:
        return (
            self.enabled
            and tensor.is_cuda
            and not torch.is_grad_enabled()
            and not torch.cuda.is_current_stream_capturing()
        )

    def clear(self) -> None:
        """Drop every captured graph (and the memory pools they hold), e.g. when Qwen3.5 is released."""
        self.graphs.clear()
        self.seen.clear()

    def _disable(self, error: Exception) -> None:
        self.enabled = False
        self.clear()
        logger.warning(f"Qwen3.5 DeltaNet CUDA graph disabled, using the eager loop: {error!r}")

    def run(self, *inputs: torch.Tensor) -> torch.Tensor:
        key = (tuple(tuple(t.shape) for t in inputs), inputs[0].device)
        with torch.cuda.device(inputs[0].device):
            entry = self.graphs.get(key)
            if entry is None:
                if key not in self.seen:
                    self.seen.add(key)
                    return _delta_rule_loop(*inputs)
                try:
                    entry = self._capture(inputs)
                except Exception as error:  # never let the speedup break caching
                    self._disable(error)
                    return _delta_rule_loop(*inputs)
                self.graphs[key] = entry
                while len(self.graphs) > self.max_graphs:
                    self.graphs.popitem(last=False)
            else:
                self.graphs.move_to_end(key)
            graph, static_inputs, static_output = entry
            try:
                for static, value in zip(static_inputs, inputs):
                    static.copy_(value)
                graph.replay()
                return static_output.clone()
            except Exception as error:
                self._disable(error)
                return _delta_rule_loop(*inputs)

    def _capture(self, inputs):
        # Always capture outside inference mode: the static buffers (and the CUDA generator state that
        # capture_begin registers on the first capture) must be normal tensors, or a later capture/copy outside
        # inference mode fails with "Inplace update to inference tensor". Normal tensors work in both modes.
        # The capture stream is per device: torch.cuda.graph's shared default stream is bound to the first device.
        device = inputs[0].device
        stream = self.capture_streams.get(device)
        if stream is None:
            stream = self.capture_streams[device] = torch.cuda.Stream(device=device)
        with torch.inference_mode(False), torch.no_grad():
            static_inputs = [t.clone() for t in inputs]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                static_output = _delta_rule_loop(*static_inputs)
        return graph, static_inputs, static_output


_SCAN_GRAPHS = _ScanGraphCache()


def clear_scan_graphs() -> None:
    """Free the DeltaNet CUDA graphs (call when the Qwen3.5 model is released)."""
    _SCAN_GRAPHS.clear()


class SSMBlock(nn.Module):
    """Qwen3.5 GatedDeltaNet linear-attention block (reference delta-rule scan, token by token)."""

    def __init__(self, hidden_size=2560, d_inner=8192, n_groups=32, d_gate=4096, conv_kernel=4, norm_dim=128, device=None, dtype=None):
        super().__init__()
        self.num_v_heads = n_groups
        self.num_k_heads = 16
        self.head_k_dim = norm_dim
        self.head_v_dim = norm_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = CastLinear(hidden_size, self.conv_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_z = CastLinear(hidden_size, self.value_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_a = CastLinear(hidden_size, self.num_v_heads, bias=False, device=device, dtype=dtype)
        self.in_proj_b = CastLinear(hidden_size, self.num_v_heads, bias=False, device=device, dtype=dtype)
        self.conv1d = CastConv1d(
            self.conv_dim, self.conv_dim, conv_kernel, groups=self.conv_dim, padding=conv_kernel - 1, bias=False, device=device, dtype=dtype
        )
        self.out_proj = CastLinear(self.value_dim, hidden_size, bias=False, device=device, dtype=dtype)
        self.norm = RMSNorm(self.head_v_dim, device=device, dtype=dtype)
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, device=device, dtype=dtype))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads, device=device, dtype=dtype))

    def _delta_rule_scan(self, query, key, value, g, beta):
        initial_dtype = query.dtype
        query = l2norm(query, dim=-1).transpose(1, 2).contiguous().float()
        key = l2norm(key, dim=-1).transpose(1, 2).contiguous().float()
        value = value.transpose(1, 2).contiguous().float()
        beta = beta.transpose(1, 2).contiguous().float()
        g = g.transpose(1, 2).contiguous().float()

        k_dim = key.shape[-1]
        query = query * (1.0 / (k_dim**0.5))
        g_exp = g.exp()  # once for the whole sequence: same per-element exp as the per-step slice, one launch instead of seq_len

        if _SCAN_GRAPHS.usable(query):
            out = _SCAN_GRAPHS.run(query, key, value, g_exp, beta)
        else:
            out = _delta_rule_loop(query, key, value, g_exp, beta)
        return out.transpose(1, 2).contiguous().to(initial_dtype)

    def forward(self, hidden_states, attention_mask=None):
        # SSM input masking: only for padded batches (batch > 1), exactly like the reference
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            hidden_states = hidden_states * attention_mask[:, :, None].to(dtype=hidden_states.dtype)

        batch, seq_len, _ = hidden_states.shape
        z = self.in_proj_z(hidden_states).reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len]).transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

        beta = self.in_proj_b(hidden_states).sigmoid()
        g = -self.A_log.float().exp().to(hidden_states.device) * F.softplus(
            self.in_proj_a(hidden_states).float() + self.dt_bias.to(hidden_states.device).float()
        )
        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        y = self._delta_rule_scan(query, key, value, g, beta)
        y = self.norm(y.reshape(-1, self.head_v_dim))
        y = y * F.silu(z.reshape(-1, self.head_v_dim).float()).to(y.dtype)
        return self.out_proj(y.reshape(batch, seq_len, self.value_dim))


class GatedSelfAttention(nn.Module):
    """Self-attention with gated Q projection (16 heads x 256, 4 KV heads, Q/gate interleaved per head)."""

    def __init__(self, hidden_size=2560, num_heads=16, num_kv_heads=4, head_dim=256, rope_theta=10000000.0, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gqa_ratio = num_heads // num_kv_heads
        self.inner_dim = num_heads * head_dim

        self.q_proj = CastLinear(hidden_size, 2 * self.inner_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = CastLinear(hidden_size, num_kv_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = CastLinear(hidden_size, num_kv_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = CastLinear(self.inner_dim, hidden_size, bias=False, device=device, dtype=dtype)

        self.q_norm = Qwen35RMSNorm(head_dim, device=device, dtype=dtype)
        self.k_norm = Qwen35RMSNorm(head_dim, device=device, dtype=dtype)

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None):
        B, L, _ = hidden_states.shape

        qg = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim * 2)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(B, L, self.inner_dim)

        q = q.view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if freqs_cis is not None:
            cos, sin = freqs_cis
            q = _apply_rotary_emb(q, cos, sin)
            k = _apply_rotary_emb(k, cos, sin)

        k = k.repeat_interleave(self.gqa_ratio, dim=1)
        v = v.repeat_interleave(self.gqa_ratio, dim=1)

        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.to(dtype=q.dtype)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=(attention_mask is None))

        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.inner_dim)
        attn_out = attn_out * torch.sigmoid(gate)

        return self.o_proj(attn_out)


def _apply_rotary_emb(x, cos, sin):
    """Apply rotary position embeddings to the first ``cos.shape[-1]`` channels."""
    rotary_dim = cos.shape[-1]
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x1 = x_rot[..., : rotary_dim // 2]
    x2 = x_rot[..., rotary_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    x_embed = (x_rot * cos) + (rotated * sin)
    return torch.cat((x_embed, x_pass), dim=-1)


def _precompute_freqs_cis(head_dim, max_seq_len, theta=10000000.0, device=None, dtype=None):
    """Precompute RoPE frequencies."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    cos = cos.repeat(1, 1, 1, 2)
    sin = sin.repeat(1, 1, 1, 2)
    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)
    return cos, sin


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, hidden_size=2560, intermediate_size=9216, device=None, dtype=None):
        super().__init__()
        self.gate_proj = CastLinear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = CastLinear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = CastLinear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HybridBlock(nn.Module):
    """One Qwen3.5 layer: Gated DeltaNet (SSM) or gated self-attention, then an optional SwiGLU MLP."""

    def __init__(self, hidden_size=2560, intermediate_size=9216, use_ssm=True, has_mlp=True, device=None, dtype=None):
        super().__init__()
        self.use_ssm = use_ssm
        self.has_mlp = has_mlp

        self.input_layernorm = Qwen35RMSNorm(hidden_size, device=device, dtype=dtype)

        if use_ssm:
            self.linear_attn = SSMBlock(hidden_size=hidden_size, device=device, dtype=dtype)
        else:
            self.self_attn = GatedSelfAttention(hidden_size=hidden_size, device=device, dtype=dtype)

        if has_mlp:
            self.post_attention_layernorm = Qwen35RMSNorm(hidden_size, device=device, dtype=dtype)
            self.mlp = MLP(hidden_size=hidden_size, intermediate_size=intermediate_size, device=device, dtype=dtype)

    def forward(self, x, attention_mask=None, freqs_cis=None, linear_attention_mask=None):
        residual = x
        x_norm = self.input_layernorm(x)

        if self.use_ssm:
            x = residual + self.linear_attn(x_norm, attention_mask=linear_attention_mask)
        else:
            x = residual + self.self_attn(x_norm, attention_mask=attention_mask, freqs_cis=freqs_cis)

        if self.has_mlp:
            residual = x
            x = residual + self.mlp(self.post_attention_layernorm(x))

        return x


# endregion

# region model (sam3ext/anima38/qwen35.py)


class Qwen35HybridModel(nn.Module):
    """Qwen3.5 4B hybrid text backbone used by the Anima adapter (same state-dict keys as the checkpoint)."""

    SELF_ATTN_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}
    NUM_LAYERS = 32
    HIDDEN_SIZE = 2560
    INTERMEDIATE_SIZE = 9216
    VOCAB_SIZE = 248320
    OUTPUT_DIM = 1024
    ROTARY_DIM = 64
    ROPE_THETA = 10000000.0

    def __init__(self, config_dict=None, dtype=None, device=None, compute_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.num_layers = self.NUM_LAYERS
        self.dtype = dtype
        # None = run in the embedding table's dtype (bfloat16 for qwen35_4b.safetensors), as Forge does
        self.compute_dtype = compute_dtype
        self.embed_tokens = CastEmbedding(self.VOCAB_SIZE, self.HIDDEN_SIZE, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                HybridBlock(
                    hidden_size=self.HIDDEN_SIZE,
                    intermediate_size=self.INTERMEDIATE_SIZE,
                    use_ssm=index not in self.SELF_ATTN_LAYERS,
                    has_mlp=index != 31,
                    device=device,
                    dtype=dtype,
                )
                for index in range(self.NUM_LAYERS)
            ]
        )
        self.norm = nn.Sequential(
            CastLinear(self.HIDDEN_SIZE, self.OUTPUT_DIM, bias=True, device=device, dtype=dtype),
            Qwen35RMSNorm(self.OUTPUT_DIM, device=device, dtype=dtype),
            nn.SiLU(),
            CastLinear(self.OUTPUT_DIM, self.OUTPUT_DIM, bias=True, device=device, dtype=dtype),
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.embed_tokens = embeddings

    def _prepare(self, input_ids, attention_mask, embeds, compute_dtype):
        if embeds is not None:
            hidden_states = embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
            compute_dtype = compute_dtype if compute_dtype is not None else self.compute_dtype
            if compute_dtype is not None:
                hidden_states = hidden_states.to(compute_dtype)

        sequence_length = hidden_states.shape[1]
        rotary = _precompute_freqs_cis(
            self.ROTARY_DIM, sequence_length, theta=self.ROPE_THETA, device=hidden_states.device, dtype=hidden_states.dtype
        )
        linear_attention_mask = attention_mask
        attention_bias = None
        if attention_mask is not None:
            mask_fill = torch.finfo(hidden_states.dtype).min / 4
            causal = (
                torch.empty(sequence_length, sequence_length, dtype=hidden_states.dtype, device=hidden_states.device)
                .fill_(mask_fill)
                .triu_(1)
            )
            padding = 1.0 - attention_mask.to(hidden_states.dtype).reshape(attention_mask.shape[0], 1, -1, attention_mask.shape[-1]).expand(
                attention_mask.shape[0], 1, sequence_length, attention_mask.shape[-1]
            )
            padding = padding.masked_fill(padding.to(torch.bool), mask_fill)
            attention_bias = causal + padding
        elif sequence_length > 1:
            mask_fill = torch.finfo(hidden_states.dtype).min / 4
            attention_bias = (
                torch.empty(sequence_length, sequence_length, dtype=hidden_states.dtype, device=hidden_states.device)
                .fill_(mask_fill)
                .triu_(1)
            )
        return hidden_states, attention_bias, rotary, linear_attention_mask

    def forward(
        self,
        input_ids,
        attention_mask=None,
        embeds=None,
        num_tokens=None,
        intermediate_output=None,
        final_layer_norm_intermediate=True,
        dtype=None,
        embeds_info=None,
        compute_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Reference-compatible forward: returns ``(norm_head(last_hidden), intermediate)``.

        ``dtype`` is accepted and ignored exactly like the reference (Forge passes ``dtype=torch.float32``
        but computes in the embedding dtype). Use ``compute_dtype`` to change the compute dtype.
        """
        del num_tokens, final_layer_norm_intermediate, embeds_info, kwargs, dtype
        hidden_states, attention_bias, rotary, linear_attention_mask = self._prepare(input_ids, attention_mask, embeds, compute_dtype)

        intermediate = None
        for index, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask=attention_bias, freqs_cis=rotary, linear_attention_mask=linear_attention_mask)
            if isinstance(intermediate_output, int) and index == intermediate_output:
                intermediate = hidden_states.clone()
            elif isinstance(intermediate_output, list) and index in intermediate_output:
                if intermediate is None:
                    intermediate = {}
                intermediate[index] = hidden_states.clone()

        return self.norm(hidden_states), intermediate

    def forward_semantic_layers(
        self,
        input_ids,
        attention_mask=None,
        layer_indices: Sequence[int] = LAYER_INDICES,
        embeds=None,
        compute_dtype: Optional[torch.dtype] = None,
    ) -> List[torch.Tensor]:
        """Hidden states after the given layers (default 7/15/23/31), in that order.

        Same values as ``forward(..., intermediate_output=list(layer_indices))[1]``; stops after the last
        requested layer and skips the 2560->1024 ``norm`` head, whose output Forge discards.
        """
        wanted = [int(index) for index in layer_indices]
        if not wanted or min(wanted) < 0 or max(wanted) >= len(self.layers):
            raise ValueError(f"layer_indices out of range: {wanted}")
        hidden_states, attention_bias, rotary, linear_attention_mask = self._prepare(input_ids, attention_mask, embeds, compute_dtype)

        captured: Dict[int, torch.Tensor] = {}
        last = max(wanted)
        for index, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask=attention_bias, freqs_cis=rotary, linear_attention_mask=linear_attention_mask)
            if index in wanted:
                captured[index] = hidden_states.clone()
            if index == last:
                break
        return [captured[index] for index in wanted]


# endregion

# region loading / encoding helpers


def load_qwen35_model(
    path: str,
    device: Optional[Union[str, torch.device]] = None,
    compute_dtype: Optional[torch.dtype] = None,
) -> Qwen35HybridModel:
    """Build the port and load ``qwen35_4b.safetensors`` READ-ONLY (``safe_open``), strict.

    Storage dtypes are kept (F8_E4M3 Linear weights, BF16 embedding/norms/conv/SSM parameters), like
    Forge's ``_load_qwen``; the Cast* layers upcast one layer at a time during forward. The model is
    frozen (``requires_grad_(False)``) and in eval mode. ``device`` moves it after loading (about
    4.8 GB at storage dtype).
    """
    from safetensors import safe_open

    state = {}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            state[key] = checkpoint.get_tensor(key)

    with torch.device("meta"):
        model = Qwen35HybridModel(compute_dtype=compute_dtype)
    model.requires_grad_(False)
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Qwen3.5 checkpoint mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    del state
    model.eval()
    if device is not None:
        model.to(device)
    return model


def pad_token_batch(token_lists: Sequence[Sequence[int]], pad_token_id: int):
    """Right-pad token id lists -> (input_ids [B, L] long, attention_mask [B, L] long, 1 = token)."""
    batch = len(token_lists)
    length = max(len(tokens) for tokens in token_lists)
    input_ids = torch.full((batch, length), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((batch, length), dtype=torch.long)
    for row, tokens in enumerate(token_lists):
        input_ids[row, : len(tokens)] = torch.tensor(list(tokens), dtype=torch.long)
        attention_mask[row, : len(tokens)] = 1
    return input_ids, attention_mask


@torch.no_grad()
def encode_semantic_layers(
    model: Qwen35HybridModel,
    tokenizer,
    texts: Union[str, Sequence[str]],
    device: Optional[Union[str, torch.device]] = None,
    layer_indices: Sequence[int] = LAYER_INDICES,
    compute_dtype: Optional[torch.dtype] = None,
    zero_padding: bool = True,
):
    """Qwen3.5 semantic features for Semantic Connector v2.

    Forge's ``Anima3BRuntime._semantic_layers`` generalised to batches: raw text, no special tokens, at
    most 1024 tokens (the tokenizer does that), attention mask of ones. One text gives exactly Forge's
    call. Several texts are right-padded; the reference model then uses its padding bias in the attention
    layers and masks the SSM inputs, so valid positions match the unpadded result (causal model).

    Returns ``(states, attention_mask)``: ``states`` is a list of ``len(layer_indices)`` tensors
    [B, L, 2560] in the compute dtype; ``attention_mask`` is [B, L] long (1 = token). With
    ``zero_padding`` the padded positions are set to 0 (the connector masks them anyway).
    """
    if isinstance(texts, str):
        texts = [texts]
    token_lists = tokenizer(list(texts))["input_ids"]
    input_ids, attention_mask = pad_token_batch(token_lists, tokenizer.pad_token_id)
    if device is None:
        device = model.embed_tokens.weight.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    states = model.forward_semantic_layers(input_ids, attention_mask=attention_mask, layer_indices=layer_indices, compute_dtype=compute_dtype)
    if zero_padding and input_ids.shape[0] > 1:
        keep = attention_mask.to(torch.bool)[:, :, None]
        states = [state.masked_fill(~keep, 0) for state in states]
    return states, attention_mask


# endregion
