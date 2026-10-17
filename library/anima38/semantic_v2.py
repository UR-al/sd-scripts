# Semantic Connector v2 (Anima-3.8B v1.1 "quality anchored semantic connector"), trainer port.
#
# Port of forge_sam3_extension/sam3ext/anima38/semantic_v2.py and the parameter part of adapter.py
# (forge-anima-3.8B, MIT, (c) 2026 GumGum10 contributors; see LICENSE-forge-anima-3.8B.txt).
#
# Math and state-dict layout are the reference's. Differences:
# - Forge's ``backend.nn.anima.Attention`` is replaced by the trainer's
#   ``library.anima_models.LLMAdapterAttention``. Parameter names (q_proj, q_norm, k_proj, k_norm, v_proj,
#   o_proj) and math are the same, but masks are 2-D [B, L] bool (True = token) instead of Forge's
#   4-D [B, 1, 1, L]. 4-D masks given to this port are squeezed back to 2-D.
# - The native adapter must be the trainer's ``LLMAdapter`` (its blocks take 2-D masks; Forge's take
#   4-D and would mis-broadcast a 2-D mask). It is referenced, not registered, so the connector's
#   state_dict is exactly the bundle's 190 connector tensors (after runtime.py's parameter_bank renames).
# - ``forward_context`` adds what Forge's runtime does around the connector call
#   (``Anima3BRuntime._expand_v2_context``): T5 ids cut to 512, prompt weights, zero padding to 512.
# - ProgressiveCrossAdapter keeps its parameters and ``_mixed_source``; its v1 ``forward`` (separate
#   expanded_adapter file, Forge "v1 mode") is not ported.
#
# The connector is frozen and meant to run under ``torch.no_grad()``.

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from library.anima_models import LLMAdapterAttention, LLMAdapterTransformerBlock

V2_ARCHITECTURE = "anima_qwen35_quality_anchored_semantic_connector_v2"
DEFAULT_LAYER_INDICES = (7, 15, 23, 31)
CONTEXT_LENGTH = 512


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half, 1))
    angles = timesteps.float().reshape(-1, 1) * 1_000.0 * frequencies.reshape(1, -1)
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return F.pad(embedding, (0, dim % 2)) if dim % 2 else embedding


def _mask_2d(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Bool [B, L] mask (True = token). Accepts [B, L] or Forge-style [B, 1, 1, L], any dtype."""
    if mask is None:
        return None
    mask = mask.to(torch.bool)
    if mask.ndim == 4:
        mask = mask[:, 0, 0, :]
    if mask.ndim != 2:
        raise ValueError(f"attention masks must be [B, L] (or [B, 1, 1, L]), got {tuple(mask.shape)}")
    return mask


def _as_layer_list(semantic_hidden_states) -> List[torch.Tensor]:
    """A sequence of [B, L, D] tensors, or one stacked [B, n_layers, L, D] tensor."""
    if isinstance(semantic_hidden_states, torch.Tensor):
        if semantic_hidden_states.ndim != 4:
            raise ValueError("stacked semantic states must be [B, n_layers, L, D]")
        return list(semantic_hidden_states.unbind(1))
    return list(semantic_hidden_states)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if query_dim % num_heads:
            raise ValueError("query_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.query_dim = query_dim
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.o_proj = nn.Linear(query_dim, query_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, query_tokens, _ = query.shape
        context_tokens = context.shape[1]

        def heads(value: torch.Tensor, tokens: int) -> torch.Tensor:
            return value.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

        q = heads(self.q_proj(query), query_tokens)
        k = heads(self.k_proj(context), context_tokens)
        v = heads(self.v_proj(context), context_tokens)
        mask = None
        if context_mask is not None:
            mask = context_mask.to(torch.bool).reshape(batch, 1, 1, context_tokens)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attended = attended.transpose(1, 2).reshape(batch, query_tokens, self.query_dim)
        return self.o_proj(attended)


class TimestepModulatedNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(
        self,
        value: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return self.norm(value) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SemanticResamplerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        qwen_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.cross_norm = TimestepModulatedNorm(dim)
        self.self_norm = TimestepModulatedNorm(dim)
        self.mlp_norm = TimestepModulatedNorm(dim)
        self.source_norm = nn.LayerNorm(qwen_dim)
        self.cross_attention = MultiHeadAttention(dim, qwen_dim, num_heads)
        self.self_attention = MultiHeadAttention(dim, dim, num_heads)
        self.mlp_in = nn.Linear(dim, 2 * mlp_hidden_dim, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden_dim, dim, bias=False)
        self.time_modulation = nn.Linear(dim, 6 * dim, bias=True)

    def forward(
        self,
        queries: torch.Tensor,
        qwen_features: torch.Tensor,
        timestep_embedding: torch.Tensor,
        qwen_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        modulation = self.time_modulation(F.silu(timestep_embedding))
        cross_scale, cross_shift, self_scale, self_shift, mlp_scale, mlp_shift = modulation.chunk(6, dim=-1)
        queries = queries + self.cross_attention(
            self.cross_norm(queries, cross_scale, cross_shift),
            self.source_norm(qwen_features),
            qwen_mask,
        )
        normalized = self.self_norm(queries, self_scale, self_shift)
        queries = queries + self.self_attention(normalized, normalized)
        normalized = self.mlp_norm(queries, mlp_scale, mlp_shift)
        gate, value = self.mlp_in(normalized).chunk(2, dim=-1)
        return queries + self.mlp_out(F.silu(gate) * value)


class _ResamplerParameterBank(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_queries: int,
        model_dim: int,
        qwen_dim: int,
    ) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.empty(1, num_queries, model_dim))
        self.layer_embeddings = nn.Parameter(torch.empty(num_layers, 1, qwen_dim))


class TimestepAwareSemanticResampler(nn.Module):
    def __init__(
        self,
        qwen_dim: int,
        output_dim: int,
        num_layers: int,
        num_queries: int,
        num_blocks: int,
        model_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.parameter_bank = _ResamplerParameterBank(num_layers, num_queries, model_dim, qwen_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.blocks = nn.ModuleList(SemanticResamplerBlock(model_dim, qwen_dim, num_heads, mlp_hidden_dim) for _ in range(num_blocks))
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, output_dim, bias=False)

    @property
    def query_tokens(self) -> torch.Tensor:
        return self.parameter_bank.query_tokens

    @property
    def layer_embeddings(self) -> torch.Tensor:
        return self.parameter_bank.layer_embeddings

    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        timesteps: torch.Tensor,
        source_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if len(hidden_states) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} Qwen layers, got {len(hidden_states)}")
        batch = hidden_states[0].shape[0]
        layer_streams = [
            hidden + self.layer_embeddings[index].to(device=hidden.device, dtype=hidden.dtype).unsqueeze(0)
            for index, hidden in enumerate(hidden_states)
        ]
        qwen_features = torch.cat(layer_streams, dim=1)
        qwen_mask = None
        if source_mask is not None:
            qwen_mask = torch.cat([source_mask] * self.num_layers, dim=1)
        time = sinusoidal_timestep_embedding(timesteps, self.model_dim).to(dtype=qwen_features.dtype)
        time = self.time_mlp(time)
        queries = self.query_tokens.to(device=qwen_features.device, dtype=qwen_features.dtype).expand(batch, -1, -1)
        for block in self.blocks:
            queries = block(queries, qwen_features, time, qwen_mask)
        return self.output_projection(self.output_norm(queries))


class _AdapterParameterBank(nn.Module):
    def __init__(self, blocks: int, layers: int) -> None:
        super().__init__()
        self.layer_mix_logits = nn.Parameter(torch.zeros(blocks, layers))


class ProgressiveCrossAdapter(nn.Module):
    """Parameters of the trained progressive cross adapter ("quality anchor") used by the v2 connector."""

    def __init__(
        self,
        native_adapter: nn.Module,
        semantic_source_dim: int = 2560,
        layer_indices: Sequence[int] = DEFAULT_LAYER_INDICES,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "native_adapter", native_adapter)
        self.layer_indices = tuple(layer_indices)
        model_dim = native_adapter.embed.weight.shape[1]
        num_heads = native_adapter.blocks[0].self_attn.n_heads
        head_dim = model_dim // num_heads

        self.query_norms = nn.ModuleList(nn.RMSNorm(model_dim, eps=1e-6) for _ in native_adapter.blocks)
        self.source_norms = nn.ModuleList(nn.RMSNorm(semantic_source_dim, eps=1e-6) for _ in native_adapter.blocks)
        self.semantic_attentions = nn.ModuleList(
            LLMAdapterAttention(model_dim, semantic_source_dim, num_heads, head_dim) for _ in native_adapter.blocks
        )
        self.parameter_bank = _AdapterParameterBank(len(native_adapter.blocks), len(self.layer_indices))

    @property
    def layer_mix_logits(self) -> torch.Tensor:
        return self.parameter_bank.layer_mix_logits

    @staticmethod
    def _mixed_source(
        semantic_states: Sequence[torch.Tensor],
        mix: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor:
        return sum(state * mix[block_index, state_index] for state_index, state in enumerate(semantic_states))


def _check_native_adapter(native_adapter: nn.Module) -> None:
    blocks = getattr(native_adapter, "blocks", None)
    if not blocks or not all(isinstance(block, LLMAdapterTransformerBlock) for block in blocks):
        raise TypeError(
            "Semantic Connector v2 port needs the trainer's library.anima_models.LLMAdapter as native_adapter "
            "(its blocks take 2-D [B, L] masks)"
        )


class QualityAnchoredSemanticConnectorV2(nn.Module):
    architecture = V2_ARCHITECTURE

    def __init__(
        self,
        native_adapter: nn.Module,
        semantic_source_dim: int = 2560,
        layer_indices: Sequence[int] = DEFAULT_LAYER_INDICES,
        num_queries: int = 64,
        resampler_blocks: int = 6,
        resampler_dim: int = 2048,
        resampler_heads: int = 16,
        mlp_hidden_dim: int = 5632,
    ) -> None:
        super().__init__()
        _check_native_adapter(native_adapter)
        self.layer_indices = tuple(int(index) for index in layer_indices)
        self.quality_anchor = ProgressiveCrossAdapter(native_adapter, semantic_source_dim=semantic_source_dim, layer_indices=self.layer_indices)
        model_dim = native_adapter.embed.weight.shape[1]
        num_heads = native_adapter.blocks[0].self_attn.n_heads
        head_dim = model_dim // num_heads
        self.semantic_resampler = TimestepAwareSemanticResampler(
            qwen_dim=semantic_source_dim,
            output_dim=model_dim,
            num_layers=len(self.layer_indices),
            num_queries=num_queries,
            num_blocks=resampler_blocks,
            model_dim=resampler_dim,
            num_heads=resampler_heads,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self.v2_query_norms = nn.ModuleList(nn.RMSNorm(model_dim, eps=1e-6) for _ in native_adapter.blocks)
        self.v2_semantic_norms = nn.ModuleList(nn.RMSNorm(model_dim, eps=1e-6) for _ in native_adapter.blocks)
        self.v2_attentions = nn.ModuleList(LLMAdapterAttention(model_dim, model_dim, num_heads, head_dim) for _ in native_adapter.blocks)

    @property
    def native_adapter(self) -> nn.Module:
        return self.quality_anchor.native_adapter

    @staticmethod
    def _mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return _mask_2d(mask)

    def forward(
        self,
        native_source: torch.Tensor,
        target_input_ids: torch.Tensor,
        semantic_hidden_states: Union[Sequence[torch.Tensor], torch.Tensor],
        target_attention_mask: Optional[torch.Tensor] = None,
        native_source_mask: Optional[torch.Tensor] = None,
        semantic_source_mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        include_inserted_blocks: bool = True,
        include_v2: bool = True,
        **_ignored,
    ) -> torch.Tensor:
        """Same inputs/outputs as the reference: native Qwen3-0.6B source [B, Ls, 1024], T5 ids [B, Lt],
        Qwen3.5 layer states (4 x [B, Lq, 2560] or [B, 4, Lq, 2560]), optional masks, diffusion
        timesteps (sigma in [0, 1], [B] or [B, 1]). Returns the raw context [B, Lt, 1024] (no weights,
        no padding); see ``forward_context`` for the llm_adapter replacement."""
        if timesteps is None:
            raise ValueError("Semantic Connector v2 requires diffusion timesteps")
        semantic_hidden_states = _as_layer_list(semantic_hidden_states)
        if len(semantic_hidden_states) != len(self.layer_indices):
            raise ValueError(f"Expected {len(self.layer_indices)} semantic layers, got {len(semantic_hidden_states)}")

        target_attention_mask = self._mask(target_attention_mask)
        native_source_mask = self._mask(native_source_mask)
        semantic_attention_mask = self._mask(semantic_source_mask)
        semantic_bank = None
        if include_inserted_blocks and include_v2:
            semantic_bank = self.semantic_resampler(semantic_hidden_states, timesteps, semantic_attention_mask)

        native = self.native_adapter
        x = native.in_proj(native.embed(target_input_ids))
        query_positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        native_positions = torch.arange(native_source.shape[1], device=x.device).unsqueeze(0)
        anchor_positions = torch.arange(semantic_hidden_states[0].shape[1], device=x.device).unsqueeze(0)
        bank_rope = None
        if semantic_bank is not None:
            bank_positions = torch.arange(semantic_bank.shape[1], device=x.device).unsqueeze(0)
            bank_rope = native.rotary_emb(x, bank_positions)
        query_rope = native.rotary_emb(x, query_positions)
        native_rope = native.rotary_emb(x, native_positions)
        anchor_rope = native.rotary_emb(x, anchor_positions)
        anchor_mix = self.quality_anchor.layer_mix_logits.float().softmax(dim=-1).to(device=x.device, dtype=x.dtype)

        for index, native_block in enumerate(native.blocks):
            x = native_block(
                x,
                native_source,
                target_attention_mask=target_attention_mask,
                source_attention_mask=native_source_mask,
                position_embeddings=query_rope,
                position_embeddings_context=native_rope,
            )
            if not include_inserted_blocks:
                continue
            anchor_source = self.quality_anchor.source_norms[index](
                self.quality_anchor._mixed_source(semantic_hidden_states, anchor_mix, index)
            )
            x = x + self.quality_anchor.semantic_attentions[index](
                self.quality_anchor.query_norms[index](x),
                kv_mask=semantic_attention_mask,
                context=anchor_source,
                position_embeddings=query_rope,
                position_embeddings_context=anchor_rope,
            )
            if semantic_bank is not None:
                x = x + self.v2_attentions[index](
                    self.v2_query_norms[index](x),
                    context=self.v2_semantic_norms[index](semantic_bank),
                    position_embeddings=query_rope,
                    position_embeddings_context=bank_rope,
                )
        return native.norm(native.out_proj(x))

    def forward_context(
        self,
        native_source: torch.Tensor,
        target_input_ids: torch.Tensor,
        semantic_hidden_states: Union[Sequence[torch.Tensor], torch.Tensor],
        timesteps: torch.Tensor,
        target_attention_mask: Optional[torch.Tensor] = None,
        native_source_mask: Optional[torch.Tensor] = None,
        semantic_source_mask: Optional[torch.Tensor] = None,
        target_weights: Optional[torch.Tensor] = None,
        context_length: int = CONTEXT_LENGTH,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Cross-attention context that replaces ``llm_adapter(...)`` in the DiT forward.

        Mirrors ``Anima3BRuntime._encode_v2`` / ``_expand_v2_context``:
        - T5 ids (and their mask / prompt weights) are cut to ``context_length`` (Forge ``[:, :512]``);
        - native source and semantic states are cast to ``dtype`` (default: the native adapter's
          embedding dtype, as Forge does);
        - the connector output is multiplied by ``target_weights`` ([B, Lt] or [B, Lt, 1]; None = 1);
        - rows where ``target_attention_mask`` is False are set to 0 (the trainer's ``context[~mask] = 0``);
        - the result is zero-padded to ``context_length`` rows (Forge ``F.pad(..., 512)``).
        Returns [B, context_length, 1024].
        """
        if dtype is None:
            dtype = self.native_adapter.embed.weight.dtype
        target_input_ids = target_input_ids[:, :context_length]
        target_mask = _mask_2d(target_attention_mask)
        if target_mask is not None:
            target_mask = target_mask[:, :context_length]
        semantic_hidden_states = [state.to(dtype=dtype) for state in _as_layer_list(semantic_hidden_states)]
        expanded = self.forward(
            native_source.to(dtype=dtype),
            target_input_ids,
            semantic_hidden_states,
            target_attention_mask=target_mask,
            native_source_mask=native_source_mask,
            semantic_source_mask=semantic_source_mask,
            timesteps=timesteps,
        )
        if target_weights is not None:
            weights = target_weights[:, :context_length].to(device=expanded.device, dtype=expanded.dtype)
            if weights.ndim == 2:
                weights = weights.unsqueeze(-1)
            expanded = expanded * weights[:, : expanded.shape[1]]
        if target_mask is not None:
            expanded = expanded.masked_fill(~target_mask[:, : expanded.shape[1], None], 0)
        if expanded.shape[1] < context_length:
            expanded = F.pad(expanded, (0, 0, 0, context_length - expanded.shape[1]))
        return expanded
