# MOSS-Image DiT for training.
# Adapted from OpenMOSS MOSS-Image ``pipeline_moss_image.py`` (MOSSImageTransformer), Copyright OpenMOSS Team,
# Apache License 2.0. Module names and math are unchanged so the released checkpoint loads strictly and LoRA keys
# match the checkpoint keys. Changes for training: batched forward with padded text (key-padding mask through
# library.attention instead of the official per-sample loop - same result, see tests/test_moss_models.py),
# gradient checkpointing (plain / CPU offload / unsloth offload) and block swap like library/anima_models.py.

from typing import Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from library import attention, custom_offloading_utils
from library.elementwise_compile import ElementwiseChains
from library.anima_models import to_cpu, to_device, unsloth_checkpoint
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

# Released MOSS-Image-Base / -SFT transformer/config.json
MOSS_DIT_CONFIG = dict(dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, freq_dim=256,
                       latent_channels=128, rope_base=10000.0)


class MossRMSNorm(nn.Module):
    """Direct-scale RMSNorm, eps 1e-6, fp32 internals (official ``_RMSNorm``)."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return rms_norm(x, self.weight)


def rms_norm(x, weight):
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    return y.to(x.dtype) * weight


def situ(hidden, gate):
    """SiTU-GLU activation: soft-capped (4) gate * sigmoid(gate) times soft-capped (25) hidden."""
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def rotate(x, rope):
    """Rotate-half RoPE over all 128 head dims, pairs (d, d+64). x [B,L,H,D], rope [B,L,D] = (sin | cos)."""
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


# ---- elementwise chains of MossBlock, optionally fused with torch.compile (CUDA only) ---------------------------
# Eager, each chain between two matmuls is a string of separate memory-bound kernels (fp32 RMSNorm, RoPE, SiTU-GLU,
# gated residual): ~2 s of a ~9 s step at 1536 batch 2. Compiled, each becomes one or two fused kernels. The LoRA-wrapped
# Linears and attention stay eager (library/elementwise_compile.py).
def norm_modulate(x, weight, scale, shift):
    return rms_norm(x, weight) * (1 + scale[:, None]) + shift[:, None]


def qk_norm_rope(t, weight, rope, heads: int):
    """[B, L, H*D] projection -> per-head RMSNorm -> RoPE -> [B, L, H, D]."""
    return rotate(rms_norm(t.unflatten(-1, (heads, -1)), weight), rope)


def gated_residual(x, h, gate):
    return x + h * gate[:, None]


_CHAINS = ElementwiseChains("MossBlock", norm_modulate=norm_modulate, qk_norm_rope=qk_norm_rope, situ=situ,
                            gated_residual=gated_residual)


def enable_elementwise_compile(enabled: bool = True):
    """Use torch.compile'd elementwise chains in MossBlock for CUDA tensors (training and sampling)."""
    _CHAINS.enable(enabled)


def _chains(x: torch.Tensor):
    return _CHAINS.get(x)


def joint_attention(q, k, v, rope, heads, attn_params: attention.AttentionParams):
    """Official ``_attention`` (RoPE on q/k, SDPA default scale) with padding handled by ``library.attention``."""
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = rotate(q.reshape(shape), rope)
    k = rotate(k.reshape(shape), rope)
    v = v.reshape(shape)
    return attention.attention([q.to(v.dtype), k.to(v.dtype), v], attn_params=attn_params)  # [B, L, H*D]


class _CheckpointMixin:
    """Per-block gradient checkpointing switch, same three modes as library/anima_models.py Block."""

    def _init_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False

    def enable_gradient_checkpointing(self, cpu_offload: bool = False, unsloth_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload if not unsloth_offload else False
        self.unsloth_offload_checkpointing = unsloth_offload

    def disable_gradient_checkpointing(self):
        self._init_checkpointing()

    def _run(self, *args):
        if torch.is_grad_enabled() and self.training and self.gradient_checkpointing:
            if self.unsloth_offload_checkpointing:
                return unsloth_checkpoint(self._forward, *args)
            if self.cpu_offload_checkpointing:
                # Saved inputs (the previous block's output) stay in CPU RAM; the block runs on its own device and
                # hands a CPU tensor to the next block. The callers move the stream back after their loops.
                func = self._forward
                device = next(self.parameters()).device

                def custom_forward(*inputs):
                    return to_cpu(func(*to_device(inputs, device)))

                return torch_checkpoint(custom_forward, *args, use_reentrant=False)
            return torch_checkpoint(self._forward, *args, use_reentrant=False)
        return self._forward(*args)


class MossTextBlock(_CheckpointMixin, nn.Module):
    """Text-only refiner block (official ``_TextBlock``): pre-norm, no modulation, biases."""

    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = MossRMSNorm(dim), MossRMSNorm(dim)
        self.q_norm, self.k_norm = MossRMSNorm(dim // heads), MossRMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)
        self._init_checkpointing()

    def _forward(self, x, rope, attn_params):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(joint_attention(q, k, v, rope, self.heads, attn_params))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))

    def forward(self, x, rope, attn_params):
        return self._run(x, rope, attn_params)


class MossTextProjection(nn.Module):
    """Official ``_TextProjection``: Linear 4096->3072 then two text-only blocks."""

    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([MossTextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope, attn_params):
        x = self.proj(x.to(self.proj.weight.dtype))
        device = x.device
        if torch.is_grad_enabled() and not x.requires_grad and any(b.gradient_checkpointing for b in self.blocks):
            # The cached text is a constant and proj is frozen: without this, unsloth's custom autograd checkpoint
            # returns outputs without grad and LoRA on the text refiner blocks (preset moss-text) never learns.
            x.requires_grad_(True)
        for block in self.blocks:
            x = block(x, rope, attn_params)
        return x.to(device)  # CPU-offload checkpointing hands back CPU tensors


class MossBlock(_CheckpointMixin, nn.Module):
    """Joint single-stream block (official ``_Block``): shared modulation, gated residuals, no biases.

    ``proj`` is an ``nn.ModuleDict`` of plain Linears (q, k, v, o, ffn_up_gate, ffn_down): this is what LoRA/LyCORIS
    wrap, giving keys like ``lora_unet_layers_0_proj_q``. Never fuse these weights at runtime (LoRA would be bypassed).
    """

    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = MossRMSNorm(dim), MossRMSNorm(dim)
        self.q_norm, self.k_norm = MossRMSNorm(dim // heads), MossRMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)
        self._init_checkpointing()

    def _forward(self, x, rope, modulation, attn_params):
        # Same math as the official _Block; the elementwise chains come from _chains() (eager or compiled).
        fn = _chains(x)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        dtype = self.proj["q"].weight.dtype
        y = fn["norm_modulate"](x, self.norm1.weight, scale_a, shift_a).to(dtype)
        q = fn["qk_norm_rope"](self.proj["q"](y), self.q_norm.weight, rope, self.heads)
        k = fn["qk_norm_rope"](self.proj["k"](y), self.k_norm.weight, rope, self.heads)
        v = self.proj["v"](y)
        v = v.reshape(*v.shape[:2], self.heads, -1)
        attn = attention.attention([q.to(v.dtype), k.to(v.dtype), v], attn_params=attn_params)  # [B, L, H*D]
        x = fn["gated_residual"](x, self.proj["o"](attn), gate_a)
        y = fn["norm_modulate"](x, self.norm2.weight, scale_f, shift_f)
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)  # first half hidden(up), second gate
        return fn["gated_residual"](x, self.proj["ffn_down"](fn["situ"](hidden, gate)), gate_f)

    def forward(self, x, rope, modulation, attn_params):
        return self._run(x, rope, modulation, attn_params)


class MossRoPE(nn.Module):
    """4D RoPE (r, f, h, w). Bands are derived fp32 constants; the ``rope.bands`` key is kept for strict loading."""

    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self._default_bands(torch.device("cpu")))

    def _default_bands(self, device):
        exponent = torch.arange(self.num_bands, device="cpu", dtype=torch.float32) / self.num_bands
        return (1 / (self.base ** exponent)).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if self.bands.device.type != "meta":
            self.bands = self._default_bands(self.bands.device)  # .to(bf16) must never degrade the frequencies
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        if self.bands.device.type != "meta":
            self.bands = self._default_bands(self.bands.device)

    def _bands(self, device):
        if self.bands.dtype != torch.float32 or self.bands.device != device:
            return self._default_bands(device)
        return self.bands

    def forward(self, height, width, text_length, device):
        """Official layout for one sample: [1, HW + T, 128] fp32; image first, image f = T, h/w centred."""
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self._bands(device)
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)

    def batched(self, height, width, lengths: Iterable[int], padded_length: int, device):
        """[B, HW + padded_length, 128]: each sample uses its own true text length; padded rows are zeros."""
        rows = []
        for length in lengths:
            rope = self(height, width, int(length), device)
            rows.append(F.pad(rope, (0, 0, 0, padded_length - int(length))))
        return torch.cat(rows, 0)


class MossTransformer(nn.Module):
    """MOSS-Image DiT (official MOSSImageTransformer, ~5.08B params). Attribute names = checkpoint keys."""

    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, freq_dim=256,
                 latent_channels=128, rope_base=10000.0, attn_mode: str = "torch", split_attn: bool = False):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError("Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.")
        self.config = dict(dim=dim, ffn_dim=ffn_dim, num_heads=num_heads, num_layers=num_layers, text_dim=text_dim,
                           freq_dim=freq_dim, latent_channels=latent_channels, rope_base=rope_base)
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = MossTextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = MossRoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([MossBlock(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = MossRMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

        self.attn_mode = attn_mode
        self.split_attn = split_attn
        self.num_blocks = num_layers
        self.blocks_to_swap = None
        self.offloader: Optional[custom_offloading_utils.ModelOffloader] = None

    # ---- properties / switches -----------------------------------------------------------------------------
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, cpu_offload: bool = False, unsloth_offload: bool = False):
        for block in [*self.layers, *self.text_embedder.blocks]:
            block.enable_gradient_checkpointing(cpu_offload=cpu_offload, unsloth_offload=unsloth_offload)

    def disable_gradient_checkpointing(self):
        for block in [*self.layers, *self.text_embedder.blocks]:
            block.disable_gradient_checkpointing()

    # ---- block swap (same API as library/anima_models.py) --------------------------------------------------
    def enable_block_swap(self, num_blocks: int, device: torch.device):
        self.blocks_to_swap = num_blocks
        assert self.blocks_to_swap <= self.num_blocks - 2, (
            f"Cannot swap more than {self.num_blocks - 2} blocks. Requested: {self.blocks_to_swap} blocks.")
        self.offloader = custom_offloading_utils.ModelOffloader(self.layers, self.blocks_to_swap, device)
        logger.info(f"MOSS: block swap enabled, swapping {num_blocks} of {self.num_blocks} blocks, device {device}.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap:
            saved = self.layers
            self.layers = None
        self.to(device)
        if self.blocks_to_swap:
            self.layers = saved

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap:
            try:
                self.offloader.prepare_block_devices_before_forward(self.layers)
            except TypeError:
                self.offloader.prepare_block_devices_before_forward(self.layers, free_cache=True)

    # ---- forward -------------------------------------------------------------------------------------------
    def time_conditioning(self, timestep, batch, device):
        """float64 sinusoid -> fp32 time MLP -> shared block modulation [B,6,D] and head modulation [B,2,D]."""
        times = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1).expand(batch)
        half = self.config["freq_dim"] // 2
        freq = 10000 ** (-torch.arange(half, device=device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        return mod, head

    @staticmethod
    def _right_align_text(text: torch.Tensor, mask: torch.Tensor):
        """Move valid text rows to the front of each sample (the official forward drops padded rows anywhere)."""
        lengths = mask.sum(1)
        positions = torch.arange(mask.shape[1], device=mask.device)[None]
        if torch.equal(mask, positions < lengths[:, None]):
            return text, mask, lengths
        order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)
        text = torch.gather(text, 1, order[..., None].expand_as(text))
        return text, positions < lengths[:, None], lengths

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, text: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None, skip_layers: Optional[Iterable[int]] = None) -> torch.Tensor:
        """x [B,128,h,w] noisy latent, timesteps [B] = 1000*sigma (fp32), text [B,S,4096], text_mask [B,S] bool.

        Returns the velocity prediction (noise - x0) [B,128,h,w]. ``skip_layers`` skips main blocks (inference
        guidance only; the official sampler skips block 22 in the negative branch)."""
        skip = frozenset(() if skip_layers is None else skip_layers)
        if any(not (0 <= i < len(self.layers)) for i in skip):
            raise ValueError(f"skip_layers must be block indices in [0, {len(self.layers)}).")
        if x.ndim != 4 or x.shape[1] != self.config["latent_channels"]:
            raise ValueError("x must be [B, latent_channels, h, w].")
        batch, _, height, width = x.shape
        if text.ndim != 3 or text.shape[0] != batch:
            raise ValueError("text must be [B, S, text_dim].")
        if text_mask is None:
            text_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=x.device)
        text_mask = text_mask.to(device=x.device).bool()
        if text_mask.shape != text.shape[:2]:
            raise ValueError("text_mask must be [B, S].")
        text, text_mask, lengths = self._right_align_text(text.to(x.device), text_mask)
        if bool((lengths == 0).any()):
            raise ValueError("Each sample needs at least one valid text token (an empty caption still has 10).")
        max_len = int(lengths.max())
        text, text_mask = text[:, :max_len], text_mask[:, :max_len]
        img_len = height * width

        mod, head = self.time_conditioning(timesteps, batch, x.device)
        video = x.flatten(2).transpose(1, 2)
        video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())  # [B, HW, D]
        # fp32 phases, then the activation dtype (official: rope.to(video.dtype)).
        rope = self.rope.batched(height, width, lengths.tolist(), max_len, x.device).to(video.dtype)

        # xformers' block-diagonal bias needs packed sequences; with padded batches use per-sample attention instead.
        split = self.split_attn or (self.attn_mode == "xformers" and batch > 1)
        text_params = attention.AttentionParams.create_attention_params_from_mask(self.attn_mode, split, 0, text_mask)
        hidden = torch.cat((video, self.text_embedder(text, rope[:, img_len:], text_params)), 1)
        joint_params = attention.AttentionParams.create_attention_params_from_mask(self.attn_mode, split, img_len, text_mask)

        for index, layer in enumerate(self.layers):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)
            if index not in skip:
                hidden = layer(hidden, rope, mod, joint_params)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks(self.layers, index)
        hidden = hidden.to(x.device)  # CPU-offload checkpointing hands back CPU tensors

        hidden = self.final_norm_video(hidden[:, :img_len].to(self.final_norm_video.weight.dtype))
        hidden = hidden * (1 + head[:, 1, None]) + head[:, 0, None]
        hidden = self.final_linear_video(hidden.to(self.final_linear_video.weight.dtype))
        return hidden.transpose(1, 2).reshape(batch, -1, height, width)
