# MOSS-Image DiT for training.
# Adapted from OpenMOSS MOSS-Image ``pipeline_moss_image.py`` (MOSSImageTransformer), Copyright OpenMOSS Team,
# Apache License 2.0. Module names and math are unchanged so the released checkpoint loads strictly and LoRA keys
# match the checkpoint keys. Changes for training: batched forward with padded text (key-padding mask through
# library.attention instead of the official per-sample loop - same result, see tests/test_moss_models.py),
# gradient checkpointing (plain / CPU offload / unsloth offload) and block swap like library/anima_models.py.

import contextlib
import math
import sys
from typing import Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from library import anima_models as _ckpt  # the checkpointers and the no-grad first-pass switches, shared with Anima
from library import attention, custom_offloading_utils, lycoris_fast
from library.elementwise_compile import ElementwiseChains
from library.anima_models import to_cpu, to_device, unsloth_checkpoint
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

# Released MOSS-Image-Base / -SFT transformer/config.json
MOSS_DIT_CONFIG = dict(dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, freq_dim=256,
                       latent_channels=128, rope_base=10000.0)


def split_fused_qkv(state: dict) -> int:
    """Split ComfyUI-style fused joint-block ``layers.N.proj.qkv.weight`` [3 dim, dim] into q / k / v, in place.

    KBlueLeaf/MOSS-Image-heal's ComfyUI file (moss_image_heal_fp16.safetensors) fuses each joint block's attention input
    projections; every other key and shape is Base's. ComfyUI runs it as ``proj["qkv"](y).chunk(3, dim=-1)``, so the rows
    are q, then k, then v. The trainer keeps separate q / k / v Linears (the LoRA / LyCORIS targets; ComfyUI maps their
    keys onto the fused rows, comfy/lora.py). ``text_embedder.blocks.N.qkv`` is a fused Linear in the model itself and is
    not touched. Returns the number of blocks split (0 for a Base-layout file)."""
    count = 0
    for key in [k for k in state if k.startswith("layers.") and k.endswith(".proj.qkv.weight")]:
        prefix = key[:-len("qkv.weight")]
        clash = [prefix + name + ".weight" for name in ("q", "k", "v") if prefix + name + ".weight" in state]
        if clash:
            raise ValueError(f"{key} and separate {clash} in the same file")
        weight = state.pop(key)
        if weight.dim() != 2 or weight.shape[0] % 3:
            raise ValueError(f"{key} has shape {tuple(weight.shape)}: expected [3 * dim, dim]")
        for name, part in zip(("q", "k", "v"), weight.chunk(3, 0)):
            state[prefix + name + ".weight"] = part.clone()  # own storage: the fused block is not kept behind 3 views
        count += 1
    return count


class MossRMSNorm(nn.Module):
    """Direct-scale RMSNorm, eps 1e-6, fp32 internals (official ``_RMSNorm``)."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        # None: the official 1e-6. A float only on the norms that read the joint stream under --moss_stream_scale
        # (MossTransformer.set_stream_scale). A plain attribute: never in a state dict.
        self.eps: Optional[float] = None

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


def rms_norm(x, weight, eps: Optional[float] = None):
    # eps None keeps the literal 1e-6 in the expression: the same eager kernels and the same torch.compile graph as
    # before --moss_stream_scale existed (dynamo specialises None; a float eps is a graph input on torch 2.14)
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + (1e-6 if eps is None else eps))
    return y.to(x.dtype) * weight


# ---- --moss_stream_scale C (2026-11-15, default 1.0 = off): an exact fold of the joint residual stream -------------------
# Under fp16 the joint stream of the 42 MossBlocks grows past fp16's largest value (65504); --moss_stream_fp32 keeps it in
# fp32 instead (~4 GiB more at the 2048 tier). The fold keeps it in the training dtype, C times smaller:
# * everything that reads the joint stream is an RMSNorm - norm1 / norm2 of every block (inside norm_modulate) and
#   final_norm_video - and RMSNorm(C * x, eps * C**2) == RMSNorm(x, eps). Those norms get eps = 1e-6 * C**2.
#   q_norm / k_norm read the projections of the normalised stream and the text refiner's norms read its own residual
#   stream (not scaled): they keep 1e-6.
# * everything that writes into it is multiplied by C, out of place, in MossTransformer.forward: the joint input (the
#   video embedder output and the text refiner output, concatenated) and the two gates of the shared modulation
#   (gate_a, gate_f: rows 2 and 5 of [shift_a, scale_a, gate_a, shift_f, scale_f, gate_f], the order MossBlock._forward
#   unbinds) - once per forward, before the joint blocks, so every block, every checkpoint / recompute / compiled graph
#   and every sampling forward get the scaled modulation without a change inside MossBlock.
# * C is a power of two in (0, 1]: each multiplication by C or C**2 changes only the exponent, so it is exact in fp16 /
#   bf16 / fp32 unless a value leaves the normal range (fp16 values below ~6.1e-5 * C round as subnormals). The block
#   outputs are exactly C times the unscaled ones, the norm outputs - and with them every LoRA / LyCORIS input, output and
#   gradient - are the same, and the stream's gradient is exactly 1 / C times its unscaled one (CPU tests: bit-identical,
#   tests/test_moss_stream_scale.py). No weight changes: a --resume state or --network_weights trained without the
#   option continue with it and back.
# * C == 1.0 is the code before the option: forward multiplies nothing (a Python float check) and every eps stays None.
def stream_scale_problem(scale) -> Optional[str]:
    """Why `scale` cannot be --moss_stream_scale (a sentence), or None for a power of two in (0, 1] (2**-k, k >= 0)."""
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        return f"must be a number (a power of two in (0, 1], e.g. 0.0625 = 1/16), got {scale!r}"
    value = float(scale)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        return f"must be in (0, 1] (1.0 = off, 0.0625 = 1/16 recommended for fp16), got {scale!r}"
    if math.frexp(value)[0] != 0.5:
        return (f"must be a power of two (1, 0.5, 0.25, ..., 0.0625 = 1/16) so that multiplying by it is exact, got"
                f" {scale!r}")
    return None


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
def norm_modulate(x, weight, scale, shift, eps: Optional[float] = None):
    return rms_norm(x, weight, eps) * (1 + scale[:, None]) + shift[:, None]


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
    if v.dtype == torch.float32 and not attn_params.supports_fp32:
        # --moss_stream_fp32 runs the text refiner outside autocast, and flash-attn has no fp32 kernels
        return attention.attention([t.to(torch.bfloat16) for t in (q, k, v)], attn_params=attn_params).float()
    return attention.attention([q.to(v.dtype), k.to(v.dtype), v], attn_params=attn_params)  # [B, L, H*D]


# ---- no-grad first pass + merged LoKr weight cache / joint-attention reuse (--moss_ckpt_nograd_merged,
# --moss_ckpt_reuse_attn; 2026-11-14, both off by default) ------------------------------------------------------------
# The --anima_ckpt_nograd_merged / --anima_ckpt_reuse_attn machinery (library/anima_models.py NoGradFirstPassCheckpointer,
# library/lycoris_fast.py "checkpoint merged-weight cache", library/attention.py _FlashReuse - read those section comments)
# applied to the joint MossBlocks. Nothing is duplicated here: the plain (no offload) checkpoint of a MossBlock becomes
# that node, which calls MossBlock._forward(..., attn_stash=list) for the no-grad first pass and
# MossBlock._forward(..., attn_reuse=(out, softmax_lse)) for the recompute in backward.
# * The switches are the process-wide ones of library/anima_models.py (CKPT_NOGRAD_MERGED, CKPT_REUSE_ATTN, and the
#   CKPT_RECOMPUTE_DIFFS check): one trainer runs per process, like block_compile.pause. set_ckpt_reuse() sets them.
# * What is cached: the merged LoKr+DoRA weight of every eligible adapted Linear of the block (proj.q / k / v / o /
#   ffn_up_gate / ffn_down). A block left out of the network (--moss_lora_exclude_blocks) has none and still gets the
#   no-grad first pass and the attention reuse.
# * What is reused: the block's ONE attention call - the joint text+image self-attention - and only when it is the plain
#   flash call: one sample, or samples of equal text length (with or without a mask in the params: attention() trims a
#   uniform mask away). A padded batch (different text lengths) goes through flash's varlen kernel: nothing is stored, the
#   recompute runs the forward kernel as before. MOSS has no cross-attention.
# * Under --moss_block_compile a padded batch keeps torch.utils.checkpoint (_CheckpointMixin._ckpt_keeps_plain): its
#   masked attention params carry per-length Python ints, so every padded length compiles a graph of its own, and the
#   no-grad node would compile two per length (first pass and recompute differ in grad mode and _CKPT_PASS; the plain
#   checkpoint runs both passes through one graph) for a call that has nothing to reuse.
# * Not the two text refiner blocks (MossTextBlock): they keep torch.utils.checkpoint. Their output enters joint block 0
#   as part of its x, a tensor with a single consumer, so LoRA on them (preset moss-text) gets the same gradients.
# * Tensors that enter every joint block from outside: rope (no grad) and the shared modulation. Each block reads the
#   modulation once (unbind), so the per-block gradients reach its producer one per block in both checkpoints - unlike
#   Anima's three adaLN uses per Block there is no per-block pre-sum that could round differently.
# * Sampling and validation run under no_grad: _run calls _forward directly, lycoris_fast._CKPT_PASS stays 0 and neither
#   the cache nor a stash is ever consulted. A sample round drops the cache first (library/moss/train_utils.py
#   sample_images: ~8.65 GiB on the real model that sampling would only carry), the next training step builds it again.
# Both off: _run and MossBlock._forward run exactly as before.
#
# Reuse-only (2026-11-15): --moss_ckpt_reuse_attn WITHOUT --moss_ckpt_nograd_merged. The cache is one bf16 / fp16 copy
# of every adapted Linear (~8.65 GiB on MOSS-Image) and measured +-0 % on its own on Anima; the gain is the attention
# reuse, and with the cache the 2048 tier does not fit a 32 GB card. So the two switches are independent here:
# * CKPT_REUSE_ATTN alone (CKPT_NOGRAD_MERGED off): a joint block still goes through the same node (no-grad first pass,
#   stored attention, recompute + backward in the node's backward), but the node gets an empty stand-in module where it
#   expects the block (_NO_CACHE), so it finds no module to cache and builds nothing, and the function it runs
#   (MossBlock._forward_uncached) keeps lycoris_fast._CKPT_PASS at 0: every adapted Linear rebuilds its merged LoKr+DoRA
#   weight in both passes exactly as under torch.utils.checkpoint, and a cache tensor left on a module by an earlier
#   cached run in the same process is never read. library/anima_models.py is not touched: Anima's path is the same code.
#   The node's check "the parameters are still what the first pass read" works on the modules it caches, i.e. on none
#   here, so that function makes it itself with the same keys (MossBlock._ckpt_adapter_keys) before it lets the
#   recompute take the stored attention.
#   Where nothing is compiled, both passes run the same kernels on the same values (CPU tests: bit-identical outputs
#   and gradients). On the GPU that is only the run with --moss_no_compile and --moss_no_lokr_compile and without
#   --moss_block_compile. Every compile splits by grad mode, so the no-grad first pass and the grad-mode recompute are
#   two compiled graphs:
#   - under --moss_block_compile, of the whole block (its fused weight rebuilds and chains);
#   - without it, in the default CUDA run, of the LoKr+DoRA rebuild (lycoris_fast.enable_compiled_rebuild) and of the
#     elementwise chains above (_CHAINS; prepare_moss_runtime switches both on). The plain checkpoint runs both of its
#     passes in grad mode, through one graph of each.
#   The two graphs may round their reductions differently (DoRA's vector_norm, the RMSNorm means): then the stored
#   attention output belongs to q / k / v that differ from the recompute's at rounding level. The cached mode hands
#   the recompute the first pass's weight bits (not its q / k chains, which split the same way); this mode has no such
#   tensor. Not measurable on CPU (aot_eager: bit-identical; the chain and rebuild compiles are CUDA-only), for the GPU
#   harness to report (CKPT_RECOMPUTE_DIFFS; tests/manual_test_moss_ckpt_reuse_gpu.py modes "chains" and "compiled").
# * a byte budget for the stored attention (CKPT_REUSE_BUDGET_BYTES, --moss_ckpt_reuse_attn_budget_gib): every block
#   stores joint tokens x rows x ckpt_stash_bytes_per_token bytes until its backward, and at the start of backward all of
#   them are alive. Per call, from the shape of x (Python ints, outside any compiled graph, no device read):
#   n = floor(budget / bytes per block); the blocks with index < n store and reuse, the others do not - they keep
#   torch.utils.checkpoint (reuse-only) or the node without a stash (with the cache). The index is set when the model is
#   built. The compiled entry is shared by all blocks, so n moving between micro-steps compiles nothing new: the kinds of
#   call (plain grad forward / no-grad first pass with a stash / recompute with the stored tensors) each compile once
#   (x2 with a block left out of the network: 6 graphs). With the cache on, the blocks over the budget run the node
#   without a stash (_forward_no_stash): first pass and recompute each exist with and without the stored attention - 8
#   graphs, which is why that combination needs --moss_block_compile_recompile_limit >= 16
#   (train_utils.validate_moss_ckpt_reuse_args; the default is 64).
CKPT_REUSE_BUDGET_BYTES: Optional[int] = None  # None: no limit, every joint block stores and reuses
_NO_CACHE = nn.Module()  # what the node gets instead of the block in reuse-only mode: nothing in it to cache
_NO_LIMIT = sys.maxsize


def ckpt_stash_bytes_per_token(dim: int, heads: int) -> int:
    """Bytes one token of the joint sequence adds to a block's stored attention: the flash output [L, heads, head_dim]
    in the autocast dtype (bf16 / fp16: 2 bytes) + softmax_lse [heads, L] in fp32. MOSS-Image: 3072 * 2 + 24 * 4 = 6240."""
    return int(dim) * 2 + int(heads) * 4


def ckpt_reuse_block_count(budget_bytes: Optional[int], tokens: int, rows: int, per_token: int) -> int:
    """How many joint blocks (those with index < the result) may store their attention under `budget_bytes` when one
    block stores tokens * rows * per_token bytes: floor(budget / that). No budget, or nothing to store: no limit (a
    number above any block count; callers compare an index with it or clip it)."""
    if budget_bytes is None:
        return _NO_LIMIT
    if budget_bytes < 0:
        raise ValueError(f"the stored-attention budget must be >= 0 bytes, got {budget_bytes}")
    per_block = int(tokens) * int(rows) * int(per_token)
    if per_block <= 0:
        return _NO_LIMIT
    return int(budget_bytes) // per_block


def set_ckpt_reuse(nograd_merged: bool, reuse_attn: bool = False, reuse_budget_gib: Optional[float] = None) -> None:
    """--moss_ckpt_nograd_merged / --moss_ckpt_reuse_attn / --moss_ckpt_reuse_attn_budget_gib: the process-wide switches
    (library/anima_models.py's two, and the budget here). Either of the first two sends the joint blocks through the
    no-grad first-pass node; nograd_merged alone decides whether merged weights are cached."""
    budget = None
    if reuse_budget_gib is not None:
        gib = float(reuse_budget_gib)
        if not math.isfinite(gib) or gib < 0:
            raise ValueError(f"--moss_ckpt_reuse_attn_budget_gib must be a finite number >= 0, got {reuse_budget_gib!r}")
        budget = int(gib * 2**30)
    global CKPT_REUSE_BUDGET_BYTES
    _ckpt.CKPT_NOGRAD_MERGED = bool(nograd_merged)
    _ckpt.CKPT_REUSE_ATTN = bool(reuse_attn)
    CKPT_REUSE_BUDGET_BYTES = budget


def ckpt_nograd_merged_enabled() -> bool:
    """--moss_ckpt_nograd_merged: the merged-weight cache (and with it the node) is on."""
    return bool(_ckpt.CKPT_NOGRAD_MERGED)


def ckpt_node_enabled() -> bool:
    """The joint blocks' plain checkpoint may be the no-grad first-pass node: either switch is on."""
    return bool(_ckpt.CKPT_NOGRAD_MERGED or _ckpt.CKPT_REUSE_ATTN)


def ckpt_reuse_attn_enabled() -> bool:
    """--moss_ckpt_reuse_attn: the joint attention is stored and reused (with or without the cache)."""
    return bool(_ckpt.CKPT_REUSE_ATTN)


def ckpt_reuse_budget_bytes() -> Optional[int]:
    return CKPT_REUSE_BUDGET_BYTES


class _CheckpointMixin:
    """Per-block gradient checkpointing switch, same three modes as library/anima_models.py Block."""

    _ckpt_nograd = False  # True on MossBlock: its plain checkpoint may be the no-grad first-pass one (section above)

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

    def _ckpt_keeps_plain(self, args) -> bool:
        """--moss_ckpt_nograd_merged / --moss_ckpt_reuse_attn: True for a call that stays on torch.utils.checkpoint
        although a switch is on - a
        compiled block (library/moss/block_compile.py installs `_forward` on the instance) called with masked attention
        params: a padded batch, or --split_attn. Those params carry per-length Python ints (seqlens_host, max_seqlen,
        img_len), so each padded length specialises a compiled graph. The no-grad node would make that two graphs per
        length - its first pass runs under no_grad with lycoris_fast._CKPT_PASS == 1 (and a stash list), its recompute in
        grad mode with _CKPT_PASS == 2, where the plain checkpoint runs both passes through one graph - and has nothing
        to offer there: flash varlen stores no attention output, and the no-grad pass alone measured +-0 % on Anima."""
        return "_forward" in self.__dict__ and getattr(args[-1], "attention_mask", None) is not None

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
            if ((_ckpt.CKPT_NOGRAD_MERGED or _ckpt.CKPT_REUSE_ATTN) and self._ckpt_nograd and args[0].requires_grad
                    and not self._ckpt_keeps_plain(args)):
                # --moss_ckpt_nograd_merged / --moss_ckpt_reuse_attn: no offload, the first pass under no_grad
                # (NoGradFirstPassCheckpointer; an autograd node needs a grad-bearing input, so a block whose x has none
                # stays on torch.utils.checkpoint, and so does a compiled block's padded call: _ckpt_keeps_plain)
                reuse = _ckpt.CKPT_REUSE_ATTN and self._ckpt_reuses(args[0])
                if _ckpt.CKPT_NOGRAD_MERGED:
                    # the cache is on: every joint block gets the node; a block over the stored-attention budget runs
                    # it without the stash (the node makes one whenever CKPT_REUSE_ATTN is set)
                    function = self._forward if reuse or not _ckpt.CKPT_REUSE_ATTN else self._forward_no_stash
                    return _ckpt.nograd_checkpoint(function, self, *args)
                if reuse:  # reuse-only: the node with nothing to cache; a block over the budget keeps the plain checkpoint
                    return _ckpt.nograd_checkpoint(self._forward_uncached(), _NO_CACHE, *args)
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
        # --moss_ckpt_reuse_attn_budget_gib (plain Python attributes, not buffers: nothing reaches a state dict or a
        # compiled graph): the bytes per token this block's stored attention takes, and the block's position in
        # MossTransformer.layers (set there; a standalone block is 0)
        self._ckpt_stash_per_token = ckpt_stash_bytes_per_token(dim, heads)
        self._ckpt_index = 0

    _ckpt_nograd = True  # --moss_ckpt_nograd_merged / --moss_ckpt_reuse_attn apply to the joint blocks

    def _ckpt_reuses(self, x) -> bool:
        """--moss_ckpt_reuse_attn: whether this block stores and reuses its attention for a call with the stream x
        [rows, joint tokens, dim] under the byte budget (no budget: always). Shape ints only: no device read."""
        budget = CKPT_REUSE_BUDGET_BYTES
        if budget is None:
            return True
        return self._ckpt_index < ckpt_reuse_block_count(budget, x.shape[1], x.shape[0], self._ckpt_stash_per_token)

    def _ckpt_adapter_keys(self, x) -> tuple:
        """Reuse-only: what the adapted Linears of this block are built from right now (lycoris_fast._ckpt_key per LoKr
        module: version counter and data pointer of its factors, DoRA scale and base weight, the multiplier, the
        invalidation generation) - the cached mode's check that the stored attention still belongs to the parameters,
        without a cache. Host values only. The module list is found once per block and kept under a name of its own
        (not the node's _ckpt_lokrs: nothing here may make the node cache); an empty list (a block left out of the
        network, a non-LyCORIS network) is looked up again, like the node's."""
        lokrs = self.__dict__.get("_ckpt_guard_lokrs")
        if not lokrs:
            linears = [m for m in self.modules() if isinstance(m, nn.Linear)]
            lokrs = self.__dict__["_ckpt_guard_lokrs"] = lycoris_fast.ckpt_merged_modules(self, linears)
        dtype = torch.get_autocast_dtype(x.device.type)
        return tuple(lycoris_fast._ckpt_key(lokr, dtype) for lokr in lokrs)

    def _forward_uncached(self):
        """The node's function in reuse-only mode (one object per call of the block: it carries the first pass's keys
        to the recompute). It runs _forward (the compiled one under --moss_block_compile) with lycoris_fast._CKPT_PASS
        held at 0, so no adapted Linear looks for a cached merged weight in either pass, and it hands the recompute the
        stored attention only while the adapter is what the first pass read: a parameter written between a forward and
        its backward (nothing in the trainer does that) makes the recompute run the attention plainly, as
        torch.utils.checkpoint would and as the cached mode does. Like there, a write through `p.data` is not seen."""
        block, keys = self, []

        def _forward_uncached(*args, **kwargs):
            if "attn_stash" in kwargs:  # the no-grad first pass
                keys.append(block._ckpt_adapter_keys(args[0]))
            elif "attn_reuse" in kwargs and (not keys or block._ckpt_adapter_keys(args[0]) != keys[-1]):
                del kwargs["attn_reuse"]
            with lycoris_fast.ckpt_pass(0):
                return block._forward(*args, **kwargs)

        return _forward_uncached

    def _forward_no_stash(self, *args, attn_stash=None, attn_reuse=None):
        """The node's function for a block over the stored-attention budget while the cache is on: the stash list the
        node offers is left empty, so nothing is stored and the recompute runs the attention as the plain checkpoint."""
        return self._forward(*args)

    def _forward(self, x, rope, modulation, attn_params, attn_stash=None, attn_reuse=None):
        # Same math as the official _Block; the elementwise chains come from _chains() (eager or compiled).
        # attn_stash / attn_reuse: only NoGradFirstPassCheckpointer passes them (--moss_ckpt_reuse_attn): the first pass
        # fills the list with the joint attention's flash output and softmax_lse, the recompute gets them back as a tuple
        # --moss_stream_scale: x and the gates arrive scaled by C (MossTransformer.forward), norm1 / norm2 carry
        # eps = 1e-6 * C**2 (None when off)
        fn = _chains(x)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        dtype = self.proj["q"].weight.dtype
        y = fn["norm_modulate"](x, self.norm1.weight, scale_a, shift_a, self.norm1.eps).to(dtype)
        q = fn["qk_norm_rope"](self.proj["q"](y), self.q_norm.weight, rope, self.heads)
        k = fn["qk_norm_rope"](self.proj["k"](y), self.k_norm.weight, rope, self.heads)
        v = self.proj["v"](y)
        v = v.reshape(*v.shape[:2], self.heads, -1)
        if attn_stash is None and attn_reuse is None:
            attn = attention.attention([q.to(v.dtype), k.to(v.dtype), v], attn_params=attn_params)  # [B, L, H*D]
        else:  # a checkpointed block's joint attention under --moss_ckpt_reuse_attn
            attn = attention.attention([q.to(v.dtype), k.to(v.dtype), v], attn_params=attn_params, ckpt_stash=attn_stash,
                                       ckpt_reuse=attn_reuse)
        x = fn["gated_residual"](x, self.proj["o"](attn), gate_a)
        y = fn["norm_modulate"](x, self.norm2.weight, scale_f, shift_f, self.norm2.eps)
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
        for index, layer in enumerate(self.layers):
            layer._ckpt_index = index  # --moss_ckpt_reuse_attn_budget_gib: the blocks with index < n store and reuse
        self.final_norm_video = MossRMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

        self.attn_mode = attn_mode
        self.split_attn = split_attn
        self.num_blocks = num_layers
        self.blocks_to_swap = None
        self.offloader: Optional[custom_offloading_utils.ModelOffloader] = None
        self.last_text_tokens: Optional[int] = None  # set by forward (a host int it computes anyway)
        self.stream_fp32 = False  # --moss_stream_fp32, see enable_stream_fp32
        self.stream_scale = 1.0  # --moss_stream_scale, see set_stream_scale (a Python float: 1.0 multiplies nothing)

    def set_stream_scale(self, scale: float = 1.0):
        """--moss_stream_scale C (section above rms_norm): carry the joint residual stream multiplied by C, a power of
        two in (0, 1]. The norms reading it get eps = 1e-6 * C**2; forward scales what writes into it. No parameter or
        buffer changes. 1.0 restores the plain forward (eps None). Any time before a forward; also on a compiled model
        (block compile reads the eps at trace time - the graphs traced before a change are guarded by its value)."""
        problem = stream_scale_problem(scale)
        if problem:
            raise ValueError(f"--moss_stream_scale {problem}")
        scale = float(scale)
        eps = None if scale == 1.0 else 1e-6 * scale * scale  # exact: C**2 is a power of two
        for layer in self.layers:
            layer.norm1.eps = eps
            layer.norm2.eps = eps
        self.final_norm_video.eps = eps
        self.stream_scale = scale

    def _scale_stream(self, hidden: torch.Tensor, mod: torch.Tensor, scale: float):
        """--moss_stream_scale: the joint input and the shared modulation's gate_a / gate_f (rows 2 and 5, MossBlock's
        unbind order) times C, out of place. The other four rows pass through untouched (the same values)."""
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod.unbind(1)
        mod = torch.stack((shift_a, scale_a, gate_a * scale, shift_f, scale_f, gate_f * scale), 1)
        return hidden * scale, mod

    def enable_stream_fp32(self):
        """Run the residual stream, modulation, norms, embedders and head in fp32 around the joint blocks' Linears.

        Those 252 Linears stay in the training dtype (they are almost all of the weights, and LoRA wraps them), and each
        block casts its stream input for them. Call this after the trainer's last whole-model ``.to(dtype)``."""
        for name, parameter in self.named_parameters():
            if not (name.startswith("layers.") and ".proj." in name):
                parameter.data = parameter.data.float()
        self.stream_fp32 = True

    def _around_blocks(self, device: torch.device):
        """Autocast off for what runs in fp32: under bf16 autocast those Linears would hand back a bf16 stream."""
        return torch.autocast(device.type, enabled=False) if self.stream_fp32 else contextlib.nullcontext()

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
        """Move valid text rows to the front of each sample (the official forward drops padded rows anywhere). Returns
        (text, mask, lengths as Python ints): the lengths and the already-aligned test reach the host in ONE transfer,
        and every later decision of the forward (empty check, trim, RoPE rows, the attention trims of every block) reads
        these ints - reading the device tensors was a GPU->CPU sync each, two per attention call in every block."""
        lengths = mask.sum(1)
        positions = torch.arange(mask.shape[1], device=mask.device)[None]
        aligned = torch.eq(mask, positions < lengths[:, None]).all()  # torch.equal of two same-shape bool tensors
        host = torch.cat((lengths.to(torch.int64), aligned.reshape(1).to(torch.int64))).tolist()
        lengths_host = host[:-1]
        if host[-1]:
            return text, mask, lengths_host
        order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)
        text = torch.gather(text, 1, order[..., None].expand_as(text))
        return text, positions < lengths[:, None], lengths_host

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
        text, text_mask, lengths = self._right_align_text(text.to(x.device), text_mask)  # lengths: Python ints
        if min(lengths) == 0:
            raise ValueError("Each sample needs at least one valid text token (an empty caption still has 10).")
        max_len = max(lengths)
        text, text_mask = text[:, :max_len], text_mask[:, :max_len]
        img_len = height * width
        self.last_text_tokens = max_len  # the text length this forward ran with (--moss_tier_timing reads it)

        with self._around_blocks(x.device):
            mod, head = self.time_conditioning(timesteps, batch, x.device)
            video = x.flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())  # [B, HW, D]
            # fp32 phases, then the activation dtype (official: rope.to(video.dtype)).
            rope = self.rope.batched(height, width, lengths, max_len, x.device).to(video.dtype)

            # xformers' block-diagonal bias needs packed sequences; with padded batches use per-sample attention instead.
            split = self.split_attn or (self.attn_mode == "xformers" and batch > 1)
            # the host lengths go with the params: attention() decides its trim / per-sample split without a device read
            text_params = attention.AttentionParams.create_attention_params_from_mask(self.attn_mode, split, 0, text_mask,
                                                                                      text_lengths=lengths)
            hidden = torch.cat((video, self.text_embedder(text, rope[:, img_len:], text_params)), 1)
            if self.stream_scale != 1.0:
                # --moss_stream_scale: the video embedder and text refiner outputs and every block's gates write into the
                # joint stream C times smaller (the text refiner's own stream stays as it is)
                hidden, mod = self._scale_stream(hidden, mod, self.stream_scale)
            if getattr(self, "_block_compile", False) and not split and all(n == max_len for n in lengths):
                # --moss_block_compile (library/moss/block_compile.py): no padding (one sample, or equal text lengths) -
                # the same attention without a mask, and no per-length Python ints in the params for dynamo to guard on
                # (a compiled graph per caption length otherwise)
                joint_params = attention.AttentionParams.create_attention_params(self.attn_mode, split)
            else:
                joint_params = attention.AttentionParams.create_attention_params_from_mask(self.attn_mode, split, img_len,
                                                                                           text_mask, text_lengths=lengths)

        for index, layer in enumerate(self.layers):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)
            if index not in skip:
                hidden = layer(hidden, rope, mod, joint_params)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks(self.layers, index)
        hidden = hidden.to(x.device)  # CPU-offload checkpointing hands back CPU tensors

        with self._around_blocks(x.device):
            hidden = self.final_norm_video(hidden[:, :img_len].to(self.final_norm_video.weight.dtype))
            hidden = hidden * (1 + head[:, 1, None]) + head[:, 0, None]
            hidden = self.final_linear_video(hidden.to(self.final_linear_video.weight.dtype))
        return hidden.transpose(1, 2).reshape(batch, -1, height, width)
