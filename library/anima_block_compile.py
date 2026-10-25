# Per-block torch.compile for the Anima DiT: --anima_block_compile (default off; 2026-10-26).
#
# Each Block's `_forward` (the attention / MLP math, NOT the gradient-checkpointing wrapper `forward`) is compiled with
# the LyCORIS-adapted Linears INSIDE the graph, so a micro-step runs ~3 compiled graph calls per Block instead of ~16
# Python module calls plus their kernels. The small-tier micro-steps of a multi-resolution Anima 2.9B set are bound by
# exactly that host work (the 640 LyCORIS module calls of a micro-step are ~0.5 s of Python, the launches another
# ~0.3-0.5 s; the GPU is idle), so this is where the step time of the 512 / 1024 tiers goes. Measured on a CPU harness
# (torch 2.7.1, aot_eager, the fork's own Block and LyCORIS fixed paths): one graph per Block, no graph breaks, outputs
# and gradients identical to eager for LoKr+DoRA, TSM stage 1 / 2 and TSM-LoKr (plain LoRA: the six adaLN-modulation
# up-weight gradients within a bf16 ulp or two, a K = 1 outer product rounded at another site in the AOT backward); per-Block
# launches TSM 672 -> 235 (GC) / 475 -> 153, LoKr+DoRA 980 -> 121 (GC) / 736 -> 77 (inductor CPU wrapper counts). GPU
# numbers: measure with the A/B in docs/anima_train_network.md.
#
# Pieces (and why each exists):
# * compile `_forward`, not `forward`: the unsloth / torch checkpoint wrappers hold `@torch._disable_dynamo` or are not
#   worth tracing; the checkpointed callable is the compiled `_forward`, so the recompute pass runs the compiled graph.
# * fake-5D flattening: the Block input (B, T, H, W, D) is passed as (B, 1, L, 1, D) (Anima.forward_mini_train_dit),
#   so the graph is keyed on the token count L alone and not on H and W separately (one graph per resolution otherwise).
#   Block._forward's `B, T, H, W, D = x.shape` / flatten / unflatten are the same ops on the fake shape (bit-exact).
# * dynamic sequence axis: the token axis of x and of the two RoPE tables is marked dynamic for the band of token counts
#   the dataset's buckets populate (torch._dynamo.mark_dynamic with the band's min / max), so every bucket of a tier
#   shares one graph. The marks are applied inside the (checkpointed) wrapper, not once up front: the recompute pass
#   receives detached inputs that would otherwise lose them. Under attn_mode=torch (SDPA) the marks are soft
#   (maybe_mark_dynamic): inductor's SDPA lowering guards the backward's logsumexp stride on seq % 8, which no strict
#   range satisfies (anima_lora GH #107); flash / xformers take the strict range.
# * the compiled path is taken only while the Block is in training mode and block compile is not paused (sampling,
#   validation stay eager: their resolutions and no-grad graphs would only add cache entries), and never for a
#   mixed-timestep TSM chunk (several rows with different timesteps in one forward: the masked path's expert union
#   changes per chunk and would recompile per combination; such chunks run the eager masked path).
# * the LyCORIS module forwards must be traceable: lycoris_algo_fixes.rebind_forwards (the adapter forward reached
#   through a closure cell instead of a bound method - dynamo 2.7.1 cannot otherwise read `org_module[0]`),
#   _tsm_scale / _tsm_experts / _get_timestep / _tsm_compute_gates read per-step hoisted data under compile (no host
#   syncs, no Python-int timestep in the graph: one graph per expert choice instead of one per integer timestep), and
#   library/attention.py's split-attention loops no longer create closure cells (`del q, k, v` on a cell variable is
#   DELETE_DEREF, which dynamo 2.7.1 does not handle). The re-bound closures must not carry a `__self__` attribute:
#   dynamo refuses to inline any function that has one ("Inline attempt with __self__").
# * known limitation: --split_attn slices each row by its device seqlens tensor (data-dependent): one graph break per
#   row inside every attention call. Anima's default attention (attn_mode torch, no split) traces whole.
# * the fork's own ElementwiseChains and LoKr weight-rebuild compiles are switched off under block compile (nested
#   torch.compile inside a compiled frame exhausts the recompile budget).
# * the per-module caches the fixed paths fill lazily (lycoris_fast._static_eligible: a float(scalar) .item() per LoKr
#   module) are primed before the first trace (lycoris_algo_fixes.prime_module_caches, called by the trainer after the
#   weights are loaded): traced cold, that .item() is a graph break and dynamo keeps the ~20 fragments it compiled on
#   the first call for the rest of the run.
# * the inductor cache is routed to a per-signature directory (the persistent caches key on the FX graph but not on the
#   mark_dynamic ranges; two processes compiled with different bands poison each other through the shared default dir).
#
# Cache entries = 2 (Block 0's input is bf16, Blocks 1+ see the fp32 residual) x expert choices (TSM stage 1: n_core;
# LoKr / LoRA: 1) x checkpointing regimes (GC: no-grad forward + grad forward + backward; the un-checkpointed tiers:
# forward + backward) x bands (1 with --anima_block_compile_dynamic union). The recompile limit is raised accordingly.
# All Blocks share the entries (dynamo keys on the code object; the parameters are graph inputs).
#
# The design follows sorryhyun/anima_lora (library/anima/models.py compile_blocks / _make_dynamic_seq_forward,
# library/datasets/buckets.py cluster_token_bands / band_for_seq, library/runtime/harness.py isolate_compile_cache),
# Copyright (c) 2026 Seunghyun Ji, MIT License - see THIRD_PARTY_NOTICES.md. Re-implemented for this fork's Block and
# LyCORIS modules.

from __future__ import annotations

import bisect
import contextlib
import hashlib
import logging
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

DYNAMIC_MODES = ("union", "bands", "static")
VAE_SPATIAL_DOWNSCALE = 8  # pixels -> latents (library/qwen_image_autoencoder_kl.py)

_active = True  # pause() clears it: sampling / validation run the eager Block._forward
_warned_oob: set = set()


@contextlib.contextmanager
def pause():
    """Run the eager Block._forward inside the block (sample images, validation): their resolutions and no-grad graphs
    would only add compile cache entries. Re-entrant."""
    global _active
    prev = _active
    _active = False
    try:
        yield
    finally:
        _active = prev


def is_active() -> bool:
    return _active


# ---------------------------------------------------------------------------------------------------------------------
# token bands
# ---------------------------------------------------------------------------------------------------------------------


def tokens_of_pixels(width: int, height: int, patch: int = 2) -> int:
    """DiT tokens of a width x height pixel bucket: ceil(latent / patch) per axis, latent = pixels / 8."""
    lh = -(-int(height) // VAE_SPATIAL_DOWNSCALE)
    lw = -(-int(width) // VAE_SPATIAL_DOWNSCALE)
    return (-(-lh // patch)) * (-(-lw // patch))


def cluster_token_bands(counts: Iterable[int], rel_gap: float = 0.10) -> List[Tuple[int, int]]:
    """Cluster a set of token counts into sorted (lo, hi) bands: split wherever the relative gap to the previous count
    exceeds rel_gap (one band per resolution tier for the usual 512 / 1024 / 1536 / 2048 buckets). [] for no counts."""
    ordered = sorted({int(c) for c in counts if int(c) > 0})
    if not ordered:
        return []
    bands: List[Tuple[int, int]] = []
    lo = prev = ordered[0]
    for c in ordered[1:]:
        if (c - prev) > rel_gap * prev:
            bands.append((lo, prev))
            lo = c
        prev = c
    bands.append((lo, prev))
    return bands


def band_for_seq(bands: Sequence[Tuple[int, int]], seq: int) -> Optional[Tuple[int, int]]:
    """The band containing seq, or None (bands sorted and non-overlapping)."""
    if not bands:
        return None
    i = bisect.bisect_right([b[0] for b in bands], seq) - 1
    if i < 0 or seq > bands[i][1]:
        return None
    return bands[i]


def union_band(bands: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """One band from the smallest to the largest count (one graph for every tier)."""
    if not bands:
        return []
    return [(min(b[0] for b in bands), max(b[1] for b in bands))]


def widen_band(band: Tuple[int, int], frac: float = 0.0, min_lo: int = 2) -> Tuple[int, int]:
    """Widen a band by frac on each side (mark_dynamic ranges are strict; a resolution outside every band runs an
    unmarked, static specialization). lo never drops below min_lo: dynamo treats 0 / 1 as special sizes."""
    lo, hi = band
    lo = max(min_lo, int(lo * (1.0 - frac)))
    hi = max(lo, int(-(-hi * (1.0 + frac) // 1)))
    return lo, hi


def token_counts_from_dataset_group(dataset_group: Any, patch: int = 2) -> List[int]:
    """The distinct DiT token counts of the buckets the dataset group's BucketManagers populate (pixel resos -> latents
    -> patches). A dataset without a bucket manager (or with resolution jitter, whose buckets are resampled per epoch)
    contributes its resolution / max_bucket_reso as a hint and is reported through the returned list being wider."""
    datasets = getattr(dataset_group, "datasets", None) or [dataset_group]
    counts: set = set()
    for ds in datasets:
        delegate = getattr(ds, "dreambooth_dataset_delegate", None)
        src = ds if delegate is None else delegate
        bm = getattr(src, "bucket_manager", None)
        resos = list(getattr(bm, "resos", []) or []) if bm is not None else []
        for reso in resos:
            w, h = int(reso[0]), int(reso[1])
            counts.add(tokens_of_pixels(w, h, patch))
        if getattr(src, "has_resolution_jitter", False):
            # the jittered buckets are resampled every epoch: cover the configured range with its corners
            lo_reso = getattr(src, "min_bucket_reso", None)
            hi_reso = getattr(src, "max_bucket_reso", None)
            if lo_reso:
                counts.add(tokens_of_pixels(int(lo_reso), int(lo_reso), patch))
            if hi_reso:
                counts.add(tokens_of_pixels(int(hi_reso), int(hi_reso), patch))
        if not resos:
            reso = getattr(src, "resolution", None)
            if reso:
                counts.add(tokens_of_pixels(int(reso[0]), int(reso[1]), patch))
    return sorted(counts)


def plan_bands(counts: Iterable[int], mode: str = "union", widen: float = 0.0) -> List[Tuple[int, int]]:
    """The mark_dynamic bands for a token-count set under --anima_block_compile_dynamic: "union" (one band, one graph),
    "bands" (one tight band per tier), "static" ([]: no marks, one graph per distinct token count)."""
    if mode not in DYNAMIC_MODES:
        raise ValueError(f"--anima_block_compile_dynamic must be one of {DYNAMIC_MODES}, got {mode!r}")
    if mode == "static":
        return []
    bands = cluster_token_bands(counts)
    if mode == "union":
        bands = union_band(bands)
    return [widen_band(b, widen) for b in bands]


# ---------------------------------------------------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------------------------------------------------

_ROPE_ARG_INDEX = 4  # Block._forward(x, emb, crossattn, attn_params, use_fp32, rope_cos_sin, ...) after binding


def _rope_of(args: tuple, kwargs: dict):
    if "rope_cos_sin" in kwargs:
        return kwargs["rope_cos_sin"]
    return args[_ROPE_ARG_INDEX] if len(args) > _ROPE_ARG_INDEX else None


def _mark_dynamic(x: torch.Tensor, rope, bands: Sequence[Tuple[int, int]], soft: bool) -> None:
    """Mark the token axis (dim 2 of the fake-5D x, dim 0 of each RoPE table) dynamic for the band containing it."""
    seq = int(x.shape[2])
    band = band_for_seq(bands, seq)
    if band is None:
        if seq not in _warned_oob:
            _warned_oob.add(seq)
            logger.warning(f"block compile: {seq} tokens is outside every compiled band {list(bands)}; this resolution"
                           " runs a static (unmarked) specialization of its own")
        return
    if soft:
        torch._dynamo.maybe_mark_dynamic(x, 2)
        if rope is not None:
            torch._dynamo.maybe_mark_dynamic(rope[0], 0)
            torch._dynamo.maybe_mark_dynamic(rope[1], 0)
    else:
        lo, hi = band
        torch._dynamo.mark_dynamic(x, 2, min=lo, max=hi)
        if rope is not None:
            torch._dynamo.mark_dynamic(rope[0], 0, min=lo, max=hi)
            torch._dynamo.mark_dynamic(rope[1], 0, min=lo, max=hi)


class CompiledBlocks:
    """What compile_blocks installed: the original `_forward`s (restore() puts them back) and counters."""

    def __init__(self):
        self.originals: List[Tuple[torch.nn.Module, str, Callable]] = []
        self.bands: List[Tuple[int, int]] = []
        self.compiled_calls = 0
        self.eager_calls = 0

    def restore(self) -> None:
        for block, name, fn in self.originals:
            if name in block.__dict__:
                del block.__dict__[name]  # the class method shows through again
        self.originals.clear()


def raise_recompile_limit(limit: int) -> int:
    """torch._dynamo.config.recompile_limit (alias cache_size_limit) and the accumulated limit, never lowered."""
    cfg = torch._dynamo.config
    target = max(int(getattr(cfg, "recompile_limit", 8)), int(limit))
    cfg.recompile_limit = target
    try:
        cfg.cache_size_limit = target  # the alias (older name; kept in sync for anything reading it)
    except Exception:  # noqa: BLE001 - alias may be read-only in a future torch
        pass
    try:
        cfg.accumulated_recompile_limit = max(int(getattr(cfg, "accumulated_recompile_limit", 256)), 4 * target)
    except Exception:  # noqa: BLE001
        pass
    return target


def compile_signature(bands: Sequence[Tuple[int, int]], backend: str, mode: Optional[str], dynamic: str) -> str:
    return f"torch={torch.__version__};backend={backend};mode={mode or None};dynamic={dynamic};bands={list(bands)}"


def isolate_inductor_cache(signature: str) -> str:
    """Route this process's inductor / Triton compile caches to a per-signature subdirectory of the default cache dir
    (TORCHINDUCTOR_CACHE_DIR; must run before the first torch.compile trace). Returns the directory."""
    base = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not base:
        try:
            from torch._inductor.runtime.cache_dir_utils import default_cache_dir

            base = default_cache_dir()
        except Exception:  # noqa: BLE001 - torch internals move across versions
            import getpass
            import tempfile

            base = os.path.join(tempfile.gettempdir(), f"torchinductor_{getpass.getuser()}")
    elif os.path.basename(base).startswith("anima-blocks-"):
        base = os.path.dirname(base)  # a second compile_blocks in the process: do not nest
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]
    target = os.path.join(base, f"anima-blocks-{digest}")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = target
    return target


def apply_activation_memory_budget(budget: float, gradient_checkpointing: bool) -> bool:
    """torch._functorch.config.activation_memory_budget (< 1.0: the AOT partitioner recomputes the cheapest intermediates
    in backward instead of saving them). Only for runs without gradient checkpointing: with it the recompute pass can pick
    a different graph than the forward (pytorch #166926), and checkpointing already minimises the saved set. Returns
    whether it was applied."""
    if budget >= 1.0:
        return False
    if gradient_checkpointing:
        logger.info("--anima_activation_memory_budget ignored: not applied under gradient checkpointing")
        return False
    import torch._functorch.config as functorch_config

    functorch_config.activation_memory_budget = float(budget)
    logger.info(f"block compile: torch._functorch.config.activation_memory_budget = {budget:g}")
    return True


def compile_blocks(
    anima: torch.nn.Module,
    *,
    bands: Sequence[Tuple[int, int]],
    backend: str = "inductor",
    mode: Optional[str] = None,
    dynamic: str = "union",
    recompile_limit: int = 128,
    soft_marks: Optional[bool] = None,
    skip_call: Optional[Callable[[], bool]] = None,
    isolate_cache: bool = True,
) -> CompiledBlocks:
    """torch.compile every Block's `_forward` (and `_forward_hoisted`) of `anima`; the model flattens its token grid to
    the fake-5D layout around the Block loop from now on (anima._block_compile). Call AFTER the network's apply_to and
    weight loading (the compiled graph traces the patched Linear forwards). `bands`: plan_bands() output ([]: static).
    `soft_marks`: None = soft under attn_mode "torch" (SDPA), strict otherwise. `skip_call()`: when it returns True the
    call runs the eager `_forward` (a mixed-timestep TSM chunk)."""
    if dynamic not in DYNAMIC_MODES:
        raise ValueError(f"dynamic must be one of {DYNAMIC_MODES}, got {dynamic!r}")
    bands = [tuple(int(v) for v in b) for b in bands] if dynamic != "static" else []
    if soft_marks is None:
        soft_marks = getattr(anima, "attn_mode", "torch") in (None, "torch", "sdpa")
    limit = raise_recompile_limit(recompile_limit)
    if isolate_cache and backend == "inductor":
        cache_dir = isolate_inductor_cache(compile_signature(bands, backend, mode, dynamic))
        logger.info(f"block compile: inductor cache dir {cache_dir}")
    kwargs: Dict[str, Any] = {"backend": backend, "dynamic": False}
    if mode:
        kwargs["mode"] = mode
    state = CompiledBlocks()
    state.bands = list(bands)

    def make(block: torch.nn.Module, name: str):
        orig = getattr(block, name)  # the bound class method
        compiled = torch.compile(orig, **kwargs)

        def wrapped(x, *args, **kw):
            if not (_active and block.training) or (skip_call is not None and skip_call()):
                state.eager_calls += 1
                return orig(x, *args, **kw)
            if bands and x.dim() == 5:
                _mark_dynamic(x, _rope_of(args, kw), bands, soft_marks)
            state.compiled_calls += 1
            return compiled(x, *args, **kw)

        wrapped.__name__ = f"compiled_{name}"
        wrapped.__wrapped__ = orig
        return wrapped

    for block in anima.blocks:
        for name in ("_forward", "_forward_hoisted"):
            if hasattr(block, name):
                state.originals.append((block, name, getattr(block, name)))
                setattr(block, name, make(block, name))
    anima._block_compile = True
    anima._block_compile_state = state
    logger.info(f"block compile: {len(anima.blocks)} Blocks, backend={backend}, mode={mode or 'default'}, dynamic={dynamic}"
                f" bands={bands or 'static'}, soft_marks={soft_marks}, recompile_limit={limit}")
    return state


def restore_blocks(anima: torch.nn.Module) -> None:
    state = getattr(anima, "_block_compile_state", None)
    if state is not None:
        state.restore()
    anima._block_compile = False
    anima._block_compile_state = None
