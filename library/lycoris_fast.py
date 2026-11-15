# Faster LyCORIS LoKr + DoRA for the DiT trainers (MOSS-Image, Anima), bit-identical to LyCORIS 3.2 (site-packages
# untouched).
#
# 1) Training forward (install_fast_lokr_forward): LokrModule.forward upcasts every bf16 input activation to fp32 and copies
#    the frozen bf16 base weight to fp32, and bf16 autocast casts both straight back inside F.linear. The patched forward
#    passes the bf16 activation through and lets type promotion do the (identical) fp32 add, and skips "* scalar" (a 1.0
#    buffer without use_scalar). Same float ops in the same order otherwise: output, input grad and the lokr_w1 / lokr_w2 /
#    dora_scale grads are bit-identical (checked on CPU and on the GPU at MOSS and Anima shapes, bf16 and fp32 inputs).
#    Other configurations, and calls outside bf16/fp16 autocast, fall back to the stock forward.
# 1b) Compiled weight rebuild (enable_compiled_rebuild): the per-call kron + add + DoRA norm + scale + bf16 cast over the
#    full weight is a string of memory-bound kernels (~0.4-0.5 s of every Anima-3.8B step regardless of batch size);
#    torch.compile fuses it (forward and backward). Same float ops; the norm's reduction order may differ (~1 ulp).
# 2) Sampling (cached_lokr_weights): the sampler runs many no-grad forwards with fixed weights; build each merged
#    LoKr+DoRA weight once per sampling round instead of on every call. Bit-identical to the per-call rebuild under
#    autocast. While cached, each base weight waits in CPU RAM (or stays there with RamTorch), so the cache takes the
#    base weights' place in VRAM instead of adding to it.
# 3) Checkpointed Blocks (--anima_ckpt_nograd_merged, 2026-11-14; off by default): the no-grad first pass of
#    anima_models.nograd_checkpoint reads each merged weight from a per-module tensor built once per optimizer step (the
#    adapter only changes there; with gradient accumulation 10 that is one rebuild instead of ten), see the "checkpoint
#    merged-weight cache" section below. The grad-mode recompute keeps rebuilding (it needs the LoKr gradients).
#
# The class-level patch only lives in the process that calls install_fast_lokr_forward (moss_train_network.py,
# anima_train_network.py).

from contextlib import contextmanager
from functools import partial
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from library.elementwise_compile import ElementwiseChains
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

_STOCK_FORWARD = None


def _lokr_class():
    try:
        from lycoris.modules.lokr import LokrModule
    except ImportError:
        return None
    return LokrModule


def _compute_static_eligible(module) -> bool:
    scalar = getattr(module, "scalar", None)
    return bool(
        getattr(module, "module_type", None) == "linear"
        and not module.bypass_mode
        and module.use_w1
        and module.use_w2
        and module.wd
        and not module.tucker
        and not module.rank_dropout
        and not module.dropout
        and not module.module_dropout
        and not getattr(module, "use_orthogonal_weights", False)
        and "_forward_rebuild_core" not in module.__dict__  # network_args torch_compile=True: keep LyCORIS's compiled path
        and isinstance(scalar, torch.Tensor)
        and not isinstance(scalar, nn.Parameter)
        and scalar.numel() == 1
        and float(scalar) == 1.0  # read once per module (a CUDA buffer: .item() syncs)
    )


def _static_eligible(module) -> bool:
    """LoKr linear, full w1/w2, DoRA, rebuild mode, no dropout/orthogonal/learnable scalar. Cached per module (fixed after
    apply_to), so the per-call check does no GPU->CPU sync."""
    cached = module.__dict__.get("_moss_fast_static")
    if cached is None:
        cached = _compute_static_eligible(module)
        module.__dict__["_moss_fast_static"] = cached
    return cached


_FLOAT_INPUTS = (torch.float32, torch.bfloat16, torch.float16)


def _rebuild_weight(weight, w1, w2, lokr_scale, dora_scale, multiplier, eps, norm_dims, out_dtype):
    """The merged LoKr+DoRA weight of the fast forward, cast to the autocast dtype (what F.linear casts it to)."""
    from lycoris.functional.lokr import make_kron

    diff = make_kron(w1, w2, lokr_scale)  # == get_weight(shape) here (fp32)
    merged = weight + diff.to(dora_scale.dtype)  # bf16 + fp32 -> fp32: the same exact upcast as .to(fp32)
    norm = torch.linalg.vector_norm(merged, dim=norm_dims, keepdim=True) + eps
    scale = dora_scale / norm
    scale = multiplier * (scale - 1) + 1
    return (merged * scale).to(out_dtype)


_REBUILD = ElementwiseChains("LyCORIS LoKr+DoRA weight rebuild", rebuild=_rebuild_weight)


def enable_compiled_rebuild(enabled: bool = True) -> None:
    """torch.compile the merged-weight rebuild of the fast forward (CUDA only; eager fallback on a compile failure)."""
    _REBUILD.enable(enabled)


# ---- checkpoint merged-weight cache (--anima_ckpt_nograd_merged, 2026-11-14) -------------------------------------------
# A checkpointed Block runs its forward twice per micro-step (first pass + recompute) and the merged LoKr+DoRA weight was
# rebuilt on both (2.77B elements over Anima 2.9B's adapted Linears). With gradient accumulation the adapter only changes
# at the optimizer step, so the no-grad first pass of anima_models.nograd_checkpoint reads a per-module tensor instead:
# * built by the same rebuild function, inputs and output dtype as _fast_forward (ckpt_merged_refresh), in place into a
#   tensor that lives for the run (a compiled Block graph takes it as an input: no recompile, no new allocations);
# * rebuilt when its inputs changed, by two independent checks:
#   - ckpt_merged_invalidate(): the trainer calls it from a global optimizer-step hook (every optimizer.step() of the
#     process), after each optimizer step of its loop (weight noise and max-norm scaling run before that call) and after
#     sample images. This one is load-bearing: optimizers that write through `p.data` (LoraEasyCustomOptimizer's CAME:
#     p.data.copy_) do not bump the parameter's version counter;
#   - the key: version counter and data pointer of lokr_w1 / lokr_w2 / dora_scale / the base weight, the multiplier and
#     the output dtype. It sees every write through the parameter itself (load_state_dict, p.copy_ / p.add_, an optimizer
#     updating p in place) and a `.data = ...` swap, whoever made it. NOT seen by either: an in-place write through
#     `p.data` outside optimizer.step and outside the trainer's step - the Anima trainer has none between optimizer steps;
# * read only while _CKPT_PASS says the caller is that checkpoint: sampling, validation and un-checkpointed Blocks never
#   see it (their per-call rebuild is unchanged).
# The recompute (_CKPT_PASS == 2) still rebuilds with grad for the LoKr gradients, but hands F.linear the cached tensor's
# bits (_FirstPassBits): its activations are then the first pass's whatever rounding the two rebuilds differ by (an eager
# rebuild here, one fused into the compiled Block graph there), which --anima_ckpt_reuse_attn relies on.
# Memory: one weight in the autocast dtype per cached module, i.e. 2 bytes per weight element of the cached Linears
# (ckpt_merged_bytes reports it).
_CKPT_PASS = 0  # 1: the no-grad first pass of anima_models.nograd_checkpoint, 2: its grad-mode recompute, 0: anything else
_CKPT_GENERATION = 0  # part of every key; ckpt_merged_invalidate() bumps it


class _FirstPassBits(torch.autograd.Function):
    """forward: `cached` (the merged weight the first pass used); backward: the gradient goes to `rebuilt` unchanged, as
    if F.linear had read it. No arithmetic on the weight, so the value is `cached` bit for bit."""

    @staticmethod
    def forward(ctx, rebuilt, cached):
        return cached.view_as(cached)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


@contextmanager
def ckpt_pass(n: int):
    """_CKPT_PASS = n inside the block (anima_models.NoGradFirstPassCheckpointer: 1 around the first pass, 2 around the
    recompute). A module global: a compiled Block graph guards on it, one graph per value."""
    global _CKPT_PASS
    prev = _CKPT_PASS
    _CKPT_PASS = n
    try:
        yield
    finally:
        _CKPT_PASS = prev


def ckpt_merged_modules(model: nn.Module, linears) -> list:
    """The LoKr modules wrapping these Linears of `model` that _fast_forward can serve from a cached merged weight (the
    sample cache's rule, _cache_targets, without RamTorch-streamed bases: their weights are not in VRAM to merge from).
    Nothing while the fast forward is not installed (--anima_stock_lycoris, which the trainer refuses with the flag):
    LyCORIS's stock forward never reads the cache, building it would only cost VRAM and a rebuild per optimizer step."""
    if _STOCK_FORWARD is None:
        return []
    keep = {id(m) for m in linears}
    return [lokr for module, _, lokr in _cache_targets(model)
            if id(module) in keep and not getattr(lokr, "is_ramtorch_org", False)]


def _ckpt_key(lokr, out_dtype) -> Optional[tuple]:
    """What the cached merged weight of `lokr` was built from; None where _fast_forward would not take the fast path."""
    weight = lokr.org_module[0].weight
    if not (out_dtype in (torch.bfloat16, torch.float16) and lokr._cached_dtype == lokr.dora_scale.dtype
            and torch.promote_types(weight.dtype, lokr._cached_dtype) == lokr._cached_dtype):
        return None
    w1, w2, dora = lokr.lokr_w1, lokr.lokr_w2, lokr.dora_scale
    return (_CKPT_GENERATION, out_dtype, lokr._multiplier, w1._version, w1.data_ptr(), w2._version, w2.data_ptr(),
            dora._version, dora.data_ptr(), weight._version, weight.data_ptr())


def ckpt_merged_refresh(lokrs, out_dtype) -> tuple:
    """Make every module's cached merged weight current for its parameters as they are now (rebuilt only where the key
    changed) and return the keys. Call it right before the no-grad first pass, under that pass's autocast dtype."""
    keys = []
    for lokr in lokrs:
        d = lokr.__dict__
        key = _ckpt_key(lokr, out_dtype)
        keys.append(key)
        if key is None:  # not a fast-path module under this autocast: no cache, the forward rebuilds per call
            d.pop("_ckpt_merged", None)
            d.pop("_ckpt_key", None)
            continue
        if d.get("_ckpt_key") == key:
            continue
        with torch.no_grad():
            weight = lokr.get_org_weight_for_compute(lokr.org_module[0].weight.device).detach()
            lokr_scale = d.get("_fast_lokr_scale")
            if lokr_scale is None:
                lokr_scale = d["_fast_lokr_scale"] = float(lokr.scale)
            merged = _REBUILD.get(weight)["rebuild"](weight, lokr.lokr_w1, lokr.lokr_w2, lokr_scale, lokr.dora_scale,
                                                    lokr.multiplier_buf, lokr._dora_eps, lokr._dora_norm_dims, out_dtype)
            cached = d.get("_ckpt_merged")
            if cached is None or cached.shape != merged.shape or cached.dtype != merged.dtype or cached.device != merged.device:
                d["_ckpt_merged"] = merged  # the first build: this tensor is the cache from now on
            else:
                cached.copy_(merged)  # in place: the object a compiled Block graph reads stays the same
        d["_ckpt_key"] = key
    return tuple(keys)


def ckpt_merged_current(lokrs, keys: tuple, out_dtype) -> bool:
    """True while every cached weight still is what `keys` (ckpt_merged_refresh's result at the first pass) describes:
    neither a parameter nor the cache changed since. The recompute checks it before it trusts the cached bits."""
    return len(keys) == len(lokrs) and all(
        key == _ckpt_key(lokr, out_dtype) and (key is None or lokr.__dict__.get("_ckpt_key") == key)
        for lokr, key in zip(lokrs, keys))


def ckpt_merged_invalidate() -> None:
    """Every cached merged weight is rebuilt at its next use (the tensors stay). Called whenever the adapter may have
    changed: every optimizer.step(), the end of the trainer's optimizer step, after sample images."""
    global _CKPT_GENERATION
    _CKPT_GENERATION += 1


_CKPT_STEP_HOOK = None  # the handle of ckpt_merged_invalidate_on_optimizer_steps' hook (one per process)


def ckpt_merged_invalidate_on_optimizer_steps() -> None:
    """Every optimizer.step() of the process invalidates the checkpoint merged-weight cache: a global torch.optim hook, so
    it does not depend on where the training loop steps (the keys cannot see an optimizer that writes through p.data;
    CAME does). Registered once. moss_train_network.py uses this one; anima_train_network.py keeps its own older copy
    (invalidate_ckpt_merged_on_optimizer_steps, the same hook)."""
    global _CKPT_STEP_HOOK
    if _CKPT_STEP_HOOK is None:
        from torch.optim.optimizer import register_optimizer_step_post_hook

        _CKPT_STEP_HOOK = register_optimizer_step_post_hook(lambda optimizer, args, kwargs: ckpt_merged_invalidate())


def ckpt_merged_release(model: nn.Module) -> int:
    """Drop the cached merged weights of the LoKr modules applied to `model`; returns the bytes they held."""
    freed = 0
    for _, _, lokr in _cache_targets(model):
        cached = lokr.__dict__.pop("_ckpt_merged", None)
        lokr.__dict__.pop("_ckpt_key", None)
        if cached is not None:
            freed += cached.numel() * cached.element_size()
    return freed


def ckpt_merged_bytes(model: nn.Module) -> int:
    """Bytes held by the cached merged weights of the LoKr modules applied to `model` (what --anima_ckpt_nograd_merged
    adds to VRAM)."""
    cached = (lokr.__dict__.get("_ckpt_merged") for _, _, lokr in _cache_targets(model))
    return sum(c.numel() * c.element_size() for c in cached if c is not None)


def _fast_forward(self, x, *args, **kwargs):
    # Under autocast F.linear casts its float input to the autocast dtype, and the stock x.to(fp32) before it is exact for
    # these dtypes, so any float input takes the same path (Anima feeds fp32 LayerNorm outputs, MOSS bf16).
    device_type = x.device.type
    org = self.org_module[0]
    if not (
        _static_eligible(self)
        and torch.is_autocast_enabled(device_type)
        and torch.get_autocast_dtype(device_type) in (torch.bfloat16, torch.float16)
        and x.dtype in _FLOAT_INPUTS
        and self._cached_dtype == self.dora_scale.dtype
        and torch.promote_types(org.weight.dtype, self._cached_dtype) == self._cached_dtype  # bf16/fp16 base, fp32 net
    ):
        return _STOCK_FORWARD(self, x, *args, **kwargs)
    # --anima_ckpt_nograd_merged: inside anima_models.nograd_checkpoint a module may carry a merged weight that was built
    # for exactly these parameter values (ckpt_merged_refresh). _CKPT_PASS is 0 everywhere else: cached stays None and the
    # lines below are the forward as it was.
    cached = self.__dict__.get("_ckpt_merged") if _CKPT_PASS else None
    if cached is not None and cached.dtype != torch.get_autocast_dtype(device_type):
        cached = None  # built for another autocast dtype than this call runs under: rebuild as usual
    if cached is not None and _CKPT_PASS == 1:  # the no-grad first pass: no rebuild
        merged = cached
    else:
        weight = self.get_org_weight_for_compute(x.device).detach()  # stock uses .data (no grad into the base)
        lokr_scale = self.__dict__.get("_fast_lokr_scale")
        if lokr_scale is None:  # alpha / factor: a numpy scalar in LyCORIS; a plain float keeps torch.compile's guards simple
            lokr_scale = self.__dict__["_fast_lokr_scale"] = float(self.scale)
        rebuild = _REBUILD.get(x)["rebuild"]
        merged = rebuild(weight, self.lokr_w1, self.lokr_w2, lokr_scale, self.dora_scale, self.multiplier_buf, self._dora_eps,
                         self._dora_norm_dims, torch.get_autocast_dtype(device_type))
        if cached is not None:  # the recompute (_CKPT_PASS == 2): the first pass's weight bits, this rebuild's gradients
            merged = _FirstPassBits.apply(merged, cached)
    bias = self.get_org_bias_for_compute(x.device)  # moves a RamTorch CPU bias like the stock forward
    if bias is not None:
        bias = bias.to(x.dtype)
    return F.linear(x, merged, bias)  # the weight is already in the autocast dtype (the cast F.linear used to do)


def install_fast_lokr_forward() -> bool:
    """Patch LokrModule.forward for this process. Call before the network is created (apply_to binds forward)."""
    global _STOCK_FORWARD
    cls = _lokr_class()
    if cls is None:
        return False
    if _STOCK_FORWARD is None:
        _STOCK_FORWARD = cls.forward
        cls.forward = _fast_forward
        logger.info("fast LyCORIS LoKr+DoRA forward installed (bit-identical, no fp32 activation round trip)")
    return True


def _cached_linear(weight, bias, x, *args, **kwargs):
    return F.linear(x, weight, None if bias is None else bias.to(x.dtype))


def _cache_targets(model: nn.Module):
    cls = _lokr_class()
    if cls is None:
        return []
    targets = []
    for module in model.modules():
        forward = module.__dict__.get("forward")
        # the bound method LyCORIS installs, or the closure --anima_block_compile re-binds it to (lycoris_algo_fixes.
        # rebind_forwards: __wrapped_lora__ names the module; a closure must not carry __self__, dynamo refuses to inline it)
        lokr = getattr(forward, "__wrapped_lora__", None) or getattr(forward, "__self__", None)
        if not isinstance(lokr, cls) or lokr.org_module[0] is not module or not _static_eligible(lokr):
            continue
        if isinstance(module, nn.Linear) or getattr(lokr, "is_ramtorch_org", False):  # plain or RamTorch-streamed Linear
            targets.append((module, forward, lokr))
    return targets


def _free_bytes(device: torch.device) -> int:
    free, _ = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    from library.cuda_memory import cap_bytes  # --cuda_memory_fraction: the allocator stops at the cap, not at free VRAM

    cap = cap_bytes(device)
    if cap is not None:
        free = min(free, max(0, cap - reserved))
    return free + reserved - torch.cuda.memory_allocated(device)


def _restore(patched, parked):
    for module, forward in patched:  # drops the merged weights first ...
        module.forward = forward
    for module, device in parked:  # ... then brings the base weights back
        module.weight.data = module.weight.data.to(device)


@contextmanager
def cached_lokr_weights(model: nn.Module, dtype: torch.dtype, device: Optional[torch.device] = None,
                        park_base: bool = True, reserve_bytes: int = 4 << 30):
    """Within the block, every eligible LoKr-wrapped Linear of ``model`` uses a merged weight built once (no grad).

    device: where the merged weights live (the sampling device); defaults to the first adapted weight's device. RamTorch
    Linears (base weights in pinned CPU RAM) are streamed once into their merge instead of on every call.
    park_base: while a module's merged weight is cached its frozen base weight waits in CPU RAM (nothing reads it then),
    so the cache replaces the base weights in VRAM instead of doubling them; restored exactly on exit (~1 s of copies).
    Skipped (per-call rebuild, same results) when the needed VRAM plus ``reserve_bytes`` for the sampler's activations is
    not free, or when building runs out of memory."""
    targets = _cache_targets(model)
    patched, parked = [], []
    try:
        if targets:
            device = torch.device(device) if device is not None else targets[0][0].weight.device
            item = torch.empty((), dtype=dtype).element_size()
            streamed = [m.weight.numel() for m, _, lokr in targets if getattr(lokr, "is_ramtorch_org", False)]
            resident = [m.weight.numel() for m, _, lokr in targets if not getattr(lokr, "is_ramtorch_org", False)]
            need = sum(streamed) * item  # their base weights are not in VRAM: the merged copies are all extra
            if resident:
                if park_base and device.type == "cuda":
                    need += max(resident) * (item + 3 * 4)  # one module's merge temporaries
                else:
                    need += sum(resident) * item
            if device.type == "cuda" and _free_bytes(device) < need + reserve_bytes:
                logger.info(f"samples: merged-weight cache skipped ({need / 2**30:.1f} GiB needed, "
                            f"{_free_bytes(device) / 2**30:.1f} GiB free); rebuilding per call")
                targets = []
            try:
                with torch.no_grad():
                    for module, forward, lokr in targets:
                        if getattr(lokr, "is_ramtorch_org", False):
                            # streamed into a reused ring buffer: copy out now (the .to() below) and clone the bias
                            weight = lokr.get_org_weight_for_compute(device).data.to(lokr._cached_dtype)
                            bias = lokr.get_org_bias_for_compute(device)
                            bias = None if bias is None else bias.clone()
                        else:
                            weight = module.weight.data.to(lokr._cached_dtype)
                            bias = module.bias
                        diff = lokr.get_weight(lokr.shape).to(lokr._cached_dtype) * lokr.scalar
                        # a wider base keeps its width: MOSS's fp32 text Linears run outside autocast (--moss_stream_fp32)
                        merged = lokr.apply_weight_decompose(weight + diff, lokr.multiplier_buf).to(
                            torch.promote_types(dtype, module.weight.dtype))
                        del weight, diff
                        module.forward = partial(_cached_linear, merged, bias)
                        patched.append((module, forward))
                        if park_base and isinstance(module, nn.Linear) and module.weight.device.type == "cuda":
                            parked.append((module, module.weight.device))
                            module.weight.data = module.weight.data.to("cpu")
            except torch.OutOfMemoryError:
                _restore(patched, parked)
                patched, parked = [], []
                torch.cuda.empty_cache()
                logger.warning("samples: out of memory while caching merged LoKr weights; rebuilding per call")
        if patched:
            logger.info(f"samples: {len(patched)} merged LoKr weights cached for this round"
                        + (" (base weights parked in RAM)" if parked else ""))
        yield len(patched)
    finally:
        _restore(patched, parked)
