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
    weight = self.get_org_weight_for_compute(x.device).detach()  # stock uses .data (no grad into the base)
    lokr_scale = self.__dict__.get("_fast_lokr_scale")
    if lokr_scale is None:  # alpha / factor: a numpy scalar in LyCORIS; a plain float keeps torch.compile's guards simple
        lokr_scale = self.__dict__["_fast_lokr_scale"] = float(self.scale)
    rebuild = _REBUILD.get(x)["rebuild"]
    merged = rebuild(weight, self.lokr_w1, self.lokr_w2, lokr_scale, self.dora_scale, self.multiplier_buf, self._dora_eps,
                     self._dora_norm_dims, torch.get_autocast_dtype(device_type))
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
        lokr = getattr(forward, "__self__", None)
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
                        merged = lokr.apply_weight_decompose(weight + diff, lokr.multiplier_buf).to(dtype)
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
