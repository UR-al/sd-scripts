# MOSS-Image full fine-tune, Carousel mode (docs/moss_ft_carousel_plan.md 3.6, 3.10): the optimizers and their state
# codecs.
#
#   make_optimizer(kind, groups, lr, kwargs)   adamw: torch.optim.AdamW(fused=True); adamw8bit: bitsandbytes AdamW8bit
#                                              (not paged, CUDA only); came: CarouselCAME
#   CarouselCAME                               ft_optim's CAME cores (came.py's math bit for bit, + the read-out bias
#                                              correction, on by default) on the device, with a per-parameter step count
#   state_to_host / state_to_device            exact moves of one tensor's optimizer state (keep policy, switches)
#   state_entries / state_from_entries         the state as named tensors of the training state file and back
#
# Every optimizer here steps only the parameters whose .grad is set (the active units and the always group): a
# parameter's state, step count and weight decay are untouched while its unit is inactive (LISA "keep" semantics).

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from library.moss.ft_optim import HostCAME, came_bias_correction, came_factored_core, came_scalars, came_unfactored_core

PREFIX = "MOSS full FT: "
SHARED_KEYS = {"adamw8bit": ("qmap1", "qmap2")}  # bitsandbytes' quantization maps: one object per optimizer
INT_KEYS = {"adamw8bit": ("step",), "came": ("step",)}  # Python ints in the state (torch AdamW's step is a tensor)
BF16_KEYS = {"came": ("exp_avg",)}  # what --moss_carousel_state_dtype bf16 stores in bf16 on the host
QMAP_NAMES = {"qmap1": "dynamic", "qmap2": "udynamic"}


class CarouselCAME(torch.optim.Optimizer):
    """CAME for the Carousel mode: the optimizer_args rules of came_host (ft_optim.HostCAME validates them: beta3 >
    0.99 needs bias_correction, ams_bound / kahan_sum / grams refused), the cores of ft_optim on the parameters' own
    device (fp32 masters), state per parameter: exp_avg, the factored statistics, and an int step (the t of the bias
    correction counts only the parameter's own updates). Parameters without .grad are skipped."""

    def __init__(self, params, lr: float = 5e-5, **kwargs):
        dummy = torch.nn.Parameter(torch.zeros(1))
        defaults = dict(HostCAME([dummy], lr=lr, **kwargs).defaults)
        super().__init__(params, defaults)

    @staticmethod
    def factored(shape) -> bool:
        return len(shape) >= 2

    def _init_state(self, p: torch.Tensor, group: dict) -> dict:
        shape = tuple(p.shape)
        state = {"step": 0, "exp_avg": torch.zeros(shape, dtype=torch.float32, device=p.device)}
        if self.factored(shape):
            rows, cols = shape[:-1], shape[:-2] + shape[-1:]
            for key, size in (("exp_avg_sq_row", rows), ("exp_avg_sq_col", cols), ("exp_avg_res_row", rows),
                              ("exp_avg_res_col", cols)):
                state[key] = torch.zeros(size, dtype=torch.float32, device=p.device)
        else:
            state["exp_avg_sq"] = torch.zeros(shape, dtype=torch.float32, device=p.device)
            if group["non_factored_confidence"]:
                state["exp_avg_res"] = torch.zeros(shape, dtype=torch.float32, device=p.device)
        return state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            scalars: Dict[torch.device, dict] = {}
            # the betas at their fp32 values as Python floats (what came_bias_correction reads from the 0-dim tensors):
            # the same divisors without a device read per parameter
            betas32 = tuple(float(torch.tensor(float(b), dtype=torch.float32)) for b in group["betas"])
            flags = dict(weight_decouple=bool(group["weight_decouple"]), fixed_decay=bool(group["fixed_decay"]),
                         cautious_weight_decay=bool(group["cautious_weight_decay"]),
                         cautious=group["update_strategy"] == "cautious")
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dtype != torch.float32 or p.grad.dtype != torch.float32:
                    raise RuntimeError(f"{PREFIX}the rotating-block CAME steps fp32 masters with fp32 gradients")
                hp = scalars.get(p.device)
                if hp is None:
                    hp = scalars[p.device] = came_scalars(group["betas"], group["eps1"], group["eps2"],
                                                          group["clip_threshold"], group["lr"], group["weight_decay"],
                                                          device=p.device)
                state = self.state[p]
                if len(state) == 0:
                    state.update(self._init_state(p, group))
                state["step"] = int(state["step"]) + 1
                bc = came_bias_correction(*betas32, state["step"]) if group["bias_correction"] else None
                core = {k: hp[k] for k in ("beta1", "beta2", "beta3", "eps1", "eps2", "clip_threshold", "lr",
                                           "weight_decay", "alphas")}
                if self.factored(tuple(p.shape)):
                    came_factored_core(p.grad, state["exp_avg"], state["exp_avg_sq_row"], state["exp_avg_sq_col"],
                                       state["exp_avg_res_row"], state["exp_avg_res_col"], p, **core, **flags,
                                       bias_correction=bc)
                else:
                    came_unfactored_core(p.grad, state["exp_avg"], state["exp_avg_sq"], state.get("exp_avg_res"), p,
                                         **core, **flags, non_factored_confidence=bool(group["non_factored_confidence"]),
                                         bias_correction=bc)
        return loss


def make_optimizer(kind: str, groups: List[dict], lr: float, kwargs: Dict[str, Any]):
    """The Carousel optimizer of `kind` over the param groups (handles of the fp32 masters)."""
    kwargs = dict(kwargs)
    if kind == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=tuple(kwargs["betas"]), eps=float(kwargs["eps"]),
                                 weight_decay=float(kwargs["weight_decay"]), fused=True)
    if kind == "adamw8bit":
        import bitsandbytes as bnb  # CUDA only: ft_args refuses adamw8bit without CUDA

        return bnb.optim.AdamW8bit(groups, lr=lr, betas=tuple(kwargs["betas"]), eps=float(kwargs["eps"]),
                                   weight_decay=float(kwargs["weight_decay"]), is_paged=False)
    if kind == "came":
        return CarouselCAME(groups, lr=lr, **kwargs)
    raise ValueError(f"{PREFIX}unknown rotating-block optimizer kind {kind!r}")


# ---- state codecs ---------------------------------------------------------------------------------------------------
def _host_copy(value: torch.Tensor, existing: Optional[torch.Tensor], dtype: Optional[torch.dtype],
               pin: bool) -> torch.Tensor:
    dtype = dtype or value.dtype
    if existing is None or existing.shape != value.shape or existing.dtype != dtype:
        existing = torch.empty(value.shape, dtype=dtype, device="cpu", pin_memory=pin)
    existing.copy_(value)  # a separate host buffer, also when `value` already is on the CPU
    return existing


def state_to_host(kind: str, state: Dict[str, Any], existing: Optional[Dict[str, Any]] = None,
                  state_dtype: str = "fp32", pin: bool = False) -> Dict[str, Any]:
    """One tensor's optimizer state as host values: tensors copied into host buffers (`existing` ones reused when the
    shape and dtype match), exact except the opt-in bf16 of BF16_KEYS; ints kept; the shared keys (bitsandbytes'
    quantization maps) left out - state_to_device puts the optimizer's own objects back."""
    out: Dict[str, Any] = {}
    existing = existing or {}
    lossy = BF16_KEYS.get(kind, ()) if state_dtype == "bf16" else ()
    shared = SHARED_KEYS.get(kind, ())
    for key, value in state.items():
        if key in shared:
            continue
        if torch.is_tensor(value):
            out[key] = _host_copy(value, existing.get(key), torch.bfloat16 if key in lossy else None, pin)
        else:
            out[key] = value
    return out


def state_to_device(kind: str, host: Dict[str, Any], optimizer, device, param: torch.Tensor) -> Dict[str, Any]:
    """The device state of one tensor from its host values: new device tensors (fp32 where the host keeps bf16), ints
    as they are, and for bitsandbytes' 8-bit states the optimizer's shared quantization maps (moved to the device as
    bitsandbytes' init_state does, so every parameter refers to the same two objects)."""
    out: Dict[str, Any] = {}
    for key, value in host.items():
        if torch.is_tensor(value):
            dtype = torch.float32 if value.dtype == torch.bfloat16 else value.dtype
            tensor = torch.empty(value.shape, dtype=dtype, device=device)
            tensor.copy_(value)
            out[key] = tensor
        else:
            out[key] = value
    if kind == "adamw8bit" and torch.is_tensor(out.get("state1")) and out["state1"].dtype == torch.uint8:
        qmaps = optimizer.name2qmap
        if "dynamic" not in qmaps:
            optimizer.fill_qmap()
        for key, name in QMAP_NAMES.items():
            qmaps[name] = qmaps[name].to(param.device)
            out[key] = qmaps[name]
    return out


def state_entries(kind: str, state: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """[(key, value)] of one tensor's state for the state file: tensors as they are (a producer reads them at write
    time), ints as int64 0-dim tensors; the shared keys left out."""
    out = []
    shared = SHARED_KEYS.get(kind, ())
    for key, value in state.items():
        if key in shared:
            continue
        if torch.is_tensor(value):
            out.append((key, value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append((key, torch.tensor(int(value), dtype=torch.int64)))
        else:
            raise ValueError(f"{PREFIX}optimizer state key {key} holds {type(value).__name__}: not saveable")
    return out


def state_from_entries(kind: str, values: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """The host state of one tensor from the state file's values (INT_KEYS back to Python ints)."""
    ints = INT_KEYS.get(kind, ())
    return {key: (int(value.reshape(-1)[0]) if key in ints else value) for key, value in values.items()}


def state_numel_bytes(state: Dict[str, Any]) -> int:
    return sum(v.numel() * v.element_size() for v in state.values() if torch.is_tensor(v))
