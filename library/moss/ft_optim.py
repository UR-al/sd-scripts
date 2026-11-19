# MOSS full fine-tune: the optimizer math of the host pipeline (docs/moss_full_finetune_design.md 3.1 and 3.3).
#
# HostAdamW and HostCAME are torch.optim.Optimizer subclasses only so that LR schedulers have param_groups to drive and
# a step() to see. They hold no tensors in self.state: the fp32 master and the first moment live in ft_host.HostPool,
# HostCAME's factored second moments (small) in HostCAME itself. ft_host.GradPipeline calls
#   HostAdamW.step_chunk  per piece and chunk of a parameter (AdamW is elementwise: torch's fused kernel on fp32
#                         chunks),
#   HostCAME.step_param   per parameter (CAME needs whole matrices for its row / column means),
# with the hyperparameters of a StepContext snapshot (snapshot_group), never with param_groups read by the worker.
#
# CAME cores: this fork's implementation of the paper algorithm (Luo et al. 2023, "CAME: Confidence-guided Adaptive
# Memory Efficient Optimization") in the exact operation order of the user's LoraEasyCustomOptimizer/came.py eager
# cores, so that without the correction the results are bit-identical to came.py (tests/test_moss_ft_came_bc.py),
# plus the opt-in "read-out" bias correction of design 3.3: every EMA is updated exactly as before and only divided by
# 1 - beta^t where it is read. came.py's beta3 EMA has no bias correction; with beta3 = 0.9999 the first updates are
# x32.7 the steady ones (3.3), so HostCAME refuses beta3 > 0.99 unless the correction is on.

import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch

from library.utils import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)

# lr * weight_decay below this is lost even in fp32 (relative resolution ~6e-8, design 1.5)
WD_RESOLUTION = 6e-8


def _as_bool(value, name: str) -> bool:
    """A flag from optimizer_args: sd-scripts hands over values it cannot literal_eval ("false", "on") as strings, and
    bool("false") is True."""
    if isinstance(value, str):
        flag = value.strip().lower()
        if flag not in ("true", "false", "1", "0", "yes", "no", "on", "off"):
            raise ValueError(f"{name} must be True or False, got {value!r}")
        return flag in ("true", "1", "yes", "on")
    return bool(value)


def _betas(betas, count: int, name: str) -> Tuple[float, ...]:
    values = tuple(float(b) for b in betas)
    if len(values) != count:
        raise ValueError(f"{name} needs {count} betas, got {betas!r}")
    for b in values:
        if not 0.0 <= b < 1.0:
            raise ValueError(f"{name} betas must be in [0, 1), got {betas!r}")
    return values


def _fp32(value) -> torch.Tensor:
    return torch.tensor(float(value), dtype=torch.float32)


# ----------------------------------------------------------------------------------------------------------- AdamW


class HostAdamW(torch.optim.Optimizer):
    """AdamW of `adamw_host`: param_groups (lr, betas, eps, weight_decay) for the LR scheduler, the per-chunk math for
    the pipeline. Decoupled weight decay as torch.optim.AdamW; bias-corrected; fp32 master, m, v in the HostPool.

    Default weight_decay 0.0: at the LRs of a full fine-tune lr * wd is below fp32 resolution (design 1.5)."""

    POOL_ROLES = ("master", "exp_avg", "exp_avg_sq")
    per_parameter = False

    def __init__(self, params, lr: float = 1e-5, betas: Sequence[float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        if not lr >= 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if not eps >= 0.0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        if not weight_decay >= 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        betas = _betas(betas, 2, "AdamW")
        super().__init__(params, dict(lr=lr, betas=betas, eps=float(eps), weight_decay=float(weight_decay)))
        for index, group in enumerate(self.param_groups):
            product = float(group["lr"]) * float(group["weight_decay"])
            if 0.0 < product < WD_RESOLUTION:
                logger.info(f"adamw_host group {index}: lr x weight_decay = {product:.2e} per step is below fp32 "
                            f"resolution (~{WD_RESOLUTION:.0e}); the weight decay does nothing")

    def bind(self, slots, pool) -> None:
        missing = [role for role in self.POOL_ROLES if role not in pool.roles]
        if missing:
            raise ValueError(f"HostAdamW needs the pool roles {self.POOL_ROLES}, missing {missing}")

    @staticmethod
    def snapshot_group(group: dict) -> dict:
        """One group's hyperparameters as plain Python floats (the worker never reads param_groups)."""
        beta1, beta2 = group["betas"]
        return dict(lr=float(group["lr"]), beta1=float(beta1), beta2=float(beta2), eps=float(group["eps"]),
                    weight_decay=float(group["weight_decay"]))

    @staticmethod
    def step_chunk(hp: dict, master: torch.Tensor, grad: torch.Tensor, exp_avg: torch.Tensor,
                   exp_avg_sq: torch.Tensor, step: torch.Tensor) -> None:
        """AdamW on one fp32 chunk with torch's fused kernel (the kernel of torch.optim.AdamW(fused=True)); `step` is
        the parameter's float32 step counter, already incremented for this step. Elementwise, so chunks starting at
        multiples of ft_host.ALIGN_ELEMENTS give the bits of one call on the whole tensor (design 1.4)."""
        torch._fused_adamw_([master], [grad], [exp_avg], [exp_avg_sq], [], [step], lr=hp["lr"], beta1=hp["beta1"],
                            beta2=hp["beta2"], weight_decay=hp["weight_decay"], eps=hp["eps"], amsgrad=False,
                            maximize=False)

    def step(self, closure=None):
        """Updates nothing: the pipeline stepped the parameters during the window's final micro-step. GradPipeline calls
        it once per window so an LR scheduler sees optimizer.step() before lr_scheduler.step()."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        return loss


# ------------------------------------------------------------------------------------------------------------ CAME


def came_scalars(betas: Sequence[float], eps1: float, eps2: float, clip_threshold: float, lr: float,
                 weight_decay: float, device="cpu") -> dict:
    """The scalars as came.py's eager CAME.step hands them to its cores: 0-dim float32 tensors (CAME._get_scalar_tensors
    fills them from the group) and the alphas 1 - beta formed in float32 and passed as Python floats
    (CAME._host_alphas). Keys = the keyword arguments of came_factored_core / came_unfactored_core."""
    def t(value):
        return torch.tensor(float(value), dtype=torch.float32, device=device)

    betas = tuple(float(b) for b in betas)
    return dict(beta1=t(betas[0]), beta2=t(betas[1]), beta3=t(betas[2]), eps1=t(eps1), eps2=t(eps2),
                clip_threshold=t(clip_threshold), lr=t(lr), weight_decay=t(weight_decay),
                alphas=tuple(float(_fp32(1.0) - _fp32(b)) for b in betas[:3]))


def came_bias_correction(beta1, beta2, beta3, step: int) -> Tuple[float, float, float]:
    """(1 - beta1^t, 1 - beta2^t, 1 - beta3^t) for a parameter's step count t (1-based, after its increment). The betas
    are taken at their fp32 values (the 0-dim tensors of came_scalars, or floats); the powers in float64; each result
    rounded to fp32, the value an fp32 division by it uses. Late in training every entry is exactly 1.0 and the
    corrected step is the uncorrected one bit for bit."""
    t = int(step)
    if t < 1:
        raise ValueError(f"the bias correction needs a step count >= 1, got {step}")
    return tuple(float(_fp32(1.0 - float(b) ** t)) for b in (beta1, beta2, beta3))


def _read_out(bias_correction) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Divisors of the read-out correction, None where there is nothing to divide (off, or exactly 1.0)."""
    if bias_correction is None:
        return None, None, None
    return tuple(None if float(bc) == 1.0 else float(bc) for bc in bias_correction)


def _decay_and_apply(update, grad, p_fp32, lr, weight_decay, weight_decouple, fixed_decay, cautious_weight_decay,
                     cautious) -> None:
    """The tail of came.py's cores: weight decay, LR, cautious mask, parameter update (same operations and order;
    p - update instead of p + (-update), which is the same IEEE result without the negated copy)."""
    if cautious_weight_decay:
        cwd_mask = (grad * p_fp32 >= 0).to(p_fp32.dtype)
        p_fp32.mul_(1.0 - weight_decay * lr * cwd_mask)
    elif weight_decouple:
        wd_factor = 1.0 if fixed_decay else lr
        p_fp32.mul_(1.0 - weight_decay * wd_factor)
    else:
        # came.py's non-decoupled decay: added to the gradient AFTER the update was built, so it changes only the
        # cautious mask below (kept as came.py does it; HostCAME warns about it)
        grad.add_(p_fp32, alpha=weight_decay)
    update.mul_(lr)
    if cautious:
        mask = (update * grad > 0).to(grad.dtype)
        mask.div_(mask.mean().clamp_(min=1e-3))
        update.mul_(mask)
    p_fp32.sub_(update)


@torch.no_grad()
def came_factored_core(grad: torch.Tensor, exp_avg: torch.Tensor, exp_avg_sq_row: torch.Tensor,
                       exp_avg_sq_col: torch.Tensor, exp_avg_res_row: torch.Tensor, exp_avg_res_col: torch.Tensor,
                       p_fp32: torch.Tensor, *, beta1, beta2, beta3, eps1, eps2, clip_threshold, lr, weight_decay,
                       alphas: Tuple[float, float, float], weight_decouple: bool = True, fixed_decay: bool = False,
                       cautious_weight_decay: bool = False, cautious: bool = False,
                       bias_correction: Optional[Tuple[float, float, float]] = None) -> None:
    """One CAME step of a >= 2-D parameter, in place on every state tensor and p_fp32 (all fp32, one device).

    beta1..weight_decay: 0-dim fp32 tensors and alphas = (1 - beta1, 1 - beta2, 1 - beta3) as came.py passes them
    (came_scalars). bias_correction: None = came.py's math bit for bit; (bc1, bc2, bc3) from came_bias_correction = the
    read-out correction of design 3.3: the column factors read exp_avg_sq_col / bc2 and exp_avg_res_col / bc3, the
    final update reads exp_avg / bc1; the row factors are ratios and need nothing; the residual uses the uncorrected
    exp_avg. Every EMA update is unchanged, so states are interchangeable with and without it."""
    bc1, bc2, bc3 = _read_out(bias_correction)
    update = torch.mul(grad, grad).add_(eps1)

    a1, a2, a3 = alphas
    exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=a2)
    exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=a2)

    r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
    col = exp_avg_sq_col if bc2 is None else exp_avg_sq_col / bc2
    c_factor = col.unsqueeze(-2).rsqrt()
    torch.mul(r_factor, c_factor, out=update)

    update.mul_(grad)

    rms = update.norm(2) / math.sqrt(update.numel())
    clip_factor = (rms / clip_threshold).clamp_(min=1.0)
    update.div_(clip_factor)

    exp_avg.mul_(beta1).add_(update, alpha=a1)

    res = update - exp_avg
    res.pow_(2).add_(eps2)

    exp_avg_res_row.mul_(beta3).add_(res.mean(dim=-1), alpha=a3)
    exp_avg_res_col.mul_(beta3).add_(res.mean(dim=-2), alpha=a3)
    del res

    r_factor_res = (exp_avg_res_row / exp_avg_res_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
    col_res = exp_avg_res_col if bc3 is None else exp_avg_res_col / bc3
    c_factor_res = col_res.unsqueeze(-2).rsqrt()
    torch.mul(r_factor_res, c_factor_res, out=update)
    update.mul_(exp_avg)
    if bc1 is not None:
        update.div_(bc1)

    _decay_and_apply(update, grad, p_fp32, lr, weight_decay, weight_decouple, fixed_decay, cautious_weight_decay,
                     cautious)


@torch.no_grad()
def came_unfactored_core(grad: torch.Tensor, exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor,
                         exp_avg_res: Optional[torch.Tensor], p_fp32: torch.Tensor, *, beta1, beta2, beta3, eps1, eps2,
                         clip_threshold, lr, weight_decay, alphas: Tuple[float, float, float],
                         non_factored_confidence: bool = False, weight_decouple: bool = True,
                         fixed_decay: bool = False, cautious_weight_decay: bool = False, cautious: bool = False,
                         bias_correction: Optional[Tuple[float, float, float]] = None) -> None:
    """One CAME step of a 0-D / 1-D parameter (came.py's _core_unfactored_full_fp32 without AMSBound / grams). With the
    correction: rsqrt(exp_avg_sq / bc2), the update exp_avg / bc1 and, with non_factored_confidence, the denominator
    sqrt(exp_avg_res / bc3) + eps2. exp_avg_res may be None when non_factored_confidence is off."""
    bc1, bc2, bc3 = _read_out(bias_correction)
    update = torch.mul(grad, grad).add_(eps1)

    a1, a2, a3 = alphas
    exp_avg_sq.mul_(beta2).add_(update, alpha=a2)
    torch.rsqrt(exp_avg_sq if bc2 is None else exp_avg_sq / bc2, out=update)

    update.mul_(grad)

    rms = update.norm(2) / math.sqrt(update.numel())
    clip_factor = (rms / clip_threshold).clamp_(min=1.0)
    update.div_(clip_factor)

    exp_avg.mul_(beta1).add_(update, alpha=a1)

    if non_factored_confidence:
        if exp_avg_res is None:
            raise ValueError("non_factored_confidence needs the exp_avg_res state")
        res = update.sub(exp_avg).pow_(2).add_(eps2)  # update still holds the pre-momentum value
        exp_avg_res.mul_(beta3).add_(res, alpha=a3)
        del res
        update.copy_(exp_avg)
        if bc1 is not None:
            update.div_(bc1)
        update.div_((exp_avg_res if bc3 is None else exp_avg_res / bc3).sqrt().add_(eps2))
    else:
        update.copy_(exp_avg)
        if bc1 is not None:
            update.div_(bc1)

    _decay_and_apply(update, grad, p_fp32, lr, weight_decay, weight_decouple, fixed_decay, cautious_weight_decay,
                     cautious)


# The measured first-step amplification of uncorrected CAME (design 3.3), quoted by the refusal
_BETA3_TABLE = ("beta3 0.9999: update x32.7 the steady one at step 1, x19.2 at 24, x11.5 at 71, x3.2 at 1,000; "
                "beta3 0.999: x10.6 / x6.3 / x3.8 / x1.3; with bias_correction=True (either beta3) "
                "x3.4 / x1.0 / x1.0 / x1.0, like bias-corrected AdamW")
# came.py options for its GPU / staging paths: meaningless on the host, ignored with one warning (design 3.3)
CAME_IGNORED_KEYS = ("state_storage_device", "state_storage_dtype", "compile_step", "group_shapes",
                     "group_shapes_max_numel", "foreach", "sync_chunk_size")
CAME_MAX_BETA3_UNCORRECTED = 0.99


class HostCAME(torch.optim.Optimizer):
    """CAME of `came_host`: came.py's math on the host (fp32 master and exp_avg in the HostPool, factored statistics
    here), per parameter, with the opt-in read-out bias correction (`bias_correction`, default False).

    Start-up rules (design 3.3): beta3 > 0.99 without bias_correction=True, ams_bound, kahan_sum and the update
    strategies grams / both are refused; came.py's GPU-path options are ignored with one warning; anything else unknown
    is refused."""

    POOL_ROLES = ("master", "exp_avg")
    per_parameter = True

    def __init__(self, params, lr: float = 5e-5, betas: Sequence[float] = (0.9, 0.999, 0.99),
                 weight_decay: float = 0.0, weight_decouple=True, fixed_decay=False, clip_threshold: float = 1.0,
                 eps1: float = 1e-30, eps2: float = 1e-16, update_strategy: Optional[str] = "unmodified",
                 cautious_weight_decay=False, non_factored_confidence=False, bias_correction=False,
                 ams_bound=False, kahan_sum=False, **kwargs):
        ignored = sorted(k for k in kwargs if k in CAME_IGNORED_KEYS)
        if "cautious" in kwargs:
            raise ValueError("came_host: use update_strategy='cautious' instead of came.py's deprecated `cautious`")
        unknown = sorted(k for k in kwargs if k not in CAME_IGNORED_KEYS)
        if unknown:
            raise ValueError(f"came_host: unknown optimizer_args {unknown}")
        if ignored:
            logger.warning(f"came_host: optimizer_args {ignored} are came.py GPU-path options and are ignored "
                           "(the step runs on the host in fp32)")
        if _as_bool(ams_bound, "ams_bound"):
            raise ValueError("came_host: ams_bound=True is not supported (refused, also with the bias correction)")
        if _as_bool(kahan_sum, "kahan_sum"):
            raise ValueError("came_host: kahan_sum=True is not supported (the master is fp32 already)")
        strategy = "unmodified" if update_strategy is None else str(update_strategy).strip().lower()
        if strategy not in ("unmodified", "cautious"):
            raise ValueError(f"came_host: update_strategy must be 'unmodified' or 'cautious', got {update_strategy!r} "
                             "(grams / both are refused)")
        betas = _betas(betas, 3, "CAME")
        bias_correction = _as_bool(bias_correction, "bias_correction")
        if betas[2] > CAME_MAX_BETA3_UNCORRECTED and not bias_correction:
            raise ValueError(f"came_host: beta3 = {betas[2]} > {CAME_MAX_BETA3_UNCORRECTED} needs "
                             f"bias_correction=True. came.py's beta3 EMA has no bias correction ({_BETA3_TABLE}; "
                             "docs/moss_full_finetune_design.md 3.3). Use bias_correction=True or beta3 <= 0.99.")
        for name, value in (("lr", lr), ("weight_decay", weight_decay), ("eps1", eps1), ("eps2", eps2)):
            if not float(value) >= 0.0:
                raise ValueError(f"came_host: {name} must be >= 0, got {value}")
        if not float(clip_threshold) > 0.0:
            raise ValueError(f"came_host: clip_threshold must be > 0, got {clip_threshold}")
        weight_decouple = _as_bool(weight_decouple, "weight_decouple")
        defaults = dict(lr=lr, betas=betas, weight_decay=float(weight_decay), weight_decouple=weight_decouple,
                        fixed_decay=_as_bool(fixed_decay, "fixed_decay"), clip_threshold=float(clip_threshold),
                        eps1=float(eps1), eps2=float(eps2), update_strategy=strategy,
                        cautious_weight_decay=_as_bool(cautious_weight_decay, "cautious_weight_decay"),
                        non_factored_confidence=_as_bool(non_factored_confidence, "non_factored_confidence"),
                        bias_correction=bias_correction)
        super().__init__(params, defaults)
        for index, group in enumerate(self.param_groups):  # a group may override the defaults: the same rules
            group["betas"] = _betas(group["betas"], 3, f"CAME group {index}")
            for key in ("weight_decouple", "fixed_decay", "cautious_weight_decay", "non_factored_confidence",
                        "bias_correction"):
                group[key] = _as_bool(group[key], key)
            if group["betas"][2] > CAME_MAX_BETA3_UNCORRECTED and not group["bias_correction"]:
                raise ValueError(f"came_host group {index}: beta3 = {group['betas'][2]} needs bias_correction=True "
                                 f"({_BETA3_TABLE})")
            if group["update_strategy"] not in ("unmodified", "cautious"):
                raise ValueError(f"came_host group {index}: update_strategy {group['update_strategy']!r} is refused")
            if not group["weight_decouple"] and not group["cautious_weight_decay"] and group["weight_decay"] > 0:
                logger.warning(f"came_host group {index}: weight_decouple=False - came.py adds this weight decay to "
                               "the gradient after the update is built, so it only changes the cautious mask, never "
                               "the step itself (kept as came.py does it). Use weight_decouple=True for a weight "
                               "decay.")
        self._slot_state: Dict[int, Dict[str, torch.Tensor]] = {}
        self._slot_names: Dict[int, str] = {}

    @staticmethod
    def factored(shape) -> bool:
        return len(shape) >= 2  # came.py get_options

    def bind(self, slots, pool) -> None:
        """Allocate the per-parameter statistics (fp32 zeros, came.py's shapes) for the pipeline's slots. Idempotent:
        a slot that already has its statistics (same name and shapes, e.g. loaded from a state before the pipeline was
        built) keeps them."""
        missing = [role for role in self.POOL_ROLES if role not in pool.roles]
        if missing:
            raise ValueError(f"HostCAME needs the pool roles {self.POOL_ROLES}, missing {missing}")
        groups = self.param_groups
        for slot in slots:
            shape = tuple(slot.shape)
            if self.factored(shape):
                rows, cols = shape[:-1], shape[:-2] + shape[-1:]  # came.py's _init_param_state shapes
                state = dict(exp_avg_sq_row=torch.zeros(rows), exp_avg_sq_col=torch.zeros(cols),
                             exp_avg_res_row=torch.zeros(rows), exp_avg_res_col=torch.zeros(cols))
            else:
                state = dict(exp_avg_sq=torch.zeros(shape))
                if groups[slot.group]["non_factored_confidence"]:
                    state["exp_avg_res"] = torch.zeros(shape)
            old = self._slot_state.get(slot.index)
            if (old is not None and self._slot_names.get(slot.index) == slot.name and old.keys() == state.keys()
                    and all(old[k].shape == state[k].shape for k in state)):
                continue
            self._slot_state[slot.index] = state
            self._slot_names[slot.index] = slot.name

    def slot_state(self, index: int) -> Dict[str, torch.Tensor]:
        return self._slot_state[index]

    def extra_state(self) -> Dict[str, torch.Tensor]:
        """The factored statistics by "<parameter name>.<came.py state key>" (views of the live tensors): what a
        training state saves next to the pool (came_factored.safetensors) and loads back in place."""
        return {f"{self._slot_names[i]}.{key}": tensor for i, state in self._slot_state.items()
                for key, tensor in state.items()}

    @staticmethod
    def snapshot_group(group: dict) -> dict:
        hp = came_scalars(group["betas"], group["eps1"], group["eps2"], group["clip_threshold"], group["lr"],
                          group["weight_decay"])
        hp["flags"] = dict(weight_decouple=bool(group["weight_decouple"]), fixed_decay=bool(group["fixed_decay"]),
                           cautious_weight_decay=bool(group["cautious_weight_decay"]),
                           cautious=group["update_strategy"] == "cautious")
        hp["non_factored_confidence"] = bool(group["non_factored_confidence"])
        hp["bias_correction"] = bool(group["bias_correction"])
        return hp

    def step_param(self, slot, hp: dict, master: torch.Tensor, grad: torch.Tensor, exp_avg: torch.Tensor,
                   step: torch.Tensor) -> None:
        """One CAME step of a whole parameter: flat fp32 views of its master, (accumulated, clipped) gradient and
        exp_avg; `step` already incremented for this step (t of the bias correction)."""
        shape = tuple(slot.shape)
        state = self._slot_state[slot.index]
        scalars = {k: hp[k] for k in ("beta1", "beta2", "beta3", "eps1", "eps2", "clip_threshold", "lr",
                                      "weight_decay", "alphas")}
        bc = came_bias_correction(hp["beta1"], hp["beta2"], hp["beta3"], int(step)) if hp["bias_correction"] else None
        grad, master, exp_avg = grad.view(shape), master.view(shape), exp_avg.view(shape)
        if self.factored(shape):
            came_factored_core(grad, exp_avg, state["exp_avg_sq_row"], state["exp_avg_sq_col"],
                               state["exp_avg_res_row"], state["exp_avg_res_col"], master, **scalars, **hp["flags"],
                               bias_correction=bc)
        else:
            came_unfactored_core(grad, exp_avg, state["exp_avg_sq"], state.get("exp_avg_res"), master, **scalars,
                                 **hp["flags"], non_factored_confidence=hp["non_factored_confidence"],
                                 bias_correction=bc)

    def step(self, closure=None):
        """Updates nothing (see HostAdamW.step)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        return loss
