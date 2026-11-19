# MOSS-Image full fine-tune: dynamic loss scaling for fp16 compute (docs/moss_ft_te_live_fp16_plan.md 2.2, phase 1b).
#
#   make(args)                       a DynamicLossScaler for mixed_precision "fp16", else None
#   DynamicLossScaler.scale          multiply the micro-step loss by it before backward() (a power of two: exact)
#   DynamicLossScaler.inv_scale      StepContext.inv_grad_scale: the host step folds it into the clip coefficient
#   DynamicLossScaler.update(stats, loss_finite)   after the window: halve on an overflow window (not when the
#                                    loss itself was non-finite: a forward overflow S cannot cause), double after
#                                    growth_interval clean windows in a row (0 = never grow)
#   DynamicLossScaler.check(max_skipped_steps)     stop a run that keeps overflowing where the scale cannot help
#   state_dict() / load_state_dict() the training state's "loss_scaler" entry (train_state.json)
#
# The scale is always a power of two, so scaling the loss and unscaling the gradient on the host never round: the
# scaled run's updates equal the unscaled run's, bit for bit, whenever no fp16 gradient overflowed (tested on the CPU).
# torch's GradScaler is not used: the optimizer steps inside the backward on the host (ft_host.GradPipeline), and the
# per-slot non-finite check of the pipeline is the overflow detector.

from __future__ import annotations

import math
from typing import Optional

from library.utils import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)

PREFIX = "MOSS full FT: "
DEFAULT_INIT_SCALE = 2.0 ** 15
DEFAULT_GROWTH_INTERVAL = 2000
DEFAULT_MIN_SCALE = 1.0
MAX_SCALE = 2.0 ** 24
GROW, BACKOFF, KEEP = "grow", "backoff", "keep"


def is_power_of_two(value) -> bool:
    """A finite positive float that is an exact power of two (2^-1074 .. 2^1023)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(value) or value <= 0.0:
        return False
    mantissa, _ = math.frexp(value)
    return mantissa == 0.5


class DynamicLossScaler:
    """Dynamic loss scale S for fp16 compute with the in-backward host step.

    update(): a window with an overflow (WindowStats.skipped_params > 0: a slot's gradient was not finite) and a finite
    loss halves S (not below min_scale) and restarts the clean-window count; growth_interval clean windows in a row
    double S (not above max_scale); growth_interval 0 never grows S (an overflow still halves it). A window whose loss
    is not finite keeps S: the loss is formed before it is multiplied by S, so a non-finite loss is a forward overflow
    (fp16 activations) that no scale causes or cures - halving would only push the gradients towards underflow and
    take growth_interval clean windows per lost factor of 2 to undo. Such a window counts in overflow_streak.
    check(): raises after max_skipped_steps consecutive overflow windows that the scale cannot explain - a non-finite
    forward loss, or an overflow at min_scale."""

    def __init__(self, init_scale: float = DEFAULT_INIT_SCALE, growth_interval: int = DEFAULT_GROWTH_INTERVAL,
                 min_scale: float = DEFAULT_MIN_SCALE, max_scale: float = MAX_SCALE):
        for name, value in (("init_scale", init_scale), ("min_scale", min_scale), ("max_scale", max_scale)):
            if not is_power_of_two(value):
                raise ValueError(f"{PREFIX}loss scaler: {name} must be a power of two, got {value!r}")
        if not float(min_scale) <= float(init_scale) <= float(max_scale):
            raise ValueError(f"{PREFIX}loss scaler: need min_scale <= init_scale <= max_scale, got {min_scale!r},"
                             f" {init_scale!r}, {max_scale!r}")
        if isinstance(growth_interval, bool) or not isinstance(growth_interval, int) or growth_interval < 0:
            raise ValueError(f"{PREFIX}loss scaler: growth_interval must be an integer >= 0, got {growth_interval!r}")
        self.init_scale = float(init_scale)
        self.growth_interval = int(growth_interval)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.scale = self.init_scale
        self.good_windows = 0  # clean windows since the last overflow or growth
        self.overflow_windows = 0  # every overflow window of the run (a statistic)
        self.overflow_streak = 0  # consecutive overflow windows the scale cannot explain (check)

    @property
    def inv_scale(self) -> float:
        return 1.0 / self.scale  # exact: a power of two

    def update(self, stats, loss_finite: bool = True) -> str:
        """After a window's drain (stats: its ft_host.WindowStats, before logging). Returns "grow", "backoff" or
        "keep". The window's scale is the one before this call."""
        overflow = int(getattr(stats, "skipped_params", 0)) > 0
        if not loss_finite:  # a forward overflow: independent of S (the loss is scaled after it is formed)
            if overflow:
                self.overflow_windows += 1
            self.overflow_streak += 1
            return KEEP  # the clean-window count neither advances nor restarts
        if overflow:
            self.overflow_windows += 1
            self.overflow_streak = self.overflow_streak + 1 if self.scale <= self.min_scale else 0
            self.good_windows = 0
            if self.scale > self.min_scale:
                self.scale = max(self.min_scale, self.scale * 0.5)
                return BACKOFF
            return KEEP
        self.overflow_streak = 0
        if self.growth_interval <= 0:
            return KEEP
        self.good_windows += 1
        if self.good_windows >= self.growth_interval:
            self.good_windows = 0
            if self.scale < self.max_scale:
                self.scale = min(self.max_scale, self.scale * 2.0)
                return GROW
        return KEEP

    def check(self, max_skipped_steps: int) -> None:
        """--moss_ft_max_skipped_steps with fp16 (replaces GradPipeline.check_skip_streak): raise after that many
        consecutive overflow windows with a non-finite loss or at min_scale (0 = never). Overflow windows with a finite
        loss above min_scale are the scaler's normal search and never count."""
        limit = int(max_skipped_steps)
        if limit > 0 and self.overflow_streak >= limit:
            raise RuntimeError(
                f"non-finite gradients in {self.overflow_streak} optimizer steps in a row that the fp16 loss scale"
                f" cannot explain (a non-finite loss, or an overflow at the minimum scale {self.min_scale:g};"
                f" --moss_ft_max_skipped_steps {limit}). The run diverged: resume from an earlier state, e.g. with a"
                " lower learning rate")

    # ---- state ------------------------------------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"scale": self.scale, "good_windows": self.good_windows, "overflow_windows": self.overflow_windows,
                "overflow_streak": self.overflow_streak, "init_scale": self.init_scale,
                "growth_interval": self.growth_interval, "min_scale": self.min_scale, "max_scale": self.max_scale}

    def load_state_dict(self, state: dict) -> None:
        """The saved scale and counters; this run's constants stay (a change is logged; the scale is clamped into
        this run's [min_scale, max_scale]). One exception, the explicit reset: when --moss_ft_loss_scale (init_scale)
        differs from the saved run's, the scale restarts at this run's init_scale with fresh clean-window and streak
        counts (e.g. after a scale that collapsed towards 1 in the saved run); set it to the saved run's value to
        continue exactly."""
        scale = state.get("scale")
        if not is_power_of_two(scale):
            raise ValueError(f"{PREFIX}loss scaler state: scale {scale!r} is not a power of two")
        self.overflow_windows = int(state.get("overflow_windows", 0))
        if "init_scale" in state and float(state["init_scale"]) != self.init_scale:
            logger.warning(f"{PREFIX}--moss_ft_loss_scale changed since the state was saved ({float(state['init_scale']):g}"
                           f" -> {self.init_scale:g}): the loss scale restarts at {self.init_scale:g} (the saved scale"
                           f" was {float(scale):g}); keep {float(state['init_scale']):g} to continue from the saved scale")
            self.scale, self.good_windows, self.overflow_streak = self.init_scale, 0, 0
            return
        changed = [key for key in ("growth_interval", "min_scale", "max_scale")
                   if key in state and state[key] != getattr(self, key)]
        if changed:
            logger.warning(f"{PREFIX}the loss scaler settings changed since the state was saved ({', '.join(changed)}):"
                           f" continuing from the saved scale {float(scale):g} with this run's settings")
        self.scale = min(self.max_scale, max(self.min_scale, float(scale)))
        self.good_windows = int(state.get("good_windows", 0))
        if self.growth_interval > 0:
            self.good_windows = min(self.good_windows, self.growth_interval - 1)
        else:
            self.good_windows = 0
        self.overflow_streak = int(state.get("overflow_streak", 0))

    def __repr__(self) -> str:
        return (f"DynamicLossScaler(scale={self.scale:g}, growth_interval={self.growth_interval},"
                f" min_scale={self.min_scale:g}, good_windows={self.good_windows})")


def make(args) -> Optional[DynamicLossScaler]:
    """The run's loss scaler: mixed_precision "fp16" only (bf16 / fp32 compute need none)."""
    if getattr(args, "mixed_precision", "bf16") != "fp16":
        return None
    return DynamicLossScaler(init_scale=float(getattr(args, "moss_ft_loss_scale", DEFAULT_INIT_SCALE)),
                             growth_interval=int(getattr(args, "moss_ft_loss_scale_growth_interval",
                                                         DEFAULT_GROWTH_INTERVAL)),
                             min_scale=float(getattr(args, "moss_ft_loss_scale_min", DEFAULT_MIN_SCALE)))


def restore(scaler: Optional[DynamicLossScaler], saved: Optional[dict]) -> None:
    """Resume (plan 2.4): fp16 state -> fp16 run exact (unless --moss_ft_loss_scale changed: then it restarts at the
    new value, load_state_dict); bf16 state -> fp16 run starts at init_scale; fp16 state -> bf16 run: ignored. Each
    mismatch is a log line."""
    if scaler is None:
        if saved:
            logger.info(f"{PREFIX}the state has an fp16 loss scale ({saved.get('scale')}); this run does not scale"
                        " the loss (not fp16): ignored")
        return
    if not saved:
        logger.info(f"{PREFIX}the state has no loss scale (not an fp16 run): starting at {scaler.scale:g}")
        return
    scaler.load_state_dict(saved)
    logger.info(f"{PREFIX}loss scale restored: {scaler.scale:g} ({scaler.good_windows} clean windows,"
                f" {scaler.overflow_windows} overflow windows so far)")
