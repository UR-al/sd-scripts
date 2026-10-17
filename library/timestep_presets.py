# Named sigmoid timestep presets and a low-noise "scale_resume" stage for the MOSS-Image and Anima trainers.
#
# ``--sigmoid_scale`` accepts a number (sigma = sigmoid(scale * N(0,1)), unchanged) or one of three presets that follow
# ai-toolkit's "Timestep Type: Sigmoid" + "Timestep Bias": a sorted table of sigmoid(N(0,1)) indexed by u (Balanced),
# u^3 from the high-noise end (High Noise) or u^3 from the low-noise end (Low Noise). The same distributions as
# quantiles: sigma = sigmoid(ndtri(p)) with
#   balanced: p = u          sigma < 0.3: 20%, 0.3..0.7: 60%, > 0.7: 20%   (= sigmoid_scale 1.0)
#   content:  p = 1 - u^3    7% / 35% / 58%   high noise: composition, shapes, colour layout
#   style:    p = u^3        59% / 34% / 7%   low noise: line work, texture, rendering detail
# ``--scale_resume N`` trains N more epochs after max_train_epochs with the "style" preset (also on --resume of a run
# that already finished its epochs). train_network reports the epoch through set_epoch().

import argparse
from typing import Dict, Optional, Union

import torch

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

PRESETS = ("balanced", "content", "style")
RESUME_PRESET = "style"
_EPS = 1e-7


def parse_sigmoid_scale(value) -> Union[float, str]:
    """A number or a preset name (case-insensitive). Used as the argparse type; TOML values arrive unconverted."""
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError(f"sigmoid_scale must be a number or one of {PRESETS}, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in PRESETS:
        return text
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"sigmoid_scale must be a number or one of {PRESETS}, got {value!r}") from None


def add_timestep_preset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scale_resume", type=int, default=0,
                        help="after max_train_epochs, train this many more epochs with the low-noise 'style' sigmoid preset"
                             " (needs timestep_sampling = sigmoid and max_train_epochs; works with --resume of a finished run)")


SIGMOID_SCALE_HELP = ("sigmoid / shift timestep sampling: a number (sigmoid(scale * N(0,1))) or, for timestep_sampling ="
                      " sigmoid, a preset: balanced (= 1.0), content (high noise), style (low noise)")


def resolve(args) -> None:
    """Turn a preset name into args._sigmoid_preset (numeric scale 1.0) and apply --scale_resume to max_train_epochs.

    Idempotent: a second call (measurement harnesses call the runtime setup again) changes nothing."""
    if hasattr(args, "_sigmoid_preset"):
        return
    try:
        value = parse_sigmoid_scale(getattr(args, "sigmoid_scale", 1.0))
    except argparse.ArgumentTypeError as e:
        raise ValueError(str(e)) from None
    sampling = getattr(args, "timestep_sampling", None)
    preset = value if isinstance(value, str) else None
    if preset is not None and sampling != "sigmoid":
        raise ValueError(f"sigmoid_scale = \"{preset}\" needs timestep_sampling = \"sigmoid\", got {sampling!r}")
    extra = getattr(args, "scale_resume", 0) or 0
    if isinstance(extra, bool) or int(extra) != extra or extra < 0:
        raise ValueError(f"scale_resume must be a whole number >= 0, got {extra!r}")
    extra = int(extra)
    if extra and sampling != "sigmoid":
        raise ValueError(f"scale_resume needs timestep_sampling = \"sigmoid\", got {sampling!r}")
    if extra and getattr(args, "max_train_epochs", None) is None:
        raise ValueError("scale_resume counts epochs: set max_train_epochs (not only max_train_steps)")

    args.sigmoid_scale = 1.0 if preset is not None else value
    args._sigmoid_preset = preset
    args._scale_resume_from = None
    args._timestep_epoch = 0
    if extra:
        args._scale_resume_from = int(args.max_train_epochs)
        args.max_train_epochs = int(args.max_train_epochs) + extra


def set_epoch(args, epoch: int) -> None:
    """Called by train_network at the start of every epoch (1-based)."""
    args._timestep_epoch = int(epoch)
    start = getattr(args, "_scale_resume_from", None)
    if start is not None and epoch > start and not getattr(args, "_scale_resume_logged", False):
        args._scale_resume_logged = True
        logger.info(f"scale_resume: epoch {epoch} is past max_train_epochs {start}; sampling timesteps with the"
                    f" low-noise '{RESUME_PRESET}' sigmoid preset until epoch {args.max_train_epochs}")


def active_sigmoid_preset(args) -> Optional[str]:
    """The preset the sigmoid sampler uses now, or None for the numeric sigmoid_scale."""
    start = getattr(args, "_scale_resume_from", None)
    if start is not None and getattr(args, "_timestep_epoch", 0) > start:
        return RESUME_PRESET
    return getattr(args, "_sigmoid_preset", None)


def preset_sigmas(preset: str, u: torch.Tensor) -> torch.Tensor:
    """Map base uniforms u in [0,1] to sigmas of the preset (float32, same device)."""
    if preset not in PRESETS:
        raise ValueError(f"unknown sigmoid preset {preset!r}")
    u = u.to(torch.float64).clamp(0.0, 1.0)
    if preset == "content":
        p = 1.0 - u**3
    elif preset == "style":
        p = u**3
    else:
        p = u
    return torch.sigmoid(torch.special.ndtri(p.clamp(_EPS, 1.0 - _EPS))).float()


def describe(args) -> str:
    preset = getattr(args, "_sigmoid_preset", None)
    text = f"sigmoid_scale={preset}" if preset else f"sigmoid_scale={getattr(args, 'sigmoid_scale', 1.0)}"
    start = getattr(args, "_scale_resume_from", None)
    if start is not None:
        text += (f", scale_resume: epochs {start + 1}..{args.max_train_epochs} use the '{RESUME_PRESET}' preset"
                 f" (max_train_epochs {start} -> {args.max_train_epochs})")
    return text


def metadata(args) -> Dict[str, str]:
    preset = getattr(args, "_sigmoid_preset", None)
    start = getattr(args, "_scale_resume_from", None)
    extra = 0 if start is None else int(args.max_train_epochs) - start
    out = {"ss_sigmoid_scale": str(preset or getattr(args, "sigmoid_scale", 1.0))}
    if preset:
        out["ss_sigmoid_preset"] = preset
    if extra:
        out["ss_scale_resume"] = str(extra)
        out["ss_scale_resume_from_epoch"] = str(start + 1)
    return out
