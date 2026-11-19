# MOSS-Image full fine-tune (moss_train.py): CLI arguments, start-up validation, optimizer mode and the two-phase LR.
#
# Design: docs/moss_full_finetune_design.md 4.2 (parser composition), 4.5 (arguments), 4.6 (two-phase LR); the budget
# checks of 4.5 live in ft_budget.py. Implementation notes: docs/moss_ft_impl_notes.md (T1).
#
#   setup_parser()                                  the 4.2 parser with moss_train.py's defaults
#   validate_moss_ft_args(args)                     start-up step 2 (design 4.3): everything that needs no dataset
#   validate_moss_ft_args(args, train_group, ...)   step 4: steps per epoch, LR phase boundaries, budget checks
#   optimizer_mode(args)                            "adamw_host" | "adafactor_fused" | "came_host" | ft_budget.CAROUSEL
#   carousel_kind(args) / carousel_arg(args, key)   the Carousel mode's optimizer kind and flags (built from CAROUSEL)
#   parse_optimizer_args(args, mode)                that mode's optimizer_args with defaults (unknown keys refused)
#   parse_lr_phases / lr_phase_bounds / make_lr_phase_scheduler      --moss_ft_lr_phases
#   make_lr_scheduler(args, optimizer, steps_per_epoch)               the run's scheduler (phases or get_scheduler_fix)
#   check_lr_scheduler(args, steps_per_epoch)                         step 4: it builds and its state is resumable
#   val_enabled(args) / parse_val_sigmas(value)     --moss_ft_val_* (the validation itself: ft_val.py)
#
# Every refusal is a ValueError whose message starts with "MOSS full FT: " and gives the reason.

from __future__ import annotations

import argparse
import ast
import math
import os
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from library.utils import setup_logging

from . import captions as moss_captions
from . import ft_budget
from . import ft_scaler
from .te_live_weights import HOST_MODES as TE_HOST_MODES

setup_logging()
import logging

logger = logging.getLogger(__name__)

PREFIX = "MOSS full FT: "
ADAMW_HOST, ADAFACTOR_FUSED, CAME_HOST = ft_budget.ADAMW_HOST, ft_budget.ADAFACTOR_FUSED, ft_budget.CAME_HOST
CAROUSEL = ft_budget.CAROUSEL  # the rotating-blocks mode (docs/moss_ft_carousel_plan.md); its name lives in ft_budget
MODES = (ADAMW_HOST, ADAFACTOR_FUSED, CAME_HOST, CAROUSEL)
CLIP_MODES = ("stale", "exact")
PHASE_SCHEDULERS = ("constant", "constant_with_warmup")
SAVE_PRECISIONS = ("bf16", "fp16", "float")
MEMORY_CACHE_IMAGE_LIMIT = 2000  # above this many images both *_to_disk cache flags are required (design 2.4)
KOHYA_LEARNING_RATE = 2e-6  # train_util.add_optimizer_arguments' default, used when nothing sets the LR
KOHYA_MAX_TRAIN_STEPS = 1600  # train_util.add_training_arguments' default
WD_RESOLUTION = 6e-8  # lr x weight_decay below this is lost even in an fp32 master (design 1.5)
CAME_BETA3_LIMIT = 0.99  # above it CAME needs bias_correction=True (design 3.3)
TE_LIVE_DEFAULTS = {"moss_te_live_ahead": 32, "moss_te_live_ring_slots": 2, "moss_te_live_host": "copy",
                    "moss_te_live_threads": 8, "moss_te_live_batch": 1}
TE_RING_SLOTS = (2, 4)
OVERFLOW_POLICIES = ("partial", "window")
LOSS_SCALE_DEFAULTS = {"moss_ft_loss_scale": ft_scaler.DEFAULT_INIT_SCALE,
                       "moss_ft_loss_scale_growth_interval": ft_scaler.DEFAULT_GROWTH_INTERVAL,
                       "moss_ft_loss_scale_min": ft_scaler.DEFAULT_MIN_SCALE, "moss_ft_overflow_policy": "partial"}
# validation loss (library/moss/ft_val.py): 8 noise levels at the midpoints of [0, 1] cut into 8 equal bins - the mean
# over them is the midpoint-rule estimate of the loss averaged over uniform sigma; each is k/16, exact in fp32 and x1000
VAL_DEFAULT_SIGMAS = (0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375)
VAL_DEFAULTS = {"moss_ft_val_every_n_steps": 0, "moss_ft_val_sigmas": None, "moss_ft_val_seed": 1234}
VAL_MAX_SIGMAS = 64
VAL_LOG_FILE = "val_log.jsonl"

OPTIMIZER_TABLE = (
    "optimizer_type -> mode (design 4.5):\n"
    '  "" or AdamW (any case)                     -> adamw_host (default: fp32 master, exp_avg, exp_avg_sq in host RAM,'
    " AdamW on the CPU inside the backward)\n"
    "  Adafactor + --fused_backward_pass           -> adafactor_fused (GA 1, stochastic rounding, no host state)\n"
    "  LoraEasyCustomOptimizer.came.CAME or CAME   -> came_host (CAME on the host; bias_correction=True for beta3 > 0.99)\n"
    f"  with --moss_{CAROUSEL}: AdamW -> adamw (fused, GPU), AdamW8bit -> adamw8bit (bitsandbytes, not paged), CAME ->"
    " came (GPU, bias_correction on)\n"
    "  anything else (Prodigy, schedule-free, ...) is not supported")
CAROUSEL_KIND_TABLE = (
    f"optimizer_type with --moss_{CAROUSEL} (plan 3.6):\n"
    '  "" or AdamW (any case)                         -> adamw (torch AdamW fused=True on the GPU, fp32 states)\n'
    "  AdamW8bit / bitsandbytes.optim.AdamW8bit / --use_8bit_adam -> adamw8bit (bitsandbytes, CUDA, not paged)\n"
    "  CAME / LoraEasyCustomOptimizer.came.CAME      -> came (ft_optim's CAME cores on the GPU, bias_correction on)\n"
    "  anything else (paged 8-bit, Adafactor, Lion, Prodigy, ...) is not supported")
CAROUSEL_DEFAULTS = {"active": None, "vram_gib": None, "period": ft_budget.CAROUSEL_DEFAULT_PERIOD, "select": "random",
                     "seed": None, "always": "small", "state": "keep", "state_dtype": "fp32", "host_pinned": "none"}
# GradPipeline flags the Carousel mode does not use (no host worker, no rings): an info line when changed
CAROUSEL_UNUSED_DEFAULTS = {"moss_ft_host_threads": 16, "moss_ft_piece_mib": 64, "moss_ft_ring_mib": 1024,
                            "moss_ft_upload_ring_mib": 512}

CAME_AMPLIFICATION = (
    "RMS update over its steady value, measured (design 3.3): CAME beta3 0.9999: t1 x32.7, t24 x19.2, t71 x11.5, t1000"
    " x3.2; beta3 0.999: t1 x10.6, t24 x6.3, t71 x3.8, t1000 x1.3; with bias_correction=True (any beta3), like torch"
    " AdamW: t1 x3.4, t24 x1.0. The Anima notes: beta3 0.9999 broke a full fine-tune in 24 steps")

STREAM_PRECISION_NOTE = (
    "the user's 11-16 precision run measured LoKr gradients against fp32: bf16 with a bf16 stream 82 % error, cos 0.58"
    " (blocks 0-13: cos 0.11); fp16 with an fp32 stream 18.6 %, cos 0.989 (design 1.5)")


def _refuse(message: str):
    raise ValueError(PREFIX + message)


# ---- the parser (4.2, 4.5) ------------------------------------------------------------------------------------------
def add_moss_ft_arguments(parser) -> None:
    """The new arguments of design 4.5."""
    parser.add_argument("--moss_ft_resume_new_lr_schedule", action="store_true",
                        help="on resume, start the current CLI LR schedule from its beginning instead of loading the"
                             " saved LR/scheduler state; keep optimizer state, global step and dataset progress")
    parser.add_argument("--moss_ft_lr_phases", type=lr_phases_arg, default=None,
                        help='LR phases "E1:LR1,E2:LR2[,...]" (epochs and LR of each phase, e.g. "2:5e-6,6:1e-6";'
                             " fractional epochs allowed). Phase i ends after round(cumsum(E)_i x steps per epoch)"
                             " optimizer steps; --lr_warmup_steps warms phase 1 up and must end inside it. Sets"
                             " max_train_epochs (when the sum is whole) and learning_rate (to LR1); not with"
                             " --max_train_steps; lr_scheduler must be constant or constant_with_warmup")
    parser.add_argument("--moss_ft_freeze", type=freeze_arg, default="",
                        help="comma-separated regexes (re.fullmatch on parameter names such as layers.22.proj.q.weight)"
                             " of parameters kept frozen; a TOML array takes regexes with commas. A pattern that"
                             " matches nothing, or freezing everything, is refused. Default: all 5.08B trained")
    parser.add_argument("--moss_ft_clip_mode", choices=CLIP_MODES, default="stale",
                        help="--max_grad_norm clipping: stale (default) scales by the previous window's gradient norm so"
                             " the step can run inside the backward (the first window runs exact); exact uses the"
                             " window's own norm and steps after the backward (needs the 18.93 GiB host accumulator"
                             " even at GA 1)")
    parser.add_argument("--moss_ft_host_threads", type=int, default=16,
                        help="intra-op threads of the host optimizer worker, 1 .. cpu count (the host step is DRAM-bound"
                             " above ~16 on the 285K, design 1.4)")
    parser.add_argument("--moss_ft_piece_mib", type=int, default=64,
                        help="gradient / upload piece size in MiB, a power of two in [16, 256]")
    parser.add_argument("--moss_ft_ring_mib", type=int, default=1024,
                        help="pinned device-to-host gradient ring in MiB: a power of two (torch's pinned allocator"
                             " rounds up to one), at least 4 pieces")
    parser.add_argument("--moss_ft_upload_ring_mib", type=int, default=512,
                        help="pinned host-to-device weight upload ring in MiB: a power of two, at least 2 pieces")
    parser.add_argument("--moss_ft_max_skipped_steps", type=int, default=8,
                        help="stop (RuntimeError, after logging) after this many consecutive optimizer steps that"
                             " skipped a parameter for non-finite gradients; 0 = never stop")
    parser.add_argument("--moss_ft_state_dir", type=str, default=None,
                        help="directory of the training states (61 GB each with adamw_host); default output_dir")
    parser.add_argument("--moss_ft_no_keep_phase_states", action="store_true",
                        help="also delete the epoch state at the end of each LR phase but the last (kept by default so"
                             " a later phase can be re-run from it)")
    parser.add_argument("--moss_ft_allow_bf16_stream", action="store_true",
                        help="allow bf16 compute without --moss_stream_fp32 (e.g. a heal-type model); Base's residual"
                             " stream needs fp32 (design 1.5)")
    parser.add_argument("--moss_ft_skip_resource_check", action="store_true",
                        help="do not refuse to start on the host-RAM and disk checks (the estimates are still logged)")
    parser.add_argument("--moss_ft_log_every", type=int, default=10,
                        help="optimizer steps between the pipeline timing / memory log lines (default 10)")
    # the live text encoder (docs/moss_ft_te_live_fp16_plan.md 1.7)
    parser.add_argument("--moss_te_live", action="store_true",
                        help="encode the captions at train time with the frozen Qwen3.5-9B streamed from host RAM (no TE"
                             " cache): shuffle_caption / keep_tokens(_separator), multi-line captions with"
                             " enable_wildcard, caption_dropout_rate and caption_tag_dropout_rate draw anew at every"
                             " fetch. Not with --cache_text_encoder_outputs(_to_disk) / --cache_caption_variants")
    parser.add_argument("--moss_te_live_ahead", type=int, default=TE_LIVE_DEFAULTS["moss_te_live_ahead"],
                        help="micro-batches whose captions one encoder pass encodes (rounded up to a multiple of"
                             " gradient_accumulation_steps; default 32): one weight transfer serves them all")
    parser.add_argument("--moss_te_live_ring_slots", type=int, default=TE_LIVE_DEFAULTS["moss_te_live_ring_slots"],
                        help="device / staging slots of the layer ring: 2 (default, ~0.8 GiB VRAM for the 9B) or 4")
    parser.add_argument("--moss_te_live_host", choices=TE_HOST_MODES, default=TE_LIVE_DEFAULTS["moss_te_live_host"],
                        help="the encoder's decoder layers in host RAM: copy (default, 12.9 GiB private + 1 GiB pinned"
                             " staging), pinned (12.9 GiB page-locked, no staging; CUDA) or mmap (no private copy; reads"
                             " the file through the OS file cache, which then wants ~13 GB)")
    parser.add_argument("--moss_te_live_threads", type=int, default=TE_LIVE_DEFAULTS["moss_te_live_threads"],
                        help="intra-op threads of the stager (memcpy of a layer into the pinned staging entry), 1 .."
                             " cpu count (default 8)")
    parser.add_argument("--moss_te_live_batch", type=int, default=TE_LIVE_DEFAULTS["moss_te_live_batch"],
                        help="captions per batched layer call of an encoder pass (default 1: each caption runs alone and"
                             " unpadded, bit for bit the cached path). N > 1: the pass's captions are sorted by length,"
                             " cut into groups of at most N, right-padded to the group's longest and run through each"
                             " layer once per group (far fewer kernel launches; close to, not bit-identical with, the"
                             " cached encoding). A call holds at most N x 300 padded tokens, the call the VRAM estimate"
                             " charges: captions past 300 tokens run in smaller groups of their own")
    # fp16 compute with dynamic loss scaling (docs/moss_ft_te_live_fp16_plan.md 2.5); only with mixed_precision fp16
    parser.add_argument("--moss_ft_loss_scale", type=float, default=LOSS_SCALE_DEFAULTS["moss_ft_loss_scale"],
                        help='fp16: initial loss scale, a power of two in [1, 2^24] (default 32768); mixed_precision'
                             ' = "fp16" only. A resume continues the saved scale; a value different from the saved'
                             ' run (its init scale) restarts the scale at this value (the explicit reset)')
    parser.add_argument("--moss_ft_loss_scale_growth_interval", type=int,
                        default=LOSS_SCALE_DEFAULTS["moss_ft_loss_scale_growth_interval"],
                        help="fp16: optimizer steps without an overflow before the loss scale doubles (default 2000);"
                             " 0 = never grows (an overflow still halves it)")
    parser.add_argument("--moss_ft_loss_scale_min", type=float, default=LOSS_SCALE_DEFAULTS["moss_ft_loss_scale_min"],
                        help="fp16: the lowest loss scale, a power of two (default 1); overflows at it count toward"
                             " --moss_ft_max_skipped_steps")
    parser.add_argument("--moss_ft_overflow_policy", choices=OVERFLOW_POLICIES,
                        default=LOSS_SCALE_DEFAULTS["moss_ft_overflow_policy"],
                        help="fp16: a step in which some gradient overflowed: partial (default) steps the parameters"
                             " whose gradient is finite and skips the others (the step runs inside the backward;"
                             " with --moss_ft_clip_mode exact every step runs after the backward and the whole step"
                             " is skipped, since its true norm is not finite); window skips the whole step (torch"
                             " GradScaler; needs --moss_ft_clip_mode exact with max_grad_norm > 0)")
    # deterministic validation loss on a held-out set (library/moss/ft_val.py)
    parser.add_argument("--moss_ft_val_dataset_config", type=str, default=None,
                        help="a kohya dataset TOML of the held-out images: the validation loss is computed on them at"
                             " fixed noise levels with fixed noise (no validation without it). Latents cached like the"
                             " training set's; the captions without any random draw (no shuffle / dropout; a"
                             " multi-line caption gives its first line); one line per validation in"
                             f" <output_dir>/{VAL_LOG_FILE}, TensorBoard val/loss and val/loss_s<sigma>")
    parser.add_argument("--moss_ft_val_every_n_steps", type=int, default=VAL_DEFAULTS["moss_ft_val_every_n_steps"],
                        help="validate every N optimizer steps; 0 (default) = only before the first step (step 0) and"
                             " after the last. Before the first and after the last step it always runs")
    parser.add_argument("--moss_ft_val_sigmas", type=val_sigmas_arg, default=VAL_DEFAULTS["moss_ft_val_sigmas"],
                        help='the fixed flow-matching noise levels, comma-separated in (0, 1] (default "'
                             + ",".join(f"{s:g}" for s in VAL_DEFAULT_SIGMAS) + '": the midpoints of 8 equal bins, so'
                             " the mean is the loss averaged over uniform sigma)")
    parser.add_argument("--moss_ft_val_seed", type=int, default=VAL_DEFAULTS["moss_ft_val_seed"],
                        help="seed of the validation noise (default 1234): the noise of an image at a sigma depends only"
                             " on it, the image's file name and the sigma; the training RNG is never used")
    add_carousel_arguments(parser)


def carousel_dest(key: str = "") -> str:
    """The argparse dest of a Carousel flag: moss_<CAROUSEL> / moss_<CAROUSEL>_<key> (ft_budget.CAROUSEL is the one
    place the name lives)."""
    return f"moss_{CAROUSEL}" + (f"_{key}" if key else "")


def carousel_flag(key: str = "") -> str:
    return "--" + carousel_dest(key)


def carousel_arg(args, key: str = "", default=None):
    return getattr(args, carousel_dest(key), CAROUSEL_DEFAULTS.get(key, default) if key else default)


def carousel_enabled(args) -> bool:
    return bool(getattr(args, carousel_dest(), False))


def add_carousel_arguments(parser) -> None:
    """The Carousel flags (plan 4.2), every name built from ft_budget.CAROUSEL."""
    from . import ft_carousel_units as cu

    name = CAROUSEL
    parser.add_argument(carousel_flag(), action="store_true",
                        help=f"{name}: full fine-tune with rotating blocks (LISA / BAdam style): only K of the 44 units"
                             " (2 text refiner + 42 joint blocks; 45 with --moss_" + name + "_always rotate_stem) are"
                             " trainable at a time, with fp32 master, gradient"
                             f" accumulator and optimizer state on the GPU; every --moss_{name}_period steps K new units"
                             " are chosen. optimizer_type AdamW (default), AdamW8bit or CAME")
    parser.add_argument(carousel_flag("active"), type=int, default=None,
                        help=f"{name}: active units K, 1 .. 44 (45 with rotate_stem; all = no switching). Default: as"
                             " many as"
                             f" --moss_{name}_vram_gib allows")
    parser.add_argument(carousel_flag("vram_gib"), type=float, default=None,
                        help=f"{name}: device GiB the run may use when K comes from the budget (default 30.5, the"
                             " warning line); not with --moss_" + name + "_active")
    parser.add_argument(carousel_flag("period"), type=int, default=CAROUSEL_DEFAULTS["period"],
                        help=f"{name}: optimizer steps between switches of the active set (default 50)")
    parser.add_argument(carousel_flag("select"), choices=cu.SELECTIONS, default=CAROUSEL_DEFAULTS["select"],
                        help=f"{name}: how the active set is chosen: random (default; every unit once per cycle, no"
                             " repeats), round_robin (fixed sets spread over the depth) or window (K contiguous units)")
    parser.add_argument(carousel_flag("seed"), type=int, default=None,
                        help=f"{name}: seed of the random selection (default --seed, 0 without it); the training RNG"
                             " is not used")
    parser.add_argument(carousel_flag("always"), choices=cu.ALWAYS_POLICIES, default=CAROUSEL_DEFAULTS["always"],
                        help=f"{name}: always-trainable small groups: small (default: embedders, time MLP, shared"
                             " modulation, final norm / linear / head modulation, 99.1M; every step then backpropagates"
                             " through all blocks), head (only the head, 19.3M: nothing below the lowest active block"
                             " is recomputed or backpropagated; the stem is never trained), rotate_stem (the head"
                             " always; the stem, 79.8M, becomes a 45th rotating unit: trained in K/45 of the steps, and"
                             " the truncation of head while it is inactive) or none")
    parser.add_argument(carousel_flag("state"), choices=cu.STATE_POLICIES, default=CAROUSEL_DEFAULTS["state"],
                        help=f"{name}: optimizer state of the inactive units: keep (default, in host RAM, exact) or"
                             " reset (LISA: a unit starts with a fresh state every time it becomes active)")
    parser.add_argument(carousel_flag("state_dtype"), choices=cu.STATE_DTYPES, default=CAROUSEL_DEFAULTS["state_dtype"],
                        help=f"{name}: host storage of kept states: fp32 (default, exact) or bf16 (CAME's exp_avg only,"
                             " lossy, halves its host RAM)")
    parser.add_argument(carousel_flag("host_pinned"), choices=cu.HOST_PINNED, default=CAROUSEL_DEFAULTS["host_pinned"],
                        help=f"{name}: none (default, pageable copies ~8.5 GB/s) or masters (page-lock the 18.9 GiB"
                             " master pool for faster switches; refused past 40 %% of physical RAM in pinned total)")


def _set_help(parser, dest: str, text: str) -> None:
    for action in parser._actions:  # argparse has no public setter; config_validation reads _actions too
        if action.dest == dest:
            action.help = text


def setup_parser() -> argparse.ArgumentParser:
    """moss_train.py's parser: the composition of design 4.2 (checked to build without argparse conflicts) +
    add_moss_ft_arguments, with this script's defaults: mixed_precision and save_precision bf16, cfm_lambda 0.02 (as the
    MOSS LoRA trainer), and learning_rate / max_train_steps None, so --moss_ft_lr_phases can tell "not given" from
    kohya's defaults (validate_moss_ft_args fills those in: 2e-6 / 1600)."""
    from library import config_util, deepspeed_utils, sai_model_spec, train_util
    from library.custom_train_functions import add_custom_train_arguments
    from library.utils import add_logging_arguments

    from . import train_utils as moss_train_utils

    parser = argparse.ArgumentParser(description="MOSS-Image full fine-tune (moss_train.py)")
    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)
    train_util.add_dit_training_arguments(parser)
    moss_train_utils.add_moss_training_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)
    # what anima_train.setup_parser adds itself
    parser.add_argument("--cpu_offload_checkpointing", action="store_true",
                        help="refused by moss_train.py: the gradient hooks are verified with the plain, unsloth and"
                             " no-grad-node checkpoints only; use --unsloth_offload_checkpointing")
    parser.add_argument("--unsloth_offload_checkpointing", action="store_true",
                        help="keep the checkpointed block inputs in pinned host RAM (async copies): ~12 %% slower, moves"
                             " the block-input row of the VRAM budget (8.45 GiB at 1024 batch 4) to the host, where"
                             " torch's pinned allocator keeps 42 power-of-two blocks per size class (about 18.4 GiB"
                             " there with short last batches)")
    parser.add_argument("--flow_use_ot", action="store_true",
                        help="pair latents and noise with cosine optimal transport (batch > 1)")
    parser.add_argument("--contrastive_flow_matching", action="store_true",
                        help="Contrastive Flow Matching (ΔFM) objective (batch > 1)")
    parser.add_argument("--cfm_lambda", type=float, default=0.02,
                        help="ΔFM: weight of the contrastive term (default 0.02, as the MOSS LoRA trainer)")
    add_moss_ft_arguments(parser)
    # validation_split: no argument here (validation is phase 2), but train_util.get_scheduler_fix reads it when
    # lr_scheduler_args has first_cycle_max_steps (the LoraEasyCustomOptimizer restart schedulers)
    parser.set_defaults(mixed_precision="bf16", save_precision="bf16", learning_rate=None, max_train_steps=None,
                        validation_split=0.0)
    _set_help(parser, "learning_rate", "learning rate (default 2e-6; with --moss_ft_lr_phases leave it out: LR1)")
    _set_help(parser, "max_train_steps", "training steps (default 1600 without --max_train_epochs; not with"
                                         " --moss_ft_lr_phases, whose epochs give the length)")
    return parser


# ---- argparse types -------------------------------------------------------------------------------------------------
def lr_phases_arg(value):
    """argparse type of --moss_ft_lr_phases: checks the text and keeps it as given (TOML strings are converted through
    it by config_validation; a TOML array reaches parse_lr_phases unconverted)."""
    try:
        parse_lr_phases(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    return value


def val_sigmas_arg(value):
    """argparse type of --moss_ft_val_sigmas: checks the text and keeps it as given (a TOML array reaches
    parse_val_sigmas unconverted)."""
    try:
        parse_val_sigmas(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    return value


def parse_val_sigmas(value) -> Tuple[float, ...]:
    """--moss_ft_val_sigmas -> the noise levels in the given order: None / "" -> VAL_DEFAULT_SIGMAS; a comma-separated
    string or a list of numbers, each finite in (0, 1], no repeats, at most VAL_MAX_SIGMAS."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return VAL_DEFAULT_SIGMAS
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ValueError(f"{PREFIX}--moss_ft_val_sigmas must be comma-separated numbers in (0, 1], got {value!r}")
    sigmas = []
    for part in parts:
        if isinstance(part, bool) or part == "":
            raise ValueError(f"{PREFIX}--moss_ft_val_sigmas must be comma-separated numbers in (0, 1], got {value!r}")
        try:
            sigma = float(part)
        except (TypeError, ValueError):
            raise ValueError(f"{PREFIX}--moss_ft_val_sigmas: {part!r} is not a number (comma-separated numbers in"
                             " (0, 1])") from None
        if not math.isfinite(sigma) or not 0.0 < sigma <= 1.0:
            raise ValueError(f"{PREFIX}--moss_ft_val_sigmas: {part!r} is outside (0, 1] (a flow-matching noise level:"
                             " 0 = the clean latent, 1 = pure noise)")
        if sigma in sigmas:
            raise ValueError(f"{PREFIX}--moss_ft_val_sigmas: {sigma:g} is given twice")
        sigmas.append(sigma)
    if not 1 <= len(sigmas) <= VAL_MAX_SIGMAS:
        raise ValueError(f"{PREFIX}--moss_ft_val_sigmas takes 1 to {VAL_MAX_SIGMAS} noise levels, got {len(sigmas)}")
    keys = [f"{sigma:g}" for sigma in sigmas]
    if len(set(keys)) != len(keys):  # the log keys (val/loss_s<sigma>, loss_by_sigma) would collide
        raise ValueError(f"{PREFIX}--moss_ft_val_sigmas: {value!r} has noise levels that print alike ({keys}); use"
                         " values that differ in their first 6 significant digits")
    return tuple(sigmas)


def freeze_arg(value):
    """argparse type of --moss_ft_freeze: compiles every regex and keeps the value as given."""
    try:
        _compile_freeze(split_freeze_patterns(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    return value


# ---- --moss_ft_freeze -----------------------------------------------------------------------------------------------
def _split_top_level_commas(text: str) -> List[str]:
    """Split on commas outside (), {} and [] and not escaped: "layers\\.\\d{1,2}\\..*,x" is two patterns."""
    parts, current, depth, escaped, in_class = [], [], 0, False, False
    for char in text:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif in_class:
            in_class = char != "]"
        elif char == "[":
            in_class = True
        elif char in "({":
            depth += 1
        elif char in ")}" and depth > 0:
            depth -= 1
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return parts


def split_freeze_patterns(value) -> List[str]:
    """--moss_ft_freeze as a list of regexes: a comma-separated string (commas inside groups, quantifiers and classes,
    or escaped, stay in their regex) or a TOML array."""
    if value is None:
        return []
    items = [str(item) for item in value] if isinstance(value, (list, tuple)) else _split_top_level_commas(str(value))
    return [item.strip() for item in items if item.strip()]


def _compile_freeze(patterns: Sequence[str]) -> List[Tuple[str, "re.Pattern"]]:
    compiled = []
    for pattern in patterns:
        try:
            compiled.append((pattern, re.compile(pattern)))
        except re.error as error:
            raise ValueError(f"{PREFIX}--moss_ft_freeze pattern {pattern!r} is not a regular expression: {error}") from None
    return compiled


def frozen_parameter_names(names: Iterable[str], patterns) -> List[str]:
    """The parameter names (in their order) that a --moss_ft_freeze value freezes: re.fullmatch of any pattern.
    `names` may be a mapping (its keys) such as dict(model.named_parameters()) or ft_budget.moss_param_shapes().
    ValueError when a pattern matches no name."""
    names = list(names)
    compiled = _compile_freeze(split_freeze_patterns(patterns))
    unmatched = [text for text, regex in compiled if not any(regex.fullmatch(name) for name in names)]
    if unmatched:
        more = f" (nor do {unmatched[1:]})" if len(unmatched) > 1 else ""
        _refuse(f"--moss_ft_freeze pattern {unmatched[0]!r} matches no parameter{more}: re.fullmatch on names such as"
                " 'layers.0.proj.q.weight', 'text_embedder.blocks.0.qkv.weight', 'time_embedding.0.weight'")
    return [name for name in names if any(regex.fullmatch(name) for _, regex in compiled)]


def model_sizes(args) -> ft_budget.ModelSizes:
    """ft_budget.ModelSizes of the trained set: the whole DiT, minus what --moss_ft_freeze freezes."""
    patterns = split_freeze_patterns(getattr(args, "moss_ft_freeze", ""))
    if not patterns:
        return ft_budget.ModelSizes()
    shapes = ft_budget.moss_param_shapes()
    return ft_budget.ModelSizes.from_shapes(shapes, frozen_parameter_names(shapes, patterns))


# ---- --moss_ft_lr_phases (4.6) --------------------------------------------------------------------------------------
LR_PHASES_FORMAT = '"E1:LR1,E2:LR2[,...]" (epochs > 0 and LR > 0 per phase, e.g. "2:5e-6,6:1e-6")'


def parse_lr_phases(text) -> List[Tuple[float, float]]:
    """--moss_ft_lr_phases "E1:LR1,E2:LR2[,...]" -> [(epochs, lr), ...]. Epochs and LRs are finite floats > 0; one
    phase is a constant LR with warmup. A TOML array of "E:LR" strings is the same list. ValueError otherwise."""
    if isinstance(text, (list, tuple)) and text and all(isinstance(item, str) for item in text):
        text = ",".join(text)
    if not isinstance(text, str) or not text.strip():
        _refuse(f"--moss_ft_lr_phases takes {LR_PHASES_FORMAT}, got {text!r}")
    phases = []
    for index, item in enumerate(text.split(","), start=1):
        parts = item.split(":")
        if len(parts) != 2:
            _refuse(f"--moss_ft_lr_phases phase {index} {item.strip()!r} is not EPOCHS:LR; it takes {LR_PHASES_FORMAT}")
        try:
            epochs, lr = float(parts[0]), float(parts[1])
        except ValueError:
            _refuse(f"--moss_ft_lr_phases phase {index} {item.strip()!r} is not two numbers; it takes {LR_PHASES_FORMAT}")
        for what, value in (("epochs", epochs), ("LR", lr)):
            if not (math.isfinite(value) and value > 0):
                _refuse(f"--moss_ft_lr_phases phase {index}: {what} must be a finite number > 0, got {item.strip()!r}")
        phases.append((epochs, lr))
    return phases


def _validated_phases(phases) -> List[Tuple[float, float]]:
    """parse_lr_phases' result from a string, a list of strings or a list of (epochs, lr) pairs."""
    if isinstance(phases, str) or (isinstance(phases, (list, tuple)) and phases and isinstance(phases[0], str)):
        return parse_lr_phases(phases)
    out = []
    for index, phase in enumerate(phases or (), start=1):
        epochs, lr = (float(value) for value in phase)
        if not (math.isfinite(epochs) and epochs > 0 and math.isfinite(lr) and lr > 0):
            _refuse(f"--moss_ft_lr_phases phase {index}: epochs and LR must be finite numbers > 0, got {tuple(phase)}")
        out.append((epochs, lr))
    if not out:
        _refuse(f"--moss_ft_lr_phases needs at least one phase: {LR_PHASES_FORMAT}")
    return out


def steps_per_epoch(num_batches: int, gradient_accumulation_steps: int) -> int:
    """ceil(len(train_dataloader) / GA): kohya's optimizer steps per epoch (the last window may be short)."""
    ga = int(gradient_accumulation_steps)
    if ga < 1:
        _refuse(f"gradient_accumulation_steps must be >= 1, got {gradient_accumulation_steps!r}")
    return -(-int(num_batches) // ga)


def lr_phase_bounds(phases, steps_per_epoch: int) -> List[int]:
    """b_i = round(cumsum(E)_i x steps_per_epoch) (float noise rounded away, a tie rounds up): phase i runs for the
    0-based optimizer steps [b_(i-1), b_i). The last bound is the run's length. ValueError when a phase gets no step."""
    phases = _validated_phases(phases)
    if isinstance(steps_per_epoch, bool) or int(steps_per_epoch) != steps_per_epoch or steps_per_epoch < 1:
        _refuse(f"--moss_ft_lr_phases needs at least 1 optimizer step per epoch, got {steps_per_epoch!r} steps per epoch")
    steps = int(steps_per_epoch)
    bounds = []
    for index in range(len(phases)):
        cumulative = math.fsum(epochs for epochs, _ in phases[:index + 1])
        bounds.append(int(math.floor(round(cumulative * steps, 6) + 0.5)))
    previous = 0
    for index, bound in enumerate(bounds):
        if bound <= previous:
            _refuse(f"--moss_ft_lr_phases phase {index + 1} ({phases[index][0]:g} epochs) gets no optimizer step at"
                    f" {steps} steps per epoch (boundaries {bounds}): lengthen it")
        previous = bound
    return bounds


def lr_phase_end_epochs(phases) -> List[float]:
    """The epoch (cumulative, may be fractional) at which each phase ends; the states at the ends of all phases but
    the last are the pinned phase-boundary states of design 4.7 (when the epoch is whole)."""
    phases = _validated_phases(phases)
    return [math.fsum(epochs for epochs, _ in phases[:index + 1]) for index in range(len(phases))]


def resolve_warmup_steps(value, total_steps: int) -> int:
    """--lr_warmup_steps as steps: an int (or integral float) is a step count, a float in (0, 1) a fraction of
    total_steps converted like kohya's get_scheduler_fix (train_util.fraction_to_steps)."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (
            isinstance(value, float) and not value.is_integer() and value >= 1):
        _refuse(f"lr_warmup_steps must be a step count >= 0 or a fraction of the steps in [0, 1), got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        from library.train_util import fraction_to_steps

        return fraction_to_steps(value, int(total_steps))
    return int(value)


def _phase_lambda(bounds: Sequence[int], lrs: Sequence[float], warmup_steps: int) -> Callable[[int], float]:
    bounds, lrs, warmup, last = list(bounds), list(lrs), int(warmup_steps), len(bounds) - 1

    def lr_lambda(step: int) -> float:  # step = 0-based optimizer step (design 4.6)
        phase = next((index for index, bound in enumerate(bounds) if step < bound), last)
        warm = min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0
        return warm * lrs[phase] / lrs[0]

    return lr_lambda


class LrPhaseScheduler(torch.optim.lr_scheduler.LambdaLR):
    """--moss_ft_lr_phases as a LambdaLR (design 4.6): every param group's base LR is LR1 and
    lr(s) = min(1, (s + 1) / W) x LR_phase(s) for the 0-based optimizer step s (step 0 runs at LR1 / W, not 0).

    state_dict() keeps only the step count (and, for the record, the schedule it ran with): load_state_dict()
    continues at that step with the schedule this object was built with from the current config, writes the resumed
    step's LR into the param groups at once, and warns when the steps per epoch, boundaries, LRs or warmup changed."""

    _CONFIG_KEYS = ("phases", "phase_lrs", "steps_per_epoch", "bounds", "total_steps", "warmup_steps")
    STATE_KEY = "moss_ft_lr_phases"

    def __init__(self, optimizer, phases, steps_per_epoch: int, warmup_steps=0, last_epoch: int = -1):
        self.phases = _validated_phases(phases)
        self.phase_lrs = [lr for _, lr in self.phases]
        self.steps_per_epoch = int(steps_per_epoch)
        self.bounds = lr_phase_bounds(self.phases, self.steps_per_epoch)
        self.total_steps = self.bounds[-1]
        self.warmup_steps = resolve_warmup_steps(warmup_steps, self.total_steps)
        if self.warmup_steps > self.bounds[0]:
            _refuse(f"lr_warmup_steps {warmup_steps!r} ({self.warmup_steps} steps) must end inside LR phase 1 (optimizer"
                    f" steps [0, {self.bounds[0]})): shorten the warmup or lengthen phase 1")
        for group in optimizer.param_groups:  # one LR for every group (design 4.4.1); the scheduler owns it
            group["lr"] = self.phase_lrs[0]
            group["initial_lr"] = self.phase_lrs[0]
        super().__init__(optimizer, _phase_lambda(self.bounds, self.phase_lrs, self.warmup_steps), last_epoch=last_epoch)

    def schedule(self) -> Dict[str, object]:
        return {"phases": [[epochs, lr] for epochs, lr in self.phases], "steps_per_epoch": self.steps_per_epoch,
                "bounds": list(self.bounds), "warmup_steps": self.warmup_steps}

    def phase_of(self, step: int) -> int:
        """0-based LR phase of a 0-based optimizer step."""
        return next((index for index, bound in enumerate(self.bounds) if step < bound), len(self.bounds) - 1)

    def state_dict(self):
        state = super().state_dict()
        for key in self._CONFIG_KEYS:
            state.pop(key, None)
        state[self.STATE_KEY] = self.schedule()
        return state

    def load_state_dict(self, state_dict) -> None:
        state = dict(state_dict)
        saved = state.pop(self.STATE_KEY, None)
        for key in self._CONFIG_KEYS:
            state.pop(key, None)
        super().load_state_dict(state)
        now = self.schedule()
        if saved is not None and saved != now:
            logger.warning(f"{PREFIX}the LR schedule changed since the state was saved (dataset size, GA or"
                           f" --moss_ft_lr_phases / --lr_warmup_steps): saved {saved}, now {now}. Continuing at optimizer"
                           f" step {self.last_epoch} with the new schedule")
        base = self.phase_lrs[0]
        self.base_lrs = [base for _ in self.optimizer.param_groups]
        values = [base * fn(self.last_epoch) for fn in self.lr_lambdas]
        for group, value in zip(self.optimizer.param_groups, values):
            group["initial_lr"] = base
            group["lr"] = value
        self._last_lr = values


def make_lr_phase_scheduler(optimizer, phases, steps_per_epoch: int, warmup_steps=0) -> LrPhaseScheduler:
    """The --moss_ft_lr_phases scheduler (a torch.optim.lr_scheduler.LambdaLR). phases: parse_lr_phases' list, the
    option's text, or (epochs, lr) pairs; steps_per_epoch = ceil(len(train_dataloader) / GA); warmup_steps: the
    --lr_warmup_steps value (steps, or a fraction of the run). ValueError when the warmup does not end inside phase 1."""
    return LrPhaseScheduler(optimizer, phases, steps_per_epoch, warmup_steps)


def make_lr_scheduler(args, optimizer, steps_per_epoch: int, num_processes: int = 1):
    """moss_train.py's LR scheduler: the --moss_ft_lr_phases scheduler, else train_util.get_scheduler_fix (any kohya
    scheduler, lr_scheduler_type included; design 4.5). Validated args: max_train_steps is the run's length."""
    if getattr(args, "moss_ft_lr_phases", None):
        return make_lr_phase_scheduler(optimizer, args.moss_ft_lr_phases, steps_per_epoch, args.lr_warmup_steps)
    from library import train_util

    return train_util.get_scheduler_fix(args, optimizer, num_processes)


def param_group_values(optimizer) -> List[dict]:
    """Every param group's values but its parameters: what a training state keeps of the groups (adamw_host /
    came_host: param_groups.pt), loaded back with torch.load(weights_only=True)."""
    return [{key: value for key, value in group.items() if key != "params"} for group in optimizer.param_groups]


def check_scheduler_resumable(lr_scheduler, optimizer) -> None:
    """A training state stores lr_scheduler.state_dict() and the param groups (ft_state) and reads them back with
    torch.load(weights_only=True): plain values and tensors only. A scheduler whose state holds objects (kohya's
    ZeroLRWarmupScheduler keeps its inner scheduler, which holds the optimizer and every parameter) could be saved
    but never resumed: refused before the run starts."""
    import io

    buffer = io.BytesIO()
    try:  # a state that cannot even be pickled (a local lambda inside an inner scheduler) fails at the save already
        torch.save({"scheduler": lr_scheduler.state_dict(), "param_groups": param_group_values(optimizer)}, buffer)
        buffer.seek(0)
        torch.load(buffer, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - pickle refusals have several types
        first = re.sub(r"\x1b\[[0-9;]*m", "", (str(error).strip().splitlines() or [""])[0])  # torch colours the text
        _refuse(f"the LR scheduler {type(lr_scheduler).__name__} cannot be resumed from a training state: its state"
                " does not round-trip through torch.save / torch.load(weights_only=True)"
                f" ({type(error).__name__}: {first[:120]}). Use another scheduler (lr_scheduler / lr_scheduler_type) or"
                " --moss_ft_lr_phases")


def check_lr_scheduler(args, steps_per_epoch: int) -> None:
    """Start-up step 4 (once max_train_steps is known, before the caches and the DiT load): build the run's scheduler
    on a stand-in optimizer with the two param groups and step it twice, so a setting get_scheduler_fix refuses (a
    warmup for "constant", a custom scheduler's arguments) fails now instead of after hours of caching, and check that
    its state is resumable (check_scheduler_resumable)."""
    import logging as _logging

    params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(2)]
    optimizer = torch.optim.SGD([{"params": [params[0]], "name": "decay"}, {"params": [params[1]], "name": "no_decay"}],
                                lr=float(args.learning_rate))
    train_util_logger = _logging.getLogger("library.train_util")
    level = train_util_logger.level
    train_util_logger.setLevel(_logging.WARNING)  # its [SCHEDULER] info lines come again when the run builds it
    try:
        try:
            lr_scheduler = make_lr_scheduler(args, optimizer, steps_per_epoch, 1)
            for _ in range(2):
                optimizer.step()
                lr_scheduler.step()
        except ValueError as error:
            if str(error).startswith(PREFIX):
                raise
            _refuse(_scheduler_problem(args, error))
        except Exception as error:  # noqa: BLE001 - a custom scheduler can raise anything
            _refuse(_scheduler_problem(args, error))
    finally:
        train_util_logger.setLevel(level)
    check_scheduler_resumable(lr_scheduler, optimizer)


def _scheduler_problem(args, error) -> str:
    what = getattr(args, "lr_scheduler_type", None) or getattr(args, "lr_scheduler", None)
    text = f"the LR scheduler {what!r} cannot be built with these settings ({type(error).__name__}: {error})"
    if "num_warmup_steps" in str(error):
        text += ("; lr_warmup_steps needs a scheduler with a warmup (constant_with_warmup, linear, cosine, ...) - a custom"
                 " lr_scheduler_type takes its warmup in lr_scheduler_args")
    return text


# ---- optimizer_type -> mode, optimizer_args (4.5) -------------------------------------------------------------------
def carousel_kind(args) -> str:
    """The Carousel optimizer kind of optimizer_type (plan 3.6): adamw, adamw8bit or came; ValueError otherwise."""
    if getattr(args, "fused_backward_pass", False):
        _refuse(f"--fused_backward_pass is not used with --moss_{CAROUSEL}: the active units step on the GPU after the"
                " backward. Remove it. " + CAROUSEL_KIND_TABLE)
    if getattr(args, "use_lion_optimizer", False):
        _refuse(f"--use_lion_optimizer selects Lion, which --moss_{CAROUSEL} does not support. " + CAROUSEL_KIND_TABLE)
    name = str(getattr(args, "optimizer_type", "") or "").strip()
    lower = name.lower()
    if "paged" in lower:
        _refuse(f"optimizer_type {name!r}: paged optimizers are refused with --moss_{CAROUSEL} (unified memory would"
                " fight the explicit switches of the states). Use AdamW8bit. " + CAROUSEL_KIND_TABLE)
    if getattr(args, "use_8bit_adam", False):
        if lower not in ("", "adamw", "adamw8bit", "bitsandbytes.optim.adamw8bit"):
            _refuse(f"--use_8bit_adam with optimizer_type {name!r}: choose one. " + CAROUSEL_KIND_TABLE)
        return "adamw8bit"
    if lower in ("", "adamw"):
        return "adamw"
    if lower in ("adamw8bit", "bitsandbytes.optim.adamw8bit"):
        return "adamw8bit"
    if lower in ("came", "loraeasycustomoptimizer.came.came"):
        return "came"
    _refuse(f"optimizer_type {name!r} is not supported with --moss_{CAROUSEL}. " + CAROUSEL_KIND_TABLE)


def optimizer_mode(args) -> str:
    """The mode of design 4.5's table; ValueError (quoting the table) for anything else."""
    if carousel_enabled(args):
        carousel_kind(args)
        return CAROUSEL
    for flag, name in (("use_8bit_adam", "AdamW8bit"), ("use_lion_optimizer", "Lion")):
        if getattr(args, flag, False):
            _refuse(f"--{flag} selects {name}, which moss_train.py does not support. " + OPTIMIZER_TABLE)
    name = str(getattr(args, "optimizer_type", "") or "").strip()
    lower = name.lower()
    fused = bool(getattr(args, "fused_backward_pass", False))
    if lower in ("", "adamw"):
        mode = ADAMW_HOST
    elif lower == "adafactor":
        if not fused:
            _refuse("optimizer_type Adafactor needs --fused_backward_pass (mode adafactor_fused: the hook steps every"
                    " parameter in the backward). " + OPTIMIZER_TABLE)
        return ADAFACTOR_FUSED
    elif lower in ("came", "loraeasycustomoptimizer.came.came"):
        mode = CAME_HOST
    else:
        _refuse(f"optimizer_type {name!r} is not supported. " + OPTIMIZER_TABLE)
    if fused:
        _refuse(f"--fused_backward_pass selects adafactor_fused (with optimizer_type Adafactor); {mode} steps every"
                " parameter inside the backward on the host already. Remove --fused_backward_pass")
    return mode


def _optimizer_arg_items(args) -> List[Tuple[str, object]]:
    """optimizer_args "key=value" entries, values through ast.literal_eval like train_util.get_optimizer (a value that
    is not a literal stays a string: update_strategy=cautious)."""
    entries = getattr(args, "optimizer_args", None) or []
    if isinstance(entries, str):
        entries = [entries]
    items, seen = [], set()
    for entry in entries:
        text = str(entry)
        key, separator, raw = text.partition("=")
        key = key.strip()
        if not separator or not key:
            _refuse(f"optimizer_args entry {text!r} is not key=value")
        if key in seen:
            _refuse(f"optimizer_args {key} is given twice")
        seen.add(key)
        raw = raw.strip()
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        items.append((key, value))
    return items


def _number(key: str, value, *, low: Optional[float] = None, low_open: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (
            low is not None and (value <= low if low_open else value < low)):
        bound = "" if low is None else f" {'>' if low_open else '>='} {low:g}"
        _refuse(f"optimizer_args {key}={value!r}: needs a finite number{bound}")
    return float(value)


def _flag(key: str, value) -> bool:
    if not isinstance(value, bool):
        _refuse(f"optimizer_args {key}={value!r}: needs True or False")
    return value


def _betas(key: str, value, count: int) -> Tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != count or any(
            isinstance(beta, bool) or not isinstance(beta, (int, float)) or not 0.0 <= beta < 1.0 for beta in value):
        _refuse(f"optimizer_args {key}={value!r}: needs {count} numbers in [0, 1)")
    return tuple(float(beta) for beta in value)


def _non_negative(key, value):
    return _number(key, value, low=0.0)


def _positive(key, value):
    return _number(key, value, low=0.0, low_open=True)


ADAMW_ARGS = {  # key: (default, check)
    "betas": ((0.9, 0.999), lambda key, value: _betas(key, value, 2)),
    "eps": (1e-8, _positive),
    "weight_decay": (0.0, _non_negative),
}
CAME_ARGS = {
    "betas": ((0.9, 0.999, 0.99), lambda key, value: _betas(key, value, 3)),
    "bias_correction": (False, _flag),
    "weight_decay": (0.0, _non_negative),
    "weight_decouple": (True, _flag),
    "fixed_decay": (False, _flag),
    "cautious_weight_decay": (False, _flag),
    "eps1": (1e-30, _non_negative),
    "eps2": (1e-16, _non_negative),
    "clip_threshold": (1.0, _positive),
    "update_strategy": ("unmodified", None),
    "non_factored_confidence": (False, _flag),
}
CAME_GPU_ONLY_ARGS = ("state_storage_device", "state_storage_dtype", "compile_step", "group_shapes",
                      "group_shapes_max_numel", "foreach", "sync_chunk_size")
CAME_FALSE_ONLY_ARGS = ("ams_bound", "kahan_sum")
CAME_UPDATE_STRATEGIES = ("unmodified", "cautious")
ADAFACTOR_REQUIRED_FALSE = ("scale_parameter", "relative_step", "warmup_init")
ADAFACTOR_ARGS = {
    "eps": lambda key, value: tuple(_non_negative(key, item) for item in _pair(key, value)),
    "clip_threshold": _positive,
    "decay_rate": lambda key, value: _number(key, value),
    "weight_decay": _non_negative,
}


def _pair(key, value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        _refuse(f"optimizer_args {key}={value!r}: needs two numbers")
    return value


def parse_optimizer_args(args, mode: str, quiet: bool = False) -> Dict[str, object]:
    """That mode's optimizer_args (design 4.5) with every default filled in; unknown keys refused.

    adamw_host: betas (0.9, 0.999), eps 1e-8, weight_decay 0.0.
    came_host: betas (0.9, 0.999, 0.99), bias_correction False, weight_decay 0.0, weight_decouple True, fixed_decay
      False, cautious_weight_decay False, eps1 1e-30, eps2 1e-16, clip_threshold 1.0, update_strategy "unmodified"
      (or "cautious"), non_factored_confidence False: the keyword arguments of ft_optim.HostCAME. Its GPU-path tuning
      keys are ignored with one warning (not with quiet=True); ams_bound / kahan_sum are refused unless False, came.py's
      deprecated `cautious` is refused (update_strategy=cautious); beta3 > 0.99 needs bias_correction=True.
    adafactor_fused: scale_parameter / relative_step / warmup_init False (filled in when missing), no beta1 (None
      accepted), plus the given eps / clip_threshold / decay_rate / weight_decay: the kwargs for
      train_util.get_optimizer(args, params, optimizer_kwargs=...)."""
    items = _optimizer_arg_items(args)
    if mode == ADAMW_HOST:
        return _with_defaults(mode, items, ADAMW_ARGS)
    if mode == CAME_HOST:
        return _came_args(items, quiet)
    if mode == CAROUSEL:
        kind = carousel_kind(args)
        if kind == "came":
            out = _came_args(items, quiet, mode_name=f"{CAROUSEL} came", default_bias_correction=True)
            if not out["bias_correction"] and carousel_arg(args, "state") == "reset":
                _refuse(f"{CAROUSEL} came with {carousel_flag('state')} reset needs bias_correction=True: reset starts"
                        " every incoming unit at t = 1, so without the correction the early-update amplification"
                        f" repeats at every switch, not once per run. {CAME_AMPLIFICATION}")
            return out
        for key, _ in items:
            if key in ("is_paged", "paged", "optim_bits", "percentile_clipping", "block_wise", "min_8bit_size"):
                _refuse(f"optimizer_args {key} is not supported by {CAROUSEL} {kind} (bitsandbytes' defaults; paged"
                        " optimizers are refused)")
        return _with_defaults(f"{CAROUSEL} {kind}", items, ADAMW_ARGS)
    if mode == ADAFACTOR_FUSED:
        return _adafactor_args(items)
    raise ValueError(f"{PREFIX}unknown optimizer mode {mode!r} (one of {MODES})")


def _unknown(mode: str, key: str, known: Iterable[str]):
    _refuse(f"optimizer_args {key} is not used by {mode} (known keys: {', '.join(known)})")


def _with_defaults(mode: str, items, spec) -> Dict[str, object]:
    out = {key: default for key, (default, _) in spec.items()}
    for key, value in items:
        if key not in spec:
            _unknown(mode, key, spec)
        check = spec[key][1]
        out[key] = check(key, value) if check else value
    return out


def _came_args(items, quiet: bool, mode_name: str = CAME_HOST, default_bias_correction: bool = False
               ) -> Dict[str, object]:
    kept, ignored = [], []
    for key, value in items:
        if key in CAME_GPU_ONLY_ARGS:
            ignored.append(key)
        elif key in CAME_FALSE_ONLY_ARGS:
            if value is not False:
                _refuse(f"optimizer_args {key}={value!r} is not supported by {mode_name}"
                        + (" (with the bias correction it is undefined, design 3.3)" if key == "ams_bound" else
                           " (the master is fp32: there is no low-precision write-back to compensate)"))
        elif key == "cautious":  # came.py's deprecated switch (ft_optim.HostCAME refuses it too)
            _refuse("optimizer_args cautious is came.py's deprecated switch: use update_strategy=cautious")
        else:
            kept.append((key, value))
    given = {key for key, _ in kept}
    out = _with_defaults(mode_name, kept, CAME_ARGS)
    if "bias_correction" not in given:
        out["bias_correction"] = bool(default_bias_correction)
    strategy = "unmodified" if out["update_strategy"] is None else out["update_strategy"]
    if strategy not in CAME_UPDATE_STRATEGIES:
        _refuse(f"optimizer_args update_strategy={strategy!r} is not supported by {mode_name} (unmodified or cautious;"
                " grams / both are not ported)")
    out["update_strategy"] = strategy
    if ignored and not quiet:
        logger.warning(f"{PREFIX}{mode_name}: optimizer_args {', '.join(ignored)} are ignored (GPU-path tuning of"
                       " came.py; this mode runs ft_optim's CAME cores, design 3.3)")
    beta3 = out["betas"][2]
    if default_bias_correction and not out["bias_correction"] and not quiet:
        logger.warning(f"{PREFIX}{mode_name}: bias_correction=False: the EMAs are read uncorrected and the first"
                       f" updates of every unit are amplified ({CAME_AMPLIFICATION})")
    if beta3 > CAME_BETA3_LIMIT and not out["bias_correction"]:
        _refuse(f"{mode_name} with beta3 {beta3:g} needs bias_correction=True in optimizer_args (or beta3 <="
                f" {CAME_BETA3_LIMIT:g}): without it the EMAs are read uncorrected and the early updates are amplified."
                f" {CAME_AMPLIFICATION}")
    return out


def _adafactor_args(items) -> Dict[str, object]:
    out: Dict[str, object] = {key: False for key in ADAFACTOR_REQUIRED_FALSE}
    known = (*ADAFACTOR_REQUIRED_FALSE, "beta1", *ADAFACTOR_ARGS)
    for key, value in items:
        if key in ADAFACTOR_REQUIRED_FALSE:
            if value is not False:
                _refuse(f"optimizer_args {key}={value!r}: adafactor_fused needs {key}=False (design 3.2: a fixed LR"
                        " from --learning_rate / --moss_ft_lr_phases, no relative-step schedule)")
        elif key == "beta1":
            if value is not None:
                _refuse(f"optimizer_args beta1={value!r}: adafactor_fused needs beta1 unset (a beta1 allocates a full"
                        " fp32 first moment on the GPU, design 3.2)")
            out["beta1"] = None
        elif key in ADAFACTOR_ARGS:
            out[key] = ADAFACTOR_ARGS[key](key, value)
        else:
            _unknown(ADAFACTOR_FUSED, key, known)
    return out


# ---- the validation (4.5) -------------------------------------------------------------------------------------------
LORA_ONLY_ARGS = {
    "moss_ckpt_nograd_merged": "it caches merged LoKr weights",
    "moss_lora_exclude_blocks": "it leaves blocks out of the network; --moss_ft_freeze 'layers\\.22\\..*' keeps a"
                                " block's weights fixed here",
    "moss_stock_lycoris": "LyCORIS's stock forward",
    "moss_no_lokr_compile": "the LoKr+DoRA weight rebuild",
    "moss_no_sample_cache": "the merged LoKr weight cache of sample images",
}


def check_single_process(num_processes) -> None:
    """Design 4.3 step 5: refuse num_processes > 1 (call it with accelerator.num_processes)."""
    if int(num_processes) > 1:
        _refuse(f"{int(num_processes)} processes: multi-GPU / multi-process training is not supported (the host pools,"
                " the pinned rings and the in-backward step serve one GPU in one process). Launch one process")


def _int_option(args, name: str, low: int, high: Optional[int], what: str) -> int:
    value = getattr(args, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
        bounds = f"in [{low}, {high}]" if high is not None else f">= {low}"
        _refuse(f"--{name} must be an integer {bounds} ({what}), got {value!r}")
    return value


def _check_new_arguments(args) -> None:
    cpus = os.cpu_count() or 1
    _int_option(args, "moss_ft_host_threads", 1, cpus, "intra-op threads of the host worker")
    piece = getattr(args, "moss_ft_piece_mib", None)
    if isinstance(piece, bool) or not isinstance(piece, int) or not 16 <= piece <= 256 or piece & (piece - 1):
        _refuse(f"--moss_ft_piece_mib must be a power of two in [16, 256], got {piece!r}")
    for name, pieces in (("moss_ft_ring_mib", 4), ("moss_ft_upload_ring_mib", 2)):
        value = getattr(args, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value % piece or value < pieces * piece:
            _refuse(f"--{name} must be a multiple of --moss_ft_piece_mib ({piece}) and at least {pieces} pieces"
                    f" ({pieces * piece} MiB), got {value!r}")
        if value & (value - 1):  # torch's pinned host allocator rounds every allocation up to a power of two
            _refuse(f"--{name} must be a power of two (e.g. {1 << (value.bit_length() - 1)} or {1 << value.bit_length()}"
                    f" MiB): the pinned ring is one allocation, which torch's pinned host allocator rounds up to"
                    f" {1 << value.bit_length()} MiB - the rest would be page-locked RAM the budget does not count"
                    " (design 2.2)")
    _int_option(args, "moss_ft_max_skipped_steps", 0, None, "0 = never stop")
    _int_option(args, "moss_ft_log_every", 1, None, "optimizer steps between pipeline log lines")
    if getattr(args, "moss_ft_clip_mode", None) not in CLIP_MODES:
        _refuse(f"--moss_ft_clip_mode must be one of {CLIP_MODES}, got {getattr(args, 'moss_ft_clip_mode', None)!r}")
    state_dir = getattr(args, "moss_ft_state_dir", None)
    if state_dir and os.path.exists(state_dir) and not os.path.isdir(state_dir):
        _refuse(f"--moss_ft_state_dir {state_dir!r} is a file, not a directory")


def _check_precision(args) -> str:
    precision = getattr(args, "mixed_precision", "bf16")
    if precision not in ("bf16", "fp16", "no"):
        _refuse(f'mixed_precision must be bf16 (default), fp16 (with --moss_stream_fp32) or no (fp32 compute), got'
                f' {precision!r}')
    for flag in ("full_bf16", "full_fp16"):
        if getattr(args, flag, False):
            _refuse(f"--{flag} is not supported: the fp32 master in host RAM replaces full_bf16 / full_fp16 (the GPU"
                    " holds only the compute copy; every update is applied to the fp32 master and cast, design 3.1)")
    for flag in ("fp8_base", "fp8_base_unet"):
        if getattr(args, flag, False):
            _refuse(f"--{flag} is not supported: every base weight is trained (an fp32 master cast to the bf16 compute"
                    " copy); fp8 weights cannot take the updates")
    if precision == "no":
        logger.warning(f'{PREFIX}mixed_precision = "no": fp32 compute (the precision reference / CPU test mode). The'
                       " GPU copy is fp32: about 24.7 GiB of VRAM at 1024^2 batch 1 (design 2.1); train with bf16")
    return precision


def _check_loss_scaling(args, precision: str, mode: str) -> None:
    """fp16 compute (plan 2.1 / 2.5): the fp32 stream, a host optimizer mode, valid loss-scale settings; the
    loss-scale flags only with fp16."""
    if precision != "fp16":
        changed = [f"--{name}" for name, default in LOSS_SCALE_DEFAULTS.items()
                   if getattr(args, name, default) != default]
        if changed:
            _refuse(f"{', '.join(changed)} {'is' if len(changed) == 1 else 'are'} for fp16 compute (dynamic loss"
                    f' scaling) only; mixed_precision is {precision!r}: leave {"it" if len(changed) == 1 else "them"}'
                    ' at the default or set mixed_precision = "fp16"')
        return
    if mode == ADAFACTOR_FUSED:
        _refuse(f"fp16 needs a host optimizer mode (adamw_host / came_host) or {CAROUSEL}: the loss scale is undone in"
                " the optimizer step; adafactor_fused steps on the GPU inside the hooks")
    init = getattr(args, "moss_ft_loss_scale", ft_scaler.DEFAULT_INIT_SCALE)
    low = getattr(args, "moss_ft_loss_scale_min", ft_scaler.DEFAULT_MIN_SCALE)
    for name, value in (("moss_ft_loss_scale", init), ("moss_ft_loss_scale_min", low)):
        if isinstance(value, bool) or not ft_scaler.is_power_of_two(value) or not 1.0 <= float(value) <= ft_scaler.MAX_SCALE:
            _refuse(f"--{name} must be a power of two in [1, 2^24] (the unscale on the host is then exact), got"
                    f" {value!r}")
    if float(low) > float(init):
        _refuse(f"--moss_ft_loss_scale_min {low:g} is above --moss_ft_loss_scale {init:g}")
    _int_option(args, "moss_ft_loss_scale_growth_interval", 0, None, "0 = never grows; an overflow still halves it")
    policy = getattr(args, "moss_ft_overflow_policy", "partial")
    if policy not in OVERFLOW_POLICIES:
        _refuse(f"--moss_ft_overflow_policy must be one of {OVERFLOW_POLICIES}, got {policy!r}")
    norm = float(getattr(args, "max_grad_norm", 0.0) or 0.0)
    if mode == CAROUSEL:
        if policy != "window":
            logger.info(f"{PREFIX}{CAROUSEL}: --moss_ft_overflow_policy {policy} is not used: a step with a non-finite"
                        " gradient is skipped as a whole (the optimizer steps after the backward, GradScaler semantics)")
        policy = "window"
    elif policy == "window" and not (getattr(args, "moss_ft_clip_mode", "stale") == "exact" and norm > 0):
        _refuse("--moss_ft_overflow_policy window needs --moss_ft_clip_mode exact with max_grad_norm > 0: only then"
                " does every step run after the backward, where a whole step can still be skipped (with the"
                " in-backward step the parameters whose gradient arrived first are already stepped)")
    logger.info(f"{PREFIX}fp16 compute with the fp32 stream: dynamic loss scale from {float(init):g} (x0.5 on an"
                f" overflow, x2 after {args.moss_ft_loss_scale_growth_interval} clean steps (0 = never), min"
                f" {float(low):g}),"
                f" overflow policy {policy}")


def _check_refused_options(args) -> None:
    if getattr(args, "deepspeed", False):
        _refuse("--deepspeed is not supported: the optimizer state lives in host RAM and is stepped inside the backward"
                " by this script (design 3.1); DeepSpeed would own the optimizer and the gradients")
    if (getattr(args, "blocks_to_swap", None) or 0) > 0:
        _refuse("--blocks_to_swap is not supported: block swap moves weights between devices, while the in-backward"
                " step uploads the new weights into the resident GPU parameters (design 3.1). For VRAM use a smaller"
                " batch or --unsloth_offload_checkpointing")
    if getattr(args, "use_ramtorch", False) or getattr(args, "use_ramtorch_network", False):
        _refuse("RamTorch (use_ramtorch) is not supported: it streams weights between devices, while the in-backward"
                " step uploads the new weights into the resident GPU parameters (design 3.1)")
    if getattr(args, "cpu_offload_checkpointing", False):
        _refuse("--cpu_offload_checkpointing is not supported: the per-parameter gradient hooks are verified with no"
                " checkpoint, the plain one, unsloth offload and the no-grad first-pass node (design 1.4); use"
                " --unsloth_offload_checkpointing (faster)")
    if getattr(args, "moss_block_compile", False):
        _refuse("--moss_block_compile is not supported yet: block compile is not validated with trainable base weights"
                " and the gradient hooks (phase 2, design 8.4). The elementwise chains stay compiled (--moss_no_compile"
                " turns that off)")
    if getattr(args, "torch_compile", False):
        _refuse("--torch_compile is not supported: a whole-model compile is not validated with the in-backward"
                " gradient hooks; the MOSS elementwise chains are compiled already")
    if (getattr(args, "fused_optimizer_groups", None) or 0) > 0:
        _refuse("--fused_optimizer_groups is not supported: the host optimizer steps every parameter inside the"
                " backward already (design 3.1); the option belongs to the GPU fused optimizers of sdxl / flux_train")
    if getattr(args, "blockwise_fused_optimizers", False):
        _refuse("--blockwise_fused_optimizers is not supported: the host optimizer steps every parameter inside the"
                " backward already (design 3.1); the option belongs to the GPU fused optimizers of flux_train")
    if float(getattr(args, "validation_split", 0.0) or 0.0) > 0.0:
        _refuse(f"validation_split > 0 is not supported: {VAL_HELD_OUT_HINT}")
    world = os.environ.get("WORLD_SIZE", "").strip()
    if world.isdigit():
        check_single_process(int(world))


def _check_lora_only(args) -> None:
    for name, why in LORA_ONLY_ARGS.items():
        value = getattr(args, name, None)
        if name == "moss_lora_exclude_blocks":  # default "" (a TOML array / int reaches here unconverted)
            changed = value not in (None, [], ()) and str(value).strip() != ""
        else:
            changed = bool(value)
        if changed:
            _refuse(f"--{name} is a LoRA / LyCORIS option ({why}); moss_train.py trains the base weights without a"
                    " network: leave it at its default")


def _check_checkpointing(args, cuda: Optional[bool]) -> None:
    if getattr(args, "unsloth_offload_checkpointing", False):
        if not getattr(args, "gradient_checkpointing", False):
            logger.info(f"{PREFIX}--unsloth_offload_checkpointing is on: enabling --gradient_checkpointing")
            args.gradient_checkpointing = True
        logger.info(f"{PREFIX}--unsloth_offload_checkpointing: allowed (the gradient hooks are verified with it); costs"
                    " ~12 % time and keeps the checkpointed block inputs in pinned host RAM")
    if cuda is None:
        cuda = torch.cuda.is_available()
    if cuda and not getattr(args, "gradient_checkpointing", False):
        _refuse("--gradient_checkpointing is required on CUDA: the budget (design 2.1) counts every block checkpointed;"
                " without it the 42 blocks' activations do not fit")
    if getattr(args, "moss_ckpt_reuse_attn", False) and getattr(args, "attn_mode", None) != "flash":
        _refuse("--moss_ckpt_reuse_attn needs attn_mode = \"flash\" (it reuses flash-attn's stored output and"
                f" softmax_lse), attn_mode is {getattr(args, 'attn_mode', None) or 'torch'!r}")


def _check_stream(args, precision: str) -> None:
    if precision == "fp16":
        if not getattr(args, "moss_stream_fp32", False):
            _refuse("mixed_precision fp16 needs --moss_stream_fp32: Base's residual stream exceeds fp16's 65504 and the"
                    " stream must be fp32 for full fine-tuning (--moss_stream_scale is for LoRA);"
                    f" {STREAM_PRECISION_NOTE}")
        return
    if precision != "bf16" or getattr(args, "moss_stream_fp32", False):
        return  # fp32 compute runs the stream in fp32 anyway
    if not getattr(args, "moss_ft_allow_bf16_stream", False):
        _refuse(f"--moss_stream_fp32 is required with bf16 compute: Base's residual stream needs fp32; {STREAM_PRECISION_NOTE}."
                " --moss_ft_allow_bf16_stream allows it (e.g. a heal-type model)")
    logger.warning(f"{PREFIX}training with a bf16 residual stream (--moss_ft_allow_bf16_stream): {STREAM_PRECISION_NOTE}")


def _check_workers(args) -> None:
    """Every DataLoader worker is a spawned process on Windows that imports torch and moss_train.py again: about 1 GiB
    of host RAM each (the host budget counts the ones above ft_budget.PROCESS_WORKERS, ft_budget.WORKER_BYTES each).
    The caches make a batch a few file reads, so 2-4 are enough; kohya's default is 8."""
    workers = ft_budget.dataloader_workers(args)
    if workers > ft_budget.PROCESS_WORKERS:
        respawn = ("" if getattr(args, "persistent_data_loader_workers", False) else
                   "; without persistent_data_loader_workers they are started again every epoch")
        logger.warning(f"{PREFIX}max_data_loader_n_workers {workers}: each worker process imports torch and this script"
                       f" again, about {ft_budget.WORKER_BYTES / (1 << 30):.0f} GiB of host RAM each next to the"
                       f" optimizer's pools (counted in the host budget){respawn}. With cached latents and text-encoder"
                       f" outputs {ft_budget.PROCESS_WORKERS} or fewer are enough")


def _check_latent_cache(args) -> None:
    if getattr(args, "cache_latents_to_disk", False) and not getattr(args, "cache_latents", False):
        args.cache_latents = True  # as train_util.verify_training_args
    if not getattr(args, "cache_latents", False):
        _refuse("--cache_latents is required: the VAE stays on the CPU while training (design 2.4)")


def te_live(args) -> bool:
    """--moss_te_live: the captions are encoded at train time (no TE cache)."""
    return bool(getattr(args, "moss_te_live", False))


def _check_text_path(args, cuda: Optional[bool]) -> None:
    """Cached TE outputs (default) or the live text encoder (--moss_te_live), never both (plan 1.7)."""
    if not te_live(args):
        changed = [f"--{name}" for name, default in TE_LIVE_DEFAULTS.items() if getattr(args, name, default) != default]
        if changed:
            batch_hint = (". --moss_te_live_batch batches the live encoder's layer calls; the TE cache encodes each"
                          " caption alone once (the official per-prompt encoding), so it has nothing to batch"
                          if "--moss_te_live_batch" in changed else "")
            _refuse(f"{', '.join(changed)} {'is' if len(changed) == 1 else 'are'} only used with --moss_te_live (the"
                    f" live text encoder); set --moss_te_live or leave them at their defaults{batch_hint}")
        if getattr(args, "cache_text_encoder_outputs_to_disk", False) and not getattr(args, "cache_text_encoder_outputs",
                                                                                     False):
            args.cache_text_encoder_outputs = True  # as validate_moss_args
        if not getattr(args, "cache_text_encoder_outputs", False):
            _refuse("--cache_text_encoder_outputs is required: the 9B text encoder is only loaded while caching (design"
                    " 2.4); or encode the captions at train time with --moss_te_live")
        return
    for name in ("cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk"):
        if getattr(args, name, False):
            _refuse(f"--{name} with --moss_te_live: choose one. --moss_te_live encodes every caption at train time and"
                    " writes no TE cache; the cached path encodes each caption once before training")
    if int(getattr(args, "cache_caption_variants", 0) or 0) > 1:
        _refuse("--cache_caption_variants is a TE-cache feature: --moss_te_live draws a new caption (shuffle, wildcard"
                " line, dropout) at every fetch, so nothing is cached. Remove cache_caption_variants")
    _int_option(args, "moss_te_live_ahead", 1, None, "micro-batches per encoder pass")
    slots = getattr(args, "moss_te_live_ring_slots", None)
    if isinstance(slots, bool) or slots not in TE_RING_SLOTS:
        _refuse(f"--moss_te_live_ring_slots must be one of {TE_RING_SLOTS} (2 = one layer uploads while one computes),"
                f" got {slots!r}")
    _int_option(args, "moss_te_live_threads", 1, os.cpu_count() or 1, "intra-op threads of the stager")
    batch = _int_option(args, "moss_te_live_batch", 1, None, "captions per batched layer call of an encoder pass; 1 ="
                        " each caption alone, bit for bit the cached path")
    host = getattr(args, "moss_te_live_host", None)
    if host not in TE_HOST_MODES:
        _refuse(f"--moss_te_live_host must be one of {TE_HOST_MODES}, got {host!r}")
    if host == "pinned":
        if cuda is None:
            cuda = torch.cuda.is_available()
        if not cuda:
            _refuse("--moss_te_live_host pinned page-locks the encoder with cudaHostRegister: it needs CUDA (use copy or"
                    " mmap)")
    calls = ("each caption alone" if batch == 1 else
             f"layer calls of up to {batch} captions sorted by length and right-padded, at most {batch} x"
             f" {ft_budget.DEFAULT_TEXT_TOKENS} padded tokens per call (--moss_te_live_batch; close to, not"
             " bit-identical with, the cached encoding)")
    logger.info(f"{PREFIX}--moss_te_live: the captions are encoded at train time ({args.moss_te_live_ahead} micro-batches"
                f" per pass, {slots} layer slots, host {host}, {calls}); no TE cache is read or written")


CACHED_CAPTION_HINT = "or use --moss_te_live (a new draw at every fetch)"


def _check_cached_captions(dataset_group) -> None:
    """The caption features a TE cache cannot serve (were the assert of validate_moss_args), as clear refusals."""
    for dataset in moss_captions.datasets_of(dataset_group):
        variants = int(getattr(dataset, "caption_aug_variant_count", 0) or 0)
        for subset in getattr(dataset, "subsets", None) or []:
            where = f" (subset {getattr(subset, 'image_dir', None) or '?'})"
            if variants <= 1:
                if getattr(subset, "shuffle_caption", False):
                    _refuse("shuffle_caption needs --cache_caption_variants N>1 with the TE cache (a cached caption is"
                            f" one fixed text), {CACHED_CAPTION_HINT}{where}")
                if float(getattr(subset, "caption_tag_dropout_rate", 0.0) or 0.0) > 0.0:
                    _refuse("caption_tag_dropout_rate > 0 needs --cache_caption_variants N>1 with the TE cache,"
                            f" {CACHED_CAPTION_HINT}{where}")
                if float(getattr(subset, "caption_dropout_rate", 0.0) or 0.0) > 0.0:
                    _refuse("caption_dropout_rate > 0 needs --cache_caption_variants N>1 with the TE cache (a dropped"
                            " caption is encoded as '' through the chat template, the official unconditional prompt),"
                            f" {CACHED_CAPTION_HINT}{where}")
            if float(getattr(subset, "token_warmup_step", 0) or 0) > 0:
                _refuse(f"token_warmup_step is step-dependent and cannot be cached, {CACHED_CAPTION_HINT}{where}")
            if int(getattr(subset, "caption_dropout_every_n_epochs", 0) or 0) > 0:
                _refuse("caption_dropout_every_n_epochs is not supported with the TE cache (it zeroes cached outputs;"
                        " MOSS needs >= 1 text token). Use caption_dropout_rate with --cache_caption_variants N>1,"
                        f" {CACHED_CAPTION_HINT}{where}")


def check_captions(args, dataset_group) -> None:
    """Step 4: the caption settings against the text path (refusals on the cached path), the `|||` warnings (both
    paths) and, in live mode, a summary of what is drawn per fetch."""
    if not te_live(args):
        _check_cached_captions(dataset_group)
    for line in moss_captions.separator_warnings(dataset_group):
        logger.warning(f"{PREFIX}{line}")
    if te_live(args):
        for line in moss_captions.live_caption_summary(dataset_group):
            logger.info(f"{PREFIX}live captions: {line}")


def _check_carousel(args, mode: str, cuda: Optional[bool]) -> None:
    """The Carousel flags (plan 4.3): only with --moss_carousel; ranges and choices; the kinds' device needs."""
    from . import ft_carousel_units as cu

    if mode != CAROUSEL:
        changed = [carousel_flag(key) for key, default in CAROUSEL_DEFAULTS.items()
                   if getattr(args, carousel_dest(key), default) != default]
        if changed:
            _refuse(f"{', '.join(changed)} {'is' if len(changed) == 1 else 'are'} only used with {carousel_flag()}; set"
                    f" it or leave {'it' if len(changed) == 1 else 'them'} at the default")
        return
    kind = carousel_kind(args)
    if cuda is None:
        cuda = torch.cuda.is_available()
    shapes = ft_budget.moss_param_shapes()
    patterns = split_freeze_patterns(getattr(args, "moss_ft_freeze", ""))
    frozen = set(frozen_parameter_names(shapes, patterns)) if patterns else set()
    always = carousel_arg(args, "always")
    if always not in cu.ALWAYS_POLICIES:
        _refuse(f"{carousel_flag('always')} must be one of {cu.ALWAYS_POLICIES}, got {always!r}")
    units = cu.discover_units([(n, s) for n, s in shapes.items() if n not in frozen],
                              stem_unit=cu.stem_rotates(always))
    if not units:
        _refuse(f"{CAROUSEL}: --moss_ft_freeze leaves no block trainable: nothing would rotate (use adamw_host)")
    active, vram = carousel_arg(args, "active"), carousel_arg(args, "vram_gib")
    if active is not None and vram is not None:
        _refuse(f"{carousel_flag('active')} and {carousel_flag('vram_gib')}: choose one (K given, or K from the budget)")
    if active is not None:
        _int_option(args, carousel_dest("active"), 1, len(units), f"active units of the {len(units)} trainable ones")
    if vram is not None and (isinstance(vram, bool) or not isinstance(vram, (int, float)) or not math.isfinite(vram)
                             or vram <= 0):
        _refuse(f"{carousel_flag('vram_gib')} must be a number of GiB > 0, got {vram!r}")
    _int_option(args, carousel_dest("period"), 1, None, "optimizer steps between switches")
    seed = carousel_arg(args, "seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        _refuse(f"{carousel_flag('seed')} must be an integer, got {seed!r}")
    for key, choices in (("select", cu.SELECTIONS), ("always", cu.ALWAYS_POLICIES), ("state", cu.STATE_POLICIES),
                         ("state_dtype", cu.STATE_DTYPES), ("host_pinned", cu.HOST_PINNED)):
        if carousel_arg(args, key) not in choices:
            _refuse(f"{carousel_flag(key)} must be one of {choices}, got {carousel_arg(args, key)!r}")
    if carousel_arg(args, "state_dtype") == "bf16":
        if kind != "came":
            _refuse(f"{carousel_flag('state_dtype')} bf16 is for came only (its exp_avg): {kind} keeps"
                    + (" its second moment, which bf16 would ruin" if kind == "adamw" else " 8-bit states already"))
        if carousel_arg(args, "state") != "keep":
            _refuse(f"{carousel_flag('state_dtype')} bf16 stores kept states: it needs {carousel_flag('state')} keep")
    if kind == "adamw8bit" and not cuda:
        _refuse(f"{CAROUSEL} adamw8bit needs CUDA: bitsandbytes' AdamW8bit has no CPU kernels (use adamw or came)")
    if carousel_arg(args, "host_pinned") == "masters" and not cuda:
        _refuse(f"{carousel_flag('host_pinned')} masters page-locks host memory for CUDA copies: it needs CUDA")
    unused = [f"--{name}" for name, default in CAROUSEL_UNUSED_DEFAULTS.items() if getattr(args, name, default) != default]
    if unused:
        logger.info(f"{PREFIX}{CAROUSEL}: {', '.join(unused)} {'is' if len(unused) == 1 else 'are'} not used (no host"
                    " worker, no pinned rings)")
    logger.info(f"{PREFIX}{CAROUSEL}: {kind}, {len(units)} units, K "
                + (f"{active}" if active is not None else "from the VRAM budget")
                + f", switch every {carousel_arg(args, 'period')} steps ({carousel_arg(args, 'select')}), always"
                f" {carousel_arg(args, 'always')}, state {carousel_arg(args, 'state')}"
                + (f" ({carousel_arg(args, 'state_dtype')})" if carousel_arg(args, "state") == "keep" else ""))


def _check_accumulation_and_clipping(args, mode: str) -> None:
    ga = getattr(args, "gradient_accumulation_steps", None)
    if isinstance(ga, bool) or not isinstance(ga, int) or ga < 1:
        _refuse(f"gradient_accumulation_steps must be an integer >= 1, got {ga!r}")
    if mode == ADAFACTOR_FUSED and ga != 1:
        _refuse(f"adafactor_fused steps every parameter in its gradient hook, so gradient_accumulation_steps must be 1"
                f" (got {ga}); adamw_host accumulates on the host")
    norm = getattr(args, "max_grad_norm", 0.0)
    if isinstance(norm, bool) or not isinstance(norm, (int, float)) or not math.isfinite(norm) or norm < 0:
        _refuse(f"max_grad_norm must be a finite number >= 0 (0 = no clipping), got {norm!r}")
    clip_mode = getattr(args, "moss_ft_clip_mode", "stale")
    if mode == CAROUSEL:
        if clip_mode != "stale":
            logger.info(f"{PREFIX}{CAROUSEL}: --moss_ft_clip_mode {clip_mode} is not used: the optimizer steps after the"
                        " backward, so clipping is always exact over the trainable tensors")
        logger.info(f"{PREFIX}{CAROUSEL}: max_grad_norm {norm:g}: "
                    + ("exact global clipping over the active units and the always group" if norm > 0 else
                       "no gradient clipping (grad_norm is still logged)"))
        return
    if mode == ADAFACTOR_FUSED:
        if clip_mode == "exact":
            _refuse("--moss_ft_clip_mode exact needs the host accumulator of adamw_host / came_host; adafactor_fused clips"
                    " each tensor in its hook (as anima_train.py)")
        if norm > 0:
            logger.info(f"{PREFIX}max_grad_norm {norm:g}: adafactor_fused clips each tensor in its hook, not the global"
                        " norm (as anima_train.py)")
        return
    if norm == 0:
        if clip_mode == "exact":
            logger.warning(f"{PREFIX}--moss_ft_clip_mode exact changes nothing without clipping (max_grad_norm 0): the"
                           " step stays inside the backward")
        logger.info(f"{PREFIX}max_grad_norm 0: no gradient clipping (grad_norm is still logged)")
    elif clip_mode == "stale":
        logger.info(f"{PREFIX}max_grad_norm {norm:g}, stale clipping: coefficient min(1, max / previous window's norm) so"
                    " the step runs inside the backward; a window without a known previous norm runs exact (design 3.1)")
    else:
        logger.info(f"{PREFIX}max_grad_norm {norm:g}, exact clipping: the window's own norm, the host step runs after the"
                    " backward (the tail becomes the whole host step)")


def _check_saving(args, mode: str) -> None:
    precision = getattr(args, "save_precision", None) or "bf16"
    if precision not in SAVE_PRECISIONS:
        _refuse(f"save_precision must be one of {SAVE_PRECISIONS} (bf16 = the Base file's dtype), got {precision!r}")
    args.save_precision = precision
    size = ft_budget.state_bytes(mode) / 1e9
    if mode == CAROUSEL:
        size = ft_budget.carousel_state_bytes(carousel_kind(args), state_policy=carousel_arg(args, "state"),
                                              state_dtype=carousel_arg(args, "state_dtype"),
                                              k=int(carousel_arg(args, "active") or 8),
                                              always=carousel_arg(args, "always")) / 1e9
    if getattr(args, "save_state", False):
        for name in ("save_last_n_epochs_state", "save_last_n_steps_state"):
            if getattr(args, name, None) is None:
                setattr(args, name, 1)
                logger.info(f"{PREFIX}--{name} not set with save_state: keeping 1 (a state is {size:.1f} GB, design 2.3)")
    else:
        logger.warning(f"{PREFIX}save_state is off: a crash or power loss loses the run. Recommended: save_state = true,"
                       f" save_every_n_steps ~1000, save_last_n_steps_state = 1 ({size:.1f} GB per state, design 4.7)")


def _check_resume(args) -> None:
    resume = getattr(args, "resume", None)
    if not resume:
        return
    if getattr(args, "resume_from_huggingface", False):
        _refuse("--resume_from_huggingface is not supported: moss_train.py resumes from a local state directory of its"
                " own format (design 4.7)")
    if not os.path.isdir(resume):
        _refuse(f"--resume {resume!r} is not a directory (moss_train.py resumes from its own state directories)")
    marker = os.path.join(resume, "COMPLETE.json")
    if not os.path.exists(marker) and any(os.path.exists(os.path.join(resume, name))
                                          for name in ("optimizer.bin", "random_states_0.pkl")):
        _refuse(f"--resume {resume!r} is an accelerate state (optimizer.bin, no COMPLETE.json): moss_train.py resumes"
                " only its own state format (design 4.7)")


def _check_lr(args) -> Optional[List[Tuple[float, float]]]:
    scheduler = str(getattr(args, "lr_scheduler", "constant") or "constant")
    if scheduler.lower().startswith("adafactor"):
        _refuse(f"lr_scheduler {scheduler!r} (relative-step Adafactor) is not supported: the LR comes from"
                " --learning_rate or --moss_ft_lr_phases")
    resolve_warmup_steps(getattr(args, "lr_warmup_steps", 0), 1)  # type and range; the steps follow the dataset
    if getattr(args, "zero_lr_warmup", False):
        _refuse("--zero_lr_warmup is not supported: kohya's ZeroLRWarmupScheduler keeps its inner scheduler (with the"
                " optimizer and every parameter) in its state, so a training state could be written but never resumed"
                " (scheduler.pt is read with torch.load(weights_only=True)). Use --lr_warmup_steps (a linear warmup)")
    text = getattr(args, "moss_ft_lr_phases", None)
    if not text:
        if getattr(args, "learning_rate", None) is None:
            args.learning_rate = KOHYA_LEARNING_RATE
        if getattr(args, "max_train_steps", None) is None and getattr(args, "max_train_epochs", None) is None:
            args.max_train_steps = KOHYA_MAX_TRAIN_STEPS
        return None
    phases = parse_lr_phases(text)
    lr1 = phases[0][1]
    given = getattr(args, "learning_rate", None)
    if given is not None and not math.isclose(float(given), lr1, rel_tol=1e-9, abs_tol=0.0):
        _refuse(f"--learning_rate {given:g} conflicts with --moss_ft_lr_phases (LR1 {lr1:g}): leave learning_rate out"
                " (it is set to LR1) or set it to LR1")
    args.learning_rate = lr1
    if getattr(args, "max_train_steps", None) is not None:
        _refuse("--max_train_steps cannot be combined with --moss_ft_lr_phases: the phases' epochs give the length")
    if scheduler not in PHASE_SCHEDULERS:
        _refuse(f"--moss_ft_lr_phases replaces the LR scheduler (one schedule source): lr_scheduler must be constant or"
                f" constant_with_warmup (the warmup is --lr_warmup_steps), got {scheduler!r}")
    for name, value in (("lr_scheduler_type", getattr(args, "lr_scheduler_type", "")),
                        ("lr_scheduler_args", getattr(args, "lr_scheduler_args", None)),
                        ("lr_decay_steps", _non_zero(getattr(args, "lr_decay_steps", 0)))):
        if value:
            _refuse(f"--{name} is not used with --moss_ft_lr_phases (one schedule source): remove it")
    total = math.fsum(epochs for epochs, _ in phases)
    epochs = getattr(args, "max_train_epochs", None)
    if epochs is not None:
        if not math.isclose(float(epochs), total, rel_tol=0.0, abs_tol=1e-9):
            _refuse(f"--max_train_epochs {epochs} does not equal the sum of the --moss_ft_lr_phases epochs ({total:g}):"
                    " set it to that sum or leave it out")
    elif abs(total - round(total)) < 1e-9:
        args.max_train_epochs = int(round(total))
    logger.info(f"{PREFIX}LR phases {text!r}: " + ", ".join(f"{lr:g} for {epochs:g} epochs" for epochs, lr in phases)
                + f" ({total:g} epochs); learning_rate = LR1 = {lr1:g}")
    return phases


def _non_zero(value) -> bool:
    try:
        return float(value or 0) != 0.0
    except (TypeError, ValueError):
        return bool(value)


def _check_freeze(args) -> None:
    patterns = split_freeze_patterns(getattr(args, "moss_ft_freeze", ""))
    if not patterns:
        return
    shapes = ft_budget.moss_param_shapes()
    frozen = frozen_parameter_names(shapes, patterns)
    if len(frozen) == len(shapes):
        _refuse(f"--moss_ft_freeze {patterns} freezes every parameter: nothing would be trained")
    sizes = ft_budget.ModelSizes.from_shapes(shapes, frozen)
    logger.info(f"{PREFIX}--moss_ft_freeze froze {len(frozen)} of {len(shapes)} tensors"
                f" ({(sizes.total - sizes.trainable) / 1e9:.3f}B parameters); {sizes.trainable / 1e9:.3f}B are trained")


def _weight_decay_note(args, mode: str, kwargs: Dict[str, object], phases) -> None:
    decay = float(kwargs.get("weight_decay", 0.0) or 0.0)
    if mode == ADAFACTOR_FUSED or decay <= 0:
        return
    lowest = min(lr for _, lr in phases) if phases else float(args.learning_rate)
    if lowest * decay < WD_RESOLUTION:
        logger.info(f"{PREFIX}weight_decay {decay:g} x lr {lowest:g} = {lowest * decay:.3g} per step is below the fp32"
                    f" resolution of the master ({WD_RESOLUTION:g} relative): that decay rounds away (design 1.5)")


def _validate_static(args, cuda: Optional[bool]) -> str:
    _check_new_arguments(args)
    precision = _check_precision(args)
    _check_refused_options(args)
    _check_lora_only(args)
    _check_checkpointing(args, cuda)
    _check_stream(args, precision)
    _check_latent_cache(args)
    _check_text_path(args, cuda)
    _check_workers(args)
    mode = optimizer_mode(args)
    _check_carousel(args, mode, cuda)
    _check_accumulation_and_clipping(args, mode)
    _check_loss_scaling(args, precision, mode)
    _check_saving(args, mode)
    _check_resume(args)
    _check_validation_args(args)
    kwargs = parse_optimizer_args(args, mode)
    phases = _check_lr(args)
    _weight_decay_note(args, mode, kwargs, phases)
    _check_freeze(args)
    logger.info(f"{PREFIX}optimizer mode {mode} {kwargs}, learning_rate {args.learning_rate:g}")
    return mode


def _check_validation_datasets(dataset_group, val_dataset_group) -> None:
    """kohya's own validation sets inside the training TOML (validation_split, is_val subsets) are refused: the held-out
    set is its own TOML (--moss_ft_val_dataset_config, ft_val)."""
    for dataset in getattr(dataset_group, "datasets", None) or []:
        if float(getattr(dataset, "validation_split", 0.0) or 0.0) > 0.0:
            _refuse(f"validation_split > 0 in the dataset config is not supported: {VAL_HELD_OUT_HINT}")
    if val_dataset_group is not None and (getattr(val_dataset_group, "datasets", None) or []):
        _refuse(f"validation subsets in the training dataset config (is_val / validation_split) are not supported:"
                f" {VAL_HELD_OUT_HINT}")


VAL_HELD_OUT_HINT = ("put the held-out images in a dataset TOML of their own and pass it as"
                     " --moss_ft_val_dataset_config (a deterministic validation loss: fixed captions, noise levels and"
                     " noise)")


def val_enabled(args) -> bool:
    """--moss_ft_val_dataset_config: a validation loss is computed (library/moss/ft_val.py)."""
    return bool(getattr(args, "moss_ft_val_dataset_config", None))


def _check_validation_args(args) -> None:
    """The --moss_ft_val_* flags: only with --moss_ft_val_dataset_config; the TOML exists; N >= 0; the sigmas; the
    seed."""
    path = getattr(args, "moss_ft_val_dataset_config", None)
    if not path:
        changed = [f"--{name}" for name, default in VAL_DEFAULTS.items() if getattr(args, name, default) != default]
        if changed:
            _refuse(f"{', '.join(changed)} {'is' if len(changed) == 1 else 'are'} only used with"
                    " --moss_ft_val_dataset_config (the held-out dataset TOML); set it or leave"
                    f" {'it' if len(changed) == 1 else 'them'} at the default")
        return
    if not isinstance(path, str) or not os.path.isfile(path):
        _refuse(f"--moss_ft_val_dataset_config {path!r} is not a file (a kohya dataset TOML of the held-out images)")
    dataset_config = getattr(args, "dataset_config", None)
    if dataset_config and os.path.isfile(dataset_config) and os.path.samefile(dataset_config, path):
        logger.warning(f"{PREFIX}--moss_ft_val_dataset_config is the training dataset_config: the validation loss is"
                       " then measured on the training images (fixed noise), not on held-out ones")
    _int_option(args, "moss_ft_val_every_n_steps", 0, None, "0 = only before the first and after the last step")
    sigmas = parse_val_sigmas(getattr(args, "moss_ft_val_sigmas", None))
    seed = getattr(args, "moss_ft_val_seed", None)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < (1 << 63):
        _refuse(f"--moss_ft_val_seed must be an integer in [0, 2^63), got {seed!r}")
    every = int(args.moss_ft_val_every_n_steps)
    logger.info(f"{PREFIX}validation loss on {path}: "
                + (f"every {every} optimizer steps" if every else "before the first and after the last step")
                + f" (and always at step 0 and the last step), {len(sigmas)} noise levels"
                f" {', '.join(f'{s:g}' for s in sigmas)}, noise seed {seed}; {VAL_LOG_FILE} in output_dir")


def _validate_dataset(args, dataset_group, val_dataset_group, mode: str, moss_val_group=None, **checks) -> None:
    _check_validation_datasets(dataset_group, val_dataset_group)
    check_captions(args, dataset_group)
    images = ft_budget.count_images(dataset_group)
    if images > MEMORY_CACHE_IMAGE_LIMIT:
        flags = ("cache_latents_to_disk",) if te_live(args) else ("cache_latents_to_disk",
                                                                   "cache_text_encoder_outputs_to_disk")
        missing = [f"--{name}" for name in flags if not getattr(args, name, False)]
        if missing:
            _refuse(f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} required above"
                    f" {MEMORY_CACHE_IMAGE_LIMIT:,} images ({images:,} images): in-memory caches would take tens of GB of"
                    " host RAM next to the optimizer's pools (design 2.4)")
    ga = int(args.gradient_accumulation_steps)
    batches = len(dataset_group)
    per_epoch = steps_per_epoch(batches, ga)
    text = getattr(args, "moss_ft_lr_phases", None)
    if text:
        phases = parse_lr_phases(text)
        bounds = lr_phase_bounds(phases, per_epoch)
        warmup = resolve_warmup_steps(getattr(args, "lr_warmup_steps", 0), bounds[-1])
        if warmup > bounds[0]:
            _refuse(f"lr_warmup_steps {getattr(args, 'lr_warmup_steps', 0)!r} ({warmup} steps) must end inside LR"
                    f" phase 1 (optimizer steps [0, {bounds[0]}) at {per_epoch} steps per epoch)")
        args.max_train_steps = bounds[-1]
        logger.info(f"{PREFIX}{batches} batches per epoch / GA {ga} = {per_epoch} optimizer steps per epoch; LR phase"
                    f" boundaries {bounds} (max_train_steps {bounds[-1]}), warmup {warmup} steps")
    elif getattr(args, "max_train_epochs", None) is not None:
        args.max_train_steps = int(args.max_train_epochs) * per_epoch
    check_lr_scheduler(args, per_epoch)  # the run's scheduler builds and is resumable (before caching, design 4.3)
    ft_budget.run_startup_checks(args, dataset_group, mode, sizes=model_sizes(args), val_dataset_group=moss_val_group,
                                 **checks)


def validate_moss_ft_args(args, dataset_group=None, *, val_dataset_group=None, moss_val_group=None,
                          cuda: Optional[bool] = None, virtual_memory: Optional[Callable] = None,
                          disk_usage: Optional[Callable] = None, exists: Optional[Callable[[str], bool]] = None) -> None:
    """Design 4.5 for moss_train.py. ValueError("MOSS full FT: <reason>") on the first refused setting.

    Without a dataset group (start-up step 2, design 4.3): the new arguments, the optimizer mode and its optimizer_args,
    the fixed handling of the existing arguments, the LR phases against the other LR / length arguments, the freeze
    patterns. Fills in: learning_rate (LR1 with phases, else kohya's 2e-6 when unset), max_train_epochs (the phases'
    sum when whole), max_train_steps (1600 when neither it nor max_train_epochs is set and there are no phases),
    save_precision bf16, save_last_n_*_state 1 with save_state, gradient_checkpointing with unsloth offload.

    With the train dataset group (step 4; runs the step-2 part first if it has not run on these args): no validation
    datasets inside the training TOML (val_dataset_group: what generate_dataset_group_by_blueprint returned; the
    held-out set is moss_val_group, built by ft_val from --moss_ft_val_dataset_config and only counted by the budget
    checks), both *_to_disk cache flags above
    2,000 images, steps per epoch = ceil(len(group) / GA) -> the LR phase boundaries (each phase >= 1 step, the warmup
    inside phase 1) and max_train_steps, then ft_budget.run_startup_checks (host RAM / disk refusals, GPU / cache
    warnings; virtual_memory / disk_usage / exists are injectable for tests).

    cuda: whether the run is on CUDA (default torch.cuda.is_available()); gradient checkpointing is required there.
    num_processes > 1 is refused here through WORLD_SIZE and again by check_single_process(accelerator.num_processes)."""
    if not getattr(args, "_moss_ft_static_checked", False):
        _validate_static(args, cuda)
        args._moss_ft_static_checked = True
    if dataset_group is not None:
        _validate_dataset(args, dataset_group, val_dataset_group, optimizer_mode(args), moss_val_group=moss_val_group,
                          virtual_memory=virtual_memory, disk_usage=disk_usage, exists=exists)
