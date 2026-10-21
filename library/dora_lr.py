# DoRA magnitude parameters (LyCORIS dora_scale) in their own optimizer group: --learning_rate_DoRA and --weight_decay_DoRA.
#
# AdamW moves every parameter by about lr per step in absolute terms, while dora_scale holds whole row/column norms of the base
# weight (MOSS-Image input-axis norms ~2.4, Anima-3.8B output-axis ~0.45). --learning_rate_DoRA sets their lr separately.
# Weight decay pulls a parameter toward 0, which for a norm means shrinking the base weight; LyCORIS tags dora_scale with
# weight_decay_ratio=0 but torch optimizers ignore the tag. The DoRA group therefore gets weight_decay_DoRA (default 0).
# The lr scheduler scales every group the same way (warmup, decay).
# A --resume state saved before this split (no DoRA group) keeps its own layout: torch cannot load it into more groups.

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

DORA_PARAM_SUFFIXES = (".dora_scale",)
DECAY_KEYS = ("weight_decay", "adam_weight_decay")  # torch AdamW / adv_optm Muon_adv's AdamW path


def add_dora_lr_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--learning_rate_DoRA", "--learning_rate_dora", dest="learning_rate_DoRA", type=float, default=None,
                        help="learning rate for DoRA magnitude parameters (LyCORIS dora_scale); default: the same lr as the rest."
                             " The lr scheduler applies to it as to every other group")
    parser.add_argument("--weight_decay_DoRA", "--weight_decay_dora", dest="weight_decay_DoRA", type=float, default=0.0,
                        help="weight decay for DoRA magnitude parameters (default 0: decaying a norm shrinks the base weight;"
                             " LyCORIS marks them weight_decay_ratio=0). Set the optimizer's value to restore the old behaviour")


def _arg(args, name: str):
    value = getattr(args, name, None)
    return getattr(args, name.lower(), None) if value is None else value  # TOML keys in lower case


def dora_learning_rate(args) -> Optional[float]:
    value = _arg(args, "learning_rate_DoRA")
    if value is None:
        return None
    value = float(value)
    if not value > 0:
        raise ValueError(f"learning_rate_DoRA must be > 0, got {value}")
    return value


def dora_weight_decay(args) -> float:
    value = _arg(args, "weight_decay_DoRA")
    value = 0.0 if value is None else float(value)
    if value < 0:
        raise ValueError(f"weight_decay_DoRA must be >= 0, got {value}")
    return value


def _is_dora_param(param: torch.nn.Parameter, name: str) -> bool:
    return bool(getattr(param, "_is_dora_scale", False)) or name.endswith(DORA_PARAM_SUFFIXES)


def _description(group: Any, index: int) -> str:
    if isinstance(group, dict):
        return str(group.get("lr_description") or group.get("name") or f"group{index}")
    return f"group{index}"


def split_dora_param_groups(network, trainable_params, lr_descriptions: Optional[List[str]], lr_dora: Optional[float],
                            default_lr: Optional[float] = None, weight_decay: float = 0.0,
                            decay_keys: Optional[Sequence[str]] = ()) -> Tuple[Any, Optional[List[str]]]:
    """Move DoRA magnitude parameters of every group into a following group with lr ``lr_dora`` (None: the source group's lr)
    and ``weight_decay`` for each of ``decay_keys`` (the weight-decay arguments the optimizer has; None: they could not be
    read, the DoRA magnitudes keep the optimizer's weight decay and a warning says so).

    Returns (trainable_params, lr_descriptions) unchanged when there is nothing to change: no DoRA parameters, or neither a DoRA
    lr nor a weight-decay argument."""
    if lr_dora is None and decay_keys is not None and not decay_keys:
        return trainable_params, lr_descriptions
    dora_ids = {id(p) for name, p in network.named_parameters() if _is_dora_param(p, name)}
    if not dora_ids:
        if lr_dora is not None:
            logger.warning("learning_rate_DoRA is set but the network has no DoRA parameters (dora_scale); ignored")
        return trainable_params, lr_descriptions
    if decay_keys is None:
        logger.warning("could not read the optimizer's arguments: DoRA magnitudes keep the optimizer's weight decay, not"
                       f" weight_decay_DoRA ({weight_decay}); give weight_decay in optimizer_args to apply it")
        if lr_dora is None:
            return trainable_params, lr_descriptions

    if isinstance(trainable_params, (list, tuple)) and trainable_params and not isinstance(trainable_params[0], dict):
        trainable_params = [{"params": list(trainable_params)}]  # plain tensor list: one group at the optimizer's lr
    groups: List[Dict[str, Any]] = [dict(g) for g in trainable_params]
    descriptions = list(lr_descriptions) if lr_descriptions is not None else [_description(g, i) for i, g in enumerate(groups)]

    out_groups, out_desc, moved = [], [], 0
    for index, group in enumerate(groups):
        params = list(group["params"])
        dora = [p for p in params if id(p) in dora_ids]
        rest = [p for p in params if id(p) not in dora_ids]
        description = descriptions[index] if index < len(descriptions) else _description(group, index)
        if not dora:
            out_groups.append(group)
            out_desc.append(description)
            continue
        if rest:
            out_groups.append({**group, "params": rest})
            out_desc.append(description)
        dora_group = {**group, "params": dora}
        if lr_dora is not None:
            base_lr = group.get("lr", default_lr)
            dora_group["lr"] = torch.tensor(lr_dora, dtype=base_lr.dtype) if isinstance(base_lr, torch.Tensor) else lr_dora
        for key in decay_keys or ():
            dora_group[key] = weight_decay
        dora_group.pop("initial_lr", None)  # the scheduler must start from this group's own lr
        if "name" in dora_group:
            dora_group["name"] = f"{group['name']}_dora"
        if "lr_description" in dora_group:
            dora_group["lr_description"] = f"{group['lr_description']}_dora"
        out_groups.append(dora_group)
        out_desc.append(f"{description}_dora")
        moved += len(dora)
    lr_text = lr_dora if lr_dora is not None else f"same as the others ({default_lr})"
    if decay_keys is None:
        decay_text = "the optimizer's own (its arguments could not be read)"
    else:
        decay_text = f"{weight_decay} ({', '.join(decay_keys) or 'optimizer has none'})"
    logger.info(f"DoRA magnitudes: {moved} tensors in their own optimizer group(s), lr {lr_text}, weight decay {decay_text}")
    return out_groups, out_desc


def group_sizes(trainable_params) -> Optional[List[int]]:
    """Number of parameters in each optimizer group (a plain tensor list is one group); None when a group's params is not a
    list."""
    if isinstance(trainable_params, (list, tuple)) and trainable_params and not isinstance(trainable_params[0], dict):
        return [len(trainable_params)]
    sizes = []
    for group in trainable_params:
        params = group.get("params") if isinstance(group, dict) else None
        if not isinstance(params, (list, tuple)):
            return None
        sizes.append(len(params))
    return sizes


def saved_group_sizes(resume_dir: str) -> Optional[List[int]]:
    """Number of parameters in each optimizer group of an accelerate state folder (its optimizer.bin, memory-mapped: the
    moments are not read); None when there is no optimizer.bin or it cannot be read."""
    path = os.path.join(resume_dir, "optimizer.bin")
    if not os.path.isfile(path):
        return None
    try:
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        return [len(group["params"]) for group in state["param_groups"]]
    except Exception as e:
        logger.info(f"could not read the optimizer groups of {path}: {e}")
        return None


def keep_pre_split_layout(before: Optional[Sequence[int]], after: Optional[Sequence[int]],
                          saved: Optional[Sequence[int]]) -> bool:
    """True when the saved optimizer groups have the sizes of the layout without the DoRA group (``before``) and not those of
    the split one (``after``)."""
    if saved is None or before is None:
        return False
    return list(saved) == list(before) and list(saved) != list(after or ())


def resume_predates_split(args, before, after) -> bool:
    """True when --resume is a local state saved before the DoRA magnitudes had their own group: its optimizer groups match
    ``before`` (the groups without the split) and not ``after``, so load_state would fail on the split layout. Warns then."""
    resume = getattr(args, "resume", None)
    if not resume or getattr(args, "resume_from_huggingface", False):
        return False
    saved = saved_group_sizes(resume)
    if not keep_pre_split_layout(group_sizes(before), group_sizes(after), saved):
        return False
    logger.warning(f"resume state {resume} predates the DoRA optimizer group (saved groups {saved}, split {group_sizes(after)}):"
                   " keeping the saved groups; learning_rate_DoRA / weight_decay_DoRA do not apply to this run")
    return True


def drop_dora_group(args) -> None:
    """The run keeps the DoRA magnitudes in their source groups: clear --learning_rate_DoRA / --weight_decay_DoRA so the
    metadata does not record values that were not applied."""
    args.learning_rate_DoRA = None
    args.weight_decay_DoRA = None


def mark_weight_decay_unknown(args) -> None:
    """The optimizer's weight-decay arguments could not be read: the DoRA magnitudes keep the optimizer's own decay, so the
    saved metadata says so instead of weight_decay_DoRA."""
    args.weight_decay_DoRA_unknown = True


def metadata(args) -> Dict[str, str]:
    out = {}
    value = dora_learning_rate(args)
    if value is not None:
        out["ss_learning_rate_DoRA"] = str(value)
    if getattr(args, "weight_decay_DoRA", 0.0) is not None:  # None: no DoRA group in this run (drop_dora_group)
        unknown = getattr(args, "weight_decay_DoRA_unknown", False)
        out["ss_weight_decay_DoRA"] = "optimizer default" if unknown else str(dora_weight_decay(args))
    return out
