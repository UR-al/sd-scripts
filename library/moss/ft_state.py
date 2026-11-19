# MOSS-Image full fine-tune: the training-state directory (docs/moss_full_finetune_design.md 4.7).
#
#   <name>/
#     train_state.json             progress (global step, completed epochs, batches done in the epoch, steps per epoch,
#                                  GA, LR schedule), optimizer mode + hyperparameters, slot list hash, last window grad
#                                  norm, skip streak, the fp16 loss scaler (null without fp16), torch version, args
#                                  digest (+ the sanitized args)
#     master.safetensors           fp32, every Base key incl. frozen parameters and rope.bands: a loadable fp32 DiT
#     exp_avg.safetensors          fp32, the trained parameters' names (adamw_host, came_host)
#     exp_avg_sq.safetensors       fp32 (adamw_host)            | came_factored.safetensors (came_host)
#     optimizer_steps.safetensors  int64 per trained parameter (adamw_host, came_host)
#     optimizer.pt                 adafactor_fused only: the torch optimizer's state_dict (no host pool in that mode)
#     rotation.json, rotation_optimizer.safetensors   carousel only (ft_carousel's writer through `extra_writer`): the
#                                  rotation config / active set, the optimizer state of every unit that has one
#     param_groups.pt              adamw_host, came_host: every param group's values but the parameters (lr, initial_lr
#                                  and what an LR scheduler keeps there; optimizer.pt holds them in adafactor_fused)
#     scheduler.pt                 lr_scheduler.state_dict()
#     rng.pt                       torch CPU (+ CUDA), python random, numpy states
#     COMPLETE.json                written last: format "moss-ft-state/1", every file's byte size
#
# Writing (write_state): into <name>.tmp/, every file fsync'ed, COMPLETE.json last; then an existing <name> becomes
# <name>.old, <name>.tmp becomes <name>, <name>.old is deleted. At every moment a complete copy exists. Call it only
# after a window's drain (the pool is stable then). The pool views are written without copies.
#
# Loading (load_state): COMPLETE.json and the recorded sizes are required (else refused, naming the newest complete
# state next to it; an accelerate state directory is refused as such), the slot list hash and the optimizer mode must
# match (hyperparameter and argument changes are warnings); the files are read in place into the pool (peak +1 tensor
# for the frozen ones), the compute copies set from the masters, the param groups' scheduled values (lr, initial_lr,
# a scheduler's own keys), the scheduler and the RNG states restored.
#
# Keep policy (apply_keep_policy): the newest save_last_n_epochs_state epoch states and save_last_n_steps_state step
# states of this run, the pinned LR-phase-boundary epoch states and the train-end state stay; nothing that is not a
# complete state of this format is ever deleted, nor the state just written, nor a state numbered past it (a run
# resumed from an earlier state leaves the abandoned run's later states alone until it passes them).

import hashlib
import json
import math
import os
import random
import re
import shutil
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from library.moss import ft_args, ft_host, ft_model, ft_safetensors
from library.moss.ft_safetensors import SafetensorsReader, TensorEntry, fsync_dir, write_safetensors
from library.utils import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)

PREFIX = ft_args.PREFIX
STATE_FORMAT = "moss-ft-state/1"
COMPLETE_FILE = "COMPLETE.json"
TRAIN_STATE_FILE = "train_state.json"
MASTER_FILE = "master.safetensors"
STEPS_FILE = "optimizer_steps.safetensors"
CAME_FILE = "came_factored.safetensors"
OPTIMIZER_FILE = "optimizer.pt"
PARAM_GROUPS_FILE = "param_groups.pt"
SCHEDULER_FILE = "scheduler.pt"
RNG_FILE = "rng.pt"
ROTATION_FILE = "rotation.json"  # carousel (stable file names: ft_budget.ROTATION_FORMAT, not the mode's display name)
ROTATION_OPTIMIZER_FILE = "rotation_optimizer.safetensors"
ACCELERATE_MARKERS = ("optimizer.bin", "random_states_0.pkl", "scheduler.bin", "pytorch_model.bin")
SENSITIVE_ARGS = ("wandb_api_key", "huggingface_token")
# kohya's state names (library/train_util.py) and their defaults when output_name is not set
EPOCH_STATE_NAME, STEP_STATE_NAME, LAST_STATE_NAME = "{}-{:06d}-state", "{}-step{:08d}-state", "{}-state"
DEFAULT_EPOCH_NAME, DEFAULT_STEP_NAME, DEFAULT_LAST_NAME = "epoch", "at", "last"


class StateError(ValueError):
    """A training state that cannot be resumed from: incomplete, truncated, another layout or optimizer mode."""


@dataclass
class TrainProgress:
    """Where training stands when a state is written (after a window's drain).
    global_step: optimizer steps done; completed_epochs: epochs finished; epoch_step: batches (micro-steps) of the
    current epoch already consumed (0 for an epoch-end state); steps_per_epoch / gradient_accumulation_steps /
    lr_schedule (LrPhaseScheduler.schedule()): recorded, compared on resume; images_seen: training images consumed so
    far (the validation log's x axis; None in states written before it was recorded)."""
    global_step: int
    completed_epochs: int
    epoch_step: int = 0
    steps_per_epoch: Optional[int] = None
    gradient_accumulation_steps: int = 1
    lr_schedule: Optional[dict] = None
    images_seen: Optional[int] = None


@dataclass
class ResumeInfo:
    path: str
    progress: TrainProgress
    mode: str
    last_grad_norm: Optional[float]
    skip_streak: int
    train_state: dict = field(repr=False, default_factory=dict)
    loss_scaler: Optional[dict] = None  # ft_scaler.DynamicLossScaler.state_dict() of an fp16 run, else None

    @property
    def start_epoch(self) -> int:
        """The 0-based epoch to continue in (epoch completed_epochs + 1 in kohya's 1-based numbering)."""
        return self.progress.completed_epochs

    @property
    def batches_to_skip(self) -> int:
        """Batches of start_epoch already trained (accelerate.skip_first_batches); 0 for an epoch-end state."""
        return self.progress.epoch_step


# ---- names --------------------------------------------------------------------------------------------------------
def epoch_state_name(output_name: Optional[str], epoch_no: int) -> str:
    return EPOCH_STATE_NAME.format(output_name or DEFAULT_EPOCH_NAME, int(epoch_no))


def step_state_name(output_name: Optional[str], step_no: int) -> str:
    return STEP_STATE_NAME.format(output_name or DEFAULT_STEP_NAME, int(step_no))


def last_state_name(output_name: Optional[str]) -> str:
    return LAST_STATE_NAME.format(output_name or DEFAULT_LAST_NAME)


def state_dir_of(args) -> str:
    """--moss_ft_state_dir, else output_dir."""
    return getattr(args, "moss_ft_state_dir", None) or args.output_dir


# ---- RNG ----------------------------------------------------------------------------------------------------------
def capture_rng_state() -> dict:
    """torch CPU (+ every CUDA device), python random and numpy states, as tensors and plain values only (rng.pt loads
    with torch.load(weights_only=True))."""
    name, keys, pos, has_gauss, cached = np.random.get_state()
    state = {"torch": torch.get_rng_state(), "python": random.getstate(),
             "numpy": {"name": str(name), "keys": torch.from_numpy(np.asarray(keys, dtype=np.int64).copy()),
                       "pos": int(pos), "has_gauss": int(has_gauss), "cached_gaussian": float(cached)}}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    version, internal, gauss = state["python"]
    random.setstate((version, tuple(internal), gauss))
    numpy = state["numpy"]
    np.random.set_state((numpy["name"], numpy["keys"].numpy().astype(np.uint32), numpy["pos"], numpy["has_gauss"],
                         numpy["cached_gaussian"]))
    cuda = state.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        if len(cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda)
        else:
            logger.warning(f"{PREFIX}the state has CUDA RNG states of {len(cuda)} devices, this process sees"
                           f" {torch.cuda.device_count()}: the CUDA RNG is not restored")


# ---- small helpers ------------------------------------------------------------------------------------------------
def _json_safe(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return _json_safe(value.item())
    return str(value)


def sanitized_args(args) -> dict:
    """vars(args) as JSON values, without secrets (kohya's sensitive keys) and private runtime attributes (_*)."""
    if args is None:
        return {}
    return {k: _json_safe(v) for k, v in sorted(vars(args).items())
            if k not in SENSITIVE_ARGS and not k.startswith("_")}


def args_digest(sanitized: dict) -> str:
    return hashlib.sha256(json.dumps(sanitized, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


SCHEDULED_KEYS = ("lr", "initial_lr")  # set by the LR scheduler, not by the run's optimizer settings


def _own_keys(optimizer) -> Optional[set]:
    """The optimizer's own hyperparameter names (its defaults: lr, betas, eps, weight_decay, ...) plus the group label
    "name"; None for an optimizer without defaults. Other group keys belong to an LR scheduler (initial_lr, a restart
    scheduler's cycle counters, ...)."""
    defaults = getattr(optimizer, "defaults", None)
    return None if defaults is None else set(defaults) | {"name"}


def hyperparameters(optimizer) -> List[dict]:
    """Each param group's optimizer settings: the optimizer's own keys without the scheduled LR (compared on resume; a
    scheduler's keys in the groups are restored, not compared)."""
    own = _own_keys(optimizer)
    return [{k: _json_safe(v) for k, v in group.items()
             if k != "params" and k not in SCHEDULED_KEYS and (own is None or k in own)}
            for group in optimizer.param_groups]


def _restore_scheduled_values(optimizer, saved_groups, path: str) -> None:
    """Resume (adamw_host / came_host, whose optimizer state is not loaded through Optimizer.load_state_dict): put the
    saved lr, initial_lr and the keys an LR scheduler keeps in the param groups back, before the scheduler's own
    load_state_dict (which never touches the groups: a LambdaLR would leave the freshly built scheduler's step-0 LR, a
    recursive or cycle-keeping scheduler would continue from the wrong values). The optimizer's own settings (betas,
    eps, weight_decay, ...) stay as this run configures them; a change is warned about (hyperparameters)."""
    groups = optimizer.param_groups
    if not isinstance(saved_groups, (list, tuple)) or len(saved_groups) != len(groups):
        raise StateError(f"{PREFIX}{path}: {PARAM_GROUPS_FILE} holds {len(saved_groups or ())} param groups, the"
                         f" optimizer has {len(groups)}")
    own = _own_keys(optimizer) or set()
    for group, saved in zip(groups, saved_groups):
        for key, value in dict(saved).items():
            if key == "params" or (key in own and key not in SCHEDULED_KEYS):
                continue
            group[key] = value


def _write_json(path: str, value, fsync: bool) -> int:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=1, sort_keys=False)
        f.write("\n")
        f.flush()
        if fsync:
            os.fsync(f.fileno())
    return os.path.getsize(path)


def _torch_save(path: str, value, fsync: bool) -> int:
    with open(path, "wb") as f:
        torch.save(value, f)
        f.flush()
        if fsync:
            os.fsync(f.fileno())
    return os.path.getsize(path)


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---- writing ------------------------------------------------------------------------------------------------------
def _moment_entries(pool: ft_host.HostPool, slots: Sequence[ft_host.ParamSlot], role: str) -> List[TensorEntry]:
    return [TensorEntry(s.name, torch.float32, tuple(s.shape), lambda s=s: pool.view(role, s).view(s.shape))
            for s in slots]


def _steps_entries(pool: ft_host.HostPool, slots: Sequence[ft_host.ParamSlot]) -> List[TensorEntry]:
    return [TensorEntry(s.name, torch.int64, (), lambda s=s: pool.steps[s.index].to(torch.int64)) for s in slots]


def _extra_entries(optimizer) -> List[TensorEntry]:
    extra = optimizer.extra_state()
    return [TensorEntry(k, torch.float32, tuple(t.shape), lambda t=t: t) for k, t in extra.items()]


def write_state(state_dir: str, name: str, *, model, slots: Sequence[ft_host.ParamSlot],
                pool: Optional[ft_host.HostPool], optimizer, lr_scheduler, mode: str, progress: TrainProgress,
                last_grad_norm: Optional[float] = None, skip_streak: int = 0, args=None, fsync: bool = True,
                loss_scaler=None, extra_writer: Optional[Callable] = None) -> str:
    """Write the training state <state_dir>/<name> atomically (see the module comment). pool None: adafactor_fused
    (the model values and the torch optimizer's state_dict). last_grad_norm / skip_streak: the pipeline's counters
    (GradPipeline.last_grad_norm / skip_streak). loss_scaler: the fp16 run's ft_scaler.DynamicLossScaler (its
    state_dict goes into train_state.json; None: null). extra_writer(tmp_dir, fsync) -> ({file: size}, dict): a mode
    that keeps its optimizer state itself (carousel) writes its files and gives what goes into train_state.json under
    "extra"; the host modes' moment / step / CAME files are then not written. Returns the state's path."""
    if pool is None and mode != ft_args.ADAFACTOR_FUSED:
        raise ValueError(f"{PREFIX}write_state: mode {mode} keeps its optimizer state in the host pool; pool is None")
    os.makedirs(state_dir, exist_ok=True)
    final = os.path.join(state_dir, name)
    tmp, old = final + ".tmp", final + ".old"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)  # an earlier write of this name was interrupted
    os.makedirs(tmp)
    files: Dict[str, int] = {}
    try:
        files[MASTER_FILE] = write_safetensors(os.path.join(tmp, MASTER_FILE),
                                               ft_model.dit_entries(model, pool, torch.float32), {"format": "pt"},
                                               fsync=fsync)
        extra_train_state = None
        if extra_writer is not None:
            written, extra_train_state = extra_writer(tmp, fsync)
            files.update(written)
        if pool is not None:
            for role in pool.roles:
                if role != "master":
                    files[f"{role}.safetensors"] = write_safetensors(os.path.join(tmp, f"{role}.safetensors"),
                                                                     _moment_entries(pool, slots, role), fsync=fsync)
            if extra_writer is None:
                files[STEPS_FILE] = write_safetensors(os.path.join(tmp, STEPS_FILE), _steps_entries(pool, slots),
                                                      fsync=fsync)
            if extra_writer is None and hasattr(optimizer, "extra_state"):
                files[CAME_FILE] = write_safetensors(os.path.join(tmp, CAME_FILE), _extra_entries(optimizer),
                                                     fsync=fsync)
            # the LR and what an LR scheduler keeps in the groups (Optimizer.load_state_dict restores them in mode b)
            files[PARAM_GROUPS_FILE] = _torch_save(os.path.join(tmp, PARAM_GROUPS_FILE),
                                                   ft_args.param_group_values(optimizer), fsync)
        else:
            files[OPTIMIZER_FILE] = _torch_save(os.path.join(tmp, OPTIMIZER_FILE), optimizer.state_dict(), fsync)
        if lr_scheduler is not None:
            files[SCHEDULER_FILE] = _torch_save(os.path.join(tmp, SCHEDULER_FILE), lr_scheduler.state_dict(), fsync)
        files[RNG_FILE] = _torch_save(os.path.join(tmp, RNG_FILE), capture_rng_state(), fsync)
        sanitized = sanitized_args(args)
        norm = None if last_grad_norm is None or not math.isfinite(float(last_grad_norm)) else float(last_grad_norm)
        train_state = {"format": STATE_FORMAT, **asdict(progress), "mode": mode,
                       "hyperparameters": hyperparameters(optimizer),
                       "lr": [_json_safe(g.get("lr")) for g in optimizer.param_groups],
                       "slot_hash": ft_host.slot_list_hash(slots), "slots": len(slots),
                       "trained_numel": int(sum(s.numel for s in slots)), "last_grad_norm": norm,
                       "skip_streak": int(skip_streak),
                       "loss_scaler": None if loss_scaler is None else _json_safe(loss_scaler.state_dict()),
                       "torch_version": torch.__version__,
                       "args_digest": args_digest(sanitized), "args": sanitized}
        if extra_train_state is not None:
            train_state["extra"] = _json_safe(extra_train_state)
        train_state["lr_schedule"] = _json_safe(progress.lr_schedule)
        files[TRAIN_STATE_FILE] = _write_json(os.path.join(tmp, TRAIN_STATE_FILE), train_state, fsync)
        for file_name, size in files.items():  # what is on disk is what was written
            if os.path.getsize(os.path.join(tmp, file_name)) != size:
                raise OSError(f"{PREFIX}{file_name} in {tmp} is not {size} bytes after writing")
        _write_json(os.path.join(tmp, COMPLETE_FILE), {"format": STATE_FORMAT, "name": name,
                                                       "global_step": int(progress.global_step), "mode": mode,
                                                       "files": files}, fsync)
        if fsync:
            fsync_dir(tmp)
        if os.path.exists(old):
            shutil.rmtree(old)  # left by an interrupted replace; the new copy in tmp is complete
        if os.path.exists(final):
            os.replace(final, old)
        try:
            os.replace(tmp, final)
        except BaseException:
            if not os.path.exists(final) and os.path.exists(old):
                os.replace(old, final)  # put the previous copy back under its name
            raise
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if fsync:
        fsync_dir(state_dir)
    if os.path.exists(old):
        shutil.rmtree(old)
    total = sum(files.values())
    logger.info(f"saved the training state {final} (step {progress.global_step}, {total / 1e9:.2f} GB)")
    return final


# ---- verifying / finding ------------------------------------------------------------------------------------------
def _complete_info(path: str) -> Optional[dict]:
    """COMPLETE.json of a complete state of this format whose files all have their recorded sizes, else None."""
    marker = os.path.join(path, COMPLETE_FILE)
    try:
        complete = _read_json(marker)
        if complete.get("format") != STATE_FORMAT:
            return None
        for file_name, size in complete["files"].items():
            if os.path.getsize(os.path.join(path, file_name)) != int(size):
                return None
        return complete
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _written_at(path: str) -> int:
    """When the state was completed: the modification time of its COMPLETE.json (written last; copies keep it)."""
    try:
        return os.stat(os.path.join(path, COMPLETE_FILE)).st_mtime_ns
    except OSError:
        return -1


def complete_states(state_dir: str) -> List[str]:
    """The complete states directly under state_dir, newest first: the most recently written (COMPLETE.json's time),
    then the higher global step. Write order, not the step: after a resume from an earlier state into the same folder
    the abandoned run's later states have higher steps but are not the run's newest."""
    found = []
    try:
        entries = list(os.scandir(state_dir))
    except OSError:
        return []
    for entry in entries:
        if not entry.is_dir():
            continue
        complete = _complete_info(entry.path)
        if complete is not None:
            found.append((_written_at(entry.path), int(complete.get("global_step", -1)), entry.path))
    return [path for _, _, path in sorted(found, reverse=True)]


def _newest_hint(path: str) -> str:
    folder = os.path.dirname(os.path.abspath(path))
    others = [p for p in complete_states(folder) if os.path.abspath(p) != os.path.abspath(path)]
    if not others:
        return f" There is no complete state in {folder}."
    newest = others[0]
    return (f" The newest complete state (the last written) in {folder} is {os.path.basename(newest)} (step"
            f" {_complete_info(newest)['global_step']}).")


def verify_state(path: str) -> dict:
    """COMPLETE.json of the state at `path`; StateError (naming the newest complete state next to it) when it is not a
    directory, an accelerate state, has no COMPLETE.json, another format, or a file is missing or of another size."""
    if not os.path.isdir(path):
        raise StateError(f"{PREFIX}the training state {path} is not a directory." + _newest_hint(path))
    marker = os.path.join(path, COMPLETE_FILE)
    if not os.path.isfile(marker):
        if any(os.path.exists(os.path.join(path, m)) for m in ACCELERATE_MARKERS):
            raise StateError(f"{PREFIX}{path} is an accelerate state (optimizer.bin / random_states_0.pkl, no"
                             f" {COMPLETE_FILE}): moss_train.py resumes only its own state format (design 4.7)."
                             + _newest_hint(path))
        inside = complete_states(path)
        if inside:  # --resume <output_dir>: the folder that holds the states, not one of them
            newest = inside[0]
            raise StateError(f"{PREFIX}{path} is a folder of training states, not a state: --resume takes one of the"
                             f" state folders inside it. The newest complete one (the last written) is"
                             f" {os.path.basename(newest)} (step {_complete_info(newest)['global_step']}): resume ="
                             f" {newest!r}")
        raise StateError(f"{PREFIX}{path} has no {COMPLETE_FILE}: the state was not finished (interrupted while it was"
                         " written?) and cannot be resumed from." + _newest_hint(path))
    try:
        complete = _read_json(marker)
        files = dict(complete["files"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise StateError(f"{PREFIX}{marker} is unreadable ({error})." + _newest_hint(path)) from None
    if complete.get("format") != STATE_FORMAT:
        raise StateError(f"{PREFIX}{path} has format {complete.get('format')!r}, this script reads {STATE_FORMAT!r}.")
    for file_name, size in files.items():
        file_path = os.path.join(path, file_name)
        if not os.path.isfile(file_path):
            raise StateError(f"{PREFIX}{path}: {file_name} is missing." + _newest_hint(path))
        actual = os.path.getsize(file_path)
        if actual != int(size):
            raise StateError(f"{PREFIX}{path}: {file_name} has size {actual} bytes, {COMPLETE_FILE} recorded {size}"
                             " (truncated or replaced)." + _newest_hint(path))
    return complete


# ---- loading ------------------------------------------------------------------------------------------------------
def _reader(path: str) -> SafetensorsReader:
    try:
        return SafetensorsReader(path)
    except ft_safetensors.SafetensorsError as error:
        raise StateError(f"{PREFIX}{error}") from None


def _expect_keys(reader: SafetensorsReader, want: Sequence[str], what: str) -> None:
    have, want = set(reader.keys()), list(want)
    missing = [k for k in want if k not in have]
    extra = sorted(have - set(want))
    if missing or extra:
        raise StateError(f"{PREFIX}{reader.path} does not hold the {what} of this run: missing {missing[:8]}"
                         f" ({len(missing)}), unexpected {extra[:8]} ({len(extra)})")


@torch.no_grad()
def _load_master(path: str, model, slots, pool) -> None:
    params = dict(model.named_parameters())
    slot_of = {s.name: s for s in slots} if pool is not None else {}
    with _reader(path) as reader:
        _expect_keys(reader, sorted(model.state_dict()), "Base keys")
        for key in reader.keys():
            if key == ft_model.ROPE_KEY:
                continue  # derived constants
            info, param = reader.info(key), params[key]
            if tuple(info.shape) != tuple(param.shape):
                raise StateError(f"{PREFIX}{path}: {key} has shape {list(info.shape)}, the model {list(param.shape)}")
            slot = slot_of.get(key)
            if slot is not None:
                master = pool.view("master", slot)
                if info.dtype != torch.float32:
                    raise StateError(f"{PREFIX}{path}: {key} is {info.dtype}, a master is fp32")
                reader.read(key, master)  # in place
                param.copy_(master.view(slot.shape))  # the compute copy: the master rounded to its dtype
            else:
                param.copy_(reader.read(key))


def _load_pool_role(path: str, pool, slots, role: str) -> None:
    with _reader(path) as reader:
        _expect_keys(reader, [s.name for s in slots], role)
        for slot in slots:
            reader.read(slot.name, pool.view(role, slot))


def _load_steps(path: str, pool, slots) -> None:
    with _reader(path) as reader:
        _expect_keys(reader, [s.name for s in slots], "optimizer steps")
        for slot in slots:
            value = int(reader.read(slot.name).reshape(-1)[0])
            if value < 0 or value >= 1 << 24:
                raise StateError(f"{PREFIX}{path}: {slot.name} has step count {value}")
            pool.steps[slot.index] = float(value)


def _load_extra(path: str, optimizer) -> None:
    extra = optimizer.extra_state()
    with _reader(path) as reader:
        _expect_keys(reader, list(extra), "CAME factored statistics")
        for key, tensor in extra.items():
            reader.read(key, tensor)


def _restore_state_dtypes(optimizer, saved: dict) -> None:
    """torch.optim.Optimizer.load_state_dict casts every floating state tensor (but "step") to its parameter's dtype:
    Adafactor's fp32 statistics of a bf16 joint Linear (adafactor_fused) would continue in bf16. Put the saved dtypes
    back (on the device load_state_dict chose)."""
    params = [p for group in optimizer.param_groups for p in group["params"]]
    for index, values in (saved.get("state") or {}).items():
        state = optimizer.state.get(params[int(index)])
        if state is None:
            continue
        for key, value in values.items():
            now = state.get(key)
            if torch.is_tensor(value) and torch.is_tensor(now) and now.dtype != value.dtype:
                state[key] = value.to(device=now.device)


def _compare(saved, now, what: str) -> None:
    if saved is not None and saved != now:
        logger.warning(f"{PREFIX}{what} changed since the state was saved: saved {saved}, now {now}")


def load_state(path: str, *, model, slots: Sequence[ft_host.ParamSlot], pool: Optional[ft_host.HostPool], optimizer,
               lr_scheduler, mode: str, args=None, steps_per_epoch: Optional[int] = None,
               restore_rng: bool = True, extra_reader: Optional[Callable] = None) -> ResumeInfo:
    """Resume from the state at `path` into the run's objects (made as for a fresh start: model and pool from
    load_moss_for_full_ft, the optimizer bound to the slots, the scheduler from the current config). Refuses (StateError)
    an incomplete / truncated / accelerate state, another optimizer mode or another trainable set; warns when the
    hyperparameters, the steps per epoch (this run's, when given), GA or other arguments changed (the LR phase
    scheduler warns about its own schedule). The param groups' scheduled values (lr, initial_lr, what the scheduler keeps
    there) come back from param_groups.pt before the scheduler's state (modes a / c; mode b: optimizer.pt), so the
    first resumed window steps with the LR the saved run would have used. The caller continues at
    info.progress.global_step, in epoch info.start_epoch, skipping info.batches_to_skip batches, and restores the
    pipeline's counters: pipeline.restore_counters(info.last_grad_norm, info.skip_streak), and an fp16 run's loss
    scaler: ft_scaler.restore(scaler, info.loss_scaler). extra_reader(path, files, train_state): the counterpart of
    write_state's extra_writer (carousel), called after the masters are loaded, instead of the host modes' moment /
    step / CAME files."""
    complete = verify_state(path)
    try:
        state = _read_json(os.path.join(path, TRAIN_STATE_FILE))
    except (OSError, ValueError) as error:
        raise StateError(f"{PREFIX}{path}: {TRAIN_STATE_FILE} is unreadable ({error})") from None
    if state.get("format") != STATE_FORMAT:
        raise StateError(f"{PREFIX}{path}: {TRAIN_STATE_FILE} has format {state.get('format')!r}")
    same_rotation = mode == ft_args.CAROUSEL and ROTATION_FILE in (complete.get("files") or {})
    if state.get("mode") != mode and not same_rotation:
        raise StateError(f"{PREFIX}{path} was trained with optimizer mode {state.get('mode')}, this run uses {mode}:"
                         " the optimizer states differ (set optimizer_type back, or start from its master.safetensors as"
                         " a new model)")
    slot_hash = ft_host.slot_list_hash(slots)
    if state.get("slot_hash") != slot_hash:
        raise StateError(f"{PREFIX}{path} holds another trainable set: {state.get('slots')} tensors /"
                         f" {state.get('trained_numel')} parameters (hash {str(state.get('slot_hash'))[:12]}), this run"
                         f" trains {len(slots)} tensors / {sum(s.numel for s in slots)} parameters (hash {slot_hash[:12]})."
                         " Use the --moss_ft_freeze of the saved run")
    _compare(state.get("hyperparameters"), hyperparameters(optimizer), "the optimizer hyperparameters")
    progress = TrainProgress(**{k: state.get(k) for k in TrainProgress.__dataclass_fields__})
    if steps_per_epoch is not None:
        _compare(progress.steps_per_epoch, int(steps_per_epoch),
                 "steps per epoch (dataset size or GA; epoch_step and the epoch of the resume are the saved run's)")
    if args is not None:
        ga = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
        _compare(progress.gradient_accumulation_steps, ga, "gradient_accumulation_steps")
        now = sanitized_args(args)
        if state.get("args_digest") != args_digest(now):
            saved_args = state.get("args") or {}
            changed = sorted(k for k in set(saved_args) | set(now) if k != "resume" and saved_args.get(k) != now.get(k))
            if changed:
                logger.warning(f"{PREFIX}arguments changed since the state was saved: {changed[:16]}"
                               f"{' ...' if len(changed) > 16 else ''}")
    if state.get("torch_version") != torch.__version__:
        logger.info(f"{PREFIX}the state was saved with torch {state.get('torch_version')}, this is {torch.__version__}")
    files = complete["files"]

    _load_master(os.path.join(path, MASTER_FILE), model, slots, pool)
    if extra_reader is not None:
        extra_reader(path, files, state)
    if pool is not None:
        for role in pool.roles:
            if role != "master":
                if f"{role}.safetensors" not in files:
                    raise StateError(f"{PREFIX}{path} has no {role}.safetensors")
                _load_pool_role(os.path.join(path, f"{role}.safetensors"), pool, slots, role)
        if extra_reader is None:
            _load_steps(os.path.join(path, STEPS_FILE), pool, slots)
        if extra_reader is None and hasattr(optimizer, "extra_state"):
            if hasattr(optimizer, "bind"):
                optimizer.bind(slots, pool)  # idempotent: allocates CAME's statistics when no pipeline did yet
            _load_extra(os.path.join(path, CAME_FILE), optimizer)
    else:
        saved = torch.load(os.path.join(path, OPTIMIZER_FILE), map_location="cpu", weights_only=True)
        optimizer.load_state_dict(saved)
        _restore_state_dtypes(optimizer, saved)
    groups_restored = pool is None  # adafactor_fused: optimizer.load_state_dict restored the param groups
    if pool is not None and PARAM_GROUPS_FILE in files:
        saved_groups = torch.load(os.path.join(path, PARAM_GROUPS_FILE), map_location="cpu", weights_only=True)
        _restore_scheduled_values(optimizer, saved_groups, path)
        groups_restored = True
    if lr_scheduler is not None:
        if SCHEDULER_FILE in files:
            lr_scheduler.load_state_dict(torch.load(os.path.join(path, SCHEDULER_FILE), map_location="cpu",
                                                    weights_only=True))
            if not groups_restored:  # a state without param_groups.pt: at least the LR of the next step
                logger.warning(f"{PREFIX}{path} has no {PARAM_GROUPS_FILE}: the param groups get the scheduler's last"
                               " LR; a scheduler that keeps more in them (restart cycles) restarts")
                for group, lr in zip(optimizer.param_groups, lr_scheduler.get_last_lr()):
                    group["lr"] = lr
        else:
            logger.warning(f"{PREFIX}{path} has no {SCHEDULER_FILE}: the LR scheduler starts from its beginning")
    if restore_rng:
        restore_rng_state(torch.load(os.path.join(path, RNG_FILE), map_location="cpu", weights_only=True))
    info = ResumeInfo(path=path, progress=progress, mode=mode, last_grad_norm=state.get("last_grad_norm"),
                      skip_streak=int(state.get("skip_streak") or 0), train_state=state,
                      loss_scaler=state.get("loss_scaler") or None)
    logger.info(f"resumed from the training state {path}: step {progress.global_step}, {progress.completed_epochs} epochs"
                f" done, {progress.epoch_step} batches into the next")
    return info


# ---- keep policy --------------------------------------------------------------------------------------------------
def pinned_phase_epochs(phases) -> List[int]:
    """The (whole) epochs at which every LR phase but the last ends: their epoch states are kept (design 4.6 / 4.7).
    A fractional phase end has no epoch state."""
    ends = ft_args.lr_phase_end_epochs(phases)[:-1]
    return [int(round(e)) for e in ends if abs(e - round(e)) < 1e-9]


def keep_policy_from_args(args) -> dict:
    """apply_keep_policy's keyword arguments from the run's arguments: output_name, save_last_n_epochs_state,
    save_last_n_steps_state, and the phase-boundary epochs of --moss_ft_lr_phases unless
    --moss_ft_no_keep_phase_states."""
    pinned = []
    if getattr(args, "moss_ft_lr_phases", None) and not getattr(args, "moss_ft_no_keep_phase_states", False):
        pinned = pinned_phase_epochs(args.moss_ft_lr_phases)
    return dict(output_name=getattr(args, "output_name", None),
                keep_epoch_states=getattr(args, "save_last_n_epochs_state", None),
                keep_step_states=getattr(args, "save_last_n_steps_state", None), pinned_epochs=pinned)


_REPORTED_AHEAD: set = set()  # states past this run's position already logged (once per process)


def _position_of(written: Optional[str]) -> Dict[str, Optional[int]]:
    """{"epoch": completed epochs, "step": global step} of the state just written (its train_state.json)."""
    if written is None:
        return {"epoch": None, "step": None}
    try:
        state = _read_json(os.path.join(written, TRAIN_STATE_FILE))
        return {"epoch": int(state["completed_epochs"]), "step": int(state["global_step"])}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise StateError(f"{PREFIX}{written}: {TRAIN_STATE_FILE} is unreadable ({error})") from None


def apply_keep_policy(state_dir: str, output_name: Optional[str], *, keep_epoch_states: Optional[int],
                      keep_step_states: Optional[int], pinned_epochs: Iterable[int] = (),
                      written: Optional[str] = None) -> List[str]:
    """Delete this run's older complete states: keep the newest `keep_epoch_states` epoch states (None or < 1: all) and
    the newest `keep_step_states` step states, the epoch states of `pinned_epochs` and the train-end state. Only
    directories named like this run's states AND holding a complete state of this format are candidates.

    written: the state just written (pass it after every write_state). It is never deleted, and its train_state.json
    gives this run's position: epoch states numbered past its completed epochs and step states past its global step
    belong to another run in the same folder (one resumed from an earlier state leaves the abandoned run's later states
    behind) - they are neither deleted nor counted until this run passes them (a state of the same name is replaced by
    write_state). Without it every state counts (the newest by number). Returns the deleted paths."""
    patterns = (
        ("epoch", re.compile("^" + re.escape(output_name or DEFAULT_EPOCH_NAME) + r"-(\d{6,})-state$"), keep_epoch_states,
         {int(e) for e in pinned_epochs}),
        ("step", re.compile("^" + re.escape(output_name or DEFAULT_STEP_NAME) + r"-step(\d{8,})-state$"),
         keep_step_states, set()),
    )
    position = _position_of(written)
    protected = os.path.normcase(os.path.abspath(written)) if written is not None else None
    try:
        names = sorted(os.listdir(state_dir))
    except OSError:
        return []
    deleted = []
    for kind, pattern, keep, pinned in patterns:
        if keep is None or int(keep) < 1:
            continue
        found, ahead = [], []
        for name in names:
            match = pattern.match(name)
            path = os.path.join(state_dir, name)
            if match and os.path.isdir(path) and _complete_info(path) is not None:
                number = int(match.group(1))
                if position[kind] is not None and number > position[kind]:
                    ahead.append(path)
                else:
                    found.append((number, path))
        found.sort(reverse=True)
        for number, path in found[int(keep):]:
            if number in pinned or os.path.normcase(os.path.abspath(path)) == protected:
                continue
            logger.info(f"removing old {kind} state: {path}")
            shutil.rmtree(path)
            deleted.append(path)
        new = [path for path in ahead if path not in _REPORTED_AHEAD]
        if new:
            _REPORTED_AHEAD.update(new)
            logger.info(f"keeping {len(new)} {kind} state(s) numbered past this run's {kind} {position[kind]} (another"
                        f" run's, e.g. before a resume from an earlier state): {', '.join(os.path.basename(p) for p in new)}."
                        " The keep policy leaves them alone until this run passes them; delete them yourself if they are"
                        " not needed")
    return deleted
