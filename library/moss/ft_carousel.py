# MOSS-Image full fine-tune, Carousel mode (docs/moss_ft_carousel_plan.md): rotating active blocks (LISA / BAdam style).
#
# Only K of the 44 units (2 text refiner blocks, 42 joint blocks; 45 with the stem unit of --moss_carousel_always
# rotate_stem; ft_carousel_units) and the always group (the small stem / head tensors, --moss_carousel_always) are
# trainable at a time. Every --moss_carousel_period optimizer steps a
# new set of K units is chosen (ft_carousel_units.Selector), so over the run every block gets real full-weight updates.
#
#   GPU (per trainable tensor): the compute copy (the module parameter, fp16 / bf16 joint Linears, fp32 otherwise -
#     unchanged from the other modes, every block keeps it), and for the ACTIVE ones only: the fp32 master (a view of
#     its group's flat device buffer, wrapped in a persistent `handle` parameter the optimizer steps), an fp32 gradient
#     accumulator, the optimizer state. Inactive compute copies have requires_grad False: autograd computes no weight
#     gradient for them, and below the lowest block that needs gradients nothing runs with grad at all
#     (ft_carousel_units.apply_grad_floor).
#   host: the fp32 master of every trainable tensor (ft_host.HostPool, role "master": authoritative for the inactive
#     units), and with --moss_carousel_state keep the optimizer states of the inactive units.
#
# Window protocol (GradPipeline's interface, used by moss_train.py): the post-accumulate-grad hook of an active tensor
# copies / adds its gradient into the fp32 accumulator (exactly autograd's accumulation order) and drops it; after the
# window's last backward the engine unscales (fp16 loss scale) and checks for non-finite values (one sync), skips the
# whole window on an overflow, clips exactly over the trainable tensors, steps the optimizer on the device and writes
# every stepped master back into its compute copy (the master rounded to the compute dtype: invariant I1).
#
# Switch (after_step, between windows): outgoing units first (master D2H into the pool, state D2H with keep, grads
# off, device buffers freed), then the incoming ones (master H2D, state H2D, I1 checked per tensor, grads on). fp32
# round trips are bit-exact; the compute copies never leave the GPU.
#
# Saving: flush_to_host() puts the active masters into the pool (the DiT save and master.safetensors read the pool);
# state_writer / state_reader are ft_state's extra_writer / extra_reader (rotation.json, rotation_optimizer.safetensors).

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from library.moss import ft_args, ft_budget, ft_host
from library.moss import ft_carousel_optim as cco
from library.moss import ft_carousel_units as cu
from library.moss.ft_safetensors import SafetensorsReader, TensorEntry, write_safetensors
from library.utils import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)

CAROUSEL = ft_budget.CAROUSEL
ROTATION_FORMAT = ft_budget.ROTATION_FORMAT
LOG = CAROUSEL.title()  # the log prefix
PREFIX = ft_args.PREFIX
GIB = 1 << 30
ALWAYS = "always"  # the always group's key among the device groups
ROTATION_FILE = "rotation.json"
ROTATION_OPTIMIZER_FILE = "rotation_optimizer.safetensors"
NO_DECAY_PREFIXES = ("video_embedder.", "text_embedder.proj.")  # ft_model.NO_DECAY_PREFIXES


def scalar(name: str) -> str:
    """A per-step log key of this mode (derived from the mode's name)."""
    return f"{CAROUSEL}/{name}"


def param_groups(slots: Sequence[ft_host.ParamSlot], handles: Sequence[torch.nn.Parameter],
                 weight_decay: float = 0.0) -> List[dict]:
    """ft_model.param_groups over the handles: "decay" (>= 2-D weights) / "no_decay" (1-D, the input embedders), each
    in slot order; decided by the slots' shapes (a handle is empty while its unit is inactive)."""
    decay, no_decay = [], []
    for slot, handle in zip(slots, handles):
        (decay if len(slot.shape) >= 2 and not slot.name.startswith(NO_DECAY_PREFIXES) else no_decay).append(handle)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay), "name": "decay"})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "name": "no_decay"})
    return groups


def window_text(stats) -> str:
    """The carousel part of moss_train's periodic step line ("" for the other modes)."""
    extra = getattr(stats, "extra", None) or {}
    if scalar("active_units") not in extra:
        return ""
    return (f"; {CAROUSEL}: {int(extra[scalar('active_units')])} active units, lowest active joint block"
            f" {int(extra[scalar('lowest_active_block')])}, backward down to block {int(extra[scalar('grad_floor')])},"
            f" optimizer step {extra.get('time/optimizer_step', 0.0):.3f} s")


def _is_power_of_two(value) -> bool:
    value = float(value)
    return math.isfinite(value) and value > 0 and math.frexp(value)[0] == 0.5


class CarouselEngine:
    """The Carousel mode behind GradPipeline's interface (attach / begin_window / begin_microstep / end_microstep /
    check_skip_streak / restore_counters / close, last_grad_norm / skip_streak / pinned_bytes) plus after_step(step),
    flush_to_host(), state_writer / state_reader, activate(step) and switch(units).

    handles[i]: the optimizer's parameter of slot i (its data is the device master view while active, empty
    otherwise). units: ft_carousel_units.discover_units of the slots. always: the slot names trained at every step."""

    def __init__(self, model, pool: ft_host.HostPool, slots: Sequence[ft_host.ParamSlot], optimizer,
                 handles: Sequence[torch.nn.Parameter], *, kind: str, units: Sequence[cu.Unit], always: Sequence[str],
                 k: int, period: int, select: str = "random", seed: int = 0, state_policy: str = "keep",
                 state_dtype: str = "fp32", device="cpu", max_grad_norm: float = 0.0, max_skipped_steps: int = 8,
                 host_pinned: bool = False, always_policy: str = "small", seed_given: bool = True):
        self.model, self.pool, self.slots, self.optimizer = model, pool, list(slots), optimizer
        self.handles = list(handles)
        if len(self.handles) != len(self.slots):
            raise ValueError(f"{LOG}: one handle per slot")
        if pool.slot_hash() != ft_host.slot_list_hash(self.slots):
            raise ValueError(f"{LOG}: the pool was laid out for other slots")
        if kind not in cu.KINDS or state_policy not in cu.STATE_POLICIES or state_dtype not in cu.STATE_DTYPES:
            raise ValueError(f"{LOG}: kind {kind!r}, state {state_policy!r}, state_dtype {state_dtype!r}")
        self.kind, self.state_policy, self.state_dtype = kind, state_policy, state_dtype
        self.keep = state_policy == "keep"
        self.device = torch.device(device)
        self.units = list(units)
        self.period = int(period)
        if self.period < 1:
            raise ValueError(f"{LOG}: the period must be >= 1, got {period}")
        self.k = int(k)
        self.select_kind, self.seed, self.always_policy = select, int(seed), always_policy
        # seed_given False (no --moss_carousel_seed): a resume takes the saved seed, so the selection goes on as saved
        # even when --seed was unset (train_util.args_set_seed draws a new one per run)
        self.seed_given = bool(seed_given)
        self.selector = cu.Selector(len(self.units), self.k, select, self.seed)
        self.max_grad_norm = float(max_grad_norm or 0.0)
        self.max_skipped_steps = int(max_skipped_steps)
        self.host_pinned = bool(host_pinned)
        self.num_layers = len(getattr(model, "layers", ()))

        index_of = {s.name: s.index for s in self.slots}
        self.group_slots: Dict[Any, List[int]] = {}
        for unit in self.units:
            self.group_slots[unit.index] = [index_of[n] for n in unit.names]
        self.always_slots = [index_of[n] for n in always]
        if self.always_slots:
            self.group_slots[ALWAYS] = list(self.always_slots)
        self.trainable = sorted(i for idx in self.group_slots.values() for i in idx)
        if len(set(self.trainable)) != len(self.trainable):
            raise ValueError(f"{LOG}: a tensor is in two groups")

        self.active: Tuple[int, ...] = ()
        self.activated = False
        self.floor, self.text_grad = self.num_layers, False
        self._dev: Dict[Any, Dict[str, torch.Tensor]] = {}  # group -> {"master", "acc"}
        self._loc: Dict[int, Tuple[Any, int, int]] = {}  # slot -> (group, offset, numel) while active
        self.host_state: Dict[int, Dict[str, Any]] = {}  # keep: the states of inactive (and once-active) tensors
        self._pending: Dict[int, Dict[str, Any]] = {}  # states read from a training state, until activation
        self._filled: set = set()
        self._win: Optional[dict] = None
        self._j: Optional[int] = None
        self._hooks: list = []
        self._closed = False
        self._registered = False
        self._empty = torch.empty(0, dtype=torch.float32, device=self.device)
        self.last_grad_norm: Optional[float] = None
        self.skip_streak = 0
        self.switches = 0

    # ------------------------------------------------------------------------------------------------ properties
    @property
    def pinned_bytes(self) -> int:
        """Host bytes this engine page-locked (the master pool with --moss_carousel_host_pinned masters)."""
        return self.pool.nbytes()["master"] if self._registered else 0

    @property
    def active_slots(self) -> List[int]:
        return sorted(self._loc)

    @property
    def trainable_params(self) -> int:
        return sum(self.slots[i].numel for i in self._loc)

    def active_names(self) -> List[str]:
        return [self.units[u].name for u in self.active]

    @property
    def lowest_active_block(self) -> int:
        """The lowest joint block among the active units (num_layers when none is active: refiners / stem only)."""
        joints = [self.units[u].joint for u in self.active if self.units[u].joint is not None]
        return min(joints) if joints else self.num_layers

    # ------------------------------------------------------------------------------------------------ attach / close
    def attach(self) -> None:
        """Register the gradient hook of every tensor that can be trained (temporarily requires_grad), then switch
        every compute copy off until activation (the first begin_window, after a resume has loaded its state)."""
        if self._hooks:
            raise RuntimeError(f"{LOG}: already attached")
        for i in self.trainable:
            param = self.slots[i].param
            param.requires_grad_(True)
            self._hooks.append(param.register_post_accumulate_grad_hook(self._make_hook(i)))
        for slot in self.slots:
            slot.param.requires_grad_(False)
            slot.param.grad = None
        if self.host_pinned and self.device.type == "cuda" and not self._registered:  # pragma: no cover - CUDA only
            master = self.pool.roles["master"]
            result = torch.cuda.cudart().cudaHostRegister(master.data_ptr(), master.numel() * master.element_size(), 0)
            if int(getattr(result, "value", result)) != 0:
                raise RuntimeError(f"{LOG}: cudaHostRegister of the {master.numel() * 4 / GIB:.2f} GiB master pool failed"
                                   f" (error {result}): use --moss_{CAROUSEL}_host_pinned none")
            self._registered = True

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._closed = True
        self._win, self._j = None, None
        cu.clear_grad_floor(self.model)
        if self._registered:  # pragma: no cover - CUDA only
            try:
                torch.cuda.cudart().cudaHostUnregister(self.pool.roles["master"].data_ptr())
            except Exception:  # noqa: BLE001 - closing
                pass
            self._registered = False

    # ------------------------------------------------------------------------------------------------ activation
    def activate(self, global_step: int = 0) -> None:
        """The first active set: the always group and select(global_step // period) (a resumed run continues in its
        set and period position). States read by state_reader are used here (keep: the inactive units' ones go to
        the host store; reset: only the active ones are kept)."""
        if self.activated:
            raise RuntimeError(f"{LOG}: already activated")
        self._sync()
        units = self.selector(int(global_step) // self.period)
        if self.always_slots:
            self._activate(ALWAYS)
        for unit in units:
            self._activate(unit)
        if self.keep:
            for i, state in self._pending.items():
                self.host_state[i] = state
        self._pending.clear()
        self.active = tuple(units)
        self.activated = True
        self._apply_floor()
        logger.info(f"{LOG}: step {int(global_step)}: active units {self.active_names()}"
                    f" ({len(units)} of {len(self.units)}, {self.trainable_params / 1e6:.1f}M trainable parameters"
                    f" with the always group), lowest block with gradients {self.floor}")

    def _apply_floor(self) -> None:
        names = [self.slots[i].name for i in self._loc]
        self.floor, self.text_grad = cu.grad_floor(names, self.num_layers)
        cu.apply_grad_floor(self.model, self.floor, self.text_grad)

    def _sync(self) -> None:
        if self.device.type == "cuda":  # pragma: no cover - CUDA only
            torch.cuda.synchronize(self.device)

    def _pool_range(self, idx: Sequence[int]) -> Optional[Tuple[int, int]]:
        """(start, end) in the pool's flat master when the slots are consecutive in it, else None."""
        for a, b in zip(idx, idx[1:]):
            if b != a + 1:
                return None
        first, last = self.slots[idx[0]], self.slots[idx[-1]]
        return first.offset, last.offset + last.numel

    def _move_masters(self, group, to_device: bool) -> int:
        idx = self.group_slots[group]
        flat = self._dev[group]["master"]
        pool = self.pool.roles["master"]
        span = self._pool_range(idx)
        if span is not None:
            if to_device:
                flat.copy_(pool[span[0]:span[1]], non_blocking=self._registered)
            else:
                pool[span[0]:span[1]].copy_(flat)
        else:
            for i in idx:
                _, off, n = self._loc[i]
                if to_device:
                    flat[off:off + n].copy_(self.pool.view("master", self.slots[i]), non_blocking=self._registered)
                else:
                    self.pool.view("master", self.slots[i]).copy_(flat[off:off + n])
        return flat.numel() * 4

    @torch.no_grad()
    def _activate(self, group) -> int:
        idx = self.group_slots[group]
        total = sum(self.slots[i].numel for i in idx)
        master = torch.empty(total, dtype=torch.float32, device=self.device)
        acc = torch.empty(total, dtype=torch.float32, device=self.device)
        self._dev[group] = {"master": master, "acc": acc}
        off = 0
        for i in idx:
            self._loc[i] = (group, off, self.slots[i].numel)
            off += self.slots[i].numel
        moved = self._move_masters(group, to_device=True)
        self._sync()
        for i in idx:
            slot, handle = self.slots[i], self.handles[i]
            _, off, n = self._loc[i]
            handle.data = master[off:off + n].view(slot.shape)
            handle.grad = None
            state = self._pending.pop(i, None)
            if state is None and self.keep:
                state = self.host_state.get(i)
            self.optimizer.state.pop(handle, None)
            if state:
                self.optimizer.state[handle] = cco.state_to_device(self.kind, state, self.optimizer, self.device, handle)
                moved += cco.state_numel_bytes(state)
            compute = slot.param.detach()
            if not torch.equal(compute, handle.detach().to(compute.dtype)):
                raise RuntimeError(f"{LOG}: the compute copy of {slot.name} is not its fp32 master rounded to"
                                   f" {compute.dtype} (invariant I1 broken: drift between the host master and the GPU"
                                   " copy). Stop and resume from a saved state")
            slot.param.requires_grad_(True)
        return moved

    @torch.no_grad()
    def _deactivate(self, group) -> int:
        idx = self.group_slots[group]
        moved = self._move_masters(group, to_device=False)
        for i in idx:
            slot, handle = self.slots[i], self.handles[i]
            state = self.optimizer.state.pop(handle, None)
            if self.keep and state:
                self.host_state[i] = cco.state_to_host(self.kind, state, self.host_state.get(i), self.state_dtype)
                moved += cco.state_numel_bytes(state)
            elif not self.keep:
                self.host_state.pop(i, None)
            del state
            slot.param.requires_grad_(False)
            slot.param.grad = None
            handle.grad = None
            handle.data = self._empty
            del self._loc[i]
        del self._dev[group]
        return moved

    def after_step(self, global_step: int, max_steps: Optional[int] = None) -> Dict[str, float]:
        """Called after every optimizer step (global_step already incremented): at a period boundary the next set
        becomes active - not after the run's last step (global_step >= max_steps): nothing would train it, and a
        resume past it activates select(global_step // period) itself. Returns the switch's log scalars ({} without
        a switch)."""
        if not self.activated or int(global_step) % self.period:
            return {}
        if max_steps is not None and int(global_step) >= int(max_steps):
            return {}
        new = self.selector(int(global_step) // self.period)
        if tuple(new) == self.active:
            return {}
        return self.switch(new, global_step)

    def switch(self, new: Sequence[int], global_step: int = -1) -> Dict[str, float]:
        """Make `new` the active units: outgoing first (their memory is freed before the incoming one is allocated),
        then incoming; the backward floor follows."""
        if self._win is not None:
            raise RuntimeError(f"{LOG}: switch inside a window")
        new = tuple(sorted(int(u) for u in new))
        self._sync()
        start = time.perf_counter()
        outgoing = [u for u in self.active if u not in new]
        incoming = [u for u in new if u not in self.active]
        d2h = sum(self._deactivate(u) for u in outgoing)
        self._sync()
        h2d = sum(self._activate(u) for u in incoming)
        self._sync()
        self.active = new
        self._apply_floor()
        seconds = time.perf_counter() - start
        self.switches += 1
        logger.info(f"{LOG} switch at step {global_step}: out {[self.units[u].name for u in outgoing]}, in"
                    f" {[self.units[u].name for u in incoming]}, active {self.active_names()}, lowest block with"
                    f" gradients {self.floor}; D2H {d2h / 1e9:.2f} GB, H2D {h2d / 1e9:.2f} GB, {seconds:.2f} s")
        return {scalar("switch_s"): seconds, scalar("switch_d2h_gib"): d2h / GIB, scalar("switch_h2d_gib"): h2d / GIB}

    # ------------------------------------------------------------------------------------------------ the window
    def _make_hook(self, index: int):
        def hook(param):
            grad = param.grad
            group, off, n = self._loc[index]
            acc = self._dev[group]["acc"][off:off + n].view(param.shape)
            if index in self._filled:
                acc.add_(grad)
            else:
                acc.copy_(grad)
                self._filled.add(index)
            param.grad = None

        return hook

    def begin_window(self, n_micro: int, ctx: ft_host.StepContext) -> None:
        if self._closed:
            raise RuntimeError(f"{LOG}: closed")
        if self._win is not None:
            raise RuntimeError(f"{LOG}: begin_window before the previous window ended")
        if not _is_power_of_two(ctx.inv_grad_scale):
            raise ValueError(f"{LOG}: inv_grad_scale must be a power of two, got {ctx.inv_grad_scale!r}")
        if int(n_micro) < 1:
            raise ValueError(f"{LOG}: n_micro must be >= 1, got {n_micro}")
        if not self.activated:
            self.activate(ctx.step_index)
        self._win = {"n": int(n_micro), "ctx": ctx, "next": 0}
        self._filled = set()

    def begin_microstep(self, j: int) -> None:
        w = self._win
        if w is None or self._j is not None or int(j) != w["next"]:
            raise RuntimeError(f"{LOG}: begin_microstep({j}) out of order")
        self._j = int(j)
        w["next"] = self._j + 1

    def end_microstep(self) -> Optional[ft_host.WindowStats]:
        w, j = self._win, self._j
        if w is None or j is None:
            raise RuntimeError(f"{LOG}: end_microstep without begin_microstep")
        self._j = None
        if j < w["n"] - 1:
            return None
        return self._finish_window(w)

    def _finish_window(self, w: dict) -> ft_host.WindowStats:
        start = time.perf_counter()
        ctx = w["ctx"]
        inv = float(ctx.inv_grad_scale)
        idx = sorted(self._filled)
        grads = []
        for i in idx:
            group, off, n = self._loc[i]
            grads.append(self._dev[group]["acc"][off:off + n].view(self.slots[i].shape))
        overflow = False
        if grads:
            found = torch.zeros(1, dtype=torch.float32, device=self.device)
            inv_t = torch.full((1,), inv, dtype=torch.float32, device=self.device)
            torch._amp_foreach_non_finite_check_and_unscale_(grads, found, inv_t)  # x 1/S: exact (a power of two)
            overflow = bool(found.item())
        opt_s, clip_coef, grad_norm = 0.0, 1.0, float("nan")
        skipped: List[str] = []
        if overflow:
            skipped = [self.slots[i].name for i in idx]
            self.skip_streak += 1
            self.optimizer.step()  # no handle has a gradient: updates nothing (the LR scheduler sees the call)
            text = (f"{LOG} step {ctx.step_index}: non-finite gradients in {len(skipped)} tensors, the whole step is"
                    f" skipped (streak {self.skip_streak})")
            if inv != 1.0:
                logger.info(text + f"; fp16 loss scale {1.0 / inv:g}")
            else:
                logger.warning(text + f": {', '.join(skipped[:8])}{' ...' if len(skipped) > 8 else ''}")
        else:
            handles = [self.handles[i] for i in idx]
            for handle, grad in zip(handles, grads):
                handle.grad = grad
            if handles and self.max_grad_norm > 0:
                total = torch.nn.utils.clip_grad_norm_(handles, self.max_grad_norm)
                clip_coef = float(torch.clamp(self.max_grad_norm / (total + 1e-6), max=1.0))
            elif grads:
                total = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)), 2.0)
            else:
                total = torch.zeros(())
            opt_start = time.perf_counter()
            self.optimizer.step()
            for handle in handles:
                handle.grad = None
            self._copy_back(idx)
            self._sync()
            opt_s = time.perf_counter() - opt_start
            grad_norm = float(total)
            self.last_grad_norm = grad_norm
            self.skip_streak = 0
        self._win = None
        extra = {scalar("lowest_active_block"): float(self.lowest_active_block),
                 scalar("grad_floor"): float(self.floor),
                 scalar("grad_depth"): float(self.num_layers - self.floor),
                 scalar("active_units"): float(len(self.active)),
                 scalar("trainable_params"): float(self.trainable_params),
                 "time/optimizer_step": opt_s}
        return ft_host.WindowStats(step_index=ctx.step_index, n_micro=w["n"], grad_norm=grad_norm,
                                   clip_coef=clip_coef, clip_mode="exact" if self.max_grad_norm > 0 else "none",
                                   skipped_params=len(skipped), skipped_names=skipped, skip_streak=self.skip_streak,
                                   stepped_params=0 if overflow else len(idx), tail_s=time.perf_counter() - start,
                                   hook_wait_s=0.0, worker_busy_s=opt_s, loss_scale=1.0 / inv, overflow=overflow,
                                   extra=extra)

    @torch.no_grad()
    def _copy_back(self, idx: Sequence[int]) -> None:
        """compute copy <- master rounded to its dtype (I1); fp16: a master past 65504 stops the run."""
        fp16 = []
        for i in idx:
            param = self.slots[i].param
            param.copy_(self.handles[i])
            if param.dtype == torch.float16:
                fp16.append(i)
        if fp16:
            bad = torch.stack([torch.isinf(self.slots[i].param).any() for i in fp16])
            if bool(bad.any()):
                names = [self.slots[i].name for i, b in zip(fp16, bad.tolist()) if b]
                raise RuntimeError(f"{LOG}: the fp32 masters of {names[:8]} passed fp16's 65504: their fp16 compute"
                                   " copies would be inf. Resume from an earlier state with bf16 compute or a lower LR")

    def check_skip_streak(self, stats: ft_host.WindowStats) -> None:
        if self.max_skipped_steps > 0 and stats.skip_streak >= self.max_skipped_steps:
            raise RuntimeError(f"non-finite gradients made {LOG} skip {stats.skip_streak} optimizer steps in a row"
                               f" (--moss_ft_max_skipped_steps {self.max_skipped_steps}). The run diverged: resume from"
                               " an earlier state, e.g. with a lower learning rate.")

    def restore_counters(self, last_grad_norm: Optional[float], skip_streak: int = 0) -> None:
        self.last_grad_norm = None if last_grad_norm is None else float(last_grad_norm)
        self.skip_streak = int(skip_streak)

    # ------------------------------------------------------------------------------------------------ saving
    @torch.no_grad()
    def flush_to_host(self) -> None:
        """The active masters into the pool (the DiT save and master.safetensors read it); the device is unchanged."""
        if self._win is not None:
            raise RuntimeError(f"{LOG}: flush inside a window")
        self._sync()
        for group in list(self._dev):
            self._move_masters(group, to_device=False)
        self._sync()

    def config(self) -> Dict[str, Any]:
        return {"name": CAROUSEL, "optimizer": self.kind, "state_policy": self.state_policy,
                "state_dtype": self.state_dtype, "select": self.select_kind, "k": self.k, "period": self.period,
                "seed": self.seed, "always": self.always_policy}

    def _state_of(self, i: int) -> Optional[Dict[str, Any]]:
        if i in self._loc:
            return self.optimizer.state.get(self.handles[i]) or None
        if i in self._pending:
            return self._pending[i]
        return self.host_state.get(i) if self.keep else None

    def state_writer(self, folder: str, fsync: bool = True) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """ft_state's extra_writer: rotation_optimizer.safetensors (`<tensor>::<state key>` of every tensor that has
        an optimizer state, in its stored dtype; active ones read from the device one at a time) and rotation.json.
        Call flush_to_host() before the state's master file is written."""
        if self._win is not None:
            raise RuntimeError(f"{LOG}: state save inside a window")
        self._sync()
        entries, has_state = [], []
        for i in self.trainable:
            state = self._state_of(i)
            if not state:
                continue
            name = self.slots[i].name
            has_state.append(name)
            for key, value in cco.state_entries(self.kind, state):
                entries.append(TensorEntry(f"{name}::{key}", value.dtype, tuple(value.shape),
                                           lambda v=value: v.detach().to("cpu")))
        files = {ROTATION_OPTIMIZER_FILE: write_safetensors(os.path.join(folder, ROTATION_OPTIMIZER_FILE), entries,
                                                            {"format": ROTATION_FORMAT}, fsync=fsync)}
        info = dict(format=ROTATION_FORMAT, **self.config(), units=[u.name for u in self.units],
                    active=self.active_names(), has_state=has_state, switches=self.switches,
                    torch_version=torch.__version__, bitsandbytes_version=_bnb_version() if self.kind == "adamw8bit"
                    else None)
        path = os.path.join(folder, ROTATION_FILE)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1)
            f.write("\n")
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        files[ROTATION_FILE] = os.path.getsize(path)
        return files, {key: info[key] for key in ("format", "name", "optimizer", "state_policy", "state_dtype",
                                                    "select", "k", "period", "seed", "always", "active")}

    def state_reader(self, path: str, files: Dict[str, int], train_state: Dict[str, Any]) -> None:
        """ft_state's extra_reader: refuse another format / optimizer kind / unit list, warn about a changed rotation
        config, and keep every saved optimizer state until activate() (the masters were loaded by ft_state)."""
        from library.moss.ft_state import StateError

        if self.activated:
            raise RuntimeError(f"{LOG}: a state can only be read before the first window")
        if ROTATION_FILE not in files or ROTATION_OPTIMIZER_FILE not in files:
            raise StateError(f"{PREFIX}{path} is not a {CAROUSEL} state (no {ROTATION_FILE}): it was written by"
                             f" {train_state.get('mode')}; start from its master.safetensors as a new model instead")
        with open(os.path.join(path, ROTATION_FILE), "r", encoding="utf-8") as f:
            info = json.load(f)
        if info.get("format") != ROTATION_FORMAT:
            raise StateError(f"{PREFIX}{path}: {ROTATION_FILE} has format {info.get('format')!r}, this script reads"
                             f" {ROTATION_FORMAT!r}")
        if info.get("optimizer") != self.kind:
            raise StateError(f"{PREFIX}{path} holds {info.get('optimizer')} optimizer states, this run uses {self.kind}:"
                             " set optimizer_type back, or start from its master.safetensors as a new model")
        names = [u.name for u in self.units]
        if list(info.get("units") or []) != names:
            raise StateError(f"{PREFIX}{path} rotates other units ({len(info.get('units') or [])} saved, {len(names)}"
                             f" now): use the --moss_ft_freeze and {ft_args.carousel_flag('always')} of the saved run"
                             f" (saved always {info.get('always')!r}; rotate_stem adds the stem unit)")
        saved_seed = info.get("seed")
        if (not self.seed_given and isinstance(saved_seed, int) and not isinstance(saved_seed, bool)
                and saved_seed != self.seed):
            logger.info(f"{LOG}: no {ft_args.carousel_flag('seed')}: the selection continues with the saved seed"
                        f" {saved_seed} (this run's --seed gave {self.seed})")
            self.seed = int(saved_seed)
            self.selector = cu.Selector(len(self.units), self.k, self.select_kind, self.seed)
        changed = {key: (info.get(key), value) for key, value in self.config().items()
                   if key not in ("name", "optimizer") and info.get(key) != value}
        if changed:
            logger.warning(f"{PREFIX}{CAROUSEL} settings changed since the state was saved: "
                           + ", ".join(f"{key} {old!r} -> {new!r}" for key, (old, new) in changed.items())
                           + ". The active set follows the new settings from the saved step; kept states stay valid")
        index_of = {s.name: s.index for s in self.slots}
        grouped: Dict[str, Dict[str, torch.Tensor]] = {}
        with SafetensorsReader(os.path.join(path, ROTATION_OPTIMIZER_FILE)) as reader:
            for key in reader.keys():
                name, _, state_key = key.partition("::")
                if name not in index_of or not state_key:
                    raise StateError(f"{PREFIX}{path}: {ROTATION_OPTIMIZER_FILE} has {key!r}, not a state of this run")
                grouped.setdefault(name, {})[state_key] = reader.read(key)
        trainable = set(self.trainable)
        self._pending = {}
        for name, values in grouped.items():
            i = index_of[name]
            if i not in trainable:
                continue  # an always-group change (e.g. small -> head): that tensor is not trained any more
            self._pending[i] = cco.state_from_entries(self.kind, values)
        self.switches = int(info.get("switches", 0))
        logger.info(f"{LOG}: read the optimizer states of {len(self._pending)} tensors from {path} (saved active"
                    f" units {info.get('active')})")


def _bnb_version() -> Optional[str]:
    try:
        import bitsandbytes

        return str(bitsandbytes.__version__)
    except Exception:  # noqa: BLE001 - informational
        return None


def resolve_k(args, num_units: int) -> int:
    """K: --moss_carousel_active, else the start-up budget's (args._moss_rotation_k, set by ft_budget's report), else
    the budget at 1024^2 batch 1."""
    k = ft_args.carousel_arg(args, "active") or getattr(args, "_moss_rotation_k", None)
    if not k:
        gpu = ft_budget.gpu_budget(ft_budget.REFERENCE_IMAGE_TOKENS, 1,
                                   mixed_precision=getattr(args, "mixed_precision", "bf16"),
                                   stream_fp32=bool(getattr(args, "moss_stream_fp32", False)), mode=ft_budget.CAROUSEL)
        k = ft_budget.carousel_budget_from_args(args, gpu).k
    return max(1, min(int(k), int(num_units)))


def build(args, model, pool: ft_host.HostPool, slots: Sequence[ft_host.ParamSlot], optimizer_kwargs: Dict[str, Any],
          device) -> Tuple[Any, CarouselEngine]:
    """(optimizer, engine) of a Carousel run: one handle per slot, the "decay" / "no_decay" groups, the kind's
    optimizer at learning_rate, the engine with the run's flags. The engine is attached by the caller."""
    kind = ft_args.carousel_kind(args)
    always_policy = ft_args.carousel_arg(args, "always")
    units = cu.discover_units([(s.name, s.shape) for s in slots], stem_unit=cu.stem_rotates(always_policy))
    if not units:
        raise ValueError(f"{PREFIX}{CAROUSEL}: no block is trainable (--moss_ft_freeze): nothing would rotate")
    always = cu.always_names([s.name for s in slots], always_policy)
    k = resolve_k(args, len(units))
    device = torch.device(device)
    handles = [torch.nn.Parameter(torch.empty(0, dtype=torch.float32, device=device), requires_grad=False)
               for _ in slots]
    groups = param_groups(slots, handles, float(optimizer_kwargs.get("weight_decay", 0.0) or 0.0))
    optimizer = cco.make_optimizer(kind, groups, float(args.learning_rate), optimizer_kwargs)
    seed = ft_args.carousel_arg(args, "seed")
    seed_given = seed is not None
    if seed is None:
        seed = int(getattr(args, "seed", None) or 0)
    engine = CarouselEngine(model, pool, slots, optimizer, handles, kind=kind, units=units, always=always, k=k,
                            period=int(ft_args.carousel_arg(args, "period")), select=ft_args.carousel_arg(args, "select"),
                            seed=int(seed), state_policy=ft_args.carousel_arg(args, "state"),
                            state_dtype=ft_args.carousel_arg(args, "state_dtype"), device=device,
                            max_grad_norm=float(getattr(args, "max_grad_norm", 0.0) or 0.0),
                            max_skipped_steps=int(getattr(args, "moss_ft_max_skipped_steps", 8)),
                            host_pinned=ft_args.carousel_arg(args, "host_pinned") == "masters",
                            always_policy=always_policy, seed_given=seed_given)
    never = [s.name for s in slots if s.index not in set(engine.trainable)]
    logger.info(f"{LOG}: {kind}, K = {k} of {len(units)} units, switch every {engine.period} steps"
                f" ({engine.select_kind}, seed {engine.seed}), always {always_policy} ({len(always)} tensors,"
                f" {sum(s.numel for s in slots if s.name in set(always)) / 1e6:.2f}M), state {engine.state_policy}"
                + (f" ({engine.state_dtype})" if engine.keep else "")
                + (f"; {len(never)} tensors are never trained with always {always_policy}" if never else ""))
    return optimizer, engine
