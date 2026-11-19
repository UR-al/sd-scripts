# MOSS full fine-tune: the host-resident optimizer pipeline of `adamw_host` / `came_host`
# (docs/moss_full_finetune_design.md 3.1 and 3.3).
#
# The GPU holds only the compute copy of the trained weights (bf16 joint Linears, fp32 elsewhere under
# --moss_stream_fp32). The fp32 master, the optimizer moments and - for GA > 1 or exact clipping - an fp32 gradient
# accumulator live in ordinary pageable host RAM (HostPool). Each micro-step's gradients leave the GPU inside the
# backward: a post-accumulate-grad hook per parameter copies the gradient in pieces through a pinned ring to ONE FIFO
# worker thread, which sums them in fp32 (non-final micro-steps) or, in the window's final micro-step, steps the master
# piece by piece (AdamW is elementwise) or parameter by parameter (CAME needs whole matrices) and sends the new compute
# copy back through a second pinned ring. The end of the final micro-step waits for the worker (the step's "tail") and
# makes the compute stream wait for the uploads.
#
# All logic is shared by the CPU and the CUDA path. Only the copy / event / stream primitives differ (_CpuOps: plain
# memory, synchronous copies, no events - the CPU tests run the hooks, worker, pool, clipping and skipping code that
# training runs; _CudaOps: pinned rings, a D2H and an H2D stream, events). The ordering argument for the in-backward
# step (design 3.1 "Why the in-backward step is safe") is written out at _CudaOps.
#
# fp16 compute (docs/moss_ft_te_live_fp16_plan.md 2.3): the loss is multiplied by a power-of-two scale S before the
# backward (ft_scaler.DynamicLossScaler), so every gradient that reaches the hooks is S x the true gradient. The
# window's StepContext.inv_grad_scale (1 / S) is folded into the clip coefficient the worker multiplies the fp32 chunk by
# (one multiply unscales and clips; exact, since S is a power of two), and every norm is measured on the scaled values
# and multiplied by 1 / S where it is used (grad_norm, the exact clip total, last_grad_norm). An overflowed slot is the
# existing non-finite skip ("partial" policy); overflow_policy "window" steps no slot of such a window, and neither does
# "partial" when the window clips exactly (every step runs after the drain, and its true norm is not finite). An fp16
# compute copy is checked as it is cast from the master: a master past the fp16 range stops the run (HostPipelineError
# naming the parameter) instead of uploading inf.

import collections
import hashlib
import json
import math
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from library.moss.ft_budget import pinned_alloc_bytes
from library.utils import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)

MIB = 1 << 20
DEFAULT_PIECE_BYTES = 64 * MIB  # --moss_ft_piece_mib
DEFAULT_RING_BYTES = 1024 * MIB  # --moss_ft_ring_mib (D2H gradients)
DEFAULT_UPLOAD_RING_BYTES = 512 * MIB  # --moss_ft_upload_ring_mib (H2D compute copies)
ACC_CHUNK_ELEMENTS = 4 << 20  # accumulation pass: tmp.copy_(bf16 piece chunk); acc.add_(tmp) (design 1.4)
STEP_CHUNK_ELEMENTS = 1 << 20  # final pass: convert + add acc + AdamW + cast, cache-resident chunks (design 1.4)
# torch's CPU fused AdamW kernel works on 64-byte units with a vector body and a scalar tail; chunks that start at
# multiples of 64 elements (any vector width up to 64 floats) split a tensor exactly where one whole-tensor call does,
# so chunked steps are bit-identical to it (an odd chunk boundary is not: measured 7e-9 differences).
ALIGN_ELEMENTS = 64
ALIGN_BYTES = 256  # pieces: a multiple of 64 elements for 4-byte and of 128 for 2-byte gradients
POLL_SECONDS = 0.5  # blocked waits wake this often to re-raise a worker error
CLIP_MODES = ("stale", "exact")
OVERFLOW_POLICIES = ("partial", "window")

_PIECE, _STEP, _NORM = "piece", "step", "norm"
_STOP = object()


class HostPipelineError(RuntimeError):
    """The host pipeline failed (a worker exception, carrying its traceback, or a broken protocol). The pool may be
    half-stepped: the run must stop and resume from a saved state."""


# ------------------------------------------------------------------------------------------------------ slots / pool


@dataclass(eq=False)
class ParamSlot:
    """One trainable parameter's place in the HostPool. Slots are in model.named_parameters() order of the trainable
    set: the order of the pool, of the state files and of the hash. `param` / `compute_dtype` / `group` may be bound
    later (bind_slots), e.g. when the pool is allocated from a model on the meta device."""
    index: int
    name: str
    shape: Tuple[int, ...]
    numel: int
    offset: int
    param: Optional[torch.nn.Parameter] = None
    compute_dtype: Optional[torch.dtype] = None
    group: int = 0


def _group_index(optimizer, param) -> int:
    for index, group in enumerate(optimizer.param_groups):
        if any(p is param for p in group["params"]):
            return index
    raise ValueError("a trainable parameter is in no param group of the optimizer")


def make_slots(named_params: Iterable[Tuple[str, torch.Tensor]], optimizer=None) -> List[ParamSlot]:
    """Slots for [(name, parameter)] in the given order, offsets packed from 0. With `optimizer`, each slot's group is
    the param group holding that parameter (by identity) and every optimizer parameter must have a slot."""
    named = list(named_params)
    if len({id(p) for _, p in named}) != len(named) or len({n for n, _ in named}) != len(named):
        raise ValueError("make_slots: a parameter or a name appears twice")
    slots, offset = [], 0
    for index, (name, param) in enumerate(named):
        numel = int(param.numel())
        slots.append(ParamSlot(index=index, name=str(name), shape=tuple(int(d) for d in param.shape), numel=numel,
                               offset=offset, param=param, compute_dtype=param.dtype))
        offset += numel
    if optimizer is not None:
        bind_slots(slots, named, optimizer)
    return slots


def bind_slots(slots: Sequence[ParamSlot], named_params: Iterable[Tuple[str, torch.Tensor]], optimizer=None) -> None:
    """(Re)bind slots to live parameters by name: param, compute dtype (the parameter's dtype now - call it after
    enable_stream_fp32) and, with `optimizer`, the param group. Names and shapes must match the slots."""
    named = list(named_params)
    if [s.name for s in slots] != [n for n, _ in named]:
        raise ValueError("bind_slots: the parameter names / order differ from the slots")
    for slot, (name, param) in zip(slots, named):
        if tuple(param.shape) != slot.shape:
            raise ValueError(f"bind_slots: {name} has shape {tuple(param.shape)}, the slot {slot.shape}")
        slot.param, slot.compute_dtype = param, param.dtype
        if optimizer is not None:
            slot.group = _group_index(optimizer, param)
    if optimizer is not None:
        known = {id(s.param) for s in slots}
        extra = sum(1 for g in optimizer.param_groups for p in g["params"] if id(p) not in known)
        if extra:
            raise ValueError(f"bind_slots: {extra} optimizer parameters have no slot")


def slot_list_hash(slots: Sequence[ParamSlot]) -> str:
    """Stable hash of the pool layout (name, shape, offset of every slot, in order): a state written for one layout is
    refused by another."""
    layout = [[s.name, list(s.shape), int(s.offset)] for s in slots]
    return hashlib.sha256(json.dumps(layout, separators=(",", ":")).encode("utf-8")).hexdigest()


def needs_accumulator(gradient_accumulation_steps: int, clip_mode: str = "stale", max_grad_norm: float = 0.0,
                      overlap: bool = True) -> bool:
    """Whether the pool needs the fp32 gradient accumulator `acc` (design 2.2): GA > 1, exact clipping, or the
    after-backward step (overlap=False, a test / debugging switch)."""
    return (int(gradient_accumulation_steps) > 1 or (clip_mode == "exact" and float(max_grad_norm or 0.0) > 0.0)
            or not overlap)


class HostPool:
    """One contiguous fp32 CPU tensor per role, with per-slot flat views (pageable memory: the step runs on the CPU and
    nothing DMAs from it). Roles: the optimizer's POOL_ROLES (master, exp_avg[, exp_avg_sq]); `acc` (the gradient
    accumulator) only with with_acc. Moments start at zero; master and acc are uninitialised (the loader / the first
    contribution of a window fill them). steps: one float32 counter per slot (torch's fused AdamW step dtype);
    step(i) is slot i's 0-dim view."""

    def __init__(self, slots: Sequence[ParamSlot], roles: Sequence[str] = ("master", "exp_avg", "exp_avg_sq"),
                 with_acc: bool = False):
        self.slots = list(slots)
        offset = 0
        for index, slot in enumerate(self.slots):
            if slot.index != index or slot.offset != offset or slot.numel != math.prod(slot.shape):
                raise ValueError(f"HostPool: slot {slot.name} is not packed in order (index {slot.index}, offset "
                                 f"{slot.offset}, expected {index} / {offset})")
            offset += slot.numel
        self.total_numel = offset
        if "master" not in roles:
            raise ValueError("HostPool needs a master role")
        self.roles: Dict[str, torch.Tensor] = {}
        for role in roles:
            if role == "acc" or role in self.roles:
                raise ValueError(f"HostPool: bad role list {tuple(roles)}")
            alloc = torch.empty if role == "master" else torch.zeros
            self.roles[role] = alloc(self.total_numel, dtype=torch.float32)
        self.acc: Optional[torch.Tensor] = torch.empty(self.total_numel, dtype=torch.float32) if with_acc else None
        self.steps = torch.zeros(len(self.slots), dtype=torch.float32)

    def _slot(self, slot) -> ParamSlot:
        return self.slots[slot] if isinstance(slot, int) else slot

    def view(self, role: str, slot) -> torch.Tensor:
        """Flat fp32 view of `slot` (a ParamSlot or its index) in `role` ("acc" for the accumulator)."""
        slot = self._slot(slot)
        flat = self.acc if role == "acc" else self.roles[role]
        if flat is None:
            raise ValueError("HostPool has no accumulator (with_acc=False)")
        return flat[slot.offset:slot.offset + slot.numel]

    def step(self, slot) -> torch.Tensor:
        return self.steps[self._slot(slot).index]

    def layout(self) -> List[dict]:
        return [dict(index=s.index, name=s.name, shape=list(s.shape), numel=s.numel, offset=s.offset, group=s.group,
                     compute_dtype=str(s.compute_dtype).replace("torch.", "")) for s in self.slots]

    def slot_hash(self) -> str:
        return slot_list_hash(self.slots)

    def nbytes(self) -> Dict[str, int]:
        out = {role: t.numel() * t.element_size() for role, t in self.roles.items()}
        if self.acc is not None:
            out["acc"] = self.acc.numel() * self.acc.element_size()
        return out

    @torch.no_grad()
    def copy_master_from_params(self) -> None:
        """master <- the bound parameters (exact: every compute dtype converts to fp32 exactly)."""
        for slot in self.slots:
            self.view("master", slot).copy_(slot.param.detach().reshape(-1))

    @torch.no_grad()
    def copy_params_from_master(self) -> None:
        """The bound parameters <- master rounded to their dtype (start-up / resume; synchronous, one tensor at a
        time)."""
        for slot in self.slots:
            slot.param.data.copy_(self.view("master", slot).view(slot.shape))


class PinnedRing:
    """`total_bytes` of (pinned on CUDA) host memory in entries of `piece_bytes`. acquire() blocks until an entry is
    free, waking every POLL_SECONDS to run `check` (which raises a stored worker error); release(entry) frees it.
    view(entry, dtype, numel) sees an entry as bf16 / fp32 (piece_bytes / 2 or / 4 elements)."""

    def __init__(self, total_bytes: int, piece_bytes: int, pin: bool = False):
        total_bytes, piece_bytes = int(total_bytes), int(piece_bytes)
        if piece_bytes <= 0 or total_bytes < piece_bytes or total_bytes % piece_bytes:
            raise ValueError(f"PinnedRing: {total_bytes} bytes is not a positive multiple of the piece ({piece_bytes})")
        self.piece_bytes = piece_bytes
        self.entries = total_bytes // piece_bytes
        self.buffer = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=pin)
        self._free: "queue.Queue[int]" = queue.Queue()
        for entry in range(self.entries):
            self._free.put(entry)

    def acquire(self, check=None, poll: float = POLL_SECONDS) -> int:
        while True:
            if check is not None:
                check()
            try:
                entry = self._free.get(timeout=poll)
            except queue.Empty:
                continue
            if check is not None:
                try:
                    check()
                except BaseException:
                    self._free.put(entry)
                    raise
            return entry

    def try_acquire(self) -> Optional[int]:
        try:
            return self._free.get_nowait()
        except queue.Empty:
            return None

    def release(self, entry: int) -> None:
        self._free.put(entry)

    @property
    def free(self) -> int:
        return self._free.qsize()

    def view(self, entry: int, dtype: torch.dtype, numel: Optional[int] = None) -> torch.Tensor:
        start = entry * self.piece_bytes
        flat = self.buffer[start:start + self.piece_bytes].view(dtype)
        return flat if numel is None else flat[:numel]


# --------------------------------------------------------------------------------------------- context / statistics


@dataclass(frozen=True)
class StepContext:
    """What one window's optimizer step uses, fixed at begin_window: each param group's hyperparameters as the
    optimizer's snapshot_group returns them (plain values / host tensors: the lr the scheduler set for this step, betas,
    eps, weight decay, ...) and the optimizer step index. The worker reads only this, never optimizer.param_groups.
    The clip coefficient of the window is decided by the pipeline at begin_window (WindowStats reports it).
    inv_grad_scale: 1 / the fp16 loss scale the window's losses were multiplied by (a power of two; 1.0 = no scaling)."""
    groups: Tuple[dict, ...]
    step_index: int = 0
    inv_grad_scale: float = 1.0

    @classmethod
    def snapshot(cls, optimizer, step_index: int = 0, inv_grad_scale: float = 1.0) -> "StepContext":
        return cls(groups=tuple(optimizer.snapshot_group(g) for g in optimizer.param_groups),
                   step_index=int(step_index), inv_grad_scale=float(inv_grad_scale))


@dataclass
class WindowStats:
    """One optimizer step (window). grad_norm: the window's global L2 norm of the accumulated gradients before clipping;
    NaN for a window that skipped parameters (its norm is not finite; the next stale coefficient keeps the last clean
    window's norm). clip_coef: the factor applied (1.0 = none); clip_mode: "none", "stale" (coefficient from the
    previous window's norm), "exact" (this window's, applied after the drain) or "measure" (GA 1 without the
    accumulator and without a known previous norm: the window only measured its norm, nothing was stepped -
    stepped_params 0). skip_streak: consecutive windows (this one included) that skipped a parameter for non-finite
    gradients. Times in seconds: tail = the CPU wait at the end of the final micro-step; hook_wait = the autograd thread
    blocked on the D2H ring; worker_busy = the worker's processing time. loss_scale: the fp16 loss scale of the window
    (1.0 without scaling); overflow: some slot's gradient was not finite (the window skipped parameters). grad_norm is
    always the unscaled norm."""
    step_index: int
    n_micro: int
    grad_norm: float
    clip_coef: float
    clip_mode: str
    skipped_params: int
    skipped_names: List[str]
    skip_streak: int
    stepped_params: int
    tail_s: float
    hook_wait_s: float
    worker_busy_s: float
    loss_scale: float = 1.0
    overflow: bool = False
    extra: Dict[str, float] = field(default_factory=dict)  # mode-specific per-step scalars (the carousel's), logged as is


# ---------------------------------------------------------------------------------------- device primitives (only)


class _CpuOps:
    """Device "cpu" (the CPU tests): the gradients and the parameters are host tensors already, copies are synchronous
    and complete when they return, so there is nothing to record or wait for."""
    cuda = False

    def __init__(self, device: torch.device):
        self.device = device

    def grad_ready(self):
        return None

    def begin_d2h(self, ready) -> None:
        pass

    def grad_norm(self, flat, norm_dev, norm_host, slot: int, j: int) -> None:
        torch.linalg.vector_norm(flat, 2.0, dtype=torch.float32, out=norm_host[slot, j])

    def copy_d2h(self, dst, src):
        dst.copy_(src)
        return None

    def release_grad(self, flat) -> None:
        pass

    def wait(self, event) -> None:
        pass

    def copy_h2d(self, dst, src, after):
        dst.copy_(src)
        return None

    def record_compute(self):
        return None

    def finish_window(self) -> None:
        pass

    def worker_init(self) -> None:
        pass


class _CudaOps:
    """Device "cuda": a D2H stream for the gradients, an H2D stream for the uploads, events between them.

    Ordering (design 3.1, "Why the in-backward step is safe"), line by line:
    * grad_ready (autograd device thread, inside p's post-accumulate-grad hook) records R(p) on the current stream - the
      stream the AccumulateGrad node of p ran on, i.e. after the kernel that produced p's gradient. The node that read
      p (dX = dY.W and dW in one node, or the cast / recompute of a checkpointed block) ran before AccumulateGrad(p) -
      the engine runs AccumulateGrad only once every gradient contribution to p is summed - so every kernel that reads
      p in this backward is enqueued before R(p). Block i's weights are read only by block i's forward, recompute and
      backward; blocks not yet backpropagated keep their old weights, as their recompute needs. (Assumption: no
      backward node reads a parameter without feeding its gradient, e.g. through p.detach() in a forward. MOSS has
      none; the inline CPU test test_in_backward_step_equals_after_backward_step uploads inside the hook and would see
      it.)
    * begin_d2h makes the D2H stream wait R(p): the norm and the piece copies read the finished gradient.
    * grad_norm writes the fp32 norm of p's gradient into norm_dev[slot] and copies it to the pinned norm_host[slot, j]
      on the D2H stream BEFORE the piece copies, so a piece's D event implies its norm arrived (the worker reads it
      after waiting D of the slot's first piece).
    * copy_d2h copies one piece into a pinned ring entry and records D(piece) after it. The entry was acquired from the
      ring, i.e. the worker had finished reading its previous content (it releases only after processing).
    * release_grad: g.record_stream(d2h). The hook then sets p.grad = None; when the last reference goes, the caching
      allocator keeps the block until the D2H work enqueued on it is done (an event on d2h recorded at free time).
    * wait (worker thread) synchronizes on D(piece) before the host reads the entry, and on U(entry) before an upload
      entry is overwritten.
    * copy_h2d: the H2D stream waits R(p) (the item carries it) - or, for uploads after the drain, an event recorded
      on the compute stream when backward() had returned - then copies the pinned upload entry into p.data and records
      U. After R(p) no kernel of this backward reads p, so overwriting it is ordered after every reader.
    * finish_window (main thread, after the worker went idle, so every upload of the window is enqueued): the compute
      stream waits for the H2D stream: the next forward and any sampling see the stepped weights. The CPU never waits
      for the uploads themselves.
    No wait can form a cycle: D2H and H2D wait only on events already recorded on the compute stream, the worker waits
    only on D / U events, and the hook waits only for ring entries the worker releases without waiting on autograd."""
    cuda = True

    def __init__(self, device: torch.device):
        device = torch.device(device)
        if device.index is None:  # GPU smoke 11-17: torch.cuda.set_device in the worker thread refuses a bare "cuda"
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.d2h = torch.cuda.Stream(device)
        self.h2d = torch.cuda.Stream(device)

    def grad_ready(self):
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        return event

    def begin_d2h(self, ready) -> None:
        self.d2h.wait_event(ready)

    def grad_norm(self, flat, norm_dev, norm_host, slot: int, j: int) -> None:
        with torch.cuda.stream(self.d2h):
            out = norm_dev[slot]
            torch.linalg.vector_norm(flat, 2.0, dtype=torch.float32, out=out)
            norm_host[slot, j].copy_(out, non_blocking=True)

    def copy_d2h(self, dst, src):
        with torch.cuda.stream(self.d2h):
            dst.copy_(src, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.d2h)
        return done

    def release_grad(self, flat) -> None:
        flat.record_stream(self.d2h)

    def wait(self, event) -> None:
        if event is not None:
            event.synchronize()

    def copy_h2d(self, dst, src, after):
        with torch.cuda.stream(self.h2d):
            if after is not None:
                self.h2d.wait_event(after)
            dst.copy_(src, non_blocking=True)
            uploaded = torch.cuda.Event()
            uploaded.record(self.h2d)
        return uploaded

    def record_compute(self):
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        return event

    def finish_window(self) -> None:
        torch.cuda.current_stream(self.device).wait_stream(self.h2d)

    def worker_init(self) -> None:
        torch.cuda.set_device(self.device)  # the current device is per host thread


# ------------------------------------------------------------------------------------------------- worker plumbing


class _Fifo:
    """The worker's queue: FIFO, with a join that wakes every `poll` seconds to run `check` (queue.Queue.join cannot be
    interrupted when the worker thread dies)."""

    def __init__(self):
        self._items = collections.deque()
        self._cond = threading.Condition()
        self._unfinished = 0

    def put(self, item) -> None:
        with self._cond:
            self._items.append(item)
            self._unfinished += 1
            self._cond.notify_all()

    def get(self):
        with self._cond:
            while not self._items:
                self._cond.wait()
            return self._items.popleft()

    def task_done(self) -> None:
        with self._cond:
            self._unfinished -= 1
            if self._unfinished <= 0:
                self._cond.notify_all()

    def join(self, check, poll: float = POLL_SECONDS) -> None:
        with self._cond:
            while self._unfinished > 0:
                self._cond.wait(poll)
                check()

    @property
    def unfinished(self) -> int:
        return self._unfinished


class _Item:
    """One unit of worker work: a gradient piece (kind _PIECE), a slot to step from the accumulator (_STEP: the flush
    of a slot without a final-micro-step gradient, or the after-backward step) or a slot norm (_NORM, exact mode)."""
    __slots__ = ("kind", "win", "slot", "j", "start", "numel", "entry", "dtype", "done", "ready", "first", "last")

    def __init__(self, kind, win, slot, j=-1, start=0, numel=0, entry=-1, dtype=None, done=None, ready=None,
                 first=False, last=False):
        self.kind, self.win, self.slot, self.j = kind, win, slot, j
        self.start, self.numel, self.entry, self.dtype = start, numel, entry, dtype
        self.done, self.ready, self.first, self.last = done, ready, first, last


_Plan = collections.namedtuple("_Plan", "mode after_backward coef")


class _Window:
    """Per-window bookkeeping, written by the worker context only (the main thread reads it after a drain)."""

    def __init__(self, n_micro: int, ctx: StepContext, plan: _Plan, n_slots: int):
        self.n_micro, self.final_j, self.ctx, self.plan = n_micro, n_micro - 1, ctx, plan
        self.hyper = ctx.groups
        self.inv_scale = float(ctx.inv_grad_scale)  # 1 / the fp16 loss scale (power of two; 1.0 = none)
        self.clip = _coefficient(plan.coef)  # the clip coefficient alone (WindowStats.clip_coef), or None
        # 0-dim fp32 tensor multiplied into the gradients (clip coefficient x 1 / loss scale), or None
        self.coef = _coefficient(_combine(plan.coef, self.inv_scale))
        self.next_j = 0
        self.first_j = [-1] * n_slots  # micro-step of the slot's first contribution (-1: none yet)
        self.poisoned = [False] * n_slots
        self.finalized = [False] * n_slots  # stepped this window
        self.bumped = [False] * n_slots  # step counter incremented this window
        self.slot_norm: List[Optional[torch.Tensor]] = [None] * n_slots  # exact mode
        self.sumsq = 0.0
        self.after = None  # compute-stream event at the end of the final backward (uploads after the drain)
        self.scratch_owner = -1  # came_host at GA = 1: the slot whose gradient is in the scratch


# ---------------------------------------------------------------------------------------------------- the pipeline


class GradPipeline:
    """Gradient streaming, accumulation and the host optimizer step (design 3.1; 3.3 for CAME).

    Per window: begin_window(n_micro, StepContext) once, then per micro-step j: begin_microstep(j), loss.backward()
    (the hooks stream the gradients; in the final micro-step the parameters are stepped and uploaded as their hooks
    fire), end_microstep() - None for non-final micro-steps, the WindowStats after the drain for the final one. Then
    the LR scheduler may step. close() at the end.

    device: "cpu" (tests: synchronous copies) or a CUDA device. optimizer: HostAdamW / HostCAME (ft_optim), bound to the
    pool here. max_grad_norm > 0 clips: clip_mode "stale" (the coefficient from the previous clean window's norm, KMT's
    idea; a window without a known previous norm runs exact when the pool has the accumulator, else it only measures
    its norm and steps nothing) or "exact" (the final micro-step only accumulates, the step runs after the drain -
    needs the accumulator). Parameters whose
    gradient is non-finite in any micro-step of a window keep master, moments, step and compute copy for that window
    (check_skip_streak stops the run after max_skipped_steps such windows in a row; 0 = never). inline: the hook runs
    the worker's code itself (no thread). overlap=False: every window steps after the drain (tests / debugging).
    host_threads: torch intra-op threads of the worker thread (thread-local). overflow_policy (fp16 loss scaling):
    "partial" (default) steps the finite slots of a window in which some slot overflowed; "window" steps none of them
    (needs every window to step after the drain: exact clipping with max_grad_norm > 0, or overlap=False). A window
    clipped exactly (plan "exact": clip_mode exact, or stale without a previous norm) skips every slot once one slot
    is non-finite, under either policy: nothing of it was stepped in the backward, and a coefficient from the norm of
    the finite slots alone would under-clip a window whose true norm is not finite."""

    def __init__(self, slots: Sequence[ParamSlot], pool: HostPool, optimizer, *, device,
                 gradient_accumulation_steps: int, max_grad_norm: float = 0.0, clip_mode: str = "stale",
                 max_skipped_steps: int = 8, piece_bytes: int = DEFAULT_PIECE_BYTES,
                 ring_bytes: int = DEFAULT_RING_BYTES, upload_ring_bytes: int = DEFAULT_UPLOAD_RING_BYTES,
                 acc_chunk: int = ACC_CHUNK_ELEMENTS, step_chunk: int = STEP_CHUNK_ELEMENTS,
                 host_threads: Optional[int] = None, inline: bool = False, overlap: bool = True,
                 overflow_policy: str = "partial"):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError(f"GradPipeline: device {self.device} is neither cpu nor cuda")
        self.ga = int(gradient_accumulation_steps)
        if self.ga < 1:
            raise ValueError(f"gradient_accumulation_steps must be >= 1, got {gradient_accumulation_steps}")
        if clip_mode not in CLIP_MODES:
            raise ValueError(f"clip_mode must be one of {CLIP_MODES}, got {clip_mode!r}")
        self.max_grad_norm = float(max_grad_norm or 0.0)
        if not (math.isfinite(self.max_grad_norm) and self.max_grad_norm >= 0.0):
            raise ValueError(f"max_grad_norm must be a finite number >= 0, got {max_grad_norm}")
        self.clip_mode = clip_mode
        self.max_skipped_steps = int(max_skipped_steps)
        if self.max_skipped_steps < 0:
            raise ValueError(f"max_skipped_steps must be >= 0, got {max_skipped_steps}")
        self.piece_bytes = int(piece_bytes)
        if self.piece_bytes <= 0 or self.piece_bytes % ALIGN_BYTES:
            raise ValueError(f"piece_bytes must be a positive multiple of {ALIGN_BYTES}, got {piece_bytes}")
        self.acc_chunk, self.step_chunk = int(acc_chunk), int(step_chunk)
        for name, value in (("acc_chunk", self.acc_chunk), ("step_chunk", self.step_chunk)):
            if value <= 0 or value % ALIGN_ELEMENTS:
                raise ValueError(f"{name} must be a positive multiple of {ALIGN_ELEMENTS} elements, got {value}")
        self.overlap = bool(overlap)
        self.inline = bool(inline)
        if overflow_policy not in OVERFLOW_POLICIES:
            raise ValueError(f"overflow_policy must be one of {OVERFLOW_POLICIES}, got {overflow_policy!r}")
        if overflow_policy == "window" and self.overlap and not (clip_mode == "exact" and self.max_grad_norm > 0):
            raise ValueError("overflow_policy 'window' needs every window to step after the backward: clip_mode exact"
                             " with max_grad_norm > 0 (or overlap=False); the in-backward step cannot take back the"
                             " slots it already stepped")
        self.overflow_policy = overflow_policy
        self.host_threads = None if host_threads is None else int(host_threads)

        self.slots = list(slots)
        for index, slot in enumerate(self.slots):
            if slot.index != index or slot.param is None or slot.compute_dtype is None:
                raise ValueError(f"GradPipeline: slot {slot.name} is not bound (make_slots / bind_slots)")
            if slot.group >= len(optimizer.param_groups) or not any(
                    p is slot.param for p in optimizer.param_groups[slot.group]["params"]):
                raise ValueError(f"GradPipeline: {slot.name} is not in param group {slot.group} of the optimizer")
            if self.piece_bytes % torch.empty((), dtype=slot.compute_dtype).element_size():
                raise ValueError(f"GradPipeline: {slot.name}: piece size not a multiple of its element size")
        if pool.slot_hash() != slot_list_hash(self.slots):
            raise ValueError("GradPipeline: the pool was laid out for other slots")
        if needs_accumulator(self.ga, clip_mode, self.max_grad_norm, self.overlap) and pool.acc is None:
            raise ValueError(f"GradPipeline: GA {self.ga} / clip_mode {clip_mode} with max_grad_norm "
                             f"{self.max_grad_norm} / overlap {self.overlap} needs the fp32 accumulator "
                             "(HostPool(with_acc=True))")
        self.pool, self.optimizer = pool, optimizer
        self.per_parameter = bool(getattr(optimizer, "per_parameter", False))
        optimizer.bind(self.slots, pool)

        self.ops = _CudaOps(self.device) if self.device.type == "cuda" else _CpuOps(self.device)
        pin = self.ops.cuda
        self.d2h_ring = PinnedRing(ring_bytes, self.piece_bytes, pin)
        self.up_ring = PinnedRing(upload_ring_bytes, self.piece_bytes, pin)
        self._up_pending = collections.deque()  # (entry, upload event) in submission order; worker context only
        n = len(self.slots)
        self.norm_host = torch.zeros((n, self.ga), dtype=torch.float32, pin_memory=pin)
        self.norm_dev = torch.zeros(n, dtype=torch.float32, device=self.device) if self.ops.cuda else None
        self._scratch = torch.empty(max(self.acc_chunk, self.step_chunk), dtype=torch.float32)
        # came_host at GA = 1 has no accumulator: each parameter's gradient is assembled here (its pieces arrive back
        # to back, one hook enqueues them all) and stepped when its last piece is in
        self._param_scratch = None
        if self.per_parameter and pool.acc is None:
            self._param_scratch = torch.empty(max([s.numel for s in self.slots] + [1]), dtype=torch.float32)

        self.last_grad_norm: Optional[float] = None  # the previous window's G (stale clipping; saved in states)
        self.skip_streak = 0
        self.micro_hook_counts: List[int] = [0] * n  # hook calls per slot in the last ended micro-step
        self._hook_counts = [0] * n
        self._count_lock = threading.Lock()
        self._handles = []
        self._win: Optional[_Window] = None
        self._j: Optional[int] = None
        self._error = None  # (exception, traceback text) of the first worker failure
        self._failure_obj: Optional[HostPipelineError] = None
        self._hook_wait = 0.0
        self._worker_busy = 0.0
        self._warned_noncontiguous = False
        self._closed = False
        self._fifo: Optional[_Fifo] = None
        self._thread: Optional[threading.Thread] = None
        if not self.inline:
            self._fifo = _Fifo()
            self._thread = threading.Thread(target=self._worker_main, name="moss-ft-host-worker", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------------------------------------ public API

    def attach(self) -> None:
        """Register the post-accumulate-grad hook on every slot's parameter."""
        if self._handles:
            raise HostPipelineError("GradPipeline.attach: already attached")
        for slot in self.slots:
            if not slot.param.requires_grad:
                raise ValueError(f"GradPipeline.attach: {slot.name} does not require grad")
            self._handles.append(slot.param.register_post_accumulate_grad_hook(self._make_hook(slot.index)))

    def begin_window(self, n_micro: int, ctx: StepContext) -> None:
        self._raise_if_failed()
        if self._closed:
            raise HostPipelineError("GradPipeline is closed")
        if self._win is not None:
            raise HostPipelineError("begin_window: the previous window has not ended (end_microstep of its final "
                                    "micro-step)")
        n_micro = int(n_micro)
        if not 1 <= n_micro <= self.ga:
            raise ValueError(f"begin_window: n_micro must be in [1, {self.ga}], got {n_micro}")
        if len(ctx.groups) != len(self.optimizer.param_groups):
            raise ValueError("begin_window: the StepContext has another number of param groups")
        inv = ctx.inv_grad_scale
        if not _is_power_of_two(inv):
            raise ValueError(f"begin_window: inv_grad_scale must be a power of two (exact unscaling), got {inv!r}")
        self._drain()  # idle already (the previous window drained); this also surfaces a late worker error
        self._hook_wait = 0.0
        self._worker_busy = 0.0
        self._win = _Window(n_micro, ctx, self._plan(), len(self.slots))

    def begin_microstep(self, j: int) -> None:
        w = self._win
        if w is None:
            raise HostPipelineError("begin_microstep outside a window")
        if self._j is not None or int(j) != w.next_j:
            raise HostPipelineError(f"begin_microstep({j}): expected micro-step {w.next_j}, after end_microstep")
        self._raise_if_failed()
        with self._count_lock:
            self._hook_counts = [0] * len(self.slots)
        self._j = int(j)
        w.next_j = self._j + 1

    def end_microstep(self) -> Optional[WindowStats]:
        """Non-final micro-steps return None at once (the worker keeps accumulating behind the next forward). The final
        one drains, finishes the step (flush, or the exact / after-backward step) and returns the WindowStats."""
        w, j = self._win, self._j
        if w is None or j is None:
            raise HostPipelineError("end_microstep without begin_microstep")
        self._raise_if_failed()
        with self._count_lock:
            self.micro_hook_counts = list(self._hook_counts)
        self._j = None
        if j < w.final_j:
            return None
        return self._finish_window(w)

    def step_from_accumulator(self, clip_coef=None) -> None:
        """Step every slot that got a gradient this window and is not finalized yet from the accumulator, the gradient
        multiplied by clip_coef (None / 1.0: not at all; a float or an fp32 0-dim tensor), and upload it. Part of
        end_microstep for exact clipping; callable on its own only between the final micro-step's drain and the end of
        the window (a window whose final micro-step only accumulated)."""
        w = self._win
        if w is None or self._j is not None or w.next_j <= w.final_j:
            raise HostPipelineError("step_from_accumulator: only after the final micro-step of a window")
        if self.pool.acc is None:
            raise HostPipelineError("step_from_accumulator needs the accumulator")
        w.clip = _coefficient(clip_coef)
        w.coef = _coefficient(_combine(clip_coef, w.inv_scale))  # the loss scale undone in the same multiply
        self._run([_Item(_STEP, w, s, ready=w.after) for s in self._unstepped(w)])

    def sync(self) -> None:
        """Wait until the worker has processed everything submitted so far (re-raises a worker error)."""
        self._drain()

    @property
    def pinned_bytes(self) -> int:
        """Host bytes this pipeline page-locked (the two rings and the norm table; 0 on the CPU device), as torch's
        pinned host allocator holds them: each allocation rounded up to a power of two."""
        if not self.ops.cuda:
            return 0
        return (pinned_alloc_bytes(self.d2h_ring.buffer.numel()) + pinned_alloc_bytes(self.up_ring.buffer.numel())
                + pinned_alloc_bytes(self.norm_host.numel() * self.norm_host.element_size()))

    def check_skip_streak(self, stats: WindowStats) -> None:
        """--moss_ft_max_skipped_steps: raise after that many consecutive windows that skipped a parameter for
        non-finite gradients (call it after logging the window)."""
        if self.max_skipped_steps > 0 and stats.skip_streak >= self.max_skipped_steps:
            raise RuntimeError(
                f"non-finite gradients made the host optimizer skip parameters in {stats.skip_streak} optimizer steps "
                f"in a row (--moss_ft_max_skipped_steps {self.max_skipped_steps}; last window: "
                f"{', '.join(stats.skipped_names[:8])}{' ...' if len(stats.skipped_names) > 8 else ''}). The run "
                "diverged: resume from an earlier state, e.g. with a lower learning rate.")

    def restore_counters(self, last_grad_norm: Optional[float], skip_streak: int = 0) -> None:
        """Resume: the previous window's grad norm (stale clipping) and the skip streak from a training state."""
        self.last_grad_norm = None if last_grad_norm is None else float(last_grad_norm)
        self.skip_streak = int(skip_streak)

    def close(self) -> None:
        """Remove the hooks and join the worker thread. Idempotent; safe after a failure."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._closed = True
        self._win, self._j = None, None
        thread, self._thread = self._thread, None
        if thread is not None:
            self._fifo.put(_STOP)
            thread.join(timeout=60.0)
            if thread.is_alive():
                logger.warning("GradPipeline.close: the host worker did not stop within 60 s (left as a daemon)")

    # ------------------------------------------------------------------------------------------- window control

    def _coef_from_norm(self, norm) -> torch.Tensor:
        # clip_grad_norm_'s coefficient, formed the same way: max_norm / (G + 1e-6) on an fp32 0-dim tensor, clamped
        total = norm if torch.is_tensor(norm) else torch.tensor(float(norm), dtype=torch.float32)
        clip_coef = self.max_grad_norm / (total + 1e-6)
        return torch.clamp(clip_coef, max=1.0)

    def _plan(self) -> _Plan:
        after = not self.overlap
        if self.max_grad_norm <= 0.0:
            return _Plan("none", after, None)
        if self.clip_mode == "exact":
            return _Plan("exact", True, None)
        if self.last_grad_norm is None or not math.isfinite(self.last_grad_norm):
            if self.pool.acc is not None:
                return _Plan("exact", True, None)
            # GA = 1 without the accumulator: this window cannot wait for its own norm, and stepping it unclipped would
            # put the unclipped gradient into the moments (exp_avg_sq at (G / max_norm)^2 scale shrinks every clipped
            # update after it for ~1 / (1 - beta2) steps). It only measures G; the next windows clip with it.
            logger.info("moss_ft: no previous gradient norm and no accumulator (GA = 1, stale clipping): this window "
                        "only measures its gradient norm (no update); the next ones clip with it")
            return _Plan("measure", False, None)
        return _Plan("stale", after, self._coef_from_norm(self.last_grad_norm))

    def _unstepped(self, w: _Window) -> List[int]:
        return [s for s in range(len(self.slots)) if w.first_j[s] >= 0 and not w.poisoned[s] and not w.finalized[s]]

    def _finish_window(self, w: _Window) -> WindowStats:
        t0 = time.perf_counter()
        # uploads made after the drain wait for everything the backward enqueued on the compute stream
        w.after = self.ops.record_compute()
        self._drain()
        # the sums of squares and the slot norms are of the scaled gradients: x inv_scale (a power of two) is exact
        whole_window_skip = False
        if w.plan.mode == "measure":  # nothing was stepped (GA 1: no earlier micro-step to flush either)
            grad_norm = math.sqrt(w.sumsq) * w.inv_scale
        elif w.plan.after_backward:
            total = None
            # "window", or an exact-clipped window under "partial": nothing was stepped in the backward, so skipping
            # it all is possible, and the norm of the finite slots alone would under-clip it
            whole_window_skip = any(w.poisoned) and (self.overflow_policy == "window" or w.plan.mode == "exact")
            if w.plan.mode == "exact" and not whole_window_skip:
                self._run([_Item(_NORM, w, s) for s in range(len(self.slots))
                           if w.first_j[s] >= 0 and not w.poisoned[s] and w.slot_norm[s] is None])
                norms = [w.slot_norm[s] for s in range(len(self.slots)) if w.first_j[s] >= 0 and not w.poisoned[s]]
                # clip_grad_norm_'s total: the norm of the stacked per-parameter norms (slot order)
                total = torch.linalg.vector_norm(torch.stack(norms), 2.0) if norms else torch.zeros(())
                if w.inv_scale != 1.0:
                    total = total * w.inv_scale
                coef = self._coef_from_norm(total)
            else:
                coef = w.plan.coef
            if not whole_window_skip:  # overflow_policy "window": no slot of an overflowed window steps
                self.step_from_accumulator(coef)
            grad_norm = float(total) if total is not None else math.sqrt(w.sumsq) * w.inv_scale
        else:
            # parameters with a gradient in an earlier micro-step but none in the final one: step them from acc now
            self._run([_Item(_STEP, w, s, ready=w.after) for s in self._unstepped(w)])
            grad_norm = math.sqrt(w.sumsq) * w.inv_scale
        self.ops.finish_window()

        overflow = any(w.poisoned)
        if whole_window_skip:
            skipped = [self.slots[s].name for s in range(len(self.slots)) if w.first_j[s] >= 0]
        else:
            skipped = [self.slots[s].name for s in range(len(self.slots)) if w.poisoned[s]]
        self.skip_streak = self.skip_streak + 1 if skipped else 0
        stepped_norm = grad_norm
        if skipped:
            # the window's norm is not finite and its stepped part understates it (a fully skipped window would read
            # 0.0 and turn stale clipping off right after a divergence event): the next window keeps the last clean
            # window's norm
            grad_norm = float("nan")
        else:
            self.last_grad_norm = grad_norm
        mode = w.plan.mode
        stats = WindowStats(step_index=w.ctx.step_index, n_micro=w.n_micro, grad_norm=grad_norm,
                            clip_coef=1.0 if w.clip is None else float(w.clip), clip_mode=mode,
                            skipped_params=len(skipped), skipped_names=skipped, skip_streak=self.skip_streak,
                            stepped_params=sum(w.finalized), tail_s=time.perf_counter() - t0,
                            hook_wait_s=self._hook_wait, worker_busy_s=self._worker_busy,
                            loss_scale=1.0 / w.inv_scale, overflow=overflow)
        if skipped:
            text = (f"moss_ft step {w.ctx.step_index}: non-finite gradients, {len(skipped)} parameter(s) not "
                    f"stepped: {', '.join(skipped[:8])}{' ...' if len(skipped) > 8 else ''} "
                    f"(streak {self.skip_streak}; norm of the stepped part {stepped_norm:.4g}, the next clip "
                    f"keeps the previous norm {self.last_grad_norm})")
            if whole_window_skip:
                text += ("; whole window skipped (" + ("overflow policy window" if self.overflow_policy == "window"
                                                       else "exact clipping: its norm is not finite") + ")")
            if w.inv_scale != 1.0:
                # fp16 loss scaling: an overflow is the scaler's search for the largest safe scale (ft_scaler); the
                # scale halves after this window (unless the loss itself was not finite)
                logger.info(text + f"; fp16 loss scale {1.0 / w.inv_scale:g}")
            else:
                logger.warning(text)
        self._win = None
        self.optimizer.step()  # updates nothing; an LR scheduler sees the call order it expects
        return stats

    # ----------------------------------------------------------------------------------------------- the hook

    def _make_hook(self, index: int):
        def hook(param):
            self._on_grad(index, param)
        return hook

    def _on_grad(self, s: int, param) -> None:
        """Autograd thread, right after p's gradient is complete for this backward."""
        slot = self.slots[s]
        self._raise_if_failed()
        w, j = self._win, self._j
        if w is None or j is None:
            raise HostPipelineError(f"a gradient of {slot.name} arrived outside a window: begin_window and "
                                    "begin_microstep come before backward()")
        with self._count_lock:
            self._hook_counts[s] += 1
            count = self._hook_counts[s]
        if count > 1:
            raise HostPipelineError(f"the post-accumulate-grad hook of {slot.name} fired twice in micro-step {j} "
                                    "(two backward passes in one micro-step?)")
        grad = param.grad
        if grad is None:
            return
        if grad.is_sparse or grad.dtype != slot.compute_dtype or grad.numel() != slot.numel:
            raise HostPipelineError(f"{slot.name}: gradient {grad.dtype} x {grad.numel()} (sparse {grad.is_sparse}) "
                                    f"does not match the slot ({slot.compute_dtype} x {slot.numel}); bind the slots "
                                    "after the last dtype change of the model")
        if grad.is_contiguous():
            flat = grad.view(-1)
        else:  # Linear weight gradients are contiguous; anything else is copied once (before R(p) is recorded)
            if not self._warned_noncontiguous:
                self._warned_noncontiguous = True
                logger.warning(f"moss_ft: the gradient of {slot.name} is not contiguous; copied with reshape")
            flat = grad.reshape(-1)
        ready = self.ops.grad_ready()  # R(p)
        self.ops.begin_d2h(ready)
        self.ops.grad_norm(flat, self.norm_dev, self.norm_host, s, j)
        per = self.piece_bytes // flat.element_size()
        n = slot.numel
        for start in range(0, n, per):
            k = min(per, n - start)
            entry = self._acquire_d2h()
            done = self.ops.copy_d2h(self.d2h_ring.view(entry, flat.dtype, k), flat[start:start + k])
            self._submit(_Item(_PIECE, w, s, j, start, k, entry, flat.dtype, done, ready, start == 0,
                               start + k >= n))
        self.ops.release_grad(flat)
        param.grad = None

    def _acquire_d2h(self) -> int:
        t0 = time.perf_counter()
        try:
            return self.d2h_ring.acquire(self._check_worker)
        finally:
            self._hook_wait += time.perf_counter() - t0

    # ---------------------------------------------------------------------------------- submission / errors

    def _submit(self, item: _Item) -> None:
        if self.inline:
            self._run_inline(item)
        else:
            self._fifo.put(item)

    def _run(self, items: List[_Item]) -> None:
        """Submit and wait (main thread, after a drain)."""
        for item in items:
            self._submit(item)
        self._drain()

    def _run_inline(self, item: _Item) -> None:
        t0 = time.perf_counter()
        try:
            self._process(item)
        except BaseException as error:
            self._fail(error, traceback.format_exc())
            raise self._failure() from error
        finally:
            self._worker_busy += time.perf_counter() - t0

    def _drain(self) -> None:
        if not self.inline and self._fifo is not None:
            self._fifo.join(self._check_worker)
        self._raise_if_failed()

    def _fail(self, error, text: str) -> None:
        if self._error is None:
            self._error = (error, text)

    def _failure(self) -> HostPipelineError:
        if self._failure_obj is None:
            error, text = self._error
            self._failure_obj = HostPipelineError(f"the moss_ft host worker failed: {type(error).__name__}: {error}\n"
                                                  f"worker traceback:\n{text}")
        return self._failure_obj

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._failure()

    def _check_worker(self) -> None:
        """Run while blocked on the ring or the queue: a worker error, or a worker thread that is gone, raises."""
        self._raise_if_failed()
        thread = self._thread
        if not self.inline and (thread is None or not thread.is_alive()):
            self._fail(RuntimeError("the host worker thread is not running"), "(no traceback: the thread is gone)")
            raise self._failure()

    def _worker_main(self) -> None:
        if self.host_threads:
            torch.set_num_threads(self.host_threads)  # thread-local (design 1.4): the main thread keeps its count
        try:
            self.ops.worker_init()
        except BaseException as error:  # pragma: no cover - CUDA only
            self._fail(error, traceback.format_exc())
        fifo = self._fifo
        while True:
            item = fifo.get()
            if item is _STOP:
                fifo.task_done()
                return
            t0 = time.perf_counter()
            try:
                if self._error is None:
                    self._process(item)
                elif item.kind == _PIECE:  # after a failure: drop the rest, but give their ring entries back
                    self.d2h_ring.release(item.entry)
            except BaseException as error:
                self._fail(error, traceback.format_exc())
            finally:
                self._worker_busy += time.perf_counter() - t0
                fifo.task_done()

    # ------------------------------------------------------------------------------------ worker: processing

    def _process(self, item: _Item) -> None:
        if item.kind == _PIECE:
            self._on_piece(item)
        elif item.kind == _STEP:
            self._step_slot_from_acc(item.win, item.slot, item.ready)
        elif item.kind == _NORM:
            self._norm_of_acc(item.win, item.slot)
        else:  # pragma: no cover
            raise HostPipelineError(f"unknown work item {item.kind!r}")

    def _on_piece(self, item: _Item) -> None:
        w, s = item.win, item.slot
        slot = self.slots[s]
        try:
            self.ops.wait(item.done)  # the piece (and, for a first piece, the slot's norm) is on the host
            if item.first:
                if not math.isfinite(float(self.norm_host[s, item.j])):
                    w.poisoned[s] = True  # this window: no step, no upload; counted in WindowStats
                if w.first_j[s] < 0:
                    w.first_j[s] = item.j
                if self.pool.acc is None and self.per_parameter:
                    w.scratch_owner = s
            if w.poisoned[s]:
                return
            if w.plan.mode == "measure":  # GA 1, no known norm: only the window's norm (no accumulation, no step)
                self._measure_range(w, slot, self.d2h_ring.view(item.entry, item.dtype, item.numel))
                return
            if self.pool.acc is None and self.per_parameter and w.scratch_owner != s:
                raise HostPipelineError(f"{slot.name}: gradient pieces of two parameters interleaved; came_host at "
                                        "GA = 1 assembles one parameter at a time")
            src = self.d2h_ring.view(item.entry, item.dtype, item.numel)
            earlier = w.first_j[s] < item.j
            final = item.j == w.final_j
            if final and not w.plan.after_backward and not self.per_parameter:
                self._bump(w, s)
                acc = self._acc_flat(slot)[item.start:item.start + item.numel] if earlier else None
                self._step_range(w, slot, item.start, item.numel, src, acc, item.ready)
                if item.last:
                    w.finalized[s] = True
                return
            self._accumulate(slot, item.start, src, first=not earlier)
            if final and item.last:
                if w.plan.after_backward:
                    if w.plan.mode == "exact":
                        self._norm_of_acc(w, s)
                else:  # came_host: the whole gradient is in (acc or the scratch): step the parameter now
                    self._step_param(w, slot, item.ready)
        finally:
            self.d2h_ring.release(item.entry)

    def _acc_flat(self, slot: ParamSlot) -> torch.Tensor:
        if self.pool.acc is not None:
            return self.pool.view("acc", slot)
        if self._param_scratch is None:
            raise HostPipelineError(f"{slot.name}: no accumulator for an accumulated gradient")
        return self._param_scratch[:slot.numel]

    def _accumulate(self, slot: ParamSlot, start: int, src: torch.Tensor, first: bool) -> None:
        """acc[piece] = piece (the slot's first contribution in the window: no zero fill) or += piece, in fp32."""
        dst = self._acc_flat(slot)[start:start + src.numel()]
        if first:
            dst.copy_(src)
        elif src.dtype == torch.float32:
            dst.add_(src)
        else:  # converting chunks first is ~2x faster than a mixed-dtype add_ (design 1.4); same values
            n, chunk = src.numel(), self.acc_chunk
            for a in range(0, n, chunk):
                b = min(n, a + chunk)
                tmp = self._scratch[:b - a]
                tmp.copy_(src[a:b])
                dst[a:b].add_(tmp)

    def _bump(self, w: _Window, s: int) -> None:
        """The slot's step counter += 1 once per window, before its first stepped piece (torch's fused AdamW
        increments before its kernel; the kernel reads it)."""
        if not w.bumped[s]:
            self.pool.step(s).add_(1)
            w.bumped[s] = True

    def _measure_range(self, w: _Window, slot: ParamSlot, src: torch.Tensor) -> None:
        """A "measure" window: the piece's sum of squares in fp32 chunks (as the in-backward step forms it)."""
        n, chunk = src.numel(), self.step_chunk
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            t = self._scratch[:b - a]
            t.copy_(src[a:b])
            self._add_sumsq(w, t, slot)

    def _add_sumsq(self, w: _Window, t: torch.Tensor, slot: ParamSlot) -> None:
        value = torch.dot(t, t).item()
        if not math.isfinite(value):
            raise HostPipelineError(f"{slot.name}: non-finite accumulated gradient although every micro-step's norm "
                                    "was finite (fp32 overflow); the pool may be partly stepped - resume from a state")
        w.sumsq += value

    def _step_range(self, w: _Window, slot: ParamSlot, start: int, n: int, src: Optional[torch.Tensor],
                    acc: Optional[torch.Tensor], ready) -> None:
        """AdamW on elements [start, start + n) of a slot (at most one upload entry): per chunk t = piece (+ acc), the
        window's sum of squares, t *= clip coefficient, the fused kernel on master / m / v, cast into the upload entry;
        then the upload."""
        hp = w.hyper[slot.group]
        step = self.pool.step(slot.index)
        master = self.pool.view("master", slot)[start:start + n]
        exp_avg = self.pool.view("exp_avg", slot)[start:start + n]
        exp_avg_sq = self.pool.view("exp_avg_sq", slot)[start:start + n]
        entry = self._upload_entry()
        up = self.up_ring.view(entry, slot.compute_dtype, n)
        coef = w.coef
        chunk = self.step_chunk
        try:
            for a in range(0, n, chunk):
                b = min(n, a + chunk)
                t = self._scratch[:b - a]
                if src is not None:
                    t.copy_(src[a:b])
                    if acc is not None:
                        t.add_(acc[a:b])
                else:
                    t.copy_(acc[a:b])
                self._add_sumsq(w, t, slot)
                if coef is not None:
                    t.mul_(coef)
                self.optimizer.step_chunk(hp, master[a:b], t, exp_avg[a:b], exp_avg_sq[a:b], step)
                up[a:b].copy_(master[a:b])
            self._check_fp16_copy(slot, start, up)
        except BaseException:
            self.up_ring.release(entry)
            raise
        self._upload(slot, start, n, entry, up, ready)

    def _step_param(self, w: _Window, slot: ParamSlot, ready) -> None:
        """came_host: one parameter at a time from its whole fp32 gradient, then all its pieces uploaded."""
        s = slot.index
        grad = self._acc_flat(slot)
        n, chunk = slot.numel, self.step_chunk
        for a in range(0, n, chunk):
            self._add_sumsq(w, grad[a:min(n, a + chunk)], slot)
        if w.coef is not None:
            grad.mul_(w.coef)
        self._bump(w, s)
        master = self.pool.view("master", slot)
        self.optimizer.step_param(slot, w.hyper[slot.group], master, grad, self.pool.view("exp_avg", slot),
                                  self.pool.step(s))
        self._upload_all(slot, ready)
        w.finalized[s] = True
        if w.scratch_owner == s:
            w.scratch_owner = -1

    def _step_slot_from_acc(self, w: _Window, s: int, ready) -> None:
        """_STEP: a slot whose window gradient is complete in acc (flush / after-backward step)."""
        if w.first_j[s] < 0 or w.poisoned[s] or w.finalized[s]:
            return
        slot = self.slots[s]
        if self.per_parameter:
            self._step_param(w, slot, ready)
            return
        self._bump(w, s)
        acc = self._acc_flat(slot)
        per = self.piece_bytes // torch.empty((), dtype=slot.compute_dtype).element_size()
        for start in range(0, slot.numel, per):
            n = min(per, slot.numel - start)
            self._step_range(w, slot, start, n, None, acc[start:start + n], ready)
        w.finalized[s] = True

    def _norm_of_acc(self, w: _Window, s: int) -> None:
        """Exact clipping: the slot's accumulated gradient norm as clip_grad_norm_ takes it (torch.linalg.vector_norm
        of the parameter-shaped gradient, fp32)."""
        slot = self.slots[s]
        if w.slot_norm[s] is None:
            w.slot_norm[s] = torch.linalg.vector_norm(self._acc_flat(slot).view(slot.shape), 2.0)

    # ---------------------------------------------------------------------------------------- worker: uploads

    def _upload_entry(self) -> int:
        entry = self.up_ring.try_acquire()
        if entry is not None:
            return entry
        if not self._up_pending:
            raise HostPipelineError("the upload ring has no free and no pending entry")
        entry, event = self._up_pending.popleft()  # the oldest upload: wait until the DMA has read its entry
        self.ops.wait(event)
        return entry

    def _upload(self, slot: ParamSlot, start: int, n: int, entry: int, up: torch.Tensor, ready) -> None:
        # p.data: a write that does not bump the parameter's autograd version counter
        dst = slot.param.data.view(-1)[start:start + n]
        event = self.ops.copy_h2d(dst, up, ready)
        if event is None:
            self.up_ring.release(entry)
        else:
            self._up_pending.append((entry, event))

    def _upload_all(self, slot: ParamSlot, ready) -> None:
        master = self.pool.view("master", slot)
        per = self.piece_bytes // torch.empty((), dtype=slot.compute_dtype).element_size()
        for start in range(0, slot.numel, per):
            n = min(per, slot.numel - start)
            entry = self._upload_entry()
            up = self.up_ring.view(entry, slot.compute_dtype, n)
            try:
                up.copy_(master[start:start + n])
                self._check_fp16_copy(slot, start, up)
            except BaseException:
                self.up_ring.release(entry)
                raise
            self._upload(slot, start, n, entry, up, ready)

    def _check_fp16_copy(self, slot: ParamSlot, start: int, up: torch.Tensor) -> None:
        """fp16 compute copies: the cast of the master must stay finite (|w| < 65520 rounds to <= 65504). One aminmax
        over the upload entry (cache-warm, no temporary); fp16_range_report checks only the loaded weights."""
        if up.dtype != torch.float16 or up.numel() == 0:
            return
        low, high = torch.aminmax(up)
        if math.isfinite(float(low)) and math.isfinite(float(high)):
            return
        master = self.pool.view("master", slot)[start:start + up.numel()]
        bad = (~torch.isfinite(up)).nonzero()
        at = start + int(bad[0]) if bad.numel() else start
        raise HostPipelineError(
            f"{slot.name}: the fp32 master left the fp16 range (|w| up to {float(master.abs().max()):.6g} at element"
            f" {at}; fp16 holds up to 65504), so its fp16 compute copy would be inf and every later step's loss"
            " non-finite. The weights diverged: resume from an earlier state (e.g. with a lower learning rate), or"
            " train this parameter in bf16 (mixed_precision bf16). The pool may be partly stepped in this window")


def _is_power_of_two(value) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0 and math.frexp(value)[0] == 0.5


def _combine(coef, inv_scale: float):
    """The clip coefficient (None, a float or an fp32 0-dim tensor) times 1 / the loss scale, as an fp32 0-dim tensor
    (None when both are 1). inv_scale is a power of two, so the product is exact and t * (c * s) == round(t * c) * s:
    the worker's single multiply unscales and clips (design 8.3 "unscale fused into the host convert")."""
    inv_scale = float(inv_scale)
    if inv_scale == 1.0:
        return coef
    if coef is None:
        return torch.tensor(inv_scale, dtype=torch.float32)
    c = coef.detach().to(device="cpu", dtype=torch.float32).reshape(()) if torch.is_tensor(coef) \
        else torch.tensor(float(coef), dtype=torch.float32)
    return c * inv_scale


def _coefficient(value) -> Optional[torch.Tensor]:
    """A clip coefficient as the 0-dim fp32 tensor the gradients are multiplied by; None for none / 1.0 (x * 1 = x)."""
    if value is None:
        return None
    t = value.detach().to(device="cpu", dtype=torch.float32).reshape(()) if torch.is_tensor(value) \
        else torch.tensor(float(value), dtype=torch.float32)
    return None if float(t) == 1.0 else t
