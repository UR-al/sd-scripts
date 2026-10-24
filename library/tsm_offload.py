# TSM-LoKr core-expert offload for training (network_args algo=tsm-lokr, LyCORIS fixed paths = the default).
# Opt-in through plain settings, never switched on or changed at runtime: tsm_offload_states, tsm_resident_cores,
# tsm_grouped_accumulation.
#
# A core expert "group" i is the scale-0 expert i of every TSM-LoKr module (all modules route the same t to the same
# interval). One optimizer step only touches the groups its micro-steps used, so:
# * tsm_offload_states=True: the AdamW exp_avg / exp_avg_sq (and the fused step counters) of every core group live in
#   pinned CPU RAM between optimizer steps. The optimizer step pre-hook copies the used groups' states to one GPU buffer
#   per group (activations are already freed then), the post-hook copies them back on a side stream (overlapping the
#   next forward) and points the optimizer at the host copies again. Copies only, never casts or recomputation: the run
#   is bit-identical to keeping everything on the GPU. torch's fused AdamW stays as it is (only params with a grad are
#   stepped or touched, torch/optim/adam.py:149).
# * tsm_resident_cores=K < n_core: only the K most recently used core groups keep their fp32 weights on the GPU; the
#   others wait in pinned RAM (p.data is a pinned host view). A missing group is uploaded at set_timestep, before the
#   forward; a group with gradients or in use by the current micro-step is never evicted. If no group can be evicted,
#   the micro-step stops with an error (the start-up check sizes K so this cannot happen in a normal run).
# * tsm_grouped_accumulation=True: all samples of one optimizer step come from the core interval of the window's first
#   draw (an exact draw from the trainer's sampler); the other samples are redrawn until they fall in that interval
#   (exact conditional). Each sample's marginal t distribution is unchanged; see grouped_draw().
#
# Host memory comes in 64 MiB slabs (exact sizes: torch's pinned allocator rounds every allocation up to a power of
# two), one set per group, filled first-fit. At the optimizer step each used slab gets a GPU buffer of its own (one copy
# per slab each way): 64 MiB pieces fit in the blocks the caching allocator keeps from the freed activations, where one
# buffer per group (1.29 GiB for an Anima-2.9B factor-4 core expert) would need new segments on top of them. The next
# micro-step waits for the write-backs before its forward allocates, so their buffers are free again by then (peak =
# max(activations, step-time states), not their sum); it costs the overlap of the write-back with the next forward.

from __future__ import annotations

import itertools
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

SLAB_NUMEL = 16 * 1024 * 1024  # 64 MiB of fp32
# TSM_LOKR_MEMLOG=1: log CUDA memory before each forward and at each optimizer step (GPU check of the memory table);
# any other value is a file that gets one JSON line per entry instead
_MEMLOG = os.environ.get("TSM_LOKR_MEMLOG", "").strip()
_MEMLOG = "" if _MEMLOG == "0" else _MEMLOG


class SlabLayout:
    """Tensors (numel each) packed first-fit into slabs of exactly `slab_numel` elements (a tensor larger than a slab
    gets a slab of its own size). places[k] = (slab, offset); the GPU mirror uses base[slab] + offset."""

    def __init__(self, numels: Sequence[int], slab_numel: int, dtype: torch.dtype, pin: bool):
        self.places: List[Tuple[int, int]] = []
        sizes: List[int] = []
        used: List[int] = []
        for n in numels:
            n = int(n)
            k = next((s for s in range(len(sizes)) if sizes[s] - used[s] >= n), None)
            if k is None:
                sizes.append(max(slab_numel, n))
                used.append(0)
                k = len(sizes) - 1
            self.places.append((k, used[k]))
            used[k] += n
        self.used = used
        self.slabs = [torch.empty(s, dtype=dtype, pin_memory=pin) for s in sizes]
        self.base = list(itertools.accumulate([0] + used[:-1])) if used else []
        self.total = sum(used)
        self.numels = [int(n) for n in numels]
        self.host_bytes = sum(s for s in sizes) * torch.empty((), dtype=dtype).element_size()

    def host_view(self, k: int, shape) -> torch.Tensor:
        s, off = self.places[k]
        return self.slabs[s][off:off + self.numels[k]].view(shape)

    def dev_view(self, buf, k: int, shape) -> torch.Tensor:
        """Tensor k in a GPU mirror: `buf` is one buffer per slab (a list, as the manager uses) or one flat buffer."""
        s, off = self.places[k]
        if isinstance(buf, (list, tuple)):
            return buf[s][off:off + self.numels[k]].view(shape)
        start = self.base[s] + off
        return buf[start:start + self.numels[k]].view(shape)


class _Group:
    def __init__(self, index: int, params: List[nn.Parameter]):
        self.index = index
        self.params = params
        self.state: Optional[SlabLayout] = None  # 2 entries per param: exp_avg, exp_avg_sq
        self.steps_host: Optional[torch.Tensor] = None
        self.has_state = [False] * len(params)
        self.weights: Optional[SlabLayout] = None  # LRU mode: fp32 master of a non-resident group
        self.resident = True
        self.dirty = False  # GPU weights newer than the host master
        self.host_valid = False  # the host master holds the current weights
        self.last_used = 0
        self.state_event = None  # pending D2H of the state
        self.weight_event = None  # pending D2H of the weights
        self.dev_buf: Optional[List[Optional[torch.Tensor]]] = None  # one GPU buffer per used state slab
        self.dev_steps: Optional[torch.Tensor] = None


class ExpertOffloadManager:
    """See the module comment. `groups[i]` = parameters of core expert i over all TSM-LoKr modules.

    device: the training device. On a CPU device the "device" buffers are separate CPU tensors and every copy is
    synchronous (CPU tests); on CUDA the host slabs are pinned and the write-backs run on a side stream."""

    def __init__(self, groups: Sequence[Sequence[nn.Parameter]], optimizer: Optional[torch.optim.Optimizer],
                 device, *, core_index: Callable[[int], int], offload_states: bool, resident_cores: int,
                 grouped: bool = False, core_index_t: Optional[Callable[[float], int]] = None,
                 slab_numel: int = SLAB_NUMEL, pin: Optional[bool] = None):
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.pin = self.cuda if pin is None else bool(pin)
        self.copy_stream = torch.cuda.Stream(self.device) if self.cuda else None
        self.groups = [_Group(i, list(ps)) for i, ps in enumerate(groups)]
        self.optimizer = optimizer
        self.core_index = core_index
        self.core_index_t = core_index_t or (lambda t: core_index(int(round(float(t)))))
        self.offload_states = bool(offload_states)
        self.capacity = int(resident_cores)
        self.grouped = bool(grouped)
        self.window_core: Optional[int] = None
        self.slab_numel = int(slab_numel)
        self._clock = 0
        self._current: set = set()
        self._stepped: List[_Group] = []
        self._stash: Dict[int, dict] = {}
        self._where: Dict[int, Tuple[_Group, int]] = {id(p): (g, j) for g in self.groups for j, p in enumerate(g.params)}
        self._handles = []
        self.stream_steps = False
        self._pending = False  # copies queued on copy_stream that the CPU has not waited for
        self._memlog_calls = 0
        self.uploads = 0
        self.evictions = 0

        for g in self.groups:
            for p in g.params:
                if self.offload_states and p.dtype != torch.float32:
                    raise ValueError(
                        f"tsm-lokr: tsm_offload_states needs fp32 expert weights (got {p.dtype}); turn off full_bf16/"
                        f"full_fp16 or set tsm_offload_states=False / tsm_offload_states 는 fp32 전문가 가중치가 필요합니다"
                    )

        if optimizer is not None and (self.offload_states or self.capacity < len(self.groups)):
            managed = {id(p) for g in self.groups for p in g.params}
            in_opt = [p for pg in optimizer.param_groups for p in pg["params"] if id(p) in managed]
            fused = [bool(pg.get("fused")) or bool(pg.get("capturable")) for pg in optimizer.param_groups
                     if any(id(p) in managed for p in pg["params"])]
            self.stream_steps = bool(fused) and all(fused)
            if self.offload_states and in_opt:
                for g in self.groups:
                    g.state = SlabLayout([n for p in g.params for n in (p.numel(), p.numel())], self.slab_numel,
                                         torch.float32, self.pin)
                    g.steps_host = torch.zeros(len(g.params), dtype=torch.float32, pin_memory=self.pin)
                    for j, p in enumerate(g.params):
                        if optimizer.state.get(p):
                            self._adopt_state(g, j, optimizer.state[p], sync=True)
            self._handles.append(optimizer.register_step_pre_hook(self._step_pre))
            self._handles.append(optimizer.register_step_post_hook(self._step_post))
            self._handles.append(optimizer.register_state_dict_pre_hook(lambda opt: self.flush()))
            self._handles.append(optimizer.register_load_state_dict_pre_hook(self._load_pre))
            self._handles.append(optimizer.register_load_state_dict_post_hook(self._load_post))
        elif optimizer is not None and self.grouped:
            self._handles.append(optimizer.register_step_pre_hook(self._step_pre))
            self._handles.append(optimizer.register_step_post_hook(self._step_post))

        if self.capacity < len(self.groups):
            for g in self.groups:
                g.weights = SlabLayout([p.numel() for p in g.params], self.slab_numel, torch.float32, self.pin)
            for g in self.groups[self.capacity:]:
                self._evict(g)
            self.flush()

    # ---------------------------------------------------------------------------------------------------- helpers

    def flush(self) -> None:
        """Wait for every pending copy before the CPU reads or writes host copies (saves, loads, state_dict): the
        write-backs run on copy_stream, but a group's first eviction and the uploads run on the compute stream."""
        if self.cuda:
            torch.cuda.synchronize(self.device)
            self._pending = False

    def _memlog(self, where: str) -> None:
        """TSM_LOKR_MEMLOG: allocated / reserved / peaks at this phase (every call for the first 400, then 1 in 200)."""
        if not (_MEMLOG and self.cuda):
            return
        self._memlog_calls += 1
        if self._memlog_calls > 400 and self._memlog_calls % 200:
            return
        gib = 2 ** 30
        d = self.device
        row = {"where": where, "allocated": torch.cuda.memory_allocated(d) / gib,
               "reserved": torch.cuda.memory_reserved(d) / gib, "max_allocated": torch.cuda.max_memory_allocated(d) / gib,
               "max_reserved": torch.cuda.max_memory_reserved(d) / gib}
        if _MEMLOG == "1":
            logger.info(f"tsm-lokr memlog {where}: allocated {row['allocated']:.2f}, reserved {row['reserved']:.2f},"
                        f" max allocated {row['max_allocated']:.2f}, max reserved {row['max_reserved']:.2f} GiB")
        else:
            import json

            with open(_MEMLOG, "a", encoding="utf-8") as f:
                print(json.dumps(row), file=f)

    def _wait_copies(self) -> None:
        """Wait for the queued write-backs, so that the device buffers they hold (record_stream) can be reused by the
        allocations that follow instead of reserving new memory next to them."""
        if self._pending:
            self.copy_stream.synchronize()
            self._pending = False

    def host_bytes(self) -> int:
        n = 0
        for g in self.groups:
            for lay in (g.state, g.weights):
                if lay is not None:
                    n += lay.host_bytes
        return n

    def resident_groups(self) -> List[int]:
        return [g.index for g in self.groups if g.resident]

    def _cur_stream(self):
        return torch.cuda.current_stream(self.device) if self.cuda else None

    def _adopt_state(self, g: _Group, j: int, st: dict, sync: bool = False) -> None:
        """Copy a param's optimizer state into its host slots and point the optimizer at them."""
        p = g.params[j]
        hm = g.state.host_view(2 * j, p.shape)
        hv = g.state.host_view(2 * j + 1, p.shape)
        nb = not sync and self.cuda
        hm.copy_(st["exp_avg"], non_blocking=nb)
        hv.copy_(st["exp_avg_sq"], non_blocking=nb)
        if self.cuda and nb:
            st["exp_avg"].record_stream(self.copy_stream)
            st["exp_avg_sq"].record_stream(self.copy_stream)
        st["exp_avg"], st["exp_avg_sq"] = hm, hv
        if self.stream_steps and isinstance(st.get("step"), torch.Tensor):
            g.steps_host[j].copy_(st["step"], non_blocking=nb)
            if self.cuda and nb and st["step"].is_cuda:
                st["step"].record_stream(self.copy_stream)
            st["step"] = g.steps_host[j]
        g.has_state[j] = True

    # ------------------------------------------------------------------------------------------------- micro-steps

    def prepare_micro_step(self, ints: Sequence[int]) -> None:
        """Called by set_timestep with the integer timesteps of the coming forward: mark the core groups in use and
        upload missing ones (tsm_resident_cores < n_core)."""
        self._wait_copies()  # last step's state write-backs: free their buffers before this forward allocates
        self._memlog("before forward")  # the max values include the previous micro-step's backward
        needed = sorted({self.core_index(int(t)) for t in ints})
        self._clock += 1
        for i in needed:
            self.groups[i].last_used = self._clock
        self._current = set(needed)
        for i in needed:
            g = self.groups[i]
            if g.resident:
                continue
            while sum(1 for x in self.groups if x.resident) >= self.capacity:
                victims = [x for x in self.groups if x.resident and x.index not in self._current
                           and all(p.grad is None for p in x.params)]
                if not victims:
                    raise RuntimeError(
                        f"tsm-lokr: tsm_resident_cores={self.capacity} is too small for this step (core experts"
                        f" {sorted(x.index for x in self.groups if x.resident)} hold gradients or are in use, {i} is"
                        f" needed); raise tsm_resident_cores / tsm_resident_cores={self.capacity} 가 이 스텝에 너무"
                        " 작습니다. 값을 올리세요"
                    )
                self._evict(min(victims, key=lambda x: x.last_used))
            self._wait_copies()  # a dirty eviction's device copy is reusable only after its write-back
            self._upload(g)

    def _evict(self, g: _Group) -> None:
        if not g.resident:
            return
        lay = g.weights
        if g.dirty:
            if self.cuda:
                self.copy_stream.wait_stream(self._cur_stream())
                with torch.cuda.stream(self.copy_stream):
                    for j, p in enumerate(g.params):
                        lay.host_view(j, p.shape).copy_(p.data, non_blocking=True)
                        p.data.record_stream(self.copy_stream)
                    ev = torch.cuda.Event()
                    ev.record(self.copy_stream)
                g.weight_event = ev
                self._pending = True
            else:
                for j, p in enumerate(g.params):
                    lay.host_view(j, p.shape).copy_(p.data)
            g.dirty = False
        elif not g.host_valid:
            # first eviction of a clean group: the host master has never been written
            for j, p in enumerate(g.params):
                lay.host_view(j, p.shape).copy_(p.data, non_blocking=self.cuda)
            if self.cuda:
                ev = torch.cuda.Event()
                ev.record(self._cur_stream())
                g.weight_event = ev
        g.host_valid = True
        for j, p in enumerate(g.params):
            p.data = lay.host_view(j, p.shape)
        g.resident = False
        self.evictions += 1

    def _upload(self, g: _Group) -> None:
        lay = g.weights
        if self.cuda and g.weight_event is not None:
            self._cur_stream().wait_event(g.weight_event)
            g.weight_event = None
        for j, p in enumerate(g.params):
            dev = torch.empty(p.shape, dtype=p.dtype, device=self.device)
            dev.copy_(lay.host_view(j, p.shape), non_blocking=self.cuda)
            p.data = dev
        g.resident = True
        self.uploads += 1

    # --------------------------------------------------------------------------------------------- optimizer hooks

    def _step_pre(self, opt, args, kwargs) -> None:
        self._stepped = [g for g in self.groups if any(p.grad is not None for p in g.params)]
        if not self.offload_states:
            return
        for g in self._stepped:
            if g.state is None or not any(g.has_state):
                continue  # first use: torch creates the state on the GPU in this step
            if self.cuda and g.state_event is not None:
                self._cur_stream().wait_event(g.state_event)
                g.state_event = None
            lay = g.state
            bufs: List[Optional[torch.Tensor]] = []
            for s, slab in enumerate(lay.slabs):  # one buffer per slab (see the module comment)
                n = lay.used[s]
                buf = torch.empty(n, dtype=torch.float32, device=self.device) if n else None
                if buf is not None:
                    buf.copy_(slab[:n], non_blocking=self.cuda)
                bufs.append(buf)
            steps = g.steps_host.to(self.device, non_blocking=self.cuda, copy=True) if self.stream_steps else None
            for j, p in enumerate(g.params):
                if not g.has_state[j]:
                    continue
                st = opt.state[p]
                st["exp_avg"] = lay.dev_view(bufs, 2 * j, p.shape)
                st["exp_avg_sq"] = lay.dev_view(bufs, 2 * j + 1, p.shape)
                if steps is not None:
                    st["step"] = steps[j]
            g.dev_buf, g.dev_steps = bufs, steps
        self._memlog("optimizer step")  # the used cores' states are on the GPU now, the activations are freed

    def _step_post(self, opt, args, kwargs) -> None:
        self.window_core = None  # grouped accumulation: the next micro-step starts a new window
        for g in self._stepped:
            if g.weights is not None:
                g.dirty = True
        if not self.offload_states:
            self._stepped = []
            return
        ctx = None
        if self.cuda:
            self.copy_stream.wait_stream(self._cur_stream())
            ctx = torch.cuda.stream(self.copy_stream)
            ctx.__enter__()
            self._pending = True
        try:
            for g in self._stepped:
                if g.state is None:
                    continue
                lay = g.state
                bufs, steps = g.dev_buf, g.dev_steps
                if bufs is not None:
                    for s, slab in enumerate(lay.slabs):
                        if bufs[s] is not None:
                            slab[:lay.used[s]].copy_(bufs[s], non_blocking=self.cuda)
                    if steps is not None:
                        g.steps_host.copy_(steps, non_blocking=self.cuda)
                    if self.cuda:
                        for buf in bufs:
                            if buf is not None:
                                buf.record_stream(self.copy_stream)
                        if steps is not None:
                            steps.record_stream(self.copy_stream)
                for j, p in enumerate(g.params):
                    st = opt.state.get(p)
                    if not st:
                        continue
                    if g.has_state[j]:
                        st["exp_avg"] = lay.host_view(2 * j, p.shape)
                        st["exp_avg_sq"] = lay.host_view(2 * j + 1, p.shape)
                        if steps is not None:
                            st["step"] = g.steps_host[j]
                    else:
                        self._adopt_state(g, j, st)  # created by torch in this step
                g.dev_buf = g.dev_steps = None
                if self.cuda:
                    ev = torch.cuda.Event()
                    ev.record(self.copy_stream)
                    g.state_event = ev
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        self._stepped = []

    def _load_pre(self, opt, state_dict):
        self.flush()
        saved = list(itertools.chain.from_iterable(pg["params"] for pg in state_dict["param_groups"]))
        current = list(itertools.chain.from_iterable(pg["params"] for pg in opt.param_groups))
        self._stash = {}
        if len(saved) != len(current):
            return None  # torch raises its own error
        state = dict(state_dict["state"])
        for sid, p in zip(saved, current):
            if id(p) in self._where and sid in state:
                self._stash[id(p)] = state.pop(sid)
        out = dict(state_dict)
        out["state"] = state
        return out

    def _load_post(self, opt) -> None:
        for g in self.groups:
            for j, p in enumerate(g.params):
                st = self._stash.get(id(p))
                if st is None:
                    g.has_state[j] = False
                    opt.state.pop(p, None)
                    continue
                st = dict(st)
                if self.offload_states and g.state is not None:
                    if self.stream_steps and isinstance(st.get("step"), torch.Tensor):
                        st["step"] = st["step"].to(torch.float32)
                    self._adopt_state(g, j, st, sync=True)
                    opt.state[p] = st
                else:
                    for k, v in list(st.items()):
                        if isinstance(v, torch.Tensor):
                            if k == "step" and not self.stream_steps:
                                continue  # for-loop/foreach AdamW keeps its step on the CPU
                            st[k] = v.to(device=self.device, dtype=torch.float32 if k == "step" else p.dtype)
                    opt.state[p] = st
        self._stash = {}

    # --------------------------------------------------------------------------------------- network state hooks

    def attach_network_hooks(self, network: nn.Module) -> None:
        def sd_pre(module, prefix, keep_vars):
            self.flush()

        def load_pre(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            self.flush()

        def load_post(module, incompatible_keys):
            for g in self.groups:
                if g.resident and g.weights is not None:
                    g.dirty = True

        def sd_post(module, state_dict, prefix, local_metadata):
            # Swapped-out experts are views into shared host slabs. accelerate's save_state drops every tensor that
            # shares storage with another but the first (clean_state_dict_for_safetensors: 30 of 120 keys lost in the
            # CPU test) and safetensors' save_file refuses them (LyCORIS save_weights without a dtype). Give those
            # entries own copies.
            for k, v in list(state_dict.items()):
                if isinstance(v, torch.Tensor) and v.device.type == "cpu" and \
                        v.untyped_storage().nbytes() > v.numel() * v.element_size():
                    state_dict[k] = v.clone()

        self._handles.append(network.register_state_dict_pre_hook(sd_pre))
        if any(g.weights is not None for g in self.groups):
            self._handles.append(network.register_state_dict_post_hook(sd_post))
        self._handles.append(network.register_load_state_dict_pre_hook(load_pre))
        self._handles.append(network.register_load_state_dict_post_hook(load_post))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    # ------------------------------------------------------------------------------------- grouped accumulation

    def grouped_draw(self, draw: Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                     max_draws: int = 100000):
        """(noisy_model_input, timesteps, sigmas) with every row in the window's core interval.

        The first draw of a window fixes the interval (its first row: an exact draw from the trainer's sampler, so the
        interval is distributed like the sampler's); every other row is redrawn until its timestep falls in that
        interval (rejection: the exact conditional). Row r of the result is row r of the accepting draw."""
        nmi, ts, sg = draw()
        cores = [self.core_index_t(v) for v in ts.detach().float().reshape(-1).tolist()]
        if self.window_core is None:
            self.window_core = cores[0]
        target = self.window_core
        missing = [r for r, c in enumerate(cores) if c != target]
        if not missing:
            return nmi, ts, sg
        nmi, ts, sg = nmi.clone(), ts.clone(), sg.clone()
        draws = 1
        while missing:
            draws += 1
            if draws > max_draws:
                raise RuntimeError(f"tsm-lokr grouped accumulation: no timestep in core interval {target} after"
                                   f" {max_draws} draws / {max_draws}번 뽑아도 구간 {target} 의 타임스텝이 나오지 않았습니다")
            n2, t2, s2 = draw()
            c2 = [self.core_index_t(v) for v in t2.detach().float().reshape(-1).tolist()]
            still = []
            for r in missing:
                if c2[r] == target:
                    nmi[r], ts[r], sg[r] = n2[r], t2[r], s2[r]
                else:
                    still.append(r)
            missing = still
        return nmi, ts, sg
