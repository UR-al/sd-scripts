"""Shared helpers of tests/test_moss_ft_carousel_*.py (library/moss/ft_carousel*.py, docs/moss_ft_carousel_plan.md 6):
the tiny MOSS DiT (tests/_moss_ft_common.TINY: 3 joint + 2 refiner blocks = 5 units), an engine on the CPU, the
trainer's window loop for it, and the plain-autograd reference it must reproduce ("masked plain FT": the same
requires_grad toggling, one torch optimizer over every parameter). Not collected by pytest (no test_ prefix)."""
import copy
import re
from typing import Callable, Dict, List, Optional, Sequence

import torch

from _moss_ft_common import TINY, make_batch
from library.moss import ft_carousel, ft_host
from library.moss import ft_carousel_units as cu
from library.moss.models import MossTransformer

NO_DECAY_PREFIXES = ("video_embedder.", "text_embedder.proj.")


def build_model(seed: int = 0, dtype=torch.float32, ckpt: str = "plain", stream_fp32: bool = False) -> MossTransformer:
    torch.manual_seed(seed)
    model = MossTransformer(**TINY)
    g = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.02)
    model.to(dtype)
    if stream_fp32:
        model.enable_stream_fp32()
    if ckpt != "none":
        model.enable_gradient_checkpointing(unsloth_offload=(ckpt == "unsloth"))
    model.train()
    return model


def carousel_loss(model, seed: int) -> torch.Tensor:
    """One micro-step's loss as moss_train forms it in the carousel mode: the noisy input does NOT require grad."""
    x, t, emb, mask = make_batch(seed, dtype=torch.float32 if model.stream_fp32 else model.video_embedder.weight.dtype)
    y = model(x, t, emb, mask)
    return y.float().square().mean()


def named_trainable(model, frozen: Optional[str] = None):
    out = []
    for name, p in model.named_parameters():
        if frozen is not None and re.fullmatch(frozen, name):
            p.requires_grad_(False)
            continue
        out.append((name, p))
    return out


class Fake8bit(torch.optim.Optimizer):
    """A CPU stand-in with bitsandbytes AdamW8bit's state layout (it has no CPU kernels): uint8 state1 / state2, fp32
    absmax1 / absmax2 per 256 values, the shared qmap1 / qmap2 objects of name2qmap, an int step; fp32 states below
    4096 values. Its update is deterministic (a sign step scaled by the step count), only the layout matters."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, min_8bit_size=64):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.min_8bit_size = min_8bit_size
        self.name2qmap = {}
        self.fill_qmap()

    def fill_qmap(self):
        self.name2qmap["dynamic"] = torch.linspace(-1, 1, 256)
        self.name2qmap["udynamic"] = torch.linspace(0, 1, 256)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    if p.numel() < self.min_8bit_size:
                        state["state1"] = torch.zeros_like(p, dtype=torch.float32)
                        state["state2"] = torch.zeros_like(p, dtype=torch.float32)
                    else:
                        blocks = -(-p.numel() // 256)
                        state["state1"] = torch.zeros_like(p, dtype=torch.uint8)
                        state["qmap1"] = self.name2qmap["dynamic"]
                        state["state2"] = torch.zeros_like(p, dtype=torch.uint8)
                        state["qmap2"] = self.name2qmap["udynamic"]
                        state["absmax1"] = torch.zeros(blocks)
                        state["absmax2"] = torch.zeros(blocks)
                state["step"] += 1
                g = p.grad
                if state["state1"].dtype == torch.uint8:
                    state["state1"].add_((g > 0).to(torch.uint8))
                    state["state2"].add_(1)
                    state["absmax1"].add_(g.abs().max())
                    state["absmax2"].add_(1.0)
                    scale = state["absmax1"][0] / state["step"]
                else:
                    state["state1"].mul_(0.9).add_(g)
                    state["state2"].mul_(0.99).add_(g * g)
                    scale = 1.0
                p.add_(torch.sign(g) * scale, alpha=-group["lr"])


def make_optimizer(kind: str, groups, lr: float, wd: float = 0.0, came_kwargs: Optional[dict] = None):
    if kind == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=wd, fused=True)
    if kind == "came":
        return ft_carousel.cco.CarouselCAME(groups, lr=lr, **(came_kwargs or dict(betas=(0.9, 0.999, 0.9999),
                                                                                 bias_correction=True, weight_decay=wd)))
    if kind == "adamw8bit":
        return Fake8bit(groups, lr=lr, weight_decay=wd)
    raise ValueError(kind)


def make_engine(model, kind: str = "adamw", *, k: int = 5, period: int = 2, select: str = "random", seed: int = 3,
                always: str = "small", state: str = "keep", state_dtype: str = "fp32", lr: float = 1e-3,
                wd: float = 0.01, max_grad_norm: float = 0.0, frozen: Optional[str] = None, came_kwargs=None,
                max_skipped_steps: int = 8, attach: bool = True, seed_given: bool = True):
    named = named_trainable(model, frozen)
    slots = ft_host.make_slots(named)
    pool = ft_host.HostPool(slots, ("master",))
    pool.copy_master_from_params()
    handles = [torch.nn.Parameter(torch.empty(0), requires_grad=False) for _ in slots]
    groups = ft_carousel.param_groups(slots, handles, wd)
    optimizer = make_optimizer(kind, groups, lr, wd, came_kwargs)
    units = cu.discover_units([(s.name, s.shape) for s in slots], stem_unit=cu.stem_rotates(always))
    names = cu.always_names([s.name for s in slots], always)
    engine = ft_carousel.CarouselEngine(model, pool, slots, optimizer, handles, kind=kind, units=units, always=names,
                                        k=k, period=period, select=select, seed=seed, state_policy=state,
                                        state_dtype=state_dtype, device="cpu", max_grad_norm=max_grad_norm,
                                        max_skipped_steps=max_skipped_steps, always_policy=always,
                                        seed_given=seed_given)
    if attach:
        engine.attach()
    return engine


def train_engine(engine, windows: Sequence[Sequence[int]], *, start_step: int = 0, inv_scale: Callable = None,
                 loss_fn=carousel_loss, scheduler=None, after_window=None) -> List[ft_host.WindowStats]:
    """The moss_train loop: windows[w] = the batch seeds of window w. Returns every window's stats."""
    stats = []
    step = start_step
    for w, seeds in enumerate(windows):
        n = len(seeds)
        inv = 1.0 if inv_scale is None else inv_scale(step)
        engine.begin_window(n, ft_host.StepContext(groups=(), step_index=step, inv_grad_scale=inv))
        result = None
        for j, seed in enumerate(seeds):
            engine.begin_microstep(j)
            loss = loss_fn(engine.model, seed) / n
            (loss if inv == 1.0 else loss * (1.0 / inv)).backward()
            result = engine.end_microstep()
        stats.append(result)
        if scheduler is not None:
            scheduler.step()
        step += 1
        engine.after_step(step)
        if after_window is not None:
            after_window(w, result)
    return stats


def reference_groups(named, wd):
    decay = [p for n, p in named if p.dim() >= 2 and not n.startswith(NO_DECAY_PREFIXES)]
    rest = [p for n, p in named if not (p.dim() >= 2 and not n.startswith(NO_DECAY_PREFIXES))]
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": wd, "name": "decay"})
    if rest:
        groups.append({"params": rest, "weight_decay": 0.0, "name": "no_decay"})
    return groups


def train_reference(model, kind: str, windows, *, k: int, period: int, select: str, seed: int, always: str,
                    state: str = "keep", lr: float = 1e-3, wd: float = 0.01, max_grad_norm: float = 0.0,
                    frozen: Optional[str] = None, came_kwargs=None, start_step: int = 0, optimizer=None):
    """Masked plain FT: requires_grad on the always tensors and the selected units' tensors (select(step // period)),
    plain autograd accumulation, clip_grad_norm_ over the parameters with a gradient, one optimizer over every
    parameter (a parameter without gradient is skipped: its state is kept; reset pops the deactivated ones)."""
    named = named_trainable(model, frozen)
    units = cu.discover_units([(n, tuple(p.shape)) for n, p in named], stem_unit=cu.stem_rotates(always))
    selector = cu.Selector(len(units), k, select, seed)
    always_set = set(cu.always_names([n for n, _ in named], always))
    if optimizer is None:
        optimizer = make_optimizer(kind, reference_groups(named, wd), lr, wd, came_kwargs)
    params = dict(named)
    current = None
    step = start_step
    for seeds in windows:
        active = selector(step // period)
        if active != current:
            names = always_set | {n for u in active for n in units[u].names}
            for name, p in named:
                was = p.requires_grad
                p.requires_grad_(name in names)
                if state == "reset" and was and name not in names:
                    optimizer.state.pop(p, None)
            current = active
        n = len(seeds)
        for s in seeds:
            (carousel_loss(model, s) / n).backward()
        with_grad = [p for _, p in named if p.grad is not None]
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(with_grad, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
    return optimizer, units, params


def master_of(engine, name: str) -> torch.Tensor:
    """The authoritative fp32 master of a tensor: the device copy while active, the pool otherwise."""
    for slot in engine.slots:
        if slot.name == name:
            if slot.index in engine._loc:
                return engine.handles[slot.index].detach().reshape(-1)
            return engine.pool.view("master", slot)
    raise KeyError(name)


def state_of(engine, name: str) -> Dict:
    for slot in engine.slots:
        if slot.name == name:
            return engine._state_of(slot.index) or {}
    raise KeyError(name)


def assert_matches_reference(engine, ref_params: Dict[str, torch.Tensor], ref_opt, what: str = "") -> None:
    for slot in engine.slots:
        p = ref_params[slot.name]
        mine = master_of(engine, slot.name)
        assert torch.equal(mine, p.detach().reshape(-1).float()), (
            f"{what}: master of {slot.name} differs, max |d| {(mine - p.detach().reshape(-1)).abs().max().item():.3e}")
        assert torch.equal(slot.param.detach().reshape(-1), mine.to(slot.param.dtype)), f"{what}: I1 {slot.name}"
        theirs = ref_opt.state.get(p) or {}
        ours = state_of(engine, slot.name)
        assert set(k for k in theirs if k not in ("qmap1", "qmap2")) == set(
            k for k in ours if k not in ("qmap1", "qmap2")), (what, slot.name, sorted(theirs), sorted(ours))
        for key, value in theirs.items():
            if key in ("qmap1", "qmap2"):
                continue  # bitsandbytes' shared maps (the device state refers to the optimizer's objects)
            other = ours[key]
            if torch.is_tensor(value):
                assert torch.equal(other.float() if other.dtype == torch.bfloat16 else other, value), (
                    f"{what}: state {key} of {slot.name}")
            else:
                assert other == value, (what, slot.name, key)


def clone(model):
    return copy.deepcopy(model)
