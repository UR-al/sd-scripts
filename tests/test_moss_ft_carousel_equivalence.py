"""CPU tests: the Carousel engine reproduces plain training bit for bit (docs/moss_ft_carousel_plan.md 6, items
10-14).

* K = U (no switching) == plain fused-AdamW full FT (masters, compute copies, exp_avg / exp_avg_sq, steps), GA 1 / 3
  with a partial window, clipping on / off, weight decay, an LR scheduler;
* CarouselCAME == ft_optim.HostCAME's step_param math (the came_host mode), tensor by tensor;
* bf16 compute copies, K = U == the existing adamw_host GradPipeline (masters and compute copies);
* K < U with switches == "masked plain FT" (the same requires_grad toggling, one torch optimizer, keep / reset), for
  every selection, GA 2, adamw and came, with and without backward truncation;
* fp16 loss scale: a power-of-two scale changes no bit; an inf skips the whole window; the skip streak stops a run.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_carousel_equivalence.py -q -p no:cacheprovider
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

import _moss_ft_common as common
from _moss_ft_carousel_common import (assert_matches_reference, build_model, carousel_loss, clone, make_engine,
                                      train_engine, train_reference)
from library.moss import ft_carousel, ft_host
from library.moss.ft_carousel_optim import CarouselCAME
from library.moss.ft_optim import HostAdamW, HostCAME


def windows_of(sizes, start=0):
    return common.seeds(sizes, start)


@pytest.mark.parametrize("ga,clip,sched", [(1, 0.0, False), (3, 1.0, True), (3, 0.05, False)])
def test_k_equals_u_is_plain_fused_adamw(ga, clip, sched):
    model = build_model()
    ref = clone(model)
    engine = make_engine(model, "adamw", k=5, period=1, max_grad_norm=clip, lr=2e-3, wd=0.05)
    sizes = [ga] * 3 + ([2] if ga > 1 else [1])  # a partial last window
    windows = windows_of(sizes)
    scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda s: 1.0 / (1 + s)) if sched else None
    stats = train_engine(engine, windows, scheduler=scheduler)
    ref_named = list(ref.named_parameters())
    from _moss_ft_carousel_common import make_optimizer, reference_groups

    opt = make_optimizer("adamw", reference_groups(ref_named, 0.05), 2e-3, 0.05)
    ref_sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 / (1 + s)) if sched else None
    norms = []
    for seeds in windows:
        for s in seeds:
            (carousel_loss(ref, s) / len(seeds)).backward()
        with_grad = [p for _, p in ref_named if p.grad is not None]
        if clip > 0:
            norms.append(float(torch.nn.utils.clip_grad_norm_(with_grad, clip)))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if ref_sched is not None:
            ref_sched.step()
    assert_matches_reference(engine, dict(ref_named), opt, f"GA {ga}")
    assert engine.switches == 0 and all(s.stepped_params == len(engine.slots) for s in stats)
    if clip > 0:
        assert [s.grad_norm for s in stats] == norms
        assert any(s.clip_coef < 1.0 for s in stats) == (clip < 0.1)


def test_came_matches_the_host_came_math():
    """CarouselCAME (device cores, per-parameter step) == HostCAME.step_param (came_host) on fp32 tensors, with the
    bias correction and beta3 0.9999, factored and unfactored tensors, three steps."""
    torch.manual_seed(0)
    shapes = [(12, 8), (5, 3, 4), (7,)]
    kwargs = dict(betas=(0.9, 0.999, 0.9999), bias_correction=True, weight_decay=0.02, non_factored_confidence=True)
    params = [torch.nn.Parameter(torch.randn(s)) for s in shapes]
    mine = CarouselCAME([{"params": params, "weight_decay": 0.02}], lr=1e-3, **kwargs)
    host = HostCAME([{"params": [torch.nn.Parameter(torch.zeros(s)) for s in shapes], "weight_decay": 0.02}], lr=1e-3,
                    **kwargs)
    slots = ft_host.make_slots([(f"p{i}", p) for i, p in enumerate(host.param_groups[0]["params"])])
    pool = ft_host.HostPool(slots, HostCAME.POOL_ROLES)
    for slot, p in zip(slots, params):
        pool.view("master", slot).copy_(p.detach().reshape(-1))
    host.bind(slots, pool)
    for step in range(1, 4):
        grads = [torch.randn(s) for s in shapes]
        for p, g in zip(params, grads):
            p.grad = g.clone()
        mine.step()
        hp = HostCAME.snapshot_group(host.param_groups[0])
        for slot, g in zip(slots, grads):
            pool.steps[slot.index] += 1
            host.step_param(slot, hp, pool.view("master", slot), g.clone().reshape(-1), pool.view("exp_avg", slot),
                            pool.step(slot))
        for slot, p in zip(slots, params):
            assert torch.equal(p.detach().reshape(-1), pool.view("master", slot)), (step, slot.name)
            state = mine.state[p]
            assert state["step"] == step
            assert torch.equal(state["exp_avg"].reshape(-1), pool.view("exp_avg", slot))
            for key, value in host.slot_state(slot.index).items():
                assert torch.equal(state[key], value), (step, slot.name, key)


def test_bf16_k_equals_u_matches_the_adamw_host_pipeline():
    """bf16 compute copies (fp32 masters): K = U equals the existing adamw_host GradPipeline (GA 2, no clipping):
    masters and compute copies bit for bit."""
    model = build_model(dtype=torch.bfloat16)
    ref = clone(model)
    engine = make_engine(model, "adamw", k=5, period=1, lr=2e-3, wd=0.05)
    engine.optimizer.param_groups[0]["betas"] = engine.optimizer.param_groups[1]["betas"] = (0.9, 0.99)
    windows = windows_of([2, 2, 1])
    train_engine(engine, windows)
    named = list(ref.named_parameters())
    opt = HostAdamW(common.param_groups(named, 0.05), lr=2e-3, betas=(0.9, 0.99), eps=1e-8)
    pipe, pool, slots = common.make_pipeline(named, opt, 2, inline=True)
    closures = [[(lambda s=s: carousel_loss(ref, s)) for s in seeds] for seeds in windows]
    common.train_pipeline(pipe, opt, closures)
    pipe.close()
    for slot, mine in zip(slots, engine.slots):
        assert torch.equal(pool.view("master", slot), engine.pool.view("master", mine) if mine.index not in engine._loc
                           else engine.handles[mine.index].detach().reshape(-1)), slot.name
        assert torch.equal(slot.param.detach(), mine.param.detach()), slot.name


CASES = [
    ("adamw", 2, "random", "small", "keep", 0.0),
    ("adamw", 2, "round_robin", "small", "reset", 0.7),
    ("adamw", 2, "window", "head", "keep", 0.0),
    ("adamw", 1, "window", "none", "reset", 0.3),
    ("came", 2, "random", "head", "keep", 0.5),
    ("came", 3, "window", "small", "reset", 0.0),
    ("adamw", 2, "random", "rotate_stem", "keep", 0.4),
    ("came", 2, "window", "rotate_stem", "reset", 0.0),
]


@pytest.mark.parametrize("kind,k,select,always,state,clip", CASES)
def test_switching_equals_masked_plain_fine_tuning(kind, k, select, always, state, clip):
    model = build_model()
    ref = clone(model)
    engine = make_engine(model, kind, k=k, period=2, select=select, always=always, state=state, max_grad_norm=clip)
    windows = windows_of([2] * 9)
    switches = []
    train_engine(engine, windows, after_window=lambda w, s: switches.append(engine.switches))
    assert engine.switches >= 3
    opt, _, params = train_reference(ref, kind, windows, k=k, period=2, select=select, seed=3, always=always,
                                     state=state, max_grad_norm=clip)
    assert_matches_reference(engine, params, opt, f"{kind} {select} {always} {state}")


def test_power_of_two_loss_scale_changes_no_bit():
    a, b = build_model(), build_model()
    ea = make_engine(a, "adamw", k=2, period=2, max_grad_norm=0.5)
    eb = make_engine(b, "adamw", k=2, period=2, max_grad_norm=0.5)
    windows = windows_of([2] * 5)
    sa = train_engine(ea, windows)
    sb = train_engine(eb, windows, inv_scale=lambda step: 2.0 ** -12)
    assert [s.grad_norm for s in sa] == [s.grad_norm for s in sb]
    assert all(s.loss_scale == 4096.0 for s in sb)
    for x, y in zip(ea.slots, eb.slots):
        assert torch.equal(x.param, y.param), x.name


def test_an_inf_skips_the_whole_window_and_the_next_one_steps():
    model = build_model()
    engine = make_engine(model, "adamw", k=2, period=100, max_skipped_steps=2)
    train_engine(engine, windows_of([1]))
    before = {s.name: s.param.detach().clone() for s in engine.slots}
    states = {s.name: {k: v.clone() for k, v in (engine._state_of(s.index) or {}).items()} for s in engine.slots}

    def poisoned(model, seed):
        return carousel_loss(model, seed) * float("inf")

    stats = train_engine(engine, windows_of([1], 1), start_step=1, loss_fn=poisoned, inv_scale=lambda s: 2.0 ** -10)
    assert stats[0].overflow and stats[0].skipped_params > 0 and stats[0].stepped_params == 0
    assert stats[0].skip_streak == 1 and stats[0].grad_norm != stats[0].grad_norm  # NaN
    for slot in engine.slots:
        assert torch.equal(slot.param, before[slot.name])
        for key, value in states[slot.name].items():
            assert torch.equal(engine._state_of(slot.index)[key], value)
    engine.check_skip_streak(stats[0])
    stats = train_engine(engine, windows_of([1], 2), start_step=2, loss_fn=poisoned)
    with pytest.raises(RuntimeError, match="in a row"):
        engine.check_skip_streak(stats[0])
    stats = train_engine(engine, windows_of([1], 3), start_step=3)
    assert not stats[0].overflow and stats[0].skip_streak == 0 and stats[0].stepped_params > 0
    assert any(not torch.equal(s.param, before[s.name]) for s in engine.slots)


def test_window_protocol_refusals():
    engine = make_engine(build_model(), "adamw", k=2)
    ctx = ft_host.StepContext(groups=(), step_index=0, inv_grad_scale=0.3)
    with pytest.raises(ValueError, match="power of two"):
        engine.begin_window(1, ctx)
    engine.begin_window(2, ft_host.StepContext(groups=(), step_index=0))
    with pytest.raises(RuntimeError):
        engine.begin_microstep(1)
    with pytest.raises(RuntimeError):
        engine.switch((0,))
    engine.close()
    assert not any("forward" in m.__dict__ for m in [*engine.model.layers, engine.model.text_embedder])
