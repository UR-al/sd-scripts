"""CPU tests of the Carousel switches (docs/moss_ft_carousel_plan.md 6, items 15-18): exact round trips of masters and
optimizer states for every kind (bitsandbytes' 8-bit layout through a CPU stand-in with its exact state structure),
invariant I1 and its guard, reset / bf16 state policies, and that an outgoing unit leaves nothing on the device.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_carousel_switch.py -q -p no:cacheprovider
"""
import os
import weakref

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

import _moss_ft_common as common
from _moss_ft_carousel_common import (assert_matches_reference, build_model, clone, make_engine, master_of, state_of,
                                      train_engine, train_reference)


def snapshot(engine):
    out = {}
    for slot in engine.slots:
        state = state_of(engine, slot.name)
        out[slot.name] = (master_of(engine, slot.name).clone(), slot.param.detach().clone(),
                          {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state.items()
                           if k not in ("qmap1", "qmap2")})
    return out


def assert_same(a, b):
    assert a.keys() == b.keys()
    for name in a:
        ma, pa, sa = a[name]
        mb, pb, sb = b[name]
        assert torch.equal(ma, mb) and torch.equal(pa, pb), name
        assert sa.keys() == sb.keys(), name
        for key in sa:
            assert (torch.equal(sa[key], sb[key]) and sa[key].dtype == sb[key].dtype) if torch.is_tensor(sa[key]) \
                else sa[key] == sb[key], (name, key)


@pytest.mark.parametrize("kind", ["adamw", "came", "adamw8bit"])
def test_round_trips_are_exact(kind):
    """Train, then switch every unit out and back in several times without training: every master, compute copy and
    state bit for bit; the 8-bit states keep their uint8 dtype, int step and the optimizer's shared qmap objects."""
    engine = make_engine(build_model(), kind, k=2, period=1, select="round_robin")
    train_engine(engine, common.seeds([2] * 6))
    before = snapshot(engine)
    active = engine.active
    for rounds in range(3):
        for units in ((0, 1), (2, 3), (4,), active):
            engine.switch(units)
    assert_same(before, snapshot(engine))
    if kind == "adamw8bit":
        qmaps = engine.optimizer.name2qmap
        seen_8bit = False
        for i in engine.active_slots:
            state = engine.optimizer.state[engine.handles[i]]
            if state["state1"].dtype == torch.uint8:
                seen_8bit = True
                assert state["qmap1"] is qmaps["dynamic"] and state["qmap2"] is qmaps["udynamic"]
                assert isinstance(state["step"], int) and state["absmax1"].dtype == torch.float32
        assert seen_8bit
        for i, state in engine.host_state.items():
            assert "qmap1" not in state and "qmap2" not in state


@pytest.mark.parametrize("kind", ["adamw8bit"])
def test_eight_bit_layout_matches_its_masked_reference(kind):
    model = build_model()
    ref = clone(model)
    engine = make_engine(model, kind, k=2, period=2, select="random")
    windows = common.seeds([2] * 8)
    train_engine(engine, windows)
    opt, _, params = train_reference(ref, kind, windows, k=2, period=2, select="random", seed=3, always="small")
    assert_matches_reference(engine, params, opt, kind)


def test_inactive_host_masters_are_never_written():
    engine = make_engine(build_model(), "adamw", k=2, period=3, select="window")
    pool = engine.pool
    train_engine(engine, common.seeds([1]))
    inactive = [s for s in engine.slots if s.index not in engine._loc]
    frozen = {s.name: pool.view("master", s).clone() for s in inactive}
    train_engine(engine, common.seeds([1, 1], 1), start_step=1)  # steps 1, 2: no switch before step 3
    for s in inactive:
        assert torch.equal(pool.view("master", s), frozen[s.name]), s.name


def test_invariant_holds_after_every_switch_and_the_guard_fires():
    engine = make_engine(build_model(dtype=torch.bfloat16), "adamw", k=2, period=1, select="random")

    def check(w, stats):
        for slot in engine.slots:
            assert torch.equal(slot.param.detach().reshape(-1), master_of(engine, slot.name).to(slot.param.dtype))

    train_engine(engine, common.seeds([1] * 6), after_window=check)
    outgoing = next(u for u in range(len(engine.units)) if u not in engine.active)
    slot = engine.slots[engine.group_slots[outgoing][0]]
    with torch.no_grad():
        slot.param.view(-1)[0] += 1.0  # a compute copy drifted from its master while inactive
    with pytest.raises(RuntimeError, match="invariant I1"):
        engine.switch(tuple(sorted({outgoing, *engine.active[1:]})))


def test_reset_drops_states_and_restarts_the_step():
    engine = make_engine(build_model(), "adamw", k=1, period=1, select="window", state="reset", always="none")
    train_engine(engine, common.seeds([1] * 2))  # unit 0 at step 0, unit 1 at step 1, unit 2 active now
    assert engine.active == (2,)
    assert not engine.host_state
    for slot in engine.slots:
        assert not state_of(engine, slot.name), slot.name  # the trained units' states were dropped
    engine.switch((0,))
    engine.period = 1000  # no switch after the next window
    train_engine(engine, common.seeds([1]), start_step=2)
    for i in engine.active_slots:
        assert float(engine.optimizer.state[engine.handles[i]]["step"]) == 1.0  # fresh: one step since activation


def test_bf16_momentum_rounds_once_per_deactivation():
    engine = make_engine(build_model(), "came", k=1, period=1000, select="window", state_dtype="bf16", always="none")
    train_engine(engine, common.seeds([1]))
    unit = engine.active[0]
    live = {i: {k: (v.clone() if torch.is_tensor(v) else v) for k, v in engine.optimizer.state[engine.handles[i]].items()}
            for i in engine.group_slots[unit]}
    engine.switch((unit + 1,))
    for i, state in live.items():
        host = engine.host_state[i]
        assert host["exp_avg"].dtype == torch.bfloat16
        assert torch.equal(host["exp_avg"], state["exp_avg"].bfloat16())
        for key, value in state.items():
            if key not in ("exp_avg", "step"):
                assert host[key].dtype == torch.float32 and torch.equal(host[key], value), key
        assert host["step"] == state["step"] == 1
    engine.switch((unit,))
    for i, state in live.items():
        back = engine.optimizer.state[engine.handles[i]]
        assert back["exp_avg"].dtype == torch.float32
        assert torch.equal(back["exp_avg"], state["exp_avg"].bfloat16().float())


def test_an_outgoing_unit_leaves_nothing_on_the_device():
    engine = make_engine(build_model(), "adamw", k=2, period=1000, select="window")
    train_engine(engine, common.seeds([1]))
    unit = engine.active[0]
    flat = weakref.ref(engine._dev[unit]["master"])
    states = [weakref.ref(engine.optimizer.state[engine.handles[i]]["exp_avg"]) for i in engine.group_slots[unit]]
    engine.switch(tuple(u for u in range(len(engine.units)) if u not in engine.active)[:2])
    assert unit not in engine._dev and flat() is None and all(ref() is None for ref in states)
    for i in engine.group_slots[unit]:
        handle = engine.handles[i]
        assert handle.numel() == 0 and handle.grad is None and handle not in engine.optimizer.state
        assert not engine.slots[i].param.requires_grad and i not in engine._loc
    assert set(engine._dev) == {*engine.active, "always"}
