"""CPU tests of the Carousel training states and DiT saves (docs/moss_ft_carousel_plan.md 6, items 23-25): a run
saved and resumed into fresh objects continues bit for bit (mid-period and at a switch boundary; keep and reset;
adamw and came, bf16 momentum), refusals (another optimizer kind, another trainable set, a host-mode state), the
warning on a changed rotation config, and the DiT saved while units are active on the device.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_carousel_state.py -q -p no:cacheprovider
"""
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
from safetensors import safe_open

import _moss_ft_common as common
from _moss_ft_carousel_common import build_model, make_engine, master_of, state_of, train_engine
from library.moss import ft_args, ft_carousel, ft_model, ft_state

WINDOWS = common.seeds([2] * 9)
PERIOD = 3


def scheduler_of(engine):
    return torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda s: 1.0 / (1.0 + 0.1 * s))


def save(engine, scheduler, folder, step, name="s"):
    engine.flush_to_host()
    return ft_state.write_state(str(folder), name, model=engine.model, slots=engine.slots, pool=engine.pool,
                                optimizer=engine.optimizer, lr_scheduler=scheduler, mode=ft_args.CAROUSEL,
                                progress=ft_state.TrainProgress(global_step=step, completed_epochs=0, epoch_step=step),
                                last_grad_norm=engine.last_grad_norm, skip_streak=engine.skip_streak, fsync=False,
                                extra_writer=engine.state_writer)


def load(engine, scheduler, path):
    return ft_state.load_state(path, model=engine.model, slots=engine.slots, pool=engine.pool,
                               optimizer=engine.optimizer, lr_scheduler=scheduler, mode=ft_args.CAROUSEL,
                               restore_rng=False, extra_reader=engine.state_reader)


def assert_same_runs(a, b):
    assert a.active == b.active and a.floor == b.floor
    for sa, sb in zip(a.slots, b.slots):
        assert torch.equal(master_of(a, sa.name), master_of(b, sb.name)), sa.name
        assert torch.equal(sa.param, sb.param) and sa.param.requires_grad == sb.param.requires_grad, sa.name
        xa, xb = state_of(a, sa.name), state_of(b, sb.name)
        assert xa.keys() == xb.keys(), sa.name
        for key in xa:
            if torch.is_tensor(xa[key]):
                assert xa[key].dtype == xb[key].dtype and torch.equal(xa[key], xb[key]), (sa.name, key)
            else:
                assert xa[key] == xb[key], (sa.name, key)
    assert [g["lr"] for g in a.optimizer.param_groups] == [g["lr"] for g in b.optimizer.param_groups]


@pytest.mark.parametrize("kind,state,dtype,cut", [("adamw", "keep", "fp32", 4), ("adamw", "keep", "fp32", 6),
                                                  ("adamw", "reset", "fp32", 4), ("came", "keep", "bf16", 6),
                                                  ("came", "reset", "fp32", 3)])
def test_resume_is_bit_identical(tmp_path, kind, state, dtype, cut):
    """cut 6 / 3 are switch boundaries (period 3), 4 is inside a period."""
    kw = dict(k=2, period=PERIOD, select="random", always="small", state=state, state_dtype=dtype, max_grad_norm=0.8)
    a = make_engine(build_model(), kind, **kw)
    sched_a = scheduler_of(a)
    path = None
    stats_a = train_engine(a, WINDOWS[:cut], scheduler=sched_a)
    saved_switches = a.switches
    assert saved_switches > 0
    path = save(a, sched_a, tmp_path, cut)
    stats_a += train_engine(a, WINDOWS[cut:], start_step=cut, scheduler=sched_a)

    b = make_engine(build_model(seed=9), kind, **kw)  # other initial values: everything must come from the state
    sched_b = scheduler_of(b)
    info = load(b, sched_b, path)
    assert b.switches == saved_switches
    assert info.progress.global_step == cut
    b.restore_counters(info.last_grad_norm, info.skip_streak)
    stats_b = train_engine(b, WINDOWS[cut:], start_step=cut, scheduler=sched_b)
    assert [s.grad_norm for s in stats_b] == [s.grad_norm for s in stats_a[cut:]]
    assert_same_runs(a, b)
    rotation = json.load(open(os.path.join(path, ft_carousel.ROTATION_FILE), encoding="utf-8"))
    assert rotation["format"] == ft_carousel.ROTATION_FORMAT and rotation["optimizer"] == kind
    assert rotation["k"] == 2 and rotation["period"] == PERIOD and len(rotation["units"]) == 5
    train = json.load(open(os.path.join(path, ft_state.TRAIN_STATE_FILE), encoding="utf-8"))
    assert train["mode"] == ft_args.CAROUSEL and train["extra"]["active"] == rotation["active"]
    names = os.listdir(path)
    assert ft_state.STEPS_FILE not in names and "exp_avg.safetensors" not in names


def test_state_file_dtypes(tmp_path):
    engine = make_engine(build_model(), "came", k=1, period=1, select="window", state_dtype="bf16", always="head")
    sched = scheduler_of(engine)
    train_engine(engine, WINDOWS[:3], scheduler=sched)
    path = save(engine, sched, tmp_path, 3)
    with safe_open(os.path.join(path, ft_carousel.ROTATION_OPTIMIZER_FILE), "pt") as f:
        keys = list(f.keys())
        active = {engine.slots[i].name for i in engine.active_slots}
        for key in keys:
            name, _, state_key = key.partition("::")
            tensor = f.get_tensor(key)
            if state_key == "step":
                assert tensor.dtype == torch.int64
            elif state_key == "exp_avg":
                # inactive units: stored as kept (bf16); active ones / the always group: fp32 from the device
                assert tensor.dtype == (torch.float32 if name in active else torch.bfloat16), key
            else:
                assert tensor.dtype == torch.float32
        assert {k.partition("::")[0] for k in keys} == {s.name for s in engine.slots if state_of(engine, s.name)}


def test_refusals_and_warnings(tmp_path, monkeypatch):
    a = make_engine(build_model(), "adamw", k=2, period=PERIOD)
    sched = scheduler_of(a)
    train_engine(a, WINDOWS[:2], scheduler=sched)
    path = save(a, sched, tmp_path, 2)

    came = make_engine(build_model(), "came", k=2, period=PERIOD)
    with pytest.raises(ft_state.StateError, match="adamw optimizer states"):
        load(came, scheduler_of(came), path)

    frozen = make_engine(build_model(), "adamw", k=2, period=PERIOD, frozen=r"layers\.0\..*")
    with pytest.raises(ft_state.StateError, match="another trainable set"):
        load(frozen, scheduler_of(frozen), path)

    warnings = []
    monkeypatch.setattr(ft_carousel.logger, "warning", lambda m, *x, **k: warnings.append(str(m)))
    other = make_engine(build_model(), "adamw", k=3, period=5, select="window", seed=4)
    load(other, scheduler_of(other), path)
    assert warnings and all(word in warnings[-1] for word in ("k 2 -> 3", "period 3 -> 5", "select 'random' -> 'window'",
                                                               "seed 3 -> 4"))
    train_engine(other, WINDOWS[2:3], start_step=2)  # the set follows the new settings from the saved step
    assert len(other.active) == 3

    # a host-mode state (adamw_host) cannot resume a carousel run
    model = build_model()
    named = list(model.named_parameters())
    from library.moss.ft_optim import HostAdamW

    opt = HostAdamW(common.param_groups(named), lr=1e-3)
    slots = __import__("library.moss.ft_host", fromlist=["x"]).make_slots(named, opt)
    pool = __import__("library.moss.ft_host", fromlist=["x"]).HostPool(slots, HostAdamW.POOL_ROLES)
    pool.copy_master_from_params()
    host_path = ft_state.write_state(str(tmp_path), "host", model=model, slots=slots, pool=pool, optimizer=opt,
                                     lr_scheduler=None, mode=ft_args.ADAMW_HOST,
                                     progress=ft_state.TrainProgress(global_step=1, completed_epochs=0), fsync=False)
    fresh = make_engine(build_model(), "adamw", k=2, period=PERIOD)
    with pytest.raises(ft_state.StateError, match="optimizer mode adamw_host"):
        load(fresh, None, host_path)


def test_dit_save_while_units_are_active(tmp_path):
    engine = make_engine(build_model(), "adamw", k=2, period=PERIOD)
    train_engine(engine, WINDOWS[:2])
    assert engine.active_slots
    masters = {s.name: master_of(engine, s.name).clone() for s in engine.slots}
    stale = [s.name for s in engine.slots if s.index in engine._loc
             and not torch.equal(engine.pool.view("master", s), masters[s.name])]
    assert stale  # the pool is behind the device for the active units until the flush
    engine.flush_to_host()
    path = tmp_path / "dit.safetensors"
    ft_model.save_moss_dit(str(path), engine.model, engine.pool, "fp16", {"x": "y"})
    with safe_open(str(path), "pt") as f:
        assert sorted(f.keys()) == sorted(engine.model.state_dict())
        for name, master in masters.items():
            shape = dict(engine.model.named_parameters())[name].shape
            assert torch.equal(f.get_tensor(name), master.view(shape).to(torch.float16)), name
    for s in engine.slots:  # the flush changed nothing on the device
        assert torch.equal(master_of(engine, s.name), masters[s.name])
