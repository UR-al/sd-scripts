"""CPU tests for the two-phase LR of the MOSS full fine-tune (library/moss/ft_args.py; design
docs/moss_full_finetune_design.md 4.6, test plan 5 "test_moss_ft_lr_phases.py").

--moss_ft_lr_phases "E1:LR1,E2:LR2[,...]": phase i runs until b_i = round(cumsum(E)_i x steps_per_epoch) optimizer steps
(steps_per_epoch = ceil(batches / GA), kohya's formula), the LR is LR_i times a linear warmup over the first W steps
(lr(s) = min(1, (s + 1) / W) x LR_phase(s); step 0 uses LR1 / W, not 0, on purpose), and W must end inside phase 1.
The scheduler is a LambdaLR whose state_dict keeps only the step count: on resume the lambda is rebuilt from the config.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_lr_phases.py -q -p no:cacheprovider
"""
import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

from library.moss import ft_args


def _optimizer(groups: int = 1, lr: float = 1.0):
    params = [torch.nn.Parameter(torch.zeros(2)) for _ in range(groups)]
    return torch.optim.SGD([{"params": [p]} for p in params], lr=lr)


def _lrs(scheduler, optimizer, steps: int):
    """lr(s) for s = 0 .. steps-1: the group LR an optimizer step s runs with (read before the scheduler steps)."""
    out = []
    for _ in range(steps):
        out.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    return out


# ---- parsing -----------------------------------------------------------------------------------------------------
def test_parse_lr_phases():
    assert ft_args.parse_lr_phases("2:5e-6,6:1e-6") == [(2.0, 5e-6), (6.0, 1e-6)]
    assert ft_args.parse_lr_phases(" 2 : 5e-6 , 6:1e-6 ") == [(2.0, 5e-6), (6.0, 1e-6)]
    assert ft_args.parse_lr_phases("0.5:1e-5,1:1e-6,1.5:2e-7") == [(0.5, 1e-5), (1.0, 1e-6), (1.5, 2e-7)]
    assert ft_args.parse_lr_phases("8:5e-6") == [(8.0, 5e-6)]  # one phase: a constant LR with warmup
    # a TOML array of "E:LR" strings is the same list
    assert ft_args.parse_lr_phases(["2:5e-6", "6:1e-6"]) == [(2.0, 5e-6), (6.0, 1e-6)]


@pytest.mark.parametrize("text", [
    "", "  ", "2", "2:", ":5e-6", "2:5e-6:3", "a:1e-6", "2:x", "2:5e-6,", ",2:5e-6", "2:5e-6,,6:1e-6", "2;5e-6",
    "0:1e-6", "-1:1e-6", "2:0", "2:-5e-6", "nan:1e-6", "2:nan", "inf:1e-6", "2:inf", None, 2, 2.5, [2, 5e-6],
])
def test_parse_errors(text):
    with pytest.raises(ValueError, match="moss_ft_lr_phases"):
        ft_args.parse_lr_phases(text)


def test_argparse_type_keeps_the_text():
    assert ft_args.lr_phases_arg("2:5e-6,6:1e-6") == "2:5e-6,6:1e-6"
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        ft_args.lr_phases_arg("2:5e-6,x")


# ---- boundaries --------------------------------------------------------------------------------------------------
def test_bounds_and_steps_per_epoch():
    assert ft_args.steps_per_epoch(4000, 2) == 2000  # design 4.6 example: 16k images at batch 4 x GA 2
    assert ft_args.steps_per_epoch(4001, 2) == 2001  # the last window of an epoch may be short (kohya's ceil)
    assert ft_args.steps_per_epoch(7, 8) == 1
    phases = ft_args.parse_lr_phases("2:5e-6,6:1e-6")
    assert ft_args.lr_phase_bounds(phases, 2000) == [4000, 16000]  # 5e-6 for [0, 4000), 1e-6 for [4000, 16000)
    assert ft_args.lr_phase_bounds(phases, 100) == [200, 800]
    assert ft_args.lr_phase_end_epochs(phases) == [2.0, 8.0]
    # round(): float noise does not lose a step, a tie rounds up
    assert ft_args.lr_phase_bounds([(0.1, 1.0), (0.2, 1.0)], 10) == [1, 3]  # 0.1 + 0.2 = 0.30000000000000004
    assert ft_args.lr_phase_bounds([(0.25, 1.0), (0.5, 1.0)], 10) == [3, 8]  # 2.5 -> 3, 7.5 -> 8
    assert ft_args.lr_phase_bounds([(0.29, 1.0), (1.0, 1.0)], 100) == [29, 129]


def test_bounds_refuse_empty_phases():
    with pytest.raises(ValueError, match="phase 1"):
        ft_args.lr_phase_bounds([(0.001, 1e-5), (1.0, 1e-6)], 100)  # 0.1 steps -> 0
    with pytest.raises(ValueError, match="phase 2"):
        ft_args.lr_phase_bounds([(1.0, 1e-5), (0.001, 1e-6)], 100)
    with pytest.raises(ValueError, match="steps per epoch"):
        ft_args.lr_phase_bounds([(1.0, 1e-5)], 0)


# ---- values ------------------------------------------------------------------------------------------------------
def test_values():
    """Design 5: S = 100, "2:5e-6,6:1e-6", W = 10: lr(0) = 5e-7, lr(9) = 5e-6, lr(199) = 5e-6, lr(200) = 1e-6,
    lr(799) = 1e-6, total 800."""
    optimizer = _optimizer(groups=2, lr=123.0)  # the optimizer's own LR is replaced by LR1
    scheduler = ft_args.make_lr_phase_scheduler(optimizer, ft_args.parse_lr_phases("2:5e-6,6:1e-6"), 100, 10)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert scheduler.bounds == [200, 800] and scheduler.total_steps == 800 and scheduler.warmup_steps == 10
    lrs = _lrs(scheduler, optimizer, 800)
    approx = lambda value: pytest.approx(value, rel=1e-12)  # noqa: E731
    assert lrs[0] == approx(5e-7)  # deliberately lr0 / W at step 0, not 0
    assert lrs[4] == approx(2.5e-6)
    assert lrs[9] == approx(5e-6)
    assert lrs[10] == approx(5e-6)
    assert lrs[199] == approx(5e-6)
    assert lrs[200] == approx(1e-6)
    assert lrs[799] == approx(1e-6)
    assert all(lr == approx(5e-6) for lr in lrs[9:200]) and all(lr == approx(1e-6) for lr in lrs[200:])
    assert [lr == approx((s + 1) / 10 * 5e-6) for s, lr in enumerate(lrs[:10])] == [True] * 10
    # every group gets the same LR (design 4.4.1: both groups use the same LR)
    assert optimizer.param_groups[1]["lr"] == optimizer.param_groups[0]["lr"]
    # past the end the last phase holds (the loop stops at max_train_steps = 800 anyway)
    assert optimizer.param_groups[0]["lr"] == approx(1e-6)


def test_no_warmup():
    optimizer = _optimizer()
    scheduler = ft_args.make_lr_phase_scheduler(optimizer, "1:1e-5,1:1e-6", 10, 0)
    lrs = _lrs(scheduler, optimizer, 20)
    assert lrs[0] == pytest.approx(1e-5) and lrs[9] == pytest.approx(1e-5) and lrs[10] == pytest.approx(1e-6)


def test_fractional_epochs():
    """Design 5: "0.5:1e-5,1:1e-6" -> boundary 50 at S = 100; the run is 1.5 epochs = 150 steps."""
    phases = ft_args.parse_lr_phases("0.5:1e-5,1:1e-6")
    assert ft_args.lr_phase_bounds(phases, 100) == [50, 150]
    optimizer = _optimizer()
    scheduler = ft_args.make_lr_phase_scheduler(optimizer, phases, 100, 5)
    lrs = _lrs(scheduler, optimizer, 150)
    assert lrs[0] == pytest.approx(2e-6) and lrs[4] == pytest.approx(1e-5)
    assert lrs[49] == pytest.approx(1e-5) and lrs[50] == pytest.approx(1e-6) and lrs[149] == pytest.approx(1e-6)
    assert scheduler.total_steps == 150


def test_warmup_fraction_like_kohya():
    """A float lr_warmup_steps < 1 is a fraction of the total steps, converted like kohya's get_scheduler_fix."""
    from library.train_util import fraction_to_steps

    for fraction in (0.05, 0.1, 0.25, 0.29, 0.07):
        assert ft_args.resolve_warmup_steps(fraction, 800) == fraction_to_steps(fraction, 800)
    assert ft_args.resolve_warmup_steps(0.05, 800) == 40
    assert ft_args.resolve_warmup_steps(None, 800) == 0
    assert ft_args.resolve_warmup_steps(0, 800) == 0
    assert ft_args.resolve_warmup_steps(100, 800) == 100
    assert ft_args.resolve_warmup_steps(100.0, 800) == 100  # an integral TOML float is a step count
    scheduler = ft_args.make_lr_phase_scheduler(_optimizer(), "2:5e-6,6:1e-6", 100, 0.05)
    assert scheduler.warmup_steps == 40
    for bad in (-1, -0.5, 1.5, float("nan"), True, "10"):
        with pytest.raises(ValueError, match="lr_warmup_steps"):
            ft_args.resolve_warmup_steps(bad, 800)


def test_warmup_must_end_inside_phase_1():
    phases = ft_args.parse_lr_phases("2:5e-6,6:1e-6")
    ft_args.make_lr_phase_scheduler(_optimizer(), phases, 100, 200)  # ends exactly at the boundary: allowed
    with pytest.raises(ValueError, match="phase 1"):
        ft_args.make_lr_phase_scheduler(_optimizer(), phases, 100, 201)
    with pytest.raises(ValueError, match="phase 1"):
        ft_args.make_lr_phase_scheduler(_optimizer(), phases, 100, 0.3)  # 240 of 800 steps


# ---- resume ------------------------------------------------------------------------------------------------------
def test_resume_keeps_schedule():
    """A scheduler state_dict round trip continues the same values; the lambda (and the base LR) come from the config
    the new scheduler was built with, and the optimizer's group LR is the resumed step's LR right after the load."""
    phases = ft_args.parse_lr_phases("2:5e-6,6:1e-6")
    optimizer = _optimizer(groups=2)
    scheduler = ft_args.make_lr_phase_scheduler(optimizer, phases, 100, 10)
    straight = _lrs(scheduler, optimizer, 800)

    optimizer = _optimizer(groups=2)
    scheduler = ft_args.make_lr_phase_scheduler(optimizer, phases, 100, 10)
    first = _lrs(scheduler, optimizer, 250)
    state = scheduler.state_dict()
    saved = torch.load(_roundtrip(state), weights_only=False)  # scheduler.pt is written with torch.save

    optimizer2 = _optimizer(groups=2, lr=0.5)
    scheduler2 = ft_args.make_lr_phase_scheduler(optimizer2, phases, 100, 10)
    scheduler2.load_state_dict(saved)
    assert scheduler2.last_epoch == 250
    assert [g["lr"] for g in optimizer2.param_groups] == [pytest.approx(straight[250])] * 2
    rest = _lrs(scheduler2, optimizer2, 550)
    assert first + rest == straight  # bit for bit: same arithmetic on the same step counts
    assert scheduler2.get_last_lr() == [optimizer2.param_groups[0]["lr"]] * 2


def test_resume_rebuilds_the_lambda_from_the_config(monkeypatch):
    """A changed dataset (steps per epoch) or LR list: the step count is kept, the schedule is the new config's, and
    both versions are logged as a warning (design 4.6)."""
    warnings = []
    monkeypatch.setattr(ft_args.logger, "warning", lambda m, *a, **k: warnings.append(str(m)))
    optimizer = _optimizer()
    scheduler = ft_args.make_lr_phase_scheduler(optimizer, "2:5e-6,6:1e-6", 100, 10)
    _lrs(scheduler, optimizer, 150)
    state = scheduler.state_dict()

    optimizer2 = _optimizer()
    scheduler2 = ft_args.make_lr_phase_scheduler(optimizer2, "1:4e-6,6:1e-6", 120, 10)
    scheduler2.load_state_dict(state)
    assert scheduler2.last_epoch == 150 and scheduler2.bounds == [120, 840]
    assert optimizer2.param_groups[0]["lr"] == pytest.approx(1e-6)  # step 150 is in the new config's phase 2
    assert scheduler2.base_lrs == [4e-6]
    assert len(warnings) == 1 and "[200, 800]" in warnings[0] and "[120, 840]" in warnings[0]

    # the same config: nothing to warn about
    optimizer3 = _optimizer()
    scheduler3 = ft_args.make_lr_phase_scheduler(optimizer3, "2:5e-6,6:1e-6", 100, 10)
    scheduler3.load_state_dict(state)
    assert len(warnings) == 1 and optimizer3.param_groups[0]["lr"] == pytest.approx(5e-6)


def _roundtrip(state):
    import io

    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return buffer


def test_lambda_matches_the_design_formula():
    """The lambda is the design's (4.6) for every step, with warmup and three phases."""
    phases = ft_args.parse_lr_phases("1.5:3e-6,2:2e-6,0.5:1e-6")
    optimizer = _optimizer()
    scheduler = ft_args.make_lr_phase_scheduler(optimizer, phases, 40, 25)
    bounds, lrs, W = [60, 140, 160], [3e-6, 2e-6, 1e-6], 25

    def reference(s):
        phase = next((i for i, b in enumerate(bounds) if s < b), len(bounds) - 1)
        warm = min(1.0, (s + 1) / W) if W > 0 else 1.0
        return warm * lrs[phase] / lrs[0]

    assert scheduler.bounds == bounds
    got = _lrs(scheduler, optimizer, 160)
    assert got == [3e-6 * reference(s) for s in range(160)]
    assert math.isclose(got[59], 3e-6) and math.isclose(got[60], 2e-6) and math.isclose(got[159], 1e-6)
