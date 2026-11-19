"""CPU tests of library/moss/ft_scaler.py (docs/moss_ft_te_live_fp16_plan.md 2.2): the dynamic loss scale's update
rules, its bounds, growth_interval 0 (never grows, still backs off), a non-finite loss keeps the scale, the divergence
check (finite-loss overflows above the minimum do not count; overflows
with a non-finite loss or at the minimum do) and the state round trip / restore cases of plan 2.4.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_loss_scaler.py -q -p no:cacheprovider
"""
import json
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest

from library.moss import ft_scaler
from library.moss.ft_scaler import BACKOFF, GROW, KEEP, DynamicLossScaler

CLEAN = types.SimpleNamespace(skipped_params=0)
OVERFLOW = types.SimpleNamespace(skipped_params=3)


def test_power_of_two():
    for value in (1, 1.0, 2.0, 0.5, 2.0 ** -24, 2.0 ** 15, 32768, 2.0 ** 24):
        assert ft_scaler.is_power_of_two(value), value
    for value in (0, -2.0, 3.0, 1000.0, 65504.0, float("inf"), float("nan"), "x", None):
        assert not ft_scaler.is_power_of_two(value), value


def test_defaults_and_inverse():
    s = DynamicLossScaler()
    assert (s.scale, s.growth_interval, s.min_scale, s.max_scale) == (32768.0, 2000, 1.0, 2.0 ** 24)
    assert s.inv_scale == 2.0 ** -15 and s.scale * s.inv_scale == 1.0


def test_backoff_and_growth():
    s = DynamicLossScaler(init_scale=1024.0, growth_interval=3)
    assert s.update(OVERFLOW) == BACKOFF and s.scale == 512.0 and s.good_windows == 0
    assert [s.update(CLEAN) for _ in range(3)] == [KEEP, KEEP, GROW] and s.scale == 1024.0
    assert s.good_windows == 0
    assert [s.update(CLEAN) for _ in range(2)] == [KEEP, KEEP]
    assert s.update(OVERFLOW) == BACKOFF and s.scale == 512.0 and s.good_windows == 0  # the count restarts
    assert [s.update(CLEAN) for _ in range(3)] == [KEEP, KEEP, GROW] and s.scale == 1024.0
    assert s.overflow_windows == 2


def test_bounds():
    s = DynamicLossScaler(init_scale=4.0, growth_interval=1, min_scale=2.0, max_scale=16.0)
    assert s.update(OVERFLOW) == BACKOFF and s.scale == 2.0
    assert s.update(OVERFLOW) == KEEP and s.scale == 2.0  # not below min_scale
    for _ in range(5):
        s.update(CLEAN)
    assert s.scale == 16.0 and s.update(CLEAN) == KEEP and s.scale == 16.0  # not above max_scale


def test_growth_interval_zero_never_grows_but_still_backs_off():
    s = DynamicLossScaler(init_scale=256.0, growth_interval=0)
    assert all(s.update(CLEAN) == KEEP for _ in range(10)) and s.scale == 256.0 and s.good_windows == 0
    assert s.update(OVERFLOW) == BACKOFF and s.scale == 128.0  # an overflow still backs off (not a fixed scale)


def test_non_finite_loss_keeps_the_scale():
    """Review fix: the loss is formed (float64) before it is multiplied by S, so a non-finite loss is a forward
    overflow of the fp16 activations that S neither causes nor cures. The scale and the clean-window count stay; the
    window counts towards the divergence streak."""
    s = DynamicLossScaler(init_scale=1024.0, growth_interval=3)
    assert [s.update(CLEAN) for _ in range(2)] == [KEEP, KEEP] and s.good_windows == 2
    assert s.update(OVERFLOW, loss_finite=False) == KEEP and s.scale == 1024.0
    assert s.good_windows == 2 and s.overflow_windows == 1 and s.overflow_streak == 1
    assert s.update(CLEAN, loss_finite=False) == KEEP and s.overflow_streak == 2 and s.overflow_windows == 1
    assert s.update(CLEAN) == GROW and s.scale == 2048.0 and s.overflow_streak == 0  # the third clean window


def test_divergence_check_counts_only_what_the_scale_cannot_explain():
    s = DynamicLossScaler(init_scale=2.0 ** 15, growth_interval=2000, min_scale=1.0)
    for _ in range(14):  # the normal search: 2^15 -> 2^1, finite losses above the minimum
        s.update(OVERFLOW, loss_finite=True)
        s.check(2)
    assert s.scale == 2.0 and s.overflow_streak == 0
    s.update(OVERFLOW, loss_finite=False)  # a non-finite loss counts even above the minimum (and keeps the scale)
    assert s.overflow_streak == 1 and s.scale == 2.0
    s.check(2)
    s.update(OVERFLOW, loss_finite=False)
    assert s.overflow_streak == 2
    with pytest.raises(RuntimeError, match="2 optimizer steps in a row"):
        s.check(2)
    s.update(OVERFLOW, loss_finite=True)  # a finite-loss overflow above the minimum: the search, the streak restarts
    assert s.scale == 1.0 and s.overflow_streak == 0
    s.update(OVERFLOW, loss_finite=True)  # at min_scale: counts
    assert s.overflow_streak == 1 and s.scale == 1.0
    s.check(0)  # 0 = never stop
    s.update(CLEAN)
    assert s.overflow_streak == 0
    s.check(2)


def test_invalid_settings_refused():
    for kw in (dict(init_scale=1000.0), dict(min_scale=3.0), dict(init_scale=2.0, min_scale=4.0),
               dict(init_scale=2.0 ** 25), dict(growth_interval=-1), dict(growth_interval=1.5)):
        with pytest.raises(ValueError, match="MOSS full FT: loss scaler"):
            DynamicLossScaler(**kw)


def test_state_round_trip_through_json():
    a = DynamicLossScaler(init_scale=2.0 ** 12, growth_interval=5)
    for stats in (OVERFLOW, CLEAN, CLEAN, OVERFLOW, CLEAN, CLEAN, CLEAN):
        a.update(stats)
    saved = json.loads(json.dumps(a.state_dict()))  # what train_state.json holds
    b = DynamicLossScaler(init_scale=2.0 ** 12, growth_interval=5)
    b.load_state_dict(saved)
    assert b.state_dict() == a.state_dict()
    for stats in (CLEAN, CLEAN, OVERFLOW, CLEAN):
        assert a.update(stats) == b.update(stats)
    assert a.state_dict() == b.state_dict()


def test_load_with_other_settings_keeps_the_saved_scale(caplog):
    a = DynamicLossScaler(init_scale=2.0 ** 10, growth_interval=100)
    a.update(OVERFLOW)
    for _ in range(40):
        a.update(CLEAN)
    b = DynamicLossScaler(init_scale=2.0 ** 10, growth_interval=10, min_scale=2.0)
    with caplog.at_level("WARNING"):
        b.load_state_dict(a.state_dict())
    assert b.scale == 512.0 and b.growth_interval == 10
    assert b.good_windows == 9  # clamped below this run's interval: grows at its next clean window
    assert "settings changed" in caplog.text
    with pytest.raises(ValueError, match="power of two"):
        b.load_state_dict({"scale": 3.0})


def test_a_changed_init_scale_resets_the_saved_scale(caplog):
    """Review fix: a scale that collapsed in the saved run could not be raised on resume. A different
    --moss_ft_loss_scale is the explicit reset: the scale restarts there with fresh counters (the overflow total stays)."""
    a = DynamicLossScaler(init_scale=2.0 ** 4, growth_interval=100)
    for _ in range(4):
        a.update(OVERFLOW)
    a.update(OVERFLOW, loss_finite=False)
    assert a.scale == 1.0 and a.overflow_streak == 1
    b = DynamicLossScaler(init_scale=2.0 ** 12, growth_interval=100)
    with caplog.at_level("WARNING"):
        b.load_state_dict(json.loads(json.dumps(a.state_dict())))
    assert (b.scale, b.good_windows, b.overflow_streak, b.overflow_windows) == (4096.0, 0, 0, 5)
    assert "restarts at 4096" in caplog.text
    c = DynamicLossScaler(init_scale=2.0 ** 4, growth_interval=100)  # the saved run's value: exact
    c.load_state_dict(a.state_dict())
    assert c.state_dict() == a.state_dict()


def test_make_and_restore(caplog):
    assert ft_scaler.make(types.SimpleNamespace(mixed_precision="bf16")) is None
    assert ft_scaler.make(types.SimpleNamespace(mixed_precision="no")) is None
    s = ft_scaler.make(types.SimpleNamespace(mixed_precision="fp16", moss_ft_loss_scale=2048,
                                             moss_ft_loss_scale_growth_interval=7, moss_ft_loss_scale_min=4.0))
    assert (s.scale, s.growth_interval, s.min_scale) == (2048.0, 7, 4.0)
    with caplog.at_level("INFO"):
        ft_scaler.restore(s, None)  # bf16 state -> fp16 run: starts at init_scale
        assert s.scale == 2048.0 and "no loss scale" in caplog.text
        caplog.clear()
        saved = DynamicLossScaler(init_scale=2048, growth_interval=7, min_scale=4.0)
        saved.update(OVERFLOW)
        ft_scaler.restore(s, saved.state_dict())  # fp16 -> fp16: exact
        assert s.state_dict() == saved.state_dict() and "restored" in caplog.text
        caplog.clear()
        ft_scaler.restore(None, saved.state_dict())  # fp16 state -> bf16 run: ignored
        assert "ignored" in caplog.text
