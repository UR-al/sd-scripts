"""Warmup / decay ratios become steps without losing a step to float noise (review 11-05 follow-up).

The GUI backend sends lr_warmup_steps (and may send lr_decay_steps) as a fraction of the run. get_scheduler_fix used
int(fraction * num_training_steps), and float noise made that lose a step: 0.29 * 100 = 28.999999999999996 -> 28.
fraction_to_steps rounds the noise away first and keeps upstream's floor for genuinely fractional products
(0.115 * 100 = 11.5 -> 11). Integers stay absolute steps. CPU only.
"""

import argparse
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library import train_util  # noqa: E402
from library.train_util import get_scheduler_fix  # noqa: E402

BASE_LR = 1e-4


def test_library_is_this_checkout():
    # sys.path.insert above must win over any other sd-scripts checkout on the path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.dirname(os.path.abspath(train_util.__file__)) == os.path.join(repo_root, "library")


@pytest.mark.parametrize(
    "fraction,total,expected",
    [
        (0.29, 100, 29),  # 28.999999999999996 with a plain int()
        (0.57, 100, 57),  # 56.99999999999999 with a plain int()
        (0.07, 100, 7),  # 7.000000000000001, noise upward, stays 7
        (0.105, 100, 10),  # a real 10.5 keeps upstream's floor
        (0.115, 100, 11),  # a real 11.5 keeps the floor too; round() (half to even) would give 12
        (0.5, 3, 1),  # an exact 1.5 -> 1; round() would give 2
        (0.0, 100, 0),
        (1.0, 57, 57),
    ],
)
def test_fraction_to_steps(fraction, total, expected):
    assert train_util.fraction_to_steps(fraction, total) == expected


def build_args(lr_scheduler, lr_warmup_steps, lr_decay_steps=None, max_train_steps=100):
    return argparse.Namespace(
        lr_scheduler_type="",
        lr_warmup_steps=lr_warmup_steps,
        zero_lr_warmup=False,
        lr_scheduler_args=None,
        lr_scheduler_num_cycles=1,
        max_train_steps=max_train_steps,
        lr_scheduler=lr_scheduler,
        lr_decay_steps=lr_decay_steps,
        lr_scheduler_power=1.0,
        lr_scheduler_timescale=None,
        lr_scheduler_min_lr_ratio=None,
        optimizer_type="AdamW",
        validation_split=0.0,
    )


def lrs_at(args, steps):
    """Learning rate the scheduler from get_scheduler_fix produces at each of the given step indices."""
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
    scheduler = get_scheduler_fix(args, optimizer, num_processes=1)
    out = {}
    for step in range(max(steps) + 1):
        if step in steps:
            out[step] = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
    return out


@pytest.mark.parametrize("warmup", [0.29, 29])
def test_constant_with_warmup_warms_up_for_29_steps(warmup):
    lrs = lrs_at(build_args("constant_with_warmup", warmup), [1, 28, 29, 50])
    assert lrs[1] == pytest.approx(BASE_LR / 29)
    assert lrs[28] == pytest.approx(BASE_LR * 28 / 29)
    assert lrs[29] == pytest.approx(BASE_LR)
    assert lrs[50] == pytest.approx(BASE_LR)


@pytest.mark.parametrize("warmup", [0.29, 29])
def test_linear_warms_up_for_29_steps(warmup):
    lrs = lrs_at(build_args("linear", warmup), [28, 29, 30])
    assert lrs[28] == pytest.approx(BASE_LR * 28 / 29)
    assert lrs[29] == pytest.approx(BASE_LR)  # peak, then linear decay over the remaining 71 steps
    assert lrs[30] == pytest.approx(BASE_LR * 70 / 71)


@pytest.mark.parametrize("decay", [0.29, 29])
def test_warmup_stable_decay_decays_for_29_steps(decay):
    # no warmup, 29 decay steps -> 71 stable steps [0, 71), cosine decay over [71, 100)
    lrs = lrs_at(build_args("warmup_stable_decay", 0, lr_decay_steps=decay), [70, 71, 72, 99])
    assert lrs[70] == pytest.approx(BASE_LR)
    assert lrs[71] == pytest.approx(BASE_LR)
    assert lrs[72] == pytest.approx(BASE_LR * 0.5 * (1.0 + math.cos(math.pi * 1 / 29)))
    assert lrs[99] == pytest.approx(BASE_LR * 0.5 * (1.0 + math.cos(math.pi * 28 / 29)))


def test_genuine_half_step_keeps_floor():
    # 0.115 of 100 steps = 11.5 -> 11 warmup steps, as int() gave before (round() would give 12)
    lrs = lrs_at(build_args("constant_with_warmup", 0.115), [10, 11])
    assert lrs[10] == pytest.approx(BASE_LR * 10 / 11)
    assert lrs[11] == pytest.approx(BASE_LR)
