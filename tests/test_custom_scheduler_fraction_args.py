"""Custom restart schedulers get warmup_steps / first_cycle_max_steps as fractions of the run (review 11-05, item 3).

The GUI backend only estimates the step count (global batch size; the trainer batches each folder with its own batch
size, resolution pools and validation split). For default schedulers it already sends lr_warmup_steps as a ratio,
which get_scheduler_fix turns into steps. The custom restart schedulers (LoraEasyCustomOptimizer.*WarmRestarts) take
warmup_steps and first_cycle_max_steps as constructor arguments in steps, so the backend now sends those as floats in
[0, 1] and get_scheduler_fix converts them with the real step count:
- first_cycle_max_steps = max(1, ceil(round(fraction * num_training_steps, 6)))
- warmup_steps = train_util.fraction_to_steps(fraction, num_training_steps) // lr_scheduler_num_cycles, i.e. the floor
  after rounding away float noise, per cycle as the backend used to do (a default scheduler's lr_warmup_steps ratio
  uses the same helper)
Integers stay absolute steps, so hand-written TOMLs behave as before. CPU only.
"""

import argparse
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("LoraEasyCustomOptimizer")

from library.train_util import get_scheduler_fix  # noqa: E402

COSINE = "LoraEasyCustomOptimizer.CosineAnnealingWarmRestarts.CosineAnnealingWarmRestarts"
REX = "LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts"


def build_args(lr_scheduler_type, lr_scheduler_args, max_train_steps, num_cycles=1, validation_split=0.0):
    return argparse.Namespace(
        lr_scheduler_type=lr_scheduler_type,
        lr_warmup_steps=0,
        zero_lr_warmup=False,
        lr_scheduler_args=lr_scheduler_args,
        lr_scheduler_num_cycles=num_cycles,
        max_train_steps=max_train_steps,
        lr_scheduler="constant",
        lr_decay_steps=None,
        lr_scheduler_power=1.0,
        lr_scheduler_timescale=None,
        lr_scheduler_min_lr_ratio=None,
        optimizer_type="AdamW",
        validation_split=validation_split,
    )


def group_of(scheduler_type, scheduler_args, max_train_steps, num_cycles=1, num_processes=1, validation_split=0.0):
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    args = build_args(scheduler_type, scheduler_args, max_train_steps, num_cycles, validation_split)
    get_scheduler_fix(args, optimizer, num_processes=num_processes)
    group = optimizer.param_groups[0]
    return group["warmup_steps"], group["current_cycle_max_steps"]


@pytest.mark.parametrize("scheduler_type", [COSINE, REX])
def test_fractions_use_the_trainers_step_count(scheduler_type):
    # the reviewer's case: the backend estimated 20 steps, the trainer runs 80; 10 % warmup, one cycle
    warmup, first_cycle = group_of(scheduler_type, ["warmup_steps=0.1", "first_cycle_max_steps=1.0", "min_lr=1e-6",
                                                    "gamma=0.9"], max_train_steps=80)
    assert (warmup, first_cycle) == (8, 80)


def test_fractions_with_several_cycles_are_per_cycle():
    # 3 cycles over 300 steps: each cycle 100 steps, 10 % of the run as warmup = 30 steps split over 3 cycles = 10
    warmup, first_cycle = group_of(COSINE, ["warmup_steps=0.1", f"first_cycle_max_steps={1 / 3!r}", "min_lr=1e-6",
                                            "gamma=0.9"], max_train_steps=300, num_cycles=3)
    assert (warmup, first_cycle) == (10, 100)


@pytest.mark.parametrize("cycles,total", [(10, 30), (10, 1000), (7, 700), (3, 3), (4, 1001)])
def test_cycle_fraction_does_not_overshoot_from_float_rounding(cycles, total):
    # 1 / cycles of the run is one cycle's length (ceil of the exact quotient); these products come out exact
    # (30 * 0.1 == 3.0), the float-noise cases are in the next two tests
    _, first_cycle = group_of(COSINE, [f"first_cycle_max_steps={1 / cycles!r}", "min_lr=1e-6", "gamma=0.9"],
                              max_train_steps=total, num_cycles=cycles)
    assert first_cycle == -(-total // cycles)


@pytest.mark.parametrize("fraction,total,expected", [(0.07, 100, 7), (0.035, 200, 7), (0.14, 100, 14)])
def test_cycle_fraction_float_noise_does_not_add_a_step(fraction, total, expected):
    # 0.07 * 100 = 7.000000000000001 and 0.14 * 100 = 14.000000000000002; a plain ceil would give 8 and 15
    _, first_cycle = group_of(COSINE, [f"first_cycle_max_steps={fraction!r}", "min_lr=1e-6", "gamma=0.9"],
                              max_train_steps=total)
    assert first_cycle == expected


@pytest.mark.parametrize("fraction,expected", [(0.29, 29), (0.57, 57)])
def test_warmup_fraction_float_noise_does_not_lose_a_step(fraction, expected):
    # 0.29 * 100 = 28.999999999999996 and 0.57 * 100 = 56.99999999999999; a plain int() would give 28 and 56
    warmup, first_cycle = group_of(COSINE, [f"warmup_steps={fraction!r}", "first_cycle_max_steps=1.0", "min_lr=1e-6",
                                            "gamma=0.9"], max_train_steps=100)
    assert (warmup, first_cycle) == (expected, 100)


def test_num_processes_scale_like_the_default_warmup():
    # num_training_steps = max_train_steps * num_processes, the same base as lr_warmup_steps ratios
    warmup, first_cycle = group_of(COSINE, ["warmup_steps=0.1", "first_cycle_max_steps=1.0", "min_lr=1e-6",
                                            "gamma=0.9"], max_train_steps=50, num_processes=2)
    assert (warmup, first_cycle) == (10, 100)


def test_integers_stay_absolute_steps():
    # a hand-written TOML: the same numbers as before this change
    warmup, first_cycle = group_of(COSINE, ["warmup_steps=5", "first_cycle_max_steps=40", "min_lr=1e-6",
                                            "gamma=0.9"], max_train_steps=80)
    assert (warmup, first_cycle) == (5, 40)


def test_validation_split_fix_still_applies_to_integers():
    # the existing 'temp fix': with a validation split the trainer recomputes the first cycle from its own count
    _, first_cycle = group_of(COSINE, ["first_cycle_max_steps=999", "min_lr=1e-6", "gamma=0.9"], max_train_steps=60,
                              num_cycles=2, validation_split=0.1)
    assert first_cycle == 30
