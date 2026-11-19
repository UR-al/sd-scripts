"""CPU tests of the Carousel mode's pure parts (library/moss/ft_carousel_units.py; docs/moss_ft_carousel_plan.md 6,
items 1-3): the units of the real MOSS DiT (meta model), the always groups, the selection, the backward floor.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_carousel_units.py -q -p no:cacheprovider
"""
import math
import os
import random

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import torch

from library.moss import ft_args, ft_budget
from library.moss import ft_carousel_units as cu


@pytest.fixture(scope="module")
def shapes():
    return ft_budget.moss_param_shapes()


def test_units_of_the_real_model(shapes):
    units = cu.discover_units(shapes.items())
    assert len(units) == 44
    assert [u.name for u in units[:3]] == ["text_embedder.blocks.0", "text_embedder.blocks.1", "layers.0"]
    assert [u.name for u in units[2:]] == [f"layers.{i}" for i in range(42)]
    assert [u.joint for u in units] == [None, None] + list(range(42))
    assert all(u.numel == 113_284_352 for u in units[:2]) and all(u.numel == 113_252_608 for u in units[2:])
    joint = units[2]
    assert len(joint.names) == 10 and "layers.0.norm1.weight" in joint.names and "layers.0.k_norm.weight" in joint.names
    assert len(units[0].names) == 14
    # every tensor is a unit, stem or head tensor; units + small = everything
    names = list(shapes)
    small = cu.always_names(names, "small")
    head = cu.always_names(names, "head")
    count = {n: math.prod(s) for n, s in shapes.items()}
    assert sum(count[n] for n in small) == 99_130_368
    assert sum(count[n] for n in head) == 19_276_800
    assert head == ["final_norm_video.weight", "final_linear_video.weight", "video_head_modulation.weight",
                    "video_head_modulation.bias"]
    assert cu.always_names(names, "none") == []
    assert sum(u.numel for u in units) + sum(count[n] for n in small) == ft_budget.TOTAL_PARAMS
    with pytest.raises(ValueError, match="neither a block"):
        cu.tensor_role("lora_down.weight")


def test_freeze_drops_tensors_and_whole_units(shapes):
    frozen = set(ft_args.frozen_parameter_names(shapes, ["layers\\.(0|1)\\..*", "layers\\.2\\.proj\\.q\\.weight"]))
    units = cu.discover_units([(n, s) for n, s in shapes.items() if n not in frozen])
    assert len(units) == 42 and units[2].name == "layers.2"
    assert "layers.2.proj.q.weight" not in units[2].names and len(units[2].names) == 9


@pytest.mark.parametrize("k", [1, 3, 7, 11])
def test_random_selection_covers_every_unit_once_per_cycle(k):
    n = 11
    sel = cu.Selector(n, k, "random", seed=5)
    sets = [sel(i) for i in range(40)]
    assert all(len(s) == k and len(set(s)) == k and list(s) == sorted(s) for s in sets)
    stream = []
    for s in sets:
        stream.extend(s)
    # every window of n consecutive draws of a cycle covers all units: the first ceil(n / k) sets cover every unit
    covered = set()
    for s in sets[:-(-n // k)]:
        covered.update(s)
    assert covered == set(range(n))
    again = cu.Selector(n, k, "random", seed=5)
    assert [again(i) for i in reversed(range(40))][::-1] == sets  # a cold start (any order) gives the cached sets
    other = cu.Selector(n, k, "random", seed=6)
    assert [other(i) for i in range(40)] != sets or k == n


def test_selection_does_not_touch_the_global_rngs():
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    before = (torch.get_rng_state(), random.getstate(), np.random.get_state()[1].copy())
    sel = cu.Selector(44, 7, "random", seed=0)
    for i in range(30):
        sel(i)
    assert torch.equal(before[0], torch.get_rng_state()) and before[1] == random.getstate()
    assert (before[2] == np.random.get_state()[1]).all()


def test_round_robin_and_window():
    rr = cu.Selector(10, 3, "round_robin")
    assert [rr(i) for i in range(5)] == [(0, 4, 8), (1, 5, 9), (2, 6), (3, 7), (0, 4, 8)]
    win = cu.Selector(10, 3, "window")
    assert [win(i) for i in range(5)] == [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 1, 9), (2, 3, 4)]
    for kind in cu.SELECTIONS:
        assert cu.Selector(6, 6, kind, seed=2)(7) == tuple(range(6))  # K = U: no switching
    with pytest.raises(ValueError):
        cu.Selector(5, 6)
    with pytest.raises(ValueError):
        cu.Selector(5, 2, "spiral")


def test_grad_floor():
    assert cu.grad_floor(["final_linear_video.weight", "layers.7.proj.q.weight", "layers.3.norm1.weight"], 42) == (3, False)
    assert cu.grad_floor(["final_linear_video.weight"], 42) == (42, False)
    assert cu.grad_floor(["layers.9.proj.q.weight", "text_embedder.blocks.1.qkv.weight"], 42) == (0, True)
    assert cu.grad_floor(["layers.9.proj.q.weight", "shared_block_modulation.weight"], 42) == (0, False)
    assert cu.grad_floor(["layers.9.proj.q.weight", "text_embedder.proj.weight"], 42) == (0, True)
    assert cu.grad_floor(["time_embedding.0.weight"], 42) == (0, False)


def test_bytes_per_tensor():
    assert cu.tensor_device_bytes("adamw", (3072, 3072)) == 16 * 3072 * 3072
    assert cu.state_bytes("adamw8bit", (3072, 3072)) == 2 * 3072 * 3072 + 8 * (3072 * 3072 // 256)
    assert cu.state_bytes("adamw8bit", (3072,)) == 8 * 3072  # below 4096 values: fp32 states
    assert cu.state_bytes("came", (16, 8)) == 4 * 128 + 4 * 2 * (16 + 8)
    assert cu.state_bytes("came", (16, 8), "bf16") == 2 * 128 + 4 * 2 * (16 + 8)
    assert cu.optimizer_temp_bytes("came", [(4, 4), (8, 2)]) == 2 * 4 * 16
    assert cu.optimizer_temp_bytes("adamw", [(4, 4)]) == 0
