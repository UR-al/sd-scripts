"""CPU tests of the Carousel budget (library/moss/ft_budget.py carousel section; docs/moss_ft_carousel_plan.md 6,
items 4-7): the per-unit device bytes of plan 3.8, K from a VRAM budget at its boundaries and the K < 1 refusal, the
host RAM of keep / reset / bf16, the state size per kind, the start-up report lines, the pinned-masters refusal.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_carousel_budget.py -q -p no:cacheprovider
"""
import math
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest

from library.moss import ft_args, ft_budget
from library.moss import ft_carousel_units as cu
from test_moss_ft_args import FakeGroup, make_args, plenty_disk, plenty_memory

GIB = 1 << 30
C = ft_budget.CAROUSEL


@pytest.fixture(scope="module")
def parser():
    return ft_args.setup_parser()


@pytest.fixture(scope="module")
def layout():
    units, trained = ft_budget.carousel_layout(())
    return units, dict(trained)


@pytest.mark.parametrize("kind,joint,refiner,small,head", [
    ("adamw", 1_812_041_728, 1_812_549_632, 1.4772, 0.2872),
    ("adamw8bit", 1_136_103_424, 1_136_458_528, 0.9262, 0.1801),
    ("came", 1_359_524_864, 1_360_135_168, 1.1086, 0.2156),
])
def test_per_unit_bytes_match_the_plan(layout, kind, joint, refiner, small, head):
    units, shapes = layout
    assert cu.group_device_bytes(kind, cu.unit_shapes(units[2], shapes)) == joint
    assert cu.group_device_bytes(kind, cu.unit_shapes(units[0], shapes)) == refiner
    names = list(shapes)
    for policy, gib in (("small", small), ("head", head)):
        got = cu.group_device_bytes(kind, [shapes[n] for n in cu.always_names(names, policy)]) / GIB
        assert got == pytest.approx(gib, abs=6e-5), (policy, got)
    gpu = ft_budget.gpu_budget(4096, 1, mode=C)
    cb = ft_budget.carousel_budget(kind, gpu)
    assert cb.unit_bytes == refiner and cb.units == 44  # the budget takes the largest unit


def test_k_from_the_vram_budget_at_its_boundaries():
    gpu = ft_budget.gpu_budget(4096, 1, mode=C)
    base = ft_budget.carousel_budget("adamw", gpu, vram_bytes=40 * GIB)
    fixed = base.weights + base.always_bytes + base.temp_bytes + base.activations
    for k in (1, 5, 9):
        exact = ft_budget.OUTSIDE_BYTES + (1 + ft_budget.ALLOCATOR_SLACK) * (fixed + k * base.unit_bytes)
        assert ft_budget.carousel_budget("adamw", gpu, vram_bytes=int(math.ceil(exact)) + 1).k_budget == k
        assert ft_budget.carousel_budget("adamw", gpu, vram_bytes=int(exact) - (1 << 20)).k_budget == k - 1
    assert ft_budget.carousel_budget("adamw", gpu, vram_bytes=200 * GIB).k == 44  # clamped to the units
    # the measured activations of the 11-17 smoke at b1 (3.5 GiB, plan 3.8) on the full 31.84 GiB card: K 7 / 12 / 9
    smoke = ft_budget.GpuBudget(image_tokens=4096, text_tokens=300, batch=1, weights=gpu.weights, block_inputs=0,
                                recompute=0, grads_in_flight=ft_budget.CAROUSEL_GRADS_IN_FLIGHT_BYTES,
                                misc=int(3.5 * GIB) - ft_budget.CAROUSEL_GRADS_IN_FLIGHT_BYTES)
    card = int(31.84 * GIB)
    assert [ft_budget.carousel_budget(k, smoke, vram_bytes=card).k for k in ("adamw", "adamw8bit", "came")] == [7, 12, 9]
    assert [ft_budget.carousel_budget(k, smoke, vram_bytes=card, always="head").k
            for k in ("adamw", "adamw8bit", "came")] == [8, 13, 10]


def test_host_and_state_sizes():
    gpu = ft_budget.gpu_budget(4096, 1, mode=C)
    keep = {k: ft_budget.carousel_budget(k, gpu).host_states / GIB for k in cu.KINDS}
    assert keep["adamw"] == pytest.approx(37.13, abs=0.01) and keep["adamw8bit"] == pytest.approx(9.43, abs=0.01)
    assert keep["came"] == pytest.approx(18.58, abs=0.01)
    assert ft_budget.carousel_budget("came", gpu, state_dtype="bf16").host_states / GIB == pytest.approx(9.30, abs=0.01)
    assert ft_budget.carousel_budget("adamw", gpu, state_policy="reset").host_states == 0
    assert ft_budget.carousel_budget("adamw", gpu).host_masters / GIB == pytest.approx(18.93, abs=0.01)
    assert ft_budget.carousel_budget("adamw", gpu, pinned_masters=True).pinned_master == 4 * ft_budget.TOTAL_PARAMS
    sizes = {k: ft_budget.carousel_state_bytes(k) / 1e9 for k in cu.KINDS}
    assert sizes["adamw"] == pytest.approx(61.0, abs=0.1) and sizes["adamw8bit"] == pytest.approx(30.7, abs=0.1)
    assert sizes["came"] == pytest.approx(40.7, abs=0.1)
    assert ft_budget.carousel_state_bytes("came", state_dtype="bf16") / 1e9 == pytest.approx(30.7, abs=0.15)
    reset = ft_budget.carousel_state_bytes("adamw", state_policy="reset", k=7) / 1e9
    assert reset == pytest.approx(20.33 + 7 * 0.906 + 0.79, abs=0.1)
    cb = ft_budget.carousel_budget("adamw", gpu, active=7)
    assert cb.switch_bytes() == 113_284_352 * 12 and cb.switch_seconds() == pytest.approx(7 * 2 * 1.36e9 / 8.5e9, rel=0.01)


def carousel_args(parser, *argv, **overrides):
    return make_args(parser, ft_args.carousel_flag(), "--moss_ft_skip_resource_check", *argv, **overrides)


def test_startup_report_and_refusals(parser, tmp_path):
    group = FakeGroup(tmp_path, images=8, batch_size=1)
    args = carousel_args(parser, ft_args.carousel_flag("vram_gib"), "24", output_dir=str(tmp_path))
    ft_args.validate_moss_ft_args(args, cuda=False)
    report = ft_budget.run_startup_checks(args, group, ft_args.CAROUSEL, virtual_memory=plenty_memory,
                                          disk_usage=plenty_disk, exists=lambda p: True)
    cb = report.carousel
    assert cb is not None and cb.k == cb.k_budget >= 1 and args._moss_rotation_k == cb.k
    text = "\n".join(report.lines)
    for needle in (f"{C}: 44 units", "GPU per active unit", f"K = {cb.k} active units", "switch:", "truncation:",
                   "host fp32 masters", "host kept optimizer states"):
        assert needle in text, needle
    assert report.host.rings == 0 and report.host.accumulator == 0 and report.host.exp_avg == cb.host_states

    small = carousel_args(parser, ft_args.carousel_flag("vram_gib"), "12", output_dir=str(tmp_path))
    ft_args.validate_moss_ft_args(small, cuda=False)
    with pytest.raises(ValueError, match="no unit fits"):
        ft_budget.run_startup_checks(small, group, ft_args.CAROUSEL, virtual_memory=plenty_memory,
                                     disk_usage=plenty_disk, exists=lambda p: True)

    pinned = carousel_args(parser, ft_args.carousel_flag("host_pinned"), "masters", output_dir=str(tmp_path))
    ft_args.validate_moss_ft_args(pinned, cuda=True)
    ram = lambda: types.SimpleNamespace(available=1 << 50, total=40 * GIB)  # noqa: E731 - 18.9 GiB > 40 % of 40
    with pytest.raises(ValueError, match="host_pinned masters"):
        ft_budget.run_startup_checks(pinned, group, ft_args.CAROUSEL, virtual_memory=ram, disk_usage=plenty_disk,
                                     exists=lambda p: True)
    ft_budget.run_startup_checks(pinned, group, ft_args.CAROUSEL, virtual_memory=plenty_memory, disk_usage=plenty_disk,
                                 exists=lambda p: True)
