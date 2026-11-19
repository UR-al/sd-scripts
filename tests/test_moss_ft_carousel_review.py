"""CPU tests of the Carousel review fixes (2026-11-17 numerics / integration reviews): the random selection is fair
across cycle boundaries, the stem as a rotating unit (--moss_carousel_always rotate_stem) with its backward
truncation, no switch after the last step, a resume without --moss_carousel_seed keeps the saved seed, the log keys
(lowest active joint block vs the gradient floor), CAME reset needs the bias correction, and the measured activation
model of the K budget.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_carousel_review.py -q -p no:cacheprovider
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

import _moss_ft_common as common
from _moss_ft_carousel_common import build_model, carousel_loss, make_engine, train_engine
from library.moss import ft_args, ft_budget, ft_carousel, ft_host
from library.moss import ft_carousel_units as cu
from test_moss_ft_args import FakeGroup, make_args, plenty_disk, plenty_memory
from test_moss_ft_carousel_state import load, save, scheduler_of

GIB = 1 << 30
C = ft_budget.CAROUSEL
F = ft_args.carousel_flag


@pytest.fixture(scope="module")
def parser():
    return ft_args.setup_parser()


# ---- selection fairness (numerics 4) --------------------------------------------------------------------------------
@pytest.mark.parametrize("n,k", [(44, 7), (45, 6), (44, 5), (11, 3), (10, 4), (7, 5), (44, 1)])
def test_random_selection_is_exactly_fair_across_cycles(n, k):
    sel = cu.Selector(n, k, "random", seed=11)
    sets = [sel(i) for i in range(3 * n)]
    assert all(len(s) == k and len(set(s)) == k for s in sets)
    # n sets draw n * k = k whole cycles: every unit exactly k times (the old stream skipped a carry unit drawn twice)
    for count in (n, 2 * n, 3 * n):
        seen = [0] * n
        for s in sets[:count]:
            for u in s:
                seen[u] += 1
        assert seen == [count * k // n] * n, (count, seen)
    if 2 * k <= n:  # a carry unit is not active again in the set right after the boundary set
        for a, b in zip(sets, sets[1:]):
            assert not set(a) & set(b), (a, b)
    again = cu.Selector(n, k, "random", seed=11)
    assert [again(i) for i in reversed(range(3 * n))][::-1] == sets


# ---- the stem as a rotating unit (integration 2) --------------------------------------------------------------------
def test_stem_unit_layout():
    shapes = ft_budget.moss_param_shapes()
    plain = cu.discover_units(shapes.items())
    units = cu.discover_units(shapes.items(), stem_unit=True)
    assert len(plain) == 44 and len(units) == 45
    stem = units[0]
    assert stem.name == cu.STEM_UNIT and stem.joint is None and stem.index == 0
    assert all(cu.tensor_role(n) == "stem" for n in stem.names)
    assert {n for n in shapes if cu.block_of(n) is None and cu.tensor_role(n) == "stem"} == set(stem.names)
    assert 79e6 < stem.numel < 81e6  # 79.8M: one block's size or less, so the per-unit budget is unchanged
    assert [u.name for u in units[1:]] == [u.name for u in plain]
    names = list(shapes)
    assert cu.always_names(names, "rotate_stem") == cu.always_names(names, "head")
    assert cu.stem_rotates("rotate_stem") and not cu.stem_rotates("small")
    # the stem inactive: the floor is the lowest active joint block; active: 0 with the text path
    head = cu.always_names(names, "head")
    assert cu.grad_floor(head + list(units[1 + 2 + 9].names), 42) == (9, False)
    assert cu.grad_floor(head + list(stem.names) + list(units[1 + 2 + 9].names), 42) == (0, True)


def test_rotate_stem_truncates_the_backward_while_the_stem_is_inactive():
    model = build_model()
    engine = make_engine(model, "adamw", k=1, period=1, select="window", always="rotate_stem")
    assert [u.name for u in engine.units][0] == cu.STEM_UNIT and len(engine.units) == 6  # stem + 2 refiners + 3 joint
    floors, lowest = [], []
    train_engine(engine, common.seeds([1] * 6),
                 after_window=lambda w, s: (floors.append(int(s.extra[f"{C}/grad_floor"])),
                                            lowest.append(int(s.extra[f"{C}/lowest_active_block"]))))
    # window K=1: stem, refiner 0, refiner 1, layers.0, layers.1, layers.2
    assert floors == [0, 0, 0, 0, 1, 2]
    assert lowest == [3, 3, 3, 0, 1, 2]  # num_layers (3) when no joint block is active


def test_lowest_active_block_differs_from_the_floor_with_small():
    engine = make_engine(build_model(), "adamw", k=1, period=1, select="window", always="small")
    stats = train_engine(engine, common.seeds([1] * 5))
    assert [int(s.extra[f"{C}/grad_floor"]) for s in stats] == [0] * 5  # the stem is trained at every step
    assert [int(s.extra[f"{C}/lowest_active_block"]) for s in stats] == [3, 3, 0, 1, 2]
    text = ft_carousel.window_text(stats[-1])
    assert "lowest active joint block 2" in text and "backward down to block 0" in text


# ---- no switch after the last step (numerics 2, integration 7) ------------------------------------------------------
def test_no_switch_after_the_last_step():
    engine = make_engine(build_model(), "adamw", k=2, period=2, select="window", always="head")
    for step in (1, 2):
        engine.begin_window(1, ft_host.StepContext(groups=(), step_index=step - 1))
        engine.begin_microstep(0)
        carousel_loss(engine.model, step).backward()
        engine.end_microstep()
        assert engine.after_step(step, max_steps=2) == {}
    assert engine.switches == 0  # step 2 is a period boundary but the run's last step
    assert engine.after_step(4, max_steps=10)  # an ordinary boundary still switches
    assert engine.switches == 1


# ---- the selection seed on resume (integration 3) -------------------------------------------------------------------
def test_resume_without_a_carousel_seed_keeps_the_saved_seed(tmp_path, monkeypatch):
    windows = common.seeds([1] * 8)
    a = make_engine(build_model(), "adamw", k=2, period=2, seed=101, seed_given=False)
    sched = scheduler_of(a)
    train_engine(a, windows[:3], scheduler=sched)
    path = save(a, sched, tmp_path, 3)
    train_engine(a, windows[3:], start_step=3, scheduler=sched)

    warnings = []
    monkeypatch.setattr(ft_carousel.logger, "warning", lambda m, *x, **k: warnings.append(str(m)))
    b = make_engine(build_model(seed=9), "adamw", k=2, period=2, seed=555, seed_given=False)  # a new random --seed
    sb = scheduler_of(b)
    load(b, sb, path)
    assert b.seed == 101 and not warnings
    train_engine(b, windows[3:], start_step=3, scheduler=sb)
    assert b.active == a.active
    for sa, sb_ in zip(a.slots, b.slots):
        assert torch.equal(sa.param, sb_.param), sa.name

    c = make_engine(build_model(), "adamw", k=2, period=2, seed=555, seed_given=True)  # --moss_carousel_seed 555
    load(c, scheduler_of(c), path)
    assert c.seed == 555 and warnings and "seed 101 -> 555" in warnings[-1]


# ---- CAME reset needs the bias correction (numerics 3) --------------------------------------------------------------
def test_came_reset_without_bias_correction_is_refused(parser):
    args = make_args(parser, F(), "--optimizer_type", "CAME", F("state"), "reset", "--optimizer_args",
                     "bias_correction=False", "betas=(0.9,0.999,0.99)")
    with pytest.raises(ValueError, match="reset needs bias_correction=True"):
        ft_args.validate_moss_ft_args(args, cuda=False)
    keep = make_args(parser, F(), "--optimizer_type", "CAME", "--optimizer_args", "bias_correction=False",
                     "betas=(0.9,0.999,0.99)")
    ft_args.validate_moss_ft_args(keep, cuda=False)  # keep: warned once (the first steps of each unit), allowed
    ok = make_args(parser, F(), "--optimizer_type", "CAME", F("state"), "reset")
    ft_args.validate_moss_ft_args(ok, cuda=False)


def test_rotate_stem_flag_counts_45_units(parser):
    args = make_args(parser, F(), F("always"), "rotate_stem", F("active"), "45")
    ft_args.validate_moss_ft_args(args, cuda=False)
    with pytest.raises(ValueError, match="active"):
        ft_args.validate_moss_ft_args(make_args(parser, F(), F("active"), "45"), cuda=False)


# ---- the measured activation model (integration 1) ------------------------------------------------------------------
def gpu(tokens, batch, **kw):
    return ft_budget.gpu_budget(tokens, batch, mixed_precision="fp16", stream_fp32=True, mode=C, **kw)


def test_measured_activation_model():
    te = ft_budget.TeLive(host="mmap", ring_slots=2, ahead=8)
    b4 = ft_budget.carousel_budget("adamw8bit", gpu(4096, 4, te_live=te))
    assert b4.activation_model == "measured" and b4.activations < b4.design_activations - 3 * GIB
    assert b4.k_budget >= 1  # the design estimate refused 1024 b4 (K = -1)
    # the fit reproduces the 11-17 smoke peaks (minus the compute copies) within its 0.5 GiB margin
    for batch, peak in ((1, 4.03), (4, 11.63)):
        fit = ft_budget.CAROUSEL_MEASURED_ACT_FIXED_BYTES + ft_budget.CAROUSEL_MEASURED_ACT_BYTES_PER_TOKEN * 4396 * batch
        assert 0 <= fit / GIB - peak <= 0.55, (batch, fit / GIB)
    b1 = ft_budget.carousel_budget("adamw", gpu(4096, 1))
    assert b1.activation_model == "design"  # lower than the fit at b1: the design estimate stays
    for kw in (dict(unsloth_offload=True), dict(reuse_attn=True)):
        assert ft_budget.measured_activations(gpu(4096, 4, **kw)) is None
    assert ft_budget.measured_activations(ft_budget.gpu_budget(4096, 4, mixed_precision="no", mode=C)) is None
    assert ft_budget.measured_activations(ft_budget.gpu_budget(4096, 4, stream_fp32=False, mode=C)) is None


def test_refusal_names_the_ways_out(parser, tmp_path):
    group = FakeGroup(tmp_path, images=8, batch_size=1)
    args = make_args(parser, F(), "--moss_ft_skip_resource_check", F("vram_gib"), "12", output_dir=str(tmp_path))
    ft_args.validate_moss_ft_args(args, cuda=False)
    with pytest.raises(ValueError) as err:
        ft_budget.run_startup_checks(args, group, C, virtual_memory=plenty_memory, disk_usage=plenty_disk,
                                     exists=lambda p: True)
    text = str(err.value)
    for needle in ("no unit fits", F("active"), "--unsloth_offload_checkpointing", "batch 1", "largest bucket"):
        assert needle in text, needle


def test_rotate_stem_budget_and_report(parser, tmp_path):
    g = gpu(4096, 1)
    small = ft_budget.carousel_budget("adamw", g, always="small")
    stem = ft_budget.carousel_budget("adamw", g, always="rotate_stem")
    assert stem.units == 45 and small.units == 44 and stem.unit_bytes == small.unit_bytes
    assert stem.always_params < small.always_params and stem.k_budget >= small.k_budget
    assert stem.host_masters == small.host_masters and stem.host_states > small.host_states  # + the stem state (keep)
    group = FakeGroup(tmp_path, images=8, batch_size=1)
    args = make_args(parser, F(), "--moss_ft_skip_resource_check", F("always"), "rotate_stem",
                     output_dir=str(tmp_path))
    ft_args.validate_moss_ft_args(args, cuda=False)
    report = ft_budget.run_startup_checks(args, group, C, virtual_memory=plenty_memory, disk_usage=plenty_disk,
                                          exists=lambda p: True)
    text = "\n".join(report.lines)
    assert "45 units (the stem + 2 text refiner + 42 joint blocks)" in text
    assert "while the stem unit is inactive" in text
