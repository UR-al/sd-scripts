"""CPU tests of the `adamw_host` pipeline (library/moss/ft_host.py GradPipeline + library/moss/ft_optim.py HostAdamW;
docs/moss_full_finetune_design.md 3.1 and the section-5 rows of this file).

The pipeline runs on device "cpu": no streams, events are no-ops, the rings are plain memory and every copy is
synchronous; the hooks, the FIFO worker (or the inline call), the accumulation, the per-piece AdamW, the uploads, the
clipping plans, the non-finite skip and the flush are the code the CUDA path runs. With fp32 compute every result is
compared bit for bit with torch.optim.AdamW(fused=True) on an fp32 copy of the model driven with loss / n_micro.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_host_adamw.py -q -p no:cacheprovider
"""
import math
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _moss_ft_common as C  # noqa: E402

from library.moss import ft_host  # noqa: E402
from library.moss.ft_optim import HostAdamW  # noqa: E402
from library.moss.models import MossTransformer  # noqa: E402

LR = 1e-3
WD = 0.1
FROZEN = r"layers\.1\..*"


def lr_lambda(step):
    return 1.0 / (1.0 + 0.5 * step)  # the LR changes every window: the snapshot must follow the scheduler


def setup(ga, inline, frozen=FROZEN, extra_params=(), sched=True, **kw):
    """(pipeline model, its pipeline objects, reference model, reference optimizer + scheduler)."""
    model = C.build_model(seed=3)
    ref_model = C.clone_model(model)
    named = C.trainable(model, frozen) + list(extra_params[0] if extra_params else [])
    ref_named = C.trainable(ref_model, frozen) + list(extra_params[1] if extra_params else [])
    opt = HostAdamW(C.param_groups(named, WD), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    ref_opt = torch.optim.AdamW(C.param_groups(ref_named, WD), lr=LR, betas=(0.9, 0.999), eps=1e-8, fused=True)
    lam = torch.optim.lr_scheduler.LambdaLR
    scheds = (lam(opt, lr_lambda), lam(ref_opt, lr_lambda)) if sched else (None, None)
    pipe, pool, slots = C.make_pipeline(named, opt, ga, inline=inline, **kw)
    return dict(model=model, named=named, opt=opt, pipe=pipe, pool=pool, slots=slots, sched=scheds[0],
                ref_model=ref_model, ref_named=ref_named, ref_opt=ref_opt, ref_sched=scheds[1])


@pytest.fixture
def closing():
    pipes = []
    yield pipes.append
    for pipe in pipes:
        pipe.close()


@pytest.mark.parametrize("inline", [True, False], ids=["inline", "threaded"])
@pytest.mark.parametrize("ga", [1, 3])
def test_adamw_host_equals_torch_adamw(ga, inline, closing):
    s = setup(ga, inline)
    closing(s["pipe"])
    sizes = [ga] * 5
    seeds = C.seeds(sizes)
    stats = C.train_pipeline(s["pipe"], s["opt"], C.model_windows(s["model"], seeds), s["sched"])
    C.train_reference(s["ref_opt"], [p for _, p in s["ref_named"]], C.model_windows(s["ref_model"], seeds),
                      s["ref_sched"])
    C.assert_pool_equals_torch(s["pool"], s["slots"], s["ref_named"], s["ref_opt"], f"ga {ga}")
    C.assert_compute_copies(s["pool"], s["slots"])
    # the frozen block is not in the pool and never changed
    assert not any(slot.name.startswith("layers.1.") for slot in s["slots"])
    for (name, p), (_, q) in zip(s["model"].named_parameters(), s["ref_model"].named_parameters()):
        if name.startswith("layers.1."):
            assert torch.equal(p, q) and p.grad is None
    assert all(st.skipped_params == 0 and st.n_micro == ga for st in stats)
    assert all(st.clip_mode == "none" and st.grad_norm > 0 for st in stats)
    assert [g["weight_decay"] for g in s["opt"].param_groups] == [WD, 0.0]
    assert float(s["pool"].steps.max()) == 5.0


@pytest.mark.parametrize("inline", [True, False], ids=["inline", "threaded"])
@pytest.mark.parametrize("ga", [1, 3])
def test_exact_clip_equals_clip_grad_norm(ga, inline, closing):
    max_norm = 0.05
    s = setup(ga, inline, max_grad_norm=max_norm, clip_mode="exact")
    closing(s["pipe"])
    assert s["pool"].acc is not None  # exact mode needs the accumulator even at GA = 1
    seeds = C.seeds([ga] * 4)
    stats = C.train_pipeline(s["pipe"], s["opt"], C.model_windows(s["model"], seeds), s["sched"])
    norms = C.train_reference(s["ref_opt"], [p for _, p in s["ref_named"]], C.model_windows(s["ref_model"], seeds),
                              s["ref_sched"], max_grad_norm=max_norm)
    C.assert_pool_equals_torch(s["pool"], s["slots"], s["ref_named"], s["ref_opt"], "exact clip")
    C.assert_compute_copies(s["pool"], s["slots"])
    for st, total in zip(stats, norms):
        assert st.clip_mode == "exact"
        assert st.grad_norm == float(total)  # torch's total norm, bit for bit
        assert st.clip_coef == float(torch.clamp(max_norm / (total + 1e-6), max=1.0))
        assert st.clip_coef < 1.0  # clipping really happened


@pytest.mark.parametrize("inline", [True, False], ids=["inline", "threaded"])
def test_stale_clip_uses_previous_norm(inline, closing):
    max_norm = 0.05
    s = setup(2, inline, max_grad_norm=max_norm, clip_mode="stale")
    closing(s["pipe"])
    seeds = C.seeds([2] * 4)
    stats = C.train_pipeline(s["pipe"], s["opt"], C.model_windows(s["model"], seeds), s["sched"])
    assert stats[0].clip_mode == "exact"  # no previous norm: window 1 runs exact
    assert [st.clip_mode for st in stats[1:]] == ["stale"] * 3
    coefs = [None]
    for prev, st in zip(stats, stats[1:]):
        # c_t = min(1, max_norm / (G_{t-1} + 1e-6)), formed like clip_grad_norm_ forms it (an fp32 0-dim tensor)
        expected = torch.clamp(max_norm / (torch.tensor(prev.grad_norm, dtype=torch.float32) + 1e-6), max=1.0)
        assert st.clip_coef == float(expected)
        assert math.isclose(st.clip_coef, min(1.0, max_norm / (prev.grad_norm + 1e-6)), rel_tol=1e-6)
        coefs.append(expected)
    assert stats[1].clip_coef < 1.0

    first = {}

    def coef_for(w):
        return coefs[w]

    def before_step(w):  # window 1 of the reference: exact clipping, as the pipeline ran it
        if w == 0:
            with_grad = [p for _, p in s["ref_named"] if p.grad is not None]
            first["norm"] = torch.nn.utils.clip_grad_norm_(with_grad, max_norm)

    C.train_reference(s["ref_opt"], [p for _, p in s["ref_named"]], C.model_windows(s["ref_model"], seeds),
                      s["ref_sched"], coef_for=coef_for, before_step=before_step)
    assert stats[0].grad_norm == float(first["norm"])
    C.assert_pool_equals_torch(s["pool"], s["slots"], s["ref_named"], s["ref_opt"], "stale clip")
    assert s["pipe"].last_grad_norm == stats[-1].grad_norm


@pytest.mark.parametrize("inline", [True, False], ids=["inline", "threaded"])
def test_stale_clip_without_accumulator_measures_the_first_window(inline, closing):
    """Review fix (design 3.1: a window without a known previous norm runs exact). GA = 1, stale, max_grad_norm > 0 has
    no accumulator, so that window cannot wait for its own norm. It used to step unclipped, which put the unclipped
    gradient into exp_avg / exp_avg_sq at (G / max_norm)^2 scale and shrank every clipped update after it for ~1 /
    (1 - beta2) steps. Now it only measures G: no update, no moment or step-counter change; the next windows clip with
    it. From window 2 on bit-identical to torch AdamW with the same coefficients."""
    max_norm = 0.05
    s = setup(1, inline, max_grad_norm=max_norm, clip_mode="stale", sched=False)
    closing(s["pipe"])
    assert s["pool"].acc is None
    roles_before = {role: t.clone() for role, t in s["pool"].roles.items()}
    params_before = [p.detach().clone() for _, p in s["named"]]
    after_first = {}

    def after_window(w, st):
        if w == 0:
            after_first.update(roles={r: t.clone() for r, t in s["pool"].roles.items()}, steps=s["pool"].steps.clone(),
                               params=[p.detach().clone() for _, p in s["named"]])

    seeds = C.seeds([1] * 4)
    stats = C.train_pipeline(s["pipe"], s["opt"], C.model_windows(s["model"], seeds), after_window=after_window)
    assert [st.clip_mode for st in stats] == ["measure", "stale", "stale", "stale"]
    assert stats[0].stepped_params == 0 and math.isfinite(stats[0].grad_norm) and stats[0].grad_norm > max_norm
    assert all(torch.equal(after_first["roles"][role], t) for role, t in roles_before.items())
    assert not after_first["steps"].any()
    assert all(torch.equal(a, b) for a, b in zip(after_first["params"], params_before))
    coefs = [None] + [torch.clamp(max_norm / (torch.tensor(st.grad_norm, dtype=torch.float32) + 1e-6), max=1.0)
                      for st in stats[:-1]]
    assert stats[1].clip_coef == float(coefs[1]) < 1.0

    measured = {}

    def before_step(w):  # window 1 of the reference: the norm only, no step
        if w == 0:
            grads = [p.grad for _, p in s["ref_named"] if p.grad is not None]
            measured["norm"] = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
            for _, p in s["ref_named"]:
                p.grad = None

    C.train_reference(s["ref_opt"], [p for _, p in s["ref_named"]], C.model_windows(s["ref_model"], seeds),
                      coef_for=lambda w: coefs[w], before_step=before_step)
    assert stats[0].grad_norm == pytest.approx(float(measured["norm"]), rel=1e-5)
    C.assert_pool_equals_torch(s["pool"], s["slots"], s["ref_named"], s["ref_opt"], "measure first window")
    C.assert_compute_copies(s["pool"], s["slots"])


def test_came_host_measures_the_first_window_without_accumulator(closing):
    """The same for came_host at GA = 1 (per-parameter step, no accumulator): window 1 changes no state at all."""
    from library.moss.ft_optim import HostCAME

    model = C.build_model(seed=3)
    named = C.trainable(model)
    opt = HostCAME(C.param_groups(named, 0.0), lr=1e-3, betas=(0.9, 0.999, 0.99))
    pipe, pool, slots = C.make_pipeline(named, opt, 1, inline=True, max_grad_norm=0.05, clip_mode="stale")
    closing(pipe)
    assert pool.acc is None
    master = pool.roles["master"].clone()
    stats = C.train_pipeline(pipe, opt, C.model_windows(model, C.seeds([1])))
    assert stats[0].clip_mode == "measure" and stats[0].stepped_params == 0 and stats[0].grad_norm > 0.05
    assert torch.equal(pool.roles["master"], master) and not pool.roles["exp_avg"].any() and not pool.steps.any()
    assert not any(t.any() for t in opt.extra_state().values())
    stats = C.train_pipeline(pipe, opt, C.model_windows(model, C.seeds([1], start=1)))
    assert stats[0].clip_mode == "stale" and stats[0].clip_coef < 1.0 and stats[0].stepped_params == len(slots)


@pytest.mark.parametrize("inline", [True, False], ids=["inline", "threaded"])
def test_skipped_window_keeps_the_previous_norm(inline, closing):
    """Review fix: a window that skipped parameters for non-finite gradients has no usable norm - its stepped part
    understates G, and a fully skipped window measured 0.0, which turned stale clipping off for the next window (a
    spike right after a divergence event went through unclipped). Such a window reports grad_norm NaN and leaves the
    last clean window's norm for the next coefficient (and for the training state)."""
    max_norm = 0.05
    s = setup(2, inline, sched=False, max_grad_norm=max_norm, clip_mode="stale")
    closing(s["pipe"])
    target = "layers.2.proj.q.weight"
    p_target = dict(s["named"])[target]
    handles = []

    def inject(w, j):  # window 3, micro-step 0: an inf in one element of one parameter's gradient
        for h in handles:
            h.remove()
        handles.clear()
        if w == 3 and j == 0:
            def poison(g):
                g = g.clone()
                g.view(-1)[3] = float("inf")
                return g
            handles.append(p_target.register_hook(poison))
        return None

    def scaled(fn, factor):
        return lambda: fn() * factor

    windows = C.model_windows(s["model"], C.seeds([2] * 5), extra=inject)
    windows[1][0] = scaled(windows[1][0], float("nan"))  # every gradient non-finite: every parameter skipped
    windows[2] = [scaled(fn, 200.0) for fn in windows[2]]  # a spike right after it
    norms = []
    stats = C.train_pipeline(s["pipe"], s["opt"], windows,
                             after_window=lambda w, st: norms.append(s["pipe"].last_grad_norm))
    g0, g2 = stats[0].grad_norm, stats[2].grad_norm
    assert stats[0].clip_mode == "exact" and math.isfinite(g0)
    assert stats[1].skipped_params == len(s["slots"]) and math.isnan(stats[1].grad_norm)
    assert stats[3].skipped_params == 1 and math.isnan(stats[3].grad_norm)
    assert norms == [g0, g0, g2, g2, stats[4].grad_norm]

    def coef(g):
        return float(torch.clamp(max_norm / (torch.tensor(g, dtype=torch.float32) + 1e-6), max=1.0))

    assert [st.clip_mode for st in stats[1:]] == ["stale"] * 4
    assert stats[2].clip_coef == coef(g0) < 1.0  # the spike is clipped with the last clean norm
    assert g2 > 50 * g0 and stats[4].clip_coef == coef(g2)


def test_pinned_bytes_count_the_allocator_rounding(closing):
    """host/pinned_gib: what torch's pinned host allocator holds for the two rings and the norm table - each allocation
    rounded up to a power of two (the review found a 1.5 GiB ring logged as 1.5 GiB while it page-locks 2 GiB)."""
    from library.moss import ft_budget

    model = C.build_model(seed=3)
    named = C.trainable(model)
    opt = HostAdamW(C.param_groups(named, WD), lr=LR)
    pipe, pool, slots = C.make_pipeline(named, opt, 2, ring_bytes=3 * C.PIECE_BYTES, upload_ring_bytes=3 * C.PIECE_BYTES)
    closing(pipe)
    assert pipe.pinned_bytes == 0  # the CPU device page-locks nothing
    pipe.ops.cuda = True  # what the same buffers would page-lock on CUDA
    try:
        table = pipe.norm_host.numel() * pipe.norm_host.element_size()
        assert pipe.pinned_bytes == 2 * 4 * C.PIECE_BYTES + ft_budget.pinned_alloc_bytes(table)
    finally:
        del pipe.ops.cuda


@pytest.mark.parametrize("ckpt", ["none", "plain"])
def test_in_backward_step_equals_after_backward_step(ckpt, closing):
    """Inline: every upload happens inside the hook, before the rest of the backward runs - the strongest CPU check that
    nothing reads a parameter after its post-accumulate-grad hook (design 3.1, "Why the in-backward step is safe")."""
    runs = []
    for overlap in (True, False):
        model = C.build_model(seed=5, ckpt=ckpt)
        named = C.trainable(model)
        opt = HostAdamW(C.param_groups(named, WD), lr=LR)
        pipe, pool, slots = C.make_pipeline(named, opt, 2, inline=True, overlap=overlap)
        closing(pipe)
        stats = C.train_pipeline(pipe, opt, C.model_windows(model, C.seeds([2] * 3)))
        runs.append((model, pool, slots, stats))
    (m0, p0, s0, st0), (m1, p1, s1, st1) = runs
    for a, b in zip(s0, s1):
        for role in ("master", "exp_avg", "exp_avg_sq"):
            assert torch.equal(p0.view(role, a), p1.view(role, b)), (role, a.name)
        assert torch.equal(a.param, b.param)
    assert torch.equal(p0.steps, p1.steps)
    assert [st.clip_mode for st in st0] == [st.clip_mode for st in st1] == ["none"] * 3


@pytest.mark.parametrize("inline", [True, False], ids=["inline", "threaded"])
def test_partial_last_window(inline, closing):
    s = setup(3, inline)
    closing(s["pipe"])
    sizes = C.window_sizes(7, 3)
    assert sizes == [3, 3, 1]
    seeds = C.seeds(sizes)
    stats = C.train_pipeline(s["pipe"], s["opt"], C.model_windows(s["model"], seeds), s["sched"])
    C.train_reference(s["ref_opt"], [p for _, p in s["ref_named"]], C.model_windows(s["ref_model"], seeds),
                      s["ref_sched"])
    assert [st.n_micro for st in stats] == sizes
    C.assert_pool_equals_torch(s["pool"], s["slots"], s["ref_named"], s["ref_opt"], "partial window")
    C.assert_compute_copies(s["pool"], s["slots"])


@pytest.mark.parametrize("inline", [True, False], ids=["inline", "threaded"])
def test_nonfinite_grad_skips_only_that_param(inline, closing):
    s = setup(3, inline, sched=False, max_skipped_steps=2)
    closing(s["pipe"])
    target = "layers.2.proj.q.weight"
    p_target = dict(s["named"])[target]
    slot = next(sl for sl in s["slots"] if sl.name == target)
    handles = []

    def inject(w, j):  # a NaN in one element of the target's gradient, micro-step 1 of window 1 (0-based)
        for h in handles:
            h.remove()
        handles.clear()
        if w == 1 and j == 1:
            def poison(g):
                g = g.clone()
                g.view(-1)[7] = float("nan")
                return g
            handles.append(p_target.register_hook(poison))
        return None

    before = {}

    def after_window(w, st):
        if w == 0:
            before.update(master=s["pool"].view("master", slot).clone(), m=s["pool"].view("exp_avg", slot).clone(),
                          v=s["pool"].view("exp_avg_sq", slot).clone(), step=float(s["pool"].steps[slot.index]),
                          param=p_target.detach().clone())

    seeds = C.seeds([3] * 3)
    stats = C.train_pipeline(s["pipe"], s["opt"], C.model_windows(s["model"], seeds, extra=inject),
                             after_window=after_window)
    assert [st.skipped_params for st in stats] == [0, 1, 0]
    assert stats[1].skipped_names == [target]
    assert [st.skip_streak for st in stats] == [0, 1, 0]
    for st in stats:
        s["pipe"].check_skip_streak(st)  # never two in a row: no error

    # the reference drops the target's gradient in window 1: torch AdamW then leaves it (and its step) alone
    ref_target = dict(s["ref_named"])[target]

    def before_step(w):
        if w == 1:
            ref_target.grad = None

    C.train_reference(s["ref_opt"], [p for _, p in s["ref_named"]], C.model_windows(s["ref_model"], seeds),
                      before_step=before_step)
    C.assert_pool_equals_torch(s["pool"], s["slots"], s["ref_named"], s["ref_opt"], "non-finite skip")
    C.assert_compute_copies(s["pool"], s["slots"])
    assert float(s["pool"].steps[slot.index]) == 2.0 and float(s["pool"].steps.max()) == 3.0

    # window 1 alone: the poisoned slot kept master / m / v / step and its compute copy
    s2 = setup(3, inline, sched=False, max_skipped_steps=2)
    closing(s2["pipe"])
    p2 = dict(s2["named"])[target]
    slot2 = next(sl for sl in s2["slots"] if sl.name == target)
    snap = {}

    def snap_after(w, st):
        if w == 0:
            snap.update(master=s2["pool"].view("master", slot2).clone(), m=s2["pool"].view("exp_avg", slot2).clone(),
                        v=s2["pool"].view("exp_avg_sq", slot2).clone(), param=p2.detach().clone())

    handles2 = []

    def inject2(w, j):
        for h in handles2:
            h.remove()
        handles2.clear()
        if w >= 1 and j == 1:
            def poison(g):
                g = g.clone()
                g.view(-1)[0] = float("inf")
                return g
            handles2.append(p2.register_hook(poison))
        return None

    windows = C.model_windows(s2["model"], C.seeds([3] * 3), extra=inject2)
    st_a = C.train_pipeline(s2["pipe"], s2["opt"], windows[:2], after_window=snap_after)
    assert torch.equal(s2["pool"].view("master", slot2), snap["master"])
    assert torch.equal(s2["pool"].view("exp_avg", slot2), snap["m"])
    assert torch.equal(s2["pool"].view("exp_avg_sq", slot2), snap["v"])
    assert torch.equal(p2.detach(), snap["param"])
    assert float(s2["pool"].steps[slot2.index]) == 1.0
    s2["pipe"].check_skip_streak(st_a[-1])  # streak 1 < 2
    st_b = C.train_pipeline(s2["pipe"], s2["opt"], windows[2:])  # second window in a row with a skip
    assert st_b[0].skip_streak == 2
    with pytest.raises(RuntimeError, match="2 optimizer steps in a row"):
        s2["pipe"].check_skip_streak(st_b[0])


@pytest.mark.parametrize("inline", [True, False], ids=["inline", "threaded"])
def test_param_without_final_grad_is_flushed(inline, closing):
    """extra0 gets a gradient only in micro-step 0, extra_last only in the final micro-step: both are stepped at the
    window end exactly like torch (whose p.grad keeps micro-step 0's gradient)."""
    def extras():
        g = torch.Generator().manual_seed(9)
        return [torch.nn.Parameter(torch.randn(16, 80, generator=g) * 0.1),
                torch.nn.Parameter(torch.randn(1300, generator=g) * 0.1)]

    mine, ref = extras(), extras()
    s = setup(3, inline, extra_params=([("extra0", mine[0]), ("extra_last", mine[1])],
                                       [("extra0", ref[0]), ("extra_last", ref[1])]))
    closing(s["pipe"])
    assert sum(1 for sl in s["slots"] if sl.name.startswith("extra")) == 2
    k = torch.linspace(-1.0, 1.0, 1280).view(16, 80)
    k2 = torch.linspace(0.5, -0.25, 1300)

    def term(params):
        def extra(w, j):
            if j == 0:
                return (params[0] * k).sum() * 1e-2
            if j == 2:
                return (params[1] * k2).square().sum() * 1e-2
            return None
        return extra

    seeds = C.seeds([3] * 3)
    C.train_pipeline(s["pipe"], s["opt"], C.model_windows(s["model"], seeds, extra=term(mine)), s["sched"])
    C.train_reference(s["ref_opt"], [p for _, p in s["ref_named"]],
                      C.model_windows(s["ref_model"], seeds, extra=term(ref)), s["ref_sched"])
    C.assert_pool_equals_torch(s["pool"], s["slots"], s["ref_named"], s["ref_opt"], "flush")
    C.assert_compute_copies(s["pool"], s["slots"])
    for sl in s["slots"]:
        if sl.name.startswith("extra"):
            assert float(s["pool"].steps[sl.index]) == 3.0, sl.name


def test_worker_error_surfaces_in_backward(monkeypatch, closing):
    s = setup(2, False, sched=False, ring_bytes=C.PIECE_BYTES)  # a one-entry ring: the hook waits on the worker
    pipe = s["pipe"]
    closing(pipe)

    def boom(*args, **kwargs):
        raise ValueError("boom in the host worker")

    monkeypatch.setattr(pipe, "_accumulate", boom)
    pipe.begin_window(2, ft_host.StepContext.snapshot(s["opt"]))
    pipe.begin_microstep(0)
    with pytest.raises(RuntimeError) as info:
        (C.loss_of(s["model"], C.make_batch(0)) / 2).backward()
        pipe.end_microstep()
    text = str(info.value)
    assert "boom in the host worker" in text and "Traceback" in text and "boom" in text
    with pytest.raises(RuntimeError, match="boom in the host worker"):
        pipe.end_microstep()  # the pipeline stays failed


@pytest.mark.parametrize("inline", [True, False], ids=["inline", "threaded"])
def test_worker_error_surfaces_at_drain(monkeypatch, inline, closing):
    """Exact clipping steps after the drain, so an error in the AdamW kernel call can only surface there."""
    s = setup(2, inline, sched=False, max_grad_norm=0.05, clip_mode="exact")
    pipe = s["pipe"]
    closing(pipe)

    def boom(*args, **kwargs):
        raise ValueError("kernel failed on the host")

    monkeypatch.setattr(s["opt"], "step_chunk", boom)
    pipe.begin_window(2, ft_host.StepContext.snapshot(s["opt"]))
    for j in range(2):
        pipe.begin_microstep(j)
        (C.loss_of(s["model"], C.make_batch(j)) / 2).backward()
        if j == 0:
            assert pipe.end_microstep() is None
    with pytest.raises(RuntimeError) as info:
        pipe.end_microstep()
    assert "kernel failed on the host" in str(info.value)
    if not inline:
        assert "Traceback" in str(info.value)


def test_dead_worker_does_not_block_the_hook(closing):
    """A worker thread that is gone (here stopped on purpose, as after a crash) must not leave the hook waiting for a
    ring entry forever: acquire() polls the worker every 0.5 s (design 6, risk 7)."""
    s = setup(2, False, sched=False, ring_bytes=C.PIECE_BYTES)  # one ring entry
    pipe = s["pipe"]
    closing(pipe)
    pipe._fifo.put(ft_host._STOP)
    pipe._thread.join(5.0)
    assert not pipe._thread.is_alive()
    pipe.begin_window(2, ft_host.StepContext.snapshot(s["opt"]))
    pipe.begin_microstep(0)
    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="not running"):
        (C.loss_of(s["model"], C.make_batch(0)) / 2).backward()
    assert time.perf_counter() - t0 < 10.0


def test_slots_from_meta_model_bind_later(closing):
    """The loader's order (design 4.4.1): slots and pool from the model on the meta device, master filled per tensor,
    the live parameters (and the optimizer's groups) bound afterwards."""
    with torch.device("meta"):
        meta = MossTransformer(**C.TINY)
    slots = ft_host.make_slots(list(meta.named_parameters()))
    pool = ft_host.HostPool(slots, HostAdamW.POOL_ROLES, with_acc=True)
    model = C.build_model(seed=3)
    named = C.trainable(model)
    for slot, (_, p) in zip(slots, named):
        pool.view("master", slot).copy_(p.detach().reshape(-1))
    opt = HostAdamW(C.param_groups(named, WD), lr=LR)
    ft_host.bind_slots(slots, named, opt)
    assert all(sl.param is p for sl, (_, p) in zip(slots, named))
    pipe = ft_host.GradPipeline(slots, pool, opt, device="cpu", gradient_accumulation_steps=2,
                                piece_bytes=C.PIECE_BYTES, ring_bytes=C.RING_BYTES,
                                upload_ring_bytes=C.UPLOAD_RING_BYTES, acc_chunk=C.CHUNK, step_chunk=C.CHUNK)
    closing(pipe)
    pipe.attach()
    stats = C.train_pipeline(pipe, opt, C.model_windows(model, C.seeds([2, 2])))
    assert all(st.stepped_params == len(slots) for st in stats)
    C.assert_compute_copies(pool, slots)
    with pytest.raises(ValueError):
        ft_host.bind_slots(slots, list(reversed(named)))


def test_slot_layout_and_hash():
    model = C.build_model(seed=1)
    named = C.trainable(model, FROZEN)
    opt = HostAdamW(C.param_groups(named, WD), lr=LR)
    slots = ft_host.make_slots(named, opt)
    assert [sl.name for sl in slots] == [n for n, _ in named]
    assert [sl.index for sl in slots] == list(range(len(slots)))
    offset = 0
    for sl, (_, p) in zip(slots, named):
        assert sl.offset == offset and sl.numel == p.numel() and sl.shape == tuple(p.shape) and sl.param is p
        assert sl.group == (0 if C.is_decay(sl.name, p) else 1)
        assert sl.compute_dtype == torch.float32
        offset += p.numel()
    pool = ft_host.HostPool(slots, HostAdamW.POOL_ROLES, with_acc=False)
    assert pool.total_numel == offset and pool.acc is None
    assert set(pool.roles) == {"master", "exp_avg", "exp_avg_sq"}
    assert pool.view("exp_avg", slots[3]).data_ptr() == pool.roles["exp_avg"][slots[3].offset:].data_ptr()
    h = ft_host.slot_list_hash(slots)
    assert h == pool.slot_hash() == ft_host.slot_list_hash(ft_host.make_slots(named, opt))
    other = ft_host.make_slots(list(reversed(named)), opt)
    assert ft_host.slot_list_hash(other) != h
    assert ft_host.needs_accumulator(1, "stale", 1.0) is False
    assert ft_host.needs_accumulator(2, "stale", 0.0) is True
    assert ft_host.needs_accumulator(1, "exact", 1.0) is True
    assert ft_host.needs_accumulator(1, "exact", 0.0) is False
    with pytest.raises(ValueError):
        ft_host.GradPipeline(slots, pool, opt, device="cpu", gradient_accumulation_steps=2)  # GA 2 needs acc
    with pytest.raises(ValueError, match="twice"):
        ft_host.make_slots(named + named[:1])
    for bad in (dict(piece_bytes=1000), dict(step_chunk=100), dict(acc_chunk=0), dict(clip_mode="global"),
                dict(ring_bytes=C.PIECE_BYTES + 256)):
        kw = dict(piece_bytes=C.PIECE_BYTES, ring_bytes=C.RING_BYTES, upload_ring_bytes=C.UPLOAD_RING_BYTES,
                  acc_chunk=C.CHUNK, step_chunk=C.CHUNK)
        kw.update(bad)
        with pytest.raises(ValueError):
            ft_host.GradPipeline(slots, pool, opt, device="cpu", gradient_accumulation_steps=1, inline=True, **kw)
