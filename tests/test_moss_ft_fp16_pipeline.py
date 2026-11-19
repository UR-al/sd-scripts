"""CPU tests of the loss scale inside the host optimizer pipeline (library/moss/ft_host.py, ft_scaler.py;
docs/moss_ft_te_live_fp16_plan.md 2.3).

* Exactness: the toy DiT in fp32, the loss multiplied by a power-of-two scale (a DynamicLossScaler starting at 2^10 and
  doubling every 2 clean windows) and StepContext.inv_grad_scale = 1 / scale: masters, moments, steps, compute copies,
  grad_norm and clip_coef equal the unscaled run's bit for bit - GA 1 / 3, clipping none / stale / exact, adamw_host
  and came_host, inline and threaded.
* Overflow: an inf in one parameter's gradient - "partial" steps every other parameter exactly as the unscaled run with
  the same inf does and the scale halves; "window" (exact clipping) steps nothing in that window.
* fp16 production precision: fp16 joint Linears + the fp32 stream under fp16 autocast with a scaler: the compute
  copies stay the master rounded to fp16.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_fp16_pipeline.py -q -p no:cacheprovider
"""
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _moss_ft_common as C  # noqa: E402

from library.moss import ft_host, ft_scaler  # noqa: E402
from library.moss.ft_optim import HostAdamW, HostCAME  # noqa: E402

TARGET = "layers.2.proj.q.weight"


@pytest.mark.parametrize("inline", [True, False])
@pytest.mark.parametrize("clip", ["none", "stale", "exact"])
@pytest.mark.parametrize("kind", ["adamw", "came"])
def test_finite_accumulated_gradients_can_exceed_fp32_squared_norm(kind, clip, inline):
    """Each micro norm fits fp32; the accumulated squared norm does not. Unscale exactly once."""
    param = torch.nn.Parameter(torch.ones(4))
    named = [("weight", param)]
    opt = make_optimizer(kind, named)
    kw = {} if clip == "none" else dict(max_grad_norm=0.05, clip_mode=clip)
    pipe, pool, slots = C.make_pipeline(named, opt, 2, inline=inline, **kw)
    scaler = ft_scaler.DynamicLossScaler(init_scale=2.0 ** 24)
    try:
        stats, _ = train([[lambda: param.sum() * (1e19 / 2.0 ** 24)] * 2], pipe, opt, scaler)
        st = stats[0]
        assert st.stepped_params == 1 and not st.overflow
        assert st.grad_norm == pytest.approx(2e19 / 2.0 ** 24, rel=1e-6)
        if clip != "none":
            assert st.clip_coef == pytest.approx(0.05 / (2e19 / 2.0 ** 24), rel=1e-6, abs=0)
        assert all(torch.isfinite(t).all() for t in pool.roles.values())
        assert not torch.equal(param.detach(), torch.ones(4))
        C.assert_compute_copies(pool, slots)
    finally:
        pipe.close()


def make_optimizer(kind, named):
    if kind == "adamw":
        return HostAdamW(C.param_groups(named, 0.1), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    return HostCAME(C.param_groups(named, 0.1), lr=1e-3, betas=(0.9, 0.999, 0.99), bias_correction=True,
                    non_factored_confidence=True)


def train(windows, pipe, opt, scaler=None, after_window=None):
    """moss_train._train_loop's order: the window's StepContext with 1 / scale, (loss / n x scale).backward(), the
    scaler updated from the window's stats."""
    stats, scales = [], []
    for w, micro in enumerate(windows):
        n = len(micro)
        inv = 1.0 if scaler is None else scaler.inv_scale
        pipe.begin_window(n, ft_host.StepContext.snapshot(opt, step_index=w, inv_grad_scale=inv))
        result = None
        for j, fn in enumerate(micro):
            pipe.begin_microstep(j)
            loss = fn() / n
            (loss if scaler is None else loss * scaler.scale).backward()
            result = pipe.end_microstep()
        stats.append(result)
        scales.append(result.loss_scale)
        if scaler is not None:
            scaler.update(result, True)
        if after_window is not None:
            after_window(w, result)
    return stats, scales


def run(kind, ga, clip, inline, scaled, windows=5, inject=None, policy="partial"):
    model = C.build_model(seed=7)
    named = C.trainable(model)
    opt = make_optimizer(kind, named)
    kw = {}
    if clip != "none":
        kw = dict(max_grad_norm=0.05, clip_mode=clip)
    pipe, pool, slots = C.make_pipeline(named, opt, ga, inline=inline, overflow_policy=policy, **kw)
    scaler = ft_scaler.DynamicLossScaler(init_scale=2.0 ** 10, growth_interval=2) if scaled else None
    extra = None
    if inject is not None:
        extra = inject(dict(named)[TARGET])
    try:
        stats, scales = train(C.model_windows(model, C.seeds([ga] * windows), extra=extra), pipe, opt, scaler)
    finally:
        pipe.close()
    return dict(model=model, opt=opt, pool=pool, slots=slots, stats=stats, scales=scales, scaler=scaler)


def assert_same(a, b):
    for role in a["pool"].roles:
        assert torch.equal(a["pool"].roles[role], b["pool"].roles[role]), role
    assert torch.equal(a["pool"].steps, b["pool"].steps)
    for (name, p), (_, q) in zip(a["model"].named_parameters(), b["model"].named_parameters()):
        assert torch.equal(p.detach(), q.detach()), name
    if hasattr(a["opt"], "extra_state"):
        ea, eb = a["opt"].extra_state(), b["opt"].extra_state()
        assert ea.keys() == eb.keys() and all(torch.equal(ea[k], eb[k]) for k in ea)
    for sa, sb in zip(a["stats"], b["stats"]):
        assert (sa.clip_mode, sa.skipped_params, sa.stepped_params) == (sb.clip_mode, sb.skipped_params,
                                                                        sb.stepped_params)
        assert sa.grad_norm == sb.grad_norm or (math.isnan(sa.grad_norm) and math.isnan(sb.grad_norm))
        assert sa.clip_coef == sb.clip_coef


@pytest.mark.parametrize("inline", [True, False], ids=["inline", "threaded"])
@pytest.mark.parametrize("clip", ["none", "stale", "exact"])
@pytest.mark.parametrize("ga", [1, 3])
@pytest.mark.parametrize("kind", ["adamw", "came"])
def test_scaled_run_equals_unscaled_bitwise(kind, ga, clip, inline):
    plain = run(kind, ga, clip, inline, scaled=False)
    scaled = run(kind, ga, clip, inline, scaled=True)
    assert_same(plain, scaled)
    assert plain["scales"] == [1.0] * 5
    assert scaled["scales"] == [1024.0, 1024.0, 2048.0, 2048.0, 4096.0]  # growth every 2 clean windows
    assert scaled["scaler"].scale == 4096.0 and scaled["scaler"].overflow_windows == 0
    assert all(math.isfinite(st.grad_norm) and st.grad_norm > 0 for st in scaled["stats"] if st.clip_mode != "measure")
    if clip != "none":
        assert any(st.clip_coef < 1.0 for st in scaled["stats"])  # clipping happened, with the unscaled norms
    C.assert_compute_copies(scaled["pool"], scaled["slots"])


def poison_window(index, value=float("inf")):
    """An extra=... factory: in window `index`, micro-step 0, one element of the target's gradient becomes `value`."""
    def factory(param):
        handles = []

        def extra(w, j):
            for h in handles:
                h.remove()
            handles.clear()
            if w == index and j == 0:
                def poison(g):
                    g = g.clone()
                    g.view(-1)[3] = value
                    return g
                handles.append(param.register_hook(poison))
            return None
        return extra
    return factory


@pytest.mark.parametrize("inline", [True, False], ids=["inline", "threaded"])
@pytest.mark.parametrize("ga", [1, 3])
def test_partial_overflow_steps_the_rest_and_halves_the_scale(ga, inline):
    plain = run("adamw", ga, "none", inline, scaled=False, inject=poison_window(1))
    scaled = run("adamw", ga, "none", inline, scaled=True, inject=poison_window(1))
    assert_same(plain, scaled)  # the finite slots took exactly the unscaled steps; the target kept its state
    st = scaled["stats"][1]
    assert st.overflow and st.skipped_params == 1 and st.skipped_names == [TARGET]
    assert st.stepped_params == len(scaled["slots"]) - 1 and math.isnan(st.grad_norm)
    # 1024 (overflow) -> 512, two clean -> 1024, ...
    assert scaled["scales"] == [1024.0, 1024.0, 512.0, 512.0, 1024.0]
    assert scaled["scaler"].overflow_windows == 1 and scaled["scaler"].overflow_streak == 0
    slot = next(s for s in scaled["slots"] if s.name == TARGET)
    assert float(scaled["pool"].steps[slot.index]) == 4.0 and float(scaled["pool"].steps.max()) == 5.0
    C.assert_compute_copies(scaled["pool"], scaled["slots"])


@pytest.mark.parametrize("inline", [True, False], ids=["inline", "threaded"])
def test_window_policy_skips_the_whole_window(inline):
    model = C.build_model(seed=7)
    named = C.trainable(model)
    opt = make_optimizer("adamw", named)
    pipe, pool, slots = C.make_pipeline(named, opt, 2, inline=inline, max_grad_norm=0.05, clip_mode="exact",
                                        overflow_policy="window")
    scaler = ft_scaler.DynamicLossScaler(init_scale=2.0 ** 10, growth_interval=2)
    snap = {}

    def after_window(w, st):
        if w == 0:
            snap.update({role: t.clone() for role, t in pool.roles.items()}, steps=pool.steps.clone(),
                        params=[p.detach().clone() for _, p in named])
        if w == 1:
            for role, t in pool.roles.items():
                assert torch.equal(t, snap[role]), role  # nothing stepped
            assert torch.equal(pool.steps, snap["steps"])
            assert all(torch.equal(p.detach(), q) for (_, p), q in zip(named, snap["params"]))

    try:
        stats, scales = train(C.model_windows(model, C.seeds([2] * 3), extra=poison_window(1)(dict(named)[TARGET])),
                              pipe, opt, scaler, after_window=after_window)
    finally:
        pipe.close()
    st = stats[1]
    assert st.overflow and st.stepped_params == 0 and st.skipped_params == len(slots) and math.isnan(st.grad_norm)
    assert scales == [1024.0, 1024.0, 512.0]
    assert stats[2].stepped_params == len(slots) and math.isfinite(stats[2].grad_norm)
    assert float(pool.steps.max()) == 2.0


def test_window_policy_needs_steps_after_the_backward():
    model = C.build_model(seed=7)
    named = C.trainable(model)
    opt = make_optimizer("adamw", named)
    for kw in (dict(), dict(max_grad_norm=0.05, clip_mode="stale"), dict(clip_mode="exact")):
        with pytest.raises(ValueError, match="overflow_policy 'window'"):
            C.make_pipeline(named, opt, 2, inline=True, overflow_policy="window", **kw)
    with pytest.raises(ValueError, match="overflow_policy must be"):
        C.make_pipeline(named, opt, 2, inline=True, overflow_policy="skip")
    pipe, _, _ = C.make_pipeline(named, opt, 2, inline=True, overflow_policy="window", overlap=False)
    pipe.close()


def test_inv_grad_scale_must_be_a_power_of_two():
    model = C.build_model(seed=7)
    named = C.trainable(model)
    opt = make_optimizer("adamw", named)
    pipe, _, _ = C.make_pipeline(named, opt, 1, inline=True)
    try:
        for bad in (0.0, -0.5, 0.3, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="power of two"):
                pipe.begin_window(1, ft_host.StepContext.snapshot(opt, 0, inv_grad_scale=bad))
    finally:
        pipe.close()


def test_combine_is_exact():
    c = torch.tensor(0.0123456789, dtype=torch.float32)
    assert ft_host._combine(None, 1.0) is None and ft_host._combine(c, 1.0) is c
    assert float(ft_host._combine(None, 2.0 ** -10)) == 2.0 ** -10
    t = torch.randn(4096, generator=torch.Generator().manual_seed(1))
    for s in (2.0 ** -1, 2.0 ** -10, 2.0 ** -24):
        combined = ft_host._combine(c, s)
        assert float(combined) == float(c) * s
        assert torch.equal((t / s) * combined, t * c)  # round(t * c * s / s) == round(t * c)
        assert float(ft_host._combine(0.5, s)) == 0.5 * s


@pytest.fixture
def _no_mkldnn():
    with torch.backends.mkldnn.flags(enabled=False):
        yield


def test_fp16_compute_copies_follow_the_master(_no_mkldnn):
    """Production fp16 precision on the CPU: fp16 joint Linears, the fp32 stream, fp16 autocast, the loss scaled by a
    dynamic scaler, GA 2, threaded worker: after every window each compute copy == master rounded to its dtype."""
    model = C.build_model(seed=21, dtype=torch.float16, stream_fp32=True)
    named = C.trainable(model)
    opt = HostAdamW(C.param_groups(named), lr=1e-3)
    pipe, pool, slots = C.make_pipeline(named, opt, 2)
    scaler = ft_scaler.DynamicLossScaler(init_scale=2.0 ** 8, growth_interval=2)
    try:
        assert {s.compute_dtype for s in slots} == {torch.float16, torch.float32}
        before = pool.roles["master"].clone()
        checked = []

        def after_window(w, st):
            C.assert_compute_copies(pool, slots, f"window {w}")
            checked.append(st.loss_scale)

        stats, _ = train(C.model_windows(model, C.seeds([2] * 4), autocast=torch.float16), pipe, opt, scaler,
                         after_window=after_window)
        assert len(checked) == 4 and not torch.equal(before, pool.roles["master"])
        assert all(st.skipped_params == 0 and math.isfinite(st.grad_norm) for st in stats)
        assert checked == [256.0, 256.0, 512.0, 512.0]
    finally:
        pipe.close()


@pytest.mark.parametrize("inline", [True, False], ids=["inline", "threaded"])
@pytest.mark.parametrize("scaled", [True, False], ids=["fp16", "unscaled"])
def test_exact_clipping_skips_the_whole_window_under_partial(inline, scaled):
    """Review fix: with exact clipping every step runs after the drain, so "partial" (the default) skips the whole
    window once a slot is non-finite instead of clipping the finite slots with the norm of the finite slots alone (the
    true norm is not finite). Same for an unscaled (bf16) run."""
    model = C.build_model(seed=7)
    named = C.trainable(model)
    opt = make_optimizer("adamw", named)
    pipe, pool, slots = C.make_pipeline(named, opt, 2, inline=inline, max_grad_norm=0.05, clip_mode="exact")
    scaler = ft_scaler.DynamicLossScaler(init_scale=2.0 ** 10, growth_interval=2) if scaled else None
    snap = {}

    def after_window(w, st):
        if w == 0:
            snap.update({role: t.clone() for role, t in pool.roles.items()}, steps=pool.steps.clone(),
                        params=[p.detach().clone() for _, p in named])
        if w == 1:
            for role, t in pool.roles.items():
                assert torch.equal(t, snap[role]), role  # nothing stepped
            assert torch.equal(pool.steps, snap["steps"])
            assert all(torch.equal(p.detach(), q) for (_, p), q in zip(named, snap["params"]))

    try:
        stats, scales = train(C.model_windows(model, C.seeds([2] * 3), extra=poison_window(1)(dict(named)[TARGET])),
                              pipe, opt, scaler, after_window=after_window)
    finally:
        pipe.close()
    st = stats[1]
    assert st.overflow and st.stepped_params == 0 and st.skipped_params == len(slots) and math.isnan(st.grad_norm)
    assert stats[2].stepped_params == len(slots) and math.isfinite(stats[2].grad_norm)
    assert float(pool.steps.max()) == 2.0
    if scaled:
        assert scales == [1024.0, 1024.0, 512.0]


@pytest.mark.parametrize("inline", [True, False], ids=["inline", "threaded"])
@pytest.mark.parametrize("ga", [1, 2])
def test_fp16_master_past_the_range_stops_the_run(_no_mkldnn, inline, ga):
    """Review fix: a master past the fp16 range (|w| >= 65520) is not cast into an inf compute copy silently; the step
    raises HostPipelineError naming the parameter (the in-backward step at GA 1 and the after-backward one at GA 2)."""
    model = C.build_model(seed=21, dtype=torch.float16, stream_fp32=True)
    named = C.trainable(model)
    opt = HostAdamW(C.param_groups(named), lr=1e-3)
    pipe, pool, slots = C.make_pipeline(named, opt, ga, inline=inline)
    target = next(s for s in slots if s.compute_dtype == torch.float16)
    pool.view("master", target)[5] = 70000.0  # what a diverged run's master would hold
    try:
        with pytest.raises(ft_host.HostPipelineError, match=rf"{target.name}: the fp32 master left the fp16 range"):
            train(C.model_windows(model, C.seeds([ga] * 2), autocast=torch.float16), pipe, opt,
                  ft_scaler.DynamicLossScaler(init_scale=2.0 ** 8, growth_interval=2))
        assert torch.isfinite(target.param).all()  # nothing inf was uploaded
    finally:
        pipe.close()
