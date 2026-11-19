"""CPU tests of `came_host` (docs/moss_full_finetune_design.md 3.3, section 5 rows of test_moss_ft_came_bc.py):
library/moss/ft_optim.py's CAME cores (the paper algorithm as the user's came.py computes it, plus the opt-in
"read-out" bias correction), HostCAME's start-up rules, and HostCAME in the host pipeline.

* without the correction the cores are bit-identical to came.py's eager cores (_core_factored_full_fp32 /
  _core_unfactored_full_fp32) given the scalars as CAME.step passes them: fp32 0-dim tensors and the alphas 1 - beta
  formed in fp32 (skipped when LoraEasyCustomOptimizer is not importable);
* the correction removes the beta3 amplification of the first steps (the synthetic gradients of 3.3);
* it is state-compatible: same buffers, the second-moment EMAs untouched, identical to the uncorrected step once
  1 - beta^t rounds to 1;
* HostCAME refuses beta3 > 0.99 without the correction, AMSBound, Kahan summation and the grams strategies;
* the pipeline (per-parameter final step, accumulator or the GA = 1 scratch) equals came.py's CAME on fp32 parameters,
  and with the correction it equals the cores driven directly with each parameter's step count.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_came_bc.py -q -p no:cacheprovider
"""
import logging
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _moss_ft_common as C  # noqa: E402

from library.moss import ft_host  # noqa: E402
from library.moss.ft_optim import (HostCAME, came_bias_correction, came_factored_core, came_scalars,  # noqa: E402
                                   came_unfactored_core)

EPS1, EPS2 = 1e-30, 1e-16
DECAYS = {
    "decoupled": dict(weight_decay=0.1, weight_decouple=True),
    "fixed": dict(weight_decay=0.1, weight_decouple=True, fixed_decay=True),
    "coupled": dict(weight_decay=0.1, weight_decouple=False),
    "cautious_wd": dict(weight_decay=0.1, cautious_weight_decay=True),
}


def came_py():
    try:
        from LoraEasyCustomOptimizer.came import CAME
    except Exception as error:  # noqa: BLE001 - an editable install outside this repository
        pytest.skip(f"LoraEasyCustomOptimizer.came is not importable: {error}")
    return CAME


def flags(decay: str) -> dict:
    spec = dict(DECAYS[decay])
    spec.pop("weight_decay")
    return dict(dict(weight_decouple=True, fixed_decay=False, cautious_weight_decay=False), **spec)


@pytest.mark.parametrize("decay", list(DECAYS))
@pytest.mark.parametrize("strategy", ["unmodified", "cautious"])
def test_core_matches_came_py_without_correction(strategy, decay):
    CAME = came_py()
    torch.manual_seed(0)
    sc = came_scalars((0.9, 0.999, 0.9999), EPS1, EPS2, 1.0, 1e-3, DECAYS[decay]["weight_decay"])
    fl = flags(decay)
    cautious = strategy == "cautious"
    empty = torch.empty(0)

    # factored (2-D)
    shape = (24, 40)
    p0 = torch.randn(shape) * 0.1
    mine = [p0.clone(), torch.zeros(shape), torch.zeros(24), torch.zeros(40), torch.zeros(24), torch.zeros(40)]
    theirs = [t.clone() for t in mine]
    for _ in range(5):
        g = torch.randn(shape) + 0.2
        p, m, r, c, rr, rc = mine
        came_factored_core(g.clone(), m, r, c, rr, rc, p, **sc, **fl, cautious=cautious)
        p, m, r, c, rr, rc = theirs
        CAME._core_factored_full_fp32(g.clone(), m, r, c, rr, rc, empty, sc["beta1"], sc["beta2"], sc["beta3"],
                                      sc["eps1"], sc["eps2"], sc["clip_threshold"], False, sc["lr"],
                                      sc["weight_decay"], fl["weight_decouple"], fl["fixed_decay"],
                                      fl["cautious_weight_decay"], cautious, False, p, alphas=sc["alphas"])
    for name, a, b in zip(("p", "exp_avg", "sq_row", "sq_col", "res_row", "res_col"), mine, theirs):
        assert torch.equal(a, b), f"factored {name}: max |d| = {(a - b).abs().max().item():.3e}"

    # unfactored (1-D), with and without non_factored_confidence
    for nfc in (False, True):
        q0 = torch.randn(40) * 0.1
        mine = [q0.clone(), torch.zeros(40), torch.zeros(40), torch.zeros(40)]
        theirs = [t.clone() for t in mine]
        for _ in range(5):
            g = torch.randn(40) + 0.2
            p, m, v, res = mine
            came_unfactored_core(g.clone(), m, v, res if nfc else None, p, **sc, **fl, cautious=cautious,
                                 non_factored_confidence=nfc)
            p, m, v, res = theirs
            CAME._core_unfactored_full_fp32(g.clone(), m, v, res if nfc else empty, empty, sc["beta1"], sc["beta2"],
                                            sc["beta3"], sc["eps1"], sc["eps2"], sc["clip_threshold"], False, nfc,
                                            sc["lr"], sc["weight_decay"], fl["weight_decouple"], fl["fixed_decay"],
                                            fl["cautious_weight_decay"], cautious, False, p, alphas=sc["alphas"])
        for name, a, b in zip(("p", "exp_avg", "exp_avg_sq", "exp_avg_res"), mine, theirs):
            assert torch.equal(a, b), f"unfactored nfc={nfc} {name}: max |d| = {(a - b).abs().max().item():.3e}"


def factored_state(shape):
    rows, cols = shape
    return [torch.zeros(shape), torch.zeros(rows), torch.zeros(cols), torch.zeros(rows), torch.zeros(cols)]


def update_profile(bias_correction: bool, beta3: float = 0.999, n: int = 64, steps: int = 3000, seed: int = 0):
    """RMS(update)/lr per step over the mean of the last 2,000 steps, for 3.3's synthetic gradients
    (0.2 * mu + N(0, 1), mu ~ N(0, 1) fixed)."""
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(n, n, generator=g)
    lr = 1e-3
    sc = came_scalars((0.9, 0.999, beta3), EPS1, EPS2, 1.0, lr, 0.0)
    m, r, c, rr, rc = factored_state((n, n))
    p = torch.zeros(n, n)
    rms = []
    for t in range(1, steps + 1):
        grad = 0.2 * mu + torch.randn(n, n, generator=g)
        p.zero_()  # p after the step = -lr * update exactly (no weight decay)
        bc = came_bias_correction(sc["beta1"], sc["beta2"], sc["beta3"], t) if bias_correction else None
        came_factored_core(grad, m, r, c, rr, rc, p, **sc, bias_correction=bc)
        rms.append(p.norm().item() / n / lr)
    late = sum(rms[-2000:]) / 2000
    return [x / late for x in rms]


@pytest.fixture(scope="module")
def adamw_reference():
    return adamw_profile()


def adamw_profile(n: int = 64, steps: int = 3000, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(n, n, generator=g)
    lr = 1e-3
    p = torch.nn.Parameter(torch.zeros(n, n))
    opt = torch.optim.AdamW([p], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    rms = []
    for _ in range(steps):
        p.grad = 0.2 * mu + torch.randn(n, n, generator=g)
        with torch.no_grad():
            p.zero_()
        opt.step()
        rms.append(p.detach().norm().item() / n / lr)
    late = sum(rms[-2000:]) / 2000
    return [x / late for x in rms]


@pytest.mark.parametrize("beta3", [0.999, 0.9999])
def test_bias_correction_removes_amplification(beta3, adamw_reference):
    """64x64, T 3,000 (design section 5: beta3 0.999; 0.9999 too, the failing default of 3.3, where bc2 != bc3)."""
    corrected = update_profile(True, beta3=beta3)
    uncorrected = update_profile(False, beta3=beta3)
    adamw = adamw_reference
    assert corrected[0] <= 4.0 and corrected[23] <= 1.2, (corrected[0], corrected[23])
    assert uncorrected[0] >= 8.0, uncorrected[0]
    # the corrected profile is AdamW's (3.3: x3.4 at t1, x1.0 from t24)
    assert abs(corrected[0] - adamw[0]) <= 0.15 * adamw[0], (corrected[0], adamw[0])
    assert abs(corrected[23] - adamw[23]) <= 0.15, (corrected[23], adamw[23])


def test_correction_is_state_compatible():
    torch.manual_seed(1)
    shape, steps = (16, 24), 20
    sc = came_scalars((0.9, 0.999, 0.99), EPS1, EPS2, 1.0, 1e-3, 0.1)
    grads = [torch.randn(shape) + 0.3 for _ in range(steps)]
    vgrads = [torch.randn(24) + 0.3 for _ in range(steps)]
    runs = {}
    for corrected in (False, True):
        p = torch.ones(shape) * 0.05
        st = factored_state(shape)
        q = torch.ones(24) * 0.05
        vst = [torch.zeros(24), torch.zeros(24), torch.zeros(24)]
        for t in range(1, steps + 1):
            bc = came_bias_correction(sc["beta1"], sc["beta2"], sc["beta3"], t) if corrected else None
            came_factored_core(grads[t - 1].clone(), *st, p, **sc, bias_correction=bc)
            came_unfactored_core(vgrads[t - 1].clone(), vst[0], vst[1], vst[2], q, **sc, bias_correction=bc,
                                 non_factored_confidence=True)
        runs[corrected] = (p, st, q, vst)
    (p0, st0, q0, v0), (p1, st1, q1, v1) = runs[False], runs[True]
    # the same buffers (shapes, dtypes); the second-moment EMAs see only the gradients: bit-identical
    for a, b in zip(st0 + v0, st1 + v1):
        assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(st0[1], st1[1]) and torch.equal(st0[2], st1[2])  # exp_avg_sq_row / _col
    assert torch.equal(v0[1], v1[1])  # exp_avg_sq of the 1-D parameter
    # the corrected run moved differently in these first steps (that is the point of it)
    assert not torch.equal(p0, p1)

    # once 1 - beta^t rounds to 1 in fp32, a corrected step IS the uncorrected step: a state from either run can be
    # continued by either optimizer
    t = 10 ** 6
    assert came_bias_correction(sc["beta1"], sc["beta2"], sc["beta3"], t) == (1.0, 1.0, 1.0)
    g = torch.randn(shape)
    vg = torch.randn(24)
    outs = []
    for corrected in (False, True):
        p, st, q, vst = (x.clone() if torch.is_tensor(x) else [y.clone() for y in x] for x in runs[True])
        bc = came_bias_correction(sc["beta1"], sc["beta2"], sc["beta3"], t) if corrected else None
        came_factored_core(g.clone(), *st, p, **sc, bias_correction=bc)
        came_unfactored_core(vg.clone(), vst[0], vst[1], vst[2], q, **sc, bias_correction=bc,
                             non_factored_confidence=True)
        outs.append([p, *st, q, *vst])
    for a, b in zip(*outs):
        assert torch.equal(a, b)

    # HostCAME keeps the same state tensors with and without the correction
    layouts = []
    for corrected in (False, True):
        w = torch.nn.Parameter(torch.randn(shape))
        bvec = torch.nn.Parameter(torch.randn(24))
        opt = HostCAME([w, bvec], lr=1e-3, bias_correction=corrected, non_factored_confidence=True)
        slots = ft_host.make_slots([("w", w), ("b", bvec)], opt)
        pool = ft_host.HostPool(slots, HostCAME.POOL_ROLES, with_acc=False)
        opt.bind(slots, pool)
        layouts.append({k: (tuple(v.shape), v.dtype) for k, v in opt.extra_state().items()})
    assert layouts[0] == layouts[1]
    assert set(layouts[0]) == {"w.exp_avg_sq_row", "w.exp_avg_sq_col", "w.exp_avg_res_row", "w.exp_avg_res_col",
                               "b.exp_avg_sq", "b.exp_avg_res"}
    assert HostCAME.POOL_ROLES == ("master", "exp_avg")


def test_came_state_survives_rebind():
    """Statistics loaded into HostCAME before the pipeline exists (a resume) are kept when the pipeline binds it."""
    w = torch.nn.Parameter(torch.randn(8, 12))
    b = torch.nn.Parameter(torch.randn(12))
    opt = HostCAME([w, b], lr=1e-3, non_factored_confidence=True)
    slots = ft_host.make_slots([("w", w), ("b", b)], opt)
    pool = ft_host.HostPool(slots, HostCAME.POOL_ROLES, with_acc=False)
    pool.copy_master_from_params()
    opt.bind(slots, pool)
    for i, tensor in enumerate(opt.extra_state().values()):
        tensor.fill_(i + 1.0)
    saved = {k: v.clone() for k, v in opt.extra_state().items()}
    pipe = ft_host.GradPipeline(slots, pool, opt, device="cpu", gradient_accumulation_steps=1, inline=True,
                                piece_bytes=C.PIECE_BYTES, ring_bytes=C.RING_BYTES,
                                upload_ring_bytes=C.UPLOAD_RING_BYTES, acc_chunk=C.CHUNK, step_chunk=C.CHUNK)
    try:
        for key, value in opt.extra_state().items():
            assert torch.equal(value, saved[key]), key
    finally:
        pipe.close()


def test_refuses_high_beta3_without_correction(caplog):
    p = torch.nn.Parameter(torch.randn(8, 8))
    with pytest.raises(ValueError, match="bias_correction"):
        HostCAME([p], lr=1e-5, betas=(0.9, 0.999, 0.9999))
    HostCAME([p], lr=1e-5, betas=(0.9, 0.999, 0.9999), bias_correction=True)
    HostCAME([p], lr=1e-5, betas=(0.9, 0.999, 0.99))  # 0.99 without the correction is allowed
    assert HostCAME([p], lr=1e-5).defaults["betas"] == (0.9, 0.999, 0.99)
    for bad in (dict(ams_bound=True), dict(kahan_sum=True), dict(update_strategy="grams"),
                dict(update_strategy="both"), dict(betas=(0.9, 0.999)), dict(not_a_came_key=1)):
        with pytest.raises(ValueError):
            HostCAME([p], lr=1e-5, **bad)
    with caplog.at_level(logging.WARNING):
        HostCAME([p], lr=1e-5, state_storage_device="cpu", state_storage_dtype="float32", compile_step=True,
                 foreach=True, sync_chunk_size=64, group_shapes=True)
    warned = [r for r in caplog.records if "ignored" in r.getMessage()]
    assert len(warned) == 1 and "compile_step" in warned[0].getMessage()


@pytest.fixture
def came_py_on_cpu(monkeypatch):
    """came.py's CAME assumes a CUDA build with a visible GPU even for CPU parameters (pin_memory, the current CUDA
    device as compute device, a final synchronize): pointed at the CPU here (as tests/test_moss_ckpt_reuse.py does)."""
    CAME = came_py()
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: self)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    return CAME


CAME_ARGS = dict(lr=1e-3, betas=(0.9, 0.999, 0.99), clip_threshold=1.0, eps1=EPS1, eps2=EPS2,
                 update_strategy="cautious", non_factored_confidence=True)


@pytest.mark.parametrize("inline", [False, True], ids=["threaded", "inline"])
@pytest.mark.parametrize("ga", [1, 2])
def test_came_host_pipeline_equals_came_py(ga, inline, came_py_on_cpu):
    CAME = came_py_on_cpu
    model = C.build_model(seed=31)
    ref = C.clone_model(model)
    named, ref_named = C.trainable(model), C.trainable(ref)
    opt = HostCAME(C.param_groups(named, 0.1), **CAME_ARGS)
    ref_opt = CAME(C.param_groups(ref_named, 0.1), state_storage_device="cpu", state_storage_dtype="float32",
                   **CAME_ARGS)
    pipe, pool, slots = C.make_pipeline(named, opt, ga, inline=inline)
    try:
        assert (pool.acc is None) == (ga == 1)  # GA 1: the per-parameter step assembles each gradient in a scratch
        seeds = C.seeds([ga] * 3)
        C.train_pipeline(pipe, opt, C.model_windows(model, seeds))
        C.train_reference(ref_opt, [p for _, p in ref_named], C.model_windows(ref, seeds))
    finally:
        pipe.close()
    extra = opt.extra_state()
    refs = dict(ref_named)
    for s in slots:
        p = refs[s.name]
        state = ref_opt.state[p]
        assert torch.equal(pool.view("master", s), p.detach().reshape(-1)), s.name
        assert torch.equal(s.param.detach(), p.detach()), s.name
        assert torch.equal(pool.view("exp_avg", s), state["exp_avg"].reshape(-1)), s.name
        keys = ("exp_avg_sq_row", "exp_avg_sq_col", "exp_avg_res_row", "exp_avg_res_col") if len(s.shape) >= 2 \
            else ("exp_avg_sq", "exp_avg_res")
        for key in keys:
            assert torch.equal(extra[f"{s.name}.{key}"], state[key]), (s.name, key)
        assert float(pool.steps[s.index]) == 3.0


def test_came_host_bias_correction_in_pipeline():
    """HostCAME(bias_correction=True) in the pipeline == the cores driven directly, each parameter with its own step
    count t: an extra parameter gets no gradient in window 1, so its t lags the others'."""
    betas = (0.9, 0.999, 0.9999)
    model = C.build_model(seed=41)
    ref = C.clone_model(model)
    e0 = torch.randn(24, 40, generator=torch.Generator().manual_seed(5)) * 0.1
    extra, ref_extra = torch.nn.Parameter(e0.clone()), torch.nn.Parameter(e0.clone())
    named, ref_named = C.trainable(model) + [("extra", extra)], C.trainable(ref) + [("extra", ref_extra)]
    k = torch.linspace(-1.0, 1.0, 960).view(24, 40)

    def term(param):
        def fn(w, j):
            return None if w == 1 else (param * k).square().sum() * 1e-2
        return fn

    opt = HostCAME(C.param_groups(named, 0.1), lr=2e-3, betas=betas, bias_correction=True,
                   non_factored_confidence=True)
    pipe, pool, slots = C.make_pipeline(named, opt, 2)
    try:
        stats = C.train_pipeline(pipe, opt, C.model_windows(model, C.seeds([2] * 3), extra=term(extra)))
    finally:
        pipe.close()

    # reference: autograd accumulation, then the cores per parameter with t = that parameter's step count
    groups = C.param_groups(ref_named, 0.1)
    wd_of = {id(p): g["weight_decay"] for g in groups for p in g["params"]}
    state, steps = {}, {}
    for w, micro in enumerate(C.model_windows(ref, C.seeds([2] * 3), extra=term(ref_extra))):
        for fn in micro:
            (fn() / len(micro)).backward()
        for name, p in ref_named:
            if p.grad is None:
                continue
            sc = came_scalars(betas, EPS1, EPS2, 1.0, 2e-3, wd_of[id(p)])
            steps[name] = steps.get(name, 0) + 1
            bc = came_bias_correction(sc["beta1"], sc["beta2"], sc["beta3"], steps[name])
            with torch.no_grad():
                if p.dim() >= 2:
                    st = state.setdefault(name, factored_state(tuple(p.shape)))
                    came_factored_core(p.grad, *st, p.data, **sc, bias_correction=bc)
                else:
                    st = state.setdefault(name, [torch.zeros_like(p), torch.zeros_like(p), torch.zeros_like(p)])
                    came_unfactored_core(p.grad, st[0], st[1], st[2], p.data, **sc, bias_correction=bc,
                                         non_factored_confidence=True)
            p.grad = None
    refs = dict(ref_named)
    for s in slots:
        assert torch.equal(pool.view("master", s), refs[s.name].detach().reshape(-1)), s.name
        assert torch.equal(pool.view("exp_avg", s), state[s.name][0].reshape(-1)), s.name
        assert float(pool.steps[s.index]) == steps[s.name], s.name
    assert steps["extra"] == 2 and steps["layers.0.proj.q.weight"] == 3
    assert all(st.skipped_params == 0 for st in stats)
