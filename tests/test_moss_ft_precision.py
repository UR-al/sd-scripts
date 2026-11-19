"""CPU tests of the fp32 master in bf16 training (docs/moss_full_finetune_design.md 1.5, section 5 rows of
test_moss_ft_precision.py): with a bf16 compute copy and an fp32 host master, AdamW steps of lr 1e-6 are kept; on bf16
parameters alone they round away; the compute copy is always the master rounded to nearest.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_precision.py -q -p no:cacheprovider
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _moss_ft_common as C  # noqa: E402

from library.moss import ft_host  # noqa: E402
from library.moss.ft_optim import HostAdamW  # noqa: E402

LR = 1e-6
STEPS = 100


@pytest.fixture(autouse=True)
def _no_mkldnn():
    with torch.backends.mkldnn.flags(enabled=False):  # oneDNN has no bf16 backward on some CPUs
        yield


def weights_and_grads(shape=(64, 64), seed=0):
    """Base-like weights (|w| >= 0.005: no element close enough to 0 for a 1e-6 step to show in bf16) and a fixed
    gradient, both exactly representable in bf16."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(shape, generator=g) * 0.02
    w = (w.sign() * w.abs().clamp(min=0.005)).to(torch.bfloat16).float()
    grad = (torch.randn(shape, generator=g) * 1e-3 + 2e-4).to(torch.bfloat16).float()
    return w, grad


def reference_movement(w0, grad):
    ref = torch.nn.Parameter(w0.clone())
    opt = torch.optim.AdamW([ref], lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=True)
    for _ in range(STEPS):
        ref.grad = grad.clone()
        opt.step()
    return ref.detach() - w0


def test_fp32_master_keeps_1e_6_updates():
    w0, grad = weights_and_grads()
    p = torch.nn.Parameter(w0.to(torch.bfloat16))
    opt = HostAdamW([p], lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    pipe, pool, slots = C.make_pipeline([("w", p)], opt, 1, inline=True)
    try:
        for step in range(STEPS):
            pipe.begin_window(1, ft_host.StepContext.snapshot(opt, step_index=step))
            pipe.begin_microstep(0)
            (p.float() * grad).sum().backward()  # dL/dp = grad, exactly (bf16-representable)
            stats = pipe.end_microstep()
            assert stats.skipped_params == 0
            assert torch.equal(p.detach(), pool.view("master", slots[0]).view_as(p).to(torch.bfloat16))
    finally:
        pipe.close()
    moved = pool.view("master", slots[0]).view_as(w0) - w0
    expected = reference_movement(w0, grad)
    assert expected.abs().min() > 0  # every element should move
    assert (moved - expected).norm() <= 1e-3 * expected.norm()


def test_pure_bf16_loses_them():
    """torch AdamW on the bf16 parameter itself: >= 90 % of the intended movement is lost to round-to-nearest (here all
    of it: lr 1e-6 is far below half a bf16 ulp of |w| >= 0.005). Documents why the fp32 master exists."""
    w0, grad = weights_and_grads()
    q = torch.nn.Parameter(w0.to(torch.bfloat16))
    opt = torch.optim.AdamW([q], lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    for _ in range(STEPS):
        q.grad = grad.to(torch.bfloat16)
        opt.step()
    lost = 1.0 - ((q.detach().float() - w0).norm() / reference_movement(w0, grad).norm()).item()
    assert lost >= 0.9


def test_compute_copy_is_rtn_of_master():
    """The real toy DiT in production precision (bf16 joint Linears, fp32 stream parameters, bf16 autocast), GA 2,
    threaded worker: after every optimizer step each compute copy == master rounded to its dtype."""
    model = C.build_model(seed=21, dtype=torch.bfloat16, stream_fp32=True)
    named = C.trainable(model)
    opt = HostAdamW(C.param_groups(named), lr=1e-3)
    pipe, pool, slots = C.make_pipeline(named, opt, 2)
    try:
        assert {s.compute_dtype for s in slots} == {torch.bfloat16, torch.float32}
        before = pool.roles["master"].clone()
        checked = []

        def after_window(w, stats):
            C.assert_compute_copies(pool, slots, f"window {w}")
            checked.append(w)

        C.train_pipeline(pipe, opt, C.model_windows(model, C.seeds([2] * 4), autocast=torch.bfloat16),
                         after_window=after_window)
        assert checked == [0, 1, 2, 3]
        assert not torch.equal(before, pool.roles["master"])
        assert float(pool.steps.min()) == 4.0
    finally:
        pipe.close()
