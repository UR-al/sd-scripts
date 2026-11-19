"""CPU check that the host pipeline's post-accumulate-grad hooks see every gradient exactly once, in each gradient
checkpointing mode of the MOSS DiT (docs/moss_full_finetune_design.md 1.4, kept as a test; section 5 row
test_moss_ft_hooks.py): none, plain (torch.utils.checkpoint), unsloth offload, the no-grad first-pass node
(set_ckpt_reuse(True, False)) and reuse-only (set_ckpt_reuse(False, True), attention on tests/_fake_flash.py).

Production precision: bf16 joint Linears, --moss_stream_fp32 (fp32 everywhere else), bf16 autocast. One micro-step of
a two-micro-step window, so the pipeline only accumulates: its fp32 accumulator must hold the gradients autograd gives a
hook-free copy of the model, bit for bit, and every hook must have fired once.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_hooks.py -q -p no:cacheprovider
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _moss_ft_common as C  # noqa: E402

from library import anima_models  # noqa: E402
from library.moss import ft_host  # noqa: E402
from library.moss import models as moss_models  # noqa: E402
from library.moss.ft_optim import HostAdamW  # noqa: E402
from tests import _fake_flash as fake_flash  # noqa: E402

# checkpoint calls per forward of the toy model (3 joint blocks, 2 text refiner blocks): plain, unsloth, no-grad node
EXPECTED_CALLS = {"none": (0, 0, 0), "plain": (5, 0, 0), "unsloth": (0, 5, 0), "nograd": (2, 0, 3), "reuse": (2, 0, 3)}


@pytest.fixture(autouse=True)
def _no_mkldnn():
    with torch.backends.mkldnn.flags(enabled=False):  # oneDNN has no bf16 backward on some CPUs
        yield


def spy_checkpoints(monkeypatch):
    seen = {"plain": 0, "unsloth": 0, "node": 0}
    real = (moss_models.torch_checkpoint, moss_models.unsloth_checkpoint, anima_models.nograd_checkpoint)

    def plain(*args, **kwargs):
        seen["plain"] += 1
        return real[0](*args, **kwargs)

    def unsloth(*args, **kwargs):
        seen["unsloth"] += 1
        return real[1](*args, **kwargs)

    def node(*args, **kwargs):
        seen["node"] += 1
        return real[2](*args, **kwargs)

    monkeypatch.setattr(moss_models, "torch_checkpoint", plain)
    monkeypatch.setattr(moss_models, "unsloth_checkpoint", unsloth)
    monkeypatch.setattr(anima_models, "nograd_checkpoint", node)
    return seen


@pytest.mark.parametrize("inline", [False, True], ids=["threaded", "inline"])
@pytest.mark.parametrize("ckpt", C.CKPT_MODES)
def test_hooks_fire_once_per_param(ckpt, inline, monkeypatch):
    C.set_ckpt_flags(monkeypatch, ckpt)
    attn_mode = "torch"
    if ckpt == "reuse":
        fake_flash.install(monkeypatch)
        attn_mode = "flash"
    seen = spy_checkpoints(monkeypatch)
    model = C.build_model(seed=11, dtype=torch.bfloat16, stream_fp32=True, ckpt=ckpt, attn_mode=attn_mode)
    ref = C.clone_model(model)
    named = C.trainable(model)
    opt = HostAdamW(C.param_groups(named), lr=1e-4)
    pipe, pool, slots = C.make_pipeline(named, opt, 2, inline=inline)
    try:
        assert {s.compute_dtype for s in slots} == {torch.bfloat16, torch.float32}
        batch = C.make_batch(4, dtype=torch.float32)
        pipe.begin_window(2, ft_host.StepContext.snapshot(opt))
        pipe.begin_microstep(0)
        fake_flash.CALLS["fwd"] = 0
        (C.loss_of(model, batch, autocast=torch.bfloat16) / 2).backward()
        flash_fwd = fake_flash.CALLS["fwd"]
        assert pipe.end_microstep() is None
        pipe.sync()
        assert (seen["plain"], seen["unsloth"], seen["node"]) == EXPECTED_CALLS[ckpt]
        if ckpt == "reuse":  # the joint blocks' recompute reused the stored attention: no second forward kernel
            assert flash_fwd == 3 + 2 * 2
        assert pipe.micro_hook_counts == [1] * len(slots)
        assert all(s.param.grad is None for s in slots)  # the hook freed every gradient

        (C.loss_of(ref, batch, autocast=torch.bfloat16) / 2).backward()
        ref_params = dict(ref.named_parameters())
        for s in slots:
            g = ref_params[s.name].grad
            assert g is not None and g.dtype == s.compute_dtype, s.name
            assert torch.equal(pool.view("acc", s), g.float().reshape(-1)), (
                f"{ckpt}: accumulated gradient of {s.name} differs from autograd's")
    finally:
        pipe.close()


def test_second_hook_call_in_one_backward_is_refused():
    """A parameter whose post-accumulate-grad hook fires twice in one micro-step would be counted twice: refused."""
    p = torch.nn.Parameter(torch.randn(256))
    opt = HostAdamW([p], lr=1e-3)
    pipe, pool, slots = C.make_pipeline([("p", p)], opt, 2, inline=True)
    try:
        pipe.begin_window(2, ft_host.StepContext.snapshot(opt))
        pipe.begin_microstep(0)
        (p * 2.0).sum().backward()
        with pytest.raises(RuntimeError, match="fired twice"):
            (p * 3.0).sum().backward()
    finally:
        pipe.close()


def test_gradient_outside_a_window_is_refused():
    p = torch.nn.Parameter(torch.randn(64))
    opt = HostAdamW([p], lr=1e-3)
    pipe, pool, slots = C.make_pipeline([("p", p)], opt, 1, inline=True)
    try:
        with pytest.raises(RuntimeError, match="outside a window"):
            p.sum().backward()
    finally:
        pipe.close()
