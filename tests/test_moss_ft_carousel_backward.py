"""CPU tests of the Carousel backward (docs/moss_ft_carousel_plan.md 6, items 19-22): only the active tensors get
gradients (equal to the same model's gradients with everything trainable), nothing below the lowest active block is
recomputed or backpropagated (no checkpoint, plain checkpoint, unsloth offload), the unsloth floor block keeps its
weight gradients, and the text path runs without grad when it is not trained.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_carousel_backward.py -q -p no:cacheprovider
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

from _moss_ft_carousel_common import build_model, carousel_loss, clone, make_engine
from library.moss import ft_host
from library.moss import models as moss_models
from library.moss.models import MossBlock, MossTextBlock


def one_micro_step(engine, seed=0, step=0):
    """begin_window + one micro-step's backward, without the step (end_microstep of a 2-micro-step window)."""
    engine.begin_window(2, ft_host.StepContext(groups=(), step_index=step))
    engine.begin_microstep(0)
    carousel_loss(engine.model, seed).backward()
    assert engine.end_microstep() is None


def acc_of(engine, i):
    group, off, n = engine._loc[i]
    return engine._dev[group]["acc"][off:off + n].view(engine.slots[i].shape)


def test_only_active_tensors_get_gradients_equal_to_full_backward():
    model = build_model(ckpt="none")
    full = clone(model)
    engine = make_engine(model, "adamw", k=2, period=1000, select="window", always="small")
    one_micro_step(engine)
    active = set(engine.active_slots)
    assert active and len(active) < len(engine.slots)
    for slot in engine.slots:
        assert slot.param.grad is None, slot.name  # moved into the accumulator by the hook
        assert slot.param.requires_grad == (slot.index in active), slot.name
    assert engine._filled == active
    carousel_loss(full, 0).backward()
    grads = dict(full.named_parameters())
    for i in active:
        assert torch.equal(acc_of(engine, i), grads[engine.slots[i].name].grad), engine.slots[i].name


class Counter:
    """Calls of each block's _forward with / without grad, and unsloth checkpoint calls per block."""

    def __init__(self, monkeypatch):
        self.calls = {}
        self.unsloth = {}
        self.outputs = {}
        for cls in (MossBlock, MossTextBlock):
            original = cls._forward
            counter = self

            def _forward(block, *args, _original=original, **kwargs):
                key = counter.key(block)
                grad = "grad" if torch.is_grad_enabled() else "no_grad"
                counter.calls.setdefault(key, []).append(grad)
                out = _original(block, *args, **kwargs)
                counter.outputs[key] = out
                return out

            monkeypatch.setattr(cls, "_forward", _forward)
        real = moss_models.unsloth_checkpoint

        def unsloth(function, *args):
            key = self.key(function.__self__)
            self.unsloth[key] = self.unsloth.get(key, 0) + 1
            return real(function, *args)

        monkeypatch.setattr(moss_models, "unsloth_checkpoint", unsloth)
        self.model = None

    def key(self, block):
        for i, layer in enumerate(self.model.layers):
            if layer is block:
                return f"layers.{i}"
        for i, layer in enumerate(self.model.text_embedder.blocks):
            if layer is block:
                return f"refiner.{i}"
        return "?"


@pytest.mark.parametrize("ckpt", ["none", "plain", "unsloth"])
def test_nothing_below_the_floor_is_recomputed_or_backpropagated(monkeypatch, ckpt):
    model = build_model(ckpt=ckpt)
    counter = Counter(monkeypatch)
    counter.model = model
    # always head, window K = 1: units 0, 1 are the refiners, unit 3 = layers.1 -> floor 1
    engine = make_engine(model, "adamw", k=1, period=1000, select="window", always="head")
    engine.activate(0)
    engine.switch((3,))
    assert engine.floor == 1 and not engine.text_grad
    inputs = {}
    handle = model.layers[1].register_forward_pre_hook(lambda mod, args: inputs.update(x=args[0]))
    one_micro_step(engine)
    handle.remove()
    calls, unsloth = counter.calls, counter.unsloth
    assert calls["layers.0"] == ["no_grad"]  # once, without grad: no graph, no recompute
    assert calls["refiner.0"] == ["no_grad"] and calls["refiner.1"] == ["no_grad"]
    assert not counter.outputs["layers.0"].requires_grad
    assert not inputs["x"].requires_grad and inputs["x"].grad_fn is None  # the floor's input: a constant
    passes = {"none": ["grad"], "plain": ["grad", "grad"], "unsloth": ["no_grad", "grad"]}[ckpt]
    for name in ("layers.1", "layers.2"):
        assert calls[name] == passes, (name, calls[name])  # forward (+ recompute)
    if ckpt == "unsloth":
        assert unsloth == {"layers.1": 1, "layers.2": 1}  # no unsloth node (no CPU copy) below the floor
    else:
        assert not unsloth
    active = set(engine.active_slots)
    assert {engine.slots[i].name.split(".")[0] for i in active} == {"layers", "final_norm_video", "final_linear_video",
                                                                     "video_head_modulation"}
    assert engine._filled == active  # the floor block's weights got their gradients


def test_unsloth_floor_block_keeps_its_weight_gradients():
    """unsloth's autograd.Function returns a grad-free output for grad-free inputs: the floor block's input is made a
    leaf, so its weight gradients exist and equal the run without checkpointing."""
    grads = {}
    for ckpt in ("none", "unsloth"):
        engine = make_engine(build_model(ckpt=ckpt), "adamw", k=1, period=1000, select="window", always="head")
        engine.activate(0)
        engine.switch((2,))  # layers.0: the floor is block 0
        assert engine.floor == 0 and not engine.text_grad
        one_micro_step(engine)
        grads[ckpt] = {engine.slots[i].name: acc_of(engine, i).clone() for i in engine.active_slots}
        assert set(grads[ckpt]) == set(engine.slots[i].name for i in engine.active_slots)
    assert grads["none"].keys() == grads["unsloth"].keys()
    assert any(name.startswith("layers.0.") for name in grads["unsloth"])
    for name, value in grads["none"].items():
        assert torch.equal(grads["unsloth"][name], value), name


@pytest.mark.parametrize("ckpt", ["plain", "unsloth"])
def test_text_path_with_and_without_gradients(monkeypatch, ckpt):
    model = build_model(ckpt=ckpt)
    counter = Counter(monkeypatch)
    counter.model = model
    engine = make_engine(model, "adamw", k=1, period=1000, select="window", always="small")
    engine.activate(0)
    engine.switch((4,))  # layers.2 only, but always small trains the stem: full depth, text path trained
    assert engine.floor == 0 and engine.text_grad
    one_micro_step(engine, step=0)
    assert "grad" in counter.calls["refiner.0"] and "grad" in counter.calls["layers.0"]
    names = {engine.slots[i].name for i in engine._filled}
    assert "text_embedder.proj.weight" in names and "video_embedder.weight" in names
    assert not any(n.startswith(("layers.0.", "layers.1.", "text_embedder.blocks.")) for n in names)

    model = build_model(ckpt=ckpt)
    counter.calls.clear()
    counter.model = model
    engine = make_engine(model, "adamw", k=1, period=1000, select="window", always="head")
    engine.activate(0)
    engine.switch((1,))  # the second refiner: text path trained, floor 0
    assert engine.floor == 0 and engine.text_grad
    one_micro_step(engine)
    names = {engine.slots[i].name for i in engine._filled}
    assert any(n.startswith("text_embedder.blocks.1.") for n in names)
    assert not any(n.startswith("text_embedder.blocks.0.") or n.startswith("text_embedder.proj") for n in names)
