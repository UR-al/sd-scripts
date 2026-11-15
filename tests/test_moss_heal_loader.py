"""CPU tests for loading a MOSS DiT file whose joint blocks carry a fused q/k/v (2026-11-15).

KBlueLeaf/MOSS-Image-heal's ComfyUI file (moss_image_heal_fp16.safetensors) stores each joint block's attention input
projections as one ``layers.N.proj.qkv.weight`` [3 * dim, dim] in place of Base's ``layers.N.proj.{q,k,v}.weight``; every
other key and shape is Base's. ComfyUI reads it as ``q, k, v = proj["qkv"](y).chunk(3, dim=-1)``, so the rows are q, then
k, then v. load_moss_dit splits it back (library/moss/models.split_fused_qkv): the trainer keeps separate q / k / v
Linears, the LoRA / LyCORIS targets, whose keys ComfyUI maps onto the fused rows (comfy/lora.py). The text refiner's
``text_embedder.blocks.N.qkv`` is a fused Linear in the model itself and is left alone.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_heal_loader.py -q -p no:cacheprovider
(an EMPTY CUDA_VISIBLE_DEVICES does not hide the GPU on this Windows build: use -1)
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
from safetensors.torch import save_file

from library.moss.models import MossTransformer, split_fused_qkv

TINY = dict(dim=192, ffn_dim=512, num_heads=4, num_layers=3, text_dim=64, freq_dim=32, latent_channels=128)


def _state(seed=0):
    torch.manual_seed(seed)
    model = MossTransformer(**TINY).eval()
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.05)
    return {k: v.clone() for k, v in model.state_dict().items()}


def _fuse(state):
    """The heal file's layout: q, k, v rows concatenated into layers.N.proj.qkv.weight."""
    fused = dict(state)
    for key in [k for k in state if k.startswith("layers.") and k.endswith(".proj.q.weight")]:
        prefix = key[:-len("q.weight")]
        fused[prefix + "qkv.weight"] = torch.cat([fused.pop(prefix + name + ".weight") for name in ("q", "k", "v")], 0)
    return fused


def test_split_restores_q_k_v():
    state = _state()
    fused = _fuse(state)
    assert sum(k.endswith(".proj.qkv.weight") for k in fused) == TINY["num_layers"]
    assert split_fused_qkv(fused) == TINY["num_layers"]
    assert set(fused) == set(state)
    for key, value in state.items():
        assert torch.equal(fused[key], value), key


def test_text_refiner_qkv_is_left_alone():
    state = _state()
    refiner = [k for k in state if k.startswith("text_embedder.blocks.") and k.endswith(".qkv.weight")]
    assert refiner  # the model's own fused Linear
    fused = _fuse(state)
    split_fused_qkv(fused)
    for key in refiner:
        assert key in fused and torch.equal(fused[key], state[key])
        assert key[:-len("qkv.weight")] + "q.weight" not in fused


def test_base_layout_is_untouched():
    state = _state()
    copy = dict(state)
    assert split_fused_qkv(copy) == 0
    assert copy.keys() == state.keys() and all(copy[k] is state[k] for k in state)


def test_rejects_rows_not_divisible_by_three():
    state = _fuse(_state())
    state["layers.0.proj.qkv.weight"] = state["layers.0.proj.qkv.weight"][:-1]
    with pytest.raises(ValueError, match="layers.0.proj.qkv.weight"):
        split_fused_qkv(state)


def test_rejects_fused_and_separate_together():
    state = _state()
    state["layers.1.proj.qkv.weight"] = torch.cat([state[f"layers.1.proj.{n}.weight"] for n in ("q", "k", "v")], 0)
    with pytest.raises(ValueError, match="layers.1.proj"):
        split_fused_qkv(state)


def test_split_parts_do_not_share_storage():
    fused = _fuse(_state())
    split_fused_qkv(fused)
    pointers = {fused[f"layers.0.proj.{n}.weight"].untyped_storage().data_ptr() for n in ("q", "k", "v")}
    assert len(pointers) == 3  # own tensors: the fused [3 dim, dim] block is not kept alive behind three views


@pytest.mark.parametrize("file_dtype", [torch.float16, torch.bfloat16])
def test_load_moss_dit_reads_a_fused_file(tmp_path, monkeypatch, file_dtype):
    import moss_train_network as mtn

    monkeypatch.setattr(mtn, "MOSS_DIT_CONFIG", TINY)
    state = {k: (v if k == "rope.bands" else v.to(file_dtype)) for k, v in _state().items()}
    plain_path, fused_path = tmp_path / "base.safetensors", tmp_path / "heal.safetensors"
    save_file({k: v.contiguous() for k, v in state.items()}, str(plain_path))
    save_file({k: v.contiguous() for k, v in _fuse(state).items()}, str(fused_path))

    plain = mtn.load_moss_dit(str(plain_path), "torch", False, torch.device("cpu"), torch.float32)
    healed = mtn.load_moss_dit(str(fused_path), "torch", False, torch.device("cpu"), torch.float32)
    want, got = plain.state_dict(), healed.state_dict()
    assert want.keys() == got.keys()
    for key in want:
        assert torch.equal(want[key], got[key]), key
        assert got[key].dtype == want[key].dtype, key
