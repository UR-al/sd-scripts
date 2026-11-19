"""CPU tests of library/moss/te_live_weights.py (docs/moss_ft_te_live_fp16_plan.md 1.2): the aligned layer layouts,
the host forms (copy / mmap) holding exactly the values the cached path's loader gives, the embedding gather, and the
refusals (pickling, pinned without CUDA, a file that is not this encoder).

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_te_live_weights.py -q -p no:cacheprovider
"""
import os
import pickle
from dataclasses import replace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
from safetensors.torch import save_file

from _moss_te_tiny import PREFIX, TINY_TE, open_weights, random_ids, reference_model, write_tiny_te
from library.moss import ft_budget, te_live_weights
from library.moss.qwen35 import MossQwen35Config


@pytest.fixture(scope="module")
def te_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("te_live_weights") / "te.safetensors"
    state = write_tiny_te(path)
    return path, state


def test_layouts_are_aligned_and_cover_every_layer_parameter():
    layouts = te_live_weights.layer_layouts(TINY_TE)
    assert len(layouts) == TINY_TE.num_hidden_layers
    assert [layout.kind for layout in layouts] == ["full" if TINY_TE.is_full_attention(i) else "linear"
                                                   for i in range(TINY_TE.num_hidden_layers)]
    for layout in layouts:
        end = 0
        for entry in layout.entries:
            assert entry.offset % te_live_weights.ALIGN == 0 and entry.offset >= end, entry
            end = entry.offset + entry.numel * 2
        assert layout.nbytes % te_live_weights.ALIGN == 0 and layout.nbytes >= end
    linear = [e.name for e in layouts[0].entries]
    assert "linear_attn.in_proj_qkv.weight" in linear and "linear_attn.A_log" in linear and "mlp.down_proj.weight" in linear
    full = [e.name for e in layouts[3].entries]
    assert "self_attn.q_proj.weight" in full and "self_attn.k_norm.weight" in full
    assert te_live_weights.slot_bytes(layouts) == max(layout.nbytes for layout in layouts)
    entry = te_live_weights.staging_entry_bytes(layouts)
    assert entry >= te_live_weights.slot_bytes(layouts) and entry & (entry - 1) == 0


def test_real_encoder_layout_matches_the_budget_constants():
    """The 9B's layout on the meta device (no weights): what ft_budget counts for the host RAM and the device ring."""
    sizes = te_live_weights.te_layer_bytes(MossQwen35Config())
    assert len(sizes) == 32
    assert sum(sizes) == ft_budget.TE_LAYER_BYTES_TOTAL  # 12.889 GiB
    assert max(sizes) == ft_budget.TE_SLOT_BYTES  # 416.6 MiB
    assert sorted(set(sizes)) == [419_447_808, 436_814_592]
    assert sizes.count(436_814_592) == 24 and sizes.count(419_447_808) == 8
    layouts = te_live_weights.layer_layouts(MossQwen35Config())
    assert te_live_weights.staging_entry_bytes(layouts) == ft_budget.TE_STAGING_ENTRY_BYTES
    assert te_live_weights.te_embed_bytes() == ft_budget.TE_EMBED_BYTES
    assert abs(sum(sizes) / (1 << 30) - 12.889) < 1e-3 and abs(max(sizes) / (1 << 20) - 416.6) < 0.05


@pytest.mark.parametrize("host", ["copy", "mmap"])
def test_host_forms_hold_the_cached_loaders_values(te_file, host):
    path, state = te_file
    ref = dict(reference_model(path).named_parameters())
    weights = open_weights(path, host)
    try:
        assert weights.num_layers == TINY_TE.num_hidden_layers
        for index in range(weights.num_layers):
            sources = weights.layer_sources(index)
            assert len(sources) == len(weights.layouts[index].entries)
            for entry, tensor in sources:
                expected = ref[f"layers.{index}.{entry.name}"]
                assert tuple(tensor.shape) == tuple(expected.shape)
                assert torch.equal(tensor.to(torch.bfloat16), expected), (host, index, entry.name)
                if entry.name.endswith("A_log"):  # stored fp32: copy casts once (RNE), mmap keeps the file's fp32
                    stored = state[f"{PREFIX}layers.{index}.{entry.name}"]
                    assert tensor.dtype == (torch.bfloat16 if host == "copy" else torch.float32)
                    assert torch.equal(tensor.to(torch.bfloat16), stored.to(torch.bfloat16))
        assert torch.equal(weights.final_norm_weight, ref["norm.weight"])
        if host == "copy":
            assert weights.private_bytes == sum(layout.nbytes for layout in weights.layouts)
            for index in range(weights.num_layers):
                assert weights.layer_flat(index).numel() == weights.layouts[index].nbytes
        else:
            assert weights.private_bytes == 0 and weights.layer_flat(0) is None
        assert weights.locked_bytes == 0
    finally:
        weights.close()


@pytest.mark.parametrize("host", ["copy", "mmap"])
def test_embed_rows_is_the_embedding_lookup(te_file, host):
    path, _ = te_file
    model = reference_model(path)
    weights = open_weights(path, host)
    try:
        ids = random_ids(57, seed=3)
        rows = weights.embed_rows(ids)
        assert rows.dtype == torch.bfloat16 and rows.device.type == "cpu"
        assert torch.equal(rows, model.embed_tokens(ids[None])[0])
    finally:
        weights.close()


def test_never_pickled(te_file):
    weights = open_weights(te_file[0])
    try:
        with pytest.raises(TypeError, match="never pickled"):
            pickle.dumps(weights)
    finally:
        weights.close()


def test_pinned_needs_cuda(te_file):
    with pytest.raises(ValueError, match="pinned"):
        open_weights(te_file[0], "pinned")
    with pytest.raises(ValueError, match="host mode"):
        open_weights(te_file[0], "ram")


def test_a_file_of_another_encoder_is_refused(te_file, tmp_path):
    _, state = te_file
    missing = dict(state)
    missing.pop(PREFIX + "layers.2.mlp.up_proj.weight")
    save_file({k: v.contiguous() for k, v in missing.items()}, str(tmp_path / "missing.safetensors"))
    with pytest.raises(ValueError, match="missing"):
        open_weights(tmp_path / "missing.safetensors")
    wide = replace(TINY_TE, intermediate_size=96)
    with pytest.raises(ValueError, match="expected"):
        open_weights(te_file[0], config=wide)
    extra = dict(state)
    extra[PREFIX + "layers.0.mlp.extra.weight"] = torch.zeros(2, dtype=torch.bfloat16)
    save_file({k: v.contiguous() for k, v in extra.items()}, str(tmp_path / "extra.safetensors"))
    with pytest.raises(ValueError, match="unexpected"):
        open_weights(tmp_path / "extra.safetensors")


def test_the_9b_check_runs_by_default(te_file):
    """check=True (the trainer's call) is qwen35.check_moss_text_encoder: the tiny file is not the 9B."""
    with pytest.raises(ValueError, match="Qwen3.5-9B"):
        te_live_weights.HostTextEncoderWeights.open(str(te_file[0]), config=TINY_TE)


def test_close_releases_the_mapping(te_file):
    weights = open_weights(te_file[0], "mmap")
    weights.close()
    assert weights._mm is None and weights.layer_flat(0) is None
    weights.close()  # idempotent
