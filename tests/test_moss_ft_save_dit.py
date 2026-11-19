"""CPU tests of the trained-DiT save (library/moss/ft_model.py save_moss_dit, library/moss/ft_safetensors.py;
docs/moss_full_finetune_design.md 4.7, section-5 rows test_base_layout / test_round_trip / test_heal_layout_input).

The DiT file must be the Base layout ComfyUI detects (1.2): exactly the Base key set, sorted, bare names, bf16 (or the
save precision) for every parameter, rope.bands F32, header metadata {"format": "pt"}, split q / k / v. The full-size
key / dtype / shape / offset plan is compared with the real Base file's header when that file is on this PC.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_save_dit.py -q -p no:cacheprovider
"""
import json
import os
import struct
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _moss_ft_model_common as M  # noqa: E402

from library.moss import ft_args, ft_model, ft_safetensors  # noqa: E402
from library.moss.models import MOSS_DIT_CONFIG, MossTransformer  # noqa: E402

BASE_FILE = os.environ.get("MOSS_BASE_FILE",
                           r"C:\sd-webui-forge-classic\models\Stable-diffusion\moss_image_base_bf16.safetensors")


def _raw_header(path):
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        return length, f.read(length)


def _loaded(tmp_path, **kw):
    path, base = M.write_base(tmp_path)
    model, pool, slots = ft_model.load_moss_for_full_ft(str(path), M.args(**kw), "cpu", config=M.TINY)
    return model, pool, slots, base


def _train_a_little(pool, seed=7):
    """Move every master off its Base value (as training does) and keep the compute copies in step."""
    g = torch.Generator().manual_seed(seed)
    for slot in pool.slots:
        master = pool.view("master", slot)
        master.add_(torch.randn(master.shape, generator=g) * 1e-3)
    pool.copy_params_from_master()


def test_base_layout(tmp_path):
    model, pool, slots, base = _loaded(tmp_path)
    out = tmp_path / "dit.safetensors"
    size = ft_model.save_moss_dit(str(out), model, pool, torch.bfloat16, {"modelspec.title": "tiny"})
    assert size == os.path.getsize(out) and not os.path.exists(str(out) + ".tmp")
    length, raw = _raw_header(out)
    assert (8 + length) % 256 == 0 and raw.startswith(b'{"__metadata__":{"format":"pt",')
    header = json.loads(raw)
    metadata = header.pop("__metadata__")
    assert metadata == {"format": "pt", "modelspec.title": "tiny"}
    keys = list(header)
    assert keys == sorted(model.state_dict()) == sorted(base)  # the Base key set, in sorted order
    offset = 0
    for key in keys:  # back to back in key order, no gaps
        begin, end = header[key]["data_offsets"]
        assert begin == offset
        offset = end
    assert 8 + length + offset == size
    for key in keys:
        want = "F32" if key == "rope.bands" else "BF16"
        assert header[key]["dtype"] == want, key
        assert header[key]["shape"] == list(base[key].shape), key
    assert header["rope.bands"]["shape"] == [M.TINY["dim"] // M.TINY["num_heads"] // 8]
    assert "shared_block_modulation.weight" in header and "layers.0.proj.ffn_up_gate.weight" in header
    assert not any(k.endswith(".proj.qkv.weight") for k in keys)
    # the official reader reads it, with the master values (untrained: the Base values)
    with safe_open(str(out), framework="pt", device="cpu") as f:
        assert f.metadata() == metadata
        for key in keys:
            assert torch.equal(f.get_tensor(key), base[key]), key


@pytest.mark.skipif(not os.path.isfile(BASE_FILE), reason="the MOSS Base file is not on this PC")
def test_full_size_plan_matches_the_base_file():
    """MOSS_DIT_CONFIG on the meta device: the planned keys, dtypes, shapes and data offsets are the Base file's (463
    tensors, 5,082,308,624 elements); only the header padding differs (256 here, 8 there)."""
    with torch.device("meta"):
        model = MossTransformer(**MOSS_DIT_CONFIG)
    entries = ft_model.dit_entries(model, None, torch.bfloat16)
    planned = json.loads(ft_safetensors.header_bytes(entries, {"format": "pt"})[8:])
    _, raw = _raw_header(BASE_FILE)
    base = json.loads(raw)
    assert planned.pop("__metadata__") == base.pop("__metadata__") == {"format": "pt"}
    assert list(planned) == list(base)
    assert planned == base
    assert len(base) == 463 and sum(e.nbytes for e in entries) == base["video_head_modulation.weight"]["data_offsets"][1]
    assert sum(e.dtype == torch.bfloat16 for e in entries) == 462


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32], ids=["bf16", "fp16", "float"])
def test_round_trip(tmp_path, monkeypatch, dtype):
    """save_moss_dit -> load_moss_dit: trained tensors == master cast to the save dtype, frozen ones keep the Base
    values; master.safetensors-style fp32 saves are the masters bit for bit."""
    import moss_train_network as mtn

    model, pool, slots, base = _loaded(tmp_path, moss_ft_freeze=r"layers\.1\..*,time_embedding\.0\.bias")
    _train_a_little(pool)
    out = tmp_path / "dit.safetensors"
    ft_model.save_moss_dit(str(out), model, pool, dtype, {})
    monkeypatch.setattr(mtn, "MOSS_DIT_CONFIG", M.TINY)
    loaded = mtn.load_moss_dit(str(out), "torch", False, torch.device("cpu"), dtype).state_dict()
    trained = {s.name: s for s in slots}
    frozen = [n for n, _ in model.named_parameters() if n not in trained]
    assert frozen and all(n.startswith("layers.1.") or n == "time_embedding.0.bias" for n in frozen)
    for name, slot in trained.items():
        assert torch.equal(loaded[name], pool.view("master", slot).view(slot.shape).to(dtype)), name
    for name in frozen:
        assert torch.equal(loaded[name], base[name].to(dtype)), name


def test_heal_layout_input(tmp_path):
    """A fused-qkv file (heal layout) loaded for full FT is saved in the split Base layout."""
    state = M.base_state()
    fused = M.fuse_qkv(state)
    path = tmp_path / "heal.safetensors"
    save_file({k: v.contiguous() for k, v in fused.items()}, str(path))
    model, pool, slots = ft_model.load_moss_for_full_ft(str(path), M.args(), "cpu", config=M.TINY)
    out = tmp_path / "dit.safetensors"
    ft_model.save_moss_dit(str(out), model, pool, "bf16", {})
    with safe_open(str(out), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        assert sorted(keys) == sorted(state)
        assert not any(k.endswith(".proj.qkv.weight") for k in keys)
        for n in range(M.TINY["num_layers"]):
            for part in ("q", "k", "v"):
                key = f"layers.{n}.proj.{part}.weight"
                assert torch.equal(f.get_tensor(key), state[key]), key


def test_save_without_a_pool_uses_the_model(tmp_path):
    """adafactor_fused has no host master: the DiT comes from the compute copies."""
    path, base = M.write_base(tmp_path)
    args = M.args(optimizer_type="Adafactor", fused_backward_pass=True)
    model, pool, slots = ft_model.load_moss_for_full_ft(str(path), args, "cpu", config=M.TINY)
    assert pool is None
    with torch.no_grad():
        model.final_linear_video.weight.add_(1.0)
    out = tmp_path / "dit.safetensors"
    ft_model.save_moss_dit(str(out), model, None, torch.bfloat16, {})
    got = M.load(out)
    assert torch.equal(got["final_linear_video.weight"], model.final_linear_video.weight.detach().bfloat16())
    assert torch.equal(got["layers.0.proj.q.weight"], base["layers.0.proj.q.weight"])


def test_failed_save_leaves_the_old_file(tmp_path, monkeypatch):
    model, pool, slots, base = _loaded(tmp_path)
    out = tmp_path / "dit.safetensors"
    out.write_bytes(b"previous save")

    def broken(path, entries, metadata=None, **kw):
        with open(path, "wb") as f:
            f.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(ft_model, "write_safetensors", broken)
    with pytest.raises(OSError, match="disk full"):
        ft_model.save_moss_dit(str(out), model, pool, torch.bfloat16, {})
    assert out.read_bytes() == b"previous save" and not os.path.exists(str(out) + ".tmp")


def test_dit_metadata():
    parser = ft_args.setup_parser()
    args = parser.parse_args(["--output_name", "run1", "--pretrained_model_name_or_path", "D:/m/moss_image_base_bf16.safetensors",
                              "--moss_text_encoder", "C:/te/qwen.safetensors", "--moss_ft_lr_phases", "2:5e-6,6:1e-6",
                              "--moss_ft_freeze", r"layers\.0\..*"])
    meta = ft_model.moss_dit_metadata(args, mode="adamw_host", optimizer_kwargs={"betas": (0.9, 0.999)}, global_step=1234,
                                      epoch=3, train_images=16000)
    assert list(meta)[:4] == ["format", "modelspec.architecture", "modelspec.implementation", "modelspec.title"]
    assert meta["format"] == "pt" and meta["modelspec.architecture"] == "moss-image-base"
    assert meta["modelspec.implementation"] == "https://github.com/OpenMOSS/MOSS-Image"
    assert meta["modelspec.title"] == "run1" and "ss_moss_lora_exclude_blocks" not in meta
    assert meta["ss_base_model_version"] == "moss_image" and meta["ss_moss_text_encoder"] == "qwen.safetensors"
    assert meta["ss_moss_ft_mode"] == "adamw_host" and meta["ss_moss_ft_betas"] == "0.9,0.999"
    assert meta["ss_moss_ft_lr_phases"] == "2:5e-6,6:1e-6" and meta["ss_moss_ft_global_step"] == "1234"
    assert meta["ss_moss_ft_epoch"] == "3" and meta["ss_moss_ft_train_images"] == "16000"
    assert meta["ss_moss_ft_base_model"] == "moss_image_base_bf16.safetensors"
    assert meta["ss_moss_ft_freeze"] == r"layers\.0\..*"
    assert all(isinstance(v, str) for v in meta.values())


# ---- ft_safetensors writer / reader ---------------------------------------------------------------------------------
def test_writer_and_reader(tmp_path):
    g = torch.Generator().manual_seed(0)
    tensors = {"a": torch.randn(3, 5, generator=g), "b": torch.randn(7, generator=g).to(torch.bfloat16),
               "c": torch.tensor(5, dtype=torch.int64), "d": torch.zeros(0, 4), "e": torch.randn(2, 3, generator=g).t()}
    entries = [ft_safetensors.TensorEntry(k, v.dtype, tuple(v.shape), lambda v=v: v) for k, v in tensors.items()]
    path = tmp_path / "x.safetensors"
    size = ft_safetensors.write_safetensors(str(path), entries, {"k": 1})
    assert size == os.path.getsize(path)
    with safe_open(str(path), framework="pt", device="cpu") as f:  # the official reader agrees
        assert f.metadata() == {"k": "1"}
        for key, value in tensors.items():
            assert torch.equal(f.get_tensor(key), value), key
    with ft_safetensors.SafetensorsReader(str(path)) as reader:
        assert reader.keys() == list(tensors)
        out = torch.empty(15)
        assert reader.read("a", out) is out and torch.equal(out.view(3, 5), tensors["a"])
        assert torch.equal(reader.read("e"), tensors["e"]) and torch.equal(reader.read("c"), tensors["c"])
        with pytest.raises(ft_safetensors.SafetensorsError, match="cannot be read into"):
            reader.read("a", torch.empty(15, dtype=torch.float64))
        with pytest.raises(ft_safetensors.SafetensorsError, match="no tensor"):
            reader.read("zzz")


def test_writer_refuses_a_wrong_producer(tmp_path):
    entries = [ft_safetensors.TensorEntry("a", torch.float32, (2, 2), lambda: torch.zeros(4))]
    with pytest.raises(ValueError, match="the producer returned"):
        ft_safetensors.write_safetensors(str(tmp_path / "x.safetensors"), entries)
    entries = [ft_safetensors.TensorEntry("a", torch.float32, (2,), lambda: torch.zeros(2)),
               ft_safetensors.TensorEntry("a", torch.float32, (2,), lambda: torch.zeros(2))]
    with pytest.raises(ValueError, match="twice"):
        ft_safetensors.write_safetensors(str(tmp_path / "y.safetensors"), entries)


def test_reader_detects_truncation(tmp_path):
    path = tmp_path / "x.safetensors"
    entries = [ft_safetensors.TensorEntry(k, torch.float32, (64,), lambda: torch.ones(64)) for k in ("a", "b")]
    ft_safetensors.write_safetensors(str(path), entries)
    data = path.read_bytes()
    path.write_bytes(data[:-10])
    with pytest.raises(ft_safetensors.SafetensorsError, match="truncated"):
        ft_safetensors.SafetensorsReader(str(path))
    path.write_bytes(data[:5])
    with pytest.raises(ft_safetensors.SafetensorsError, match="shorter"):
        ft_safetensors.SafetensorsReader(str(path))
    path.write_bytes(struct.pack("<Q", 4) + b"[1] " + b"\0" * 8)
    with pytest.raises(ft_safetensors.SafetensorsError, match="not a JSON object"):
        ft_safetensors.SafetensorsReader(str(path))
