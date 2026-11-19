"""CPU tests of library/moss/ft_model.py's loading side (docs/moss_full_finetune_design.md 4.4.1): select_trainable
(--moss_ft_freeze), param_groups (decay / no_decay), and load_moss_for_full_ft - the DiT built on the meta device, the
host pool allocated for the trainable set, the file read one tensor at a time into the fp32 master and the compute
copy (heal-layout q/k/v split, prefixes stripped, strict keys), the fp32 stream, the stream scale, requires_grad.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_model.py -q -p no:cacheprovider
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _moss_ft_model_common as M  # noqa: E402

from library.moss import ft_host, ft_model  # noqa: E402
from library.moss.models import MossTransformer  # noqa: E402


# ---- select_trainable / param_groups --------------------------------------------------------------------------------
def test_select_trainable_freezes_by_fullmatch_in_registration_order():
    model = MossTransformer(**M.TINY)
    trainable = ft_model.select_trainable(model, r"layers\.1\..*,final_.*")
    names = [n for n, _ in trainable]
    every = [n for n, _ in model.named_parameters()]
    assert names == [n for n in every if not (n.startswith("layers.1.") or n.startswith("final_"))]
    for name, p in model.named_parameters():
        assert p.requires_grad == (name in names), name
    assert all(p is dict(model.named_parameters())[n] for n, p in trainable)
    # nothing frozen by default; a regex matches whole names only
    assert len(ft_model.select_trainable(MossTransformer(**M.TINY), "")) == len(every)
    assert len(ft_model.select_trainable(MossTransformer(**M.TINY), r"layers\.0\.proj\.q\.weight")) == len(every) - 1


def test_select_trainable_refusals():
    with pytest.raises(ValueError, match="matches no parameter"):
        ft_model.select_trainable(MossTransformer(**M.TINY), r"layers\.9\..*")
    with pytest.raises(ValueError, match="every parameter"):
        ft_model.select_trainable(MossTransformer(**M.TINY), ".*")


def test_param_groups_decay_and_no_decay():
    model = MossTransformer(**M.TINY)
    named = list(model.named_parameters())
    groups = ft_model.param_groups(named, weight_decay=0.05)
    assert [g["name"] for g in groups] == ["decay", "no_decay"]
    assert [g["weight_decay"] for g in groups] == [0.05, 0.0]
    decay = {id(p) for p in groups[0]["params"]}
    no_decay = {id(p) for p in groups[1]["params"]}
    assert not decay & no_decay and len(decay) + len(no_decay) == len(named)
    for name, p in named:
        embedder = name.startswith(("video_embedder.", "text_embedder.proj."))
        assert (id(p) in decay) == (p.dim() >= 2 and not embedder), name
    names = {n for n, p in named if id(p) in decay}
    for expected in ("layers.0.proj.q.weight", "layers.2.proj.ffn_down.weight", "text_embedder.blocks.0.qkv.weight",
                     "shared_block_modulation.weight", "video_head_modulation.weight", "time_embedding.0.weight",
                     "time_embedding.2.weight", "final_linear_video.weight"):
        assert expected in names, expected
    for expected in ("video_embedder.weight", "text_embedder.proj.weight", "shared_block_modulation.bias",
                     "layers.0.norm1.weight", "text_embedder.blocks.1.qkv.bias", "final_norm_video.weight"):
        assert expected not in names, expected
    # the group order of each group follows the given order; an empty group is left out
    only_1d = [(n, p) for n, p in named if p.dim() == 1]
    groups = ft_model.param_groups(only_1d, weight_decay=0.05)
    assert [g["name"] for g in groups] == ["no_decay"] and len(groups[0]["params"]) == len(only_1d)


# ---- load_moss_for_full_ft ---------------------------------------------------------------------------------------
def test_load_matches_load_moss_dit_and_fills_master(tmp_path, monkeypatch):
    import moss_train_network as mtn

    path, base = M.write_base(tmp_path)
    args = M.args()
    model, pool, slots = ft_model.load_moss_for_full_ft(str(path), args, "cpu", config=M.TINY)
    assert isinstance(model, MossTransformer) and model.training and model.stream_fp32
    # compute copies: bf16 joint Linears, fp32 everything else (--moss_stream_fp32)
    for name, p in model.named_parameters():
        joint = name.startswith("layers.") and ".proj." in name
        assert p.dtype == (torch.bfloat16 if joint else torch.float32), name
        assert p.requires_grad and p.device.type == "cpu"
        assert torch.equal(p.detach(), base[name].to(p.dtype)), name
    assert model.rope.bands.dtype == torch.float32 and torch.equal(model.rope.bands,
                                                                   model.rope._default_bands(torch.device("cpu")))
    # the pool: every parameter trainable, master = the file values in fp32, moments zero, no accumulator at GA 1
    assert [s.name for s in slots] == [n for n, _ in model.named_parameters()]
    assert tuple(pool.roles) == ("master", "exp_avg", "exp_avg_sq") and pool.acc is None
    for s in slots:
        assert s.param is dict(model.named_parameters())[s.name] and s.compute_dtype == s.param.dtype
        assert torch.equal(pool.view("master", s), base[s.name].float().reshape(-1)), s.name
    assert all(float(pool.roles[r].abs().max()) == 0.0 for r in ("exp_avg", "exp_avg_sq"))
    # the same values as the LoRA trainer's strict loader
    monkeypatch.setattr(mtn, "MOSS_DIT_CONFIG", M.TINY)
    reference = mtn.load_moss_dit(str(path), "torch", False, torch.device("cpu"), torch.bfloat16)
    reference.enable_stream_fp32()
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
        assert torch.equal(p.detach(), q.detach()), name
    ft_model.check_compute_copies(pool, slots)


def test_load_modes_freeze_and_accumulator(tmp_path):
    path, base = M.write_base(tmp_path)
    args = M.args(optimizer_type="CAME", optimizer_args=["bias_correction=True"], gradient_accumulation_steps=2,
                  moss_ft_freeze=r"layers\.2\..*")
    model, pool, slots = ft_model.load_moss_for_full_ft(str(path), args, "cpu", config=M.TINY)
    assert tuple(pool.roles) == ("master", "exp_avg") and pool.acc is not None and pool.acc.numel() == pool.total_numel
    assert not any(s.name.startswith("layers.2.") for s in slots)
    for name, p in model.named_parameters():
        assert p.requires_grad == (not name.startswith("layers.2.")), name
        assert torch.equal(p.detach(), base[name].to(p.dtype)), name  # frozen ones loaded too
    # adafactor_fused: no host pool, the trainable set still made
    args = M.args(optimizer_type="Adafactor", fused_backward_pass=True)
    model, pool, slots = ft_model.load_moss_for_full_ft(str(path), args, "cpu", config=M.TINY)
    assert pool is None and len(slots) == len(list(model.parameters()))
    # exact clipping needs the accumulator even at GA 1
    args = M.args(moss_ft_clip_mode="exact", max_grad_norm=1.0)
    assert ft_model.load_moss_for_full_ft(str(path), args, "cpu", config=M.TINY)[1].acc is not None


def test_load_fp32_compute_and_trainable_fn(tmp_path):
    path, base = M.write_base(tmp_path)
    picked = []

    def trainable_fn(model):
        out = [(n, p) for n, p in model.named_parameters() if n.startswith("layers.")]
        picked.append(len(out))
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith("layers."))
        return out

    model, pool, slots = ft_model.load_moss_for_full_ft(str(path), M.args(mixed_precision="no", moss_stream_fp32=False),
                                                        "cpu", trainable_fn, config=M.TINY)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert picked and [s.name for s in slots] == [n for n, _ in model.named_parameters() if n.startswith("layers.")]
    assert all(p.requires_grad == n.startswith("layers.") for n, p in model.named_parameters())


def test_fp32_source_keeps_the_fp32_stream_exact(tmp_path):
    """An fp32 file (a state's master.safetensors, an fp32 DiT): the fp32-stream parameters are its values exactly and
    the bf16 joint Linears its round to nearest - never an fp32 value rounded through bf16 first."""
    path, base = M.write_base(tmp_path, dtype=torch.float32)
    model, pool, slots = ft_model.load_moss_for_full_ft(str(path), M.args(), "cpu", config=M.TINY)
    for s in slots:
        master = pool.view("master", s)
        assert torch.equal(master, base[s.name].reshape(-1)), s.name
        assert torch.equal(s.param.detach().reshape(-1), master.to(s.compute_dtype)), s.name
    stream = [s for s in slots if s.compute_dtype == torch.float32]
    assert stream and any(not torch.equal(s.param.detach(), s.param.detach().bfloat16().float()) for s in stream)
    ft_model.check_compute_copies(pool, slots)


def test_prefixed_and_fused_files(tmp_path):
    path, base = M.write_base(tmp_path)
    state = {k: v for k, v in M.load(path).items()}
    fused = M.fuse_qkv(state)
    prefixed = tmp_path / "prefixed_heal.safetensors"
    save_file({"model.diffusion_model." + k: v.contiguous() for k, v in fused.items()}, str(prefixed))
    model, pool, slots = ft_model.load_moss_for_full_ft(str(prefixed), M.args(), "cpu", config=M.TINY)
    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), base[name].to(p.dtype)), name
    assert not any(".qkv." in s.name and s.name.startswith("layers.") for s in slots)


def test_strict_keys(tmp_path):
    path, base = M.write_base(tmp_path)
    state = M.load(path)

    def attempt(edit, match):
        bad = dict(state)
        edit(bad)
        file = tmp_path / "bad.safetensors"
        save_file({k: v.contiguous() for k, v in bad.items()}, str(file))
        with pytest.raises(ValueError, match=match):
            ft_model.load_moss_for_full_ft(str(file), M.args(), "cpu", config=M.TINY)

    attempt(lambda s: s.pop("layers.1.proj.o.weight"), "missing.*layers.1.proj.o.weight")
    attempt(lambda s: s.__setitem__("extra.weight", torch.zeros(2)), "unexpected.*extra.weight")
    attempt(lambda s: s.__setitem__("final_norm_video.weight", torch.zeros(3)), "shape")
    attempt(lambda s: s.__setitem__("layers.0.proj.qkv.weight", torch.zeros(3 * 64, 64)), "layers.0.proj")
    # rope.bands is derived: a file without it loads
    no_rope = dict(state)
    no_rope.pop("rope.bands")
    file = tmp_path / "no_rope.safetensors"
    save_file({k: v.contiguous() for k, v in no_rope.items()}, str(file))
    model, _, _ = ft_model.load_moss_for_full_ft(str(file), M.args(), "cpu", config=M.TINY)
    assert torch.equal(model.rope.bands, model.rope._default_bands(torch.device("cpu")))


def test_stream_scale_and_attention_options(tmp_path):
    path, _ = M.write_base(tmp_path)
    model, _, _ = ft_model.load_moss_for_full_ft(str(path), M.args(moss_stream_scale=0.25, attn_mode="sdpa",
                                                                   split_attn=True), "cpu", config=M.TINY)
    assert model.stream_scale == 0.25 and model.layers[0].norm1.eps == 1e-6 * 0.25 * 0.25
    assert model.attn_mode == "torch" and model.split_attn


def test_check_compute_copies_detects_a_mismatch(tmp_path):
    path, _ = M.write_base(tmp_path)
    model, pool, slots = ft_model.load_moss_for_full_ft(str(path), M.args(), "cpu", config=M.TINY)
    with torch.no_grad():
        slots[3].param.add_(1.0)
    with pytest.raises(RuntimeError, match=slots[3].name):
        ft_model.check_compute_copies(pool, slots)


def test_trainable_named_and_groups_bind_the_pipeline(tmp_path):
    """The slots of the loader, the optimizer over param_groups, bind_slots, and the pipeline accept each other."""
    from library.moss.ft_optim import HostAdamW

    path, _ = M.write_base(tmp_path)
    model, pool, slots = ft_model.load_moss_for_full_ft(str(path), M.args(), "cpu", config=M.TINY)
    named = ft_model.trainable_named(slots)
    opt = HostAdamW(ft_model.param_groups(named, weight_decay=0.01), lr=1e-4)
    ft_host.bind_slots(slots, named, opt)
    pipe = ft_host.GradPipeline(slots, pool, opt, device="cpu", gradient_accumulation_steps=1, inline=True,
                                piece_bytes=2048, ring_bytes=8 * 2048, upload_ring_bytes=4 * 2048, acc_chunk=128,
                                step_chunk=128)
    try:
        assert {s.group for s in slots} == {0, 1}
    finally:
        pipe.close()
