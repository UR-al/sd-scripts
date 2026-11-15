"""CPU tests for the reuse-only mode of --moss_ckpt_reuse_attn and for --moss_ckpt_reuse_attn_budget_gib (2026-11-15).

Why: the merged-weight cache of --moss_ckpt_nograd_merged (~8.65 GiB on MOSS-Image) does not fit next to the 2048 tier's
activations on a 32 GB card, and on Anima the speed came from the attention reuse, not from the cache. So
--moss_ckpt_reuse_attn now works WITHOUT --moss_ckpt_nograd_merged: the joint blocks still go through the no-grad first
pass + reuse node (anima_models.NoGradFirstPassCheckpointer, untouched), but nothing is cached - both passes rebuild the
LoKr+DoRA weights per call, as the plain checkpoint does. And the stored attention can be capped: per micro-step
n = floor(budget / (joint tokens * rows * bytes per token and block)) blocks (index < n) store and reuse, the rest keep
torch.utils.checkpoint.

The toy model, the fake flash kernels and the helpers are tests/test_moss_ckpt_reuse.py's (imported under the name
pytest gives that module: the custom ops it registers can exist only once per process).

Checked here:
* reuse-only vs the plain checkpoint - eager and aot_eager, bf16 stream and --moss_stream_fp32, one sample / changing
  lengths / equal-length batches / padded batches: output and every LoKr/DoRA gradient bit-identical, the recompute's
  block outputs equal the first pass's, one flash forward kernel per joint block instead of two;
* nothing is cached: no _ckpt_merged tensor on any module, the node never looks a real block's modules up, the weights
  the no-grad first pass rebuilt are dead before backward starts, lycoris_fast._CKPT_PASS is 0 in every attention call,
  and a stale (poisoned) cache left by an earlier cached run is never read;
* the budget: n at several token counts incl. 0 blocks and all blocks (the real model's 6,240 bytes per token and block),
  the block index stored at build time, no device read in the decision;
* mixed mode (some blocks reuse, some plain; n changing between micro-steps): bit-identical to the plain checkpoint,
  eager and compiled, reuse-only and with the cache; training runs with CAME / AdamW driven like train_network.py;
* torch.compile: mixed mode needs at most 6 graphs (plain / first pass / recompute x adapted / left-out block) however
  n moves between micro-steps, no eager fallback;
* the glue: parser + TOML key, validation (reuse-only accepted, the refusals kept, the budget checked),
  prepare_moss_runtime's switches, the per-tier start-up estimate and the peak warning in reuse-only mode.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ckpt_reuse_only.py -q -p no:cacheprovider
(an EMPTY CUDA_VISIBLE_DEVICES does not hide the GPU on this Windows build: use -1)
"""
import argparse
import gc
import os
import sys
import types
import weakref

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
import torch._dynamo
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the name pytest imports the sibling test module under
import test_moss_ckpt_reuse as base  # noqa: E402
from test_moss_ckpt_reuse import _env, came_cls, warned  # noqa: E402,F401 - fixtures (autouse _env included)

from library import anima_models, attention, lycoris_fast, train_util  # noqa: E402
from library import lycoris_algo_fixes as laf  # noqa: E402
from library.moss import models as moss_models  # noqa: E402
from library.moss import train_utils as moss_train_utils  # noqa: E402
from tests import _device_reads  # noqa: E402
from tests import _fake_flash as fake_flash  # noqa: E402

TINY, EXCLUDED, PER_BLOCK = base.TINY, base.EXCLUDED, base.PER_BLOCK
LAYERS = TINY["num_layers"]
PER_TOKEN = TINY["dim"] * 2 + TINY["num_heads"] * 4  # bytes of one token's stored out (bf16) + softmax_lse (fp32): 400
GIB = 2**30


@pytest.fixture(autouse=True)
def _budget(monkeypatch, _env):
    monkeypatch.setattr(moss_models, "CKPT_REUSE_BUDGET_BYTES", None)  # every test starts (and ends) without a budget
    yield


def set_flags(monkeypatch, nograd: bool, reuse: bool, budget=None, diffs: bool = True):
    """The process-wide switches as set_ckpt_reuse leaves them. budget: bytes (None = every block)."""
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", nograd)
    monkeypatch.setattr(anima_models, "CKPT_REUSE_ATTN", reuse)
    monkeypatch.setattr(anima_models, "CKPT_RECOMPUTE_DIFFS", [] if ((nograd or reuse) and diffs) else None)
    monkeypatch.setattr(moss_models, "CKPT_REUSE_BUDGET_BYTES", budget)


def run(monkeypatch, nograd, reuse, budget=None, batches=((dict(),) * 2), **kw):
    model, net, state = base.build(**kw)
    set_flags(monkeypatch, nograd, reuse, budget)
    results = [base.micro_step(model, net, base.batch(stream=kw.get("stream", False), **b)) for b in batches]
    return results, model, net, state


def joint_tokens(spec) -> int:
    hw, text = spec.get("hw", (4, 4)), spec.get("text", (6,))
    return hw[0] * hw[1] + max(text)


def expected_n(spec, budget) -> int:
    """Blocks that store and reuse for a batch() spec under `budget` bytes (the toy model: 400 bytes per token)."""
    if budget is None:
        return LAYERS
    return min(LAYERS, budget // (joint_tokens(spec) * len(spec.get("text", (6,))) * PER_TOKEN))


def spy_nodes(monkeypatch):
    """Records (function name, block argument) of every no-grad node call and every torch.utils.checkpoint call."""
    seen = {"node": [], "plain": 0}
    real_node, real_plain = anima_models.nograd_checkpoint, moss_models.torch_checkpoint

    def node(fn, block, *args):
        seen["node"].append((getattr(fn, "__name__", "?"), block))
        return real_node(fn, block, *args)

    def plain(fn, *args, **kwargs):
        seen["plain"] += 1
        return real_plain(fn, *args, **kwargs)

    monkeypatch.setattr(anima_models, "nograd_checkpoint", node)
    monkeypatch.setattr(moss_models, "torch_checkpoint", plain)
    return seen


# -------------------------------------------------------------------------------------------------------- reuse-only


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
@pytest.mark.parametrize("stream", [False, True], ids=["bf16_stream", "fp32_stream"])
@pytest.mark.parametrize("kind", list(base.BATCHES))
def test_reuse_only_matches_the_plain_checkpoint(kind, stream, compiled, monkeypatch):
    """--moss_ckpt_reuse_attn without --moss_ckpt_nograd_merged: output (hence the loss) and every LoKr / DoRA gradient
    equal torch.utils.checkpoint's bit for bit over consecutive micro-steps; the recompute's block outputs equal the
    first pass's; nothing is cached."""
    batches = base.BATCHES[kind]
    ref, *_ = run(monkeypatch, False, False, batches=batches, compiled=compiled, stream=stream)
    new, model, net, state = run(monkeypatch, False, True, batches=batches, compiled=compiled, stream=stream)
    for i, (a, b) in enumerate(zip(ref, new)):
        base.assert_same(a, b, f"{kind}, micro-step {i}: reuse-only vs torch.utils.checkpoint")
    diffs = anima_models.CKPT_RECOMPUTE_DIFFS
    if kind == "padded" and compiled:  # a compiled block's padded call keeps torch.utils.checkpoint
        assert diffs == []
    else:
        assert len(diffs) == len(batches) * len(model.layers) and set(diffs) == {0.0}, diffs
    assert lycoris_fast.ckpt_merged_bytes(model) == 0
    assert all("_ckpt_merged" not in lokr.__dict__ for lokr in laf._all_loras(net))
    if compiled:
        assert state.eager_calls == 0 and state.compiled_calls == 2 * len(batches) * len(model.layers)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
@pytest.mark.parametrize("kind, reused", [("one_sample", True), ("equal_lengths", True), ("padded", False)])
def test_reuse_only_skips_one_flash_forward_per_joint_block(kind, reused, compiled, monkeypatch):
    """The kernel counts of reuse-only are those of --moss_ckpt_nograd_merged + --moss_ckpt_reuse_attn: one forward
    kernel per joint block and micro-step instead of two, the backward count unchanged; a padded batch stores nothing."""
    batches = base.BATCHES[kind][:1]
    counts = {}
    for name, flags in (("off", (False, False)), ("reuse_only", (False, True)), ("both", (True, True))):
        _, model, net, _ = run(monkeypatch, *flags, batches=batches, compiled=compiled)
        fake_flash.CALLS["fwd"] = fake_flash.CALLS["bwd"] = base.VARLEN_CALLS["fwd"] = base.VARLEN_CALLS["bwd"] = 0
        base.micro_step(model, net, base.batch(**batches[0]))
        counts[name] = (dict(fake_flash.CALLS), dict(base.VARLEN_CALLS))
    assert counts["reuse_only"] == counts["both"]
    if reused:
        assert counts["off"] == ({"fwd": 2 * (2 + LAYERS), "bwd": 2 + LAYERS}, {"fwd": 0, "bwd": 0})
        assert counts["reuse_only"] == ({"fwd": 2 * 2 + LAYERS, "bwd": 2 + LAYERS}, {"fwd": 0, "bwd": 0})
    else:
        assert counts["reuse_only"] == counts["off"]


def test_reuse_only_caches_nothing(monkeypatch):
    """Reuse-only: every joint block goes through the node with an empty stand-in module in the place of the block (the
    node finds no module to cache), the function it runs keeps lycoris_fast._CKPT_PASS at 0 (a cached weight is never
    looked for), both passes rebuild every adapted weight (no-grad in the first pass, with grad in the recompute, as
    often as the plain checkpoint's two grad passes), and the first pass's merged weights are gone before backward."""
    model, net, _ = base.build()
    set_flags(monkeypatch, False, True)
    seen = spy_nodes(monkeypatch)
    refs, passes = [], []
    inner = lycoris_fast._REBUILD.eager["rebuild"]
    counts = {"nograd": 0, "grad": 0}

    def rebuild(*args):
        merged = inner(*args)
        counts["grad" if torch.is_grad_enabled() else "nograd"] += 1
        if not torch.is_grad_enabled():
            refs.append(weakref.ref(merged))
        return merged

    real_attention = attention.attention

    def spy_attention(qkv, attn_params=None, **kwargs):
        passes.append((lycoris_fast._CKPT_PASS, sorted(kwargs)))
        return real_attention(qkv, attn_params=attn_params, **kwargs)

    monkeypatch.setitem(lycoris_fast._REBUILD.eager, "rebuild", rebuild)
    monkeypatch.setattr(attention, "attention", spy_attention)
    net.zero_grad(set_to_none=True)
    y = base.forward(model, base.batch())
    loss = y.float().square().mean()
    adapted = (LAYERS - 1) * PER_BLOCK  # one joint block is left out of the network
    assert counts == {"nograd": adapted, "grad": 0}
    gc.collect()
    assert len(refs) == adapted and all(ref() is None for ref in refs), "a first-pass merged weight is still alive"
    assert lycoris_fast.ckpt_merged_bytes(model) == 0
    assert all("_ckpt_merged" not in lokr.__dict__ and "_ckpt_key" not in lokr.__dict__ for lokr in laf._all_loras(net))
    loss.backward()
    assert counts == {"nograd": adapted, "grad": adapted}
    # the node got a stand-in, never a block: no per-block module list was built on the real blocks
    assert len(seen["node"]) == LAYERS and seen["plain"] == 2  # the two text refiner blocks keep the plain checkpoint
    assert all(not isinstance(block, moss_models.MossBlock) for _, block in seen["node"])
    assert all(len(list(block.modules())) == 1 for _, block in seen["node"])
    assert all("_ckpt_lokrs" not in block.__dict__ for block in model.layers)
    joint = [entry for entry in passes if entry[1]]
    assert joint == [(0, ["ckpt_reuse", "ckpt_stash"])] * (2 * LAYERS)
    assert lycoris_fast._CKPT_PASS == 0 and lycoris_fast.ckpt_merged_bytes(model) == 0


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
def test_reuse_only_never_reads_a_stale_cache(compiled, monkeypatch):
    """A cache left behind by --moss_ckpt_nograd_merged earlier in the process (only tests switch modes; the trainer's
    flags are fixed) and poisoned: the cached mode reads it, reuse-only does not - its output and gradients are the
    plain checkpoint's - and it leaves the stale tensors alone."""
    model, net, _ = base.build(compiled=compiled)
    data = base.batch()
    set_flags(monkeypatch, False, False)
    ref = base.micro_step(model, net, data)
    set_flags(monkeypatch, True, True)
    base.assert_same(ref, base.micro_step(model, net, data), "cached mode")
    tensors = [lokr.__dict__["_ckpt_merged"] for lokr in base.cached_lokrs(model)]
    assert len(tensors) == (LAYERS - 1) * PER_BLOCK
    with torch.no_grad():
        for cached in tensors:
            cached.mul_(1.25)
    assert not torch.equal(base.micro_step(model, net, data)[0], ref[0]), "the cached mode did not read the cache"
    set_flags(monkeypatch, False, True)
    base.assert_same(ref, base.micro_step(model, net, data), "reuse-only with a poisoned cache on the modules")
    assert set(anima_models.CKPT_RECOMPUTE_DIFFS) == {0.0}
    assert all(a is b for a, b in zip(tensors, (lokr.__dict__["_ckpt_merged"] for lokr in base.cached_lokrs(model))))


def test_reuse_only_parameters_changed_before_backward_recompute_plainly(monkeypatch):
    """A parameter written between a forward and its backward (nothing in the trainer does this): without a cache the
    node has no keys of its own, so the reuse-only function checks the adapter itself and the recompute runs the
    attention plainly on the new values - gradients equal to torch.utils.checkpoint's; the block left out of the network
    has nothing that could change and still reuses."""
    results = []
    for reuse in (False, True):
        model, net, _ = base.build()
        set_flags(monkeypatch, False, reuse)
        base.micro_step(model, net, base.batch())
        net.zero_grad(set_to_none=True)
        before = fake_flash.CALLS["fwd"]
        y = base.forward(model, base.batch(seed=2))
        with torch.no_grad():
            for p in net.parameters():
                p.mul_(1.01)  # bumps the version counters
        y.float().square().mean().backward()
        results.append((y.detach().float().clone(), {n: p.grad.clone() for n, p in net.named_parameters()},
                        fake_flash.CALLS["fwd"] - before))
    (y0, g0, fwd0), (y1, g1, fwd1) = results
    base.assert_same((y0, g0), (y1, g1), "parameters changed between forward and backward, reuse-only")
    assert (fwd0, fwd1) == (2 * (2 + LAYERS), 2 * 2 + LAYERS + (LAYERS - 1))
    diffs = anima_models.CKPT_RECOMPUTE_DIFFS[-LAYERS:]  # backward visits the blocks last to first
    assert diffs[LAYERS - 1 - EXCLUDED] == 0.0 and all(d > 0.0 for i, d in enumerate(diffs) if i != LAYERS - 1 - EXCLUDED)
    assert lycoris_fast.ckpt_merged_bytes(model) == 0 and all("_ckpt_lokrs" not in b.__dict__ for b in model.layers)


def test_reuse_only_stored_tensors_are_released_after_backward(monkeypatch):
    model, net, _ = base.build()
    set_flags(monkeypatch, False, True)
    refs = []
    real = attention._flash_forward_with_lse

    def spy(q, k, v):
        out, lse = real(q, k, v)
        refs.append((weakref.ref(out), weakref.ref(lse)))
        return out, lse

    monkeypatch.setattr(attention, "_flash_forward_with_lse", spy)
    net.zero_grad(set_to_none=True)
    y = base.forward(model, base.batch())
    loss = y.float().square().mean()
    assert len(refs) == LAYERS and all(o() is not None and l() is not None for o, l in refs)
    loss.backward()
    del y, loss
    gc.collect()
    assert all(o() is None and l() is None for o, l in refs), "a stored attention output outlived its block's backward"


def test_reuse_only_sampling_and_validation_run_the_eager_blocks(monkeypatch):
    """Sample images (CFG, a skipped block) and a validation forward under reuse-only: no node, no stash, the same
    results as with the switches off."""
    model, net, _ = base.build()
    data = base.batch()

    def validation():
        model.eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(*data).float().clone()
        model.train()
        return out

    set_flags(monkeypatch, False, False)
    ref_sample, ref_val = base.sample(model), validation()
    set_flags(monkeypatch, False, True)
    base.micro_step(model, net, data)
    seen = spy_nodes(monkeypatch)
    assert torch.equal(base.sample(model), ref_sample) and torch.equal(validation(), ref_val)
    assert seen == {"node": [], "plain": 0}


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
def test_reuse_only_plain_lora_and_text_refiner_training(compiled, monkeypatch):
    """networks.lora_moss (no LyCORIS module at all) and preset moss-text (LoKr on the text refiner, whose blocks keep
    torch.utils.checkpoint): gradients bit-identical under reuse-only."""
    batches = (dict(text=(6,)), dict(text=(9,), seed=2))
    for kw in (dict(module="networks.lora_moss"), dict(preset="moss-text")):
        ref, *_ = run(monkeypatch, False, False, batches=batches, compiled=compiled, **kw)
        new, model, net, _ = run(monkeypatch, False, True, batches=batches, compiled=compiled, **kw)
        for i, (a, b) in enumerate(zip(ref, new)):
            base.assert_same(a, b, f"{kw}, micro-step {i}")
        assert lycoris_fast.ckpt_merged_bytes(model) == 0 and set(anima_models.CKPT_RECOMPUTE_DIFFS) == {0.0}


@pytest.mark.parametrize("stream", [False, True], ids=["bf16_stream", "fp32_stream"])
def test_reuse_only_trained_time_conditioning_gets_the_same_gradients(stream, monkeypatch):
    results = []
    for flags in ((False, False), (False, True)):
        model, net, _ = base.build(stream=stream)
        extra = [model.shared_block_modulation.weight, model.time_embedding[0].weight, model.time_embedding[2].bias]
        for p in extra:
            p.requires_grad_(True)
        set_flags(monkeypatch, *flags)
        results.append([base.micro_step(model, net, base.batch(stream=stream, **b), extra=extra)
                        for b in (dict(text=(6,)), dict(text=(5, 9), seed=2))])
    for i, (a, b) in enumerate(zip(*results)):
        base.assert_same(a, b, f"trained time conditioning under reuse-only, micro-step {i}")


# ------------------------------------------------------------------------------------------------------------ budget


def test_budget_block_count():
    """n = floor(budget / (joint tokens * rows * bytes per token and block)): 0 blocks, some, all, and no budget."""
    count = moss_models.ckpt_reuse_block_count
    assert moss_models.ckpt_stash_bytes_per_token(3072, 24) == 6240  # out [L, 3072] 2 bytes + softmax_lse [24, L] fp32
    assert moss_models.ckpt_stash_bytes_per_token(TINY["dim"], TINY["num_heads"]) == PER_TOKEN == 400
    three = 3 * GIB
    # the user's 2048 tier: 16,384 image tokens + captions; worst sample of the dataset 17,041 joint tokens
    assert count(three, 16384, 1, 6240) == 31
    assert count(three, 16384 + 286, 1, 6240) == 30
    assert count(three, 17041, 1, 6240) == 30
    assert count(three, 9216 + 286, 1, 6240) == 54  # 1536 tier: more than the model has - every block (callers clip)
    assert count(three, 4096 + 286, 1, 6240) == 117
    assert count(three, 16384, 2, 6240) == 15  # two rows store twice as much per block
    assert count(0, 16384, 1, 6240) == 0
    assert count(16384 * 6240 - 1, 16384, 1, 6240) == 0  # one byte short of one block
    assert count(16384 * 6240, 16384, 1, 6240) == 1
    assert count(42 * 17041 * 6240, 17041, 1, 6240) == 42  # exactly all blocks
    assert count(42 * 17041 * 6240 - 1, 17041, 1, 6240) == 41
    assert count(None, 17041, 1, 6240) >= 10**9  # no budget: every block
    assert count(three, 0, 1, 6240) >= 10**9 and count(three, 16384, 0, 6240) >= 10**9  # nothing to store
    with pytest.raises(ValueError):
        count(-1, 16384, 1, 6240)


def test_block_index_and_token_bytes_are_stored_at_build_time():
    model, net, _ = base.build(layers=5, exclude=())
    assert [block._ckpt_index for block in model.layers] == [0, 1, 2, 3, 4]
    assert {block._ckpt_stash_per_token for block in model.layers} == {PER_TOKEN}
    assert not any("ckpt" in key for key in list(model.state_dict()) + list(net.state_dict()))
    real = moss_models.MossBlock(3072, 16, 24)  # the released model's width and heads (a standalone block: index 0)
    assert real._ckpt_stash_per_token == 6240 and real._ckpt_index == 0


def test_budget_setter_takes_gib_and_reports_bytes(monkeypatch):
    set_flags(monkeypatch, False, False)
    moss_models.set_ckpt_reuse(False, True, 3.0)
    assert (anima_models.CKPT_NOGRAD_MERGED, anima_models.CKPT_REUSE_ATTN) == (False, True)
    assert moss_models.ckpt_node_enabled() and not moss_models.ckpt_nograd_merged_enabled()
    assert moss_models.ckpt_reuse_budget_bytes() == 3 * GIB
    moss_models.set_ckpt_reuse(True, True)
    assert moss_models.ckpt_node_enabled() and moss_models.ckpt_nograd_merged_enabled()
    assert moss_models.ckpt_reuse_budget_bytes() is None
    moss_models.set_ckpt_reuse(True, False, 0.5)  # a budget without the reuse: kept, changes nothing
    assert moss_models.ckpt_node_enabled() and moss_models.ckpt_reuse_budget_bytes() == GIB // 2
    moss_models.set_ckpt_reuse(False, False)
    assert not moss_models.ckpt_node_enabled() and moss_models.ckpt_reuse_budget_bytes() is None
    with pytest.raises(ValueError):
        moss_models.set_ckpt_reuse(False, True, -0.1)
    with pytest.raises(ValueError):
        moss_models.set_ckpt_reuse(False, True, float("nan"))


# n for the toy model (400 bytes per token and block) under 20,000 bytes: joint tokens * rows -> blocks
MIXED = (dict(),                                    # 22 tokens: 8,800 B per block -> 2 of 3 blocks
         dict(text=(11,), hw=(4, 6), seed=2),       # 35: 14,000 -> 1
         dict(text=(15,), hw=(6, 6), seed=3),       # 51: 20,400 -> 0 (all plain)
         dict(text=(2,), hw=(2, 2), seed=4),        # 6: 2,400 -> 8, i.e. all 3
         dict(text=(7, 7), seed=5),                 # 2 rows x 23: 18,400 -> 1
         dict(text=(13,), hw=(6, 6), seed=6),       # 49: 19,600 -> 1
         dict(text=(4,), seed=7))                   # 20: 8,000 -> 2
MIXED_BUDGET = 20000
MIXED_N = [2, 1, 0, 3, 1, 1, 2]


def test_mixed_plan_covers_every_block_count():
    assert [expected_n(spec, MIXED_BUDGET) for spec in MIXED] == MIXED_N
    assert [moss_models.ckpt_reuse_block_count(MIXED_BUDGET, joint_tokens(s), len(s.get("text", (6,))), PER_TOKEN)
            for s in MIXED] == [2, 1, 0, 8, 1, 1, 2]


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
@pytest.mark.parametrize("stream", [False, True], ids=["bf16_stream", "fp32_stream"])
@pytest.mark.parametrize("nograd", [False, True], ids=["reuse_only", "with_cache"])
def test_mixed_mode_matches_the_plain_checkpoint(nograd, stream, compiled, monkeypatch):
    """A budget that lets 2, 1, 0, 3, 1, 1, 2 of the 3 joint blocks reuse over consecutive micro-steps: output and
    gradients equal the plain checkpoint's bit for bit, the blocks with index < n (and only they) store one attention
    output each, and the others take torch.utils.checkpoint (reuse-only) or the node without a stash (with the cache)."""
    ref, *_ = run(monkeypatch, False, False, batches=MIXED, compiled=compiled, stream=stream)
    model, net, state = base.build(compiled=compiled, stream=stream)
    set_flags(monkeypatch, nograd, True, MIXED_BUDGET)
    seen = spy_nodes(monkeypatch)
    stored = []
    real = attention._flash_forward_with_lse
    if not compiled:  # (a compiled graph calls the kernel op directly; the kernel counters below cover it)
        monkeypatch.setattr(attention, "_flash_forward_with_lse", lambda q, k, v: stored.append(1) or real(q, k, v))
    for i, (spec, n) in enumerate(zip(MIXED, MIXED_N)):
        seen["node"].clear()
        seen["plain"] = 0
        stored.clear()
        fake_flash.CALLS["fwd"] = fake_flash.CALLS["bwd"] = 0
        new = base.micro_step(model, net, base.batch(stream=stream, **spec))
        base.assert_same(ref[i], new, f"mixed mode, micro-step {i} (n = {n}, cache {nograd})")
        nodes = LAYERS if nograd else n
        assert len(seen["node"]) == nodes and seen["plain"] == 2 + LAYERS - nodes, (i, seen)
        if not compiled:
            assert len(stored) == n, (i, len(stored))
        # per block: reuse = 1 forward kernel, otherwise 2 (first pass + recompute); the text refiner's 2 x 2 on top
        assert fake_flash.CALLS == {"fwd": 2 * 2 + n + 2 * (LAYERS - n), "bwd": 2 + LAYERS}, (i, fake_flash.CALLS)
    assert set(anima_models.CKPT_RECOMPUTE_DIFFS) == {0.0}
    assert (lycoris_fast.ckpt_merged_bytes(model) > 0) == nograd
    if compiled:
        assert state.eager_calls == 0


def test_budget_decision_reads_no_device_value(monkeypatch):
    """The per-block decision uses the shape ints of x only, and the reuse-only function's adapter check reads version
    counters and data pointers: a training forward + backward reads exactly as many tensor values on the host with the
    plain checkpoint, with reuse-only and with a budget (MossTransformer.forward's one text-length transfer)."""
    reads = {}
    for name, reuse, budget in (("plain", False, None), ("all", True, None), ("budget", True, MIXED_BUDGET)):
        model, net, _ = base.build()
        set_flags(monkeypatch, False, reuse, budget, diffs=False)
        data = base.batch(**MIXED[1])
        net.zero_grad(set_to_none=True)
        with _device_reads.count_device_reads() as rec:
            base.forward(model, data).float().square().mean().backward()
        reads[name] = list(rec.ops)
    assert reads["budget"] == reads["all"] == reads["plain"], reads
    assert len(reads["plain"]) <= 2, reads  # the text lengths, once per forward


@pytest.mark.parametrize("make", ["came", "adamw"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
def test_training_runs_match_the_plain_checkpoint(compiled, make, came_cls, monkeypatch):
    """3 optimizer steps x 3 accumulated micro-steps driven like train_network.py (clipping, optimizer.step, the
    trainer's on_optimizer_step_end; one sample, an equal-length batch, a padded batch): reuse-only, with and without a
    budget that changes n from micro-step to micro-step, trains to the plain checkpoint's parameters bit for bit."""
    opt = (lambda params: base.make_came(came_cls, params)) if make == "came" else base.adamw
    ref = base.simulate(monkeypatch, (False, False), make_opt=opt, compiled=compiled)
    for budget in (None, 12000):  # 12,000 B: n = 1 (22 tokens), 0 (35 and the two-row batches), 1 (20), ...
        monkeypatch.setattr(moss_models, "CKPT_REUSE_BUDGET_BYTES", budget)
        new = base.simulate(monkeypatch, (False, True), make_opt=opt, compiled=compiled)
        base.assert_same_run(ref, new, f"reuse-only training, budget {budget}, {make}, compiled={compiled}")
        assert lycoris_fast.ckpt_merged_bytes(new[2]["model"]) == 0
    assert lycoris_fast._CKPT_STEP_HOOK is None


# ------------------------------------------------------------------------------------------------------ torch.compile


def test_mixed_mode_graph_count_stays_flat_when_n_changes(monkeypatch):
    """Block compile + a budget: the 42 (here 3) blocks share one compiled entry, so which blocks reuse is not part of
    any graph. The kinds of call are: the plain checkpoint's grad forward, the node's no-grad first pass (a stash list)
    and its recompute (the stored tensors) - each for an adapted block and for the block left out of the network: 6
    graphs at most, however n moves between micro-steps and whatever the caption length / image size. Far below
    --moss_block_compile_recompile_limit (64, at least 8), and no eager fallback."""
    counter = base.Counter()
    model, net, state = base.build(compiled=True, backend=counter)
    set_flags(monkeypatch, False, True, MIXED_BUDGET)
    one_sample = tuple(spec for spec in MIXED if len(spec.get("text", (6,))) == 1)  # the user's batch size is 1
    one_n = [n for spec, n in zip(MIXED, MIXED_N) if len(spec.get("text", (6,))) == 1]
    plan = one_sample + tuple(dict(spec, seed=spec.get("seed", 1) + 10) for spec in one_sample) + (
        dict(text=(9,), hw=(4, 6), seed=30), dict(text=(17,), hw=(6, 6), seed=31), dict(text=(3,), seed=32))
    counts, ns = [], []
    seen = spy_nodes(monkeypatch)
    for spec in plan:
        seen["node"].clear()
        base.micro_step(model, net, base.batch(**spec))
        counts.append(counter.graphs)
        ns.append(len(seen["node"]))
    assert ns == one_n * 2 + [1, 0, 2] and set(ns) == {0, 1, 2, 3}
    assert state.eager_calls == 0
    assert max(counts) <= 6, counts
    # every kind of call has been seen once n has been 2 (blocks 0 and 1: adapted + left out, node), 1 and 0 (plain
    # adapted and left out): from the third micro-step on nothing compiles any more
    assert counts[2:] == [counts[2]] * (len(plan) - 2), counts
    assert counts[-1] < 8  # the smallest --moss_block_compile_recompile_limit the trainer accepts


def test_reuse_only_graph_count_without_a_budget(monkeypatch):
    """Reuse-only, compiled, no budget: first pass + recompute for the adapted blocks and for the left-out block (4
    graphs, like the cached mode), flat across caption lengths, image sizes and optimizer steps."""
    counter = base.Counter()
    counts = []
    new = base.simulate(monkeypatch, (False, True), make_opt=base.adamw, compiled=True, backend=counter,
                        plan=base.SAMPLE_PLAN, between=lambda i, model, net, trainer: counts.append(counter.graphs))
    assert new[2]["state"].eager_calls == 0
    assert counts == [4] * len(base.SAMPLE_PLAN), counts


# ---------------------------------------------------------------------------------------------------- the trainer glue


def _args(**changes):
    base_args = dict(moss_ckpt_nograd_merged=False, moss_ckpt_reuse_attn=True, moss_ckpt_reuse_attn_budget_gib=None,
                     gradient_checkpointing=True, attn_mode="flash", split_attn=False,
                     unsloth_offload_checkpointing=False, cpu_offload_checkpointing=False, blocks_to_swap=None,
                     use_ramtorch=False, use_ramtorch_network=False, moss_stock_lycoris=False, train_batch_size=1)
    base_args.update(changes)
    return types.SimpleNamespace(**base_args)


def test_validation_accepts_reuse_only(monkeypatch, warned):
    monkeypatch.setattr(anima_models, "CKPT_REUSE_ATTN", True)  # prepare_moss_runtime ran
    validate = moss_train_utils.validate_moss_ckpt_reuse_args
    validate(_args())
    validate(_args(moss_ckpt_reuse_attn_budget_gib=3.0))
    validate(_args(moss_ckpt_reuse_attn_budget_gib=0))  # no block reuses: allowed (the start-up line says so)
    validate(_args(moss_ckpt_reuse_attn_budget_gib=3))  # a TOML integer
    assert warned == []
    old = base._args()  # the test module's older namespace: no budget attribute at all
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", True)
    validate(old)
    assert warned == []


@pytest.mark.parametrize("change, word", [
    (dict(unsloth_offload_checkpointing=True), "unsloth_offload_checkpointing"),
    (dict(cpu_offload_checkpointing=True), "cpu_offload_checkpointing"),
    (dict(blocks_to_swap=4), "blocks_to_swap"),
    (dict(use_ramtorch=True), "RamTorch"),
    (dict(use_ramtorch_network=True), "RamTorch"),
    (dict(moss_stock_lycoris=True), "moss_stock_lycoris"),
    (dict(attn_mode="torch"), "flash"),
    (dict(attn_mode=None), "flash"),
    (dict(split_attn=True), "split_attn"),
    (dict(moss_ckpt_reuse_attn_budget_gib=-1.0), "moss_ckpt_reuse_attn_budget_gib"),
    (dict(moss_ckpt_reuse_attn_budget_gib=float("nan")), "moss_ckpt_reuse_attn_budget_gib"),
    (dict(moss_ckpt_reuse_attn_budget_gib=float("inf")), "moss_ckpt_reuse_attn_budget_gib"),
    (dict(moss_ckpt_reuse_attn_budget_gib="3"), "moss_ckpt_reuse_attn_budget_gib"),
])
def test_validation_keeps_refusing_in_reuse_only(change, word):
    with pytest.raises(ValueError, match=word) as error:
        moss_train_utils.validate_moss_ckpt_reuse_args(_args(**change))
    if "budget" not in word and word not in ("flash", "split_attn"):
        assert "--moss_ckpt_reuse_attn" in str(error.value)  # the message names the option that was given


def test_validation_refuses_a_bad_budget_with_the_cache_too():
    with pytest.raises(ValueError, match="moss_ckpt_reuse_attn_budget_gib"):
        moss_train_utils.validate_moss_ckpt_reuse_args(_args(moss_ckpt_nograd_merged=True,
                                                             moss_ckpt_reuse_attn_budget_gib=-2))
    with pytest.raises(ValueError, match="moss_ckpt_reuse_attn_budget_gib"):  # also with both options off
        moss_train_utils.validate_moss_ckpt_reuse_args(_args(moss_ckpt_reuse_attn=False,
                                                             moss_ckpt_reuse_attn_budget_gib=-2))


def test_validation_warnings_in_reuse_only(monkeypatch, warned):
    validate = moss_train_utils.validate_moss_ckpt_reuse_args
    validate(_args())  # the runtime switch is off (no prepare_moss_runtime)
    assert len(warned) == 1 and "--moss_ckpt_reuse_attn has no effect here" in warned[0]
    warned.clear()
    monkeypatch.setattr(anima_models, "CKPT_REUSE_ATTN", True)
    validate(_args(gradient_checkpointing=False))
    assert len(warned) == 1 and "--moss_ckpt_reuse_attn changes nothing without --gradient_checkpointing" in warned[0]
    warned.clear()
    validate(_args(train_batch_size=4))
    assert len(warned) == 1 and "--moss_ckpt_reuse_attn with batch > 1 (largest batch size 4)" in warned[0]
    warned.clear()
    # a budget without the reuse: nothing to cap
    validate(_args(moss_ckpt_reuse_attn=False, moss_ckpt_reuse_attn_budget_gib=3.0))
    assert len(warned) == 1 and "--moss_ckpt_reuse_attn_budget_gib changes nothing without --moss_ckpt_reuse_attn" in warned[0]
    warned.clear()
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", True)
    validate(_args(moss_ckpt_nograd_merged=True, moss_ckpt_reuse_attn=False, attn_mode="torch",
                   moss_ckpt_reuse_attn_budget_gib=3.0))
    assert len(warned) == 1 and "changes nothing without --moss_ckpt_reuse_attn" in warned[0]


def test_parser_toml_and_runtime_switches(monkeypatch, tmp_path):
    """--moss_ckpt_reuse_attn_budget_gib: a float option, default None (no limit), readable from a config TOML like the
    other moss_ckpt keys; prepare_moss_runtime turns reuse-only into the node switch without the cache switch, sets the
    budget and registers no optimizer-step hook (there is no cache to invalidate)."""
    import moss_train_network as mtn

    parser = mtn.setup_parser()
    parser.format_help()
    off = parser.parse_args([])
    assert off.moss_ckpt_reuse_attn_budget_gib is None
    cli = parser.parse_args(["--moss_ckpt_reuse_attn", "--moss_ckpt_reuse_attn_budget_gib", "3", "--attn_mode", "flash"])
    assert cli.moss_ckpt_reuse_attn is True and cli.moss_ckpt_nograd_merged is False
    assert cli.moss_ckpt_reuse_attn_budget_gib == 3.0 and isinstance(cli.moss_ckpt_reuse_attn_budget_gib, float)

    for text, value in (("moss_ckpt_reuse_attn_budget_gib = 2.5\n", 2.5), ("moss_ckpt_reuse_attn_budget_gib = 3\n", 3)):
        config = tmp_path / f"config_{value}.toml"
        config.write_text('attn_mode = "flash"\nmoss_ckpt_reuse_attn = true\n' + text, encoding="utf-8")
        # read_config_from_file parses sys.argv again on top of the TOML values (the command line wins)
        monkeypatch.setattr(sys, "argv", ["moss_train_network.py", "--config_file", str(config)])
        args = train_util.read_config_from_file(parser.parse_args(), parser, unknown_keys="error")
        assert args.moss_ckpt_reuse_attn is True and args.moss_ckpt_nograd_merged is False
        assert args.moss_ckpt_reuse_attn_budget_gib == value and args.attn_mode == "flash"

    monkeypatch.setattr(attention, "is_flash_attn_supported", lambda: True)
    chains, rebuild = moss_models._CHAINS.enabled, lycoris_fast._REBUILD.enabled
    infos = []
    monkeypatch.setattr(mtn.logger, "info", lambda message, *a, **k: infos.append(str(message)))
    try:
        mtn.prepare_moss_runtime(cli)
        assert anima_models.CKPT_NOGRAD_MERGED is False and anima_models.CKPT_REUSE_ATTN is True
        assert moss_models.ckpt_node_enabled() and not moss_models.ckpt_nograd_merged_enabled()
        assert moss_models.ckpt_reuse_budget_bytes() == 3 * GIB
        assert lycoris_fast._CKPT_STEP_HOOK is None
        line = [m for m in infos if "no-grad first pass" in m]
        assert len(line) == 1 and "nothing is cached" in line[0] and "3.00 GiB" in line[0]
        mtn.prepare_moss_runtime(off)
        assert not moss_models.ckpt_node_enabled() and moss_models.ckpt_reuse_budget_bytes() is None
        both = parser.parse_args(["--moss_ckpt_nograd_merged", "--moss_ckpt_reuse_attn", "--attn_mode", "flash"])
        mtn.prepare_moss_runtime(both)
        assert moss_models.ckpt_nograd_merged_enabled() and moss_models.ckpt_reuse_budget_bytes() is None
        assert lycoris_fast._CKPT_STEP_HOOK is not None
    finally:
        moss_models._CHAINS.enable(chains)
        lycoris_fast.enable_compiled_rebuild(rebuild)
        base._remove_step_hook()


def test_trainer_hooks_in_reuse_only(monkeypatch):
    """Reuse-only: nothing to invalidate after an optimizer step or a sample round (no cache), but the VRAM peak is
    watched like with the cache, and the trained-time-conditioning note applies (it is about the node)."""
    import moss_train_network as mtn

    trainer = mtn.MossNetworkTrainer()
    watched = []
    monkeypatch.setattr(trainer, "_watch_ckpt_vram", lambda report=False: watched.append(report))
    monkeypatch.setattr(mtn.moss_train_utils, "sample_images", lambda *a, **k: None)
    set_flags(monkeypatch, False, True)
    before = lycoris_fast._CKPT_GENERATION
    trainer.on_optimizer_step_end(argparse.Namespace(), None, 1, False)
    trainer.sample_images(None, argparse.Namespace(), 0, 1, "cpu", None, None, None, None)
    trainer.on_epoch_end(argparse.Namespace(), types.SimpleNamespace(unwrap_model=lambda m: m), 0, 1)
    assert lycoris_fast._CKPT_GENERATION == before and watched == [False, True]
    assert trainer._ckpt_merged_logged is False  # the cache size line belongs to --moss_ckpt_nograd_merged

    class Net:
        def named_parameters(self):
            return [("lora_unet_time_embedding_0.lokr_w1", nn.Parameter(torch.zeros(1)))]

    note = moss_train_utils.ckpt_reuse_outside_grad_note(Net())
    assert note is not None and "--moss_ckpt_reuse_attn" in note and "time_embedding" in note
    set_flags(monkeypatch, False, False)
    assert moss_train_utils.ckpt_reuse_outside_grad_note(Net()) is None


# ---------------------------------------------------------------------------------------------- estimate and log lines


def _dataset(batch_size, *resos):
    image_data = {f"image{i}": types.SimpleNamespace(bucket_reso=reso, jitter_bucket_info=None)
                  for i, reso in enumerate(resos)}
    return types.SimpleNamespace(batch_size=batch_size, image_data=image_data)


def test_dataset_group_tiers_keep_the_largest_bucket_of_every_tier():
    tiers = moss_train_utils.dataset_group_tiers
    group = types.SimpleNamespace(datasets=[
        _dataset(1, (1024, 1024), (832, 1216), (512, 512)), _dataset(1, (2048, 2048), (1536, 2816)),
        _dataset(1, (1536, 1536), (1280, 1856))])
    assert tiers(group) == [("512", 32 * 32, (512, 512)), ("1024", 64 * 64, (1024, 1024)),
                            ("1536", 80 * 116, (1280, 1856)), ("2048", 96 * 176, (1536, 2816))]
    assert tiers(None) == tiers(object()) == tiers(types.SimpleNamespace(datasets=[object()])) == []


def test_vram_estimate_reports_the_stash_per_tier_under_the_budget(monkeypatch):
    """The start-up numbers with a budget: per tier the bytes one block stores, how many blocks reuse (captions of 0 and
    of 1,000 tokens) and what they hold; for the largest bucket the same in `blocks` / `stash`. Checked against what a
    micro-step of the toy model really stores, and with the real model's dimensions (3.0 GiB: 31 of 42 blocks at a
    2048^2 image, 29 with a 1,000-token caption; every block at 1536^2 and below)."""
    stored = []
    real = attention._flash_forward_with_lse

    def spy(q, k, v):
        out, lse = real(q, k, v)
        stored.append(out.numel() * out.element_size() + lse.numel() * lse.element_size())
        return out, lse

    monkeypatch.setattr(attention, "_flash_forward_with_lse", spy)
    estimate = moss_train_utils.ckpt_reuse_vram_estimate
    for spec, n in zip(MIXED, MIXED_N):
        model, net, _ = base.build()
        set_flags(monkeypatch, False, True, MIXED_BUDGET)
        hw, text = spec.get("hw", (4, 4)), spec.get("text", (6,))
        est = estimate(model, True, hw[0] * hw[1], batch_size=len(text), cached=False, budget_bytes=MIXED_BUDGET,
                       caption_tokens=max(text))
        assert all("_ckpt_lokrs" not in block.__dict__ for block in model.layers)
        stored.clear()
        base.micro_step(model, net, base.batch(**spec))
        assert est["cache"] == 0 == lycoris_fast.ckpt_merged_bytes(model)
        assert est["blocks"] == n == len(stored) and est["stash"] == sum(stored) <= MIXED_BUDGET
        assert est["budget"] == MIXED_BUDGET and est["total_blocks"] == LAYERS

    real_size = types.SimpleNamespace(layers=[nn.Identity() for _ in range(42)], config=moss_models.MOSS_DIT_CONFIG)
    tiers = [("512", 1024, (512, 512)), ("1024", 4096, (1024, 1024)), ("1536", 9216, (1536, 1536)),
             ("2048", 16384, (2048, 2048))]
    est = estimate(real_size, True, 16384, cached=False, budget_bytes=3 * GIB, tiers=tiers)
    assert est["blocks"] == 31 and est["stash"] == 31 * 16384 * 6240 and est["cache"] == 0
    assert [(t["tier"], t["blocks"], t["blocks_long"]) for t in est["tiers"]] == [
        ("512", 42, 42), ("1024", 42, 42), ("1536", 42, 42), ("2048", 31, 29)]
    assert [t["per_block"] for t in est["tiers"]] == [tokens * 6240 for _, tokens, _ in tiers]
    assert [t["stash"] for t in est["tiers"]] == [42 * 1024 * 6240, 42 * 4096 * 6240, 42 * 9216 * 6240, 31 * 16384 * 6240]
    unlimited = estimate(real_size, True, 16384, cached=False, tiers=tiers)
    assert unlimited["budget"] is None and unlimited["blocks"] == 42 and unlimited["stash"] == 42 * 16384 * 6240
    assert [t["blocks"] for t in unlimited["tiers"]] == [42] * 4
    no_reuse = estimate(real_size, False, 16384, cached=False, budget_bytes=3 * GIB, tiers=tiers)
    assert no_reuse["stash"] == 0 and no_reuse["blocks"] == 0 and all(t["stash"] == 0 for t in no_reuse["tiers"])


def test_vram_notes_for_reuse_only_and_the_budget():
    """The start-up line names what is on: without the cache it does not talk about one; with a budget it says how many
    blocks reuse per tier; tight when weights + kept inputs + the stash pass 80 % of the card. The peak line (the 94 %
    warning) names the options that are on and the budget as the knob."""
    real_size = types.SimpleNamespace(layers=[nn.Identity() for _ in range(42)], config=moss_models.MOSS_DIT_CONFIG)
    tiers = [("1024", 4096, (1024, 1024)), ("2048", 16384, (2048, 2048))]
    card = dict(total=int(31.84 * GIB), allocated=int(9.6 * GIB))
    estimate, note = moss_train_utils.ckpt_reuse_vram_estimate, moss_train_utils.ckpt_reuse_vram_note
    est = estimate(real_size, True, 16384, cached=False, budget_bytes=3 * GIB, tiers=tiers, stream_bytes=4)
    line, tight = note(est, (2048, 2048), card)
    assert "merged-weight cache" not in line and "--moss_ckpt_reuse_attn" in line
    assert "2.95 GiB stored joint attention at the largest bucket 2048x2048" in line
    assert "31 of 42 blocks" in line and "budget 3.00 GiB" in line
    assert "1024 (1024x1024): 42 of 42 blocks, 1.00 GiB" in line
    assert "2048 (2048x2048): 31 of 42 blocks (29 with a 1000-token caption), 2.95 GiB" in line
    assert "This card: 31.84 GiB, 9.60 GiB allocated now" in line and not tight
    est = estimate(real_size, True, 16384, cached=False, tiers=tiers, stream_bytes=4)
    line, tight = note(est, (2048, 2048), dict(card, allocated=int(14.0 * GIB)))
    assert "4.00 GiB stored joint attention" in line and "42 of 42 blocks" in line and "no budget" in line
    assert tight and "That is tight" in line and "--moss_ckpt_reuse_attn_budget_gib" in line
    line, tight = note(estimate(real_size, True, 16384, cached=False, budget_bytes=0), None, None)
    assert "unknown largest bucket" in line and "0 of 42 blocks" in line and not tight

    peak = moss_train_utils.ckpt_reuse_peak_note
    total = card["total"]
    line, at_limit = peak(dict(total=total, peak_allocated=22 * GIB, peak_reserved=26 * GIB, free=4 * GIB),
                          options="--moss_ckpt_reuse_attn")
    assert not at_limit and line.startswith("MOSS-Image: --moss_ckpt_reuse_attn VRAM peak so far: 22.00 GiB allocated")
    line, at_limit = peak(dict(total=total, peak_allocated=29 * GIB, peak_reserved=int(0.94 * total) + 1),
                          options="--moss_ckpt_reuse_attn")
    assert at_limit and "at the card's limit" in line and "--moss_ckpt_reuse_attn_budget_gib" in line
    assert not peak(dict(total=total, peak_allocated=29 * GIB, peak_reserved=int(0.93 * total)))[1]


def test_trainer_logs_the_estimate_and_watches_the_peak_in_reuse_only(monkeypatch):
    import moss_train_network as mtn

    lines = {"info": [], "warning": []}
    monkeypatch.setattr(mtn.logger, "info", lambda message, *a, **k: lines["info"].append(str(message)))
    monkeypatch.setattr(mtn.logger, "warning", lambda message, *a, **k: lines["warning"].append(str(message)))
    stats = dict(total=32 * GIB, allocated=1 * GIB, peak_allocated=2 * GIB, peak_reserved=3 * GIB, free=20 * GIB)
    monkeypatch.setattr(mtn.moss_train_utils, "cuda_memory_stats",
                        lambda device, with_free=True: {k: v for k, v in stats.items() if with_free or k != "free"})
    monkeypatch.setattr(mtn.moss_train_utils, "validate_moss_args", lambda *a, **k: None)
    model, net, _ = base.build()
    accelerator = types.SimpleNamespace(unwrap_model=lambda m: m, device=torch.device("cpu"))
    args = argparse.Namespace(moss_block_compile=False, moss_ckpt_nograd_merged=False, moss_ckpt_reuse_attn=True,
                              moss_ckpt_reuse_attn_budget_gib=None, mixed_precision="fp16", moss_stream_fp32=True,
                              moss_text_encoder=None)
    set_flags(monkeypatch, False, True, 6 * 16 * PER_TOKEN)  # as prepare_moss_runtime leaves them: 6 blocks of 16 tokens
    trainer = mtn.MossNetworkTrainer()
    group = types.SimpleNamespace(datasets=[_dataset(1, (64, 64), (32, 32))])
    with pytest.raises(ValueError, match="moss_text_encoder"):  # the next check of assert_extra_args; the limits are set
        trainer.assert_extra_args(args, group, None)
    assert trainer._dataset_limits == (1, 16, (64, 64)) and [t[1] for t in trainer._dataset_tiers] == [16]
    trainer.after_network_loaded(args, accelerator, net, [], model)
    estimate = [line for line in lines["info"] if "estimated extra VRAM" in line]
    assert len(estimate) == 1 and "the largest bucket 64x64" in estimate[0] and "merged-weight cache" not in estimate[0]
    assert f"{LAYERS} of {LAYERS} blocks" in estimate[0] and "This card: 32.00 GiB" in estimate[0]
    trainer.on_optimizer_step_end(argparse.Namespace(), None, 1, False)
    trainer.on_epoch_end(argparse.Namespace(), accelerator, 0, 1)
    peaks = [line for line in lines["info"] if "VRAM peak so far" in line]
    assert len(peaks) == 1 and "--moss_ckpt_reuse_attn VRAM peak" in peaks[0] and "nograd_merged" not in peaks[0]
    assert not any("cached merged LoKr weights" in line for line in lines["info"])
    stats["peak_reserved"] = int(0.95 * 32 * GIB)  # the 94 % warning, once
    trainer.on_optimizer_step_end(argparse.Namespace(), None, 2, False)
    trainer.on_optimizer_step_end(argparse.Namespace(), None, 3, False)
    assert len(lines["warning"]) == 1 and "at the card's limit" in lines["warning"][0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
