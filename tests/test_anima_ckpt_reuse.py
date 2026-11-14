"""CPU tests for --anima_ckpt_nograd_merged and --anima_ckpt_reuse_attn (2026-11-14; both off by default):
library/anima_models.py NoGradFirstPassCheckpointer (a checkpointed Block's first pass under no_grad on the device, the
recompute + backward inside the node's backward), library/lycoris_fast.py's checkpoint merged-weight cache (the first pass
reads each merged LoKr+DoRA weight from a tensor rebuilt once per optimizer step) and library/attention.py's flash
stash / reuse (the recompute returns the first pass's self-attention output and only runs the flash backward kernel).

flash-attn has no CPU kernels, so attn_mode "flash" runs on tests/_fake_flash.py: custom ops with flash's forward /
backward split (the backward reads the STORED output and softmax_lse). The toy Anima, the LyCORIS network and the
micro-step are test_anima_block_compile's (bf16 autocast, LoKr factor 4 + DoRA, preset anima-exp), eager and compiled
with the aot_eager backend (dynamo's trace and AOTAutograd's joint graph as inductor gets them, without codegen).

Checked here:
* flags off: Block.forward calls torch.utils.checkpoint as before, nothing is cached, lycoris_fast._CKPT_PASS stays 0;
* flags on vs off, eager and compiled, bf16 stream and --anima_stream_fp32, all Blocks / every other Block checkpointed,
  --anima_adaln_hoist: the output and every LoKr / DoRA gradient are bit-identical, and the recompute's output equals the
  first pass's bit for bit (anima_models.CKPT_RECOMPUTE_DIFFS);
* the first pass builds no autograd graph and rebuilds no weight once the cache is current: one no-grad rebuild per cached
  module per optimizer step however many micro-steps accumulate, in place into the same tensor; the keys see
  optimizer.step, load_state_dict, a `.data =` swap, the multiplier, the base weight and ckpt_merged_invalidate;
* the cache is read only inside the checkpoint (a poisoned cache changes neither validation, sampling-style no-grad
  forwards nor un-checkpointed Blocks) and the compiled first pass really reads it (a poisoned cache changes the output,
  the recompute still agrees with the first pass: _FirstPassBits);
* a parameter changed between a forward and its backward: the recompute runs plainly (no cached bits, no stored output);
* reuse: one flash forward kernel run per checkpointed Block's self-attention per micro-step instead of two, the backward
  kernel count unchanged; a head dim that flash would pad, dropout and split attention leave the stash empty;
* torch.compile: the first-pass and recompute calls of Block._forward trace whole (no graph break), the Blocks share the
  graphs, a resolution sweep inside the band compiles nothing new (soft and strict marks, the stored tensors marked);
* RNG: gradients through LoKr dropout / rank_dropout / module_dropout match torch.utils.checkpoint's and the caller's RNG
  stream ends where it does; gradients of the Block's other tensor inputs (emb, adaLN-LoRA) match too;
* the real flash-attn (when installed) on fake CUDA tensors, no GPU: flash_attn_func(..., return_attn_probs=True) gives
  the output and softmax_lse the stash keeps, and the backward op's schema takes the reuse path's call;
* the trainer glue: the parser, validate_ckpt_reuse_args, prepare_anima_runtime's switches, the invalidation hooks;
* the 2026-11-14 review fixes: --anima_stock_lycoris is refused, a flash-attn without the separate backward op stores
  nothing (plain recompute, a start-up warning), llm_adapter / text encoder training and scripts that parse the options
  without applying them get a warning.

GPU-only questions (inductor's fused rebuild vs the eager one, the real flash kernels, VRAM, speed) are measured by
tests/manual_test_anima_ckpt_reuse_gpu.py.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_anima_ckpt_reuse.py
"""
import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
import torch._dynamo
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from library import anima_block_compile as bc
from library import anima_models, attention, lycoris_fast
from library import lycoris_algo_fixes as laf
from tests import _fake_flash as fake_flash
from tests import test_anima_block_compile as T

torch.set_num_threads(4)

PRESET = "lokr_dora"  # the production network: LoKr factor 4, full matrix, DoRA on the output axis


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # the order of test_anima_block_compile._env: the fast LoKr+DoRA forward first, the fixed paths on top of it
    lycoris_fast.install_fast_lokr_forward()
    laf.apply_module_patches()
    torch._dynamo.reset()
    bc._warned_oob.clear()
    fake_flash.install(monkeypatch)
    monkeypatch.setitem(T.TOY, "attn_mode", "flash")
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", False)
    monkeypatch.setattr(anima_models, "CKPT_REUSE_ATTN", False)
    monkeypatch.setattr(anima_models, "CKPT_RECOMPUTE_DIFFS", None)
    monkeypatch.setattr(anima_models, "ADALN_HOIST", False)
    with torch.backends.mkldnn.flags(enabled=False):  # oneDNN has no bf16 backward on some CPUs
        yield
    laf.set_block_compile(False)
    assert lycoris_fast._CKPT_PASS == 0, "a checkpoint pass left lycoris_fast._CKPT_PASS set"
    torch._dynamo.reset()


# ------------------------------------------------------------------------------------------------------------- helpers


def set_flags(monkeypatch, nograd: bool, reuse: bool = False):
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", nograd)
    monkeypatch.setattr(anima_models, "CKPT_REUSE_ATTN", reuse)
    monkeypatch.setattr(anima_models, "CKPT_RECOMPUTE_DIFFS", [] if nograd else None)


def perturb(net, seed: int = 7, scale: float = 0.02) -> None:
    """LoKr starts with lokr_w2 = 0 (a zero delta: lokr_w1 would get no gradient and a cached weight would just be the
    DoRA-normalised base): move every trained parameter so each one matters to the output and the gradients."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            if p.requires_grad and p.is_floating_point():
                p.add_(torch.randn(p.shape, generator=g).to(p.dtype) * scale)


def build(num_blocks: int = 2, compiled: bool = False, stream: bool = False, half: bool = False, preset: str = PRESET):
    """The toy DiT (train mode, checkpointing on: every Block, or every other one like the trainer's 1024 tier) with the
    network applied; compiled: compile_blocks as the trainer calls it."""
    unet, net, _ = T.build(preset, num_blocks=num_blocks, block_compile=compiled)
    perturb(net)
    if stream:
        unet.enable_stream_fp32()
    unet.enable_gradient_checkpointing()
    if half:
        for i, block in enumerate(unet.blocks):
            if i % 2:
                block.disable_gradient_checkpointing()
    state = T.compile_toy(unet, net) if compiled else None
    return unet, net, state


def batch(stream: bool = False, latent_hw=(8, 8), seed: int = 1):
    x, t, ctx, pad = T.inputs(latent_hw=latent_hw, t=[0.3], seed=seed)
    if stream:  # --anima_stream_fp32: the trainer keeps the noisy input and the timestep in fp32
        x, t = x.float(), t.float()
    return x, t, ctx, pad


def run(monkeypatch, nograd: bool, reuse: bool = False, steps: int = 1, **kw):
    """`steps` micro-steps of a fresh model under the flags; returns the last (output, grads) and the model."""
    stream = kw.get("stream", False)
    unet, net, state = build(**kw)
    set_flags(monkeypatch, nograd, reuse)
    result = None
    for _ in range(steps):
        result = T.step(unet, net, *batch(stream), checkpoint=True)
    return result, unet, net, state


def cached_lokrs(unet) -> list:
    return [lokr for block in unet.blocks for lokr in anima_models._ckpt_cached_lokrs(block)]


class RebuildSpy:
    """Counts lycoris_fast's eager weight rebuilds by grad mode (the cache builds and the first pass run without grad)."""

    def __init__(self, monkeypatch):
        self.nograd = self.grad = 0
        inner = lycoris_fast._REBUILD.eager["rebuild"]

        def rebuild(*args):
            if torch.is_grad_enabled():
                self.grad += 1
            else:
                self.nograd += 1
            return inner(*args)

        monkeypatch.setitem(lycoris_fast._REBUILD.eager, "rebuild", rebuild)

    def take(self):
        counts = (self.nograd, self.grad)
        self.nograd = self.grad = 0
        return counts


# ---------------------------------------------------------------------------------------------------- the fake kernels


def test_fake_flash_matches_autograd():
    """The CPU stand-in: its forward is softmax attention and its backward (from the stored output and softmax_lse) is
    autograd's gradient; a wrong stored output changes dq / dk, so the reuse tests would see a mix-up."""
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 6, 2, 8, generator=g, dtype=torch.float64).float().requires_grad_(True) for _ in range(3))
    out = fake_flash.flash_attn_func(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    torch.testing.assert_close(out, ref.transpose(1, 2), rtol=1e-5, atol=1e-6)
    w = torch.randn(out.shape, generator=g)
    got = torch.autograd.grad((out * w).sum(), (q, k, v))
    want = torch.autograd.grad((ref.transpose(1, 2) * w).sum(), (q, k, v))
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
    # reuse: the stored output handed back, only the backward kernel runs - same output and gradients
    stored_out, stored_lse = (t.detach() for t in attention._flash_forward_with_lse(q, k, v))
    before = dict(fake_flash.CALLS)
    again = attention._FlashReuse.apply(q, k, v, stored_out, stored_lse)
    grads = torch.autograd.grad((again * w).sum(), (q, k, v))
    assert fake_flash.CALLS["fwd"] == before["fwd"] and fake_flash.CALLS["bwd"] == before["bwd"] + 1
    assert torch.equal(again, out) and all(torch.equal(a, b) for a, b in zip(grads, got))
    stale = attention._FlashReuse.apply(q, k, v, stored_out * 1.5, stored_lse)
    bad = torch.autograd.grad((stale * w).sum(), (q, k, v))
    assert not torch.equal(bad[0], got[0]) and not torch.equal(bad[1], got[1])


# ------------------------------------------------------------------------------------------------------ flags off / on


def test_flags_off_keep_the_plain_checkpoint(monkeypatch):
    """Both switches off (the defaults): Block.forward hands the same arguments to torch.utils.checkpoint, nothing is
    cached, attention gets neither a stash nor a stored output, and lycoris_fast._CKPT_PASS is 0 in every module call."""
    assert anima_models.CKPT_NOGRAD_MERGED is False and anima_models.CKPT_REUSE_ATTN is False
    unet, net, _ = build()
    seen = {"torch": 0, "nograd": 0, "passes": set(), "attn": 0}
    real_ckpt, real_attention = anima_models.torch_checkpoint, attention.attention

    def spy_ckpt(fn, *args, **kwargs):
        seen["torch"] += 1
        assert kwargs == {"use_reentrant": False} and len(args) == 8
        return real_ckpt(fn, *args, **kwargs)

    def spy_nograd(*args):
        seen["nograd"] += 1

    def spy_attention(qkv, attn_params=None):  # the call as it was before the flags existed: no other argument
        seen["attn"] += 1
        seen["passes"].add(lycoris_fast._CKPT_PASS)
        return real_attention(qkv, attn_params=attn_params)

    monkeypatch.setattr(anima_models, "torch_checkpoint", spy_ckpt)
    monkeypatch.setattr(anima_models, "nograd_checkpoint", spy_nograd)
    monkeypatch.setattr(attention, "attention", spy_attention)
    T.step(unet, net, *batch(), checkpoint=True)
    # per Block: self- and cross-attention, in the first pass and in the recompute
    assert seen == {"torch": len(unet.blocks), "nograd": 0, "passes": {0}, "attn": 4 * len(unet.blocks)}
    assert lycoris_fast.ckpt_merged_bytes(unet) == 0
    assert all("_ckpt_merged" not in lokr.__dict__ for lokr in laf._all_loras(net))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
@pytest.mark.parametrize("stream", [False, True], ids=["bf16_stream", "fp32_stream"])
def test_flags_on_match_the_plain_checkpoint(compiled, stream, monkeypatch):
    """Output and every network gradient: torch.utils.checkpoint == the no-grad first pass with cached merged weights ==
    the same with the stored self-attention output, on a first and a second micro-step (the cache built, then reused).
    The recompute's Block outputs equal the first pass's bit for bit."""
    ref, *_ = run(monkeypatch, False, steps=2, compiled=compiled, stream=stream)
    for reuse in (False, True):
        new, unet, net, state = run(monkeypatch, True, reuse, steps=2, compiled=compiled, stream=stream)
        T.assert_same(ref, new, f"nograd first pass (reuse={reuse}) vs torch.utils.checkpoint")
        diffs = anima_models.CKPT_RECOMPUTE_DIFFS
        assert len(diffs) == 2 * len(unet.blocks) and set(diffs) == {0.0}, diffs
        assert lycoris_fast.ckpt_merged_bytes(unet) > 0
        if compiled:
            assert state.eager_calls == 0 and state.compiled_calls == 2 * 2 * len(unet.blocks)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
def test_half_checkpointed_blocks_match(compiled, monkeypatch):
    """The trainer's 1024 tier: even-indexed Blocks checkpointed, odd-indexed ones not (they keep the per-call rebuild
    and the flash forward as before). 4 Blocks, flags on vs off."""
    ref, *_ = run(monkeypatch, False, num_blocks=4, half=True, compiled=compiled)
    new, unet, *_ = run(monkeypatch, True, True, num_blocks=4, half=True, compiled=compiled)
    T.assert_same(ref, new, "every other Block checkpointed")
    assert anima_models.CKPT_RECOMPUTE_DIFFS == [0.0, 0.0]  # Blocks 0 and 2
    cached = [any("_ckpt_merged" in lokr.__dict__ for lokr in anima_models._ckpt_cached_lokrs(b)) for b in unet.blocks]
    assert cached == [True, False, True, False]


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
def test_adaln_hoist_matches(compiled, monkeypatch):
    """--anima_adaln_hoist: the modulations are computed outside the checkpoint and reach the node as three tensor inputs
    (their gradients come back through it); the checkpointed callable is _forward_hoisted, which passes the stash on."""
    ref, *_ = run(monkeypatch, False, compiled=compiled)
    monkeypatch.setattr(anima_models, "ADALN_HOIST", True)
    hoisted_plain, *_ = run(monkeypatch, False, compiled=compiled)
    new, *_ = run(monkeypatch, True, True, compiled=compiled)
    T.assert_same(ref, hoisted_plain, "hoist, plain checkpoint")
    T.assert_same(ref, new, "hoist, no-grad first pass + reuse")
    assert set(anima_models.CKPT_RECOMPUTE_DIFFS) == {0.0}


def test_block_without_grad_input_keeps_the_plain_checkpoint(monkeypatch):
    """An autograd node needs a grad-bearing input: a Block whose x does not require grad (a caller that did not mark the
    noisy input) stays on torch.utils.checkpoint, which handles it."""
    unet, net, _ = build()
    set_flags(monkeypatch, True, True)
    calls = []
    real = anima_models.nograd_checkpoint
    monkeypatch.setattr(anima_models, "nograd_checkpoint", lambda *a: calls.append(1) or real(*a))
    ref_y, ref_g = T.step(unet, net, *batch(), checkpoint=True)
    assert len(calls) == len(unet.blocks)
    calls.clear()
    y, g = T.step(unet, net, *batch(), checkpoint=False)  # x without requires_grad: Block 0 plain, Block 1 the new node
    assert len(calls) == len(unet.blocks) - 1
    T.assert_same((ref_y, ref_g), (y, g), "unmarked input")


# ------------------------------------------------------------------------------------------------- merged-weight cache


def test_first_pass_builds_no_graph_and_rebuilds_once_per_optimizer_step(monkeypatch):
    """Eager, 3 micro-steps accumulated per optimizer step, 2 optimizer steps. Plain checkpoint: every micro-step
    rebuilds each cached module's weight twice in grad mode (first pass + recompute). Flags on: once per micro-step in
    grad mode (the recompute) and once per OPTIMIZER step without grad (the cache), into the same tensor object; the
    trained parameters end bit-identical."""
    spy = RebuildSpy(monkeypatch)

    def train(nograd):
        unet, net, _ = build()
        set_flags(monkeypatch, nograd, nograd)
        params = [p for p in net.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=1e-2)
        counts, tensors = [], []
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            for micro in range(3):
                laf.set_timestep(net, torch.tensor([300.0]))
                x, t, ctx, pad = batch(seed=micro + 1)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    y = unet(x.requires_grad_(True), t, ctx, padding_mask=pad)
                (y.float().square().mean() / 3).backward()
                counts.append(spy.take())
                tensors.append([lokr.__dict__.get("_ckpt_merged") for lokr in cached_lokrs(unet)])
            opt.step()
        return [p.detach().clone() for p in params], counts, tensors, len(cached_lokrs(unet))

    ref, plain_counts, _, n = train(False)
    new, counts, tensors, n2 = train(True)
    assert n == n2 == 2 * 10  # per Block: self / cross attention q, k, v, out and the two MLP Linears
    assert plain_counts == [(0, 2 * n)] * 6, plain_counts
    assert counts == [(n, n), (0, n), (0, n)] * 2, counts
    assert all(a is b for row in tensors[1:] for a, b in zip(tensors[0], row)), "a cached tensor was replaced"
    assert all(torch.equal(a, b) for a, b in zip(ref, new)), "trained parameters differ"


def test_cache_keys_and_invalidation(monkeypatch):
    """ckpt_merged_refresh rebuilds a module's cached weight when what it was built from changed - seen by the key for
    every write through the parameter itself and for a `.data =` swap - and after ckpt_merged_invalidate(), which is
    what covers an in-place write through `p.data` (its own version counter: the key cannot see it; CAME writes that
    way, hence the trainer's hooks). The cached value is the weight the module's own forward builds."""
    unet, net, _ = build(num_blocks=1)
    spy = RebuildSpy(monkeypatch)
    lokrs = cached_lokrs(unet)
    lokr = lokrs[0]
    n = len(lokrs)

    def refresh():
        keys = lycoris_fast.ckpt_merged_refresh(lokrs, torch.bfloat16)
        return spy.take()[0], keys

    built, keys = refresh()
    assert built == n and all(k is not None for k in keys)
    assert refresh()[0] == 0  # nothing changed
    assert lycoris_fast.ckpt_merged_current(lokrs, keys, torch.bfloat16)
    tensor = lokr.__dict__["_ckpt_merged"]
    args = (lokr.org_module[0].weight.detach(), lokr.lokr_w1, lokr.lokr_w2, float(lokr.scale), lokr.dora_scale,
            lokr.multiplier_buf, lokr._dora_eps, lokr._dora_norm_dims, torch.bfloat16)
    with torch.no_grad():
        assert tensor.dtype == torch.bfloat16 and torch.equal(tensor, lycoris_fast._rebuild_weight(*args))

    seen_by_key = {
        "optimizer.step on the parameter": lambda: torch.optim.SGD([lokr.lokr_w1], lr=0.1).step(),
        "in-place write": lambda: lokr.lokr_w2.mul_(1.01),
        "dora_scale": lambda: lokr.dora_scale.add_(0.01),
        "load_state_dict": lambda: lokr.load_state_dict({k: v * 1.01 for k, v in lokr.state_dict().items()}),
        ".data swap": lambda: setattr(lokr.lokr_w1, "data", lokr.lokr_w1.data.clone()),
        "base weight": lambda: lokr.org_module[0].weight.mul_(1.01),
        "base weight moved (the sample cache parks it in RAM)": lambda: setattr(
            lokr.org_module[0].weight, "data", lokr.org_module[0].weight.data.clone()),
    }
    lokr.lokr_w1.grad = torch.ones_like(lokr.lokr_w1)
    for what, change in seen_by_key.items():
        with torch.no_grad():
            change()
        assert not lycoris_fast.ckpt_merged_current(lokrs, keys, torch.bfloat16), what
        built, keys = refresh()
        assert built == 1, f"{what}: {built} rebuilds"
        assert lokr.__dict__["_ckpt_merged"] is tensor, f"{what}: the cached tensor was replaced"
        with torch.no_grad():
            assert torch.equal(tensor, lycoris_fast._rebuild_weight(*args)), what
    lokr.lokr_w1.grad = None

    # a write through p.data (CAME's p.data.copy_, weight noise): invisible to the key, the invalidation covers it
    with torch.no_grad():
        lokr.lokr_w2.data.mul_(1.01)
    assert lycoris_fast.ckpt_merged_current(lokrs, keys, torch.bfloat16) and refresh()[0] == 0
    with torch.no_grad():
        assert not torch.equal(tensor, lycoris_fast._rebuild_weight(*args))  # stale until someone invalidates

    lycoris_fast.ckpt_merged_invalidate()  # the trainer's hooks: everything again
    assert not lycoris_fast.ckpt_merged_current(lokrs, keys, torch.bfloat16)
    built, keys = refresh()
    assert built == n
    with torch.no_grad():
        assert torch.equal(tensor, lycoris_fast._rebuild_weight(*args))
    # the output dtype is part of the key (bf16 -> fp16 mixed precision would not reuse a bf16 weight)
    assert lycoris_fast.ckpt_merged_refresh(lokrs, torch.float16) != keys and spy.take()[0] == n
    assert lokr.__dict__["_ckpt_merged"].dtype == torch.float16
    # a multiplier other than 1 changes the key (the fixed-path forward does not take the fast path then anyway)
    built, keys = refresh()
    lokr.multiplier = 0.5
    assert not lycoris_fast.ckpt_merged_current(lokrs, keys, torch.bfloat16)
    lokr.multiplier = 1.0
    assert lycoris_fast.ckpt_merged_current(lokrs, keys, torch.bfloat16)
    # outside bf16 / fp16 autocast the fast path is not taken: no key, the cache is dropped
    assert lycoris_fast.ckpt_merged_refresh(lokrs, torch.float32) == (None,) * n
    assert lycoris_fast.ckpt_merged_bytes(unet) == 0
    refresh()
    held = lycoris_fast.ckpt_merged_bytes(unet)
    assert held == sum(l.org_module[0].weight.numel() for l in lokrs) * 2  # one bf16 copy of each cached weight
    assert lycoris_fast.ckpt_merged_release(unet) == held and lycoris_fast.ckpt_merged_bytes(unet) == 0


def test_a_cache_of_another_dtype_is_not_used():
    """The module forward only takes a cached weight of the autocast dtype it runs under (the node refreshes the cache
    under the same autocast as the Block's Linears; this is the guard should a caller ever differ)."""
    unet, net, _ = build(num_blocks=1)
    lokr = cached_lokrs(unet)[0]
    linear = lokr.org_module[0]
    x = torch.randn(2, 5, linear.in_features)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        ref = linear(x)
        lycoris_fast.ckpt_merged_refresh([lokr], torch.bfloat16)
        lokr.__dict__["_ckpt_merged"].mul_(0.5)
        with lycoris_fast.ckpt_pass(1):
            assert not torch.equal(linear(x), ref)  # the (halved) cache is read
        lokr.__dict__["_ckpt_merged"] = lokr.__dict__["_ckpt_merged"].to(torch.float16)
        with lycoris_fast.ckpt_pass(1):
            assert torch.equal(linear(x), ref)  # a float16 cache under bf16 autocast: the per-call rebuild
        assert torch.equal(linear(x), ref)  # and never outside a checkpoint pass


def test_block_seen_before_its_network_is_looked_up_again():
    """_ckpt_cached_lokrs keeps a Block's module list once it found one; an empty result (a checkpointed forward before
    any network was applied) is not kept."""
    torch.manual_seed(0)
    unet = anima_models.Anima(**{**T.TOY, "num_blocks": 1}).to(torch.bfloat16)
    assert anima_models._ckpt_cached_lokrs(unet.blocks[0]) == []
    unet2, net, _ = build(num_blocks=1)
    block = unet2.blocks[0]
    block.__dict__["_ckpt_lokrs"] = []  # as if it had been asked before apply_to
    found = anima_models._ckpt_cached_lokrs(block)
    assert len(found) == 10 and anima_models._ckpt_cached_lokrs(block) is found


def test_only_the_attention_and_mlp_linears_are_cached():
    """The adaLN modulation Linears run with autocast off (LyCORIS's stock forward): building their weights would only
    cost time and memory."""
    unet, net, _ = build(num_blocks=1)
    block = unet.blocks[0]
    names = {id(m): n for n, m in block.named_modules()}
    cached = sorted(names[id(lokr.org_module[0])] for lokr in anima_models._ckpt_cached_lokrs(block))
    assert cached == sorted([f"{a}.{p}" for a in ("self_attn", "cross_attn") for p in ("q_proj", "k_proj", "v_proj",
                                                                                      "output_proj")]
                            + ["mlp.layer1", "mlp.layer2"])
    assert len(laf._all_loras(net)) == 16  # the six modulation Linears are adapted too


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
def test_cache_is_only_read_inside_the_checkpoint(compiled, monkeypatch):
    """With every cached weight overwritten (keys left valid): the checkpointed training step changes - the first pass
    reads the cache, compiled too - while the recompute still agrees with the first pass (it gets the cached bits), and
    a validation forward (no_grad), an eval forward and an un-checkpointed training step are untouched."""
    unet, net, _ = build(compiled=compiled)
    set_flags(monkeypatch, True, True)
    good = T.step(unet, net, *batch(), checkpoint=True)

    def others():
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            val = unet(*batch()[:3], padding_mask=batch()[3]).float().clone()
        unet.eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16), bc.pause():
            sample = unet(*batch()[:3], padding_mask=batch()[3]).float().clone()
        unet.train()
        unet.disable_gradient_checkpointing()
        plain = T.step(unet, net, *batch())
        unet.enable_gradient_checkpointing()
        return val, sample, plain

    before = others()
    with torch.no_grad():
        for lokr in cached_lokrs(unet):
            lokr.__dict__["_ckpt_merged"].mul_(0.5)  # in place: the version of the parameters, i.e. the key, is unchanged
    anima_models.CKPT_RECOMPUTE_DIFFS.clear()
    poisoned = T.step(unet, net, *batch(), checkpoint=True)
    assert not torch.equal(poisoned[0], good[0]), "the first pass did not read the cache"
    assert set(anima_models.CKPT_RECOMPUTE_DIFFS) == {0.0}, "the recompute did not get the cached bits"
    after = others()
    assert torch.equal(before[0], after[0]) and torch.equal(before[1], after[1])
    T.assert_same(before[2], after[2], "un-checkpointed step")
    lycoris_fast.ckpt_merged_invalidate()
    T.assert_same(good, T.step(unet, net, *batch(), checkpoint=True), "after invalidation")


def test_parameters_changed_before_backward_recompute_plainly(monkeypatch):
    """Nothing in the trainer changes the adapter between a forward and its backward. If something does, the cached
    weight and the stored attention output no longer belong to the parameters: the recompute then rebuilds per call and
    runs the flash forward again, like the plain checkpoint's recompute (same gradients as it gives in that situation)."""

    def go(nograd):
        unet, net, _ = build()
        set_flags(monkeypatch, nograd, nograd)
        seen = []
        real = attention.attention

        def spy(qkv, **kwargs):
            seen.append((lycoris_fast._CKPT_PASS, kwargs.get("ckpt_stash") is not None, kwargs.get("ckpt_reuse") is not None))
            return real(qkv, **kwargs)

        monkeypatch.setattr(attention, "attention", spy)
        net.zero_grad(set_to_none=True)
        laf.set_timestep(net, torch.tensor([300.0]))
        x, t, ctx, pad = batch()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = unet(x.requires_grad_(True), t, ctx, padding_mask=pad)
        lokr = cached_lokrs(unet)[0]
        with torch.no_grad():
            lokr.lokr_w1.mul_(1.5)  # Block 0, between forward and backward
        y.float().square().mean().backward()
        monkeypatch.setattr(attention, "attention", real)
        return {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}, seen

    ref, _ = go(False)
    grads, seen = go(True)
    # forward: both Blocks' first passes fill a stash (self-attention only); backward runs Block 1 first: still current
    # (pass 2, stored output reused), then Block 0: its parameters changed -> pass 0, nothing reused
    assert seen == [(1, True, False), (1, False, False)] * 2 + [(2, False, True), (2, False, False),
                                                                (0, False, False), (0, False, False)], seen
    assert ref.keys() == grads.keys() and all(torch.equal(ref[n], grads[n]) for n in ref)


# --------------------------------------------------------------------------------------------------- attention reuse


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
def test_reuse_skips_one_flash_forward_per_checkpointed_block(compiled, monkeypatch):
    """Kernel runs of one micro-step over 2 Blocks (self- and cross-attention each, both on the fake flash): the plain
    checkpoint and the no-grad first pass run the forward kernel twice per attention, with reuse the self-attention runs
    it once; every variant runs the backward kernel once per attention."""
    counts = {}
    for name, (nograd, reuse) in {"plain": (False, False), "nograd": (True, False), "reuse": (True, True)}.items():
        unet, net, _ = build(compiled=compiled)
        set_flags(monkeypatch, nograd, reuse)
        T.step(unet, net, *batch(), checkpoint=True)  # compiles
        fake_flash.CALLS["fwd"] = fake_flash.CALLS["bwd"] = 0
        T.step(unet, net, *batch(), checkpoint=True)
        counts[name] = dict(fake_flash.CALLS)
    assert counts == {"plain": {"fwd": 8, "bwd": 4}, "nograd": {"fwd": 8, "bwd": 4}, "reuse": {"fwd": 6, "bwd": 4}}


def test_stash_is_left_empty_where_flash_is_not_the_plain_call(monkeypatch):
    """Only flash_attn_func(q, k, v, 0.0) with a head dim that is a multiple of 8 is stored (FlashAttnFunc pads other
    head dims; dropout draws random numbers; split attention and masks take other calls). Elsewhere the stash stays
    empty and the result is the plain call's."""
    g = torch.Generator().manual_seed(0)
    params = attention.AttentionParams.create_attention_params("flash", False)

    def call(head_dim=8, **kw):
        q, k, v = (torch.randn(1, 5, 2, head_dim, generator=torch.Generator().manual_seed(3)) for _ in range(3))
        return attention.attention([q, k, v], attn_params=kw.pop("attn_params", params), **kw)

    stash = []
    out = call(ckpt_stash=stash)
    assert len(stash) == 2 and stash[0].shape == (1, 5, 2, 8) and stash[1].shape == (1, 2, 5)
    assert torch.equal(out, call()) and torch.equal(stash[0].reshape(1, 5, 16), out)
    assert torch.equal(call(ckpt_reuse=tuple(stash)), out)
    for kw in (dict(head_dim=12), dict(attn_params=attention.AttentionParams.create_attention_params("flash", True)),
               dict(attn_params=attention.AttentionParams.create_attention_params("torch", False))):
        stash = []
        ref = call(**kw)
        assert torch.equal(call(ckpt_stash=stash, **kw), ref) and stash == [], kw
    del g


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
def test_no_backward_op_means_no_stash_and_the_plain_recompute(compiled, monkeypatch):
    """A flash-attn without flash_attn_interface._wrapped_flash_attn_backward (library.attention keeps None then): the
    reuse path would have nothing to call in backward (TypeError in the first backward before the 2026-11-14 fix). The
    first pass stores nothing, so the recompute runs flash_attn_func again: outputs and gradients are the plain
    checkpoint's, the kernel counts those of --anima_ckpt_nograd_merged alone. The fake's own autograd Function keeps its
    backward kernel (only library.attention's handle is hidden), as an old flash-attn's FlashAttnFunc would."""
    assert attention.flash_reuse_available()
    monkeypatch.setattr(attention, "_wrapped_flash_attn_backward", None)
    assert not attention.flash_reuse_available()

    q, k, v = (torch.randn(1, 5, 2, 8, generator=torch.Generator().manual_seed(3)) for _ in range(3))
    params = attention.AttentionParams.create_attention_params("flash", False)
    stash = []
    out = attention.attention([q, k, v], attn_params=params, ckpt_stash=stash)
    assert stash == [] and torch.equal(out, attention.attention([q, k, v], attn_params=params))

    results, counts = {}, {}
    for name, (nograd, reuse) in {"plain": (False, False), "reuse": (True, True)}.items():
        unet, net, _ = build(compiled=compiled)
        set_flags(monkeypatch, nograd, reuse)
        T.step(unet, net, *batch(), checkpoint=True)  # compiles
        fake_flash.CALLS["fwd"] = fake_flash.CALLS["bwd"] = 0
        results[name] = T.step(unet, net, *batch(), checkpoint=True)
        counts[name] = dict(fake_flash.CALLS)
    assert counts == {"plain": {"fwd": 8, "bwd": 4}, "reuse": {"fwd": 8, "bwd": 4}}
    (y0, g0), (y1, g1) = results["plain"], results["reuse"]
    assert torch.equal(y0, y1) and g0.keys() == g1.keys() and all(torch.equal(g0[n], g1[n]) for n in g0)
    assert anima_models.CKPT_RECOMPUTE_DIFFS and set(anima_models.CKPT_RECOMPUTE_DIFFS) == {0.0}


def test_real_flash_attn_takes_the_stash_and_reuse_calls(monkeypatch):
    """The installed flash-attn's Python wrappers and custom-op schemas, on fake CUDA tensors (FakeTensorMode: shapes and
    dtypes from the ops' registered fake kernels, no GPU and no CUDA initialisation): the first pass's call returns the
    output and an fp32 softmax_lse of flash's shapes, the recompute returns the stored output, and the backward op
    accepts the arguments the reuse path gives it (the ones FlashAttnFunc.backward passes for the plain call)."""
    monkeypatch.undo()  # the real library.attention entry points, not the CPU stand-ins the fixture installed
    try:
        from flash_attn import flash_attn_interface
    except ImportError:
        pytest.skip("flash-attn is not installed")
    from torch._subclasses.fake_tensor import FakeTensorMode

    if attention.flash_attn_func is not flash_attn_interface.flash_attn_func:
        pytest.skip("library.attention disabled flash-attn for this GPU")
    assert attention._wrapped_flash_attn_backward is flash_attn_interface._wrapped_flash_attn_backward
    params = attention.AttentionParams.create_attention_params("flash", False)
    shape = (1, 48, 16, 128)  # (B, tokens, heads, head_dim): Anima 2.9B's heads
    initialized = torch.cuda.is_initialized()
    with FakeTensorMode():
        try:
            q, k, v = (torch.empty(shape, dtype=torch.float16, device="cuda") for _ in range(3))
        except Exception as error:  # noqa: BLE001 - a torch build that cannot fake a CUDA tensor without a device
            pytest.skip(f"no fake CUDA tensors here: {error}")
        stash = []
        with torch.no_grad():
            first = attention.attention([q, k, v], attn_params=params, ckpt_stash=stash)
        out, softmax_lse = stash
        assert first.shape == (1, 48, 16 * 128) and first.dtype == torch.float16
        assert out.shape == shape and out.dtype == torch.float16
        assert softmax_lse.shape == (1, 16, 48) and softmax_lse.dtype == torch.float32
        again = attention.attention([q, k, v], attn_params=params, ckpt_reuse=(out, softmax_lse))
        assert again.shape == first.shape and again.dtype == first.dtype
        grads = attention._flash_backward(torch.empty(shape, dtype=torch.float16, device="cuda"), q, k, v, out, softmax_lse)
        assert [g.shape for g in grads] == [shape] * 3 and all(g.dtype == torch.float16 for g in grads)
        with pytest.raises(RuntimeError):  # the schema is checked: a wrong argument list would not pass silently
            attention._wrapped_flash_attn_backward(grads[0], q, k, v, out, softmax_lse, *grads, 0.0, 128 ** -0.5, False,
                                                   -1, -1, 0.0, None, False, None, "one too many")
    assert torch.cuda.is_initialized() == initialized


def test_stored_tensors_are_released_after_backward(monkeypatch):
    """The node drops its (out, softmax_lse) when its backward starts and the recompute graph when it ends: nothing of a
    Block's stored attention survives the micro-step."""
    import gc
    import weakref

    unet, net, _ = build()
    set_flags(monkeypatch, True, True)
    refs = []
    real = attention._flash_forward_with_lse

    def spy(q, k, v):
        out, lse = real(q, k, v)
        refs.append((weakref.ref(out), weakref.ref(lse)))
        return out, lse

    monkeypatch.setattr(attention, "_flash_forward_with_lse", spy)
    net.zero_grad(set_to_none=True)
    laf.set_timestep(net, torch.tensor([300.0]))
    x, t, ctx, pad = batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = unet(x.requires_grad_(True), t, ctx, padding_mask=pad)
    loss = y.float().square().mean()
    assert len(refs) == len(unet.blocks) and all(o() is not None and l() is not None for o, l in refs)  # held for backward
    loss.backward()
    del y, loss
    gc.collect()
    assert all(o() is None and l() is None for o, l in refs), "a stored attention output outlived its Block's backward"


# ------------------------------------------------------------------------------------------------------ torch.compile


def _block_call(unet, net, stream=False):
    """The (args, kwargs) Block 1 gets in a training forward (its x requires grad, as in every Block of the trainer)."""
    cap = {}
    h = unet.blocks[1].register_forward_pre_hook(lambda m, a, k: cap.update(args=a, kwargs=k), with_kwargs=True)
    T.step(unet, net, *batch(stream), checkpoint=True, backward=False)
    h.remove()
    return cap["args"], cap["kwargs"]


@pytest.mark.parametrize("stream", [False, True], ids=["bf16_stream", "fp32_stream"])
def test_first_pass_and_recompute_trace_whole(stream):
    """torch._dynamo.explain on Block._forward as the node calls it: the no-grad first pass (cached weights read from the
    modules, the stash list filled) and the grad-mode recompute (_FirstPassBits, _FlashReuse with the stored tensors) are
    one graph each, no graph break."""
    unet, net, _ = build(stream=stream)
    laf.set_block_compile(True)
    laf.rebind_forwards(net)
    laf.prime_module_caches(net)
    unet._block_compile = True
    args, kwargs = _block_call(unet, net, stream)
    block = unet.blocks[1]
    lokrs = anima_models._ckpt_cached_lokrs(block)
    assert len(lokrs) == 10
    with torch.autocast("cpu", dtype=torch.bfloat16):
        lycoris_fast.ckpt_merged_refresh(lokrs, torch.bfloat16)
        stash = []
        with torch.no_grad(), lycoris_fast.ckpt_pass(1):
            first = torch._dynamo.explain(block._forward)(*args, **kwargs, attn_stash=stash)
            block._forward(*args, **kwargs, attn_stash=stash)
        with lycoris_fast.ckpt_pass(2):
            second = torch._dynamo.explain(block._forward)(*args, **kwargs, attn_reuse=tuple(stash[:2]))
    for what, ex in (("first pass", first), ("recompute", second)):
        reasons = sorted({str(r.reason).splitlines()[0][:120] for r in ex.break_reasons})
        assert ex.graph_break_count == 0 and ex.graph_count == 1, f"{what}: {ex.graph_count} graphs, breaks {reasons}"


@pytest.mark.parametrize("soft", [True, False], ids=["soft_marks", "strict_marks"])
def test_blocks_and_resolutions_share_the_graphs(soft, monkeypatch):
    """4 Blocks, --anima_stream_fp32 (one dtype regime), flags on: 2 graphs for the whole model - the no-grad first pass
    and the recompute - and the other resolutions of the band (16 .. 64 tokens) compile nothing new: the stored attention
    tensors' token axis is marked like x's. Results still match the plain checkpoint at every grid."""
    grids = ((8, 8), (8, 12), (12, 12), (16, 16))
    unet, net, _ = build(num_blocks=4, stream=True)
    refs = [T.step(unet, net, *batch(True, hw), checkpoint=True) for hw in grids]
    unet, net, _ = T.build(PRESET, num_blocks=4)
    perturb(net)
    unet.enable_stream_fp32()
    unet.enable_gradient_checkpointing()
    backend, calls = T.counting_backend("aot_eager")
    T.compile_toy(unet, net, backend=backend, soft_marks=soft, stream_fp32=True)
    set_flags(monkeypatch, True, True)
    for i, hw in enumerate(grids):
        new = T.step(unet, net, *batch(True, hw), checkpoint=True)
        T.assert_same(refs[i], new, f"grid {hw}")
        assert len(calls) == 2, f"grid {hw}: {len(calls)} graphs"
    assert set(anima_models.CKPT_RECOMPUTE_DIFFS) == {0.0} and not bc._warned_oob


def test_cached_weights_are_graph_inputs_updated_in_place(monkeypatch):
    """Compiled: after an optimizer step the cache is rebuilt in place and the same first-pass graph reads the new values
    (no recompile), giving the plain checkpoint's results for the new parameters."""
    def two_steps(nograd, backend="aot_eager"):
        unet, net, _ = T.build(PRESET)
        perturb(net)
        unet.enable_gradient_checkpointing()
        T.compile_toy(unet, net, backend=backend)
        set_flags(monkeypatch, nograd, nograd)
        opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=1e-2)
        first = T.step(unet, net, *batch(), checkpoint=True)
        opt.step()
        return first, T.step(unet, net, *batch(), checkpoint=True)

    ref = two_steps(False)
    backend, calls = T.counting_backend("aot_eager")
    new = two_steps(True, backend)
    T.assert_same(ref[0], new[0], "first step")
    T.assert_same(ref[1], new[1], "after the optimizer step")
    assert not torch.equal(ref[0][0], ref[1][0])  # the step did change the model
    assert len(calls) == 4  # first pass + recompute, for Block 0's bf16 input and Block 1's fp32 stream


# ------------------------------------------------------------------------------------------------- RNG / other inputs


LOKR_DROPOUTS = (dict(rank_dropout=0.3), dict(module_dropout=0.5), dict(dropout=0.3))


def _block_lokr_run(drop, nograd, monkeypatch, grad_inputs=False):
    """test_unsloth_checkpoint_rng's Anima Block + LoKr/DoRA (fp32, attn_mode torch) with one dropout knob, under
    torch.utils.checkpoint or the no-grad first pass."""
    from lycoris import kohya

    torch.manual_seed(0)
    root = nn.Module()
    root.block = anima_models.Block(x_dim=64, context_dim=48, num_heads=4)
    with torch.no_grad():
        for p in root.parameters():
            p.normal_(0, 0.05)
    root.requires_grad_(False)
    net = kohya.create_network(1.0, 10000, 10000, None, [], root, algo="lokr", preset="anima-exp", factor=4,
                               full_matrix=True, dora_wd=True, wd_on_output=True, **drop)
    net.apply_to([], root, False, True)
    with torch.no_grad():
        g = torch.Generator().manual_seed(7)
        for p in net.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.02)
    root.train()
    root.block.enable_gradient_checkpointing()
    set_flags(monkeypatch, nograd, False)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 1, 3, 4, 64, generator=g).requires_grad_(True)
    emb = torch.randn(2, 1, 64, generator=g).requires_grad_(grad_inputs)
    crossattn = torch.randn(2, 7, 48, generator=g).requires_grad_(grad_inputs)
    rope = (torch.randn(12, 1, 1, 16, generator=g), torch.randn(12, 1, 1, 16, generator=g))
    params = attention.AttentionParams.create_attention_params("torch", False)
    torch.manual_seed(123)
    y = root.block(x, emb, crossattn, params, False, rope)
    y.square().mean().backward()
    grads = [None if t.grad is None else t.grad.clone() for t in list(net.parameters()) + [x, emb, crossattn]]
    return y.detach(), grads, torch.get_rng_state()


@pytest.mark.parametrize("drop", LOKR_DROPOUTS, ids=lambda d: next(iter(d)))
def test_lokr_dropouts_match_the_plain_checkpoint(drop, monkeypatch):
    """The recompute replays the forward's RNG state (the dropout masks the loss saw) and the caller's stream ends where
    torch.utils.checkpoint leaves it."""
    (y0, g0, rng0), (y1, g1, rng1) = (_block_lokr_run(drop, nograd, monkeypatch) for nograd in (False, True))
    assert torch.equal(y0, y1) and torch.equal(rng0, rng1)
    assert any(g is not None for g in g0)
    for i, (a, b) in enumerate(zip(g0, g1)):
        assert (a is None) == (b is None) and (a is None or torch.equal(a, b)), f"{drop}: grad {i}"


def test_other_tensor_inputs_get_their_gradients(monkeypatch):
    """emb and the cross-attention context as grad-bearing inputs (a trained t_embedder / llm_adapter): their gradients
    come back through the node, equal to the plain checkpoint's for one Block."""
    (y0, g0, _), (y1, g1, _) = (_block_lokr_run({}, nograd, monkeypatch, grad_inputs=True) for nograd in (False, True))
    assert torch.equal(y0, y1) and g0[-1] is not None and g0[-2] is not None
    for i, (a, b) in enumerate(zip(g0, g1)):
        assert (a is None) == (b is None) and (a is None or torch.equal(a, b)), f"grad {i}"


# ------------------------------------------------------------------------------------------------------- trainer glue


def _args(**over):
    base = dict(anima_ckpt_nograd_merged=True, anima_ckpt_reuse_attn=True, gradient_checkpointing=True,
                unsloth_offload_checkpointing=False, cpu_offload_checkpointing=False, blocks_to_swap=None,
                attn_mode="flash", split_attn=False, anima_stock_lycoris=False)
    base.update(over)
    return argparse.Namespace(**base)


def test_validate_ckpt_reuse_args():
    from anima_train_network import validate_ckpt_reuse_args

    validate_ckpt_reuse_args(_args())
    validate_ckpt_reuse_args(_args(anima_ckpt_reuse_attn=False, attn_mode="torch", split_attn=True))  # the cache alone
    validate_ckpt_reuse_args(_args(anima_ckpt_nograd_merged=False, anima_ckpt_reuse_attn=False,  # off: anything goes
                                   unsloth_offload_checkpointing=True, blocks_to_swap=8, attn_mode="torch"))
    validate_ckpt_reuse_args(_args(gradient_checkpointing=False))  # only a warning: nothing is checkpointed
    validate_ckpt_reuse_args(_args(anima_ckpt_nograd_merged=False, anima_ckpt_reuse_attn=False,  # off: stock LyCORIS is fine
                                   anima_stock_lycoris=True))
    for bad, match in ((dict(anima_ckpt_nograd_merged=False), "requires --anima_ckpt_nograd_merged"),
                       (dict(unsloth_offload_checkpointing=True), "unsloth_offload_checkpointing"),
                       (dict(cpu_offload_checkpointing=True), "cpu_offload_checkpointing"),
                       (dict(blocks_to_swap=4), "blocks_to_swap"),
                       # the stock LoKr forward never reads the cache (2026-11-14 review): with or without the reuse
                       (dict(anima_stock_lycoris=True), "not with --anima_stock_lycoris"),
                       (dict(anima_stock_lycoris=True, anima_ckpt_reuse_attn=False), "not with --anima_stock_lycoris"),
                       (dict(attn_mode="torch"), "attn_mode flash"), (dict(attn_mode=None), "attn_mode flash"),
                       (dict(split_attn=True), "split_attn")):
        with pytest.raises(ValueError, match=match):
            validate_ckpt_reuse_args(_args(**bad))


def test_reuse_without_the_flash_backward_op_is_a_startup_warning(monkeypatch):
    """--anima_ckpt_reuse_attn with a flash-attn that has no separate backward op: not refused (the run is right, the
    recompute just runs the forward kernel), one warning that the option changes nothing."""
    import anima_train_network as atn

    warned = []
    monkeypatch.setattr(atn.logger, "warning", lambda message, *a, **k: warned.append(str(message)))
    atn.validate_ckpt_reuse_args(_args())
    assert warned == []
    monkeypatch.setattr(attention, "_wrapped_flash_attn_backward", None)
    atn.validate_ckpt_reuse_args(_args())
    assert len(warned) == 1 and "--anima_ckpt_reuse_attn changes nothing" in warned[0]
    atn.validate_ckpt_reuse_args(_args(anima_ckpt_reuse_attn=False))  # the cache alone does not need flash
    assert len(warned) == 1


def test_text_training_and_unapplied_flags_are_warned_about(monkeypatch):
    """Two start-up warnings of the 2026-11-14 review, neither an error.
    * A trained llm_adapter or text encoder: the node returns the text conditioning's gradient summed per Block (in its
      fp16 / bf16 dtype), so what is trained behind it is not bit-identical to the flag-off run
      (tests/test_anima_ckpt_reuse_review.py::test_text_conditioning_gradient_through_many_blocks measures it).
      AnimaNetworkTrainer.after_network_loaded says so once, only with the switch on.
    * anima_train_addift.py and anima_train_leco.py's dataset-backed iLECO parse the options (anima_train_network's
      parser) but never call prepare_anima_runtime: the switches stay off, assert_extra_args says the options do nothing."""
    import inspect
    import types

    import anima_train_network as atn

    on = argparse.Namespace(anima_ckpt_nograd_merged=True, anima_ckpt_reuse_attn=True)
    off = argparse.Namespace(anima_ckpt_nograd_merged=False, anima_ckpt_reuse_attn=False)

    # ---- the options were parsed, the switches never set (prepare_anima_runtime not called)
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", False)
    message = atn.ckpt_reuse_not_applied_warning(on)
    assert message and "--anima_ckpt_nograd_merged / --anima_ckpt_reuse_attn has no effect" in message
    assert "--anima_ckpt_reuse_attn" not in atn.ckpt_reuse_not_applied_warning(
        argparse.Namespace(anima_ckpt_nograd_merged=True, anima_ckpt_reuse_attn=False))
    assert atn.ckpt_reuse_not_applied_warning(off) is None and atn.ckpt_reuse_not_applied_warning(argparse.Namespace()) is None
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", True)
    assert atn.ckpt_reuse_not_applied_warning(on) is None  # anima_train_network.py: applied
    assert "ckpt_reuse_not_applied_warning(args)" in inspect.getsource(atn.AnimaNetworkTrainer.assert_extra_args)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for script, trainer_use in (("anima_train_addift.py", "anima_train_network.AnimaNetworkTrainer()"),
                                ("anima_train_leco.py", "AnimaNetworkTrainer()")):
        with open(os.path.join(root, script), encoding="utf-8") as f:
            source = f.read()
        # the scripts this warning is for: the trainer (and with it assert_extra_args) without the runtime switches.
        # Whoever wires prepare_anima_runtime into one of them drops it from this list
        assert trainer_use in source and "prepare_anima_runtime" not in source, script

    # ---- llm_adapter / text encoder training with the switch on
    llm, plain = types.SimpleNamespace(train_llm_adapter=True), types.SimpleNamespace(train_llm_adapter=False)
    assert atn.ckpt_reuse_text_grad_warning(plain, False) is None and atn.ckpt_reuse_text_grad_warning(None, False) is None
    assert "with llm_adapter trained" in atn.ckpt_reuse_text_grad_warning(llm, False)
    assert "with the text encoder trained" in atn.ckpt_reuse_text_grad_warning(plain, True)
    assert "with llm_adapter and the text encoder trained" in atn.ckpt_reuse_text_grad_warning(llm, True)
    unet, net, _ = build()  # the production shape of network: LoKr+DoRA on the Blocks only
    assert atn.ckpt_reuse_text_grad_warning(net, False) is None
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", False)
    assert atn.ckpt_reuse_text_grad_warning(llm, True) is None  # switch off: nothing to say

    warned = []
    monkeypatch.setattr(atn.logger, "warning", lambda message, *a, **k: warned.append(str(message)))
    trainer = atn.AnimaNetworkTrainer()
    # switch off: after_network_loaded does not even look (no network_train_unet_only on this namespace)
    trainer.after_network_loaded(argparse.Namespace(anima_block_compile=False), None, llm, [], None)
    assert warned == []
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", True)
    args = argparse.Namespace(anima_block_compile=False, network_train_unet_only=True)
    trainer.after_network_loaded(args, None, plain, [], None)
    assert warned == []
    trainer.after_network_loaded(args, None, llm, [], None)
    assert len(warned) == 1 and "llm_adapter trained" in warned[0] and "LoKr / DoRA gradients are unchanged" in warned[0]
    args.network_train_unet_only = False
    trainer.after_network_loaded(args, None, plain, [], None)
    assert len(warned) == 2 and "the text encoder trained" in warned[1]


def test_parser_runtime_switches_and_invalidation_hooks(monkeypatch):
    """Off by default; prepare_anima_runtime copies the two options to the module switches; with the switch on the
    trainer invalidates the cache after every optimizer step and after sample images (and not with it off)."""
    import inspect

    import anima_train_network as atn

    assert "validate_ckpt_reuse_args(args)" in inspect.getsource(atn.AnimaNetworkTrainer.assert_extra_args)
    parser = atn.setup_parser()
    ns = parser.parse_args(["--timestep_sampling", "sigmoid"])
    assert ns.anima_ckpt_nograd_merged is False and ns.anima_ckpt_reuse_attn is False
    chains, rebuild, hoist = anima_models._CHAINS.enabled, lycoris_fast._REBUILD.enabled, anima_models.ADALN_HOIST
    try:
        atn.prepare_anima_runtime(ns)
        assert anima_models.CKPT_NOGRAD_MERGED is False and anima_models.CKPT_REUSE_ATTN is False
        on = parser.parse_args(["--timestep_sampling", "sigmoid", "--anima_ckpt_nograd_merged", "--anima_ckpt_reuse_attn"])
        atn.prepare_anima_runtime(on)
        assert anima_models.CKPT_NOGRAD_MERGED is True and anima_models.CKPT_REUSE_ATTN is True
    finally:
        anima_models._CHAINS.enable(chains)
        lycoris_fast.enable_compiled_rebuild(rebuild)
        anima_models.ADALN_HOIST = hoist

    trainer = atn.AnimaNetworkTrainer()
    monkeypatch.setattr(atn.anima_train_utils, "sample_images", lambda *a, **k: None)
    monkeypatch.setattr(atn.strategy_base.TextEncodingStrategy, "get_strategy", staticmethod(lambda: None))
    monkeypatch.setattr(atn.strategy_base.TokenizeStrategy, "get_strategy", staticmethod(lambda: None))
    monkeypatch.setattr(trainer, "get_models_for_text_encoding", lambda *a: None)

    def hooks():
        before = lycoris_fast._CKPT_GENERATION
        trainer.on_optimizer_step_end(argparse.Namespace(), None, 1, False)
        after_step = lycoris_fast._CKPT_GENERATION
        trainer.sample_images(None, argparse.Namespace(), 0, 1, "cpu", None, None, [None], None)
        return after_step - before, lycoris_fast._CKPT_GENERATION - after_step

    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", False)
    assert hooks() == (0, 0)
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", True)
    assert hooks() == (1, 1)

    # prepare_anima_runtime (flag on) registered the global optimizer-step hook, once: any optimizer.step() invalidates,
    # also one that writes through p.data like CAME (which the keys cannot see)
    assert atn._CKPT_STEP_HOOK is not None
    handle = atn._CKPT_STEP_HOOK
    atn.invalidate_ckpt_merged_on_optimizer_steps()
    assert atn._CKPT_STEP_HOOK is handle

    class DataWriter(torch.optim.Optimizer):
        def __init__(self, params):
            super().__init__(params, {})

        def step(self, closure=None):
            for group in self.param_groups:
                for p in group["params"]:
                    p.data.mul_(0.5)

    p = nn.Parameter(torch.ones(2))
    before, version = lycoris_fast._CKPT_GENERATION, p._version
    DataWriter([p]).step()
    assert p._version == version and lycoris_fast._CKPT_GENERATION == before + 1
    handle.remove()  # leave the process as it was for the other test files
    atn._CKPT_STEP_HOOK = None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
