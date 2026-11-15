"""CPU tests for --moss_ckpt_nograd_merged and --moss_ckpt_reuse_attn (2026-11-14; both off by default): the
--anima_ckpt_nograd_merged / --anima_ckpt_reuse_attn machinery (library/anima_models.py NoGradFirstPassCheckpointer,
library/lycoris_fast.py's checkpoint merged-weight cache, library/attention.py's flash stash / reuse - tested for Anima
in tests/test_anima_ckpt_reuse*.py) applied to the joint MossBlocks of library/moss/models.py.

flash-attn has no CPU kernels: attn_mode "flash" runs on tests/_fake_flash.py (custom ops with flash's forward / backward
split, the backward reads the STORED output and softmax_lse), plus a varlen stand-in defined here for padded batches (the
same kernels per cu_seqlens segment). The toy MossTransformer and the LyCORIS network are test_moss_block_compile's
(frozen bf16, LoKr factor 4 full matrix + DoRA on the output axis, preset moss, built the way train_network.py builds
it), with one joint block left out of the network like --moss_lora_exclude_blocks "22", every factor moved off its init,
gradient checkpointing on; eager and per-block compiled with the aot_eager backend (dynamo's trace and AOTAutograd's
joint graph as inductor gets them, without codegen).

Checked here:
* flags off: MossBlock._run hands the same arguments to torch.utils.checkpoint, attention gets no extra argument,
  nothing is cached, lycoris_fast._CKPT_PASS stays 0;
* flags on vs off - eager and compiled, bf16 stream and --moss_stream_fp32, with and without the attention reuse: the
  output and every network gradient are bit-identical, and the recompute's block outputs equal the first pass's;
* batches: one sample with changing text lengths and image sizes, several samples of equal text length (the plain flash
  call: reused) and a padded batch (flash varlen: nothing stored, the recompute runs the forward kernel as before);
* the block left out of the network has no cache and still gets the no-grad first pass and the reuse;
* training runs driven like train_network.py (accumulation 3 x 3 optimizer steps; CAME, which writes through p.data, and
  AdamW; gradient clipping; a sample round with CFG and the negative branch's skipped block in between; a resume-like
  load_state_dict in the middle of an accumulation; a validation forward): every micro-step's output and the trained
  parameters equal the flags-off run's; with no invalidation layer at all a CAME run goes stale (negative control);
* the cache is read only inside the checkpoint: a poisoned cache changes the checkpointed training output (so it IS
  read) but neither sample images (CFG + skip layers), validation nor un-checkpointed blocks; sampling never passes a
  stash or a stored output to attention;
* text refiner training (preset moss-text), trained time conditioning (the shared modulation enters every block) and a
  non-LyCORIS network (networks.lora_moss): gradients bit-identical;
* torch.compile: no eager fallback, the graph count stays flat across optimizer steps and caption lengths;
* the trainer glue: the parser, validate_moss_ckpt_reuse_args, prepare_moss_runtime's switches, the invalidation hooks;
* review fixes (2026-11-15): a padded batch under block compile keeps torch.utils.checkpoint (no second graph per padded
  length); a sample round releases the merged-weight cache and training rebuilds it; the VRAM estimate (checked against
  what the cache and the stash really hold), its start-up line and the peak watch; the batch > 1 warnings read a dataset
  TOML's batch sizes; the fixture takes lycoris_algo_fixes' class patches off again.

Reuse-only (--moss_ckpt_reuse_attn without --moss_ckpt_nograd_merged, 2026-11-15) and the stored-attention budget
(--moss_ckpt_reuse_attn_budget_gib) are tested in tests/test_moss_ckpt_reuse_only.py, which imports this module's toy
model and helpers.

GPU-only questions (inductor's fused rebuild vs the eager cache, the real flash kernels, VRAM, speed) are for
tests/manual_test_moss_ckpt_reuse_gpu.py.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ckpt_reuse.py -q -p no:cacheprovider
"""
import argparse
import gc
import os
import types
import weakref
from typing import Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
import torch._dynamo
from torch import nn
from torch._dynamo.backends.registry import lookup_backend

from library import anima_models, attention, lycoris_args, lycoris_fast
from library import lycoris_algo_fixes as laf
from library.moss import block_compile as mbc
from library.moss import models as moss_models
from library.moss import train_utils as moss_train_utils
from library.moss.lycoris_presets import register_moss_lycoris_presets
from library.moss.models import MossTransformer, enable_elementwise_compile
from tests import _fake_flash as fake_flash

torch.set_num_threads(4)

TINY = dict(dim=192, ffn_dim=512, num_heads=4, num_layers=3, text_dim=64, freq_dim=32, latent_channels=128,
            attn_mode="flash")
LOKR = ["algo=lokr", "preset=moss", "factor=4", "full_matrix=True", "decompose_both=False", "dora_wd=True",
        "wd_on_output=True"]
EXCLUDED = 1  # the joint block left out of the network (production: 22, the block the sampler's negative branch skips)
PER_BLOCK = 6  # adapted Linears of a joint block: proj.q / k / v / o / ffn_up_gate / ffn_down


# ---------------------------------------------------------------------------------- a CPU stand-in for flash varlen
# library.attention sends a padded batch (different text lengths) through flash_attn_varlen_func with the rows flattened
# to [(B*L), heads, head_dim] and cu_seqlens marking, per sample, its valid tokens and its padding as two sequences.
# The same split as tests/_fake_flash.py: a forward and a backward kernel as custom ops (opaque to dynamo), an autograd
# Function around them.
_NS = "moss_test_flash"
VARLEN_CALLS = {"fwd": 0, "bwd": 0}


def _segments(cu_seqlens: torch.Tensor):
    bounds = cu_seqlens.tolist()
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


@torch.library.custom_op(f"{_NS}::varlen_fwd", mutates_args=())
def _varlen_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor) -> Tuple[torch.Tensor,
                                                                                                    torch.Tensor]:
    VARLEN_CALLS["fwd"] += 1
    out = torch.zeros_like(q)
    softmax_lse = torch.zeros((q.shape[1], q.shape[0]), dtype=torch.float32)
    for a, b in _segments(cu_seqlens):
        o, lse = torch.ops.anima_test_flash.fwd(q[a:b][None], k[a:b][None], v[a:b][None])
        out[a:b] = o[0]
        softmax_lse[:, a:b] = lse[0]
    return out, softmax_lse


@torch.library.register_fake(f"{_NS}::varlen_fwd")
def _varlen_fwd_fake(q, k, v, cu_seqlens):
    return torch.empty_like(q), torch.empty((q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device)


@torch.library.custom_op(f"{_NS}::varlen_bwd", mutates_args=())
def _varlen_bwd(dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
                softmax_lse: torch.Tensor, cu_seqlens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    VARLEN_CALLS["bwd"] += 1
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    for a, b in _segments(cu_seqlens):
        sq, sk, sv = (torch.empty_like(t[a:b][None]) for t in (q, k, v))
        torch.ops.anima_test_flash.bwd(dout[a:b][None].contiguous(), q[a:b][None], k[a:b][None], v[a:b][None],
                                       out[a:b][None], softmax_lse[:, a:b][None].contiguous(), sq, sk, sv)
        dq[a:b], dk[a:b], dv[a:b] = sq[0], sk[0], sv[0]
    return dq, dk, dv


@torch.library.register_fake(f"{_NS}::varlen_bwd")
def _varlen_bwd_fake(dout, q, k, v, out, softmax_lse, cu_seqlens):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


class FakeFlashVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens):
        out, softmax_lse = torch.ops.moss_test_flash.varlen_fwd(q, k, v, cu_seqlens)
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, softmax_lse, cu_seqlens = ctx.saved_tensors
        dq, dk, dv = torch.ops.moss_test_flash.varlen_bwd(dout, q, k, v, out, softmax_lse, cu_seqlens)
        return dq, dk, dv, None


def fake_flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0):
    assert dropout_p == 0.0
    return FakeFlashVarlenFunc.apply(q, k, v, cu_seqlens_q)


# ------------------------------------------------------------------------------------------------------------ fixtures


def _remove_step_hook():
    if lycoris_fast._CKPT_STEP_HOOK is not None:
        lycoris_fast._CKPT_STEP_HOOK.remove()
        lycoris_fast._CKPT_STEP_HOOK = None


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    lycoris_fast.install_fast_lokr_forward()
    laf.apply_module_patches()
    register_moss_lycoris_presets()
    enable_elementwise_compile(False)  # production: off under block compile (prepare_moss_runtime); CUDA only anyway
    torch._dynamo.reset()
    fake_flash.install(monkeypatch)
    monkeypatch.setattr(attention, "flash_attn_varlen_func", fake_flash_attn_varlen_func)
    VARLEN_CALLS["fwd"] = VARLEN_CALLS["bwd"] = 0
    for name, value in (("CKPT_NOGRAD_MERGED", False), ("CKPT_REUSE_ATTN", False), ("CKPT_RECOMPUTE_DIFFS", None)):
        monkeypatch.setattr(anima_models, name, value)
    _remove_step_hook()  # every run here chooses its invalidation layers itself
    with torch.backends.mkldnn.flags(enabled=False):  # oneDNN has no bf16 backward on some CPUs
        yield
    _teardown_env()
    assert lycoris_fast._CKPT_PASS == 0, "a checkpoint pass left lycoris_fast._CKPT_PASS set"


def _teardown_env():
    """What `_env` undoes after every test (idempotent). The class patches of lycoris_algo_fixes go too: a test module
    that runs later in the same process (tests/test_moss_pipeline.py) expects LyCORIS's classes without them, as
    tests/test_lycoris_algo_fixes.py leaves them. Every test here applies them again and builds its own model."""
    _remove_step_hook()
    laf.set_block_compile(False)
    laf._revert_module_patches()
    torch._dynamo.reset()


@pytest.fixture
def came_cls(monkeypatch):
    """The user's optimizer class (LoraEasyCustomOptimizer.came.CAME). Its step assumes a CUDA build with a visible GPU
    even for CPU parameters (pin_memory, torch.cuda.current_device as the compute device, a final synchronize); the
    three calls are pointed at the CPU so the real write path (p.data.copy_) runs here."""
    try:
        from LoraEasyCustomOptimizer.came import CAME
    except Exception as error:  # noqa: BLE001 - the package is an editable install outside this repository
        pytest.skip(f"LoraEasyCustomOptimizer.came is not importable: {error}")
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: self)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    return CAME


def make_came(came_cls, params, lr=2e-3):
    # the optimizer_args of the user's Anima config; the state on the CPU, compile_step off (inductor needs a compiler)
    return came_cls(params, lr=lr, state_storage_device="cpu", state_storage_dtype="float32", weight_decay=0.1,
                    weight_decouple=True, fixed_decay=False, update_strategy="cautious", clip_threshold=1.0, eps1=1e-30,
                    eps2=1e-16, betas=(0.9, 0.999, 0.9999))


# ------------------------------------------------------------------------------------------------------------- helpers


def set_flags(monkeypatch, nograd: bool, reuse: bool = False, diffs: bool = True):
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", nograd)
    monkeypatch.setattr(anima_models, "CKPT_REUSE_ATTN", reuse)
    monkeypatch.setattr(anima_models, "CKPT_RECOMPUTE_DIFFS", [] if (nograd and diffs) else None)


def perturb(net, seed: int = 7, scale: float = 0.02) -> None:
    """LoKr starts with lokr_w2 = 0 (a zero delta): move every trained parameter so each one matters to the output."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            if p.requires_grad and p.is_floating_point():
                p.add_(torch.randn(p.shape, generator=g).to(p.dtype) * scale)


def build(compiled: bool = False, stream: bool = False, preset: str = "moss", exclude=(EXCLUDED,), layers: int = 3,
          module: str = "lycoris.kohya", backend="aot_eager", seed: int = 0):
    """The toy MOSS DiT (frozen bf16, train mode, every block checkpointed) with the network applied the way
    train_network.py does it; `exclude`: joint blocks left out of it (--moss_lora_exclude_blocks). compiled: the joint
    blocks compiled as moss_train_network.after_network_loaded does. Returns (model, network, compile state)."""
    torch.manual_seed(seed)
    model = MossTransformer(**{**TINY, "num_layers": layers}).to(torch.bfloat16)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.05)
    model.requires_grad_(False)
    model.train()
    network_args = [a.replace("preset=moss", f"preset={preset}") for a in LOKR] if module == "lycoris.kohya" else []
    args = argparse.Namespace(network_args=network_args, network_dim=10000, network_alpha=10000, network_module=module,
                              network_dropout=None, lycoris_stock=False, lycoris_algo_fixes=False, network_weights=None,
                              moss_lora_exclude_blocks=",".join(str(i) for i in exclude))
    moss_train_utils.apply_moss_block_exclusion(args)  # -> network_args exclude_patterns, as prepare_moss_runtime does
    net_kwargs = dict(i.split("=", 1) for i in args.network_args)
    laf.set_block_compile(compiled)
    if module == "lycoris.kohya":
        import lycoris.kohya as kohya

        net_kwargs = lycoris_args.fix_lycoris_bool_kwargs(args.network_module, net_kwargs)
        create_kwargs = laf.prepare_network_kwargs("lokr", dict(net_kwargs))
        create_kwargs.setdefault("dropout", 0.0)
        network = kohya.create_network(1.0, args.network_dim, args.network_alpha, None, None, model, neuron_dropout=None,
                                       **create_kwargs)
        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        network.apply_to(None, model, False, True)
        laf.after_network_created(network, args)
    else:
        from networks import lora_moss

        network = lora_moss.create_network(1.0, 4, 4, None, [], model, **net_kwargs)
        network.apply_to([], model, False, True)
    network.requires_grad_(True)
    network.train()
    perturb(network)
    if module == "lycoris.kohya":
        laf.prime_module_caches(network)
    if stream:
        model.enable_stream_fp32()
    model.enable_gradient_checkpointing()
    state = None
    if compiled:
        state = mbc.compile_moss_blocks(model, backend=backend, isolate_cache=False, stream_fp32=stream)
    return model, network, state


def batch(hw=(4, 4), text=(6,), seed: int = 1, stream: bool = False):
    """Latents, timesteps, padded text and its mask for len(text) samples with those text lengths."""
    g = torch.Generator().manual_seed(seed)
    rows, longest = len(text), max(text)
    dtype = torch.float32 if stream else torch.bfloat16  # the trainer: the noisy input and text in the stream's dtype
    x = torch.randn(rows, 128, *hw, generator=g).to(dtype)
    t = torch.full((rows,), 100.0 + 70.0 * (seed % 11))
    emb = torch.randn(rows, longest, TINY["text_dim"], generator=g).to(dtype)
    mask = torch.arange(longest)[None] < torch.tensor(text)[:, None]
    return x, t, emb, mask


def forward(model, data):
    """The trainer's forward of one micro-step (moss_train_network.get_noise_pred_and_target): the noisy input marked
    as grad-bearing under gradient checkpointing, bf16 autocast."""
    x, t, emb, mask = data
    x = x.detach().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        return model(x, t, emb, mask)


def micro_step(model, network, data, extra=()):
    """One micro-step; returns the output and the gradients of the network (and of the `extra` parameters)."""
    network.zero_grad(set_to_none=True)
    for p in extra:
        p.grad = None
    y = forward(model, data)
    y.float().square().mean().backward()
    grads = {n: p.grad.detach().clone() for n, p in network.named_parameters() if p.grad is not None}
    grads.update({f"extra{i}": p.grad.detach().clone() for i, p in enumerate(extra)})
    assert grads, "the network got no gradients"
    return y.detach().float().clone(), grads


def assert_same(ref, new, what: str):
    (y0, g0), (y1, g1) = ref, new
    assert torch.equal(y0, y1), f"{what}: output differs, max |d| = {(y0 - y1).abs().max().item():.3e}"
    assert g0.keys() == g1.keys(), f"{what}: different gradient sets"
    for name in g0:
        assert torch.equal(g0[name], g1[name]), (
            f"{what}: gradient of {name} differs, max |d| = {(g0[name].float() - g1[name].float()).abs().max().item():.3e}")


def cached_lokrs(model) -> list:
    return [lokr for block in model.layers for lokr in anima_models._ckpt_cached_lokrs(block)]


def run(monkeypatch, nograd: bool, reuse: bool = False, batches=((dict(),) * 2), **kw):
    """One micro-step per entry of `batches` (batch() keyword sets) on a fresh model under the flags; returns the list of
    (output, grads), the model, the network and the compile state."""
    model, net, state = build(**kw)
    set_flags(monkeypatch, nograd, reuse)
    results = [micro_step(model, net, batch(stream=kw.get("stream", False), **b)) for b in batches]
    return results, model, net, state


def sample(model, compiled: bool = False, seed: int = 3):
    """A sample round as moss_train_utils.sample_images runs it: eval mode (train_network.py switches the DiT to eval
    around sampling under gradient checkpointing), no_grad + autocast, the per-round merged LoKr cache, CFG with the
    negative branch skipping the block the network leaves out; the compiled blocks paused."""
    g = torch.Generator().manual_seed(seed)
    cond = (torch.randn(1, 7, TINY["text_dim"], generator=g), torch.ones(1, 7, dtype=torch.bool))
    neg = (torch.randn(1, 4, TINY["text_dim"], generator=g), torch.ones(1, 4, dtype=torch.bool))
    model.eval()
    try:
        with mbc.pause(), torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            with lycoris_fast.cached_lokr_weights(model, torch.bfloat16, torch.device("cpu")):
                return moss_train_utils.moss_sample(model, cond, neg, 64, 64, 3, 3.0, 5.0, seed, torch.device("cpu"),
                                                    skip_layers=(EXCLUDED,)).clone()
    finally:
        model.train()


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


class Counter:
    """aot_eager backend that counts the graphs dynamo hands it (one per compile)."""

    def __init__(self):
        self.graphs = 0
        self.inner = lookup_backend("aot_eager")

    def __call__(self, gm, example_inputs):
        self.graphs += 1
        return self.inner(gm, example_inputs)


# ------------------------------------------------------------------------------------------------------ flags off / on


def test_flags_off_keep_the_plain_checkpoint(monkeypatch):
    """Both switches off (the defaults): every block hands the arguments it always did to torch.utils.checkpoint, the
    no-grad node is never built, attention gets neither a stash nor a stored output, lycoris_fast._CKPT_PASS is 0 in
    every attention call and nothing is cached."""
    assert anima_models.CKPT_NOGRAD_MERGED is False and anima_models.CKPT_REUSE_ATTN is False
    assert moss_models.ckpt_nograd_merged_enabled() is False
    model, net, _ = build()
    seen = {"torch": [], "nograd": 0, "passes": set(), "attn": 0}
    real_ckpt, real_attention = moss_models.torch_checkpoint, attention.attention

    def spy_ckpt(fn, *args, **kwargs):
        assert kwargs == {"use_reentrant": False}
        seen["torch"].append(len(args))
        return real_ckpt(fn, *args, **kwargs)

    def spy_nograd(*args):
        seen["nograd"] += 1

    def spy_attention(qkv, attn_params=None):  # the call as it was before the flags existed: no other argument
        seen["attn"] += 1
        seen["passes"].add(lycoris_fast._CKPT_PASS)
        return real_attention(qkv, attn_params=attn_params)

    monkeypatch.setattr(moss_models, "torch_checkpoint", spy_ckpt)
    monkeypatch.setattr(anima_models, "nograd_checkpoint", spy_nograd)
    monkeypatch.setattr(attention, "attention", spy_attention)
    micro_step(model, net, batch())
    # two text refiner blocks (x, rope, params) then the joint blocks (x, rope, modulation, params); one attention per
    # block, in the first pass and in the recompute
    assert seen == {"torch": [3, 3] + [4] * len(model.layers), "nograd": 0, "passes": {0},
                    "attn": 2 * (2 + len(model.layers))}
    assert lycoris_fast.ckpt_merged_bytes(model) == 0
    assert all("_ckpt_merged" not in lokr.__dict__ for lokr in laf._all_loras(net))


BATCHES = {
    "one_sample": (dict(), dict(seed=2)),
    "text_lengths_and_sizes": (dict(text=(6,)), dict(text=(11,), hw=(4, 6), seed=2), dict(text=(3,), hw=(6, 6), seed=3)),
    "equal_lengths": (dict(text=(7, 7)), dict(text=(5, 5, 5), seed=2)),
    "padded": (dict(text=(5, 9)), dict(text=(9, 4, 6), hw=(4, 6), seed=2)),
}


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
@pytest.mark.parametrize("stream", [False, True], ids=["bf16_stream", "fp32_stream"])
@pytest.mark.parametrize("kind", list(BATCHES))
def test_flags_on_match_the_plain_checkpoint(kind, stream, compiled, monkeypatch):
    """Output and every network gradient: torch.utils.checkpoint == the no-grad first pass with cached merged weights ==
    the same with the stored joint-attention output, over consecutive micro-steps (the cache built, then reused) of one
    sample, of changing text lengths / image sizes, of equal-length batches and of padded batches. The recompute's block
    outputs equal the first pass's bit for bit."""
    batches = BATCHES[kind]
    ref, *_ = run(monkeypatch, False, batches=batches, compiled=compiled, stream=stream)
    for reuse in (False, True):
        new, model, net, state = run(monkeypatch, True, reuse, batches=batches, compiled=compiled, stream=stream)
        for i, (a, b) in enumerate(zip(ref, new)):
            assert_same(a, b, f"{kind}, micro-step {i}, reuse={reuse}: no-grad first pass vs torch.utils.checkpoint")
        diffs = anima_models.CKPT_RECOMPUTE_DIFFS
        if kind == "padded" and compiled:  # a compiled block's padded call keeps torch.utils.checkpoint: no node at all
            assert diffs == [] and lycoris_fast.ckpt_merged_bytes(model) == 0
        else:
            assert len(diffs) == len(batches) * len(model.layers) and set(diffs) == {0.0}, diffs
            assert lycoris_fast.ckpt_merged_bytes(model) > 0
        if compiled:
            assert state.eager_calls == 0 and state.compiled_calls == 2 * len(batches) * len(model.layers)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
@pytest.mark.parametrize("kind, reused", [("one_sample", True), ("equal_lengths", True), ("padded", False)])
def test_reuse_skips_one_flash_forward_per_joint_block(kind, reused, compiled, monkeypatch):
    """The plain flash call (one sample, or equal text lengths: attention() trims a uniform mask away) is stored and
    reused: one forward kernel run per joint block and micro-step instead of two, the backward kernel count unchanged.
    A padded batch goes through flash varlen: nothing is stored, the kernel counts are the plain checkpoint's."""
    batches = BATCHES[kind][:1]
    counts = {}
    for name, flags in (("off", (False, False)), ("nograd", (True, False)), ("reuse", (True, True))):
        _, model, net, _ = run(monkeypatch, *flags, batches=batches, compiled=compiled)  # compiles, builds the cache
        fake_flash.CALLS["fwd"] = fake_flash.CALLS["bwd"] = VARLEN_CALLS["fwd"] = VARLEN_CALLS["bwd"] = 0
        micro_step(model, net, batch(**batches[0]))
        counts[name] = (dict(fake_flash.CALLS), dict(VARLEN_CALLS))
    layers = TINY["num_layers"]
    assert counts["nograd"] == counts["off"]
    if reused:
        # per micro-step: 2 text refiner blocks and `layers` joint blocks, each a first pass and a recompute
        assert counts["off"] == ({"fwd": 2 * (2 + layers), "bwd": 2 + layers}, {"fwd": 0, "bwd": 0})
        assert counts["reuse"] == ({"fwd": 2 * 2 + layers, "bwd": 2 + layers}, {"fwd": 0, "bwd": 0})
    else:
        assert counts["off"][1] == {"fwd": 2 * (2 + layers), "bwd": 2 + layers}
        assert counts["reuse"] == counts["off"]


def test_the_block_left_out_of_the_network_has_no_cache_but_the_node_and_the_reuse(monkeypatch):
    """--moss_lora_exclude_blocks: the excluded block has no LoKr module (nothing to cache) and still runs the no-grad
    first pass with the stored joint attention; the other blocks cache their six adapted Linears."""
    model, net, _ = build()
    set_flags(monkeypatch, True, True)
    calls, stored = [], []
    real_node, real_lse = anima_models.nograd_checkpoint, attention._flash_forward_with_lse

    def spy_node(fn, block, *args):
        calls.append(block)
        return real_node(fn, block, *args)

    def spy_lse(q, k, v):
        stored.append(1)
        return real_lse(q, k, v)

    monkeypatch.setattr(anima_models, "nograd_checkpoint", spy_node)
    monkeypatch.setattr(attention, "_flash_forward_with_lse", spy_lse)
    micro_step(model, net, batch())
    assert calls == list(model.layers) and len(stored) == len(model.layers)  # joint blocks only, not the text refiner
    per_block = [len(anima_models._ckpt_cached_lokrs(b)) for b in model.layers]
    assert per_block == [PER_BLOCK, 0, PER_BLOCK]
    names = {id(m): n for n, m in model.layers[0].named_modules()}
    assert sorted(names[id(l.org_module[0])] for l in anima_models._ckpt_cached_lokrs(model.layers[0])) == sorted(
        f"proj.{n}" for n in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"))
    held = lycoris_fast.ckpt_merged_bytes(model)
    assert held == sum(l.org_module[0].weight.numel() for l in cached_lokrs(model)) * 2  # one bf16 copy each
    # the cache lives in the modules' __dict__: neither a saved LoRA nor a --save_state / --resume state sees it
    assert not any("ckpt" in key for key in list(net.state_dict()) + list(model.state_dict()))
    assert lycoris_fast.ckpt_merged_release(model) == held and lycoris_fast.ckpt_merged_bytes(model) == 0


def test_block_without_grad_input_keeps_the_plain_checkpoint(monkeypatch):
    """An autograd node needs a grad-bearing input. The trainer marks the noisy latents; MossTextProjection marks the
    text when a text block checkpoints, so the joint stream carries grad either way - a joint block called directly with
    an x that has none stays on torch.utils.checkpoint."""
    model, net, _ = build()
    set_flags(monkeypatch, True, True)
    calls = []
    real = anima_models.nograd_checkpoint
    monkeypatch.setattr(anima_models, "nograd_checkpoint", lambda *a: calls.append(1) or real(*a))
    x, t, emb, mask = batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = model(x, t, emb, mask)  # the latents not marked
    y.float().square().mean().backward()
    assert len(calls) == len(model.layers)  # the text refiner's output requires grad
    calls.clear()
    block = model.layers[0]
    hidden = torch.randn(1, 22, TINY["dim"]).to(torch.bfloat16)
    rope = model.rope.batched(4, 4, [6], 6, hidden.device).to(hidden.dtype)
    mod, _ = model.time_conditioning(t, 1, hidden.device)
    params = attention.AttentionParams.create_attention_params("flash", False)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = block(hidden, rope, mod, params)
    assert calls == [] and out.requires_grad  # the plain checkpoint: the LoKr parameters carry the grad
    out.float().sum().backward()


# ------------------------------------------------------------------------------------------------- merged-weight cache


def test_first_pass_rebuilds_once_per_optimizer_step(monkeypatch):
    """Eager, 3 micro-steps accumulated per optimizer step, 2 optimizer steps. Plain checkpoint: every micro-step
    rebuilds each cached module's weight twice in grad mode (first pass + recompute). Flags on: once per micro-step in
    grad mode (the recompute) and once per OPTIMIZER step without grad (the cache), into the same tensor object; the
    trained parameters end bit-identical."""
    spy = RebuildSpy(monkeypatch)

    def train(nograd):
        model, net, _ = build()
        set_flags(monkeypatch, nograd, nograd)
        params = [p for p in net.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=1e-2)
        counts, tensors = [], []
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            for micro in range(3):
                y = forward(model, batch(seed=micro + 1, text=(5 + micro,)))
                (y.float().square().mean() / 3).backward()
                counts.append(spy.take())
                tensors.append([lokr.__dict__.get("_ckpt_merged") for lokr in cached_lokrs(model)])
            opt.step()
        return [p.detach().clone() for p in params], counts, tensors, len(cached_lokrs(model))

    ref, plain_counts, _, n = train(False)
    new, counts, tensors, n2 = train(True)
    assert n == n2 == 2 * PER_BLOCK  # two adapted joint blocks
    assert plain_counts == [(0, 2 * n)] * 6, plain_counts
    assert counts == [(n, n), (0, n), (0, n)] * 2, counts
    assert all(a is b for row in tensors[1:] for a, b in zip(tensors[0], row)), "a cached tensor was replaced"
    assert all(torch.equal(a, b) for a, b in zip(ref, new)), "trained parameters differ"


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
def test_cache_is_only_read_inside_the_checkpoint(compiled, monkeypatch):
    """A poisoned cache (every cached weight scaled, the keys still current): the checkpointed training forward changes -
    the first pass really reads it, eager and compiled - and its recompute still agrees with the first pass
    (_FirstPassBits). Sample images (CFG, the negative branch skipping a block), a validation forward (eval, no grad) and
    un-checkpointed blocks do not change: they never read it. After ckpt_merged_invalidate the cache is rebuilt in place
    and the training output is the reference again."""
    model, net, state = build(compiled=compiled)
    data = batch()

    def validation():
        model.eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(*data).float().clone()
        model.train()
        return out

    def unchecked():
        model.disable_gradient_checkpointing()
        out = micro_step(model, net, data)
        model.enable_gradient_checkpointing()
        return out

    set_flags(monkeypatch, False)
    ref_train, ref_sample, ref_val, ref_plain = micro_step(model, net, data), sample(model), validation(), unchecked()
    set_flags(monkeypatch, True, True)
    assert_same(ref_train, micro_step(model, net, data), "clean cache")
    tensors = [lokr.__dict__["_ckpt_merged"] for lokr in cached_lokrs(model)]
    assert len(tensors) == 2 * PER_BLOCK
    with torch.no_grad():
        for cached in tensors:
            cached.mul_(1.25)
    anima_models.CKPT_RECOMPUTE_DIFFS.clear()
    poisoned = micro_step(model, net, data)
    assert not torch.equal(poisoned[0], ref_train[0]), "the first pass did not read the cache"
    assert set(anima_models.CKPT_RECOMPUTE_DIFFS) == {0.0}  # the recompute computes with the first pass's weight bits
    assert torch.equal(sample(model), ref_sample), "sample images read the checkpoint cache"
    assert torch.equal(validation(), ref_val), "validation read the checkpoint cache"
    assert_same(ref_plain, unchecked(), "un-checkpointed blocks with a poisoned cache")
    assert lycoris_fast._CKPT_PASS == 0
    lycoris_fast.ckpt_merged_invalidate()
    assert_same(ref_train, micro_step(model, net, data), "after ckpt_merged_invalidate")
    assert all(a is b for a, b in zip(tensors, (lokr.__dict__["_ckpt_merged"] for lokr in cached_lokrs(model))))


def test_sampling_passes_no_stash_and_no_stored_output(monkeypatch):
    """Sample images and validation with both switches on: MossBlock._run calls _forward directly (no checkpoint node),
    so attention never gets a stash or a stored output and lycoris_fast._CKPT_PASS stays 0; a training micro-step right
    after still stores and reuses."""
    model, net, _ = build()
    set_flags(monkeypatch, True, True)
    micro_step(model, net, batch())  # the cache exists and is current
    seen = []
    real_attention, real_node = attention.attention, anima_models.nograd_checkpoint

    def spy_attention(qkv, attn_params=None, **kwargs):
        seen.append((lycoris_fast._CKPT_PASS, sorted(kwargs)))
        return real_attention(qkv, attn_params=attn_params, **kwargs)

    nodes = []
    monkeypatch.setattr(attention, "attention", spy_attention)
    monkeypatch.setattr(anima_models, "nograd_checkpoint", lambda *a: nodes.append(1) or real_node(*a))
    sample(model)
    model.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        model(*batch())
    model.train()
    assert seen and all(entry == (0, []) for entry in seen) and nodes == []
    seen.clear()
    micro_step(model, net, batch())
    layers = len(model.layers)
    assert len(nodes) == layers
    joint = [entry for entry in seen if entry[1]]
    assert joint == [(1, ["ckpt_reuse", "ckpt_stash"])] * layers + [(2, ["ckpt_reuse", "ckpt_stash"])] * layers


def test_parameters_changed_before_backward_recompute_plainly(monkeypatch):
    """A parameter written between a forward and its backward (nothing in the trainer does this): the recompute must not
    trust the cached bits or the stored attention output. It runs plainly on the new values, as torch.utils.checkpoint's
    recompute does, and the gradients equal the plain checkpoint's."""
    results = []
    for nograd in (False, True):
        model, net, _ = build()
        set_flags(monkeypatch, nograd, nograd)
        micro_step(model, net, batch())
        net.zero_grad(set_to_none=True)
        before = fake_flash.CALLS["fwd"]
        y = forward(model, batch(seed=2))
        with torch.no_grad():
            for p in net.parameters():
                p.mul_(1.01)  # bumps the version counters: the keys see it
        y.float().square().mean().backward()
        results.append((y.detach().float().clone(), {n: p.grad.clone() for n, p in net.named_parameters()},
                        fake_flash.CALLS["fwd"] - before))
    (y0, g0, fwd0), (y1, g1, fwd1) = results
    assert_same((y0, g0), (y1, g1), "parameters changed between forward and backward")
    layers = TINY["num_layers"]
    # the adapted blocks ran the forward kernel again (their keys changed); the block left out of the network has no
    # parameter that could change, its recompute sees the first pass's input again and may reuse the stored output
    assert (fwd0, fwd1) == (2 * (2 + layers), 2 * 2 + layers + (layers - 1))
    diffs = anima_models.CKPT_RECOMPUTE_DIFFS[-layers:]  # backward visits the blocks last to first
    assert diffs[layers - 1 - EXCLUDED] == 0.0 and all(d > 0.0 for i, d in enumerate(diffs) if i != layers - 1 - EXCLUDED)


def test_stored_tensors_are_released_after_backward(monkeypatch):
    """The node drops its (out, softmax_lse) when its backward starts and the recompute graph when it ends: nothing of a
    block's stored attention survives the micro-step."""
    model, net, _ = build()
    set_flags(monkeypatch, True, True)
    refs = []
    real = attention._flash_forward_with_lse

    def spy(q, k, v):
        out, lse = real(q, k, v)
        refs.append((weakref.ref(out), weakref.ref(lse)))
        return out, lse

    monkeypatch.setattr(attention, "_flash_forward_with_lse", spy)
    net.zero_grad(set_to_none=True)
    y = forward(model, batch())
    loss = y.float().square().mean()
    assert len(refs) == len(model.layers) and all(o() is not None and l() is not None for o, l in refs)
    loss.backward()
    del y, loss
    gc.collect()
    assert all(o() is None and l() is None for o, l in refs), "a stored attention output outlived its block's backward"


# ------------------------------------------------------------------------------------- training runs like the trainer's

# 3 optimizer steps x 3 accumulated micro-steps: one sample with changing text lengths and image sizes (the user's batch
# 1), an equal-length batch and a padded batch
PLAN = (dict(text=(6,)), dict(text=(11,), hw=(4, 6)), dict(text=(7, 7)),
        dict(text=(4,), hw=(6, 6)), dict(text=(5, 9)), dict(text=(6,)),
        dict(text=(13,)), dict(text=(6,), hw=(4, 6)), dict(text=(8,)))


def simulate(monkeypatch, flags, *, make_opt, invalidate=("hook", "trainer"), compiled=False, stream=False, plan=PLAN,
             accum=3, between=None, bypass_step_hooks=False, backend="aot_eager"):
    """A short training run driven like train_network.py's loop: per micro-step forward, loss / accumulation, backward;
    every `accum` micro-steps gradient clipping (max_grad_norm 1.0), optimizer.step, zero_grad and then the trainer's
    on_optimizer_step_end.

    invalidate: which invalidation layers exist - "hook" (the global optimizer-step hook prepare_moss_runtime registers)
    and / or "trainer" (MossNetworkTrainer.on_optimizer_step_end).
    bypass_step_hooks: step through the undecorated optimizer.step (no torch.optim step hooks run), like step_func.
    between(i, model, net, trainer): called after micro-step i (0-based) and its optimizer step, if any.
    Returns (outputs of every micro-step, the trained parameters, extras)."""
    import moss_train_network as mtn

    nograd, reuse = flags
    model, net, state = build(compiled=compiled, stream=stream, backend=backend)
    set_flags(monkeypatch, nograd, reuse)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = make_opt(params)
    if "hook" in invalidate and nograd:
        lycoris_fast.ckpt_merged_invalidate_on_optimizer_steps()
    trainer = mtn.MossNetworkTrainer()
    outputs, rebuilds = [], []
    spy = {"nograd": 0}
    inner = lycoris_fast._REBUILD.eager["rebuild"]

    def counting_rebuild(*args):
        spy["nograd"] += not torch.is_grad_enabled()
        return inner(*args)

    if not compiled:  # traced into a compiled block graph, the counter would be guarded (a recompile per count)
        monkeypatch.setitem(lycoris_fast._REBUILD.eager, "rebuild", counting_rebuild)
    for i, spec in enumerate(plan):
        y = forward(model, batch(seed=i + 1, stream=stream, **spec))
        (y.float().square().mean() / accum).backward()
        outputs.append(y.detach().float().clone())
        del y
        if (i + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            if bypass_step_hooks:
                type(opt).step.__wrapped__(opt)
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            if "trainer" in invalidate:
                trainer.on_optimizer_step_end(argparse.Namespace(), None, (i + 1) // accum, False)
            rebuilds.append(spy["nograd"])
            spy["nograd"] = 0
        if between is not None:
            between(i, model, net, trainer)
    monkeypatch.setitem(lycoris_fast._REBUILD.eager, "rebuild", inner)
    _remove_step_hook()  # one run's hook must not serve the next run of the same test
    extras = {"model": model, "net": net, "state": state, "nograd_rebuilds": rebuilds,
              "diffs": anima_models.CKPT_RECOMPUTE_DIFFS}
    return outputs, [p.detach().clone() for p in params], extras


def assert_same_run(ref, new, what: str):
    (y0, p0, _), (y1, p1, _) = ref, new
    assert len(y0) == len(y1) and len(p0) == len(p1)
    for i, (a, b) in enumerate(zip(y0, y1)):
        assert torch.equal(a, b), f"{what}: output of micro-step {i} differs, max |d| = {(a - b).abs().max().item():.3e}"
    for i, (a, b) in enumerate(zip(p0, p1)):
        assert torch.equal(a, b), f"{what}: trained parameter {i} differs, max |d| = {(a - b).abs().max().item():.3e}"


def adamw(params):
    return torch.optim.AdamW(params, lr=2e-3, weight_decay=0.1)


def test_came_steps_are_invisible_to_the_cache_keys(came_cls):
    """The premise of the invalidation hooks, with the real optimizer: a CAME step changes every trained parameter but
    neither its version counter nor its storage, so the keys still call the cached weights current."""
    model, net, _ = build()
    lokrs = cached_lokrs(model)
    keys = lycoris_fast.ckpt_merged_refresh(lokrs, torch.bfloat16)
    params = [p for p in net.parameters() if p.requires_grad]
    before = [(p._version, p.data_ptr(), p.detach().clone()) for p in params]
    opt = make_came(came_cls, params)
    for p in params:
        p.grad = torch.randn(p.shape, generator=torch.Generator().manual_seed(1))
    opt.step()
    assert all(p._version == v and p.data_ptr() == ptr for p, (v, ptr, _) in zip(params, before))
    assert all(not torch.equal(p, old) for p, (_, _, old) in zip(params, before))
    assert lycoris_fast.ckpt_merged_current(lokrs, keys, torch.bfloat16)  # stale, and the keys cannot tell


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
@pytest.mark.parametrize("layers", [("hook",), ("trainer",), ("hook", "trainer")], ids=["hook", "trainer", "both"])
def test_came_training_matches_the_plain_checkpoint(layers, compiled, came_cls, monkeypatch):
    """3 optimizer steps x 3 accumulated micro-steps with CAME (it writes through p.data): every micro-step's output and
    the trained parameters are bit-identical to the flags-off run, with either invalidation layer alone and with both.
    At most one no-grad rebuild per cached module and optimizer step (the cache is not rebuilt per micro-step)."""
    opt = lambda params: make_came(came_cls, params)  # noqa: E731
    ref = simulate(monkeypatch, (False, False), make_opt=opt, compiled=compiled)
    new = simulate(monkeypatch, (True, True), make_opt=opt, invalidate=layers, compiled=compiled)
    assert_same_run(ref, new, f"CAME, invalidation by {layers}, compiled={compiled}")
    extras = new[2]
    # compiled, the one padded micro-step of the plan keeps torch.utils.checkpoint (no node, no diff recorded)
    nodes = len(PLAN) - (sum(len(set(spec.get("text", (6,)))) > 1 for spec in PLAN) if compiled else 0)
    assert len(extras["diffs"]) == nodes * TINY["num_layers"] and set(extras["diffs"]) == {0.0}, extras["diffs"]
    if not compiled:
        assert extras["nograd_rebuilds"] == [2 * PER_BLOCK] * 3, extras["nograd_rebuilds"]
    else:
        assert extras["state"].eager_calls == 0


def test_came_training_without_any_invalidation_goes_stale(came_cls, monkeypatch):
    """Negative control: with neither layer the first pass keeps reading the weights of step 0 while CAME moves the
    parameters - the outputs leave the flags-off run at the first micro-step after the first optimizer step. (Nothing in
    the trainer runs like this; it shows what the two layers hold up, and that the comparison above would catch a cache
    that stopped refreshing.)"""
    opt = lambda params: make_came(came_cls, params)  # noqa: E731
    ref = simulate(monkeypatch, (False, False), make_opt=opt)
    stale = simulate(monkeypatch, (True, True), make_opt=opt, invalidate=())
    same = [torch.equal(a, b) for a, b in zip(ref[0], stale[0])]
    assert same[:3] == [True, True, True], same  # before the first optimizer step nothing is stale
    assert not any(same[3:]), same
    assert stale[2]["nograd_rebuilds"][1:] == [0, 0]  # nothing was rebuilt after the first optimizer step


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
@pytest.mark.parametrize("layers", [(), ("hook", "trainer")], ids=["keys_only", "both"])
def test_adamw_training_matches_the_plain_checkpoint(layers, compiled, monkeypatch):
    """AdamW (the optimizer of the current MOSS config) updates the parameters in place: their version counters move, so
    the cache keys alone already follow it; with the two invalidation layers on top the result is the same."""
    ref = simulate(monkeypatch, (False, False), make_opt=adamw, compiled=compiled)
    new = simulate(monkeypatch, (True, True), make_opt=adamw, invalidate=layers, compiled=compiled)
    assert_same_run(ref, new, f"AdamW, invalidation by {layers}, compiled={compiled}")
    assert set(new[2]["diffs"]) == {0.0}


def test_optimizer_stepping_past_the_step_hooks_is_covered_by_the_trainer(came_cls, monkeypatch):
    """step_func (train_network.py: Polyak-step optimizers) and any optimizer whose step does not run torch.optim's step
    hooks: the global hook never fires, MossNetworkTrainer.on_optimizer_step_end alone keeps the cache right."""
    opt = lambda params: make_came(came_cls, params)  # noqa: E731
    ref = simulate(monkeypatch, (False, False), make_opt=opt, bypass_step_hooks=True)
    new = simulate(monkeypatch, (True, True), make_opt=opt, bypass_step_hooks=True)
    assert_same_run(ref, new, "optimizer stepped without step hooks")


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
@pytest.mark.parametrize("stream", [False, True], ids=["bf16_stream", "fp32_stream"])
def test_sampling_resume_and_validation_inside_a_run(stream, compiled, came_cls, monkeypatch):
    """What happens between micro-steps of a real run, flags on vs off with CAME: a sample round (per-round merged LoKr
    cache, CFG with the skipped block, then the trainer's after_sample_images) after an optimizer step and in the middle
    of an accumulation, a resume-like load_state_dict of other adapter values in the middle of an accumulation, and a
    validation forward. Outputs of every micro-step, the samples and the trained parameters are bit-identical."""
    opt = lambda params: make_came(came_cls, params)  # noqa: E731
    seen = {}

    def between(key):
        def call(i, model, net, trainer):
            if i in (1, 2, 5):  # mid-accumulation, right after an optimizer step, after the second one
                generation = lycoris_fast._CKPT_GENERATION
                seen.setdefault(key, []).append(sample(model, seed=i))
                trainer.after_sample_images()
                assert lycoris_fast._CKPT_GENERATION == generation + int(anima_models.CKPT_NOGRAD_MERGED)
            if i == 4:  # a state loaded in the middle of an accumulation: every adapter value replaced
                g = torch.Generator().manual_seed(11)
                state = {k: (v + torch.randn(v.shape, generator=g).to(v.dtype) * 0.01 if v.is_floating_point() and v.dim()
                             else v.clone()) for k, v in net.state_dict().items()}
                net.load_state_dict(state)
            if i == 6:
                model.eval()
                with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                    seen.setdefault(key, []).append(model(*batch(seed=20, stream=stream)).float().clone())
                model.train()
        return call

    ref = simulate(monkeypatch, (False, False), make_opt=opt, compiled=compiled, stream=stream, between=between("off"))
    new = simulate(monkeypatch, (True, True), make_opt=opt, compiled=compiled, stream=stream, between=between("on"))
    assert_same_run(ref, new, f"sampling / resume / validation inside the run, compiled={compiled}, stream={stream}")
    assert len(seen["off"]) == len(seen["on"]) == 4
    assert all(torch.equal(a, b) for a, b in zip(seen["off"], seen["on"])), "samples / validation differ"
    assert set(new[2]["diffs"]) == {0.0}
    if compiled:
        assert new[2]["state"].eager_calls > 0  # the sample rounds and the validation forward ran the eager blocks


# --------------------------------------------------------------------- gradients that leave the joint blocks' Linears


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
def test_text_refiner_training_matches(compiled, monkeypatch):
    """preset moss-text: LoKr on the two text refiner blocks too. They keep torch.utils.checkpoint; their output enters
    joint block 0 as part of its x, so their gradients come through the first joint block's node - bit-identical."""
    batches = (dict(text=(6,)), dict(text=(5, 9), seed=2))
    ref, _, net0, _ = run(monkeypatch, False, batches=batches, compiled=compiled, preset="moss-text")
    new, model, net, _ = run(monkeypatch, True, True, batches=batches, compiled=compiled, preset="moss-text")
    text_grads = [n for n in ref[0][1] if "text_embedder" in n]
    assert len(text_grads) >= 2 * 5 and all(bool(ref[0][1][n].any()) for n in text_grads)
    for i, (a, b) in enumerate(zip(ref, new)):
        assert_same(a, b, f"moss-text, micro-step {i}")
    assert set(anima_models.CKPT_RECOMPUTE_DIFFS) == {0.0}
    # the text refiner's modules are not in any joint block's cache
    assert len(cached_lokrs(model)) == 2 * PER_BLOCK < len(laf._all_loras(net))


@pytest.mark.parametrize("stream", [False, True], ids=["bf16_stream", "fp32_stream"])
def test_trained_time_conditioning_gets_the_same_gradients(stream, monkeypatch):
    """The shared modulation enters every joint block from outside the checkpoints. Each block reads it once, so the
    node hands its producer one gradient per block exactly as the plain checkpoint does: with the time MLP and the
    shared modulation Linear trained, their gradients (and everything else) are bit-identical."""
    results = []
    for flags in ((False, False), (True, True)):
        model, net, _ = build(stream=stream)
        extra = [model.shared_block_modulation.weight, model.time_embedding[0].weight, model.time_embedding[2].bias]
        for p in extra:
            p.requires_grad_(True)
        set_flags(monkeypatch, *flags)
        results.append([micro_step(model, net, batch(stream=stream, **b), extra=extra)
                        for b in (dict(text=(6,)), dict(text=(5, 9), seed=2))])
    for i, (a, b) in enumerate(zip(*results)):
        assert all(bool(a[1][f"extra{j}"].any()) for j in range(3)), "the time conditioning got no gradient"
        assert_same(a, b, f"trained time conditioning, micro-step {i}")


def test_outside_grad_note_names_the_trained_time_modules(monkeypatch):
    class Net:
        def __init__(self, names):
            self.names = names

        def named_parameters(self):
            return [(n, nn.Parameter(torch.zeros(1))) for n in self.names]

    plain = Net(["lora_unet_layers_0_proj_q.lokr_w1", "lora_unet_text_embedder_blocks_0_qkv.lokr_w1"])
    timed = Net(["lora_unet_time_embedding_0.lokr_w1", "lora_unet_shared_block_modulation.lokr_w2"])
    assert moss_train_utils.ckpt_reuse_outside_grad_note(timed) is None  # the switch is off
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", True)
    assert moss_train_utils.ckpt_reuse_outside_grad_note(plain) is None
    assert moss_train_utils.ckpt_reuse_outside_grad_note(None) is None
    note = moss_train_utils.ckpt_reuse_outside_grad_note(timed)
    assert "shared_block_modulation and time_embedding" in note


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
def test_plain_lora_network_gets_the_node_without_a_cache(compiled, monkeypatch):
    """networks.lora_moss (not LyCORIS): nothing to cache, the no-grad first pass and the attention reuse still apply
    and the gradients are the plain checkpoint's."""
    batches = (dict(text=(6,)), dict(text=(9,), seed=2))
    ref, *_ = run(monkeypatch, False, batches=batches, compiled=compiled, module="networks.lora_moss")
    new, model, net, _ = run(monkeypatch, True, True, batches=batches, compiled=compiled, module="networks.lora_moss")
    for i, (a, b) in enumerate(zip(ref, new)):
        assert_same(a, b, f"networks.lora_moss, micro-step {i}")
    assert lycoris_fast.ckpt_merged_bytes(model) == 0 and cached_lokrs(model) == []
    assert set(anima_models.CKPT_RECOMPUTE_DIFFS) == {0.0}


# ------------------------------------------------------------------------------------------------------ torch.compile


def test_graph_count_is_flat_across_optimizer_steps_and_caption_lengths(monkeypatch):
    """Compiled, flags on, AdamW: the first micro-step compiles a first-pass and a recompute graph for the adapted blocks
    and for the block left out of the network (4 graphs; flags off: 2), and nothing after it - the refreshed cache
    tensors are graph inputs updated in place, and other caption lengths / image sizes of one sample share the graphs.
    The token axis of x, rope and of the stored attention handed to every recompute is soft-marked."""
    counter = Counter()
    one_sample = tuple(dict(text=(n,), hw=hw) for n, hw in ((6, (4, 4)), (11, (4, 6)), (4, (6, 6)), (9, (4, 4)),
                                                           (13, (4, 6)), (5, (4, 4)), (7, (6, 6)), (12, (4, 4)),
                                                           (10, (4, 6))))
    counts, stored_marks = [], []
    real_mark = torch._dynamo.maybe_mark_dynamic

    def mark(tensor, index):
        if tensor.dim() == 4 or index == 2:  # out [B, L, heads, head_dim] at 1, softmax_lse [B, heads, L] at 2
            stored_marks.append((tensor.dim(), index))
        return real_mark(tensor, index)

    def between(i, model, net, trainer):
        counts.append(counter.graphs)

    monkeypatch.setattr(torch._dynamo, "maybe_mark_dynamic", mark)
    new = simulate(monkeypatch, (True, True), make_opt=adamw, compiled=True, plan=one_sample, backend=counter,
                   between=between)
    assert new[2]["state"].eager_calls == 0
    assert counts == [4] * len(one_sample), counts
    assert stored_marks == [(4, 1), (3, 2)] * (len(one_sample) * TINY["num_layers"])


# ---------------------------------------------------------------------------------------------------- the trainer glue


def _args(**changes):
    base = dict(moss_ckpt_nograd_merged=True, moss_ckpt_reuse_attn=True, gradient_checkpointing=True, attn_mode="flash",
                split_attn=False, unsloth_offload_checkpointing=False, cpu_offload_checkpointing=False,
                blocks_to_swap=None, use_ramtorch=False, use_ramtorch_network=False, moss_stock_lycoris=False,
                train_batch_size=1)
    base.update(changes)
    return types.SimpleNamespace(**base)


@pytest.fixture
def warned(monkeypatch):
    messages = []
    monkeypatch.setattr(moss_train_utils.logger, "warning", lambda message, *a, **k: messages.append(str(message)))
    return messages


def test_validation_accepts_the_production_setup_and_off_is_unchecked(monkeypatch, warned):
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", True)  # prepare_moss_runtime ran
    moss_train_utils.validate_moss_ckpt_reuse_args(_args())
    moss_train_utils.validate_moss_ckpt_reuse_args(_args(moss_ckpt_reuse_attn=False, attn_mode="torch"))
    moss_train_utils.validate_moss_ckpt_reuse_args(_args(moss_ckpt_nograd_merged=False, moss_ckpt_reuse_attn=False,
                                                         use_ramtorch=True, blocks_to_swap=4, attn_mode=None,
                                                         unsloth_offload_checkpointing=True))
    assert warned == []


# (--moss_ckpt_reuse_attn without --moss_ckpt_nograd_merged is no longer refused: it is the reuse-only mode,
# tests/test_moss_ckpt_reuse_only.py)
@pytest.mark.parametrize("change, word", [
    (dict(unsloth_offload_checkpointing=True), "unsloth_offload_checkpointing"),
    (dict(cpu_offload_checkpointing=True), "cpu_offload_checkpointing"),
    (dict(blocks_to_swap=4), "blocks_to_swap"),
    (dict(use_ramtorch=True), "RamTorch"),
    (dict(use_ramtorch_network=True), "RamTorch"),
    (dict(moss_stock_lycoris=True), "moss_stock_lycoris"),
    (dict(attn_mode="torch"), "flash"),
    (dict(attn_mode=None), "flash"),
    (dict(attn_mode="xformers"), "flash"),
    (dict(split_attn=True), "split_attn"),
])
def test_validation_refuses(change, word):
    with pytest.raises(ValueError, match=word):
        moss_train_utils.validate_moss_ckpt_reuse_args(_args(**change))


def test_validation_refuses_a_flash_attn_without_the_backward_op(monkeypatch):
    monkeypatch.setattr(attention, "_wrapped_flash_attn_backward", None)
    with pytest.raises(ValueError, match="no separate backward op"):
        moss_train_utils.validate_moss_ckpt_reuse_args(_args())
    moss_train_utils.validate_moss_ckpt_reuse_args(_args(moss_ckpt_reuse_attn=False))  # the cache alone does not need it


def test_validation_warns_where_a_flag_has_no_effect(monkeypatch, warned):
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", True)
    moss_train_utils.validate_moss_ckpt_reuse_args(_args(gradient_checkpointing=False))
    assert len(warned) == 1 and "changes nothing without --gradient_checkpointing" in warned[0]
    warned.clear()
    moss_train_utils.validate_moss_ckpt_reuse_args(_args(train_batch_size=4))
    assert len(warned) == 1 and "batch > 1 (largest batch size 4)" in warned[0]
    warned.clear()
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", False)  # the options parsed, the runtime never switched on
    moss_train_utils.validate_moss_ckpt_reuse_args(_args())
    assert len(warned) == 1 and "has no effect here" in warned[0]


def test_parser_runtime_switches_and_invalidation_hooks(monkeypatch):
    """Off by default; prepare_moss_runtime copies the two options to the process-wide switches and registers the global
    optimizer-step hook once; with the switch on the trainer invalidates the cache after every optimizer step and after
    sample images - also when sampling raises - and not with it off; validate_moss_args runs the new validation."""
    import inspect

    import moss_train_network as mtn

    assert "validate_moss_ckpt_reuse_args(args, train_dataset_group)" in inspect.getsource(
        moss_train_utils.validate_moss_args)
    parser = mtn.setup_parser()
    parser.format_help()  # the help strings are %-formatted by argparse
    off = parser.parse_args([])
    assert off.moss_ckpt_nograd_merged is False and off.moss_ckpt_reuse_attn is False
    monkeypatch.setattr(attention, "is_flash_attn_supported", lambda: True)
    chains, rebuild = moss_models._CHAINS.enabled, lycoris_fast._REBUILD.enabled
    try:
        mtn.prepare_moss_runtime(off)
        assert anima_models.CKPT_NOGRAD_MERGED is False and anima_models.CKPT_REUSE_ATTN is False
        assert lycoris_fast._CKPT_STEP_HOOK is None
        on = parser.parse_args(["--moss_ckpt_nograd_merged", "--moss_ckpt_reuse_attn", "--attn_mode", "flash"])
        mtn.prepare_moss_runtime(on)
        assert anima_models.CKPT_NOGRAD_MERGED is True and anima_models.CKPT_REUSE_ATTN is True
        assert moss_models.ckpt_nograd_merged_enabled() is True
        handle = lycoris_fast._CKPT_STEP_HOOK
        assert handle is not None
        mtn.prepare_moss_runtime(on)
        assert lycoris_fast._CKPT_STEP_HOOK is handle  # registered once
    finally:
        moss_models._CHAINS.enable(chains)
        lycoris_fast.enable_compiled_rebuild(rebuild)

    class DataWriter(torch.optim.Optimizer):  # writes through p.data like CAME: no version counter moves
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
    _remove_step_hook()

    trainer = mtn.MossNetworkTrainer()
    sampled = []
    monkeypatch.setattr(mtn.moss_train_utils, "sample_images", lambda *a, **k: sampled.append(lycoris_fast._CKPT_PASS))

    def hooks():
        before = lycoris_fast._CKPT_GENERATION
        trainer.on_optimizer_step_end(argparse.Namespace(), None, 1, False)
        after_step = lycoris_fast._CKPT_GENERATION
        trainer.sample_images(None, argparse.Namespace(), 0, 1, "cpu", None, None, None, None)
        return after_step - before, lycoris_fast._CKPT_GENERATION - after_step

    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", False)
    assert hooks() == (0, 0)
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", True)
    assert hooks() == (1, 1) and sampled == [0, 0]

    def failing(*a, **k):
        raise RuntimeError("out of memory while sampling")

    monkeypatch.setattr(mtn.moss_train_utils, "sample_images", failing)
    before = lycoris_fast._CKPT_GENERATION
    with pytest.raises(RuntimeError):
        trainer.sample_images(None, argparse.Namespace(), 0, 1, "cpu", None, None, None, None)
    assert lycoris_fast._CKPT_GENERATION == before + 1

    # the one-time cache size line and the trained-time-conditioning note
    model, net, _ = build()
    accelerator = types.SimpleNamespace(unwrap_model=lambda m: m)
    assert trainer.after_network_loaded(argparse.Namespace(moss_block_compile=False), accelerator, net, [], model) is None
    assert trainer._ckpt_merged_unet is model
    micro_step(model, net, batch())
    trainer.on_epoch_end(argparse.Namespace(), accelerator, 0, 1)
    assert trainer._ckpt_merged_logged is True

# -------------------------------------------------------------------------- review fixes (2026-11-15): tests per finding


def test_env_teardown_leaves_no_lycoris_class_patch():
    """The `_env` fixture applies lycoris_algo_fixes' class patches (LokrModule.forward becomes its _lokr_forward) and
    must take them off again: tests/test_moss_pipeline.py::test_fast_lokr_forward_is_bit_identical_to_lycoris, which
    sorts after this module, expects LyCORIS's classes with only the fast LoKr forward installed and failed when this
    module ran before it in one process."""
    lokr_class = lycoris_fast._lokr_class()
    assert laf.patched_targets() and lokr_class.forward is not lycoris_fast._fast_forward  # what the fixture set up
    _teardown_env()
    assert laf.patched_targets() == []
    assert lokr_class.forward is lycoris_fast._fast_forward


PADDED = (dict(text=(5, 9)), dict(text=(4, 7), seed=2), dict(text=(9, 4, 6), hw=(4, 6), seed=3), dict(text=(3, 8), seed=4))


def test_padded_batches_under_block_compile_add_no_graphs(monkeypatch):
    """Block compile + batch > 1: a padded batch's masked attention params carry per-length ints, so every padded length
    compiles its own graphs - one per kind of block (adapted / left out of the network), with the options as without
    them. The no-grad node would compile two per kind and length (first pass and recompute are different graphs), using
    up --moss_block_compile_recompile_limit twice as fast for a call that has nothing to reuse: such a call keeps
    torch.utils.checkpoint. One-sample micro-steps afterwards still get the node (and its two graphs per kind, once)."""
    nodes = []
    real_node = anima_models.nograd_checkpoint
    monkeypatch.setattr(anima_models, "nograd_checkpoint", lambda *a: nodes.append(1) or real_node(*a))
    one_sample = (dict(text=(6,)), dict(text=(10,), hw=(4, 6), seed=2))
    counts, results, node_counts = {}, {}, {}
    for name, flags in (("off", (False, False)), ("nograd", (True, False)), ("reuse", (True, True))):
        counter = Counter()
        model, net, state = build(compiled=True, backend=counter)
        set_flags(monkeypatch, *flags)
        nodes.clear()
        counts[name], results[name] = [], []
        for spec in PADDED + one_sample:
            results[name].append(micro_step(model, net, batch(**spec)))
            counts[name].append(counter.graphs)
            if spec is PADDED[-1]:
                node_counts[name] = len(nodes)
                nodes.clear()
        assert state.eager_calls == 0
        assert len(nodes) == (len(one_sample) * len(model.layers) if flags[0] else 0)
    assert node_counts == {"off": 0, "nograd": 0, "reuse": 0}  # no padded call went through the no-grad node
    padded = len(PADDED)
    assert counts["off"][:padded] == [2, 4, 6, 8], counts["off"]
    assert counts["nograd"][:padded] == counts["reuse"][:padded] == counts["off"][:padded], counts
    # the first one-sample micro-step: 2 graphs without the options, first pass + recompute (4) with them; then flat
    assert [c - counts["off"][padded - 1] for c in counts["off"][padded:]] == [2, 2]
    for name in ("nograd", "reuse"):
        assert [c - counts[name][padded - 1] for c in counts[name][padded:]] == [4, 4], counts[name]
        for i, (a, b) in enumerate(zip(results["off"], results[name])):
            assert_same(a, b, f"padded under block compile, {name}, micro-step {i}")


def test_padded_batches_keep_the_no_grad_node_without_block_compile(monkeypatch):
    """Eager blocks have no per-length graphs: a padded batch keeps the no-grad first pass (nothing is stored)."""
    nodes = []
    real_node = anima_models.nograd_checkpoint
    monkeypatch.setattr(anima_models, "nograd_checkpoint", lambda *a: nodes.append(1) or real_node(*a))
    model, net, _ = build()
    set_flags(monkeypatch, True, True)
    micro_step(model, net, batch(**PADDED[0]))
    assert len(nodes) == len(model.layers) and lycoris_fast.ckpt_merged_bytes(model) > 0


def _dataset(batch_size, *resos, jitter=None):
    image_data = {f"image{i}": types.SimpleNamespace(bucket_reso=reso, jitter_bucket_info=None)
                  for i, reso in enumerate(resos)}
    if jitter:
        image_data["image0"].jitter_bucket_info = {1536: (jitter, jitter)}
    return types.SimpleNamespace(batch_size=batch_size, image_data=image_data)


def test_dataset_group_limits_read_the_dataset_toml_values():
    """The batch size and the resolution tiers of a dataset TOML are per dataset (the user's config: four datasets with
    batch_size 1 and max_bucket_reso 4096 in the args): the largest batch size and the largest bucket an image got."""
    limits = moss_train_utils.dataset_group_limits
    group = types.SimpleNamespace(datasets=[_dataset(1, (1024, 1024), (832, 1216)), _dataset(2, (2048, 2048), (1536, 2816)),
                                            _dataset(1, (1536, 1536))])
    assert limits(group) == (2, 96 * 176, (1536, 2816))
    assert limits(types.SimpleNamespace(datasets=[_dataset(1, (512, 512), jitter=(768, 1024))])) == (1, 48 * 64, (768, 1024))
    assert limits(None) == limits(object()) == limits(types.SimpleNamespace(datasets=[object()])) == (1, 0, None)
    odd = types.SimpleNamespace(datasets=[_dataset(2, (1024, 1024)), types.SimpleNamespace(batch_size=3, image_data=[1]),
                                          types.SimpleNamespace(batch_size="x")])
    assert limits(odd) == (3, 64 * 64, (1024, 1024))  # a layout it cannot read never raises: what was read so far
    args = types.SimpleNamespace(train_batch_size=1)
    assert moss_train_utils.largest_batch_size(args, limits(group)) == 2
    assert moss_train_utils.largest_batch_size(types.SimpleNamespace(train_batch_size=4), limits(group)) == 4
    assert moss_train_utils.largest_batch_size(args) == 1


def test_batch_warning_covers_either_option_and_dataset_batch_sizes(monkeypatch, warned):
    """The batch > 1 warning: also for --moss_ckpt_nograd_merged alone, also when only a dataset of the dataset TOML has
    batch_size > 1 (args.train_batch_size stays 1), and it says what a padded batch does under block compile."""
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", True)
    validate = moss_train_utils.validate_moss_ckpt_reuse_args
    group = types.SimpleNamespace(datasets=[_dataset(1, (1024, 1024)), _dataset(3, (1024, 1024))])
    validate(_args(), types.SimpleNamespace(datasets=[_dataset(1, (1024, 1024))]))
    validate(_args(), None)
    assert warned == []
    validate(_args(), group)
    assert len(warned) == 1 and "largest batch size 3" in warned[0] and "keeps the plain gradient checkpoint" in warned[0]
    warned.clear()
    validate(_args(moss_ckpt_reuse_attn=False, attn_mode="torch", train_batch_size=4))
    assert len(warned) == 1 and "--moss_ckpt_nograd_merged with batch > 1 (largest batch size 4)" in warned[0]
    import inspect

    assert "validate_moss_ckpt_reuse_args(args, train_dataset_group)" in inspect.getsource(moss_train_utils.validate_moss_args)


def test_block_compile_batch_warning_sees_a_dataset_batch_size(monkeypatch):
    """after_network_loaded's --moss_block_compile batch > 1 warning read args.train_batch_size only; the trainer now
    remembers the datasets' own batch sizes (assert_extra_args)."""
    import moss_train_network as mtn

    warnings = []
    monkeypatch.setattr(mtn.logger, "warning", lambda message, *a, **k: warnings.append(str(message)))
    monkeypatch.setattr(mtn.moss_block_compile, "compile_moss_blocks", lambda model, **kw: "state")
    monkeypatch.setattr(mtn.moss_train_utils, "validate_moss_args", lambda *a, **k: None)
    args = argparse.Namespace(moss_block_compile=True, train_batch_size=1, moss_block_compile_backend="aot_eager",
                              moss_block_compile_mode=None, moss_block_compile_recompile_limit=64, moss_stream_fp32=False,
                              moss_text_encoder=None)
    accelerator = types.SimpleNamespace(unwrap_model=lambda m: m, print=lambda *a, **k: None)
    model, net, _ = build()
    trainer = mtn.MossNetworkTrainer()
    assert trainer.after_network_loaded(args, accelerator, net, [], model) == "state" and warnings == []
    group = types.SimpleNamespace(datasets=[_dataset(1, (1024, 1024)), _dataset(2, (2048, 2048))])
    with pytest.raises(ValueError, match="moss_text_encoder"):  # the next check of assert_extra_args; the limits are set
        trainer.assert_extra_args(args, group, None)
    assert trainer._dataset_limits == (2, 128 * 128, (2048, 2048))
    trainer.after_network_loaded(args, accelerator, net, [], model)
    assert len(warnings) == 1 and "--moss_block_compile with batch > 1" in warnings[0]


SAMPLE_PLAN = tuple(dict(text=(n,), hw=hw) for n, hw in ((6, (4, 4)), (11, (4, 6)), (4, (6, 6)), (9, (4, 4)), (13, (4, 6)),
                                                         (5, (4, 4)), (7, (6, 6)), (12, (4, 4)), (10, (4, 6))))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
def test_sample_round_releases_the_cache_and_training_rebuilds_it(compiled, came_cls, monkeypatch, tmp_path):
    """The merged-weight cache (~8.65 GiB on the real model) is not held while sample images are made:
    moss_train_utils.sample_images, driven through MossNetworkTrainer.sample_images, drops it once a round really runs
    (after an optimizer step and in the middle of an accumulation here) and the next training first pass builds it
    again. A call that makes no samples (the trainer calls it after every step) keeps the cached tensors. Outputs,
    samples and trained parameters equal the flags-off run; compiled, the new cache tensors need no new graph."""
    opt = lambda params: make_came(came_cls, params)  # noqa: E731
    g = torch.Generator().manual_seed(5)
    cond = (torch.randn(1, 7, TINY["text_dim"], generator=g), torch.ones(1, 7, dtype=torch.bool))
    neg = (torch.randn(1, 4, TINY["text_dim"], generator=g), torch.ones(1, 4, dtype=torch.bool))
    accelerator = types.SimpleNamespace(is_main_process=True, unwrap_model=lambda m: m, device=torch.device("cpu"),
                                        autocast=lambda: torch.autocast("cpu", dtype=torch.bfloat16),
                                        wait_for_everyone=lambda: None)
    sampling = argparse.Namespace(sample_at_first=False, sample_every_n_steps=1, sample_every_n_epochs=None,
                                  sample_prompts="unused", output_dir=str(tmp_path), moss_stock_lycoris=False,
                                  moss_sample_uncond_skip_layers=str(EXCLUDED), moss_no_sample_cache=False)
    not_due = argparse.Namespace(**{**vars(sampling), "sample_every_n_steps": None})
    record = {}

    def generate(accelerator_, args, dit, prompt_dict, te_outputs, skip):
        record["during"].append(lycoris_fast.ckpt_merged_bytes(dit))
        dit.eval()  # train_network.py switches the DiT to eval around sampling
        try:
            latents = moss_train_utils.moss_sample(dit, cond, neg, 64, 64, 3, 3.0, 5.0, 3, torch.device("cpu"),
                                                   skip_layers=skip)
        finally:
            dit.train()
        record["samples"].append(latents.clone())
        return latents

    monkeypatch.setattr(moss_train_utils.train_util, "load_prompts", lambda path: [dict(prompt="a")])
    monkeypatch.setattr(moss_train_utils, "_generate_one", generate)
    monkeypatch.setattr(moss_train_utils, "_decode_and_save", lambda *a, **k: None)

    def between(i, model, net, trainer):
        record["graphs"].append(record["counter"].graphs)
        if i in (2, 4):  # right after the first optimizer step; in the middle of the second accumulation
            trainer.sample_prompts_te_outputs = {"a": cond, "": neg}
            record["before"].append(lycoris_fast.ckpt_merged_bytes(model))
            trainer.sample_images(accelerator, sampling, None, i + 1, "cpu", None, None, None, model)
            record["after"].append(lycoris_fast.ckpt_merged_bytes(model))
        if i == 6:  # no round is due: nothing is released
            held = [lokr.__dict__.get("_ckpt_merged") for lokr in cached_lokrs(model)]
            trainer.sample_images(accelerator, not_due, None, i + 1, "cpu", None, None, None, model)
            assert all(a is b for a, b in zip(held, (lokr.__dict__.get("_ckpt_merged") for lokr in cached_lokrs(model))))
            record["kept"].append(lycoris_fast.ckpt_merged_bytes(model))

    runs = {}
    for name, flags in (("off", (False, False)), ("on", (True, True))):
        record = {key: [] for key in ("during", "samples", "graphs", "before", "after", "kept")}
        record["counter"] = Counter()
        run = simulate(monkeypatch, flags, make_opt=opt, compiled=compiled, plan=SAMPLE_PLAN, between=between,
                       backend=record["counter"])
        runs[name] = (run, record)
    (ref, off), (new, on) = runs["off"], runs["on"]
    assert_same_run(ref, new, f"sample rounds that release the cache, compiled={compiled}")
    assert len(off["samples"]) == len(on["samples"]) == 2
    assert all(torch.equal(a, b) for a, b in zip(off["samples"], on["samples"])), "samples differ"
    assert off["before"] == off["during"] == off["after"] == [0, 0] and off["kept"] == [0]  # nothing is ever cached
    full = sum(lokr.org_module[0].weight.numel() for lokr in cached_lokrs(new[2]["model"])) * 2
    assert full > 0 and on["before"] == [full, full] and on["kept"] == [full]  # rebuilt by the step after each round
    assert on["during"] == [0, 0] and on["after"] == [0, 0]
    assert set(new[2]["diffs"]) == {0.0}
    if compiled:
        assert on["graphs"] == [4] * len(SAMPLE_PLAN), on["graphs"]  # the rebuilt cache tensors are plain graph inputs
        assert new[2]["state"].eager_calls > 0  # the sample rounds ran the eager blocks


def test_vram_estimate_counts_what_the_cache_and_the_stash_really_hold(monkeypatch):
    """ckpt_reuse_vram_estimate, made before the first forward from the modules alone, equals what a micro-step then
    holds: the bytes of the merged-weight cache, and the bytes of every joint block's stored attention output +
    softmax_lse (all alive when backward starts) for one sample and for an equal-length batch. With the real model's
    dimensions: 6,240 bytes per token and block, ~4.0 GiB over 42 blocks at a 2048^2 image."""
    stored = []
    real = attention._flash_forward_with_lse

    def spy(q, k, v):
        out, lse = real(q, k, v)
        stored.append(out.numel() * out.element_size() + lse.numel() * lse.element_size())
        return out, lse

    monkeypatch.setattr(attention, "_flash_forward_with_lse", spy)
    estimate = moss_train_utils.ckpt_reuse_vram_estimate
    for rows, text, hw in ((1, 11, (4, 6)), (2, 7, (4, 4))):
        model, net, _ = build()
        set_flags(monkeypatch, True, True)
        est = estimate(model, True, hw[0] * hw[1], batch_size=rows)
        assert all("_ckpt_lokrs" not in block.__dict__ for block in model.layers)  # the estimate leaves no state behind
        stored.clear()
        micro_step(model, net, batch(hw=hw, text=(text,) * rows))
        assert est["cache"] == lycoris_fast.ckpt_merged_bytes(model) > 0
        assert len(stored) == len(model.layers) and sum(stored) == est["stash"] + text * est["stash_per_text_token"]
        assert est["inputs"] == len(model.layers) * hw[0] * hw[1] * rows * TINY["dim"] * 2
        no_reuse = estimate(model, False, hw[0] * hw[1], batch_size=rows, cached=False, stream_bytes=4)
        assert (no_reuse["cache"], no_reuse["stash"], no_reuse["stash_per_text_token"]) == (0, 0, 0)
        assert no_reuse["inputs"] == 2 * est["inputs"]
    real_size = types.SimpleNamespace(layers=[nn.Identity() for _ in range(moss_models.MOSS_DIT_CONFIG["num_layers"])],
                                      config=moss_models.MOSS_DIT_CONFIG)
    est = estimate(real_size, True, 128 * 128)
    assert est["stash_per_text_token"] == 42 * 6240 and est["stash"] == 42 * 6240 * 16384 and est["cache"] == 0
    assert 3.9 < est["stash"] / 2**30 < 4.1 and est["inputs"] == 42 * 16384 * 3072 * 2


def test_vram_notes_warn_when_the_card_is_tight_or_at_its_limit():
    """The start-up line: a warning when weights + kept block inputs + the options' extra pass 80 % of the card (the
    user's 32 GB card with the 2048 tier: 8.65 GiB cache + ~4 GiB stash on ~9.6 GiB of weights), an info line at 1024.
    The peak line: a warning once the peak reserved memory is at the card's size or the device has nothing free."""
    gib = 2**30
    card = dict(total=int(31.84 * gib), allocated=int(9.6 * gib))
    per_token = 42 * 6240

    def est(tokens):
        return dict(cache=int(8.65 * gib), stash=per_token * tokens, stash_per_text_token=per_token,
                    inputs=42 * tokens * 3072 * 2)

    line, tight = moss_train_utils.ckpt_reuse_vram_note(est(16384), (2048, 2048), card)
    assert tight and "8.65 GiB merged-weight cache" in line and "4.00 GiB stored joint attention" in line
    assert "the largest bucket 2048x2048" in line and "12.65 GiB on top" in line and "That is tight" in line
    assert "0.24 GiB per 1000 caption tokens" in line and "31.84 GiB, 9.60 GiB allocated now" in line
    line, tight = moss_train_utils.ckpt_reuse_vram_note(est(4096), (1024, 1024), card)
    assert not tight and "That is tight" not in line and "This card" in line
    line, tight = moss_train_utils.ckpt_reuse_vram_note(est(16384), (2048, 2048), None)  # not a CUDA device: no verdict
    assert not tight and "This card" not in line and "12.65 GiB on top" in line
    line, tight = moss_train_utils.ckpt_reuse_vram_note(est(0), None, None)
    assert "unknown largest bucket" in line

    peak = moss_train_utils.ckpt_reuse_peak_note
    total = card["total"]
    line, at_limit = peak(dict(total=total, peak_allocated=20 * gib, peak_reserved=24 * gib, free=6 * gib))
    assert not at_limit and "20.00 GiB allocated, 24.00 GiB reserved of 31.84 GiB (6.00 GiB free" in line
    line, at_limit = peak(dict(total=total, peak_allocated=29 * gib, peak_reserved=int(0.95 * total)))
    assert at_limit and "at the card's limit" in line and "free on the device" not in line
    assert peak(dict(total=total, peak_allocated=26 * gib, peak_reserved=28 * gib, free=int(0.01 * total)))[1]
    assert moss_train_utils.cuda_memory_stats(torch.device("cpu")) is None
    assert moss_train_utils.cuda_memory_stats(None) is None


def test_trainer_logs_the_vram_estimate_and_watches_the_peak(monkeypatch):
    """With the switch on: after_network_loaded logs the estimate for the datasets' largest bucket (a warning when it is
    tight for the card), every optimizer step checks the peak (one warning when the card is at its limit), every epoch
    end reports the measured peak when it grew. With the switch off nothing reads a memory counter."""
    import moss_train_network as mtn

    lines = {"info": [], "warning": []}
    monkeypatch.setattr(mtn.logger, "info", lambda message, *a, **k: lines["info"].append(str(message)))
    monkeypatch.setattr(mtn.logger, "warning", lambda message, *a, **k: lines["warning"].append(str(message)))
    gib = 2**30
    stats = dict(total=32 * gib, allocated=1 * gib, peak_allocated=2 * gib, peak_reserved=3 * gib, free=20 * gib)
    reads = []
    real_stats = moss_train_utils.cuda_memory_stats

    def fake_stats(device, with_free=True):
        reads.append(with_free)
        return {k: v for k, v in stats.items() if with_free or k != "free"}

    monkeypatch.setattr(mtn.moss_train_utils, "cuda_memory_stats", fake_stats)
    model, net, _ = build()
    accelerator = types.SimpleNamespace(unwrap_model=lambda m: m, device=torch.device("cpu"))
    args = argparse.Namespace(moss_block_compile=False, moss_ckpt_reuse_attn=True, mixed_precision="bf16",
                              moss_stream_fp32=False)

    def drive(trainer):
        trainer._dataset_limits = (1, 16, (64, 64))
        trainer.after_network_loaded(args, accelerator, net, [], model)
        trainer.on_optimizer_step_end(argparse.Namespace(), None, 1, False)
        trainer.on_epoch_end(argparse.Namespace(), accelerator, 0, 1)

    drive(mtn.MossNetworkTrainer())  # both switches off
    assert reads == [] and lines == {"info": [], "warning": []}

    set_flags(monkeypatch, True, True)
    trainer = mtn.MossNetworkTrainer()
    drive(trainer)
    estimate = [line for line in lines["info"] if "estimated extra VRAM" in line]
    assert len(estimate) == 1 and "the largest bucket 64x64" in estimate[0] and "This card: 32.00 GiB" in estimate[0]
    # start-up, the step, the epoch end: each reads the device's free memory too (what other processes hold counts)
    assert lines["warning"] == [] and reads == [True, True, True]
    peaks = [line for line in lines["info"] if "VRAM peak so far" in line]
    assert len(peaks) == 1 and "3.00 GiB reserved of 32.00 GiB (20.00 GiB free" in peaks[0]
    trainer.on_epoch_end(argparse.Namespace(), accelerator, 1, 2)  # the peak did not grow: no second line
    assert len([line for line in lines["info"] if "VRAM peak so far" in line]) == 1
    stats["peak_reserved"] = 5 * gib
    trainer.on_epoch_end(argparse.Namespace(), accelerator, 2, 3)
    assert len([line for line in lines["info"] if "VRAM peak so far" in line]) == 2
    stats["peak_reserved"] = 31 * gib  # at the card's limit: one warning, at the optimizer step that saw it
    trainer.on_optimizer_step_end(argparse.Namespace(), None, 4, False)
    trainer.on_optimizer_step_end(argparse.Namespace(), None, 5, False)
    assert len(lines["warning"]) == 1 and "at the card's limit" in lines["warning"][0]
    before = len(reads)
    trainer.on_optimizer_step_end(argparse.Namespace(), None, 6, False)
    assert len(reads) == before  # warned: the per-step check is over

    stats.update(allocated=30 * gib, peak_reserved=3 * gib)  # weights already fill the card: the estimate is a warning
    lines["warning"].clear()
    tight = mtn.MossNetworkTrainer()
    tight._dataset_limits = (1, 16, (64, 64))
    tight.after_network_loaded(args, accelerator, net, [], model)
    assert len(lines["warning"]) == 1 and "That is tight" in lines["warning"][0]

    # no CUDA device (the real cuda_memory_stats answers None): the estimate without a verdict, nothing watched
    monkeypatch.setattr(mtn.moss_train_utils, "cuda_memory_stats", real_stats)
    lines["info"].clear()
    cpu = mtn.MossNetworkTrainer()
    drive(cpu)
    assert cpu._ckpt_vram_device is None and not any("VRAM peak" in line for line in lines["info"])
    assert any("estimated extra VRAM" in line and "This card" not in line for line in lines["info"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
