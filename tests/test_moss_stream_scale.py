"""CPU tests for --moss_stream_scale C (library/moss/models.py, 2026-11-15; default 1.0 = off).

The joint residual stream of the MossBlocks is carried multiplied by C (a power of two in (0, 1]): the joint input (video
embedder + text refiner outputs) and the two gates of the shared modulation are multiplied by C in
MossTransformer.forward, and the RMSNorms that read the stream (norm1 / norm2 of every block, final_norm_video) use
eps = 1e-6 * C**2. RMSNorm(C x, C**2 eps) == RMSNorm(x, eps), and every multiplication by a power of two is exact, so:

* the output, every LoKr / DoRA gradient (also of the text refiner and of a trained time conditioning) and the sampled
  latents are bit-identical with and without the option - fp32, fp64, bf16, bf16 autocast, --moss_stream_fp32; eager,
  per-block compiled (aot_eager), every gradient checkpoint (plain, CPU offload, unsloth), --moss_ckpt_nograd_merged /
  --moss_ckpt_reuse_attn (also reuse-only with a budget), one sample / changing lengths / padded batches, CFG with the
  negative branch's skipped block;
* every block's input and output are exactly C times the unscaled ones (so the stream's largest value is), and an fp16
  forward whose stream passes 65504 is inf / nan without the option and finite with it;
* the compiled block entry needs no more graphs than without the option, a compiled elementwise chain takes the float eps
  without a graph break and keeps the old graph for eps None;
* C = 1.0 is the previous forward: nothing multiplied, every eps None, the chain graph unchanged;
* the glue: stream_scale_problem / set_stream_scale / validate_moss_stream_scale_args refuse what is not a power of two
  in (0, 1], the parser and a config TOML take the key, load_unet_lazily sets it on the DiT, block compile gets its own
  inductor cache signature.

The toy model, the LoKr+DoRA network and the fake flash kernels are tests/test_moss_ckpt_reuse.py's (imported under the
name pytest gives that module, like tests/test_moss_ckpt_reuse_only.py does).

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_stream_scale.py -q -p no:cacheprovider
(an EMPTY CUDA_VISIBLE_DEVICES does not hide the GPU on this Windows build: use -1)
"""
import math
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
import torch._dynamo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the name pytest imports the sibling test module under
import test_moss_ckpt_reuse as base  # noqa: E402
from test_moss_ckpt_reuse import _env  # noqa: E402,F401 - the autouse fixture

from library import anima_block_compile, anima_models, train_util  # noqa: E402
from library.moss import block_compile as mbc  # noqa: E402
from library.moss import models as moss_models  # noqa: E402
from library.moss import train_utils as moss_train_utils  # noqa: E402
from library.moss.models import MossTransformer, norm_modulate, rms_norm, stream_scale_problem  # noqa: E402

C = 0.0625  # the recommended 1/16
SCALES = (0.5, C, 2.0 ** -10)
PLAIN = dict(dim=192, ffn_dim=512, num_heads=4, num_layers=4, text_dim=64, freq_dim=32, latent_channels=128)


# ------------------------------------------------------------------------------------------------------------- helpers


def plain_model(dtype=torch.float32, scale=1.0, seed=0, gate_bias=0.0):
    """A MossTransformer without a network (attn_mode torch), eval mode; gate_bias is added to gate_a / gate_f."""
    torch.manual_seed(seed)
    model = MossTransformer(**PLAIN).eval()
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.05)
        if gate_bias:
            bias = model.shared_block_modulation.bias.view(6, -1)
            bias[2] += gate_bias
            bias[5] += gate_bias
    model = model.to(dtype)
    if scale != 1.0:
        model.set_stream_scale(scale)
    return model


def plain_inputs(dtype, text=(6,), hw=(6, 6), seed=1):
    g = torch.Generator().manual_seed(seed)
    rows, longest = len(text), max(text)
    x = torch.randn(rows, 128, *hw, generator=g).to(dtype)
    emb = torch.randn(rows, longest, PLAIN["text_dim"], generator=g).to(dtype)
    mask = torch.arange(longest)[None] < torch.tensor(text)[:, None]
    return x, torch.full((rows,), 500.0), emb, mask


def plain_forward(model, data, autocast=None, skip_layers=None):
    """Output, and the input and output of every joint block (forward hooks)."""
    seen = {"in": [], "out": []}
    hooks = []
    for layer in model.layers:
        hooks.append(layer.register_forward_pre_hook(lambda _m, args: seen["in"].append(args[0].detach().clone())))
        hooks.append(layer.register_forward_hook(lambda _m, _a, out: seen["out"].append(out.detach().clone())))
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=autocast or torch.bfloat16, enabled=autocast is not None):
            out = model(*data, skip_layers=skip_layers)
    finally:
        for hook in hooks:
            hook.remove()
    return out, seen


def set_flags(monkeypatch, nograd: bool, reuse: bool, budget=None):
    """The process-wide checkpoint switches as set_ckpt_reuse leaves them (budget: bytes, None = every block)."""
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", nograd)
    monkeypatch.setattr(anima_models, "CKPT_REUSE_ATTN", reuse)
    monkeypatch.setattr(anima_models, "CKPT_RECOMPUTE_DIFFS", [] if (nograd or reuse) else None)
    monkeypatch.setattr(moss_models, "CKPT_REUSE_BUDGET_BYTES", budget)


def run(monkeypatch, scale, batches, flags=(False, False), budget=None, extra_time=False, ckpt=None, **kw):
    """A fresh toy model + LoKr+DoRA network (tests/test_moss_ckpt_reuse.build) under `scale`, one micro-step per batch
    spec. ckpt: None (the plain checkpoint build() sets), "off", "cpu" or "unsloth". extra_time: the time MLP and the
    shared modulation trained too (their gradients are compared). Returns ([(output, grads)], model, net, state)."""
    model, net, state = base.build(**kw)
    if scale != 1.0:
        model.set_stream_scale(scale)
    if ckpt == "off":
        model.disable_gradient_checkpointing()
    elif ckpt == "cpu":
        model.enable_gradient_checkpointing(cpu_offload=True)
    elif ckpt == "unsloth":
        model.enable_gradient_checkpointing(unsloth_offload=True)
    extra = ()
    if extra_time:
        extra = (model.shared_block_modulation.weight, model.shared_block_modulation.bias, model.time_embedding[0].weight)
        for p in extra:
            p.requires_grad_(True)
    set_flags(monkeypatch, *flags, budget=budget)
    stream = kw.get("stream", False)
    results = [base.micro_step(model, net, base.batch(stream=stream, **b), extra=extra) for b in batches]
    return results, model, net, state


def assert_runs_same(ref, new, what):
    assert len(ref) == len(new)
    for i, (a, b) in enumerate(zip(ref, new)):
        base.assert_same(a, b, f"{what}, micro-step {i}")


# ------------------------------------------------------------------------------------- (1) the forward, bit for bit


@pytest.mark.parametrize("dtype, autocast", [(torch.float32, None), (torch.float64, None), (torch.bfloat16, None),
                                             (torch.bfloat16, torch.bfloat16), (torch.float32, torch.bfloat16)],
                         ids=["fp32", "fp64", "bf16", "bf16_autocast", "fp32_bf16_autocast"])
@pytest.mark.parametrize("scale", SCALES)
def test_forward_is_bit_identical(dtype, autocast, scale):
    """One sample, a padded batch (different text lengths) and the negative branch's skipped block: the same output."""
    for data, skip in ((plain_inputs(dtype), None), (plain_inputs(dtype, text=(5, 9, 3), hw=(4, 6), seed=2), None),
                       (plain_inputs(dtype, text=(4,), seed=3), (2,))):
        off, _ = plain_forward(plain_model(dtype), data, autocast, skip)
        on, _ = plain_forward(plain_model(dtype, scale), data, autocast, skip)
        assert off.dtype == on.dtype and torch.isfinite(off).all()
        assert torch.equal(off, on), f"max |d| {(off.double() - on.double()).abs().max().item():.3e}"


def test_stream_fp32_forward_is_bit_identical():
    for scale in SCALES:
        outs = []
        for s in (1.0, scale):
            model = plain_model(torch.bfloat16, s)
            model.enable_stream_fp32()
            outs.append(plain_forward(model, plain_inputs(torch.float32), torch.bfloat16)[0])
        assert outs[0].dtype == torch.float32 and torch.equal(*outs)


# ------------------------------------------------------------------------- (3) the stream is exactly C times smaller


@pytest.mark.parametrize("dtype, autocast", [(torch.float32, None), (torch.bfloat16, torch.bfloat16)], ids=["fp32", "bf16"])
@pytest.mark.parametrize("scale", SCALES)
def test_every_block_input_and_output_is_exactly_c_times(dtype, autocast, scale):
    data = plain_inputs(dtype, text=(5, 9), hw=(4, 6))
    _, off = plain_forward(plain_model(dtype), data, autocast)
    _, on = plain_forward(plain_model(dtype, scale), data, autocast)
    for kind in ("in", "out"):
        assert len(off[kind]) == len(on[kind]) == PLAIN["num_layers"]
        for i, (a, b) in enumerate(zip(off[kind], on[kind])):
            assert torch.equal(b, a * scale), f"block {i} {kind}put is not exactly C times the unscaled one"
            assert b.abs().max().item() == a.abs().max().item() * scale  # the stream's largest value, too


def test_fp16_stream_overflows_without_and_stays_finite_with_the_option():
    """Gates of ~30000 (fp16 still holds them) push the toy model's joint stream to ~9.6e4 in fp32: past fp16's 65504.
    A plain fp16 forward turns it into inf / nan; with C = 1/16 the stream peaks near 6e3 and the output matches the fp32
    forward as closely as an fp16 forward whose stream fits (the same model with 1/2 the gates: ~2e-3)."""
    data32 = plain_inputs(torch.float32)
    truth, stream32 = plain_forward(plain_model(torch.float32, gate_bias=30000.0), data32)
    peak = max(t.abs().max().item() for t in stream32["out"])
    assert 65504 < peak < 65504 / C, peak
    x, t, emb, mask = data32
    data16 = (x.half(), t, emb.half(), mask)  # the timesteps stay fp32, as the trainer passes them
    off, _ = plain_forward(plain_model(torch.float16, gate_bias=30000.0), data16)
    on, stream16 = plain_forward(plain_model(torch.float16, C, gate_bias=30000.0), data16)
    assert not torch.isfinite(off).all()
    assert torch.isfinite(on).all()
    assert max(t.float().abs().max().item() for t in stream16["out"]) < 65504 * C * 2  # ~peak * C, far below fp16's max
    error = float((on.float() - truth).norm() / truth.norm())
    assert error < 1e-2, error


# ---------------------------------------------------------------------- (2) LoKr + DoRA gradients and the checkpoints


@pytest.mark.parametrize("stream", [False, True], ids=["bf16_stream", "fp32_stream"])
@pytest.mark.parametrize("kind", ["text_lengths_and_sizes", "padded"])
def test_lokr_dora_gradients_are_bit_identical(kind, stream, monkeypatch):
    """The production network (LoKr factor 4 full matrix + DoRA on the output, one block left out) under the plain
    gradient checkpoint, with the time MLP and the shared modulation trained too (the gates' rows of its gradient pass
    through the C multiplication): output and every gradient equal, over consecutive micro-steps."""
    batches = base.BATCHES[kind]
    ref, *_ = run(monkeypatch, 1.0, batches, stream=stream, extra_time=True)
    new, *_ = run(monkeypatch, C, batches, stream=stream, extra_time=True)
    assert all(bool(g.any()) for _, grads in new for g in grads.values())
    assert_runs_same(ref, new, f"{kind}, stream_fp32={stream}")


def test_text_refiner_preset_gradients_are_bit_identical(monkeypatch):
    """preset moss-text: the text refiner's own stream is not scaled, its output enters the joint stream times C."""
    batches = base.BATCHES["text_lengths_and_sizes"]
    ref, _, net, _ = run(monkeypatch, 1.0, batches, preset="moss-text")
    new, *_ = run(monkeypatch, C, batches, preset="moss-text")
    assert any("text_embedder" in name for name in ref[0][1])
    assert_runs_same(ref, new, "preset moss-text")


@pytest.mark.parametrize("ckpt", ["off", "cpu", "unsloth"])
def test_other_checkpoint_modes_are_bit_identical(ckpt, monkeypatch):
    batches = base.BATCHES["text_lengths_and_sizes"][:2]
    ref, *_ = run(monkeypatch, 1.0, batches, ckpt=ckpt)
    new, *_ = run(monkeypatch, C, batches, ckpt=ckpt)
    assert_runs_same(ref, new, f"checkpoint {ckpt}")


def test_sampling_with_cfg_and_the_skipped_block_is_bit_identical(monkeypatch):
    """moss_sample as sample_images runs it (eval, no_grad, autocast, the per-round merged LoKr cache, CFG with the
    negative branch skipping the block left out of the network), between training micro-steps."""
    latents = []
    for scale in (1.0, C):
        _, model, net, _ = run(monkeypatch, scale, base.BATCHES["one_sample"][:1])
        latents.append(base.sample(model))
    assert torch.isfinite(latents[0]).all() and torch.equal(*latents)


# -------------------------------------------------------------------------------------- (4) per-block torch.compile


@pytest.mark.parametrize("stream", [False, True], ids=["bf16_stream", "fp32_stream"])
def test_block_compile_is_bit_identical_and_adds_no_graph(stream, monkeypatch):
    """--moss_block_compile (aot_eager): the scaled run equals the unscaled compiled run bit for bit, needs exactly as
    many graphs over changing caption lengths and image sizes (the eps is one value for all blocks and the whole run),
    never falls back to the eager blocks, and stays as close to the eager run as the unscaled one is."""
    batches = base.BATCHES["text_lengths_and_sizes"] + (dict(text=(8,), seed=4),)
    results, graphs = {}, {}
    for scale in (1.0, C):
        torch._dynamo.reset()
        counter = base.Counter()
        res, model, _, state = run(monkeypatch, scale, batches, compiled=True, stream=stream, backend=counter)
        assert state.eager_calls == 0 and state.compiled_calls == 2 * len(batches) * len(model.layers)
        results[scale], graphs[scale] = res, counter.graphs
    assert graphs[C] == graphs[1.0], graphs
    assert_runs_same(results[1.0], results[C], f"block compile, stream_fp32={stream}")


def test_block_compile_graph_reads_the_eps_from_the_block(monkeypatch):
    """The traced graph of a scaled block takes its norm eps (1e-6 * C**2) from the block: the same scale reuses the
    graphs, and setting the scale back to 1.0 compiles the plain run's graphs again (the guard on the eps) - no more
    than there were."""
    torch._dynamo.reset()
    counter = base.Counter()
    _, model, net, _ = run(monkeypatch, C, base.BATCHES["one_sample"][:1], compiled=True, backend=counter)
    assert all(layer.norm1.eps == layer.norm2.eps == 1e-6 * C * C for layer in model.layers)
    assert model.final_norm_video.eps == 1e-6 * C * C
    first = counter.graphs
    base.micro_step(model, net, base.batch(seed=5))
    assert counter.graphs == first  # same scale, same graphs
    model.set_stream_scale(1.0)
    assert all(layer.norm1.eps is None and layer.norm2.eps is None for layer in model.layers)
    base.micro_step(model, net, base.batch(seed=5))
    assert first < counter.graphs <= 2 * first


def test_compiled_chain_takes_the_float_eps_without_a_graph_break():
    """The elementwise chains (CUDA runs compile them with dynamic=True, fullgraph=True; here aot_eager on CPU): a float
    eps traces without a graph break (fullgraph would raise), equals eager, and one graph serves every shape at one eps."""
    counter = base.Counter()
    compiled = torch.compile(norm_modulate, dynamic=True, fullgraph=True, backend=counter)
    eps = 1e-6 * C * C
    g = torch.Generator().manual_seed(0)
    for tokens in (7, 11, 13):
        x = torch.randn(1, tokens, 32, generator=g).to(torch.bfloat16) * C
        w, scale, shift = (torch.randn(1, 32, generator=g).to(torch.bfloat16)[0] for _ in range(3))
        scale, shift = scale[None], shift[None]
        assert torch.equal(compiled(x, w, scale, shift, eps), norm_modulate(x, w, scale, shift, eps))
    assert counter.graphs <= 2, counter.graphs  # the first shape may specialise, then the token axis is symbolic


# ---------------------------------------------------------------------------------- (5) checkpoint reuse paths


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
@pytest.mark.parametrize("flags", [(False, True), (True, True), (True, False)], ids=["reuse_only", "cache_reuse", "cache"])
@pytest.mark.parametrize("kind", ["one_sample", "equal_lengths", "padded"])
def test_ckpt_reuse_paths_are_bit_identical(kind, flags, compiled, monkeypatch):
    """--moss_ckpt_reuse_attn (reuse-only and with the cache) and --moss_ckpt_nograd_merged with the option on equal the
    plain checkpoint without it, and the recompute's block outputs equal the first pass's (CKPT_RECOMPUTE_DIFFS 0.0)."""
    batches = base.BATCHES[kind]
    ref, *_ = run(monkeypatch, 1.0, batches, compiled=compiled)
    new, model, _, _ = run(monkeypatch, C, batches, flags=flags, compiled=compiled)
    assert_runs_same(ref, new, f"{kind}, flags {flags}, compiled={compiled}")
    diffs = anima_models.CKPT_RECOMPUTE_DIFFS
    if kind == "padded" and compiled:  # a compiled block's padded call keeps torch.utils.checkpoint
        assert diffs == []
    else:
        assert len(diffs) == len(batches) * len(model.layers) and set(diffs) == {0.0}, diffs


@pytest.mark.parametrize("nograd", [False, True], ids=["reuse_only", "with_cache"])
def test_ckpt_reuse_with_a_budget_is_bit_identical(nograd, monkeypatch):
    """--moss_ckpt_reuse_attn_budget_gib: some blocks reuse, the others keep the plain checkpoint (or the node without a
    stash); the budget reads the shape of x, which the scale does not change."""
    batches = (dict(), dict(text=(11,), hw=(4, 6), seed=2), dict(text=(7, 7), seed=3))
    ref, *_ = run(monkeypatch, 1.0, batches)
    budget = 2 * 22 * (base.TINY["dim"] * 2 + base.TINY["num_heads"] * 4)  # two blocks of a 22-token sample
    new, *_ = run(monkeypatch, C, batches, flags=(nograd, True), budget=budget)
    assert_runs_same(ref, new, f"budget, nograd={nograd}")


# --------------------------------------------------------------------------------- (6) C = 1.0 is the previous code


def _previous_rms_norm(x, weight):  # library/moss/models.py before --moss_stream_scale
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    return y.to(x.dtype) * weight


def _previous_norm_modulate(x, weight, scale, shift):
    return _previous_rms_norm(x, weight) * (1 + scale[:, None]) + shift[:, None]


def test_scale_one_multiplies_nothing_and_keeps_every_eps_none(monkeypatch):
    model = plain_model()
    model.set_stream_scale(C)
    model.set_stream_scale(1.0)  # back to off: the same state as a model never touched
    for m in (model, plain_model()):
        assert m.stream_scale == 1.0 and isinstance(m.stream_scale, float)
        assert all(n.eps is None for n in m.modules() if isinstance(n, moss_models.MossRMSNorm))

    def fail(*_a, **_k):
        raise AssertionError("C = 1.0 multiplied the stream")

    monkeypatch.setattr(MossTransformer, "_scale_stream", fail)
    data = plain_inputs(torch.float32, text=(5, 9))
    assert torch.equal(plain_forward(model, data)[0], plain_forward(plain_model(), data)[0])
    # and the no-argument default of the trainer (set_stream_scale is only called for C != 1.0) is that state
    assert stream_scale_problem(1.0) is None and stream_scale_problem(1) is None


def test_scale_one_norms_equal_the_previous_functions():
    g = torch.Generator().manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        x = torch.randn(2, 9, 48, generator=g).to(dtype) * 300
        w, scale, shift = (torch.randn(2, 48, generator=g).to(dtype) for _ in range(3))
        assert torch.equal(rms_norm(x, w[0]), _previous_rms_norm(x, w[0]))
        assert torch.equal(rms_norm(x, w[0], None), _previous_rms_norm(x, w[0]))
        assert torch.equal(norm_modulate(x, w[0], scale, shift), _previous_norm_modulate(x, w[0], scale, shift))
        assert torch.equal(norm_modulate(x, w[0], scale, shift, None), _previous_norm_modulate(x, w[0], scale, shift))
        # and the exact fold: C x with eps 1e-6 C**2 gives the same bits
        assert torch.equal(rms_norm(x * C, w[0], 1e-6 * C * C), rms_norm(x, w[0]))


def test_scale_one_chain_graph_is_the_previous_graph():
    """eps None (C = 1.0) traces to the same FX graph as the previous chain function: the compiled elementwise chains of
    a CUDA run (and the block graphs, which inline them) compile what they compiled before."""

    def graph_of(fn, *args):
        codes = []

        def backend(gm, example_inputs):
            codes.append([(n.op, str(n.target)) for n in gm.graph.nodes])
            return gm.forward

        torch._dynamo.reset()
        torch.compile(fn, dynamic=True, fullgraph=True, backend=backend)(*args)
        assert len(codes) == 1
        return codes[0]

    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 9, 32, generator=g).to(torch.bfloat16)
    w, scale, shift = torch.randn(32, generator=g), torch.randn(1, 32, generator=g), torch.randn(1, 32, generator=g)
    previous = graph_of(_previous_norm_modulate, x, w, scale, shift)
    assert graph_of(norm_modulate, x, w, scale, shift, None) == previous
    assert graph_of(norm_modulate, x, w, scale, shift, 1e-6 * C * C) != previous  # the eps is in the graph
    torch._dynamo.reset()


def test_scale_one_training_step_is_bit_identical_to_an_untouched_model(monkeypatch):
    ref, *_ = run(monkeypatch, 1.0, base.BATCHES["one_sample"])
    model, net, _ = base.build()
    model.set_stream_scale(1.0)
    set_flags(monkeypatch, False, False)
    new = [base.micro_step(model, net, base.batch(**b)) for b in base.BATCHES["one_sample"]]
    assert_runs_same(ref, new, "set_stream_scale(1.0)")


# ------------------------------------------------------------------------------------------ (7) validation and glue

GOOD = (1, 1.0, 0.5, 0.25, C, 2.0 ** -10, 2.0 ** -20)
BAD = (0, 0.0, -0.5, -C, 2.0, 1.5, 4, 0.1, 0.75, 3 / 16, 0.3, math.inf, -math.inf, math.nan, True, False, "0.0625", None)


def test_stream_scale_problem_accepts_only_powers_of_two_in_range():
    for value in GOOD:
        assert stream_scale_problem(value) is None, value
    for value in BAD:
        assert stream_scale_problem(value), value


def test_model_setter_refuses_and_keeps_its_state():
    model = plain_model()
    model.set_stream_scale(C)
    for value in BAD:
        with pytest.raises(ValueError, match="--moss_stream_scale"):
            model.set_stream_scale(value)
        assert model.stream_scale == C and model.layers[0].norm1.eps == 1e-6 * C * C


def _args(**changes):
    values = dict(moss_stream_scale=C, moss_stream_fp32=False, mixed_precision="fp16", full_fp16=False)
    return types.SimpleNamespace(**{**values, **changes})


@pytest.mark.parametrize("value", BAD)
def test_validation_refuses(value):
    with pytest.raises(ValueError, match="--moss_stream_scale"):
        moss_train_utils.validate_moss_stream_scale_args(_args(moss_stream_scale=value))


def test_validation_through_validate_moss_args_refuses_before_the_datasets():
    args = types.SimpleNamespace(pretrained_model_name_or_path=__file__, vae=__file__, timestep_sampling="moss",
                                 moss_max_token_length=512, cache_text_encoder_outputs=True,
                                 cache_text_encoder_outputs_to_disk=False, network_train_unet_only=True,
                                 mixed_precision="fp16", moss_stream_fp32=False, moss_stream_scale=0.1)
    with pytest.raises(ValueError, match="power of two"):
        moss_train_utils.validate_moss_args(args, None, None)


def test_validation_logs_and_warns(monkeypatch):
    infos, warnings = [], []
    monkeypatch.setattr(moss_train_utils.logger, "info", lambda m, *a, **k: infos.append(str(m)))
    monkeypatch.setattr(moss_train_utils.logger, "warning", lambda m, *a, **k: warnings.append(str(m)))

    args = _args(moss_stream_scale=1)  # a TOML int: off, stored back as a float, nothing logged
    moss_train_utils.validate_moss_stream_scale_args(args)
    assert args.moss_stream_scale == 1.0 and isinstance(args.moss_stream_scale, float) and not infos and not warnings

    args = _args()  # fp16 mixed precision: the start-up line, no warning
    moss_train_utils.validate_moss_stream_scale_args(args)
    assert args.moss_stream_scale == C and len(infos) == 1 and not warnings
    assert "--moss_stream_scale 0.0625 (1/16)" in infos[0] and "16x smaller" in infos[0] and "3.90625e-09" in infos[0]

    moss_train_utils.validate_moss_stream_scale_args(_args(mixed_precision="no", full_fp16=True))
    assert not warnings  # full fp16
    moss_train_utils.validate_moss_stream_scale_args(_args(moss_stream_fp32=True))
    assert len(warnings) == 1 and "redundant" in warnings[0]
    moss_train_utils.validate_moss_stream_scale_args(_args(mixed_precision="bf16"))
    assert len(warnings) == 2 and "only matters for fp16" in warnings[1]


def test_parser_and_toml_key(monkeypatch, tmp_path):
    import moss_train_network as mtn

    parser = mtn.setup_parser()
    parser.format_help()
    assert parser.parse_args([]).moss_stream_scale == 1.0
    cli = parser.parse_args(["--moss_stream_scale", "0.0625"])
    assert cli.moss_stream_scale == C and isinstance(cli.moss_stream_scale, float)
    for text, value in (("moss_stream_scale = 0.0625\n", C), ("moss_stream_scale = 0.5\n", 0.5)):
        config = tmp_path / f"config_{value}.toml"
        config.write_text('mixed_precision = "fp16"\n' + text, encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["moss_train_network.py", "--config_file", str(config)])
        args = train_util.read_config_from_file(parser.parse_args(), parser, unknown_keys="error")
        assert args.moss_stream_scale == value and args.mixed_precision == "fp16"


def test_load_unet_lazily_sets_the_scale(monkeypatch):
    import moss_train_network as mtn

    parser = mtn.setup_parser()
    for argv, want in ((["--moss_stream_scale", "0.0625"], C), ([], 1.0)):
        calls = []
        model = plain_model()
        real = model.set_stream_scale
        model.set_stream_scale = lambda s: (calls.append(s), real(s))[1]
        monkeypatch.setattr(mtn, "load_moss_dit", lambda *a, **k: model)
        trainer = mtn.MossNetworkTrainer()
        trainer.is_swapping_blocks = False
        got, _ = trainer.load_unet_lazily(parser.parse_args(argv), torch.float32,
                                          types.SimpleNamespace(device=torch.device("cpu")), [])
        assert got is model and got.stream_scale == want
        assert calls == ([C] if want != 1.0 else [])  # off: the DiT is not touched at all
        assert all((layer.norm1.eps is None) == (want == 1.0) for layer in got.layers)


def test_block_compile_gets_a_cache_signature_of_its_own(monkeypatch):
    import moss_train_network as mtn

    signatures = []
    monkeypatch.setattr(anima_block_compile, "isolate_inductor_cache", lambda s: signatures.append(s) or "dir")
    for scale in (1.0, C):
        model = plain_model()
        mbc.compile_moss_blocks(model, backend="inductor", stream_scale=scale)  # lazy: nothing traced, no compiler
        mbc.restore_moss_blocks(model)
    assert "stream_scale" not in signatures[0] and signatures[1] == signatures[0] + ";stream_scale=0.0625"

    # the trainer hands the option to the compile (after_network_loaded)
    seen = {}
    monkeypatch.setattr(mtn.moss_block_compile, "compile_moss_blocks", lambda model, **kw: seen.update(kw))
    args = mtn.setup_parser().parse_args(["--moss_block_compile", "--moss_stream_scale", "0.0625"])
    accelerator = types.SimpleNamespace(unwrap_model=lambda m: m, print=lambda *a, **k: None)
    mtn.MossNetworkTrainer().after_network_loaded(args, accelerator, None, [], plain_model())
    assert seen["stream_scale"] == C


def test_state_dict_is_unchanged():
    """No parameter, buffer or key changes: a state saved with or without the option loads into the other."""
    off, on = plain_model(), plain_model(scale=C)
    a, b = off.state_dict(), on.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    off.load_state_dict(b, strict=True)
