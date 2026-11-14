"""Second reviewer pass (2026-11-14, after the review12 fixes) for --anima_ckpt_nograd_merged / --anima_ckpt_reuse_attn.
CPU only (flash runs on tests/_fake_flash.py); nothing here touches the implementation.

What the fix round claimed and what this file pins:

Findings of this pass (strict xfail when written, fixed the same day: plain tests now):
* test_t_embedder_training_gets_the_start_up_warning: a LyCORIS preset that adapts TimestepEmbedding (the shipped
  anima-full-no-excludes and anima-inpaint do) trains t_embedder; with the flag its LoKr / DoRA gradients are NOT those of
  the plain checkpoint (test_t_embedder_gradients_differ_from_the_plain_checkpoint measures it: up to ~1e-2 relative on a
  bf16 stream, ~4e-7 on the fp32 stream), the option's help text says "(a warning at start-up)" for the timestep
  embedding too, and ckpt_reuse_text_grad_warning only knew llm_adapter and the text encoder. Fix:
  anima_train_network.network_trains_t_embedder, and the warning names t_embedder (after_network_loaded logs it).
* test_gpu_harness_has_a_flags_off_baseline: finding 4 of the first review (the GPU harness compared flags on with
  flags off but never flags off with flags off, so on the GPU a non-zero "differing" could not be told from flash's
  non-deterministic backward). Fix: tests/manual_test_anima_ckpt_reuse_gpu.py repeats the flags-off micro-step
  (--baseline_runs), reports the worst run-to-run difference as correctness["off_vs_off"] and reads every flags-on entry
  against it ("reading"); test_gpu_harness_reads_a_difference_against_the_noise_floor covers the three readings.

Regression guards for what this pass checked and found right (they pass):
* the trainer wiring by behaviour, not by source text: AnimaNetworkTrainer.assert_extra_args on the real parser's
  namespace refuses --anima_stock_lycoris with the flag, warns once where the options were parsed but never applied
  (anima_train_addift.py / dataset-backed iLECO) and says nothing where prepare_anima_runtime applied them or where no
  option is given;
* every other adapter family of the toy presets (lora, loha+DoRA, tsm stage 1, tsm-lokr stage 1 / 2, tlora, full,
  abba+DoRA, diag-oft, boft, dylora - the first review and the implementer's tests only ran LoKr+DoRA with the flags on):
  outputs and every gradient equal the plain checkpoint's, the recompute equals the first pass, the caller's RNG stream
  ends in the same state (dylora draws its rank from Python's random: reused by the recompute as under the plain one);
* micro-steps of several rows with their own timesteps and other text lengths (--anima_micro_batch_max_tokens), with
  --anima_adaln_hoist, an optimizer step in between, all / every other Block checkpointed, both streams;
* --lycoris_stock (the fixed paths not loaded, the fast LoKr forward serves EVERY multiplier, so the cache is read at
  m != 1 too - the first review's multiplier test ran with the fixed paths, which leave the fast path there):
  m = 1, 0.5, 0, -1, 1 equal the plain checkpoint (a fresh interpreter: the fixed paths are a class-level patch);
* the fix's new condition in library/attention.py (`_wrapped_flash_attn_backward is not None`) inside a dynamo trace
  with the REAL flash-attn handle (a torch._ops.OpOverloadPacket; the CPU stand-in is a Python function): one graph, no
  graph break, and hiding the handle afterwards re-specialises to "nothing stored".

Not here (needs the GPU, see the review report): inductor's graphs, the real flash kernels, VRAM, speed.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_anima_ckpt_reuse_review2.py
"""
import argparse
import json
import os
import random
import subprocess
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
import torch._dynamo

from library import anima_block_compile as bc
from library import anima_models, attention, lycoris_args, lycoris_fast
from library import lycoris_algo_fixes as laf
from tests import _fake_flash as fake_flash
from tests import test_anima_block_compile as T

torch.set_num_threads(4)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    lycoris_fast.install_fast_lokr_forward()
    laf.apply_module_patches()
    torch._dynamo.reset()
    bc._warned_oob.clear()
    fake_flash.install(monkeypatch)
    monkeypatch.setitem(T.TOY, "attn_mode", "flash")
    for name, value in (("CKPT_NOGRAD_MERGED", False), ("CKPT_REUSE_ATTN", False), ("CKPT_RECOMPUTE_DIFFS", None),
                        ("ADALN_HOIST", False)):
        monkeypatch.setattr(anima_models, name, value)
    import anima_train_network as atn

    if atn._CKPT_STEP_HOOK is not None:  # another test file left the global optimizer-step hook registered
        atn._CKPT_STEP_HOOK.remove()
        atn._CKPT_STEP_HOOK = None
    with torch.backends.mkldnn.flags(enabled=False):  # oneDNN has no bf16 backward on some CPUs
        yield
    laf.set_block_compile(False)
    # what tests/test_anima_block_compile.build changes per preset (its own fixture puts these back)
    laf._OFT_MAX_BLOCK = T._OFT_MAX_BLOCK_DEFAULT
    laf._TSM_EXPERT_FN = False
    laf._TSM_SAMPLE_T.clear()
    from lycoris.modules.tsm import clear_tsm_timestep

    clear_tsm_timestep(0)
    assert lycoris_fast._CKPT_PASS == 0, "a checkpoint pass left lycoris_fast._CKPT_PASS set"
    torch._dynamo.reset()


# ------------------------------------------------------------------------------------------------------------- helpers


def set_flags(monkeypatch, nograd: bool, reuse: bool = False):
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", nograd)
    monkeypatch.setattr(anima_models, "CKPT_REUSE_ATTN", reuse)
    monkeypatch.setattr(anima_models, "CKPT_RECOMPUTE_DIFFS", [] if nograd else None)


def perturb(net, seed: int = 7, scale: float = 0.02) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            if p.requires_grad and p.is_floating_point():
                p.add_(torch.randn(p.shape, generator=g).to(p.dtype) * scale)


def build_lokr(preset: str, blocks: int = 4, stream: bool = False):
    """The toy DiT with the production LoKr factor 4 + DoRA network under another LyCORIS preset (which modules get an
    adapter), applied the way train_network.py does it; every Block checkpointed."""
    import lycoris.kohya as kohya

    torch.manual_seed(0)
    unet = anima_models.Anima(**{**T.TOY, "num_blocks": blocks}).to(torch.bfloat16)
    unet.requires_grad_(False)
    unet.train()
    args = T.make_args(["algo=lokr", f"preset={preset}", "factor=4", "full_matrix=True", "decompose_both=False",
                        "dora_wd=True", "wd_on_output=True"], 10000, 10000)
    kwargs = lycoris_args.fix_lycoris_bool_kwargs(args.network_module, dict(i.split("=", 1) for i in args.network_args))
    create = laf.prepare_network_kwargs("lokr", dict(kwargs))
    create.setdefault("dropout", 0.0)
    laf.set_block_compile(False)
    net = kohya.create_network(1.0, args.network_dim, args.network_alpha, None, None, unet, neuron_dropout=None, **create)
    if hasattr(net, "prepare_network"):
        net.prepare_network(args)
    net.apply_to(None, unet, False, True)
    laf.after_network_created(net, args)
    net.requires_grad_(True)
    net.train()
    perturb(net)
    if stream:
        unet.enable_stream_fp32()
    unet.enable_gradient_checkpointing()
    return unet, net


def micro_step(unet, net, stream: bool, seed: int = 5, latent_hw=(16, 16)):
    x, t, ctx, pad = T.inputs(latent_hw=latent_hw, t=[0.3], seed=seed)
    if stream:  # --anima_stream_fp32: the trainer keeps the noisy input and the timestep in fp32
        x, t = x.float(), t.float()
    return T.step(unet, net, x, t, ctx, pad, checkpoint=True)


T_EMBEDDER_PRESET = "anima-full-no-excludes"  # lycoris/config.py: Block, PatchEmbed, TimestepEmbedding, FinalLayer


def _t_embedder_runs(monkeypatch, stream: bool):
    results = {}
    for name, flags in (("off", (False, False)), ("on", (True, True))):
        unet, net = build_lokr(T_EMBEDDER_PRESET, stream=stream)
        set_flags(monkeypatch, *flags)
        results[name] = (*micro_step(unet, net, stream), net)
    return results["off"], results["on"]


# ------------------------------------------------------------------------------------------- finding: t_embedder training


@pytest.mark.parametrize("stream", [False, True], ids=["bf16_stream", "fp32_stream"])
def test_t_embedder_gradients_differ_from_the_plain_checkpoint(stream, monkeypatch):
    """The measurement behind the finding below (this part passes): with t_embedder adapted (preset
    anima-full-no-excludes; anima-inpaint targets TimestepEmbedding too) the flag changes exactly the t_embedder
    gradients - emb / adaln_lora enter every Block, the node sums a Block's uses before the Blocks are summed - and
    nothing else: the output and every Block / x_embedder / final_layer gradient stay bit-identical. On a bf16 stream the
    partial sums are rounded to bf16 (~1e-2 relative here), on the fp32 stream (--anima_stream_fp32) to fp32 (~4e-7)."""
    (y0, g0, _), (y1, g1, _) = _t_embedder_runs(monkeypatch, stream)
    assert torch.equal(y0, y1) and g0.keys() == g1.keys()
    differing = sorted(n for n in g0 if not torch.equal(g0[n], g1[n]))
    t_embedder = sorted(n for n in g0 if "t_embedder" in n)
    assert len(t_embedder) == 6  # linear_1 / linear_2 of the TimestepEmbedding: lokr_w1, lokr_w2, dora_scale each
    assert differing and set(differing) <= set(t_embedder), differing
    worst = max((g0[n].float() - g1[n].float()).abs().max().item() / g0[n].float().abs().max().item() for n in differing)
    print(f"\nt_embedder grads ({'fp32' if stream else 'bf16'} stream): {len(differing)} of {len(t_embedder)} tensors differ,"
          f" worst max|d| / max|g| = {worst:.3e}")
    assert worst < (1e-5 if stream else 5e-2)  # rounding of the partial sums, not another function


def test_t_embedder_training_gets_the_start_up_warning(monkeypatch):
    """The first fix round added the warning for llm_adapter / text encoder training only, while the option's help text,
    written in the same round, reads "gradients that pass through the text conditioning (llm_adapter or text encoder
    training) or the timestep embedding are summed per Block first and differ from it by rounding (a warning at
    start-up)". Now the warning covers a trained t_embedder: named in the message, alone and next to the text encoder,
    logged once by AnimaNetworkTrainer.after_network_loaded; nothing for the production preset (anima-exp: Blocks only),
    for a frozen t_embedder adapter, or with the switch off."""
    import anima_train_network as atn

    unet, net = build_lokr(T_EMBEDDER_PRESET, blocks=1)
    trained = [n for n, p in net.named_parameters() if p.requires_grad and "t_embedder" in n]
    assert trained, "the preset no longer adapts t_embedder: pick another one for this test"
    parser = atn.setup_parser()
    help_text = " ".join(next(a.help for a in parser._actions if "--anima_ckpt_nograd_merged" in a.option_strings).split())
    assert "timestep embedding" in help_text and "(a warning at start-up)" in help_text  # what the help promises
    assert atn.network_trains_t_embedder(net) and not atn.network_trains_t_embedder(None)

    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", False)
    assert atn.ckpt_reuse_text_grad_warning(net, False) is None  # switch off: nothing to say
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", True)
    warning = atn.ckpt_reuse_text_grad_warning(net, False)
    assert warning is not None, "t_embedder is trained, the help text promises a warning, none is given"
    assert "with t_embedder trained" in warning and "the timestep-embedding gradient is summed per Block first" in warning
    assert "--anima_stream_fp32" in warning and "all other LoKr / DoRA gradients are unchanged" in warning
    assert "text-conditioning" not in warning  # only what is trained is named
    both = atn.ckpt_reuse_text_grad_warning(net, True)
    assert "with the text encoder and t_embedder trained" in both
    assert "the text-conditioning and timestep-embedding gradients are summed per Block first" in both
    text_only = atn.ckpt_reuse_text_grad_warning(argparse.Namespace(train_llm_adapter=True), True)  # as before the fix
    assert "with llm_adapter and the text encoder trained" in text_only and "timestep" not in text_only
    assert "the DiT's LoKr / DoRA gradients are unchanged" in text_only

    warned = []
    monkeypatch.setattr(atn.logger, "warning", lambda message, *a, **k: warned.append(str(message)))
    args = argparse.Namespace(anima_block_compile=False, network_train_unet_only=True)
    atn.AnimaNetworkTrainer().after_network_loaded(args, None, net, [], None)
    assert warned == [warning]  # the trainer says it once at start-up

    _, production = build_lokr("anima-exp", blocks=1)  # the shipped config: LoKr+DoRA on the Blocks, t_embedder untouched
    assert not atn.network_trains_t_embedder(production)
    assert atn.ckpt_reuse_text_grad_warning(production, False) is None
    for name, p in net.named_parameters():
        if "t_embedder" in name:
            p.requires_grad_(False)
    assert not atn.network_trains_t_embedder(net) and atn.ckpt_reuse_text_grad_warning(net, False) is None  # frozen


# ------------------------------------------------------------------- finding: the GPU harness had no flags-off baseline


def test_gpu_harness_has_a_flags_off_baseline(tmp_path):
    """The harness's CPU self-test (toy shapes, the fake flash kernels), smallest run: every flags-on correctness entry
    compares a flag set with "off"; the flags-off micro-step repeated and compared with the first (the run-to-run noise
    of flash's default, non-deterministic backward on the GPU) is what a non-zero "differing" has to be read against. The
    kernel-level floor of tests/manual_test_flash_reuse_kernel_gpu.py does not say how far that noise travels through the
    Blocks into the LoKr gradients this harness compares. On the CPU the noise is zero and every reading bit-identical."""
    out = tmp_path / "harness.json"
    done = subprocess.run([sys.executable, "-B", os.path.join("tests", "manual_test_anima_ckpt_reuse_gpu.py"), "--device", "cpu",
                           "--tokens", "16", "--blocks", "1", "--iters", "1", "--warmup", "0", "--modes", "eager",
                           "--json", str(out)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=900, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    assert done.returncode == 0, done.stderr[-2000:]
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["runs"], "the self-test produced no run"
    for run in report["runs"]:
        assert set(run["correctness"]) >= {"nograd", "nograd+reuse"}  # flags on vs off
        baseline = [k for k in run["correctness"] if k.lower().startswith("off")]
        baseline += [k for k in run if any(word in k.lower() for word in ("noise", "baseline", "off_vs_off"))]
        assert baseline, f"no flags-off vs flags-off entry in the report (correctness keys: {sorted(run['correctness'])})"
        floor = run["correctness"]["off_vs_off"]
        assert floor["runs"] == 3 and floor["same_keys"] and floor["grad_tensors"] > 0  # --baseline_runs default
        assert floor["output"]["differing"] == 0 and floor["grads"]["differing"] == 0  # the CPU is deterministic
        for flags in ("nograd", "nograd+reuse"):
            entry = run["correctness"][flags]
            assert entry["reading"] == "bit-identical" and entry["grad_tensors"] == floor["grad_tensors"]


def test_gpu_harness_reads_a_difference_against_the_noise_floor():
    """The two functions behind that entry, on made-up results (output, {name: gradient}): noise_floor keeps the worst of
    every number over the repeats (a count is the worst repeat's, not a sum), reading says bit-identical only for no
    differing element, "inside" up to twice the floor's max |d| for the output AND the gradients (the kernel script's
    rule), OUTSIDE otherwise - also against a zero floor, which is what the GPU measured on 2026-11-14 (flash's backward
    repeated bit for bit there, so the compiled-mode difference is a property of the two graphs, not noise)."""
    from tests import manual_test_anima_ckpt_reuse_gpu as harness

    y, grads = torch.ones(4), {"a": torch.ones(3), "b": torch.full((2,), 2.0)}
    ref = (y, grads)

    def moved(dy: float, da: float):
        return y + dy, {"a": grads["a"] + da, "b": grads["b"]}

    floor = harness.noise_floor(ref, [moved(0.0, 0.0), moved(1e-3, 2e-3), moved(5e-4, 1e-3)])
    assert floor["runs"] == 3 and floor["same_keys"] and floor["grad_tensors"] == 2
    assert floor["output"]["max_abs"] == pytest.approx(1e-3, rel=1e-3) and floor["output"]["differing"] == 4
    assert floor["grads"]["max_abs"] == pytest.approx(2e-3, rel=1e-3) and floor["grads"]["max_rel"] == pytest.approx(2e-3, rel=1e-3)
    assert floor["grads"]["differing"] == 3 and floor["grads"]["tensors_differing"] == 1 and floor["grads"]["worst"] == "a"
    zero = harness.noise_floor(ref, [moved(0.0, 0.0)])
    assert zero["runs"] == 1 and zero["output"]["differing"] == 0 and zero["grads"]["differing"] == 0

    def read(dy: float, da: float, against):
        return harness.reading(harness.compare_results(ref, moved(dy, da)), against)

    assert read(0.0, 0.0, zero) == read(0.0, 0.0, floor) == "bit-identical"
    assert read(1e-3, 3e-3, floor) == "inside the flags-off run-to-run noise"  # 1x the output's floor, 1.5x the gradients'
    assert read(1e-3, 5e-3, floor).startswith("OUTSIDE")  # gradients 2.5x
    assert read(3e-3, 0.0, floor).startswith("OUTSIDE")  # output 3x, gradients equal
    assert read(0.0, 1e-6, zero).startswith("OUTSIDE")  # no noise measured: any difference is the code path's


# ------------------------------------------------------------------------------- the trainer wiring, by behaviour


class _Group:
    """What AnimaNetworkTrainer.assert_extra_args asks of a dataset group."""
    datasets = []

    def is_text_encoder_output_cacheable(self, **kwargs):
        return True

    def verify_bucket_reso_steps(self, steps):
        pass


def _parsed(*extra):
    import anima_train_network as atn

    return atn.setup_parser().parse_args(["--timestep_sampling", "sigmoid", "--gradient_checkpointing",
                                          "--network_train_unet_only", "--attn_mode", "flash", *extra])


def test_assert_extra_args_refuses_and_warns_as_the_fix_says(monkeypatch):
    """The fix round's tests look for the two call names in assert_extra_args's source; here the method runs on the real
    parser's namespace: (a) prepare_anima_runtime applied the options -> silent, (b) parsed but never applied (the switch
    is off: anima_train_addift.py, dataset-backed iLECO) -> one "has no effect" warning, (c) --anima_stock_lycoris with
    the flag -> ValueError, with or without the switch, (d) no option -> silent, nothing raised."""
    import anima_train_network as atn

    warned = []
    monkeypatch.setattr(atn.logger, "warning", lambda message, *a, **k: warned.append(str(message)))
    flags = ("--anima_ckpt_nograd_merged", "--anima_ckpt_reuse_attn")

    def run(*extra, switch):
        warned.clear()
        monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", switch)
        atn.AnimaNetworkTrainer().assert_extra_args(_parsed(*extra), _Group(), None)
        return [m for m in warned if "anima_ckpt" in m]

    assert run(*flags, switch=True) == []  # (a) anima_train_network.py
    not_applied = run(*flags, switch=False)  # (b)
    assert len(not_applied) == 1 and "has no effect in this script" in not_applied[0]
    assert "--anima_ckpt_nograd_merged / --anima_ckpt_reuse_attn" in not_applied[0]
    for switch in (True, False):  # (c)
        with pytest.raises(ValueError, match="not with --anima_stock_lycoris"):
            run(*flags, "--anima_stock_lycoris", switch=switch)
    assert run(switch=False) == [] and run("--anima_stock_lycoris", switch=False) == []  # (d)


# ---------------------------------------------------------------------------------- other adapter families, flags on


OTHER_PRESETS = ("lora", "loha_dora", "tsm", "tsm_lokr", "tsm_lokr_stage2", "tlora", "full", "abba_dora", "diag_oft", "boft",
                 "dylora")


def _preset_run(preset: str, flags, monkeypatch):
    torch._dynamo.reset()
    laf._TSM_SAMPLE_T.clear()
    torch.manual_seed(0)
    unet, net, _ = T.build(preset, num_blocks=3, block_compile=False)
    T.seed_zero_factors(net)
    T.perturb_tsm(net)
    T.perturb_tlora(net)
    perturb(net)
    unet.enable_gradient_checkpointing()
    set_flags(monkeypatch, *flags)
    torch.manual_seed(123)
    random.seed(123)  # dylora draws its rank per forward from Python's random
    results = []
    for i, hw in enumerate(((8, 8), (8, 12))):
        x, t, ctx, pad = T.inputs(latent_hw=hw, t=[0.3 + 0.2 * i], seed=i + 1)
        results.append(T.step(unet, net, x, t, ctx, pad, checkpoint=True))
    return results, list(anima_models.CKPT_RECOMPUTE_DIFFS or []), torch.get_rng_state(), unet


@pytest.mark.parametrize("preset", OTHER_PRESETS)
def test_other_adapters_match_the_plain_checkpoint(preset, monkeypatch):
    """docs: "other adapters than LoKr+DoRA get the no-grad first pass without a cache". Nothing of theirs may depend on
    the first pass running under no_grad instead of grad mode (a path chosen by grad mode, a random draw outside the
    replayed torch RNG): two micro-steps (two grids, two timesteps), eager, flags on with the stored self-attention."""
    ref, _, rng0, _ = _preset_run(preset, (False, False), monkeypatch)
    new, diffs, rng1, unet = _preset_run(preset, (True, True), monkeypatch)
    for i, (a, b) in enumerate(zip(ref, new)):
        T.assert_same(a, b, f"{preset}, micro-step {i}: no-grad first pass + reuse vs torch.utils.checkpoint")
    assert len(diffs) == 2 * 3 and set(diffs) == {0.0}, diffs  # one node per Block and micro-step, recompute == first pass
    assert torch.equal(rng0, rng1), "the caller's RNG stream ends elsewhere"
    assert lycoris_fast.ckpt_merged_bytes(unet) == 0  # no LokrModule here: nothing to cache, nothing cached


# ----------------------------------------------------------------------- rows per micro-step, text length, tier mixes


@pytest.mark.parametrize("half", [False, True], ids=["all_blocks", "every_other_block"])
@pytest.mark.parametrize("stream", [True, False], ids=["fp32_stream", "bf16_stream"])
def test_multi_row_micro_steps_and_text_lengths(stream, half, monkeypatch):
    """What --anima_micro_batch_max_tokens feeds the Blocks and the earlier tests did not: micro-steps of 3, 2, 1 and 3
    rows (each row its own timestep), text of 8 and 13 tokens, --anima_adaln_hoist on, an optimizer step in the middle
    (the cache rebuilt for the new values); LoKr+DoRA, 4 Blocks, all or every other one checkpointed. Flags on with the
    stored self-attention == the plain checkpoint, micro-step by micro-step."""
    def run(flags):
        monkeypatch.setattr(anima_models, "ADALN_HOIST", True)
        unet, net, _ = T.build("lokr_dora", num_blocks=4, block_compile=False)
        perturb(net)
        if stream:
            unet.enable_stream_fp32()
        unet.enable_gradient_checkpointing()
        if half:
            for i, block in enumerate(unet.blocks):
                if i % 2:
                    block.disable_gradient_checkpointing()
        set_flags(monkeypatch, *flags)
        opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=1e-2)
        results = []
        for i, (hw, rows, text_len) in enumerate((((8, 8), 3, 8), ((8, 12), 2, 13), ((16, 16), 1, 8), ((8, 8), 3, 8))):
            x, t, ctx, pad = T.inputs(latent_hw=hw, batch=rows, text_len=text_len, seed=i + 1)
            if stream:
                x, t = x.float(), t.float()
            results.append(T.step(unet, net, x, t, ctx, pad, checkpoint=True))
            if i == 1:
                opt.step()  # SGD writes the parameters in place: the keys see it, no invalidation call needed
        return results, list(anima_models.CKPT_RECOMPUTE_DIFFS or [])

    ref, _ = run((False, False))
    new, diffs = run((True, True))
    for i, (a, b) in enumerate(zip(ref, new)):
        T.assert_same(a, b, f"micro-step {i}")
    assert not torch.equal(ref[0][0], ref[3][0])  # same inputs before and after the optimizer step: it changed the model
    assert len(diffs) == 4 * (2 if half else 4) and set(diffs) == {0.0}, diffs


# -------------------------------------------------------------------------------- --lycoris_stock: the cache at m != 1


_LYCORIS_STOCK_PROBE = r"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import torch
torch.set_num_threads(4)
from library import anima_models, attention, lycoris_fast
from tests import _fake_flash as fake_flash

assert "library.lycoris_algo_fixes" not in sys.modules  # --lycoris_stock: the fixed paths are never imported
lycoris_fast.install_fast_lokr_forward()  # prepare_anima_runtime without --anima_stock_lycoris
attention.flash_attn_func = fake_flash.flash_attn_func
attention._wrapped_flash_attn_backward = fake_flash.wrapped_flash_attn_backward
TOY = dict(max_img_h=32, max_img_w=32, max_frames=1, in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
           model_channels=64, num_blocks=3, num_heads=2, mlp_ratio=4.0, crossattn_emb_channels=32, use_adaln_lora=True,
           adaln_lora_dim=16, use_llm_adapter=False, attn_mode="flash", pos_emb_cls="rope3d")
MULTIPLIERS = (1.0, 0.5, 0.0, -1.0, 1.0)


def build():
    import lycoris.kohya as kohya

    torch.manual_seed(0)
    unet = anima_models.Anima(**TOY).to(torch.bfloat16)
    unet.requires_grad_(False)
    unet.train()
    net = kohya.create_network(1.0, 10000, 10000, None, None, unet, neuron_dropout=None, algo="lokr", preset="anima-exp",
                               factor=4, full_matrix=True, decompose_both=False, dora_wd=True, wd_on_output=True, dropout=0.0)
    net.apply_to(None, unet, False, True)
    net.requires_grad_(True)
    net.train()
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in net.parameters():
            if p.requires_grad and p.is_floating_point():
                p.add_(torch.randn(p.shape, generator=g).to(p.dtype) * 0.02)
    unet.enable_gradient_checkpointing()
    return unet, net


def run(flags):
    unet, net = build()
    anima_models.CKPT_NOGRAD_MERGED, anima_models.CKPT_REUSE_ATTN = flags
    anima_models.CKPT_RECOMPUTE_DIFFS = [] if flags[0] else None
    results = []
    for i, m in enumerate(MULTIPLIERS):
        net.set_multiplier(m)
        net.zero_grad(set_to_none=True)
        g = torch.Generator().manual_seed(i + 1)
        x = torch.randn(1, 16, 1, 8, 8, generator=g).to(torch.bfloat16).requires_grad_(True)
        ctx = torch.randn(1, 8, 32, generator=g).to(torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = unet(x, torch.tensor([0.3]).to(torch.bfloat16), ctx, padding_mask=torch.zeros(1, 1, 8, 8, dtype=torch.bfloat16))
        y.float().square().mean().backward()
        results.append((y.detach().float().clone(),
                        {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}))
    return results, list(anima_models.CKPT_RECOMPUTE_DIFFS or []), lycoris_fast.ckpt_merged_bytes(unet)


with torch.backends.mkldnn.flags(enabled=False):
    ref, _, _ = run((False, False))
    new, diffs, held = run((True, True))
for i, ((y0, g0), (y1, g1)) in enumerate(zip(ref, new)):
    bad = [n for n in g0 if n not in g1 or not torch.equal(g0[n], g1[n])]
    print("STEP", i, MULTIPLIERS[i], int(torch.equal(y0, y1)), len(g0), len(bad))
print("CHANGED", int(not torch.equal(ref[0][0], ref[1][0])))  # same weights, another multiplier: another output
print("NODES", len(diffs), max(diffs, default=-1.0), "CACHE_BYTES", held)
"""


def test_lycoris_stock_paths_with_the_cache_read_at_every_multiplier():
    """--lycoris_stock keeps LyCORIS's own multiplier formula and the fast LoKr forward serves it for every m, so the
    checkpoint cache is READ at m != 1 (with the fixed paths, the default, m != 1 leaves the fast path and the first
    review's multiplier test never read it there). iLECO / ADDifT / reverse pairs set m per forward: 1, 0.5, 0, -1 and 1
    again, each micro-step equal to the plain checkpoint's (the key carries the multiplier, the cache is built with
    multiplier_buf). A fresh interpreter: the fixed paths are a class-level patch this process already carries."""
    done = subprocess.run([sys.executable, "-B", "-c", _LYCORIS_STOCK_PROBE], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=900, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    assert done.returncode == 0, done.stderr[-2000:]
    steps = [line.split() for line in done.stdout.splitlines() if line.startswith("STEP ")]
    assert len(steps) == 5
    for _, i, m, same_output, grads, differing in steps:
        assert int(same_output) == 1 and int(grads) > 0 and int(differing) == 0, f"multiplier {m} (micro-step {i})"
    assert done.stdout.split("CHANGED", 1)[1].split()[0] == "1"
    nodes = done.stdout.rsplit("NODES", 1)[1].split()
    assert int(nodes[0]) == 5 * 3 and float(nodes[1]) == 0.0 and int(nodes[3]) > 0, nodes


# ------------------------------------------------------------------- the fix's new condition with the real flash handle


def test_stash_condition_traces_with_the_real_backward_op_handle(monkeypatch):
    """library/attention.py now reads `_wrapped_flash_attn_backward is not None` before it stores anything; inside a
    compiled first-pass Block graph that is a global dynamo has to guard. The CPU tests put a Python function there
    (tests/_fake_flash.py); the installed flash-attn's handle is a torch._ops.OpOverloadPacket. With the real handle and
    the stand-in forward kernel: the first-pass call traces to one graph without a graph break and fills the list;
    with the handle hidden the same compiled function re-specialises and stores nothing."""
    try:
        from flash_attn import flash_attn_interface
    except ImportError:
        pytest.skip("flash-attn is not installed")
    real = getattr(flash_attn_interface, "_wrapped_flash_attn_backward", None)
    if real is None:
        pytest.skip("this flash-attn has no separate backward op")
    monkeypatch.setattr(attention, "_wrapped_flash_attn_backward", real)  # the forward stays the CPU stand-in
    assert attention.flash_reuse_available()
    params = attention.AttentionParams.create_attention_params("flash", False)

    def first_pass(q, k, v, stash):
        return attention.attention([q, k, v], attn_params=params, ckpt_stash=stash)

    q, k, v = (torch.randn(1, 5, 2, 8, generator=torch.Generator().manual_seed(3)) for _ in range(3))
    with torch.no_grad():
        explained = torch._dynamo.explain(first_pass)(q, k, v, [])
        reasons = sorted({str(r.reason).splitlines()[0][:120] for r in explained.break_reasons})
        assert explained.graph_count == 1 and explained.graph_break_count == 0, reasons
        compiled = torch.compile(first_pass, backend="eager", fullgraph=True)
        stash = []
        out = compiled(q, k, v, stash)
        assert len(stash) == 2 and torch.equal(stash[0].reshape(1, 5, 16), out)
        monkeypatch.setattr(attention, "_wrapped_flash_attn_backward", None)
        stash = []
        assert torch.equal(compiled(q, k, v, stash), out) and stash == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
