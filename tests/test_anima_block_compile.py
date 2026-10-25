"""CPU tests for --anima_block_compile (library/anima_block_compile.py): per-Block torch.compile of the Anima DiT with the
LyCORIS fixed paths inside the graphs.

Checked on torch 2.7.1 CPU with the aot_eager / eager backends (the same dynamo tracing and the same AOTAutograd joint
graph as inductor, without codegen) on a 2-Block toy Anima and networks built the way train_network.py builds them
(lycoris.kohya + library/lycoris_algo_fixes + library/lycoris_fast, preset anima-exp):
* token bands: pixels -> tokens, tier clustering, union / bands / static plans, the dataset-group reader;
* the fake-5D Block loop (Anima.forward_mini_train_dit with _block_compile) is bit-identical to the plain loop;
* every Block._forward traces to ONE graph with no graph break for lokr+DoRA, lora, tsm stage 1 / 2 and tsm-lokr;
* compiled forward + backward reproduce the eager outputs and every network gradient exactly, with and without gradient
  checkpointing (unsloth offload and torch.utils.checkpoint) - plain LoRA's six adaLN-modulation up-weight gradients
  within a bf16 ulp or two (see assert_same);
* TSM: timesteps of one expert interval share the graphs (no recompile), another interval adds at most one graph per
  Block dtype regime; a mixed-timestep chunk (batch > 1 with different timesteps) runs eager through skip_call;
* a resolution sweep inside the union band runs on the first graphs (soft marks under attn_mode torch);
* pause() / eval run the eager Blocks; restore_blocks puts the class methods back;
* rebind_forwards keeps lycoris_fast's merged-weight cache targets; library/attention.py's split path traces without a
  graph break; the trainer glue: validate_block_compile_args, after_network_loaded, prepare_anima_runtime, the parser.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -B -m pytest tests/test_anima_block_compile.py
"""
import argparse
import logging
import os
import types
from typing import Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
import torch._dynamo
from torch._dynamo.backends.registry import lookup_backend

from library import anima_block_compile as bc
from library import anima_models, attention, lycoris_args, lycoris_fast
from library import lycoris_algo_fixes as laf

torch.set_num_threads(4)
LOG = logging.getLogger("test_anima_block_compile")

# the toy Anima: 2 Blocks, rope3d, AdaLN-LoRA, attn_mode torch (the production attention), like the 2.9B / 3.8B configs
TOY = dict(
    max_img_h=32, max_img_w=32, max_frames=1, in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
    model_channels=64, num_blocks=2, num_heads=2, mlp_ratio=4.0, crossattn_emb_channels=32,
    use_adaln_lora=True, adaln_lora_dim=16, use_llm_adapter=False, attn_mode="torch", pos_emb_cls="rope3d",
)
# network_args per algo, the production settings where the toy allows (docs/tsm_lokr.md "production LoKr settings")
PRESETS = {
    "lokr_dora": dict(args=["algo=lokr", "preset=anima-exp", "factor=4", "full_matrix=True", "decompose_both=False",
                            "dora_wd=True", "wd_on_output=True"], dim=10000, alpha=10000),
    "lora": dict(args=["algo=lora", "preset=anima-exp"], dim=8, alpha=8),
    "tsm": dict(args=["algo=tsm", "preset=anima-exp", "tsm_n_scales=8,1", "tsm_stage=1"], dim=8, alpha=8),
    "tsm_stage2": dict(args=["algo=tsm", "preset=anima-exp", "tsm_n_scales=8,1", "tsm_stage=2"], dim=8, alpha=8),
    "tsm_lokr": dict(args=["algo=tsm-lokr", "preset=anima-exp", "tsm_n_scales=8,1", "tsm_stage=1", "factor=4",
                           "full_matrix=True", "decompose_both=False", "dora_wd=True", "wd_on_output=True",
                           "tsm_offload_states=False"], dim=10000, alpha=10000),
}
TOY_TOKENS = [16, 24, 36, 64]  # latent grids 8x8, 8x12, 12x12, 16x16 at patch 2
TOY_BANDS = bc.plan_bands(TOY_TOKENS, "union")  # [(16, 64)]


@pytest.fixture(autouse=True)
def _env():
    laf.apply_module_patches()
    lycoris_fast.install_fast_lokr_forward()
    torch._dynamo.reset()
    bc._warned_oob.clear()
    yield
    laf.set_block_compile(False)
    laf._TSM_SAMPLE_T.clear()
    from lycoris.modules.tsm import clear_tsm_timestep

    clear_tsm_timestep(0)
    torch._dynamo.reset()


# ------------------------------------------------------------------------------------------------------------- helpers


def make_args(network_args, dim, alpha):
    return argparse.Namespace(network_args=list(network_args), network_dim=dim, network_alpha=alpha,
                              network_module="lycoris.kohya", network_dropout=None, lycoris_stock=False,
                              lycoris_algo_fixes=False, network_weights=None)


def build(preset: str, num_blocks: int = 2, block_compile: bool = True, seed: int = 0):
    """Toy Anima (frozen bf16, train mode) + the LyCORIS network applied to it the way train_network.py does it
    (prepare_network_kwargs -> create_network -> apply_to -> after_network_created, which re-binds the forwards when
    block compile is on). Returns (unet, network, algo)."""
    import lycoris.kohya as kohya

    torch.manual_seed(seed)
    kw = dict(TOY)
    kw["num_blocks"] = num_blocks
    unet = anima_models.Anima(**kw).to(torch.bfloat16)
    unet.requires_grad_(False)
    unet.train()
    p = PRESETS[preset]
    args = make_args(p["args"], p["dim"], p["alpha"])
    net_kwargs = dict(item.split("=", 1) for item in args.network_args)
    net_kwargs = lycoris_args.fix_lycoris_bool_kwargs(args.network_module, net_kwargs)
    algo = (net_kwargs.get("algo", "lora") or "lora").lower()
    create_kwargs = laf.prepare_network_kwargs(algo, dict(net_kwargs))
    create_kwargs.setdefault("dropout", 0.0)
    laf.set_block_compile(block_compile)
    network = kohya.create_network(1.0, args.network_dim, args.network_alpha, None, None, unet, neuron_dropout=None,
                                   **create_kwargs)
    if hasattr(network, "prepare_network"):
        network.prepare_network(args)
    network.apply_to(None, unet, False, True)
    laf.after_network_created(network, args)
    network.requires_grad_(True)
    network.train()
    return unet, network, algo


def inputs(latent_hw=(8, 8), batch=1, text_len=8, seed=1, t=None):
    g = torch.Generator().manual_seed(seed)
    h, w = latent_hw
    x = torch.randn(batch, 16, 1, h, w, generator=g).to(torch.bfloat16)  # (B, C, T=1, H, W) noisy latents
    if t is None:
        t = torch.rand(batch, generator=g)
    t = torch.as_tensor(t, dtype=torch.float32).reshape(-1).to(torch.bfloat16)  # sigma in [0, 1] as the trainer passes it
    ctx = torch.randn(batch, text_len, TOY["crossattn_emb_channels"], generator=g).to(torch.bfloat16)
    pad = torch.zeros(batch, 1, h, w, dtype=torch.bfloat16)
    return x, t, ctx, pad


def step(unet, network, x, t, ctx, pad, checkpoint=False, backward=True):
    """One micro-step as the trainer runs it: set_timestep(sigma * 1000), bf16 autocast forward, backward."""
    network.zero_grad(set_to_none=True)
    if checkpoint:  # the trainer marks the noisy input when checkpointing (unsloth's Function needs a grad-bearing input)
        x = x.detach().requires_grad_(True)
    laf.set_timestep(network, t.float() * 1000.0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = unet(x, t, ctx, padding_mask=pad)
    if not backward:
        return y.detach().float().clone(), {}
    y.float().square().mean().backward()
    grads = {n: p.grad.detach().clone() for n, p in network.named_parameters() if p.grad is not None}
    assert grads, "the network got no gradients"
    return y.detach().float().clone(), grads


def assert_same(ref, new, what: str, ulp_grads: Sequence[str] = ()):
    """Output and every gradient bit-identical; the gradients whose name contains one of `ulp_grads` may differ by a
    bf16 rounding step (rtol 2^-6: one ulp, two across a binade boundary): under AOTAutograd the backward of plain LoRA's
    adaLN-modulation up weights (a K = 1 outer product: grad_out^T @ mid over the one token of the modulation input) is
    rounded at another site than eager's, so those six gradients differ by a bf16 ulp in a few elements while everything
    else is identical (LoKr, TSM: all exact)."""
    y0, g0 = ref
    y1, g1 = new
    assert torch.equal(y0, y1), f"{what}: output differs, max |d| = {(y0 - y1).abs().max().item():.3e}"
    assert g0.keys() == g1.keys(), f"{what}: gradient sets differ: {sorted(set(g0) ^ set(g1))[:5]}"
    loose = []
    for n in g0:
        if torch.equal(g0[n], g1[n]):
            continue
        assert any(k in n for k in ulp_grads), f"{what}: grad of {n} differs, max |d| = {(g0[n] - g1[n]).abs().max().item():.3e}"
        torch.testing.assert_close(g1[n], g0[n], rtol=2 ** -6, atol=1e-5, msg=lambda m: f"{what}: grad of {n}: {m}")
        loose.append(n)
    return loose


def counting_backend(inner: str = "aot_eager"):
    """A torch.compile backend that counts the graphs dynamo hands it (one per compile / recompile)."""
    wrapped = lookup_backend(inner)
    calls = []

    def backend(gm, example_inputs, **kwargs):
        calls.append(len(example_inputs))
        return wrapped(gm, example_inputs, **kwargs)

    return backend, calls


def compile_toy(unet, backend="aot_eager", bands=TOY_BANDS, skip_call=None, **kw):
    return bc.compile_blocks(unet, bands=bands, backend=backend, dynamic="union" if bands else "static",
                             recompile_limit=64, skip_call=skip_call, isolate_cache=False, **kw)


_BF16_BWD = None


def bf16_backward_or_skip():
    """oneDNN on some CPUs has no bf16 backward for the Block's ops (tests/test_anima_adaln_hoist.py); probed once."""
    global _BF16_BWD
    if _BF16_BWD is None:
        try:
            unet, net, _ = build("lora", num_blocks=1, block_compile=False)
            step(unet, net, *inputs())
            _BF16_BWD = True
        except RuntimeError as error:
            assert "bf16" in str(error), error
            _BF16_BWD = False
    if not _BF16_BWD:
        pytest.skip("no bf16 backward on this CPU")


# ---------------------------------------------------------------------------------------------------------- token bands


def test_tokens_of_pixels_and_band_clustering():
    assert bc.tokens_of_pixels(512, 512) == 1024
    assert bc.tokens_of_pixels(1024, 1024) == 4096
    assert bc.tokens_of_pixels(768, 512) == 1536  # 96 x 64 latents -> 48 x 32 patches
    assert bc.tokens_of_pixels(1032, 520) == 65 * 33  # partial latents and patches round up
    assert bc.tokens_of_pixels(2048, 2048) == 16384
    # the aspect-ratio buckets of a tier (bucket_reso_steps 64, area <= 512^2 / 1024^2) cluster into one band per tier
    # (10% relative gap, as anima_lora): 512 tier 704x320 .. 512x512, 1024 tier 1536x640 .. 1024x1024
    counts = [bc.tokens_of_pixels(w, h) for w, h in ((512, 512), (576, 448), (640, 384), (704, 320), (768, 320),
                                                      (1024, 1024), (1152, 896), (1216, 832), (1344, 768), (1536, 640))]
    assert sorted(set(counts)) == [880, 960, 1008, 1024, 3840, 3952, 4032, 4096]
    bands = bc.cluster_token_bands(counts)
    assert bands == [(880, 1024), (3840, 4096)]
    assert bc.cluster_token_bands([1024, 1152, 1536]) == [(1024, 1024), (1152, 1152), (1536, 1536)]  # gaps > 10%
    assert bc.cluster_token_bands([]) == []
    assert bc.cluster_token_bands([0, -1, 5, 5]) == [(5, 5)]
    assert bc.band_for_seq(bands, 1000) == (880, 1024)
    assert bc.band_for_seq(bands, 4096) == (3840, 4096)
    assert bc.band_for_seq(bands, 2000) is None and bc.band_for_seq(bands, 100) is None and bc.band_for_seq([], 5) is None
    assert bc.union_band(bands) == [(880, 4096)] and bc.union_band([]) == []
    assert bc.widen_band((1024, 4096), 0.0) == (1024, 4096)
    assert bc.widen_band((1024, 4096), 0.25) == (768, 5120)
    assert bc.widen_band((1, 3), 0.9) == (2, 6)  # lo never below 2 (0 / 1 are special sizes for dynamo)


def test_plan_bands_modes():
    counts = [1024, 1536, 4096, 16384]
    assert bc.plan_bands(counts, "union") == [(1024, 16384)]
    assert bc.plan_bands(counts, "bands") == [(1024, 1024), (1536, 1536), (4096, 4096), (16384, 16384)]
    assert bc.plan_bands(counts, "static") == []
    assert bc.plan_bands([], "union") == []
    assert bc.plan_bands(counts, "union", widen=0.5) == [(512, 24576)]
    with pytest.raises(ValueError, match="anima_block_compile_dynamic"):
        bc.plan_bands(counts, "auto")


def test_token_counts_from_dataset_group():
    bm = types.SimpleNamespace(resos=[(512, 512), (768, 512), (1024, 1024)])
    plain = types.SimpleNamespace(bucket_manager=bm, has_resolution_jitter=False)
    wrapped = types.SimpleNamespace(dreambooth_dataset_delegate=plain)  # a FineTuning-style wrapper
    no_buckets = types.SimpleNamespace(bucket_manager=None, resolution=(640, 640))
    jitter = types.SimpleNamespace(bucket_manager=types.SimpleNamespace(resos=[(512, 512)]), has_resolution_jitter=True,
                                   min_bucket_reso=256, max_bucket_reso=1024)
    group = types.SimpleNamespace(datasets=[plain, wrapped, no_buckets])
    assert bc.token_counts_from_dataset_group(group) == [1024, 1536, 1600, 4096]
    assert bc.token_counts_from_dataset_group(jitter) == [256, 1024, 4096]  # the jitter corners (256^2, 1024^2) added
    assert bc.token_counts_from_dataset_group(types.SimpleNamespace(datasets=[])) == []


# -------------------------------------------------------------------------------------------------- fake-5D block loop


def test_fake5d_block_loop_is_bitwise():
    """Anima.forward_mini_train_dit with _block_compile (the Blocks see (B, 1, L, 1, D)) against the plain loop: output
    and every network gradient bit-identical, eager, with the LoKr+DoRA network applied."""
    bf16_backward_or_skip()
    unet, net, _ = build("lokr_dora", block_compile=False)
    x, t, ctx, pad = inputs(latent_hw=(8, 12))
    ref = step(unet, net, x, t, ctx, pad)
    seen = []
    orig = anima_models.Block._forward

    def spy(self, x_in, *a, **k):
        seen.append(tuple(x_in.shape))
        return orig(self, x_in, *a, **k)

    anima_models.Block._forward = spy
    try:
        unet._block_compile = True
        new = step(unet, net, x, t, ctx, pad)
    finally:
        anima_models.Block._forward = orig
        unet._block_compile = False
    assert seen and all(s == (1, 1, 24, 1, TOY["model_channels"]) for s in seen), seen
    assert_same(ref, new, "fake-5D loop vs plain loop")


# ----------------------------------------------------------------------------------------------- tracing / correctness


@pytest.mark.parametrize("preset", ["lokr_dora", "lora", "tsm", "tsm_stage2", "tsm_lokr"])
def test_block_forward_traces_to_one_graph(preset):
    """torch._dynamo.explain on Block._forward with the real inputs of a training micro-step: one graph, no graph break
    (the LyCORIS forwards reached through the re-bound closures, the TSM step data hoisted by set_timestep)."""
    unet, net, _ = build(preset)
    unet._block_compile = True
    cap = {}
    h = unet.blocks[0].register_forward_pre_hook(lambda m, a, k: cap.update(args=a, kwargs=k), with_kwargs=True)
    x, t, ctx, pad = inputs(t=[0.3])
    step(unet, net, x, t, ctx, pad, backward=False)
    h.remove()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        ex = torch._dynamo.explain(unet.blocks[0]._forward)(*cap["args"], **cap["kwargs"])
    reasons = sorted({str(r.reason).splitlines()[0][:120] for r in ex.break_reasons})
    assert ex.graph_break_count == 0, f"{preset}: {ex.graph_break_count} graph breaks: {reasons}"
    assert ex.graph_count == 1, f"{preset}: {ex.graph_count} graphs"


@pytest.mark.parametrize("preset,ckpt", [("lokr_dora", "none"), ("lokr_dora", "unsloth"), ("lokr_dora", "torch"),
                                         ("tsm", "none"), ("tsm", "unsloth"), ("tsm_stage2", "none"),
                                         ("tsm_lokr", "none"), ("lora", "torch")])
def test_compiled_blocks_match_eager(preset, ckpt):
    """compile_blocks (aot_eager): the compiled forward + backward give the eager output and every network gradient
    exactly, without checkpointing and under the unsloth offload / torch.utils.checkpoint recompute (the checkpointed
    callable is the compiled _forward, the marks are re-applied on the recompute pass)."""
    bf16_backward_or_skip()
    unet, net, _ = build(preset)
    if ckpt != "none":
        unet.enable_gradient_checkpointing(unsloth_offload=ckpt == "unsloth")
    x, t, ctx, pad = inputs(t=[0.3])
    ref = step(unet, net, x, t, ctx, pad, checkpoint=ckpt != "none")
    state = compile_toy(unet)
    assert unet._block_compile is True
    new = step(unet, net, x, t, ctx, pad, checkpoint=ckpt != "none")
    assert state.compiled_calls >= len(unet.blocks) and state.eager_calls == 0, (state.compiled_calls, state.eager_calls)
    ulp = ("adaln_modulation",) if preset == "lora" else ()  # see assert_same: plain LoRA's K = 1 up-weight grads
    loose = assert_same(ref, new, f"{preset} {ckpt}: compiled vs eager", ulp_grads=ulp)
    assert all(n.endswith("lora_up.weight") for n in loose), loose
    # a second step (the cached graphs) gives the same results as the first compiled one
    again = step(unet, net, x, t, ctx, pad, checkpoint=ckpt != "none")
    assert_same(new, again, f"{preset} {ckpt}: second compiled step vs first")


def test_tsm_timesteps_of_one_expert_interval_share_the_graphs():
    """TSM stage 1, 8 core experts over 1000 timesteps (one every 125): timesteps 260 / 300 / 370 pick the same experts
    and must not recompile (the graph guards on the hoisted expert tuple, not on the integer timestep); timestep 900
    picks other experts: at most one new graph per Block dtype regime (the first step's count)."""
    unet, net, _ = build("tsm")
    backend, calls = counting_backend("aot_eager")
    compile_toy(unet, backend=backend)
    x, t, ctx, pad = inputs(t=[0.3])
    step(unet, net, x, t, ctx, pad, backward=False)
    first = len(calls)
    assert first >= 1
    for sigma in (0.26, 0.37, 0.3):
        step(unet, net, x, torch.tensor([sigma]).to(torch.bfloat16), ctx, pad, backward=False)
    assert len(calls) == first, f"recompiled inside one expert interval: {len(calls) - first} new graphs"
    step(unet, net, x, torch.tensor([0.9]).to(torch.bfloat16), ctx, pad, backward=False)
    assert first < len(calls) <= 2 * first, (first, len(calls))
    # and the module-level hoist is what the graphs read
    tsm = [m for m in laf._all_loras(net) if type(m).__name__ == "TSMModule"]
    assert tsm and all("_hx_experts" in m.__dict__ for m in tsm)
    assert all(m.__dict__["_hx_experts"] == laf._tsm_experts(m, 900) for m in tsm)


def test_hoisted_tsm_step_data_and_mixed_chunk_clears_it():
    """set_timestep under block compile: stage 1 stores the expert tuple, stage 2 also the embedding row index tensor,
    tsm-lokr its expert indices; a mixed-timestep chunk removes them and mixed_timestep_chunk() reports it. Outside
    a compiled graph _get_timestep still returns the stock value."""
    for preset in ("tsm", "tsm_stage2", "tsm_lokr"):
        unet, net, _ = build(preset)
        mods = [m for m in laf._all_loras(net) if type(m).__name__ in laf._TSM_KINDS]
        assert mods, preset
        laf.set_timestep(net, torch.tensor([300.0]))
        assert laf.mixed_timestep_chunk() is False
        for m in mods:
            if type(m).__name__ == "TSMModule":
                assert m.__dict__["_hx_experts"] == laf._tsm_experts(m, 300)
                assert laf._tsm_get_timestep(m) == 300  # eager: the stock global
                if preset == "tsm_stage2":
                    assert m.__dict__["_hx_tidx"].tolist() == [299] and m.__dict__["_hx_tidx"].dtype == torch.long
                else:
                    assert "_hx_tidx" not in m.__dict__
            else:
                assert m.__dict__["_hx_idx"] == m._expert_indices(300)
        if preset == "tsm_stage2":
            tidx = mods[0].__dict__["_hx_tidx"]
            laf.set_timestep(net, torch.tensor([500.0]))
            assert mods[0].__dict__["_hx_tidx"] is tidx and tidx.tolist() == [499]  # rewritten in place
        laf.set_timestep(net, torch.tensor([300.0, 900.0]))
        assert laf.mixed_timestep_chunk() is True
        for m in mods:
            assert not any(k in m.__dict__ for k in ("_hx_experts", "_hx_tidx", "_hx_idx")), preset
        laf.set_timestep(net, torch.tensor([300.0, 300.0]))  # two rows, one timestep: hoisted again
        assert laf.mixed_timestep_chunk() is False
        assert all(any(k in m.__dict__ for k in ("_hx_experts", "_hx_idx")) for m in mods)


def test_mixed_timestep_chunk_runs_eager_and_single_timestep_batch_compiles():
    """TSM, batch 2: rows with different timesteps run the eager Block._forward through skip_call (the masked path's
    expert union changes per chunk) and give the eager results; two rows with one timestep take the compiled path."""
    bf16_backward_or_skip()
    unet, net, _ = build("tsm")
    x, t, ctx, pad = inputs(batch=2, t=[0.3, 0.9])
    ref_mixed = step(unet, net, x, t, ctx, pad)
    x1, t1, ctx1, pad1 = inputs(batch=2, t=[0.3, 0.3])
    ref_same = step(unet, net, x1, t1, ctx1, pad1)
    state = compile_toy(unet, skip_call=laf.mixed_timestep_chunk)
    new_mixed = step(unet, net, x, t, ctx, pad)
    assert state.eager_calls == len(unet.blocks) and state.compiled_calls == 0, (state.eager_calls, state.compiled_calls)
    assert_same(ref_mixed, new_mixed, "mixed-timestep chunk (eager under block compile)")
    new_same = step(unet, net, x1, t1, ctx1, pad1)
    assert state.compiled_calls >= len(unet.blocks), state.compiled_calls
    assert_same(ref_same, new_same, "batch 2, one timestep (compiled)")


def test_resolution_sweep_inside_the_union_band_shares_the_graphs():
    """Latent grids 8x8, 8x12, 12x12, 16x16 (16 .. 64 tokens) with the union band (16, 64): the first grid compiles, the
    others run on its graphs (soft marks on the token axis of x and the RoPE tables under attn_mode torch)."""
    unet, net, _ = build("lokr_dora")
    backend, calls = counting_backend("aot_eager")
    state = compile_toy(unet, backend=backend)
    assert state.bands == [(16, 64)]
    x, t, ctx, pad = inputs(latent_hw=(8, 8), t=[0.3])
    step(unet, net, x, t, ctx, pad, backward=False)
    first = len(calls)
    assert first >= 1
    for hw in ((8, 12), (12, 12), (16, 16)):
        x, t, ctx, pad = inputs(latent_hw=hw, t=[0.3])
        step(unet, net, x, t, ctx, pad, backward=False)
        assert len(calls) == first, f"{hw}: recompiled ({len(calls) - first} new graphs)"
    assert not bc._warned_oob
    # a grid outside every band runs its own static specialization and is reported once
    x, t, ctx, pad = inputs(latent_hw=(4, 4), t=[0.3])
    step(unet, net, x, t, ctx, pad, backward=False)
    assert bc._warned_oob == {4}


def test_static_plan_compiles_per_token_count():
    """dynamic=static (no marks): every distinct token count compiles its own graphs."""
    unet, net, _ = build("lora")
    backend, calls = counting_backend("eager")
    bc.compile_blocks(unet, bands=[], backend=backend, dynamic="static", recompile_limit=64, isolate_cache=False)
    x, t, ctx, pad = inputs(latent_hw=(8, 8), t=[0.3])
    step(unet, net, x, t, ctx, pad, backward=False)
    first = len(calls)
    x, t, ctx, pad = inputs(latent_hw=(8, 12), t=[0.3])
    step(unet, net, x, t, ctx, pad, backward=False)
    assert len(calls) == 2 * first, (first, len(calls))


def test_pause_eval_and_restore_run_the_eager_blocks():
    """pause() (sample images) and eval mode (validation) take the eager _forward; restore_blocks removes the wrappers
    and the fake-5D flag."""
    unet, net, _ = build("lora")
    state = compile_toy(unet, backend="eager")
    x, t, ctx, pad = inputs(t=[0.3])
    with bc.pause():
        assert not bc.is_active()
        with bc.pause():  # re-entrant
            step(unet, net, x, t, ctx, pad, backward=False)
        assert not bc.is_active()
    assert bc.is_active()
    assert state.eager_calls == len(unet.blocks) and state.compiled_calls == 0
    unet.eval()
    with torch.no_grad():
        step(unet, net, x, t, ctx, pad, backward=False)
    unet.train()
    assert state.eager_calls == 2 * len(unet.blocks) and state.compiled_calls == 0
    step(unet, net, x, t, ctx, pad, backward=False)
    assert state.compiled_calls == len(unet.blocks)
    assert all("_forward" in b.__dict__ for b in unet.blocks)
    bc.restore_blocks(unet)
    assert unet._block_compile is False and unet._block_compile_state is None
    assert all("_forward" not in b.__dict__ and "_forward_hoisted" not in b.__dict__ for b in unet.blocks)
    step(unet, net, x, t, ctx, pad, backward=False)
    assert state.compiled_calls == len(unet.blocks)  # the wrappers are gone


def test_rebind_forwards_keeps_the_lokr_sample_cache_targets():
    """rebind_forwards: every adapted Linear's forward becomes a closure (dynamo-traceable) carrying __self__, so
    lycoris_fast._cache_targets (the merged-weight cache of sample images) finds the same LoKr modules; idempotent;
    a foreign forward patch is left alone."""
    unet, net, _ = build("lokr_dora", block_compile=False)
    before = lycoris_fast._cache_targets(unet)
    assert before, "no LoKr cache targets on the toy"
    n = laf.rebind_forwards(net)
    assert n == len(laf._all_loras(net)) == len(before)
    after = lycoris_fast._cache_targets(unet)
    assert [(m, lokr) for m, _, lokr in after] == [(m, lokr) for m, _, lokr in before]
    assert all(not hasattr(f, "__func__") and f.__wrapped_lora__ is lokr for _, f, lokr in after)
    assert laf.rebind_forwards(net) == 0
    lin, fwd, _ = after[0]
    lin.forward = lambda x_: x_  # someone else's patch (a cached merge): not touched
    assert laf.rebind_forwards(net) == 0 and lin.forward(1) == 1
    lin.forward = fwd


def test_attention_traces_without_a_graph_break():
    """library/attention.py: the split-attention lists are built with loops, so q / k / v are plain locals and the
    `del q, k, v` of the SDPA path is DELETE_FAST: attention() traces whole (the comprehensions made them closure cells
    for the whole function, so the del was DELETE_DEREF - a graph break under dynamo 2.7 even without --split_attn).
    With --split_attn the per-row slice by the device seqlens tensor is data-dependent (one graph break per row, a known
    limitation, documented); the results stay those of eager."""
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 12, 2, 16, generator=g) for _ in range(3))
    params = attention.AttentionParams.create_attention_params("torch", False)
    ex = torch._dynamo.explain(attention.attention)([q, k, v], attn_params=params)
    reasons = sorted({str(r.reason).splitlines()[0][:120] for r in ex.break_reasons})
    assert ex.graph_break_count == 0 and ex.graph_count == 1, f"{ex.graph_break_count} breaks: {reasons}"
    split = attention.AttentionParams.create_attention_params("torch", True)
    split.seqlens = torch.tensor([12, 9])
    split.max_seqlen = 12
    for p in (params, split):
        ref = attention.attention([q.clone(), k.clone(), v.clone()], attn_params=p)
        out = torch.compile(attention.attention, backend="eager")([q.clone(), k.clone(), v.clone()], attn_params=p)
        assert torch.equal(ref, out)


# ----------------------------------------------------------------------------------------------------- config helpers


def test_raise_recompile_limit_never_lowers(monkeypatch):
    cfg = torch._dynamo.config
    monkeypatch.setattr(cfg, "recompile_limit", 8)
    monkeypatch.setattr(cfg, "accumulated_recompile_limit", 256)
    assert bc.raise_recompile_limit(128) == 128 and cfg.recompile_limit == 128 and cfg.accumulated_recompile_limit == 512
    assert bc.raise_recompile_limit(16) == 128 and cfg.recompile_limit == 128


def test_isolate_inductor_cache_is_per_signature_and_does_not_nest(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    sig_a = bc.compile_signature([(1024, 16384)], "inductor", None, "union")
    sig_b = bc.compile_signature([(1024, 4096)], "inductor", None, "union")
    a = bc.isolate_inductor_cache(sig_a)
    assert os.path.dirname(a) == str(tmp_path) and os.path.basename(a).startswith("anima-blocks-")
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == a
    b = bc.isolate_inductor_cache(sig_b)
    assert os.path.dirname(b) == str(tmp_path) and b != a  # a second call does not nest under the first
    assert bc.isolate_inductor_cache(sig_a) == a  # deterministic
    assert "torch=" in sig_a and "bands=[(1024, 16384)]" in sig_a


def test_apply_activation_memory_budget(monkeypatch):
    import torch._functorch.config as fc

    monkeypatch.setattr(fc, "activation_memory_budget", 1.0)
    assert bc.apply_activation_memory_budget(1.0, False) is False and fc.activation_memory_budget == 1.0
    assert bc.apply_activation_memory_budget(0.5, True) is False and fc.activation_memory_budget == 1.0  # not under GC
    assert bc.apply_activation_memory_budget(0.5, False) is True and fc.activation_memory_budget == 0.5


def test_compile_blocks_rejects_an_unknown_dynamic_mode():
    unet, _, _ = build("lora")
    with pytest.raises(ValueError, match="dynamic"):
        bc.compile_blocks(unet, bands=[], backend="eager", dynamic="auto", isolate_cache=False)
    assert unet._block_compile is False


# ------------------------------------------------------------------------------------------------------- trainer glue


def _trainer_args(**over):
    base = dict(anima_block_compile=True, anima_block_compile_backend="eager", anima_block_compile_mode=None,
                anima_block_compile_dynamic="union", anima_block_compile_band_widen=0.0,
                anima_block_compile_recompile_limit=128, anima_activation_memory_budget=1.0,
                gradient_checkpointing=False, use_ramtorch=False, use_ramtorch_network=False, blocks_to_swap=None,
                torch_compile=False, lycoris_stock=False, network_module="lycoris.kohya")
    base.update(over)
    return argparse.Namespace(**base)


def test_validate_block_compile_args():
    from anima_train_network import validate_block_compile_args

    validate_block_compile_args(_trainer_args())
    validate_block_compile_args(_trainer_args(anima_block_compile=False, blocks_to_swap=10, use_ramtorch=True))  # off: anything
    validate_block_compile_args(_trainer_args(lycoris_stock=True, network_module="networks.lora"))  # stock only matters for LyCORIS
    for bad, match in ((dict(use_ramtorch=True), "RamTorch"), (dict(use_ramtorch_network=True), "RamTorch"),
                       (dict(blocks_to_swap=4), "blocks_to_swap"), (dict(torch_compile=True), "torch_compile"),
                       (dict(lycoris_stock=True), "lycoris_stock"), (dict(anima_block_compile_dynamic="auto"), "dynamic"),
                       (dict(anima_activation_memory_budget=0.0), "memory_budget"),
                       (dict(anima_activation_memory_budget=1.5), "memory_budget"),
                       (dict(anima_block_compile_recompile_limit=4), "recompile_limit")):
        with pytest.raises(ValueError, match=match):
            validate_block_compile_args(_trainer_args(**bad))


def test_parser_defaults_and_runtime_switches():
    """The options exist with block compile off by default; prepare_anima_runtime with --anima_block_compile turns the
    LyCORIS re-bind on and the nested compiles (elementwise chains, LoKr rebuild) off."""
    import anima_train_network as atn
    import inspect

    assert "validate_block_compile_args(args)" in inspect.getsource(atn.AnimaNetworkTrainer.assert_extra_args)
    parser = atn.setup_parser()
    ns = parser.parse_args([])
    assert ns.anima_block_compile is False and ns.anima_block_compile_backend == "inductor"
    assert ns.anima_block_compile_mode is None and ns.anima_block_compile_dynamic == "union"
    assert ns.anima_block_compile_band_widen == 0.0 and ns.anima_block_compile_recompile_limit == 128
    assert ns.anima_activation_memory_budget == 1.0
    ns = parser.parse_args(["--anima_block_compile", "--anima_block_compile_backend", "aot_eager",
                            "--anima_block_compile_dynamic", "bands", "--anima_activation_memory_budget", "0.7",
                            "--timestep_sampling", "sigmoid"])
    chains_before, rebuild_before = anima_models._CHAINS.enabled, lycoris_fast._REBUILD.enabled
    try:
        atn.prepare_anima_runtime(ns)
        assert laf.BLOCK_COMPILE is True
        assert anima_models._CHAINS.enabled is False and lycoris_fast._REBUILD.enabled is False
        laf.set_block_compile(False)
        off = parser.parse_args(["--timestep_sampling", "sigmoid"])
        atn.prepare_anima_runtime(off)
        assert laf.BLOCK_COMPILE is False
        assert anima_models._CHAINS.enabled is True and lycoris_fast._REBUILD.enabled is True
    finally:
        anima_models._CHAINS.enable(chains_before)
        lycoris_fast.enable_compiled_rebuild(rebuild_before)


def test_after_network_loaded_compiles_from_the_bucket_token_counts():
    """AnimaNetworkTrainer.after_network_loaded: the bands come from the training buckets (512^2, 768x512, 1024^2 ->
    union (1024, 4096)), skip_call is the fixed paths' mixed-timestep test, the Blocks run compiled afterwards; with the
    option off nothing happens."""
    import anima_train_network as atn

    unet, net, _ = build("lora")
    trainer = atn.AnimaNetworkTrainer()
    trainer._lycoris_fix_on = True
    trainer._lycoris_fixes = laf
    bm = types.SimpleNamespace(resos=[(512, 512), (768, 512), (1024, 1024)])
    trainer._train_dataset_group = types.SimpleNamespace(datasets=[types.SimpleNamespace(bucket_manager=bm)])
    printed = []
    accel = types.SimpleNamespace(unwrap_model=lambda m: m, print=lambda *a, **k: printed.append(" ".join(map(str, a))))
    trainer.after_network_loaded(_trainer_args(anima_block_compile=False), accel, net, [], unet)
    assert unet._block_compile is False and not printed
    trainer.after_network_loaded(_trainer_args(), accel, net, [], unet)
    state = unet._block_compile_state
    assert unet._block_compile is True and state.bands == [(1024, 4096)]
    assert printed and "1024, 1536, 4096" in printed[0]
    x, t, ctx, pad = inputs(t=[0.3])
    step(unet, net, x, t, ctx, pad, backward=False)
    assert state.compiled_calls == len(unet.blocks)
    bc.restore_blocks(unet)
