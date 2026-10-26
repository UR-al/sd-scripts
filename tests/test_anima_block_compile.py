"""CPU tests for --anima_block_compile (library/anima_block_compile.py): per-Block torch.compile of the Anima DiT with the
LyCORIS fixed paths inside the graphs.

Checked on torch 2.7.1 CPU with the aot_eager / eager backends (the same dynamo tracing and the same AOTAutograd joint
graph as inductor, without codegen) on a 2-Block toy Anima and networks built the way train_network.py builds them
(lycoris.kohya + library/lycoris_algo_fixes + library/lycoris_fast, preset anima-exp):
* token bands: pixels -> tokens, tier clustering, union / bands / static plans, the dataset-group reader;
* the fake-5D Block loop (Anima.forward_mini_train_dit with _block_compile) is bit-identical to the plain loop;
* every Block._forward traces to ONE graph with no graph break for lokr+DoRA, lora, tsm stage 1 / 2, tsm-lokr, loha
  (+DoRA; both of its fixed paths, the dense Hadamard delta and the factored (x KA^T) KB^T), full, tlora, abba (+DoRA),
  diag-oft and boft (its token rows on the blockwise rotation path, the adaLN rows on the few-rows stage path);
* compiled forward + backward reproduce the eager outputs and every network gradient exactly (zero-init factors seeded:
  seed_zero_factors, perturb_tlora, perturb_tsm), with and without gradient checkpointing (unsloth offload and
  torch.utils.checkpoint); the compiled call hands the caller's autocast state to the graph, which re-enters it
  (compile_blocks), so the backward is traced without the ambient autocast like eager's - before, AOTAutograd traced it
  under the trainer's autocast and every fp32 matmul of the backward ran in bf16 (plain LoRA's six adaLN-up lora_up
  gradients off by a bf16 ulp or two, LoHa's hada_w2_a and BOFT's oft_blocks on the attention / MLP layers too). The
  pinned exception is in _FP32_GRADS (assert_same);
* the Blocks share one compiled entry (2 graphs for 4 Blocks, as for 2); a training forward of an eval-mode DiT (the
  trainer without gradient checkpointing) runs compiled, eval under no_grad (validation) eager;
* TSM stage 2 (TSMModule, the tsm_expert_fn path, TSM-LoKr) reads the step's embedding row inside the graphs;
* dylora is refused by validate_block_compile_args: it draws a random rank per forward (tensor shapes from a Python
  random number) and reads torch._C._current_graph_task_id() to reuse it in a checkpoint recompute - untraceable by
  design, and a graph per rank otherwise (kohya's networks.dylora too: it flips requires_grad per drawn rank);
* TSM: timesteps of one expert interval share the graphs (no recompile), another interval adds at most one graph per
  Block dtype regime; a mixed-timestep chunk (batch > 1 with different timesteps) runs eager through skip_call;
* T-LoRA: every timestep shares the graphs (the rank mask is hoisted onto the modules and rewritten in place per step;
  clear_timestep refills it with ones in place, no recompile); LoHa / ABBA: the factored / dense path choice is fixed
  inside the graphs (plan_block_compile_rows: the rule at the planned row count; without a plan factored) - a strict
  dynamic band across the roofline rule's flip compiles to one graph (it was a ConstraintViolationError);
* --anima_adaln_hoist with gradient checkpointing reaches the compiled _forward through the eager _forward_hoisted;
* a resolution sweep inside the union band runs on the first graphs (soft marks under attn_mode torch), per preset;
* pause() / eval run the eager Blocks; restore_blocks puts the class methods back;
* rebind_forwards keeps lycoris_fast's merged-weight cache targets; library/attention.py's split path traces without a
  graph break; the trainer glue: validate_block_compile_args, after_network_loaded, prepare_anima_runtime, the parser;
* a cold network (the trainer's first forward is the first traced call) traces whole once prime_module_caches has filled
  the lazy per-module caches - without it LoKr+DoRA breaks on a .item() and keeps the fragments (loha / full trace whole
  cold: the fork's default-free _HadaWeight Function, the 0-d alpha/dim operand as a graph constant, {**a, **b}).

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
    "tsm_stage2_fn": dict(args=["algo=tsm", "preset=anima-exp", "tsm_n_scales=8,1", "tsm_stage=2", "tsm_expert_fn=True"],
                          dim=8, alpha=8),  # the opt-in _TSMExpertLinear path
    "tsm_lokr_stage2": dict(args=["algo=tsm-lokr", "preset=anima-exp", "tsm_n_scales=8,1", "tsm_stage=2", "factor=4",
                                  "full_matrix=True", "decompose_both=False", "dora_wd=True", "wd_on_output=True",
                                  "tsm_offload_states=False"], dim=10000, alpha=10000),
    "loha": dict(args=["algo=loha", "preset=anima-exp"], dim=8, alpha=8),
    "loha_dora": dict(args=["algo=loha", "preset=anima-exp", "dora_wd=True", "wd_on_output=True"], dim=8, alpha=8),
    "full": dict(args=["algo=full", "preset=anima-exp"], dim=8, alpha=8),
    "tlora": dict(args=["algo=tlora", "preset=anima-exp"], dim=8, alpha=8),  # rank 8, mask min rank 4 (ceil(dim / 2))
    "abba": dict(args=["algo=abba", "preset=anima-exp"], dim=8, alpha=8),  # r1 = r2 = 4: delta rank 16
    "abba_dora": dict(args=["algo=abba", "preset=anima-exp", "dora_wd=True", "wd_on_output=True"], dim=8, alpha=8),
    "diag_oft": dict(args=["algo=diag-oft", "preset=anima-exp"], dim=4, alpha=4),
    # boft: network_dim >= 6. LyCORIS' butterfly factorization (power2factorization) takes the largest even block size
    # b <= network_dim with out_dim / b a power of two; the adaLN-LoRA up Linears are 3 x model_channels wide (192 here,
    # 6144 on Anima 2.9B), so their b is 3 x 2^k, at least 6 (dim 6: b = 6 there, 4 on the 64-wide Linears; dim 4
    # cannot build). oft_max_block 8: the toy's 16 .. 64 token rows take the blockwise rotation path (_BlockResidualMM,
    # chunked and permuted stages) that production's >= 880 token rows take under the default _OFT_MAX_BLOCK 256; the
    # adaLN rows (one per sample) and here the toy's 8 text rows take the few-rows stage path (production's text is
    # padded to 512 rows: blockwise).
    "boft": dict(args=["algo=boft", "preset=anima-exp"], dim=6, alpha=6, oft_max_block=8),
    "boft_rows": dict(args=["algo=boft", "preset=anima-exp"], dim=6, alpha=6),  # default 256: token rows on the few-rows path
    "dylora": dict(args=["algo=dylora", "preset=anima-exp", "block_size=2"], dim=8, alpha=8),  # refused, see test_dylora_*
}
# what a COLD trace (before prime_module_caches) breaks on: LoKr+DoRA's float(scalar) .item() (lycoris_fast); every other
# preset traces whole cold (loha: the stock HadaWeight's tensor default argument broke every .apply, the fork's _HadaWeight
# has none, and its 0-d alpha/dim operand is a graph constant under compile, _loha_gamma)
COLD_BREAK = {"lokr_dora": "item"}
PRIMED = {"lokr_dora", "lora"}  # presets whose every module has a lazily filled cache to prime
TOY_TOKENS = [16, 24, 36, 64]  # latent grids 8x8, 8x12, 12x12, 16x16 at patch 2
TOY_BANDS = bc.plan_bands(TOY_TOKENS, "union")  # [(16, 64)]
_OFT_MAX_BLOCK_DEFAULT = laf._OFT_MAX_BLOCK  # build() lowers it for a preset's oft_max_block; _env puts it back


@pytest.fixture(autouse=True)
def _env():
    # production order (anima_train_network.prepare_anima_runtime, then train()): the fast LoKr+DoRA forward first, the
    # fixed paths on top of it. Reversed in a fresh process, install_fast_lokr_forward would record the fixed-path LoKr
    # forward as "stock" and later test files (test_lycoris_algo_fixes, test_moss_pipeline) would find the fast forward
    # gone.
    lycoris_fast.install_fast_lokr_forward()
    laf.apply_module_patches()
    torch._dynamo.reset()
    bc._warned_oob.clear()
    yield
    laf.set_block_compile(False)
    laf._OFT_MAX_BLOCK = _OFT_MAX_BLOCK_DEFAULT
    laf._TSM_EXPERT_FN = False  # tsm_stage2_fn's network_args set it
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
    laf._OFT_MAX_BLOCK = p.get("oft_max_block", _OFT_MAX_BLOCK_DEFAULT)
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


_ZERO_INIT = ("lora_up2.weight", "hada_w2_a", "oft_blocks")  # ABBA (abba.py:255), LoHa, diag-OFT / BOFT (R = I)


def seed_zero_factors(network, seed: int = 3, scale: float = 0.05) -> int:
    """The factor each of these algos starts at zero, so that its delta is zero at init - ABBA's second up factor,
    LoHa's hada_w2_a, (diag-)OFT / BOFT's oft_blocks - seeded small: with it zero every path or delta form a graph can
    compute returns the base layer's output and the other factors get no gradient, so a compiled-vs-eager comparison
    would only see that one factor's gradient. Returns the number of modules seeded."""
    g = torch.Generator().manual_seed(seed)
    n = 0
    with torch.no_grad():
        for m in laf._all_loras(network):
            hit = False
            for name, p in m.named_parameters():
                if name in _ZERO_INIT and not bool(p.any()):
                    p.copy_(torch.randn(p.shape, generator=g).to(p.dtype) * scale)
                    hit = True
            n += hit
    return n


def perturb_tsm(network, seed: int = 7, scale: float = 0.05) -> int:
    """TSM's stage-2 router, timestep embedding and context experts' up factors start at zero (tsm.py:228-230, 267;
    tsm_lokr.py:307, 310, 382-384): the gated term is zero whichever embedding row a graph reads. Every float parameter of
    the TSM modules perturbed, so outputs and gradients depend on the step's experts and embedding row. Returns the
    number of tensors perturbed."""
    g = torch.Generator().manual_seed(seed)
    n = 0
    with torch.no_grad():
        for m in laf._all_loras(network):
            if type(m).__name__ in laf._TSM_KINDS:
                for p in m.parameters():
                    if p.is_floating_point():
                        p.add_(torch.randn(p.shape, generator=g).to(p.dtype) * scale)
                        n += 1
    return n


def perturb_tlora(network, seed: int = 5) -> int:
    """T-LoRA starts as a zero delta for EVERY mask (q, p, lambda equal their base copies: p diag(lambda m) q -
    p_b diag(lambda_b m) q_b = 0), so an output or gradient check cannot see which mask a graph read: move the trained
    factors off their bases. Returns the number of modules perturbed."""
    g = torch.Generator().manual_seed(seed)
    n = 0
    with torch.no_grad():
        for m in laf._all_loras(network):
            if type(m).__name__ != "TLoraModule":
                continue
            for w in (m.q_layer.weight, m.p_layer.weight, m.lambda_layer):
                w.add_(torch.randn(w.shape, generator=g).to(w.dtype) * 0.2 * w.abs().mean().clamp_min(1e-3))
            n += 1
    return n


def assert_same(ref, new, what: str, ulp_grads: Sequence[str] = (), fp32_grads: Sequence[str] = ()):
    """Output and every gradient bit-identical; the gradients whose name contains one of `ulp_grads` may differ by a
    bf16 rounding step (rtol 2^-6: one ulp, two across a binade boundary), those containing one of `fp32_grads` by fp32
    rounding (rtol 1e-5, atol 1e-7) - for a rounding site that AOTAutograd provably moves (_ULP_GRADS / _FP32_GRADS). The
    "plain LoRA adaLN up-weight ulps" the bf16 allowance once covered were the traced backward running the fp32 adaLN
    region's matmuls under the trainer's autocast (bf16), which compile_blocks now keeps out of the graph (the state is
    handed to the graph as arguments). Returns the names that used an allowance."""
    y0, g0 = ref
    y1, g1 = new
    assert torch.equal(y0, y1), f"{what}: output differs, max |d| = {(y0 - y1).abs().max().item():.3e}"
    assert g0.keys() == g1.keys(), f"{what}: gradient sets differ: {sorted(set(g0) ^ set(g1))[:5]}"
    loose = []
    for n in g0:
        if torch.equal(g0[n], g1[n]):
            continue
        d = (g0[n] - g1[n]).abs().max().item()
        if any(k in n for k in fp32_grads):
            torch.testing.assert_close(g1[n], g0[n], rtol=1e-5, atol=1e-7, msg=lambda m: f"{what}: grad of {n}: {m}")
        else:
            assert any(k in n for k in ulp_grads), f"{what}: grad of {n} differs, max |d| = {d:.3e}"
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


def compile_toy(unet, net=None, backend="aot_eager", bands=TOY_BANDS, skip_call=None, **kw):
    """compile_blocks as the trainer calls it (after_network_loaded): the network's lazy per-module caches primed and the
    LoHa / ABBA path rule's row count planned (the band's largest token count) first."""
    if net is not None:
        laf.prime_module_caches(net)
        laf.plan_block_compile_rows(net, unet, max((hi for _, hi in bands), default=0))
    return bc.compile_blocks(unet, bands=bands, backend=backend, dynamic="union" if bands else "static",
                             recompile_limit=64, skip_call=skip_call, isolate_cache=False, **kw)


def capture_block_args(unet, net, x, t, ctx, pad):
    """The (args, kwargs) Block 0 receives in a training forward, captured WITHOUT running any Block (a hook raises), so
    the network's lazy caches stay cold."""

    class Stop(Exception):
        pass

    cap = {}

    def stop(m, a, k):
        cap.update(args=a, kwargs=k)
        raise Stop()

    h = unet.blocks[0].register_forward_pre_hook(stop, with_kwargs=True)
    laf.set_timestep(net, t.float() * 1000.0)
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            unet(x, t, ctx, padding_mask=pad)
    except Stop:
        pass
    finally:
        h.remove()
    return cap


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


@pytest.mark.parametrize("preset", ["lokr_dora", "lora", "tsm", "tsm_stage2", "tsm_lokr", "loha", "loha_dora",
                                    "full", "tlora", "abba", "abba_dora", "diag_oft", "boft"])
def test_block_forward_traces_to_one_graph(preset):
    """torch._dynamo.explain on Block._forward with the real inputs of a training micro-step: one graph, no graph break
    (the LyCORIS forwards reached through the re-bound closures, the TSM step data and the T-LoRA mask hoisted by
    set_timestep, T-LoRA's base matmul read through org_module[0] instead of the bound org_forward)."""
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


@pytest.mark.parametrize("preset", ["lokr_dora", "lora", "tsm", "tsm_stage2", "tsm_lokr", "loha", "loha_dora",
                                    "full", "tlora", "abba", "abba_dora", "diag_oft", "boft"])
def test_cold_start_traces_whole_after_priming(preset):
    """The trainer compiles right after the weights are loaded, so the first traced call is the network's FIRST forward:
    its lazily filled per-module caches are cold. Traced cold, LoKr+DoRA breaks the graph on lycoris_fast's float(scalar)
    (.item(), _compute_static_eligible) and dynamo keeps the fragments (21 graphs per Block call, re-specialised per
    resolution); the other presets trace whole cold (COLD_BREAK; diag-oft / boft have no lazy cache, their rotation
    Function takes the out=-free branch while tracing). prime_module_caches (called by after_network_loaded before
    compile_blocks) fills the caches eagerly: every preset then traces to one graph with no break from a cold network."""
    unet, net, _ = build(preset)
    unet._block_compile = True
    x, t, ctx, pad = inputs(t=[0.3])
    cap = capture_block_args(unet, net, x, t, ctx, pad)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cold = torch._dynamo.explain(unet.blocks[0]._forward)(*cap["args"], **cap["kwargs"])
    want = COLD_BREAK.get(preset)
    if want:
        assert cold.graph_break_count > 0 and all(want in str(r.reason) for r in cold.break_reasons), cold.break_reasons
    else:
        assert cold.graph_break_count == 0, [str(r.reason)[:100] for r in cold.break_reasons]
    torch._dynamo.reset()
    unet, net, _ = build(preset)  # fresh again: the cold explain above filled the caches while breaking
    unet._block_compile = True
    cap = capture_block_args(unet, net, x, t, ctx, pad)
    n = laf.prime_module_caches(net)
    assert n == (len(laf._all_loras(net)) if preset in PRIMED else 0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        primed = torch._dynamo.explain(unet.blocks[0]._forward)(*cap["args"], **cap["kwargs"])
    assert primed.graph_break_count == 0 and primed.graph_count == 1, [str(r.reason)[:100] for r in primed.break_reasons]


# gradients allowed to differ from eager (assert_same), per preset. By a bf16 step: none. Plain LoRA's adaLN up weights
# needed it while the backward was traced under the trainer's autocast (compile_blocks now hands the state to the graph);
# ABBA+DoRA's factor / scale gradients while the graph took the factored path and eager the dense one (the path rule now
# runs at the planned row count, eager's choice on the toy). By fp32 rounding: boft_rows' oft_blocks (token rows on the
# few-rows stage path, a toy-only regime: the AOT backward sums the einsum over the rows in another order than eager's
# bmm; 5.8e-8 on a 4.2e-2 gradient measured; production's >= 880 token rows take the blockwise path, exact as `boft`)
_ULP_GRADS: dict = {}
_FP32_GRADS = {"boft_rows": ("oft_blocks",)}


@pytest.mark.parametrize("preset,ckpt", [("lokr_dora", "none"), ("lokr_dora", "unsloth"), ("lokr_dora", "torch"),
                                         ("tsm", "none"), ("tsm", "unsloth"), ("tsm_stage2", "none"),
                                         ("tsm_lokr", "none"), ("lora", "torch"),
                                         ("loha", "none"), ("loha", "unsloth"), ("loha_dora", "none"),
                                         ("loha_dora", "unsloth"), ("full", "none"), ("full", "unsloth"),
                                         ("tlora", "none"), ("tlora", "unsloth"),
                                         ("abba", "none"), ("abba", "unsloth"), ("abba_dora", "none"),
                                         ("abba_dora", "unsloth"), ("diag_oft", "none"), ("diag_oft", "unsloth"),
                                         ("boft", "none"), ("boft", "unsloth"), ("boft", "torch"), ("boft_rows", "none")])
def test_compiled_blocks_match_eager(preset, ckpt):
    """compile_blocks (aot_eager): the compiled forward + backward give the eager output and every network gradient
    exactly, without checkpointing and under the unsloth offload / torch.utils.checkpoint recompute (the checkpointed
    callable is the compiled _forward, the marks are re-applied on the recompute pass). ABBA, LoHa and the OFTs run with
    their zero-init factor seeded (seed_zero_factors), so the delta and every factor's gradient are compared. Inside the graphs LoHa's
    and ABBA's path rule runs at the planned row count (compile_toy: the band's 64 tokens), which on the toy is the path
    eager picks at every grid of the band (test_planned_rows_fix_the_path_choice_under_block_compile covers a band the
    rule flips in). boft_rows (token rows on the few-rows stage path): oft_blocks within fp32 rounding (_FP32_GRADS)."""
    bf16_backward_or_skip()
    unet, net, _ = build(preset)
    if preset.startswith(("abba", "loha", "diag_oft", "boft")):
        assert seed_zero_factors(net) == len(laf._all_loras(net))
    if ckpt != "none":
        unet.enable_gradient_checkpointing(unsloth_offload=ckpt == "unsloth")
    x, t, ctx, pad = inputs(t=[0.3])
    ref = step(unet, net, x, t, ctx, pad, checkpoint=ckpt != "none")
    state = compile_toy(unet, net)
    assert unet._block_compile is True
    new = step(unet, net, x, t, ctx, pad, checkpoint=ckpt != "none")
    assert state.compiled_calls >= len(unet.blocks) and state.eager_calls == 0, (state.compiled_calls, state.eager_calls)
    ulp, fp32 = _ULP_GRADS.get(preset, ()), _FP32_GRADS.get(preset, ())
    assert_same(ref, new, f"{preset} {ckpt}: compiled vs eager", ulp_grads=ulp, fp32_grads=fp32)
    # a second step (the cached graphs) gives the same results as the first compiled one
    again = step(unet, net, x, t, ctx, pad, checkpoint=ckpt != "none")
    assert_same(new, again, f"{preset} {ckpt}: second compiled step vs first")


@pytest.mark.parametrize("preset,factored", [("loha", True), ("loha", False), ("loha_dora", True), ("loha_dora", False),
                                             ("abba", False), ("abba_dora", False)])
def test_forced_path_traces_and_matches_eager(preset, factored, monkeypatch):
    """Both fixed paths of LoHa / ABBA (+DoRA) inside the graph, whatever the cost rule picks for the toy shapes (at
    Anima's shapes the planned row count picks dense on many layers from dim 16 / 32): _hada_factored_cheaper forced to
    the factored (x KA^T) KB^T path (_HadaDelta; with DoRA _HadaDoRAOut) or to the dense rebuild (LoHa: _HadaWeight).
    One graph, no break, and the compiled forward + backward equal eager exactly (zero-init factors seeded)."""
    bf16_backward_or_skip()
    monkeypatch.setattr(laf, "_hada_factored_cheaper", lambda *a, **k: factored)
    unet, net, _ = build(preset)
    assert seed_zero_factors(net) == len(laf._all_loras(net))
    unet._block_compile = True
    x, t, ctx, pad = inputs(t=[0.3])
    cap = capture_block_args(unet, net, x, t, ctx, pad)
    laf.prime_module_caches(net)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        ex = torch._dynamo.explain(unet.blocks[0]._forward)(*cap["args"], **cap["kwargs"])
    assert ex.graph_break_count == 0 and ex.graph_count == 1, [str(r.reason)[:100] for r in ex.break_reasons]
    unet._block_compile = False
    torch._dynamo.reset()
    ref = step(unet, net, x, t, ctx, pad)
    compile_toy(unet, net)
    new = step(unet, net, x, t, ctx, pad)
    assert_same(ref, new, f"{preset} forced factored={factored}: compiled vs eager")


@pytest.mark.parametrize("preset", ["lora", "loha", "tsm"])
def test_blocks_share_the_compiled_graphs(preset):
    """compile_blocks compiles one entry per Block class and calls it with the Block as an argument: a 4-Block toy runs
    on the same 2 graphs as a 2-Block one (Block 0's bf16 input, the fp32 residual of Blocks 1+). An entry closing over
    each Block's bound _forward gave every Block graphs of its own (4 here; 28 per regime on Anima 2.9B, past the
    recompile limit, the rest then eager)."""
    unet, net, _ = build(preset, num_blocks=4)
    backend, calls = counting_backend("aot_eager")
    state = compile_toy(unet, net, backend=backend)
    x, t, ctx, pad = inputs(t=[0.3])
    step(unet, net, x, t, ctx, pad, backward=False)
    assert state.compiled_calls == 4 and state.eager_calls == 0, (state.compiled_calls, state.eager_calls)
    assert len(calls) == 2, f"{len(calls)} graphs for 4 Blocks"


def test_eval_mode_training_forward_runs_compiled():
    """Without --gradient_checkpointing the trainer keeps the DiT in eval mode for training too (train_network.py:
    unet.eval() unless gradient checkpointing). A forward with grad enabled is a training forward and runs compiled -
    it ran eager, so block compile (and --anima_activation_memory_budget with it) did nothing without checkpointing; the
    compiled step equals eager. Eval under no_grad (validation) still runs the eager Blocks."""
    bf16_backward_or_skip()
    unet, net, _ = build("lora")
    unet.eval()
    x, t, ctx, pad = inputs(t=[0.3])
    ref = step(unet, net, x, t, ctx, pad)
    state = compile_toy(unet, net)
    new = step(unet, net, x, t, ctx, pad)
    assert state.compiled_calls == len(unet.blocks) and state.eager_calls == 0, (state.compiled_calls, state.eager_calls)
    assert_same(ref, new, "eval-mode training step: compiled vs eager")
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        unet(x, t, ctx, padding_mask=pad)
    assert state.eager_calls == len(unet.blocks)


def _embed_rows(grads):
    """The timestep-embedding rows that got a gradient, per module (sorted names)."""
    return [g.abs().sum(1).nonzero().flatten().tolist() for n, g in sorted(grads.items()) if n.endswith("timestep_embed.weight")]


@pytest.mark.parametrize("preset", ["tsm_stage2", "tsm_stage2_fn", "tsm_lokr_stage2"])
def test_tsm_stage2_compiled_reads_the_step_embedding_row(preset):
    """Stage 2's router adds the step's timestep-embedding row. Inside a graph the integer timestep is the constant 0
    (the step is hoisted), so the row must come from the hoisted index tensor _hx_tidx: TSMModule's gates read it; before
    this fix TSM-LoKr's gates and the opt-in tsm_expert_fn Function (now the factor path inside the graphs) read row 0
    for every timestep. With the stage-2 weights perturbed (router, embedding and context experts are zero at init, so the
    gated term vanished whichever row was read), timesteps 260 and 369 (one expert interval, different rows) run on the
    same graphs and equal eager, and the embedding gradient lands on the step's row; 900 (other experts) too."""
    bf16_backward_or_skip()
    unet, net, _ = build(preset)
    assert perturb_tsm(net) > 0
    x, _, ctx, pad = inputs(t=[0.3])
    sigmas = (0.26, 0.37, 0.9)
    fn = laf._TSM_EXPERT_FN
    try:
        laf._TSM_EXPERT_FN = False  # the eager reference: the factor path, which the graphs take for stage 2
        refs = {s: step(unet, net, x, torch.tensor([s]).to(torch.bfloat16), ctx, pad) for s in sigmas}
    finally:
        laf._TSM_EXPERT_FN = fn
    assert _embed_rows(refs[0.26][1]) != _embed_rows(refs[0.37][1]) and all(r for r in _embed_rows(refs[0.26][1]))
    backend, calls = counting_backend("aot_eager")
    compile_toy(unet, net, backend=backend)
    first = None
    for s in sigmas:
        new = step(unet, net, x, torch.tensor([s]).to(torch.bfloat16), ctx, pad)
        if first is None:
            first = len(calls)
        elif s == 0.37:
            assert len(calls) == first, f"sigma {s}: recompiled inside one expert interval"
        assert_same(refs[s], new, f"{preset} sigma {s}: compiled vs eager")
        assert _embed_rows(new[1]) == _embed_rows(refs[s][1]), f"{preset} sigma {s}: embedding rows differ"


def test_tsm_timesteps_of_one_expert_interval_share_the_graphs():
    """TSM stage 1, 8 core experts over 1000 timesteps (one every 125): timesteps 260 / 300 / 370 pick the same experts
    and must not recompile (the graph guards on the hoisted expert tuple, not on the integer timestep); timestep 900
    picks other experts: at most one new graph per Block dtype regime (the first step's count)."""
    unet, net, _ = build("tsm")
    backend, calls = counting_backend("aot_eager")
    compile_toy(unet, net, backend=backend)
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


def test_tlora_timesteps_share_the_graphs_through_the_hoisted_mask():
    """T-LoRA rank 8 with mask min rank 4: timesteps 300 / 260 / 370 keep 6 ranks, 898 keeps 4, 50 keeps 7. No timestep
    recompiles: the graphs read the mask set_timestep hoisted onto each module (_hx_tmask, float32 (1, rank), rewritten
    in place per step) instead of LyCORIS' module-global dict, which holds a new tensor every step; each compiled output
    is the eager one (the factors perturbed off their bases, so the output depends on the mask). A batch of two timesteps hoists the per-sample (2, rank) mask (a new tensor), two rows again
    rewrite it in place. Eager _tlora_mask keeps reading the global."""
    from lycoris.modules.tlora import get_timestep_mask

    unet, net, _ = build("tlora")
    assert perturb_tlora(net) == len(laf._all_loras(net))  # a mask-dependent delta: the output shows the mask read
    x, _, ctx, pad = inputs(t=[0.3])
    sigmas = (0.3, 0.26, 0.37, 0.9, 0.05)
    refs = {s: step(unet, net, x, torch.tensor([s]).to(torch.bfloat16), ctx, pad, backward=False)[0] for s in sigmas}
    backend, calls = counting_backend("aot_eager")
    compile_toy(unet, net, backend=backend)
    y, _ = step(unet, net, x, torch.tensor([0.3]).to(torch.bfloat16), ctx, pad, backward=False)
    first = len(calls)
    assert first >= 1 and torch.equal(y, refs[0.3])
    mods = [m for m in laf._all_loras(net) if type(m).__name__ == "TLoraModule"]
    assert len(mods) == len(laf._all_loras(net))
    hx = mods[0].__dict__["_hx_tmask"]
    assert hx.shape == (1, 8) and hx.dtype == torch.float32 and hx.sum().item() == 6
    hx0 = {id(m): m.__dict__["_hx_tmask"] for m in mods}
    for s in sigmas[1:]:
        y, _ = step(unet, net, x, torch.tensor([s]).to(torch.bfloat16), ctx, pad, backward=False)
        assert len(calls) == first, f"sigma {s}: recompiled ({len(calls) - first} new graphs)"
        assert torch.equal(y, refs[s]), f"sigma {s}: compiled output differs from eager"
        assert all(m.__dict__["_hx_tmask"] is hx0[id(m)] for m in mods)  # every module's copy rewritten in place
    assert mods[0].__dict__["_hx_tmask"] is hx and hx.tolist() == [[1.0] * 7 + [0.0]]  # sigma 0.05: 7 ranks, in place
    assert all(torch.equal(m.__dict__["_hx_tmask"], hx) for m in mods)
    # eager reads the global; the hoisted copy holds the same values
    assert laf._tlora_mask(mods[0], torch.device("cpu")) is get_timestep_mask(mods[0].mask_group_id)
    assert torch.equal(laf._tlora_mask(mods[0], torch.device("cpu")), hx)
    laf.set_timestep(net, torch.tensor([300.0, 900.0]))  # per-sample masks: another shape, a new tensor per module
    hx2 = mods[0].__dict__["_hx_tmask"]
    assert hx2 is not hx and hx2.shape == (2, 8) and hx2.sum(1).tolist() == [6.0, 4.0]
    laf.set_timestep(net, torch.tensor([100.0, 100.0]))
    assert mods[0].__dict__["_hx_tmask"] is hx2 and hx2.sum(1).tolist() == [7.0, 7.0]  # rewritten in place
    assert all(m.__dict__["_hx_tmask"].shape == (2, 8) for m in mods)
    laf.set_block_compile(False)
    laf.set_timestep(net, torch.tensor([300.0]))  # block compile off: set_timestep leaves the hoisted copy alone
    assert mods[0].__dict__["_hx_tmask"] is hx2 and hx2.shape == (2, 8)


def test_tlora_mask_values_reach_the_compiled_gradients():
    """Perturbed T-LoRA (a mask-dependent delta), compiled at 300 (6 ranks): forward + backward at 898 (4 ranks) and 50
    (7 ranks) run on the same graphs and give eager's output and every gradient exactly - the graph reads the step's mask
    values, not the traced ones. After clear_timestep (the trainer's call after every backward) a compiled forward
    computes eager's no-mask result (every rank): the hoisted copies are filled with ones in place."""
    bf16_backward_or_skip()
    unet, net, _ = build("tlora")
    perturb_tlora(net)
    x, _, ctx, pad = inputs(t=[0.3])
    sigmas = (0.3, 0.9, 0.05)
    refs = {s: step(unet, net, x, torch.tensor([s]).to(torch.bfloat16), ctx, pad) for s in sigmas}
    laf.clear_timestep(net)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        ref_nomask = unet(x, torch.tensor([0.3]).to(torch.bfloat16), ctx, padding_mask=pad).detach().float()
    backend, calls = counting_backend("aot_eager")
    state = compile_toy(unet, net, backend=backend)
    new = step(unet, net, x, torch.tensor([0.3]).to(torch.bfloat16), ctx, pad)
    first = len(calls)
    assert_same(refs[0.3], new, "sigma 0.3 (traced): compiled vs eager")
    for s in sigmas[1:]:
        new = step(unet, net, x, torch.tensor([s]).to(torch.bfloat16), ctx, pad)
        assert len(calls) == first, f"sigma {s}: recompiled ({len(calls) - first} new graphs)"
        assert_same(refs[s], new, f"sigma {s}: compiled vs eager")
    mods = [m for m in laf._all_loras(net) if type(m).__name__ == "TLoraModule"]
    hx = [m.__dict__["_hx_tmask"] for m in mods]
    laf.clear_timestep(net)
    assert all(m.__dict__["_hx_tmask"] is h and bool((h == 1).all()) for m, h in zip(mods, hx))  # in place, every rank
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = unet(x, torch.tensor([0.3]).to(torch.bfloat16), ctx, padding_mask=pad).detach().float()
    assert state.eager_calls == 0 and len(calls) == first
    assert torch.equal(y, ref_nomask), "compiled forward after clear_timestep differs from eager without a mask"


@pytest.mark.parametrize("preset", ["tlora", "tsm", "tsm_stage2", "tsm_lokr"])
def test_clear_timestep_resets_the_hoists_without_a_recompile(preset):
    """The trainer's sequence per micro-step: set_timestep, forward (+ backward), clear_timestep. clear_timestep puts the
    block-compile hoists back to the no-timestep state (T-LoRA's _hx_tmask: ones, in place; TSM's expert tuples
    _hx_experts / _hx_idx: removed, stage 2's _hx_tidx kept for the in-place rewrite) and the next set_timestep restores
    them: the steps after the first add no graph."""
    unet, net, _ = build(preset)
    backend, calls = counting_backend("aot_eager")
    compile_toy(unet, net, backend=backend)
    x, _, ctx, pad = inputs(t=[0.3])
    kinds = ("TLoraModule",) if preset == "tlora" else laf._TSM_KINDS
    mods = [m for m in laf._all_loras(net) if type(m).__name__ in kinds]
    assert mods
    first = None
    for sigma in (0.3, 0.26, 0.37, 0.3, 0.3):
        step(unet, net, x, torch.tensor([sigma]).to(torch.bfloat16), ctx, pad, backward=False)
        if first is None:
            first = len(calls)
            assert first >= 1
            hx = {id(m): m.__dict__.get("_hx_tmask", m.__dict__.get("_hx_tidx")) for m in mods}
        assert len(calls) == first, f"sigma {sigma}: recompiled after clear_timestep ({len(calls) - first} new graphs)"
        laf.clear_timestep(net)
        for m in mods:
            d = m.__dict__
            assert "_hx_experts" not in d and "_hx_idx" not in d
            if preset == "tlora":
                assert d["_hx_tmask"] is hx[id(m)] and bool((d["_hx_tmask"] == 1).all())
            elif preset == "tsm_stage2" and type(m).__name__ == "TSMModule":
                assert d["_hx_tidx"] is hx[id(m)]  # kept: the next step rewrites it in place


@pytest.mark.parametrize("ckpt", ["unsloth", "torch"])
def test_adaln_hoist_checkpointed_blocks_run_the_compiled_forward(ckpt, monkeypatch):
    """--anima_adaln_hoist: a checkpointed Block runs Block._forward_hoisted (the modulations as flat checkpoint inputs),
    which forwards to self._forward. compile_blocks wraps only _forward, so the hoisted entry reaches the compiled graph
    through it - compiled itself, _forward_hoisted's trace hit the _forward wrapper's mark_dynamic ("Attempt to trace
    forbidden callable maybe_mark_dynamic") at the first checkpointed call. The compiled step equals the eager hoisted
    one exactly."""
    bf16_backward_or_skip()
    monkeypatch.setattr(anima_models, "ADALN_HOIST", True)
    hoisted = []
    orig = anima_models.Block._forward_hoisted

    def spy(self, *a, **k):
        hoisted.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(anima_models.Block, "_forward_hoisted", spy)
    unet, net, _ = build("lora")
    unet.enable_gradient_checkpointing(unsloth_offload=ckpt == "unsloth")
    x, t, ctx, pad = inputs(t=[0.3])
    ref = step(unet, net, x, t, ctx, pad, checkpoint=True)
    assert hoisted, "the hoisted path did not run"
    state = compile_toy(unet, net)
    assert all("_forward_hoisted" not in b.__dict__ and "_forward" in b.__dict__ for b in unet.blocks)
    hoisted.clear()
    new = step(unet, net, x, t, ctx, pad, checkpoint=True)
    assert hoisted and state.compiled_calls >= len(unet.blocks) and state.eager_calls == 0
    assert_same(ref, new, f"adaln hoist {ckpt}: compiled vs eager")


def test_abba_path_choice_is_fixed_under_compile():
    """_abba_forward picks the factored or the dense rebuild path by a roofline rule that reads the token count; inside a
    compiled graph that count is the symbolic size of the dynamic token axis, so the rule became a guard on it: with a
    strict mark_dynamic band across the rule's flip (the toy's 64 x 64 layers flip between 256 and 1024 rows) dynamo
    raised ConstraintViolationError before the first graph, with soft marks it recompiled at the flip. Under block
    compile without a row plan (a graph compiled outside after_network_loaded) the factored path is taken: one graph
    over the whole band, the output identical to the factored eager path at every row count (eager keeps the rule and
    picks dense past the flip)."""
    unet, net, _ = build("abba")
    seed_zero_factors(net)
    m = laf._all_loras(net)[0]
    lin = m.org_module[0]
    assert tuple(m.shape) == (64, 64) and not m.wd
    rank = m.r1 * m.r2
    rows = (16, 64, 256, 1024, 2000)
    rule = [laf._hada_factored_cheaper(m, torch.empty(1, n, 64, device="meta"), "abba", rank, False, False) for n in rows]
    assert rule[0] is True and rule[-1] is False, rule  # the eager rule flips inside the band
    backend, calls = counting_backend("aot_eager")
    compiled = torch.compile(lambda x_: lin(x_), backend=backend, dynamic=False)
    g = torch.Generator().manual_seed(0)
    orig_rule = laf._hada_factored_cheaper
    for n, factored in zip(rows, rule):
        x = torch.randn(1, n, 64, generator=g).to(torch.bfloat16)
        torch._dynamo.mark_dynamic(x, 1, min=16, max=4096)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = compiled(x)
            try:
                laf._hada_factored_cheaper = lambda *a, **k: True  # the factored eager path as the reference
                y_factored = lin(x)
            finally:
                laf._hada_factored_cheaper = orig_rule
            y_eager = lin(x)
        assert len(calls) == 1, f"{n} rows: {len(calls)} graphs"
        assert torch.equal(y, y_factored), f"{n} rows: compiled differs from the factored eager path"
        if not factored:  # eager took the dense path here: the same math, rounded elsewhere
            torch.testing.assert_close(y_eager.float(), y.float(), rtol=2 ** -6, atol=1e-2)


@pytest.mark.parametrize("preset,where", [("loha", "mlp.layer1"), ("abba", "self_attn.q_proj")])
def test_planned_rows_fix_the_path_choice_under_block_compile(preset, where):
    """The LoHa / ABBA rule (_hada_factored_cheaper) flips inside a strict band of 16 .. 4096 rows on these toy Linears
    (LoHa's 256 x 64 MLP Linear between 64 and 256 rows, ABBA's 64 x 64 projection between 256 and 1024), as it does
    inside a 4-tier band at Anima's shapes (LoHa with fp32 factors from dim 16, with DoRA from dim 8; ABBA from dim 32).
    plan_block_compile_rows (after_network_loaded) gives the token-row Linears a fixed row count - the cross-attention
    key / value and the adaLN Linears 0 (their rows are static in the graph) - and the rule runs at it inside the graphs:
    one graph over the band, every output equal to the eager path the rule picks at the planned count (factored at 64,
    dense at 4096) and different from the other path's."""
    unet, net, _ = build(preset)
    assert seed_zero_factors(net) == len(laf._all_loras(net))  # a zero delta made both paths return the base output
    blk = unet.blocks[0]
    lin = blk.get_submodule(where)
    of = {id(m.org_module[0]): m for m in laf._all_loras(net)}
    m = of[id(lin)]
    rank = m.r1 * m.r2 if preset == "abba" else m.hada_w1_a.shape[1] * m.hada_w2_a.shape[1]
    fp32_grads = preset == "loha"  # as the forwards pass it: LoHa's fp32 factors under bf16 autocast; ABBA always False
    rows = (16, 64, 256, 1024, 2000, 4096)
    rule = [laf._hada_factored_cheaper(m, torch.empty(1, n, lin.in_features, device="meta"), preset, rank, False,
                                       fp32_grads) for n in rows]
    assert rule[0] is True and rule[-1] is False, rule  # the eager rule flips inside the band
    n_token = laf.plan_block_compile_rows(net, unet, 4096)
    static = [of[id(blk.cross_attn.k_proj)], of[id(blk.cross_attn.v_proj)], of[id(blk.adaln_modulation_self_attn[1])],
              of[id(blk.adaln_modulation_mlp[2])]]
    token = [m, of[id(blk.self_attn.k_proj)], of[id(blk.self_attn.v_proj)], of[id(blk.cross_attn.output_proj)]]
    assert all(s.__dict__["_bc_rows"] == 0 for s in static) and all(t.__dict__["_bc_rows"] == 4096 for t in token)
    assert 0 < n_token < len(laf._all_loras(net))
    orig_rule = laf._hada_factored_cheaper
    g = torch.Generator().manual_seed(0)
    for planned, factored in ((64, True), (4096, False)):
        laf.plan_block_compile_rows(net, unet, planned)
        torch._dynamo.reset()
        backend, calls = counting_backend("aot_eager")
        compiled = torch.compile(lambda x_: lin(x_), backend=backend, dynamic=False)
        for n in rows:
            x = torch.randn(1, n, lin.in_features, generator=g).to(torch.bfloat16)
            torch._dynamo.mark_dynamic(x, 1, min=16, max=4096)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                y = compiled(x)
                try:
                    laf._hada_factored_cheaper = lambda *a, **k: factored  # the eager path the plan picks
                    y_ref = lin(x)
                    laf._hada_factored_cheaper = lambda *a, **k: not factored  # and the other one
                    y_other = lin(x)
                finally:
                    laf._hada_factored_cheaper = orig_rule
            assert len(calls) == 1, f"planned {planned}, {n} rows: {len(calls)} graphs"
            assert not torch.equal(y_ref, y_other), f"{n} rows: the two paths round alike, the check would be blind"
            assert torch.equal(y, y_ref), f"planned {planned}, {n} rows: compiled differs from the planned eager path"


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
    state = compile_toy(unet, net, skip_call=laf.mixed_timestep_chunk)
    new_mixed = step(unet, net, x, t, ctx, pad)
    assert state.eager_calls == len(unet.blocks) and state.compiled_calls == 0, (state.eager_calls, state.compiled_calls)
    assert_same(ref_mixed, new_mixed, "mixed-timestep chunk (eager under block compile)")
    new_same = step(unet, net, x1, t1, ctx1, pad1)
    assert state.compiled_calls >= len(unet.blocks), state.compiled_calls
    assert_same(ref_same, new_same, "batch 2, one timestep (compiled)")


@pytest.mark.parametrize("preset", ["lokr_dora", "loha", "loha_dora", "full", "tlora", "abba", "diag_oft", "boft"])
def test_resolution_sweep_inside_the_union_band_shares_the_graphs(preset):
    """Latent grids 8x8, 8x12, 12x12, 16x16 (16 .. 64 tokens) with the union band (16, 64): the first grid compiles, the
    others run on its graphs (soft marks on the token axis of x and the RoPE tables under attn_mode torch). LoHa's and
    ABBA's path rule (_hada_factored_cheaper, a cost rule on the row count) runs at the planned row count inside the
    graphs, no guard on the token axis (test_planned_rows_fix_the_path_choice_under_block_compile); boft's row count
    only reaches the few-rows / blockwise choice (rows <= _OFT_MAX_BLOCK), one side of it for every grid."""
    unet, net, _ = build(preset)
    backend, calls = counting_backend("aot_eager")
    state = compile_toy(unet, net, backend=backend)  # cold network, caches primed as the trainer does
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
    laf.prime_module_caches(net)
    bc.compile_blocks(unet, bands=[], backend=backend, dynamic="static", recompile_limit=64, isolate_cache=False)
    x, t, ctx, pad = inputs(latent_hw=(8, 8), t=[0.3])
    step(unet, net, x, t, ctx, pad, backward=False)
    first = len(calls)
    x, t, ctx, pad = inputs(latent_hw=(8, 12), t=[0.3])
    step(unet, net, x, t, ctx, pad, backward=False)
    assert len(calls) == 2 * first, (first, len(calls))


def test_pause_eval_and_restore_run_the_eager_blocks():
    """pause() (sample images) and eval mode under no_grad (validation) take the eager _forward; restore_blocks removes
    the wrappers and the fake-5D flag."""
    unet, net, _ = build("lora")
    state = compile_toy(unet, net, backend="eager")
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
    """rebind_forwards: every adapted Linear's forward becomes a closure (dynamo-traceable) carrying the module as __wrapped_lora__ (not
    __self__: dynamo refuses to inline a function that has one), so
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
                torch_compile=False, lycoris_stock=False, network_module="lycoris.kohya", network_args=None)
    base.update(over)
    return argparse.Namespace(**base)


def test_validate_block_compile_args():
    from anima_train_network import validate_block_compile_args

    validate_block_compile_args(_trainer_args())
    validate_block_compile_args(_trainer_args(anima_block_compile=False, blocks_to_swap=10, use_ramtorch=True))  # off: anything
    validate_block_compile_args(_trainer_args(lycoris_stock=True, network_module="networks.lora"))  # stock only matters for LyCORIS
    validate_block_compile_args(_trainer_args(network_args=["algo=lokr", "preset=anima-exp"]))
    validate_block_compile_args(_trainer_args(network_args=["algo=dylora"], network_module="networks.lora"))  # not LyCORIS' algo
    for bad, match in ((dict(use_ramtorch=True), "RamTorch"), (dict(use_ramtorch_network=True), "RamTorch"),
                       (dict(network_args=["algo=dylora", "preset=anima-exp", "block_size=4"]), "dylora"),
                       (dict(network_args=["algo=DyLoRA"]), "dylora"), (dict(network_module="networks.dylora"), "dylora"),
                       (dict(blocks_to_swap=4), "blocks_to_swap"), (dict(torch_compile=True), "torch_compile"),
                       (dict(lycoris_stock=True), "lycoris_stock"), (dict(anima_block_compile_dynamic="auto"), "dynamic"),
                       (dict(anima_activation_memory_budget=0.0), "memory_budget"),
                       (dict(anima_activation_memory_budget=1.5), "memory_budget"),
                       (dict(anima_block_compile_recompile_limit=4), "recompile_limit")):
        with pytest.raises(ValueError, match=match):
            validate_block_compile_args(_trainer_args(**bad))


def test_dylora_is_unfit_for_block_compile_by_design():
    """algo=dylora is refused with --anima_block_compile (validate_block_compile_args) rather than fixed: the fixed path
    draws a random block count per forward (dylora.py:114-116; the tensor shapes of the step follow a Python random
    number, a graph per rank at best) and reads torch._C._current_graph_task_id() to hand a checkpoint recompute the
    forward's draw - dynamo has no graph for either, so a cold Block trace breaks at every adapted Linear."""
    from anima_train_network import validate_block_compile_args

    with pytest.raises(ValueError, match="dylora.*random rank"):
        validate_block_compile_args(_trainer_args(network_args=list(PRESETS["dylora"]["args"])))
    unet, net, _ = build("dylora")
    unet._block_compile = True
    x, t, ctx, pad = inputs(t=[0.3])
    cap = capture_block_args(unet, net, x, t, ctx, pad)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cold = torch._dynamo.explain(unet.blocks[0]._forward)(*cap["args"], **cap["kwargs"])
    assert cold.graph_break_count > 0
    assert any("_current_graph_task_id" in str(r.reason) for r in cold.break_reasons), [str(r.reason)[:100] for r in cold.break_reasons]


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
    assert printed and "1024, 1536, 4096" in printed[0] and f"{len(laf._all_loras(net))} LyCORIS module caches primed" in printed[0]
    assert all("_laf_fast" in m.__dict__ for m in laf._all_loras(net))  # the lazy caches were filled before compiling
    planned = {m.__dict__.get("_bc_rows") for m in laf._all_loras(net)}
    assert planned == {0, 4096}, planned  # the path-rule row plan: token-row Linears 4096, text / adaLN rows 0
    x, t, ctx, pad = inputs(t=[0.3])
    step(unet, net, x, t, ctx, pad, backward=False)
    assert state.compiled_calls == len(unet.blocks)
    bc.restore_blocks(unet)
