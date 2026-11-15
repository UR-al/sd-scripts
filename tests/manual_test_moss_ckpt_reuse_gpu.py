"""GPU harness for --moss_ckpt_nograd_merged / --moss_ckpt_reuse_attn / --moss_ckpt_reuse_attn_budget_gib (2026-11-14;
reuse-only, the budget and the "chains" mode 2026-11-15): what the CPU tests (tests/test_moss_ckpt_reuse.py,
tests/test_moss_ckpt_reuse_only.py) cannot answer - the real flash-attn kernels on MOSS's joint attention, inductor's
compiled joint-block graphs, VRAM and time. The MOSS counterpart of tests/manual_test_anima_ckpt_reuse_gpu.py (its
comparison helpers are imported from there). NOT collected by pytest (run it by hand); it loads no checkpoint unless
--dit is given and runs no training.

What it builds: a MossTransformer with MOSS-Image's block shapes (3072 wide, 24 heads, ffn 8192, text 4096) and --blocks
joint blocks (default 2; weights random, or joint blocks 0..N-1 of --dit), bf16 (the user's mixed precision; --stream_fp32
adds --moss_stream_fp32), the production LyCORIS network (LoKr factor 4 full matrix + DoRA on the output axis, preset
moss, the fast forward and the fixed paths; --exclude leaves joint blocks out of it like --moss_lora_exclude_blocks),
gradient checkpointing on every block, attn_mode flash, one sample per micro-step (no padding: the plain flash call).
Three modes (--modes), the three ways a training run executes a joint block:
* eager    - nothing compiled (a run with --moss_no_compile --moss_no_lokr_compile): both passes of a checkpoint run
             the same kernels;
* chains   - eager blocks with the elementwise chains and the LoKr+DoRA rebuild torch.compile'd on their own, as
             prepare_moss_runtime leaves a default CUDA run WITHOUT --moss_block_compile. The no-grad first pass and the
             grad-mode recompute then go through different compiled graphs of both (the plain checkpoint runs both
             passes in grad mode through one), so "bit-identical in eager" does not carry over: this mode measures it.
             On the CPU self-test it equals eager (those compiles are CUDA-only);
* compiled - per-block compiled (--backend; the chain and rebuild compiles off, as under --moss_block_compile).

The configurations (CONFIGS), each run through the trainer's own switch (moss_models.set_ckpt_reuse):
  off                    plain torch.utils.checkpoint
  nograd                 --moss_ckpt_nograd_merged
  nograd+reuse           --moss_ckpt_nograd_merged --moss_ckpt_reuse_attn
  reuse-only             --moss_ckpt_reuse_attn alone: the node with nothing cached (MossBlock._forward_uncached,
                         lycoris_fast._CKPT_PASS 0 in both passes, the adapter-key guard) - the mode a 32 GB card needs
  reuse-only, budget     ... + --moss_ckpt_reuse_attn_budget_gib sized so that --budget_blocks of the joint blocks
                         (default: half of --blocks) reuse and the others keep torch.utils.checkpoint: a micro-step that
                         mixes node blocks and plain blocks, and the peak VRAM a budget gives
  nograd+reuse, budget   the same budget with the cache on (blocks over it keep the node without a stored output)

What it reports, per image token count (--tokens, default 1024 and 16384 = a 2048^2 image; --text_tokens text tokens are
added to the joint sequence) and per mode (eager / chains / compiled):
* correctness, flags on vs off: max |d|, max |d| / max |ref| and the count of differing elements of the output and of
  every LoKr / DoRA gradient (0 = bit-identical); max |recompute output - first-pass output| over the joint blocks
  (anima_models.CKPT_RECOMPUTE_DIFFS); eager and chains: whether the recompute's q, k, v equal the first pass's bit for
  bit, for nograd+reuse and for reuse-only (chains: the direct answer to whether the two compiled graphs of the chains
  and of the rebuild round alike);
* the baseline those numbers are read against, flags off vs flags off (correctness["off_vs_off"]): the same micro-step
  repeated --baseline_runs times with both flags off, the worst of every number. flash-attn's default backward is
  documented as non-deterministic, so on the GPU a non-zero "differing" of flags on vs off is only a finding when it is
  larger than this run-to-run noise; every flags-on entry carries the comparison as "reading" (bit-identical / inside
  the flags-off run-to-run noise / OUTSIDE it). OUTSIDE in the compiled mode means a reproducible difference between
  the two code paths (inductor rounding the no-grad and the grad graph differently, the eager-built cache against the
  rebuild fused into the graph) - on Anima it was as large as block compile's own difference from eager;
* that scale, measured here (chains / compiled mode after the eager mode of the same run):
  run["block_compile_vs_eager"] (chains: run["chain_compile_vs_eager"]) is the mode's flags-off micro-step against the
  eager flags-off one (same weights, same batch), and every flags-on entry of the mode carries "vs_block_compile"
  ("vs_chain_compile"): its relative L2 difference from the mode's flags-off (output; all gradients pooled) divided by
  the compile's own, with a reading (bit-identical / no larger than the compile's own difference / LARGER). Every
  correctness entry also has the relative L2 numbers themselves ("l2"): the max-norm numbers above are decided by
  single elements;
* the cached merged weight against a fresh eager rebuild and against the same rebuild torch.compile'd on its own;
* time per micro-step (forward + backward, median of --iters after --warmup) for every configuration, with the
  cache current (micro-steps 2..N of an optimizer step) and, for the cached ones, for the first micro-step after an
  optimizer step (cache rebuilt), and the per-block saving (the model is timed with --blocks and with 1 joint block:
  difference / (blocks - 1); not for the budget configurations, where the blocks differ in kind);
* VRAM: peak allocated per configuration (a configuration without the cache is measured without another one's cache
  left in VRAM), the cache's bytes and the stored attention's bytes (under a budget: of the blocks that reuse). Scale
  them to a run: cache bytes per adapted block x 41, stored attention bytes per block x the blocks that reuse (42
  without a budget, at the largest tier), on top of the run's peak.

VRAM: 2 joint blocks at 16384 + 256 tokens are estimated at 8-11 GiB (not measured: MOSS blocks are 1.5x wider than
Anima 2.9B's, whose harness stays under 7 GiB); the process is capped at --vram_cap_gb (default 12) and refuses to start
when other processes already hold more than --busy_gb of the GPU (a training run). Use --tokens 1024 4096 for a run
under ~4 GiB.

Run (GPU):  venv_t214/Scripts/python.exe -B tests/manual_test_moss_ckpt_reuse_gpu.py --json moss_ckpt_reuse_gpu.json
Self-test (CPU, toy shapes, the fake flash kernels, aot_eager):  ... tests/manual_test_moss_ckpt_reuse_gpu.py --device cpu
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from library import anima_models, attention, lycoris_args, lycoris_fast  # noqa: E402
from library import lycoris_algo_fixes as laf  # noqa: E402
from library.moss import block_compile as mbc  # noqa: E402
from library.moss import models as moss_models  # noqa: E402
from library.moss import train_utils as moss_train_utils  # noqa: E402
from library.moss.lycoris_presets import register_moss_lycoris_presets  # noqa: E402
from tests.manual_test_anima_ckpt_reuse_gpu import BASELINE, FLAGS, compare, compare_results, noise_floor, reading  # noqa: E402

NETWORK_ARGS = ["algo=lokr", "preset=moss", "factor=4", "full_matrix=True", "decompose_both=False", "dora_wd=True",
                "wd_on_output=True"]
PROD = dict(dim=3072, ffn_dim=8192, num_heads=24, text_dim=4096, freq_dim=256, latent_channels=128, attn_mode="flash")
TOY = dict(dim=192, ffn_dim=512, num_heads=4, text_dim=64, freq_dim=32, latent_channels=128, attn_mode="flash")


def build(cfg: dict, blocks: int, device: torch.device, dtype: torch.dtype, exclude=(), stream_fp32: bool = False,
          dit: str = None, seed: int = 0):
    """The DiT (frozen, `dtype`, train mode, every block checkpointed) and the LoKr+DoRA network, applied the way
    train_network.py does it (tests/test_moss_ckpt_reuse.build)."""
    import lycoris.kohya as kohya

    torch.manual_seed(seed)
    model = moss_models.MossTransformer(**cfg, num_layers=blocks)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.02)
    if dit:
        load_blocks(model, dit, blocks)
    model = model.to(device=device, dtype=dtype)
    model.requires_grad_(False)
    model.train()
    args = argparse.Namespace(network_args=list(NETWORK_ARGS), network_dim=10000, network_alpha=10000,
                              network_module="lycoris.kohya", network_dropout=None, lycoris_stock=False,
                              lycoris_algo_fixes=False, network_weights=None,
                              moss_lora_exclude_blocks=",".join(str(i) for i in exclude if i < blocks))
    moss_train_utils.apply_moss_block_exclusion(args)
    kwargs = lycoris_args.fix_lycoris_bool_kwargs(args.network_module, dict(i.split("=", 1) for i in args.network_args))
    create = laf.prepare_network_kwargs("lokr", dict(kwargs))
    create.setdefault("dropout", 0.0)
    laf.set_block_compile(True)  # the adapted forwards re-bound through closures (needed compiled, harmless eager)
    network = kohya.create_network(1.0, args.network_dim, args.network_alpha, None, None, model, neuron_dropout=None, **create)
    if hasattr(network, "prepare_network"):
        network.prepare_network(args)
    network.apply_to(None, model, False, True)
    laf.after_network_created(network, args)
    network.to(device)
    network.requires_grad_(True)
    network.train()
    g = torch.Generator().manual_seed(seed + 7)
    with torch.no_grad():  # LoKr starts as a zero delta (lokr_w2 = 0): move every factor, as after some training
        for p in network.parameters():
            p.add_((torch.randn(p.shape, generator=g) * 0.02).to(p))
    if stream_fp32:
        model.enable_stream_fp32()
    model.enable_gradient_checkpointing()
    laf.prime_module_caches(network)
    return model, network


def load_blocks(model, path: str, blocks: int) -> None:
    """Joint blocks 0..blocks-1 of a real MOSS-Image DiT checkpoint into the model (everything else stays random)."""
    from safetensors import safe_open

    loaded = 0
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        prefix = next((p for p in ("", "model.diffusion_model.", "diffusion_model.", "transformer.", "model.")
                       if any(k.startswith(p + "layers.") for k in keys)), "")
        state = model.state_dict()
        for name in state:
            if name.startswith("layers.") and int(name.split(".")[1]) < blocks and prefix + name in keys:
                state[name].copy_(f.get_tensor(prefix + name).to(state[name].dtype))
                loaded += 1
    print(f"loaded {loaded} joint block tensors from {path}")


def batch(cfg: dict, tokens: int, text_tokens: int, device, dtype, seed: int = 1):
    """One sample: latents of a square grid with `tokens` image tokens (one latent pixel = one token), a timestep,
    `text_tokens` text embeddings with an all-valid mask, and the weights of the scalar the gradients are taken of."""
    side = int(round(tokens ** 0.5))
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, cfg["latent_channels"], side, side, generator=g).to(device=device, dtype=dtype)
    t = torch.tensor([500.0], device=device)
    text = torch.randn(1, text_tokens, cfg["text_dim"], generator=g).to(device=device, dtype=dtype)
    mask = torch.ones(1, text_tokens, dtype=torch.bool, device=device)
    w = torch.randn(1, cfg["latent_channels"], side, side, generator=g).to(device)
    return x, t, text, mask, w


def micro_step(model, network, data, autocast_dtype):
    """One micro-step as the trainer runs it (the noisy input marked for the checkpoint); returns the output and grads."""
    x, t, text, mask, w = data
    network.zero_grad(set_to_none=True)
    x = x.detach().requires_grad_(True)
    with torch.autocast(x.device.type, dtype=autocast_dtype):
        y = model(x, t, text, mask)
    (y.float() * w).mean().backward()
    return y.detach().float(), {n: p.grad.detach().clone() for n, p in network.named_parameters() if p.grad is not None}


# The configurations beyond off / nograd / nograd+reuse (2026-11-15). Each goes through the trainer's own switch
# (moss_models.set_ckpt_reuse): nothing is patched, so what is measured is what a training run executes.
# * reuse-only: --moss_ckpt_reuse_attn alone - the no-grad first pass and the attention reuse WITHOUT the merged-weight
#   cache (MossBlock._forward_uncached: lycoris_fast._CKPT_PASS 0 in both passes, the node gets an empty stand-in module,
#   the adapter-key guard). The cache is ~8.65 GiB on the real model and gave +-0 % on its own on Anima; with it a 2048
#   tier does not fit a 32 GB card. Eager: both passes rebuild with the same kernels, q / k / v are expected
#   bit-identical; chains / compiled: the no-grad and the grad graph may round differently.
# * the budget ones: --moss_ckpt_reuse_attn_budget_gib sized per token count so that --budget_blocks of the joint blocks
#   reuse (set_budget below). Reuse-only: the other blocks keep torch.utils.checkpoint, so one micro-step mixes node
#   blocks and plain blocks (the compiled entry then holds the plain grad graph next to the node's two); with the
#   cache: they keep the node without a stored output (MossBlock._forward_no_stash).
# (Until 2026-11-15 a "nograd+reuse, no cache" proxy stood here: the cached mode with anima_models._ckpt_cached_lokrs
# patched to find nothing. It ran _CKPT_PASS 1 / 2 on the real block - not the path the trainer got - and is gone.)
REUSE_ONLY = "reuse-only"
REUSE_ONLY_BUDGET = "reuse-only, budget"
CACHED_BUDGET = "nograd+reuse, budget"
CONFIGS = {**FLAGS, REUSE_ONLY: (False, True), REUSE_ONLY_BUDGET: (False, True), CACHED_BUDGET: (True, True)}
BUDGETED = (REUSE_ONLY_BUDGET, CACHED_BUDGET)
_BUDGET_GIB = None  # the budget of the BUDGETED configurations for the token count being measured (set_budget)


def budget_gib_for(cfg: dict, joint_tokens: int, blocks: int) -> float:
    """A --moss_ckpt_reuse_attn_budget_gib that lets exactly `blocks` joint blocks reuse for one sample of
    `joint_tokens` tokens: (blocks + 0.5) x the bytes one block stores (the trainer takes floor(budget / that))."""
    per_block = int(joint_tokens) * moss_models.ckpt_stash_bytes_per_token(cfg["dim"], cfg["num_heads"])
    return (int(blocks) + 0.5) * per_block / 2**30


def set_budget(gib) -> None:
    global _BUDGET_GIB
    _BUDGET_GIB = gib


def set_flags(name: str, check: bool = True, release=()) -> None:
    """Switch to a configuration of CONFIGS through the trainer's switch. A configuration without the cache drops the
    merged weights the models in `release` still carry from a cached one: reuse-only never reads them, but they would
    sit in VRAM and in its peak."""
    nograd, reuse = CONFIGS[name]
    moss_models.set_ckpt_reuse(nograd, reuse, _BUDGET_GIB if name in BUDGETED else None)
    anima_models.CKPT_RECOMPUTE_DIFFS = [] if (check and (nograd or reuse)) else None
    if not nograd:
        for model in release:
            lycoris_fast.ckpt_merged_release(model)


def cached_lokrs(model) -> list:
    return [lokr for block in model.layers for lokr in anima_models._ckpt_cached_lokrs(block)]


def to_cpu(result):
    """A micro-step result (output, grads) on the CPU: kept across the modes of a run."""
    y, grads = result
    return y.cpu(), {name: g.cpu() for name, g in grads.items()}


def l2_compare(ref, new) -> dict:
    """Relative L2 differences of two micro-step results: ||new - ref|| / ||ref|| of the output, of all gradients pooled
    (grads_rel_l2) and of the worst single gradient tensor. compare_results' max-norm numbers are decided by single
    elements; these say how far the whole result moved (the measure the Anima check was read by, fp32 reference)."""
    (y0, g0), (y1, g1) = ref, new
    num = den = 0.0
    worst, worst_name = 0.0, None
    for name in g0:
        a, b = g0[name].double(), g1[name].double()
        d2, r2 = float((b - a).pow(2).sum()), float(a.pow(2).sum())
        num += d2
        den += r2
        if r2 > 0.0 and (d2 / r2) ** 0.5 > worst:
            worst, worst_name = (d2 / r2) ** 0.5, name
    y0, y1 = y0.double(), y1.double()
    return {"output_rel_l2": float((y1 - y0).norm() / y0.norm().clamp_min(1e-300)),
            "grads_rel_l2": (num / den) ** 0.5 if den > 0.0 else 0.0,
            "worst_tensor_rel_l2": worst, "worst_tensor": worst_name}


def worst_l2(items: list) -> dict:
    """The largest of each number over several l2_compare results (the flags-off repeats)."""
    out = dict(items[0])
    for item in items[1:]:
        if item["worst_tensor_rel_l2"] > out["worst_tensor_rel_l2"]:
            out["worst_tensor"] = item["worst_tensor"]
        for key in ("output_rel_l2", "grads_rel_l2", "worst_tensor_rel_l2"):
            out[key] = max(out[key], item[key])
    return out


SAME_SCALE = 3.0  # flags on vs off up to this many times the compile's own difference reads as "no larger"
OWN_KEYS = {"compiled": ("block_compile_vs_eager", "vs_block_compile", "block compile"),
            "chains": ("chain_compile_vs_eager", "vs_chain_compile", "the chain / rebuild compile")}


def scale_reading(on: dict, own: dict, what: str = "block compile") -> dict:
    """A compiled flags-on vs flags-off difference (l2_compare) against what the compile itself changes (`own`: the
    mode's flags-off vs eager flags-off; `what` names the compile): the two ratios and a reading. The compiled first pass reads the eager-built
    cache where flags off rebuilds inside the graph, and the no-grad and grad graphs round differently - the same kind of
    difference block compile already makes against eager, so the ratio is expected near 1 (Anima: 1.00)."""
    ratios = {}
    for key in ("output_rel_l2", "grads_rel_l2"):
        ratios[key] = (on[key] / own[key]) if own[key] > 0.0 else (0.0 if on[key] == 0.0 else None)
    if on["output_rel_l2"] == 0.0 and on["grads_rel_l2"] == 0.0:
        text = "bit-identical"
    elif any(ratio is None for ratio in ratios.values()):
        text = f"{what} is bit-identical to eager here and flags on is not: a difference of the flags' own"
    elif max(ratios.values()) <= SAME_SCALE:
        text = f"no larger than {what}'s own difference from eager"
    else:
        text = f"LARGER than {what}'s own difference from eager"
    return {"output_ratio": ratios["output_rel_l2"], "grads_ratio": ratios["grads_rel_l2"], "reading": text}


def qkv_passes(model, network, data, autocast_dtype, flags: str = "nograd+reuse") -> dict:
    """Blocks run in Python (eager / chains), a reusing configuration: the joint attention's q, k, v of the first pass
    and of the recompute, compared bit for bit. The passes are told apart by what the node hands the attention (a stash
    list / the stored tensors), not by lycoris_fast._CKPT_PASS: reuse-only keeps that at 0. Under a budget only the
    blocks that reuse are seen."""
    seen = {1: [], 2: []}
    real = attention.attention

    def spy(qkv, **kwargs):
        if kwargs.get("ckpt_stash") is not None:
            seen[1].append([t.detach().clone() for t in qkv])
        elif kwargs.get("ckpt_reuse") is not None:
            seen[2].append([t.detach().clone() for t in qkv])
        return real(qkv, **kwargs)

    attention.attention = spy
    try:
        set_flags(flags, release=[model])
        micro_step(model, network, data, autocast_dtype)
    finally:
        attention.attention = real
    first, second = seen[1], seen[2][::-1]  # backward visits the blocks last to first
    out = {"blocks": len(first), "recomputes": len(second)}
    for i, name in enumerate("qkv"):
        diffs = [compare(a[i], b[i]) for a, b in zip(first, second)]
        out[name] = {"bit_identical": all(d["differing"] == 0 for d in diffs),
                     "max_abs": max((d["max_abs"] for d in diffs), default=None),
                     "differing": sum(d["differing"] for d in diffs)}
    return out


def merged_weight_checks(model, autocast_dtype, backend: str) -> dict:
    """The cached merged weights against (a) a fresh eager rebuild and (b) the same rebuild inside a torch.compile'd
    function (a proxy for what a compiled block graph computes in its recompute and what the flag-off first pass
    computed; the end-to-end answer is the compiled mode's output comparison)."""
    lokrs = cached_lokrs(model)
    lycoris_fast.ckpt_merged_refresh(lokrs, autocast_dtype)
    compiled = torch.compile(lycoris_fast._rebuild_weight, backend=backend, dynamic=False)
    eager = {"differing": 0, "of": 0, "max_abs": 0.0}
    graph = {"differing": 0, "of": 0, "max_abs": 0.0, "max_ulp": 0.0}
    with torch.no_grad():
        for lokr in lokrs:
            args = (lokr.org_module[0].weight.detach(), lokr.lokr_w1, lokr.lokr_w2, float(lokr.scale), lokr.dora_scale,
                    lokr.multiplier_buf, lokr._dora_eps, lokr._dora_norm_dims, autocast_dtype)
            cached = lokr.__dict__["_ckpt_merged"]
            for stats, other in ((eager, lycoris_fast._rebuild_weight(*args)), (graph, compiled(*args))):
                c = compare(cached, other)
                stats["differing"] += c["differing"]
                stats["of"] += c["of"]
                stats["max_abs"] = max(stats["max_abs"], c["max_abs"])
                if stats is graph and c["differing"]:
                    ulp = (cached.float().abs() * torch.finfo(autocast_dtype).eps).clamp_min(1e-30)
                    graph["max_ulp"] = max(graph["max_ulp"], float(((cached.float() - other.float()).abs() / ulp).max()))
    adapted = sum(1 for block in model.layers if anima_models._ckpt_cached_lokrs(block))
    held = lycoris_fast.ckpt_merged_bytes(model)
    return {"modules": len(lokrs), "bytes": held, "adapted_blocks": adapted,
            "gib_for_41_adapted_blocks": (held / adapted * 41 / 2**30) if adapted else 0.0,
            "vs_eager_rebuild": eager, "vs_compiled_rebuild": graph}


def timed(model, network, data, autocast_dtype, flags: str, iters: int, warmup: int, invalidate: bool) -> dict:
    """Median time of a micro-step under `flags`. invalidate: the cache is rebuilt before every step (the first
    micro-step after an optimizer step) instead of staying current (the other micro-steps of an accumulation)."""
    set_flags(flags, check=False, release=[model])
    cuda = data[0].is_cuda
    stored = []
    real = attention._flash_forward_with_lse

    def spy(q, k, v):
        out, lse = real(q, k, v)
        stored.append(out.numel() * out.element_size() + lse.numel() * lse.element_size())
        return out, lse

    times = []
    for i in range(warmup + iters):
        if invalidate:
            lycoris_fast.ckpt_merged_invalidate()
        if i == warmup and cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        micro_step(model, network, data, autocast_dtype)
        if cuda:
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - start)
    out = {"median_ms": statistics.median(times) * 1e3, "min_ms": min(times) * 1e3, "max_ms": max(times) * 1e3,
           "merged_cache_bytes": lycoris_fast.ckpt_merged_bytes(model)}  # what this configuration holds (no cache: 0)
    if cuda:
        out["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    if CONFIGS[flags][1] and not getattr(model, "_block_compile", False):
        attention._flash_forward_with_lse = spy  # blocks run in Python only: a compiled graph does not pass through it
        try:
            micro_step(model, network, data, autocast_dtype)
        finally:
            attention._flash_forward_with_lse = real
        out["stored_attention_bytes"] = sum(stored)  # under a budget: of the blocks that reuse
        out["blocks_storing"] = len(stored)
        if stored and flags not in BUDGETED:
            out["stored_attention_gib_for_42_blocks"] = stored[0] * 42 / 2**30
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--tokens", type=int, nargs="+", default=None, help="image tokens per sample (square grids)")
    parser.add_argument("--text_tokens", type=int, default=None, help="text tokens of the joint sequence (default 256)")
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--exclude", type=int, nargs="*", default=[], help="joint blocks left out of the network")
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--baseline_runs", type=int, default=3,
                        help="flags-off micro-steps repeated for the off vs off noise floor (0 = no baseline)")
    parser.add_argument("--backend", default=None, help="torch.compile backend of the compiled mode (inductor on cuda)")
    parser.add_argument("--modes", nargs="+", default=["eager", "chains", "compiled"], choices=["eager", "chains", "compiled"],
                        help="eager: nothing compiled; chains: eager blocks + the compiled elementwise chains and LoKr+DoRA"
                             " rebuild of a default CUDA run without --moss_block_compile; compiled: block compile")
    parser.add_argument("--budget_blocks", type=int, default=None,
                        help="joint blocks that reuse in the budget configurations (default: half of --blocks)")
    parser.add_argument("--mixed_precision", default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--stream_fp32", action="store_true", help="--moss_stream_fp32")
    parser.add_argument("--dit", default=None, help="a MOSS-Image DiT .safetensors: its first joint blocks (else random)")
    parser.add_argument("--vram_cap_gb", type=float, default=12.0)
    parser.add_argument("--busy_gb", type=float, default=6.0)
    parser.add_argument("--json", default=None, help="also write the report to this file")
    args = parser.parse_args()

    cuda = args.device == "cuda"
    device = torch.device(args.device)
    cfg = PROD if cuda else TOY
    tokens = args.tokens or ([1024, 16384] if cuda else [16, 64])
    text_tokens = args.text_tokens or (256 if cuda else 6)
    backend = args.backend or ("inductor" if cuda else "aot_eager")
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    stream_dtype = torch.float32 if args.stream_fp32 else autocast_dtype  # the noisy input and the text (the trainer's)
    if cuda:
        if not attention.is_flash_attn_supported() or not attention.flash_reuse_available():
            raise SystemExit("flash-attn with the separate backward op is needed (attn_mode flash)")
        free, total = torch.cuda.mem_get_info()
        if (total - free) / 2**30 > args.busy_gb:
            raise SystemExit(f"{(total - free) / 2**30:.1f} GiB of the GPU is in use (> --busy_gb {args.busy_gb}): not"
                             " starting next to another job")
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * 2**30 / total))
    else:
        from tests import _fake_flash  # the CPU stand-in for flash-attn (tests/_fake_flash.py)

        attention.flash_attn_func = _fake_flash.flash_attn_func
        attention._wrapped_flash_attn_backward = _fake_flash.wrapped_flash_attn_backward
        torch.backends.mkldnn.enabled = False  # oneDNN has no bf16 backward on some CPUs

    lycoris_fast.install_fast_lokr_forward()
    laf.apply_module_patches()
    register_moss_lycoris_presets()
    budget_blocks = args.blocks // 2 if args.budget_blocks is None else args.budget_blocks
    if not 0 <= budget_blocks <= args.blocks:
        raise SystemExit(f"--budget_blocks must be in 0..{args.blocks} (--blocks)")
    report = {"device": args.device, "torch": torch.__version__, "backend": backend, "autocast": str(autocast_dtype),
              "stream": str(stream_dtype), "blocks": args.blocks, "exclude": list(args.exclude),
              "text_tokens": text_tokens, "weights": args.dit or "random", "budget_blocks": budget_blocks, "runs": []}
    eager_off = {}  # per token count: the eager flags-off result (CPU copies) - a compile's own difference is read off it
    for mode in args.modes:
        # chains: as prepare_moss_runtime leaves a default CUDA run without --moss_block_compile (both compiles are
        # CUDA-only: on the CPU self-test this mode equals eager); eager / compiled: both off (under --moss_block_compile
        # they are inside the block graphs)
        moss_models.enable_elementwise_compile(mode == "chains")
        lycoris_fast.enable_compiled_rebuild(mode == "chains")
        models = {}
        for blocks in sorted({1, args.blocks}):
            torch._dynamo.reset()
            model, network = build(cfg, blocks, device, autocast_dtype, args.exclude, args.stream_fp32, args.dit)
            if mode == "compiled":
                mbc.compile_moss_blocks(model, backend=backend, recompile_limit=64, isolate_cache=False,
                                        stream_fp32=args.stream_fp32)
            models[blocks] = (model, network)
        model, network = models[args.blocks]
        for n in tokens:
            data = batch(cfg, n, text_tokens, device, stream_dtype)
            set_budget(budget_gib_for(cfg, n + text_tokens, budget_blocks))  # the BUDGETED configurations at this size
            run = {"mode": mode, "tokens": n, "joint_tokens": n + text_tokens, "correctness": {}, "timing": {},
                   "budget_gib": _BUDGET_GIB, "budget_blocks": budget_blocks}
            results = {}
            for flags in CONFIGS:
                set_flags(flags, release=[model])
                micro_step(model, network, data, autocast_dtype)  # compiles / builds the cache
                results[flags] = micro_step(model, network, data, autocast_dtype)
                if flags == "off":
                    if args.baseline_runs > 0:  # the same flags-off micro-step again: what "differing" means below
                        repeats = [micro_step(model, network, data, autocast_dtype) for _ in range(args.baseline_runs)]
                        run["correctness"][BASELINE] = noise_floor(results["off"], repeats)
                        run["correctness"][BASELINE]["l2"] = worst_l2([l2_compare(results["off"], r) for r in repeats])
                        del repeats
                else:
                    diffs = anima_models.CKPT_RECOMPUTE_DIFFS  # one per node block (a budget: fewer than --blocks)
                    run["correctness"][flags] = {**compare_results(results["off"], results[flags]),
                                                 "l2": l2_compare(results["off"], results[flags]),
                                                 "recompute_vs_first_pass_max_abs": max(diffs, default=None),
                                                 "recomputes": len(diffs)}
                    if BASELINE in run["correctness"]:
                        run["correctness"][flags]["reading"] = reading(run["correctness"][flags],
                                                                       run["correctness"][BASELINE])
            if mode == "eager":
                eager_off[n] = to_cpu(results["off"])
            elif n in eager_off:  # chains / compiled, after the eager mode of this run: what the compile itself changes
                own_key, entry_key, what = OWN_KEYS[mode]
                compiled_off = to_cpu(results["off"])
                own = l2_compare(eager_off[n], compiled_off)
                run[own_key] = {**compare_results(eager_off[n], compiled_off), "l2": own}
                for flags in CONFIGS:
                    if flags != "off":
                        run["correctness"][flags][entry_key] = scale_reading(run["correctness"][flags]["l2"], own, what)
                del compiled_off
            del results
            if mode != "compiled":  # the blocks run in Python: the attention call of each pass can be watched
                run["qkv_first_pass_vs_recompute"] = qkv_passes(model, network, data, autocast_dtype)
                run["qkv_first_pass_vs_recompute_reuse_only"] = qkv_passes(model, network, data, autocast_dtype, REUSE_ONLY)
            for flags in CONFIGS:
                for other, _ in models.values():  # the reported peak is this model's: no other model's cache next to it
                    if other is not model:
                        lycoris_fast.ckpt_merged_release(other)
                run["timing"][flags] = timed(model, network, data, autocast_dtype, flags, args.iters, args.warmup, False)
                if CONFIGS[flags][0]:  # the cached configurations: the first micro-step after an optimizer step
                    run["timing"][flags + " (cache rebuilt every step)"] = timed(
                        model, network, data, autocast_dtype, flags, args.iters, args.warmup, True)
                if args.blocks > 1 and flags not in BUDGETED:  # the same with 1 joint block: the difference is
                    # (blocks - 1) blocks (not under a budget: there the blocks differ in kind, and the 1-block model
                    # would get another budget share)
                    one = timed(*models[1], data, autocast_dtype, flags, args.iters, args.warmup, False)
                    run["timing"][flags]["per_block_ms"] = (run["timing"][flags]["median_ms"] - one["median_ms"]) / (
                        args.blocks - 1)
            if args.blocks > 1:
                base = run["timing"]["off"]["per_block_ms"]
                run["per_block_saving_ms"] = {f: base - run["timing"][f]["per_block_ms"] for f in CONFIGS
                                              if f != "off" and f not in BUDGETED}
            report["runs"].append(run)
            print(json.dumps(run, indent=1))
        set_flags("nograd", check=False)  # the cache on for the merged-weight checks (the last configuration timed is budgeted)
        report.setdefault("merged_weight", {})[mode] = merged_weight_checks(model, autocast_dtype, backend)
        print(json.dumps({"merged_weight": report["merged_weight"][mode]}, indent=1))
        if mode == "compiled":
            state = model._block_compile_state
            report["compiled_calls"], report["eager_calls"] = state.compiled_calls, state.eager_calls
        set_flags("off", check=False)
        set_budget(None)
        for m, _ in models.values():
            lycoris_fast.ckpt_merged_release(m)
        del models, model, network
        if cuda:
            torch.cuda.empty_cache()
    moss_models.enable_elementwise_compile(False)
    lycoris_fast.enable_compiled_rebuild(False)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print(f"report written to {os.path.abspath(args.json)}")


if __name__ == "__main__":
    main()
