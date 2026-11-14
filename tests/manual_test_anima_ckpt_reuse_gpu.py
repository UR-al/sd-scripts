"""GPU harness for --anima_ckpt_nograd_merged / --anima_ckpt_reuse_attn (2026-11-14): what the CPU tests
(tests/test_anima_ckpt_reuse.py) cannot answer - the real flash-attn kernels, inductor's compiled Block graphs, VRAM and
time. NOT collected by pytest (run it by hand); it loads no checkpoint unless --dit is given and runs no training.

What it builds: a DiT with Anima 2.9B's Block shapes (2048 wide, 16 heads, context 1024, adaLN-LoRA 256) and --blocks
Blocks (default 2; weights random, or Blocks 0..N-1 of --dit), fp16 with the fp32 stream (--anima_stream_fp32), the
production LyCORIS network (LoKr factor 4 full matrix + DoRA on the output axis, preset anima-exp, the fast forward and
the fixed paths), gradient checkpointing on every Block, attn_mode flash. Eager and per-block compiled (--backend).

What it reports, per token count (--tokens, default 1024 and 16384 = a 2048-tier image) and per mode (eager / compiled):
* correctness, flags on vs off: max |d|, max |d| / max |ref| and the count of differing elements of the output and of
  every LoKr / DoRA gradient (0 = bit-identical); max |recompute output - first-pass output| over the Blocks
  (anima_models.CKPT_RECOMPUTE_DIFFS); eager only: whether the recompute's q, k, v equal the first pass's bit for bit;
* the baseline those numbers are read against, flags off vs flags off (correctness["off_vs_off"]): the same micro-step
  repeated --baseline_runs times with both flags off, each compared with the flags-off reference, the worst of every
  number. flash-attn's default backward is documented as non-deterministic, so on the GPU a non-zero "differing" of
  flags on vs off is only a finding when it is larger than this run-to-run noise; every flags-on entry carries the
  comparison as "reading" (bit-identical / inside the flags-off run-to-run noise / OUTSIDE it). OUTSIDE means a
  reproducible difference between the two code paths (e.g. inductor rounding the no-grad and the grad graph
  differently), not kernel noise; how large it is against what block compile itself changes is another measurement;
* the cached merged weight against a fresh eager rebuild and against the same rebuild torch.compile'd on its own (a
  proxy for the weight a compiled Block graph builds in place; the end-to-end answer is the compiled mode's output
  comparison, flags on vs off);
* time per micro-step (forward + backward, median of --iters after --warmup) for off / nograd / nograd+reuse, with the
  cache current (micro-steps 2..N of an optimizer step) and for the first micro-step after an optimizer step (cache
  rebuilt), and the per-Block saving (the model is timed with --blocks and with 1 Block: the difference / (blocks - 1));
* VRAM: peak allocated per configuration, the cache's bytes and the stored attention's bytes.

VRAM: stays under ~7 GiB at --blocks 2 --tokens 16384 (the process is capped at --vram_cap_gb, default 8); it refuses to
start when other processes already hold more than --busy_gb of the GPU (a training run).

Run (GPU):  venv_t214/Scripts/python.exe -B tests/manual_test_anima_ckpt_reuse_gpu.py
Self-test (CPU, toy shapes, the fake flash kernels, aot_eager):  ... tests/manual_test_anima_ckpt_reuse_gpu.py --device cpu
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from library import anima_block_compile as bc  # noqa: E402
from library import anima_models, attention, lycoris_args, lycoris_fast  # noqa: E402
from library import lycoris_algo_fixes as laf  # noqa: E402

NETWORK_ARGS = ["algo=lokr", "preset=anima-exp", "factor=4", "full_matrix=True", "decompose_both=False", "dora_wd=True",
                "wd_on_output=True"]
PROD = dict(max_img_h=512, max_img_w=512, max_frames=128, in_channels=16, out_channels=16, patch_spatial=2,
            patch_temporal=1, model_channels=2048, concat_padding_mask=True, crossattn_emb_channels=1024,
            pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop", min_fps=1, max_fps=30,
            use_adaln_lora=True, adaln_lora_dim=256, num_heads=16, extra_per_block_abs_pos_emb=False,
            rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
            use_llm_adapter=False, attn_mode="flash", split_attn=False)
TOY = dict(max_img_h=32, max_img_w=32, max_frames=1, in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
           model_channels=64, num_heads=2, mlp_ratio=4.0, crossattn_emb_channels=32, use_adaln_lora=True,
           adaln_lora_dim=16, use_llm_adapter=False, attn_mode="flash", pos_emb_cls="rope3d")
FLAGS = {"off": (False, False), "nograd": (True, False), "nograd+reuse": (True, True)}


def build(cfg: dict, blocks: int, device: torch.device, dtype: torch.dtype, dit: str = None, seed: int = 0):
    """The DiT (frozen, `dtype` with the fp32 stream, train mode, every Block checkpointed) and the LoKr+DoRA network,
    applied the way train_network.py does it (tests/test_anima_block_compile.build)."""
    import lycoris.kohya as kohya

    torch.manual_seed(seed)
    unet = anima_models.Anima(**{**cfg, "num_blocks": blocks})
    if dit:
        load_blocks(unet, dit, blocks)
    unet = unet.to(device=device, dtype=dtype)
    unet.requires_grad_(False)
    unet.train()
    args = argparse.Namespace(network_args=list(NETWORK_ARGS), network_dim=10000, network_alpha=10000,
                              network_module="lycoris.kohya", network_dropout=None, lycoris_stock=False,
                              lycoris_algo_fixes=False, network_weights=None)
    kwargs = lycoris_args.fix_lycoris_bool_kwargs(args.network_module, dict(i.split("=", 1) for i in args.network_args))
    create = laf.prepare_network_kwargs("lokr", dict(kwargs))
    create.setdefault("dropout", 0.0)
    laf.set_block_compile(True)  # the adapted forwards re-bound through closures (needed by the compiled Blocks, harmless eager)
    network = kohya.create_network(1.0, args.network_dim, args.network_alpha, None, None, unet, neuron_dropout=None, **create)
    if hasattr(network, "prepare_network"):
        network.prepare_network(args)
    network.apply_to(None, unet, False, True)
    laf.after_network_created(network, args)
    network.to(device)
    network.requires_grad_(True)
    network.train()
    g = torch.Generator().manual_seed(seed + 7)
    with torch.no_grad():  # LoKr starts as a zero delta (lokr_w2 = 0): move every factor, as after some training
        for p in network.parameters():
            p.add_((torch.randn(p.shape, generator=g) * 0.02).to(p))
    unet.enable_stream_fp32()
    unet.enable_gradient_checkpointing()
    laf.prime_module_caches(network)
    return unet, network


def load_blocks(unet, path: str, blocks: int) -> None:
    """Blocks 0..blocks-1 of a real Anima DiT checkpoint into the model (the embedders stay random)."""
    from safetensors import safe_open

    loaded = 0
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        prefix = "net." if any(k.startswith("net.blocks.") for k in keys) else ""
        state = unet.state_dict()
        for name in state:
            if name.startswith("blocks.") and int(name.split(".")[1]) < blocks and prefix + name in keys:
                state[name].copy_(f.get_tensor(prefix + name).to(state[name].dtype))
                loaded += 1
    print(f"loaded {loaded} Block tensors from {path}")


def batch(cfg: dict, tokens: int, device, seed: int = 1):
    """Latents of a square grid with `tokens` DiT tokens, a timestep, a 512-token context and the padding mask."""
    side = int(round(tokens ** 0.5)) * cfg["patch_spatial"]
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, cfg["in_channels"], 1, side, side, generator=g).to(device)  # fp32: the stream is
    t = torch.tensor([0.3], device=device)
    ctx = torch.randn(1, 512 if cfg is PROD else 8, cfg["crossattn_emb_channels"], generator=g).to(device)
    pad = torch.zeros(1, 1, side, side, device=device)
    w = torch.randn(1, cfg["out_channels"], 1, side, side, generator=g).to(device)
    return x, t, ctx, pad, w


def micro_step(unet, network, data, autocast_dtype):
    """One micro-step as the trainer runs it (the noisy input marked for the checkpoint); returns the output and grads."""
    x, t, ctx, pad, w = data
    network.zero_grad(set_to_none=True)
    laf.set_timestep(network, t.float() * 1000.0)
    x = x.detach().requires_grad_(True)
    with torch.autocast(x.device.type, dtype=autocast_dtype):
        y = unet(x, t, ctx, padding_mask=pad)
    (y.float() * w).mean().backward()
    return y.detach().float(), {n: p.grad.detach().clone() for n, p in network.named_parameters() if p.grad is not None}


def set_flags(name: str, check: bool = True) -> None:
    anima_models.CKPT_NOGRAD_MERGED, anima_models.CKPT_REUSE_ATTN = FLAGS[name]
    anima_models.CKPT_RECOMPUTE_DIFFS = [] if (check and FLAGS[name][0]) else None


def compare(ref: torch.Tensor, new: torch.Tensor) -> dict:
    d = (ref.float() - new.float()).abs()
    return {"max_abs": float(d.max()), "max_rel": float(d.max() / ref.float().abs().max().clamp_min(1e-30)),
            "differing": int((ref != new).sum()), "of": ref.numel()}


def compare_results(ref, new) -> dict:
    (y0, g0), (y1, g1) = ref, new
    worst = {"max_abs": 0.0, "max_rel": 0.0, "differing": 0, "of": 0, "tensors_differing": 0, "worst": None}
    for name in g0:
        c = compare(g0[name], g1[name])
        worst["differing"] += c["differing"]
        worst["of"] += c["of"]
        worst["tensors_differing"] += c["differing"] > 0
        if c["max_rel"] > worst["max_rel"]:
            worst.update(max_rel=c["max_rel"], worst=name)
        worst["max_abs"] = max(worst["max_abs"], c["max_abs"])
    return {"output": compare(y0, y1), "grads": worst, "grad_tensors": len(g0), "same_keys": g0.keys() == g1.keys()}


BASELINE = "off_vs_off"  # the correctness entry of the flags-off run-to-run noise


def noise_floor(ref, repeats: list) -> dict:
    """compare_results of every repeat with `ref` (all of them flags off: the run-to-run noise of the kernels), the worst
    of each number over the repeats."""
    floor = compare_results(ref, repeats[0])
    for new in repeats[1:]:
        c = compare_results(ref, new)
        if c["grads"]["max_rel"] > floor["grads"]["max_rel"]:
            floor["grads"]["worst"] = c["grads"]["worst"]
        for part, keys in (("output", ("max_abs", "max_rel", "differing")),
                           ("grads", ("max_abs", "max_rel", "differing", "tensors_differing"))):
            for key in keys:
                floor[part][key] = max(floor[part][key], c[part][key])
        floor["same_keys"] = floor["same_keys"] and c["same_keys"]
    floor["runs"] = len(repeats)
    return floor


def reading(result: dict, floor: dict) -> str:
    """A flags-on vs flags-off comparison read against the flags-off run-to-run noise (the rule of
    tests/manual_test_flash_reuse_kernel_gpu.py: inside = at most twice the noise's max |d|, output and gradients)."""
    if result["output"]["differing"] == 0 and result["grads"]["differing"] == 0:
        return "bit-identical"
    if all(result[part]["max_abs"] <= 2 * floor[part]["max_abs"] for part in ("output", "grads")):
        return "inside the flags-off run-to-run noise"
    return "OUTSIDE the flags-off run-to-run noise: a reproducible difference between the two code paths"


def qkv_passes(unet, network, data, autocast_dtype) -> dict:
    """Eager, nograd+reuse: the self-attention q, k, v of the first pass and of the recompute, compared bit for bit."""
    seen = {1: [], 2: []}
    real = attention.attention

    def spy(qkv, **kwargs):
        if lycoris_fast._CKPT_PASS and (kwargs.get("ckpt_stash") is not None or kwargs.get("ckpt_reuse") is not None):
            seen[lycoris_fast._CKPT_PASS].append([t.detach().clone() for t in qkv])
        return real(qkv, **kwargs)

    attention.attention = spy
    try:
        set_flags("nograd+reuse")
        micro_step(unet, network, data, autocast_dtype)
    finally:
        attention.attention = real
    first, second = seen[1], seen[2][::-1]  # backward visits the Blocks last to first
    out = {"blocks": len(first), "recomputes": len(second)}
    for i, name in enumerate("qkv"):
        diffs = [compare(a[i], b[i]) for a, b in zip(first, second)]
        out[name] = {"bit_identical": all(d["differing"] == 0 for d in diffs),
                     "max_abs": max((d["max_abs"] for d in diffs), default=None),
                     "differing": sum(d["differing"] for d in diffs)}
    return out


def merged_weight_checks(unet, autocast_dtype, backend: str) -> dict:
    """The cached merged weights against (a) a fresh eager rebuild and (b) the same rebuild inside a torch.compile'd
    function (what a compiled Block graph computes in its recompute and what the flag-off first pass computed)."""
    lokrs = [l for b in unet.blocks for l in anima_models._ckpt_cached_lokrs(b)]
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
    return {"modules": len(lokrs), "bytes": lycoris_fast.ckpt_merged_bytes(unet), "vs_eager_rebuild": eager,
            "vs_compiled_rebuild": graph}


def timed(unet, network, data, autocast_dtype, flags: str, iters: int, warmup: int, invalidate: bool) -> dict:
    """Median time of a micro-step under `flags`. invalidate: the cache is rebuilt before every step (the first
    micro-step after an optimizer step) instead of staying current (the other micro-steps of an accumulation)."""
    set_flags(flags, check=False)
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
        micro_step(unet, network, data, autocast_dtype)
        if cuda:
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - start)
    out = {"median_ms": statistics.median(times) * 1e3, "min_ms": min(times) * 1e3, "max_ms": max(times) * 1e3}
    if cuda:
        out["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    if flags == "nograd+reuse" and not getattr(unet, "_block_compile", False):
        attention._flash_forward_with_lse = spy  # eager only: a compiled graph does not pass through this function
        try:
            micro_step(unet, network, data, autocast_dtype)
        finally:
            attention._flash_forward_with_lse = real
        out["stored_attention_bytes"] = sum(stored)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--tokens", type=int, nargs="+", default=None, help="DiT tokens per sample (square grids)")
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--baseline_runs", type=int, default=3,
                        help="flags-off micro-steps repeated for the off vs off noise floor (0 = no baseline)")
    parser.add_argument("--backend", default=None, help="torch.compile backend of the compiled mode (inductor on cuda)")
    parser.add_argument("--modes", nargs="+", default=["eager", "compiled"], choices=["eager", "compiled"])
    parser.add_argument("--mixed_precision", default=None, choices=["fp16", "bf16"])
    parser.add_argument("--dit", default=None, help="an Anima DiT .safetensors: its first Blocks' weights (else random)")
    parser.add_argument("--vram_cap_gb", type=float, default=8.0)
    parser.add_argument("--busy_gb", type=float, default=6.0)
    parser.add_argument("--json", default=None, help="also write the report to this file")
    args = parser.parse_args()

    cuda = args.device == "cuda"
    device = torch.device(args.device)
    cfg = PROD if cuda else TOY
    tokens = args.tokens or ([1024, 16384] if cuda else [16, 64])
    backend = args.backend or ("inductor" if cuda else "aot_eager")
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, None: torch.float16 if cuda else torch.bfloat16}[
        args.mixed_precision]
    if cuda:
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
    report = {"device": args.device, "torch": torch.__version__, "backend": backend, "autocast": str(autocast_dtype),
              "blocks": args.blocks, "weights": args.dit or "random", "runs": []}
    for mode in args.modes:
        models = {}
        for blocks in sorted({1, args.blocks}):
            torch._dynamo.reset()
            unet, network = build(cfg, blocks, device, autocast_dtype, args.dit)
            if mode == "compiled":
                bands = bc.plan_bands(tokens, "union")
                bc.compile_blocks(unet, bands=bands, backend=backend, dynamic="union", recompile_limit=128,
                                  isolate_cache=False, stream_fp32=True)
            models[blocks] = (unet, network)
        unet, network = models[args.blocks]
        for n in tokens:
            data = batch(cfg, n, device)
            run = {"mode": mode, "tokens": n, "correctness": {}, "timing": {}}
            results = {}
            for flags in FLAGS:
                set_flags(flags)
                micro_step(unet, network, data, autocast_dtype)  # compiles / builds the cache
                results[flags] = micro_step(unet, network, data, autocast_dtype)
                if flags == "off":
                    if args.baseline_runs > 0:  # the same flags-off micro-step again: what "differing" means below
                        repeats = [micro_step(unet, network, data, autocast_dtype) for _ in range(args.baseline_runs)]
                        run["correctness"][BASELINE] = noise_floor(results["off"], repeats)
                else:
                    diffs = anima_models.CKPT_RECOMPUTE_DIFFS
                    run["correctness"][flags] = {**compare_results(results["off"], results[flags]),
                                                 "recompute_vs_first_pass_max_abs": max(diffs), "recomputes": len(diffs)}
                    if BASELINE in run["correctness"]:
                        run["correctness"][flags]["reading"] = reading(run["correctness"][flags],
                                                                       run["correctness"][BASELINE])
            if mode == "eager":
                run["qkv_first_pass_vs_recompute"] = qkv_passes(unet, network, data, autocast_dtype)
            for flags in FLAGS:
                run["timing"][flags] = timed(unet, network, data, autocast_dtype, flags, args.iters, args.warmup, False)
                if flags != "off":
                    run["timing"][flags + " (cache rebuilt every step)"] = timed(
                        unet, network, data, autocast_dtype, flags, args.iters, args.warmup, True)
                if args.blocks > 1:  # the same with 1 Block: the difference is (blocks - 1) Blocks
                    one = timed(*models[1], data, autocast_dtype, flags, args.iters, args.warmup, False)
                    run["timing"][flags]["per_block_ms"] = (run["timing"][flags]["median_ms"] - one["median_ms"]) / (
                        args.blocks - 1)
            if args.blocks > 1:
                base = run["timing"]["off"]["per_block_ms"]
                run["per_block_saving_ms"] = {f: base - run["timing"][f]["per_block_ms"] for f in FLAGS if f != "off"}
            report["runs"].append(run)
            print(json.dumps(run, indent=1))
        report.setdefault("merged_weight", {})[mode] = merged_weight_checks(unet, autocast_dtype, backend)
        print(json.dumps({"merged_weight": report["merged_weight"][mode]}, indent=1))
        if mode == "compiled":
            state = unet._block_compile_state
            report["compiled_calls"], report["eager_calls"] = state.compiled_calls, state.eager_calls
        set_flags("off", check=False)
        for u, _ in models.values():
            lycoris_fast.ckpt_merged_release(u)
        del models, unet, network
        if cuda:
            torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print(f"report written to {os.path.abspath(args.json)}")


if __name__ == "__main__":
    main()
