"""Reviewer GPU check for --anima_ckpt_reuse_attn (2026-11-14), kernel level: NOT collected by pytest and NOT run on the
GPU yet (the reviewer had no go-ahead for GPU work; only the CPU self-test below has run).

What tests/manual_test_anima_ckpt_reuse_gpu.py does not separate: flash-attn's default backward kernel is documented as
non-deterministic (flash_attn_func's `deterministic` argument), so "flags on vs off, N elements differ" from that harness
can be kernel noise or a real difference. This script measures both on the bare kernels, no model:
* out of attention._flash_forward_with_lse == out of flash_attn_func(q, k, v, 0.0), bit for bit (the stored output is the
  plain call's);
* the run-to-run noise of the plain flash backward (same q, k, v and cotangent, --repeats times): max |d| and the number
  of differing elements of dq, dk, dv - the floor any on / off comparison has to be read against;
* the reuse path (attention._FlashReuse: the stored out and softmax_lse, only the backward kernel) against the plain
  path: its dq, dk, dv must sit inside that floor (bit-identical when the floor is zero).

VRAM: under 2 GiB at the default sizes (16384 tokens x 16 heads x 128: 67 MB per fp16 tensor; q, k, v, the cotangent,
one reference and one current set of gradients are alive at a time; the process is capped at --vram_cap_gb, default 3).
A few seconds. It refuses to start when other processes hold more than --busy_gb of the GPU.

Run (GPU):       venv_t214/Scripts/python.exe -B tests/manual_test_flash_reuse_kernel_gpu.py
Self-test (CPU): venv_t214/Scripts/python.exe -B tests/manual_test_flash_reuse_kernel_gpu.py --device cpu
MOSS-Image's joint attention (24 heads x 128, bf16, a 2048^2 image + 256 text tokens; every tensor is 1.5x the default's,
so the cap is raised):  ... --heads 24 --head_dim 128 --mixed_precision bf16 --tokens 1024 4096 16640 --vram_cap_gb 6
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from library import attention  # noqa: E402


def stats(ref, new) -> dict:
    """Worst case over a tuple of tensors: max |d|, max |d| / max |ref|, differing elements."""
    out = {"max_abs": 0.0, "max_rel": 0.0, "differing": 0, "of": 0}
    for a, b in zip(ref, new):
        d = (a.float() - b.float()).abs()
        out["max_abs"] = max(out["max_abs"], float(d.max()))
        out["max_rel"] = max(out["max_rel"], float(d.max() / a.float().abs().max().clamp_min(1e-30)))
        out["differing"] += int((a != b).sum())
        out["of"] += a.numel()
    return out


def worst(items) -> dict:
    items = list(items)
    return {"max_abs": max(i["max_abs"] for i in items), "max_rel": max(i["max_rel"] for i in items),
            "differing": max(i["differing"] for i in items), "of": items[0]["of"]}


def plain(q, k, v, cot):
    out = attention.flash_attn_func(q, k, v, 0.0)
    return out.detach(), torch.autograd.grad((out.float() * cot).sum(), (q, k, v))


def reuse(q, k, v, cot):
    with torch.no_grad():  # the checkpoint's first pass
        stored_out, softmax_lse = attention._flash_forward_with_lse(q, k, v)
    out = attention._FlashReuse.apply(q, k, v, stored_out, softmax_lse)  # the recompute
    return stored_out.detach(), torch.autograd.grad((out.float() * cot).sum(), (q, k, v))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--tokens", type=int, nargs="+", default=None)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head_dim", type=int, default=128)  # Anima 2.9B: 2048 / 16
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--mixed_precision", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--busy_gb", type=float, default=6.0)
    parser.add_argument("--vram_cap_gb", type=float, default=3.0,
                        help="per-process VRAM cap (the default sizes stay under 2 GiB; more heads / tokens need more)")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    cuda = args.device == "cuda"
    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    tokens = args.tokens or ([1024, 4096, 16384] if cuda else [16, 64])
    if cuda:
        free, total = torch.cuda.mem_get_info()
        if (total - free) / 2**30 > args.busy_gb:
            raise SystemExit(f"{(total - free) / 2**30:.1f} GiB of the GPU is in use (> --busy_gb {args.busy_gb}): not"
                             " starting next to another job")
        # the default sizes stay under 2 GiB (cap 3)
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * 2**30 / total))
        if attention.flash_attn_func is None or attention._wrapped_flash_attn_backward is None:
            raise SystemExit("flash-attn (with _wrapped_flash_attn_backward) is not usable here")
    else:
        from tests import _fake_flash  # the CPU stand-in (deterministic: the floor is zero)

        attention.flash_attn_func = _fake_flash.flash_attn_func
        attention._wrapped_flash_attn_backward = _fake_flash.wrapped_flash_attn_backward
        args.heads, args.head_dim = 2, 8

    report = {"device": args.device, "torch": torch.__version__, "dtype": str(dtype), "runs": []}
    for n in tokens:
        g = torch.Generator().manual_seed(n)
        shape = (1, n, args.heads, args.head_dim)
        # q / k after the Block's RMS norm have unit RMS per head; v is a Linear output
        q, k, v = (torch.randn(shape, generator=g).to(args.device, dtype).requires_grad_(True) for _ in range(3))
        cot = torch.randn(shape, generator=g).to(args.device)
        ref_out, ref_grads = plain(q, k, v, cot)
        # one run alive at a time (a set of gradients is 3 x 67 MB at 16384 tokens): only the statistics are kept
        plain_noise, stored, reuse_diff, reuse_noise, first_reuse = [], [], [], [], None
        for _ in range(args.repeats):
            plain_noise.append(stats(ref_grads, plain(q, k, v, cot)[1]))
        for _ in range(args.repeats):
            out, grads = reuse(q, k, v, cot)
            stored.append(stats((ref_out,), (out,)))
            reuse_diff.append(stats(ref_grads, grads))
            if first_reuse is None:
                first_reuse = grads
            else:
                reuse_noise.append(stats(first_reuse, grads))
            del out, grads
        run = {
            "tokens": n,
            "stored_out_vs_plain_out": worst(stored),
            "plain_backward_run_to_run": worst(plain_noise),
            "reuse_backward_vs_plain": worst(reuse_diff),
            "reuse_backward_run_to_run": worst(reuse_noise) if reuse_noise else None,
            "finite": bool(all(torch.isfinite(t).all() for t in ref_grads)),
        }
        floor, got = run["plain_backward_run_to_run"], run["reuse_backward_vs_plain"]
        run["verdict"] = ("stored output differs from the plain call" if run["stored_out_vs_plain_out"]["differing"] else
                          "bit-identical" if not got["differing"] else
                          "inside the kernel's own run-to-run noise" if got["max_abs"] <= 2 * floor["max_abs"] else
                          "OUTSIDE the kernel's run-to-run noise: do not enable --anima_ckpt_reuse_attn")
        report["runs"].append(run)
        print(json.dumps(run, indent=1))
        del q, k, v, cot, ref_out, ref_grads, first_reuse
    if cuda:
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        print(f"peak allocated: {report['peak_allocated_gib']:.2f} GiB")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print(f"report written to {os.path.abspath(args.json)}")


if __name__ == "__main__":
    main()
