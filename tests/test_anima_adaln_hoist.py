"""CPU tests for the AdaLN hoist (library/anima_models.py ADALN_HOIST; opt-in --anima_adaln_hoist, default off): a
checkpointed Block's three adaLN modulations are computed outside the checkpoint and passed in, so their six Linears run
forward once and backward once instead of forward + recompute + backward. The tests set the module flag themselves.
Checks: outputs and every gradient bit-identical to the unhoisted path (no checkpoint,
torch_checkpoint(use_reentrant=False), cpu-offload torch_checkpoint, UnslothOffloadedGradientCheckpointer), the Unsloth
Function gives gradients to its extra tensor arguments, the Linears are not called in the recompute, no hoist with block
swap / eval / no_grad, and the aten op count of a 2-Block forward + backward drops.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe tests/test_anima_adaln_hoist.py   (or with pytest)
"""
import types
import warnings

import torch

from library import anima_models, attention
from library.anima_models import Block, UnslothOffloadedGradientCheckpointer

D, CTX, HEADS, B, T, H, W = 64, 48, 4, 2, 1, 3, 4  # 2 samples (2 timesteps), T=1 like the trainer
CKPT_MODES = ("none", "torch", "cpu_offload", "unsloth")


def _bf16_backward_supported() -> bool:
    """oneDNN on some CPUs has no bf16 backward for the Block's ops (RuntimeError 'DNNL does not support bf16/f16
    backward on the platform with avx2_vnni_2'); probed with a real Block, as a lone small Linear can still pass."""
    try:
        _run_blocks(_blocks(1, False, torch.bfloat16), "none", False, False, False, torch.bfloat16, autocast=True)
        return True
    except RuntimeError as error:
        assert "bf16" in str(error), error
        return False


def _set_checkpointing(blocks, mode: str):
    for b in blocks:
        if mode == "none":
            b.disable_gradient_checkpointing()
        else:
            b.enable_gradient_checkpointing(cpu_offload=mode == "cpu_offload", unsloth_offload=mode == "unsloth")


def _blocks(n: int, use_adaln_lora: bool, dtype=torch.float32):
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList(
        Block(x_dim=D, context_dim=CTX, num_heads=HEADS, use_adaln_lora=use_adaln_lora, adaln_lora_dim=16) for _ in range(n)
    )
    with torch.no_grad():
        for p in blocks.parameters():
            p.normal_(0, 0.05)
    return blocks.to(dtype).train()


def _inputs(dtype, use_adaln_lora: bool):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(B, T, H, W, D, generator=g).to(dtype).requires_grad_(True)
    emb_src = torch.randn(B, T, D, generator=g).to(dtype).requires_grad_(True)  # one row per timestep
    lora_src = torch.randn(B, T, 3 * D, generator=g).to(dtype).requires_grad_(True) if use_adaln_lora else None
    crossattn = torch.randn(B, 7, CTX, generator=g).to(dtype)
    rope = (torch.randn(12, 1, 1, 16, generator=g).to(dtype), torch.randn(12, 1, 1, 16, generator=g).to(dtype))
    extra = torch.randn(B, T, H, W, D, generator=g).to(dtype)
    return x, emb_src, lora_src, crossattn, rope, extra


def _run_blocks(blocks, mode: str, hoist: bool, use_fp32: bool, use_adaln_lora: bool, dtype=torch.float32, autocast=False,
                backward=True):
    """The block loop of Anima.forward_mini_train_dit over free-standing Blocks; emb / adaln_lora are non-leaf tensors
    (like t_embedding_norm's output) so their gradient accumulation order is exercised too."""
    params = attention.AttentionParams.create_attention_params("torch", False)
    x, emb_src, lora_src, crossattn, rope, extra = _inputs(dtype, use_adaln_lora)
    _set_checkpointing(blocks, mode)
    blocks.zero_grad(set_to_none=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # use_fp32 asks for an fp32 autocast region, which CPU autocast ignores with a warning
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            emb = emb_src * 1.0
            lora = lora_src * 1.0 if use_adaln_lora else None
            h = x
            for b in blocks:
                mods = None
                if hoist and b.gradient_checkpointing:
                    emb_in, lora_in = emb, lora
                    if b.unsloth_offload_checkpointing:
                        emb_in, lora_in = anima_models._per_block_grad_sum(emb), anima_models._per_block_grad_sum(lora)
                    mods = b.compute_adaln_modulations(emb_in, lora_in, use_fp32)
                h = b(h, emb, crossattn, params, use_fp32, rope, lora, extra, adaln_mods=mods)
            if not backward:
                return h.detach().clone(), {}
            h.float().square().mean().backward()
    grads = {n: p.grad.detach().clone() for n, p in blocks.named_parameters()}
    grads["x"], grads["emb"] = x.grad.detach().clone(), emb_src.grad.detach().clone()
    if use_adaln_lora:
        grads["adaln_lora"] = lora_src.grad.detach().clone()
    return h.detach().clone(), grads


def _assert_same(ref, new, tag):
    (ref_out, ref_g), (out, g) = ref, new
    assert torch.equal(out, ref_out), f"{tag}: output"
    assert set(g) == set(ref_g), f"{tag}: gradient keys {set(g) ^ set(ref_g)}"
    bad = [k for k in ref_g if not torch.equal(g[k], ref_g[k])]
    assert not bad, f"{tag}: gradients differ for {bad}"


def test_hoisted_block_matches_unhoisted_bitwise_fp32():
    """Single Block and two Blocks (gradient accumulation into emb / adaln_lora across Blocks), every checkpoint path,
    use_fp32 on and off, with and without AdaLN-LoRA (the 3.8B style): outputs and all gradients bit-identical."""
    for use_adaln_lora in (False, True):
        for n in (1, 2):
            blocks = _blocks(n, use_adaln_lora)
            for use_fp32 in (False, True):
                for mode in CKPT_MODES:
                    ref = _run_blocks(blocks, mode, False, use_fp32, use_adaln_lora)
                    new = _run_blocks(blocks, mode, True, use_fp32, use_adaln_lora)
                    _assert_same(ref, new, f"lora={use_adaln_lora} n={n} fp32={use_fp32} {mode}")
                    assert len(ref[1]) == sum(1 for _ in blocks.parameters()) + 2 + int(use_adaln_lora)  # + x, emb[, lora]


def test_hoisted_block_matches_unhoisted_bf16_autocast():
    """bf16 autocast on CPU: outputs bit-identical on every checkpoint path; gradients too where this CPU has a bf16
    backward (oneDNN lacks one on some CPUs, then the forward-only check stands)."""
    grads_ok = _bf16_backward_supported()
    for use_adaln_lora in (False, True):
        blocks = _blocks(2, use_adaln_lora, torch.bfloat16)
        for mode in CKPT_MODES:
            ref = _run_blocks(blocks, mode, False, False, use_adaln_lora, torch.bfloat16, autocast=True, backward=grads_ok)
            new = _run_blocks(blocks, mode, True, False, use_adaln_lora, torch.bfloat16, autocast=True, backward=grads_ok)
            _assert_same(ref, new, f"bf16 lora={use_adaln_lora} {mode}")


def test_unsloth_checkpointer_gives_gradients_to_extra_args():
    """UnslothOffloadedGradientCheckpointer.backward re-runs the function on detached copies of *args and returns
    inp.grad for each (anima_models.py:119-138): tensors passed after hidden_states do get their gradients, bit-identical
    to the plain graph, and a non-tensor / no-grad argument gets None."""

    def fn(h, a, b, flag, c):
        return h * a + b.sin() * (2.0 if flag else 1.0) + c

    results = []
    for use_ckpt in (False, True):
        g = torch.Generator().manual_seed(2)  # the same inputs for both runs
        make = lambda req: torch.randn(2, 5, generator=g).requires_grad_(req)  # noqa: E731
        h, a, b, c = make(True), make(True), make(True), make(False)
        out = UnslothOffloadedGradientCheckpointer.apply(fn, h, a, b, True, c) if use_ckpt else fn(h, a, b, True, c)
        out.square().sum().backward()
        assert c.grad is None
        results.append((out.detach(), h.grad, a.grad, b.grad))
    assert all(torch.equal(u, v) for u, v in zip(*results))


def test_per_block_grad_sum_is_a_view_or_passthrough():
    t = torch.randn(2, 3, requires_grad=True) * 1.0
    v = anima_models._per_block_grad_sum(t)
    assert v is not t and v.shape == t.shape and v.data_ptr() == t.data_ptr() and v.grad_fn.name().startswith("ViewBackward")
    plain = torch.randn(2, 3)
    assert anima_models._per_block_grad_sum(plain) is plain
    assert anima_models._per_block_grad_sum(None) is None


def _modulation_linears(block):
    return [block.adaln_modulation_self_attn[-1], block.adaln_modulation_cross_attn[-1], block.adaln_modulation_mlp[-1]]


def _count_linear_calls(blocks, mode: str, hoist: bool):
    counts = {}
    handles = []
    for i, b in enumerate(blocks):
        for j, lin in enumerate(_modulation_linears(b)):
            counts[(i, j)] = 0
            handles.append(lin.register_forward_hook(lambda m, inp, out, key=(i, j): counts.__setitem__(key, counts[key] + 1)))
    try:
        _run_blocks(blocks, mode, hoist, False, False)
    finally:
        for hd in handles:
            hd.remove()
    return counts


def test_hoist_runs_the_modulation_linears_once_per_micro_step():
    """Checkpointed: without the hoist each modulation Linear runs twice (forward + recompute in backward); hoisted it
    runs once. Without checkpointing it is once either way."""
    blocks = _blocks(2, False)
    for mode in ("torch", "cpu_offload", "unsloth"):
        assert set(_count_linear_calls(blocks, mode, False).values()) == {2}, mode
        assert set(_count_linear_calls(blocks, mode, True).values()) == {1}, mode
    assert set(_count_linear_calls(blocks, "none", False).values()) == {1}
    assert set(_count_linear_calls(blocks, "none", True).values()) == {1}


def _tiny_anima(use_adaln_lora: bool, extra_pos: bool = False):
    torch.manual_seed(0)
    m = anima_models.Anima(
        max_img_h=16, max_img_w=16, max_frames=1, in_channels=4, out_channels=4, patch_spatial=2, patch_temporal=1,
        concat_padding_mask=False, model_channels=D, num_blocks=2, num_heads=HEADS, crossattn_emb_channels=CTX,
        pos_emb_cls="rope3d", use_adaln_lora=use_adaln_lora, adaln_lora_dim=16, extra_per_block_abs_pos_emb=extra_pos,
    )
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.05)
    return m


def _run_anima(m, mode: str, hoist: bool, backward: bool = True, grad: bool = True, train: bool = True):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 4, 1, 8, 8, generator=g).requires_grad_(grad)
    t = torch.tensor([0.3, 0.7])
    ctx = torch.randn(2, 7, CTX, generator=g)
    if mode == "none":
        m.disable_gradient_checkpointing()
    else:
        m.enable_gradient_checkpointing(cpu_offload=mode == "cpu_offload", unsloth_offload=mode == "unsloth")
    m.train(train)
    m.zero_grad(set_to_none=True)
    old = anima_models.ADALN_HOIST
    anima_models.ADALN_HOIST = hoist
    try:
        with torch.set_grad_enabled(grad):
            out = m.forward_mini_train_dit(x, t, ctx)
            if not backward:
                return out.detach().clone(), {}
            out.square().mean().backward()
    finally:
        anima_models.ADALN_HOIST = old
    grads = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
    grads["x"] = x.grad.detach().clone()
    return out.detach().clone(), grads


def test_forward_mini_train_dit_hoist_is_bitwise_and_actually_hoists():
    """The real block loop (Anima.forward_mini_train_dit, 2 Blocks): ADALN_HOIST on vs off, every checkpoint path,
    with / without AdaLN-LoRA and the extra per-block pos emb: outputs and every parameter gradient (t_embedder and
    adaLN Linears included) bit-identical; with the hoist on, the modulation Linears run once per micro-step."""
    for use_adaln_lora in (False, True):
        for extra_pos in (False, True):
            m = _tiny_anima(use_adaln_lora, extra_pos)
            for mode in CKPT_MODES:
                ref = _run_anima(m, mode, False)
                new = _run_anima(m, mode, True)
                _assert_same(ref, new, f"anima lora={use_adaln_lora} extra={extra_pos} {mode}")
                assert any(k.startswith("t_embedder.") for k in ref[1]), list(ref[1])[:5]  # emb's producers got grads
    m = _tiny_anima(False)
    calls = []
    for b in m.blocks:
        for lin in _modulation_linears(b):
            lin.register_forward_hook(lambda mod, inp, out: calls.append(1))
    for mode, expected in (("torch", 6), ("unsloth", 6), ("none", 6)):
        calls.clear()
        _run_anima(m, mode, True)
        assert len(calls) == expected, (mode, len(calls))
    calls.clear()
    _run_anima(m, "torch", False)
    assert len(calls) == 12  # forward + recompute without the hoist


def test_no_hoist_with_block_swap_eval_or_no_grad():
    """blocks_to_swap set, eval mode or no_grad: compute_adaln_modulations is not called from the loop (the Blocks compute
    their modulations as before); outputs stay bitwise equal to ADALN_HOIST off."""
    m = _tiny_anima(False)
    calls = []
    orig = Block.compute_adaln_modulations

    def counting(self, *args, **kwargs):
        calls.append(1)
        return orig(self, *args, **kwargs)

    Block.compute_adaln_modulations = counting
    try:
        m.enable_gradient_checkpointing()
        # block swap: offloader stub with the two calls the loop makes; the loop keeps the Blocks' own modulation code
        m.blocks_to_swap = 1
        m.offloader = types.SimpleNamespace(wait_for_block=lambda i: None, submit_move_blocks=lambda blocks, i: None)
        ref = _run_anima(m, "torch", False)
        hoist_calls_before = len(calls)
        new = _run_anima(m, "torch", True)
        _assert_same(ref, new, "block swap")
        # compute_adaln_modulations ran only from inside Block._forward (forward + recompute = 2 per Block per run)
        assert len(calls) - hoist_calls_before == 2 * len(m.blocks)
        m.blocks_to_swap, m.offloader = None, None
        # eval + no_grad (sampling) with the argument absent
        calls.clear()
        ref = _run_anima(m, "torch", False, backward=False, grad=False, train=False)
        new = _run_anima(m, "torch", True, backward=False, grad=False, train=False)
        assert torch.equal(ref[0], new[0]) and len(calls) == 2 * len(m.blocks)  # one _forward per Block per run
        # training but no checkpointing: nothing to hoist
        calls.clear()
        _run_anima(m, "none", True)
        assert len(calls) == len(m.blocks)
    finally:
        Block.compute_adaln_modulations = orig


def _aten_ops(fn) -> int:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fn()
    return sum(e.count for e in prof.key_averages() if e.key.startswith("aten::"))


def test_hoist_reduces_aten_ops_of_a_checkpointed_micro_step():
    """2-Block tiny Anima, forward + backward with torch_checkpoint: the hoist removes the recompute of the six
    modulation Linears per Block (SiLU + matmul [+ add] each) and the extra graph plumbing."""
    m = _tiny_anima(False)
    counts = {}
    for mode in ("torch", "unsloth"):
        for hoist in (False, True):
            counts[(mode, hoist)] = _aten_ops(lambda: _run_anima(m, mode, hoist))
        assert counts[(mode, True)] < counts[(mode, False)], counts
    print("aten ops per micro-step (2 Blocks, fwd+bwd):", counts)


if __name__ == "__main__":
    import sys
    import traceback

    failed = 0
    for name, fn in [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print("failed:", failed)
    sys.exit(1 if failed else 0)
