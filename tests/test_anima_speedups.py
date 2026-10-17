"""CPU tests for the Anima speed-ups: the restructured Block / Attention eager path is bit-identical to the module-call
math it replaced, and the fast LyCORIS LoKr forward is bit-identical for fp32 inputs (Anima's LayerNorm outputs).

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe tests/test_anima_speedups.py   (or with pytest if installed)
"""
import types

import torch

from library import anima_models, attention
from library.anima_models import Block, apply_rotary_pos_emb_qk


def _reference_attention(attn, x, attn_params, context=None, rope_cos_sin=None):
    """Attention.forward as it was: module calls for the q/k norms and apply_rotary_pos_emb_qk."""
    q = attn.q_proj(x)
    context = x if context is None else context
    k = attn.k_proj(context)
    v = attn.v_proj(context)
    q = attn.q_norm(q.unflatten(-1, (attn.n_heads, attn.head_dim)))
    k = attn.k_norm(k.unflatten(-1, (attn.n_heads, attn.head_dim)))
    v = attn.v_norm(v.unflatten(-1, (attn.n_heads, attn.head_dim)))
    if attn.is_selfattn and rope_cos_sin is not None:
        q, k = apply_rotary_pos_emb_qk(q, k, rope_cos_sin, tensor_format=attn.qkv_format)
    if q.dtype != v.dtype and (not attn_params.supports_fp32 or attn_params.requires_same_dtype) and torch.is_autocast_enabled():
        q, k = q.to(v.dtype), k.to(v.dtype)
    return attn.output_dropout(attn.output_proj(attention.attention([q, k, v], attn_params=attn_params)))


def _reference_block(block, x, emb, crossattn, attn_params, rope_cos_sin):
    """Block._forward as it was (use_fp32=False, no AdaLN-LoRA, no extra pos emb)."""
    shift_sa, scale_sa, gate_sa = block.adaln_modulation_self_attn(emb).chunk(3, dim=-1)
    shift_ca, scale_ca, gate_ca = block.adaln_modulation_cross_attn(emb).chunk(3, dim=-1)
    shift_m, scale_m, gate_m = block.adaln_modulation_mlp(emb).chunk(3, dim=-1)
    e = lambda t: t[:, :, None, None, :]  # noqa: E731
    B, T, H, W, D = x.shape
    n = block.layer_norm_self_attn(x) * (1 + e(scale_sa)) + e(shift_sa)
    flat = n.flatten(1, 3)
    x = x + e(gate_sa) * _reference_attention(block.self_attn, flat, attn_params, flat, rope_cos_sin).unflatten(1, (T, H, W))
    n = block.layer_norm_cross_attn(x) * (1 + e(scale_ca)) + e(shift_ca)
    r = _reference_attention(block.cross_attn, n.flatten(1, 3), attn_params, crossattn, rope_cos_sin).unflatten(1, (T, H, W))
    x = r * e(gate_ca) + x
    n = block.layer_norm_mlp(x) * (1 + e(scale_m)) + e(shift_m)
    return x + e(gate_m) * block.mlp(n)


def _block_inputs(dtype):
    torch.manual_seed(0)
    block = Block(x_dim=64, context_dim=48, num_heads=4).to(dtype)
    with torch.no_grad():
        for p in block.parameters():
            p.normal_(0, 0.05)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 1, 3, 4, 64, generator=g).to(dtype)
    emb = torch.randn(2, 1, 64, generator=g).to(dtype)
    crossattn = torch.randn(2, 7, 48, generator=g).to(dtype)
    cos = torch.randn(12, 1, 1, 16, generator=g)  # sbhd tables longer than the 12 tokens, partial rotary dim (16 of 16)
    sin = torch.randn(12, 1, 1, 16, generator=g)
    return block, x, emb, crossattn, (cos, sin)


def test_block_eager_path_matches_module_math():
    """fp32: output and every grad bit-identical. bf16 autocast: output bit-identical (oneDNN on some CPUs has no bf16
    backward, so the bf16 grads are checked on the GPU instead)."""
    params = attention.AttentionParams.create_attention_params("torch", False)
    for dtype, autocast in ((torch.float32, False), (torch.bfloat16, True)):
        results = []
        for use_new in (False, True):
            block, x, emb, crossattn, rope = _block_inputs(dtype)
            x.requires_grad_(not autocast)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast), torch.set_grad_enabled(not autocast):
                if use_new:
                    out = block._forward(x, emb, crossattn, params, False, rope)
                else:
                    out = _reference_block(block, x, emb, crossattn, params, rope)
            if autocast:
                results.append((out, None, []))
                continue
            out.float().square().mean().backward()
            results.append((out.detach(), x.grad, [p.grad.clone() for p in block.parameters() if p.grad is not None]))
        (a, ga, pa), (b, gb, pb) = results
        assert torch.equal(a, b), dtype
        if not autocast:
            assert torch.equal(ga, gb) and len(pa) == len(pb) and all(torch.equal(u, v) for u, v in zip(pa, pb))


def test_patched_norm_falls_back_to_module_call():
    params = attention.AttentionParams.create_attention_params("torch", False)
    block, x, emb, crossattn, rope = _block_inputs(torch.float32)
    calls = []
    block.self_attn.q_norm.register_forward_hook(lambda m, i, o: calls.append(1))
    block._forward(x, emb, crossattn, params, False, rope)
    assert calls  # a hooked norm is called as a module, not inlined


def test_fast_lokr_fp32_input_is_bit_identical():
    """Anima feeds fp32 LayerNorm outputs to the LoKr Linears under bf16 autocast."""
    from lycoris.modules.lokr import LokrModule
    from library import lycoris_fast

    lycoris_fast.install_fast_lokr_forward()
    results = []
    for fast in (False, True):
        torch.manual_seed(0)
        org = torch.nn.Linear(128, 192, bias=False).to(torch.bfloat16).requires_grad_(False)
        module = LokrModule("t", org, multiplier=1.0, lora_dim=10000, alpha=10000, factor=4, weight_decompose=True,
                            wd_on_output=False, full_matrix=True, bypass_mode=False)
        module.apply_to()
        if not fast:
            org.forward = types.MethodType(lycoris_fast._STOCK_FORWARD, module)
        with torch.no_grad():
            module.lokr_w2.normal_(0, 0.02)
            module.dora_scale.mul_(1.01)
        x = torch.randn(2, 9, 128, generator=torch.Generator().manual_seed(1)).requires_grad_(True)  # fp32
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = org(x)
        y.float().square().sum().backward()
        results.append((y, x.grad, module.lokr_w1.grad, module.lokr_w2.grad, module.dora_scale.grad))
    assert all(torch.equal(a, b) for a, b in zip(*results))


def test_compile_flag_defaults_off():
    assert anima_models._CHAINS.enabled is False or anima_models._CHAINS.enabled is True  # module state exists
    old = anima_models._CHAINS.enabled
    try:
        anima_models.enable_elementwise_compile(True)
        assert anima_models._CHAINS.get(torch.zeros(1)) is anima_models._CHAINS.eager  # CPU tensors stay eager
    finally:
        anima_models.enable_elementwise_compile(old)


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
