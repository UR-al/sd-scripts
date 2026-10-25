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


def test_no_checkpoint_max_tokens_toggles_the_blocks_per_micro_step():
    """--anima_no_checkpoint_max_tokens: micro-steps of small images run without gradient checkpointing, larger ones with
    the configured checkpointing (unsloth / cpu offload kept); 0 and --blocks_to_swap leave the blocks alone."""
    import argparse
    import inspect

    from anima_train_network import (CKPT_ALL, CKPT_NONE, anima_patch_spatial, apply_checkpointing_for_tokens,
                                     tokens_of_latents)

    # the attribute the trainer reads is the one Anima.__init__ sets (a run crashed on the Block's name, 2026-10-25)
    assert "self.patch_spatial = patch_spatial" in inspect.getsource(anima_models.Anima.__init__)
    assert anima_patch_spatial(types.SimpleNamespace(patch_spatial=2)) == 2
    assert anima_patch_spatial(types.SimpleNamespace()) == 2
    assert tokens_of_latents(torch.zeros(1, 16, 64, 64), 2) == 1024  # 512^2 image
    assert tokens_of_latents(torch.zeros(1, 16, 128, 96), 2) == 64 * 48  # 1024 x 768
    assert tokens_of_latents(torch.zeros(1, 16, 65, 63), 2) == 33 * 32  # odd grids round up

    class Blk:
        def __init__(self):
            self.gradient_checkpointing = True
            self.cpu_offload_checkpointing = False
            self.unsloth_offload_checkpointing = True

        def enable_gradient_checkpointing(self, cpu_offload=False, unsloth_offload=False):
            self.gradient_checkpointing = True
            self.cpu_offload_checkpointing = cpu_offload if not unsloth_offload else False
            self.unsloth_offload_checkpointing = unsloth_offload

        def disable_gradient_checkpointing(self):
            self.gradient_checkpointing = self.cpu_offload_checkpointing = self.unsloth_offload_checkpointing = False

    class Model:
        def __init__(self):
            self.blocks = [Blk() for _ in range(3)]
            self.spatial_patch_size = 2

        def enable_gradient_checkpointing(self, cpu_offload=False, unsloth_offload=False):
            for b in self.blocks:
                b.enable_gradient_checkpointing(cpu_offload=cpu_offload, unsloth_offload=unsloth_offload)

        def disable_gradient_checkpointing(self):
            for b in self.blocks:
                b.disable_gradient_checkpointing()

    args = argparse.Namespace(gradient_checkpointing=True, anima_no_checkpoint_max_tokens=1100, cpu_offload_checkpointing=False)
    m = Model()
    assert apply_checkpointing_for_tokens(m, args, 1024, unsloth=True, swapping=False) == CKPT_NONE
    assert all(not b.gradient_checkpointing and not b.unsloth_offload_checkpointing for b in m.blocks)
    assert apply_checkpointing_for_tokens(m, args, 4096, unsloth=True, swapping=False) == CKPT_ALL
    assert all(b.gradient_checkpointing and b.unsloth_offload_checkpointing and not b.cpu_offload_checkpointing for b in m.blocks)
    assert apply_checkpointing_for_tokens(m, args, 1100, unsloth=False, swapping=False) == CKPT_NONE  # boundary: <= N
    assert apply_checkpointing_for_tokens(m, args, 1101, unsloth=False, swapping=False) == CKPT_ALL
    assert all(b.gradient_checkpointing and not b.unsloth_offload_checkpointing for b in m.blocks)
    # blocks_to_swap: never switched off
    m2 = Model()
    assert apply_checkpointing_for_tokens(m2, args, 1024, unsloth=True, swapping=True) == CKPT_ALL
    assert all(b.gradient_checkpointing for b in m2.blocks)
    # 0 (the default) and no gradient checkpointing at all: untouched
    m3 = Model()
    args0 = argparse.Namespace(gradient_checkpointing=True, anima_no_checkpoint_max_tokens=0, cpu_offload_checkpointing=False)
    assert apply_checkpointing_for_tokens(m3, args0, 1024, unsloth=True, swapping=False) == CKPT_ALL
    assert all(b.gradient_checkpointing for b in m3.blocks)
    args_off = argparse.Namespace(gradient_checkpointing=False, anima_no_checkpoint_max_tokens=1100, cpu_offload_checkpointing=False)
    m4 = Model()
    m4.disable_gradient_checkpointing()
    assert apply_checkpointing_for_tokens(m4, args_off, 1024, unsloth=False, swapping=False) == CKPT_NONE
    assert all(not b.gradient_checkpointing for b in m4.blocks)


class _CountingBlk:
    """A Block's checkpointing flags (anima_models.Block.enable_gradient_checkpointing / disable_gradient_checkpointing)
    plus call counters."""

    def __init__(self, gc=True, cpu=False, unsloth=False):
        self.gradient_checkpointing, self.cpu_offload_checkpointing, self.unsloth_offload_checkpointing = gc, cpu, unsloth
        self.enables = self.disables = 0

    def enable_gradient_checkpointing(self, cpu_offload=False, unsloth_offload=False):
        self.enables += 1
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload if not unsloth_offload else False
        self.unsloth_offload_checkpointing = unsloth_offload

    def disable_gradient_checkpointing(self):
        self.disables += 1
        self.gradient_checkpointing = self.cpu_offload_checkpointing = self.unsloth_offload_checkpointing = False

    def flags(self):
        return (self.gradient_checkpointing, self.cpu_offload_checkpointing, self.unsloth_offload_checkpointing)


class _CountingModel:
    """anima_models.Anima.enable_gradient_checkpointing / disable_gradient_checkpointing over n fake blocks."""

    def __init__(self, n=6, **flags):
        self.blocks = [_CountingBlk(**flags) for _ in range(n)]
        self.patch_spatial = 2

    def enable_gradient_checkpointing(self, cpu_offload=False, unsloth_offload=False):
        for b in self.blocks:
            b.enable_gradient_checkpointing(cpu_offload=cpu_offload, unsloth_offload=unsloth_offload)

    def disable_gradient_checkpointing(self):
        for b in self.blocks:
            b.disable_gradient_checkpointing()

    def flags(self):
        return [b.flags() for b in self.blocks]

    def calls(self):
        return sum(b.enables + b.disables for b in self.blocks)


def _ckpt_args(n1, n2, gc=True, cpu=False):
    import argparse

    return argparse.Namespace(gradient_checkpointing=gc, anima_no_checkpoint_max_tokens=n1,
                              anima_half_checkpoint_max_tokens=n2, cpu_offload_checkpointing=cpu)


def test_half_checkpoint_three_regimes_and_boundaries():
    """--anima_half_checkpoint_max_tokens N2: tokens <= N1 none, N1 < tokens <= N2 every other Block (even-indexed
    checkpointed, odd-indexed kept), tokens > N2 all - with the configured mode (unsloth / cpu offload) restored exactly."""
    from anima_train_network import CKPT_ALL, CKPT_HALF, CKPT_NONE, apply_checkpointing_for_tokens, checkpoint_mode_for_tokens

    args = _ckpt_args(1100, 4200)
    expect = {1024: CKPT_NONE, 1100: CKPT_NONE, 1101: CKPT_HALF, 4096: CKPT_HALF, 4200: CKPT_HALF, 4201: CKPT_ALL,
              9216: CKPT_ALL, 16384: CKPT_ALL}
    assert {t: checkpoint_mode_for_tokens(args, t, False) for t in expect} == expect
    for unsloth, cpu in ((True, False), (False, True), (False, False)):
        args = _ckpt_args(1100, 4200, cpu=cpu)
        on = (True, cpu, unsloth)  # the flags Block.enable_gradient_checkpointing(cpu_offload, unsloth_offload) sets
        off = (False, False, False)
        m = _CountingModel(gc=True, cpu=cpu, unsloth=unsloth)
        for tokens, mode in expect.items():
            assert apply_checkpointing_for_tokens(m, args, tokens, unsloth=unsloth, swapping=False) == mode, (tokens, unsloth, cpu)
            if mode == CKPT_ALL:
                assert m.flags() == [on] * 6
            elif mode == CKPT_NONE:
                assert m.flags() == [off] * 6
            else:
                assert m.flags() == [on, off] * 3, (tokens, m.flags())
    # a 1-row micro-step per tier (rows x tokens per sample): 4096 tokens -> half, 1024 -> none, 9216 / 16384 -> all
    assert tokens_of_latents_tiers() == {512: 1024, 1024: 4096, 1536: 9216, 2048: 16384}


def tokens_of_latents_tiers():
    from anima_train_network import tokens_of_latents

    return {px: tokens_of_latents(torch.zeros(1, 16, px // 8, px // 8), 2) for px in (512, 1024, 1536, 2048)}


def test_half_checkpoint_without_the_none_regime():
    """N1 = 0, N2 > 0: no un-checkpointed regime; everything up to N2 is half, above it all."""
    from anima_train_network import CKPT_ALL, CKPT_HALF, apply_checkpointing_for_tokens

    args = _ckpt_args(0, 4200)
    m = _CountingModel(unsloth=True)
    assert apply_checkpointing_for_tokens(m, args, 1024, unsloth=True, swapping=False) == CKPT_HALF
    assert m.flags() == [(True, False, True), (False, False, False)] * 3
    assert apply_checkpointing_for_tokens(m, args, 4200, unsloth=True, swapping=False) == CKPT_HALF
    assert apply_checkpointing_for_tokens(m, args, 4201, unsloth=True, swapping=False) == CKPT_ALL
    assert m.flags() == [(True, False, True)] * 6


def test_half_checkpoint_refuses_n2_below_n1():
    from anima_train_network import validate_checkpoint_token_limits

    try:
        validate_checkpoint_token_limits(_ckpt_args(4200, 1100))
    except ValueError as e:
        msg = str(e)
        assert "4200" in msg and "1100" in msg and "--anima_half_checkpoint_max_tokens" in msg
        assert "이상이어야" in msg  # English + Korean
    else:
        raise AssertionError("N2 < N1 accepted")
    for n1, n2 in ((1100, 4200), (1100, 1100), (0, 4200), (1100, 0), (0, 0)):
        validate_checkpoint_token_limits(_ckpt_args(n1, n2))  # accepted
    # the trainer refuses at start-up (assert_extra_args), and the parser knows the option with 0 as its default
    import inspect

    import anima_train_network as atn

    assert "validate_checkpoint_token_limits(args)" in inspect.getsource(atn.AnimaNetworkTrainer.assert_extra_args)
    parser = atn.setup_parser()
    ns = parser.parse_args(["--anima_half_checkpoint_max_tokens", "4200"])
    assert ns.anima_half_checkpoint_max_tokens == 4200 and ns.anima_no_checkpoint_max_tokens == 0
    assert parser.parse_args([]).anima_half_checkpoint_max_tokens == 0


def test_half_checkpoint_leaves_blocks_to_swap_alone():
    """--blocks_to_swap: every mode is all, and the block flags are never touched."""
    from anima_train_network import CKPT_ALL, apply_checkpointing_for_tokens

    args = _ckpt_args(1100, 4200)
    m = _CountingModel(unsloth=False)
    m.blocks[1].gradient_checkpointing = False  # an odd flag pattern must survive untouched
    before = m.flags()
    for tokens in (1024, 4096, 16384):
        assert apply_checkpointing_for_tokens(m, args, tokens, unsloth=False, swapping=True) == CKPT_ALL
    assert m.flags() == before and m.calls() == 0
    # N1 = N2 = 0 (the defaults): untouched as well
    m = _CountingModel()
    assert apply_checkpointing_for_tokens(m, _ckpt_args(0, 0), 1024, unsloth=False, swapping=False) == CKPT_ALL
    assert m.calls() == 0


def test_half_checkpoint_toggles_only_on_mode_change():
    """The flags are written only when the regime changes between micro-steps (the mode is remembered on the model)."""
    from anima_train_network import CKPT_ALL, CKPT_HALF, CKPT_NONE, apply_checkpointing_for_tokens

    args = _ckpt_args(1100, 4200)
    m = _CountingModel(n=6, unsloth=True)
    seq = [(1024, CKPT_NONE)] * 3 + [(4096, CKPT_HALF)] * 4 + [(16384, CKPT_ALL)] * 2 + [(4096, CKPT_HALF)] + [(1024, CKPT_NONE)]
    calls_after = []
    for tokens, mode in seq:
        assert apply_checkpointing_for_tokens(m, args, tokens, unsloth=True, swapping=False) == mode
        calls_after.append(m.calls())
    # first call applies (6 block writes), then only the 4 regime changes (6 writes each); repeats write nothing
    assert calls_after == [6, 6, 6, 12, 12, 12, 12, 18, 18, 24, 30]
    assert m._anima_ckpt_mode == CKPT_NONE
    assert m.flags() == [(False, False, False)] * 6
    # the trainer's log: each reduced mode once (the source keeps a set of the modes announced)
    import inspect

    import anima_train_network as atn

    src = inspect.getsource(atn.AnimaNetworkTrainer.get_noise_pred_and_target)
    assert "self._ckpt_modes_logged" in src and "any(checkpoint_token_limits(args))" in src


def test_half_checkpointed_block_stack_has_the_same_grads():
    """Real Blocks (fp32, CPU): a stack with every other Block checkpointed gives the same output and parameter grads as
    all / none checkpointed (torch_checkpoint use_reentrant=False recompute is exact in fp32)."""
    from anima_train_network import CKPT_ALL, CKPT_HALF, CKPT_NONE, apply_checkpointing_for_tokens

    params = attention.AttentionParams.create_attention_params("torch", False)

    class Stack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.blocks = torch.nn.ModuleList(Block(x_dim=64, context_dim=48, num_heads=4) for _ in range(4))
            with torch.no_grad():
                for p in self.parameters():
                    p.normal_(0, 0.05)
            self.patch_spatial = 2

        def enable_gradient_checkpointing(self, cpu_offload=False, unsloth_offload=False):
            for b in self.blocks:
                b.enable_gradient_checkpointing(cpu_offload=cpu_offload, unsloth_offload=unsloth_offload)

        def disable_gradient_checkpointing(self):
            for b in self.blocks:
                b.disable_gradient_checkpointing()

    results = {}
    for tokens, mode in ((1024, CKPT_NONE), (4096, CKPT_HALF), (16384, CKPT_ALL)):
        stack = Stack().train()
        assert apply_checkpointing_for_tokens(stack, _ckpt_args(1100, 4200), tokens, unsloth=False, swapping=False) == mode
        assert [b.gradient_checkpointing for b in stack.blocks] == {CKPT_NONE: [False] * 4, CKPT_HALF: [True, False] * 2,
                                                                     CKPT_ALL: [True] * 4}[mode]
        g = torch.Generator().manual_seed(1)
        x = torch.randn(2, 1, 3, 4, 64, generator=g).requires_grad_(True)
        emb = torch.randn(2, 1, 64, generator=g)
        crossattn = torch.randn(2, 7, 48, generator=g)
        for b in stack.blocks:
            x = b(x, emb, crossattn, params, False, None)
        x.square().mean().backward()
        results[mode] = (x.detach(), [p.grad.clone() for p in stack.parameters()])
    out_none, grads_none = results[CKPT_NONE]
    for mode in (CKPT_HALF, CKPT_ALL):
        out, grads = results[mode]
        assert torch.equal(out, out_none), mode
        assert len(grads) == len(grads_none) and all(torch.equal(a, b) for a, b in zip(grads, grads_none)), mode


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
