"""A CPU stand-in for flash-attn (tests/test_anima_ckpt_reuse.py): the same shape of code as flash_attn_interface - a
forward and a backward kernel registered as torch custom ops (the backward writes dq / dk / dv in place, mutates_args), an
autograd Function around them that saves q, k, v, out, softmax_lse, and flash_attn_func on top - so library/attention.py's
attn_mode "flash" path, its checkpoint stash / reuse and their torch.compile traces can run without a GPU.

The backward is flash's formulation, not autograd's: it rebuilds the probabilities from q, k and the STORED softmax_lse
and takes the row sum of dP * P from the STORED output (sum_j dP_ij P_ij = <dout_i, out_i>), so a wrong or stale out / lse
reaching it changes the gradients (test_fake_flash_matches_autograd pins the kernels against plain autograd).
"""
from typing import Tuple

import torch

_NS = "anima_test_flash"
CALLS = {"fwd": 0, "bwd": 0}  # kernel runs, counted inside the ops: eager and compiled graphs both pass through them


def _scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """(B, heads, Lq, Lk) scaled dot products in fp32 from (B, L, heads, head_dim) q and k."""
    scale = q.shape[-1] ** (-0.5)
    return (q.float().transpose(1, 2) @ k.float().transpose(1, 2).transpose(-1, -2)) * scale


@torch.library.custom_op(f"{_NS}::fwd", mutates_args=())
def fake_flash_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    CALLS["fwd"] += 1
    with torch.autocast(q.device.type, enabled=False):  # a kernel does not see autocast (eager calls this under it)
        s = _scores(q, k)
        softmax_lse = torch.logsumexp(s, dim=-1)  # (B, heads, Lq), fp32 like flash's
        p = torch.exp(s - softmax_lse.unsqueeze(-1))
        out = (p @ v.float().transpose(1, 2)).transpose(1, 2).to(q.dtype)
    return out.contiguous(), softmax_lse


@torch.library.register_fake(f"{_NS}::fwd")
def _fake_flash_fwd_fake(q, k, v):
    batch, seqlen_q, heads, _ = q.shape
    return torch.empty_like(q), torch.empty((batch, heads, seqlen_q), dtype=torch.float32, device=q.device)


@torch.library.custom_op(f"{_NS}::bwd", mutates_args=("dq", "dk", "dv"))
def fake_flash_bwd(dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
                   softmax_lse: torch.Tensor, dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
    CALLS["bwd"] += 1
    scale = q.shape[-1] ** (-0.5)
    with torch.autocast(q.device.type, enabled=False):
        p = torch.exp(_scores(q, k) - softmax_lse.unsqueeze(-1))  # from the stored lse
        d_out = dout.float().transpose(1, 2)  # (B, heads, Lq, D)
        d_p = d_out @ v.float().transpose(1, 2).transpose(-1, -2)
        row = (d_out * out.float().transpose(1, 2)).sum(-1, keepdim=True)  # from the stored output
        d_s = p * (d_p - row)
        dq.copy_(((d_s @ k.float().transpose(1, 2)) * scale).transpose(1, 2).to(q.dtype))
        dk.copy_(((d_s.transpose(-1, -2) @ q.float().transpose(1, 2)) * scale).transpose(1, 2).to(k.dtype))
        dv.copy_((p.transpose(-1, -2) @ d_out).transpose(1, 2).to(v.dtype))
    return row.squeeze(-1)  # flash returns softmax_d; unused by the callers


@torch.library.register_fake(f"{_NS}::bwd")
def _fake_flash_bwd_fake(dout, q, k, v, out, softmax_lse, dq, dk, dv):
    batch, seqlen_q, heads, _ = q.shape
    return torch.empty((batch, heads, seqlen_q), dtype=torch.float32, device=q.device)


def wrapped_flash_attn_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_p, softmax_scale, causal,
                                window_size_left, window_size_right, softcap, alibi_slopes, deterministic, rng_state=None):
    """flash_attn_interface._wrapped_flash_attn_backward's signature; only the plain call is supported."""
    assert dropout_p == 0.0 and not causal and softcap == 0.0 and alibi_slopes is None
    assert softmax_scale == q.shape[-1] ** (-0.5) and (window_size_left, window_size_right) == (-1, -1)
    return torch.ops.anima_test_flash.bwd(dout, q, k, v, out, softmax_lse, dq, dk, dv)


class FakeFlashAttnFunc(torch.autograd.Function):
    """FlashAttnFunc for the plain call: forward kernel, q / k / v / out / softmax_lse saved when a grad is needed, the
    backward kernel on them."""

    @staticmethod
    def forward(ctx, q, k, v, return_softmax, is_grad_enabled):
        out, softmax_lse = torch.ops.anima_test_flash.fwd(q, k, v)
        if is_grad_enabled and any(t.requires_grad for t in (q, k, v)):
            ctx.save_for_backward(q, k, v, out, softmax_lse)
        return out if not return_softmax else (out, softmax_lse, torch.empty(0, dtype=q.dtype))

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        wrapped_flash_attn_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, 0.0, q.shape[-1] ** (-0.5), False, -1, -1,
                                    0.0, None, False, rng_state=None)
        return dq, dk, dv, None, None


def flash_attn_func(q, k, v, dropout_p=0.0, return_attn_probs=False):
    assert dropout_p == 0.0
    return FakeFlashAttnFunc.apply(q, k, v, return_attn_probs, torch.is_grad_enabled())


def install(monkeypatch) -> None:
    """Point library.attention's flash entry points at the CPU stand-ins for one test."""
    from library import attention

    monkeypatch.setattr(attention, "flash_attn_func", flash_attn_func)
    monkeypatch.setattr(attention, "_wrapped_flash_attn_backward", wrapped_flash_attn_backward)
    CALLS["fwd"] = CALLS["bwd"] = 0
