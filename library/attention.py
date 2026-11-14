# Unified attention function supporting various implementations

import logging
from dataclasses import dataclass
import torch
from typing import List, Optional, Sequence, Union

_logger = logging.getLogger(__name__)

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from flash_attn.flash_attn_interface import flash_attn_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None
    flash_attn_func = None

try:  # the backward kernel alone, for --anima_ckpt_reuse_attn (the torch custom op from torch 2.4, the plain function before)
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_backward
except ImportError:
    _wrapped_flash_attn_backward = None

# Verify flash_attn GPU compatibility: requires Ampere (sm_80, compute capability 8.0) or newer.
# T4 (Turing, sm_75) and older GPUs will hit "FlashAttention only supports Ampere GPUs or newer"
# at runtime. Detect this early and disable flash_attn so callers fall back gracefully.
if flash_attn is not None:
    try:
        if torch.cuda.is_available():
            _capability = torch.cuda.get_device_capability()
            if _capability < (8, 0):
                _logger.warning(
                    "flash_attn is installed but requires Ampere GPU (sm_80) or newer. "
                    f"Current GPU compute capability: {_capability[0]}.{_capability[1]}. "
                    "Disabling flash_attn and falling back to other attention implementations."
                )
                flash_attn = None
                flash_attn_varlen_func = None
                _flash_attn_forward = None
                flash_attn_func = None
    except Exception:
        _logger.debug("Could not determine GPU capability for flash_attn compatibility check")


def is_flash_attn_supported() -> bool:
    """Return True if flash_attn is importable AND the current GPU supports it (Ampere+)."""
    if flash_attn_varlen_func is None:
        return False
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability() >= (8, 0)
    except Exception:
        pass
    return False


try:
    from sageattention import sageattn_varlen, sageattn
except ImportError:
    sageattn_varlen = None
    sageattn = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None


@dataclass
class AttentionParams:
    attn_mode: Optional[str] = None
    split_attn: bool = False
    img_len: Optional[int] = None
    attention_mask: Optional[torch.Tensor] = None
    seqlens: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    max_seqlen: Optional[int] = None
    # host copy of seqlens (Python ints), set when the caller already holds the text lengths on the CPU
    # (create_attention_params_from_mask(text_lengths=...)): attention() then decides the trim / per-sample split from
    # it instead of reading the device tensor - one GPU->CPU sync per call otherwise, i.e. per block
    seqlens_host: Optional[List[int]] = None

    @property
    def supports_fp32(self) -> bool:
        return self.attn_mode not in ["flash"]

    @property
    def requires_same_dtype(self) -> bool:
        return self.attn_mode in ["xformers"]

    @staticmethod
    def create_attention_params(attn_mode: Optional[str], split_attn: bool) -> "AttentionParams":
        return AttentionParams(attn_mode, split_attn)

    @staticmethod
    def create_attention_params_from_mask(
        attn_mode: Optional[str], split_attn: bool, img_len: Optional[int], attention_mask: Optional[torch.Tensor],
        text_lengths: Optional[Sequence[int]] = None,
    ) -> "AttentionParams":
        """text_lengths: attention_mask.sum(1) as Python ints when the caller already has them on the CPU (computed once
        per forward); the params then carry a host copy of seqlens and attention() never reads the device tensor."""
        if attention_mask is None:
            # No attention mask provided: assume all tokens are valid
            return AttentionParams(attn_mode, split_attn, None, None, None, None, None)
        else:
            # Note: attention_mask is only for text tokens, not including image tokens
            seqlens = attention_mask.sum(dim=1).to(torch.int32) + img_len  # [B]
            max_seqlen = attention_mask.shape[1] + img_len
            seqlens_host = None if text_lengths is None else [int(n) + img_len for n in text_lengths]

            if split_attn:
                # cu_seqlens is not needed for split attention
                return AttentionParams(attn_mode, split_attn, img_len, attention_mask, seqlens, None, max_seqlen, seqlens_host)

            # Convert attention mask to cumulative sequence lengths for flash attention
            batch_size = attention_mask.shape[0]
            cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device=attention_mask.device)
            for i in range(batch_size):
                cu_seqlens[2 * i + 1] = i * max_seqlen + seqlens[i]  # end of valid tokens for query
                cu_seqlens[2 * i + 2] = (i + 1) * max_seqlen  # end of all tokens for query

            # Expand attention mask to include image tokens
            attention_mask = torch.nn.functional.pad(attention_mask, (img_len, 0), value=1)  # [B, img_len + L]

            if attn_mode == "xformers":
                seqlens_list = list(seqlens_host) if seqlens_host is not None else seqlens.cpu().tolist()
                attention_mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                    seqlens_list, seqlens_list, device=attention_mask.device
                )
            elif attn_mode == "torch":
                attention_mask = attention_mask[:, None, None, :].to(torch.bool)  # [B, 1, 1, img_len + L]

            return AttentionParams(attn_mode, split_attn, img_len, attention_mask, seqlens, cu_seqlens, max_seqlen, seqlens_host)


# ---- self-attention reuse in a checkpoint recompute (--anima_ckpt_reuse_attn, 2026-11-14; off by default) ---------------
# anima_models.nograd_checkpoint runs a checkpointed Block twice: a no-grad first pass, and a grad-mode recompute in
# backward that used to run the flash forward kernel again only to get the same output and softmax_lse (the recompute's
# q, k, v come from the same weights and inputs). With the flag the first pass keeps both (ckpt_stash, a list it fills)
# and the recompute returns the stored output and saves q, k, v, out, lse for the flash BACKWARD kernel, which is all the
# flash autograd node needs (FlashAttnFunc.backward calls the same op with the same arguments). Only on the plain flash
# call (no mask, no split, no dropout, head dim a multiple of 8: FlashAttnFunc pads other head dims) and only when this
# flash-attn exposes the backward kernel on its own (flash_reuse_available); anything else leaves the stash empty and the
# recompute runs the forward as before.


def flash_reuse_available() -> bool:
    """True when the recompute can call flash's backward kernel alone: flash_attn_interface._wrapped_flash_attn_backward
    was importable (flash-attn 2.8.4 has it; an older build with only FlashAttnFunc would fail in the first backward).
    False: nothing is stored for reuse, the recompute runs flash_attn_func again, as without --anima_ckpt_reuse_attn."""
    return _wrapped_flash_attn_backward is not None


def _flash_forward_with_lse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """flash_attn_func(q, k, v, 0.0) plus its softmax_lse. The kernel call is the same (return_softmax only reaches it with
    dropout), so the output is the one the plain call gives."""
    out, softmax_lse, _ = flash_attn_func(q, k, v, 0.0, return_attn_probs=True)
    return out, softmax_lse


def _flash_backward(dout, q, k, v, out, softmax_lse):
    """dq, dk, dv from the flash backward kernel, called as FlashAttnFunc.backward calls it for flash_attn_func's
    defaults (no dropout: the rng state is not read)."""
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _wrapped_flash_attn_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, 0.0, q.shape[-1] ** (-0.5), False, -1, -1,
                                 0.0, None, False, rng_state=None)
    return dq, dk, dv


class _FlashReuse(torch.autograd.Function):
    """The flash attention node of a recompute: the forward kernel is skipped (the first pass stored its results), the
    backward kernel runs on this pass's q, k, v."""

    @staticmethod
    def forward(ctx, q, k, v, out, softmax_lse):
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        return out.view_as(out)

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = _flash_backward(dout, q, k, v, out, softmax_lse)
        return dq, dk, dv, None, None


def attention(
    qkv_or_q: Union[torch.Tensor, list],
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    attn_params: Optional[AttentionParams] = None,
    drop_rate: float = 0.0,
    ckpt_stash: Optional[list] = None,
    ckpt_reuse: Optional[tuple] = None,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention with variable sequence lengths.

    Handles batches with different sequence lengths by splitting and
    processing each sequence individually.

    Args:
        qkv_or_q: Query tensor [B, L, H, D]. or list of such tensors.
        k: Key tensor [B, L, H, D].
        v: Value tensor [B, L, H, D].
        attn_params: Attention parameters including mask and sequence lengths.
        drop_rate: Attention dropout rate.
        ckpt_stash: --anima_ckpt_reuse_attn, the no-grad first pass of a checkpointed Block: an empty list that receives
            (out, softmax_lse) of the plain flash call. None (the default): nothing is kept.
        ckpt_reuse: the recompute of that Block: the stored (out, softmax_lse); the flash forward is not run again.

    Returns:
        Attention output tensor [B, L, H*D].
    """
    if isinstance(qkv_or_q, list):
        q, k, v = qkv_or_q
        q: torch.Tensor = q
        qkv_or_q.clear()
        del qkv_or_q
    else:
        q: torch.Tensor = qkv_or_q
        del qkv_or_q
        assert k is not None and v is not None, "k and v must be provided if qkv_or_q is a tensor"
    if attn_params is None:
        attn_params = AttentionParams.create_attention_params("torch", False)

    # If split attn is False, attention mask is provided and all sequence lengths are same, we can trim the sequence.
    # With a host copy of the lengths (seqlens_host) the decision costs no GPU->CPU sync; without one it reads the
    # device tensor as before (a sync per call).
    seqlen_trimmed = False
    if not attn_params.split_attn and attn_params.attention_mask is not None and attn_params.seqlens is not None:
        host = attn_params.seqlens_host
        if host is not None:
            uniform = all(n == host[0] for n in host)
        else:
            uniform = bool(torch.all(attn_params.seqlens == attn_params.seqlens[0]))
        if uniform:
            seqlen = host[0] if host is not None else attn_params.seqlens[0].item()
            q = q[:, :seqlen]
            k = k[:, :seqlen]
            v = v[:, :seqlen]
            max_seqlen = attn_params.max_seqlen
            attn_params = AttentionParams.create_attention_params(attn_params.attn_mode, False)  # do not in-place modify
            attn_params.max_seqlen = max_seqlen  # keep max_seqlen for padding
            seqlen_trimmed = True

    # Determine tensor layout based on attention implementation
    if attn_params.attn_mode == "torch" or (
        attn_params.attn_mode == "sageattn" and (attn_params.split_attn or attn_params.cu_seqlens is None)
    ):
        transpose_fn = lambda x: x.transpose(1, 2)  # [B, H, L, D] for SDPA and sageattn with fixed length
        # pad on sequence length dimension
        pad_fn = lambda x, pad_to: torch.nn.functional.pad(x, (0, 0, 0, pad_to - x.shape[-2]), value=0)
    else:
        transpose_fn = lambda x: x  # [B, L, H, D] for other implementations
        # pad on sequence length dimension
        pad_fn = lambda x, pad_to: torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_to - x.shape[-3]), value=0)

    # Process each batch element with its valid sequence lengths
    if attn_params.split_attn:
        if attn_params.seqlens is None:
            # If no seqlens provided, assume all tokens are valid
            attn_params = AttentionParams.create_attention_params(attn_params.attn_mode, True)  # do not in-place modify
            attn_params.seqlens = torch.tensor([q.shape[1]] * q.shape[0], device=q.device)
            attn_params.max_seqlen = q.shape[1]
        # Loops, not comprehensions: a comprehension makes q / k / v closure cells, which turns the `del q, k, v` below
        # into DELETE_DEREF - a bytecode torch.compile (dynamo, torch 2.7) does not trace, i.e. a graph break per
        # attention call under --anima_block_compile. With plain locals the del is DELETE_FAST.
        qs, ks, vs = [], [], []
        host = attn_params.seqlens_host
        for i in range(len(q)):
            n = host[i] if host is not None else attn_params.seqlens[i]  # a device element as a slice end: a sync each
            qs.append(transpose_fn(q[i : i + 1, :n]))
            ks.append(transpose_fn(k[i : i + 1, :n]))
            vs.append(transpose_fn(v[i : i + 1, :n]))
        q, k, v = qs, ks, vs
    else:
        q = transpose_fn(q)
        k = transpose_fn(k)
        v = transpose_fn(v)

    if attn_params.attn_mode == "torch":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                x_i = torch.nn.functional.scaled_dot_product_attention(q[i], k[i], v[i], dropout_p=drop_rate)
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, H, L, D
            x = torch.cat(x, dim=0)
            del q, k, v

        else:
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_params.attention_mask, dropout_p=drop_rate)
            del q, k, v

    elif attn_params.attn_mode == "xformers":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                x_i = xops.memory_efficient_attention(q[i], k[i], v[i], p=drop_rate)
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, L, H, D
            x = torch.cat(x, dim=0)
            del q, k, v

        else:
            x = xops.memory_efficient_attention(q, k, v, attn_bias=attn_params.attention_mask, p=drop_rate)
            del q, k, v

    elif attn_params.attn_mode == "sageattn":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                # HND seems to cause an error
                x_i = sageattn(q[i], k[i], v[i])  # B, H, L, D. No dropout support
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, H, L, D
            x = torch.cat(x, dim=0)
            del q, k, v
        elif attn_params.cu_seqlens is None:  # all tokens are valid
            x = sageattn(q, k, v)  # B, L, H, D. No dropout support
            del q, k, v
        else:
            # Reshape to [(bxs), a, d]
            batch_size, seqlen = q.shape[0], q.shape[1]
            q = q.view(q.shape[0] * q.shape[1], *q.shape[2:])  # [B*L, H, D]
            k = k.view(k.shape[0] * k.shape[1], *k.shape[2:])  # [B*L, H, D]
            v = v.view(v.shape[0] * v.shape[1], *v.shape[2:])  # [B*L, H, D]

            # Assume cu_seqlens_q == cu_seqlens_kv and max_seqlen_q == max_seqlen_kv. No dropout support
            x = sageattn_varlen(
                q, k, v, attn_params.cu_seqlens, attn_params.cu_seqlens, attn_params.max_seqlen, attn_params.max_seqlen
            )
            del q, k, v

            # Reshape x with shape [(bxs), a, d] to [b, s, a, d]
            x = x.view(batch_size, seqlen, x.shape[-2], x.shape[-1])  # B, L, H, D

    elif attn_params.attn_mode == "flash":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                # HND seems to cause an error
                x_i = flash_attn_func(q[i], k[i], v[i], drop_rate)  # B, L, H, D
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, L, H, D
            x = torch.cat(x, dim=0)
            del q, k, v
        elif attn_params.cu_seqlens is None:  # all tokens are valid
            if ckpt_reuse is not None:  # a checkpoint recompute: the first pass's output, only the backward kernel later
                x = _FlashReuse.apply(q, k, v, ckpt_reuse[0], ckpt_reuse[1])
            elif (ckpt_stash is not None and _wrapped_flash_attn_backward is not None and drop_rate == 0.0
                  and q.shape[-1] % 8 == 0):  # no backward op to call later: nothing stored, the recompute runs the forward
                x, softmax_lse = _flash_forward_with_lse(q, k, v)  # the same kernel call, its softmax_lse kept too
                ckpt_stash.append(x)
                ckpt_stash.append(softmax_lse)
            else:
                x = flash_attn_func(q, k, v, drop_rate)  # B, L, H, D
            del q, k, v
        else:
            # Reshape to [(bxs), a, d]
            batch_size, seqlen = q.shape[0], q.shape[1]
            q = q.view(q.shape[0] * q.shape[1], *q.shape[2:])  # [B*L, H, D]
            k = k.view(k.shape[0] * k.shape[1], *k.shape[2:])  # [B*L, H, D]
            v = v.view(v.shape[0] * v.shape[1], *v.shape[2:])  # [B*L, H, D]

            # Assume cu_seqlens_q == cu_seqlens_kv and max_seqlen_q == max_seqlen_kv
            x = flash_attn_varlen_func(
                q, k, v, attn_params.cu_seqlens, attn_params.cu_seqlens, attn_params.max_seqlen, attn_params.max_seqlen, drop_rate
            )
            del q, k, v

            # Reshape x with shape [(bxs), a, d] to [b, s, a, d]
            x = x.view(batch_size, seqlen, x.shape[-2], x.shape[-1])  # B, L, H, D

    else:
        # Currently only PyTorch SDPA and xformers are implemented
        raise ValueError(f"Unsupported attention mode: {attn_params.attn_mode}")

    x = transpose_fn(x)  # [B, L, H, D]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, L, H*D]

    if seqlen_trimmed:
        x = torch.nn.functional.pad(x, (0, 0, 0, attn_params.max_seqlen - x.shape[1]), value=0)  # pad back to max_seqlen

    return x
