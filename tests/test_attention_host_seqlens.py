"""CPU tests for the host copy of the attention lengths (library/attention.py AttentionParams.seqlens_host).

create_attention_params_from_mask(text_lengths=...) gives the params the lengths as Python ints; attention() then decides
the uniform-length trim and the per-sample split from them instead of reading the device tensor (torch.all(...) and
.item() per call before, a slice end per sample with split attention: a GPU->CPU sync each, in every block). The output
must stay bit-identical to the device path, and the host path must not read a tensor value at all.

A "device read" is counted the way it would sync on a GPU (tests/_device_reads.py).

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -B -m pytest tests/test_attention_host_seqlens.py
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

from library import attention
from library.attention import AttentionParams
from tests._device_reads import count_device_reads


def _qkv(b, img_len, text_len, dtype, seed=0, heads=2, dim=8):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(b, img_len + text_len, heads, dim, generator=g).to(dtype) for _ in range(3)]


def _mask(lengths, width):
    mask = torch.zeros(len(lengths), width, dtype=torch.bool)
    for i, n in enumerate(lengths):
        mask[i, :n] = True
    return mask


CASES = [((5, 5, 5), 7), ((5, 9, 2), 11), ((3,), 6), ((4, 4), 4)]  # (text lengths, padded text width)


def test_params_carry_the_host_lengths():
    mask = _mask((5, 9, 2), 11)
    for split in (False, True):
        p = AttentionParams.create_attention_params_from_mask("torch", split, 4, mask, text_lengths=[5, 9, 2])
        assert p.seqlens_host == [9, 13, 6] and p.seqlens.tolist() == [9, 13, 6]
        assert AttentionParams.create_attention_params_from_mask("torch", split, 4, mask).seqlens_host is None
    assert AttentionParams.create_attention_params_from_mask("torch", False, 4, None, text_lengths=[1]).seqlens_host is None


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("img_len", [0, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_host_lengths_are_bit_identical_to_the_device_path(split, img_len, dtype):
    for lengths, width in CASES:
        mask = _mask(lengths, width)
        q, k, v = _qkv(len(lengths), img_len, width, dtype)
        dev = AttentionParams.create_attention_params_from_mask("torch", split, img_len, mask)
        host = AttentionParams.create_attention_params_from_mask("torch", split, img_len, mask, text_lengths=list(lengths))
        want = attention.attention(q.clone(), k.clone(), v.clone(), dev)
        got = attention.attention(q.clone(), k.clone(), v.clone(), host)
        assert got.dtype == want.dtype and torch.equal(got, want), (lengths, split, img_len, dtype)
        # list input (the per-call form) as well
        got_list = attention.attention([q.clone(), k.clone(), v.clone()], attn_params=host)
        assert torch.equal(got_list, want)


@pytest.mark.parametrize("split", [False, True])
def test_host_lengths_read_no_tensor_value(split):
    for lengths, width in CASES:
        mask = _mask(lengths, width)
        q, k, v = _qkv(len(lengths), 4, width, torch.float32)
        host = AttentionParams.create_attention_params_from_mask("torch", split, 4, mask, text_lengths=list(lengths))
        dev = AttentionParams.create_attention_params_from_mask("torch", split, 4, mask)
        with count_device_reads() as reads:
            attention.attention(q, k, v, host)
        assert reads.ops == [], (lengths, split, reads.ops)
        with count_device_reads() as reads:
            attention.attention(q, k, v, dev)
        # the counter sees the old reads: torch.all(...) (+ .item() when uniform) batched, a slice end per sample split
        assert reads.ops, (lengths, split)


def test_gradients_are_bit_identical():
    lengths, width = (5, 9, 2), 11
    mask = _mask(lengths, width)
    grads = []
    for text_lengths in (None, list(lengths)):
        q, k, v = [t.requires_grad_(True) for t in _qkv(3, 4, width, torch.float32, seed=3)]
        p = AttentionParams.create_attention_params_from_mask("torch", True, 4, mask, text_lengths=text_lengths)
        attention.attention(q, k, v, p).square().sum().backward()
        grads.append([t.grad for t in (q, k, v)])
    assert all(torch.equal(a, b) for a, b in zip(*grads))
