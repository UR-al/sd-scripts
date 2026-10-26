"""RoPE angles of the Anima DiT must be computed in fp32 whatever dtype the position buffers end up in.

Anima checkpoints (e.g. Anima-2.9B-preview-v1) store ``pos_embedder.seq`` / ``dim_spatial_range`` /
``dim_temporal_range`` in bf16 (0.047607 instead of 1/21 = 0.047619), and ``load_anima_model`` assigns them with
``load_state_dict(assign=True)``; a ``model.to(bf16)`` casts them too. diffusion-pipe (the model author's trainer) and
ComfyUI compute these values in fp32, so the angles must not depend on the buffer dtype.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m pytest tests/test_anima_rope_precision.py -q
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.anima_models import VideoRopePosition3DEmb  # noqa: E402

# Same settings as library/anima_utils.load_anima_model: max_img 512 / patch 2, 128 frames, head_dim 2048 / 16.
ANIMA_ROPE = dict(head_dim=128, len_h=256, len_w=256, len_t=128,
                  h_extrapolation_ratio=4.0, w_extrapolation_ratio=4.0, t_extrapolation_ratio=1.0,
                  enable_fps_modulation=False)
H_1536 = 96  # 1536 px -> 192 latent -> 96 patches


def _reference_angles(H: int, W: int) -> torch.Tensor:
    """Exact (float64) RoPE angles for a T=1 image, laid out like generate_embeddings: [t, h, w] * 2."""
    rope = VideoRopePosition3DEmb(**ANIMA_ROPE)
    dim_h, dim_t = rope._dim_h, rope._dim_t
    spatial = torch.arange(0, dim_h, 2, dtype=torch.float64)[: dim_h // 2] / dim_h
    temporal = torch.arange(0, dim_t, 2, dtype=torch.float64)[: dim_t // 2] / dim_t
    h_freqs = 1.0 / ((10000.0 * rope.h_ntk_factor) ** spatial)
    w_freqs = 1.0 / ((10000.0 * rope.w_ntk_factor) ** spatial)
    t_freqs = 1.0 / ((10000.0 * rope.t_ntk_factor) ** temporal)
    seq = torch.arange(max(H, W, 1), dtype=torch.float64)
    emb_t = torch.outer(seq[:1], t_freqs)
    emb_h = torch.outer(seq[:H], h_freqs)
    emb_w = torch.outer(seq[:W], w_freqs)
    em = torch.cat([emb_t[:, None, None, :].expand(-1, H, W, -1),
                    emb_h[None, :, None, :].expand(1, -1, W, -1),
                    emb_w[None, None, :, :].expand(1, H, -1, -1)] * 2, dim=-1)
    return em.flatten(0, 2)


def _max_cos_sin_error(rope: VideoRopePosition3DEmb, H: int, W: int) -> float:
    cos, sin = rope.generate_embeddings(torch.Size((1, 1, H, W, 128)))
    ref = _reference_angles(H, W)
    cos = cos.reshape(ref.shape).double()
    sin = sin.reshape(ref.shape).double()
    return max((cos - ref.cos()).abs().max().item(), (sin - ref.sin()).abs().max().item())


def _bf16_checkpoint_rope() -> VideoRopePosition3DEmb:
    """What load_anima_model builds: the buffers replaced by the checkpoint's bf16 tensors (assign=True)."""
    rope = VideoRopePosition3DEmb(**ANIMA_ROPE)
    sd = {k: v.to(torch.bfloat16) for k, v in rope.state_dict().items()}
    rope.load_state_dict(sd, assign=True)
    assert rope.dim_spatial_range.dtype == torch.bfloat16
    return rope


def test_fresh_fp32_module_matches_reference():
    assert _max_cos_sin_error(VideoRopePosition3DEmb(**ANIMA_ROPE), H_1536, H_1536) < 1e-4


@pytest.mark.parametrize("H", [32, 64, H_1536])  # 512 / 1024 / 1536 px
def test_bf16_checkpoint_buffers_do_not_change_angles(H):
    assert _max_cos_sin_error(_bf16_checkpoint_rope(), H, H) < 1e-4


def test_module_cast_to_bf16_does_not_change_angles():
    rope = VideoRopePosition3DEmb(**ANIMA_ROPE).to(torch.bfloat16)
    assert rope.seq.dtype == torch.bfloat16
    assert _max_cos_sin_error(rope, H_1536, H_1536) < 1e-4


def test_tables_are_fp32():
    cos, sin = _bf16_checkpoint_rope().generate_embeddings(torch.Size((1, 1, 8, 8, 128)))
    assert cos.dtype == torch.float32 and sin.dtype == torch.float32


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
