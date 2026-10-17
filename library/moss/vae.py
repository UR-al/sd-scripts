# MOSS-Image VAE: FLUX.2 VAE (32 ch, 8x, small decoder) + 2x2 patchify + BatchNorm running-stat normalisation.
# The DiT sees 128-channel latents at stride 16. Decode follows the official pipeline_moss_image.py
# (Copyright OpenMOSS Team, Apache-2.0); encode is its exact inverse and uses the posterior mode like BFL FLUX.2 and
# diffusers Flux2Pipeline (MOSS does not publish its encode code).

from typing import Optional

import torch
from safetensors.torch import load_file

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

# OpenMOSS-Team/MOSS-Image-Base vae/config.json (identical in -SFT), diffusers AutoencoderKLFlux2.
MOSS_VAE_CONFIG = {
    "act_fn": "silu",
    "batch_norm_eps": 0.0001,
    "batch_norm_momentum": 0.1,
    "block_out_channels": [128, 256, 512, 512],
    "decoder_block_out_channels": [96, 192, 384, 384],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "force_upcast": True,
    "in_channels": 3,
    "latent_channels": 32,
    "layers_per_block": 2,
    "mid_block_add_attention": True,
    "norm_num_groups": 32,
    "out_channels": 3,
    "patch_size": [2, 2],
    "sample_size": 1024,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "use_post_quant_conv": True,
    "use_quant_conv": True,
}
MOSS_VAE_SCALE = 16  # pixels per DiT token (8x VAE * 2x2 patch)


def load_moss_vae(path: str, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32):
    """Load the diffusers-format FLUX.2 small-decoder VAE strictly. Keep it fp32 (official default)."""
    from diffusers import AutoencoderKLFlux2

    with torch.device("meta"):
        vae = AutoencoderKLFlux2(**MOSS_VAE_CONFIG)
    state = load_file(path, device="cpu")
    if any(key.startswith("vae.") for key in state):
        state = {key[4:]: value for key, value in state.items() if key.startswith("vae.")}
    vae.load_state_dict(state, strict=True, assign=True)
    vae = vae.to(device=device or torch.device("cpu"), dtype=dtype)
    vae.bn.num_batches_tracked = vae.bn.num_batches_tracked.to(torch.long)
    vae.eval().requires_grad_(False)
    logger.info(f"MOSS VAE loaded from {path} ({dtype}, eps {vae.config.batch_norm_eps})")
    return vae


def _bn_stats(vae, like: torch.Tensor):
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(like)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1).to(like) + vae.config.batch_norm_eps)
    return mean, std


@torch.no_grad()
def moss_vae_encode(vae, pixels: torch.Tensor) -> torch.Tensor:
    """pixels [B,3,H,W] in [-1,1] (H, W multiples of 16) -> DiT latents [B,128,H/16,W/16] (BN-normalised)."""
    if pixels.shape[-1] % MOSS_VAE_SCALE or pixels.shape[-2] % MOSS_VAE_SCALE:
        raise ValueError(f"MOSS image sides must be multiples of 16, got {tuple(pixels.shape[-2:])}")
    vae.eval()
    z = vae.encode(pixels.to(device=vae.device, dtype=vae.dtype)).latent_dist.mode()  # [B,32,H/8,W/8]
    b, c, h, w = z.shape
    z = z.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    mean, std = _bn_stats(vae, z)
    return (z - mean) / std


@torch.no_grad()
def moss_vae_decode(vae, latents: torch.Tensor) -> torch.Tensor:
    """DiT latents [B,128,h,w] -> pixels [B,3,16h,16w] in [-1,1] (official decode, fp32 VAE)."""
    z = latents.to(device=vae.device, dtype=vae.dtype)
    mean, std = _bn_stats(vae, z)
    z = z * std + mean
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    return vae.decode(z, return_dict=False)[0].clamp(-1, 1)
