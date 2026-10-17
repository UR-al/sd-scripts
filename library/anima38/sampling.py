# Training-time sample images for Anima-3.8B v1.1 (--anima38_connector).
#
# Same flow as anima_train_utils.sample_images / _sample_image_inference, and the denoising loop IS
# anima_train_utils.do_sample (reused unchanged). The difference: the prompt is not turned into a fixed
# llm_adapter context up front. Each DiT call receives the raw Qwen3 source, T5 ids and the cached Qwen3.5
# features, so the attached conditioner runs Semantic Connector v2 with that step's sigma, as Forge does.
# The negative prompt goes through the connector too ('connector', the user's Forge setting) or through the
# plain llm_adapter once ('native', Forge's default).

from __future__ import annotations

import gc
import logging
import os
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from library import anima_train_utils, train_util
from library.device_utils import clean_memory_on_device, synchronize_device

from .training import get_anima38_conditioner

logger = logging.getLogger(__name__)


class DiTConditioning:
    """One prompt's conditioning for a DiT call: ``dit(x, t, context, **kwargs)``."""

    def __init__(self, context: torch.Tensor, kwargs: Optional[Dict[str, Any]] = None):
        self.context = context
        self.kwargs = dict(kwargs or {})


class ConditionedDiT:
    """Adapter for ``anima_train_utils.do_sample``: it calls ``dit(x, t, crossattn_emb, padding_mask=...)`` and passes
    ``crossattn_emb`` through untouched, so a DiTConditioning can stand in for it."""

    def __init__(self, dit):
        self.dit = dit

    def __call__(self, x, t, conditioning, padding_mask=None):
        if not isinstance(conditioning, DiTConditioning):
            return self.dit(x, t, conditioning, padding_mask=padding_mask)
        return self.dit(x, t, conditioning.context, padding_mask=padding_mask, **conditioning.kwargs)


def _as_batch_tensor(value, device, dtype=None) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).unsqueeze(0)  # cached per-item arrays have no batch dim
    return value.to(device=device, dtype=dtype) if dtype is not None else value.to(device=device)


def connector_conditioning(encoded: Sequence, device, dtype) -> DiTConditioning:
    """[prompt_embeds, attn_mask, t5_ids, t5_mask, qwen35_states, qwen35_mask] -> per-step connector conditioning."""
    if len(encoded) < 6 or encoded[4] is None:
        raise RuntimeError("Anima-3.8B sampling: the prompt has no cached Qwen3.5 features")
    prompt_embeds, attn_mask, t5_ids, t5_mask, qwen35_states, qwen35_mask = encoded[:6]
    return DiTConditioning(
        _as_batch_tensor(prompt_embeds, device, dtype),
        {
            "target_input_ids": _as_batch_tensor(t5_ids, device, torch.long),
            "target_attention_mask": _as_batch_tensor(t5_mask, device),
            "source_attention_mask": _as_batch_tensor(attn_mask, device),
            "semantic_hidden_states": _as_batch_tensor(qwen35_states, device, dtype),
            "semantic_attention_mask": _as_batch_tensor(qwen35_mask, device),
        },
    )


def native_conditioning(dit, encoded: Sequence, device, dtype) -> DiTConditioning:
    """Plain llm_adapter context computed once (anima_train_utils._sample_image_inference); the conditioner passes a
    call without T5 ids through unchanged."""
    prompt_embeds, attn_mask, t5_ids, t5_mask = encoded[:4]
    prompt_embeds = _as_batch_tensor(prompt_embeds, device, dtype)
    attn_mask = _as_batch_tensor(attn_mask, device)
    t5_ids = _as_batch_tensor(t5_ids, device, torch.long)
    t5_mask = _as_batch_tensor(t5_mask, device)
    context = dit.llm_adapter(
        source_hidden_states=prompt_embeds,
        target_input_ids=t5_ids,
        target_attention_mask=t5_mask,
        source_attention_mask=attn_mask,
    )
    context[~t5_mask.bool()] = 0
    return DiTConditioning(context)


def sample_images(
    accelerator,
    args,
    epoch,
    steps,
    dit,
    vae,
    sample_prompts_te_outputs: Optional[Dict[str, Sequence]],
    negative_mode: str = "connector",
    prompt_replacement=None,
):
    """anima_train_utils.sample_images for the connector path (same schedule checks, RNG handling, block swap)."""
    if steps == 0:
        if not args.sample_at_first:
            return
    else:
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return
        if args.sample_every_n_epochs is not None:
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        else:
            if steps % args.sample_every_n_steps != 0 or epoch is not None:
                return

    logger.info(f"Generating sample images at step {steps} (Anima-3.8B connector, negative: {negative_mode})")
    if not os.path.isfile(args.sample_prompts):
        logger.error(f"No prompt file: {args.sample_prompts}")
        return
    if not sample_prompts_te_outputs:
        logger.error("Anima-3.8B sampling needs the sample prompts cached with Qwen3.5 (done while caching TE outputs)")
        return

    dit = accelerator.unwrap_model(dit)
    if get_anima38_conditioner(dit) is None:
        raise RuntimeError("Anima-3.8B sampling: the DiT has no Semantic Connector v2 attached")
    dit.switch_block_swap_for_inference()

    prompts = train_util.load_prompts(args.sample_prompts)
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)

    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    with torch.no_grad(), accelerator.autocast():
        for prompt_dict in prompts:
            dit.prepare_block_swap_before_forward()
            sample_image_inference(
                accelerator,
                args,
                dit,
                vae,
                save_dir,
                prompt_dict,
                epoch,
                steps,
                sample_prompts_te_outputs,
                negative_mode,
                prompt_replacement,
            )

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    dit.switch_block_swap_for_training()
    clean_memory_on_device(accelerator.device)


def sample_image_inference(
    accelerator,
    args,
    dit,
    vae,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    sample_prompts_te_outputs,
    negative_mode: str = "connector",
    prompt_replacement=None,
    decode: bool = True,
):
    """One sample image (anima_train_utils._sample_image_inference with per-step connector conditioning).

    Returns the latents (and saves the decoded PNG when ``decode``)."""
    prompt = prompt_dict.get("prompt", "")
    negative_prompt = prompt_dict.get("negative_prompt", "")
    sample_steps = prompt_dict.get("sample_steps", 30)
    width = prompt_dict.get("width", 512)
    height = prompt_dict.get("height", 512)
    scale = prompt_dict.get("scale", 7.5)
    seed = prompt_dict.get("seed")
    flow_shift = prompt_dict.get("flow_shift", 3.0)

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt:
            negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    height = max(64, height - height % 16)
    width = max(64, width - width % 16)

    logger.info(
        f"  prompt: {prompt}, size: {width}x{height}, steps: {sample_steps}, scale: {scale}, flow_shift: {flow_shift}, seed: {seed}"
    )

    encoded = sample_prompts_te_outputs.get(prompt)
    if encoded is None:
        logger.warning("Anima-3.8B: this prompt was not cached with Qwen3.5 features, skipping sample")
        return None

    device = accelerator.device
    dtype = dit.dtype
    positive = connector_conditioning(encoded, device, dtype)

    negative = None
    if scale > 1.0 and negative_prompt is not None:
        neg_encoded = sample_prompts_te_outputs.get(negative_prompt)
        if neg_encoded is None:
            logger.warning("Anima-3.8B: the negative prompt was not cached, sampling without CFG")
        elif negative_mode == "native":
            negative = native_conditioning(dit, neg_encoded, device, dtype)
        else:
            negative = connector_conditioning(neg_encoded, device, dtype)

    clean_memory_on_device(device)
    latents = anima_train_utils.do_sample(
        height, width, seed, ConditionedDiT(dit), positive, sample_steps, dtype, device, scale, flow_shift, negative
    )
    if not decode:
        return latents

    gc.collect()
    synchronize_device(device)
    clean_memory_on_device(device)
    org_vae_device = vae.device
    vae.to(device)
    decoded = vae.decode_to_pixels(latents)
    vae.to(org_vae_device)
    clean_memory_on_device(device)

    image = decoded.float()
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
    if image.ndim == 4:
        image = image[:, 0, :, :]
    decoded_np = (255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)).astype(np.uint8)
    image = Image.fromarray(decoded_np)

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i = prompt_dict.get("enum", 0)
    img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(save_dir, img_filename))

    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        wandb_tracker = accelerator.get_tracker("wandb")
        import wandb

        wandb_tracker.log({f"sample_{i}": wandb.Image(image, caption=prompt)}, commit=False)
    return latents
