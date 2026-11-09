# Anima Training Utilities

import argparse
import contextlib
import gc
import math
import os
import time
from typing import Optional

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm
from PIL import Image

from library.device_utils import init_ipex, clean_memory_on_device, synchronize_device
from library import anima_models, anima_utils, timestep_presets, train_util, qwen_image_autoencoder_kl
from library.lycoris_fast import cached_lokr_weights

init_ipex()

from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# Anima-specific training arguments


def add_anima_training_arguments(parser: argparse.ArgumentParser):
    """Add Anima-specific training arguments to the parser."""
    parser.add_argument(
        "--qwen3",
        type=str,
        default=None,
        help="Path to Qwen3-0.6B model (safetensors file or directory)",
    )
    parser.add_argument(
        "--llm_adapter_path",
        type=str,
        default=None,
        help="Path to separate LLM adapter weights. If None, adapter is loaded from DiT file if present",
    )
    parser.add_argument(
        "--llm_adapter_lr",
        type=float,
        default=None,
        help="Learning rate for LLM adapter. None=same as base LR, 0=freeze adapter",
    )
    parser.add_argument(
        "--self_attn_lr",
        type=float,
        default=None,
        help="Learning rate for self-attention layers. None=same as base LR, 0=freeze",
    )
    parser.add_argument(
        "--cross_attn_lr",
        type=float,
        default=None,
        help="Learning rate for cross-attention layers. None=same as base LR, 0=freeze",
    )
    parser.add_argument(
        "--mlp_lr",
        type=float,
        default=None,
        help="Learning rate for MLP layers. None=same as base LR, 0=freeze",
    )
    parser.add_argument(
        "--mod_lr",
        type=float,
        default=None,
        help="Learning rate for AdaLN modulation layers. None=same as base LR, 0=freeze. Note: mod layers are not included in LoRA by default.",
    )
    parser.add_argument(
        "--t5_tokenizer_path",
        type=str,
        default=None,
        help="Path to T5 tokenizer directory. If None, uses default configs/t5_old/",
    )
    parser.add_argument(
        "--qwen3_max_token_length",
        type=int,
        default=512,
        help="Maximum token length for Qwen3 tokenizer (default: 512)",
    )
    parser.add_argument(
        "--t5_max_token_length",
        type=int,
        default=512,
        help="Maximum token length for T5 tokenizer (default: 512)",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=1.0,
        help="Timestep distribution shift for rectified flow training (default: 1.0)",
    )
    parser.add_argument(
        "--timestep_sampling",
        type=str,
        default="sigmoid",
        choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift"],
        help="Timestep sampling method (default: sigmoid (logit normal))",
    )
    parser.add_argument(
        "--sigmoid_scale",
        type=timestep_presets.parse_sigmoid_scale,
        default=1.0,
        help=timestep_presets.SIGMOID_SCALE_HELP + " (default: 1.0)",
    )
    timestep_presets.add_timestep_preset_arguments(parser)
    parser.add_argument(
        "--attn_mode",
        choices=["torch", "xformers", "flash", "sageattn", "sdpa"],  # "sdpa" is for backward compatibility
        default=None,
        help="Attention implementation to use. Default is None (torch). xformers requires --split_attn. sageattn does not support training (inference only). This option overrides --xformers or --sdpa."
        " / 使用するAttentionの実装。デフォルトはNone（torch）です。xformersは--split_attnの指定が必要です。sageattnはトレーニングをサポートしていません（推論のみ）。このオプションは--xformersまたは--sdpaを上書きします。",
    )
    parser.add_argument(
        "--split_attn",
        action="store_true",
        help="split attention computation to reduce memory usage / メモリ使用量を減らすためにattention時にバッチを分割する",
    )
    parser.add_argument(
        "--vae_chunk_size",
        type=int,
        default=None,
        help="Spatial chunk size for VAE encoding/decoding to reduce memory usage. Must be even number. If not specified, chunking is disabled (official behavior)."
        + " / メモリ使用量を減らすためのVAEエンコード/デコードの空間チャンクサイズ。偶数である必要があります。未指定の場合、チャンク処理は無効になります（公式の動作）。",
    )
    parser.add_argument(
        "--vae_disable_cache",
        action="store_true",
        help="Disable internal VAE caching mechanism to reduce memory usage. Encoding / decoding will also be faster, but this differs from official behavior."
        + " / VAEのメモリ使用量を減らすために内部のキャッシュ機構を無効にします。エンコード/デコードも速くなりますが、公式の動作とは異なります。",
    )


# Loss weighting


def compute_loss_weighting_for_anima(weighting_scheme: str, sigmas: torch.Tensor) -> torch.Tensor:
    """Compute loss weighting for Anima training.

    Same schemes as SD3 but can add Anima-specific ones if needed in future.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    elif weighting_scheme == "none" or weighting_scheme is None:
        weighting = torch.ones_like(sigmas)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


# Parameter groups (6 groups with separate LRs)
def get_anima_param_groups(
    dit,
    base_lr: float,
    self_attn_lr: Optional[float] = None,
    cross_attn_lr: Optional[float] = None,
    mlp_lr: Optional[float] = None,
    mod_lr: Optional[float] = None,
    llm_adapter_lr: Optional[float] = None,
):
    """Create parameter groups for Anima training with separate learning rates.

    Args:
        dit: Anima model
        base_lr: Base learning rate
        self_attn_lr: LR for self-attention layers (None = base_lr, 0 = freeze)
        cross_attn_lr: LR for cross-attention layers
        mlp_lr: LR for MLP layers
        mod_lr: LR for AdaLN modulation layers
        llm_adapter_lr: LR for LLM adapter

    Returns:
        List of parameter group dicts for optimizer
    """
    if self_attn_lr is None:
        self_attn_lr = base_lr
    if cross_attn_lr is None:
        cross_attn_lr = base_lr
    if mlp_lr is None:
        mlp_lr = base_lr
    if mod_lr is None:
        mod_lr = base_lr
    if llm_adapter_lr is None:
        llm_adapter_lr = base_lr

    base_params = []
    self_attn_params = []
    cross_attn_params = []
    mlp_params = []
    mod_params = []
    llm_adapter_params = []

    for name, p in dit.named_parameters():
        # Store original name for debugging
        p.original_name = name

        if "llm_adapter" in name:
            llm_adapter_params.append(p)
        elif ".self_attn" in name:
            self_attn_params.append(p)
        elif ".cross_attn" in name:
            cross_attn_params.append(p)
        elif ".mlp" in name:
            mlp_params.append(p)
        elif ".adaln_modulation" in name:
            mod_params.append(p)
        else:
            base_params.append(p)

    logger.info(f"Parameter groups:")
    logger.info(f"  base_params: {len(base_params)} (lr={base_lr})")
    logger.info(f"  self_attn_params: {len(self_attn_params)} (lr={self_attn_lr})")
    logger.info(f"  cross_attn_params: {len(cross_attn_params)} (lr={cross_attn_lr})")
    logger.info(f"  mlp_params: {len(mlp_params)} (lr={mlp_lr})")
    logger.info(f"  mod_params: {len(mod_params)} (lr={mod_lr})")
    logger.info(f"  llm_adapter_params: {len(llm_adapter_params)} (lr={llm_adapter_lr})")

    param_groups = []
    for lr, params, name in [
        (base_lr, base_params, "base"),
        (self_attn_lr, self_attn_params, "self_attn"),
        (cross_attn_lr, cross_attn_params, "cross_attn"),
        (mlp_lr, mlp_params, "mlp"),
        (mod_lr, mod_params, "mod"),
        (llm_adapter_lr, llm_adapter_params, "llm_adapter"),
    ]:
        if lr == 0:
            for p in params:
                p.requires_grad_(False)
            logger.info(f"  Frozen {name} params ({len(params)} parameters)")
        elif len(params) > 0:
            param_groups.append({"params": params, "lr": lr})

    total_trainable = sum(p.numel() for group in param_groups for p in group["params"] if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_trainable:,}")

    return param_groups


# Save functions
def save_anima_model_on_train_end(
    args: argparse.Namespace,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    dit: anima_models.Anima,
):
    """Save Anima model at the end of training."""

    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec_dataclass(
            None, args, False, False, False, is_stable_diffusion_ckpt=True, anima="preview"
        ).to_metadata_dict()
        dit_sd = dit.state_dict()
        # Save with 'net.' prefix for ComfyUI compatibility
        anima_utils.save_anima_model(ckpt_file, dit_sd, sai_metadata, save_dtype)

    train_util.save_sd_model_on_train_end_common(args, True, True, epoch, global_step, sd_saver, None)


def save_anima_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator: Accelerator,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    dit: anima_models.Anima,
):
    """Save Anima model at epoch end or specific steps."""

    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec_dataclass(
            None, args, False, False, False, is_stable_diffusion_ckpt=True, anima="preview"
        ).to_metadata_dict()
        dit_sd = dit.state_dict()
        anima_utils.save_anima_model(ckpt_file, dit_sd, sai_metadata, save_dtype)

    train_util.save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        True,
        True,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        None,
    )


# Sampling (Euler discrete for rectified flow)
def do_sample(
    height: int,
    width: int,
    seed: Optional[int],
    dit: anima_models.Anima,
    crossattn_emb: torch.Tensor,
    steps: int,
    dtype: torch.dtype,
    device: torch.device,
    guidance_scale: float = 1.0,
    flow_shift: float = 3.0,
    neg_crossattn_emb: Optional[torch.Tensor] = None,
    like_inference: bool = False,
) -> torch.Tensor:
    """Generate a sample using Euler discrete sampling for rectified flow.

    like_inference (--anima_sample_like_inference): step the way ComfyUI's euler sampler does (Forge Neo the same) - sigmas,
    the latent between steps and the update in fp32, the model fed the latent in `dtype` and the timestep in fp32 when the
    DiT keeps its stream in fp32 (Anima's t_embedder makes its sinusoid in fp32 there). Off: the original trainer sampler,
    which rounds the sigma table and the latent to `dtype` (bf16) at every step.

    Args:
        height, width: Output image dimensions
        seed: Random seed (None for random)
        dit: Anima model
        crossattn_emb: Cross-attention embeddings (B, N, D)
        steps: Number of sampling steps
        dtype: Compute dtype
        device: Compute device
        guidance_scale: CFG scale (1.0 = no guidance)
        flow_shift: Flow shift parameter for rectified flow
        neg_crossattn_emb: Negative cross-attention embeddings for CFG
        like_inference: fp32 sampler state as ComfyUI / Forge Neo keep it (see above)

    Returns:
        Denoised latents
    """
    if like_inference:
        return _do_sample_like_inference(
            height, width, seed, dit, crossattn_emb, steps, dtype, device, guidance_scale, flow_shift, neg_crossattn_emb
        )
    # Latent shape: (1, 16, 1, H/8, W/8) for single image
    latent_h = height // 8
    latent_w = width // 8
    latent = torch.zeros(1, 16, 1, latent_h, latent_w, device=device, dtype=dtype)

    # Generate noise
    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None
    noise = torch.randn(latent.size(), dtype=torch.float32, generator=generator, device="cpu").to(dtype).to(device)

    # Timestep schedule: linear from 1.0 to 0.0
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    flow_shift = float(flow_shift)
    if flow_shift != 1.0:
        sigmas = (sigmas * flow_shift) / (1 + (flow_shift - 1) * sigmas)

    # Start from pure noise
    x = noise.clone()

    # Padding mask (zeros = no padding) — resized in prepare_embedded_sequence to match latent dims
    padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=dtype, device=device)

    use_cfg = guidance_scale > 1.0 and neg_crossattn_emb is not None

    for i in tqdm(range(steps), desc="Sampling"):
        sigma = sigmas[i]
        t = sigma.unsqueeze(0)  # (1,)

        if use_cfg:
            # CFG: two separate passes to reduce memory usage
            pos_out = dit(x, t, crossattn_emb, padding_mask=padding_mask)
            pos_out = pos_out.float()
            neg_out = dit(x, t, neg_crossattn_emb, padding_mask=padding_mask)
            neg_out = neg_out.float()

            model_output = neg_out + guidance_scale * (pos_out - neg_out)
        else:
            model_output = dit(x, t, crossattn_emb, padding_mask=padding_mask)
            model_output = model_output.float()

        # Euler step: x_{t-1} = x_t - (sigma_t - sigma_{t-1}) * model_output
        dt = sigmas[i + 1] - sigma
        x = x + model_output * dt
        x = x.to(dtype)

    return x


def comfy_simple_flow_sigmas(steps: int, flow_shift: float) -> torch.Tensor:
    """ComfyUI's "simple" schedule for a flow model (Anima: ModelSamplingDiscreteFlow, multiplier 1.0): the shifted table
    sigma(k / 1000) for k = 1..1000 (comfy/model_sampling.py set_parameters, time_snr_shift), indexed the way
    comfy/samplers.py simple_scheduler does it, then 0. The timestep is truncated to the 1/1000 grid, so this equals the
    shifted linspace only when 1000 % steps == 0 (20, 25, 40, 50 steps). fp32, on the CPU."""
    t = torch.arange(1, 1001, 1) / 1000
    shift = float(flow_shift)
    table = t if shift == 1.0 else shift * t / (1 + (shift - 1) * t)
    ss = len(table) / steps
    return torch.FloatTensor([float(table[-(1 + int(x * ss))]) for x in range(steps)] + [0.0])


def _model_input_dtype(dit, dtype: torch.dtype) -> torch.dtype:
    """The dtype ComfyUI / Forge Neo feed the DiT: its weights' compute dtype (comfy/model_base.py _apply_model casts xc to
    it). With --anima_stream_fp32 `dit.dtype` (its first parameter, the x_embedder) is fp32 while the Block Linears and the
    LLM adapter stay bf16, so the first low-precision parameter tells the compute dtype; all-fp32 models keep `dtype`."""
    parameters = getattr(dit, "parameters", None)
    if parameters is None:  # a wrapper (e.g. library/anima38 ConditionedDiT) without parameters
        return dtype
    return next((p.dtype for p in parameters() if p.dtype in (torch.bfloat16, torch.float16)), dtype)


def _do_sample_like_inference(
    height, width, seed, dit, crossattn_emb, steps, dtype, device, guidance_scale, flow_shift, neg_crossattn_emb
) -> torch.Tensor:
    """do_sample(like_inference=True): ComfyUI's euler with the "simple" scheduler, step for step.

    - noise: CPU randn in fp32 from torch.manual_seed(seed), the same draw as ComfyUI's comfy/sample.py prepare_noise and
      as the original sampler; the latent stays fp32 between steps (k-diffusion sample_euler);
    - sigmas: comfy_simple_flow_sigmas;
    - the DiT gets x in its compute dtype (_model_input_dtype) and the timestep in fp32 (model_base._apply_model passes
      t.float(); Anima builds the sinusoid from it) whenever the stream is fp32 or autocast is on - sampling runs under
      accelerator.autocast(), where t_embedder's Linears take the fp32 sinusoid; without either, the model dtype, so a
      bf16 Linear does not get an fp32 input;
    - CFG on the denoised predictions (CONST.calculate_denoised x - v * sigma, samplers.cfg_function), then the euler
      derivative (x - denoised) / sigma, all in fp32.
    Returns the fp32 latent (decode it with fp32_unnormalize=True, as ComfyUI's latent format un-normalizes in fp32)."""
    latent_h = height // 8
    latent_w = width // 8
    generator = torch.manual_seed(seed) if seed is not None else None
    x = torch.randn((1, 16, 1, latent_h, latent_w), dtype=torch.float32, generator=generator, device="cpu").to(device)
    sigmas = comfy_simple_flow_sigmas(steps, flow_shift).to(device)

    x_dtype = _model_input_dtype(dit, dtype)
    autocast_on = torch.is_autocast_enabled(device.type) if isinstance(device, torch.device) else torch.is_autocast_enabled()
    t_dtype = torch.float32 if (getattr(dit, "stream_fp32", False) or autocast_on) else x_dtype
    padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=x_dtype, device=device)
    use_cfg = guidance_scale > 1.0 and neg_crossattn_emb is not None

    for i in tqdm(range(steps), desc="Sampling"):
        sigma = sigmas[i]
        t = sigma.unsqueeze(0).to(t_dtype)
        x_in = x.to(x_dtype)
        denoised = x - dit(x_in, t, crossattn_emb, padding_mask=padding_mask).float() * sigma
        if use_cfg:
            uncond = x - dit(x_in, t, neg_crossattn_emb, padding_mask=padding_mask).float() * sigma
            denoised = uncond + (denoised - uncond) * guidance_scale
        x = x + (x - denoised) / sigma * (sigmas[i + 1] - sigma)

    return x


# ComfyUI's prompt-weight parser (comfy/sd1_clip.py parse_parentheses, token_weights, escape_important, unescape_important),
# which Forge Neo's backend/text_processing/_comfy.py uses too: only \( and \) are escapes; [ ] are plain text there.
def _comfy_parse_parentheses(string: str) -> list:
    result, current_item, nesting_level = [], "", 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result


def _comfy_token_weights(string: str, current_weight: float) -> list:
    out = []
    for x in _comfy_parse_parentheses(string):
        weight = current_weight
        if len(x) >= 2 and x[-1] == ")" and x[0] == "(":
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                try:
                    weight = float(x[xx + 1 :])
                    x = x[:xx]
                except ValueError:
                    pass
            out += _comfy_token_weights(x, weight)
        else:
            out += [(x, current_weight)]
    return out


def comfy_prompt_segments(text: str) -> list:
    """(text, weight) segments as ComfyUI tokenizes them, escapes restored."""
    escaped = text.replace("\\)", "\0\1").replace("\\(", "\0\2")
    return [(segment.replace("\0\1", ")").replace("\0\2", "("), weight) for segment, weight in _comfy_token_weights(escaped, 1.0)]


def process_prompt_escapes_like_inference(text: str) -> str:
    """A sample prompt's text as ComfyUI / Forge Neo pass it to the tokenizers: emphasis brackets and ":weight" removed,
    \\( \\) turned into plain brackets, \\[ \\] left as they are. The weights themselves are not applied to samples (warned in
    _sample_image_inference)."""
    return "".join(segment for segment, _ in comfy_prompt_segments(text))


def prompt_has_emphasis(text: str) -> bool:
    """Any segment ComfyUI would weight other than 1.0 ((x) is 1.1, (x:w) is w)."""
    return any(weight != 1.0 for _, weight in comfy_prompt_segments(text))


def sample_prompt_text(text: str, args) -> str:
    """The text a sample prompt is encoded from: unchanged, or with --anima_sample_like_inference its escapes processed the
    way ComfyUI / Forge Neo do. Used both where the sample prompts' TE outputs are cached and where they are looked up."""
    if text and getattr(args, "anima_sample_like_inference", False):
        return process_prompt_escapes_like_inference(text)
    return text


def sample_merge_cache(accelerator: Accelerator, args: argparse.Namespace, dit):
    """Context: merged LoKr weights built once per sampling round instead of on every call (bit-identical under
    autocast; library/lycoris_fast.py). Not with block swap, without autocast, or when switched off."""
    device_type = accelerator.device.type
    if (getattr(args, "anima_stock_lycoris", False) or getattr(args, "anima_no_sample_cache", False)
            or getattr(dit, "blocks_to_swap", None) or not torch.is_autocast_enabled(device_type)):
        return contextlib.nullcontext()
    return cached_lokr_weights(dit, torch.get_autocast_dtype(device_type), accelerator.device)


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    dit: anima_models.Anima,
    vae,
    text_encoder,
    tokenize_strategy,
    text_encoding_strategy,
    sample_prompts_te_outputs=None,
    prompt_replacement=None,
    on_prompt_start=None,
    on_prompt_end=None,
):
    """Generate sample images during training.

    This is a simplified sampler for Anima - it generates images using the current model state.

    on_prompt_start / on_prompt_end:
        Optional callbacks invoked around each prompt's `_sample_image_inference` call. Useful
        for injecting per-prompt state into wrapper modules (e.g. ControlNet-LLLite cond image).
        Signature: ``on_prompt_start(prompt_dict, accelerator)`` / ``on_prompt_end(prompt_dict)``.
    """
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

    logger.info(f"Generating sample images at step {steps}")
    if not os.path.isfile(args.sample_prompts) and sample_prompts_te_outputs is None:
        logger.error(f"No prompt file: {args.sample_prompts}")
        return

    # Unwrap models
    dit = accelerator.unwrap_model(dit)
    if text_encoder is not None:
        text_encoder = accelerator.unwrap_model(text_encoder)

    dit.switch_block_swap_for_inference()

    prompts = train_util.load_prompts(args.sample_prompts)
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)

    # Save RNG state
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    with torch.no_grad(), accelerator.autocast(), sample_merge_cache(accelerator, args, dit):
        for prompt_dict in prompts:
            dit.prepare_block_swap_before_forward()
            if on_prompt_start is not None:
                on_prompt_start(prompt_dict, accelerator)
            try:
                _sample_image_inference(
                    accelerator,
                    args,
                    dit,
                    text_encoder,
                    vae,
                    tokenize_strategy,
                    text_encoding_strategy,
                    save_dir,
                    prompt_dict,
                    epoch,
                    steps,
                    sample_prompts_te_outputs,
                    prompt_replacement,
                )
            finally:
                if on_prompt_end is not None:
                    on_prompt_end(prompt_dict)

    # Restore RNG state
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    dit.switch_block_swap_for_training()
    clean_memory_on_device(accelerator.device)


def _sample_image_inference(
    accelerator,
    args,
    dit,
    text_encoder,
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    tokenize_strategy,
    text_encoding_strategy,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    sample_prompts_te_outputs,
    prompt_replacement,
):
    """Generate a single sample image."""
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

    like_inference = bool(getattr(args, "anima_sample_like_inference", False))
    if like_inference:
        for text in (prompt, negative_prompt):
            if text and prompt_has_emphasis(text):
                logger.warning(f"  sample prompt has (text) / (text:weight) emphasis, which ComfyUI / Forge apply but samples do not: {text[:80]}")
    prompt = sample_prompt_text(prompt, args)  # same text as the cached TE outputs (see anima_train_network)
    negative_prompt = sample_prompt_text(negative_prompt, args)

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # seed all CUDA devices for multi-GPU

    height = max(64, height - height % 16)
    width = max(64, width - width % 16)

    logger.info(
        f"  prompt: {prompt}, size: {width}x{height}, steps: {sample_steps}, scale: {scale}, flow_shift: {flow_shift}, seed: {seed}"
    )

    # Encode prompt
    def encode_prompt(prpt):
        if sample_prompts_te_outputs and prpt in sample_prompts_te_outputs:
            return sample_prompts_te_outputs[prpt]
        if text_encoder is not None:
            tokens = tokenize_strategy.tokenize(prpt)
            encoded = text_encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
            return encoded
        return None

    encoded = encode_prompt(prompt)
    if encoded is None:
        logger.warning("Cannot encode prompt, skipping sample")
        return

    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = encoded

    # Convert to tensors if numpy
    if isinstance(prompt_embeds, np.ndarray):
        prompt_embeds = torch.from_numpy(prompt_embeds).unsqueeze(0)
        attn_mask = torch.from_numpy(attn_mask).unsqueeze(0)
        t5_input_ids = torch.from_numpy(t5_input_ids).unsqueeze(0)
        t5_attn_mask = torch.from_numpy(t5_attn_mask).unsqueeze(0)

    prompt_embeds = prompt_embeds.to(accelerator.device, dtype=dit.dtype)
    attn_mask = attn_mask.to(accelerator.device)
    t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
    t5_attn_mask = t5_attn_mask.to(accelerator.device)

    # Process through LLM adapter if available
    if dit.use_llm_adapter:
        crossattn_emb = dit.llm_adapter(
            source_hidden_states=prompt_embeds,
            target_input_ids=t5_input_ids,
            target_attention_mask=t5_attn_mask,
            source_attention_mask=attn_mask,
        )
        crossattn_emb[~t5_attn_mask.bool()] = 0
        if getattr(dit, "crossattn_like_inference", False):  # samples build the context here, outside Anima.forward
            crossattn_emb = anima_models.fit_crossattn_like_inference(crossattn_emb, t5_attn_mask)
    else:
        crossattn_emb = prompt_embeds

    # Encode negative prompt for CFG
    neg_crossattn_emb = None
    if scale > 1.0 and negative_prompt is not None:
        neg_encoded = encode_prompt(negative_prompt)
        if neg_encoded is not None:
            neg_pe, neg_am, neg_t5_ids, neg_t5_am = neg_encoded
            if isinstance(neg_pe, np.ndarray):
                neg_pe = torch.from_numpy(neg_pe).unsqueeze(0)
                neg_am = torch.from_numpy(neg_am).unsqueeze(0)
                neg_t5_ids = torch.from_numpy(neg_t5_ids).unsqueeze(0)
                neg_t5_am = torch.from_numpy(neg_t5_am).unsqueeze(0)

            neg_pe = neg_pe.to(accelerator.device, dtype=dit.dtype)
            neg_am = neg_am.to(accelerator.device)
            neg_t5_ids = neg_t5_ids.to(accelerator.device, dtype=torch.long)
            neg_t5_am = neg_t5_am.to(accelerator.device)

            if dit.use_llm_adapter:
                neg_crossattn_emb = dit.llm_adapter(
                    source_hidden_states=neg_pe,
                    target_input_ids=neg_t5_ids,
                    target_attention_mask=neg_t5_am,
                    source_attention_mask=neg_am,
                )
                neg_crossattn_emb[~neg_t5_am.bool()] = 0
                if getattr(dit, "crossattn_like_inference", False):
                    neg_crossattn_emb = anima_models.fit_crossattn_like_inference(neg_crossattn_emb, neg_t5_am)
            else:
                neg_crossattn_emb = neg_pe

    # Generate sample
    clean_memory_on_device(accelerator.device)
    latents = do_sample(
        height, width, seed, dit, crossattn_emb, sample_steps, dit.dtype, accelerator.device, scale, flow_shift, neg_crossattn_emb,
        like_inference=like_inference,
    )

    # Decode latents
    gc.collect()
    synchronize_device(accelerator.device)
    clean_memory_on_device(accelerator.device)
    org_vae_device = vae.device
    vae.to(accelerator.device)
    decoded = vae.decode_to_pixels(latents, fp32_unnormalize=True) if like_inference else vae.decode_to_pixels(latents)
    vae.to(org_vae_device)
    clean_memory_on_device(accelerator.device)

    # Convert to image
    image = decoded.float()
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
    # Remove temporal dim if present
    if image.ndim == 4:
        image = image[:, 0, :, :]
    decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
    decoded_np = decoded_np.astype(np.uint8)

    image = Image.fromarray(decoded_np)

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i = prompt_dict.get("enum", 0)
    img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(save_dir, img_filename))

    # Log to wandb if enabled
    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        wandb_tracker = accelerator.get_tracker("wandb")
        import wandb
        wandb_tracker.log({f"sample_{i}": wandb.Image(image, caption=prompt)}, commit=False)
