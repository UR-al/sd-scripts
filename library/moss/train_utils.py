# MOSS-Image training helpers: CLI arguments, option validation, metadata, sample images.

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from library import timestep_presets, train_util
from library.device_utils import clean_memory_on_device, synchronize_device
from library.utils import setup_logging

from .flow import TIMESTEP_CHOICES, describe_moss_sampling
from .models import MOSS_DIT_CONFIG
from library.lycoris_fast import cached_lokr_weights
from .vae import moss_vae_decode

setup_logging()
import logging

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SHIFT = 5.0  # MOSSImageScheduler.shift (inference)
DEFAULT_UNCOND_SKIP_LAYERS = "22"  # official negative-branch skip


def add_moss_training_arguments(parser) -> None:
    parser.add_argument("--moss_text_encoder", type=str, default=None,
                        help="Qwen3.5-9B text encoder safetensors (qwen3.5_9b_moss_image_sft_bf16.safetensors; Base and SFT"
                             " share it). Loaded only while caching text-encoder outputs")
    parser.add_argument("--moss_tokenizer_dir", type=str, default=None,
                        help="folder with the Qwen3.5 tokenizer.json (MOSS-Image tokenizer/ or any Qwen3.5 copy). Default:"
                             " $MOSS_TOKENIZER_DIR")
    parser.add_argument("--moss_max_token_length", type=int, default=4096, help="text tokens per caption incl. chat template")
    parser.add_argument("--timestep_sampling", type=str, default="moss", choices=TIMESTEP_CHOICES,
                        help="moss = the released Base's last pretraining distribution (logit-normal, see --moss_logit_*);"
                             " sigma/uniform/sigmoid/shift = the usual flow-matching options (fp32 timesteps)")
    parser.add_argument("--sigmoid_scale", type=timestep_presets.parse_sigmoid_scale, default=1.0,
                        help=timestep_presets.SIGMOID_SCALE_HELP)
    timestep_presets.add_timestep_preset_arguments(parser)
    parser.add_argument("--discrete_flow_shift", type=float, default=1.0,
                        help="shift for timestep_sampling=shift and the scheduler of sigma mode (MOSS training_shift = 1.0)")
    parser.add_argument("--moss_logit_mean", type=float, default=1.09, help="moss sampling: mean of logit(sigma)")
    parser.add_argument("--moss_logit_std", type=float, default=0.95, help="moss sampling: std of logit(sigma)")
    parser.add_argument("--moss_sigma_cutoff", type=float, default=0.95, help="moss sampling: truncation before rescaling")
    parser.add_argument("--moss_uniform_mix", type=float, default=0.01, help="moss sampling: weight of the uniform mixture")
    parser.add_argument("--attn_mode", type=str, default=None, choices=["torch", "xformers", "flash", "sageattn", "sdpa"],
                        help="attention backend (padding is handled with masks / varlen); default torch SDPA."
                             " flash (flash-attn varlen) is ~1.6x faster per step on an RTX 5090 at 1536")
    parser.add_argument("--split_attn", action="store_true", help="per-sample attention (slower, lowest memory)")
    parser.add_argument("--moss_no_compile", action="store_true",
                        help="do not torch.compile the MossBlock elementwise chains (RMSNorm/RoPE/SiTU/residual) on CUDA")
    parser.add_argument("--moss_no_lokr_compile", action="store_true",
                        help="do not torch.compile the LyCORIS LoKr+DoRA weight rebuild of the fast forward (compiled results"
                             " differ by ~1 bf16 ulp)")
    parser.add_argument("--moss_stock_lycoris", action="store_true",
                        help="use LyCORIS's stock LoKr forward and per-call sample merges (slower; same results)")
    parser.add_argument("--moss_no_sample_cache", action="store_true",
                        help="sample images without the merged LoKr weight cache (~9 GB less VRAM while sampling; slower)")
    parser.add_argument("--moss_lora_exclude_blocks", type=moss_exclude_blocks_arg, default="",
                        help="comma-separated joint block indices (0-41) left out of the LoRA/LyCORIS network, e.g. \"22\":"
                             " the block the official sampler's negative branch skips, so CFG's skipped-block contrast stays"
                             " the base model's (ai-toolkit's MOSS preset does this). Changes the network: a --resume state or"
                             " --network_weights trained with those blocks will not match")
    parser.add_argument("--moss_sample_shift", type=float, default=DEFAULT_SAMPLE_SHIFT,
                        help="sample images: sigma shift (official 5.0); --fs in a prompt line overrides it")
    parser.add_argument("--moss_sample_uncond_skip_layers", type=str, default=DEFAULT_UNCOND_SKIP_LAYERS,
                        help="sample images: main blocks skipped in the negative CFG branch (official '22'; '' = off)")
    parser.add_argument("--moss_tier_timing", action="store_true",
                        help="--anima_tier_timing for MOSS (library/tier_timing.py): time every micro-step's forward +"
                             " backward and group the times by resolution tier of its image tokens per sample (one latent"
                             " pixel = one DiT token: 512^2 = 1024, 1024^2 = 4096, 1536^2 = 9216, 2048^2 = 16384); the text"
                             " tokens of the joint sequence are reported per tier next to it (tier_time/<tier>/text_tokens,"
                             " 'text N tok'). CUDA: event pairs read only once finished (no synchronization); CPU:"
                             " perf_counter. Every --moss_tier_timing_every optimizer steps the window goes to the tracker,"
                             " at the end of every epoch one INFO line. Off (default): one attribute check per micro-step")
    parser.add_argument("--moss_tier_timing_every", type=int, default=50,
                        help="optimizer steps between --moss_tier_timing reports (default 50)")
    # Accepted and ignored so Anima command lines / train.ps1 do not fail argparse.
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="ignored for MOSS-Image")
    parser.add_argument("--vae_disable_cache", action="store_true", help="ignored for MOSS-Image")
    parser.add_argument("--show_timesteps", type=str, default=None, choices=["console", "image"],
                        help="print (console) or plot (image, matplotlib) the timestep distribution and loss weighting the"
                             " current settings sample, then exit without training (CPU only, no model loaded)")
    parser.add_argument("--show_timesteps_epoch", type=int, default=None,
                        help="--show_timesteps: preview the distribution of this 1-based epoch (scale_resume switches to the"
                             " low-noise 'style' preset after max_train_epochs)")


def _datasets(groups: Sequence) -> List:
    out = []
    for group in groups or ():
        if group is not None:
            out.extend(getattr(group, "datasets", [group]))
    return out


def validate_moss_args(args, train_dataset_group, val_dataset_group) -> None:
    """Refuse option combinations the MOSS path does not support (ValueError with the reason)."""

    def fail(message):
        raise ValueError(f"MOSS-Image: {message}")

    if not args.pretrained_model_name_or_path or not os.path.isfile(args.pretrained_model_name_or_path):
        fail(f"pretrained_model_name_or_path (moss_image_base_bf16.safetensors) not found: {args.pretrained_model_name_or_path}")
    if not args.vae or not os.path.isfile(args.vae):
        fail(f"vae (moss_image_vae_fp32.safetensors) not found: {args.vae}")
    if args.timestep_sampling not in TIMESTEP_CHOICES:  # TOML values bypass argparse choices
        fail(f"timestep_sampling must be one of {TIMESTEP_CHOICES} (flux_shift assumes 8x latents), got {args.timestep_sampling}")
    if not 1 <= int(args.moss_max_token_length) <= 4096:
        fail("moss_max_token_length must be in 1..4096")
    if getattr(args, "moss_tier_timing", False) and int(getattr(args, "moss_tier_timing_every", 50) or 0) < 1:
        fail(f"--moss_tier_timing_every must be >= 1 (optimizer steps between reports), got {args.moss_tier_timing_every}")
    if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
        args.cache_text_encoder_outputs = True
    if not args.cache_text_encoder_outputs:
        fail("--cache_text_encoder_outputs is required (the 16 GB Qwen3.5-9B is only loaded while caching; the user guide"
             " keeps the text encoder frozen)")
    if not args.network_train_unet_only:
        logger.warning("MOSS-Image: the text encoder is frozen; forcing --network_train_unet_only")
        args.network_train_unet_only = True
    for flag in ("weighted_captions", "adaptive_timestep_sampling", "network_train_text_encoder_only", "flow_model"):
        if getattr(args, flag, False):
            fail(f"--{flag} is not supported")
    if getattr(args, "fp8_base", False) or getattr(args, "fp8_base_unet", False):
        logger.warning("MOSS-Image: fp8_base is not supported, disabled")
        args.fp8_base = False
        args.fp8_base_unet = False
    args.fp8_scaled = False
    if (args.blocks_to_swap or 0) > 0 and (args.cpu_offload_checkpointing or args.unsloth_offload_checkpointing):
        fail("blocks_to_swap cannot be combined with cpu/unsloth offload checkpointing")
    if args.unsloth_offload_checkpointing:
        if args.cpu_offload_checkpointing:
            fail("use either --unsloth_offload_checkpointing or --cpu_offload_checkpointing")
        if not args.gradient_checkpointing:
            logger.warning("unsloth_offload_checkpointing is on, enabling gradient_checkpointing")
            args.gradient_checkpointing = True
    if getattr(args, "hf_scale", 0.0) and getattr(args, "hf_patch", 1) != 1:
        logger.warning("MOSS latents are already 2x2 packed tokens: setting hf_patch=1")
        args.hf_patch = 1

    assert train_dataset_group.is_text_encoder_output_cacheable(cache_supports_dropout=True), (
        "when caching text-encoder outputs, shuffle_caption / caption_tag_dropout_rate need --cache_caption_variants N>1,"
        " and token_warmup_step cannot be used")
    for dataset in _datasets([train_dataset_group, val_dataset_group]):
        variants = int(getattr(dataset, "caption_aug_variant_count", 0) or 0)
        if variants > 1 and int(getattr(dataset, "aug_refresh_epochs", 0) or 0) > 0:
            fail("--cache_aug_refresh_epochs is not supported with caption variants (it re-encodes through the text encoder)")
        for subset in getattr(dataset, "subsets", []):
            if int(getattr(subset, "caption_dropout_every_n_epochs", 0) or 0) > 0:
                fail("caption_dropout_every_n_epochs is not supported (it zeroes cached outputs; MOSS needs >= 1 text"
                     " token). Use caption_dropout_rate with --cache_caption_variants N>1")
            if float(getattr(subset, "caption_dropout_rate", 0.0) or 0.0) > 0.0 and variants <= 1:
                fail("caption_dropout_rate > 0 needs --cache_caption_variants N>1 (a dropped caption is encoded as ''"
                     " through the chat template, the official unconditional prompt)")
    train_dataset_group.verify_bucket_reso_steps(16)
    if val_dataset_group is not None:
        val_dataset_group.verify_bucket_reso_steps(16)
    logger.info("MOSS-Image: " + describe_moss_sampling(args))


def show_timesteps(args, n_try: int = 100000) -> None:
    """--show_timesteps: sample the configured timestep distribution on the CPU and print / plot it (no model).

    Goes through the trainer's own sampling path (flow.get_noisy_model_input_and_timesteps with the moss pmf sampler,
    the sigmoid presets and scale_resume) and loss weighting, so the histogram is what training samples.
    library/timestep_visualization.py is upstream sd-scripts' renderer."""
    from library import sd3_train_utils, timestep_visualization
    from library.anima_train_utils import compute_loss_weighting_for_anima

    from .flow import NUM_TRAIN_TIMESTEPS, MossSigmaSampler, get_noisy_model_input_and_timesteps

    timestep_presets.resolve(args)
    epoch = getattr(args, "show_timesteps_epoch", None)
    if epoch is not None:
        timestep_presets.set_epoch(args, int(epoch))
    scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS,
                                                                shift=args.discrete_flow_shift)
    sampler = MossSigmaSampler(args) if args.timestep_sampling == "moss" else None
    device, dtype = torch.device("cpu"), torch.float32

    def sample_timesteps(batch_size: int) -> torch.Tensor:
        latents = torch.zeros(batch_size, MOSS_DIT_CONFIG["latent_channels"], 4, 4, dtype=dtype, device=device)
        _, timesteps, _ = get_noisy_model_input_and_timesteps(args, scheduler, latents, torch.ones_like(latents), device,
                                                              dtype, sampler)
        return timesteps

    def compute_weighting(timesteps: torch.Tensor) -> torch.Tensor:
        sigmas = timesteps / NUM_TRAIN_TIMESTEPS
        weighting = compute_loss_weighting_for_anima(args.weighting_scheme, sigmas)
        if getattr(args, "min_snr_gamma", None):  # the trainer applies it on top (post_process_loss)
            from library.custom_train_functions import apply_snr_weight_for_flow_matching

            weighting = apply_snr_weight_for_flow_matching(weighting, sigmas, args.min_snr_gamma,
                                                           soft=bool(getattr(args, "min_snr_gamma_soft", False)))
        return weighting

    header = "MOSS-Image timestep distribution: " + describe_moss_sampling(args)
    if epoch is not None and args.timestep_sampling == "sigmoid":
        header += f" (epoch {int(epoch)}: preset {timestep_presets.active_sigmoid_preset(args) or 'numeric scale'})"
    elif epoch is not None:
        header += f" (epoch {int(epoch)})"
    timestep_visualization.show_timestep_distribution(args.show_timesteps, sample_timesteps, compute_weighting,
                                                      num_train_timesteps=NUM_TRAIN_TIMESTEPS, n_try=n_try, header=header)


def parse_moss_exclude_blocks(value, num_layers: int = MOSS_DIT_CONFIG["num_layers"]) -> List[int]:
    """--moss_lora_exclude_blocks: "22", "22, 23", 22 or [22, 23] (TOML values arrive unconverted) -> sorted indices."""
    if value is None or value == "" or value == []:
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    blocks = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        if isinstance(item, bool) or not text.lstrip("-").isdigit():
            raise ValueError(f"MOSS-Image: moss_lora_exclude_blocks takes block indices, got {value!r}")
        index = int(text)
        if not 0 <= index < num_layers:
            raise ValueError(f"MOSS-Image: moss_lora_exclude_blocks index {index} is outside 0..{num_layers - 1}")
        blocks.add(index)
    return sorted(blocks)


def moss_exclude_blocks_arg(value):
    """argparse type of --moss_lora_exclude_blocks: checks the indices and keeps the value as given. config_validation
    converts a TOML string with a custom type (a bad "22,x" fails when the config is read, with the file and line) and
    passes an array or integer through unconverted to parse_moss_exclude_blocks; type=str refused [22, 23]."""
    try:
        parse_moss_exclude_blocks(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    return value


def moss_block_exclude_patterns(blocks: Sequence[int]) -> List[str]:
    """Regexes for network_args exclude_patterns. LyCORIS full-matches module paths ("layers.22" and its children),
    networks.lora_moss full-matches "layers.22.proj.q" etc."""
    return [rf"layers\.{index}(\..*)?" for index in blocks]


def apply_moss_block_exclusion(args) -> List[int]:
    """Merge the excluded blocks into network_args exclude_patterns (both network modules read it). Idempotent."""
    if getattr(args, "_moss_excluded_blocks", None) is not None:
        return args._moss_excluded_blocks
    blocks = parse_moss_exclude_blocks(getattr(args, "moss_lora_exclude_blocks", ""))
    args._moss_excluded_blocks = blocks
    if not blocks:
        return blocks
    patterns = moss_block_exclude_patterns(blocks)
    network_args = list(getattr(args, "network_args", None) or [])
    if "lycoris" in str(getattr(args, "network_module", "")):
        # LyCORIS ignores a preset's exclude_name once exclude_patterns is given; the MOSS presets have none.
        preset = next((str(e).partition("=")[2].strip() for e in network_args if str(e).partition("=")[0].strip() == "preset"),
                      "full")
        if preset not in ("moss", "moss-text"):
            raise ValueError(f"MOSS-Image: moss_lora_exclude_blocks needs LyCORIS preset=moss or moss-text, got {preset!r}")
    for i, entry in enumerate(network_args):
        key, _, value = str(entry).partition("=")
        if key.strip() == "exclude_patterns":
            existing = ast.literal_eval(value.strip())  # the network modules parse it the same way
            existing = list(existing) if isinstance(existing, (list, tuple)) else [existing]
            network_args[i] = f"exclude_patterns={existing + [p for p in patterns if p not in existing]!r}"
            break
    else:
        network_args.append(f"exclude_patterns={patterns!r}")
    args.network_args = network_args
    logger.info(f"MOSS-Image: joint blocks {blocks} are left out of the network (exclude_patterns {patterns})")
    return blocks


def moss_metadata(args) -> Dict[str, str]:
    excluded = getattr(args, "_moss_excluded_blocks", None) or []
    return {
        "ss_moss_lora_exclude_blocks": ",".join(str(index) for index in excluded),
        "ss_base_model_version": "moss_image",
        "ss_moss_text_encoder": os.path.basename(str(args.moss_text_encoder)),
        "ss_moss_max_token_length": str(args.moss_max_token_length),
        "ss_timestep_sampling": str(args.timestep_sampling),
        "ss_moss_logit_mean": str(args.moss_logit_mean),
        "ss_moss_logit_std": str(args.moss_logit_std),
        "ss_moss_sigma_cutoff": str(args.moss_sigma_cutoff),
        "ss_moss_uniform_mix": str(args.moss_uniform_mix),
        **timestep_presets.metadata(args),
        "ss_discrete_flow_shift": str(args.discrete_flow_shift),
        "ss_weighting_scheme": str(args.weighting_scheme),
        "ss_logit_mean": str(args.logit_mean),
        "ss_logit_std": str(args.logit_std),
    }


# ---- sample prompt text-encoder outputs: kept on disk so a run with complete caches never loads the 9B model -------
def sample_prompt_cache_dir(args) -> Optional[str]:
    if not args.sample_prompts:
        return None
    return os.path.splitext(os.path.abspath(args.sample_prompts))[0] + "_moss_te"


def sample_prompt_cache_path(cache_dir: str, signature: str, prompt: str) -> str:
    digest = hashlib.sha1((signature + "\n" + prompt).encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, digest + ".safetensors")


def load_cached_sample_prompt(path: str):
    if not os.path.isfile(path):
        return None
    state = load_file(path)
    return [state["text_embeds"], state["text_mask"]]


def save_cached_sample_prompt(path: str, outputs) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({"text_embeds": outputs[0].contiguous().cpu(), "text_mask": outputs[1].contiguous().cpu()}, path)


def sample_prompt_texts(args) -> List[str]:
    texts: List[str] = []
    for prompt_dict in train_util.load_prompts(args.sample_prompts):
        for text in (prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "") or ""):
            if text not in texts:
                texts.append(text)
    return texts


# ---- sampling (official MOSSImageScheduler grid + Euler + CFG with negative-branch layer skip) ----------------------
def moss_inference_sigmas(steps: int, shift: float = DEFAULT_SAMPLE_SHIFT, total: int = 1000, training_shift: float = 1.0):
    """MOSSImageScheduler.set_timesteps: shifted grid snapped to the nearest training sigma (ties -> lower), + 0."""

    def grid(count, s):
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(steps, shift)
    training = grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


@torch.no_grad()
def moss_sample(dit, cond, neg, height: int, width: int, steps: int, scale: float, shift: float, seed: Optional[int],
                device, skip_layers: Sequence[int] = (22,)) -> torch.Tensor:
    """Returns DiT latents [1,128,H/16,W/16] (fp32). cond/neg = (text_embeds [1,L,4096], text_mask [1,L])."""
    generator = torch.Generator("cpu").manual_seed(seed) if seed is not None else None
    latents = torch.randn((1, 128, height // 16, width // 16), generator=generator, dtype=torch.float32).to(device)
    sigmas = moss_inference_sigmas(steps, shift).to(device)
    text, mask = cond[0].to(device, dtype=dit.dtype), cond[1].to(device).bool()
    if neg is not None:
        neg_text, neg_mask = neg[0].to(device, dtype=dit.dtype), neg[1].to(device).bool()
    for i in range(len(sigmas) - 1):
        t = (sigmas[i] * 1000.0).reshape(1)
        pred = dit(latents, t, text, mask).float()
        if scale != 1.0 and neg is not None:
            uncond = dit(latents, t, neg_text, neg_mask, skip_layers=skip_layers).float()
            pred = uncond + scale * (pred - uncond)
        latents = latents + (sigmas[i + 1] - sigmas[i]) * pred
    return latents


def sample_images(accelerator, args, epoch, steps, dit, vae, sample_prompts_te_outputs: Dict[str, list]):
    if not train_util.sample_images_check(args, epoch, steps):
        return
    if not sample_prompts_te_outputs:
        logger.warning("MOSS sample images: no cached prompt embeddings, skipping")
        return
    if not accelerator.is_main_process:  # one process samples; the others would write the same file names
        accelerator.wait_for_everyone()
        return
    logger.info(f"Generating MOSS sample images at step {steps}")
    dit = accelerator.unwrap_model(dit)
    dit.switch_block_swap_for_inference()
    prompts = train_util.load_prompts(args.sample_prompts)
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)
    skip = [int(x) for x in str(args.moss_sample_uncond_skip_layers or "").replace(",", " ").split()]

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    try:
        with torch.no_grad(), accelerator.autocast():
            device_type = accelerator.device.type
            # Merged LoKr weights built once per round (bit-identical to the per-call rebuild under autocast). Not with
            # block swap (weights move between CPU and GPU), without autocast, or when switched off. All prompts run
            # inside the cache; the VAE decodes after it is released so the two never hold VRAM at the same time.
            use_cache = (not getattr(args, "moss_stock_lycoris", False) and not getattr(args, "moss_no_sample_cache", False)
                         and not dit.blocks_to_swap and torch.is_autocast_enabled(device_type))
            cache = (cached_lokr_weights(dit, torch.get_autocast_dtype(device_type), accelerator.device) if use_cache
                     else contextlib.nullcontext())
            generated = []
            with cache:
                for prompt_dict in prompts:
                    dit.prepare_block_swap_before_forward()
                    latents = _generate_one(accelerator, args, dit, prompt_dict, sample_prompts_te_outputs, skip)
                    if latents is not None:
                        generated.append((prompt_dict, latents.cpu()))
            clean_memory_on_device(accelerator.device)
            for prompt_dict, latents in generated:
                _decode_and_save(accelerator, args, vae, save_dir, prompt_dict, epoch, steps, latents)
    finally:
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)
        dit.switch_block_swap_for_training()
        clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()


def _generate_one(accelerator, args, dit, prompt_dict, te_outputs, skip_layers) -> Optional[torch.Tensor]:
    """DiT sampling for one prompt line; returns latents [1,128,h,w] (fp32) or None when the prompt has no embedding."""
    prompt = prompt_dict.get("prompt", "")
    negative = prompt_dict.get("negative_prompt", "") or ""
    sample_steps = int(prompt_dict.get("sample_steps", 28))
    scale = float(prompt_dict.get("scale", 3.0))
    shift = float(prompt_dict.get("flow_shift", args.moss_sample_shift) or args.moss_sample_shift)
    seed = prompt_dict.get("seed")
    width = max(256, int(prompt_dict.get("width", 1024)) // 16 * 16)
    height = max(256, int(prompt_dict.get("height", 1024)) // 16 * 16)
    if prompt not in te_outputs or negative not in te_outputs:
        logger.warning(f"  no cached embedding for this prompt, skipped: {prompt[:80]}")
        return None
    logger.info(f"  prompt: {prompt}, size: {width}x{height}, steps: {sample_steps}, cfg: {scale}, shift: {shift}, seed: {seed}")
    clean_memory_on_device(accelerator.device)
    latents = moss_sample(dit, te_outputs[prompt], te_outputs[negative], height, width, sample_steps, scale, shift, seed,
                          accelerator.device, skip_layers)
    synchronize_device(accelerator.device)
    return latents


def _decode_and_save(accelerator, args, vae, save_dir, prompt_dict, epoch, steps, latents):
    seed = prompt_dict.get("seed")
    clean_memory_on_device(accelerator.device)
    org_device = vae.device
    vae.to(accelerator.device)
    with torch.autocast(device_type=accelerator.device.type, enabled=False):
        pixels = moss_vae_decode(vae, latents.to(accelerator.device))
    vae.to(org_device)
    clean_memory_on_device(accelerator.device)

    image = torch.clamp((pixels.float()[0] + 1.0) / 2.0, 0.0, 1.0)
    image = Image.fromarray((255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)).round().astype(np.uint8))
    ts = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    index = prompt_dict.get("enum", 0)
    name = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{index:02d}_{ts}{seed_suffix}.png"
    image.save(os.path.join(save_dir, name))
