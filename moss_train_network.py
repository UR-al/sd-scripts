# MOSS-Image (OpenMOSS) LoRA / LyCORIS training script.
#
# Model: MOSS-Image-Base (recommended for training: no post-training, per the official user guide) or -SFT, a 5B
# single-stream flow DiT with a frozen Qwen3.5-9B text encoder and the FLUX.2 VAE. See library/moss/.
#   - text-encoder outputs MUST be cached (the 9B encoder is loaded only while caching, then released);
#   - latents are 128-channel FLUX.2 latents at stride 16 (buckets must be multiples of 16);
#   - target = noise - latents (velocity), timestep input = 1000 * sigma in fp32;
#   - LoRA keys are lora_unet_layers_N_proj_{q,k,v,o,ffn_up_gate,ffn_down} (ComfyUI loads them);
#   - LyCORIS: network_args "preset=moss" (or "moss-text" to add the two text refiner blocks).

import argparse
from types import SimpleNamespace
from typing import Optional, Union

import torch
from accelerate import Accelerator

from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

import train_network
from library import anima_train_utils, attention, sd3_train_utils, strategy_base, tier_timing, timestep_presets, train_util
from library.custom_train_functions import apply_snr_weight_for_flow_matching, maybe_apply_antithetic_noise_pairing
from library.moss import captions, flow, strategy
from library.moss import train_utils as moss_train_utils
from library.moss import block_compile as moss_block_compile
from library.lycoris_fast import (ckpt_merged_bytes, ckpt_merged_invalidate, ckpt_merged_invalidate_on_optimizer_steps,
                                  enable_compiled_rebuild, install_fast_lokr_forward)
from library.moss.lycoris_presets import register_moss_lycoris_presets
from library.moss.models import (MOSS_DIT_CONFIG, MossTransformer, ckpt_nograd_merged_enabled, ckpt_node_enabled,
                                 ckpt_reuse_budget_bytes, enable_elementwise_compile, set_ckpt_reuse, split_fused_qkv)
from library.moss.vae import load_moss_vae, moss_vae_encode
from library.ramtorch_util import apply_ramtorch_to_module
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def load_moss_dit(path: str, attn_mode: str, split_attn: bool, device, dtype) -> MossTransformer:
    """Strict load of the single-file DiT (bare keys, as in moss_image_base_bf16.safetensors). A file with fused joint-block
    q/k/v (KBlueLeaf/MOSS-Image-heal's ComfyUI files, layers.N.proj.qkv.weight) is split back into q / k / v first."""
    from safetensors.torch import load_file

    with torch.device("meta"):
        model = MossTransformer(**MOSS_DIT_CONFIG, attn_mode=attn_mode, split_attn=split_attn)
    state = load_file(path, device="cpu")
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer.", "model."):
        if all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}
            break
    fused = split_fused_qkv(state)
    if fused:
        logger.info(f"MOSS DiT: fused q/k/v split back into q / k / v in {fused} joint blocks (ComfyUI layout, e.g. heal)")
    state = {k: (v if k == "rope.bands" else v.to(dtype)) for k, v in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    model.rope.bands = model.rope._default_bands(torch.device("cpu"))
    model.to(device)
    logger.info(f"MOSS DiT loaded: {path} ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, {dtype},"
                f" {device}, attn {attn_mode}{', split' if split_attn else ''})")
    return model


def apply_ramtorch_to_moss_blocks(model: MossTransformer, args, device, dtype, apply=None) -> list:
    """--use_ramtorch: stream the joint blocks' Linears (8.9 GB, the LoRA targets) from pinned CPU RAM; LyCORIS streams them
    into its rebuild, so training is unchanged. Blocks left out of the LoRA (--moss_lora_exclude_blocks) and the small
    layers stay ordinary GPU modules: RamTorch's own backward would compute (and stochastically round, using the RNG)
    grads for their frozen weights. Returns the indices of the streamed blocks."""
    apply = apply or apply_ramtorch_to_module
    excluded = set(moss_train_utils.parse_moss_exclude_blocks(getattr(args, "moss_lora_exclude_blocks", "")))
    streamed = []
    for index in range(len(model.layers)):
        if index in excluded:
            continue
        model.layers[index] = apply(model.layers[index], f"MOSS joint block {index}", device, dtype)
        streamed.append(index)
    kept = sorted(excluded)
    logger.info(f"RamTorch applied to the Linears of {len(streamed)} MOSS joint blocks"
                + (f"; blocks {kept} (no LoRA) stay on the GPU" if kept else ""))
    return streamed


class MossNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.hf_prediction_mode = "flow"
        self.hf_timesteps_in_sigma = True
        self.anchor_prediction_mode = "flow"
        self.anchor_timesteps_in_sigma = True
        self.is_swapping_blocks = False
        self._use_unsloth_offload_checkpointing = False
        self._sigma_sampler: Optional[flow.MossSigmaSampler] = None
        self._te_provider: Optional[strategy.MossTextEncoderProvider] = None
        self.latent_stride = strategy.MossLatentsCachingStrategy.STRIDE  # adv reference keys of disk-cached latents
        self._ot_logged = False  # --flow_use_ot: the first batch's assignment is logged once
        self._tsm_grouped_supported = True  # TSM-LoKr tsm_grouped_accumulation: see get_noise_pred_and_target
        self._tier_timer = None  # --moss_tier_timing: library/tier_timing.TierTimer, made at the first micro-step
        self._tier_dit = None  # the unwrapped DiT of the timed micro-step (its forward records the text length)
        self._ckpt_merged_unet = None  # --moss_ckpt_nograd_merged: the DiT, for the one-time cache size log (on_epoch_end)
        self._ckpt_merged_logged = False
        self._dataset_limits = (1, 0, None)  # largest batch size / image tokens / bucket of the datasets (assert_extra_args)
        self._dataset_tiers = []  # (tier, image tokens, bucket) of the largest bucket per resolution tier (the same)
        self._ckpt_vram_device = None  # the no-grad first-pass options on a CUDA device: the VRAM peak is watched there
        self._ckpt_vram_warned = False  # the at-the-limit warning, once
        self._ckpt_vram_reported = -1  # peak reserved bytes of the last epoch-end line

    def get_adaptive_model_type(self, args) -> str:
        return "flow_matching"

    # ---- checks / models ------------------------------------------------------------------------------------
    def assert_extra_args(self, args, train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
                          val_dataset_group: Optional[train_util.DatasetGroup]):
        moss_train_utils.validate_moss_args(args, train_dataset_group, val_dataset_group)
        # a dataset TOML sets batch_size and the resolution tiers per dataset: the args do not show them
        self._dataset_limits = moss_train_utils.dataset_group_limits(train_dataset_group)
        self._dataset_tiers = moss_train_utils.dataset_group_tiers(train_dataset_group)
        if not args.moss_text_encoder:
            raise ValueError("MOSS-Image: --moss_text_encoder (qwen3.5_9b_moss_image_sft_bf16.safetensors) is required")
        strategy.check_moss_text_encoder_path(args.moss_text_encoder)
        self._sigma_sampler = flow.MossSigmaSampler(args)

    def load_target_model(self, args, weight_dtype, accelerator):
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
        vae = load_moss_vae(args.vae, device="cpu", dtype=torch.float32)  # fp32 like the official pipeline
        self._te_provider = strategy.MossTextEncoderProvider(args.moss_text_encoder, accelerator.device, torch.bfloat16)
        return "moss-image", [strategy.MossTextEncoderStub(self._te_provider)], vae, None  # DiT loaded lazily

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders):
        attn_mode = "torch"
        if args.xformers:
            attn_mode = "xformers"
        if args.attn_mode is not None:
            attn_mode = "torch" if args.attn_mode == "sdpa" else args.attn_mode
        loading_device = "cpu" if (self.is_swapping_blocks or args.use_ramtorch) else accelerator.device
        model = load_moss_dit(args.pretrained_model_name_or_path, attn_mode, args.split_attn, loading_device, weight_dtype)
        stream_scale = float(getattr(args, "moss_stream_scale", 1.0))
        if stream_scale != 1.0:
            # --moss_stream_scale (validated in assert_extra_args): set at load, not next to enable_stream_fp32 in
            # prepare_unet_with_accelerator, so every forward has it - also a GoRA / RaLoRA init pre-pass, which runs
            # before that. Plain attributes (norm eps, a float): no dtype cast, RamTorch or accelerate wrap touches them.
            model.set_stream_scale(stream_scale)
        self._use_unsloth_offload_checkpointing = args.unsloth_offload_checkpointing
        if args.use_ramtorch:
            apply_ramtorch_to_moss_blocks(model, args, accelerator.device, model.dtype)
        if self.is_swapping_blocks:
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)
        return model, text_encoders

    # ---- strategies -----------------------------------------------------------------------------------------
    def get_tokenize_strategy(self, args):
        return strategy.MossTokenizeStrategy(args.moss_tokenizer_dir, args.moss_max_token_length)

    def get_tokenizers(self, tokenize_strategy: strategy.MossTokenizeStrategy):
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy.MossLatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check,
                                                   cache_dtype=getattr(args, "cache_latents_dtype", "auto"))

    def get_text_encoding_strategy(self, args):
        return strategy.MossTextEncodingStrategy()

    def get_text_encoder_outputs_caching_strategy(self, args):
        return strategy.MossTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check,
            cache_dtype=getattr(args, "cache_text_encoder_outputs_dtype", "auto"),
            signature=strategy.moss_te_signature(args.moss_text_encoder, args.moss_max_token_length, args.moss_tokenizer_dir))

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        return None  # outputs are always cached

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        pass

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        """Cache dataset captions (every variant) and sample prompts. The 9B encoder loads on the first caption that
        actually needs encoding and is released at the end, so complete caches never load it."""
        provider = self._te_provider
        caching = strategy_base.TextEncoderOutputsCachingStrategy.get_strategy()
        captions.install_canonical_captions(dataset)
        captions.prepare_canonical_captions(dataset)
        captions.register_caption_digests(dataset, caching)
        org_vae_device = vae.device
        if not args.lowram:
            vae.to("cpu")
            clean_memory_on_device(accelerator.device)
        try:
            dataset.new_cache_text_encoder_outputs([provider], accelerator)
            if args.sample_prompts is not None and self.sample_prompts_te_outputs is None:
                self.sample_prompts_te_outputs = self._cache_sample_prompts(args, provider)
            accelerator.wait_for_everyone()
        finally:
            captions.uninstall_canonical_captions(dataset)  # DataLoader workers pickle the dataset
            if provider.loaded:
                provider.release()
                logger.info("released the MOSS text encoder after caching")
            clean_memory_on_device(accelerator.device)
            if not args.lowram:
                vae.to(org_vae_device)

    def _cache_sample_prompts(self, args, provider):
        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
        signature = strategy.moss_te_signature(args.moss_text_encoder, args.moss_max_token_length, args.moss_tokenizer_dir)
        cache_dir = moss_train_utils.sample_prompt_cache_dir(args)
        outputs = {}
        for text in moss_train_utils.sample_prompt_texts(args):
            path = moss_train_utils.sample_prompt_cache_path(cache_dir, signature, text)
            cached = moss_train_utils.load_cached_sample_prompt(path)
            if cached is None:
                logger.info(f"  encode sample prompt: {text[:100]}")
                with torch.no_grad():
                    cached = encoding_strategy.encode_tokens(tokenize_strategy, [provider], tokenize_strategy.tokenize(text))
                cached = [cached[0].to("cpu"), cached[1].to("cpu")]
                moss_train_utils.save_cached_sample_prompt(path, cached)
            outputs[text] = cached
        logger.info(f"sample prompt embeddings ready: {len(outputs)} prompts ({cache_dir})")
        return outputs

    def is_text_encoder_not_needed_for_training(self, args):
        return True

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        pass

    def cast_vae(self, args):
        return False  # keep the fp32 VAE (latents are cached from fp32 encodes)

    def cast_text_encoder(self, args):
        return False

    # ---- training step ----------------------------------------------------------------------------------------
    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device):
        return sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)

    def encode_images_to_latents(self, args, vae, images: torch.FloatTensor) -> torch.FloatTensor:
        return moss_vae_encode(vae, images)

    def shift_scale_latents(self, args, latents: torch.FloatTensor) -> torch.FloatTensor:
        return latents  # already BN-normalised in moss_vae_encode

    def get_noise_pred_and_target(self, args, accelerator, noise_scheduler, latents, batch, text_encoder_conds,
                                  text_encoder_masks, unet, network, weight_dtype, train_unet, fixed_timesteps=None,
                                  is_train=True):
        stream_dtype = torch.float32 if getattr(args, "moss_stream_fp32", False) else weight_dtype
        noise = torch.randn_like(latents, dtype=torch.float32)
        if getattr(args, "flow_use_ot", False) and latents.size(0) > 1:
            noise = self._ot_reorder_noise(latents, noise)  # before antithetic pairing, as Anima / train_util do
        noise = maybe_apply_antithetic_noise_pairing(args, noise, is_train=is_train)

        def draw():
            return flow.get_noisy_model_input_and_timesteps(
                args, noise_scheduler, latents, noise, accelerator.device, stream_dtype, self._sigma_sampler,
                fixed_timesteps=fixed_timesteps, is_train=is_train)

        tsm_plan = getattr(self, "_tsm_offload", None)
        if is_train and fixed_timesteps is None and tsm_plan is not None and tsm_plan.grouped:
            # TSM-LoKr tsm_grouped_accumulation: every sample of this optimizer step in one core interval (redraws)
            noisy, timesteps, sigmas = tsm_plan.grouped_draw(draw)
        else:
            noisy, timesteps, sigmas = draw()
        if is_train:
            if getattr(self, "wavelet_masking_enabled", False):
                self._noisy_latents = noisy.detach()
            if getattr(self, "hf_scale", 0.0) > 0.0:
                self._hf_noisy_latents = noisy.detach()
            if getattr(self, "anchor_scale", 0.0) > 0.0:
                self._anchor_noisy_latents = noisy.detach()
            if getattr(self, "adv_scale", 0.0) > 0.0:
                self._adv_noisy_latents = noisy.detach()  # the stationary RFF-MMD loss (train_network._apply_adv_term)
        self.apply_tlora_mask(timesteps)
        if args.gradient_checkpointing:
            noisy.requires_grad_(True)  # unsloth offload only returns grad-carrying outputs for grad-carrying inputs

        text = text_encoder_conds[0].to(accelerator.device, dtype=stream_dtype)
        text_mask = text_encoder_conds[1].to(accelerator.device).bool()
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred = unet(noisy, timesteps, text, text_mask)
        if not (is_train and args.gradient_checkpointing):
            # With checkpointing the backward pass re-runs the blocks: the T-LoRA mask must survive until then. The
            # next step's apply_tlora_mask replaces it; sample_images clears it explicitly.
            self.clear_tlora_mask_if_needed()

        target = noise.to(torch.float64) - latents.to(torch.float64)  # velocity: eps - x0
        weighting = anima_train_utils.compute_loss_weighting_for_anima(args.weighting_scheme, sigmas)
        return model_pred, target, sigmas.flatten(), weighting, noise.to(torch.float64)

    def _ot_reorder_noise(self, latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """--flow_use_ot (anima_train_network.get_noise_pred_and_target): the batch's noise rows reassigned to the
        latents by cosine optimal transport. The cost is taken in fp32 (the noise is fp32; cached latents may be bf16)."""
        with torch.no_grad():
            b = latents.size(0)
            _, (_, col_indices) = train_util.cosine_optimal_transport(latents.reshape(b, -1).float(), noise.reshape(b, -1))
            noise = noise[col_indices.squeeze(0)]
        if not self._ot_logged:
            logger.info(f"[MOSS OT] First batch: noise reordered by cosine OT. New noise assignment indices:"
                        f" {col_indices.squeeze(0).tolist()}")
            self._ot_logged = True
        return noise

    def process_batch(self, batch, text_encoders, unet, network, vae, noise_scheduler, vae_dtype, weight_dtype,
                      accelerator, args, text_encoding_strategy, tokenize_strategy, is_train=True,
                      train_text_encoder=True, train_unet=True, edm2_model=None):
        outputs = batch.get("text_encoder_outputs_list", None)
        if outputs is not None:
            rates = outputs[-1]
            kept = text_encoding_strategy.drop_cached_text_encoder_outputs(*outputs[:-1], caption_dropout_rates=rates)
            batch["text_encoder_outputs_list"] = kept + [rates]
        return super().process_batch(batch, text_encoders, unet, network, vae, noise_scheduler, vae_dtype, weight_dtype,
                                     accelerator, args, text_encoding_strategy, tokenize_strategy, is_train,
                                     train_text_encoder, train_unet, edm2_model)

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        if args.min_snr_gamma:
            loss = apply_snr_weight_for_flow_matching(loss, timesteps, args.min_snr_gamma, soft=args.min_snr_gamma_soft)
        return loss

    def prepare_unet_with_accelerator(self, args, accelerator: Accelerator, unet: torch.nn.Module) -> torch.nn.Module:
        if getattr(args, "moss_stream_fp32", False):
            unet.enable_stream_fp32()  # here: train_network casts the whole model to weight_dtype before this call
        # (--moss_stream_scale is set when the DiT is loaded: load_unet_lazily)
        if self._use_unsloth_offload_checkpointing and args.gradient_checkpointing:
            unet.enable_gradient_checkpointing(unsloth_offload=True)
        if not self.is_swapping_blocks:
            return super().prepare_unet_with_accelerator(args, accelerator, unet)
        model = accelerator.prepare(unet, device_placement=[False])
        accelerator.unwrap_model(model).move_to_device_except_swap_blocks(accelerator.device)
        accelerator.unwrap_model(model).prepare_block_swap_before_forward()
        return model

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()

    # ---- --moss_tier_timing (library/tier_timing.py): one span per micro-step, keyed by its image tokens per sample ----
    # MOSS latents are already 2x2-packed (one latent pixel = one DiT token: patch 1, pixels / 16); the text tokens of
    # the joint sequence are reported next to each tier (the padded text length the forward ran with,
    # MossTransformer.last_text_tokens - a host int, no device read). Off: one attribute check per micro-step.
    def on_micro_step_start(self, args, accelerator, sub_batch, unet):
        if not getattr(args, "moss_tier_timing", False):
            return
        if self._tier_timer is None:
            self._tier_timer = tier_timing.TierTimer(getattr(accelerator.device, "type", "cpu"))
        self._tier_dit = accelerator.unwrap_model(unet)
        self._tier_timer.start(tier_timing.sample_tokens(sub_batch, patch=1,
                                                         vae_downscale=strategy.MossLatentsCachingStrategy.STRIDE))

    def on_micro_step_end(self, args, accelerator):
        if self._tier_timer is not None:
            self._tier_timer.stop(text_tokens=getattr(self._tier_dit, "last_text_tokens", None))

    def on_optimizer_step_end(self, args, accelerator, global_step: int, is_tracking: bool):
        if ckpt_nograd_merged_enabled():
            # --moss_ckpt_nograd_merged: the adapter just changed (optimizer.step, or step_func which has no step hook,
            # then weight noise / max-norm scaling): the cached merged weights are rebuilt at their next use
            ckpt_merged_invalidate()
        if ckpt_node_enabled():  # also reuse-only (--moss_ckpt_reuse_attn alone): the stored attention is VRAM too
            self._watch_ckpt_vram()
        timer = self._tier_timer
        if timer is None or global_step % max(1, int(getattr(args, "moss_tier_timing_every", 50) or 50)):
            return
        stats = timer.report()  # the finished spans of the window (CUDA: event pairs already completed, no sync)
        if is_tracking:
            accelerator.log(timer.scalars(stats), step=global_step)
        else:
            logger.info(timer.format_line(stats, what=f"step {global_step}"))

    def on_epoch_end(self, args, accelerator, epoch: int, global_step: int):
        if not self._ckpt_merged_logged and ckpt_nograd_merged_enabled():
            self._ckpt_merged_logged = True  # once, after an epoch has touched every tier: what the cache holds
            held = ckpt_merged_bytes(accelerator.unwrap_model(self._ckpt_merged_unet)) if self._ckpt_merged_unet else 0
            # read before this epoch's sample round, which releases the cache (0.00 GiB: a padded-only / no-autocast run)
            logger.info(f"--moss_ckpt_nograd_merged: {held / 2**30:.2f} GiB of cached merged LoKr weights")
        if ckpt_node_enabled():
            self._watch_ckpt_vram(report=True)
        if self._tier_timer is not None:
            logger.info(self._tier_timer.format_line(self._tier_timer.epoch_report(), what=f"epoch {epoch + 1}"))

    # ---- samples / metadata -----------------------------------------------------------------------------------
    def _lycoris_fix_sampler_timestep_scale(self):
        # LyCORIS fixed paths (TSM / TSM-LoKr / T-LoRA): the MOSS sampler calls dit(x, 1000 * sigma, ...), already the
        # [0, max_timestep] scale the training masks and expert choices use (train_network._lycoris_fix_wrap_sampling)
        return 1.0

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        self.clear_tlora_mask_if_needed()  # a training mask may still be set (kept for checkpoint recomputation)
        try:
            with moss_block_compile.pause():  # --moss_block_compile: sampling runs the eager blocks (no extra graphs)
                moss_train_utils.sample_images(accelerator, args, epoch, global_step, unet, vae,
                                               self.sample_prompts_te_outputs)
        finally:
            self.after_sample_images()

    def _watch_ckpt_vram(self, report: bool = False) -> None:
        """--moss_ckpt_nograd_merged / --moss_ckpt_reuse_attn on a CUDA device: the options add VRAM (merged-weight cache,
        stored joint attention), and a Windows driver with sysmem fallback answers a full card with slow steps, not with
        an error. After every optimizer step: one warning when the peak reserved memory, or the device with what other
        processes hold on it, reaches the card's limit (allocator counters and one driver query of the device's free
        memory per optimizer step, no sync: the device, not this process alone, is what spills). report (epoch end):
        the measured peak line, repeated only when the peak grew."""
        if self._ckpt_vram_device is None or (self._ckpt_vram_warned and not report):
            return
        stats = moss_train_utils.cuda_memory_stats(self._ckpt_vram_device, with_free=True)
        if stats is None:
            return
        line, at_limit = moss_train_utils.ckpt_reuse_peak_note(stats, options=moss_train_utils.ckpt_reuse_options())
        if at_limit and not self._ckpt_vram_warned:
            self._ckpt_vram_warned = True
            self._ckpt_vram_reported = stats["peak_reserved"]
            logger.warning(line)
        elif report and stats["peak_reserved"] >= self._ckpt_vram_reported + 2**28:
            self._ckpt_vram_reported = stats["peak_reserved"]
            logger.info(line)

    def _log_ckpt_vram_estimate(self, args, accelerator, unet, network=None) -> None:
        """--moss_ckpt_nograd_merged / --moss_ckpt_reuse_attn, once the network is applied: what the options add at the
        largest bucket of the dataset next to what the card has (the device: what other processes hold counts, and so
        do the adapter's parameters and gradients, which reach the GPU later), and under
        --moss_ckpt_reuse_attn_budget_gib how many blocks reuse in every resolution tier
        (library/moss/train_utils.ckpt_reuse_vram_estimate); a warning when tight."""
        batch, tokens, bucket = self._dataset_limits
        estimate = moss_train_utils.ckpt_reuse_vram_estimate(
            accelerator.unwrap_model(unet), bool(getattr(args, "moss_ckpt_reuse_attn", False)), tokens, batch_size=batch,
            cached=ckpt_nograd_merged_enabled() and getattr(args, "mixed_precision", "bf16") in ("bf16", "fp16"),
            stream_bytes=4 if getattr(args, "moss_stream_fp32", False) else 2,
            budget_bytes=ckpt_reuse_budget_bytes(), tiers=self._dataset_tiers,
            adapter_bytes=moss_train_utils.adapter_vram_bytes(network))
        stats = moss_train_utils.cuda_memory_stats(getattr(accelerator, "device", None))
        self._ckpt_vram_device = accelerator.device if stats else None
        line, tight = moss_train_utils.ckpt_reuse_vram_note(estimate, bucket, stats)
        (logger.warning if tight else logger.info)(line)

    def after_sample_images(self) -> None:
        """The end of every sample round. Sampling runs under no_grad and never reads the checkpoint merged-weight cache
        or a stored attention output (MossBlock._run calls _forward directly); it has its own cache
        (lycoris_fast.cached_lokr_weights), which parks the base weights in RAM and brings them back as new tensors.
        A round that really ran released the checkpoint cache first (moss_train_utils.sample_images)."""
        if ckpt_nograd_merged_enabled():
            ckpt_merged_invalidate()  # --moss_ckpt_nograd_merged: rebuild at the next training step

    def after_network_loaded(self, args, accelerator, network, text_encoders, unet):
        """--moss_block_compile: torch.compile every joint block's _forward with the (applied, loaded) network inside the
        graphs (library/moss/block_compile.py)."""
        if ckpt_node_enabled():  # --moss_ckpt_nograd_merged and / or --moss_ckpt_reuse_attn
            self._ckpt_merged_unet = unet  # --moss_ckpt_nograd_merged: for the cache size line at the first epoch end
            note = moss_train_utils.ckpt_reuse_outside_grad_note(network)  # the network is applied by now
            if note:
                logger.info(note)
            self._log_ckpt_vram_estimate(args, accelerator, unet, network)
        if not getattr(args, "moss_block_compile", False):
            return
        model = accelerator.unwrap_model(unet)
        fixes_on = getattr(self, "_lycoris_fix_on", False)
        # the per-module caches the fixed paths fill on the first forward (a .item() each for LoKr) must be warm before
        # the first trace: traced cold, the first call breaks the graph and dynamo keeps those fragments
        primed = self._lycoris_fixes.prime_module_caches(network) if fixes_on else 0
        if fixes_on:
            # the LoHa / ABBA path rule reads the row count: inside a graph a fixed choice at the largest joint length
            rows = int(getattr(args, "max_bucket_reso", 2048) or 2048) ** 2 // (strategy.MossLatentsCachingStrategy.STRIDE ** 2)
            # every adapted Linear of a joint block has the joint token axis as rows (plan_block_compile_rows reads .blocks)
            self._lycoris_fixes.plan_block_compile_rows(network, SimpleNamespace(blocks=model.layers),
                                                        rows + int(args.moss_max_token_length))
        if getattr(model, "split_attn", False):
            logger.warning("--moss_block_compile with --split_attn: the per-row slice by the seqlens tensor is a graph break"
                           " inside every attention call (the blocks still run, in pieces)")
        if moss_train_utils.largest_batch_size(args, self._dataset_limits) > 1:  # also a dataset TOML's batch_size
            logger.warning("--moss_block_compile with batch > 1: a batch whose captions differ in length is padded, and the"
                           " padded attention specialises a graph per text length (batch 1 / equal lengths share one)")
        state = moss_block_compile.compile_moss_blocks(
            model,
            backend=args.moss_block_compile_backend,
            mode=args.moss_block_compile_mode,
            recompile_limit=int(args.moss_block_compile_recompile_limit),
            stream_fp32=bool(getattr(args, "moss_stream_fp32", False)),
            stream_scale=float(getattr(args, "moss_stream_scale", 1.0)),
        )
        accelerator.print(
            f"MOSS block compile: {len(model.layers)} joint blocks compiled per call (backend"
            f" {args.moss_block_compile_backend}, mode {args.moss_block_compile_mode or 'default'}, {primed} LyCORIS module"
            " caches primed); the first steps of every checkpointing regime compile (minutes on Windows), later runs"
            " reuse the inductor cache. Sample images and validation run eager.")
        return state

    def process_val_batch(self, batch, text_encoders, unet, network, vae, noise_scheduler, vae_dtype, weight_dtype,
                          accelerator, args, *rest, **kwargs):
        # Validation runs several no-grad forwards in a row; with block swap they need the forward-only schedule
        # (train_network never calls on_validation_step_end).
        model = accelerator.unwrap_model(unet) if self.is_swapping_blocks else None
        if model is not None:
            model.switch_block_swap_for_inference()
        try:
            return super().process_val_batch(batch, text_encoders, unet, network, vae, noise_scheduler, vae_dtype,
                                             weight_dtype, accelerator, args, *rest, **kwargs)
        finally:
            if model is not None:
                model.switch_block_swap_for_training()

    def get_sai_model_spec(self, args):
        spec = train_util.get_sai_model_spec_dataclass(None, args, False, True, False).to_metadata_dict()
        spec["modelspec.architecture"] = "moss-image-base/lora"  # the string upstream sd-scripts writes
        spec["modelspec.implementation"] = "https://github.com/OpenMOSS/MOSS-Image"
        if not getattr(args, "resolution", None):
            spec["modelspec.resolution"] = "1024x1024"  # the dataset config sets the resolution; the 512 fallback is SD1
        return spec

    def update_metadata(self, metadata, args):
        metadata.update(moss_train_utils.moss_metadata(args))


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    moss_train_utils.add_moss_training_arguments(parser)
    parser.add_argument("--unsloth_offload_checkpointing", action="store_true",
                        help="offload activations to CPU RAM with async transfers (faster than --cpu_offload_checkpointing);"
                             " not with --cpu_offload_checkpointing or --blocks_to_swap")
    parser.set_defaults(cfm_lambda=0.02)
    return parser


def prepare_moss_runtime(args) -> None:
    """Process-wide setup shared by the trainer entry point and measurement harnesses."""
    if args.network_module == "networks.lora":
        print("Override network module: networks.lora -> networks.lora_moss")
        args.network_module = "networks.lora_moss"
    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"
    if args.attn_mode == "flash" and not attention.is_flash_attn_supported():
        raise ValueError("MOSS-Image: attn_mode = \"flash\" needs the flash-attn package and an Ampere or newer GPU;"
                         " install flash-attn or set attn_mode = \"torch\" (about 1.6x slower)")
    timestep_presets.resolve(args)  # sigmoid_scale presets; scale_resume extends max_train_epochs
    moss_train_utils.apply_moss_block_exclusion(args)  # --moss_lora_exclude_blocks -> network_args exclude_patterns
    register_moss_lycoris_presets()
    if not args.moss_stock_lycoris:
        install_fast_lokr_forward()  # before the network is created: apply_to binds LokrModule.forward
    block_compile = bool(getattr(args, "moss_block_compile", False))
    if block_compile:
        from library import lycoris_algo_fixes

        # re-bind the adapted forwards after apply_to (traceable closures) - library/moss/block_compile.py
        lycoris_algo_fixes.set_block_compile(True)
    # Under block compile the compiled block graphs contain the elementwise chains and the LoKr rebuild: their own
    # torch.compile would nest inside the block graphs (recompile-budget storms), so both stay eager there.
    enable_elementwise_compile(not args.moss_no_compile and not block_compile)
    # The plain joint-block checkpoint with a no-grad first pass + cached merged LoKr weights / the stored joint-attention
    # output (library/moss/models.py; the switches are process-wide, both off by default)
    # (--moss_ckpt_reuse_attn alone = reuse-only: the same node, nothing cached; --moss_ckpt_reuse_attn_budget_gib caps
    # the stored attention by letting only the first n joint blocks of a micro-step reuse)
    set_ckpt_reuse(getattr(args, "moss_ckpt_nograd_merged", False), getattr(args, "moss_ckpt_reuse_attn", False),
                   getattr(args, "moss_ckpt_reuse_attn_budget_gib", None))
    if ckpt_node_enabled():
        reuse = bool(getattr(args, "moss_ckpt_reuse_attn", False))
        budget = ckpt_reuse_budget_bytes()
        if ckpt_nograd_merged_enabled():
            ckpt_merged_invalidate_on_optimizer_steps()  # every optimizer.step(): CAME writes through p.data
            line = "MOSS: checkpointed joint blocks run a no-grad first pass with merged LoKr weights cached per optimizer step"
        else:
            line = ("MOSS: checkpointed joint blocks run a no-grad first pass; nothing is cached (the LoKr+DoRA weights are"
                    " rebuilt in both passes, as with the plain checkpoint)")
        if reuse:
            line += ", the recompute reuses the stored joint-attention output"
            if budget is not None:
                line += (f" in the first blocks of each micro-step that fit {budget / 2**30:.2f} GiB of stored attention"
                         " (--moss_ckpt_reuse_attn_budget_gib; the other blocks "
                         + ("run without a stored output" if ckpt_nograd_merged_enabled() else "keep the plain checkpoint")
                         + ")")
        logger.info(line)
    # The fast forward's LoKr+DoRA weight rebuild, fused by torch.compile (~1 bf16 ulp)
    enable_compiled_rebuild(not (args.moss_stock_lycoris or args.moss_no_compile or args.moss_no_lokr_compile
                                 or block_compile))
    # Rectified-flow target (noise - latents) for contrastive flow matching in train_network (historical flag name).
    args._anima_model = True


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    if args.show_timesteps:
        moss_train_utils.show_timesteps(args)  # CPU only: no model, no GPU checks, then exit
    else:
        prepare_moss_runtime(args)
        MossNetworkTrainer().train(args)
