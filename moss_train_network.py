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
from typing import Optional, Union

import torch
from accelerate import Accelerator

from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

import train_network
from library import anima_train_utils, attention, sd3_train_utils, strategy_base, timestep_presets, train_util
from library.custom_train_functions import apply_snr_weight_for_flow_matching, maybe_apply_antithetic_noise_pairing
from library.moss import captions, flow, strategy
from library.moss import train_utils as moss_train_utils
from library.lycoris_fast import install_fast_lokr_forward
from library.moss.lycoris_presets import register_moss_lycoris_presets
from library.moss.models import MOSS_DIT_CONFIG, MossTransformer, enable_elementwise_compile
from library.moss.vae import load_moss_vae, moss_vae_encode
from library.ramtorch_util import apply_ramtorch_to_module
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def load_moss_dit(path: str, attn_mode: str, split_attn: bool, device, dtype) -> MossTransformer:
    """Strict load of the single-file DiT (bare keys, as in moss_image_base_bf16.safetensors)."""
    from safetensors.torch import load_file

    with torch.device("meta"):
        model = MossTransformer(**MOSS_DIT_CONFIG, attn_mode=attn_mode, split_attn=split_attn)
    state = load_file(path, device="cpu")
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer.", "model."):
        if all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}
            break
    state = {k: (v if k == "rope.bands" else v.to(dtype)) for k, v in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    model.rope.bands = model.rope._default_bands(torch.device("cpu"))
    model.to(device)
    logger.info(f"MOSS DiT loaded: {path} ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, {dtype},"
                f" {device}, attn {attn_mode}{', split' if split_attn else ''})")
    return model


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

    def get_adaptive_model_type(self, args) -> str:
        return "flow_matching"

    # ---- checks / models ------------------------------------------------------------------------------------
    def assert_extra_args(self, args, train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
                          val_dataset_group: Optional[train_util.DatasetGroup]):
        moss_train_utils.validate_moss_args(args, train_dataset_group, val_dataset_group)
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
        self._use_unsloth_offload_checkpointing = args.unsloth_offload_checkpointing
        if args.use_ramtorch:
            # Only the joint blocks' Linears (8.9 GB, the LoRA targets) stay in pinned CPU RAM: LyCORIS streams them into its
            # rebuild, so training is unchanged. The small layers stay on the GPU: RamTorch's own backward would compute
            # (and stochastically round, using the RNG) grads for their frozen weights.
            model.layers = apply_ramtorch_to_module(model.layers, "MOSS joint blocks", accelerator.device, model.dtype)
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
        noise = torch.randn_like(latents, dtype=torch.float32)
        noise = maybe_apply_antithetic_noise_pairing(args, noise, is_train=is_train)
        noisy, timesteps, sigmas = flow.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype, self._sigma_sampler,
            fixed_timesteps=fixed_timesteps, is_train=is_train)
        if is_train:
            if getattr(self, "wavelet_masking_enabled", False):
                self._noisy_latents = noisy.detach()
            if getattr(self, "hf_scale", 0.0) > 0.0:
                self._hf_noisy_latents = noisy.detach()
            if getattr(self, "anchor_scale", 0.0) > 0.0:
                self._anchor_noisy_latents = noisy.detach()
        self.apply_tlora_mask(timesteps)
        if args.gradient_checkpointing:
            noisy.requires_grad_(True)  # unsloth offload only returns grad-carrying outputs for grad-carrying inputs

        text = text_encoder_conds[0].to(accelerator.device, dtype=weight_dtype)
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

    # ---- samples / metadata -----------------------------------------------------------------------------------
    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        self.clear_tlora_mask_if_needed()  # a training mask may still be set (kept for checkpoint recomputation)
        moss_train_utils.sample_images(accelerator, args, epoch, global_step, unet, vae, self.sample_prompts_te_outputs)

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
        spec["modelspec.architecture"] = "moss-image/lora"
        spec["modelspec.implementation"] = "https://github.com/OpenMOSS/MOSS-Image"
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
    enable_elementwise_compile(not args.moss_no_compile)
    # Rectified-flow target (noise - latents) for contrastive flow matching in train_network (historical flag name).
    args._anima_model = True


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    prepare_moss_runtime(args)
    MossNetworkTrainer().train(args)
