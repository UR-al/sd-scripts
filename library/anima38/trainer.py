# AnimaNetworkTrainer overrides for Anima-3.8B v1.1 (--anima38_connector).
#
# anima_train_network.py builds ``make_anima38_trainer(AnimaNetworkTrainer)`` only when the option is on, so the
# plain Anima trainer is untouched otherwise. The mixin:
#   - validates options and the bundle before any caching work,
#   - tokenizes/encodes/caches every caption (and every caption variant) with Qwen3-0.6B AND Qwen3.5-4B,
#     Qwen3.5 being loaded on first use during caching and released afterwards,
#   - loads the DiT without the bundled connector tensors, builds the frozen connector around the DiT's own
#     llm_adapter in the training dtype on the training device and attaches it (see training.py),
#   - passes each batch's Qwen3.5 features to the DiT call, and samples with the connector.

from __future__ import annotations

import logging

from library.device_utils import clean_memory_on_device

from library.strategy_base import TextEncoderOutputsCachingStrategy

from . import captions, sampling, strategy, training
from .bundle import build_semantic_connector, load_anima38_connector_state

logger = logging.getLogger(__name__)


class Anima38TrainerMixin:
    """Mixed in before AnimaNetworkTrainer; every override calls the Anima implementation where it applies."""

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)
        training.validate_anima38_args(args, [train_dataset_group, val_dataset_group])
        training.check_qwen35_checkpoint(args.qwen35_path)
        info = training.require_anima38_bundle(args.pretrained_model_name_or_path)
        logger.info(
            f"Anima-3.8B v1.1 bundle: {len(info['connector_keys'])} Semantic Connector v2 tensors; Qwen3.5: {args.qwen35_path};"
            f" sample negative: {args.anima38_sample_negative}"
        )

    # region text encoding

    def get_tokenize_strategy(self, args):
        return strategy.Anima38TokenizeStrategy(
            qwen35_tokenizer_dir=args.qwen35_tokenizer_dir,
            qwen3_path=args.qwen3,
            t5_tokenizer_path=args.t5_tokenizer_path,
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
        )

    def get_text_encoding_strategy(self, args):
        return strategy.Anima38TextEncodingStrategy(qwen35_batch_size=args.anima38_qwen35_batch_size)

    def get_text_encoder_outputs_caching_strategy(self, args):
        if not args.cache_text_encoder_outputs:
            return None  # refused in assert_extra_args
        return strategy.Anima38TextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            args.skip_cache_check,
            False,
            cache_dtype=getattr(args, "cache_text_encoder_outputs_dtype", "auto"),
            qwen35_signature=strategy.qwen35_cache_signature(args.qwen35_path),
        )

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        # The Anima implementation moves text_encoders[0] (Qwen3) around and passes the list to the caching and
        # sample-prompt encoding; the provider loads Qwen3.5 there only if something actually needs encoding.
        patched = captions.install_canonical_captions(dataset)
        caching = TextEncoderOutputsCachingStrategy.get_strategy()
        if patched and isinstance(caching, strategy.Anima38TextEncoderOutputsCachingStrategy):
            captions.register_caption_digests(dataset, caching)
        provider = strategy.Qwen35Provider(args.qwen35_path, accelerator.device)
        try:
            super().cache_text_encoder_outputs_if_needed(
                args, accelerator, unet, vae, [text_encoders[0], provider], dataset, weight_dtype
            )
        finally:
            captions.uninstall_canonical_captions(dataset)  # workers pickle the dataset; caching is over
            loaded = provider.loaded
            provider.release()
            if loaded:
                logger.info("released Qwen3.5-4B after caching")
                clean_memory_on_device(accelerator.device)

    # endregion

    # region DiT

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders):
        # the Anima implementation calls load_anima_model(..., anima38_connector=True), which skips the connector tensors
        model, text_encoders = super().load_unet_lazily(args, weight_dtype, accelerator, text_encoders)
        connector_state, info = load_anima38_connector_state(args.pretrained_model_name_or_path)
        connector = build_semantic_connector(
            model.llm_adapter,
            connector_state,
            config=info["config"],
            layer_indices=info["layer_indices"],
            device=accelerator.device,
            dtype=weight_dtype,
        )
        del connector_state
        training.attach_anima38_conditioner(model, connector)
        parameters = sum(p.numel() for p in connector.parameters())
        logger.info(
            f"Semantic Connector v2 attached (frozen, {parameters:,} parameters, {weight_dtype}, {accelerator.device},"
            f" layers {info['layer_indices']}, config {info['config']})"
        )
        return model, text_encoders

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        super().post_process_network(args, accelerator, network, text_encoders, unet)
        conditioner = training.get_anima38_conditioner(unet)
        if conditioner is None:
            raise RuntimeError("--anima38_connector: the DiT has no connector attached")
        conditioner.grad_through_connector = training.network_trains_llm_adapter(network)
        # LyCORIS registers its modules only in apply_to(), after this hook: look again at the first DiT call
        self._anima38_network = network
        self._anima38_conditioner = conditioner

    def _anima38_update_grad_policy(self):
        network = getattr(self, "_anima38_network", None)
        if network is None:
            return
        self._anima38_network = None
        conditioner = self._anima38_conditioner
        conditioner.grad_through_connector = training.network_trains_llm_adapter(network)
        if conditioner.grad_through_connector:
            logger.info("llm_adapter modules are trained: gradients flow through the (frozen) connector's native blocks")

    def get_extra_dit_kwargs(self, args, accelerator, text_encoder_conds, weight_dtype):
        self._anima38_update_grad_policy()
        return training.semantic_dit_kwargs(text_encoder_conds, accelerator.device, weight_dtype)

    # endregion

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        self._anima38_update_grad_policy()  # sample_at_first runs before the first training step
        sampling.sample_images(
            accelerator,
            args,
            epoch,
            global_step,
            unet,
            vae,
            self.sample_prompts_te_outputs,
            negative_mode=args.anima38_sample_negative,
        )

    def update_metadata(self, metadata, args):
        super().update_metadata(metadata, args)
        metadata.update(training.anima38_metadata(args))


def make_anima38_trainer(base_cls):
    """Instance of ``Anima38NetworkTrainer(Anima38TrainerMixin, base_cls)``."""
    trainer_cls = type("Anima38NetworkTrainer", (Anima38TrainerMixin, base_cls), {})
    return trainer_cls()
