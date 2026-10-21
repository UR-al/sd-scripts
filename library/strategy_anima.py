# Anima Strategy Classes

import hashlib
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from library import anima_utils, train_util
from library.cache_utils import CACHE_DTYPE_METADATA_KEY, save_npz
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy, caption_for_cache, variant_key
from library import qwen_image_autoencoder_kl

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class AnimaTokenizeStrategy(TokenizeStrategy):
    """Tokenize strategy for Anima: dual tokenization with Qwen3 + T5.

    Qwen3 tokens are used for the text encoder.
    T5 tokens are used as target input IDs for the LLM Adapter (NOT encoded by T5).

    Can be initialized with either pre-loaded tokenizer objects or paths to load from.
    """

    def __init__(
        self,
        qwen3_tokenizer=None,
        t5_tokenizer=None,
        qwen3_max_length: int = 512,
        t5_max_length: int = 512,
        qwen3_path: Optional[str] = None,
        t5_tokenizer_path: Optional[str] = None,
    ) -> None:
        # Load tokenizers from paths if not provided directly
        if qwen3_tokenizer is None:
            if qwen3_path is None:
                raise ValueError("Either qwen3_tokenizer or qwen3_path must be provided")
            qwen3_tokenizer = anima_utils.load_qwen3_tokenizer(qwen3_path)
        if t5_tokenizer is None:
            t5_tokenizer = anima_utils.load_t5_tokenizer(t5_tokenizer_path)

        self.qwen3_tokenizer = qwen3_tokenizer
        self.qwen3_max_length = qwen3_max_length
        self.t5_tokenizer = t5_tokenizer
        self.t5_max_length = t5_max_length

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text

        # Tokenize with Qwen3
        qwen3_encoding = self.qwen3_tokenizer(
            text, return_tensors="pt", truncation=True, padding="max_length", max_length=self.qwen3_max_length
        )
        qwen3_input_ids = qwen3_encoding["input_ids"]
        qwen3_attn_mask = qwen3_encoding["attention_mask"]

        # Tokenize with T5 (for LLM Adapter target tokens)
        t5_encoding = self.t5_tokenizer(
            text, return_tensors="pt", truncation=True, padding="max_length", max_length=self.t5_max_length
        )
        t5_input_ids = t5_encoding["input_ids"]
        t5_attn_mask = t5_encoding["attention_mask"]
        return [qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask]


class AnimaTextEncodingStrategy(TextEncodingStrategy):
    """Text encoding strategy for Anima.

    Encodes Qwen3 tokens through the Qwen3 text encoder to get hidden states.
    T5 tokens are passed through unchanged (only used by LLM Adapter).
    """

    def __init__(self) -> None:
        super().__init__()

    def encode_tokens(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Encode Qwen3 tokens and return embeddings + T5 token IDs.

        Args:
            models: [qwen3_text_encoder]
            tokens: [qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask]

        Returns:
            [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]
        """
        # Do not handle dropout here; handled dataset-side or in drop_cached_text_encoder_outputs()

        qwen3_text_encoder = models[0]
        qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask = tokens

        encoder_device = qwen3_text_encoder.device

        qwen3_input_ids = qwen3_input_ids.to(encoder_device)
        qwen3_attn_mask = qwen3_attn_mask.to(encoder_device)
        outputs = qwen3_text_encoder(input_ids=qwen3_input_ids, attention_mask=qwen3_attn_mask)
        prompt_embeds = outputs.last_hidden_state
        prompt_embeds[~qwen3_attn_mask.bool()] = 0

        return [prompt_embeds, qwen3_attn_mask, t5_input_ids, t5_attn_mask]

    def drop_cached_text_encoder_outputs(
        self,
        prompt_embeds: torch.Tensor,
        attn_mask: torch.Tensor,
        t5_input_ids: torch.Tensor,
        t5_attn_mask: torch.Tensor,
        caption_dropout_rates: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Apply dropout to cached text encoder outputs.

        Called during training when using cached outputs.
        Replaces dropped items with pre-cached unconditional embeddings (from encoding "")
        to match diffusion-pipe-main behavior.
        """
        if caption_dropout_rates is None or torch.all(caption_dropout_rates == 0.0).item():
            return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]

        # Clone to avoid in-place modification of cached tensors
        prompt_embeds = prompt_embeds.clone()
        if attn_mask is not None:
            attn_mask = attn_mask.clone()
        if t5_input_ids is not None:
            t5_input_ids = t5_input_ids.clone()
        if t5_attn_mask is not None:
            t5_attn_mask = t5_attn_mask.clone()

        for i in range(prompt_embeds.shape[0]):
            if random.random() < caption_dropout_rates[i].item():
                # Use pre-cached unconditional embeddings
                prompt_embeds[i] = 0
                if attn_mask is not None:
                    attn_mask[i] = 0
                if t5_input_ids is not None:
                    t5_input_ids[i, 0] = 1  # Set to </s> token ID
                    t5_input_ids[i, 1:] = 0
                if t5_attn_mask is not None:
                    t5_attn_mask[i, 0] = 1
                    t5_attn_mask[i, 1:] = 0

        return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]


# SHA-1 of the raw caption and the processed caption a *_anima_te.npz encodes (cache_caption_digest). The upstream
# validity check compares only the caption augmentation settings, never the caption text, so without it an edited
# caption file silently kept its old encoding. Files written before the digest existed have no such key and are
# rebuilt once (Qwen3-0.6B is fast).
CAPTION_DIGEST_KEY = "anima_caption_sha1"

TE_OUTPUT_KEYS = ("prompt_embeds", "attn_mask", "t5_input_ids", "t5_attn_mask")


def cache_caption_digest(info) -> str:
    """Digest of the raw caption + the processed caption the cache encodes as variant 0 (caption_for_cache), so editing
    the caption file or a deterministic caption setting (prefix/suffix, separators, shuffle on/off, replacements)
    changes it. Variants 1..n follow from the raw caption and the settings in caption_aug_hash. Without variants the
    file also stores the caption dropout rate the trainer applies (cache_batch_outputs; with variants it is 0 and
    caption_aug_hash covers it), so that rate is part of the digest too."""
    variants = getattr(info, "caption_variants", None)
    rate = 0.0 if variants and len(variants) > 1 else float(getattr(info, "caption_dropout_rate", 0.0) or 0.0)
    text = (info.caption or "") + "\x00" + (caption_for_cache(info) or "") + "\x00" + repr(rate)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _npz_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _read_cached(npz, key: str, stored_dtype: Optional[str], bf16_as_tensor: bool = False):
    """One npz entry as load_npz returns it (BF16 bits widened to fp32 numpy); with bf16_as_tensor the stored BF16 bits
    are viewed straight as a torch.bfloat16 tensor instead (same values, no fp32 round trip)."""
    array = npz[key]
    if stored_dtype != "bf16":
        return array
    bf16 = torch.from_numpy(np.ascontiguousarray(array.astype(np.uint16, copy=False))).view(torch.bfloat16)
    return bf16 if bf16_as_tensor else bf16.float().numpy()


class AnimaTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    """Caching strategy for Anima text encoder outputs.

    Caches: prompt_embeds (float), attn_mask (int), t5_input_ids (int), t5_attn_mask (int)
    """

    ANIMA_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_anima_te.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        cache_dtype: str = "auto",
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial, cache_dtype=cache_dtype)
        # npz_key(npz path) -> cache_caption_digest of the image's current caption (register_cache_captions)
        self.caption_text_digests: Dict[str, str] = {}
        self._logged_missing_caption_digest = False

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + self.ANIMA_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def register_cache_captions(self, infos: List) -> None:
        """Called by BaseDataset.new_cache_text_encoder_outputs once info.cache_caption (and the caption variants) are
        set: the digest each image's cache file must carry. The Anima-3.8B subclass checks its own digests."""
        if not self.cache_to_disk:
            return
        for info in infos:
            self.caption_text_digests[_npz_key(self.get_outputs_npz_path(info.absolute_path))] = cache_caption_digest(info)

    def is_disk_cached_outputs_expected(self, npz_path: str, num_caption_variants: int = 0, caption_aug_hash: Optional[str] = None) -> bool:
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            with np.load(npz_path, allow_pickle=False) as npz:  # lazy: only the entries read below are decompressed
                for key in TE_OUTPUT_KEYS + ("caption_dropout_rate",):
                    if key not in npz:
                        return False
                expected_digest = self.caption_text_digests.get(_npz_key(npz_path))
                if expected_digest is not None:
                    stored_digest = str(npz[CAPTION_DIGEST_KEY].tolist()) if CAPTION_DIGEST_KEY in npz else None
                    if stored_digest != expected_digest:
                        if stored_digest is not None:
                            logger.info(f"caption changed since caching, rebuilding: {npz_path}")
                        elif not self._logged_missing_caption_digest:
                            self._logged_missing_caption_digest = True
                            logger.info(f"Anima TE caches written before caption digests are rebuilt once (first: {npz_path})")
                        return False
                if not self._check_cached_variant_keys(npz, set(TE_OUTPUT_KEYS), num_caption_variants, caption_aug_hash):
                    return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str, variant: int = 0) -> List[Any]:
        """Reads only the chosen variant's entries and caption_dropout_rate (load_npz would decompress every variant in
        the file and widen each BF16 one to fp32). BF16 prompt_embeds come back as a compact torch.bfloat16 tensor
        straight from the stored bits, so the batch stays bf16 in RAM/queue until the weight_dtype cast at the model
        boundary; everything else, and legacy fp32/fp16 caches, exactly as load_npz returns it."""
        with np.load(npz_path, allow_pickle=False) as npz:
            files = set(npz.files)
            dtypes = json.loads(str(npz[CACHE_DTYPE_METADATA_KEY].tolist())) if CACHE_DTYPE_METADATA_KEY in files else {}
            outputs = []
            for base_key in TE_OUTPUT_KEYS:
                key = variant_key(base_key, variant) if variant > 0 and variant_key(base_key, variant) in files else base_key
                outputs.append(_read_cached(npz, key, dtypes.get(key), bf16_as_tensor=base_key == "prompt_embeds"))
            caption_dropout_rate = _read_cached(npz, "caption_dropout_rate", dtypes.get("caption_dropout_rate"))
        return outputs + [caption_dropout_rate]

    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        infos: List,
    ):
        anima_text_encoding_strategy: AnimaTextEncodingStrategy = text_encoding_strategy

        canonical_captions, variant_groups = self._get_variant_caption_groups(infos)

        def encode_captions(captions: List[str]) -> dict:
            tokens_and_masks = tokenize_strategy.tokenize(captions)
            with torch.no_grad():
                prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = anima_text_encoding_strategy.encode_tokens(
                    tokenize_strategy, models, tokens_and_masks
                )

            # Preserve the TE's native precision (bf16/fp16) end-to-end instead of
            # widening to fp32. Widening doubled the disk/RAM/queue footprint and
            # required a re-cast to weight_dtype at the model boundary anyway.
            # bf16 has no numpy dtype, so keep it as a CPU tensor; save_npz
            # encodes it to its uint16 bit layout and load_outputs_npz()
            # reconstructs bf16. fp16/fp32 are stored as numpy arrays as before.
            prompt_embeds = prompt_embeds.detach().to("cpu")
            if prompt_embeds.dtype == torch.bfloat16:
                prompt_embeds_cached = prompt_embeds
            else:
                prompt_embeds_cached = prompt_embeds.numpy()
            return {
                "prompt_embeds": prompt_embeds_cached,
                "attn_mask": attn_mask.cpu().numpy(),
                "t5_input_ids": t5_input_ids.cpu().numpy().astype(np.int32),
                "t5_attn_mask": t5_attn_mask.cpu().numpy().astype(np.int32),
            }

        batched = encode_captions(canonical_captions)
        variant_batched = [(group, encode_captions([c for _, c in group])) for group in variant_groups]

        base_keys = ["prompt_embeds", "attn_mask", "t5_input_ids", "t5_attn_mask"]

        for i, info in enumerate(infos):
            n_variants = len(info.caption_variants) if getattr(info, "caption_variants", None) else 0

            # when caption variants are cached, dropout is carried inside the variants:
            # zero the stored rate so the train-time dropout hook becomes inert
            effective_dropout_rate = info.caption_dropout_rate if n_variants <= 1 else 0.0
            caption_dropout_rate = torch.tensor(effective_dropout_rate, dtype=torch.float32)

            if self.cache_to_disk:
                save_kwargs = {key: batched[key][i] for key in base_keys}
                save_kwargs["caption_dropout_rate"] = caption_dropout_rate
                save_kwargs[CAPTION_DIGEST_KEY] = np.array(cache_caption_digest(info))
                for k, (group, vb) in enumerate(variant_batched, start=1):
                    idxs = [idx for idx, _ in group]
                    if i not in idxs:
                        continue
                    j = idxs.index(i)
                    for key in base_keys:
                        save_kwargs[variant_key(key, k)] = vb[key][j]
                if n_variants > 1:
                    save_kwargs["caption_variants"] = np.array(n_variants)
                    save_kwargs["caption_aug_hash"] = np.array(getattr(info, "caption_aug_hash", None) or "")
                save_npz(info.text_encoder_outputs_npz, save_kwargs, cache_dtype=self.cache_dtype)
            else:
                info.text_encoder_outputs = tuple([batched[key][i] for key in base_keys] + [caption_dropout_rate])
                if n_variants > 1:
                    variants_outputs = []
                    for k, (group, vb) in enumerate(variant_batched, start=1):
                        idxs = [idx for idx, _ in group]
                        if i not in idxs:
                            continue
                        j = idxs.index(i)
                        variants_outputs.append(tuple([vb[key][j] for key in base_keys] + [caption_dropout_rate]))
                    info.text_encoder_outputs_variants = variants_outputs


class AnimaLatentsCachingStrategy(LatentsCachingStrategy):
    """Latent caching strategy for Anima using WanVAE.

    WanVAE produces 16-channel latents with spatial downscale 8x.
    Latent shape for images: (B, 16, 1, H/8, W/8)
    """

    ANIMA_LATENTS_NPZ_SUFFIX = "_anima.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool, cache_dtype: str = "auto") -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, cache_dtype=cache_dtype)

    @property
    def cache_suffix(self) -> str:
        return self.ANIMA_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.ANIMA_LATENTS_NPZ_SUFFIX

    def is_disk_cached_latents_expected(
        self,
        bucket_reso: Tuple[int, int],
        npz_path: str,
        flip_aug: bool,
        alpha_mask: bool,
        num_aug_variants: int = 0,
        aug_config_hash: Optional[str] = None,
    ):
        return self._default_is_disk_cached_latents_expected(
            8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True,
            num_aug_variants=num_aug_variants, aug_config_hash=aug_config_hash,
        )

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int], variant: int = 0
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray], Optional[bool]]:
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso, variant=variant)

    def cache_batch_latents(
        self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool, random_crop_padding_percent: float = 0.05,
        num_aug_variants: int = 0, augmentor: Optional[Callable] = None, aug_config_hash: Optional[str] = None,
    ):
        """Cache batch of latents using Qwen Image VAE.

        vae is expected to be the Qwen Image VAE (AutoencoderKLQwenImage).
        The encoding function handles the mean/std normalization.
        """
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage = vae
        vae_device = vae.device
        vae_dtype = vae.dtype

        def encode_by_vae(img_tensor):
            """Encode image tensor to latents.

            img_tensor: (B, C, H, W) in [-1, 1] range (already normalized by IMAGE_TRANSFORMS)
            Qwen Image VAE accepts inputs in (B, C, H, W) or (B, C, 1, H, W) shape.
            Returns latents in (B, 16, 1, H/8, W/8) shape on CPU.
            """
            latents = vae.encode_pixels_to_latents(img_tensor)  # Keep 4D for input/output
            return latents.to("cpu")

        self._default_cache_batch_latents(
            encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True,
            random_crop_padding_percent=random_crop_padding_percent,
            num_aug_variants=num_aug_variants, augmentor=augmentor, aug_config_hash=aug_config_hash,
        )

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae_device)
