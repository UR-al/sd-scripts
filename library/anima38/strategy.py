# Anima-3.8B v1.1 tokenize / text-encoding / TE-output caching strategies (--anima38_connector).
#
# They extend the Anima strategies with the Qwen3.5-4B semantic features (layers 7/15/23/31) that feed Semantic
# Connector v2. Every text-encoder output tuple gains two entries:
#
#   [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, qwen35_states, qwen35_mask]
#
# qwen35_states is one [L, 4 * 2560] tensor per caption (the 4 layers side by side), so the dataloader's
# existing collation pads captions of different lengths along L and qwen35_mask ([L], 1 = token) marks the padding.
#
# Caching: both encoders see the SAME caption string, and both results are stored in ONE file per image
# (``*_anima38_te.npz`` next to the image, where the fork writes ``*_anima_te.npz``), variant k under the same
# ``_v{k}`` keys. The dataloader loads one variant index from that file, so the Qwen3-0.6B and Qwen3.5 features of a
# training sample always come from the same (shuffled / tag-dropped) caption. Qwen3.5 is loaded lazily on the first
# caption that needs encoding and released by the trainer after caching.
#
# Each file also stores the SHA-1 of the raw caption it was built from and of the processed caption it encodes; the
# trainer registers the current digests (captions.register_caption_digests), so an edited caption or a changed
# prefix/suffix/separator rebuilds its cache. Qwen3.5 gets the caption escaped the
# way it is typed into Forge (captions.forge_prompt_text); Qwen3 and T5 get it unchanged, as in Forge.

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from library.cache_utils import CACHE_DTYPE_METADATA_KEY, save_npz
from library.strategy_anima import (
    AnimaTextEncoderOutputsCachingStrategy,
    AnimaTextEncodingStrategy,
    AnimaTokenizeStrategy,
)
from library.strategy_base import TextEncodingStrategy, TokenizeStrategy, caption_for_cache, variant_key

from .captions import caption_digest, forge_prompt_text, npz_key
from .qwen35 import LAYER_INDICES, clear_scan_graphs, load_qwen35_model, pad_token_batch
from .tokenizer import Qwen35Tokenizer
from .training import pack_semantic_layers

logger = logging.getLogger(__name__)

QWEN35_STATES_KEY = "qwen35_states"
QWEN35_MASK_KEY = "qwen35_mask"
SIGNATURE_KEY = "anima38_signature"
CAPTION_DIGEST_KEY = "anima38_caption_sha1"
CACHE_CAPTION_DIGEST_KEY = "anima38_cache_caption_sha1"  # digest of the PROCESSED caption the file was encoded from
# v2: canonical variant without the keep-token separator, Qwen3.5 input escaped like a Forge prompt, caption digest
CACHE_SCHEMA = "anima38-te-v2"


def qwen35_cache_signature(qwen35_path: str, layer_indices: Sequence[int] = LAYER_INDICES) -> str:
    """Identifies the Qwen3.5 features in a cache file (checkpoint name + size, layers, schema). Metadata only."""
    size = os.path.getsize(qwen35_path) if qwen35_path and os.path.exists(qwen35_path) else -1
    layers = ",".join(str(int(index)) for index in layer_indices)
    return f"{CACHE_SCHEMA}|{os.path.basename(str(qwen35_path))}|{size}|layers={layers}"


class Qwen35Provider:
    """Loads Qwen3.5-4B (read-only, FP8 storage, frozen) on first use; ``release()`` drops it."""

    def __init__(self, path: str, device, loader=None):
        self.path = path
        self.device = device
        self._loader = loader or load_qwen35_model
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def get(self):
        if self._model is None:
            logger.info(f"loading Qwen3.5-4B for Anima-3.8B conditioning: {self.path} -> {self.device}")
            self._model = self._loader(self.path, device=self.device)
        return self._model

    def release(self) -> None:
        self._model = None
        clear_scan_graphs()  # the DeltaNet CUDA graphs hold small memory pools; free them before training


class Anima38TokenizeStrategy(AnimaTokenizeStrategy):
    """Anima tokenization plus Qwen3.5 ids of the same text (raw text, no special tokens, <= 1024 tokens)."""

    def __init__(self, qwen35_tokenizer_dir: Optional[str] = None, qwen35_tokenizer=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.qwen35_tokenizer = qwen35_tokenizer if qwen35_tokenizer is not None else Qwen35Tokenizer(qwen35_tokenizer_dir)

    def tokenize(self, text):
        text = [text] if isinstance(text, str) else list(text)
        tokens = super().tokenize(text)
        qwen35_ids, qwen35_mask = self.tokenize_qwen35(text)
        return tokens + [qwen35_ids, qwen35_mask]

    def tokenize_qwen35(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Right-padded [B, L] ids (pad 151643) and mask (1 = token), L = longest text in the batch."""
        token_lists = self.qwen35_tokenizer(list(texts))["input_ids"]
        return pad_token_batch(token_lists, self.qwen35_tokenizer.pad_token_id)


class Anima38TextEncodingStrategy(AnimaTextEncodingStrategy):
    """models = [qwen3_text_encoder, Qwen35HybridModel or Qwen35Provider]; tokens = Anima38TokenizeStrategy output."""

    def __init__(self, qwen35_batch_size: int = 1, layer_indices: Sequence[int] = LAYER_INDICES) -> None:
        super().__init__()
        self.qwen35_batch_size = max(1, int(qwen35_batch_size))
        self.layer_indices = tuple(int(index) for index in layer_indices)

    def encode_tokens(self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(tokens) != 6 or len(models) < 2:
            raise ValueError("Anima-3.8B encoding needs 6 token tensors (Anima38TokenizeStrategy) and [qwen3, qwen35] models")
        native = super().encode_tokens(tokenize_strategy, models[:1], list(tokens[:4]))
        states, mask = self.encode_qwen35(models[1], tokens[4], tokens[5])
        return list(native) + [states, mask]

    @torch.no_grad()
    def encode_qwen35(self, model_or_provider, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Packed Qwen3.5 layer states [B, L, 4 * 2560] (zero at padding) and the mask [B, L] (long).

        Rows are encoded ``qwen35_batch_size`` at a time, each chunk trimmed to its longest caption; with the default
        of 1 every caption runs unpadded with a mask of ones, which is exactly Forge's ``_semantic_layers`` call.
        Autocast is disabled: the port computes in the embedding dtype (bf16) like Forge.
        """
        model = model_or_provider.get() if isinstance(model_or_provider, Qwen35Provider) else model_or_provider
        device = model.embed_tokens.weight.device
        mask = attention_mask.to(torch.long).cpu()
        lengths = [int(n) for n in mask.sum(dim=1).tolist()]
        if min(lengths) < 1:
            raise ValueError("every Qwen3.5 sequence needs at least one token (empty text is tokenized as [151643])")
        batch, width = input_ids.shape
        packed = None
        with torch.autocast(device_type=device.type, enabled=False):
            for start in range(0, batch, self.qwen35_batch_size):
                stop = min(batch, start + self.qwen35_batch_size)
                chunk_width = max(lengths[start:stop])
                ids = input_ids[start:stop, :chunk_width].to(device)
                chunk_mask = mask[start:stop, :chunk_width].to(device)
                states = model.forward_semantic_layers(ids, attention_mask=chunk_mask, layer_indices=self.layer_indices)
                chunk = pack_semantic_layers(states)
                if stop - start > 1:
                    chunk = chunk.masked_fill(~chunk_mask.to(torch.bool)[:, :, None], 0)
                if packed is None:
                    packed = chunk.new_zeros(batch, width, chunk.shape[-1])
                packed[start:stop, :chunk_width] = chunk
        return packed, mask.to(packed.device)

    def drop_cached_text_encoder_outputs(
        self,
        prompt_embeds: torch.Tensor,
        attn_mask: torch.Tensor,
        t5_input_ids: torch.Tensor,
        t5_attn_mask: torch.Tensor,
        qwen35_states: Optional[torch.Tensor] = None,
        qwen35_mask: Optional[torch.Tensor] = None,
        caption_dropout_rates: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Caption dropout is baked into the cached caption variants (a dropped caption is encoded as '' by both
        encoders), so the stored rate is 0. A non-zero rate here means an unsupported cache; refuse it."""
        if caption_dropout_rates is not None and not torch.all(caption_dropout_rates == 0.0).item():
            raise RuntimeError(
                "Anima-3.8B: cached caption dropout without caption variants is not supported; use --cache_caption_variants N>1"
            )
        return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, qwen35_states, qwen35_mask]


def _decode_cached(array: np.ndarray, stored_dtype: Optional[str]):
    """BF16 entries (uint16 bit patterns, see cache_utils) -> torch.bfloat16; everything else as stored."""
    if stored_dtype == "bf16":
        return torch.from_numpy(np.ascontiguousarray(array.astype(np.uint16, copy=False))).view(torch.bfloat16)
    return array


class Anima38TextEncoderOutputsCachingStrategy(AnimaTextEncoderOutputsCachingStrategy):
    """Caches Qwen3-0.6B + T5 ids (as the Anima strategy) and the Qwen3.5 states of the same caption, per variant."""

    ANIMA38_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_anima38_te.npz"
    OUTPUT_KEYS = ("prompt_embeds", "attn_mask", "t5_input_ids", "t5_attn_mask", QWEN35_STATES_KEY, QWEN35_MASK_KEY)

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        cache_dtype: str = "auto",
        qwen35_signature: str = "",
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial, cache_dtype=cache_dtype)
        self.qwen35_signature = qwen35_signature
        self.caption_digests: Dict[str, str] = {}  # npz_key(npz path) -> caption_digest(current raw caption)
        # npz_key(npz path) -> caption_digest(processed caption: prefix/suffix, separators, replacements applied)
        self.cache_caption_digests: Dict[str, str] = {}

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + self.ANIMA38_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str, num_caption_variants: int = 0, caption_aug_hash: Optional[str] = None) -> bool:
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True
        try:
            with np.load(npz_path, allow_pickle=False) as npz:  # lazy: only the keys read below are decompressed
                for key in self.OUTPUT_KEYS + ("caption_dropout_rate", SIGNATURE_KEY):
                    if key not in npz:
                        return False
                if str(npz[SIGNATURE_KEY].tolist()) != self.qwen35_signature:
                    return False
                expected_digest = self.caption_digests.get(npz_key(npz_path))
                if expected_digest is not None and (
                    CAPTION_DIGEST_KEY not in npz or str(npz[CAPTION_DIGEST_KEY].tolist()) != expected_digest
                ):
                    logger.info(f"caption changed since caching, rebuilding: {npz_path}")
                    return False
                if num_caption_variants is None or num_caption_variants <= 1:
                    # Without caption variants the file holds the PROCESSED caption (prefix/suffix, secondary separator,
                    # replacements; strategy_base.caption_for_cache). Files written before that fix carry no processed
                    # digest: they are kept only when the processed caption equals the raw one, else rebuilt.
                    expected_processed = self.cache_caption_digests.get(npz_key(npz_path))
                    if expected_processed is not None:
                        stored = str(npz[CACHE_CAPTION_DIGEST_KEY].tolist()) if CACHE_CAPTION_DIGEST_KEY in npz else expected_digest
                        if stored != expected_processed:
                            logger.info(f"processed caption changed since caching, rebuilding: {npz_path}")
                            return False
                if not self._check_cached_variant_keys(npz, set(self.OUTPUT_KEYS), num_caption_variants, caption_aug_hash):
                    return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e
        return True

    def load_outputs_npz(self, npz_path: str, variant: int = 0) -> List[Any]:
        """Reads only the entries of one variant (the base loader decodes every variant in the file)."""
        with np.load(npz_path, allow_pickle=False) as npz:
            files = set(npz.files)
            dtypes = json.loads(str(npz[CACHE_DTYPE_METADATA_KEY].tolist())) if CACHE_DTYPE_METADATA_KEY in files else {}
            outputs = []
            for base_key in self.OUTPUT_KEYS:
                key = variant_key(base_key, variant) if variant > 0 and variant_key(base_key, variant) in files else base_key
                outputs.append(_decode_cached(npz[key], dtypes.get(key)))
            caption_dropout_rate = npz["caption_dropout_rate"]
        return outputs + [caption_dropout_rate]

    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        infos: List,
    ):
        canonical_captions, variant_groups = self._get_variant_caption_groups(infos)

        def encode_captions(captions: List[str]) -> List[Dict[str, Any]]:
            tokens_and_masks = tokenize_strategy.tokenize(captions)
            # Qwen3.5 reads the prompt line as typed in Forge (escaped brackets); Qwen3 / T5 keep the plain caption
            tokens_and_masks[4:6] = tokenize_strategy.tokenize_qwen35([forge_prompt_text(caption) for caption in captions])
            with torch.no_grad():
                prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, qwen35_states, qwen35_mask = text_encoding_strategy.encode_tokens(
                    tokenize_strategy, models, tokens_and_masks
                )
            # same storage as the Anima strategy: bf16 stays a CPU tensor (save_npz stores its bits), else numpy
            prompt_embeds = prompt_embeds.detach().to("cpu")
            qwen35_states = qwen35_states.detach().to("cpu")
            qwen35_lengths = qwen35_mask.to("cpu").sum(dim=1).tolist()
            rows = []
            for i in range(len(captions)):
                length = int(qwen35_lengths[i])
                states = qwen35_states[i, :length].clone()  # unpadded; the dataloader pads per batch
                rows.append(
                    {
                        "prompt_embeds": prompt_embeds[i] if prompt_embeds.dtype == torch.bfloat16 else prompt_embeds[i].numpy(),
                        "attn_mask": attn_mask[i].cpu().numpy(),
                        "t5_input_ids": t5_input_ids[i].cpu().numpy().astype(np.int32),
                        "t5_attn_mask": t5_attn_mask[i].cpu().numpy().astype(np.int32),
                        QWEN35_STATES_KEY: states if states.dtype == torch.bfloat16 else states.numpy(),
                        QWEN35_MASK_KEY: np.ones(length, dtype=np.int32),
                    }
                )
            return rows

        canonical_rows = encode_captions(canonical_captions)
        variant_rows = [(group, encode_captions([caption for _, caption in group])) for group in variant_groups]

        for i, info in enumerate(infos):
            n_variants = len(info.caption_variants) if getattr(info, "caption_variants", None) else 0
            # as in the Anima strategy: with variants, dropout is carried by the variants themselves
            effective_dropout_rate = info.caption_dropout_rate if n_variants <= 1 else 0.0
            caption_dropout_rate = torch.tensor(effective_dropout_rate, dtype=torch.float32)

            own_variants = []  # (k, row) for k = 1..n_variants-1, in order
            for k, (group, rows) in enumerate(variant_rows, start=1):
                indices = [index for index, _ in group]
                if i in indices:
                    own_variants.append((k, rows[indices.index(i)]))

            if self.cache_to_disk:
                save_kwargs = dict(canonical_rows[i])
                save_kwargs["caption_dropout_rate"] = caption_dropout_rate
                save_kwargs[SIGNATURE_KEY] = np.array(self.qwen35_signature)
                save_kwargs[CAPTION_DIGEST_KEY] = np.array(caption_digest(getattr(info, "caption", None)))
                save_kwargs[CACHE_CAPTION_DIGEST_KEY] = np.array(caption_digest(caption_for_cache(info)))
                for k, row in own_variants:
                    for key in self.OUTPUT_KEYS:
                        save_kwargs[variant_key(key, k)] = row[key]
                if n_variants > 1:
                    save_kwargs["caption_variants"] = np.array(n_variants)
                    save_kwargs["caption_aug_hash"] = np.array(getattr(info, "caption_aug_hash", None) or "")
                # uncompressed: zlib only shrinks these bf16 outputs by ~20-35% but costs seconds per file on the
                # main thread while the GPU waits; load_npz reads both forms, so older compressed caches stay valid
                save_npz(info.text_encoder_outputs_npz, save_kwargs, cache_dtype=self.cache_dtype, compress=False)
            else:
                info.text_encoder_outputs = tuple([canonical_rows[i][key] for key in self.OUTPUT_KEYS] + [caption_dropout_rate])
                if n_variants > 1:
                    info.text_encoder_outputs_variants = [
                        tuple([row[key] for key in self.OUTPUT_KEYS] + [caption_dropout_rate]) for _, row in own_variants
                    ]
