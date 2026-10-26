# MOSS-Image tokenize / text-encoding / caching strategies.
#
# Text: each caption (and each cached caption variant) is chat-templated and encoded ALONE, unpadded, by the frozen
# Qwen3.5-9B (official encode_prompt). The cache stores the unpadded [T, 4096] bf16 rows; the dataloader pads them
# per batch and the DiT masks the padding. The 9B encoder is loaded lazily (MossTextEncoderProvider) only when some
# caption actually needs encoding, and released right after caching, so it never shares VRAM with the DiT.
# Latents: FLUX.2 VAE + 2x2 patchify + BN -> [128, H/16, W/16], cached as ``<stem>_{W}x{H}_moss.npz`` (stride 16).

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from library import train_util
from library.cache_utils import CACHE_DTYPE_METADATA_KEY, save_npz
from library.strategy_base import (LatentsCachingStrategy, TextEncoderOutputsCachingStrategy, TextEncodingStrategy,
                                   TokenizeStrategy, variant_key)
from library.utils import setup_logging

from .captions import info_caption_digest, npz_key
from .qwen35 import MAX_TOKENS, MossQwen35TextEncoder, MossTokenizer, load_moss_text_encoder
from .vae import moss_vae_encode

setup_logging()
import logging

logger = logging.getLogger(__name__)

TEXT_EMBEDS_KEY = "text_embeds"
TEXT_MASK_KEY = "text_mask"
SIGNATURE_KEY = "moss_te_signature"
CAPTION_DIGEST_KEY = "moss_caption_sha1"
CACHE_SCHEMA = "moss-te-v1"


def check_moss_text_encoder_path(te_path: str) -> None:
    """Header-only check (no tensor is read): the file exists and is the Qwen3.5-9B MOSS text encoder."""
    from .qwen35 import check_moss_text_encoder

    if not te_path or not os.path.isfile(te_path):
        raise FileNotFoundError(f"MOSS-Image text encoder not found: {te_path}")
    check_moss_text_encoder(te_path)


def _file_fingerprint(path: str, header_only: bool = False) -> str:
    """sha1 of a whole file, or of a safetensors header (shapes/dtypes) plus size and mtime for multi-GB weights."""
    import hashlib
    import struct

    if not path or not os.path.isfile(path):
        return "missing"
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        if header_only:
            size = struct.unpack("<Q", f.read(8))[0]
            digest.update(f.read(size))
            stat = os.stat(path)
            digest.update(f"{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8"))
        else:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                digest.update(chunk)
    return digest.hexdigest()[:16]


def moss_te_signature(te_path: str, max_length: int, tokenizer_dir: Optional[str] = None) -> str:
    """Identifies encoder weights + tokenizer + prompt formatting of a cache (cheap: never reads the weights)."""
    from .qwen35 import resolve_tokenizer_dir

    try:
        tokenizer_json = os.path.join(resolve_tokenizer_dir(tokenizer_dir), "tokenizer.json")
    except FileNotFoundError:
        tokenizer_json = ""
    return (f"{CACHE_SCHEMA}|{os.path.basename(str(te_path))}|te={_file_fingerprint(te_path, header_only=True)}"
            f"|tok={_file_fingerprint(tokenizer_json)}|max={max_length}|chat-think|qwen2-regex|unescape")


class MossTextEncoderProvider:
    """Loads the Qwen3.5-9B text encoder (bf16, frozen) on first use; ``release()`` drops it."""

    def __init__(self, path: str, device, dtype: torch.dtype = torch.bfloat16, loader: Optional[Callable] = None):
        self.path = path
        self.device = device
        self.dtype = dtype
        self._loader = loader or load_moss_text_encoder
        self._model: Optional[MossQwen35TextEncoder] = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def get(self) -> MossQwen35TextEncoder:
        if self._model is None:
            logger.info(f"loading the MOSS text encoder (Qwen3.5-9B, ~16 GB) for caching: {self.path} -> {self.device}")
            self._model = self._loader(self.path, device=self.device, dtype=self.dtype)
        return self._model

    def release(self) -> None:
        self._model = None


class MossTextEncoderStub(nn.Module):
    """Placeholder in the trainer's text-encoder list (the base trainer iterates and deletes it); holds the provider
    outside the module tree so ``.to()`` / ``.parameters()`` never touch the 9B model."""

    supports_gradient_checkpointing = False

    def __init__(self, provider: MossTextEncoderProvider):
        super().__init__()
        self.__dict__["provider"] = provider
        self.register_buffer("_placeholder", torch.zeros(1), persistent=False)

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, *args, **kwargs):  # pragma: no cover - never called
        raise RuntimeError("MOSS text encoding runs through the cached outputs (cache_text_encoder_outputs)")


def resolve_text_encoder(model: Any) -> MossQwen35TextEncoder:
    if isinstance(model, MossTextEncoderProvider):
        return model.get()
    if isinstance(model, MossTextEncoderStub):
        return model.provider.get()
    return model


def pad_rows(rows: List[torch.Tensor], pad_value=0) -> Tuple[torch.Tensor, torch.Tensor]:
    """[T_i, ...] rows -> right-padded [B, max T, ...] and mask [B, max T] (long, 1 = token)."""
    width = max(r.shape[0] for r in rows)
    out = rows[0].new_full((len(rows), width, *rows[0].shape[1:]), pad_value)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, : r.shape[0]] = r
        mask[i, : r.shape[0]] = 1
    return out, mask


class MossTokenizeStrategy(TokenizeStrategy):
    def __init__(self, tokenizer_dir: Optional[str] = None, max_length: int = MAX_TOKENS) -> None:
        self.tokenizer = MossTokenizer(tokenizer_dir, max_length=max_length)
        logger.info(f"MOSS tokenizer: {self.tokenizer.tokenizer_dir} (max {max_length} tokens, chat template, thinking)")

    def tokenize(self, text) -> List[torch.Tensor]:
        """-> [input_ids [B, L] long (pad 248044), attention_mask [B, L] long]; L = longest prompt."""
        texts = [text] if isinstance(text, str) else list(text)
        ids, mask = pad_rows([torch.tensor(i, dtype=torch.long) for i in self.tokenizer(texts)], self.tokenizer.pad_token_id)
        return [ids, mask]


class MossTextEncodingStrategy(TextEncodingStrategy):
    @torch.no_grad()
    def encode_tokens(self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor],
                      attn_masks=None) -> List[torch.Tensor]:
        """-> [text_embeds [B, L, 4096] (bf16, zero at padding), text_mask [B, L] long]. Each row runs unpadded."""
        model = resolve_text_encoder(models[0])
        input_ids, mask = tokens[0], tokens[1]
        rows = []
        with torch.autocast(device_type=model.device.type, enabled=False):
            for i in range(input_ids.shape[0]):
                length = int(mask[i].sum())
                if length < 1:
                    raise ValueError("every MOSS prompt has at least the 10 chat-template tokens")
                rows.append(model(input_ids[i : i + 1, :length].to(model.device))[0])
        return list(pad_rows(rows))

    def drop_cached_text_encoder_outputs(self, text_embeds, text_mask, caption_dropout_rates=None):
        """Caption dropout lives in the cached caption variants (a dropped caption is encoded as '' through the chat
        template, the official unconditional prompt). Zeroing cached rows is not a valid MOSS encoding: refuse it.

        The rates are checked on the host where the outputs are made or read (cache_batch_outputs, load_outputs_npz), so
        a batch that reaches the trainer on the GPU (accelerate's DataLoader moves it) is not read back here: that
        .item() was a GPU->CPU sync per batch. Rates still on the CPU (a CPU tensor, a list) are checked here as before."""
        if (caption_dropout_rates is not None and _is_host(caption_dropout_rates)
                and not torch.all(torch.as_tensor(caption_dropout_rates) == 0.0).item()):
            raise RuntimeError(CAPTION_DROPOUT_ERROR)
        return [text_embeds, text_mask]


CAPTION_DROPOUT_ERROR = "MOSS: caption dropout needs --cache_caption_variants N>1 (cached outputs are never zeroed)"


def _is_host(rates) -> bool:
    return not isinstance(rates, torch.Tensor) or rates.device.type == "cpu"


def check_caption_dropout_rate(rate) -> None:
    """The per-sample half of drop_cached_text_encoder_outputs' check, on the host: a cached caption_dropout_rate other
    than 0 (a cache written with caption dropout and no caption variants) is refused when it is made or read."""
    if bool((np.asarray(rate, dtype=np.float64) != 0.0).any()):
        raise RuntimeError(CAPTION_DROPOUT_ERROR)


def _decode_cached(array: np.ndarray, stored_dtype: Optional[str]):
    if stored_dtype == "bf16":
        return torch.from_numpy(np.ascontiguousarray(array.astype(np.uint16, copy=False))).view(torch.bfloat16)
    return array


def _save_npz_uncompressed(path: str, arrays: Dict[str, Any], cache_dtype: str) -> None:
    try:
        save_npz(path, arrays, cache_dtype=cache_dtype, compress=False)  # bf16 rows barely compress; zlib only costs time
    except TypeError:  # cache_utils without the compress parameter
        save_npz(path, arrays, cache_dtype=cache_dtype)


class MossTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    MOSS_TE_NPZ_SUFFIX = "_moss_te.npz"
    OUTPUT_KEYS = (TEXT_EMBEDS_KEY, TEXT_MASK_KEY)

    def __init__(self, cache_to_disk: bool, batch_size: Optional[int], skip_disk_cache_validity_check: bool,
                 cache_dtype: str = "auto", signature: str = "") -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, False, cache_dtype=cache_dtype)
        self.signature = signature
        self.caption_digests: Dict[str, str] = {}  # npz_key(npz path) -> digest of the current raw caption

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + self.MOSS_TE_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str, num_caption_variants: int = 0,
                                        caption_aug_hash: Optional[str] = None) -> bool:
        if not self.cache_to_disk or not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True
        try:
            with np.load(npz_path, allow_pickle=False) as npz:
                for key in self.OUTPUT_KEYS + ("caption_dropout_rate", SIGNATURE_KEY):
                    if key not in npz:
                        return False
                if str(npz[SIGNATURE_KEY].tolist()) != self.signature:
                    return False
                expected = self.caption_digests.get(npz_key(npz_path))
                if expected is not None and (CAPTION_DIGEST_KEY not in npz or str(npz[CAPTION_DIGEST_KEY].tolist()) != expected):
                    logger.info(f"caption changed since caching, rebuilding: {npz_path}")
                    return False
                if not self._check_cached_variant_keys(npz, set(self.OUTPUT_KEYS), num_caption_variants, caption_aug_hash):
                    return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e
        return True

    def load_outputs_npz(self, npz_path: str, variant: int = 0) -> List[Any]:
        """Reads only one variant's entries: [text_embeds [T,4096] bf16, text_mask [T] int32, caption_dropout_rate]."""
        with np.load(npz_path, allow_pickle=False) as npz:
            files = set(npz.files)
            dtypes = json.loads(str(npz[CACHE_DTYPE_METADATA_KEY].tolist())) if CACHE_DTYPE_METADATA_KEY in files else {}
            outputs = []
            for base in self.OUTPUT_KEYS:
                key = variant_key(base, variant) if variant > 0 and variant_key(base, variant) in files else base
                outputs.append(_decode_cached(npz[key], dtypes.get(key)))
            rate = npz["caption_dropout_rate"]
        check_caption_dropout_rate(rate)  # on the host (DataLoader worker): no device read per batch later
        return outputs + [rate]

    def cache_batch_outputs(self, tokenize_strategy: TokenizeStrategy, models: List[Any],
                            text_encoding_strategy: TextEncodingStrategy, infos: List):
        # Variant 0 = the processed canonical caption (caption_for_cache: variant 0, else info.cache_caption set by
        # BaseDataset.new_cache_text_encoder_outputs / captions.prepare_canonical_captions, never the raw caption).
        canonical, variant_groups = self._get_variant_caption_groups(infos)

        def encode(captions: List[str]) -> List[Dict[str, Any]]:
            tokens = tokenize_strategy.tokenize(captions)
            embeds, mask = text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens)
            embeds = embeds.detach().to("cpu")
            rows = []
            for i, length in enumerate(mask.sum(1).tolist()):
                row = embeds[i, : int(length)].clone()
                rows.append({TEXT_EMBEDS_KEY: row if row.dtype == torch.bfloat16 else row.float().numpy(),
                             TEXT_MASK_KEY: np.ones(int(length), dtype=np.int32)})
            return rows

        canonical_rows = encode(canonical)
        variant_rows = [(group, encode([caption for _, caption in group])) for group in variant_groups]

        for i, info in enumerate(infos):
            n_variants = len(info.caption_variants) if getattr(info, "caption_variants", None) else 0
            rate = torch.tensor(info.caption_dropout_rate if n_variants <= 1 else 0.0, dtype=torch.float32)
            if not self.cache_to_disk:  # in-memory outputs are never read through load_outputs_npz: check them here
                check_caption_dropout_rate(rate)
            own = []
            for k, (group, rows) in enumerate(variant_rows, start=1):
                indices = [index for index, _ in group]
                if i in indices:
                    own.append((k, rows[indices.index(i)]))
            if self.cache_to_disk:
                arrays = dict(canonical_rows[i])
                arrays["caption_dropout_rate"] = rate
                arrays[SIGNATURE_KEY] = np.array(self.signature)
                arrays[CAPTION_DIGEST_KEY] = np.array(info_caption_digest(info))
                for k, row in own:
                    for key in self.OUTPUT_KEYS:
                        arrays[variant_key(key, k)] = row[key]
                if n_variants > 1:
                    arrays["caption_variants"] = np.array(n_variants)
                    arrays["caption_aug_hash"] = np.array(getattr(info, "caption_aug_hash", None) or "")
                _save_npz_uncompressed(info.text_encoder_outputs_npz, arrays, self.cache_dtype)
            else:
                info.text_encoder_outputs = tuple([canonical_rows[i][k] for k in self.OUTPUT_KEYS] + [rate])
                if n_variants > 1:
                    info.text_encoder_outputs_variants = [tuple([row[k] for k in self.OUTPUT_KEYS] + [rate]) for _, row in own]


class MossLatentsCachingStrategy(LatentsCachingStrategy):
    """FLUX.2 VAE + 2x2 patchify + BN: [128, H/16, W/16] per image (one latent pixel = one DiT token)."""

    MOSS_LATENTS_NPZ_SUFFIX = "_moss.npz"  # exactly one "_" token after the size: the size scan splits on "_"
    STRIDE = 16

    @property
    def cache_suffix(self) -> str:
        return self.MOSS_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.MOSS_LATENTS_NPZ_SUFFIX

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool,
                                        num_aug_variants: int = 0, aug_config_hash: Optional[str] = None):
        return self._default_is_disk_cached_latents_expected(
            self.STRIDE, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True,
            num_aug_variants=num_aug_variants, aug_config_hash=aug_config_hash)

    def load_latents_from_disk(self, npz_path: str, bucket_reso: Tuple[int, int], variant: int = 0):
        return self._default_load_latents_from_disk(self.STRIDE, npz_path, bucket_reso, variant=variant)

    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool,
                            random_crop_padding_percent: float = 0.05, num_aug_variants: int = 0,
                            augmentor: Optional[Callable] = None, aug_config_hash: Optional[str] = None):
        def encode_by_vae(pixels):
            return moss_vae_encode(vae, pixels).to("cpu")

        self._default_cache_batch_latents(
            encode_by_vae, vae.device, vae.dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True,
            random_crop_padding_percent=random_crop_padding_percent, num_aug_variants=num_aug_variants,
            augmentor=augmentor, aug_config_hash=aug_config_hash)
        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)
