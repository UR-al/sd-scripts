"""CPU tests for cache validity against what produced a cache (codex review 2026-11-04):

- SD-01: every *_anima_te.npz (and the Qwen3-0.6B / T5 part of *_anima38_te.npz) carries a signature of the schema,
  the Qwen3 file, the tokenizers and the Qwen3 / T5 max token lengths. A cache with another signature, or written
  before signatures existed, is rebuilt.
- Latent caches store the source image's file size and mtime_ns per resolution; an image edited in place makes its
  latents stale. Caches without these fields stay valid (no mass re-encoding of existing datasets).

Only temporary folders and small fake arrays; no model, no GPU.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp=<scratch dir>
     tests/test_cache_source_signature.py
"""
import json
import os
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from library import strategy_anima
from library.anima38 import strategy as a38_strategy
from library.cache_utils import load_npz, save_npz
from library.strategy_anima import (TE_OUTPUT_KEYS, TE_SIGNATURE_KEY, AnimaLatentsCachingStrategy,
                                    AnimaTextEncoderOutputsCachingStrategy, AnimaTextEncodingStrategy, AnimaTokenizeStrategy,
                                    anima_te_signature)
from library.strategy_base import TokenizeStrategy
from library.train_util import ImageInfo

# ======================================================================================================
# SD-01: Anima TE cache signature
# ======================================================================================================


def _fake_safetensors(path, tensors=None, payload=b"\x00" * 16):
    header = json.dumps(tensors or {"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(payload)
    return str(path)


def _fake_t5_dir(path, vocab="t5-vocab"):
    path.mkdir(exist_ok=True)
    (path / "tokenizer.json").write_text(json.dumps({"vocab": vocab}), encoding="utf-8")
    (path / "spiece.model").write_bytes(b"spiece-" + vocab.encode("utf-8"))
    return str(path)


def test_signature_follows_lengths_encoder_and_tokenizer(tmp_path):
    qwen3 = _fake_safetensors(tmp_path / "qwen3.safetensors")
    t5 = _fake_t5_dir(tmp_path / "t5")
    base = anima_te_signature(qwen3, t5, 512, 512)
    assert base == anima_te_signature(qwen3, t5, 512, 512)  # deterministic
    assert base.startswith(strategy_anima.TE_CACHE_SCHEMA + "|")
    assert anima_te_signature(qwen3, t5, 540, 512) != base  # qwen3_max_token_length
    assert anima_te_signature(qwen3, t5, 512, 620) != base  # t5_max_token_length

    other_t5 = _fake_t5_dir(tmp_path / "t5_other", vocab="other")
    assert anima_te_signature(qwen3, other_t5, 512, 512) != base  # another T5 tokenizer
    assert anima_te_signature(qwen3, None, 512, 512) != base  # the bundled configs/t5_old tokenizer

    other_qwen3 = _fake_safetensors(
        tmp_path / "qwen3_other.safetensors", {"w": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}, b"\x00" * 8
    )
    assert anima_te_signature(other_qwen3, t5, 512, 512) != base  # another Qwen3 file


def test_signature_ignores_the_qwen3_file_name(tmp_path):
    """Renaming / moving the Qwen3 file (same bytes, same mtime) is not an encoder change."""
    qwen3 = _fake_safetensors(tmp_path / "qwen3.safetensors")
    t5 = _fake_t5_dir(tmp_path / "t5")
    before = anima_te_signature(qwen3, t5, 512, 512)
    moved = str(tmp_path / "renamed.safetensors")
    os.replace(qwen3, moved)
    assert anima_te_signature(moved, t5, 512, 512) == before


class _FakeTokenizer:
    def __call__(self, text, return_tensors=None, truncation=True, padding="max_length", max_length=8):
        ids = torch.zeros(len(text), max_length, dtype=torch.long)
        mask = torch.zeros(len(text), max_length, dtype=torch.long)
        for row, caption in enumerate(text):
            n = min(max_length, len(caption.split(",")))
            ids[row, :n] = torch.arange(1, n + 1)
            mask[row, :n] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _FakeQwen3:
    device = torch.device("cpu")

    def __call__(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=input_ids[:, :, None].float().repeat(1, 1, 4))


def _tokenize_strategy(tmp_path, qwen3_len=8, t5_len=8):
    return AnimaTokenizeStrategy(
        qwen3_tokenizer=_FakeTokenizer(), t5_tokenizer=_FakeTokenizer(), qwen3_max_length=qwen3_len, t5_max_length=t5_len,
        qwen3_path=_fake_safetensors(tmp_path / "qwen3.safetensors"), t5_tokenizer_path=_fake_t5_dir(tmp_path / "t5"),
    )


def _te_info(tmp_path, name="img1", caption="1boy, solo, beach"):
    return SimpleNamespace(
        caption=caption, cache_caption=caption, caption_variants=None, caption_dropout_rate=0.0, caption_aug_hash=None,
        absolute_path=str(tmp_path / f"{name}.png"), text_encoder_outputs_npz=str(tmp_path / f"{name}_anima_te.npz"),
        text_encoder_outputs=None, text_encoder_outputs_variants=None,
    )


def test_tokenize_strategy_signature_matches_the_function(tmp_path):
    tokenize = _tokenize_strategy(tmp_path, 540, 620)
    assert tokenize.cache_signature() == anima_te_signature(str(tmp_path / "qwen3.safetensors"), str(tmp_path / "t5"), 540, 620)
    # tokenizers handed over as objects, no path: the encoder cannot be identified, so no signature
    assert AnimaTokenizeStrategy(qwen3_tokenizer=_FakeTokenizer(), t5_tokenizer=_FakeTokenizer()).cache_signature() is None


def test_anima_te_cache_is_rebuilt_when_the_signature_changes(tmp_path, monkeypatch):
    tokenize = _tokenize_strategy(tmp_path, 8, 8)
    monkeypatch.setattr(TokenizeStrategy, "_strategy", tokenize)  # what the trainer sets before caching
    caching = AnimaTextEncoderOutputsCachingStrategy(True, 1, False)
    info = _te_info(tmp_path)
    caching.cache_batch_outputs(tokenize, [_FakeQwen3()], AnimaTextEncodingStrategy(), [info])

    with np.load(info.text_encoder_outputs_npz) as npz:
        assert str(npz[TE_SIGNATURE_KEY].tolist()) == tokenize.cache_signature()
    assert caching.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz)

    # same files, other token lengths (config_anima29.toml 540/620 vs the 512 configs): stale
    longer = _tokenize_strategy(tmp_path, 8, 12)
    monkeypatch.setattr(TokenizeStrategy, "_strategy", longer)
    assert not AnimaTextEncoderOutputsCachingStrategy(True, 1, False).is_disk_cached_outputs_expected(info.text_encoder_outputs_npz)
    # an explicit signature wins over the global tokenize strategy
    explicit = AnimaTextEncoderOutputsCachingStrategy(True, 1, False, te_signature=tokenize.cache_signature())
    assert explicit.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz)
    # skip_cache_check still trusts any existing file
    assert AnimaTextEncoderOutputsCachingStrategy(True, 1, True).is_disk_cached_outputs_expected(info.text_encoder_outputs_npz)


def test_anima_te_cache_without_signature_is_rebuilt(tmp_path, monkeypatch):
    tokenize = _tokenize_strategy(tmp_path)
    monkeypatch.setattr(TokenizeStrategy, "_strategy", tokenize)
    path = str(tmp_path / "old_anima_te.npz")
    arrays = {key: np.zeros(4, dtype=np.int32) for key in TE_OUTPUT_KEYS}
    arrays["caption_dropout_rate"] = np.array(0.0, dtype=np.float32)
    save_npz(path, arrays)  # a cache written before the signature existed
    assert not AnimaTextEncoderOutputsCachingStrategy(True, 1, False).is_disk_cached_outputs_expected(path)
    arrays[TE_SIGNATURE_KEY] = np.array(tokenize.cache_signature())
    save_npz(path, arrays)
    assert AnimaTextEncoderOutputsCachingStrategy(True, 1, False).is_disk_cached_outputs_expected(path)


class _FakeAnima38Tokenize:
    def __init__(self, signature):
        self.signature = signature

    def cache_signature(self):
        return self.signature

    def tokenize(self, captions):
        n = len(captions)
        return [torch.ones(n, 4, dtype=torch.long), torch.ones(n, 4, dtype=torch.long), torch.ones(n, 4, dtype=torch.long),
                torch.ones(n, 4, dtype=torch.long), torch.ones(n, 3, dtype=torch.long), torch.ones(n, 3, dtype=torch.long)]

    def tokenize_qwen35(self, captions):
        n = len(captions)
        return torch.ones(n, 3, dtype=torch.long), torch.ones(n, 3, dtype=torch.long)


class _FakeAnima38Encoding:
    def encode_tokens(self, tokenize_strategy, models, tokens):
        n = tokens[0].shape[0]
        return [torch.ones(n, 4, 2), tokens[1], tokens[2], tokens[3], torch.ones(n, 3, 5), tokens[5]]


def test_anima38_cache_checks_the_qwen3_t5_signature(tmp_path, monkeypatch):
    monkeypatch.setattr(TokenizeStrategy, "_strategy", _FakeAnima38Tokenize("qwen3-sig-A"))
    caching = a38_strategy.Anima38TextEncoderOutputsCachingStrategy(True, 1, False, qwen35_signature="q35")
    info = _te_info(tmp_path)
    info.text_encoder_outputs_npz = str(tmp_path / "img1_anima38_te.npz")
    caching.cache_batch_outputs(TokenizeStrategy.get_strategy(), [None, None], _FakeAnima38Encoding(), [info])
    path = info.text_encoder_outputs_npz
    with np.load(path) as npz:
        assert str(npz[TE_SIGNATURE_KEY].tolist()) == "qwen3-sig-A"
    assert caching.is_disk_cached_outputs_expected(path)

    monkeypatch.setattr(TokenizeStrategy, "_strategy", _FakeAnima38Tokenize("qwen3-sig-B"))  # other Qwen3 / T5 / lengths
    assert not caching.is_disk_cached_outputs_expected(path)

    monkeypatch.setattr(TokenizeStrategy, "_strategy", _FakeAnima38Tokenize("qwen3-sig-A"))
    arrays = load_npz(path)
    del arrays[TE_SIGNATURE_KEY]
    save_npz(path, arrays)  # written before the Qwen3 / T5 signature existed
    assert not caching.is_disk_cached_outputs_expected(path)


# ======================================================================================================
# latent caches: source image size + mtime_ns
# ======================================================================================================


def _image(path, color=(200, 10, 10), size=(64, 64)):
    Image.new("RGB", size, color).save(path)
    return str(path)


def _latent_info(tmp_path, name="img"):
    image_path = _image(tmp_path / f"{name}.png")
    info = ImageInfo(name, 1, "a cat", False, False, image_path)
    info.bucket_reso = (64, 64)
    info.resized_size = (64, 64)
    info.image_size = (64, 64)
    info.latents_npz = AnimaLatentsCachingStrategy(True, 1, False).get_latents_npz_path(image_path, info.image_size)
    return info


def _encode(img_tensor):  # (B, 3, H, W) -> (B, 16, H/8, W/8), like the Anima VAE wrapper's 4D output
    pooled = torch.nn.functional.avg_pool2d(img_tensor.float(), 8)
    return pooled.repeat(1, 6, 1, 1)[:, :16]


def _edit_in_place(path):
    """What an image editor does: new pixels (another file size) written to the same path, newer mtime."""
    stat = os.stat(path)
    Image.new("RGB", (64, 64), (10, 200, 10)).save(path, compress_level=0)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))


@pytest.mark.parametrize("num_aug_variants", [0, 2])
def test_latent_cache_notices_an_image_edited_in_place(tmp_path, num_aug_variants):
    strategy = AnimaLatentsCachingStrategy(True, 1, False)
    info = _latent_info(tmp_path)
    strategy._default_cache_batch_latents(
        _encode, torch.device("cpu"), torch.float32, [info], False, False, False, multi_resolution=True,
        num_aug_variants=num_aug_variants, aug_config_hash="h" if num_aug_variants else None,
    )
    check = dict(num_aug_variants=num_aug_variants, aug_config_hash="h" if num_aug_variants else None)
    assert strategy.is_disk_cached_latents_expected((64, 64), info.latents_npz, False, False, **check)
    stat = os.stat(info.absolute_path)
    with np.load(info.latents_npz) as npz:
        assert npz["source_stat_8x8"].tolist() == [stat.st_size, stat.st_mtime_ns]

    _edit_in_place(info.absolute_path)
    assert not strategy.is_disk_cached_latents_expected((64, 64), info.latents_npz, False, False, **check)
    # skip_cache_check still trusts any existing file
    assert AnimaLatentsCachingStrategy(True, 1, True).is_disk_cached_latents_expected((64, 64), info.latents_npz, False, False, **check)


def test_latent_cache_notices_mtime_only_change(tmp_path):
    strategy = AnimaLatentsCachingStrategy(True, 1, False)
    info = _latent_info(tmp_path)
    strategy._default_cache_batch_latents(_encode, torch.device("cpu"), torch.float32, [info], False, False, False, multi_resolution=True)
    stat = os.stat(info.absolute_path)
    os.utime(info.absolute_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))  # same size, rewritten
    assert not strategy.is_disk_cached_latents_expected((64, 64), info.latents_npz, False, False)


def test_latent_cache_without_source_fields_stays_valid(tmp_path):
    """Caches written before this change (every existing *_anima.npz) must not be mass re-encoded."""
    strategy = AnimaLatentsCachingStrategy(True, 1, False)
    info = _latent_info(tmp_path)
    strategy.save_latents_to_disk(info.latents_npz, torch.zeros(16, 8, 8), (64, 64), (0, 0, 64, 64), key_reso_suffix="_8x8")
    assert "source_stat_8x8" not in load_npz(info.latents_npz)
    _edit_in_place(info.absolute_path)
    assert strategy.is_disk_cached_latents_expected((64, 64), info.latents_npz, False, False)


def test_latent_cache_with_missing_source_image_stays_valid(tmp_path):
    """The recorded image is gone (moved / renamed): nothing to compare against, keep the cache as before."""
    strategy = AnimaLatentsCachingStrategy(True, 1, False)
    info = _latent_info(tmp_path)
    strategy._default_cache_batch_latents(_encode, torch.device("cpu"), torch.float32, [info], False, False, False, multi_resolution=True)
    os.remove(info.absolute_path)
    assert strategy.is_disk_cached_latents_expected((64, 64), info.latents_npz, False, False)


def test_latent_source_stat_is_per_resolution(tmp_path):
    """One npz holds every bucket resolution of an image; an edit rebuilt at one resolution leaves the others stale."""
    strategy = AnimaLatentsCachingStrategy(True, 1, False)
    info = _latent_info(tmp_path)
    strategy._default_cache_batch_latents(_encode, torch.device("cpu"), torch.float32, [info], False, False, False, multi_resolution=True)
    _edit_in_place(info.absolute_path)
    info.bucket_reso = (128, 64)
    info.resized_size = (128, 64)
    strategy._default_cache_batch_latents(_encode, torch.device("cpu"), torch.float32, [info], False, False, False, multi_resolution=True)
    assert strategy.is_disk_cached_latents_expected((128, 64), info.latents_npz, False, False)  # rebuilt after the edit
    assert not strategy.is_disk_cached_latents_expected((64, 64), info.latents_npz, False, False)  # still the old pixels


def _cache(strategy, info, **kw):
    strategy._default_cache_batch_latents(
        _encode, torch.device("cpu"), torch.float32, [info], kw.pop("flip_aug", False), False, False, multi_resolution=True, **kw
    )


def _paint(path, color, size=(64, 64)):
    stat = os.stat(path)
    Image.new("RGB", size, color).save(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))


def test_rebuild_after_edit_drops_keys_it_does_not_rewrite(tmp_path):
    """flip run caches (old) -> image edited -> no-flip run rebuilds latents only -> the flip run must not take the
    old flipped latents as valid under the new stat (review 2026-11-04)."""
    strategy = AnimaLatentsCachingStrategy(True, 1, False)
    info = _latent_info(tmp_path)
    _paint(info.absolute_path, (255, 255, 255))
    _cache(strategy, info, flip_aug=True)
    _paint(info.absolute_path, (0, 0, 0))
    assert not strategy.is_disk_cached_latents_expected((64, 64), info.latents_npz, False, False)
    _cache(strategy, info, flip_aug=False)
    assert strategy.is_disk_cached_latents_expected((64, 64), info.latents_npz, False, False)
    assert "latents_flipped_8x8" not in load_npz(info.latents_npz)
    assert not strategy.is_disk_cached_latents_expected((64, 64), info.latents_npz, True, False)


def test_plain_rebuild_after_edit_drops_old_variants(tmp_path):
    strategy = AnimaLatentsCachingStrategy(True, 1, False)
    info = _latent_info(tmp_path)
    _cache(strategy, info, num_aug_variants=2, aug_config_hash="h")
    _paint(info.absolute_path, (0, 0, 0))
    _cache(strategy, info)  # plain rebuild of the same resolution
    assert not any(k.startswith("latents_v") for k in load_npz(info.latents_npz))
    assert not strategy.is_disk_cached_latents_expected((64, 64), info.latents_npz, False, False, num_aug_variants=2, aug_config_hash="h")


def test_rewrite_of_an_unchanged_image_keeps_the_other_keys(tmp_path):
    """No edit: adding / re-writing one part of a resolution keeps what is already there (behaviour as before)."""
    strategy = AnimaLatentsCachingStrategy(True, 1, False)
    info = _latent_info(tmp_path)
    _cache(strategy, info, flip_aug=True)
    _cache(strategy, info, flip_aug=False)
    assert strategy.is_disk_cached_latents_expected((64, 64), info.latents_npz, True, False)


def test_rewrite_keeps_other_resolutions(tmp_path):
    strategy = AnimaLatentsCachingStrategy(True, 1, False)
    info = _latent_info(tmp_path)
    _cache(strategy, info)
    _paint(info.absolute_path, (0, 0, 0))
    info.bucket_reso = info.resized_size = (128, 64)
    _cache(strategy, info)
    keys = load_npz(info.latents_npz)
    assert "latents_8x8" in keys and "latents_8x16" in keys  # 8x8 is stale (still checked by its stat), not deleted


def test_image_size_from_cache_name_ignores_an_edited_image(tmp_path):
    """A cache named after the old dimensions must not decide the bucket of an image cropped / resized in place."""
    strategy = AnimaLatentsCachingStrategy(True, 1, False)
    image_path = _image(tmp_path / "img.png", size=(64, 128))
    info = ImageInfo("img", 1, "a cat", False, False, image_path)
    info.image_size = info.bucket_reso = info.resized_size = (64, 128)
    info.latents_npz = strategy.get_latents_npz_path(image_path, info.image_size)
    _cache(strategy, info)
    assert strategy.get_image_size_from_disk_cache_path(image_path, info.latents_npz) == (64, 128)  # unchanged image
    assert strategy.get_image_size_from_disk_cache_path(info.latents_npz, info.latents_npz) == (64, 128)  # fine-tune: abs_path is the npz

    _paint(image_path, (10, 200, 10), size=(128, 128))
    assert strategy.get_image_size_from_disk_cache_path(image_path, info.latents_npz) == (None, None)
    assert strategy.get_image_size_from_disk_cache_path(info.latents_npz, info.latents_npz) == (64, 128)


def test_image_size_from_cache_name_of_an_old_cache(tmp_path):
    strategy = AnimaLatentsCachingStrategy(True, 1, False)
    image_path = _image(tmp_path / "img.png", size=(64, 128))
    npz_path = strategy.get_latents_npz_path(image_path, (64, 128))
    strategy.save_latents_to_disk(npz_path, torch.zeros(16, 16, 8), (64, 128), (0, 0, 64, 128), key_reso_suffix="_16x8")
    _paint(image_path, (10, 200, 10), size=(128, 128))
    assert strategy.get_image_size_from_disk_cache_path(image_path, npz_path) == (64, 128)  # no fields: as before


def test_unsigned_disk_te_cache_warns_once(tmp_path, monkeypatch):
    """anima_train.py / anima_train_control_net_lllite.py hand the tokenizers over as objects (no qwen3_path): their
    caches cannot be signed, so say so once instead of silently skipping the check (review 2026-11-04)."""
    tokenize = AnimaTokenizeStrategy(qwen3_tokenizer=_FakeTokenizer(), t5_tokenizer=_FakeTokenizer(), qwen3_max_length=8, t5_max_length=8)
    monkeypatch.setattr(TokenizeStrategy, "_strategy", tokenize)
    warnings = []
    monkeypatch.setattr(strategy_anima.logger, "warning", lambda msg, *a, **k: warnings.append(msg))
    caching = AnimaTextEncoderOutputsCachingStrategy(True, 1, False)
    infos = [_te_info(tmp_path, "img1"), _te_info(tmp_path, "img2")]
    caching.cache_batch_outputs(tokenize, [_FakeQwen3()], AnimaTextEncodingStrategy(), infos)
    for info in infos:
        assert caching.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz)  # unchanged behaviour: not checked
    assert len(warnings) == 1 and "not signed" in warnings[0]

    # memory-only caching has no files to sign: no warning
    warnings.clear()
    AnimaTextEncoderOutputsCachingStrategy(False, 1, False).cache_batch_outputs(
        tokenize, [_FakeQwen3()], AnimaTextEncodingStrategy(), [_te_info(tmp_path, "img3")]
    )
    assert warnings == []
