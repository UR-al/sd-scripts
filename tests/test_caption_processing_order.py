"""CPU tests for the caption-pipeline fixes taken from upstream sd-scripts (PRs #2454 / #2455, 2026-10-18):

- caption_prefix / caption_suffix are added AFTER the enable_wildcard line choice (every line gets both; the canonical
  functions of train_util, MOSS and Anima-3.8B agree),
- Text Encoder caching encodes the processed caption for every model (strategy_base.caption_for_cache),
- BaseSubsetParams defaults match argparse, tags-only metadata keeps its tags, "\\n" in line prompts.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m pytest tests/test_caption_processing_order.py
"""
import types
from dataclasses import asdict

from PIL import Image

from library.anima38 import captions as anima38_captions
from library.config_util import DreamBoothDatasetParams, DreamBoothSubsetParams
from library.moss import captions as moss_captions
from library.strategy_base import (TextEncoderOutputsCachingStrategy, TextEncodingStrategy, TokenizeStrategy,
                                   caption_for_cache)
from library.train_util import DreamBoothDataset, DreamBoothSubset, build_finetuning_caption, line_to_prompt_dict

CAPTION = "first line {a|b}\nsecond line\n"
LINES = ["first line a", "first line b", "second line"]


def _dataset(tmp_path, **subset_kwargs):
    tmp_path.mkdir(exist_ok=True)
    Image.new("RGB", (64, 64), (10, 20, 30)).save(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text(CAPTION, encoding="utf-8")
    subset = DreamBoothSubset(**asdict(DreamBoothSubsetParams(image_dir=str(tmp_path), caption_extension=".txt",
                                                              **subset_kwargs)))
    dataset = DreamBoothDataset(subsets=[subset], is_training_dataset=True,
                                **asdict(DreamBoothDatasetParams(resolution=(64, 64))))
    return dataset, subset, list(dataset.image_data.values())[0].caption


def test_prefix_suffix_wrap_the_chosen_line_with_wildcard(tmp_path):
    dataset, subset, caption = _dataset(tmp_path, enable_wildcard=True, caption_prefix="PRE", caption_suffix="SUF")
    seen = {dataset.process_caption(subset, caption) for _ in range(300)}
    assert seen == {f"PRE {line} SUF" for line in LINES}, seen
    first = "PRE first line a SUF"
    assert dataset.process_caption_canonical(subset, caption) == first
    assert moss_captions.canonical_caption(dataset, subset, caption) == first
    assert anima38_captions.canonical_caption(dataset, subset, caption) == first


def test_prefix_suffix_wrap_the_whole_caption_without_wildcard(tmp_path):
    dataset, subset, caption = _dataset(tmp_path, caption_prefix="PRE", caption_suffix="SUF")
    whole = "PRE first line {a|b}\nsecond line SUF"
    assert dataset.process_caption(subset, caption) == whole
    assert dataset.process_caption_canonical(subset, caption) == whole
    assert moss_captions.canonical_caption(dataset, subset, caption) == whole
    assert anima38_captions.canonical_caption(dataset, subset, caption) == whole  # used to cut to the first line


def test_subset_params_defaults_match_argparse():
    params = DreamBoothSubsetParams(image_dir="x")
    assert params.caption_separator == "," and params.keep_tokens_separator == ""


def test_build_finetuning_caption():
    wildcard = types.SimpleNamespace(enable_wildcard=True, caption_separator=", ")
    assert build_finetuning_caption(wildcard, "", "t1, t2") == "t1, t2"  # tags-only metadata keeps its tags
    assert build_finetuning_caption(wildcard, "a\n\nb", "t") == "a, t\nb, t"
    assert build_finetuning_caption(wildcard, "a", None) == "a"
    assert build_finetuning_caption(wildcard, "a", "") == "a"  # an empty tags string adds no dangling separator
    plain = types.SimpleNamespace(enable_wildcard=False, caption_separator=", ")
    assert build_finetuning_caption(plain, "a", "t") == "a, t"
    assert build_finetuning_caption(plain, "", "t") == "t"
    assert build_finetuning_caption(plain, "a", "") == "a"


def test_backslash_n_in_line_prompts_becomes_a_line_break():
    d = line_to_prompt_dict(r"hello\nworld --n bad\nneg --w 512")
    assert d["prompt"] == "hello\nworld" and d["negative_prompt"] == "bad\nneg" and d["width"] == 512
    assert line_to_prompt_dict(r"star \(symbol\) --s 20")["prompt"] == r"star \(symbol\)"  # other escapes untouched


def test_caption_for_cache_prefers_variants_then_the_processed_caption():
    info = types.SimpleNamespace(caption="raw", caption_variants=None)
    assert caption_for_cache(info) == "raw"
    info.cache_caption = "processed"
    assert caption_for_cache(info) == "processed"
    info.cache_caption = ""
    assert caption_for_cache(info) == ""  # an empty processed caption is a value, not "unset"
    info.caption_variants = ["v0", "v1"]
    assert caption_for_cache(info) == "v0"


def test_anima38_cache_validity_covers_the_processed_caption(tmp_path):
    """A pre-patch *_anima38_te.npz (raw-caption digest only) is reused only when the processed caption equals the raw
    one; a changed prefix rebuilds it; files written now carry the processed digest."""
    import numpy as np

    from library.anima38 import captions as a38_captions
    from library.anima38 import strategy as a38_strategy

    caching = a38_strategy.Anima38TextEncoderOutputsCachingStrategy(True, 1, False, False, qwen35_signature="sig")
    image = str(tmp_path / "img.png")
    npz_path = caching.get_outputs_npz_path(image)
    arrays = {key: np.zeros(2, dtype=np.float32) for key in caching.OUTPUT_KEYS}
    arrays["caption_dropout_rate"] = np.array(0.0, dtype=np.float32)
    arrays[a38_strategy.SIGNATURE_KEY] = np.array("sig")
    arrays[a38_strategy.CAPTION_DIGEST_KEY] = np.array(a38_captions.caption_digest("1girl, smile"))
    np.savez(npz_path, **arrays)  # what every run before the processed-caption fix stored

    info = types.SimpleNamespace(absolute_path=image, image_key="k", caption="1girl, smile")
    subset = types.SimpleNamespace(caption_prefix=None)

    def dataset_with(prefix):
        subset.caption_prefix = prefix
        return types.SimpleNamespace(
            image_data={"k": info}, image_to_subset={"k": subset},
            process_caption_canonical=lambda s, caption: (s.caption_prefix + " " + caption) if s.caption_prefix else caption)

    a38_captions.register_caption_digests(dataset_with(None), caching)
    assert caching.is_disk_cached_outputs_expected(npz_path, 0, None)  # processed == raw: the old file is fine
    a38_captions.register_caption_digests(dataset_with("PRE"), caching)
    assert info.cache_caption == "PRE 1girl, smile"
    assert not caching.is_disk_cached_outputs_expected(npz_path, 0, None)  # encoded without the prefix: rebuild
    arrays[a38_strategy.CACHE_CAPTION_DIGEST_KEY] = np.array(a38_captions.caption_digest("PRE 1girl, smile"))
    np.savez(npz_path, **arrays)  # a file written by the patched strategy
    assert caching.is_disk_cached_outputs_expected(npz_path, 0, None)
    assert caching.is_disk_cached_outputs_expected(npz_path, 1, None)
    a38_captions.register_caption_digests(dataset_with("OTHER"), caching)
    assert not caching.is_disk_cached_outputs_expected(npz_path, 0, None)


class _Recorder(TextEncoderOutputsCachingStrategy):
    """Caching strategy that only records which caption text each image would be encoded with."""

    def __init__(self):
        super().__init__(False, 8, True, False)
        self.seen = []

    def get_outputs_npz_path(self, image_abs_path):
        return image_abs_path + "_te.npz"

    def is_disk_cached_outputs_expected(self, *args, **kwargs):
        return False

    def cache_batch_outputs(self, tokenize_strategy, models, text_encoding_strategy, batch):
        self.seen.extend(caption_for_cache(info) for info in batch)
        for info in batch:
            info.text_encoder_outputs = ("recorded",)


def _captions_sent_to_the_cache(dataset):
    classes = (TokenizeStrategy, TextEncodingStrategy, TextEncoderOutputsCachingStrategy)
    saved = [cls._strategy for cls in classes]
    recorder = _Recorder()
    for cls, value in zip(classes, (object(), object(), recorder)):
        cls._strategy = value
    try:
        dataset.new_cache_text_encoder_outputs([None], types.SimpleNamespace(num_processes=1, process_index=0))
    finally:
        for cls, value in zip(classes, saved):
            cls._strategy = value
    return recorder.seen


def test_te_cache_encodes_the_processed_caption(tmp_path):
    dataset, _, _ = _dataset(tmp_path / "plain", caption_prefix="PRE", caption_suffix="SUF")
    assert _captions_sent_to_the_cache(dataset) == ["PRE first line {a|b}\nsecond line SUF"]
    dataset, _, _ = _dataset(tmp_path / "wildcard", enable_wildcard=True, caption_prefix="PRE")
    assert _captions_sent_to_the_cache(dataset) == ["PRE first line a"]  # first line, first option, with the prefix
    info = list(dataset.image_data.values())[0]
    assert info.cache_caption == "PRE first line a" and info.caption == "first line {a|b}\nsecond line"
