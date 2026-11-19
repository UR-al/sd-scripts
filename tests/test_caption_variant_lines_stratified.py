"""CPU tests: with enable_wildcard and cached caption variants (--cache_caption_variants K), the K variants of a multi-line
caption take the lines in proportion (stratified) instead of K independent random line picks, so a caption file that
repeats lines to set weights (e.g. 3 tag lines, 3 tag+sentence lines, 1 short, 1 medium, 1 long) gets exactly those
shares in every image's cache. Variant 0 stays the canonical caption (first line).

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -m pytest tests/test_caption_variant_lines_stratified.py
"""
import collections
import random
from dataclasses import asdict

from PIL import Image

from library.config_util import DreamBoothDatasetParams, DreamBoothSubsetParams
from library.train_util import DreamBoothDataset, DreamBoothSubset

HEAD = "@trigger, 1boy, name"
TAG = HEAD + " ||| blue hair, eagle, solo"
FULL = HEAD + " ||| blue hair, eagle, solo ||| A man sits. An eagle perches."
SHORT = HEAD + " ||| ||| A man sits with an eagle."
MEDIUM = HEAD + " ||| ||| A man with blue hair sits with an eagle on his shoulder."
LONG = HEAD + " ||| ||| A man with short blue hair sits under an arch while a brown eagle perches on his shoulder at night."
NINE = [FULL, TAG, FULL, SHORT, TAG, MEDIUM, FULL, LONG, TAG]  # 1/3 tag+sentence, 1/3 tag, 1/9 each sentence length


def _dataset(tmp_path, lines, variants, **subset_kwargs):
    Image.new("RGB", (64, 64), (10, 20, 30)).save(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    kwargs = dict(image_dir=str(tmp_path), caption_extension=".txt", keep_tokens_separator="|||", shuffle_caption=True,
                  enable_wildcard=True)
    kwargs.update(subset_kwargs)
    subset = DreamBoothSubset(**asdict(DreamBoothSubsetParams(**kwargs)))
    dataset = DreamBoothDataset(subsets=[subset], is_training_dataset=True,
                                **asdict(DreamBoothDatasetParams(resolution=(64, 64))))
    dataset.set_aug_variant_config(0, variants)
    dataset.build_caption_variants()
    return dataset, subset, list(dataset.image_data.values())[0]


def _kind(variant: str) -> str:
    """Which caption line a processed variant came from (tags are shuffled, so classify by content)."""
    if "A man sits. An eagle perches." in variant:
        return "full"
    for name, line in (("short", SHORT), ("medium", MEDIUM), ("long", LONG)):
        if line.split("||| ||| ")[1] in variant:
            return name
    assert "eagle" in variant and "A man" not in variant, variant
    return "tag"


def test_variants_follow_line_proportions_exactly(tmp_path):
    for seed in range(5):
        random.seed(seed)
        dataset, subset, info = _dataset(tmp_path, NINE, 18)
        counts = collections.Counter(_kind(v) for v in info.caption_variants)
        assert len(info.caption_variants) == 18
        assert counts == {"full": 6, "tag": 6, "short": 2, "medium": 2, "long": 2}, (seed, counts)


def test_variant_zero_is_the_canonical_first_line(tmp_path):
    random.seed(1)
    dataset, subset, info = _dataset(tmp_path, NINE, 18)
    assert info.caption_variants[0] == dataset.process_caption_canonical(subset, info.caption)
    assert _kind(info.caption_variants[0]) == "full"


def test_count_not_a_multiple_of_lines_stays_within_one(tmp_path):
    for seed in range(5):
        random.seed(seed)
        dataset, subset, info = _dataset(tmp_path, NINE, 16)
        per_line = collections.Counter(_kind(v) for v in info.caption_variants)
        assert sum(per_line.values()) == 16
        # 16 = 9 + 7: every line at least once and at most twice -> tag lines 3..6, sentence lengths 1..2 each
        assert 3 <= per_line["full"] <= 6 and 3 <= per_line["tag"] <= 6, (seed, per_line)
        assert all(1 <= per_line[k] <= 2 for k in ("short", "medium", "long")), (seed, per_line)


def test_fewer_variants_than_lines_never_repeat_a_line_slot(tmp_path):
    random.seed(3)
    lines = [FULL, TAG, SHORT, MEDIUM, LONG]
    dataset, subset, info = _dataset(tmp_path, lines, 4)
    kinds = [_kind(v) for v in info.caption_variants]
    assert kinds[0] == "full" and len(set(kinds)) == 4, kinds


def test_tags_are_still_shuffled_per_variant(tmp_path):
    random.seed(0)
    dataset, subset, info = _dataset(tmp_path, NINE, 18)
    tag_variants = {v for v in info.caption_variants[1:] if _kind(v) == "tag"}
    assert len(tag_variants) > 1  # different tag orders, not one string repeated


def test_wildcard_options_inside_a_line_still_expand(tmp_path):
    random.seed(0)
    lines = [HEAD + " ||| ||| A {red|green} bird.", HEAD + " ||| blue hair, solo"]
    dataset, subset, info = _dataset(tmp_path, lines, 40)
    sentences = {v for v in info.caption_variants[1:] if "bird" in v}
    assert any("red" in s for s in sentences) and any("green" in s for s in sentences), sentences


def test_single_line_caption_unchanged(tmp_path):
    random.seed(0)
    dataset, subset, info = _dataset(tmp_path, [FULL], 8)
    assert len(info.caption_variants) == 8
    assert all(_kind(v) == "full" for v in info.caption_variants)


def test_without_wildcard_the_whole_file_is_one_caption(tmp_path):
    random.seed(0)
    dataset, subset, info = _dataset(tmp_path, NINE, 9, enable_wildcard=False)
    # no line picking: every variant keeps every line's text (joined as one caption)
    assert all("A man with short blue hair" in v and "A man sits with an eagle." in v for v in info.caption_variants)
