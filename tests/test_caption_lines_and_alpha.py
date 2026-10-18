"""CPU tests: multi-line caption files are read whole (not cut to the first line) unless enable_wildcard picks a line, and
transparent images are composited onto white when loaded for training.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m pytest tests/test_caption_lines_and_alpha.py
"""
from dataclasses import asdict

import numpy as np
from PIL import Image

from library import train_util
from library.config_util import DreamBoothDatasetParams, DreamBoothSubsetParams
from library.image_utils import flatten_alpha
from library.moss import captions as moss_captions
from library.train_util import DreamBoothDataset, DreamBoothSubset

CAPTION = "@trigger ||| tag one, tag two ||| A long caption, first paragraph.\n\nSecond paragraph {a|b}.\n"
WHOLE = "@trigger ||| tag one, tag two ||| A long caption, first paragraph.\nSecond paragraph {a|b}."


def _dataset(tmp_path, **subset_kwargs):
    Image.new("RGB", (64, 64), (10, 20, 30)).save(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text(CAPTION, encoding="utf-8")
    subset = DreamBoothSubset(**asdict(DreamBoothSubsetParams(image_dir=str(tmp_path), caption_extension=".txt",
                                                              **subset_kwargs)))
    dataset = DreamBoothDataset(subsets=[subset], is_training_dataset=True,
                                **asdict(DreamBoothDatasetParams(resolution=(64, 64))))
    return dataset, subset, list(dataset.image_data.values())[0].caption


def test_multi_line_caption_is_used_whole(tmp_path):
    dataset, subset, caption = _dataset(tmp_path)
    assert caption == WHOLE  # blank lines dropped, every other line kept
    assert dataset.process_caption(subset, caption) == WHOLE  # no wildcard expansion, no first-line cut
    assert dataset.process_caption_canonical(subset, caption) == WHOLE
    assert moss_captions.canonical_caption(dataset, subset, caption) == WHOLE


def test_wildcard_picks_one_line(tmp_path):
    dataset, subset, caption = _dataset(tmp_path, enable_wildcard=True)
    assert caption == WHOLE
    lines = {"@trigger ||| tag one, tag two ||| A long caption, first paragraph.", "Second paragraph a.",
             "Second paragraph b."}
    assert {dataset.process_caption(subset, caption) for _ in range(200)} == lines
    first = "@trigger ||| tag one, tag two ||| A long caption, first paragraph."
    assert dataset.process_caption_canonical(subset, caption) == first
    assert moss_captions.canonical_caption(dataset, subset, caption) == first


def test_shuffled_multi_line_caption_keeps_its_paragraphs(tmp_path):
    dataset, subset, caption = _dataset(tmp_path, shuffle_caption=True, keep_tokens_separator="|||", caption_separator=",")
    expected_tail = "A long caption, first paragraph.\nSecond paragraph {a|b}."
    for _ in range(20):
        out = dataset.process_caption(subset, caption)
        assert out.startswith("@trigger, ") and out.endswith(expected_tail), out
    canonical = "@trigger, tag one, tag two, " + expected_tail
    assert moss_captions.canonical_caption(dataset, subset, caption) == canonical


def _rgba(alpha):
    pixels = np.zeros((4, 4, 4), np.uint8)
    pixels[..., :3] = (0, 0, 0)  # what transparent PNGs often store under alpha 0
    pixels[..., 3] = alpha
    return Image.fromarray(pixels, "RGBA")


def test_flatten_alpha_composites_onto_white():
    out = flatten_alpha(_rgba(0))
    assert out.mode == "RGB" and np.asarray(out).min() == 255
    half = np.asarray(flatten_alpha(_rgba(128)))
    assert np.all(half == 127)  # 255 * (1 - 128/255), rounded
    opaque = _rgba(255)
    assert flatten_alpha(opaque) is opaque  # unchanged: same pixels as before the fix
    rgb = Image.new("RGB", (4, 4))
    assert flatten_alpha(rgb) is rgb

    palette = Image.new("P", (4, 4), 0)
    palette.putpalette([0, 0, 0] * 256)
    palette.info["transparency"] = 0
    assert np.asarray(flatten_alpha(palette)).min() == 255

    icc = b"profile-bytes"
    tagged = _rgba(0)
    tagged.info["icc_profile"] = icc
    assert flatten_alpha(tagged).info["icc_profile"] == icc  # kept for to_srgb


def test_load_image_flattens_only_without_alpha_mask(tmp_path):
    path = str(tmp_path / "t.png")
    _rgba(0).save(path)
    assert np.asarray(train_util.load_image(path)).min() == 255
    with_mask = train_util.load_image(path, alpha=True)
    assert with_mask.shape == (4, 4, 4) and with_mask[..., 3].max() == 0  # alpha_mask training keeps the alpha
    opaque = str(tmp_path / "o.png")
    _rgba(255).save(opaque)
    assert np.asarray(train_util.load_image(opaque)).max() == 0  # opaque RGBA: unchanged
