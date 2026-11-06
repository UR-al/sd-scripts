"""A token_warmup_step ratio becomes steps without losing a step to float noise (review 11-05 follow-up).

token_warmup_step N < 1 means N * max_train_steps. process_caption resolved it with math.floor(N * max_train_steps),
so 0.29 of 100 steps = 28.999999999999996 gave 28. It now uses train_util.fraction_to_steps, the helper the lr warmup
and decay ratios use. Integer steps (>= 1) and 0 are unchanged. Drives the real DreamBoothDataset.process_caption.
CPU only.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m pytest tests/test_token_warmup_fraction.py
"""
import os
import sys
from dataclasses import asdict

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library import train_util  # noqa: E402
from library.config_util import DreamBoothDatasetParams, DreamBoothSubsetParams  # noqa: E402
from library.train_util import DreamBoothDataset, DreamBoothSubset  # noqa: E402

TAGS = [f"tag{i}" for i in range(10)]


def test_library_is_this_checkout():
    # sys.path.insert above must win over any other sd-scripts checkout on the path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.dirname(os.path.abspath(train_util.__file__)) == os.path.join(repo_root, "library")


def _dataset(tmp_path, max_train_steps, **subset_kwargs):
    Image.new("RGB", (64, 64), (10, 20, 30)).save(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text(", ".join(TAGS), encoding="utf-8")
    subset = DreamBoothSubset(**asdict(DreamBoothSubsetParams(image_dir=str(tmp_path), caption_extension=".txt",
                                                              **subset_kwargs)))
    dataset = DreamBoothDataset(subsets=[subset], is_training_dataset=True,
                                **asdict(DreamBoothDatasetParams(resolution=(64, 64))))
    dataset.set_max_train_steps(max_train_steps)
    return dataset, subset, list(dataset.image_data.values())[0].caption


def _tags_at(dataset, subset, caption, step):
    dataset.set_current_step(step)
    return dataset.process_caption(subset, caption).split(", ")


def test_ratio_029_of_100_steps_is_29_steps(tmp_path):
    dataset, subset, caption = _dataset(tmp_path, 100, token_warmup_step=0.29, token_warmup_min=1)
    # step 28 is still inside a 29-step warmup: floor(28 * (10 - 1) / 29) + 1 = 9 tags (a 28-step warmup gives all 10)
    assert _tags_at(dataset, subset, caption, 28) == TAGS[:9]
    assert subset.token_warmup_step == 29 and isinstance(subset.token_warmup_step, int)
    assert _tags_at(dataset, subset, caption, 29) == TAGS


def test_integer_steps_are_unchanged(tmp_path):
    dataset, subset, caption = _dataset(tmp_path, 100, token_warmup_step=29, token_warmup_min=1)
    assert _tags_at(dataset, subset, caption, 28) == TAGS[:9]
    assert subset.token_warmup_step == 29


def test_genuine_half_step_keeps_the_floor(tmp_path):
    # 0.115 of 100 steps = 11.5 -> 11, as math.floor gave before
    dataset, subset, caption = _dataset(tmp_path, 100, token_warmup_step=0.115, token_warmup_min=1)
    _tags_at(dataset, subset, caption, 0)
    assert subset.token_warmup_step == 11


def test_zero_stays_off(tmp_path):
    # shuffle_caption enters the tag branch with token_warmup_step 0; warmup stays off and every tag is kept
    dataset, subset, caption = _dataset(tmp_path, 100, token_warmup_step=0, shuffle_caption=True)
    assert sorted(_tags_at(dataset, subset, caption, 0)) == sorted(TAGS)
    assert subset.token_warmup_step == 0
