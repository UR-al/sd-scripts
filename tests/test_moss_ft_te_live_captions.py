"""CPU tests of the caption features --moss_te_live makes possible (docs/moss_ft_te_live_fp16_plan.md 1.4): a real
DreamBoothDataset of tiny images built by moss_train.make_dataset_groups from a dataset TOML, with moss_train's live
strategies (no TE-outputs caching strategy, so __getitem__ runs kohya's process_caption and the MOSS tokenize strategy
at every fetch; workers 0).

* "HEAD ||| TAGS ||| SENTENCES" with keep_tokens_separator: HEAD stays first, SENTENCES last, TAGS permute;
* keep_tokens on a plain tag list;
* a multi-line caption file with enable_wildcard: every line is drawn;
* caption_dropout_rate 1.0: the tokens of the '' prompt (the official unconditional prompt, as the cached variants);
* caption_tag_dropout_rate keeps the protected tags; token warmup;
* the `|||` warnings and the start-up summary; the dataset pickles without any encoder.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_te_live_captions.py -q -p no:cacheprovider
"""
import os
import pickle

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
from PIL import Image

import moss_train
from _moss_te_tiny import CharTokenizer, decode
from library import strategy_base, train_util
from library.moss import captions, ft_args, strategy
from library.moss.qwen35 import format_moss_prompt

HEAD = "1girl, alice, wonderland"
TAGS = ["red dress", "blue eyes", "long hair", "smile", "outdoors", "holding cup"]
SENTENCE = "She stands in a garden at noon."
STRUCTURED = f"{HEAD} ||| {', '.join(TAGS)} ||| {SENTENCE}"


def _reset_strategies():
    for cls in (strategy_base.TokenizeStrategy, strategy_base.TextEncodingStrategy,
                strategy_base.TextEncoderOutputsCachingStrategy, strategy_base.LatentsCachingStrategy):
        cls._strategy = None


@pytest.fixture
def make_group(tmp_path, monkeypatch):
    """make_group(captions, subset_lines) -> (dataset, args): one subset of 32x32 images with these caption files."""
    monkeypatch.setattr(strategy, "MossTokenizer", CharTokenizer)
    _reset_strategies()
    made = []

    def make(caption_texts, subset_lines=(), name="images"):
        images = tmp_path / name
        images.mkdir()
        rng = np.random.default_rng(len(made))
        for index, text in enumerate(caption_texts):
            Image.fromarray((rng.random((32, 32, 3)) * 255).astype(np.uint8)).save(images / f"img{index}.png")
            (images / f"img{index}.txt").write_text(text, encoding="utf-8")
        toml = tmp_path / f"{name}.toml"
        toml.write_text("[general]\nenable_bucket = false\ncaption_extension = \".txt\"\n\n[[datasets]]\nresolution = 32\n"
                        f"batch_size = 1\n\n  [[datasets.subsets]]\n  image_dir = '{images.as_posix()}'\n"
                        "  num_repeats = 1\n" + "".join(f"  {line}\n" for line in subset_lines), encoding="utf-8")
        args = moss_train.setup_parser().parse_args(["--dataset_config", str(toml), "--moss_te_live",
                                                     "--moss_text_encoder", "unused.safetensors", "--seed", "1"])
        train_util.prepare_dataset_args(args, True)  # as moss_train.train's step 1
        _reset_strategies()
        moss_train.set_moss_strategies(args)
        assert strategy_base.TextEncoderOutputsCachingStrategy.get_strategy() is None  # live: no TE cache strategy
        group, _ = moss_train.make_dataset_groups(args)
        moss_train.configure_dataset_group(args, group)
        group.set_current_strategies()
        group.set_max_train_steps(100)
        made.append(group)
        return group.datasets[0], args

    yield make
    _reset_strategies()


def _fetch(dataset, index=0):
    example = dataset[index]
    assert example["text_encoder_outputs_list"] is None  # nothing cached: the feed encodes
    ids, mask = example["input_ids_list"]
    length = int(mask[0].sum())
    caption = example["captions"][0]
    assert decode(ids[0, :length]) == format_moss_prompt(caption)  # the tokens are the processed caption's
    return caption


def test_head_tags_sentences(make_group):
    dataset, _ = make_group([STRUCTURED], ['keep_tokens_separator = "|||"', "shuffle_caption = true"])
    orders = set()
    for _ in range(40):
        caption = _fetch(dataset)
        assert "|||" not in caption
        assert caption.startswith(HEAD + ", ") and caption.endswith(", " + SENTENCE)
        middle = caption[len(HEAD) + 2: -len(SENTENCE) - 2].split(", ")
        assert sorted(middle) == sorted(TAGS)
        orders.add(tuple(middle))
    assert len(orders) > 5  # shuffled anew at every fetch


def test_keep_tokens_on_a_plain_tag_list(make_group):
    dataset, _ = make_group(["alice, 1girl, " + ", ".join(TAGS)], ["keep_tokens = 2", "shuffle_caption = true"])
    orders = set()
    for _ in range(30):
        parts = _fetch(dataset).split(", ")
        assert parts[:2] == ["alice", "1girl"] and sorted(parts[2:]) == sorted(TAGS)
        orders.add(tuple(parts[2:]))
    assert len(orders) > 3


def test_multi_line_captions_are_variants_with_enable_wildcard(make_group):
    lines = [STRUCTURED, f"{HEAD} ||| red dress, smile ||| {SENTENCE}", f"{HEAD} ||| smile"]
    dataset, _ = make_group(["\n".join(lines)], ['keep_tokens_separator = "|||"', "enable_wildcard = true",
                                                  "shuffle_caption = true"])
    seen = set()
    for _ in range(60):
        caption = _fetch(dataset)
        assert caption.startswith(HEAD) and "\n" not in caption and "|||" not in caption
        seen.add(len(caption.split(", ")))
    assert seen == {3 + len(TAGS) + 1, 3 + 2 + 1, 3 + 1}  # every line drawn (long / medium / short)


def test_caption_dropout_gives_the_unconditional_prompt(make_group):
    dataset, _ = make_group([STRUCTURED], ['keep_tokens_separator = "|||"', "shuffle_caption = true",
                                           "caption_dropout_rate = 1.0"])
    ids, mask = dataset[0]["input_ids_list"]
    empty = strategy_base.TokenizeStrategy.get_strategy().tokenize("")
    assert dataset[0]["captions"] == [""]
    assert ids[0, : int(mask[0].sum())].tolist() == empty[0][0].tolist()  # the '' prompt's chat template


def test_tag_dropout_keeps_protected_tags(make_group, tmp_path):
    dataset, _ = make_group([STRUCTURED], ['keep_tokens_separator = "|||"', "caption_tag_dropout_rate = 1.0"])
    protected = tmp_path / "protected.txt"
    protected.write_text("Blue Eyes\nsmile\n", encoding="utf-8")
    dataset.protected_tags_file = str(protected)
    for _ in range(5):
        assert _fetch(dataset) == f"{HEAD}, blue eyes, smile, {SENTENCE}"  # only the protected tags of TAGS stay


def test_token_warmup(make_group):
    dataset, _ = make_group([STRUCTURED], ['keep_tokens_separator = "|||"', "token_warmup_step = 10",
                                           "token_warmup_min = 1"])
    dataset.set_current_step(0)
    assert _fetch(dataset) == f"{HEAD}, {TAGS[0]}, {SENTENCE}"
    dataset.set_current_step(10)
    assert _fetch(dataset) == f"{HEAD}, {', '.join(TAGS)}, {SENTENCE}"


def test_separator_warnings_and_summary(make_group):
    dataset, args = make_group([STRUCTURED, "plain, tags"], ["shuffle_caption = true"])
    warnings = captions.separator_warnings(dataset)
    assert len(warnings) == 1 and "'|||'" in warnings[0] and "not set" in warnings[0] and "1 caption(s)" in warnings[0]
    summary = captions.live_caption_summary(dataset)
    assert len(summary) == 1 and "2 images" in summary[0] and "shuffle_caption True" in summary[0]


def test_separator_without_a_split_warns(make_group):
    dataset, _ = make_group([STRUCTURED], ['keep_tokens_separator = "|||"'])  # no shuffle / tag dropout / warmup
    warnings = captions.separator_warnings(dataset)
    assert len(warnings) == 1 and "splits there only with" in warnings[0]
    assert "|||" in _fetch(dataset)  # kohya leaves it literal: what the warning says


def test_check_captions_logs_in_live_mode(make_group, monkeypatch):
    dataset, args = make_group([STRUCTURED], ['keep_tokens_separator = "|||"', "shuffle_caption = true",
                                              "caption_dropout_rate = 0.05", "caption_tag_dropout_rate = 0.1"])
    lines = {"info": [], "warning": []}
    monkeypatch.setattr(ft_args.logger, "info", lambda m, *a, **k: lines["info"].append(str(m)))
    monkeypatch.setattr(ft_args.logger, "warning", lambda m, *a, **k: lines["warning"].append(str(m)))
    group = type("Group", (), {"datasets": [dataset]})()
    ft_args.check_captions(args, group)  # live: shuffle and both dropouts are fine
    assert not lines["warning"]
    assert any("live captions:" in line and "caption_dropout_rate 0.05" in line and "caption_tag_dropout_rate 0.1"
               in line for line in lines["info"])


def test_the_dataset_pickles_without_an_encoder(make_group):
    dataset, _ = make_group([STRUCTURED], ['keep_tokens_separator = "|||"', "shuffle_caption = true"])
    _fetch(dataset)
    blob = pickle.dumps(dataset)  # what a spawned DataLoader worker receives
    assert b"HostTextEncoderWeights" not in blob and b"LiveTextEncoder" not in blob
    assert len(blob) < 8 << 20


def test_multi_line_without_wildcard_warns(make_group):
    """Review fix: a multi-line caption file with enable_wildcard off is encoded whole (every variant joined) - a
    warning, not an info line."""
    lines = [STRUCTURED, f"{HEAD} ||| smile"]
    dataset, _ = make_group(["\n".join(lines), "plain, tags"], ['keep_tokens_separator = "|||"', "shuffle_caption = true"])
    warnings = captions.separator_warnings(dataset)
    assert len(warnings) == 1 and "1 multi-line caption file(s)" in warnings[0] and "enable_wildcard" in warnings[0]


def test_multi_line_with_wildcard_does_not_warn(make_group):
    lines = [STRUCTURED, f"{HEAD} ||| smile"]
    dataset, _ = make_group(["\n".join(lines)], ['keep_tokens_separator = "|||"', "shuffle_caption = true",
                                                  "enable_wildcard = true"])
    assert captions.separator_warnings(dataset) == []


def test_braces_with_wildcard_warn(make_group):
    dataset, _ = make_group([f"{HEAD} ||| {{red|blue}} dress ||| {SENTENCE}"],
                            ['keep_tokens_separator = "|||"', "shuffle_caption = true", "enable_wildcard = true"])
    warnings = captions.separator_warnings(dataset)
    assert len(warnings) == 1 and "{...}" in warnings[0]
    assert "{" not in _fetch(dataset)  # what the warning says: process_caption picked an option
