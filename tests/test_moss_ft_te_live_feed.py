"""CPU tests of library/moss/te_live_feed.py (docs/moss_ft_te_live_fp16_plan.md 1.2, 1.6): `ahead` batches per
encoder pass in their order (the epoch's tail pass shorter), the batch form equal to the cached path's - the npz files
written by the real MossTextEncoderOutputsCachingStrategy, read back by load_outputs_npz and stacked the way
BaseDataset.__getitem__ stacks them - and moss_train.live_ahead's rounding to a multiple of GA.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_te_live_feed.py -q -p no:cacheprovider
"""
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
import torch.nn.functional as F

from _moss_te_tiny import live_encoder, reference_model, tokenize_strategy, write_tiny_te
from library import train_util
from library.moss import strategy, te_live_feed

CAPTIONS = ["a red square", "two blue circles on a white table", "", "green",
            "a cat, sitting, looking at the viewer, long shadows on the floor of an empty room"]


@pytest.fixture(scope="module")
def te_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("te_live_feed") / "te.safetensors"
    write_tiny_te(path)
    return path


class _CountingEncoder:
    """encode_rows -> one row [T, 2] per caption holding (pass number, T); counts the passes and their rows."""

    def __init__(self):
        self.passes = []

    def encode_rows(self, rows):
        self.passes.append(len(rows))
        return [torch.full((int(r.numel()), 2), float(len(self.passes)), dtype=torch.bfloat16) for r in rows]


def _batch(index, lengths):
    width = max(lengths)
    ids = torch.zeros(len(lengths), width, dtype=torch.long)
    mask = torch.zeros(len(lengths), width, dtype=torch.long)
    for b, n in enumerate(lengths):
        ids[b, :n] = torch.arange(1, n + 1) + 10 * index
        mask[b, :n] = 1
    return {"index": index, "input_ids_list": [ids, mask], "latents": torch.zeros(len(lengths), 4, 2, 2),
            "text_encoder_outputs_list": None}


def test_passes_group_ahead_batches_in_order():
    encoder = _CountingEncoder()
    feed = te_live_feed.TextEncodeAhead(encoder, ahead=3)
    batches = [_batch(i, [3 + i, 5]) for i in range(7)]
    out = list(feed.wrap(iter(batches)))
    assert [b["index"] for b in out] == list(range(7))
    assert encoder.passes == [6, 6, 2]  # 3 + 3 + 1 batches of 2 rows: the tail pass is shorter
    assert feed.pass_count == 3
    for b in out:
        embeds, mask, rates = b["text_encoder_outputs_list"]
        assert "input_ids_list" not in b
        assert embeds.shape == (2, max(3 + b["index"], 5), 2) and embeds.dtype == torch.bfloat16
        assert float(embeds[0, 0, 0]) == float(b["index"] // 3 + 1)  # encoded in its group's pass
        assert mask.dtype == torch.float32 and mask.sum(1).tolist() == [3 + b["index"], 5]
        assert rates.dtype == torch.float32 and rates.tolist() == [0.0, 0.0]
        # zero padding of the shorter row
        short = 0 if 3 + b["index"] < 5 else 1
        n = int(mask[short].sum())
        assert torch.all(embeds[short, n:] == 0)
    assert feed.take_pass_seconds() >= 0.0 and feed.take_pass_seconds() == 0.0


def test_a_pass_runs_only_when_the_queue_is_empty():
    encoder = _CountingEncoder()
    feed = te_live_feed.TextEncodeAhead(encoder, ahead=2)
    gen = feed.wrap(iter([_batch(i, [4]) for i in range(5)]))
    seen = []
    for batch in gen:
        seen.append((batch["index"], len(encoder.passes)))
    assert seen == [(0, 1), (1, 1), (2, 2), (3, 2), (4, 3)]
    assert list(feed.wrap(iter([]))) == [] and encoder.passes == [2, 2, 1]
    with pytest.raises(ValueError, match="ahead"):
        te_live_feed.TextEncodeAhead(encoder, ahead=0)


def test_batches_without_tokens_or_with_cached_outputs_are_refused():
    feed = te_live_feed.TextEncodeAhead(_CountingEncoder(), ahead=1)
    with pytest.raises(ValueError, match="input_ids_list"):
        list(feed.wrap(iter([{"input_ids_list": None}])))
    cached = _batch(0, [3])
    cached["text_encoder_outputs_list"] = [torch.zeros(1)]
    with pytest.raises(ValueError, match="already has"):
        list(feed.wrap(iter([cached])))


def _cached_batch_form(caching, infos):
    """BaseDataset.__getitem__'s cached path for one batch: load_outputs_npz per sample, convert_te_output_for_batch,
    and none_or_stack_elements' stacking (F.pad of the shorter rows, torch.stack of the 0-dim rates)."""
    per_sample = [caching.load_outputs_npz(info.text_encoder_outputs_npz) for info in infos]
    out = []
    for i in range(3):
        tensors = [train_util.convert_te_output_for_batch(sample[i]) for sample in per_sample]
        if tensors[0].ndim == 0:
            out.append(torch.stack(tensors))
            continue
        width = max(len(t) for t in tensors)
        if tensors[0].ndim == 1:
            out.append(torch.stack([F.pad(t, (0, width - t.shape[0])) for t in tensors]))
        else:
            out.append(torch.stack([F.pad(t, (0, 0, 0, width - t.shape[0])) for t in tensors]))
    return out


@pytest.mark.parametrize("group", [[0, 1], [2, 3, 4], [1]])
def test_live_batch_equals_the_real_cache_batch(te_file, tmp_path, monkeypatch, group):
    tok = tokenize_strategy(monkeypatch)
    encoding = strategy.MossTextEncodingStrategy()
    caching = strategy.MossTextEncoderOutputsCachingStrategy(True, None, False, signature="sig")
    infos = [types.SimpleNamespace(caption=CAPTIONS[i], cache_caption=CAPTIONS[i], caption_variants=None,
                                   caption_dropout_rate=0.0, absolute_path=str(tmp_path / f"img{i}.png"),
                                   text_encoder_outputs_npz=str(tmp_path / f"img{i}_moss_te.npz"))
             for i in group]
    caching.cache_batch_outputs(tok, [reference_model(te_file)], encoding, infos)
    cached = _cached_batch_form(caching, infos)

    encoder = live_encoder(te_file)
    try:
        tokens = [tok.tokenize(info.caption) for info in infos]  # __getitem__ tokenizes each caption alone
        width = max(t[0].shape[1] for t in tokens)
        ids = torch.stack([F.pad(t[0][0], (0, width - t[0].shape[1])) for t in tokens])
        mask = torch.stack([F.pad(t[1][0], (0, width - t[1].shape[1])) for t in tokens])
        batch = {"input_ids_list": [ids, mask], "latents": torch.zeros(len(group), 4, 2, 2)}
        out = list(te_live_feed.TextEncodeAhead(encoder, ahead=1).wrap(iter([batch])))[0]
    finally:
        encoder.close()
    live = out["text_encoder_outputs_list"]
    assert len(live) == len(cached) == 3
    for mine, theirs in zip(live, cached):
        assert mine.dtype == theirs.dtype and mine.shape == theirs.shape and torch.equal(mine, theirs)


def test_live_ahead_rounds_up_to_a_multiple_of_ga():
    import moss_train

    for ahead, ga, expected in ((32, 1, 32), (32, 3, 33), (5, 2, 6), (1, 4, 4), (8, 4, 8)):
        args = types.SimpleNamespace(moss_te_live_ahead=ahead, gradient_accumulation_steps=ga)
        assert moss_train.live_ahead(args) == expected, (ahead, ga)


def test_pass_time_is_per_encoded_caption():
    """Review fix: a pass encodes the captions of `ahead` batches and falls into one window; te/s_per_image divides the
    pass time by the captions it encoded (take_pass / s_per_image), not by one window's images."""
    import moss_train

    encoder = _CountingEncoder()
    feed = te_live_feed.TextEncodeAhead(encoder, ahead=3)
    gen = feed.wrap(iter([_batch(i, [4, 5]) for i in range(4)]))
    next(gen)
    seconds, rows = feed.take_pass()
    assert seconds >= 0.0 and rows == 6  # 3 batches of 2 captions
    next(gen)
    assert feed.take_pass() == (0.0, 0)  # the windows between passes
    list(gen)
    assert feed.take_pass()[1] == 2 and feed.rows_total == 8 and feed.pass_count == 2
    feed.pass_s_total = 4.0  # what the run spent in its passes
    assert feed.s_per_image == 0.5 and feed.device_seconds() == (0.0, 0)  # no CUDA events on the CPU
    meter = moss_train.WindowMeter(0.0, {})
    meter.te_s, meter.te_rows = 3.0, 6
    meter.add(torch.zeros((), dtype=torch.float64), 2, 16, 4)  # a window of 2 images
    stats = types.SimpleNamespace(grad_norm=1.0, clip_coef=1.0, skipped_params=0, tail_s=0.0, hook_wait_s=0.0,
                                  worker_busy_s=0.0)
    logs = moss_train.window_logs(meter, 0.5, stats, 1.0, feed)
    assert logs["te/pass_s"] == 3.0 and logs["te/pass_images"] == 6.0 and logs["te/s_per_image"] == 0.5
    assert moss_train.window_logs(meter, 0.5, stats, 1.0)["te/s_per_image"] == 0.5  # 3 s / 6 captions


def test_live_prefetch_factor_holds_a_pass():
    """Review fix: with workers, the DataLoader keeps a whole pass of batches ready (ceil(ahead / workers), at least
    torch's default 2)."""
    import moss_train

    assert moss_train.live_prefetch_factor(32, 3) == 11 and moss_train.live_prefetch_factor(32, 1) == 32
    assert moss_train.live_prefetch_factor(2, 4) == 2 and moss_train.live_prefetch_factor(8, 4) == 2
