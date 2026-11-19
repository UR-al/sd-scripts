"""CPU tests of the opt-in batched layer calls of library/moss/te_live.py (--moss_te_live_batch N > 1): the grouping
(sorted by length, ceil(n / N) balanced groups, every row once, order restored; captions past
ft_budget.DEFAULT_TEXT_TOKENS group among themselves within the token budget of a call, the reviewer's 11-18 finding),
the padding (each row right-padded with its own last token id), and the proof points of te_live.py's header on the
tiny config:
* rotary: the group length's table starts with every row's own table, bit for bit;
* the pad content never reaches a real token: other pad ids in the same group shapes give the same rows bit for bit,
  and every pad activation stays finite (while a non-finite pad WOULD reach the real tokens through 0 x nan - why the
  pad must be a real token);
* fp32: the batched rows match the per-row rows to ~1e-5 (lengths 1, 63, 64, 65, 130 around the 64-token chunks of the
  gated delta rule, mixed lengths in one group, groups smaller than the batch size, a single row); a group of one
  unpadded row is bit-identical; bf16 stays close;
* the streaming machinery (copy / mmap hosts, 2 / 4 slots, the threaded stager) runs the group tensors unchanged;
* batch_rows validation, the pass stats, the strategy branch and te_live_feed's one-time group log line.
The default path (batch 1) is bit for bit the cached path: tests/test_moss_ft_te_live_engine.py.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_te_live_batch.py -q -p no:cacheprovider
"""
import math
import os
import random

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
import torch.nn.functional as F

from _moss_te_tiny import TINY_TE, random_ids, reference_model, tokenize_strategy, write_tiny_te
from library.moss import ft_budget, strategy, te_live, te_live_feed, te_live_weights

EDGE_LENGTHS = (1, 63, 64, 65, 130)  # around the 64-token chunks of the gated delta rule
MIXED_LENGTHS = (130, 5, 64, 1, 65, 37, 63, 2, 129)
LONG_LENGTHS = (420, 5, 350, 20, 310, 64, 333)  # four past the 300 tokens of the estimate: their own group at N = 8
# the token counts of the smoke subset's first 32 captions (te_batch_bench.py --cpu --tiny on the config's dataset and
# the real tokenizer, 11-18): min 10 / median 191 / max 770 / total 7,341, 8 of them past 300 tokens
SMOKE_LENGTHS = (770, 92, 135, 137, 173, 362, 191, 10, 203, 51, 268, 251, 183, 323, 352, 152, 209, 145, 361, 167, 122,
                 583, 10, 152, 284, 137, 151, 434, 384, 197, 200, 152)
FP32_TOLERANCE = 5e-5  # max abs of the batched vs the per-row rows in fp32 (observed 1.2e-5 .. 1.7e-5, |row| <= ~5.6)
BF16_MIN_COSINE = 0.999  # bf16: a rounding flip early in a layer travels (observed >= 0.9996 per row, tiny random model)


@pytest.fixture(scope="module")
def te_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("te_live_batch") / "te.safetensors"
    write_tiny_te(path)
    return path


def make_encoder(path, dtype=torch.bfloat16, batch_rows=1, host="copy", ring_slots=2, staging=None):
    weights = te_live_weights.HostTextEncoderWeights.open(str(path), config=TINY_TE, host=host, dtype=dtype,
                                                          check=False)
    return te_live.LiveTextEncoder(weights, "cpu", ring_slots=ring_slots, staging_threads=2, staging=staging,
                                   batch_rows=batch_rows)


def rows_of(lengths, seed=300):
    return [random_ids(n, seed=seed + 7 * i + n) for i, n in enumerate(lengths)]


def max_abs(rows_a, rows_b) -> float:
    assert len(rows_a) == len(rows_b)
    for a, b in zip(rows_a, rows_b):
        assert a.shape == b.shape and a.dtype == b.dtype
    return max(float((a.float() - b.float()).abs().max()) for a, b in zip(rows_a, rows_b))


def min_cosine(rows_a, rows_b) -> float:
    return min(float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))
               for a, b in zip(rows_a, rows_b))


def encode(path, rows, batch_rows, dtype=torch.bfloat16, **kw):
    encoder = make_encoder(path, dtype, batch_rows, **kw)
    try:
        out = encoder.encode_rows(rows)
        return out, list(encoder.last_groups), dict(encoder.stats)
    finally:
        encoder.close()


# ---- grouping and padding ------------------------------------------------------------------------------------------
def test_group_sizes_are_balanced_and_capped():
    sizes = ft_budget.te_group_sizes
    assert sizes(32, 8) == [8, 8, 8, 8]
    assert sizes(33, 32) == [17, 16]  # not 32 + a lone straggler
    assert sizes(5, 4) == [3, 2]  # groups smaller than the batch size
    assert sizes(1, 8) == [1] and sizes(0, 8) == [] and sizes(3, 100) == [3]
    assert sizes(7, 1) == [1] * 7  # batch 1: one call per row (today's path)
    for rows in range(1, 70):
        for batch in range(1, 40):
            got = sizes(rows, batch)
            assert sum(got) == rows and max(got) <= batch and len(got) == math.ceil(rows / batch), (rows, batch)
            assert max(got) - min(got) <= 1 and got == sorted(got, reverse=True)
    assert ft_budget.te_group_text(32, 8) == "4 x 8" and ft_budget.te_group_text(33, 32) == "17 + 16"


def test_length_groups_sort_by_length_and_cover_every_row():
    lengths = [5, 200, 63, 64, 1, 130, 64]
    groups = te_live.length_groups(lengths, 3)
    # longest first, equal lengths in their order (64 at 3 before 64 at 6); sizes [3, 2, 2]
    assert groups == [[1, 5, 3], [6, 2], [0, 4]]
    assert te_live.length_groups(lengths, 1) == [[1], [5], [3], [6], [2], [0], [4]]
    assert te_live.length_groups(lengths, 100) == [[1, 5, 3, 6, 2, 0, 4]]
    assert te_live.length_groups([], 4) == []
    for batch in (2, 4, 5):
        flat = [i for group in te_live.length_groups(lengths, batch) for i in group]
        assert sorted(flat) == list(range(len(lengths)))
    for bad in (0, -1, True):
        with pytest.raises(ValueError, match="layer batch"):
            te_live.length_groups(lengths, bad)
    for bad in (0, -5, True):
        with pytest.raises(ValueError, match="text tokens"):
            te_live.length_groups(lengths, 3, bad)


def _shapes(lengths, groups):
    return [(len(group), max(lengths[i] for i in group)) for group in groups]


def test_long_captions_never_pad_a_large_group():
    """Reviewer 11-18: grouped by row count alone, the smoke subset's 32 captions became ONE call of 32 rows x 770
    tokens at N = 32 (24,640 padded tokens for 7,341 real ones; about twice the pass peak the estimate charges, which
    takes every caption at 300 tokens) and a first call of 8 x 770 at N = 8. Captions past DEFAULT_TEXT_TOKENS now
    group among themselves, each call at most min(N, n) x 300 padded tokens; the rest group exactly as before."""
    T = ft_budget.DEFAULT_TEXT_TOKENS
    assert _shapes(SMOKE_LENGTHS, te_live.length_groups(SMOKE_LENGTHS, 8)) == [
        (3, 770), (5, 384), (8, 284), (8, 183), (8, 137)]
    assert _shapes(SMOKE_LENGTHS, te_live.length_groups(SMOKE_LENGTHS, 16)) == [
        (6, 770), (2, 352), (12, 284), (12, 152)]
    assert _shapes(SMOKE_LENGTHS, te_live.length_groups(SMOKE_LENGTHS, 32)) == [(8, 770), (24, 284)]
    for batch, padded in ((8, 9062), (16, 10556), (32, 12976)):  # was 10,992 / 15,248 / 24,640
        shapes = _shapes(SMOKE_LENGTHS, te_live.length_groups(SMOKE_LENGTHS, batch))
        assert sum(rows * width for rows, width in shapes) == padded
        assert all(rows * width <= batch * T for rows, width in shapes), (batch, shapes)
    # a caption past the budget of a call runs alone (as on the per-row path); the explicit text_tokens moves the line
    assert te_live.length_groups([2500, 2500, 10], 8) == [[0], [1], [2]]
    assert te_live.length_groups([2500, 2500, 10], 8, 2000) == [[0, 1], [2]]  # 2 x 2500 <= 3 x 2000; the short alone


def test_every_call_stays_within_its_token_budget():
    """Any lengths, any N: each call holds at most min(N, n) rows and min(N, n) x 300 padded tokens (or one caption
    alone), a long caption never pads short ones, the groups are consecutive runs of the length order, and when no
    caption is past 300 tokens the groups are te_group_sizes(n, N) as before."""
    T = ft_budget.DEFAULT_TEXT_TOKENS
    rng = random.Random(11)
    for _ in range(600):
        n, batch = rng.randint(1, 48), rng.randint(1, 40)
        long_share = rng.choice((0.0, 0.1, 0.5, 1.0))
        lengths = [rng.randint(T + 1, 4096) if rng.random() < long_share else rng.randint(1, T) for _ in range(n)]
        groups = te_live.length_groups(lengths, batch)
        assert [i for group in groups for i in group] == sorted(range(n), key=lambda i: (-lengths[i], i))
        rows_cap = min(batch, n)
        for rows, width in _shapes(lengths, groups):
            assert 1 <= rows <= rows_cap and (rows == 1 or rows * width <= rows_cap * T), (n, batch, lengths, groups)
        for group in groups:
            assert all(lengths[i] > T for i in group) or all(lengths[i] <= T for i in group), (lengths, group)
        if max(lengths) <= T:
            assert [len(group) for group in groups] == ft_budget.te_group_sizes(n, batch)


def test_pad_group_repeats_each_rows_last_token():
    rows = [torch.tensor([1, 2, 3]), torch.tensor([7]), torch.tensor([[4, 5]])]
    assert te_live.pad_group(rows).tolist() == [[1, 2, 3], [7, 7, 7], [4, 5, 5]]
    assert te_live.pad_group(rows).dtype == torch.long
    with pytest.raises(ValueError, match="chat-template"):
        te_live.pad_group([torch.tensor([1]), torch.zeros(0, dtype=torch.long)])


def test_rotary_prefix_equals_the_rows_own_table(te_file):
    encoder = make_encoder(te_file)
    try:
        for dtype in (torch.bfloat16, torch.float32):
            cos, sin = encoder.shell._rotary(130, torch.device("cpu"), dtype)
            assert cos.shape == (1, 130, TINY_TE.head_dim // 4)
            for length in (1, 2, 63, 64, 65, 129, 130):
                own_cos, own_sin = encoder.shell._rotary(length, torch.device("cpu"), dtype)
                assert torch.equal(cos[:, :length], own_cos) and torch.equal(sin[:, :length], own_sin), length
    finally:
        encoder.close()


# ---- numerics ------------------------------------------------------------------------------------------------------
FP32_CASES = [
    ("edge_lengths_one_group", EDGE_LENGTHS, 8),  # 1, 63, 64, 65, 130 padded together to 130
    ("edge_lengths_groups_of_2", EDGE_LENGTHS, 2),  # [2, 2, 1]
    ("groups_smaller_than_batch", EDGE_LENGTHS, 4),  # [3, 2]
    ("mixed_lengths", MIXED_LENGTHS, 3),
    ("mixed_lengths_one_group", MIXED_LENGTHS, 32),
    ("long_captions_own_group", LONG_LENGTHS, 8),  # [(4, 420), (3, 64)], not one group of 7 x 420
]


@pytest.mark.parametrize("name,lengths,batch", FP32_CASES, ids=[case[0] for case in FP32_CASES])
def test_fp32_batched_rows_match_the_per_row_rows(te_file, name, lengths, batch, record_property):
    rows = rows_of(lengths)
    per_row, _, _ = encode(te_file, rows, 1, torch.float32)
    batched, groups, stats = encode(te_file, rows, batch, torch.float32)
    assert [int(h.shape[0]) for h in batched] == list(lengths)  # every row its own length, in the caller's order
    assert groups == _shapes(lengths, te_live.length_groups(lengths, batch))
    if max(lengths) <= ft_budget.DEFAULT_TEXT_TOKENS:
        assert [count for count, _ in groups] == ft_budget.te_group_sizes(len(rows), batch)
    else:
        assert groups == [(4, 420), (3, 64)]
    assert stats["calls"] == len(groups) and stats["rows"] == len(rows)
    diff = max_abs(batched, per_row)
    record_property("fp32_max_abs", diff)
    print(f"\n{name}: fp32 max abs batched vs per row {diff:.3g} (groups {groups})")
    assert diff < FP32_TOLERANCE, (name, diff)
    assert min_cosine(batched, per_row) > 1.0 - 1e-6


def test_a_single_row_and_a_group_of_one_are_bit_identical(te_file):
    reference = reference_model(te_file)
    for length in EDGE_LENGTHS:
        ids = random_ids(length, seed=40 + length)
        out, groups, _ = encode(te_file, [ids], 8)
        assert groups == [(1, length)]
        assert torch.equal(out[0], reference(ids[None])[0]), length  # the per-row shapes: the cached row bit for bit


def test_bf16_batched_rows_stay_close(te_file):
    rows = rows_of(MIXED_LENGTHS)
    per_row, _, _ = encode(te_file, rows, 1)
    batched, _, _ = encode(te_file, rows, 4)
    diff, cosine = max_abs(batched, per_row), min_cosine(batched, per_row)
    print(f"\nbf16 max abs batched vs per row {diff:.3g}, min row cosine {cosine:.6f}")
    assert cosine >= BF16_MIN_COSINE and all(torch.isfinite(h.float()).all() for h in batched)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_pad_content_never_reaches_a_real_token(te_file, dtype, monkeypatch):
    """Same group shapes, other pad ids (random tokens instead of the row's last one): every real row is the same bit
    for bit, so a batched row differs from its per-row encoding only by the kernels of other shapes, never by the
    padding (the causal masks give a pad token an exact 0 coefficient)."""
    rows = rows_of(EDGE_LENGTHS)
    own, _, _ = encode(te_file, rows, 8, dtype)
    real_pad = te_live.pad_group
    generator = torch.Generator().manual_seed(5)

    def random_pad(group):
        out = real_pad(group)
        for j, ids in enumerate(group):
            n = int(ids.numel())
            out[j, n:] = torch.randint(0, TINY_TE.vocab_size, (out.shape[1] - n,), generator=generator)
        return out

    monkeypatch.setattr(te_live, "pad_group", random_pad)
    other, _, _ = encode(te_file, rows, 8, dtype)
    assert all(torch.equal(a, b) for a, b in zip(own, other))


def test_pad_activations_stay_finite(te_file):
    rows = rows_of(EDGE_LENGTHS)
    encoder = make_encoder(te_file, batch_rows=8)
    seen = []

    def hook(module, inputs, output):
        seen.append((tuple(output.shape), bool(torch.isfinite(output.float()).all())))

    handles = [layer.register_forward_hook(hook) for layer in encoder.shell.layers]
    try:
        encoder.encode_rows(rows)
    finally:
        for handle in handles:
            handle.remove()
        encoder.close()
    assert len(seen) == TINY_TE.num_hidden_layers
    assert all(shape == (5, 130, TINY_TE.hidden_size) and finite for shape, finite in seen), seen


def test_a_non_finite_pad_would_reach_the_real_tokens(te_file, monkeypatch):
    """Why the pad must be a real token: a NaN at a pad position enters a real token's sums as 0 x NaN = NaN (here
    through the gated delta rule's in-chunk matmuls), although its coefficient is an exact 0."""
    rows = [random_ids(130, seed=1), random_ids(63, seed=2)]
    encoder = make_encoder(te_file, batch_rows=2)
    real_pad, real_embed = te_live.pad_group, encoder.weights.embed_rows

    def marked_pad(group):
        out = real_pad(group)
        for j, ids in enumerate(group):
            out[j, int(ids.numel()):] = -1
        return out

    def embed(ids):
        ids = ids.reshape(-1)
        out = real_embed(ids.clamp_min(0))
        out[ids < 0] = float("nan")
        return out

    monkeypatch.setattr(te_live, "pad_group", marked_pad)
    monkeypatch.setattr(encoder.weights, "embed_rows", embed)
    try:
        out = encoder.encode_rows(rows)
    finally:
        encoder.close()
    assert torch.isfinite(out[0].float()).all()  # the longest row has no pad
    assert not torch.isfinite(out[1].float()).all()


# ---- the engine around it ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("host", ["copy", "mmap"])
@pytest.mark.parametrize("ring_slots", [2, 4])
@pytest.mark.parametrize("staging", [False, True])
def test_streaming_variants_run_the_groups_alike(te_file, host, ring_slots, staging):
    """The slots, host forms and the threaded stager carry the same layer bits: the batched rows are the same bit for
    bit in every variant (and the shell is unbound after the pass)."""
    rows = rows_of(MIXED_LENGTHS)
    baseline, _, _ = encode(te_file, rows, 3)
    encoder = make_encoder(te_file, batch_rows=3, host=host, ring_slots=ring_slots, staging=staging)
    try:
        out = encoder.encode_rows(rows)
        assert all(torch.equal(a, b) for a, b in zip(out, baseline)), (host, ring_slots, staging)
        assert encoder.bound_layers() == []
        again = encoder.encode_rows(rows[:2])  # a second pass, now one group of 2
        assert encoder.last_groups == [(2, max(MIXED_LENGTHS[:2]))]
        assert all(h.shape[0] == n for h, n in zip(again, MIXED_LENGTHS[:2]))
    finally:
        encoder.close()


def test_batch_rows_validation_and_stats(te_file):
    encoder = make_encoder(te_file)
    try:
        assert encoder.batch_rows == 1
        rows = rows_of((10, 20, 30))
        encoder.encode_rows(rows)
        assert encoder.last_groups == [(1, 10), (1, 20), (1, 30)]
        assert encoder.stats["calls"] == 3 and encoder.stats["padded_tokens"] == encoder.stats["tokens"] == 60
        encoder.batch_rows = 4
        encoder.encode_rows(rows)
        assert encoder.last_groups == [(3, 30)]
        assert encoder.stats["calls"] == 1 and encoder.stats["tokens"] == 60 and encoder.stats["padded_tokens"] == 90
        for bad in (0, -2, True, 2.0, "4"):
            with pytest.raises(ValueError, match="batch_rows"):
                encoder.batch_rows = bad
        assert encoder.batch_rows == 4
        with pytest.raises(ValueError, match="batch_rows"):
            te_live.LiveTextEncoder(encoder.weights, "cpu", batch_rows=0)
    finally:
        encoder.close()


def test_strategy_branch_with_batching(te_file, monkeypatch):
    """MossTextEncodingStrategy.encode_tokens through a batched live encoder: the same mask and zero padding as the
    cached strategy, embeds close to it."""
    tok = tokenize_strategy(monkeypatch)
    encoding = strategy.MossTextEncodingStrategy()
    texts = ["a red square", "two blue circles on a white table, sitting, long shadows on the floor", "", "green"]
    tokens = tok.tokenize(texts)
    cached = encoding.encode_tokens(tok, [reference_model(te_file)], tokens)
    encoder = make_encoder(te_file, batch_rows=4)
    try:
        live = encoding.encode_tokens(tok, [encoder], tokens)
    finally:
        encoder.close()
    assert torch.equal(live[1], cached[1]) and live[0].shape == cached[0].shape and live[0].dtype == torch.bfloat16
    lengths = cached[1].sum(1).tolist()
    for i, n in enumerate(lengths):
        assert torch.all(live[0][i, n:] == 0)
    rows_live = [live[0][i, :n] for i, n in enumerate(lengths)]
    rows_cached = [cached[0][i, :n] for i, n in enumerate(lengths)]
    assert min_cosine(rows_live, rows_cached) >= BF16_MIN_COSINE


def _batch(index, lengths):
    width = max(lengths)
    ids = torch.zeros(len(lengths), width, dtype=torch.long)
    mask = torch.zeros(len(lengths), width, dtype=torch.long)
    for b, n in enumerate(lengths):
        ids[b, :n] = torch.arange(1, n + 1) + 10 * index
        mask[b, :n] = 1
    return {"index": index, "input_ids_list": [ids, mask], "latents": torch.zeros(len(lengths), 4, 2, 2),
            "text_encoder_outputs_list": None}


@pytest.mark.parametrize("batch_rows", [1, 4])
def test_feed_logs_the_first_pass_groups_once(te_file, batch_rows, monkeypatch):
    lines = []
    monkeypatch.setattr(te_live_feed.logger, "info", lambda m, *a, **k: lines.append(str(m)))
    encoder = make_encoder(te_file, batch_rows=batch_rows)
    try:
        feed = te_live_feed.TextEncodeAhead(encoder, ahead=3)
        batches = [_batch(i, [12 + 3 * i, 20]) for i in range(7)]  # passes of 6, 6 and 2 rows
        out = list(feed.wrap(iter(batches)))
    finally:
        encoder.close()
    assert [b["index"] for b in out] == list(range(7)) and feed.pass_count == 3
    for b in out:
        embeds, mask, _ = b["text_encoder_outputs_list"]
        assert mask.sum(1).tolist() == [12 + 3 * b["index"], 20] and embeds.shape[1] == max(12 + 3 * b["index"], 20)
    if batch_rows == 1:
        assert lines == []
        return
    assert len(lines) == 1, lines
    assert "first pass (--moss_te_live_batch 4): 6 captions in 2 layer calls of [3, 3] rows" in lines[0]
    assert "padded to [20, 18] tokens" in lines[0] and "pad tokens" in lines[0]
    # the pass peak of these shapes (ft_budget.te_pass_gpu_bytes), next to the start-up budget line's estimate
    peak = ft_budget.te_pass_gpu_bytes([(3, 20), (3, 18)], 2)
    assert f"pass peak ~{peak / (1 << 30):.2f} GiB for these shapes" in lines[0]
