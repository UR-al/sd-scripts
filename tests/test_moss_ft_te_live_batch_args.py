"""CPU tests of --moss_te_live_batch (library/moss/ft_args.py, ft_budget.py): the flag parses (also from a TOML config,
default 1 = today's per-caption path), its refusals (an integer >= 1; only with --moss_te_live, with the reason the
cached path has nothing to batch), the start-up log line, and the live text encoder's estimates for batched layer
calls: the pass peak grows by the largest call's temporaries (min(N, rows) rows), it bounds every pass the run can make
(shorter passes, and through length_groups' token budget the smoke subset's real 10-770-token captions; reviewer
11-18), te_pass_gpu_bytes models a pass's actual call shapes, the launch overhead is paid per group, the budget line
names the groups; batch 1 keeps every earlier number.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_te_live_batch_args.py -q -p no:cacheprovider
"""
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest

from library.moss import ft_args, ft_budget as B, te_live
from test_moss_ft_te_live_args import Group, cached_args, live_args, validate
from test_moss_ft_te_live_batch import SMOKE_LENGTHS

GIB, MIB = 1 << 30, 1 << 20


@pytest.fixture(scope="module")
def parser():
    return ft_args.setup_parser()


@pytest.fixture
def logs(monkeypatch):
    out = {"info": [], "warning": []}
    for module in (ft_args, B):
        monkeypatch.setattr(module.logger, "info", lambda m, *a, **k: out["info"].append(str(m)))
        monkeypatch.setattr(module.logger, "warning", lambda m, *a, **k: out["warning"].append(str(m)))
    return out


def test_default_and_toml(parser, tmp_path, monkeypatch):
    from library import train_util

    assert parser.parse_args([]).moss_te_live_batch == 1 == ft_args.TE_LIVE_DEFAULTS["moss_te_live_batch"]
    assert parser.parse_args(["--moss_te_live_batch", "16"]).moss_te_live_batch == 16
    config = tmp_path / "live_batch.toml"
    config.write_text("moss_te_live = true\nmoss_te_live_batch = 8\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["moss_train.py", "--config_file", str(config)])
    args = train_util.read_config_from_file(parser.parse_args(), parser, unknown_keys="error")
    assert args.moss_te_live and args.moss_te_live_batch == 8
    with pytest.raises(SystemExit):
        parser.parse_args(["--moss_te_live_batch", "many"])


@pytest.mark.parametrize("value", [8, 1, 32, 1000])
def test_valid_values_pass(parser, value):
    validate(live_args(parser, "--moss_te_live_batch", str(value)))


@pytest.mark.parametrize("value", [0, -3, True, 2.5, "8", None])
def test_refused_values(parser, value):
    with pytest.raises(ValueError, match="--moss_te_live_batch must be an integer >= 1") as refused:
        validate(live_args(parser, moss_te_live_batch=value))
    assert str(refused.value).startswith(ft_args.PREFIX)


def test_refused_without_the_live_encoder(parser):
    with pytest.raises(ValueError, match="--moss_te_live_batch is only used with --moss_te_live") as refused:
        validate(cached_args(parser, moss_te_live_batch=8))
    assert "the TE cache encodes each caption alone once" in str(refused.value)
    with pytest.raises(ValueError, match="--moss_te_live_ahead, --moss_te_live_batch are only used"):
        validate(cached_args(parser, moss_te_live_ahead=8, moss_te_live_batch=8))
    validate(cached_args(parser, moss_te_live_batch=1))  # the default is no request


def test_startup_log_line(parser, logs):
    validate(live_args(parser))
    line = next(m for m in logs["info"] if "--moss_te_live: the captions are encoded at train time" in m)
    assert "each caption alone" in line
    logs["info"].clear()
    validate(live_args(parser, "--moss_te_live_batch", "8"))
    line = next(m for m in logs["info"] if "--moss_te_live: the captions are encoded at train time" in m)
    assert "layer calls of up to 8 captions sorted by length and right-padded" in line
    assert "at most 8 x 300 padded tokens per call" in line


def test_pass_peak_counts_the_largest_group():
    ring, plain = B.te_live_gpu_bytes(32, 300, 2)
    assert (ring, plain) == B.te_live_gpu_bytes(32, 300, 2, 1)  # batch 1: the earlier estimate
    activations = 2 * 32 * 300 * 4096 * 2
    assert plain - ring == activations + B.te_row_temp_bytes(300)
    _, eight = B.te_live_gpu_bytes(32, 300, 2, 8)
    assert eight - ring == activations + 8 * B.te_row_temp_bytes(300)
    # 33 captions run as 17 + 16, but a pass of 32 (the loader's last, short bucket batches) as one group of 32
    _, odd = B.te_live_gpu_bytes(33, 300, 2, 32)
    assert odd - ring == 2 * 33 * 300 * 4096 * 2 + 32 * B.te_row_temp_bytes(300)
    _, capped = B.te_live_gpu_bytes(4, 300, 2, 32)  # a group never holds more rows than the pass
    assert capped - ring == 2 * 4 * 300 * 4096 * 2 + 4 * B.te_row_temp_bytes(300)
    assert eight / GIB == pytest.approx(1.47, abs=0.02)  # ring 0.81 + rows 0.15 + 8 x 65 MiB


def _pass_bytes(shapes, ring_slots=2):
    """The memory model of one pass, written out here from the constants (not from ft_budget's helper): the ring, the
    padded layer input of every call (all held through the pass), the largest call's output and the largest call's
    temporaries (rows x one row's at the padded length)."""
    held = sum(rows * width for rows, width in shapes) * 4096 * 2
    output = max(rows * width for rows, width in shapes) * 4096 * 2
    temps = max(rows * (width * (3 * 12288 * 2 + 4 * 8192 * 4) + 6 * MIB) for rows, width in shapes)
    return ring_slots * B.TE_SLOT_BYTES + held + output + temps


def _shapes(lengths, batch):
    return [(len(g), max(lengths[i] for i in g)) for g in te_live.length_groups(lengths, batch)]


def test_pass_peak_bounds_every_pass_the_run_can_make():
    """Reviewer 11-18: the pass peak must hold for the passes the run actually makes. (a) Captions up to 300 tokens
    (what the estimate assumes): every pass of m <= rows captions - the loader's last pass and short bucket batches
    give fewer than ahead x batch, and a shorter pass can be ONE group (33 at 32: 17 + 16, but 32 at 32: 32) - stays
    within it by construction. (b) The smoke subset's real captions (10-770 tokens): by row count alone N = 32 made one
    call of 32 x 770, about twice the estimate; the token budget of a call keeps it within."""
    T = B.DEFAULT_TEXT_TOKENS
    for rows in (1, 5, 17, 32, 33):
        for batch in (1, 2, 8, 12, 16, 32, 40):
            _, peak = B.te_live_gpu_bytes(rows, T, 2, batch)
            for m in range(1, rows + 1):
                for lengths in ([T] * m, [T] + [1] * (m - 1), [1 + (37 * k) % T for k in range(m)]):
                    shapes = _shapes(lengths, batch)
                    assert _pass_bytes(shapes) <= peak, (rows, batch, m, shapes)
    for batch in (8, 16, 32):
        _, peak = B.te_live_gpu_bytes(32, T, 2, batch)
        shapes = _shapes(SMOKE_LENGTHS, batch)
        assert _pass_bytes(shapes) <= peak, (batch, shapes, _pass_bytes(shapes) / GIB, peak / GIB)
        temps = max(r * B.te_row_temp_bytes(w) for r, w in shapes)  # the part the reviewer found x N too low
        assert temps <= batch * B.te_row_temp_bytes(T), (batch, shapes)
    # the per-row path (default, unchanged) misses by its one 770-token row only (~5 MiB here): one row's worth
    _, peak = B.te_live_gpu_bytes(32, T, 2, 1)
    over = _pass_bytes(_shapes(SMOKE_LENGTHS, 1)) - peak
    assert 0 < over < B.te_row_temp_bytes(770) - B.te_row_temp_bytes(T)


def test_pass_bytes_of_the_actual_shapes():
    """ft_budget.te_pass_gpu_bytes: the model above for a pass's [(rows, padded length)] (LiveTextEncoder.last_groups;
    the bench prints it per setting before it touches CUDA). The per-row path is [(1, T_i)]."""
    for shapes in ([(1, 770), (1, 10)], [(3, 770), (5, 384), (8, 284)], [(32, 300)]):
        assert B.te_pass_gpu_bytes(shapes, 2) == _pass_bytes(shapes, 2)
        assert B.te_pass_gpu_bytes(shapes, 4) == _pass_bytes(shapes, 4)
    assert B.te_pass_gpu_bytes([], 2) == 2 * B.TE_SLOT_BYTES
    assert B.te_pass_gpu_bytes([(32, 300)], 2) <= B.te_live_gpu_bytes(32, 300, 2, 32)[1]
    assert B.te_call_rows(33, 32) == 32 and B.te_call_rows(4, 32) == 4 and B.te_call_rows(32, 1) == 1
    smoke = {batch: B.te_pass_gpu_bytes(_shapes(SMOKE_LENGTHS, batch), 2) / GIB for batch in (1, 8, 16, 32)}
    print(f"\nsmoke subset, predicted pass peak (GiB): {smoke}")
    # batch 1 ~1.03, 8 ~1.38, 16 ~1.85, 32 ~2.41 GiB (by row count alone 32 was one 32 x 770 call: ~6.1 GiB)
    assert smoke[1] == pytest.approx(1.03, abs=0.01) and smoke[8] == pytest.approx(1.38, abs=0.01)
    assert smoke[16] == pytest.approx(1.85, abs=0.01) and smoke[32] == pytest.approx(2.41, abs=0.01)


def test_settings_budget_and_speed():
    args = types.SimpleNamespace(moss_te_live=True, moss_te_live_ahead=32, gradient_accumulation_steps=1,
                                 moss_te_live_host="mmap", moss_te_live_ring_slots=2, moss_te_live_batch=8)
    live = B.te_live_settings(args)
    assert live == B.TeLive(host="mmap", ring_slots=2, ahead=32, layer_batch=8)
    assert B.te_live_settings(types.SimpleNamespace(moss_te_live=True)).layer_batch == 1
    one = B.gpu_budget(4096, 1, te_live=B.TeLive(ahead=32))
    batched = B.gpu_budget(4096, 1, te_live=B.TeLive(ahead=32, layer_batch=8))
    assert batched.te_pass - one.te_pass == 7 * B.te_row_temp_bytes(B.DEFAULT_TEXT_TOKENS)
    assert batched.te_ring == one.te_ring and batched.te_held == one.te_held and batched.dit_step == one.dit_step
    # launch overhead per layer call: 4 groups instead of 32 rows (the pass becomes staging-bound in the estimate)
    rows = 32
    flops = rows * 300 * B.TE_FLOPS_PER_TOKEN / B.TE_GPU_FLOPS
    assert B.te_seconds_per_pass(rows, 300, "pinned") == pytest.approx(flops + 32 * B.TE_LAUNCH_S_PER_ROW)
    assert B.te_seconds_per_pass(rows, 300, "pinned", layer_batch=8) == pytest.approx(flops + 4 * B.TE_LAUNCH_S_PER_ROW)
    assert B.te_seconds_per_image(B.TeLive(ahead=32, layer_batch=8), 1) <= B.te_seconds_per_image(B.TeLive(ahead=32), 1)
    assert B.throughput(1, 2, te_live=B.TeLive(ahead=32, layer_batch=8)).step <= B.throughput(
        1, 2, te_live=B.TeLive(ahead=32)).step


def test_budget_line_names_the_groups(parser, tmp_path, logs):
    validate(live_args(parser, "--moss_te_live_batch", "8"), Group(tmp_path))
    text = "\n".join(logs["info"])
    assert "GPU live text encoder: ring 2 x 416.6 MiB = 0.81 GiB" in text
    assert "for 32 captions in layer calls of 4 x 8 rows (--moss_te_live_batch 8;" in text
    assert "a call holds at most 8 captions and 8 x 300 padded tokens, captions past 300 tokens run in smaller" in text
    logs["info"].clear()
    validate(live_args(parser), Group(tmp_path))
    line = next(m for m in logs["info"] if "GPU live text encoder" in m)
    assert "for 32 captions (between micro-steps" in line and "layer calls" not in line
