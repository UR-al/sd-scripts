"""CPU tests for --anima_tier_timing (library/tier_timing.py): the tier mapping, the token count of a batch, the timer
(CPU clock and CUDA-style event pairs harvested only once finished), the tracker scalars / log line, and the trainer
wiring (hooks around each micro-step, the periodic and epoch-end reports, the parser).

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -B -m pytest tests/test_tier_timing.py
"""
import argparse
import inspect
import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

from library import tier_timing


def test_tier_of_tokens_nearest_on_a_log_scale():
    assert tier_timing.tier_of_tokens(1024) == "512" and tier_timing.tier_of_tokens(880) == "512"  # 704x320 bucket
    assert tier_timing.tier_of_tokens(4096) == "1024" and tier_timing.tier_of_tokens(3840) == "1024"  # 1536x640
    assert tier_timing.tier_of_tokens(9216) == "1536" and tier_timing.tier_of_tokens(16384) == "2048"
    assert tier_timing.tier_of_tokens(20000) == "2048" and tier_timing.tier_of_tokens(1) == "512"
    # the geometric midpoints are the boundaries; a tie goes to the lower tier
    assert tier_timing.tier_of_tokens(2048) == "512" and tier_timing.tier_of_tokens(2049) == "1024"
    assert tier_timing.tier_of_tokens(6144) == "1024" and tier_timing.tier_of_tokens(6145) == "1536"
    assert tier_timing.tier_of_tokens(12288) == "1536" and tier_timing.tier_of_tokens(12289) == "2048"
    assert tier_timing.tier_of_tokens(0) == tier_timing.UNKNOWN and tier_timing.tier_of_tokens(-5) == tier_timing.UNKNOWN


def test_sample_tokens_from_latents_or_pixels():
    assert tier_timing.sample_tokens({"latents": torch.zeros(2, 16, 64, 64)}) == 1024  # 512^2: 64x64 latents, 32x32 patches
    assert tier_timing.sample_tokens({"latents": torch.zeros(1, 16, 1, 128, 96)}) == 64 * 48  # (B, C, T, H, W)
    assert tier_timing.sample_tokens({"latents": torch.zeros(1, 16, 65, 33)}) == 33 * 17  # partial patches round up
    assert tier_timing.sample_tokens({"images": torch.zeros(1, 3, 1024, 1024)}) == 4096  # pixels: /8 then patches
    assert tier_timing.sample_tokens({"captions": ["a"]}) == 0 and tier_timing.sample_tokens(None) == 0
    assert tier_timing.sample_tokens({"latents": torch.zeros(1, 16, 64, 64)}, patch=4) == 256


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_cpu_timer_reports_window_and_epoch_statistics():
    clock = _Clock()
    timer = tier_timing.TierTimer("cpu", clock=clock)
    assert not timer.cuda
    for tokens, ms in ((1024, 300.0), (1024, 320.0), (4096, 900.0), (880, 310.0), (16384, 4000.0)):
        timer.start(tokens)
        clock.t += ms / 1000.0
        timer.stop()
        clock.t += 0.001  # optimizer / host work between micro-steps is not counted
    stats = timer.report()
    assert stats["512"]["count"] == 3 and stats["512"]["median_ms"] == pytest.approx(310.0) and stats["512"]["mean_ms"] == pytest.approx(310.0)
    assert stats["1024"] == {"count": 1.0, "mean_ms": pytest.approx(900.0), "median_ms": pytest.approx(900.0)}
    assert stats["2048"]["count"] == 1 and "1536" not in stats
    assert timer.report() == {}  # the window was cleared
    timer.start(4096)
    clock.t += 0.95
    timer.stop()
    assert timer.report()["1024"]["count"] == 1
    epoch = timer.epoch_report()  # everything since the start: 3 + 2 + 1
    assert {t: s["count"] for t, s in epoch.items()} == {"512": 3.0, "1024": 2.0, "2048": 1.0}
    assert epoch["1024"]["mean_ms"] == pytest.approx(925.0)
    assert timer.epoch_report() == {}
    scal = timer.scalars(stats := {"512": {"count": 3.0, "mean_ms": 310.0, "median_ms": 310.0}})
    assert scal == {"tier_time/512/count": 3.0, "tier_time/512/mean_ms": 310.0, "tier_time/512/median_ms": 310.0, "tier_time/pending": 0.0}
    line = timer.format_line(stats)
    assert line.startswith("tier timing (epoch; fwd+bwd per micro-step") and "512 n=3 mean 310 ms median 310 ms" in line
    assert timer.format_line({}) .endswith("no finished micro-steps")
    timer.start(1024)
    timer.start(1024)  # a start without stop drops the open span (counted) rather than mis-measuring it
    timer.stop()
    assert timer.dropped == 1 and "dropped 1" in timer.format_line({})


class _GPU:
    """A fake GPU timeline: events record the current tick; query() says whether the timeline has passed them."""

    def __init__(self):
        self.now = 0.0
        self.done_until = 0.0

    def event(self):
        gpu = self

        class Ev:
            def __init__(self):
                self.t = None

            def record(self):
                self.t = gpu.now

            def query(self):
                return self.t is not None and self.t <= gpu.done_until

            def elapsed_time(self, other):
                return (other.t - self.t) * 1000.0

        return Ev()


def test_event_timer_harvests_only_finished_pairs_without_sync():
    gpu = _GPU()
    timer = tier_timing.TierTimer("cuda", event_factory=gpu.event)
    assert timer.cuda
    spans = []
    for tokens, ms in ((1024, 300.0), (4096, 1000.0), (16384, 4000.0)):
        timer.start(tokens)
        gpu.now += ms / 1000.0  # the recorded end time; the GPU "finishes" later
        timer.stop()
        spans.append(gpu.now)
    assert timer.pending == 3
    gpu.done_until = spans[1]  # the first two micro-steps completed on the GPU, the third is still running
    stats = timer.report()
    assert {t: s["count"] for t, s in stats.items()} == {"512": 1.0, "1024": 1.0}
    assert stats["1024"]["mean_ms"] == pytest.approx(1000.0)
    assert timer.pending == 1 and timer.scalars(stats)["tier_time/pending"] == 1.0
    assert "pending 1" in timer.format_line(stats, what="step 50")
    gpu.done_until = spans[2]
    stats = timer.report()
    assert {t: s["count"] for t, s in stats.items()} == {"2048": 1.0} and timer.pending == 0
    assert timer.epoch_report()["2048"]["mean_ms"] == pytest.approx(4000.0)


def test_real_cpu_clock_runs():
    timer = tier_timing.TierTimer("cpu")
    timer.start(1024)
    timer.stop()
    stats = timer.report()
    assert stats["512"]["count"] == 1 and stats["512"]["mean_ms"] >= 0.0


# ------------------------------------------------------------------------------------------------------- trainer glue


def test_trainer_hooks_are_wired_around_the_micro_step():
    """train_network.NetworkTrainer.train (source): on_micro_step_start before the chunk's forward, on_micro_step_end after
    its backward (and the LyCORIS after-backward clear), on_optimizer_step_end right after global_step advances,
    on_epoch_end at the end of the epoch; the base hooks are no-ops."""
    import train_network

    src = inspect.getsource(train_network.NetworkTrainer.train)
    i_start = src.index("self.on_micro_step_start(args, accelerator, sub_batch, unet)")
    i_fwd = src.index("c_loss, c_pre_scaling_loss, c_loss_scaled = self.process_batch(")
    i_bwd = src.index("accelerator.backward(c_loss if backward_scale == 1.0 else c_loss * backward_scale)")
    i_end = src.index("self.on_micro_step_end(args, accelerator)")
    i_sum = src.index("loss = micro_batch.weighted_sum(loss, c_loss.detach(), chunk_weight)")
    assert i_start < i_fwd < i_bwd < i_end < i_sum
    i_gs = src.index("global_step += 1")
    i_step = src.index("self.on_optimizer_step_end(args, accelerator, global_step, is_tracking)")
    assert i_gs < i_step < src.index("train_util.sample_images_check(args, None, global_step)")
    i_epoch = src.index("# END OF EPOCH")
    assert i_epoch < src.index("self.on_epoch_end(args, accelerator, epoch, global_step)") < src.index("accelerator.wait_for_everyone()", i_epoch)
    t = train_network.NetworkTrainer()
    assert t.on_micro_step_start(None, None, {}, None) is None and t.on_micro_step_end(None, None) is None
    assert t.on_optimizer_step_end(None, None, 1, False) is None and t.on_epoch_end(None, None, 0, 1) is None


class _Accel:
    def __init__(self):
        self.logged = []
        self.device = torch.device("cpu")

    def log(self, values, step=None):
        self.logged.append((step, dict(values)))

    def unwrap_model(self, m):
        return m


def test_anima_trainer_times_micro_steps_and_reports(caplog):
    """AnimaNetworkTrainer with --anima_tier_timing: each micro-step is one span of its tier, the report every
    --anima_tier_timing_every optimizer steps goes to the tracker (only when tracking), the epoch end logs one line; off,
    the hooks do nothing (no timer is created)."""
    import anima_train_network as atn

    unet = types.SimpleNamespace(patch_spatial=2)
    args = argparse.Namespace(anima_tier_timing=False, anima_tier_timing_every=2)
    off = atn.AnimaNetworkTrainer()
    accel = _Accel()
    off.on_micro_step_start(args, accel, {"latents": torch.zeros(1, 16, 64, 64)}, unet)
    off.on_micro_step_end(args, accel)
    off.on_optimizer_step_end(args, accel, 2, True)
    off.on_epoch_end(args, accel, 0, 2)
    assert off._tier_timer is None and accel.logged == []

    args.anima_tier_timing = True
    tr = atn.AnimaNetworkTrainer()
    batches = [{"latents": torch.zeros(1, 16, 64, 64)}, {"latents": torch.zeros(1, 16, 128, 128)},
               {"latents": torch.zeros(1, 16, 64, 64)}, {"images": torch.zeros(1, 3, 2048, 2048)}]
    for step, batch in enumerate(batches, start=1):
        tr.on_micro_step_start(args, accel, batch, unet)
        tr.on_micro_step_end(args, accel)
        tr.on_optimizer_step_end(args, accel, step, True)
    assert tr._tier_timer is not None and not tr._tier_timer.cuda
    assert [s for s, _ in accel.logged] == [2, 4]  # every 2 optimizer steps
    first, second = accel.logged[0][1], accel.logged[1][1]
    assert first["tier_time/512/count"] == 1.0 and first["tier_time/1024/count"] == 1.0 and first["tier_time/pending"] == 0.0
    assert second["tier_time/512/count"] == 1.0 and second["tier_time/2048/count"] == 1.0 and "tier_time/1024/count" not in second
    tr.on_micro_step_start(args, accel, batches[0], unet)
    tr.on_micro_step_end(args, accel)
    tr.on_optimizer_step_end(args, accel, 5, False)  # not tracking: no scalar, but the window keeps accumulating
    assert len(accel.logged) == 2
    with caplog.at_level(logging.INFO, logger="anima_train_network"):
        tr.on_epoch_end(args, accel, 0, 5)
    lines = [r.getMessage() for r in caplog.records if "tier timing" in r.getMessage()]
    assert len(lines) == 1 and "512 n=3" in lines[0] and "1024 n=1" in lines[0] and "2048 n=1" in lines[0], lines
    assert tr._tier_timer.epoch_report() == {}  # the epoch window was consumed


def test_parser_and_validation():
    import anima_train_network as atn

    parser = atn.setup_parser()
    ns = parser.parse_args([])
    assert ns.anima_tier_timing is False and ns.anima_tier_timing_every == 50
    ns = parser.parse_args(["--anima_tier_timing", "--anima_tier_timing_every", "10"])
    assert ns.anima_tier_timing is True and ns.anima_tier_timing_every == 10
    assert "validate_tier_timing_args(args)" in inspect.getsource(atn.AnimaNetworkTrainer.assert_extra_args)
    atn.validate_tier_timing_args(argparse.Namespace(anima_tier_timing=True, anima_tier_timing_every=1))
    atn.validate_tier_timing_args(argparse.Namespace(anima_tier_timing=False, anima_tier_timing_every=0))
    with pytest.raises(ValueError, match="anima_tier_timing_every"):
        atn.validate_tier_timing_args(argparse.Namespace(anima_tier_timing=True, anima_tier_timing_every=0))
