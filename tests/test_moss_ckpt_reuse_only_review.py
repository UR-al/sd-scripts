"""CPU tests for the review findings on the reuse-only mode / --moss_ckpt_reuse_attn_budget_gib (2026-11-15), next to
tests/test_moss_ckpt_reuse_only.py (its toy model and helpers are used here).

* the VRAM watch is about the DEVICE, not only this process: the user's criterion is "nothing spills into Windows'
  shared memory", i.e. the device total stays under the card, and ~2.3 GiB of the card is held outside the trainer's
  allocator (other programs, the CUDA context). The per-step check reads the device's free memory too and warns when
  peak reserved + what is held outside reaches the card, although peak reserved alone is still under 94 % of it;
* the peaks of that line are the run's: CUDA_MEMLOG resets torch's peak counters at every row
  (library/cuda_memory.MemLog.mark), which now remembers what it clears (cuda_memory.run_peaks);
* the start-up estimate counts what is countable - the memory other processes hold, the adapter's parameters and
  gradients - so "left" is what the optimizer's state, one block's recompute and the allocator really get, and the
  80 % line can fire for reuse-only;
* --moss_ckpt_nograd_merged + --moss_ckpt_reuse_attn + a budget under --moss_block_compile compiles 8 graphs on the
  shared block entry: a --moss_block_compile_recompile_limit below 16 is refused.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ckpt_reuse_only_review.py -q -p no:cacheprovider
"""
import argparse
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the name pytest imports the sibling test modules under
import test_moss_ckpt_reuse as base  # noqa: E402
import test_moss_ckpt_reuse_only as only  # noqa: E402
from test_moss_ckpt_reuse import _env, warned  # noqa: E402,F401 - fixtures (autouse _env included)
from test_moss_ckpt_reuse_only import _budget  # noqa: E402,F401 - autouse: every test starts without a budget

from library import cuda_memory  # noqa: E402
from library.moss import models as moss_models  # noqa: E402
from library.moss import train_utils as moss_train_utils  # noqa: E402

GIB = 2**30
CARD = int(31.84 * GIB)  # RTX 5090
FAKE = torch.device("cuda", 0)


class FakeCuda:
    """torch.cuda's memory counters as plain numbers a test moves (bytes)."""

    def __init__(self, monkeypatch, total=CARD):
        self.total, self.free = total, total
        self.allocated = self.reserved = self.max_allocated = self.max_reserved = 0
        self.resets = 0
        monkeypatch.setattr(cuda_memory, "_cuda", lambda d: None if d is None or torch.device(d).type != "cuda" else FAKE)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d=None: self.allocated)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d=None: self.reserved)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d=None: self.max_allocated)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d=None: self.max_reserved)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (self.free, self.total))
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: argparse.Namespace(total_memory=self.total))
        monkeypatch.setattr(torch.cuda, "memory_stats", lambda d=None: {})
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", self.reset)
        monkeypatch.setattr(cuda_memory, "_RUN_PEAK", {})

    def reset(self, d=None):
        self.resets += 1
        self.max_allocated, self.max_reserved = self.allocated, self.reserved

    def use(self, allocated, reserved, outside=0):
        """This process holds `allocated` / `reserved` now (the peaks follow), other processes `outside`."""
        self.allocated, self.reserved = int(allocated), int(reserved)
        self.max_allocated, self.max_reserved = max(self.max_allocated, self.allocated), max(self.max_reserved, self.reserved)
        self.free = max(0, self.total - self.reserved - int(outside))


# ------------------------------------------------------------------------------- the run's peaks under CUDA_MEMLOG


def test_stats_keep_the_run_peak_across_memlog_resets(monkeypatch, tmp_path):
    """CUDA_MEMLOG (the A/B harness and the real launcher set it) resets torch's peak counters at every row. The peak a
    row cleared stays in cuda_memory.run_peaks, so the trainer's 'VRAM peak so far' is the run's, not the last window's."""
    cuda = FakeCuda(monkeypatch)
    monkeypatch.setenv("CUDA_MEMLOG", "10")
    monkeypatch.setenv("CUDA_MEMLOG_FILE", str(tmp_path / "memlog.jsonl"))
    memlog = cuda_memory.MemLog(FAKE)
    assert memlog.enabled
    cuda.use(27 * GIB, 28 * GIB)  # a 2048 sample in steps 1..10
    cuda.use(12 * GIB, 28 * GIB)  # ... its activations are freed again
    memlog.step(10)  # the row: resets the counters (train_network.py calls it before on_optimizer_step_end)
    assert cuda.resets == 1 and cuda.max_allocated == 12 * GIB
    cuda.use(20 * GIB, 28 * GIB)  # steps 11..: smaller samples only
    stats = moss_train_utils.cuda_memory_stats(FAKE)
    assert stats["peak_allocated"] == 27 * GIB and stats["peak_reserved"] == 28 * GIB
    assert stats["allocated"] == 20 * GIB and stats["reserved"] == 28 * GIB
    line, _ = moss_train_utils.ckpt_reuse_peak_note(stats, options="--moss_ckpt_reuse_attn")
    assert "27.00 GiB allocated" in line
    # the epoch end: empty_cache, then a row - the reserved peak before it is the run's too
    cuda.use(12 * GIB, 13 * GIB)
    cuda.max_reserved = 28 * GIB  # (torch keeps the peak until the reset)
    memlog.mark("epoch end (cache released)", 65)
    assert cuda.max_reserved == 13 * GIB
    assert cuda_memory.run_peaks(FAKE) == (27 * GIB, 28 * GIB)
    cuda.use(29 * GIB, 30 * GIB)  # a later, larger peak wins
    assert cuda_memory.run_peaks(FAKE) == (29 * GIB, 30 * GIB)
    assert cuda_memory.run_peaks(torch.device("cpu")) is None and cuda_memory.run_peaks(None) is None


def test_stats_without_memlog_are_torchs_counters(monkeypatch):
    cuda = FakeCuda(monkeypatch)
    cuda.use(5 * GIB, 6 * GIB, outside=2 * GIB)
    cuda.use(3 * GIB, 6 * GIB, outside=2 * GIB)
    stats = moss_train_utils.cuda_memory_stats(FAKE)
    assert stats == dict(total=CARD, allocated=3 * GIB, reserved=6 * GIB, peak_allocated=5 * GIB, peak_reserved=6 * GIB,
                         free=CARD - 8 * GIB)
    assert "free" not in moss_train_utils.cuda_memory_stats(FAKE, with_free=False)


# ----------------------------------------------------------------------------- the device, not only this process


def test_peak_note_counts_what_other_processes_hold():
    """The user's line is the device total (no spill into shared memory). With ~2.3 GiB of the card held outside the
    trainer's allocator the device is full at ~29.5 GiB reserved - under the 94 % (29.93 GiB) line of peak reserved."""
    peak = moss_train_utils.ckpt_reuse_peak_note
    outside = int(2.3 * GIB)

    def stats(reserved_gib, now_gib=None):
        reserved = int(reserved_gib * GIB)
        now = reserved if now_gib is None else int(now_gib * GIB)
        return dict(total=CARD, allocated=now - GIB, reserved=now, peak_allocated=reserved - GIB, peak_reserved=reserved,
                    free=max(0, CARD - now - outside))

    # --moss_ckpt_reuse_attn without a budget at the 2048 tier: 25.6 + 4.16 = 29.76 GiB reserved, 32.06 GiB wanted
    line, at_limit = peak(stats(29.76), options="--moss_ckpt_reuse_attn")
    assert at_limit and "at the card's limit" in line
    # 3.0 GiB budget: 28.57 GiB reserved, 30.87 GiB on the device - under the line, and the line says where it stands
    line, at_limit = peak(stats(28.57), options="--moss_ckpt_reuse_attn")
    assert not at_limit and "28.57 GiB reserved of 31.84 GiB" in line
    assert "2.30 GiB held outside this process" in line and "30.87 GiB" in line
    # the same peak after an empty_cache (reserved now is small, the device has room again): still what the run reached
    line, at_limit = peak(stats(29.76, now_gib=14.0), options="--moss_ckpt_reuse_attn")
    assert at_limit and "32.06 GiB" in line
    assert not peak(stats(28.57, now_gib=14.0))[1]
    # nothing outside: the 94 % line of peak reserved decides as before
    alone = dict(total=CARD, allocated=20 * GIB, reserved=int(29.5 * GIB), peak_allocated=28 * GIB,
                 peak_reserved=int(29.5 * GIB), free=CARD - int(29.5 * GIB))
    assert not peak(alone)[1]
    assert peak(dict(alone, peak_reserved=int(0.95 * CARD), reserved=int(0.95 * CARD), free=CARD - int(0.95 * CARD)))[1]


def test_trainer_reads_the_device_at_every_optimizer_step(monkeypatch):
    """The per-step watch reads the device's free memory too (one driver query per optimizer step, no synchronize): the
    warning comes at the step where the device fills, not at the epoch end."""
    import moss_train_network as mtn

    warnings, reads = [], []
    monkeypatch.setattr(mtn.logger, "info", lambda message, *a, **k: None)
    monkeypatch.setattr(mtn.logger, "warning", lambda message, *a, **k: warnings.append(str(message)))
    outside = int(2.3 * GIB)
    state = {"reserved": int(26.0 * GIB)}

    def fake_stats(device, with_free=True):
        reads.append(with_free)
        reserved = state["reserved"]
        out = dict(total=CARD, allocated=reserved - GIB, reserved=reserved, peak_allocated=reserved - GIB,
                   peak_reserved=reserved)
        if with_free:
            out["free"] = max(0, CARD - reserved - outside)
        return out

    monkeypatch.setattr(mtn.moss_train_utils, "cuda_memory_stats", fake_stats)
    only.set_flags(monkeypatch, False, True)
    trainer = mtn.MossNetworkTrainer()
    trainer._ckpt_vram_device = FAKE
    trainer.on_optimizer_step_end(argparse.Namespace(), None, 1, False)
    assert warnings == [] and reads == [True]
    state["reserved"] = int(29.76 * GIB)  # under 94 % of the card (29.93 GiB), the device is over
    trainer.on_optimizer_step_end(argparse.Namespace(), None, 2, False)
    assert len(warnings) == 1 and "at the card's limit" in warnings[0] and "--moss_ckpt_reuse_attn" in warnings[0]
    trainer.on_optimizer_step_end(argparse.Namespace(), None, 3, False)
    assert len(warnings) == 1 and reads == [True, True]  # warned once: the per-step check is over


# --------------------------------------------------------------------------------------------- the start-up estimate


def test_adapter_vram_bytes_counts_gradients_and_parameters_still_on_the_cpu():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.zeros(10, 10))  # fp32, trained
            self.b = nn.Parameter(torch.zeros(6, dtype=torch.float16), requires_grad=False)  # frozen: nothing

    assert moss_train_utils.adapter_vram_bytes(Net()) == 2 * 100 * 4  # the parameter (not on the GPU yet) + its gradient
    assert moss_train_utils.adapter_vram_bytes(None) == 0 and moss_train_utils.adapter_vram_bytes(object()) == 0
    model, net, _ = base.build()
    trained = sum(p.numel() * p.element_size() for p in net.parameters() if p.requires_grad)
    assert trained > 0 and moss_train_utils.adapter_vram_bytes(net) == 2 * trained


def test_startup_note_counts_the_device_and_the_adapter():
    """The user's setup at the 2048 tier (fp16 + --moss_stream_fp32): 9.47 GiB of DiT weights, 2.3 GiB held outside the
    trainer, 7.84 GiB of kept block inputs, a 1.09 GiB adapter (parameters + gradients: 2.17 GiB). Without a budget the
    stash (3.98 GiB at the worst sample) leaves less than 20 % of the card: tight, the run that would spill is warned
    about. With the 3.0 GiB budget it is not, and 'left' is what the optimizer's state, a block's recompute and the
    allocator get - not the 11.5 GiB a count of the DiT weights alone showed."""
    real_size = types.SimpleNamespace(layers=[nn.Identity() for _ in range(42)], config=moss_models.MOSS_DIT_CONFIG)
    estimate, note = moss_train_utils.ckpt_reuse_vram_estimate, moss_train_utils.ckpt_reuse_vram_note
    allocated, outside, adapter = int(9.47 * GIB), int(2.3 * GIB), int(2.17 * GIB)
    card = dict(total=CARD, allocated=allocated, reserved=allocated + GIB // 4, free=CARD - allocated - GIB // 4 - outside)
    worst = dict(image_tokens=16384, caption_tokens=657)  # 17,041 joint tokens
    est = estimate(real_size, True, cached=False, stream_bytes=4, adapter_bytes=adapter, **worst)
    assert est["adapter"] == adapter
    line, tight = note(est, (2048, 2048), card)
    assert tight and "That is tight" in line
    assert "2.30 GiB held outside this process" in line and "2.17 GiB for the adapter's parameters and gradients" in line
    assert "the optimizer's state" in line
    budget = estimate(real_size, True, cached=False, stream_bytes=4, adapter_bytes=adapter, budget_bytes=3 * GIB, **worst)
    line, tight = note(budget, (2048, 2048), card)
    assert not tight and "30 of 42 blocks" in line
    left = CARD - allocated - outside - budget["inputs"] - adapter - budget["stash"]
    assert f"{left / GIB:.2f} GiB left" in line and 6.5 < left / GIB < 7.5
    # counters without the device's free memory (an older caller): the weights and the inputs only, as before
    line, tight = note(est, (2048, 2048), dict(total=CARD, allocated=allocated))
    assert "held outside" not in line and not tight


def test_trainer_passes_the_adapter_to_the_estimate(monkeypatch):
    import moss_train_network as mtn

    infos = []
    monkeypatch.setattr(mtn.logger, "info", lambda message, *a, **k: infos.append(str(message)))
    monkeypatch.setattr(mtn.logger, "warning", lambda message, *a, **k: infos.append(str(message)))
    stats = dict(total=32 * GIB, allocated=1 * GIB, reserved=1 * GIB, peak_allocated=1 * GIB, peak_reserved=1 * GIB,
                 free=29 * GIB)
    monkeypatch.setattr(mtn.moss_train_utils, "cuda_memory_stats", lambda device, with_free=True: dict(stats))
    model, net, _ = base.build()
    accelerator = types.SimpleNamespace(unwrap_model=lambda m: m, device=torch.device("cpu"))
    args = argparse.Namespace(moss_block_compile=False, moss_ckpt_nograd_merged=False, moss_ckpt_reuse_attn=True,
                              moss_ckpt_reuse_attn_budget_gib=None, mixed_precision="fp16", moss_stream_fp32=True)
    only.set_flags(monkeypatch, False, True)
    trainer = mtn.MossNetworkTrainer()
    trainer._dataset_limits = (1, 16, (64, 64))
    trainer.after_network_loaded(args, accelerator, net, [], model)
    line = [m for m in infos if "estimated extra VRAM" in m]
    assert len(line) == 1 and "2.00 GiB held outside this process" in line[0]
    assert "for the adapter's parameters and gradients" in line[0]
    adapter = moss_train_utils.adapter_vram_bytes(net)
    assert adapter > 0 and f"{adapter / GIB:.2f} GiB for the adapter" in line[0]


# ------------------------------------------------------------------------ cache + reuse + budget under block compile


def _compile_args(**changes):
    values = dict(moss_ckpt_nograd_merged=True, moss_ckpt_reuse_attn=True, moss_ckpt_reuse_attn_budget_gib=3.0,
                  moss_block_compile=True, moss_block_compile_recompile_limit=64)
    values.update(changes)
    return only._args(**values)


def test_cache_with_reuse_and_budget_compiles_eight_graphs(monkeypatch):
    """With the cache on, a block over the budget keeps the node but stores nothing (MossBlock._forward_no_stash): the
    shared compiled entry then holds the first pass with and without a stash and the recompute with and without the
    stored tensors, each for the adapted blocks and for the left-out one - 8 graphs (4 without a budget)."""
    counter = base.Counter()
    model, net, state = base.build(compiled=True, backend=counter)
    only.set_flags(monkeypatch, True, True, only.MIXED_BUDGET)
    one_sample = [spec for spec in only.MIXED if len(spec.get("text", (6,))) == 1]
    for spec in one_sample * 2:
        base.micro_step(model, net, base.batch(**spec))
    assert state.eager_calls == 0
    assert counter.graphs == 8, counter.graphs


def test_validation_wants_a_larger_recompile_limit_for_cache_reuse_budget(monkeypatch, warned):
    monkeypatch.setattr(only.anima_models, "CKPT_NOGRAD_MERGED", True)
    monkeypatch.setattr(only.anima_models, "CKPT_REUSE_ATTN", True)
    validate = moss_train_utils.validate_moss_ckpt_reuse_args
    with pytest.raises(ValueError, match="moss_block_compile_recompile_limit") as error:
        validate(_compile_args(moss_block_compile_recompile_limit=8))
    assert "16" in str(error.value) and "8 graphs" in str(error.value)
    with pytest.raises(ValueError, match="moss_block_compile_recompile_limit"):
        validate(_compile_args(moss_block_compile_recompile_limit=15))
    validate(_compile_args(moss_block_compile_recompile_limit=16))
    validate(_compile_args())  # the default 64
    validate(_compile_args(moss_block_compile_recompile_limit=8, moss_ckpt_reuse_attn_budget_gib=None))  # 4 graphs
    validate(_compile_args(moss_block_compile_recompile_limit=8, moss_block_compile=False))  # no compiled entry
    monkeypatch.setattr(only.anima_models, "CKPT_NOGRAD_MERGED", False)
    validate(_compile_args(moss_block_compile_recompile_limit=8, moss_ckpt_nograd_merged=False))  # reuse-only: 6 graphs
    assert warned == []


# -------------------------------------------------------------------------------------------------- the GPU harness


def test_gpu_harness_runs_the_trainers_reuse_only_and_budget_configurations(monkeypatch):
    """tests/manual_test_moss_ckpt_reuse_gpu.py measures the modes the trainer has: reuse-only and the budget go through
    moss_models.set_ckpt_reuse (no patched node lookup), and the 'chains' mode is the default CUDA run without block
    compile. (Its CPU self-test - `--device cpu` - runs every configuration end to end; too slow for this file.)"""
    import manual_test_moss_ckpt_reuse_gpu as harness

    only.set_flags(monkeypatch, False, False)  # monkeypatch restores the switches the harness sets below
    lookup = only.anima_models._ckpt_cached_lokrs
    assert harness.CONFIGS[harness.REUSE_ONLY] == (False, True)
    assert harness.CONFIGS[harness.REUSE_ONLY_BUDGET] == (False, True) and harness.CONFIGS[harness.CACHED_BUDGET] == (True, True)
    assert not hasattr(harness, "NO_CACHE")
    try:
        harness.set_budget(harness.budget_gib_for(harness.TOY, 22, 1))  # one block of a 22-token sample
        for name in harness.CONFIGS:
            harness.set_flags(name)
            nograd, reuse = harness.CONFIGS[name]
            assert (only.anima_models.CKPT_NOGRAD_MERGED, only.anima_models.CKPT_REUSE_ATTN) == (nograd, reuse)
            assert only.anima_models._ckpt_cached_lokrs is lookup  # the node's own lookup: nothing patched
            budget = moss_models.ckpt_reuse_budget_bytes()
            if name in harness.BUDGETED:
                per_token = moss_models.ckpt_stash_bytes_per_token(harness.TOY["dim"], harness.TOY["num_heads"])
                assert moss_models.ckpt_reuse_block_count(budget, 22, 1, per_token) == 1
            else:
                assert budget is None
    finally:
        harness.set_budget(None)
    parser_modes = [line for line in open(harness.__file__, encoding="utf-8") if 'add_argument("--modes"' in line]
    assert len(parser_modes) == 1 and '"chains"' in parser_modes[0]


# ----------------------------------------------------------------------------------------------------- the help text


def test_help_names_the_compiled_chains_caveat():
    """Reuse-only is bit-identical to the plain checkpoint only where both passes run the same kernels. A default CUDA
    run without --moss_block_compile compiles the elementwise chains and the LoKr+DoRA rebuild on their own
    (prepare_moss_runtime), and their no-grad and grad-mode graphs are separate compiles too: the help says so."""
    import moss_train_network as mtn

    text = " ".join(mtn.setup_parser().format_help().split())
    start = text.index("--moss_ckpt_reuse_attn --anima_ckpt_reuse_attn for MOSS")  # the option's own entry
    reuse_help = text[start:text.index("--moss_ckpt_reuse_attn_budget_gib MOSS_CKPT", start)]
    assert "--moss_no_compile" in reuse_help and "--moss_no_lokr_compile" in reuse_help


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
