"""CPU tests for the CUDA caching-allocator controls (library/cuda_memory.py) and the cap-aware free-bytes helper in
library/lycoris_fast.py. torch.cuda calls are monkeypatched; whether the cap really trims the cache is a GPU check.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -B -m pytest tests/test_cuda_memory.py
"""
import argparse
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

from library import cuda_memory as cm

GIB = 2**30
FAKE = torch.device("cuda", 0)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("CUDA_MEMORY_FRACTION", "CUDA_MEMLOG", "CUDA_MEMLOG_FILE", "PYTORCH_CUDA_ALLOC_CONF"):
        monkeypatch.delenv(k, raising=False)
    cm._CAP.clear()
    yield
    cm._CAP.clear()


def ns(**kw):
    return argparse.Namespace(**kw)


def fake_cuda(monkeypatch, reserved=10 * GIB, allocated=8 * GIB, free=4 * GIB, total=32 * GIB, calls=None):
    calls = calls if calls is not None else []
    monkeypatch.setattr(cm, "_cuda", lambda d: None if d is None or torch.device(d).type != "cuda" else FAKE)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d=None: reserved)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d=None: allocated)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d=None: reserved + GIB)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d=None: allocated + GIB)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (free, total))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d=None: calls.append("reset"))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f, d=None: calls.append(("fraction", f)))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: argparse.Namespace(total_memory=total))
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda d=None: {
            "inactive_split_bytes.all.current": GIB // 2,
            "inactive_split_bytes.all.peak": GIB,
            "num_alloc_retries": 3,
            "num_ooms": 0,
            "num_device_alloc": 40,
            "num_device_free": 12,
        },
    )
    return calls


def test_resolve_fraction_precedence_and_range(monkeypatch):
    assert cm.resolve_fraction(ns()) is None
    monkeypatch.setenv("CUDA_MEMORY_FRACTION", "0.85")
    assert cm.resolve_fraction(ns()) == pytest.approx(0.85)
    assert cm.resolve_fraction(ns(cuda_memory_fraction=0.7)) == pytest.approx(0.7)  # argument wins over env
    for bad in (0.0, 1.5, -1.0):
        with pytest.raises(ValueError):
            cm.resolve_fraction(ns(cuda_memory_fraction=bad))


def test_everything_is_a_no_op_on_cpu(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_MEMORY_FRACTION", "0.8")
    monkeypatch.setenv("CUDA_MEMLOG_FILE", str(tmp_path / "m.jsonl"))
    assert cm.apply_memory_fraction(ns(), "cpu") is None
    assert cm.cap_bytes("cpu") is None
    assert cm.maybe_empty_cache(ns(cuda_empty_cache_every_n_steps=1), "cpu", 10) is False
    log = cm.MemLog("cpu")
    assert not log.enabled
    log.mark("x", 1)
    assert not (tmp_path / "m.jsonl").exists()


def test_apply_memory_fraction_sets_cap(monkeypatch):
    calls = fake_cuda(monkeypatch, total=32 * GIB)
    assert cm.apply_memory_fraction(ns(cuda_memory_fraction=0.8), FAKE) == pytest.approx(0.8)
    assert ("fraction", 0.8) in calls
    assert cm.cap_bytes(FAKE) == int(0.8 * 32 * GIB)
    cm._CAP.clear()
    calls.clear()
    assert cm.apply_memory_fraction(ns(), FAKE) is None  # nothing set: no call, no cap
    assert calls == [] and cm.cap_bytes(FAKE) is None


def test_alloc_conf_warnings(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.9")
    assert any("garbage_collection_threshold" in w for w in cm._alloc_conf_warnings(None))
    assert cm._alloc_conf_warnings(0.85) == []
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setattr(cm.sys, "platform", "win32")
    assert any("expandable_segments" in w for w in cm._alloc_conf_warnings(0.85))
    monkeypatch.setattr(cm.sys, "platform", "linux")
    assert cm._alloc_conf_warnings(0.85) == []


def test_maybe_empty_cache_every_n_and_above(monkeypatch):
    calls = fake_cuda(monkeypatch, reserved=20 * GIB)
    every = ns(cuda_empty_cache_every_n_steps=5, cuda_empty_cache_above_gib=None)
    fired = [s for s in range(1, 21) if cm.maybe_empty_cache(every, FAKE, s)]
    assert fired == [5, 10, 15, 20] and calls.count("empty_cache") == 4
    calls.clear()
    assert cm.maybe_empty_cache(ns(cuda_empty_cache_above_gib=19.5), FAKE, 3) is True
    assert cm.maybe_empty_cache(ns(cuda_empty_cache_above_gib=20.5), FAKE, 3) is False
    assert cm.maybe_empty_cache(ns(), FAKE, 3) is False
    assert calls.count("empty_cache") == 1


def test_memlog_writes_json_lines(monkeypatch, tmp_path):
    calls = fake_cuda(monkeypatch)
    cm._CAP[0] = 27 * GIB
    path = tmp_path / "mem.jsonl"
    monkeypatch.setenv("CUDA_MEMLOG", "10")
    monkeypatch.setenv("CUDA_MEMLOG_FILE", str(path))
    log = cm.MemLog(FAKE)
    assert log.enabled and log.every == 10
    monkeypatch.setenv("RANK", "3")  # accelerate / torchrun env: the row says which process wrote it
    for s in range(1, 21):
        log.step(s)
    monkeypatch.delenv("RANK")
    monkeypatch.setenv("LOCAL_RANK", "1")
    log.mark("after sampling", 21)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["step"] for r in rows] == [10, 20, 21] and rows[-1]["where"] == "after sampling"
    want = {"device", "rank", "allocated", "reserved", "max_allocated", "max_reserved", "inactive_split",
            "inactive_split_peak", "device_used", "cap", "alloc_retries", "ooms", "cuda_mallocs", "cuda_frees"}
    assert want <= set(rows[0])
    assert rows[0]["device"] == cm._cuda(FAKE).index and rows[0]["rank"] == 3 and rows[-1]["rank"] == 1
    assert rows[0]["inactive_split_peak"] == pytest.approx(1.0) and rows[0]["cap"] == pytest.approx(27.0)
    assert rows[0]["device_used"] == pytest.approx(28.0) and rows[0]["alloc_retries"] == 3
    assert calls.count("reset") == 3  # peaks are per window


def test_memlog_env_switches(monkeypatch, tmp_path):
    fake_cuda(monkeypatch)
    assert not cm.MemLog(FAKE).enabled
    monkeypatch.setenv("CUDA_MEMLOG", "0")
    assert not cm.MemLog(FAKE).enabled
    monkeypatch.setenv("CUDA_MEMLOG", "")
    monkeypatch.setenv("CUDA_MEMLOG_FILE", str(tmp_path / "x.jsonl"))
    log = cm.MemLog(FAKE)
    assert log.enabled and log.every == 50
    monkeypatch.setenv("CUDA_MEMLOG", "0")  # explicit 0 switches it off even with a file
    assert not cm.MemLog(FAKE).enabled


def test_memlog_file_error_switches_logging_off(monkeypatch, tmp_path):
    fake_cuda(monkeypatch)
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("CUDA_MEMLOG", "1")
    monkeypatch.setenv("CUDA_MEMLOG_FILE", str(blocker / "mem.jsonl"))  # parent is a file -> OSError
    log = cm.MemLog(FAKE)
    log.mark("step", 1)  # must not raise
    assert not log.enabled
    nested = tmp_path / "a" / "b" / "mem.jsonl"  # missing folders are created
    monkeypatch.setenv("CUDA_MEMLOG_FILE", str(nested))
    log = cm.MemLog(FAKE)
    log.mark("step", 1)
    assert log.enabled and len(nested.read_text(encoding="utf-8").splitlines()) == 1


def test_memlog_warns_next_to_tsm_lokr_memlog(monkeypatch, caplog):
    fake_cuda(monkeypatch)
    monkeypatch.setenv("CUDA_MEMLOG", "10")
    monkeypatch.setenv("TSM_LOKR_MEMLOG", "x.jsonl")
    with caplog.at_level("WARNING"):
        cm.MemLog(FAKE)
    assert any("TSM_LOKR_MEMLOG" in r.getMessage() for r in caplog.records)


def test_cuda_device_without_index_resolves_to_current(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert cm._cuda("cuda") == torch.device("cuda", 3)
    assert cm._cuda(torch.device("cuda", 1)) == torch.device("cuda", 1)
    assert cm._cuda("cpu") is None and cm._cuda(None) is None


def test_training_arguments_accept_the_flags():
    from library import train_util

    parser = argparse.ArgumentParser()
    train_util.add_training_arguments(parser, True)
    a = parser.parse_args([])
    assert a.cuda_memory_fraction is None and a.cuda_empty_cache_every_n_steps is None
    assert a.cuda_empty_cache_above_gib is None
    a = parser.parse_args(["--cuda_memory_fraction", "0.8", "--cuda_empty_cache_every_n_steps", "50",
                           "--cuda_empty_cache_above_gib", "25"])
    assert a.cuda_memory_fraction == pytest.approx(0.8) and a.cuda_empty_cache_every_n_steps == 50
    assert a.cuda_empty_cache_above_gib == pytest.approx(25.0)


def test_lycoris_fast_free_bytes_respects_the_cap(monkeypatch):
    from library import lycoris_fast

    fake_cuda(monkeypatch, reserved=20 * GIB, allocated=15 * GIB, free=10 * GIB)
    # no cap: driver-free VRAM + reusable cache
    assert lycoris_fast._free_bytes(FAKE) == 10 * GIB + 5 * GIB
    cm._CAP[0] = 22 * GIB  # only 2 GiB more may be reserved
    assert lycoris_fast._free_bytes(FAKE) == 2 * GIB + 5 * GIB
    cm._CAP[0] = 19 * GIB  # already above the cap: only the cache is usable
    assert lycoris_fast._free_bytes(FAKE) == 5 * GIB
