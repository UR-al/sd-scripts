"""End-to-end CPU test of moss_train.py with --moss_te_live --moss_te_live_batch N (the harness and the tiny encoder
file of tests/test_moss_ft_te_live_e2e.py): a cached run as the reference, then live runs with batched layer calls
(ahead 3 -> 4 with GA 2, so passes of 4 captions and a tail pass of 1; N = 2 -> groups [2, 2], N = 4 -> one group of
4, the 1-caption tail pass is a group of one = the per-row shapes). The captions differ in length, so the groups pad.
The batched runs train every step, log te/*, save the DiT, log the first pass's groups once, and stay close to the
cached run (the encoder rows are close, not bit-identical: test_moss_ft_te_live_batch.py).

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_te_live_batch_e2e.py -q -p no:cacheprovider
"""
import math
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

import test_moss_ft_te_live_e2e as L
import test_moss_ft_train_e2e as E
from library.moss import te_live_feed

LOSS_REL_TOL = 1e-3  # losses / grad norms vs the cached run (observed 4.8e-5 at N = 2 and 4)
DIT_ABS_TOL = 5e-3  # the saved bf16 DiT after 6 AdamW steps at lr <= 1e-3 (observed 1.5e-3: one bf16 step)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return L.make_env(tmp_path_factory.mktemp("moss_te_live_batch_e2e"))


@pytest.fixture(scope="module")
def cached_run(env):
    logs = []
    out = env.root / "cached"
    L.run(env, out, *L.RUN, *L.CACHED, logs=logs)
    return types.SimpleNamespace(out=out, steps=E.by_step(logs))


@pytest.mark.parametrize("batch,calls,groups", [(2, 2, "[2, 2]"), (4, 1, "[4]")])
def test_batched_live_run_trains_close_to_the_cached_run(env, cached_run, batch, calls, groups, monkeypatch):
    lines = []
    monkeypatch.setattr(te_live_feed.logger, "info", lambda m, *a, **k: lines.append(str(m)))
    logs = []
    out = env.root / f"live_batch_{batch}"
    args = L.run(env, out, *L.RUN, "--moss_te_live", "--moss_te_live_ahead", "3", "--moss_te_live_batch", str(batch),
                 logs=logs)
    assert args.moss_te_live_batch == batch
    steps = E.by_step(logs)
    assert sorted(steps) == sorted(cached_run.steps) == list(range(1, 2 * E.STEPS_PER_EPOCH + 1))
    worst = 0.0
    for step in steps:
        live, cached = steps[step], cached_run.steps[step]
        assert math.isfinite(live["loss"]) and live["skipped_params"] == 0
        assert live["lr/decay"] == cached["lr/decay"]
        assert live["te/pass_s"] >= 0.0 and "te/pass_s" not in cached
        for key in ("loss", "grad_norm"):
            worst = max(worst, abs(live[key] - cached[key]) / max(abs(cached[key]), 1e-12))
            assert math.isclose(live[key], cached[key], rel_tol=LOSS_REL_TOL), (batch, step, key, live[key], cached[key])
    assert sum(v["te/pass_images"] for v in steps.values()) == 2 * E.IMAGES  # every caption once per epoch
    final_live, _ = E.read(out / "tiny.safetensors")
    final_cached, _ = E.read(cached_run.out / "tiny.safetensors")
    assert sorted(final_live) == sorted(final_cached)
    dit = max(float((final_live[k].float() - final_cached[k].float()).abs().max()) for k in final_live)
    print(f"\n--moss_te_live_batch {batch}: worst relative loss / grad-norm difference {worst:.3g}, DiT max abs {dit:.3g}")
    assert dit < DIT_ABS_TOL
    first = [line for line in lines if "first pass (--moss_te_live_batch" in line]
    assert len(first) == 1 and f"4 captions in {calls} layer calls of {groups} rows" in first[0], lines


def test_batch_one_is_still_the_cached_run(env, cached_run):
    """--moss_te_live_batch 1 given explicitly is the default path: bit for bit the cached run."""
    logs = []
    out = env.root / "live_batch_1"
    L.run(env, out, *L.RUN, "--moss_te_live", "--moss_te_live_ahead", "3", "--moss_te_live_batch", "1", logs=logs)
    steps = E.by_step(logs)
    for step in steps:
        for key in ("loss", "grad_norm", "clip_coef", "lr/decay"):
            assert steps[step][key] == cached_run.steps[step][key], (step, key)
    final_live, _ = E.read(out / "tiny.safetensors")
    final_cached, _ = E.read(cached_run.out / "tiny.safetensors")
    assert all(torch.equal(final_live[k], final_cached[k]) for k in final_cached)
