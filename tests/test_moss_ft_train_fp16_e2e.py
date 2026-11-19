"""End-to-end CPU test of moss_train.py's loss scaling (docs/moss_ft_te_live_fp16_plan.md 2.3, 2.4) with the harness of
tests/test_moss_ft_train_e2e.py (tiny DiT, random FLUX.2 VAE, 5 images, real caches, host pipeline, LR phases, states).

Accelerate refuses fp16 mixed precision on the CPU, so the run keeps fp32 compute and ft_scaler.make is monkeypatched to
return a DynamicLossScaler (2^10, doubling every 2 clean windows): the loop multiplies each loss by the scale and the
host step divides it out. Checks:
* the scaled run logs the same loss / grad_norm / clip_coef / LR at every step as the unscaled run and ends with the same
  pool, compute copies and DiT file, bit for bit; it logs train/loss_scale (the growth schedule);
* the scaler is in every state's train_state.json; a resume from the epoch-end step state continues the scale exactly
  (same logs, same final state, same loss_scaler entry);
* a state without a scaler resumed by a scaled run starts at the initial scale; a scaled state resumed by an unscaled
  run ignores it (both logged).

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_train_fp16_e2e.py -q -p no:cacheprovider
"""
import logging
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_moss_ft_train_e2e as E  # noqa: E402

from library.moss import ft_scaler  # noqa: E402

SCALES = [1024.0, 1024.0, 2048.0, 2048.0, 4096.0, 4096.0]  # steps 1-6: x2 after every 2 clean windows


def forced_scaler(args):
    return ft_scaler.DynamicLossScaler(init_scale=2.0 ** 10, growth_interval=2)


def run(env, out, *extra, scaled):
    loaded, logs = [], []
    with pytest.MonkeyPatch.context() as mp:
        if scaled:
            mp.setattr(ft_scaler, "make", forced_scaler)
        E.run(env, out, *E.ADAMW, *extra, loaded=loaded, logs=logs)
    model, pool, slots = loaded[0]
    return types.SimpleNamespace(out=out, logs=logs, steps=E.by_step(logs), model=model, pool=pool, slots=slots)


@pytest.fixture(scope="module", autouse=True)
def _restore_logging():
    """train() resets the logging set-up (root level and handlers); later test files expect pytest's."""
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    yield
    root.setLevel(level)
    root.handlers[:] = handlers


# Each straight run gets its own copy of the files: a run that writes the caches draws other random numbers before its
# first step than one that finds them (the two straight runs must start alike; resumes restore the RNG anyway).
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return E.write_files(tmp_path_factory.mktemp("moss_ft_fp16_e2e"))


@pytest.fixture(scope="module")
def plain_env(tmp_path_factory):
    return E.write_files(tmp_path_factory.mktemp("moss_ft_fp16_e2e_plain"))


@pytest.fixture(scope="module")
def plain(plain_env):
    return run(plain_env, plain_env.root / "plain", scaled=False)


@pytest.fixture(scope="module")
def scaled(env):
    return run(env, env.root / "scaled", scaled=True)


def test_scaled_run_equals_the_unscaled_run(env, plain, scaled):
    assert sorted(scaled.steps) == sorted(plain.steps) == list(range(1, 7))
    for step in range(1, 7):
        for key in ("loss", "grad_norm", "clip_coef", "lr/decay", "lr/no_decay", "skipped_params"):
            assert scaled.steps[step][key] == plain.steps[step][key], (step, key)
        assert "train/loss_scale" not in plain.steps[step]
        assert scaled.steps[step]["train/overflow"] == 0.0 and scaled.steps[step]["train/overflow_windows"] == 0.0
    assert [scaled.steps[s]["train/loss_scale"] for s in range(1, 7)] == SCALES
    E._assert_same_training(plain, scaled)
    final_a, _ = E.read(plain.out / "tiny.safetensors")
    final_b, _ = E.read(scaled.out / "tiny.safetensors")
    assert all(torch.equal(final_a[k], final_b[k]) for k in final_a)


def test_states_hold_the_scaler(plain, scaled):
    assert E.train_state(plain.out / "tiny-step00000003-state")["loss_scaler"] is None
    for step, scale in ((1, 1024.0), (2, 2048.0), (3, 2048.0), (6, 8192.0)):
        saved = E.train_state(scaled.out / f"tiny-step{step:08d}-state")["loss_scaler"]
        assert saved["scale"] == scale and saved["growth_interval"] == 2 and saved["overflow_windows"] == 0, step
    assert E.train_state(scaled.out / "tiny-state")["loss_scaler"]["scale"] == 8192.0


def test_resume_continues_the_scale_exactly(env, scaled):
    resumed = run(env, env.root / "scaled_resumed", "--resume", str(scaled.out / "tiny-step00000003-state"),
                  scaled=True)
    assert sorted(resumed.steps) == [4, 5, 6]
    for step in (4, 5, 6):
        for key in ("loss", "grad_norm", "clip_coef", "lr/decay", "train/loss_scale"):
            assert resumed.steps[step][key] == scaled.steps[step][key], (step, key)
    E._assert_same_training(scaled, resumed)
    a, b = E.train_state(scaled.out / "tiny-state"), E.train_state(resumed.out / "tiny-state")
    for key in ("global_step", "last_grad_norm", "skip_streak", "loss_scaler"):
        assert a[key] == b[key], key


def test_cross_resumes_are_logged(env, plain, scaled, monkeypatch):
    lines = []  # train() resets the logging set-up, so the scaler's log lines are taken at the logger
    monkeypatch.setattr(ft_scaler.logger, "info", lambda message, *a, **k: lines.append(str(message)))
    start = run(env, env.root / "plain_to_scaled", "--resume", str(plain.out / "tiny-step00000003-state"), scaled=True)
    assert any("no loss scale" in line for line in lines)
    assert start.steps[4]["train/loss_scale"] == 1024.0  # the initial scale
    assert start.steps[4]["loss"] == plain.steps[4]["loss"]  # still the same training
    lines.clear()
    back = run(env, env.root / "scaled_to_plain", "--resume", str(scaled.out / "tiny-step00000003-state"),
               scaled=False)
    assert any("ignored" in line for line in lines)
    assert "train/loss_scale" not in back.steps[4]
    E._assert_same_training(plain, back)
    assert E.train_state(back.out / "tiny-state")["loss_scaler"] is None
