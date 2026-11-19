"""End-to-end CPU test of moss_train.py in the Carousel mode (docs/moss_ft_carousel_plan.md 6, item 26).

The harness of tests/test_moss_ft_train_e2e.py (tiny DiT / VAE / text encoder stand-ins, real caches, real loader):
5 images, GA 2 = windows [0, 1], [2, 3], [4] = 3 optimizer steps per epoch, 2 epochs; --moss_carousel with K 2 of the
tiny model's 5 units, fp32 compute. adamw and came, a switch every 2 steps (steps 2, 4, 6) or every 3 (steps 3, 6):
the uninterrupted run against a resume from the epoch-end state at step 3 (inside a period with 2, at a switch
boundary with 3) gives the same logs, final DiT, masters and compute copies; the carousel scalars are logged. A
mid-epoch state (step 4) resumes with a new shuffle of the epoch (kohya's behaviour, as the other modes): its run
continues in the saved set and period position with finite losses.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_carousel_train_e2e.py -q -p no:cacheprovider
"""
import json
import math
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

from library.moss import ft_args, ft_carousel, ft_state
from test_moss_ft_train_e2e import STEPS_PER_EPOCH, by_step, read, run, write_files

C = ft_args.CAROUSEL


def flags(kind, period=2):
    optimizer = {"adamw": ("--optimizer_type", "AdamW", "--optimizer_args", "weight_decay=0.01"),
                 "came": ("--optimizer_type", "CAME", "--optimizer_args", "betas=(0.9,0.999,0.9999)")}[kind]
    return ("--mixed_precision", "no", "--gradient_accumulation_steps", "2", *optimizer, "--learning_rate", "1e-3",
            "--max_train_epochs", "2", "--max_grad_norm", "1.0", "--save_state", "--save_every_n_steps", "1",
            "--save_last_n_steps_state", "100", ft_args.carousel_flag(), ft_args.carousel_flag("active"), "2",
            ft_args.carousel_flag("period"), str(period), ft_args.carousel_flag("seed"), "7")


@pytest.fixture(autouse=True)
def restore_logging():
    """moss_train.train calls setup_logging(reset=True): it drops pytest's root handlers and sets the root level to
    INFO, which later tests in the same process (caplog checks of test_moss_ft_state.py) would see."""
    import logging

    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return write_files(tmp_path_factory.mktemp("moss_ft_carousel_e2e"))


def _run(env, out, kind, *extra, period=2):
    loaded, logs = [], []
    args = run(env, out, *flags(kind, period), *extra, loaded=loaded, logs=logs)
    model, pool, slots = loaded[0]
    return types.SimpleNamespace(out=out, args=args, logs=logs, steps=by_step(logs), model=model, pool=pool,
                                 slots=slots)


@pytest.mark.parametrize("kind,period", [("adamw", 2), ("came", 2), ("adamw", 3), ("came", 3)])
def test_carousel_trains_saves_and_resumes_exactly(env, kind, period):
    a = _run(env, env.root / f"{kind}_{period}_a", kind, period=period)
    assert sorted(a.steps) == list(range(1, 2 * STEPS_PER_EPOCH + 1))
    for step, values in a.steps.items():
        assert math.isfinite(values["loss"]) and math.isfinite(values["grad_norm"]), (step, values)
        assert values[f"{C}/active_units"] == 2.0 and values[f"{C}/trainable_params"] > 0
        assert f"{C}/lowest_active_block" in values and "time/optimizer_step" in values
        # a switch at every period boundary except after the last step (nothing would train the new set)
        assert (f"{C}/switch_s" in values) == (step % period == 0 and step < 2 * STEPS_PER_EPOCH), step
        assert f"{C}/grad_floor" in values
    final_a, meta = read(a.out / "tiny.safetensors")
    assert meta["ss_moss_ft_mode"] == C
    info = json.loads(meta[f"ss_moss_ft_{C}"])
    assert info["optimizer"] == kind and info["active"] == 2 and info["period"] == period
    state = a.out / "tiny-step00000004-state"
    rotation = json.load(open(state / ft_carousel.ROTATION_FILE, encoding="utf-8"))
    assert rotation["optimizer"] == kind and len(rotation["active"]) == 2 and len(rotation["units"]) == 5
    for slot in a.slots:  # the final save: every master (flushed) rounded to bf16
        assert torch.equal(final_a[slot.name], a.pool.view("master", slot).view(slot.shape).to(torch.bfloat16)), slot.name

    b = _run(env, env.root / f"{kind}_{period}_mid", kind, "--resume", str(a.out / "tiny-step00000004-state"),
             period=period)
    assert sorted(b.steps) == [5, 6] and all(math.isfinite(v["loss"]) for v in b.steps.values())
    assert all(b.steps[s][f"{C}/active_units"] == 2.0 for s in b.steps)
    for cut in (3,):
        b = _run(env, env.root / f"{kind}_{period}_b{cut}", kind, "--resume", str(a.out / f"tiny-step{cut:08d}-state"),
                 period=period)
        assert sorted(b.steps) == list(range(cut + 1, 2 * STEPS_PER_EPOCH + 1))
        for step in b.steps:
            for key in ("loss", "grad_norm", "lr/decay", f"{C}/lowest_active_block"):
                assert b.steps[step][key] == a.steps[step][key], (cut, step, key)
        final_b, _ = read(b.out / "tiny.safetensors")
        assert all(torch.equal(final_a[k], final_b[k]) for k in final_a), cut
        assert torch.equal(a.pool.roles["master"], b.pool.roles["master"])
        for (name, p), (_, q) in zip(a.model.named_parameters(), b.model.named_parameters()):
            assert torch.equal(p.detach(), q.detach()), (cut, name)
        last_a = json.load(open(a.out / "tiny-state" / ft_state.TRAIN_STATE_FILE, encoding="utf-8"))
        last_b = json.load(open(b.out / "tiny-state" / ft_state.TRAIN_STATE_FILE, encoding="utf-8"))
        assert last_a["extra"] == last_b["extra"] and last_a["last_grad_norm"] == last_b["last_grad_norm"]
