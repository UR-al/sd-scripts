"""End-to-end CPU test of the validation loss of moss_train.py (--moss_ft_val_dataset_config, library/moss/ft_val.py).

The harness of tests/test_moss_ft_te_live_e2e.py (tiny DiT / VAE, a tiny Qwen3.5-like text encoder file both text
paths read, the real caches and loader; 5 training images, GA 2 = windows [0, 1], [2, 3], [4] = 3 steps per epoch, 2
epochs) plus a held-out TOML of 3 other images (batch 2, num_repeats 3, flip_aug, shuffle_caption with "|||",
enable_wildcard, caption_dropout_rate: all of it switched off or made fixed for validation; one multi-line caption).

Checks:
* a run with validation trains bit for bit like the same run without it (every step's loss, grad norm, LR; the final
  DiT; the host pool) - on the cached path, with the live text encoder (whose feed counters are also unchanged) and in
  the Carousel mode: no side effect on weights, optimizer, RNG, DataLoader order or the TE feed;
* val_log.jsonl: one line per validation at step 0, every N steps and the last step, with exactly the contract's keys,
  images_seen = the images consumed (0, 4, 7, 10); the TensorBoard scalars val/loss and val/loss_s<sigma> at those
  steps, logged after that step's training scalars;
* determinism: the validations at the steps two schedules share (N 2 and N 3: steps 0 and 6) are equal, so are the
  cached and the live path's (the live encoder encodes the held-out captions alone, also with --moss_te_live_batch 2);
* resume: a run resumed from the epoch-end state at step 3 validates at 4 and 6 with the uninterrupted run's numbers
  and continues images_seen from the state;
* refusals: validation_split inside the held-out TOML.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_val_e2e.py -q -p no:cacheprovider
"""
import json
import math
import os
import shutil
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import torch
from PIL import Image

import moss_train
import test_moss_ft_te_live_e2e as L
import test_moss_ft_train_e2e as E
from library.moss import ft_args, ft_state, ft_val

CONTRACT_KEYS = ["step", "images_seen", "loss", "loss_by_sigma", "n", "seconds"]
VAL_CAPTIONS = ["a green circle on grey", "cat ||| sitting, window\ncat ||| sitting", "a {red|blue} box ||| small"]
SIGMAS = "0.25,0.5,0.875"
CACHED = ("--cache_text_encoder_outputs", "--cache_text_encoder_outputs_to_disk")
IMAGES_SEEN = {0: 0, 1: 2, 2: 4, 3: 5, 4: 7, 5: 9, 6: 10}  # windows of 2, 2, 1 images per epoch


@pytest.fixture(autouse=True)
def restore_logging():
    """moss_train.train calls setup_logging(reset=True); later caplog tests need pytest's root handlers back."""
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


def write_val_set(root, name="val.toml", extra_dataset_lines=()):
    images = root / "val_images"
    if not images.exists():
        images.mkdir()
        rng = np.random.default_rng(11)
        for index, caption in enumerate(VAL_CAPTIONS):
            pixels = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
            pixels[:, : 6 + 5 * index, (index + 1) % 3] = 40
            Image.fromarray(pixels).save(images / f"val{index}.png")
            (images / f"val{index}.txt").write_text(caption, encoding="utf-8")
    toml = root / name
    toml.write_text("[general]\nenable_bucket = false\ncaption_extension = \".txt\"\n\n"
                    "[[datasets]]\nresolution = 32\nbatch_size = 2\n" + "".join(f"{line}\n" for line in extra_dataset_lines)
                    + "\n"
                    f"  [[datasets.subsets]]\n  image_dir = '{images.as_posix()}'\n  num_repeats = 3\n  flip_aug = true\n"
                    "  shuffle_caption = true\n  keep_tokens_separator = \"|||\"\n  enable_wildcard = true\n"
                    "  caption_dropout_rate = 0.5\n  caption_tag_dropout_rate = 0.5\n", encoding="utf-8")
    return toml


RUN = L.RUN  # mixed_precision no, GA 2, AdamW, LR phases "1:1e-3,1:2e-4" with warmup 2, max_grad_norm 1.0


def val_flags(toml, every=2):
    return ("--moss_ft_val_dataset_config", str(toml), "--moss_ft_val_every_n_steps", str(every),
            "--moss_ft_val_sigmas", SIGMAS)


def train(env, out, *extra):
    """L.run with every log record and the Run object captured."""
    logs, runs, loaded = [], [], []
    with pytest.MonkeyPatch.context() as mp:
        real = moss_train._train_loop

        def capture(run, *args, **kwargs):
            runs.append(run)
            return real(run, *args, **kwargs)

        mp.setattr(moss_train, "_train_loop", capture)
        args = L.run(env, out, *extra, logs=logs, loaded=loaded)
    model, pool, slots = loaded[0]
    return types.SimpleNamespace(out=out, args=args, logs=logs, steps=E.by_step(logs), run=runs[0], model=model,
                                 pool=pool, slots=slots, val=_val_lines(out))


def _val_lines(out):
    path = out / ft_val.VAL_LOG_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _same_training(a, b, keys=("loss", "grad_norm", "clip_coef", "lr/decay")):
    assert sorted(a.steps) == sorted(b.steps) == list(range(1, 2 * E.STEPS_PER_EPOCH + 1))
    for step in a.steps:
        for key in keys:
            assert a.steps[step][key] == b.steps[step][key], (step, key)
    if a.pool is not None:
        for role in a.pool.roles:
            assert torch.equal(a.pool.roles[role], b.pool.roles[role]), role
    for (name, p), (_, q) in zip(a.model.named_parameters(), b.model.named_parameters()):
        assert torch.equal(p.detach(), q.detach()), name
    L._same_dit(a.out, b.out)


def _by_step(lines):
    return {line["step"]: line for line in lines}


# ---------------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    env = L.make_env(tmp_path_factory.mktemp("moss_ft_val_e2e"))
    env.val = write_val_set(env.root)
    return env


@pytest.fixture(scope="module")
def base(env):
    return train(env, env.root / "base", *RUN, *CACHED)


@pytest.fixture(scope="module")
def cached(env):
    return train(env, env.root / "val_cached", *RUN, *CACHED, *val_flags(env.val), "--save_state",
                 "--save_every_n_steps", "1", "--save_last_n_steps_state", "100")


# ---------------------------------------------------------------------------------------------------- tests
def test_validation_does_not_change_training(base, cached):
    _same_training(base, cached)
    assert base.val == [] and base.run.validator is None


def test_val_log_lines_follow_the_contract(env, cached):
    lines = cached.val
    assert [line["step"] for line in lines] == [0, 2, 4, 6]
    for line in lines:
        assert list(line) == CONTRACT_KEYS
        assert line["images_seen"] == IMAGES_SEEN[line["step"]]
        assert line["n"] == len(VAL_CAPTIONS)  # every held-out image once (num_repeats 3 ignored)
        assert list(line["loss_by_sigma"]) == ["0.25", "0.5", "0.875"]
        assert all(isinstance(v, float) and math.isfinite(v) and v > 0 for v in line["loss_by_sigma"].values())
        assert line["loss"] == sum(line["loss_by_sigma"].values()) / 3
        assert isinstance(line["seconds"], float) and line["seconds"] >= 0
    assert lines[0]["loss"] != lines[-1]["loss"]  # the weights moved
    assert [r.step for r in cached.run.validator.records] == [0, 2, 4, 6]
    # the held-out captions: fixed (first line, first wildcard option, the separator split), cached next to the TOML
    validator = cached.run.validator
    assert [item.caption for item in validator.items] == ["a green circle on grey", "cat, sitting, window",
                                                          "a red box, small"]
    assert [len(batch) for batch in validator.batches] == [2, 1]
    assert all(item.info.latents_npz and os.path.isfile(item.info.latents_npz) for item in validator.items)
    assert not any(getattr(item.info, "latents_flipped", None) is not None for item in validator.items)
    cache_dir = env.root / f"val{ft_val.VAL_TE_SUFFIX}"
    assert len(list(cache_dir.glob("*.safetensors"))) == len(VAL_CAPTIONS)


def test_tensorboard_scalars_come_after_the_steps_logs(cached):
    order = []
    for index, (step, values) in enumerate(cached.logs):
        if "val/loss" in values:
            line = _by_step(cached.val)[step]
            assert values["val/loss"] == line["loss"]
            for key, value in line["loss_by_sigma"].items():
                assert values[f"val/loss_s{key}"] == value
            order.append((step, index))
    assert [step for step, _ in order] == [0, 2, 4, 6]
    first_train = min(i for i, (s, v) in enumerate(cached.logs) if "loss" in v)
    assert order[0][1] < first_train  # step 0: before any update
    for step, index in order[1:]:
        train_index = next(i for i, (s, v) in enumerate(cached.logs) if s == step and "loss" in v)
        assert train_index < index, step


def test_two_schedules_give_the_same_numbers_where_they_meet(env, base, cached):
    every3 = train(env, env.root / "val_every3", *RUN, *CACHED, *val_flags(env.val, every=3))
    _same_training(base, every3)
    assert [line["step"] for line in every3.val] == [0, 3, 6]
    a, b = _by_step(cached.val), _by_step(every3.val)
    for step in (0, 6):
        assert a[step]["loss_by_sigma"] == b[step]["loss_by_sigma"] and a[step]["loss"] == b[step]["loss"]
    assert b[3]["images_seen"] == 5
    only_ends = train(env, env.root / "val_ends", *RUN, *CACHED, *val_flags(env.val, every=0))
    assert [line["step"] for line in only_ends.val] == [0, 6]
    assert _by_step(only_ends.val)[6]["loss"] == a[6]["loss"]


def test_live_text_encoder_path(env, cached):
    """--moss_te_live: the held-out captions are encoded by the live encoder at start-up (another TOML stem: another
    cache folder, so nothing is read from the cached run's), each alone - the same rows as the provider's, so the same
    validation numbers. Training and the TE feed's counters equal the live run without validation."""
    live_toml = write_val_set(env.root, "val_live.toml")
    live_val = train(env, env.root / "live_val", *RUN, "--moss_te_live", "--moss_te_live_ahead", "3",
                     *val_flags(live_toml))
    live_plain = train(env, env.root / "live_plain", *RUN, "--moss_te_live", "--moss_te_live_ahead", "3")
    _same_training(live_plain, live_val, keys=("loss", "grad_norm", "clip_coef", "lr/decay", "te/pass_images"))
    for name in ("pass_count", "rows_total"):
        assert getattr(live_val.run.te_feed, name) == getattr(live_plain.run.te_feed, name), name
    assert len(list((env.root / f"val_live{ft_val.VAL_TE_SUFFIX}").glob("*.safetensors"))) == len(VAL_CAPTIONS)
    assert [line["step"] for line in live_val.val] == [0, 2, 4, 6]
    for mine, theirs in zip(live_val.val, cached.val):  # live training == cached training (the te_live e2e)
        assert mine["loss_by_sigma"] == theirs["loss_by_sigma"], mine["step"]
    # batched layer calls for training: the held-out captions are still encoded alone, so step 0 is unchanged
    b2_toml = write_val_set(env.root, "val_live_b2.toml")
    batched = train(env, env.root / "live_b2", *RUN, "--moss_te_live", "--moss_te_live_ahead", "3",
                    "--moss_te_live_batch", "2", *val_flags(b2_toml, every=0))
    assert batched.val[0]["step"] == 0 and batched.val[0]["loss_by_sigma"] == cached.val[0]["loss_by_sigma"]


def test_carousel_with_validation(env):
    flags = ("--mixed_precision", "no", "--gradient_accumulation_steps", "2", "--optimizer_type", "AdamW",
             "--optimizer_args", "weight_decay=0.01", "--learning_rate", "1e-3", "--max_train_epochs", "2",
             "--max_grad_norm", "1.0", ft_args.carousel_flag(), ft_args.carousel_flag("active"), "2",
             ft_args.carousel_flag("period"), "2", ft_args.carousel_flag("seed"), "7",
             ft_args.carousel_flag("always"), "head", *CACHED)
    plain = train(env, env.root / "carousel_plain", *flags)
    with_val = train(env, env.root / "carousel_val", *flags, *val_flags(env.val))
    _same_training(plain, with_val, keys=("loss", "grad_norm", "lr/decay", "carousel/lowest_active_block"))
    assert [line["step"] for line in with_val.val] == [0, 2, 4, 6]
    assert all(math.isfinite(line["loss"]) for line in with_val.val)


def test_resume_continues_the_validation_log(env, cached):
    state = cached.out / "tiny-step00000003-state"
    with open(state / ft_state.TRAIN_STATE_FILE, encoding="utf-8") as f:
        assert json.load(f)["images_seen"] == 5
    resumed = train(env, env.root / "val_resumed", *RUN, *CACHED, *val_flags(env.val), "--resume", str(state))
    assert sorted(resumed.steps) == [4, 5, 6]
    a = _by_step(cached.val)
    assert [line["step"] for line in resumed.val] == [4, 6]  # no step-0 validation when resuming
    for line in resumed.val:
        assert line["images_seen"] == IMAGES_SEEN[line["step"]]
        assert line["loss_by_sigma"] == a[line["step"]]["loss_by_sigma"], line["step"]
    # a state written before images_seen was recorded: the estimate from the epochs done
    old = env.root / "old_state" / "tiny-step00000003-state"
    shutil.copytree(state, old)
    with open(old / ft_state.TRAIN_STATE_FILE, encoding="utf-8") as f:
        data = json.load(f)
    data.pop("images_seen")
    with open(old / ft_state.TRAIN_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    complete = json.loads((old / ft_state.COMPLETE_FILE).read_text(encoding="utf-8"))
    complete["files"][ft_state.TRAIN_STATE_FILE] = os.path.getsize(old / ft_state.TRAIN_STATE_FILE)
    (old / ft_state.COMPLETE_FILE).write_text(json.dumps(complete), encoding="utf-8")
    estimated = train(env, env.root / "val_resumed_old", *RUN, *CACHED, *val_flags(env.val), "--resume", str(old))
    assert [line["images_seen"] for line in estimated.val] == [7, 10]


def test_validation_split_in_the_held_out_toml_is_refused(env):
    bad = write_val_set(env.root, "val_split.toml", ["validation_split = 0.5"])
    with pytest.raises(ValueError, match="held-out set itself"):
        L.run(env, env.root / "refused", *RUN, *CACHED, "--moss_ft_val_dataset_config", str(bad))
    assert not (env.root / "refused").exists()
