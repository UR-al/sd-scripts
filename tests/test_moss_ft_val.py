"""CPU unit tests of the validation loss of moss_train.py (library/moss/ft_val.py, ft_loss.moss_ft_val_loss_rows, the
--moss_ft_val_* flags of ft_args, the budget's validation forward).

The tiny MOSS DiT of tests/_moss_ft_common.py, in-memory latents and text rows behind a stand-in dataset group:
* the flags: parse_val_sigmas, the refusals of validate_moss_ft_args, the TOML defaults;
* the fixed parts: noise seeds, first-line captions, the batch plan, the per-(image, sigma) noise;
* the loss: equal to the training loss formula at a fixed sigma (the same noise) bit for bit; per-sigma means;
* determinism: two validations of an unchanged model give identical records; checkpointing and the Carousel's no-grad
  wrappers change nothing;
* no side effects: the adamw_host pipeline's hooks do not fire, master / moments / steps / the loss scaler / every global
  RNG / the compute copies are unchanged; the val_log.jsonl line has exactly the contract's keys;
* the schedule (due), the log file of a fresh run and of a resume, missing latents refused;
* the budget: a validation forward counts only where it exceeds the training peak.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_val.py -q -p no:cacheprovider
"""
import contextlib
import copy
import json
import math
import os
import random
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import torch

import moss_train
from _moss_ft_common import TINY, build_model, make_pipeline, param_groups, trainable
from _moss_ft_carousel_common import make_engine
from library import sd3_train_utils
from library.moss import flow, ft_args, ft_budget, ft_loss, ft_scaler, ft_val, strategy
from library.moss import ft_carousel_units as cu
from library.moss.ft_optim import HostAdamW

CONTRACT_KEYS = ["step", "images_seen", "loss", "loss_by_sigma", "n", "seconds"]


# ---------------------------------------------------------------------------------------------------- stand-ins
def make_args(tmp_path, *extra):
    args = moss_train.setup_parser().parse_args(["--output_dir", str(tmp_path / "out"), *extra])
    args.gradient_checkpointing = True
    return args


class FakeAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.is_main_process = True
        self.logs = []

    def autocast(self):
        return contextlib.nullcontext()

    def log(self, values, step=None):
        self.logs.append((step, dict(values)))


def subset(**kw):
    values = dict(enable_wildcard=False, caption_prefix=None, caption_suffix=None, shuffle_caption=False,
                  token_warmup_step=0, caption_tag_dropout_rate=0.0, secondary_separator=None, keep_tokens=0,
                  caption_separator=",", keep_tokens_separator="", alpha_mask=False)
    values.update(kw)
    return types.SimpleNamespace(**values)


def fake_group(captions, *, batch_size=2, resos=None, sub=None, seed=0, folder="val"):
    """A dataset group of len(captions) images with in-memory latents (128 x h x w)."""
    g = torch.Generator().manual_seed(500 + seed)
    sub = sub or subset()
    data, to_subset = {}, {}
    for i, caption in enumerate(captions):
        reso = (resos or {}).get(i, (64, 64))
        key = os.path.join(folder, f"img{i}.png")
        latents = torch.randn(TINY["latent_channels"], reso[1] // 16, reso[0] // 16, generator=g)
        data[key] = types.SimpleNamespace(image_key=key, absolute_path=key, caption=caption, bucket_reso=reso,
                                          latents=latents, latents_npz=None, alpha_mask=None, is_reg=False)
        to_subset[key] = sub
    dataset = types.SimpleNamespace(image_data=data, image_to_subset=to_subset, prior_loss_weight=1.0, replacements={},
                                    batch_size=batch_size, get_subset_batch_size=lambda s: batch_size)
    return types.SimpleNamespace(datasets=[dataset])


def text_rows_for(group, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(900 + seed)
    rows = {}
    for caption in ft_val.val_texts(group):
        rows[caption] = torch.randn(3 + len(caption) % 5, TINY["text_dim"], generator=g).to(dtype)
    return rows


def make_validator(tmp_path, group, *extra, resume=False):
    args = make_args(tmp_path, "--moss_ft_val_dataset_config", str(tmp_path / "val.toml"), *extra)
    validator = ft_val.Validator(args, group, text_rows_for(group), resume=resume, latents_strategy=object(),
                                 text_encoding_strategy=strategy.MossTextEncodingStrategy())
    return args, validator


def scheduler():
    return sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)


def validate(validator, model, step=0, images_seen=0, accelerator=None):
    return validator.run(accelerator or FakeAccelerator(), model, step, images_seen, stream_dtype=torch.float32,
                         noise_scheduler=scheduler())


CAPTIONS = ["a red square", "cat ||| sitting, window\ncat ||| sitting", "two blue circles", "green\nlong line two",
            "an empty room"]


# ---------------------------------------------------------------------------------------------------- the flags
def test_parse_val_sigmas():
    assert ft_args.parse_val_sigmas(None) == ft_args.VAL_DEFAULT_SIGMAS
    assert ft_args.parse_val_sigmas("") == ft_args.VAL_DEFAULT_SIGMAS
    assert ft_args.VAL_DEFAULT_SIGMAS == tuple((2 * k + 1) / 16 for k in range(8))  # bin midpoints, exact in fp32
    assert ft_args.parse_val_sigmas("0.5, 0.25,1") == (0.5, 0.25, 1.0)
    assert ft_args.parse_val_sigmas([0.1, 0.9]) == (0.1, 0.9)  # a TOML array
    for bad, match in (("0", r"outside \(0, 1\]"), ("1.5", "outside"), ("-0.1", "outside"), ("nan", "outside"),
                       ("0.5,0.5", "twice"), ("a", "not a number"), ("0.5,,0.6", "comma-separated"),
                       ([True], "comma-separated"), (",".join(["0.01"] * 2), "twice"),
                       (",".join(str((i + 1) / 100) for i in range(65)), "1 to 64")):
        with pytest.raises(ValueError, match=match):
            ft_args.parse_val_sigmas(bad)
    with pytest.raises(SystemExit):  # argparse rejects a bad value at parse time
        moss_train.setup_parser().parse_args(["--moss_ft_val_sigmas", "2"])


def _static(args):
    ft_args._check_validation_args(args)


def test_val_flags_need_the_config_and_valid_values(tmp_path, caplog):
    toml = tmp_path / "val.toml"
    toml.write_text("[[datasets]]\n", encoding="utf-8")
    for flags in (("--moss_ft_val_every_n_steps", "5"), ("--moss_ft_val_sigmas", "0.5"), ("--moss_ft_val_seed", "7")):
        with pytest.raises(ValueError, match="only used with --moss_ft_val_dataset_config"):
            _static(make_args(tmp_path, *flags))
    _static(make_args(tmp_path))  # no validation: nothing to check
    with pytest.raises(ValueError, match="is not a file"):
        _static(make_args(tmp_path, "--moss_ft_val_dataset_config", str(tmp_path / "missing.toml")))
    with pytest.raises(ValueError, match="moss_ft_val_every_n_steps must be an integer"):
        _static(make_args(tmp_path, "--moss_ft_val_dataset_config", str(toml), "--moss_ft_val_every_n_steps", "-1"))
    args = make_args(tmp_path, "--moss_ft_val_dataset_config", str(toml))
    args.moss_ft_val_seed = -3
    with pytest.raises(ValueError, match="moss_ft_val_seed"):
        _static(args)
    args = make_args(tmp_path, "--moss_ft_val_dataset_config", str(toml))
    args.moss_ft_val_sigmas = "0.5,7"  # a TOML string reaches the check unconverted
    with pytest.raises(ValueError, match="outside"):
        _static(args)
    args = make_args(tmp_path, "--moss_ft_val_dataset_config", str(toml), "--moss_ft_val_every_n_steps", "10",
                     "--moss_ft_val_sigmas", "0.25,0.75", "--moss_ft_val_seed", "3", "--dataset_config", str(toml))
    with caplog.at_level("INFO"):
        _static(args)
    assert "every 10 optimizer steps" in caplog.text and "0.25, 0.75" in caplog.text
    assert "is the training dataset_config" in caplog.text  # warned, not refused
    assert ft_args.val_enabled(args) and not ft_args.val_enabled(make_args(tmp_path))


def test_kohya_validation_inside_the_training_toml_points_to_the_flag():
    group = types.SimpleNamespace(datasets=[types.SimpleNamespace(validation_split=0.1)])
    with pytest.raises(ValueError, match="--moss_ft_val_dataset_config"):
        ft_args._check_validation_datasets(group, None)
    with pytest.raises(ValueError, match="--moss_ft_val_dataset_config"):
        ft_args._check_validation_datasets(types.SimpleNamespace(datasets=[]),
                                           types.SimpleNamespace(datasets=[object()]))
    args = types.SimpleNamespace(validation_split=0.2)
    with pytest.raises(ValueError, match="--moss_ft_val_dataset_config"):
        ft_args._check_refused_options(args)


# ---------------------------------------------------------------------------------------------------- the plan
def test_captions_follow_training_multiline_and_wildcard_rules():
    plain = subset()
    dataset = types.SimpleNamespace(replacements={})
    assert ft_val.val_caption(dataset, plain, "line one\nline two") == "line one\nline two"
    assert ft_val.val_caption(dataset, plain, None) == ""
    wild = subset(enable_wildcard=True)
    assert ft_val.val_caption(dataset, wild, "a {red|blue} cat\nshort") == "a red cat"
    shuffled = subset(shuffle_caption=True, keep_tokens_separator="|||", caption_tag_dropout_rate=0.5)
    # the separator split joined with ", " (as the shuffled training captions), the tags in their order, nothing dropped
    assert ft_val.val_caption(dataset, shuffled, "1girl ||| b, a, c ||| She smiles.\nshort") == \
        "1girl, b, a, c, She smiles.\nshort"
    for _ in range(3):  # no draw: the same text every time
        assert ft_val.val_caption(dataset, shuffled, "x ||| p, q") == "x, p, q"


def test_plan_batches_by_bucket_in_a_fixed_order():
    group = fake_group(CAPTIONS, batch_size=2, resos={1: (32, 64), 3: (32, 64)})
    items, batches = ft_val.val_plan(group)
    assert [item.caption for item in items] == ["a red square", "cat ||| sitting, window\ncat ||| sitting",
                                                "two blue circles", "green\nlong line two", "an empty room"]
    by_reso = [[items[i].bucket_reso for i in batch] for batch in batches]
    assert all(len(set(resos)) == 1 for resos in by_reso)
    assert sorted(i for batch in batches for i in batch) == list(range(len(CAPTIONS)))  # every image once
    assert [len(batch) for batch in batches] == [2, 2, 1]  # 32x64: 2 images; 64x64: 3 images -> 2 + 1
    assert ft_val.val_batch_shapes(group) == [(8, 2), (16, 1), (16, 2)]
    again = ft_val.val_plan(fake_group(CAPTIONS, batch_size=2, resos={1: (32, 64), 3: (32, 64)}))
    assert again[1] == batches and [i.name for i in again[0]] == [i.name for i in items]


def test_noise_seed_depends_on_seed_name_and_sigma_only():
    a = ft_val.noise_seed(1234, "img0.png", "0.5")
    assert a == ft_val.noise_seed(1234, "img0.png", "0.5") and 0 <= a < 2 ** 63
    assert len({a, ft_val.noise_seed(1235, "img0.png", "0.5"), ft_val.noise_seed(1234, "img1.png", "0.5"),
                ft_val.noise_seed(1234, "img0.png", "0.25")}) == 4
    group = fake_group(["x", "y"])
    items, _ = ft_val.val_plan(group)
    assert [item.name for item in items] == ["img0.png", "img1.png"]
    twins = fake_group(["x"], folder="a")
    twins.datasets[0].image_data.update(fake_group(["y"], folder="b").datasets[0].image_data)
    twins.datasets[0].image_to_subset.update({k: subset() for k in twins.datasets[0].image_data})
    names = sorted(item.name for item in ft_val.val_plan(twins)[0])
    assert names == ["img0.png", "img0.png#1"]  # the same file name in two folders: distinct noise


def test_missing_latents_are_refused(tmp_path):
    group = fake_group(["x", "y"])
    info = next(iter(group.datasets[0].image_data.values()))
    info.latents = None
    info.latents_npz = str(tmp_path / "gone_moss.npz")
    with pytest.raises(ValueError, match="no cached latents"):
        make_validator(tmp_path, group)
    info.latents_npz = None
    with pytest.raises(ValueError, match="no cached latents"):
        make_validator(tmp_path, group)


# ---------------------------------------------------------------------------------------------------- the loss
class FixedSampler:
    def __init__(self, sigma):
        self.sigma = sigma

    def sample(self, batch, device):
        return torch.full((batch,), self.sigma, device=device)


@pytest.mark.parametrize("extra", [(), ("--weighting_scheme", "cosmap", "--min_snr_gamma", "5"),
                                   ("--loss_type", "huber", "--huber_schedule", "exponential")])
def test_val_rows_are_the_training_loss_at_a_fixed_sigma(tmp_path, extra):
    """moss_ft_noise_and_loss with a sampler pinned to sigma and the noise its randn_like draws == the mean of
    moss_ft_val_loss_rows with that noise, bit for bit (the same chain without the draws)."""
    args = make_args(tmp_path, "--timestep_sampling", "moss", *extra)
    model = build_model(seed=3, ckpt="plain")
    accelerator = FakeAccelerator()
    encoding = strategy.MossTextEncodingStrategy()
    g = torch.Generator().manual_seed(4)
    latents = torch.randn(2, TINY["latent_channels"], 4, 4, generator=g)
    rows = [torch.randn(5, TINY["text_dim"], generator=g), torch.randn(3, TINY["text_dim"], generator=g)]
    from library.moss.te_live_feed import live_outputs

    batch = {"latents": latents, "text_encoder_outputs_list": live_outputs(rows), "loss_weights": torch.ones(2),
             "alpha_masks": None}
    for sigma in (0.0625, 0.5, 0.9375):
        torch.manual_seed(77)
        train = ft_loss.moss_ft_noise_and_loss(args, model, batch, accelerator, FixedSampler(sigma), scheduler(),
                                               torch.float32, encoding)
        torch.manual_seed(77)
        noise = torch.randn(latents.shape, dtype=torch.float32)
        with torch.no_grad():
            val = ft_loss.moss_ft_val_loss_rows(args, model, batch, accelerator, noise, sigma, scheduler(),
                                                torch.float32, encoding)
        assert val.dtype == torch.float64 and val.shape == (2,) and not val.requires_grad
        assert torch.equal(val.mean(), train.detach()), (sigma, float(val.mean()), float(train))
    with pytest.raises(ValueError, match="does not match"):
        ft_loss.moss_ft_val_loss_rows(args, model, batch, accelerator, noise[:1], 0.5, scheduler(), torch.float32,
                                      encoding)


def test_record_per_sigma_and_jsonl_format(tmp_path):
    group = fake_group(CAPTIONS)
    args, validator = make_validator(tmp_path, group, "--moss_ft_val_sigmas", "0.25,0.5,0.75")
    model = build_model(seed=1, ckpt="plain")
    accelerator = FakeAccelerator()
    record = validate(validator, model, step=0, images_seen=0, accelerator=accelerator)
    assert record.n == len(CAPTIONS) and list(record.loss_by_sigma) == ["0.25", "0.5", "0.75"]
    assert record.loss == pytest.approx(sum(record.loss_by_sigma.values()) / 3, rel=0, abs=0)
    # per sigma: the mean over the images of the per-image losses - recomputed over the same batches (exact), and image
    # by image (close: other matrix shapes and text padding round differently in fp32)
    encoding = strategy.MossTextEncodingStrategy()
    for k, (key, sigma) in enumerate(zip(validator.keys, validator.sigmas)):
        total = alone = 0.0
        for indices in validator.batches:
            batch = validator.batch(indices, accelerator.device)
            noise = torch.stack([validator.noise(i, k, batch["latents"].shape[1:]) for i in indices])
            with torch.no_grad():
                total += float(ft_loss.moss_ft_val_loss_rows(args, model, batch, accelerator, noise, sigma, scheduler(),
                                                             torch.float32, encoding).sum())
        for index in range(validator.n):
            batch = validator.batch([index], accelerator.device)
            noise = validator.noise(index, k, batch["latents"].shape[1:]).unsqueeze(0)
            with torch.no_grad():
                alone += float(ft_loss.moss_ft_val_loss_rows(args, model, batch, accelerator, noise, sigma, scheduler(),
                                                             torch.float32, encoding)[0])
        assert record.loss_by_sigma[key] == total / validator.n
        assert record.loss_by_sigma[key] == pytest.approx(alone / validator.n, rel=1e-6)
    lines = (tmp_path / "out" / ft_val.VAL_LOG_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    line = json.loads(lines[0])
    assert list(line) == CONTRACT_KEYS
    assert (line["step"], line["images_seen"], line["n"]) == (0, 0, len(CAPTIONS))
    assert isinstance(line["loss"], float) and isinstance(line["seconds"], float)
    assert line["loss_by_sigma"] == record.loss_by_sigma and line["loss"] == record.loss
    step, values = accelerator.logs[-1]
    assert step == 0 and values == {"val/loss": record.loss, "val/loss_s0.25": record.loss_by_sigma["0.25"],
                                    "val/loss_s0.5": record.loss_by_sigma["0.5"],
                                    "val/loss_s0.75": record.loss_by_sigma["0.75"]}


def _strict_json(line):
    def refuse(token):
        raise ValueError(f"not JSON: {token}")

    return json.loads(line, parse_constant=refuse)


def test_non_finite_losses_are_written_as_null(tmp_path, monkeypatch, caplog):
    """An fp16 overflow at one sigma: the line stays valid JSON (null, not NaN / Infinity), with the contract's keys;
    the finite sigmas keep their values; the record in memory, TensorBoard and the warning keep the non-finite value."""
    group = fake_group(CAPTIONS[:2])
    args, validator = make_validator(tmp_path, group, "--moss_ft_val_sigmas", "0.25,0.5,0.75")
    real = ft_loss.moss_ft_val_loss_rows
    poison = {0.5: float("nan"), 0.75: float("inf")}

    def rows(*a, **kw):
        out = real(*a, **kw)
        sigma = a[5]
        return out if sigma not in poison else torch.full_like(out, poison[sigma])

    monkeypatch.setattr(ft_loss, "moss_ft_val_loss_rows", rows)
    accelerator = FakeAccelerator()
    with caplog.at_level("WARNING"):
        record = validate(validator, build_model(seed=1, ckpt="plain"), step=3, images_seen=6, accelerator=accelerator)
    assert math.isnan(record.loss_by_sigma["0.5"]) and math.isinf(record.loss_by_sigma["0.75"])
    assert "non-finite loss at sigma 0.5, 0.75" in caplog.text
    line = _strict_json((tmp_path / "out" / ft_val.VAL_LOG_FILE).read_text(encoding="utf-8").splitlines()[0])
    assert list(line) == CONTRACT_KEYS and list(line["loss_by_sigma"]) == ["0.25", "0.5", "0.75"]
    assert line["loss"] is None and line["loss_by_sigma"]["0.5"] is None and line["loss_by_sigma"]["0.75"] is None
    assert line["loss_by_sigma"]["0.25"] == record.loss_by_sigma["0.25"] and math.isfinite(line["loss_by_sigma"]["0.25"])
    assert (line["step"], line["images_seen"], line["n"]) == (3, 6, 2)
    assert ft_val.read_val_log(str(tmp_path / "out" / ft_val.VAL_LOG_FILE))[0]["loss"] is None
    assert math.isnan(accelerator.logs[-1][1]["val/loss"])  # TensorBoard shows NaN itself
    # a finite validation afterwards: plain floats again
    monkeypatch.setattr(ft_loss, "moss_ft_val_loss_rows", real)
    validate(validator, build_model(seed=1, ckpt="plain"), step=4, images_seen=8)
    line = _strict_json((tmp_path / "out" / ft_val.VAL_LOG_FILE).read_text(encoding="utf-8").splitlines()[1])
    assert isinstance(line["loss"], float) and all(isinstance(v, float) for v in line["loss_by_sigma"].values())
    # as_json itself: a dump that would need NaN is refused, not written
    bad = ft_val.ValRecord(step=0, images_seen=0, loss=float("-inf"), loss_by_sigma={"0.5": float("nan")}, n=1,
                           seconds=float("nan"))
    assert _strict_json(json.dumps(bad.as_json(), allow_nan=False)) == {
        "step": 0, "images_seen": 0, "loss": None, "loss_by_sigma": {"0.5": None}, "n": 1, "seconds": None}


def test_console_line(tmp_path, caplog):
    _, validator = make_validator(tmp_path, fake_group(CAPTIONS[:2]))
    with caplog.at_level("INFO"):
        record = validate(validator, build_model(seed=1), step=7, images_seen=12)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("VAL step")]
    assert lines == [f"VAL step 7: loss {record.loss:.6f} (2 images, {record.seconds:.1f} s)"]


# ---------------------------------------------------------------------------------------------------- determinism
def test_two_validations_of_an_unchanged_model_are_identical(tmp_path):
    group = fake_group(CAPTIONS, resos={2: (32, 64)})
    _, validator = make_validator(tmp_path, group)
    model = build_model(seed=2)
    first = validate(validator, model, step=0)
    torch.randn(100)  # the global RNG moved on: the validation noise does not use it
    random.random()
    second = validate(validator, model, step=1)
    assert first.loss_by_sigma == second.loss_by_sigma and first.loss == second.loss
    # another Validator over the same set (a new run): the same numbers
    _, other = make_validator(tmp_path / "b", fake_group(CAPTIONS, resos={2: (32, 64)}))
    assert validate(other, model).loss_by_sigma == first.loss_by_sigma
    # another seed: other noise, another loss
    _, reseeded = make_validator(tmp_path / "c", fake_group(CAPTIONS, resos={2: (32, 64)}), "--moss_ft_val_seed", "9")
    assert validate(reseeded, model).loss != first.loss


@pytest.mark.parametrize("ckpt", ["plain", "unsloth"])
def test_checkpointing_does_not_change_the_validation(tmp_path, ckpt):
    group = fake_group(CAPTIONS[:3])
    _, validator = make_validator(tmp_path, group)
    with_ckpt = build_model(seed=5, ckpt=ckpt)
    without = build_model(seed=5, ckpt="none")
    assert validate(validator, with_ckpt).loss_by_sigma == validate(validator, without).loss_by_sigma
    assert all(p.grad is None for p in with_ckpt.parameters())


# ---------------------------------------------------------------------------------------------------- side effects
def _rng_states():
    return (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone())


def _same_rng(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and torch.equal(a[2], b[2])


def test_validation_leaves_the_host_pipeline_alone(tmp_path):
    """adamw_host: one trained window, then two validations: no hook fires, the pool (master, exp_avg, exp_avg_sq), the
    step counters, the compute copies, the loss scaler and every global RNG stay as they were, no parameter gets a
    gradient; the validation of the trained weights differs from the untrained one."""
    model = build_model(seed=7, ckpt="plain")
    named = trainable(model)
    optimizer = HostAdamW(param_groups(named), lr=1e-3)
    pipe, pool, slots = make_pipeline(named, optimizer, ga=1)
    group = fake_group(CAPTIONS[:3])
    _, validator = make_validator(tmp_path, group)
    try:
        before_training = validate(validator, model, step=0)
        from _moss_ft_common import loss_of, make_batch
        from library.moss import ft_host

        pipe.begin_window(1, ft_host.StepContext.snapshot(optimizer, step_index=0))
        pipe.begin_microstep(0)
        loss_of(model, make_batch(0)).backward()
        stats = pipe.end_microstep()
        assert stats is not None and stats.stepped_params > 0
        roles = {role: tensor.clone() for role, tensor in pool.roles.items()}
        steps = pool.steps.clone()
        copies = [p.detach().clone() for p in model.parameters()]
        hooks = list(pipe._hook_counts)
        scaler = ft_scaler.DynamicLossScaler()
        scaler_state = scaler.state_dict()
        rng = _rng_states()
        first = validate(validator, model, step=1, images_seen=1)
        second = validate(validator, model, step=2, images_seen=1)
        assert first.loss_by_sigma == second.loss_by_sigma and first.loss != before_training.loss
        assert _same_rng(rng, _rng_states())
        assert list(pipe._hook_counts) == hooks
        assert all(torch.equal(roles[role], pool.roles[role]) for role in roles)
        assert torch.equal(steps, pool.steps)
        assert all(torch.equal(a, p.detach()) for a, p in zip(copies, model.parameters()))
        assert all(p.grad is None for p in model.parameters())
        assert scaler.state_dict() == scaler_state
        # the pipeline still trains after a validation
        pipe.begin_window(1, ft_host.StepContext.snapshot(optimizer, step_index=1))
        pipe.begin_microstep(0)
        loss_of(model, make_batch(1)).backward()
        assert pipe.end_microstep().stepped_params == stats.stepped_params
    finally:
        pipe.close()


@pytest.mark.parametrize("always", ["head", "small"])
def test_validation_under_the_carousel_wrappers(tmp_path, always):
    """Carousel: K = 2 of the 5 units active, the blocks below the floor wrapped in torch.no_grad() (head). A
    validation equals that of a plain copy of the same weights, fires no hook and leaves masters, compute copies and the
    active set alone."""
    model = build_model(seed=8, ckpt="plain")
    engine = make_engine(model, k=2, always=always, period=2, seed=13)
    try:
        engine.activate(0)  # moss_train activates at the first window; units 3, 4 = joint blocks 1, 2
        group = fake_group(CAPTIONS[:3])
        _, validator = make_validator(tmp_path, group)
        wrapped = [getattr(m.__dict__.get("forward"), "_carousel", None) for m in [*model.layers, model.text_embedder]]
        if always == "head":  # joint block 0 and the text embedder run under no_grad
            assert wrapped == ["no_grad", None, None, "no_grad"]
        plain = copy.deepcopy(model)
        cu.clear_grad_floor(plain)
        for p in plain.parameters():
            p.requires_grad_(True)
        active = list(engine.active)
        masters = engine.pool.roles["master"].clone()
        copies = [p.detach().clone() for p in model.parameters()]
        record = validate(validator, model)
        assert record.loss_by_sigma == validate(validator, plain).loss_by_sigma
        assert list(engine.active) == active and torch.equal(masters, engine.pool.roles["master"])
        assert all(torch.equal(a, p.detach()) for a, p in zip(copies, model.parameters()))
        assert all(p.grad is None for p in model.parameters())
    finally:
        engine.close()


def test_last_text_tokens_is_restored(tmp_path):
    model = build_model(seed=1)
    model.last_text_tokens = 123
    _, validator = make_validator(tmp_path, fake_group(CAPTIONS[:2]))
    validate(validator, model)
    assert model.last_text_tokens == 123


# ---------------------------------------------------------------------------------------------------- schedule, log
def test_due_schedule(tmp_path):
    _, every = make_validator(tmp_path, fake_group(["x"]), "--moss_ft_val_every_n_steps", "3")
    assert [s for s in range(1, 11) if every.due(s, 10)] == [3, 6, 9, 10]
    _, ends = make_validator(tmp_path / "b", fake_group(["x"]))
    assert [s for s in range(1, 11) if ends.due(s, 10)] == [10]


def test_a_fresh_run_keeps_the_old_log_and_a_resume_appends(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    old = out / ft_val.VAL_LOG_FILE
    old.write_text('{"step": 0}\n', encoding="utf-8")
    _, validator = make_validator(tmp_path, fake_group(["x"]))
    kept = [name for name in os.listdir(out) if name.startswith("val_log.") and name != ft_val.VAL_LOG_FILE]
    assert len(kept) == 1 and (out / kept[0]).read_text(encoding="utf-8") == '{"step": 0}\n'
    assert not old.exists()
    validate(validator, build_model(seed=1), step=0)
    _, resumed = make_validator(tmp_path, fake_group(["x"]), resume=True)
    validate(resumed, build_model(seed=1), step=4, images_seen=8)
    records = [json.loads(line) for line in old.read_text(encoding="utf-8").splitlines()]
    assert [r["step"] for r in records] == [0, 4] and records[1]["images_seen"] == 8
    assert [r["step"] for r in ft_val.read_val_log(str(old))] == [0, 4]


def test_rows_alone_restores_the_live_batch():
    encoder = types.SimpleNamespace(batch_rows=8)
    with ft_val._rows_alone(encoder):
        assert encoder.batch_rows == 1
    assert encoder.batch_rows == 8


def test_preserved_rng_restores_every_generator():
    state = _rng_states()
    with ft_val.preserved_rng():
        random.random()
        np.random.rand()
        torch.rand(3)
    assert _same_rng(state, _rng_states())


# ---------------------------------------------------------------------------------------------------- budget
def test_validation_forward_counts_only_above_the_training_peak():
    small = ft_budget.gpu_budget(4096, 1, val_shapes=[(4096, 1)])
    base = ft_budget.gpu_budget(4096, 1)
    assert small.val_forward > 0 and not small.val_raises_peak and small.allocated == base.allocated
    big = ft_budget.gpu_budget(4096, 1, val_shapes=[(4096, 1), (16384, 2)])
    assert big.val_batch == 2 and big.val_image_tokens == 16384 and big.val_raises_peak
    assert big.allocated == big.val_step + big.te_held > base.allocated
    # ~3x the training batch's tokens before it matters (fp32 stream, batch 1 at 1024^2: 4396 joint tokens)
    per_token = ft_budget.val_forward_bytes(4096, 1) / (4096 + ft_budget.DEFAULT_TEXT_TOKENS)
    assert 0.25e6 < per_token < 0.35e6
    unsloth = ft_budget.gpu_budget(4096, 1, unsloth_offload=True, val_shapes=[(4096, 2)])
    assert unsloth.val_raises_peak  # without the block inputs on the device a double batch can exceed it
    cb_without = ft_budget.carousel_budget("adamw", base)
    cb_with = ft_budget.carousel_budget("adamw", big)
    assert cb_with.activations > cb_without.activations and cb_with.k_budget <= cb_without.k_budget
    assert ft_budget.carousel_budget("adamw", small).activations == cb_without.activations


def test_startup_report_counts_the_held_out_set(tmp_path):
    """The start-up report with the held-out set: a line for its forward (counted only above the training peak), its
    text outputs (and in-memory latents) in the host total, its latents to write in the disk estimate."""
    from test_moss_ft_args import FakeGroup

    args = make_args(tmp_path, "--cache_latents", "--cache_latents_to_disk", "--cache_text_encoder_outputs",
                     "--cache_text_encoder_outputs_to_disk", "--moss_stream_fp32", "--gradient_accumulation_steps", "2")
    train_group = FakeGroup(tmp_path, images=16, batch_size=1)
    plain = ft_budget.startup_report(args, train_group, "adamw_host", exists=lambda path: True)
    same = fake_group(["a", "b", "c"], batch_size=1, resos={i: (1024, 1024) for i in range(3)})
    report = ft_budget.startup_report(args, train_group, "adamw_host", exists=lambda path: True,
                                      val_dataset_group=same)
    line = next(line for line in report.lines if "validation forward" in line)
    assert "not added" in line and report.gpu.allocated == plain.gpu.allocated
    te_rows = 3 * ft_budget.te_cache_bytes(1)
    assert report.host.memory_caches == plain.host.memory_caches + te_rows and report.val_images == 3
    written = ft_budget.startup_report(args, train_group, "adamw_host", exists=lambda path: False,
                                       val_dataset_group=same)
    unwritten = ft_budget.startup_report(args, train_group, "adamw_host", exists=lambda path: False)
    assert written.latent_cache_bytes == unwritten.latent_cache_bytes + 3 * ft_budget.latent_cache_bytes(1, 4096)
    big = fake_group(["a", "b"], batch_size=2, resos={0: (2048, 2048), 1: (2048, 2048)})
    report = ft_budget.startup_report(args, train_group, "adamw_host", exists=lambda path: True, val_dataset_group=big)
    line = next(line for line in report.lines if "validation forward" in line)
    assert "counted in the total" in line and report.gpu.allocated > plain.gpu.allocated
    assert (report.gpu.val_batch, report.gpu.val_image_tokens) == (2, 16384)
