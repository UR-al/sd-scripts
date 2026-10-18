"""CPU tests for the named sigmoid timestep presets and --scale_resume (library/timestep_presets.py).

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m pytest tests/test_timestep_presets.py
"""
import argparse
import os
import sys
import tempfile
import types

import pytest
import torch

from library import flux_train_utils, timestep_presets, train_util
from library.moss import flow

SCHEDULER = types.SimpleNamespace(config=types.SimpleNamespace(num_train_timesteps=1000))
# P(sigma < 0.3), P(0.3 <= sigma <= 0.7), P(sigma > 0.7); 0.1984 = Phi(logit(0.3)), 0.5832 = (1 - 0.1984) ** (1/3)
BANDS = {"balanced": (0.1984, 0.6032, 0.1984), "content": (0.0710, 0.3458, 0.5832), "style": (0.5832, 0.3458, 0.0710)}


def _args(**kw):
    base = dict(timestep_sampling="sigmoid", sigmoid_scale=1.0, scale_resume=0, max_train_epochs=10, ip_noise_gamma=0.0,
                ip_noise_gamma_random_strength=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _bands(sigmas):
    s = sigmas.double().reshape(-1)
    return (float((s < 0.3).double().mean()), float(((s >= 0.3) & (s <= 0.7)).double().mean()),
            float((s > 0.7).double().mean()))


def _assert_bands(sigmas, preset, tol):
    got = _bands(sigmas)
    assert all(abs(a - b) < tol for a, b in zip(got, BANDS[preset])), (preset, got, BANDS[preset])


def _sample(args, batch, dtype=torch.float32):
    latents = torch.zeros(batch, 1, 1, 1, dtype=dtype)
    noise = torch.randn_like(latents)
    return flux_train_utils.get_noisy_model_input_and_timesteps(args, SCHEDULER, latents, noise, "cpu", dtype)


def test_parse_sigmoid_scale():
    parse = timestep_presets.parse_sigmoid_scale
    assert parse("Content") == "content" and parse(" style ") == "style" and parse("BALANCED") == "balanced"
    assert parse("1.5") == 1.5 and parse(2) == 2.0 and parse(1.3) == 1.3
    for bad in ("loud", True, ""):
        with pytest.raises(argparse.ArgumentTypeError):
            parse(bad)


def test_preset_bands_on_a_quantile_grid():
    u = (torch.arange(400_000, dtype=torch.float64) + 0.5) / 400_000
    for preset in timestep_presets.PRESETS:
        sigmas = timestep_presets.preset_sigmas(preset, u)
        assert sigmas.dtype == torch.float32 and float(sigmas.min()) > 0.0 and float(sigmas.max()) < 1.0
        _assert_bands(sigmas, preset, 0.002)
    # content and style are mirror images: sigma_content(u) = 1 - sigma_style(u)
    mirrored = 1.0 - timestep_presets.preset_sigmas("style", u)
    assert torch.allclose(timestep_presets.preset_sigmas("content", u), mirrored, atol=1e-6)


def test_presets_match_ai_toolkit_sigmoid_timestep_bias():
    # ai-toolkit: a table of sigmoid(N(0,1)) sorted from high to low noise, indexed by u (Balanced), u^3 (High Noise)
    # or (1 - u^3) (Low Noise).
    gen = torch.Generator().manual_seed(0)
    table = torch.sigmoid(torch.randn(200_000, generator=gen, dtype=torch.float64)).sort(descending=True).values
    u = torch.rand(400_000, generator=gen, dtype=torch.float64)
    n = len(table)
    index = {"balanced": u * n, "content": u**3 * n, "style": (1 - u**3) * n}
    for preset, idx in index.items():
        reference = table[idx.long().clamp(0, n - 1)]
        mine = timestep_presets.preset_sigmas(preset, u)
        assert all(abs(a - b) < 0.006 for a, b in zip(_bands(reference), _bands(mine))), preset
        assert abs(float(reference.mean()) - float(mine.mean())) < 0.003, preset


def test_resolve_presets_and_scale_resume():
    args = _args(sigmoid_scale="Content", scale_resume=3)
    timestep_presets.resolve(args)
    assert args.sigmoid_scale == 1.0 and args._sigmoid_preset == "content"
    assert args.max_train_epochs == 13 and args._scale_resume_from == 10
    timestep_presets.resolve(args)  # idempotent
    assert args.max_train_epochs == 13

    numeric = _args(sigmoid_scale="1.5")
    timestep_presets.resolve(numeric)
    assert numeric.sigmoid_scale == 1.5 and numeric._sigmoid_preset is None and numeric._scale_resume_from is None
    assert numeric.max_train_epochs == 10

    for bad in (_args(sigmoid_scale="content", timestep_sampling="shift"),
                _args(scale_resume=2, timestep_sampling="moss"),
                _args(scale_resume=2, max_train_epochs=None),
                _args(scale_resume=-1), _args(scale_resume=True), _args(scale_resume=1.5),
                _args(sigmoid_scale="loud")):
        with pytest.raises(ValueError):
            timestep_presets.resolve(bad)


def test_sampler_follows_preset_and_switches_after_max_train_epochs():
    torch.manual_seed(0)
    args = _args(sigmoid_scale="content", scale_resume=2)
    timestep_presets.resolve(args)
    _, timesteps, sigmas = _sample(args, 50_000)
    _assert_bands(sigmas, "content", 0.01)
    assert torch.allclose(timesteps, sigmas.reshape(-1) * 1000)
    timestep_presets.set_epoch(args, 10)
    _assert_bands(_sample(args, 50_000)[2], "content", 0.01)
    timestep_presets.set_epoch(args, 11)
    assert timestep_presets.active_sigmoid_preset(args) == "style"
    _assert_bands(_sample(args, 50_000)[2], "style", 0.01)

    # a numeric scale (with or without scale_resume) samples exactly as before until the resume stage
    numeric = _args(sigmoid_scale=1.5, scale_resume=1)
    timestep_presets.resolve(numeric)
    torch.manual_seed(1)
    latents = torch.zeros(64, 1, 1, 1)
    noise = torch.randn_like(latents)
    sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(numeric, SCHEDULER, latents, noise, "cpu", torch.float32)[2]
    torch.manual_seed(1)
    torch.randn(64, 1, 1, 1)
    assert torch.equal(sigmas.reshape(-1), torch.sigmoid(1.5 * torch.randn(64)))
    timestep_presets.set_epoch(numeric, 11)
    _assert_bands(_sample(numeric, 50_000)[2], "style", 0.01)


@pytest.mark.parametrize("flags", [dict(antithetic_timestep_sampling=True), dict(stratified_timestep_sampling=True),
                                   dict(qmc_timestep_sampling="sobol")])
def test_presets_with_variance_reduction(flags):
    torch.manual_seed(0)
    args = _args(sigmoid_scale="content", **flags)
    timestep_presets.resolve(args)
    sigmas = torch.cat([_sample(args, 512)[2].reshape(-1) for _ in range(40)])
    _assert_bands(sigmas, "content", 0.02)


def test_untouched_args_keep_the_numeric_path():
    # other trainers never call resolve(): no preset attributes, numeric sigmoid_scale as before
    args = _args(sigmoid_scale=1.3)
    assert timestep_presets.active_sigmoid_preset(args) is None
    torch.manual_seed(2)
    sigmas = _sample(args, 32)[2].reshape(-1)
    torch.manual_seed(2)
    torch.randn(32, 1, 1, 1)
    assert torch.equal(sigmas, torch.sigmoid(1.3 * torch.randn(32)))


def test_moss_flow_wrapper_uses_presets_in_fp32():
    args = _args(sigmoid_scale="style")
    timestep_presets.resolve(args)
    latents = torch.randn(8, 128, 2, 2, dtype=torch.bfloat16)
    noisy, timesteps, sigmas = flow.get_noisy_model_input_and_timesteps(args, SCHEDULER, latents, torch.randn_like(latents),
                                                                       "cpu", torch.bfloat16, sampler=None)
    assert noisy.dtype == torch.bfloat16 and timesteps.dtype == torch.float32 and sigmas.dtype == torch.float32
    assert timesteps.shape == (8,) and torch.allclose(timesteps, sigmas.reshape(-1) * 1000)
    assert "sigmoid_scale=style" in flow.describe_moss_sampling(args)


def test_metadata():
    args = _args(sigmoid_scale="content", scale_resume=3)
    timestep_presets.resolve(args)
    assert timestep_presets.metadata(args) == {"ss_sigmoid_scale": "content", "ss_sigmoid_preset": "content",
                                               "ss_scale_resume": "3", "ss_scale_resume_from_epoch": "11"}
    numeric = _args(sigmoid_scale=1.5)
    timestep_presets.resolve(numeric)
    assert timestep_presets.metadata(numeric) == {"ss_sigmoid_scale": "1.5"}


def test_toml_config_reaches_resolve(monkeypatch):
    import moss_train_network

    parser = moss_train_network.setup_parser()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.toml")
        with open(path, "w", encoding="utf-8") as f:
            f.write('timestep_sampling = "sigmoid"\nsigmoid_scale = "Content"\nscale_resume = 2\nmax_train_epochs = 4\n')
        monkeypatch.setattr(sys, "argv", ["moss_train_network.py", "--config_file", path])  # read_config re-parses argv
        args = train_util.read_config_from_file(parser.parse_args(), parser)
    assert args.sigmoid_scale == "content"  # TOML strings go through the argparse type (library/config_validation.py)
    timestep_presets.resolve(args)
    assert args._sigmoid_preset == "content" and args.max_train_epochs == 6 and args._scale_resume_from == 4


def test_cli_parsers_accept_presets():
    import anima_train_network
    import moss_train_network

    for module in (moss_train_network, anima_train_network):
        args = module.setup_parser().parse_args(["--sigmoid_scale", "Style", "--scale_resume", "2"])
        assert args.sigmoid_scale == "style" and args.scale_resume == 2, module.__name__
        assert module.setup_parser().parse_args(["--sigmoid_scale", "1.5"]).sigmoid_scale == 1.5


def test_train_loop_reports_the_epoch():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "train_network.py"), encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    i = lines.index("current_epoch.value = epoch + 1")
    assert lines[i + 1].startswith("timestep_presets.set_epoch(args, current_epoch.value)")
