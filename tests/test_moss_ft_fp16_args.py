"""CPU tests of fp16 compute's settings (docs/moss_ft_te_live_fp16_plan.md 2.1, 2.5): the new flags and their defaults,
fp16 accepted with the fp32 stream, every refusal with its message, the loss-scale flags refused without fp16, and the
loader's fp16 range report (ft_model.fp16_range_report: overflow refused, subnormal / zeroed shares logged, compute
copies = master rounded to fp16).

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_fp16_args.py -q -p no:cacheprovider
"""
import inspect
import os
import re
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _moss_ft_model_common as M  # noqa: E402

from library.moss import ft_args, ft_model  # noqa: E402


@pytest.fixture(scope="module")
def parser():
    return ft_args.setup_parser()


def make_args(parser, *argv, **overrides):
    """A valid adamw_host configuration (bf16 + fp32 stream, checkpointing, both caches) with overrides."""
    args = parser.parse_args(["--moss_stream_fp32", "--gradient_checkpointing", "--cache_latents",
                              "--cache_text_encoder_outputs", *argv])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_new_flags_and_defaults(parser):
    d = parser.parse_args([])
    assert (d.moss_ft_loss_scale, d.moss_ft_loss_scale_growth_interval, d.moss_ft_loss_scale_min,
            d.moss_ft_overflow_policy) == (32768.0, 2000, 1.0, "partial")
    a = parser.parse_args(["--moss_ft_loss_scale", "1024", "--moss_ft_loss_scale_growth_interval", "0",
                           "--moss_ft_loss_scale_min", "4", "--moss_ft_overflow_policy", "window"])
    assert (a.moss_ft_loss_scale, a.moss_ft_loss_scale_growth_interval, a.moss_ft_loss_scale_min,
            a.moss_ft_overflow_policy) == (1024.0, 0, 4.0, "window")


def test_fp16_with_the_fp32_stream_passes(parser, caplog):
    args = make_args(parser, "--mixed_precision", "fp16")
    with caplog.at_level("INFO"):
        ft_args.validate_moss_ft_args(args, cuda=False)
    assert "dynamic loss scale from 32768" in caplog.text
    ft_args.validate_moss_ft_args(make_args(parser, "--mixed_precision", "fp16", "--optimizer_type", "CAME",
                                            "--optimizer_args", "betas=(0.9,0.999,0.99)"), cuda=False)
    ft_args.validate_moss_ft_args(make_args(parser, "--mixed_precision", "fp16", "--moss_ft_overflow_policy",
                                            "window", "--moss_ft_clip_mode", "exact", "--max_grad_norm", "1.0",
                                            "--moss_ft_loss_scale", "65536", "--moss_ft_loss_scale_min", "65536",
                                            "--moss_ft_loss_scale_growth_interval", "0"), cuda=False)
    # TOML integers reach the float flags unconverted (read_config_from_file)
    ft_args.validate_moss_ft_args(make_args(parser, mixed_precision="fp16", moss_ft_loss_scale=4096,
                                            moss_ft_loss_scale_min=1), cuda=False)


REFUSALS = [
    ("fp16_bf16_stream", {"mixed_precision": "fp16", "moss_stream_fp32": False},
     "mixed_precision fp16 needs --moss_stream_fp32: Base's residual stream exceeds fp16's 65504"),
    ("fp16_bf16_stream_allowed_flag_does_not_help", {"mixed_precision": "fp16", "moss_stream_fp32": False,
                                                    "moss_ft_allow_bf16_stream": True}, "needs --moss_stream_fp32"),
    ("fp16_adafactor_fused", {"mixed_precision": "fp16", "optimizer_type": "Adafactor", "fused_backward_pass": True,
                              "gradient_accumulation_steps": 1},
     "fp16 needs a host optimizer mode (adamw_host / came_host)"),
    ("full_fp16", {"mixed_precision": "fp16", "full_fp16": True}, "--full_fp16 is not supported"),
    ("other_precision", {"mixed_precision": "fp8"},
     "mixed_precision must be bf16 (default), fp16 (with --moss_stream_fp32) or no"),
    ("scale_not_power_of_two", {"mixed_precision": "fp16", "moss_ft_loss_scale": 1000.0}, "--moss_ft_loss_scale must"),
    ("scale_too_large", {"mixed_precision": "fp16", "moss_ft_loss_scale": 2.0 ** 25}, "[1, 2^24]"),
    ("scale_below_one", {"mixed_precision": "fp16", "moss_ft_loss_scale": 0.5}, "[1, 2^24]"),
    ("min_not_power_of_two", {"mixed_precision": "fp16", "moss_ft_loss_scale_min": 3.0},
     "--moss_ft_loss_scale_min must"),
    ("min_above_initial", {"mixed_precision": "fp16", "moss_ft_loss_scale": 256.0, "moss_ft_loss_scale_min": 512.0},
     "is above --moss_ft_loss_scale"),
    ("growth_negative", {"mixed_precision": "fp16", "moss_ft_loss_scale_growth_interval": -1},
     "--moss_ft_loss_scale_growth_interval"),
    ("window_with_stale_clip", {"mixed_precision": "fp16", "moss_ft_overflow_policy": "window", "max_grad_norm": 1.0},
     "--moss_ft_overflow_policy window needs --moss_ft_clip_mode exact"),
    ("window_without_clip", {"mixed_precision": "fp16", "moss_ft_overflow_policy": "window",
                             "moss_ft_clip_mode": "exact", "max_grad_norm": 0.0}, "max_grad_norm > 0"),
    ("scale_flag_with_bf16", {"moss_ft_loss_scale": 1024.0}, "--moss_ft_loss_scale is for fp16 compute"),
    ("policy_flag_with_bf16", {"moss_ft_overflow_policy": "window"}, "--moss_ft_overflow_policy is for fp16"),
    ("scale_flags_with_fp32", {"mixed_precision": "no", "moss_ft_loss_scale_growth_interval": 10,
                               "moss_ft_loss_scale_min": 2.0},
     "--moss_ft_loss_scale_growth_interval, --moss_ft_loss_scale_min are for fp16"),
]


@pytest.mark.parametrize("name,overrides,match", REFUSALS, ids=[case[0] for case in REFUSALS])
def test_refusals(parser, name, overrides, match):
    args = make_args(parser, **overrides)
    with pytest.raises(ValueError, match=re.escape(match)) as error:
        ft_args.validate_moss_ft_args(args, cuda=False)
    assert str(error.value).startswith("MOSS full FT: ")


def test_moss_train_never_uses_accelerates_grad_scaler():
    """Accelerate builds a GradScaler for fp16; moss_train calls loss.backward() itself and steps on the host, so that
    scaler must stay unused (plan 2.1)."""
    import moss_train

    source = inspect.getsource(moss_train)
    assert "accelerator.backward" not in source.replace("never accelerator.backward", "")
    assert "accelerator.scaler" not in source.replace("accelerator.scaler stays idle", "")
    loop = inspect.getsource(moss_train._train_loop)
    assert "(loss if scaler is None else loss * scaler.scale).backward()" in loop
    assert "scaler.inv_scale" in loop and "scaler.update(stats" in loop and "scaler.check(" in loop


# ---- ft_model.fp16_range_report ----------------------------------------------------------------------------------
def _load(path, **kw):
    return ft_model.load_moss_for_full_ft(str(path), M.args(mixed_precision="fp16", moss_stream_fp32=True, **kw), "cpu",
                                          config=M.TINY)


def test_fp16_load_compute_copies_are_the_master_rounded(tmp_path, caplog):
    path, state = M.write_base(tmp_path, dtype=torch.float32)
    with caplog.at_level("INFO"):
        model, pool, slots = _load(path)
    assert "fp16 compute copies" in caplog.text and "fp16 subnormal" in caplog.text
    params = dict(model.named_parameters())
    assert params["layers.0.proj.q.weight"].dtype == torch.float16
    assert params["shared_block_modulation.weight"].dtype == torch.float32  # the fp32 stream
    for s in slots:
        assert torch.equal(s.param.detach().reshape(-1), pool.view("master", s).to(s.param.dtype)), s.name
        assert torch.equal(pool.view("master", s), state[s.name].reshape(-1)), s.name  # the file's fp32 values


def test_fp16_range_report_counts(tmp_path):
    path, state = M.write_base(tmp_path, dtype=torch.float32)
    key = "layers.1.proj.k.weight"
    w = state[key].clone()
    w.view(-1)[:5] = torch.tensor([1e-5, -3e-5, 1e-9, -2e-8, 0.0])  # 4 subnormal-range values, 2 of them cast to 0
    state[key] = w
    save_file({k: state[k].contiguous() for k in sorted(state)}, str(path), metadata={"format": "pt"})
    model, pool, slots = _load(path)
    counts = ft_model.fp16_range_report(model, pool, slots)
    fp16_values = sum(s.numel for s in slots if s.compute_dtype == torch.float16)
    assert counts["values"] == fp16_values and counts["over"] == 0
    assert counts["subnormal"] >= 4 and counts["zeroed"] >= 2


def test_fp16_overflow_is_refused(tmp_path):
    path, state = M.write_base(tmp_path, dtype=torch.float32)
    key = "layers.2.proj.ffn_down.weight"
    w = state[key].clone()
    w.view(-1)[11] = 70000.0  # beyond fp16's 65504: the compute copy would be inf
    state[key] = w
    save_file({k: state[k].contiguous() for k in sorted(state)}, str(path), metadata={"format": "pt"})
    with pytest.raises(ValueError, match=re.escape("beyond fp16's 65504")) as error:
        _load(path)
    assert key in str(error.value)
    # bf16 takes the same file
    ft_model.load_moss_for_full_ft(str(path), M.args(mixed_precision="bf16", moss_stream_fp32=True), "cpu",
                                   config=M.TINY)


def test_fp16_overflow_in_a_frozen_parameter_is_refused(tmp_path):
    path, state = M.write_base(tmp_path, dtype=torch.float32)
    key = "layers.1.proj.o.weight"
    w = state[key].clone()
    w.view(-1)[0] = -1e6
    state[key] = w
    save_file({k: state[k].contiguous() for k in sorted(state)}, str(path), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="beyond fp16") as error:
        _load(path, moss_ft_freeze=r"layers\.1\..*")
    assert key in str(error.value)
