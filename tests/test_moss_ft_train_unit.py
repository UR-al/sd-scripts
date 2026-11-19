"""CPU unit tests of moss_train.py's loop helpers (docs/moss_full_finetune_design.md 4.4, 4.6, 4.7) and of the mode (b)
state round trip with bf16 parameters (library/moss/ft_state.py; design 3.2).

* window_of: the window, micro-step and window size of a batch of the epoch (partial last window, design 3.1);
* progress_after_window: the TrainProgress of a state written after a window - an epoch's last window counts as the
  end of that epoch, so a resume from it starts the next epoch without skipping a whole epoch's batches;
* epoch_save_due / step_save_due / phase_state_due: kohya's save cadence (the last epoch is the final save), and the
  epoch state forced at the end of every LR phase but the last (design 4.6 / 4.7);
* the mode (b) state keeps Adafactor's fp32 statistics of bf16 parameters in fp32 (torch's load_state_dict casts every
  floating state tensor to its parameter's dtype).

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_train_unit.py -q -p no:cacheprovider
"""
import math
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _moss_ft_model_common as M  # noqa: E402

import moss_train  # noqa: E402
from library.moss import ft_args, ft_model, ft_state  # noqa: E402


def test_parser_is_the_ft_parser():
    args = moss_train.setup_parser().parse_args([])
    assert args.moss_ft_clip_mode == "stale" and args.mixed_precision == "bf16" and args.learning_rate is None


@pytest.mark.parametrize("batches, ga, want", [
    (5, 2, [(0, 0, 2), (0, 1, 2), (1, 0, 2), (1, 1, 2), (2, 0, 1)]),
    (4, 2, [(0, 0, 2), (0, 1, 2), (1, 0, 2), (1, 1, 2)]),
    (3, 1, [(0, 0, 1), (1, 0, 1), (2, 0, 1)]),
    (7, 3, [(0, 0, 3), (0, 1, 3), (0, 2, 3), (1, 0, 3), (1, 1, 3), (1, 2, 3), (2, 0, 1)]),
    (2, 4, [(0, 0, 2), (0, 1, 2)]),  # GA larger than the epoch: one partial window
])
def test_window_of(batches, ga, want):
    assert [moss_train.window_of(index, batches, ga) for index in range(batches)] == want
    # the final micro-step of every window: j == n_micro - 1
    finals = [index for index in range(batches) if moss_train.window_of(index, batches, ga)[1] ==
              moss_train.window_of(index, batches, ga)[2] - 1]
    assert len(finals) == ft_args.steps_per_epoch(batches, ga) and finals[-1] == batches - 1


def test_progress_after_window():
    schedule = {"bounds": [3, 6]}
    mid = moss_train.progress_after_window(global_step=2, epoch=0, consumed=4, num_batches=5, steps_per_epoch=3,
                                           gradient_accumulation_steps=2, lr_schedule=schedule)
    assert (mid.global_step, mid.completed_epochs, mid.epoch_step) == (2, 0, 4)
    assert (mid.steps_per_epoch, mid.gradient_accumulation_steps, mid.lr_schedule) == (3, 2, schedule)
    end = moss_train.progress_after_window(global_step=6, epoch=1, consumed=5, num_batches=5, steps_per_epoch=3,
                                           gradient_accumulation_steps=2, lr_schedule=None)
    assert (end.global_step, end.completed_epochs, end.epoch_step) == (6, 2, 0)
    info = ft_state.ResumeInfo(path="x", progress=end, mode="adamw_host", last_grad_norm=None, skip_streak=0)
    assert (info.start_epoch, info.batches_to_skip) == (2, 0)


def _args(**kw):
    values = dict(save_every_n_epochs=None, save_every_n_steps=None, save_state=False, moss_ft_lr_phases=None,
                  moss_ft_no_keep_phase_states=False)
    values.update(kw)
    return types.SimpleNamespace(**values)


def test_epoch_and_step_save_cadence():
    args = _args(save_every_n_epochs=2)
    assert [moss_train.epoch_save_due(args, e, 6) for e in range(1, 7)] == [False, True, False, True, False, False]
    assert not any(moss_train.epoch_save_due(_args(), e, 6) for e in range(1, 7))
    args = _args(save_every_n_steps=3)
    assert [moss_train.step_save_due(args, s) for s in range(1, 8)] == [False, False, True, False, False, True, False]
    assert not any(moss_train.step_save_due(_args(), s) for s in range(1, 8))


def test_phase_states_are_forced_at_phase_ends():
    """The pinned LR-phase-boundary states exist only if they are written: the epoch state at the end of every phase
    but the last is saved with save_state even when save_every_n_epochs misses that epoch."""
    args = _args(save_state=True, moss_ft_lr_phases="2:5e-6,3:2e-6,3:1e-6")
    assert [e for e in range(1, 9) if moss_train.phase_state_due(args, e, 8)] == [2, 5]
    assert not any(moss_train.phase_state_due(_args(save_state=False, moss_ft_lr_phases="2:5e-6,6:1e-6"), e, 8)
                   for e in range(1, 9))
    assert not any(moss_train.phase_state_due(_args(save_state=True, moss_ft_lr_phases="2:5e-6,6:1e-6",
                                                    moss_ft_no_keep_phase_states=True), e, 8) for e in range(1, 9))
    assert not any(moss_train.phase_state_due(_args(save_state=True), e, 8) for e in range(1, 9))
    assert not any(moss_train.phase_state_due(_args(save_state=True, moss_ft_lr_phases="0.5:5e-6,1:1e-6"), e, 2)
                   for e in range(1, 3))  # a fractional phase end has no epoch state


def test_fused_adafactor_steps_in_the_hooks_and_keeps_the_scheduler_order():
    """Mode (b): every parameter is stepped in its gradient hook (bf16 with stochastic rounding); the window's end calls
    optimizer.step() (a no-op: every gradient is gone) so the LR scheduler sees the order it expects; grad_norm is the
    global norm from the per-tensor clip norms."""
    import warnings

    import library.adafactor_fused as adafactor_fused
    from transformers.optimization import Adafactor

    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(8, 8).to(torch.bfloat16))
    bias = torch.nn.Parameter(torch.randn(8))
    opt = Adafactor([{"params": [weight], "name": "decay"}, {"params": [bias], "name": "no_decay"}], lr=1e-2,
                    scale_parameter=False, relative_step=False, warmup_init=False)
    adafactor_fused.patch_adafactor_fused(opt)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
        steps = moss_train.FusedAdafactorSteps(opt, max_grad_norm=1e6)
        steps.attach()
        before = (weight.detach().clone(), bias.detach().clone())
        steps.begin_window(1, moss_train.window_context(opt, 0))
        steps.begin_microstep(0)
        x = torch.randn(4, 8)
        (x @ weight.float() + bias).square().sum().backward()  # the hooks step and drop the gradients
        stats = steps.end_microstep()
        scheduler.step()
        steps.close()
    assert weight.grad is None and bias.grad is None
    assert not torch.equal(weight.detach(), before[0]) and not torch.equal(bias.detach(), before[1])
    assert stats.stepped_params == 2 and stats.clip_mode == "per_tensor" and math.isfinite(stats.grad_norm)
    assert not [w for w in caught if "lr_scheduler.step()" in str(w.message)], [str(w.message) for w in caught]


def _runtime_args(monkeypatch, *argv):
    from library.moss import strategy

    checked = []
    monkeypatch.setattr(strategy, "check_moss_text_encoder_path", lambda path: checked.append(path))
    args = moss_train.setup_parser().parse_args(["--moss_text_encoder", "te.safetensors", "--output_dir", "out",
                                                 *argv])
    return args, checked


def test_prepare_runtime_needs_an_output_dir(monkeypatch):
    """The DiT saves, states and samples go there: refused at the start, not at the first save hours later."""
    args, _ = _runtime_args(monkeypatch)
    args.output_dir = None
    with pytest.raises(ValueError, match="output_dir"):
        moss_train.prepare_runtime(args)


def test_prepare_runtime_sets_what_the_moss_checks_read(monkeypatch):
    args, checked = _runtime_args(monkeypatch, "--attn_mode", "sdpa")
    moss_train.prepare_runtime(args)
    assert args.attn_mode == "torch" and checked == ["te.safetensors"]
    assert args.network_train_unet_only is True and args.network_train_text_encoder_only is False
    assert args._sigmoid_preset is None  # timestep_presets.resolve ran


@pytest.mark.parametrize("argv, match", [
    (("--cache_aug_refresh_epochs", "2"), "cache_aug_refresh_epochs"),
    (("--save_model_as", "ckpt"), "save_model_as"),
])
def test_prepare_runtime_refusals(monkeypatch, argv, match):
    args, _ = _runtime_args(monkeypatch, *argv)
    with pytest.raises(ValueError, match=match):
        moss_train.prepare_runtime(args)


def test_prepare_runtime_needs_the_text_encoder_and_flash(monkeypatch):
    from library import attention

    args, _ = _runtime_args(monkeypatch)
    args.moss_text_encoder = None
    with pytest.raises(ValueError, match="moss_text_encoder"):
        moss_train.prepare_runtime(args)
    args, _ = _runtime_args(monkeypatch, "--attn_mode", "flash")
    monkeypatch.setattr(attention, "is_flash_attn_supported", lambda: False)
    with pytest.raises(ValueError, match="flash"):
        moss_train.prepare_runtime(args)


def test_prepare_runtime_ignores_the_cuda_allocator_options(monkeypatch):
    warnings = []
    monkeypatch.setattr(moss_train.logger, "warning", lambda message, *a, **k: warnings.append(str(message)))
    args, _ = _runtime_args(monkeypatch, "--cuda_memory_fraction", "0.9")
    moss_train.prepare_runtime(args)
    assert any("--cuda_memory_fraction is ignored" in line for line in warnings), warnings


def test_mode_b_state_keeps_fp32_adafactor_statistics_of_bf16_parameters(tmp_path):
    """Design 3.2: Adafactor keeps fp32 statistics for the bf16 joint Linears (adafactor_step_param makes them from
    the fp32 gradient). torch.optim.Optimizer.load_state_dict casts every floating state tensor to its parameter's
    dtype, so a resumed mode (b) run would continue from bf16-rounded statistics; ft_state.load_state puts the saved
    dtypes back."""
    import library.adafactor_fused as adafactor_fused
    from transformers.optimization import Adafactor

    base_path, _ = M.write_base(tmp_path)
    args = M.args(mixed_precision="bf16", moss_stream_fp32=True, optimizer_type="Adafactor", fused_backward_pass=True,
                  optimizer_args=[], max_grad_norm=0.0)

    def make():
        model, pool, slots = ft_model.load_moss_for_full_ft(str(base_path), args, "cpu", config=M.TINY)
        named = ft_model.trainable_named(slots)
        opt = Adafactor(ft_model.param_groups(named), lr=1e-3, scale_parameter=False, relative_step=False,
                        warmup_init=False)
        adafactor_fused.patch_adafactor_fused(opt)
        return model, slots, opt

    def step(model, opt, seed):
        g = torch.Generator().manual_seed(seed)
        for group in opt.param_groups:
            for p in group["params"]:
                p.grad = torch.randn(p.shape, generator=g).to(p.dtype)
                opt.step_param(p, group)
                p.grad = None

    model, slots, opt = make()
    step(model, opt, 1)
    q = model.layers[0].proj["q"].weight
    assert q.dtype == torch.bfloat16 and opt.state[q]["exp_avg_sq_row"].dtype == torch.float32
    path = ft_state.write_state(str(tmp_path / "states"), "s", model=model, slots=slots, pool=None, optimizer=opt,
                                lr_scheduler=None, mode=ft_args.ADAFACTOR_FUSED,
                                progress=ft_state.TrainProgress(global_step=1, completed_epochs=0), args=args)
    model2, slots2, opt2 = make()
    ft_state.load_state(path, model=model2, slots=slots2, pool=None, optimizer=opt2, lr_scheduler=None,
                        mode=ft_args.ADAFACTOR_FUSED, args=args, restore_rng=False)
    q2 = model2.layers[0].proj["q"].weight
    for key, value in opt.state[q].items():
        if torch.is_tensor(value):
            assert opt2.state[q2][key].dtype == value.dtype and torch.equal(opt2.state[q2][key], value), key
        else:
            assert opt2.state[q2][key] == value, key
    torch.manual_seed(3)  # the stochastic rounding of both continuations draws the same bits
    step(model, opt, 2)
    torch.manual_seed(3)
    step(model2, opt2, 2)
    for (name, p), (_, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.equal(p.detach(), p2.detach()), name


# ---- review fixes ---------------------------------------------------------------------------------------------------
def test_show_timesteps_prints_and_does_not_train(monkeypatch):
    """Review fix: --show_timesteps ("then exit without training (CPU only, no model loaded)") went into train() - the
    caching, the 85 GiB host pool, the DiT load. As moss_train_network.py: print the distribution and exit."""
    import sys

    from library.moss import train_utils as moss_train_utils

    calls = []
    monkeypatch.setattr(moss_train_utils, "show_timesteps", lambda args: calls.append(("show", args.timestep_sampling)))
    monkeypatch.setattr(moss_train, "train", lambda args: calls.append(("train", args.timestep_sampling)))
    argv = ["--show_timesteps", "console", "--timestep_sampling", "moss"]
    monkeypatch.setattr(sys, "argv", ["moss_train.py", *argv])  # read_config_from_file reads sys.argv with a config
    moss_train.main(argv)
    assert calls == [("show", "moss")]
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["moss_train.py", "--timestep_sampling", "moss"])
    moss_train.main(["--timestep_sampling", "moss"])
    assert calls == [("train", "moss")]


def test_checkpoint_switches_are_on_before_the_validation(monkeypatch, caplog):
    """Review fix: validate_moss_args (start-up step 4) logged "--moss_ckpt_reuse_attn has no effect here" because
    set_ckpt_reuse ran only at step 7, after it. prepare_runtime (step 1) sets the process-wide switches, as
    moss_train_network.prepare_moss_runtime does (they are flags: nothing runs a DiT forward before step 7)."""
    import logging

    from library import anima_models, attention
    from library.moss import models as moss_models
    from library.moss import train_utils as moss_train_utils

    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", False)  # restored after the test
    monkeypatch.setattr(anima_models, "CKPT_REUSE_ATTN", False)
    monkeypatch.setattr(moss_models, "CKPT_REUSE_BUDGET_BYTES", None)
    monkeypatch.setattr(attention, "is_flash_attn_supported", lambda: True)
    monkeypatch.setattr(attention, "flash_reuse_available", lambda: True)
    compiled = []
    monkeypatch.setattr(moss_train, "enable_elementwise_compile", lambda on: compiled.append(on))
    args, _ = _runtime_args(monkeypatch, "--attn_mode", "flash", "--moss_ckpt_reuse_attn",
                            "--moss_ckpt_reuse_attn_budget_gib", "2", "--gradient_checkpointing")
    moss_train.prepare_runtime(args)
    assert moss_models.ckpt_reuse_attn_enabled() and not moss_models.ckpt_nograd_merged_enabled()
    assert moss_models.ckpt_reuse_budget_bytes() == 2 << 30 and compiled == [True]
    with caplog.at_level(logging.WARNING):
        moss_train_utils.validate_moss_ckpt_reuse_args(args, None)
    assert "has no effect" not in caplog.text


@pytest.mark.parametrize("extra", [(), ("--max_train_epochs", "8")], ids=["no_epochs", "epochs_given"])
def test_scale_resume_with_lr_phases_is_refused_once_clearly(monkeypatch, extra):
    """Review fix: timestep_presets.resolve (prepare_runtime) ran before the phases set max_train_epochs, so the user
    got "set max_train_epochs" and, after setting it to the phases' 8, "max_train_epochs 10 does not equal the sum of
    the phases (8)". The phases give every epoch its LR, so scale_resume's extra epochs have none: one refusal."""
    args, _ = _runtime_args(monkeypatch, "--moss_ft_lr_phases", "2:5e-6,6:1e-6", "--scale_resume", "2",
                            "--timestep_sampling", "sigmoid", *extra)
    with pytest.raises(ValueError, match="scale_resume") as error:
        moss_train.prepare_runtime(args)
    message = str(error.value)
    assert "moss_ft_lr_phases" in message and "set max_train_epochs" not in message, message


def test_memory_logs_read_the_pinned_host_cache(monkeypatch):
    """For the smoke run (review of the unsloth / ring estimates): on CUDA host/pinned_cache_gib is what torch's pinned
    host allocator holds in all (rings, norm table, unsloth checkpoint blocks), next to the pipeline's own share."""
    monkeypatch.setattr(torch.cuda, "host_memory_stats", lambda: {"allocated_bytes.current": 3 << 30})
    pipeline = types.SimpleNamespace(pinned_bytes=2 << 30)
    logs = moss_train.memory_logs(torch.device("cuda"), pipeline)
    assert logs["host/pinned_gib"] == 2.0 and logs["host/pinned_cache_gib"] == 3.0
    assert "pinned_cache_gib" not in str(moss_train.memory_logs(torch.device("cpu"), pipeline))
    assert "pinned cache 3.00 GiB" in moss_train.memory_line(logs)
