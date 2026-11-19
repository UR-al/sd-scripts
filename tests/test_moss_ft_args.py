"""CPU tests for the arguments of the MOSS full fine-tune (library/moss/ft_args.py; design
docs/moss_full_finetune_design.md 4.2 parser composition, 4.5 arguments, test plan 5 "test_moss_ft_args.py").

* test_parser_builds: the 4.2 composition + add_moss_ft_arguments builds without argparse conflicts, and the example
  config of design 8.1 (read from the design document) reads with unknown keys as errors and passes the dataset-free
  validation;
* test_refusals: every refused row of 4.5 (new arguments, optimizer table, fixed handling of existing arguments, LR
  phases), one parametrized case each, plus the dataset-dependent refusals;
* test_optimizer_mode_mapping / test_optimizer_args_per_mode: the two optimizer tables of 4.5;
* test_resource_checks: mocked psutil.virtual_memory / shutil.disk_usage refuse / pass at the boundaries.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_args.py -q -p no:cacheprovider
"""
import math
import os
import re
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest

from library.moss import ft_args, ft_budget

DESIGN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "moss_full_finetune_design.md")
GIB = 1 << 30
PHASES = "2:5e-6,6:1e-6"


@pytest.fixture(scope="module")
def parser():
    return ft_args.setup_parser()


@pytest.fixture
def logs(monkeypatch):
    """INFO / WARNING lines of ft_args and ft_budget."""
    out = {"info": [], "warning": []}
    for module in (ft_args, ft_budget):
        monkeypatch.setattr(module.logger, "info", lambda m, *a, **k: out["info"].append(str(m)))
        monkeypatch.setattr(module.logger, "warning", lambda m, *a, **k: out["warning"].append(str(m)))
    return out


def make_args(parser, *argv, **overrides):
    """A valid adamw_host configuration (bf16 + fp32 stream, checkpointing, both caches) with overrides."""
    args = parser.parse_args(["--moss_stream_fp32", "--gradient_checkpointing", "--cache_latents",
                              "--cache_text_encoder_outputs", *argv])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class FakeDataset(types.SimpleNamespace):
    pass


class FakeGroup:
    """What validate_moss_ft_args / ft_budget read of a train dataset group (train_util.DatasetGroup)."""

    def __init__(self, root, images=16, batch_size=4, reso=(1024, 1024), variants=0, validation_split=0.0,
                 batches=None):
        infos = {}
        for i in range(images):
            infos[f"img{i}"] = types.SimpleNamespace(bucket_reso=reso, jitter_bucket_info=None,
                                                     absolute_path=os.path.join(str(root), f"img{i:05d}.png"))
        self.datasets = [FakeDataset(batch_size=batch_size, image_data=infos, validation_split=validation_split,
                                     caption_aug_variant_count=variants, latents_aug_variant_count=0,
                                     subsets=[types.SimpleNamespace(flip_aug=False)])]
        self.batches = batches if batches is not None else math.ceil(images / batch_size)

    def __len__(self):
        return self.batches


def plenty_memory():
    return types.SimpleNamespace(available=1 << 50, total=1 << 50)


def plenty_disk(path):
    return types.SimpleNamespace(total=1 << 60, used=0, free=1 << 60)


def validate_with(args, group, **kwargs):
    kwargs.setdefault("virtual_memory", plenty_memory)
    kwargs.setdefault("disk_usage", plenty_disk)
    kwargs.setdefault("exists", lambda path: True)
    kwargs.setdefault("cuda", False)
    ft_args.validate_moss_ft_args(args, group, **kwargs)


# ---- the parser (4.2) ------------------------------------------------------------------------------------------
def design_example_toml() -> str:
    with open(DESIGN, "r", encoding="utf-8") as handle:
        text = handle.read()
    section = text[text.index("### 8.1"):]
    match = re.search(r"```toml\n(.*?)```", section, re.DOTALL)
    assert match, "design 8.1 has no ```toml block"
    return match.group(1)


def test_parser_builds(parser, tmp_path, logs, monkeypatch):
    import sys

    from library import train_util

    parser.format_help()
    defaults = parser.parse_args([])
    assert defaults.moss_ft_lr_phases is None and defaults.moss_ft_freeze == "" and defaults.moss_ft_clip_mode == "stale"
    assert (defaults.moss_ft_host_threads, defaults.moss_ft_piece_mib, defaults.moss_ft_ring_mib,
            defaults.moss_ft_upload_ring_mib, defaults.moss_ft_max_skipped_steps, defaults.moss_ft_log_every) == (
        16, 64, 1024, 512, 8, 10)
    assert defaults.moss_ft_state_dir is None and not defaults.moss_ft_no_keep_phase_states
    assert not defaults.moss_ft_allow_bf16_stream and not defaults.moss_ft_skip_resource_check
    # moss_train.py's own defaults: bf16 compute and save, cfm_lambda as the MOSS LoRA trainer, and "not given" for the
    # two values --moss_ft_lr_phases must tell apart from kohya's defaults
    assert defaults.mixed_precision == "bf16" and defaults.save_precision == "bf16" and defaults.cfm_lambda == 0.02
    assert defaults.learning_rate is None and defaults.max_train_steps is None
    # what anima_train.setup_parser adds itself (4.2) and the MOSS / DiT arguments are there
    for name in ("cpu_offload_checkpointing", "unsloth_offload_checkpointing", "flow_use_ot", "contrastive_flow_matching",
                 "moss_stream_fp32", "moss_ckpt_reuse_attn", "cache_text_encoder_outputs", "blocks_to_swap", "deepspeed",
                 "dataset_config", "masked_loss", "metadata_title"):
        assert hasattr(defaults, name), name

    config = tmp_path / "moss_fft.toml"
    config.write_text(design_example_toml(), encoding="utf-8")
    # read_config_from_file parses sys.argv again (as moss_train.py's __main__ does), not pytest's command line
    monkeypatch.setattr(sys, "argv", ["moss_train.py", "--config_file", str(config)])
    args = train_util.read_config_from_file(parser.parse_args(), parser, unknown_keys="error")
    assert args.moss_ft_lr_phases == PHASES and args.optimizer_type == "AdamW" and args.gradient_accumulation_steps == 2
    assert args.optimizer_args == ["betas=(0.9, 0.999)", "eps=1e-8", "weight_decay=0.0"]
    ft_args.validate_moss_ft_args(args, cuda=True)
    assert ft_args.optimizer_mode(args) == "adamw_host"
    assert ft_args.parse_optimizer_args(args, "adamw_host") == {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}
    assert args.learning_rate == 5e-6 and args.max_train_epochs == 8  # both set from the phases
    assert args.max_train_steps is None  # set from the dataset (steps per epoch) by the dataset-dependent part
    assert not logs["warning"], logs["warning"]


def test_argument_types(parser):
    import argparse

    args = parser.parse_args(["--moss_ft_lr_phases", PHASES, "--moss_ft_freeze", r"layers\.0\..*,time_embedding\..*"])
    assert args.moss_ft_lr_phases == PHASES and args.moss_ft_freeze == r"layers\.0\..*,time_embedding\..*"
    with pytest.raises(SystemExit):
        parser.parse_args(["--moss_ft_lr_phases", "2:x"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--moss_ft_freeze", "layers\\.(0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--moss_ft_clip_mode", "global"])
    with pytest.raises(argparse.ArgumentTypeError):
        ft_args.freeze_arg("(")


# ---- refusals (4.5) ----------------------------------------------------------------------------------------------
def _accelerate_state(tmp_path):
    state = tmp_path / "accelerate_state"
    state.mkdir()
    (state / "optimizer.bin").write_bytes(b"x")
    (state / "model.safetensors").write_bytes(b"x")
    return {"resume": str(state)}


def _state_dir_is_file(tmp_path):
    path = tmp_path / "a_file"
    path.write_text("x", encoding="utf-8")
    return {"moss_ft_state_dir": str(path)}


REFUSALS = [
    # new arguments
    ("host_threads_0", {"moss_ft_host_threads": 0}, "moss_ft_host_threads"),
    ("host_threads_over_cpu_count", {"moss_ft_host_threads": (os.cpu_count() or 1) + 1}, "moss_ft_host_threads"),
    ("host_threads_bool", {"moss_ft_host_threads": True}, "moss_ft_host_threads"),
    ("piece_not_power_of_two", {"moss_ft_piece_mib": 48}, "moss_ft_piece_mib"),
    ("piece_too_small", {"moss_ft_piece_mib": 8}, "moss_ft_piece_mib"),
    ("piece_too_large", {"moss_ft_piece_mib": 512}, "moss_ft_piece_mib"),
    ("ring_not_a_multiple", {"moss_ft_ring_mib": 1000}, "moss_ft_ring_mib"),
    ("ring_under_4_pieces", {"moss_ft_ring_mib": 192}, "moss_ft_ring_mib"),
    ("upload_ring_not_a_multiple", {"moss_ft_upload_ring_mib": 500}, "moss_ft_upload_ring_mib"),
    ("upload_ring_under_2_pieces", {"moss_ft_upload_ring_mib": 64}, "moss_ft_upload_ring_mib"),
    # review fix: torch's pinned allocator rounds every allocation up to a power of two (2.2: "nothing is wasted")
    ("ring_not_a_power_of_two", {"moss_ft_ring_mib": 1536}, "power of two"),
    ("upload_ring_not_a_power_of_two", {"moss_ft_upload_ring_mib": 768}, "power of two"),
    ("max_skipped_negative", {"moss_ft_max_skipped_steps": -1}, "moss_ft_max_skipped_steps"),
    ("log_every_0", {"moss_ft_log_every": 0}, "moss_ft_log_every"),
    ("clip_mode", {"moss_ft_clip_mode": "global"}, "moss_ft_clip_mode"),
    ("state_dir_is_a_file", _state_dir_is_file, "moss_ft_state_dir"),
    ("freeze_bad_regex", {"moss_ft_freeze": "layers\\.(0"}, "moss_ft_freeze"),
    ("freeze_matches_nothing", {"moss_ft_freeze": r"layers\.0\..*,nothing\..*"}, "matches no parameter"),
    ("freeze_everything", {"moss_ft_freeze": ".*"}, "every parameter"),
    # optimizer_type -> mode
    ("optimizer_8bit", {"optimizer_type": "AdamW8bit"}, "adamw_host"),
    ("optimizer_prodigy", {"optimizer_type": "Prodigy"}, "adamw_host"),
    ("optimizer_schedule_free", {"optimizer_type": "AdamWScheduleFree"}, "adamw_host"),
    ("use_8bit_adam", {"use_8bit_adam": True}, "use_8bit_adam"),
    ("use_lion_optimizer", {"use_lion_optimizer": True}, "use_lion_optimizer"),
    ("adafactor_without_fused_backward_pass", {"optimizer_type": "Adafactor"}, "fused_backward_pass"),
    ("fused_backward_pass_with_adamw", {"fused_backward_pass": True}, "fused_backward_pass"),
    ("unknown_optimizer_arg", {"optimizer_args": ["amsgrad=True"]}, "amsgrad"),
    # existing arguments with fixed handling
    ("mixed_precision_fp16_bf16_stream", {"mixed_precision": "fp16", "moss_stream_fp32": False},
     "mixed_precision fp16 needs --moss_stream_fp32"),
    ("mixed_precision_other", {"mixed_precision": "fp8"}, "mixed_precision"),
    ("full_bf16", {"full_bf16": True}, "full_bf16"),
    ("full_fp16", {"full_fp16": True}, "full_fp16"),
    ("fp8_base", {"fp8_base": True}, "fp8_base"),
    ("fp8_base_unet", {"fp8_base_unet": True}, "fp8_base_unet"),
    ("deepspeed", {"deepspeed": True}, "deepspeed"),
    ("blocks_to_swap", {"blocks_to_swap": 4}, "blocks_to_swap"),
    ("use_ramtorch", {"use_ramtorch": True}, "RamTorch"),
    ("cpu_offload_checkpointing", {"cpu_offload_checkpointing": True}, "cpu_offload_checkpointing"),
    ("moss_block_compile", {"moss_block_compile": True}, "moss_block_compile"),
    ("torch_compile", {"torch_compile": True}, "torch_compile"),
    ("fused_optimizer_groups", {"fused_optimizer_groups": 4}, "fused_optimizer_groups"),
    ("blockwise_fused_optimizers", {"blockwise_fused_optimizers": True}, "blockwise_fused_optimizers"),
    ("validation_split", {"validation_split": 0.1}, "validation"),
    ("moss_ckpt_nograd_merged", {"moss_ckpt_nograd_merged": True}, "moss_ckpt_nograd_merged"),
    ("moss_lora_exclude_blocks", {"moss_lora_exclude_blocks": "22"}, "moss_lora_exclude_blocks"),
    ("moss_stock_lycoris", {"moss_stock_lycoris": True}, "moss_stock_lycoris"),
    ("moss_no_lokr_compile", {"moss_no_lokr_compile": True}, "moss_no_lokr_compile"),
    ("moss_no_sample_cache", {"moss_no_sample_cache": True}, "moss_no_sample_cache"),
    ("reuse_attn_without_flash", {"moss_ckpt_reuse_attn": True, "attn_mode": "torch"}, "flash"),
    ("bf16_stream", {"moss_stream_fp32": False}, "moss_stream_fp32"),
    ("no_cache_latents", {"cache_latents": False}, "cache_latents"),
    ("no_cache_text_encoder_outputs", {"cache_text_encoder_outputs": False}, "cache_text_encoder_outputs"),
    ("accumulation_0", {"gradient_accumulation_steps": 0}, "gradient_accumulation_steps"),
    ("adafactor_accumulation", {"optimizer_type": "Adafactor", "fused_backward_pass": True,
                                "gradient_accumulation_steps": 2}, "gradient_accumulation_steps"),
    ("adafactor_exact_clip", {"optimizer_type": "Adafactor", "fused_backward_pass": True,
                              "moss_ft_clip_mode": "exact"}, "exact"),
    ("max_grad_norm_negative", {"max_grad_norm": -1.0}, "max_grad_norm"),
    ("max_grad_norm_nan", {"max_grad_norm": float("nan")}, "max_grad_norm"),
    ("save_precision", {"save_precision": "fp8"}, "save_precision"),
    ("resume_accelerate_state", _accelerate_state, "accelerate"),
    ("resume_missing", {"resume": "Z:/no/such/state"}, "resume"),
    ("resume_from_huggingface", {"resume": "user/repo/state", "resume_from_huggingface": True}, "huggingface"),
    ("adafactor_lr_scheduler", {"lr_scheduler": "adafactor:1e-6"}, "adafactor"),
    # review fix: ZeroLRWarmupScheduler's state holds its inner scheduler (+ the optimizer): never resumable
    ("zero_lr_warmup", {"zero_lr_warmup": True, "lr_scheduler": "cosine", "lr_warmup_steps": 10}, "zero_lr_warmup"),
    # --moss_ft_lr_phases (4.6)
    ("phases_learning_rate", {"moss_ft_lr_phases": PHASES, "learning_rate": 1e-5}, "learning_rate"),
    ("phases_max_train_steps", {"moss_ft_lr_phases": PHASES, "max_train_steps": 1600}, "max_train_steps"),
    ("phases_max_train_epochs", {"moss_ft_lr_phases": PHASES, "max_train_epochs": 10}, "max_train_epochs"),
    ("phases_lr_scheduler", {"moss_ft_lr_phases": PHASES, "lr_scheduler": "cosine"}, "lr_scheduler"),
    ("phases_lr_scheduler_type", {"moss_ft_lr_phases": PHASES, "lr_scheduler_type": "CosineAnnealingLR"},
     "lr_scheduler_type"),
    ("phases_lr_scheduler_args", {"moss_ft_lr_phases": PHASES, "lr_scheduler_args": ["T_max=10"]}, "lr_scheduler_args"),
    ("phases_lr_decay_steps", {"moss_ft_lr_phases": PHASES, "lr_decay_steps": 100}, "lr_decay_steps"),
    ("phases_zero_lr_warmup", {"moss_ft_lr_phases": PHASES, "zero_lr_warmup": True}, "zero_lr_warmup"),
    ("phases_bad_text", {"moss_ft_lr_phases": "2:5e-6,0:1e-6"}, "moss_ft_lr_phases"),
    ("phases_bad_warmup", {"moss_ft_lr_phases": PHASES, "lr_warmup_steps": -5}, "lr_warmup_steps"),
]


@pytest.mark.parametrize("name,overrides,match", REFUSALS, ids=[case[0] for case in REFUSALS])
def test_refusals(parser, tmp_path, name, overrides, match):
    if callable(overrides):
        overrides = overrides(tmp_path)
    args = make_args(parser, **overrides)
    with pytest.raises(ValueError, match=re.escape(match)) as error:
        ft_args.validate_moss_ft_args(args, cuda=False)
    assert str(error.value).startswith("MOSS full FT: ")


def test_refusal_reasons(parser):
    """The rows whose reason the design states: each message carries it."""
    cases = [
        ({"full_bf16": True}, "fp32 master"),
        ({"blocks_to_swap": 2}, "between devices"),
        ({"use_ramtorch": True}, "between devices"),
        ({"moss_block_compile": True}, "not validated with trainable base weights"),
        ({"moss_stream_fp32": False}, "cos 0.58"),
        ({"optimizer_type": "Lion"}, "LoraEasyCustomOptimizer.came.CAME"),  # the mode table is quoted
    ]
    for overrides, reason in cases:
        with pytest.raises(ValueError, match=re.escape(reason)):
            ft_args.validate_moss_ft_args(make_args(parser, **overrides), cuda=False)


def test_gradient_checkpointing_required_on_cuda(parser, logs):
    args = make_args(parser, gradient_checkpointing=False)
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        ft_args.validate_moss_ft_args(args, cuda=True)
    ft_args.validate_moss_ft_args(make_args(parser, gradient_checkpointing=False), cuda=False)  # CPU tests run without
    # unsloth offload switches checkpointing on (as validate_moss_args does) and is allowed
    args = make_args(parser, gradient_checkpointing=False, unsloth_offload_checkpointing=True)
    ft_args.validate_moss_ft_args(args, cuda=True)
    assert args.gradient_checkpointing is True
    assert any("unsloth" in line for line in logs["info"])


def test_multi_process_refused(parser, monkeypatch):
    with pytest.raises(ValueError, match="process"):
        ft_args.check_single_process(2)
    ft_args.check_single_process(1)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="process"):
        ft_args.validate_moss_ft_args(make_args(parser), cuda=False)


def test_dataset_dependent_refusals(parser, tmp_path):
    # *_to_disk above 2,000 images
    validate_with(make_args(parser), FakeGroup(tmp_path, images=2000))
    with pytest.raises(ValueError, match="2,000 images"):
        validate_with(make_args(parser), FakeGroup(tmp_path, images=2001))
    with pytest.raises(ValueError, match="cache_text_encoder_outputs_to_disk"):
        validate_with(make_args(parser, cache_latents_to_disk=True), FakeGroup(tmp_path, images=2001))
    validate_with(make_args(parser, cache_latents_to_disk=True, cache_text_encoder_outputs_to_disk=True),
                  FakeGroup(tmp_path, images=2001))
    # validation datasets
    with pytest.raises(ValueError, match="validation"):
        validate_with(make_args(parser), FakeGroup(tmp_path, validation_split=0.1))
    with pytest.raises(ValueError, match="validation"):
        validate_with(make_args(parser), FakeGroup(tmp_path), val_dataset_group=FakeGroup(tmp_path, images=2))
    # LR warmup past phase 1: 16 images at batch 4 = 4 batches, GA 1 -> 4 steps per epoch, phase 1 = 8 steps
    validate_with(make_args(parser, moss_ft_lr_phases=PHASES, lr_warmup_steps=8), FakeGroup(tmp_path))
    with pytest.raises(ValueError, match="phase 1"):
        validate_with(make_args(parser, moss_ft_lr_phases=PHASES, lr_warmup_steps=9), FakeGroup(tmp_path))
    with pytest.raises(ValueError, match="phase 1"):
        validate_with(make_args(parser, moss_ft_lr_phases="0.1:1e-5,1:1e-6"), FakeGroup(tmp_path))  # 0.4 steps


def test_steps_from_the_dataset(parser, tmp_path, logs):
    # phases: max_train_steps = the last boundary; steps per epoch = ceil(batches / GA)
    args = make_args(parser, moss_ft_lr_phases=PHASES, gradient_accumulation_steps=2)
    validate_with(args, FakeGroup(tmp_path, images=40, batch_size=4))  # 10 batches -> 5 steps per epoch
    assert args.max_train_steps == 40 and args.max_train_epochs == 8 and args.learning_rate == 5e-6
    assert any("[10, 40]" in line for line in logs["info"])
    # fractional phases: max_train_epochs stays unset, max_train_steps is the last boundary
    args = make_args(parser, moss_ft_lr_phases="0.5:1e-5,1:1e-6")
    validate_with(args, FakeGroup(tmp_path, images=40, batch_size=4, batches=11))  # 11 steps per epoch
    assert args.max_train_epochs is None and args.max_train_steps == 17  # round(1.5 x 11 = 16.5) -> 17
    # no phases: kohya's max_train_epochs x steps per epoch, else max_train_steps as given (default 1600)
    args = make_args(parser, max_train_epochs=3, gradient_accumulation_steps=4)
    validate_with(args, FakeGroup(tmp_path, images=40, batch_size=4))  # 10 batches -> 3 steps per epoch
    assert args.max_train_steps == 9 and args.learning_rate == 2e-6
    args = make_args(parser)
    validate_with(args, FakeGroup(tmp_path))
    assert args.max_train_steps == 1600


# ---- values the validation fills in ------------------------------------------------------------------------------
def test_lr_defaults_and_phase_consistency(parser):
    args = make_args(parser)
    ft_args.validate_moss_ft_args(args, cuda=False)
    assert args.learning_rate == 2e-6 and args.max_train_steps == 1600 and args.max_train_epochs is None  # kohya's
    args = make_args(parser, learning_rate=1e-5, max_train_epochs=4)
    ft_args.validate_moss_ft_args(args, cuda=False)
    assert args.learning_rate == 1e-5 and args.max_train_epochs == 4
    # with phases: learning_rate unset or equal to LR1; max_train_epochs unset or equal to the sum
    args = make_args(parser, moss_ft_lr_phases=PHASES, learning_rate=5e-6, max_train_epochs=8,
                     lr_scheduler="constant_with_warmup", lr_warmup_steps=100)
    ft_args.validate_moss_ft_args(args, cuda=False)
    assert args.learning_rate == 5e-6 and args.max_train_epochs == 8
    args = make_args(parser, moss_ft_lr_phases="1.5:5e-6,1:1e-6")
    ft_args.validate_moss_ft_args(args, cuda=False)
    assert args.max_train_epochs is None and args.learning_rate == 5e-6  # 2.5 epochs: no integer epoch count


def test_save_state_defaults(parser, logs):
    args = make_args(parser, save_state=True)
    ft_args.validate_moss_ft_args(args, cuda=False)
    assert args.save_last_n_epochs_state == 1 and args.save_last_n_steps_state == 1
    assert sum("save_last_n_" in line for line in logs["info"]) == 2
    args = make_args(parser, save_state=True, save_last_n_epochs_state=3, save_last_n_steps_state=2)
    ft_args.validate_moss_ft_args(args, cuda=False)
    assert args.save_last_n_epochs_state == 3 and args.save_last_n_steps_state == 2
    logs["warning"].clear()
    args = make_args(parser)
    ft_args.validate_moss_ft_args(args, cuda=False)
    assert args.save_last_n_epochs_state is None and any("save_state" in line for line in logs["warning"])
    assert args.save_precision == "bf16"
    for precision in ("fp16", "float", None):
        args = make_args(parser, save_precision=precision)
        ft_args.validate_moss_ft_args(args, cuda=False)
        assert args.save_precision == (precision or "bf16")


def test_resume_of_this_format_passes(parser, tmp_path):
    state = tmp_path / "moss-step00001000-state"
    state.mkdir()
    (state / "COMPLETE.json").write_text("{}", encoding="utf-8")
    (state / "optimizer.bin").write_bytes(b"x")  # a stray file next to the marker does not make it an accelerate state
    ft_args.validate_moss_ft_args(make_args(parser, resume=str(state)), cuda=False)


def test_precision_and_stream_notes(parser, logs):
    args = make_args(parser, mixed_precision="no", moss_stream_fp32=False)  # fp32 compute: the stream is fp32 anyway
    ft_args.validate_moss_ft_args(args, cuda=False)
    assert any("24.7 GiB" in line for line in logs["warning"])
    logs["warning"].clear()
    args = make_args(parser, moss_stream_fp32=False, moss_ft_allow_bf16_stream=True)
    ft_args.validate_moss_ft_args(args, cuda=False)
    assert any("cos 0.58" in line for line in logs["warning"])


def test_weight_decay_note(parser, logs):
    args = make_args(parser, "--optimizer_args", "weight_decay=0.01", moss_ft_lr_phases=PHASES)
    ft_args.validate_moss_ft_args(args, cuda=False)
    assert any("weight_decay" in line and "6e-08" in line for line in logs["info"])
    logs["info"].clear()
    args = make_args(parser, "--optimizer_args", "weight_decay=0.1", learning_rate=1e-4)  # 1e-5: above resolution
    ft_args.validate_moss_ft_args(args, cuda=False)
    assert not any("6e-08" in line for line in logs["info"])


def test_freeze(parser, logs):
    shapes = ft_budget.moss_param_shapes()
    frozen = ft_args.frozen_parameter_names(shapes, r"layers\.0\..*,time_embedding\..*")
    assert len(frozen) == 10 + 4 and "layers.0.proj.q.weight" in frozen and "layers.1.proj.q.weight" not in frozen
    # commas inside {} / [] / () and escaped commas belong to the regex
    assert ft_args.split_freeze_patterns(r"layers\.\d{1,2}\.norm1\.weight,video_embedder\..*") == [
        r"layers\.\d{1,2}\.norm1\.weight", r"video_embedder\..*"]
    assert ft_args.split_freeze_patterns(r"a[,b]c, (x|y,z) ,q\,r") == [r"a[,b]c", r"(x|y,z)", r"q\,r"]
    assert ft_args.split_freeze_patterns(["layers\\.0\\..*", " time_embedding\\..* "]) == [
        "layers\\.0\\..*", "time_embedding\\..*"]
    assert ft_args.split_freeze_patterns("") == [] and ft_args.split_freeze_patterns(None) == []
    assert len(ft_args.frozen_parameter_names(shapes, r"layers\.\d{1,2}\.norm1\.weight")) == 42
    args = make_args(parser, moss_ft_freeze=r"layers\.0\..*,time_embedding\..*")
    ft_args.validate_moss_ft_args(args, cuda=False)
    assert any("froze 14 of 462 tensors" in line for line in logs["info"])


# ---- optimizer tables (4.5) --------------------------------------------------------------------------------------
@pytest.mark.parametrize("optimizer_type,fused,mode", [
    ("", False, "adamw_host"), (None, False, "adamw_host"), ("AdamW", False, "adamw_host"),
    ("adamw", False, "adamw_host"), ("ADAMW", False, "adamw_host"),
    ("Adafactor", True, "adafactor_fused"), ("adafactor", True, "adafactor_fused"),
    ("LoraEasyCustomOptimizer.came.CAME", False, "came_host"), ("CAME", False, "came_host"),
    ("came", False, "came_host"),
])
def test_optimizer_mode_mapping(parser, optimizer_type, fused, mode):
    args = make_args(parser, optimizer_type=optimizer_type, fused_backward_pass=fused)
    assert ft_args.optimizer_mode(args) == mode


@pytest.mark.parametrize("optimizer_type,fused", [
    ("AdamW8bit", False), ("PagedAdamW", False), ("PagedAdamW32bit", False), ("Lion", False), ("Prodigy", False),
    ("AdamWScheduleFree", False), ("DAdaptAdam", False), ("SGDNesterov", False), ("bitsandbytes.optim.AdEMAMix8bit", False),
    ("LoraEasyCustomOptimizer.came.CAME2", False), ("Adafactor", False), ("AdamW", True), ("CAME", True),
])
def test_optimizer_mode_refusals(parser, optimizer_type, fused):
    args = make_args(parser, optimizer_type=optimizer_type, fused_backward_pass=fused)
    with pytest.raises(ValueError) as error:
        ft_args.optimizer_mode(args)
    text = str(error.value)
    if not fused:
        assert "adamw_host" in text and "adafactor_fused" in text and "came_host" in text  # the table is quoted


def _kwargs(parser, mode, *items, **overrides):
    argv = ["--optimizer_args", *items] if items else []
    return ft_args.parse_optimizer_args(make_args(parser, *argv, **overrides), mode)


def test_optimizer_args_per_mode(parser, logs):
    # adamw_host: betas (0.9, 0.999), eps 1e-8, weight_decay 0.0
    assert _kwargs(parser, "adamw_host") == {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}
    assert _kwargs(parser, "adamw_host", "betas=(0.9, 0.95)", "eps=1e-7", "weight_decay=0.01") == {
        "betas": (0.9, 0.95), "eps": 1e-7, "weight_decay": 0.01}
    got = _kwargs(parser, "adamw_host", "betas=0.9,0.95", "weight_decay=0")  # kohya style, an int for a float
    assert got["betas"] == (0.9, 0.95) and got["weight_decay"] == 0.0 and isinstance(got["weight_decay"], float)
    for bad in ("amsgrad=True", "lr=1e-5", "foreach=True", "weight_decay", "betas=(0.9,)", "betas=(0.9, 1.0)",
                "betas=(0.9, 0.999, 0.99)", "eps=0", "eps=-1", "weight_decay=-0.1", "betas=abc", "eps=True"):
        with pytest.raises(ValueError, match="optimizer_args"):
            _kwargs(parser, "adamw_host", bad)

    # came_host: the design's defaults (betas (0.9, 0.999, 0.99), bias_correction False) and came.py's for the rest
    assert _kwargs(parser, "came_host") == {
        "betas": (0.9, 0.999, 0.99), "bias_correction": False, "weight_decay": 0.0, "weight_decouple": True,
        "fixed_decay": False, "cautious_weight_decay": False, "eps1": 1e-30, "eps2": 1e-16, "clip_threshold": 1.0,
        "update_strategy": "unmodified", "non_factored_confidence": False}
    got = _kwargs(parser, "came_host", "betas=(0.9, 0.999, 0.9999)", "bias_correction=True", "weight_decay=0.01",
                  "update_strategy=cautious", "non_factored_confidence=True", "ams_bound=False", "kahan_sum=False")
    assert got["betas"] == (0.9, 0.999, 0.9999) and got["bias_correction"] is True
    assert got["update_strategy"] == "cautious" and got["non_factored_confidence"] is True
    assert "ams_bound" not in got and "kahan_sum" not in got
    with pytest.raises(ValueError, match="update_strategy=cautious"):  # came.py's deprecated switch: not a 4.5 key
        _kwargs(parser, "came_host", "cautious=True")
    # GPU-only tuning keys: ignored with ONE warning
    logs["warning"].clear()
    got = _kwargs(parser, "came_host", "state_storage_dtype=bfloat16", "state_storage_device=cpu", "compile_step=True",
                  "group_shapes=True", "group_shapes_max_numel=1000", "foreach=True", "sync_chunk_size=64")
    assert got == _kwargs(parser, "came_host", "foreach=True")
    assert len([line for line in logs["warning"] if "ignored" in line]) == 2  # one per call above
    assert "compile_step" in logs["warning"][0] and "group_shapes_max_numel" in logs["warning"][0]
    for bad, match in (("ams_bound=True", "ams_bound"), ("kahan_sum=True", "kahan_sum"),
                       ("update_strategy=grams", "update_strategy"), ("update_strategy=both", "update_strategy"),
                       ("update_strategy=foo", "update_strategy"), ("lr=1e-5", "lr"), ("betas=(0.9, 0.999)", "betas"),
                       ("eps1=-1", "eps1"), ("clip_threshold=0", "clip_threshold")):
        with pytest.raises(ValueError, match=re.escape(match)):
            _kwargs(parser, "came_host", bad)
    # beta3 > 0.99 needs bias_correction=True (design 3.3), quoting the measured amplification
    with pytest.raises(ValueError, match=re.escape("x32.7")):
        _kwargs(parser, "came_host", "betas=(0.9, 0.999, 0.9999)")
    with pytest.raises(ValueError, match="bias_correction"):
        _kwargs(parser, "came_host", "betas=(0.9, 0.999, 0.995)")
    assert _kwargs(parser, "came_host", "betas=(0.9, 0.999, 0.99)")["betas"][2] == 0.99  # allowed uncorrected

    # adafactor_fused: scale_parameter / relative_step / warmup_init False and no beta1 (design 3.2)
    required = {"scale_parameter": False, "relative_step": False, "warmup_init": False}
    assert _kwargs(parser, "adafactor_fused") == required
    assert _kwargs(parser, "adafactor_fused", "scale_parameter=False", "relative_step=False", "warmup_init=False",
                   "beta1=None", "eps=(1e-30, 1e-3)", "clip_threshold=1.0", "decay_rate=-0.8", "weight_decay=0.0") == {
        **required, "beta1": None, "eps": (1e-30, 1e-3), "clip_threshold": 1.0, "decay_rate": -0.8, "weight_decay": 0.0}
    for bad in ("scale_parameter=True", "relative_step=True", "warmup_init=True", "beta1=0.9", "foo=1"):
        with pytest.raises(ValueError, match=re.escape(bad.split("=")[0])):
            _kwargs(parser, "adafactor_fused", bad)


def test_optimizer_kwargs_fit_the_host_optimizers(parser):
    """parse_optimizer_args' dicts are the keyword arguments of ft_optim.HostAdamW / HostCAME (T2) as they are."""
    import torch

    ft_optim = pytest.importorskip("library.moss.ft_optim")
    param = torch.nn.Parameter(torch.zeros(4, 4))
    for mode, cls, items in (("adamw_host", ft_optim.HostAdamW, ("betas=(0.9, 0.95)", "weight_decay=0.01")),
                             ("came_host", ft_optim.HostCAME, ("betas=(0.9, 0.999, 0.9999)", "bias_correction=True",
                                                               "update_strategy=cautious"))):
        kwargs = _kwargs(parser, mode, *items)
        optimizer = cls([param], lr=5e-6, **kwargs)
        group = optimizer.param_groups[0]
        assert tuple(group["betas"]) == kwargs["betas"] and group["weight_decay"] == kwargs["weight_decay"]


# ---- resource checks (ft_budget, 4.5 "Start-up checks") ----------------------------------------------------------
def test_resource_checks(parser, tmp_path, logs, monkeypatch):
    # the bare checks at their boundaries, with injected callables and with psutil / shutil mocked
    memory = lambda available: (lambda: types.SimpleNamespace(available=available, total=available))  # noqa: E731
    assert not ft_budget.check_host_memory(100, virtual_memory=memory(99)).ok
    assert ft_budget.check_host_memory(100, virtual_memory=memory(100)).ok
    disk = lambda free: (lambda path: types.SimpleNamespace(total=free, used=0, free=free))  # noqa: E731
    assert not ft_budget.check_disk_space(str(tmp_path), 100, disk_usage=disk(99)).ok
    assert ft_budget.check_disk_space(str(tmp_path / "not" / "yet"), 100, disk_usage=disk(100)).ok
    import psutil
    import shutil

    monkeypatch.setattr(psutil, "virtual_memory", memory(5))
    monkeypatch.setattr(shutil, "disk_usage", disk(5))
    assert ft_budget.check_host_memory(5).ok and not ft_budget.check_host_memory(6).ok
    assert ft_budget.check_disk_space(str(tmp_path), 5).ok and not ft_budget.check_disk_space(str(tmp_path), 6).ok

    def unreadable(path):  # a drive letter that is not mounted: refused with the reason, not a traceback
        raise FileNotFoundError(3, "The system cannot find the path specified", path)

    check = ft_budget.check_disk_space(str(tmp_path), 1, disk_usage=unreadable)
    assert not check.ok and "not readable" in check.where

    # through the start-up validation: host available < host total + 8 GiB -> refused; free space at the state
    # directory < what the run keeps at its peak (states and DiT saves, SavePlan) -> refused
    out = tmp_path / "out"
    group = FakeGroup(tmp_path, images=64, batch_size=4)

    def args_for(**extra):
        base = dict(output_dir=str(out), save_state=True, gradient_accumulation_steps=2, cache_latents_to_disk=True,
                    cache_text_encoder_outputs_to_disk=True, max_data_loader_n_workers=3)
        return make_args(parser, **{**base, **extra})

    report = ft_budget.startup_report(args_for(), group, "adamw_host", exists=lambda path: True)
    host_required = report.host.required
    assert host_required == report.host.total + 8 * GIB
    assert report.host.total / GIB == pytest.approx(85.23, abs=0.05)  # design 2.2: ~85 GiB (adamw_host, GA > 1)
    # no save_every_n_*: the train-end state and the final DiT only
    assert (report.plan.state_dirs, report.plan.dit_files) == (1, 1)
    disk_required = report.state_bytes + report.dit_bytes
    assert report.state_bytes / 1e9 == pytest.approx(61.0, abs=0.05) and report.dit_bytes / 1e9 == pytest.approx(
        10.16, abs=0.05)

    validate_with(args_for(), group, virtual_memory=memory(host_required), disk_usage=disk(disk_required))
    with pytest.raises(ValueError, match="host RAM"):
        validate_with(args_for(), group, virtual_memory=memory(host_required - 1), disk_usage=disk(disk_required))
    with pytest.raises(ValueError, match="free"):
        validate_with(args_for(), group, virtual_memory=memory(host_required), disk_usage=disk(disk_required - 1))
    # GA 1 needs no accumulator (~66 GiB); --moss_ft_clip_mode exact needs it again
    report = ft_budget.startup_report(args_for(gradient_accumulation_steps=1), group, "adamw_host",
                                      exists=lambda path: True)
    assert report.host.total / GIB == pytest.approx(66.30, abs=0.05)
    report = ft_budget.startup_report(args_for(gradient_accumulation_steps=1, moss_ft_clip_mode="exact"), group,
                                      "adamw_host", exists=lambda path: True)
    assert report.host.total / GIB == pytest.approx(85.23, abs=0.05)
    # --moss_ft_skip_resource_check: logs the estimates, refuses nothing
    logs["warning"].clear()
    validate_with(args_for(moss_ft_skip_resource_check=True), group, virtual_memory=memory(1), disk_usage=disk(1))
    assert any("moss_ft_skip_resource_check" in line for line in logs["warning"])
    # without saved states only the DiT saves need room
    validate_with(args_for(save_state=False), group, virtual_memory=memory(host_required),
                  disk_usage=disk(report.dit_bytes))


def test_cache_and_gpu_warnings(parser, tmp_path, logs):
    group = FakeGroup(tmp_path, images=64, batch_size=4)
    args = make_args(parser, output_dir=str(tmp_path / "out"), cache_latents_to_disk=True,
                     cache_text_encoder_outputs_to_disk=True)
    report = ft_budget.startup_report(args, group, "adamw_host", exists=lambda path: False)
    te = 64 * 300 * 8196  # 8 KiB per text token (+ 4 B mask), T = 300, one variant
    assert report.te_cache_bytes == te and report.latent_cache_bytes == 64 * 128 * 64 * 64 * 4
    # the caches are written next to the images: warn (do not refuse) when they do not fit
    small = lambda path: types.SimpleNamespace(total=1 << 50, used=0, free=te // 2)  # noqa: E731
    validate_with(make_args(parser, output_dir=str(tmp_path / "out"), cache_latents_to_disk=True,
                            cache_text_encoder_outputs_to_disk=True, moss_ft_skip_resource_check=True),
                  group, disk_usage=small, exists=lambda path: False)
    assert any("cache" in line and "free" in line for line in logs["warning"])
    # existing cache files are not counted
    report = ft_budget.startup_report(args, group, "adamw_host", exists=lambda path: True)
    assert report.te_cache_bytes == 0 and report.latent_cache_bytes == 0
    # 1536^2 at batch 2 is past the 30.5 GiB warning line (design 2.1: 31.2 GiB device estimate)
    logs["warning"].clear()
    validate_with(make_args(parser, output_dir=str(tmp_path / "out")),
                  FakeGroup(tmp_path, images=8, batch_size=2, reso=(1536, 1536)))
    assert any("30.5 GiB" in line for line in logs["warning"])
    # one line per budget item, for the largest bucket and batch size
    lines = report.lines
    for item in ("compute weights", "checkpointed block inputs", "recompute", "grads in flight", "rope",
                 "total allocated", "device", "fp32 master", "exp_avg", "exp_avg_sq", "accumulator", "pinned rings",
                 "process", "host total", "trained DiT", "training state", "TE cache", "latent cache", "s/step"):
        assert sum(item in line for line in lines) >= 1, item


# ---- review fixes: the LR scheduler at start-up step 4, DataLoader workers -----------------------------------------
@pytest.mark.parametrize("name, warmup", [("linear", 5), ("cosine", 5), ("cosine_with_restarts", 0), ("polynomial", 5),
                                          ("constant", 0), ("constant_with_warmup", 5), ("inverse_sqrt", 5),
                                          ("warmup_stable_decay", 5), ("CosineAnnealingLR", 0)])
def test_allowed_schedulers_build_and_are_resumable(parser, tmp_path, name, warmup):
    """Step 4 builds the run's scheduler on a stand-in optimizer and round-trips its state and the param groups through
    torch.load(weights_only=True), as a training state does (ft_state)."""
    validate_with(make_args(parser, lr_scheduler=name, lr_warmup_steps=warmup), FakeGroup(tmp_path))


@pytest.mark.parametrize("overrides, match", [
    ({"lr_scheduler": "constant", "lr_warmup_steps": 100}, "constant_with_warmup"),  # kohya: no warmup for constant
    ({"lr_scheduler": "reduce_lr_on_plateau"}, "reduce_lr_on_plateau"),  # its step() needs a metric
    ({"lr_scheduler": "cosine_with_min_lr"}, "min_lr"),
    ({"lr_scheduler_type": "ExponentialLR", "lr_scheduler_args": ["gama=0.5"]}, "ExponentialLR"),
])
def test_scheduler_problems_surface_at_step_4(parser, tmp_path, overrides, match):
    """Review fix: a scheduler setting get_scheduler_fix refuses failed only at start-up step 8 (make_lr_scheduler),
    after the latent / text-encoder caching and the DiT load. The dataset-free part cannot know the run length; the
    dataset part (step 4, before any caching) builds the scheduler and refuses with get_scheduler_fix's reason."""
    args = make_args(parser, **overrides)
    ft_args.validate_moss_ft_args(args, cuda=False)
    with pytest.raises(ValueError, match=re.escape(match)) as error:
        validate_with(args, FakeGroup(tmp_path))
    assert str(error.value).startswith("MOSS full FT: ")


def test_restart_scheduler_with_first_cycle_max_steps(parser, tmp_path):
    """Review fix: get_scheduler_fix reads args.validation_split for first_cycle_max_steps (the LoraEasyCustomOptimizer
    restart schedulers); moss_train.py's parser had no such attribute (AttributeError at step 8)."""
    pytest.importorskip("LoraEasyCustomOptimizer.CosineAnnealingWarmRestarts")
    args = make_args(parser, lr_scheduler_type="LoraEasyCustomOptimizer.CosineAnnealingWarmRestarts."
                     "CosineAnnealingWarmRestarts",
                     lr_scheduler_args=["first_cycle_max_steps=10", "warmup_steps=2", "gamma=0.9", "min_lr=1e-6"])
    assert args.validation_split == 0.0
    validate_with(args, FakeGroup(tmp_path))


def test_unresumable_scheduler_is_refused():
    """The generic guard behind the zero_lr_warmup refusal: a scheduler whose state_dict holds objects (kohya's
    ZeroLRWarmupScheduler keeps its inner scheduler, the optimizer and every parameter) is refused."""
    import torch

    from library import train_util

    param = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([param], lr=1e-3)
    inner = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    ft_args.check_scheduler_resumable(inner, optimizer)
    wrapped = train_util.ZeroLRWarmupScheduler(optimizer, inner, warmup_steps=2)
    with pytest.raises(ValueError, match="weights_only") as error:
        ft_args.check_scheduler_resumable(wrapped, optimizer)
    assert "\x1b" not in str(error.value)


def test_many_dataloader_workers_warn(parser, logs):
    """Review fix: every DataLoader worker is a spawned process (Windows) that imports torch and moss_train.py again,
    about 1 GiB of host RAM each; kohya's default is 8. Above 4 a warning (the host budget counts them, ft_budget)."""
    ft_args.validate_moss_ft_args(make_args(parser, max_data_loader_n_workers=8), cuda=False)
    if (os.cpu_count() or 1) > 4:
        assert any("max_data_loader_n_workers" in line for line in logs["warning"]), logs["warning"]
    logs["warning"].clear()
    ft_args.validate_moss_ft_args(make_args(parser, max_data_loader_n_workers=3, persistent_data_loader_workers=True),
                                  cuda=False)
    assert not any("max_data_loader_n_workers" in line for line in logs["warning"])


def test_host_budget_counts_workers_and_unsloth_classes(parser, tmp_path):
    """The start-up report feeds the review fixes of ft_budget: workers above 4 (min(workers, cpu count), as
    moss_train.py starts them) and the power-of-two classes of the unsloth block inputs of the dataset's tiers."""
    group = FakeGroup(tmp_path, images=64, batch_size=4)
    args = make_args(parser, output_dir=str(tmp_path / "out"), max_data_loader_n_workers=8)
    report = ft_budget.startup_report(args, group, "adamw_host", exists=lambda path: True)
    workers = min(8, os.cpu_count() or 1)
    assert report.host.process == ft_budget.PROCESS_BYTES + max(0, workers - 4) * GIB
    args = make_args(parser, output_dir=str(tmp_path / "out"), max_data_loader_n_workers=2,
                     unsloth_offload_checkpointing=True)
    report = ft_budget.startup_report(args, group, "adamw_host", exists=lambda path: True)
    assert report.host.pinned_checkpoints == ft_budget.unsloth_pinned_bytes([(4096, 4)]) == 42 * 448 * (1 << 20)
    assert any("pinned checkpoint inputs" in line and "18.38 GiB" in line for line in report.lines), report.lines


def _example_run(parser, tmp_path, **extra):
    """docs/moss_train.md's example config (design 8.1) at 16k images: batch 4 x GA 2 = 2,000 steps per epoch, phases
    2 + 6 epochs = 16,000 steps; an epoch DiT save every epoch, a step save + state every 1,000 steps."""
    values = dict(output_dir=str(tmp_path / "out"), moss_ft_lr_phases=PHASES, lr_warmup_steps=100,
                  gradient_accumulation_steps=2, save_every_n_epochs=1, save_state=True, save_last_n_epochs_state=1,
                  save_every_n_steps=1000, save_last_n_steps_state=1, cache_latents_to_disk=True,
                  cache_text_encoder_outputs_to_disk=True, max_data_loader_n_workers=3,
                  persistent_data_loader_workers=True)
    values.update(extra)
    return make_args(parser, **values)


def test_disk_check_counts_what_the_run_keeps(parser, tmp_path):
    """Review fix: the start-up disk check wanted 2 states + 1 DiT save (132 GB) where the guide's example keeps 1 epoch
    + 1 step + the pinned phase-1 state + 1 being written = 4 states (244 GB) and, without save_last_n_epochs /
    save_last_n_steps, 7 epoch + 16 step + 1 final DiT saves (244 GB): a drive with 200 GB free passed and filled up
    mid-run. Now it counts the run's SavePlan (kohya's removal rules, the keep policy, the pinned states)."""
    group = FakeGroup(tmp_path, images=16_000, batch_size=4)
    args = _example_run(parser, tmp_path)
    validate_with(args, group)
    assert args.max_train_steps == 16_000
    report = ft_budget.startup_report(args, group, "adamw_host", exists=lambda path: True)
    plan = report.plan
    assert (plan.epochs, plan.epoch_dit, plan.step_dit, plan.dit_files) == (8, 7, 16, 24)
    assert (plan.epoch_states, plan.step_states, plan.pinned_states, plan.state_dirs) == (1, 1, 1, 4)
    required = 24 * report.dit_bytes + 4 * report.state_bytes
    assert required / 1e9 == pytest.approx(487.9, abs=0.5)
    free = lambda free_bytes: (lambda path: types.SimpleNamespace(total=1 << 50, used=0, free=free_bytes))  # noqa: E731
    validate_with(_example_run(parser, tmp_path), group, disk_usage=free(required))
    with pytest.raises(ValueError, match="free"):
        validate_with(_example_run(parser, tmp_path), group, disk_usage=free(required - 1))
    with pytest.raises(ValueError, match="free"):  # the reviewer's 200 GB drive
        validate_with(_example_run(parser, tmp_path), group, disk_usage=free(200 * 10**9))
    # kohya's removal of old DiT saves: save_last_n_steps counts steps (keeps the saves of the last 2,000 steps: 3),
    # save_last_n_epochs counts saves
    report = ft_budget.startup_report(_example_run(parser, tmp_path, save_last_n_steps=2000, save_last_n_epochs=2,
                                                   max_train_steps=16_000), group, "adamw_host",
                                      exists=lambda path: True)
    assert (report.plan.epoch_dit, report.plan.step_dit, report.plan.dit_files) == (2, 3, 6)
    # no save_state: no states; save_state_on_train_end: the train-end state only
    report = ft_budget.startup_report(_example_run(parser, tmp_path, save_state=False, max_train_steps=16_000), group,
                                      "adamw_host", exists=lambda path: True)
    assert report.plan.state_dirs == 0
    report = ft_budget.startup_report(_example_run(parser, tmp_path, save_state=False, save_state_on_train_end=True,
                                                   max_train_steps=16_000), group, "adamw_host",
                                      exists=lambda path: True)
    assert report.plan.state_dirs == 1
    # --moss_ft_no_keep_phase_states: nothing pinned
    report = ft_budget.startup_report(_example_run(parser, tmp_path, moss_ft_no_keep_phase_states=True,
                                                   max_train_steps=16_000), group, "adamw_host",
                                      exists=lambda path: True)
    assert report.plan.pinned_states == 0 and report.plan.state_dirs == 3
