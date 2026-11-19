"""CPU tests of the --moss_te_live arguments and refusals (library/moss/ft_args.py; docs/moss_ft_te_live_fp16_plan.md
1.7): the new flags parse (also from a TOML config), every refusal of the live path, live flags without
--moss_te_live, and the cached path's caption refusals that replaced validate_moss_args' assert (each pointing to
--moss_te_live), plus the 2,000-image rule asking only for the latent cache in live mode.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_te_live_args.py -q -p no:cacheprovider
"""
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest

from library.moss import ft_args, ft_budget


@pytest.fixture(scope="module")
def parser():
    return ft_args.setup_parser()


@pytest.fixture
def logs(monkeypatch):
    out = {"info": [], "warning": []}
    for module in (ft_args, ft_budget):
        monkeypatch.setattr(module.logger, "info", lambda m, *a, **k: out["info"].append(str(m)))
        monkeypatch.setattr(module.logger, "warning", lambda m, *a, **k: out["warning"].append(str(m)))
    return out


def live_args(parser, *argv, **overrides):
    """A valid live configuration (bf16 + fp32 stream, checkpointing, latent cache, --moss_te_live)."""
    args = parser.parse_args(["--moss_stream_fp32", "--gradient_checkpointing", "--cache_latents", "--moss_te_live",
                              *argv])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def cached_args(parser, *argv, **overrides):
    args = parser.parse_args(["--moss_stream_fp32", "--gradient_checkpointing", "--cache_latents",
                              "--cache_text_encoder_outputs", *argv])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def subset(**kw):
    values = dict(flip_aug=False, shuffle_caption=False, caption_tag_dropout_rate=0.0, caption_dropout_rate=0.0,
                  token_warmup_step=0, caption_dropout_every_n_epochs=0, image_dir="/data/subset")
    values.update(kw)
    return types.SimpleNamespace(**values)


class Group:
    def __init__(self, root, images=16, variants=0, **subset_kw):
        infos = {f"img{i}": types.SimpleNamespace(bucket_reso=(1024, 1024), jitter_bucket_info=None,
                                                  absolute_path=os.path.join(str(root), f"img{i:05d}.png"))
                 for i in range(images)}
        self.datasets = [types.SimpleNamespace(batch_size=1, image_data=infos, validation_split=0.0,
                                               caption_aug_variant_count=variants, latents_aug_variant_count=0,
                                               subsets=[subset(**subset_kw)])]
        self.images = images

    def __len__(self):
        return self.images


def validate(args, group=None, cuda=False):
    ft_args.validate_moss_ft_args(args, group, cuda=cuda,
                                  virtual_memory=lambda: types.SimpleNamespace(available=1 << 50, total=1 << 50),
                                  disk_usage=lambda path: types.SimpleNamespace(total=1 << 60, used=0, free=1 << 60),
                                  exists=lambda path: True)


def test_defaults_and_toml(parser, tmp_path, monkeypatch):
    from library import train_util

    defaults = parser.parse_args([])
    assert not defaults.moss_te_live
    assert (defaults.moss_te_live_ahead, defaults.moss_te_live_ring_slots, defaults.moss_te_live_host,
            defaults.moss_te_live_threads) == (32, 2, "copy", 8)
    assert {k: getattr(defaults, k) for k in ft_args.TE_LIVE_DEFAULTS} == ft_args.TE_LIVE_DEFAULTS
    config = tmp_path / "live.toml"
    config.write_text("moss_te_live = true\nmoss_te_live_ahead = 16\nmoss_te_live_ring_slots = 4\n"
                      "moss_te_live_host = \"mmap\"\nmoss_te_live_threads = 4\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["moss_train.py", "--config_file", str(config)])
    args = train_util.read_config_from_file(parser.parse_args(), parser, unknown_keys="error")
    assert (args.moss_te_live, args.moss_te_live_ahead, args.moss_te_live_ring_slots, args.moss_te_live_host,
            args.moss_te_live_threads) == (True, 16, 4, "mmap", 4)
    with pytest.raises(SystemExit):
        parser.parse_args(["--moss_te_live_host", "ram"])


def test_user_guide_example_config_validates(parser, tmp_path, logs, monkeypatch):
    """docs/moss_train.md section 2 (the live variant of design 8.1) reads with unknown keys as errors and passes the
    dataset-free validation on CUDA."""
    import re

    from library import train_util

    guide = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "moss_train.md")
    with open(guide, "r", encoding="utf-8") as handle:
        text = handle.read()
    match = re.search(r"```toml\n(.*?)```", text[text.index("## 2. Example config"):], re.DOTALL)
    config = tmp_path / "moss_fft_live.toml"
    config.write_text(match.group(1), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["moss_train.py", "--config_file", str(config)])
    args = train_util.read_config_from_file(parser.parse_args(), parser, unknown_keys="error")
    assert args.moss_te_live and args.moss_te_live_ahead == 8 and not args.cache_text_encoder_outputs
    ft_args.validate_moss_ft_args(args, cuda=True)
    assert not logs["warning"], logs["warning"]


def test_live_passes_without_te_cache_flags(parser, logs):
    args = live_args(parser)
    validate(args)
    assert not args.cache_text_encoder_outputs
    assert any("--moss_te_live: the captions are encoded at train time" in line for line in logs["info"])


LIVE_REFUSALS = [
    ("with_te_cache", {"cache_text_encoder_outputs": True}, "choose one"),
    ("with_te_cache_to_disk", {"cache_text_encoder_outputs_to_disk": True}, "choose one"),
    ("with_caption_variants", {"cache_caption_variants": 3}, "cache_caption_variants is a TE-cache feature"),
    ("ahead_0", {"moss_te_live_ahead": 0}, "moss_te_live_ahead"),
    ("ahead_bool", {"moss_te_live_ahead": True}, "moss_te_live_ahead"),
    ("ring_3", {"moss_te_live_ring_slots": 3}, "moss_te_live_ring_slots"),
    ("ring_1", {"moss_te_live_ring_slots": 1}, "moss_te_live_ring_slots"),
    ("threads_0", {"moss_te_live_threads": 0}, "moss_te_live_threads"),
    ("threads_over_cpu_count", {"moss_te_live_threads": (os.cpu_count() or 1) + 1}, "moss_te_live_threads"),
    ("host_unknown", {"moss_te_live_host": "ram"}, "moss_te_live_host"),
    ("pinned_without_cuda", {"moss_te_live_host": "pinned"}, "needs CUDA"),
]


@pytest.mark.parametrize("name,overrides,match", LIVE_REFUSALS, ids=[case[0] for case in LIVE_REFUSALS])
def test_live_refusals(parser, name, overrides, match):
    with pytest.raises(ValueError, match=match) as refused:
        validate(live_args(parser, **overrides))
    assert str(refused.value).startswith(ft_args.PREFIX)


def test_pinned_with_cuda_passes(parser):
    validate(live_args(parser, moss_te_live_host="pinned"), cuda=True)


@pytest.mark.parametrize("name,value", [("moss_te_live_ahead", 8), ("moss_te_live_ring_slots", 4),
                                        ("moss_te_live_host", "mmap"), ("moss_te_live_threads", 2)])
def test_live_flags_need_moss_te_live(parser, name, value):
    with pytest.raises(ValueError, match=f"--{name} is only used with --moss_te_live"):
        validate(cached_args(parser, **{name: value}))


def test_cached_path_still_needs_the_te_cache(parser):
    with pytest.raises(ValueError, match="--cache_text_encoder_outputs is required.*--moss_te_live"):
        validate(parser.parse_args(["--moss_stream_fp32", "--gradient_checkpointing", "--cache_latents"]))
    args = parser.parse_args(["--moss_stream_fp32", "--gradient_checkpointing", "--cache_latents",
                              "--cache_text_encoder_outputs_to_disk"])
    validate(args)
    assert args.cache_text_encoder_outputs  # _to_disk implies it (as before)


CACHED_CAPTION_REFUSALS = [
    ("shuffle", {"shuffle_caption": True}, "shuffle_caption needs --cache_caption_variants"),
    ("tag_dropout", {"caption_tag_dropout_rate": 0.1}, "caption_tag_dropout_rate > 0 needs"),
    ("caption_dropout", {"caption_dropout_rate": 0.05}, "caption_dropout_rate > 0 needs"),
    ("token_warmup", {"token_warmup_step": 10}, "token_warmup_step is step-dependent"),
    ("dropout_every_n_epochs", {"caption_dropout_every_n_epochs": 2}, "caption_dropout_every_n_epochs is not supported"),
]


@pytest.mark.parametrize("name,subset_kw,match", CACHED_CAPTION_REFUSALS, ids=[c[0] for c in CACHED_CAPTION_REFUSALS])
def test_cached_caption_refusals_point_to_live(parser, tmp_path, name, subset_kw, match):
    with pytest.raises(ValueError, match=match) as refused:
        validate(cached_args(parser), Group(tmp_path, **subset_kw))
    assert "or use --moss_te_live" in str(refused.value) and "(subset /data/subset)" in str(refused.value)
    validate(live_args(parser), Group(tmp_path, **subset_kw))  # the live path draws them at every fetch


@pytest.mark.parametrize("subset_kw", [{"shuffle_caption": True}, {"caption_tag_dropout_rate": 0.1},
                                       {"caption_dropout_rate": 0.05}])
def test_caption_variants_still_serve_the_cached_path(parser, tmp_path, subset_kw):
    validate(cached_args(parser, "--cache_caption_variants", "3"), Group(tmp_path, variants=3, **subset_kw))


def test_two_thousand_images_need_only_the_latent_cache_on_disk_in_live_mode(parser, tmp_path):
    with pytest.raises(ValueError, match="--cache_latents_to_disk is required above") as refused:
        validate(live_args(parser), Group(tmp_path, images=2001))
    assert "cache_text_encoder_outputs_to_disk" not in str(refused.value)
    validate(live_args(parser, "--cache_latents_to_disk"), Group(tmp_path, images=2001))
    with pytest.raises(ValueError, match="cache_text_encoder_outputs_to_disk"):
        validate(cached_args(parser, "--cache_latents_to_disk"), Group(tmp_path, images=2001))


def test_live_budget_lines(parser, tmp_path, logs):
    validate(live_args(parser, "--moss_te_live_ahead", "8"), Group(tmp_path))
    text = "\n".join(logs["info"])
    assert "host live text encoder (copy): decoder layers 12.89 GiB (private), pinned staging 1.00 GiB" in text
    assert "GPU live text encoder: ring 2 x 416.6 MiB = 0.81 GiB" in text
    assert "disk TE cache: none (--moss_te_live" in text
    assert "text encoder" in next(line for line in logs["info"] if " speed ~" in line)


def test_live_host_refusal_suggests_mmap(parser, tmp_path, logs):
    args = live_args(parser, "--gradient_accumulation_steps", "2", "--max_data_loader_n_workers", "2")
    needed = []

    def tight():
        return types.SimpleNamespace(available=100 * (1 << 30), total=128 * (1 << 30))

    with pytest.raises(ValueError, match="--moss_te_live_host mmap") as refused:
        ft_args.validate_moss_ft_args(args, Group(tmp_path), cuda=False, virtual_memory=tight,
                                      disk_usage=lambda p: types.SimpleNamespace(total=1 << 60, used=0, free=1 << 60),
                                      exists=lambda p: True)
    needed.append(str(refused.value))
    assert "107." in needed[0]  # 99.1 GiB for adamw_host at GA 2 with the encoder in RAM + 8 GiB margin
    assert "trades RAM for disk reads" in needed[0]  # review fix: the hint says what mmap costs
    assert not any("--moss_te_live_host mmap:" in line for line in logs["warning"])
    args = live_args(parser, "--gradient_accumulation_steps", "2", "--max_data_loader_n_workers", "2",
                     "--moss_te_live_host", "mmap")
    ft_args.validate_moss_ft_args(args, Group(tmp_path), cuda=False, virtual_memory=tight,
                                  disk_usage=lambda p: types.SimpleNamespace(total=1 << 60, used=0, free=1 << 60),
                                  exists=lambda p: True)
    # review fix: mmap passes the check (0 GiB counted for the layers) but 100 GiB cannot also hold the 12.9 GiB of
    # layers in the file cache next to the ~94 GiB the run needs: the warning prices the passes at the disk rate
    warning = next(line for line in logs["warning"] if "--moss_te_live_host mmap:" in line)
    assert "13.8 GB from the disk" in warning and "s per pass of" in warning
