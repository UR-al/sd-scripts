"""CPU tests for library/config_validation.py: --config_file keys must name an option, values must fit it.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m pytest tests/test_config_validation.py
"""
import ast
import os
import sys

import pytest

from library import config_validation, train_util
from library.config_validation import ConfigValidationError, validate_config

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def parser():
    import moss_train_network

    return moss_train_network.setup_parser()


def _problems(config, parser, **kwargs):
    with pytest.raises(ConfigValidationError) as info:
        validate_config(config, parser, **kwargs)
    return str(info.value)


def test_unknown_keys_warn_with_line_and_suggestion(parser, caplog):
    text = 'learning_rate = 2e-5\n\n[moss]\nmoss_sigma_cutof = 0.9\nSDPA = true\n'
    config = {"learning_rate": 2e-5, "moss_sigma_cutof": 0.9, "SDPA": True}
    with caplog.at_level("WARNING"):
        out = validate_config(config, parser, source="c.toml", text=text)
    assert out == config  # kept, as before (the GUI writes its own keys)
    assert "c.toml:4: 'moss_sigma_cutof' (did you mean moss_sigma_cutoff?)" in caplog.text
    assert "c.toml:5: 'SDPA' (did you mean sdpa?)" in caplog.text and "--strict_config_keys" in caplog.text


def test_strict_and_ignore_modes(parser, caplog):
    message = _problems({"moss_sigma_cutof": 0.9, "seed": "x"}, parser, unknown="error")
    assert "'moss_sigma_cutof' (did you mean moss_sigma_cutoff?): unknown key" in message and "seed = 'x'" in message
    with caplog.at_level("WARNING"):
        assert validate_config({"network_dim": 4}, parser, unknown="ignore") == {"network_dim": 4}
    assert "network_dim" not in caplog.text
    with pytest.raises(ValueError):
        validate_config({}, parser, unknown="loud")


def test_aliases_resolve_to_the_dest_and_duplicates_fail(parser):
    assert validate_config({"learning_rate_dora": 1e-4}, parser) == {"learning_rate_DoRA": 1e-4}
    message = _problems({"learning_rate_DoRA": 1e-4, "learning_rate_dora": 2e-4}, parser)
    assert "sets the same option" in message


def test_booleans_must_be_toml_booleans(parser):
    assert validate_config({"cache_latents": True, "use_ramtorch": False}, parser) == {"cache_latents": True,
                                                                                       "use_ramtorch": False}
    message = _problems({"cache_latents": "false", "use_zero_cond_dropout": "False"}, parser)
    assert message.count("needs true or false") == 2  # store_true and type=bool: the string "False" is truthy


def test_numbers_and_strings_convert_like_the_command_line(parser):
    out = validate_config({"learning_rate": "1e-4", "max_train_epochs": 10.0, "network_dim": "16",
                           "moss_sample_uncond_skip_layers": 22, "network_alpha": 4}, parser)
    assert out == {"learning_rate": 1e-4, "max_train_epochs": 10, "network_dim": 16,
                   "moss_sample_uncond_skip_layers": "22", "network_alpha": 4.0}
    assert type(out["max_train_epochs"]) is int and type(out["network_alpha"]) is float
    message = _problems({"max_train_epochs": 2.5, "learning_rate": True, "seed": "abc", "output_name": [1]}, parser)
    for key in ("max_train_epochs = 2.5", "learning_rate = True", "seed = 'abc'", "output_name = [1]"):
        assert key in message, (key, message)


def test_lists_and_choices(parser):
    out = validate_config({"network_args": "algo=lokr", "optimizer_args": ["weight_decay=0.01"],
                           "text_encoder_lr": 1e-5, "timestep_sampling": "moss"}, parser)
    assert out["network_args"] == ["algo=lokr"]  # a bare string would be iterated character by character
    assert out["optimizer_args"] == ["weight_decay=0.01"] and out["text_encoder_lr"] == [1e-5]
    assert "is not one of" in _problems({"timestep_sampling": "mos"}, parser)


def test_custom_types_convert_strings_only(parser):
    out = validate_config({"sigmoid_scale": "Style", "lr_warmup_steps": "10%", "lr_decay_steps": 50}, parser)
    assert out == {"sigmoid_scale": "style", "lr_warmup_steps": 0.1, "lr_decay_steps": 50}
    assert "sigmoid_scale" in _problems({"sigmoid_scale": "sharp"}, parser)


def test_moss_exclude_blocks_take_an_array_and_check_strings(parser, tmp_path, monkeypatch):
    # A custom type (type=str refused an array): strings are checked, an array or integer reaches the parser as written
    from library.moss.train_utils import apply_moss_block_exclusion

    for value in ([22, 23], 22, " 22, 1 "):
        assert validate_config({"moss_lora_exclude_blocks": value}, parser) == {"moss_lora_exclude_blocks": value}
    assert "needs a string" in _problems({"moss_sample_uncond_skip_layers": [22]}, parser)  # str options still do
    assert parser.parse_args([]).moss_lora_exclude_blocks == ""
    assert parser.parse_args(["--moss_lora_exclude_blocks", "22,23"]).moss_lora_exclude_blocks == "22,23"
    with pytest.raises(SystemExit):
        parser.parse_args(["--moss_lora_exclude_blocks", "22,x"])

    path = tmp_path / "config.toml"
    path.write_text('[network_arguments]\nnetwork_args = ["algo=lokr", "preset=moss"]\n'
                    "moss_lora_exclude_blocks = [22, 23]\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["moss_train_network.py", "--config_file", str(path)])
    args = train_util.read_config_from_file(parser.parse_args(), parser)
    assert args.moss_lora_exclude_blocks == [22, 23] and apply_moss_block_exclusion(args) == [22, 23]
    key, _, patterns = args.network_args[-1].partition("=")
    assert args.network_args[:2] == ["algo=lokr", "preset=moss"] and key == "exclude_patterns"
    assert ast.literal_eval(patterns) == [r"layers\.22(\..*)?", r"layers\.23(\..*)?"]

    for text, error in (('"22,x"', "takes block indices"), ('"42"', "index 42 is outside 0..41")):
        path.write_text(f"moss_lora_exclude_blocks = {text}\n", encoding="utf-8")
        with pytest.raises(ConfigValidationError, match=r"config\.toml:1: moss_lora_exclude_blocks = .*" + error):
            train_util.read_config_from_file(parser.parse_args(), parser)


def test_read_config_from_file(parser, tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[a]\nmax_train_epochs = "4"\n[b]\nlearning_rate_dora = 1e-4\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["moss_train_network.py", "--config_file", str(path)])  # read_config re-parses argv
    args = train_util.read_config_from_file(parser.parse_args(), parser)
    assert args.max_train_epochs == 4 and args.learning_rate_DoRA == 1e-4

    path.write_text('max_train_epochs = "four"\n', encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="needs an integer"):
        train_util.read_config_from_file(parser.parse_args(), parser)

    path.write_text("max_train_epoch = 4\n", encoding="utf-8")
    assert train_util.read_config_from_file(parser.parse_args(), parser).max_train_epoch == 4  # warning only
    assert train_util.read_config_from_file(parser.parse_args(), parser, unknown_keys="ignore").max_train_epoch == 4
    monkeypatch.setattr(sys, "argv", ["moss_train_network.py", "--config_file", str(path), "--strict_config_keys"])
    with pytest.raises(ConfigValidationError, match="max_train_epochs"):
        train_util.read_config_from_file(parser.parse_args(), parser)
    path.write_text("max_train_epoch = 4\nstrict_config_keys = true\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["moss_train_network.py", "--config_file", str(path)])
    with pytest.raises(ConfigValidationError, match="max_train_epochs"):
        train_util.read_config_from_file(parser.parse_args(), parser)


def test_example_moss_config_is_valid(parser):
    import toml

    example = os.path.join(HERE, "..", "..", "..", "examples", "moss", "config_moss.toml")
    if not os.path.isfile(example):
        pytest.skip("examples/moss/config_moss.toml lives in the LoRA_Easy_Training_Scripts checkout")
    with open(example, encoding="utf-8") as f:
        text = f.read()
    flat = {}
    for key, value in toml.loads(text).items():
        flat.update(value if isinstance(value, dict) else {key: value})
    out = validate_config(flat, parser, source=example, text=text, unknown="error")
    assert out["network_args"][0] == "algo=lokr" and out["timestep_sampling"] == "moss"


def test_option_names_cover_dest_and_long_options(parser):
    names = config_validation.option_names(parser)
    assert names["learning_rate_dora"].dest == "learning_rate_DoRA" and "help" not in names
