"""CPU tests: a training run without images fails with a non-zero exit code (review 2026-11-05).

train_network.py (every *_train_network.py trainer) and anima_train.py logged "No data found" and returned, so the
process ended with exit code 0 and a launcher (the GUI backend, a queue) took the run for a finished one. They now
raise SystemExit(1) after the message. The other trainers with the same block do the same.

Runs each trainer's train() in this process up to the dataset check on an empty folder; no model is loaded, no GPU.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp=<scratch dir>
     tests/test_empty_dataset_exit.py
"""
import ast
import logging
import os

import pytest

from library import strategy_base

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def no_strategies(monkeypatch):
    """train() registers the strategies once per process, resets the root logger and (torch >= 2.14) sets the CUDA
    allocator config; start from no strategies and restore all of it afterwards."""
    for cls in (strategy_base.TokenizeStrategy, strategy_base.TextEncodingStrategy,
                strategy_base.TextEncoderOutputsCachingStrategy, strategy_base.LatentsCachingStrategy):
        monkeypatch.setattr(cls, "_strategy", None)
    handlers, level = logging.root.handlers[:], logging.root.level
    alloc_env = {name: os.environ.get(name) for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")}
    yield
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    for handler in handlers:
        logging.root.addHandler(handler)
    logging.root.setLevel(level)
    for name, value in alloc_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture
def empty_train_dir(tmp_path):
    root = tmp_path / "train"
    (root / "1_empty").mkdir(parents=True)  # a DreamBooth folder without images
    (root / "1_empty" / "notes.txt").write_text("no image here", encoding="utf-8")
    return str(root)


def test_train_network_fails_without_images(tmp_path, empty_train_dir, no_strategies):
    import train_network

    class _Trainer(train_network.NetworkTrainer):
        def get_tokenize_strategy(self, args):  # the SD tokenizer would be fetched from the hub
            return strategy_base.TokenizeStrategy()

        def get_tokenizers(self, tokenize_strategy):
            return []

        def get_latents_caching_strategy(self, args):
            return None

    args = train_network.setup_parser().parse_args([
        "--train_data_dir", empty_train_dir, "--output_dir", str(tmp_path / "out"), "--pretrained_model_name_or_path", "unused",
        "--network_module", "networks.lora", "--resolution", "512",
    ])
    with pytest.raises(SystemExit) as exited:
        _Trainer().train(args)
    assert exited.value.code == 1


def test_anima_full_trainer_fails_without_images(tmp_path, empty_train_dir, no_strategies):
    import anima_train

    args = anima_train.setup_parser().parse_args([
        "--train_data_dir", empty_train_dir, "--output_dir", str(tmp_path / "out"), "--pretrained_model_name_or_path", "unused",
        "--qwen3", "unused", "--vae", "unused", "--resolution", "512",
    ])
    with pytest.raises(SystemExit) as exited:
        anima_train.train(args)
    assert exited.value.code == 1


def _no_data_branches(path):
    """The bodies of ``if len(train_dataset_group) == 0:`` in a script."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    return [
        node.body for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "len(train_dataset_group) == 0"
    ]


@pytest.mark.parametrize("script", [
    "train_network.py", "anima_train.py", "anima_train_control_net_lllite.py", "fine_tune.py", "flux_train.py",
    "flux_train_control_net.py", "lumina_train.py", "sd3_train.py", "sdxl_train.py", "sdxl_train_control_net.py",
    "sdxl_train_control_net_lllite.py", "sdxl_train_control_net_lllite_old.py", "train_control_net.py",
    "train_textual_inversion.py", "train_textual_inversion_XTI.py",
])
def test_every_trainer_exits_non_zero_without_images(script):
    branches = _no_data_branches(os.path.join(REPO, script))
    assert branches, f"{script}: no empty-dataset check found"
    for body in branches:
        last = body[-1]
        assert not isinstance(last, ast.Return), f"{script}: the empty-dataset branch returns (exit code 0)"
        assert isinstance(last, ast.Raise) and ast.unparse(last.exc) == "SystemExit(1)", ast.unparse(last)
