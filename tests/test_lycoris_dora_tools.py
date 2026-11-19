"""CPU round trips for the standalone tools that read or rewrite adapter tensors."""
import argparse
import importlib
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    # The CLI scripts import sibling networks as top-level modules.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "networks"))
    path = str(tmp_path / "adapter.safetensors")
    scale = torch.tensor([[0.500123], [0.499876]])
    save_file({"layer.dora_scale": scale, "layer.lora_down.weight": torch.eye(2).half(),
               "layer.lora_up.weight": torch.eye(2).half(), "layer.alpha": torch.tensor(2).half()},
              path, metadata={"ss_dora_scale_dtype": "float32"})
    return path, scale


@pytest.mark.parametrize("tool", ["merge_lora", "merge_lora_old", "sdxl_merge_lora", "flux_merge_lora",
                                  "svd_merge_lora", "merger", "resize_lora"])
def test_merge_resize_reader_keeps_fp32_scale(adapter, tool):
    source, scale = adapter
    module = importlib.import_module(tool)
    if tool == "merger":
        state, _, _ = module.load_state_dict_and_collect_modules(source, torch.bfloat16)
    elif tool == "merge_lora_old":
        state = module.load_state_dict(source, torch.bfloat16)
    else:
        state, _ = module.load_state_dict(source, torch.bfloat16)
    assert state["layer.lora_down.weight"].dtype == torch.bfloat16
    assert state["layer.dora_scale"].dtype == torch.float32
    assert torch.equal(state["layer.dora_scale"], scale)


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
@pytest.mark.parametrize("tool", ["resize_lora", "merge_lora_old", "merge_models"])
def test_rewritten_adapter_keeps_fp32_scale(tmp_path, adapter, tool, precision):
    source, scale = adapter
    dest = str(tmp_path / "rewritten.safetensors")
    if tool == "resize_lora":
        import resize_lora

        args = resize_lora.setup_parser().parse_args(
            ["--model", source, "--save_to", dest, "--new_rank", "1", "--device", "cpu", "--save_precision", precision])
        resize_lora.resize(args)
    elif tool == "merge_lora_old":
        import merge_lora_old

        args = merge_lora_old.setup_parser().parse_args(
            ["--models", source, "--ratios", "1", "--save_to", dest, "--precision", precision, "--save_precision", precision])
        merge_lora_old.merge(args)
    else:
        from tools import merge_models

        merge_models.merge(argparse.Namespace(models=[source, source], ratios=[0.5, 0.5], output=dest,
                                              precision=precision, saving_precision=precision, device="cpu",
                                              unet_only=False, show_skipped=False))
    state = load_file(dest)
    assert state["layer.lora_down.weight"].dtype == {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    assert state["layer.dora_scale"].dtype == torch.float32
    assert torch.equal(state["layer.dora_scale"], scale)
