"""CPU tests for --moss_stream_fp32 (no GPU, no model files).

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe tests/test_moss_stream_fp32.py   (or with pytest if installed)
CPU autocast stands in for accelerate's bf16 autocast: the same ops are cast, on the same rule.
"""
import copy
import types

import torch
from torch.nn import functional as F

from library import attention, lycoris_fast
from library.moss.models import MossTransformer
from library.moss.train_utils import validate_moss_args

TINY = dict(dim=192, ffn_dim=512, num_heads=4, num_layers=4, text_dim=64, freq_dim=32, latent_channels=128)
BLOCK_LINEARS = 6


def _model():
    torch.manual_seed(0)
    model = MossTransformer(**TINY).eval()
    for parameter in model.parameters():
        if parameter.dim() >= 2:
            torch.nn.init.normal_(parameter, std=0.05)
    return model


def _inputs(dtype):
    torch.manual_seed(1)
    return (torch.randn(1, 128, 6, 6, dtype=dtype), torch.tensor([500.0]), torch.randn(1, 6, 64, dtype=dtype),
            torch.ones(1, 6, dtype=torch.bool))


def _run(model, dtype, autocast=True):
    seen = []
    hooks = [model.layers[0].register_forward_pre_hook(lambda _m, inputs: seen.append(inputs[0].dtype)),
             model.layers[-1].register_forward_hook(lambda _m, _i, out: seen.append(out.dtype))]
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        out = model(*_inputs(dtype))
    for hook in hooks:
        hook.remove()
    return out, seen


def _stream_model():
    model = _model().to(torch.bfloat16)
    model.enable_stream_fp32()
    return model


def test_only_the_block_linears_stay_in_the_training_dtype():
    model = _stream_model()
    kept = [name for name, parameter in model.named_parameters() if parameter.dtype == torch.bfloat16]
    assert len(kept) == TINY["num_layers"] * BLOCK_LINEARS
    assert all(name.startswith("layers.") and ".proj." in name for name in kept)
    assert model.dtype == torch.float32  # what sampling casts the text to


def test_the_stream_stays_fp32_under_bf16_autocast():
    with torch.no_grad():
        out, seen = _run(_stream_model(), torch.float32)
    assert seen == [torch.float32, torch.float32]
    assert out.dtype == torch.float32


def test_off_is_the_same_forward_as_before():
    model = _model().to(torch.bfloat16)
    untouched = copy.deepcopy(model)
    model.stream_fp32 = False
    with torch.no_grad():
        out, seen = _run(model, torch.bfloat16)
        same, _ = _run(untouched, torch.bfloat16)
    assert seen == [torch.bfloat16, torch.bfloat16]
    assert torch.equal(out, same)


def test_the_stream_is_closer_to_an_fp32_forward():
    with torch.no_grad():
        truth, _ = _run(_model().float(), torch.float32, autocast=False)
        plain, _ = _run(_model().to(torch.bfloat16), torch.bfloat16)
        stream, _ = _run(_stream_model(), torch.float32)
    error = lambda out: float((out.float() - truth).norm() / truth.norm())
    assert error(stream) < error(plain) / 2


def test_casting_the_parameters_alone_keeps_a_bf16_stream():
    """Why forward leaves autocast: the plain snippet changes the weights and not the stream."""
    model = _model().to(torch.bfloat16)
    for name, parameter in model.named_parameters():
        if not (name.startswith("layers.") and ".proj." in name):
            parameter.data = parameter.data.float()
    with torch.no_grad():
        _, seen = _run(model, torch.float32)
    assert seen == [torch.bfloat16, torch.bfloat16]


def test_gradients_reach_a_block_linear_with_checkpointing():
    model = _stream_model().train()
    model.enable_gradient_checkpointing()
    weight = model.layers[1].proj["q"].weight
    delta = torch.nn.Parameter(torch.zeros_like(weight.data, dtype=torch.float32))  # a LoRA-like fp32 trained weight
    module = model.layers[1].proj["q"]
    module.forward = lambda x: F.linear(x, module.weight + delta.to(module.weight.dtype))
    x, timesteps, text, mask = _inputs(torch.float32)
    x.requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(x, timesteps, text, mask)
    out.float().square().mean().backward()
    assert delta.grad is not None and float(delta.grad.abs().sum()) > 0


def _spy_on_attention(monkeyed):
    original = attention.attention

    def spy(tensors, attn_params):
        monkeyed.append({t.dtype for t in tensors})
        return torch.zeros_like(tensors[2]).flatten(2)  # only the dtypes matter here

    return original, spy


def test_flash_gets_16_bit_tensors_from_the_fp32_text_refiner():
    model = _model().to(torch.bfloat16)
    model.enable_stream_fp32()
    model.attn_mode = "flash"
    seen = []
    original, spy = _spy_on_attention(seen)
    attention.attention = spy
    try:
        with torch.no_grad():
            out, stream = _run(model, torch.float32)
    finally:
        attention.attention = original
    assert seen and all(dtypes <= {torch.bfloat16, torch.float16} for dtypes in seen), seen
    assert stream == [torch.float32, torch.float32] and out.dtype == torch.float32


def test_a_torch_attention_keeps_the_text_refiner_in_fp32():
    seen = []
    original, spy = _spy_on_attention(seen)
    attention.attention = spy
    try:
        with torch.no_grad():
            _run(_stream_model(), torch.float32)
    finally:
        attention.attention = original
    assert {torch.float32} in seen  # the two text blocks; the joint blocks stay bf16


def test_cached_merged_weights_keep_a_wider_base():
    class Lokr:
        _cached_dtype = torch.float32
        shape = (3, 4)
        scalar = 1.0
        multiplier_buf = 1.0
        is_ramtorch_org = False

        def get_weight(self, shape):
            return torch.zeros(shape, dtype=torch.float32)

        def apply_weight_decompose(self, weight, _multiplier):
            return weight

    model = torch.nn.ModuleDict({"text": torch.nn.Linear(4, 3, bias=False), "joint": torch.nn.Linear(4, 3, bias=False).to(torch.bfloat16)})
    targets = [(module, module.forward, Lokr()) for module in model.values()]
    original = lycoris_fast._cache_targets
    lycoris_fast._cache_targets = lambda _model: targets
    try:
        with lycoris_fast.cached_lokr_weights(model, torch.bfloat16, torch.device("cpu")):
            x = torch.ones(2, 4, dtype=torch.float32)
            assert model["text"](x).dtype == torch.float32  # outside autocast, as the fp32 text Linears run
            with torch.autocast("cpu", dtype=torch.bfloat16):
                assert model["joint"](x).dtype == torch.bfloat16
    finally:
        lycoris_fast._cache_targets = original


def _args(**changes):
    values = dict(pretrained_model_name_or_path=__file__, vae=__file__, timestep_sampling="moss", moss_max_token_length=512,
                  cache_text_encoder_outputs=True, cache_text_encoder_outputs_to_disk=False, network_train_unet_only=True,
                  mixed_precision="bf16", moss_stream_fp32=True, cache_latents=True)
    return types.SimpleNamespace(**{**values, **changes})


def test_unsupported_precision_flags_are_refused():
    for flag in ("full_bf16", "full_fp16", "fp8_base", "fp8_base_unet", "deepspeed"):
        try:
            validate_moss_args(_args(**{flag: True}), None, None)
        except ValueError as error:
            assert flag in str(error)
        else:
            raise AssertionError(f"--{flag} was accepted")


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
