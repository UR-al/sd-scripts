"""
CPU tests for the Anima-3.8B v1.1 training integration (--anima38_connector; library/anima38/{training,strategy,
sampling,trainer}.py and the guarded hooks in anima_models.Anima.forward, anima_utils.load_anima_model and
anima_train_network.py).

Covered:
  - options and validation (every refused combination),
  - the option-off path: Anima.forward, get_noise_pred_and_target's DiT call and load_anima_model are unchanged,
  - bundle handling: connector keys never reach the DiT state dict (plain and fp8 loader paths), clear errors,
  - DiT conditioner: connector output replaces the llm_adapter output with each call's timesteps, no_grad,
    padding/mask equivalence with unpadded rows,
  - Qwen3.5 caching: Qwen3 and Qwen3.5 features of one caption variant are stored and loaded together (disk and
    in-memory), and the real dataset __getitem__ picks the same variant for both,
  - sampling: anima_train_utils.do_sample is reused unchanged; connector / native negative,
  - a CPU/meta dry run of the full 3.8B training graph through the trainer's own load_unet_lazily (header reads only)
    plus the user's LyCORIS LoKr preset on it.
Opt-in (ANIMA38_REAL_WEIGHTS=1, reads multi-GB files read-only, CPU): real Qwen3.5 encoding vs the Forge call, and
an end-to-end real-weight DiT forward/backward on CPU with the real connector.

Run (CPU only, keep temp files out of the repo):
    set CUDA_VISIBLE_DEVICES=-1
    pytest -p no:cacheprovider --basetemp=<scratch dir> tests/test_anima38_integration.py -v
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library import anima_models, anima_train_utils, anima_utils, train_util  # noqa: E402
from library.anima38 import bundle as a38_bundle  # noqa: E402
from library.anima38 import sampling as a38_sampling  # noqa: E402
from library.anima38 import strategy as a38_strategy  # noqa: E402
from library.anima38 import training as a38_training  # noqa: E402
from library.anima38.semantic_v2 import QualityAnchoredSemanticConnectorV2  # noqa: E402
from library.anima38.tokenizer import DEFAULT_TOKENIZER_DIR  # noqa: E402
from library.safetensors_utils import MemoryEfficientSafeOpen, TensorWeightAdapter  # noqa: E402

BUNDLE_PATH = r"C:\sd-webui-forge-classic\models\Stable-diffusion\Anima-3.8B-v1.1.safetensors"
ANIMA_29B_PATH = r"C:\sd-webui-forge-classic\models\Stable-diffusion\Anima-2.9B-preview-v1.safetensors"
QWEN35_PATH = r"C:\sd-webui-forge-classic\models\text_encoder\qwen35_4b.safetensors"
QWEN3_PATH = r"C:\sd-webui-forge-classic\models\text_encoder\qwen_3_06b_base.safetensors"
REAL_WEIGHTS = os.environ.get("ANIMA38_REAL_WEIGHTS") == "1"
REAL_SKIP_REASON = "real-weight CPU test is opt-in: set ANIMA38_REAL_WEIGHTS=1 (reads multi-GB model files read-only)"

HAS_BUNDLE = os.path.isfile(BUNDLE_PATH)
HAS_QWEN35_TOKENIZER = os.path.isfile(os.path.join(DEFAULT_TOKENIZER_DIR, "tokenizer.json"))
SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "I64": torch.int64,
}


@pytest.fixture(autouse=True)
def _cpu_only():
    assert not torch.cuda.is_available(), "hide the GPU: CUDA_VISIBLE_DEVICES=-1"


# ======================================================================================================
# helpers
# ======================================================================================================


def _randomize_(module: nn.Module, seed: int, scale: float = 1.0) -> nn.Module:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.ndim >= 2:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * scale / math.sqrt(parameter.shape[-1]))
            else:
                parameter.copy_(0.75 + 0.5 * torch.rand(parameter.shape, generator=generator))
    return module


SEM_DIM = 40
TINY_CONNECTOR = dict(semantic_source_dim=SEM_DIM, layer_indices=(7, 15, 23, 31), num_queries=8, resampler_blocks=2, resampler_dim=32, resampler_heads=4, mlp_hidden_dim=48)


def _tiny_anima(seed: int = 0) -> anima_models.Anima:
    """Anima DiT with 2 small blocks and a small llm_adapter (Qwen3 source 32, context 64), random weights, fp32."""
    torch.manual_seed(seed)
    model = anima_models.Anima(
        max_img_h=64,
        max_img_w=64,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=True,
        model_channels=64,
        num_blocks=2,
        num_heads=4,
        crossattn_emb_channels=64,
        pos_emb_cls="rope3d",
        pos_emb_learnable=True,
        use_adaln_lora=True,
        adaln_lora_dim=16,
        rope_h_extrapolation_ratio=4.0,
        rope_w_extrapolation_ratio=4.0,
        rope_enable_fps_modulation=False,
        use_llm_adapter=False,
        attn_mode="torch",
    )
    model.llm_adapter = anima_models.LLMAdapter(source_dim=32, target_dim=64, model_dim=64, num_layers=2, num_heads=4, self_attn=True)
    model.use_llm_adapter = True
    _randomize_(model, seed)
    return model.eval().requires_grad_(False)


def _tiny_connector(model: anima_models.Anima, seed: int = 1) -> QualityAnchoredSemanticConnectorV2:
    connector = QualityAnchoredSemanticConnectorV2(native_adapter=model.llm_adapter, **TINY_CONNECTOR)
    return _randomize_(connector, seed).eval().requires_grad_(False)


def _tiny_inputs(batch: int = 2, seed: int = 3, lengths=((5, 7, 6), (3, 4, 2))):
    """DiT inputs with right padding: per row (qwen3 len, t5 len, qwen35 len)."""
    generator = torch.Generator().manual_seed(seed)
    ls, lt, lq = 9, 10, 8
    x = torch.randn(batch, 16, 1, 8, 8, generator=generator)
    timesteps = torch.tensor([0.3, 0.8][:batch])
    source = torch.randn(batch, ls, 32, generator=generator)
    t5_ids = torch.randint(2, 32000, (batch, lt), generator=generator)
    semantic = torch.randn(batch, lq, 4 * SEM_DIM, generator=generator)
    source_mask = torch.zeros(batch, ls, dtype=torch.long)
    t5_mask = torch.zeros(batch, lt, dtype=torch.long)
    semantic_mask = torch.zeros(batch, lq, dtype=torch.long)
    for row, (n_source, n_t5, n_semantic) in enumerate(lengths[:batch]):
        source_mask[row, :n_source] = 1
        t5_mask[row, :n_t5] = 1
        semantic_mask[row, :n_semantic] = 1
    source = source * source_mask[:, :, None]  # the cache zeroes padded Qwen3 rows
    t5_ids = t5_ids * t5_mask
    semantic = semantic * semantic_mask[:, :, None]
    padding_mask = torch.zeros(batch, 1, 8, 8)
    return dict(
        x=x,
        timesteps=timesteps,
        source=source,
        source_mask=source_mask,
        t5_ids=t5_ids,
        t5_mask=t5_mask,
        semantic=semantic,
        semantic_mask=semantic_mask,
        padding_mask=padding_mask,
    )


def _header(path: str):
    with MemoryEfficientSafeOpen(path) as f:  # opens 'rb', reads the header only
        return dict(f.header), dict(f.metadata())


def _meta_loader(record: list):
    """Stand-in for lora_utils.load_safetensors_with_lora_and_fp8 that builds meta tensors from the real header
    through the same TensorWeightAdapter (so the split/rename hooks run exactly as in the real loader)."""

    def load(model_files, lora_weights_list=None, lora_multipliers=None, fp8_optimization=False, calc_device=None,
             move_to_device=False, dit_weight_dtype=None, target_keys=None, exclude_keys=None,
             disable_numpy_memmap=False, weight_transform_hooks=None):
        with MemoryEfficientSafeOpen(model_files) as original_f:
            f = TensorWeightAdapter(weight_transform_hooks, original_f) if weight_transform_hooks is not None else original_f
            state = {}
            for key in f.keys():
                original = f.new_key_to_original_key_map.get(key, key) if weight_transform_hooks is not None else key
                record.append(original)
                info = original_f.header[original]
                state[key] = torch.empty(info["shape"], dtype=dit_weight_dtype or SAFETENSORS_DTYPES[info["dtype"]], device="meta")
        return state

    return load


def _meta_connector_state(path: str):
    header, metadata = _header(path)
    prefixes = a38_bundle.connector_prefixes(metadata)
    state = {
        key: torch.empty(info["shape"], dtype=SAFETENSORS_DTYPES[info["dtype"]], device="meta")
        for key, info in header.items()
        if key != "__metadata__" and key.startswith(prefixes)
    }
    _, connector_state, info = a38_bundle.split_anima38_bundle(state, metadata)
    return connector_state, info


def _args(**overrides):
    values = dict(
        anima38_connector=True,
        qwen35_path=None,
        qwen35_tokenizer_dir=None,
        anima38_sample_negative="connector",
        anima38_qwen35_batch_size=1,
        cache_text_encoder_outputs=True,
        weighted_captions=False,
        ileco=False,
        addift=False,
        adaptive_timestep_sampling=False,
        network_train_text_encoder_only=False,
        t5_max_token_length=512,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _group(caption_aug_variant_count=0, aug_refresh_epochs=0, **subset):
    subset_values = dict(caption_dropout_every_n_epochs=0, caption_dropout_rate=0.0)
    subset_values.update(subset)
    dataset = SimpleNamespace(
        subsets=[SimpleNamespace(**subset_values)],
        caption_aug_variant_count=caption_aug_variant_count,
        aug_refresh_epochs=aug_refresh_epochs,
    )
    return SimpleNamespace(datasets=[dataset])


# ======================================================================================================
# options and validation
# ======================================================================================================


def test_parser_options_and_defaults():
    anima_train_network = pytest.importorskip("anima_train_network")
    parser = anima_train_network.setup_parser()
    defaults = parser.parse_args([])
    assert defaults.anima38_connector is False
    assert defaults.qwen35_path is None and defaults.qwen35_tokenizer_dir is None
    assert defaults.anima38_sample_negative == "connector"
    assert defaults.anima38_qwen35_batch_size == 1
    args = parser.parse_args(
        ["--anima38_connector", "--qwen35_path", "q.safetensors", "--qwen35_tokenizer_dir", "tok", "--anima38_sample_negative", "native",
         "--anima38_qwen35_batch_size", "4"]
    )
    assert (args.anima38_connector, args.qwen35_path, args.qwen35_tokenizer_dir, args.anima38_sample_negative, args.anima38_qwen35_batch_size) == (
        True, "q.safetensors", "tok", "native", 4)
    with pytest.raises(SystemExit):
        parser.parse_args(["--anima38_sample_negative", "both"])


def test_validation_is_inert_when_off():
    a38_training.validate_anima38_args(SimpleNamespace(anima38_connector=False, cache_text_encoder_outputs=False, ileco=True), [_group(caption_dropout_every_n_epochs=3)])
    a38_training.validate_anima38_args(SimpleNamespace(), [])


def test_validation_accepts_the_supported_setup(tmp_path):
    qwen35 = tmp_path / "qwen35.safetensors"
    qwen35.write_bytes(b"x")
    # shuffle / tag dropout / caption dropout baked into cached variants
    a38_training.validate_anima38_args(_args(qwen35_path=str(qwen35)), [_group(8, caption_dropout_rate=0.1), None])


@pytest.mark.parametrize(
    "overrides, group, message",
    [
        (dict(qwen35_path=None), None, "--qwen35_path"),
        (dict(qwen35_path="__missing__.safetensors"), None, "not found"),
        (dict(cache_text_encoder_outputs=False), None, "--cache_text_encoder_outputs is required"),
        (dict(weighted_captions=True), None, "--weighted_captions"),
        (dict(ileco=True), None, "--ileco"),
        (dict(addift=True), None, "--addift"),
        (dict(adaptive_timestep_sampling=True), None, "--adaptive_timestep_sampling"),
        (dict(network_train_text_encoder_only=True), None, "--network_train_text_encoder_only"),
        (dict(anima38_qwen35_batch_size=0), None, "batch_size"),
        (dict(anima38_sample_negative="both"), None, "--anima38_sample_negative"),
        (dict(), _group(8, caption_dropout_every_n_epochs=2), "caption_dropout_every_n_epochs"),
        (dict(), _group(0, caption_dropout_rate=0.1), "caption_dropout_rate > 0 needs --cache_caption_variants"),
        (dict(), _group(8, aug_refresh_epochs=1), "--cache_aug_refresh_epochs"),
    ],
)
def test_validation_refuses(tmp_path, overrides, group, message):
    qwen35 = tmp_path / "qwen35.safetensors"
    qwen35.write_bytes(b"x")
    values = dict(qwen35_path=str(qwen35))
    values.update(overrides)
    with pytest.raises(ValueError, match="--anima38_connector") as info:
        a38_training.validate_anima38_args(_args(**values), [group] if group is not None else [])
    assert message in str(info.value)


@pytest.mark.skipif(not HAS_BUNDLE, reason="Anima-3.8B bundle not present")
def test_bundle_and_qwen35_header_checks():
    info = a38_training.require_anima38_bundle(BUNDLE_PATH)
    assert len(info["connector_keys"]) == 190 and info["is_bundle"]
    if os.path.isfile(ANIMA_29B_PATH) and os.path.getsize(ANIMA_29B_PATH) > 0:
        with pytest.raises(ValueError, match="needs an Anima-3.8B v1.1 bundle"):
            a38_training.require_anima38_bundle(ANIMA_29B_PATH)
    if os.path.isfile(QWEN35_PATH):
        a38_training.check_qwen35_checkpoint(QWEN35_PATH)
        with pytest.raises(ValueError, match="not a Qwen3.5-4B checkpoint"):
            a38_training.check_qwen35_checkpoint(BUNDLE_PATH)


def test_trainer_factory_mro():
    anima_train_network = pytest.importorskip("anima_train_network")
    from library.anima38.trainer import Anima38TrainerMixin, make_anima38_trainer

    trainer = make_anima38_trainer(anima_train_network.AnimaNetworkTrainer)
    assert isinstance(trainer, anima_train_network.AnimaNetworkTrainer)
    mro = type(trainer).__mro__
    assert mro[1] is Anima38TrainerMixin and mro[2] is anima_train_network.AnimaNetworkTrainer
    plain = anima_train_network.AnimaNetworkTrainer()
    assert plain.get_extra_dit_kwargs(None, None, [None] * 5, torch.float32) == {}


# ======================================================================================================
# option off: unchanged paths
# ======================================================================================================


def test_forward_without_conditioner_is_the_original_composition():
    model = _tiny_anima()
    inputs = _tiny_inputs()
    with torch.no_grad():
        got = model(
            inputs["x"],
            inputs["timesteps"],
            inputs["source"],
            padding_mask=inputs["padding_mask"],
            target_input_ids=inputs["t5_ids"],
            target_attention_mask=inputs["t5_mask"],
            source_attention_mask=inputs["source_mask"],
        )
        context = model._preprocess_text_embeds(inputs["source"], inputs["t5_ids"], inputs["t5_mask"], inputs["source_mask"])
        want = model.forward_mini_train_dit(inputs["x"], inputs["timesteps"], context, padding_mask=inputs["padding_mask"])
    assert torch.equal(got, want)
    assert "anima38_conditioner" not in model.__dict__
    with pytest.raises(TypeError):  # Qwen3.5 features without an attached connector are an error, not ignored
        model(inputs["x"], inputs["timesteps"], inputs["source"], padding_mask=inputs["padding_mask"], semantic_hidden_states=inputs["semantic"])


def _call_anima_get_noise_pred_and_target(trainer, monkeypatch, text_encoder_conds):
    anima_train_network = pytest.importorskip("anima_train_network")
    from library import flux_train_utils

    batch, channels, height, width = 2, 16, 8, 8
    latents = torch.randn(batch, channels, height, width)
    noise = torch.randn_like(latents)
    noisy = latents + noise
    timesteps = torch.tensor([300.0, 800.0])
    sigmas = (timesteps / 1000.0).view(-1, 1, 1, 1)
    monkeypatch.setattr(flux_train_utils, "get_noisy_model_input_and_timesteps", lambda *a, **k: (noisy, timesteps, sigmas))
    monkeypatch.setattr(anima_train_network, "maybe_apply_antithetic_noise_pairing", lambda args, n, **k: n)
    monkeypatch.setattr(torch, "randn_like", lambda t, *a, **k: noise.clone())

    calls = []

    def fake_anima(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.zeros(batch, channels, 1, height, width)

    trainer.hf_scale = 0.0
    trainer.anchor_scale = 0.0
    args = SimpleNamespace(ileco=False, addift=False, gradient_checkpointing=False, weighting_scheme="uniform", flow_use_ot=False)
    accelerator = SimpleNamespace(autocast=lambda: contextlib.nullcontext(), device=torch.device("cpu"))
    scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
    out = trainer.get_noise_pred_and_target(
        args, accelerator, scheduler, latents, {}, text_encoder_conds, None, fake_anima, None, torch.float32, True, is_train=True
    )
    assert len(out) == 5 and len(calls) == 1
    return calls[0]


def _cached_conds(batch=2):
    return [
        torch.randn(batch, 6, 32),
        torch.ones(batch, 6),
        torch.randint(0, 100, (batch, 7)).float(),
        torch.ones(batch, 7),
        torch.randn(batch, 5, 4 * SEM_DIM).to(torch.bfloat16),
        torch.ones(batch, 5),
        torch.zeros(batch),
    ]


def test_get_noise_pred_and_target_dit_call_unchanged_when_off(monkeypatch):
    anima_train_network = pytest.importorskip("anima_train_network")
    trainer = anima_train_network.AnimaNetworkTrainer()
    args, kwargs = _call_anima_get_noise_pred_and_target(trainer, monkeypatch, _cached_conds()[:4] + [torch.zeros(2)])
    assert len(args) == 3
    assert sorted(kwargs) == ["padding_mask", "source_attention_mask", "target_attention_mask", "target_input_ids"]


def test_get_noise_pred_and_target_passes_qwen35_features_when_on(monkeypatch):
    anima_train_network = pytest.importorskip("anima_train_network")
    from library.anima38.trainer import make_anima38_trainer

    trainer = make_anima38_trainer(anima_train_network.AnimaNetworkTrainer)
    conds = _cached_conds()
    args, kwargs = _call_anima_get_noise_pred_and_target(trainer, monkeypatch, conds)
    assert sorted(kwargs) == [
        "padding_mask", "semantic_attention_mask", "semantic_hidden_states", "source_attention_mask", "target_attention_mask", "target_input_ids"]
    assert torch.equal(kwargs["semantic_hidden_states"], conds[4].float())
    assert torch.equal(kwargs["semantic_attention_mask"], conds[5])
    with pytest.raises(RuntimeError, match="no Qwen3.5 features"):
        _call_anima_get_noise_pred_and_target(trainer, monkeypatch, conds[:4] + [torch.zeros(2)])


def _synthetic_checkpoint(path, with_connector: bool):
    from safetensors.torch import save_file

    tensors = {
        "net.blocks.0.self_attn.q_proj.weight": torch.randn(8, 8),
        "net.blocks.0.norm.weight": torch.randn(8),
        "net.llm_adapter.embed.weight": torch.randn(4, 8),
    }
    if with_connector:
        tensors["net.anima_v2_connector.semantic_resampler.blocks.0.mlp_in.weight"] = torch.randn(8, 8)
        tensors["net.anima_v2_connector.quality_anchor.layer_mix_logits"] = torch.randn(6, 4)
    save_file(tensors, str(path))
    return path


@pytest.mark.parametrize("fp8", [False, True])
def test_connector_tensors_never_reach_the_dit_loader_output(tmp_path, fp8):
    from library.lora_utils import load_safetensors_with_lora_and_fp8
    from library.safetensors_utils import WeightTransformHooks

    path = _synthetic_checkpoint(tmp_path / "bundle.safetensors", with_connector=True)
    hooks = WeightTransformHooks(split_hook=anima_utils._skip_anima38_connector_hook, rename_hook=anima_utils._rename_hook)
    state = load_safetensors_with_lora_and_fp8(
        model_files=str(path),
        lora_weights_list=None,
        lora_multipliers=None,
        fp8_optimization=fp8,
        calc_device=torch.device("cpu"),
        move_to_device=False,
        dit_weight_dtype=None if fp8 else torch.bfloat16,
        target_keys=anima_utils.FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=anima_utils.FP8_OPTIMIZATION_EXCLUDE_KEYS,
        weight_transform_hooks=hooks,
    )
    assert not any("anima_v2_connector" in key for key in state)
    assert "blocks.0.self_attn.q_proj.weight" in state and "llm_adapter.embed.weight" in state
    # the hook only drops connector keys; every other key keeps the plain rename
    plain = load_safetensors_with_lora_and_fp8(
        model_files=str(_synthetic_checkpoint(tmp_path / "plain.safetensors", with_connector=False)),
        lora_weights_list=None, lora_multipliers=None, fp8_optimization=fp8, calc_device=torch.device("cpu"),
        dit_weight_dtype=None if fp8 else torch.bfloat16, target_keys=anima_utils.FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=anima_utils.FP8_OPTIMIZATION_EXCLUDE_KEYS,
        weight_transform_hooks=WeightTransformHooks(rename_hook=anima_utils._rename_hook),
    )
    assert sorted(plain) == sorted(state)


@pytest.mark.skipif(not HAS_BUNDLE, reason="Anima-3.8B bundle not present")
def test_load_anima_model_meta_38b_bundle(monkeypatch):
    """The real loader on the real header (meta tensors): connector skipped, strict key check passes, 52 blocks."""
    requested = []
    monkeypatch.setattr(anima_utils, "load_safetensors_with_lora_and_fp8", _meta_loader(requested))
    model = anima_utils.load_anima_model("cpu", BUNDLE_PATH, "torch", False, "cpu", torch.bfloat16, False, anima38_connector=True)
    assert len(model.blocks) == 52
    assert len(requested) == 1358 - 190
    assert not any(key.startswith("net.anima_v2_connector.") for key in requested)
    assert all(p.device.type == "meta" for p in model.parameters())
    assert "anima_v2_connector" not in " ".join(model.state_dict().keys())

    requested.clear()
    with pytest.raises(RuntimeError, match="--anima38_connector"):  # without the option: explained, before loading anything
        anima_utils.load_anima_model("cpu", BUNDLE_PATH, "torch", False, "cpu", torch.bfloat16, False)
    assert requested == []


@pytest.mark.skipif(not (os.path.isfile(ANIMA_29B_PATH) and os.path.getsize(ANIMA_29B_PATH) > 0), reason="Anima-2.9B not present")
def test_load_anima_model_meta_29b_unchanged(monkeypatch):
    """2.9B (no connector): option off loads every key exactly as before; option on is refused."""
    requested = []
    monkeypatch.setattr(anima_utils, "load_safetensors_with_lora_and_fp8", _meta_loader(requested))
    model = anima_utils.load_anima_model("cpu", ANIMA_29B_PATH, "torch", False, "cpu", torch.bfloat16, False)
    header, _ = _header(ANIMA_29B_PATH)
    assert len(model.blocks) == 40
    assert sorted(requested) == sorted(key for key in header if key != "__metadata__")
    with pytest.raises(RuntimeError, match="not an Anima-3.8B v1.1 bundle"):
        anima_utils.load_anima_model("cpu", ANIMA_29B_PATH, "torch", False, "cpu", torch.bfloat16, False, anima38_connector=True)


# ======================================================================================================
# DiT conditioner
# ======================================================================================================


def _conditioned_pair(seed: int = 0):
    model = _tiny_anima(seed)
    connector = _tiny_connector(model, seed + 1)
    a38_training.attach_anima38_conditioner(model, connector)
    return model, connector


def test_conditioner_is_not_a_submodule():
    model, connector = _conditioned_pair()
    assert a38_training.get_anima38_conditioner(model).connector is connector
    names = [name for name, _ in model.named_modules()]
    assert not any("connector" in name or "quality_anchor" in name for name in names)
    assert set(model.state_dict()) == set(_tiny_anima().state_dict())
    with pytest.raises(ValueError, match="llm_adapter"):
        a38_training.attach_anima38_conditioner(_tiny_anima(5), connector)  # connector must wrap this DiT's llm_adapter


def test_forward_with_conditioner_uses_the_connector_with_each_calls_timesteps():
    model, connector = _conditioned_pair()
    inputs = _tiny_inputs()
    kwargs = dict(
        padding_mask=inputs["padding_mask"],
        target_input_ids=inputs["t5_ids"],
        target_attention_mask=inputs["t5_mask"],
        source_attention_mask=inputs["source_mask"],
    )
    with torch.no_grad():
        got = model(inputs["x"], inputs["timesteps"], inputs["source"], semantic_hidden_states=inputs["semantic"],
                    semantic_attention_mask=inputs["semantic_mask"], **kwargs)
        context = connector.forward_context(
            inputs["source"], inputs["t5_ids"], a38_training.unpack_semantic_layers(inputs["semantic"]), inputs["timesteps"],
            target_attention_mask=inputs["t5_mask"], native_source_mask=inputs["source_mask"], semantic_source_mask=inputs["semantic_mask"],
        )
        want = model.forward_mini_train_dit(inputs["x"], inputs["timesteps"], context, padding_mask=inputs["padding_mask"])
        native = model.forward_mini_train_dit(
            inputs["x"], inputs["timesteps"],
            model._preprocess_text_embeds(inputs["source"], inputs["t5_ids"], inputs["t5_mask"], inputs["source_mask"]),
            padding_mask=inputs["padding_mask"],
        )
        other_t = model(inputs["x"], inputs["timesteps"].flip(0), inputs["source"], semantic_hidden_states=inputs["semantic"],
                        semantic_attention_mask=inputs["semantic_mask"], **kwargs)
    assert context.shape == (2, 512, 64)
    assert torch.equal(got, want)
    assert (got - native).abs().max() > 1e-3  # the llm_adapter output is replaced
    assert (got - other_t).abs().max() > 1e-4  # the connector depends on the timestep
    # padded T5 rows of the context are zero, as in the native path
    assert torch.all(context[0, 7:] == 0) and torch.all(context[1, 4:] == 0)

    with pytest.raises(RuntimeError, match="no Qwen3.5 features"):
        model(inputs["x"], inputs["timesteps"], inputs["source"], **kwargs)
    # a precomputed context (no T5 ids) passes through: native negative prompts in sampling
    precomputed = torch.randn(2, 512, 64)
    with torch.no_grad():
        passthrough = model(inputs["x"], inputs["timesteps"], precomputed, padding_mask=inputs["padding_mask"])
        direct = model.forward_mini_train_dit(inputs["x"], inputs["timesteps"], precomputed, padding_mask=inputs["padding_mask"])
    assert torch.equal(passthrough, direct)


def test_conditioner_padding_matches_unpadded_rows():
    """Masked, right-padded batch == each row alone without padding (Forge runs unpadded rows)."""
    model, connector = _conditioned_pair(7)
    inputs = _tiny_inputs(lengths=((5, 7, 6), (3, 4, 2)))
    conditioner = a38_training.get_anima38_conditioner(model)
    with torch.no_grad():
        batched = conditioner(inputs["source"], inputs["timesteps"], inputs["t5_ids"], inputs["t5_mask"], inputs["source_mask"],
                              inputs["semantic"], inputs["semantic_mask"])
        for row, (n_source, n_t5, n_semantic) in enumerate(((5, 7, 6), (3, 4, 2))):
            alone = connector(
                inputs["source"][row : row + 1, :n_source],
                inputs["t5_ids"][row : row + 1, :n_t5],
                [s[:, :n_semantic] for s in a38_training.unpack_semantic_layers(inputs["semantic"][row : row + 1])],
                semantic_source_mask=torch.ones(1, n_semantic, dtype=torch.long),
                timesteps=inputs["timesteps"][row : row + 1],
            )
            scale = max(1.0, alone.abs().max().item())
            torch.testing.assert_close(batched[row, :n_t5], alone[0], rtol=1e-4, atol=1e-5 * scale)
            assert torch.all(batched[row, n_t5:] == 0)


def test_conditioner_grad_policy_and_timestep_broadcast():
    model, connector = _conditioned_pair()
    conditioner = a38_training.get_anima38_conditioner(model)
    inputs = _tiny_inputs()
    weight = model.llm_adapter.blocks[0].cross_attn.q_proj.weight
    weight.requires_grad_(True)
    call = lambda t: conditioner(inputs["source"], t, inputs["t5_ids"], inputs["t5_mask"], inputs["source_mask"],  # noqa: E731
                                 inputs["semantic"].requires_grad_(True), inputs["semantic_mask"])
    with torch.enable_grad():
        out = call(inputs["timesteps"])
        assert not out.requires_grad  # frozen connector runs under no_grad by default
        conditioner.grad_through_connector = True  # a network trains llm_adapter modules
        out = call(inputs["timesteps"])
        assert out.requires_grad
        out.float().pow(2).mean().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in connector.parameters())
    assert inputs["semantic"].grad is None  # cached Qwen3.5 features are constants
    weight.requires_grad_(False)
    conditioner.grad_through_connector = False
    with torch.no_grad():
        one = call(torch.tensor([0.4]))
        both = call(torch.tensor([0.4, 0.4]))
    assert torch.equal(one, both)
    with pytest.raises(ValueError, match="timesteps"):
        call(torch.tensor([0.1, 0.2, 0.3]))


def test_network_trains_llm_adapter_detection():
    assert not a38_training.network_trains_llm_adapter(None)
    assert a38_training.network_trains_llm_adapter(SimpleNamespace(train_llm_adapter=True))
    net = nn.Module()
    net.lycoris_blocks_0_self_attn_q_proj = nn.Linear(2, 2)
    assert not a38_training.network_trains_llm_adapter(net)
    net.lycoris_llm_adapter_blocks_0_cross_attn_q_proj = nn.Linear(2, 2)
    assert a38_training.network_trains_llm_adapter(net)
    net.lycoris_llm_adapter_blocks_0_cross_attn_q_proj.requires_grad_(False)
    assert not a38_training.network_trains_llm_adapter(net)


def test_semantic_packing_roundtrip():
    layers = [torch.randn(2, 5, SEM_DIM) for _ in range(4)]
    packed = a38_training.pack_semantic_layers(layers)
    assert packed.shape == (2, 5, 4 * SEM_DIM)
    for a, b in zip(layers, a38_training.unpack_semantic_layers(packed)):
        assert torch.equal(a, b)
    with pytest.raises(ValueError):
        a38_training.unpack_semantic_layers(torch.randn(2, 5, 4 * SEM_DIM + 1))


# ======================================================================================================
# Qwen3.5 encoding and per-variant caching
# ======================================================================================================


class _FakeQwen3(nn.Module):
    """Deterministic stand-in for Qwen3-0.6B: last_hidden_state = causal running mean of an id embedding (bf16)."""

    def __init__(self, vocab=160000, dim=32):
        super().__init__()
        self.table = nn.Embedding(vocab, dim)
        _randomize_(self.table, 11)

    @property
    def device(self):
        return self.table.weight.device

    def forward(self, input_ids, attention_mask):
        emb = self.table(input_ids) * attention_mask[:, :, None]
        hidden = emb.cumsum(1) / torch.arange(1, emb.shape[1] + 1)[None, :, None]
        return SimpleNamespace(last_hidden_state=hidden.to(torch.bfloat16))


class _FakeQwen35:
    """Stand-in for Qwen35HybridModel.forward_semantic_layers: 4 causal layers of an id embedding (bf16). Records
    batch shapes and whether autocast was on."""

    def __init__(self, vocab=248320, dim=SEM_DIM):
        generator = torch.Generator().manual_seed(12)
        self.embed_tokens = SimpleNamespace(weight=torch.randn(vocab, dim, generator=generator))
        self.calls = []

    def forward_semantic_layers(self, input_ids, attention_mask=None, layer_indices=(7, 15, 23, 31)):
        self.calls.append((tuple(input_ids.shape), torch.is_autocast_enabled("cpu"), attention_mask.tolist()))
        emb = self.embed_tokens.weight[input_ids]
        running = emb.cumsum(1) / torch.arange(1, emb.shape[1] + 1)[None, :, None]
        return [(running * (i + 1) + i).to(torch.bfloat16) for i in range(len(layer_indices))]


@pytest.fixture(scope="module")
def tokenize_strategy():
    if not HAS_QWEN35_TOKENIZER:
        pytest.skip("Qwen3.5 tokenizer files not present")
    return a38_strategy.Anima38TokenizeStrategy(
        qwen3_tokenizer=anima_utils.load_qwen3_tokenizer("configs_file.safetensors"),
        t5_tokenizer=anima_utils.load_t5_tokenizer(None),
        qwen3_max_length=48,
        t5_max_length=48,
    )


def test_tokenize_adds_qwen35_ids(tokenize_strategy):
    tokens = tokenize_strategy.tokenize(["1boy, solo, looking at viewer", ""])
    assert len(tokens) == 6
    ids, mask = tokens[4], tokens[5]
    expected = tokenize_strategy.qwen35_tokenizer(["1boy, solo, looking at viewer"])["input_ids"][0]
    assert ids[0, : len(expected)].tolist() == expected and int(mask[0].sum()) == len(expected)
    assert ids[1, 0].item() == 151643 and mask[1].tolist() == [1] + [0] * (ids.shape[1] - 1)  # '' -> [pad], like Forge
    assert torch.all(ids[1, 1:] == 151643)
    assert tokens[0].shape == (2, 48) and tokens[2].shape == (2, 48)  # Qwen3 / T5 unchanged


def test_encode_qwen35_chunking_matches_unpadded_calls(tokenize_strategy):
    captions = ["a, b, c, d, e, f", "x", "1boy, muscular male, beach, day"]
    tokens = tokenize_strategy.tokenize(captions)
    fake = _FakeQwen35()
    single, mask1 = a38_strategy.Anima38TextEncodingStrategy(qwen35_batch_size=1).encode_qwen35(fake, tokens[4], tokens[5])
    single_calls = list(fake.calls)
    fake.calls.clear()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        batched, mask3 = a38_strategy.Anima38TextEncodingStrategy(qwen35_batch_size=3).encode_qwen35(fake, tokens[4], tokens[5])
    assert torch.equal(mask1, mask3) and torch.equal(mask1, tokens[5])
    lengths = tokens[5].sum(1).tolist()
    # batch size 1: every caption alone, unpadded, mask of ones (Forge's _semantic_layers call)
    assert [c[0] for c in single_calls] == [(1, n) for n in lengths]
    assert all(all(v == 1 for v in c[2][0]) for c in single_calls)
    assert not any(c[1] for c in single_calls + fake.calls)  # autocast is disabled around Qwen3.5
    for row, n in enumerate(lengths):
        assert torch.equal(single[row, :n], batched[row, :n])
        assert torch.all(single[row, n:] == 0) and torch.all(batched[row, n:] == 0)
    assert single.shape[-1] == 4 * SEM_DIM


def _info(tmp_path, name, caption, variants=None, dropout=0.0):
    return SimpleNamespace(
        caption=caption,
        caption_variants=variants,
        caption_dropout_rate=dropout,
        caption_aug_hash="hash-a",
        text_encoder_outputs_npz=str(tmp_path / f"{name}_anima38_te.npz"),
        text_encoder_outputs=None,
        text_encoder_outputs_variants=None,
    )


def _expected(tokenize_strategy, encoding, models, caption):
    """What the cache must hold for one caption: both encoders run on this exact string."""
    pe, am, t5, t5m, q35, q35m = encoding.encode_tokens(tokenize_strategy, models, tokenize_strategy.tokenize([caption]))
    n = int(q35m[0].sum())
    return pe[0], t5[0], q35[0, :n]


def _assert_row(loaded, expected):
    pe, t5, q35 = expected
    assert torch.equal(torch.as_tensor(loaded[0]).to(pe.dtype), pe)
    assert torch.equal(torch.as_tensor(np.asarray(loaded[2])).long(), t5.long())
    assert torch.equal(torch.as_tensor(loaded[4]), q35)
    assert np.asarray(loaded[5]).tolist() == [1] * q35.shape[0]


@pytest.mark.parametrize("to_disk", [True, False])
def test_cache_keeps_qwen3_and_qwen35_of_each_variant_together(tmp_path, tokenize_strategy, to_disk):
    models = [_FakeQwen3(), _FakeQwen35()]
    encoding = a38_strategy.Anima38TextEncodingStrategy()
    caching = a38_strategy.Anima38TextEncoderOutputsCachingStrategy(to_disk, 4, False, qwen35_signature="sig-1")
    variants = ["@a, 1boy, solo, beach", "@a, 1boy, beach, solo", "@a, 1boy, solo", ""]
    infos = [
        _info(tmp_path, "img1", "@a, 1boy, solo, beach", variants),
        _info(tmp_path, "img2", "2boys, night, rain, very long caption with more tokens"),
    ]
    caching.cache_batch_outputs(tokenize_strategy, models, encoding, infos)

    for k, caption in enumerate(variants):
        if to_disk:
            loaded = caching.load_outputs_npz(infos[0].text_encoder_outputs_npz, variant=k)
        else:
            loaded = infos[0].text_encoder_outputs if k == 0 else infos[0].text_encoder_outputs_variants[k - 1]
        _assert_row(loaded, _expected(tokenize_strategy, encoding, models, caption))
        assert float(np.asarray(loaded[-1])) == 0.0  # dropout lives in the variants
    loaded = caching.load_outputs_npz(infos[1].text_encoder_outputs_npz) if to_disk else infos[1].text_encoder_outputs
    _assert_row(loaded, _expected(tokenize_strategy, encoding, models, infos[1].caption))

    if to_disk:
        assert infos[0].text_encoder_outputs is None
        assert caching.is_disk_cached_outputs_expected(infos[0].text_encoder_outputs_npz, 4, "hash-a")
        assert not caching.is_disk_cached_outputs_expected(infos[0].text_encoder_outputs_npz, 5, "hash-a")  # needs more variants
        assert not caching.is_disk_cached_outputs_expected(infos[0].text_encoder_outputs_npz, 4, "hash-b")  # other augmentation
        other = a38_strategy.Anima38TextEncoderOutputsCachingStrategy(True, 4, False, qwen35_signature="sig-2")
        assert not other.is_disk_cached_outputs_expected(infos[0].text_encoder_outputs_npz, 4, "hash-a")  # other Qwen3.5
        assert caching.is_disk_cached_outputs_expected(infos[1].text_encoder_outputs_npz, 0, None)
        with np.load(infos[0].text_encoder_outputs_npz) as raw:
            assert raw["qwen35_states_v3"].dtype == np.uint16  # bf16 bits, like the Anima prompt_embeds cache
        assert infos[0].text_encoder_outputs_npz.endswith("_anima38_te.npz")
        # the plain Anima strategy never reads these files (different suffix)
        from library.strategy_anima import AnimaTextEncoderOutputsCachingStrategy

        assert AnimaTextEncoderOutputsCachingStrategy(True, 1, False).get_outputs_npz_path(str(tmp_path / "img1.png")) != infos[0].text_encoder_outputs_npz


def test_drop_cached_outputs_refuses_nonzero_rates():
    encoding = a38_strategy.Anima38TextEncodingStrategy()
    parts = [torch.zeros(2, 3)] * 6
    assert encoding.drop_cached_text_encoder_outputs(*parts, caption_dropout_rates=torch.zeros(2)) == parts
    with pytest.raises(RuntimeError, match="--cache_caption_variants"):
        encoding.drop_cached_text_encoder_outputs(*parts, caption_dropout_rates=torch.tensor([0.0, 0.1]))


class _PairingDataset(train_util.BaseDataset):
    """Just enough BaseDataset state for the real __getitem__ with in-memory latents and disk TE caches."""

    def __init__(self, infos, subset, caching, tokenize_strategy, batch_size):
        self.image_data = {key: info for key, info in infos.items()}
        self.image_to_subset = {key: subset for key in infos}
        self.batch_buckets = [list(infos)]
        self.batch_bucket_jitter_resos = []
        self.buckets_indices = [SimpleNamespace(bucket_index=0, bucket_batch_size=batch_size, batch_index=0)]
        self.caching_mode = None
        self.prior_loss_weight = 1.0
        self.text_encoder_output_caching_strategy = caching
        self.tokenize_strategy = tokenize_strategy
        self.is_training_dataset = True
        self.current_epoch = 1
        self.network_multiplier = 1.0
        self.debug_dataset = False
        self.train_inpainting = False
        self._length = 1


def test_dataset_getitem_pairs_the_same_variant_for_both_encoders(tmp_path, tokenize_strategy):
    """The real BaseDataset.__getitem__ picks one variant index per sample and loads it from the one cache file,
    so the Qwen3 and Qwen3.5 rows always come from the same caption; captions of different Qwen3.5 lengths are
    right-padded with a zero mask."""
    models = [_FakeQwen3(), _FakeQwen35()]
    encoding = a38_strategy.Anima38TextEncodingStrategy()
    caching = a38_strategy.Anima38TextEncoderOutputsCachingStrategy(True, 4, False, qwen35_signature="sig-1")
    variants = {
        "img1": ["@a, 1boy, solo, beach, day", "@a, 1boy, day, beach, solo", "@a, 1boy, beach, solo", "@a, 1boy, solo, day, beach, rain"],
        "img2": ["@b, 2boys", "@b, 2boys, very long extra tags here to change length", "", "@b"],
    }
    infos = {}
    for key, caption_variants in variants.items():
        info = _info(tmp_path, key, caption_variants[0], caption_variants)
        info.image_key = key
        info.is_reg = False
        info.is_val = False
        info.latents = torch.zeros(16, 4, 4)
        info.latents_flipped = None
        info.latents_aug_variants = None
        info.latents_original_size = (32, 32)
        info.latents_crop_ltrb = (0, 0, 32, 32)
        info.alpha_mask = None
        info.latents_by_reso = {}
        info.bucket_reso = (32, 32)
        info.resized_size = (32, 32)
        infos[key] = info
    caching.cache_batch_outputs(tokenize_strategy, models, encoding, list(infos.values()))
    expected = {key: [_expected(tokenize_strategy, encoding, models, c) for c in caption_variants] for key, caption_variants in variants.items()}

    subset = SimpleNamespace(custom_attributes={}, flip_aug=False, caption_dropout_every_n_epochs=0)
    dataset = _PairingDataset(infos, subset, caching, tokenize_strategy, batch_size=2)
    random.seed(0)
    seen = {key: set() for key in variants}
    for _ in range(40):
        example = dataset[0]
        te = example["text_encoder_outputs_list"]
        assert len(te) == 7
        pe, am, t5, t5m, q35, q35m, rate = te
        assert q35.shape[:2] == q35m.shape and q35.shape[0] == 2
        for row, key in enumerate(infos):
            ks = [k for k, (want_pe, _, _) in enumerate(expected[key]) if torch.equal(pe[row], want_pe)]
            assert len(ks) == 1, "prompt_embeds must match exactly one cached caption variant"
            k = ks[0]
            _, want_t5, want_q35 = expected[key][k]
            n = want_q35.shape[0]
            assert torch.equal(q35[row, :n], want_q35), "Qwen3.5 row comes from a different caption than the Qwen3 row"
            assert torch.all(q35[row, n:] == 0) and q35m[row].tolist() == [1.0] * n + [0.0] * (q35.shape[1] - n)
            assert torch.equal(t5[row].long(), want_t5.long())
            if k > 0:
                assert example["captions"][row] == variants[key][k]
            seen[key].add(k)
    assert all(len(ks) == 4 for ks in seen.values()), seen  # every variant index was drawn


# ======================================================================================================
# captions: canonical variant without '|||', caption digests, Forge-style Qwen3.5 text (review F1 / F2 / F4)
# ======================================================================================================

from library.anima38 import captions as a38_captions  # noqa: E402


def _caption_subset(**overrides):
    values = dict(
        caption_prefix=None, caption_suffix=None, caption_dropout_rate=0.0, caption_dropout_every_n_epochs=0,
        enable_wildcard=False, shuffle_caption=True, token_warmup_step=0, token_warmup_min=1,
        caption_tag_dropout_rate=0.0, keep_tokens_separator="|||", keep_tokens=0, caption_separator=",",
        secondary_separator=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _bare_dataset(replacements=None):
    dataset = train_util.BaseDataset.__new__(train_util.BaseDataset)
    dataset.replacements = dict(replacements or {})
    dataset.log_caption_dropout = False
    dataset.log_caption_tag_dropout = False
    dataset.current_epoch = 1
    dataset.current_step = 0
    dataset.max_train_steps = 100
    return dataset


USER_CAPTIONS = [
    "@ennjomon, 1boy ||| male focus, muscular male, short hair ||| A large man stands on a beach.",
    "@ennjomon, 2boys ||| male focus,  towel, speech bubble,",
    "no separator, tags only, trailing ,",
    "",
]


@pytest.mark.parametrize("subset_kwargs", [{}, {"shuffle_caption": False}, {"caption_prefix": "masterpiece"}, {"secondary_separator": ";;;"}])
def test_canonical_caption_is_process_caption_without_randomness(monkeypatch, subset_kwargs):
    subset = _caption_subset(**subset_kwargs)
    dataset = _bare_dataset({"@ennjomon": "@ennjomon"})
    monkeypatch.setattr(train_util.random, "shuffle", lambda seq: None)  # process_caption with the order kept
    for caption in USER_CAPTIONS + ["a ;;; b, c ||| d"]:
        expected = train_util.BaseDataset.process_caption(dataset, subset, caption)
        assert a38_captions.canonical_caption(dataset, subset, caption) == expected, caption
        if subset.shuffle_caption:
            assert "|||" not in expected
    # the fork's own canonical variant keeps the separator (what F1 was about)
    original = train_util.BaseDataset.process_caption_canonical(dataset, _caption_subset(), USER_CAPTIONS[0])
    assert "|||" in original
    assert a38_captions.canonical_caption(dataset, _caption_subset(), USER_CAPTIONS[0]) == (
        "@ennjomon, 1boy, male focus, muscular male, short hair, A large man stands on a beach."
    )


def test_install_canonical_captions_patches_instances_and_build_caption_variants_uses_it():
    subset = _caption_subset()
    dataset = _bare_dataset()
    info = SimpleNamespace(image_key="img", caption=USER_CAPTIONS[0], caption_variants=None)
    dataset.image_data = {"img": info}
    dataset.image_to_subset = {"img": subset}
    dataset.caption_aug_variant_count = 4
    group = SimpleNamespace(datasets=[dataset, _bare_dataset()])
    assert a38_captions.install_canonical_captions(group) == 2
    assert a38_captions.install_canonical_captions(None) == 0
    import pickle

    with pytest.raises(AttributeError, match="canonical_caption"):
        pickle.loads(pickle.dumps(dataset))  # what a Windows DataLoader worker does: why the trainer removes it again
    dataset.build_caption_variants()
    assert len(info.caption_variants) == 4
    assert all("|||" not in variant for variant in info.caption_variants)
    assert info.caption_variants[0] == "@ennjomon, 1boy, male focus, muscular male, short hair, A large man stands on a beach."
    assert sorted(info.caption_variants[1].split(", ")) == sorted(info.caption_variants[0].split(", "))
    assert a38_captions.uninstall_canonical_captions(group) == 2
    assert a38_captions.uninstall_canonical_captions(group) == 0
    pickle.loads(pickle.dumps(dataset))
    # the class itself is untouched: other datasets / the plain Anima trainer keep the fork's behaviour
    assert "process_caption_canonical" not in vars(_bare_dataset())
    assert train_util.BaseDataset.process_caption_canonical is not a38_captions.canonical_caption


def test_forge_prompt_text_escapes_like_a_typed_prompt():
    assert a38_captions.forge_prompt_text("shrug (clothing), painting \(object\), [x]") == (
        "shrug \(clothing\), painting \(object\), \[x\]"
    )
    assert a38_captions.forge_prompt_text("") == "" and a38_captions.forge_prompt_text("1boy, solo") == "1boy, solo"


def test_cache_feeds_qwen35_the_escaped_caption_and_qwen3_the_plain_one(tmp_path, tokenize_strategy):
    models = [_FakeQwen3(), _FakeQwen35()]
    encoding = a38_strategy.Anima38TextEncodingStrategy()
    caching = a38_strategy.Anima38TextEncoderOutputsCachingStrategy(True, 4, False, qwen35_signature="sig-1")
    caption = "1boy, shrug (clothing), beach"
    info = _info(tmp_path, "img", caption)
    caching.cache_batch_outputs(tokenize_strategy, models, encoding, [info])
    loaded = caching.load_outputs_npz(info.text_encoder_outputs_npz)
    plain_pe, plain_t5, plain_q35 = _expected(tokenize_strategy, encoding, models, caption)
    _, _, escaped_q35 = _expected(tokenize_strategy, encoding, models, a38_captions.forge_prompt_text(caption))
    assert torch.equal(torch.as_tensor(loaded[0]).to(plain_pe.dtype), plain_pe)  # Qwen3: plain caption
    assert torch.equal(torch.as_tensor(np.asarray(loaded[2])).long(), plain_t5.long())  # T5: plain caption
    assert torch.equal(torch.as_tensor(loaded[4]), escaped_q35)  # Qwen3.5: '\(clothing\)' as typed in Forge
    assert escaped_q35.shape[0] != plain_q35.shape[0]


def test_edited_caption_invalidates_the_cache(tmp_path, tokenize_strategy):
    models = [_FakeQwen3(), _FakeQwen35()]
    encoding = a38_strategy.Anima38TextEncodingStrategy()
    caching = a38_strategy.Anima38TextEncoderOutputsCachingStrategy(True, 4, False, qwen35_signature="sig-1")
    info = _info(tmp_path, "img", "old caption, 1boy")
    info.absolute_path = str(tmp_path / "img.png")
    assert caching.get_outputs_npz_path(info.absolute_path) == info.text_encoder_outputs_npz
    group = SimpleNamespace(datasets=[SimpleNamespace(image_data={"img": info})])
    assert a38_captions.register_caption_digests(group, caching) == 1
    caching.cache_batch_outputs(tokenize_strategy, models, encoding, [info])
    assert caching.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz)

    info.caption = "new caption, 1boy"  # the user edited the .txt
    a38_captions.register_caption_digests(group, caching)
    assert not caching.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz)
    caching.cache_batch_outputs(tokenize_strategy, models, encoding, [info])
    assert caching.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz)
    # a strategy that was never told the digests (older callers) keeps accepting the file
    fresh = a38_strategy.Anima38TextEncoderOutputsCachingStrategy(True, 4, False, qwen35_signature="sig-1")
    assert fresh.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz)
    # a v1 file (no digest key) is rejected by the schema bump through the signature
    assert a38_strategy.CACHE_SCHEMA == "anima38-te-v2"


# ======================================================================================================
# sampling
# ======================================================================================================


def test_do_sample_is_reused_unchanged():
    """ConditionedDiT + DiTConditioning through anima_train_utils.do_sample == do_sample with plain contexts."""
    def fake_dit(x, t, context, padding_mask=None):
        return x * 0.5 + context.mean() * t.view(-1, 1, 1, 1, 1)

    ctx, nctx = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    want = anima_train_utils.do_sample(64, 64, 42, fake_dit, ctx, 6, torch.float32, torch.device("cpu"), 5.0, 3.0, nctx)
    got = anima_train_utils.do_sample(
        64, 64, 42, a38_sampling.ConditionedDiT(fake_dit), a38_sampling.DiTConditioning(ctx), 6, torch.float32, torch.device("cpu"),
        5.0, 3.0, a38_sampling.DiTConditioning(nctx),
    )
    assert torch.equal(got, want)


@pytest.mark.parametrize("negative_mode", ["connector", "native"])
def test_sample_inference_runs_the_connector_per_step(negative_mode):
    model, connector = _conditioned_pair()
    conditioner = a38_training.get_anima38_conditioner(model)
    seen = []
    original_call = conditioner.connector.forward_context

    def spy(*args, **kwargs):
        seen.append(args[3].clone())
        return original_call(*args, **kwargs)

    conditioner.connector.forward_context = spy
    inputs = _tiny_inputs(batch=1, lengths=((5, 7, 6),))

    def encoded(offset):
        return [inputs["source"] + offset, inputs["source_mask"], inputs["t5_ids"], inputs["t5_mask"], inputs["semantic"] + offset, inputs["semantic_mask"]]

    cache = {"pos prompt": encoded(0.0), "neg prompt": encoded(0.5)}
    accelerator = SimpleNamespace(device=torch.device("cpu"), trackers=[])
    prompt = {"prompt": "pos prompt", "negative_prompt": "neg prompt", "sample_steps": 4, "width": 64, "height": 64, "scale": 5.0, "seed": 1}
    native_calls = []
    original_adapter = model.llm_adapter.forward
    model.llm_adapter.forward = lambda *a, **k: native_calls.append(1) or original_adapter(*a, **k)
    with torch.no_grad():
        latents = a38_sampling.sample_image_inference(accelerator, SimpleNamespace(output_name=None), model, None, None, prompt, 0, 0, cache,
                                                      negative_mode=negative_mode, decode=False)
    assert latents.shape == (1, 16, 1, 8, 8) and torch.isfinite(latents).all()
    sigmas = torch.linspace(1.0, 0.0, 5)
    sigmas = (sigmas * 3.0) / (1 + 2.0 * sigmas)
    per_call = 2 if negative_mode == "connector" else 1
    assert len(seen) == 4 * per_call
    for step in range(4):
        for call in range(per_call):
            torch.testing.assert_close(seen[step * per_call + call], sigmas[step].reshape(1))
    # native negative: llm_adapter ran once, up front; the connector path itself also runs the adapter blocks
    # (not adapter.forward), so every recorded forward call is the negative context
    assert len(native_calls) == (1 if negative_mode == "native" else 0)


def test_sample_inference_skips_uncached_prompts():
    model, _ = _conditioned_pair()
    accelerator = SimpleNamespace(device=torch.device("cpu"), trackers=[])
    out = a38_sampling.sample_image_inference(accelerator, SimpleNamespace(output_name=None), model, None, None, {"prompt": "unknown"}, 0, 0, {},
                                              decode=False)
    assert out is None


# ======================================================================================================
# trainer mixin wiring
# ======================================================================================================


class _FakeAnimaTrainer:
    def __init__(self):
        self.seen_text_encoders = None

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        self.seen_text_encoders = list(text_encoders)
        text_encoders[1].get()  # something needed encoding

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        pass

    def update_metadata(self, metadata, args):
        metadata["base"] = True


def test_mixin_loads_qwen35_only_while_caching(monkeypatch):
    from library.anima38.trainer import make_anima38_trainer

    loads = []
    monkeypatch.setattr(a38_strategy, "load_qwen35_model", lambda path, device=None: loads.append((path, device)) or "QWEN35")
    trainer = make_anima38_trainer(_FakeAnimaTrainer)
    args = SimpleNamespace(qwen35_path="q.safetensors")
    accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.cache_text_encoder_outputs_if_needed(args, accelerator, None, None, ["QWEN3"], None, torch.bfloat16)
    qwen3, provider = trainer.seen_text_encoders
    assert qwen3 == "QWEN3" and isinstance(provider, a38_strategy.Qwen35Provider)
    assert loads == [("q.safetensors", torch.device("cpu"))]
    assert not provider.loaded  # released after caching

    metadata = {}
    trainer.update_metadata(metadata, SimpleNamespace(pretrained_model_name_or_path="x/Anima-3.8B-v1.1.safetensors", qwen35_path="y/qwen35_4b.safetensors",
                                                      anima38_sample_negative="connector", anima38_qwen35_batch_size=1))
    assert metadata["base"] and metadata["ss_anima38_connector"] and metadata["ss_anima38_bundle"] == "Anima-3.8B-v1.1.safetensors"


def test_mixin_installs_canonical_captions_and_digests_before_caching(monkeypatch, tmp_path):
    from library.anima38 import trainer as a38_trainer

    caching = a38_strategy.Anima38TextEncoderOutputsCachingStrategy(True, 4, False, qwen35_signature="sig-1")
    monkeypatch.setattr(a38_trainer.TextEncoderOutputsCachingStrategy, "get_strategy", classmethod(lambda cls: caching))
    monkeypatch.setattr(a38_strategy, "load_qwen35_model", lambda path, device=None: "QWEN35")
    dataset = _bare_dataset()
    dataset.image_data = {"img": SimpleNamespace(caption="@a ||| b", absolute_path=str(tmp_path / "img.png"))}
    seen = {}

    class _Base(_FakeAnimaTrainer):
        def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, text_encoders, dataset_group, weight_dtype):
            ds = dataset_group.datasets[0]
            seen["canonical"] = ds.process_caption_canonical(_caption_subset(), "@a ||| b")
            seen["digests"] = dict(caching.caption_digests)

    trainer = a38_trainer.make_anima38_trainer(_Base)
    trainer.cache_text_encoder_outputs_if_needed(
        SimpleNamespace(qwen35_path="q.safetensors"), SimpleNamespace(device=torch.device("cpu")), None, None, ["QWEN3"],
        SimpleNamespace(datasets=[dataset]), torch.bfloat16,
    )
    assert seen["canonical"] == "@a, b"
    key = a38_captions.npz_key(str(tmp_path / "img_anima38_te.npz"))
    assert seen["digests"] == {key: a38_captions.caption_digest("@a ||| b")}
    # removed after caching: Windows DataLoader workers pickle the dataset
    assert "process_caption_canonical" not in vars(dataset)
    import pickle

    pickle.loads(pickle.dumps(dataset))


def test_mixin_grad_policy_is_rechecked_after_apply_to():
    """LyCORIS registers its modules in apply_to(), after post_process_network(): the first DiT call re-checks."""
    from library.anima38.trainer import make_anima38_trainer

    model, _ = _conditioned_pair()
    conditioner = a38_training.get_anima38_conditioner(model)
    trainer = make_anima38_trainer(_FakeAnimaTrainer)
    network = nn.Module()
    trainer.post_process_network(None, None, network, [], model)
    assert conditioner.grad_through_connector is False
    network.lycoris_llm_adapter_blocks_0_cross_attn_q_proj = nn.Linear(2, 2)  # what apply_to() would register
    accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.get_extra_dit_kwargs(None, accelerator, _cached_conds(), torch.float32)
    assert conditioner.grad_through_connector is True


@pytest.mark.skipif(not HAS_BUNDLE, reason="Anima-3.8B bundle not present")
def test_meta_dry_run_full_38b_training_graph(monkeypatch):
    """The trainer's own load_unet_lazily (mixin + Anima) on the real 3.8B header, meta tensors only (no weights, no
    GPU), then the user's LyCORIS LoKr preset on that graph."""
    anima_train_network = pytest.importorskip("anima_train_network")
    from library.anima38 import trainer as a38_trainer

    requested = []
    monkeypatch.setattr(anima_utils, "load_safetensors_with_lora_and_fp8", _meta_loader(requested))
    monkeypatch.setattr(a38_trainer, "load_anima38_connector_state", lambda path, dtype=None: _meta_connector_state(path))
    trainer = a38_trainer.make_anima38_trainer(anima_train_network.AnimaNetworkTrainer)
    args = SimpleNamespace(
        pretrained_model_name_or_path=BUNDLE_PATH, blocks_to_swap=0, fp8_scaled=False, xformers=False, attn_mode="flash", split_attn=False,
        use_ramtorch=False, unsloth_offload_checkpointing=True, anima38_connector=True,
    )
    accelerator = SimpleNamespace(device=torch.device("meta"))
    trainer.is_swapping_blocks = False
    model, text_encoders = trainer.load_unet_lazily(args, torch.bfloat16, accelerator, [])
    conditioner = a38_training.get_anima38_conditioner(model)
    connector = conditioner.connector

    assert len(model.blocks) == 52 and model.attn_mode == "flash"
    assert not any(key.startswith("net.anima_v2_connector.") for key in requested)
    assert connector.native_adapter is model.llm_adapter
    dit_params = sum(p.numel() for p in model.parameters())
    connector_params = sum(p.numel() for p in connector.parameters())
    header, _ = _header(BUNDLE_PATH)
    header_dit = sum(math.prod(info["shape"]) for key, info in header.items() if key != "__metadata__" and not key.startswith("net.anima_v2_connector."))
    assert connector_params == 652_503_576
    assert sum(v.numel() for v in model.state_dict().values()) == header_dit  # every DiT tensor of the bundle, nothing else
    assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in connector.parameters())
    print(f"\n[meta dry run] DiT {dit_params:,} params in {len(model.blocks)} blocks; connector {connector_params:,} params (frozen, bf16, "
          f"not a submodule); keys loaded {len(requested)} of {len(header) - 1}")

    # LyCORIS with the user's network_args. dora_wd reads the original weights (no data on meta), so it is off
    # here; it only adds magnitude vectors to the same modules.
    import lycoris.kohya as lycoris_kohya

    network_args = dict(algo="lokr", preset="anima", full_matrix="True", factor="4", use_tucker="False", train_norm="False", dora_wd="False",
                        rescaled="False", use_scalar="False", bypass_mode="False", enable_conv="False", train_llm_adapter="False",
                        decompose_both="False", torch_compile="False", dropout="0.0", rank_dropout="0.0", module_dropout="0.0")
    network = lycoris_kohya.create_network(1.0, 10000, 10000, None, [], model, **network_args)
    names = [lora.lora_name for lora in network.unet_loras]
    assert len(names) == 520
    assert all("_blocks_" in name and "llm_adapter" not in name for name in names)
    block_ids = {int(name.split("_blocks_")[1].split("_")[0]) for name in names}
    assert block_ids == set(range(52))  # every block, as Forge requires for a 52-block LoRA
    trainer.post_process_network(args, accelerator, network, [], model)
    assert conditioner.grad_through_connector is False
    lora_params = sum(p.numel() for lora in network.unet_loras for p in lora.parameters())
    print(f"[meta dry run] LyCORIS lokr preset=anima: {len(names)} modules over {len(block_ids)} blocks, "
          f"{lora_params:,} LoKr params (without the DoRA magnitudes); none on the connector")


# ======================================================================================================
# opt-in: real weights on CPU
# ======================================================================================================


@pytest.mark.skipif(not REAL_WEIGHTS, reason=REAL_SKIP_REASON)
def test_real_qwen35_encoding_equals_forge_call(tokenize_strategy):
    from library.anima38.qwen35 import load_qwen35_model

    model = load_qwen35_model(QWEN35_PATH)
    captions = ["@ennjomon, 1boy, muscular male, beach", ""]
    tokens = tokenize_strategy.tokenize(captions)
    states, mask = a38_strategy.Anima38TextEncodingStrategy(qwen35_batch_size=1).encode_qwen35(model, tokens[4], tokens[5])
    for row, caption in enumerate(captions):
        ids = torch.tensor(tokenize_strategy.qwen35_tokenizer([caption])["input_ids"])
        _, intermediate = model(ids, attention_mask=torch.ones_like(ids), intermediate_output=[7, 15, 23, 31], dtype=torch.float32)
        forge = torch.cat([intermediate[i] for i in (7, 15, 23, 31)], dim=-1)[0]  # runtime.py _semantic_layers
        n = ids.shape[1]
        assert torch.equal(states[row, :n], forge), "cached features must be bitwise Forge's"
        print(f"\n[real qwen35] '{caption}': {n} tokens, dtype {states.dtype}, bitwise equal to the Forge call")


@pytest.mark.skipif(not REAL_WEIGHTS, reason=REAL_SKIP_REASON)
def test_real_weights_cpu_forward_backward(tokenize_strategy):
    """Real DiT + real connector + real Qwen3 / Qwen3.5 features on CPU: one training-style forward/backward.

    Runs the DiT in fp32: oneDNN on this CPU has no bf16 backward ("DNNL does not support bf16/f16 backward on the
    platform with avx2_vnni_2"); GPU training uses bf16. The text encoders stay bf16 (forward only)."""
    import time

    from library.anima38.qwen35 import load_qwen35_model

    start = time.time()
    qwen3, _ = anima_utils.load_qwen3_text_encoder(QWEN3_PATH, dtype=torch.bfloat16, device="cpu")
    encoding = a38_strategy.Anima38TextEncodingStrategy()
    caption = "@ennjomon, 1boy, muscular male, bara, solo, beach, day, looking at viewer"
    with torch.no_grad():
        pe, am, t5, t5m, q35, q35m = encoding.encode_tokens(tokenize_strategy, [qwen3, load_qwen35_model(QWEN35_PATH)], tokenize_strategy.tokenize([caption]))
    encode_time = time.time() - start

    start = time.time()
    dtype = torch.float32
    model = anima_utils.load_anima_model("cpu", BUNDLE_PATH, "torch", False, "cpu", dtype, False, anima38_connector=True)
    connector_state, info = a38_bundle.load_anima38_connector_state(BUNDLE_PATH)
    connector = a38_bundle.build_semantic_connector(model.llm_adapter, connector_state, info["config"], info["layer_indices"], dtype=dtype)
    conditioner = a38_training.attach_anima38_conditioner(model, connector)
    model.requires_grad_(False)
    load_time = time.time() - start

    targets = [next(p for n, p in model.blocks[i].named_parameters() if p.ndim == 2) for i in (0, 51)]
    for target in targets:
        target.requires_grad_(True)
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, 16, 1, 16, 16, generator=generator).to(dtype)
    t = torch.tensor([0.6], dtype=dtype)
    padding_mask = torch.zeros(1, 1, 16, 16, dtype=dtype)
    kwargs = dict(padding_mask=padding_mask, target_input_ids=t5.long(), target_attention_mask=t5m, source_attention_mask=am)
    start = time.time()
    with torch.enable_grad():
        out = model(x, t, pe.to(dtype), semantic_hidden_states=q35.to(dtype), semantic_attention_mask=q35m, **kwargs)
        loss = out.float().pow(2).mean()
        loss.backward()
    step_time = time.time() - start
    with torch.no_grad():
        v2 = conditioner(pe.to(dtype), t, t5.long(), t5m, am, q35.to(dtype), q35m)
        native = model._preprocess_text_embeds(pe.to(dtype), t5.long(), t5m, am)
    valid = t5m[0].bool()  # the connector context is zero-padded to 512 rows (Forge); the native one keeps t5_max_length
    assert v2.shape[1] == 512 and torch.all(v2[0, valid.shape[0]:] == 0)
    rel = ((v2[0, : valid.shape[0]][valid].float() - native[0, valid].float()).norm() / native[0, valid].float().norm()).item()
    assert out.shape == (1, 16, 1, 16, 16) and torch.isfinite(out.float()).all()
    for target in targets:
        assert target.grad is not None and torch.isfinite(target.grad.float()).all() and target.grad.abs().sum() > 0
    assert all(p.grad is None for p in connector.parameters())
    assert rel > 0.01
    print(f"\n[real cpu] encode {encode_time:.1f}s (qwen35 tokens {int(q35m.sum())}), load DiT+connector {load_time:.1f}s, "
          f"forward+backward {step_time:.1f}s, loss {loss.item():.4f}, |v2 - native| / |native| over {int(valid.sum())} T5 rows = {rel:.3f}")
