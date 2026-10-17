"""CPU tests for the MOSS-Image trainer plumbing (sampler, LoRA/LyCORIS naming, TE caching). No model files needed.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe tests/test_moss_pipeline.py   (or with pytest if installed)
"""
import argparse
import os
import tempfile
import types
from dataclasses import replace

import numpy as np
import torch

from library.moss import captions, flow, strategy
from library.moss.lycoris_presets import register_moss_lycoris_presets
from library.moss.models import MossTransformer
from library.moss.qwen35 import MossQwen35Config, MossQwen35TextEncoder

TINY = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=48, freq_dim=16, latent_channels=128, rope_base=10000.0)


def _args(**kw):
    base = dict(moss_logit_mean=1.09, moss_logit_std=0.95, moss_sigma_cutoff=0.95, moss_uniform_mix=0.01,
                timestep_sampling="moss", ip_noise_gamma=0.0, ip_noise_gamma_random_strength=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_moss_sigma_distribution_matches_report():
    pmf = flow.moss_sigma_pmf()
    sigma = torch.arange(1, 1001, dtype=torch.float64) / 1000
    assert abs(float(pmf.sum()) - 1.0) < 1e-9
    mean = float((pmf * sigma).sum())
    median = float(sigma[int(torch.searchsorted(pmf.cumsum(0), torch.tensor(0.5, dtype=torch.float64)))])
    assert abs(mean - 0.744) < 0.003 and abs(median - 0.781) < 0.003, (mean, median)  # spec_moss_official 8.2


def test_moss_sampler_noising_is_fp32_and_on_grid():
    args = _args()
    sampler = flow.MossSigmaSampler(args)
    torch.manual_seed(0)
    latents, noise = torch.randn(4, 128, 2, 3), torch.randn(4, 128, 2, 3)
    sched = types.SimpleNamespace(config=types.SimpleNamespace(num_train_timesteps=1000))
    noisy, t, s = flow.get_noisy_model_input_and_timesteps(args, sched, latents, noise, "cpu", torch.bfloat16, sampler)
    assert noisy.dtype == torch.bfloat16 and t.dtype == torch.float32
    assert torch.equal(t, torch.round(t)) and bool(((t >= 1) & (t <= 1000)).all())
    want = (1 - s) * latents + s * noise
    assert (noisy.float() - want).abs().max() < 0.02


def _tiny_dit():
    torch.manual_seed(0)
    return MossTransformer(**TINY)


def test_lycoris_moss_preset_keys():
    register_moss_lycoris_presets()
    from lycoris import kohya

    net = kohya.create_network(1.0, 10000, 10000, None, [], _tiny_dit(), algo="lokr", preset="moss", factor=4,
                               full_matrix=True, dora_wd=True)
    net.apply_to([], _tiny_dit(), apply_text_encoder=False, apply_unet=True) if False else None
    names = sorted({lora.lora_name for lora in net.unet_loras})
    want = sorted(f"lora_unet_layers_{i}_proj_{p}" for i in range(3) for p in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"))
    assert names == want, names


def test_lora_moss_keys_and_default_excludes():
    from networks import lora_moss

    net = lora_moss.create_network(1.0, 4, 1, None, [], _tiny_dit())
    names = sorted(lora.lora_name for lora in net.unet_loras)
    assert names == sorted(f"lora_unet_layers_{i}_proj_{p}" for i in range(3)
                           for p in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")), names
    from networks import lora_anima
    assert lora_anima.LoRANetwork.ANIMA_TARGET_REPLACE_MODULE[0] == "Block"  # restored


def _tiny_te():
    cfg = replace(MossQwen35Config(), hidden_size=64, intermediate_size=96, num_hidden_layers=4)
    torch.manual_seed(0)
    model = MossQwen35TextEncoder(cfg).to(torch.bfloat16)
    for p in model.parameters():
        p.data.normal_(0, 0.05)
    return model


class _Info:
    def __init__(self, path, caption, variants=None):
        self.absolute_path = path
        self.caption = caption
        self.caption_variants = variants
        self.caption_dropout_rate = 0.0
        self.caption_aug_hash = "h"
        self.text_encoder_outputs_npz = None


def test_te_cache_roundtrip_unpadded_per_caption():
    try:
        tokenize = strategy.MossTokenizeStrategy()
    except FileNotFoundError:
        return
    model = _tiny_te()
    provider = strategy.MossTextEncoderProvider("unused", "cpu", loader=lambda *a, **k: model)
    encoding = strategy.MossTextEncodingStrategy()
    with tempfile.TemporaryDirectory() as tmp:
        caching = strategy.MossTextEncoderOutputsCachingStrategy(True, 2, False, signature="sig")
        infos = [_Info(os.path.join(tmp, "a.png"), "a", ["1girl, smile", "smile, 1girl", ""]),
                 _Info(os.path.join(tmp, "b.png"), "b", ["a much longer caption with more words in it", "x", "y"])]
        for info in infos:
            info.text_encoder_outputs_npz = caching.get_outputs_npz_path(info.absolute_path)
        caching.cache_batch_outputs(tokenize, [provider], encoding, infos)
        for info in infos:
            assert caching.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz, 3, "h")
            for k, caption in enumerate(info.caption_variants):
                embeds, mask, rate = caching.load_outputs_npz(info.text_encoder_outputs_npz, k)
                ids = tokenize.tokenizer.encode(caption)
                assert embeds.dtype == torch.bfloat16 and tuple(embeds.shape) == (len(ids), 64)
                ref = model(torch.tensor([ids]))[0]
                assert torch.equal(embeds, ref), (info.caption, k)  # unpadded encode == stored row
                assert mask.dtype == np.int32 and mask.sum() == len(ids)
        caching.signature = "other"
        assert not caching.is_disk_cached_outputs_expected(infos[0].text_encoder_outputs_npz, 3, "h")


def test_lora_moss_roundtrip_keeps_text_refiner_adapters():
    from networks import lora_moss

    model = _tiny_dit()
    net = lora_moss.create_network(1.0, 2, 2.0, None, [], model, train_llm_adapter="true",
                                   include_patterns="['layers\\\\..*', 'text_embedder\\\\.blocks\\\\..*']")
    net.apply_to([], model, False, True)  # registers the modules: state_dict() is empty before apply_to
    target = _tiny_dit()
    loaded, weights = lora_moss.create_network_from_weights(1.0, None, None, [], target, weights_sd=net.state_dict())
    loaded.apply_to([], target, False, True)
    assert len(loaded.unet_loras) == len(net.unet_loras) == 28  # 18 joint-block + 10 text-refiner Linears
    result = loaded.load_state_dict(weights, strict=False)
    assert not result.unexpected_keys and not [k for k in result.missing_keys if "lora_" in k], result


def test_moss_text_preset_trains_text_blocks_under_unsloth_checkpointing():
    register_moss_lycoris_presets()
    from lycoris import kohya

    model = _tiny_dit()
    model.requires_grad_(False)
    net = kohya.create_network(1.0, 2, 2.0, None, [], model, algo="lokr", preset="moss-text")
    net.apply_to([], model, False, True)
    model.train().enable_gradient_checkpointing(unsloth_offload=True)
    x = torch.randn(2, 128, 2, 2, requires_grad=True)
    model(x, torch.tensor([10.0, 20.0]), torch.randn(2, 4, TINY["text_dim"]), torch.ones(2, 4, dtype=torch.bool)).square().mean().backward()
    text = [p.grad is not None for n, p in net.named_parameters() if "text_embedder" in n]
    assert text and all(text)


def test_non_variant_subsets_cache_the_processed_caption():
    subset = types.SimpleNamespace(caption_prefix="PREFIX", caption_suffix=None, enable_wildcard=False, shuffle_caption=False,
                                   token_warmup_step=0, caption_tag_dropout_rate=0.0, keep_tokens_separator="|||",
                                   caption_separator=",", keep_tokens=0, secondary_separator=";")
    info = _Info("x.png", "old; tag\nsecond line")
    info.image_key = "k"
    dataset = types.SimpleNamespace(replacements={"old": "new"}, image_to_subset={"k": subset}, image_data={"k": info})
    assert captions.prepare_canonical_captions(dataset) == 1
    assert info.moss_canonical_caption == "PREFIX new, tag"
    before = captions.info_caption_digest(info)
    subset.caption_prefix = "OTHER"  # a processing change with the same caption file must change the cache digest
    captions.prepare_canonical_captions(dataset)
    assert captions.info_caption_digest(info) != before


def _lokr_linear(w0, fast):
    """bf16 Linear wrapped by LyCORIS LoKr (full_matrix, factor 4, DoRA input axis) like config_moss.toml."""
    from lycoris.modules.lokr import LokrModule
    from library import lycoris_fast as lokr_fast

    lokr_fast.install_fast_lokr_forward()
    out_dim, in_dim = w0.shape
    org = torch.nn.Linear(in_dim, out_dim, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        org.weight.copy_(w0)
    org.weight.requires_grad_(False)
    module = LokrModule("t", org, multiplier=1.0, lora_dim=10000, alpha=10000, factor=4, weight_decompose=True,
                        wd_on_output=False, full_matrix=True, bypass_mode=False)
    module.apply_to()
    if not fast:
        org.forward = types.MethodType(lokr_fast._STOCK_FORWARD, module)
    with torch.no_grad():  # a trained-looking state: non-zero w2, w1 off identity, DoRA scale moved
        module.lokr_w2.copy_(torch.randn(module.lokr_w2.shape, generator=torch.Generator().manual_seed(3)) * 0.02)
        module.lokr_w1.copy_(torch.arange(16.0).view(4, 4) / 7 - 1)
        module.dora_scale.mul_(1.01)
    return org, module


def test_fast_lokr_forward_is_bit_identical_to_lycoris():
    from library import lycoris_fast as lokr_fast

    for out_dim, in_dim in [(192, 128), (256, 96), (96, 256)]:
        w0 = (torch.randn(out_dim, in_dim, generator=torch.Generator().manual_seed(0)) * 0.05).to(torch.bfloat16)
        results = []
        for fast in (False, True):
            org, module = _lokr_linear(w0, fast)
            assert (org.forward.__func__ is lokr_fast._fast_forward) == fast and lokr_fast._static_eligible(module)
            x = torch.randn(3, 50, in_dim, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16).requires_grad_(True)
            grad = torch.randn(3, 50, out_dim, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                with torch.no_grad():
                    first = org(x)  # unsloth's no-grad pass
                y = org(x)  # checkpoint recompute
            y.backward(grad)
            results.append((first, y, x.grad, module.lokr_w1.grad, module.lokr_w2.grad, module.dora_scale.grad))
        same = [torch.equal(a, b) for a, b in zip(*results)]
        assert all(same), ((out_dim, in_dim), same)


def test_cached_lokr_weights_match_and_restore():
    from library.lycoris_fast import cached_lokr_weights

    w0 = (torch.randn(192, 128, generator=torch.Generator().manual_seed(4)) * 0.05).to(torch.bfloat16)
    org, module = _lokr_linear(w0, fast=True)
    model = torch.nn.Sequential(org)
    x = torch.randn(2, 33, 128, generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
    before = org.forward
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        ref = model(x)
        with cached_lokr_weights(model, torch.bfloat16) as count:
            assert count == 1 and org.forward is not before
            cached = model(x)
    assert torch.equal(cached, ref)
    assert org.forward == before and org.forward.__self__ is module  # LyCORIS forward restored


def test_fast_lokr_with_bias_and_cached_static_check():
    """Biased Linears (preset moss-text) stay bit-identical; the static check is computed once per module."""
    from lycoris.modules.lokr import LokrModule
    from library import lycoris_fast as lokr_fast

    lokr_fast.install_fast_lokr_forward()
    results = []
    for fast in (False, True):
        torch.manual_seed(0)
        org = torch.nn.Linear(128, 192, bias=True).to(torch.bfloat16).requires_grad_(False)
        module = LokrModule("t", org, multiplier=1.0, lora_dim=10000, alpha=10000, factor=4, weight_decompose=True,
                            wd_on_output=False, full_matrix=True, bypass_mode=False)
        module.apply_to()
        if not fast:
            org.forward = types.MethodType(lokr_fast._STOCK_FORWARD, module)
        with torch.no_grad():
            module.lokr_w2.normal_(0, 0.02)
        x = torch.randn(2, 9, 128, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16).requires_grad_(True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = org(x)
        y.float().sum().backward()
        results.append((y, x.grad, module.lokr_w2.grad, module.dora_scale.grad))
        if fast:
            assert module.__dict__.get("_moss_fast_static") is True
    assert all(torch.equal(a, b) for a, b in zip(*results))


def test_compiled_chain_failure_falls_back_to_eager():
    import torch._dynamo.exc as dynamo_exc
    from library.moss import models

    def broken(*args):
        raise dynamo_exc.BackendCompilerFailed(lambda *a: None, RuntimeError("no triton"), None)

    old = models._CHAINS.enabled
    try:
        models.enable_elementwise_compile(True)
        call = models._CHAINS.guard("situ", broken, models.situ)
        h, g = torch.randn(3, 5), torch.randn(3, 5)
        assert torch.equal(call(h, g), models.situ(h, g))
        assert models._CHAINS.enabled is False  # switched off after the first failure
    finally:
        models.enable_elementwise_compile(old)


def test_canonical_caption_drops_separator():
    subset = types.SimpleNamespace(caption_prefix=None, caption_suffix=None, enable_wildcard=False, shuffle_caption=True,
                                   token_warmup_step=0, caption_tag_dropout_rate=0.0, keep_tokens_separator="|||",
                                   caption_separator=",", keep_tokens=0, secondary_separator=None)
    dataset = types.SimpleNamespace(replacements={})
    out = captions.canonical_caption(dataset, subset, "@mytrigger, 1girl ||| solo, smile ||| A girl smiles, softly.")
    assert out == "@mytrigger, 1girl, solo, smile, A girl smiles, softly.", out


if __name__ == "__main__":
    import sys
    import traceback

    failed = 0
    for name, fn in [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print("failed:", failed)
    sys.exit(1 if failed else 0)
