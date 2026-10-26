"""CPU tests for the MOSS-Image trainer plumbing (sampler, LoRA/LyCORIS naming, TE caching). No model files needed.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe tests/test_moss_pipeline.py   (or with pytest if installed)
"""
import argparse
import contextlib
import inspect
import io
import logging
import os
import tempfile
import types
from dataclasses import replace

import numpy as np
import torch

try:
    import pytest
except ImportError:  # `python tests/test_moss_pipeline.py` runs every test without pytest
    pytest = None

from library.moss import captions, flow, strategy
from library.moss.lycoris_presets import register_moss_lycoris_presets
from library.moss.models import MossTransformer
from library.moss.qwen35 import MossQwen35Config, MossQwen35TextEncoder

TINY = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=48, freq_dim=16, latent_channels=128, rope_base=10000.0)


class Skip(Exception):
    pass


def skip(reason):
    if pytest is not None:
        pytest.skip(reason)
    raise Skip(reason)


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


def _net_kwargs(network_args):
    return dict(entry.split("=", 1) for entry in network_args)  # train_network's parsing


def test_exclude_blocks_parsing():
    from library.moss.train_utils import parse_moss_exclude_blocks

    assert parse_moss_exclude_blocks("22") == [22] and parse_moss_exclude_blocks(" 22, 1 ") == [1, 22]
    assert parse_moss_exclude_blocks(22) == [22] and parse_moss_exclude_blocks([23, 22, 22]) == [22, 23]
    assert parse_moss_exclude_blocks("") == [] and parse_moss_exclude_blocks(None) == [] and parse_moss_exclude_blocks([]) == []
    for bad in ("abc", "42", "-1", True, "1.5"):
        try:
            parse_moss_exclude_blocks(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_block_exclusion_merges_network_args_once():
    from library.moss.train_utils import apply_moss_block_exclusion, moss_metadata

    args = argparse.Namespace(moss_lora_exclude_blocks="22", network_args=["algo=lokr", "preset=moss"])
    assert apply_moss_block_exclusion(args) == [22]
    import ast

    assert args.network_args[:2] == ["algo=lokr", "preset=moss"]
    assert ast.literal_eval(_net_kwargs(args.network_args)["exclude_patterns"]) == [r"layers\.22(\..*)?"]
    apply_moss_block_exclusion(args)  # idempotent
    assert len(args.network_args) == 3
    merged = argparse.Namespace(moss_lora_exclude_blocks="1,2", network_args=["exclude_patterns=['foo.*']"])
    apply_moss_block_exclusion(merged)
    assert ast.literal_eval(_net_kwargs(merged.network_args)["exclude_patterns"]) == [
        "foo.*", r"layers\.1(\..*)?", r"layers\.2(\..*)?"]
    other = argparse.Namespace(moss_lora_exclude_blocks="22", network_module="lycoris.kohya", network_args=["preset=full"])
    try:
        apply_moss_block_exclusion(other)
        raise AssertionError("a non-MOSS LyCORIS preset must be refused")
    except ValueError:
        pass
    ok = argparse.Namespace(moss_lora_exclude_blocks="22", network_module="lycoris.kohya", network_args=["preset=moss-text"])
    assert apply_moss_block_exclusion(ok) == [22]
    off = argparse.Namespace(moss_lora_exclude_blocks="", network_args=None)
    assert apply_moss_block_exclusion(off) == [] and off.network_args is None
    meta_args = argparse.Namespace(**{**vars(_args()), "moss_text_encoder": "te", "moss_max_token_length": 4096,
                                     "discrete_flow_shift": 1.0, "weighting_scheme": "uniform", "logit_mean": 0.0,
                                     "logit_std": 1.0, "sigmoid_scale": 1.0, "_moss_excluded_blocks": [22]})
    assert moss_metadata(meta_args)["ss_moss_lora_exclude_blocks"] == "22"


def test_excluded_blocks_get_no_adapters_in_both_network_modules():
    from library.moss.train_utils import apply_moss_block_exclusion
    from networks import lora_moss

    register_moss_lycoris_presets()
    from lycoris import kohya

    args = argparse.Namespace(moss_lora_exclude_blocks="1", network_args=[])
    apply_moss_block_exclusion(args)
    kwargs = _net_kwargs(args.network_args)
    want = sorted(f"lora_unet_layers_{i}_proj_{p}" for i in (0, 2)
                  for p in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"))
    lyco = kohya.create_network(1.0, 10000, 10000, None, [], _tiny_dit(), algo="lokr", preset="moss", factor=4,
                                full_matrix=True, dora_wd=True, **kwargs)
    assert sorted({lora.lora_name for lora in lyco.unet_loras}) == want
    lora = lora_moss.create_network(1.0, 4, 1, None, [], _tiny_dit(), **kwargs)
    assert sorted(m.lora_name for m in lora.unet_loras) == want
    full = kohya.create_network(1.0, 10000, 10000, None, [], _tiny_dit(), algo="lokr", preset="moss", factor=4,
                                full_matrix=True, dora_wd=True)
    assert len(full.unet_loras) == 18  # nothing leaks into a later network without the flag


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


class FakeMossTokenizer:
    """Character-level stand-in for MossTokenizer with the same interface (encode / __call__ / pad_token_id /
    max_length): the chat template and the ComfyUI unescaping are applied exactly like the real one."""

    pad_token_id = 248044
    tokenizer_dir = "<fake>"

    def __init__(self, max_length: int = 4096):
        self.max_length = max_length

    def encode(self, prompt: str):
        from library.moss.qwen35 import format_moss_prompt, unescape_prompt

        return [ord(c) % 250 + 1 for c in format_moss_prompt(unescape_prompt(prompt))][: self.max_length]

    def __call__(self, prompts):
        return [self.encode(p) for p in prompts]


def _fake_tokenize_strategy(max_length: int = 4096):
    tokenize = strategy.MossTokenizeStrategy.__new__(strategy.MossTokenizeStrategy)
    tokenize.tokenizer = FakeMossTokenizer(max_length)
    return tokenize


def test_tokenize_strategy_pads_right_and_masks():
    tokenize = _fake_tokenize_strategy()
    ids, mask = tokenize.tokenize(["ab", "abcd"])
    template = len(tokenize.tokenizer.encode(""))
    assert tuple(ids.shape) == (2, template + 4) == tuple(mask.shape)
    assert mask.sum(1).tolist() == [template + 2, template + 4]
    assert bool((ids[0, template + 2:] == FakeMossTokenizer.pad_token_id).all())
    single_ids, single_mask = tokenize.tokenize("ab")
    assert tuple(single_ids.shape) == (1, template + 2) and torch.equal(single_ids[0], ids[0, : template + 2])
    assert int(single_mask.sum()) == template + 2
    assert tuple(_fake_tokenize_strategy(max_length=5).tokenize("a long prompt")[0].shape) == (1, 5)


def test_te_cache_roundtrip_unpadded_per_caption():
    _te_cache_roundtrip(_fake_tokenize_strategy())  # runs on every machine


def test_te_cache_roundtrip_with_the_real_tokenizer():
    try:
        tokenize = strategy.MossTokenizeStrategy()
    except FileNotFoundError as e:
        skip(str(e))
    _te_cache_roundtrip(tokenize)


def _te_cache_roundtrip(tokenize):
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


def test_lora_moss_adapter_flag_alone_adds_the_text_refiner():
    """train_llm_adapter=true (or the upstream name train_text_embedder=true) adds the 10 refiner Linears to the 18
    joint-block ones; lora_anima's "_embedder" default exclude used to swallow them, and include_patterns (a whitelist
    in this fork) must not be needed. --moss_lora_exclude_blocks keeps working with the flag."""
    from library.moss.train_utils import apply_moss_block_exclusion
    from networks import lora_anima, lora_moss

    for flag in ("train_llm_adapter", "train_text_embedder"):
        net = lora_moss.create_network(1.0, 2, 2.0, None, [], _tiny_dit(), **{flag: "true"})
        names = sorted(lora.lora_name for lora in net.unet_loras)
        assert len(names) == 28 and sum(n.startswith("lora_unet_text_embedder_blocks_") for n in names) == 10, (flag, names)
        assert "lora_unet_text_embedder_proj" not in names  # only the two refiner blocks, not the input projection
    assert len(lora_moss.create_network(1.0, 2, 2.0, None, [], _tiny_dit(), train_llm_adapter="false").unet_loras) == 18
    args = argparse.Namespace(moss_lora_exclude_blocks="1", network_args=[])
    apply_moss_block_exclusion(args)
    net = lora_moss.create_network(1.0, 2, 2.0, None, [], _tiny_dit(), train_llm_adapter="true", **_net_kwargs(args.network_args))
    names = sorted(lora.lora_name for lora in net.unet_loras)
    assert len(names) == 22 and not any("layers_1_" in n for n in names), names
    assert lora_anima.LoRANetwork.__name__ == "LoRANetwork"  # the module global is restored after each call


def test_lora_moss_roundtrip_keeps_text_refiner_adapters():
    from networks import lora_moss

    model = _tiny_dit()
    net = lora_moss.create_network(1.0, 2, 2.0, None, [], model, train_llm_adapter="true")
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
    assert info.cache_caption == "PREFIX new, tag\nsecond line"  # no enable_wildcard: the whole caption
    before = captions.info_caption_digest(info)
    subset.caption_prefix = "OTHER"  # a processing change with the same caption file must change the cache digest
    captions.prepare_canonical_captions(dataset)
    assert captions.info_caption_digest(info) != before


def _dit_inputs():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(2, 128, 2, 3, generator=g)
    t = torch.tensor([700.0, 150.0])
    text = torch.randn(2, 5, TINY["text_dim"], generator=g)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1, 3:] = False
    return x, t, text, mask


def test_lora_moss_identity_at_init_and_merge_roundtrip():
    """A fresh LoRA changes nothing; trained weights saved, reloaded with create_network_from_weights and merged into a
    fresh DiT give the same output as the LoRA-wrapped model."""
    from networks import lora_moss

    model = _tiny_dit().eval()
    x, t, text, mask = _dit_inputs()
    with torch.no_grad():
        base = model(x, t, text, mask)
    net = lora_moss.create_network(1.0, 4, 2.0, None, [], model, train_llm_adapter="true")
    net.apply_to([], model, False, True)
    with torch.no_grad():
        assert torch.equal(model(x, t, text, mask), base)  # lora_up starts at zero
    torch.manual_seed(1)
    for lora in net.unet_loras:
        torch.nn.init.normal_(lora.lora_up.weight, std=0.1)
    with torch.no_grad():
        trained = model(x, t, text, mask)
    assert not torch.allclose(trained, base)
    weights = {k: v.clone() for k, v in net.state_dict().items()}
    fresh = _tiny_dit().eval()
    loaded, sd = lora_moss.create_network_from_weights(1.0, None, None, [], fresh, weights_sd=weights, for_inference=True)
    assert len(loaded.unet_loras) == 28
    loaded.merge_to([], fresh, sd, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        merged = fresh(x, t, text, mask)
    torch.testing.assert_close(merged, trained, atol=1e-5, rtol=1e-5)


def test_lokr_moss_identity_at_init_and_merge_roundtrip():
    """LyCORIS LoKr (preset moss, full_matrix, DoRA like config_moss.toml): identity at init, and the saved weights
    reloaded through lycoris.kohya.create_network_from_weights and merged match the wrapped model."""
    register_moss_lycoris_presets()
    from lycoris import kohya

    model = _tiny_dit().eval()
    x, t, text, mask = _dit_inputs()
    with torch.no_grad():
        base = model(x, t, text, mask)
    net = kohya.create_network(1.0, 10000, 10000, None, [], model, algo="lokr", preset="moss", factor=4,
                               full_matrix=True, dora_wd=True)
    net.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    with torch.no_grad():
        torch.testing.assert_close(model(x, t, text, mask), base, atol=1e-5, rtol=1e-5)  # lokr_w2 starts at zero
        for lora in net.unet_loras:
            lora.lokr_w2.normal_(0, 0.02, generator=torch.Generator().manual_seed(3))
        trained = model(x, t, text, mask)
    assert not torch.allclose(trained, base)
    weights = {k: v.clone() for k, v in net.state_dict().items()}
    fresh = _tiny_dit().eval()
    loaded, _ = kohya.create_network_from_weights(1.0, None, None, [], fresh, weights_sd=weights)
    assert len(loaded.unet_loras) == 18
    loaded.apply_to([], fresh, apply_text_encoder=False, apply_unet=True)  # registers the modules (like merge_lora.py)
    info = loaded.load_state_dict(weights, strict=False)
    assert not info.unexpected_keys and not info.missing_keys, info
    loaded.merge_to([], fresh, weights, torch.float32, "cpu")  # LoKr (+ DoRA scale) folded into the Linear weights
    loaded.restore()  # original forwards back: the merged weights alone must reproduce the wrapped model
    with torch.no_grad():
        merged = fresh(x, t, text, mask)
    torch.testing.assert_close(merged, trained, atol=1e-4, rtol=1e-4)


def test_show_timesteps_console_runs_every_sampling_mode():
    """--show_timesteps goes through the trainer's own sampling path (moss pmf, sigmoid presets, scale_resume epoch)."""
    import moss_train_network
    from library.moss.train_utils import show_timesteps

    parser = moss_train_network.setup_parser()
    cases = [["--timestep_sampling", "moss"],
             ["--timestep_sampling", "sigmoid", "--sigmoid_scale", "content"],
             ["--timestep_sampling", "sigmoid", "--sigmoid_scale", "1.3", "--scale_resume", "2", "--max_train_epochs", "3",
              "--show_timesteps_epoch", "4"],
             ["--timestep_sampling", "sigma", "--weighting_scheme", "logit_normal", "--logit_mean", "-1.09", "--logit_std", "0.95"],
             ["--timestep_sampling", "moss", "--min_snr_gamma", "5"]]
    for extra in cases:
        args = parser.parse_args(["--show_timesteps", "console"] + extra)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            show_timesteps(args, n_try=640)
        out = buffer.getvalue()
        assert "MOSS-Image timestep distribution" in out and "Sampled timesteps" in out and "Sampled loss weighting" in out, extra
        if "--show_timesteps_epoch" in extra:
            assert "epoch 4: preset style" in out, out[:300]
        if "--min_snr_gamma" in extra:
            import re

            peak = float(re.search(r"max weighting=([0-9.]+)", out).group(1))
            assert 0.7 <= peak <= 0.9, out[-1200:]  # min-SNR gamma 5: min(SNR, 5) / (SNR + 1) peaks at 5/6 near sigma 0.31
            assert "   0-  24:     0.00" in out, out[-400:]  # and is ~0 at the clean end, unlike the flat 1.00 of plain logit-normal


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


def test_compiled_rebuild_switch_keeps_cpu_path_identical():
    """enable_compiled_rebuild only compiles for CUDA tensors: on the CPU the fast forward stays bit-identical to
    LyCORIS with the switch on, and the trainers' flags turn it on and off."""
    from library import lycoris_fast as lokr_fast

    w0 = (torch.randn(192, 128, generator=torch.Generator().manual_seed(0)) * 0.05).to(torch.bfloat16)
    x = torch.randn(2, 9, 128, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    outs = []
    for fast, compiled in ((False, False), (True, True)):
        lokr_fast.enable_compiled_rebuild(compiled)
        org, _ = _lokr_linear(w0, fast)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            outs.append(org(x))
    lokr_fast.enable_compiled_rebuild(False)
    assert torch.equal(outs[0], outs[1])

    import moss_train_network

    parser = moss_train_network.setup_parser()
    for extra, want in (([], True), (["--moss_no_lokr_compile"], False), (["--moss_no_compile"], False),
                        (["--moss_stock_lycoris"], False)):
        args = parser.parse_args(["--network_module", "lycoris.kohya"] + extra)
        moss_train_network.prepare_moss_runtime(args)
        assert lokr_fast._REBUILD.enabled is want, extra
    lokr_fast.enable_compiled_rebuild(False)
    from library.moss import models
    models.enable_elementwise_compile(False)


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


# ------------------------------------------------------------------------------------------------- trainer paths
# MossNetworkTrainer.get_noise_pred_and_target / sampling as train_network.train runs them, on the tiny DiT: the
# LyCORIS timestep of TSM / T-LoRA / TSM-LoKr (training and sample images), TSM-LoKr grouped accumulation,
# --flow_use_ot, the adv term, the default path bit for bit, and the host-side caption-dropout check.

LOG = logging.getLogger("test_moss_pipeline")


class _Accel:
    device = torch.device("cpu")

    def autocast(self):
        return contextlib.nullcontext()

    def unwrap_model(self, model):
        return model


def _train_args(**kw):
    import moss_train_network

    args = moss_train_network.setup_parser().parse_args([])
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def _trainer(args, network=None, algo=""):
    import moss_train_network

    trainer = moss_train_network.MossNetworkTrainer()
    trainer._sigma_sampler = flow.MossSigmaSampler(args)
    if network is not None:  # what train_network.train sets up with the LyCORIS fixed paths on
        from library import lycoris_algo_fixes

        trainer.network = network
        trainer._lycoris_fix_on, trainer._lycoris_fixes, trainer._lycoris_fix_algo = True, lycoris_algo_fixes, algo
    return trainer


def _train_inputs(b=2, seed=7):
    g = torch.Generator().manual_seed(seed)
    latents = torch.randn(b, 128, 2, 3, generator=g)
    text = torch.randn(b, 5, TINY["text_dim"], generator=g)
    mask = torch.ones(b, 5, dtype=torch.bool)
    mask[-1, 3:] = False
    return latents, text, mask


def _step(trainer, args, model, latents, text, mask, net=None):
    """One micro-step: forward through get_noise_pred_and_target, backward; with LyCORIS the timestep is kept until after
    the backward (train_network sets _lycoris_fix_defer_clear around the chunk, _lycoris_fix_after_backward clears)."""
    trainer._lycoris_fix_defer_clear = net is not None
    out = trainer.get_noise_pred_and_target(args, _Accel(), trainer.get_noise_scheduler(args, "cpu"), latents, {},
                                            [text, mask], None, model, net, torch.float32, True)
    (out[0].float() - out[1].float()).square().mean().backward()
    if net is not None:
        trainer._lycoris_fix_after_backward(net)
    return out


# algo: (network_dim, network_alpha, network_args); TSM-LoKr as config_moss.toml's LoKr (full matrix, DoRA on the output)
TIMESTEP_ALGOS = {
    "tsm": (4, 4, {"tsm_n_scales": "4,1"}),
    "tlora": (8, 8, {}),
    "tsm-lokr": (10000, 10000, {"factor": 4, "full_matrix": True, "dora_wd": True, "wd_on_output": True,
                                "tsm_n_scales": "4,1"}),
}


@contextlib.contextmanager
def _fixed_paths():
    """The LyCORIS fixed paths for one test, in the production order (prepare_moss_runtime installs the fast LoKr forward,
    train() then patches LyCORIS); reverted afterwards unless they were on already."""
    from library import lycoris_algo_fixes as laf
    from library import lycoris_fast
    from lycoris.modules.tlora import clear_timestep_mask
    from lycoris.modules.tsm import clear_tsm_timestep

    was_on = bool(laf._PATCHES)
    lycoris_fast.install_fast_lokr_forward()
    laf.apply_module_patches()
    register_moss_lycoris_presets()
    try:
        yield laf
    finally:
        clear_timestep_mask()
        clear_tsm_timestep()
        laf._TSM_SAMPLE_T.clear()
        if not was_on:
            laf._revert_module_patches()


def _moss_network(laf, algo, **network_args):
    """(tiny DiT, LyCORIS network with preset moss on it), made as train_network makes it with the fixed paths on."""
    from lycoris import kohya

    dim, alpha, base = TIMESTEP_ALGOS[algo]
    raw = {"algo": algo, **{k: str(v) for k, v in {**base, **network_args}.items()}}
    model = _tiny_dit()
    net = kohya.create_network(1.0, dim, alpha, None, [], model, preset="moss", **laf.prepare_network_kwargs(algo, raw, LOG))
    net.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    laf.after_network_created(net, argparse.Namespace(network_args=[f"{k}={v}" for k, v in raw.items()], network_dim=dim,
                                                      network_alpha=alpha, lycoris_algo_fixes=True), LOG)
    net.prepare_grad_etc(None, model)
    return model, net


def _is_tlora(laf, net):
    return any(type(m).__name__ == "TLoraModule" for m in laf._all_loras(net))


def _timestep_state(laf, net):
    """What the adapters read in a forward: T-LoRA's rank mask, or TSM's (timestep, per-sample timesteps)."""
    from lycoris.modules.tlora import get_timestep_mask
    from lycoris.modules.tsm import get_tsm_timestep

    if _is_tlora(laf, net):
        mask = get_timestep_mask(0)
        return None if mask is None else mask.detach().clone()
    return get_tsm_timestep(0), list(laf._TSM_SAMPLE_T.get(0, []))


def _no_timestep(state):
    return state is None or (isinstance(state, tuple) and state == (None, []))


def _expected_state(laf, net, timesteps):
    """The state lycoris_algo_fixes.set_timestep makes from the DiT's timestep argument taken as [0, 1000]."""
    t = timesteps.detach().float().reshape(-1)
    if _is_tlora(laf, net):
        from lycoris.modules.tlora import compute_timestep_mask, compute_timestep_mask_batch

        cfg = laf._network_cfg(net)["tlora"]
        if t.numel() == 1:
            return compute_timestep_mask(int(t.item()), 1000, cfg["max_rank"], cfg["min_rank"], cfg["alpha"])
        return compute_timestep_mask_batch(t, 1000, cfg["max_rank"], cfg["min_rank"], cfg["alpha"])
    T = int(laf._all_loras(net)[0].tsm_num_timesteps)
    ints = [max(0, min(T, int(round(v * T / 1000.0)))) for v in t.tolist()]
    return ints[0], (ints if len(ints) > 1 else [])


def _same_state(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b.to(a.dtype))
    return a == b


def test_training_step_gives_tsm_tlora_tsmlokr_their_timesteps():
    """Training: every adapter of TSM / T-LoRA / TSM-LoKr (preset moss) sees the timesteps the DiT is called with
    (1000 * sigma, the [0, 1000] scale of the masks and expert intervals), per sample, in the forward and in gradient
    checkpointing's recompute; after the backward nothing is left set."""
    for algo in TIMESTEP_ALGOS:
        for gc in (False, True):
            with _fixed_paths() as laf:
                model, net = _moss_network(laf, algo)
                if gc:
                    model.enable_gradient_checkpointing()
                args = _train_args(timestep_sampling="uniform", gradient_checkpointing=gc)
                trainer = _trainer(args, net, algo)
                latents, text, mask = _train_inputs()
                ts, states = [], []
                hooks = [model.register_forward_pre_hook(lambda m, a, k: ts.append(a[1].detach().clone()), with_kwargs=True),
                         # an adapted Linear inside block 0: also runs in the checkpoint recompute (block._forward)
                         model.layers[0].proj.q.register_forward_pre_hook(
                             lambda m, a: states.append(_timestep_state(laf, net)))]
                torch.manual_seed(5)
                _step(trainer, args, model, latents, text, mask, net)
                for h in hooks:
                    h.remove()
                assert len(ts) == 1 and len(states) == (2 if gc else 1), (algo, gc, len(states))
                want = _expected_state(laf, net, ts[0])
                assert all(_same_state(s, want) for s in states), (algo, gc, states, want)
                assert _no_timestep(_timestep_state(laf, net)), algo


def test_sample_images_feed_every_sampler_step_timestep():
    """Sample images: moss_sample calls dit(x, 1000 * sigma, ...) (cond, and the skip-layer uncond branch); the sampling
    hook now hands each call's timestep to TSM / T-LoRA / TSM-LoKr (MossNetworkTrainer._lycoris_fix_sampler_timestep_scale
    = 1). The base answer (None, what MOSS inherited) ran every step with no timestep: TSM's highest-noise experts, the
    full T-LoRA rank. Nothing stays set afterwards."""
    from library.moss import train_utils as moss_train_utils

    steps, shift = 4, 3.0
    want_t = [float(v) for v in (moss_train_utils.moss_inference_sigmas(steps, shift)[:-1] * 1000.0) for _ in (0, 1)]
    for algo in TIMESTEP_ALGOS:
        with _fixed_paths() as laf:
            model, net = _moss_network(laf, algo)
            args = _train_args()
            trainer = _trainer(args, net, algo)
            _, text, mask = _train_inputs()

            def sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
                return moss_train_utils.moss_sample(unet, (text[:1], mask[:1]), (text[1:], mask[1:]), 32, 48, steps, 2.0,
                                                    shift, seed=0, device="cpu", skip_layers=(1,))

            assert trainer._lycoris_fix_sampler_timestep_scale() == 1.0
            for inherited in (False, True):
                if inherited:
                    trainer._lycoris_fix_sampler_timestep_scale = lambda: None
                calls = []
                hooks = [model.register_forward_pre_hook(lambda m, a, k: calls.append([a[1].detach().clone()]),
                                                         with_kwargs=True),  # before the wrapper's hook: the raw t
                         model.layers[0].proj.q.register_forward_pre_hook(
                             lambda m, a: calls[-1].append(_timestep_state(laf, net)))]
                trainer._lycoris_fix_wrap_sampling(sample_images)(_Accel(), args, 0, 1, "cpu", None, None, None, model)
                for h in hooks:
                    h.remove()
                assert [float(t) for t, _ in calls] == want_t, (algo, calls)
                if inherited:
                    assert all(_no_timestep(state) for _, state in calls), algo
                else:
                    assert all(_same_state(state, _expected_state(laf, net, t)) for t, state in calls), algo
                assert _no_timestep(_timestep_state(laf, net)), algo


def test_tsm_lokr_grouped_accumulation_on_moss():
    """tsm_grouped_accumulation=True: the base trainer refuses it, MOSS attaches it (_tsm_grouped_supported) and draws all
    samples of an optimizer step from one core interval (grouped_draw around flow.get_noisy_model_input_and_timesteps;
    each row of the noisy input stays its accepted draw's). Off, an optimizer step's samples spread over the intervals."""
    import train_network

    assert not train_network.NetworkTrainer()._tsm_grouped_supported
    for grouped in (True, False):
        with _fixed_paths() as laf:
            model, net = _moss_network(laf, "tsm-lokr", tsm_grouped_accumulation=grouped,
                                       tsm_resident_cores=2 if grouped else 4)
            res = net.prepare_optimizer_params(text_encoder_lr=None, unet_lr=1e-2, learning_rate=1e-2)
            opt = torch.optim.AdamW(res[0] if isinstance(res, tuple) else res, lr=1e-2, fused=True)
            accum = argparse.Namespace(gradient_accumulation_steps=2)
            args = _train_args(timestep_sampling="uniform")
            trainer = _trainer(args, net, "tsm-lokr")
            assert trainer._tsm_grouped_supported
            if grouped:
                try:
                    laf.attach_tsm_offload(net, opt, accum, torch.device("cpu"), batch_size=2, grouped_supported=False)
                    raise AssertionError("grouped accumulation attached for a trainer without it")
                except ValueError as e:
                    assert "moss_train_network.py" in str(e)
            mgr = laf.attach_tsm_offload(net, opt, accum, torch.device("cpu"), batch_size=2,
                                         grouped_supported=trainer._tsm_grouped_supported)
            assert mgr is not None and mgr.grouped is grouped
            trainer._tsm_offload = mgr
            latents, text, mask = _train_inputs()
            seen = []
            hook = model.register_forward_pre_hook(lambda m, a, k: seen.append((a[0].detach().clone(), a[1].detach().clone())),
                                                   with_kwargs=True)
            torch.manual_seed(11)
            cores_per_step = []
            for _ in range(6):
                seen.clear()
                for _ in range(accum.gradient_accumulation_steps):
                    _, _, sigmas, _, noise = _step(trainer, args, model, latents, text, mask, net)
                    s = sigmas.view(-1, 1, 1, 1)
                    torch.testing.assert_close(seen[-1][0], (1 - s) * latents + s * noise.float())  # rows of one draw
                    torch.testing.assert_close(seen[-1][1], sigmas * 1000)
                opt.step()
                opt.zero_grad(set_to_none=True)
                cores_per_step.append({mgr.core_index_t(v) for _, t in seen for v in t.tolist()})
            hook.remove()
            if grouped:
                assert all(len(c) == 1 for c in cores_per_step), cores_per_step
            else:
                assert any(len(c) > 1 for c in cores_per_step), cores_per_step


def test_flow_use_ot_reorders_the_noise_like_anima():
    """--flow_use_ot (anima_train_network.get_noise_pred_and_target): the batch's noise rows reassigned to the latents by
    cosine optimal transport (train_util.cosine_optimal_transport; fp32 cost - cached latents may be bf16), before the
    antithetic pairing; the target uses the reordered noise, and the assignment is the cheapest permutation. Off (the
    default) the noise is the plain draw; a batch of one is never reordered."""
    import itertools

    from library import train_util

    g = torch.Generator().manual_seed(3)
    latents = torch.randn(4, 128, 2, 3, generator=g).bfloat16()
    text = torch.randn(4, 5, TINY["text_dim"], generator=g)
    mask = torch.ones(4, 5, dtype=torch.bool)
    model = _tiny_dit()
    for ot in (False, True):
        for b in (4, 1):
            args = _train_args(flow_use_ot=ot)
            trainer = _trainer(args)
            torch.manual_seed(9)
            _, target, _, _, noise = trainer.get_noise_pred_and_target(
                args, _Accel(), trainer.get_noise_scheduler(args, "cpu"), latents[:b], {}, [text[:b], mask[:b]], None,
                model, None, torch.float32, True)
            torch.manual_seed(9)
            plain = torch.randn_like(latents[:b], dtype=torch.float32)
            assert torch.equal(target, noise - latents[:b].double())
            if not ot or b == 1:
                assert torch.equal(noise, plain.double()), (ot, b)
                continue
            _, (_, cols) = train_util.cosine_optimal_transport(latents.reshape(4, -1).float(), plain.reshape(4, -1))
            assert torch.equal(noise, plain[cols.squeeze(0)].double()) and not torch.equal(noise, plain.double())
            x = torch.nn.functional.normalize(latents.reshape(4, -1).float(), dim=1)
            y = torch.nn.functional.normalize(plain.reshape(4, -1), dim=1)
            cost = -(x @ y.t())
            best = min(itertools.permutations(range(4)), key=lambda p: float(sum(cost[i, p[i]] for i in range(4))))
            assert list(best) == cols.squeeze(0).tolist()


def test_adv_term_is_live_on_moss():
    """adv_scale > 0 (the stationary RFF-MMD term, train_network._apply_adv_term): MOSS stores the noisy input it fed the
    DiT (it never did: the term warned once and stayed INERT), and the reference cache keys disk-cached latents by MOSS's
    stride 16 (a 48x32 px bucket is a 2x3 latent grid; the stride-8 key 4x6 made build_adv_references refuse it)."""
    import train_network
    from library import adv_cache

    args = _train_args(adv_scale=1.0, adv_levels=1, adv_tokens_per_band=4, adv_reference_samples=16, adv_features=8)
    trainer = _trainer(args)
    trainer.setup_adv_objective(args)
    assert (trainer.adv_prediction_mode, trainer.adv_timesteps_in_sigma) == ("flow", True)

    # one image cached to disk by MossLatentsCachingStrategy (bucket 48x32 px -> 2x3 latents) with 2 augmentation variants
    latent_cache = strategy.MossLatentsCachingStrategy(True, 1, False)
    subset = types.SimpleNamespace(num_repeats=1, flip_aug=False, is_val=False, is_reg=False)
    with tempfile.TemporaryDirectory() as tmp:
        npz = os.path.join(tmp, "a_0048x0032" + latent_cache.cache_suffix)
        g = torch.Generator().manual_seed(1)
        latent_cache.save_latents_to_disk(npz, torch.randn(128, 2, 3, generator=g), [48, 32], [0, 0, 48, 32],
                                          key_reso_suffix="_2x3", aug_variants=[
                                              {"latents": torch.randn(128, 2, 3, generator=g), "crop_ltrb": [0, 0, 48, 32]}
                                              for _ in range(2)])
        info = types.SimpleNamespace(latents=None, latents_npz=npz, bucket_reso=(48, 32), is_val=False, is_reg=False)
        dataset = types.SimpleNamespace(image_data={"a": info}, image_to_subset={"a": subset},
                                        latents_caching_strategy=latent_cache, get_effective_aug_variant_count=lambda s: 3)
        group = types.SimpleNamespace(datasets=[dataset])
        assert trainer.latent_stride == 16 and train_network.NetworkTrainer().latent_stride == 8
        sources = trainer._adv_cache_sources(group, trainer.latent_stride)  # as prepare_adv_reference calls it
        assert [s.adv_grid_key for s in sources] == ["2x3"] * 3
        assert "self._adv_cache_sources(train_dataset_group, self.latent_stride)" in inspect.getsource(
            train_network.NetworkTrainer.prepare_adv_reference)
        build = dict(levels=1, token_budget=4, reference_samples=16, features=8, bandwidths="", seed=0, device="cpu")
        trainer.adv_references = adv_cache.build_adv_references(sources, **build)
        assert list(trainer.adv_references) == ["2x3"]
        stride8 = train_network.NetworkTrainer._adv_cache_sources(group)  # the key MOSS had: 4x6
        assert [s.adv_grid_key for s in stride8] == ["4x6"] * 3
        try:
            adv_cache.build_adv_references(stride8, **build)
            raise AssertionError("a 2x3 latent was accepted under a 4x6 grid key")
        except ValueError as e:
            assert "inconsistent" in str(e)
    model = _tiny_dit()
    latents = torch.randn(2, 128, 2, 3, generator=torch.Generator().manual_seed(4))
    _, text, mask = _train_inputs()
    seen = []
    hook = model.register_forward_pre_hook(lambda m, a, k: seen.append(a[0].detach().clone()), with_kwargs=True)
    sched = trainer.get_noise_scheduler(args, "cpu")
    torch.manual_seed(9)
    pred, _, sigmas, weighting, _ = trainer.get_noise_pred_and_target(args, _Accel(), sched, latents, {}, [text, mask],
                                                                      None, model, None, torch.float32, True)
    hook.remove()
    assert torch.equal(trainer._adv_noisy_latents, seen[0])
    base = torch.tensor(0.5)
    assert float(trainer._apply_adv_term(base, pred, latents, sigmas, weighting, sched, args, {})) > 0.5
    assert trainer.adv_loss_value is not None
    trainer._adv_noisy_latents = None  # MOSS before: nothing stored, the loss came back unchanged
    assert float(trainer._apply_adv_term(base, pred, latents, sigmas, weighting, sched, args, {})) == 0.5


def test_ramtorch_skips_blocks_left_out_of_the_lora():
    """--use_ramtorch streams only blocks that carry the LoRA: a block in --moss_lora_exclude_blocks keeps ordinary GPU
    Linears (before 10-27 RamTorch wrapped all of model.layers, so block 22's frozen weights went through RamTorch's own
    backward, which computes, stochastically rounds with the RNG and accumulates grads for them). RamTorch itself needs CUDA,
    so the wrapper is replaced by a recorder here; the selection is the point."""
    import moss_train_network

    for exclude, want in (("1", [0, 2]), ("", [0, 1, 2]), ("0, 2", [1])):
        model = _tiny_dit()
        before = list(model.layers)
        seen = []

        def record(module, name, device, dtype):
            seen.append(name)
            return module

        args = argparse.Namespace(moss_lora_exclude_blocks=exclude)
        got = moss_train_network.apply_ramtorch_to_moss_blocks(model, args, "cpu", torch.float32, apply=record)
        assert got == want, (exclude, got)
        assert seen == [f"MOSS joint block {i}" for i in want], (exclude, seen)
        assert all(a is b for a, b in zip(model.layers, before)), exclude  # the recorder hands every block back


def test_default_training_step_is_the_previous_one_bit_for_bit():
    """OT, adv, grouped accumulation and tier timing off (the defaults): get_noise_pred_and_target replays the previous
    step exactly (moss-image 09383988): the same RNG use, noise, timesteps, prediction, target and weighting."""
    from library import anima_train_utils
    from library.custom_train_functions import maybe_apply_antithetic_noise_pairing

    for sampling in ("moss", "uniform", "sigmoid"):
        args = _train_args(timestep_sampling=sampling)
        model = _tiny_dit()
        latents, text, mask = _train_inputs(b=3)
        trainer = _trainer(args)
        sched = trainer.get_noise_scheduler(args, "cpu")
        torch.manual_seed(21)
        got = trainer.get_noise_pred_and_target(args, _Accel(), sched, latents, {}, [text, mask], None, model, None,
                                                torch.float32, True)
        torch.manual_seed(21)
        noise = maybe_apply_antithetic_noise_pairing(args, torch.randn_like(latents, dtype=torch.float32), is_train=True)
        noisy, timesteps, sigmas = flow.get_noisy_model_input_and_timesteps(
            args, sched, latents, noise, torch.device("cpu"), torch.float32, trainer._sigma_sampler, is_train=True)
        want = (model(noisy, timesteps, text, mask), noise.double() - latents.double(), sigmas.flatten(),
                anima_train_utils.compute_loss_weighting_for_anima(args.weighting_scheme, sigmas), noise.double())
        assert all(torch.equal(a, b) for a, b in zip(got, want)), sampling


def test_caption_dropout_rate_is_checked_on_the_host():
    """drop_cached_text_encoder_outputs no longer reads the batch's rates back from the GPU every step (the .item() was a
    sync per batch): a cached caption_dropout_rate other than 0 is refused where the outputs are made (in-memory cache)
    or read (load_outputs_npz, in the DataLoader worker), and rates still on the CPU are checked as before."""
    encoding = strategy.MossTextEncodingStrategy()
    embeds, mask = torch.zeros(2, 3, 4), torch.ones(2, 3)
    assert encoding.drop_cached_text_encoder_outputs(embeds, mask, torch.zeros(2))[0] is embeds
    for rates in (torch.tensor([0.0, 0.1]), [0.0, 0.1]):
        try:
            encoding.drop_cached_text_encoder_outputs(embeds, mask, rates)
            raise AssertionError("a non-zero CPU rate was accepted")
        except RuntimeError as e:
            assert "cache_caption_variants" in str(e)
    # a rate on another device (accelerate moved the batch) is not read: a meta tensor has no value to read
    assert encoding.drop_cached_text_encoder_outputs(embeds, mask, torch.empty(2, device="meta"))[1] is mask
    model = _tiny_te()
    provider = strategy.MossTextEncoderProvider("unused", "cpu", loader=lambda *a, **k: model)
    with tempfile.TemporaryDirectory() as tmp:
        for to_disk in (True, False):
            caching = strategy.MossTextEncoderOutputsCachingStrategy(to_disk, 2, False, signature="sig")
            for rate in (0.0, 0.25):
                info = _Info(os.path.join(tmp, f"{int(to_disk)}_{rate}.png"), "a caption")
                info.caption_dropout_rate = rate
                info.text_encoder_outputs_npz = caching.get_outputs_npz_path(info.absolute_path) if to_disk else None
                try:
                    caching.cache_batch_outputs(_fake_tokenize_strategy(), [provider], encoding, [info])
                    if to_disk:
                        caching.load_outputs_npz(info.text_encoder_outputs_npz)
                    refused = False
                except RuntimeError as e:
                    assert "cache_caption_variants" in str(e)
                    refused = True
                assert refused == (rate > 0), (to_disk, rate)


if __name__ == "__main__":
    import sys
    import traceback

    failed = 0
    for name, fn in [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]:
        skips = (Skip,) + ((pytest.skip.Exception,) if pytest is not None else ())
        try:
            fn()
            print(f"PASS {name}")
        except skips as e:
            print(f"SKIP {name}: {e}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print("failed:", failed)
    sys.exit(1 if failed else 0)
