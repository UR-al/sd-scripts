"""CPU tests for the MOSS-Image trainer plumbing (sampler, LoRA/LyCORIS naming, TE caching). No model files needed.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe tests/test_moss_pipeline.py   (or with pytest if installed)
"""
import argparse
import contextlib
import io
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
