"""CPU tests of the MOSS text-encoder caching move (docs/moss_full_finetune_design.md 4.10): the two caching methods of
MossNetworkTrainer live in library/moss/te_cache.py (moss_train.py calls them too) and the trainer's methods are thin
calls into it, behavior-preserving; moss_train_network.py refuses LyCORIS algo=full with a pointer to moss_train.py.

The caching steps are recorded with fakes (dataset, provider, VAE, accelerator, the caption hooks): the same events in
the same order through the trainer method and through te_cache, also when the dataset caching raises.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_te_cache.py -q -p no:cacheprovider
"""
import argparse
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

from library import strategy_base
from library.moss import captions


class _Events(list):
    def hook(self, name, result=None):
        def record(*args, **kwargs):
            self.append(name)
            return result
        return record


class _Vae:
    def __init__(self, events, device="cuda:0"):
        self.events, self.device = events, torch.device(device)

    def to(self, device):
        self.events.append(f"vae.to({device})")
        self.device = torch.device(device) if isinstance(device, str) else device
        return self


class _Provider:
    def __init__(self, events, loaded=True):
        self.events, self.loaded = events, loaded

    def release(self):
        self.events.append("provider.release")
        self.loaded = False


class _Dataset:
    def __init__(self, events, fail=False):
        self.events, self.fail = events, fail

    def new_cache_text_encoder_outputs(self, models, accelerator):
        self.events.append("dataset.cache")
        if self.fail:
            raise RuntimeError("caching failed")


class _Accel:
    def __init__(self, events):
        self.events, self.device = events, torch.device("cpu")

    def wait_for_everyone(self):
        self.events.append("wait_for_everyone")


def _patch_hooks(monkeypatch, events):
    import moss_train_network as mtn

    for name in ("install_canonical_captions", "prepare_canonical_captions", "register_caption_digests",
                 "uninstall_canonical_captions"):
        monkeypatch.setattr(captions, name, events.hook(name, 0))
    monkeypatch.setattr(strategy_base.TextEncoderOutputsCachingStrategy, "_strategy", object())
    monkeypatch.setattr(mtn, "clean_memory_on_device", events.hook("clean_memory"))
    try:
        from library.moss import te_cache
    except ImportError:
        return
    monkeypatch.setattr(te_cache, "clean_memory_on_device", events.hook("clean_memory"))


EXPECTED = ["install_canonical_captions", "prepare_canonical_captions", "register_caption_digests", "vae.to(cpu)",
            "clean_memory", "dataset.cache", "sample_prompts", "wait_for_everyone", "uninstall_canonical_captions",
            "provider.release", "clean_memory", "vae.to(cuda:0)"]


def _args(**kw):
    values = dict(lowram=False, sample_prompts="prompts.txt")
    values.update(kw)
    return argparse.Namespace(**values)


def _run_trainer(monkeypatch, events, args, dataset, provider, vae, cached=None):
    import moss_train_network as mtn

    trainer = mtn.MossNetworkTrainer()
    trainer._te_provider = provider
    trainer.sample_prompts_te_outputs = cached
    trainer._cache_sample_prompts = lambda a, p: (events.append("sample_prompts"), {"p": [p is provider]})[1]
    trainer.cache_text_encoder_outputs_if_needed(args, _Accel(events), None, vae, [], dataset, torch.bfloat16)
    return trainer.sample_prompts_te_outputs


def test_trainer_caching_steps_in_order(monkeypatch):
    """Characterization of MossNetworkTrainer.cache_text_encoder_outputs_if_needed (before and after the move)."""
    events = _Events()
    _patch_hooks(monkeypatch, events)
    out = _run_trainer(monkeypatch, events, _args(), _Dataset(events), _Provider(events), _Vae(events))
    assert events == EXPECTED
    assert out == {"p": [True]}


def test_trainer_keeps_cached_sample_prompts_and_lowram(monkeypatch):
    events = _Events()
    _patch_hooks(monkeypatch, events)
    cached = {"already": []}
    out = _run_trainer(monkeypatch, events, _args(lowram=True), _Dataset(events), _Provider(events, loaded=False),
                       _Vae(events), cached=cached)
    assert out is cached
    assert events == ["install_canonical_captions", "prepare_canonical_captions", "register_caption_digests",
                      "dataset.cache", "wait_for_everyone", "uninstall_canonical_captions", "clean_memory"]


def test_trainer_cleans_up_when_caching_fails(monkeypatch):
    events = _Events()
    _patch_hooks(monkeypatch, events)
    with pytest.raises(RuntimeError, match="caching failed"):
        _run_trainer(monkeypatch, events, _args(), _Dataset(events, fail=True), _Provider(events), _Vae(events))
    assert events == ["install_canonical_captions", "prepare_canonical_captions", "register_caption_digests",
                      "vae.to(cpu)", "clean_memory", "dataset.cache", "uninstall_canonical_captions",
                      "provider.release", "clean_memory", "vae.to(cuda:0)"]


def test_te_cache_function_runs_the_same_steps(monkeypatch):
    from library.moss import te_cache

    events = _Events()
    _patch_hooks(monkeypatch, events)
    provider = _Provider(events)
    seen = []

    def sample(args, p):
        events.append("sample_prompts")
        seen.append(p)
        return {"q": []}

    out = te_cache.cache_moss_text_encoder_outputs(_args(), _Accel(events), _Vae(events), _Dataset(events), provider,
                                                   cache_sample_prompts=sample)
    assert events == EXPECTED and out == {"q": []} and seen == [provider]
    # already cached: returned as is, not encoded again; no sample prompts: None
    events.clear()
    again = te_cache.cache_moss_text_encoder_outputs(_args(), _Accel(events), _Vae(events), _Dataset(events), provider,
                                                     out, cache_sample_prompts=sample)
    assert again is out and "sample_prompts" not in events
    events.clear()
    none = te_cache.cache_moss_text_encoder_outputs(_args(sample_prompts=None), _Accel(events), _Vae(events),
                                                    _Dataset(events), provider, cache_sample_prompts=sample)
    assert none is None and "sample_prompts" not in events


def test_te_cache_function_without_a_vae(monkeypatch):
    """moss_train.py caches the text-encoder outputs with the VAE already back on the CPU (or released)."""
    from library.moss import te_cache

    events = _Events()
    _patch_hooks(monkeypatch, events)
    out = te_cache.cache_moss_text_encoder_outputs(_args(sample_prompts=None), _Accel(events), None, _Dataset(events),
                                                   _Provider(events))
    assert out is None
    assert events == ["install_canonical_captions", "prepare_canonical_captions", "register_caption_digests",
                      "clean_memory", "dataset.cache", "wait_for_everyone", "uninstall_canonical_captions",
                      "provider.release", "clean_memory"]


class _Tokenize:
    def tokenize(self, text):
        return [torch.tensor([[len(text)]]), torch.ones(1, 1, dtype=torch.long)]


class _Encode:
    def __init__(self):
        self.calls = []

    def encode_tokens(self, tokenize_strategy, models, tokens):
        self.calls.append(int(tokens[0]))
        n = int(tokens[0])
        return [torch.full((1, 2, 3), float(n)), torch.ones(1, 2, dtype=torch.long)]


def test_sample_prompt_cache_round_trip(tmp_path, monkeypatch):
    """cache_moss_sample_prompts (the moved _cache_sample_prompts): every prompt and negative prompt encoded once,
    written next to the prompt file under the TE signature, read back without the encoder the next time."""
    from library.moss import strategy, te_cache

    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a cat --n blurry\nsecond prompt\n", encoding="utf-8")
    monkeypatch.setattr(strategy, "moss_te_signature", lambda *a, **k: "sig")
    encode = _Encode()
    monkeypatch.setattr(strategy_base.TokenizeStrategy, "_strategy", _Tokenize())
    monkeypatch.setattr(strategy_base.TextEncodingStrategy, "_strategy", encode)
    args = argparse.Namespace(sample_prompts=str(prompts), moss_text_encoder="te.safetensors", moss_max_token_length=512,
                              moss_tokenizer_dir=None)
    first = te_cache.cache_moss_sample_prompts(args, provider=object())
    assert sorted(first) == ["", "a cat", "blurry", "second prompt"]  # "" = the second line's empty negative prompt
    assert sorted(encode.calls) == sorted(len(t) for t in first)
    assert all(torch.equal(first[t][0], torch.full((1, 2, 3), float(len(t)))) for t in first)
    cache_dir = tmp_path / "prompts_moss_te"
    assert len(list(cache_dir.iterdir())) == 4
    encode.calls.clear()
    second = te_cache.cache_moss_sample_prompts(args, provider=object())
    assert encode.calls == [] and sorted(second) == sorted(first)
    assert all(torch.equal(second[t][0], first[t][0]) and torch.equal(second[t][1], first[t][1]) for t in first)


def test_trainer_sample_prompt_method_delegates(monkeypatch):
    import moss_train_network as mtn
    from library.moss import te_cache

    calls = []
    monkeypatch.setattr(te_cache, "cache_moss_sample_prompts", lambda args, provider: calls.append((args, provider)) or {})
    trainer = mtn.MossNetworkTrainer()
    assert trainer._cache_sample_prompts("ARGS", "PROVIDER") == {} and calls == [("ARGS", "PROVIDER")]


# ---- moss_train_network.py refuses LyCORIS algo=full (design 0 / 4.10) ---------------------------------------------


@pytest.mark.parametrize("network_args", [["algo=full", "preset=moss"], ["preset=moss", "algo = Full "],
                                          ['algo="full"']])
def test_lycoris_full_is_refused_with_a_pointer(monkeypatch, network_args):
    import moss_train_network as mtn

    monkeypatch.setattr(mtn.moss_train_utils, "validate_moss_args",
                        lambda *a, **k: pytest.fail("checked after the algo=full refusal"))
    args = argparse.Namespace(network_module="lycoris.kohya", network_args=network_args, moss_text_encoder="te")
    with pytest.raises(ValueError) as error:
        mtn.MossNetworkTrainer().assert_extra_args(args, types.SimpleNamespace(datasets=[]), None)
    text = str(error.value)
    assert "moss_train.py" in text and "71 GiB" in text and "algo=full" in text


@pytest.mark.parametrize("module,network_args", [("lycoris.kohya", ["algo=lokr", "preset=moss"]),
                                                 ("networks.lora_moss", ["algo=full"]),
                                                 ("lycoris.kohya", None)])
def test_other_networks_pass_the_algo_check(monkeypatch, module, network_args):
    import moss_train_network as mtn

    class Reached(Exception):
        pass

    def reached(*a, **k):
        raise Reached()

    monkeypatch.setattr(mtn.moss_train_utils, "validate_moss_args", reached)
    args = argparse.Namespace(network_module=module, network_args=network_args, moss_text_encoder="te")
    with pytest.raises(Reached):
        mtn.MossNetworkTrainer().assert_extra_args(args, types.SimpleNamespace(datasets=[]), None)


def test_namespace_without_network_fields_passes_the_algo_check(monkeypatch):
    """Existing tests call assert_extra_args with a Namespace that has no network_module / network_args."""
    import moss_train_network as mtn

    class Reached(Exception):
        pass

    monkeypatch.setattr(mtn.moss_train_utils, "validate_moss_args", lambda *a, **k: (_ for _ in ()).throw(Reached()))
    with pytest.raises(Reached):
        mtn.MossNetworkTrainer().assert_extra_args(argparse.Namespace(moss_text_encoder=None),
                                                  types.SimpleNamespace(datasets=[]), None)
