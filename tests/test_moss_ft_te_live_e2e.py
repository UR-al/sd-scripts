"""End-to-end CPU test of moss_train.py with --moss_te_live (docs/moss_ft_te_live_fp16_plan.md 1.6): the harness of
tests/test_moss_ft_train_e2e.py (tiny DiT, random FLUX.2 VAE, dataset TOML of 5 images, real strategies), with a tiny
Qwen3.5-like text encoder written as a real safetensors file that BOTH paths read:
* the cached path through qwen35.load_moss_text_encoder (the provider's loader, given the tiny config);
* the live path through te_live_weights.HostTextEncoderWeights (DEFAULT_CONFIG patched to the tiny config, the 9B
  header check to the prefix detection).

Checks: a cached run and live runs (ahead 1, and ahead 3 -> 4 with GA 2) log the same losses / grad norms at every step
and save the same DiT bit for bit; a live run with shuffle_caption, keep_tokens_separator, enable_wildcard,
caption_dropout_rate and caption_tag_dropout_rate trains, logs te/* and resumes mid-epoch; the sample prompts encoded
through the live encoder equal the provider path's.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_te_live_e2e.py -q -p no:cacheprovider
"""
import math
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

import accelerate
import moss_train
import test_moss_ft_train_e2e as E
from _moss_te_tiny import TINY_TE, CharTokenizer, live_encoder, write_tiny_te
from library import strategy_base
from library.moss import ft_model, qwen35, strategy, te_cache, te_live_weights
from library.moss import train_utils as moss_train_utils

RUN = ("--mixed_precision", "no", "--gradient_accumulation_steps", str(E.GA), "--optimizer_type", "AdamW",
       "--optimizer_args", "betas=(0.9,0.999)", "weight_decay=0.01", "--moss_ft_lr_phases", E.PHASES,
       "--lr_warmup_steps", str(E.WARMUP), "--max_grad_norm", "1.0")
CACHED = ("--cache_text_encoder_outputs", "--cache_text_encoder_outputs_to_disk")
STRUCTURED = ["1girl, alice ||| red dress, blue eyes, smile ||| She stands in a garden.",
              "cat ||| sitting, window, shadows\ncat ||| sitting",
              "a red square", "two blue circles ||| table, white ||| A still life.", "green"]


def tiny_te_loader(path, device="cpu", dtype=torch.bfloat16):
    return qwen35.load_moss_text_encoder(path, device=device, dtype=dtype, config=TINY_TE)


def patch(mp, loaded=None, logs=None):
    mp.setattr(ft_model, "MOSS_DIT_CONFIG", dict(E.TINY))
    mp.setattr(moss_train, "load_moss_vae", E.tiny_vae)
    mp.setattr(strategy, "load_moss_text_encoder", tiny_te_loader)
    mp.setattr(strategy, "MossTokenizer", CharTokenizer)
    mp.setattr(strategy, "check_moss_text_encoder_path", lambda path: None)
    mp.setattr(te_live_weights, "DEFAULT_CONFIG", TINY_TE)
    mp.setattr(te_live_weights, "check_moss_text_encoder", qwen35.detect_text_encoder_prefix)
    if loaded is not None:
        real = ft_model.load_moss_for_full_ft

        def capture(*args, **kwargs):
            out = real(*args, **kwargs)
            loaded.append(out)
            return out

        mp.setattr(ft_model, "load_moss_for_full_ft", capture)
    if logs is not None:
        mp.setattr(accelerate.Accelerator, "log", lambda self, values, step=None, log_kwargs=None:
                   logs.append((step, dict(values))))


def make_env(root, captions=None, subset_lines=()):
    env = E.write_files(root)
    write_tiny_te(env.te)  # a real (tiny) encoder file both paths read
    if captions is not None:
        for index, caption in enumerate(captions):
            (env.images / f"img{index}.txt").write_text(caption, encoding="utf-8")
    if subset_lines:
        text = env.dataset.read_text(encoding="utf-8") + "".join(f"  {line}\n" for line in subset_lines)
        env.dataset.write_text(text, encoding="utf-8")
    return env


def run(env, out_dir, *extra, loaded=None, logs=None, sample=False):
    argv = ["--pretrained_model_name_or_path", str(env.base), "--vae", str(env.vae),
            "--moss_text_encoder", str(env.te), "--dataset_config", str(env.dataset),
            "--output_dir", str(out_dir), "--output_name", "tiny", "--seed", "1234",
            "--cache_latents", "--cache_latents_to_disk", "--max_data_loader_n_workers", "0",
            "--gradient_checkpointing", "--moss_ft_skip_resource_check",
            "--moss_ft_host_threads", "2", "--moss_ft_piece_mib", "16", "--moss_ft_ring_mib", "64",
            "--moss_ft_upload_ring_mib", "32", "--moss_ft_log_every", "2", *extra]
    if sample:
        argv += ["--sample_prompts", str(env.prompts), "--moss_sample_uncond_skip_layers", "1"]
    args = moss_train.setup_parser().parse_args(argv)
    with pytest.MonkeyPatch.context() as mp:
        patch(mp, loaded, logs)
        E.reset_process_singletons()
        try:
            moss_train.train(args)
        finally:
            E.reset_process_singletons()
    return args


def _same_dit(out_a, out_b):
    final_a, _ = E.read(out_a / "tiny.safetensors")
    final_b, _ = E.read(out_b / "tiny.safetensors")
    assert sorted(final_a) == sorted(final_b)
    for key in final_a:
        assert torch.equal(final_a[key], final_b[key]), key


@pytest.fixture(scope="module")
def plain_env(tmp_path_factory):
    return make_env(tmp_path_factory.mktemp("moss_te_live_e2e"))


@pytest.fixture(scope="module")
def cached_run(plain_env):
    logs = []
    out = plain_env.root / "cached"
    run(plain_env, out, *RUN, *CACHED, logs=logs)
    return types.SimpleNamespace(out=out, steps=E.by_step(logs))


@pytest.mark.parametrize("ahead", [1, 3])
def test_live_run_equals_the_cached_run(plain_env, cached_run, ahead):
    """The cached run wrote *_moss_te.npz through the real caching strategy; the live runs encode the same captions at
    train time. Every step's loss / grad norm / clip coefficient and the saved DiT are equal bit for bit."""
    for index in range(E.IMAGES):
        assert (plain_env.images / f"img{index}_moss_te.npz").is_file()
    logs = []
    out = plain_env.root / f"live_{ahead}"
    run(plain_env, out, *RUN, "--moss_te_live", "--moss_te_live_ahead", str(ahead), logs=logs)
    steps = E.by_step(logs)
    assert sorted(steps) == sorted(cached_run.steps) == list(range(1, 2 * E.STEPS_PER_EPOCH + 1))
    for step in steps:
        for key in ("loss", "grad_norm", "clip_coef", "lr/decay"):
            assert steps[step][key] == cached_run.steps[step][key], (ahead, step, key)
        assert steps[step]["te/pass_s"] >= 0.0 and "te/pass_s" not in cached_run.steps[step]
    _same_dit(cached_run.out, out)


@pytest.fixture(scope="module")
def caption_env(tmp_path_factory):
    return make_env(tmp_path_factory.mktemp("moss_te_live_captions"), STRUCTURED,
                    ['keep_tokens_separator = "|||"', "shuffle_caption = true", "enable_wildcard = true",
                     "caption_dropout_rate = 0.2", "caption_tag_dropout_rate = 0.2"])


@pytest.fixture(scope="module")
def caption_run(caption_env):
    loaded, logs = [], []
    out = caption_env.root / "live_captions"
    args = run(caption_env, out, *RUN, "--moss_te_live", "--moss_te_live_ahead", "2", "--save_state",
               "--save_every_n_steps", "1", "--save_last_n_steps_state", "100", loaded=loaded, logs=logs)
    return types.SimpleNamespace(out=out, args=args, steps=E.by_step(logs), loaded=loaded)


def test_live_run_with_random_captions_trains_and_logs(caption_env, caption_run):
    steps = caption_run.steps
    assert sorted(steps) == list(range(1, 2 * E.STEPS_PER_EPOCH + 1))
    for values in steps.values():
        assert math.isfinite(values["loss"]) and values["skipped_params"] == 0
        assert values["te/pass_s"] >= 0.0 and values["te/s_per_image"] >= 0.0
    assert not list(caption_env.images.glob("*_moss_te.npz"))  # live mode writes no TE cache
    model, pool, slots = caption_run.loaded[0]
    assert torch.equal(pool.steps, torch.full_like(pool.steps, 2.0 * E.STEPS_PER_EPOCH))


def test_live_run_resumes_mid_epoch(caption_env, caption_run):
    logs = []
    out = caption_env.root / "live_resumed"
    run(caption_env, out, *RUN, "--moss_te_live", "--moss_te_live_ahead", "2",
        "--resume", str(caption_run.out / "tiny-step00000002-state"), logs=logs)
    steps = E.by_step(logs)
    assert sorted(steps) == [3, 4, 5, 6]
    assert all(math.isfinite(v["loss"]) and "te/pass_s" in v for v in steps.values())
    assert (out / "tiny.safetensors").is_file()


def test_sample_prompts_through_the_live_encoder_equal_the_provider_path(plain_env, tmp_path, monkeypatch):
    for cls in (strategy_base.TokenizeStrategy, strategy_base.TextEncodingStrategy):
        monkeypatch.setattr(cls, "_strategy", None)
    monkeypatch.setattr(strategy, "MossTokenizer", CharTokenizer)
    strategy_base.TokenizeStrategy.set_strategy(strategy.MossTokenizeStrategy())
    strategy_base.TextEncodingStrategy.set_strategy(strategy.MossTextEncodingStrategy())
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(E.SAMPLE_PROMPT + "\nan empty room --n dark\n", encoding="utf-8")
    args = types.SimpleNamespace(sample_prompts=str(prompts), moss_text_encoder=str(plain_env.te),
                                 moss_max_token_length=4096, moss_tokenizer_dir=None)
    provider = strategy.MossTextEncoderProvider(str(plain_env.te), "cpu", loader=tiny_te_loader)
    from_provider = te_cache.cache_moss_sample_prompts(args, provider)
    cache_dir = moss_train_utils.sample_prompt_cache_dir(args)
    for path in list(os.scandir(cache_dir)):
        os.remove(path.path)  # encode again through the live encoder (not read from the files just written)
    encoder = live_encoder(plain_env.te)
    try:
        from_live = te_cache.cache_moss_sample_prompts(args, encoder)
    finally:
        encoder.close()
    assert sorted(from_live) == sorted(from_provider) and len(from_live) == 4
    for text in from_live:
        assert torch.equal(from_live[text][0], from_provider[text][0]) and torch.equal(from_live[text][1],
                                                                                       from_provider[text][1])


def test_live_run_with_a_dataloader_worker_equals_the_cached_run(plain_env, cached_run):
    """Review suggestion: one spawned DataLoader worker (Windows spawn: the dataset with its tokenizer pickles into the
    worker, the text encoder stays in the main process) and the live prefetch_factor (ceil(ahead 4 / 1 worker))."""
    logs = []
    out = plain_env.root / "live_worker"
    run(plain_env, out, *RUN, "--moss_te_live", "--moss_te_live_ahead", "3", "--max_data_loader_n_workers", "1",
        logs=logs)
    steps = E.by_step(logs)
    assert sorted(steps) == sorted(cached_run.steps)
    for step in steps:
        for key in ("loss", "grad_norm", "clip_coef", "lr/decay"):
            assert steps[step][key] == cached_run.steps[step][key], (step, key)
    assert sum(v["te/pass_images"] for v in steps.values()) == 2 * E.IMAGES  # every caption once per epoch
    _same_dit(cached_run.out, out)
