"""End-to-end CPU test of moss_train.py (docs/moss_full_finetune_design.md 4.3-4.9, test plan 5).

The script runs as a whole on the CPU: the dataset group from a dataset TOML (5 images of 32x32 with captions), the
latent and text-encoder caches written next to the images by the real MOSS caching strategies (the dataset's
new_cache_latents -> MossLatentsCachingStrategy -> moss_vae_encode, and te_cache ->
MossTextEncoderOutputsCachingStrategy -> MossTextEncodingStrategy), the sample-prompt cache, the DiT loaded from a
Base-layout file into the host pool, the host optimizer pipeline (threaded worker), the LR phase scheduler, sample
images, DiT saves, training states and resume.

Only the heavy models are stand-ins, injected by monkeypatching (the caches are made honestly by the real code paths):
* library.moss.ft_model.MOSS_DIT_CONFIG -> TINY: the DiT the loader builds (the Base-layout file is written in it);
* moss_train.load_moss_vae -> a randomly initialised diffusers AutoencoderKLFlux2 with 32 channels (the real VAE class:
  moss_vae_encode / moss_vae_decode run unchanged);
* library.moss.strategy.load_moss_text_encoder -> a 4-layer, width-64 MossQwen35TextEncoder (the real class) behind the
  real MossTextEncoderProvider;
* library.moss.strategy.MossTokenizer -> a character-level tokenizer with the chat template (no tokenizer.json here);
* library.moss.strategy.check_moss_text_encoder_path -> no-op (it reads the 9B file's header).

Observed through accelerate.Accelerator.log (every optimizer step's logs) and a wrapper of
ft_model.load_moss_for_full_ft (the run's model and host pool, compared after train() returns).

Shape of the run: GA 2 over 5 batches = windows [0, 1], [2, 3], [4] (the last one partial) = 3 optimizer steps per
epoch; --moss_ft_lr_phases "1:1e-3,1:2e-4" with --lr_warmup_steps 2 = 6 steps; max_grad_norm 1.0 (stale clipping, the
first window exact); fp32 compute (mixed_precision "no"), deterministic.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_train_e2e.py -q -p no:cacheprovider
"""
import json
import math
import os
import shutil
import types
from dataclasses import replace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

import accelerate
from accelerate.state import AcceleratorState, GradientState

import moss_train
from library import strategy_base
from library.moss import ft_model, ft_state, strategy
from library.moss.models import MossTransformer
from library.moss.qwen35 import MossQwen35Config, MossQwen35TextEncoder, format_moss_prompt, unescape_prompt

TINY = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=64, freq_dim=16, latent_channels=128)
IMAGES = 5
GA = 2
STEPS_PER_EPOCH = 3  # windows [0, 1], [2, 3], [4]
PHASES = "1:1e-3,1:2e-4"
WARMUP = 2
# lr of the window that ends at global step s = 1..6: 0-based step s - 1, min(1, s / 2) x LR of its phase (design 4.6)
EXPECTED_LR = [5e-4, 1e-3, 1e-3, 2e-4, 2e-4, 2e-4]
CAPTIONS = ["a red square", "two blue circles on a white table", "green", "a cat, sitting, looking at the viewer",
            "an empty room with a window and long shadows on the floor"]
SAMPLE_PROMPT = "a red square --n blurry --w 256 --h 256 --s 2 --l 2.0 --d 7"


# ---------------------------------------------------------------------------------------------------- stand-ins
class CharTokenizer:
    """MossTokenizer's interface (tokenizer_dir, max_length, encode, __call__, pad_token_id), character level, with
    the real chat template and ComfyUI unescaping."""

    pad_token_id = 248044

    def __init__(self, tokenizer_dir=None, max_length: int = 4096):
        self.tokenizer_dir = "<char tokenizer>"
        self.max_length = max_length

    def encode(self, prompt: str):
        return [ord(c) % 250 + 1 for c in format_moss_prompt(unescape_prompt(prompt))][: self.max_length]

    def __call__(self, prompts):
        return [self.encode(p) for p in prompts]


def tiny_text_encoder(path, device="cpu", dtype=torch.bfloat16):
    config = replace(MossQwen35Config(), hidden_size=TINY["text_dim"], intermediate_size=96, num_hidden_layers=4,
                     vocab_size=256)
    torch.manual_seed(11)
    model = MossQwen35TextEncoder(config)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.05)
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


def tiny_vae(path, device=None, dtype=torch.float32):
    from diffusers import AutoencoderKLFlux2

    torch.manual_seed(12)
    vae = AutoencoderKLFlux2(block_out_channels=(32, 32, 32, 32), decoder_block_out_channels=(32, 32, 32, 32),
                             layers_per_block=1, latent_channels=32, norm_num_groups=32, mid_block_add_attention=False,
                             sample_size=32)
    with torch.no_grad():
        vae.bn.running_mean.normal_(0, 0.1)
        vae.bn.running_var.uniform_(0.5, 1.5)
    return vae.to(device=device or torch.device("cpu"), dtype=dtype).eval().requires_grad_(False)


def patch_stand_ins(mp, loaded=None, logs=None):
    mp.setattr(ft_model, "MOSS_DIT_CONFIG", dict(TINY))
    mp.setattr(moss_train, "load_moss_vae", tiny_vae)
    mp.setattr(strategy, "load_moss_text_encoder", tiny_text_encoder)
    mp.setattr(strategy, "MossTokenizer", CharTokenizer)
    mp.setattr(strategy, "check_moss_text_encoder_path", lambda path: None)
    if loaded is not None:
        real = ft_model.load_moss_for_full_ft

        def capture(*args, **kwargs):
            out = real(*args, **kwargs)
            loaded.append(out)
            return out

        mp.setattr(ft_model, "load_moss_for_full_ft", capture)
    if logs is not None:
        def log(self, values, step=None, log_kwargs=None):
            logs.append((step, dict(values)))

        mp.setattr(accelerate.Accelerator, "log", log)


# ---------------------------------------------------------------------------------------------------- the files
def write_files(root):
    images = root / "images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for index, caption in enumerate(CAPTIONS):
        pixels = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
        pixels[: 8 + 4 * index, :, index % 3] = 255  # images differ in content, not only in noise
        Image.fromarray(pixels).save(images / f"img{index}.png")
        (images / f"img{index}.txt").write_text(caption, encoding="utf-8")
    dataset = root / "dataset.toml"
    dataset.write_text("[general]\nenable_bucket = false\ncaption_extension = \".txt\"\n\n"
                       "[[datasets]]\nresolution = 32\nbatch_size = 1\n\n"
                       f"  [[datasets.subsets]]\n  image_dir = '{images.as_posix()}'\n  num_repeats = 1\n",
                       encoding="utf-8")
    torch.manual_seed(5)
    model = MossTransformer(**TINY)
    g = torch.Generator().manual_seed(6)
    state = {}
    for name, tensor in model.state_dict().items():
        if name == "rope.bands":
            state[name] = tensor.float().clone()
        else:
            state[name] = (tensor.float() + torch.randn(tensor.shape, generator=g) * 0.05).to(torch.bfloat16)
    base = root / "base.safetensors"
    save_file({k: state[k].contiguous() for k in sorted(state)}, str(base), metadata={"format": "pt"})
    for name in ("vae.safetensors", "te.safetensors"):  # validate_moss_args wants the files, the TE signature reads
        save_file({"stand_in": torch.zeros(1)}, str(root / name))  # a header; the loaders are stand-ins
    prompts = root / "sample_prompts.txt"
    prompts.write_text(SAMPLE_PROMPT + "\n", encoding="utf-8")
    return types.SimpleNamespace(root=root, images=images, dataset=dataset, base=base, vae=root / "vae.safetensors",
                                 te=root / "te.safetensors", prompts=prompts, base_state=state)


def make_args(env, out_dir, *extra):
    argv = ["--pretrained_model_name_or_path", str(env.base), "--vae", str(env.vae),
            "--moss_text_encoder", str(env.te), "--dataset_config", str(env.dataset),
            "--output_dir", str(out_dir), "--output_name", "tiny", "--seed", "1234",
            "--cache_latents", "--cache_latents_to_disk", "--cache_text_encoder_outputs",
            "--cache_text_encoder_outputs_to_disk", "--max_data_loader_n_workers", "0",
            "--gradient_checkpointing", "--moss_ft_skip_resource_check",
            "--moss_ft_host_threads", "2", "--moss_ft_piece_mib", "16", "--moss_ft_ring_mib", "64",
            "--moss_ft_upload_ring_mib", "32", "--moss_ft_log_every", "2",
            "--sample_prompts", str(env.prompts), "--moss_sample_uncond_skip_layers", "1", *extra]
    return moss_train.setup_parser().parse_args(argv)


ADAMW_RUN = ("--mixed_precision", "no", "--gradient_accumulation_steps", str(GA), "--optimizer_type", "AdamW",
             "--optimizer_args", "betas=(0.9,0.999)", "weight_decay=0.01", "--moss_ft_lr_phases", PHASES,
             "--lr_warmup_steps", str(WARMUP), "--max_grad_norm", "1.0", "--save_state", "--sample_every_n_epochs", "1")
ADAMW = ADAMW_RUN + ("--save_every_n_epochs", "1", "--save_every_n_steps", "1", "--save_last_n_steps_state", "100")


def reset_process_singletons():
    """What a new process starts without: accelerate's shared state and kohya's strategy singletons (set_strategy
    refuses a second set)."""
    AcceleratorState._reset_state(True)
    GradientState._reset_state()
    for cls in (strategy_base.TokenizeStrategy, strategy_base.TextEncodingStrategy,
                strategy_base.TextEncoderOutputsCachingStrategy, strategy_base.LatentsCachingStrategy):
        cls._strategy = None


def run(env, out_dir, *extra, loaded=None, logs=None):
    """train(args) as a fresh process would run it: a new args object, the process-wide singletons reset."""
    args = make_args(env, out_dir, *extra)
    with pytest.MonkeyPatch.context() as mp:
        patch_stand_ins(mp, loaded, logs)
        reset_process_singletons()
        try:
            moss_train.train(args)
        finally:
            reset_process_singletons()
    return args


def by_step(logs):
    """{step: logs} of the per-optimizer-step records (they carry "loss")."""
    out = {}
    for step, values in logs:
        if "loss" in values:
            assert step not in out, f"step {step} logged twice"
            out[step] = values
    return out


def read(path):
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}, f.metadata()


def train_state(path):
    with open(os.path.join(path, ft_state.TRAIN_STATE_FILE), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------- run A
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return write_files(tmp_path_factory.mktemp("moss_ft_e2e"))


@pytest.fixture(scope="module")
def run_a(env):
    """The uninterrupted run: 2 epochs, every step's DiT and state saved, samples at the start and every epoch."""
    loaded, logs = [], []
    out = env.root / "run_a"
    args = run(env, out, *ADAMW, "--sample_at_first", loaded=loaded, logs=logs)
    assert len(loaded) == 1
    model, pool, slots = loaded[0]
    return types.SimpleNamespace(out=out, args=args, logs=logs, steps=by_step(logs), model=model, pool=pool,
                                 slots=slots)


def test_caches_were_written_by_the_real_strategies(env, run_a):
    for index in range(IMAGES):
        latents = env.images / f"img{index}_0032x0032_moss.npz"
        te = env.images / f"img{index}_moss_te.npz"
        assert latents.is_file() and te.is_file(), index
        with np.load(te) as npz:
            assert npz["text_embeds"].shape == (len(CharTokenizer().encode(CAPTIONS[index])), TINY["text_dim"])
    cache = env.prompts.with_name("sample_prompts_moss_te")
    assert len(list(cache.glob("*.safetensors"))) == 2  # the prompt and its negative


def test_every_step_is_logged_once_with_its_lr_and_a_finite_loss(run_a):
    assert sorted(run_a.steps) == list(range(1, 2 * STEPS_PER_EPOCH + 1))
    for step, values in run_a.steps.items():
        assert math.isfinite(values["loss"]) and values["loss"] > 0, (step, values)
        assert values["lr/decay"] == pytest.approx(EXPECTED_LR[step - 1], rel=1e-12), step
        assert values["lr/no_decay"] == pytest.approx(EXPECTED_LR[step - 1], rel=1e-12), step
        assert math.isfinite(values["grad_norm"]) and values["skipped_params"] == 0, (step, values)
        assert 0.0 < values["clip_coef"] <= 1.0
        for key in ("time/step", "time/tail", "time/hook_wait", "time/worker_busy", "time/data",
                    "throughput/tok_per_s", "throughput/img_per_s"):
            assert key in values and values[key] >= 0.0, (step, key)
    # host memory lines every --moss_ft_log_every (2) steps; no VRAM lines on the CPU
    assert all("host/rss_gib" in run_a.steps[s] for s in (2, 4, 6))
    assert not any(key.startswith("vram/") for values in run_a.steps.values() for key in values)
    epochs = [values for _, values in run_a.logs if "loss/epoch" in values]
    assert [values["epoch"] for values in epochs] == [1, 2]
    # every trained tensor was stepped once per window, also in the partial ones
    assert torch.equal(run_a.pool.steps, torch.full_like(run_a.pool.steps, 2.0 * STEPS_PER_EPOCH))


def test_the_dit_saves_have_the_base_layout(env, run_a):
    names = sorted(os.listdir(run_a.out))
    assert "tiny.safetensors" in names and "tiny-000001.safetensors" in names
    assert "tiny-000002.safetensors" not in names  # the last epoch is the final save (kohya)
    assert [f"tiny-step{s:08d}.safetensors" in names for s in range(1, 7)] == [True] * 6
    final, meta = read(run_a.out / "tiny.safetensors")
    want = sorted(run_a.model.state_dict())
    assert list(final) == want == sorted(env.base_state)
    # F32 like the Base file ([16] there: head_dim 128; this tiny config has head_dim 16)
    assert final["rope.bands"].dtype == torch.float32 and torch.equal(final["rope.bands"], run_a.model.rope.bands)
    assert all(t.dtype == torch.bfloat16 for k, t in final.items() if k != "rope.bands")
    assert "shared_block_modulation.weight" in final and "layers.0.proj.ffn_up_gate.weight" in final
    assert not any(k.startswith("layers.") and k.endswith(".qkv.weight") for k in final)  # joint q / k / v split
    assert meta["format"] == "pt" and meta["modelspec.architecture"] == "moss-image-base"
    assert meta["ss_moss_ft_global_step"] == "6" and meta["ss_moss_ft_mode"] == "adamw_host"
    for slot in run_a.slots:  # trained tensors: the final fp32 master rounded to bf16
        master = run_a.pool.view("master", slot).view(slot.shape)
        assert torch.equal(final[slot.name], master.to(torch.bfloat16)), slot.name
    moved = sum(not torch.equal(final[k], env.base_state[k]) for k in want if k != "rope.bands")
    assert moved == len(want) - 1  # every tensor was trained


def test_states_are_written_with_their_progress(run_a):
    out = run_a.out
    for step in range(1, 7):
        path = out / f"tiny-step{step:08d}-state"
        ft_state.verify_state(str(path))
        state = train_state(path)
        assert state["global_step"] == step and state["steps_per_epoch"] == STEPS_PER_EPOCH
        epoch, window = divmod(step, STEPS_PER_EPOCH)
        if window == 0:  # an epoch's last window: an epoch-end state
            assert (state["completed_epochs"], state["epoch_step"]) == (epoch, 0), step
        else:
            assert (state["completed_epochs"], state["epoch_step"]) == (epoch, 2 * window), step
    epoch_state = train_state(out / "tiny-000001-state")
    assert (epoch_state["global_step"], epoch_state["completed_epochs"], epoch_state["epoch_step"]) == (3, 1, 0)
    assert not (out / "tiny-000002-state").exists()  # the last epoch: the train-end state instead
    last = train_state(out / "tiny-state")
    assert (last["global_step"], last["completed_epochs"], last["epoch_step"]) == (6, 2, 0)
    assert last["lr_schedule"]["bounds"] == [3, 6] and last["lr_schedule"]["warmup_steps"] == WARMUP


def test_samples_are_written(run_a):
    names = sorted(os.listdir(run_a.out / "sample"))
    for tag in ("e000000", "e000001", "e000002"):
        assert any(f"_{tag}_" in name and name.endswith(".png") for name in names), (tag, names)


def _assert_same_training(a, b):
    assert [s.name for s in a.slots] == [s.name for s in b.slots]
    for role in a.pool.roles:
        assert torch.equal(a.pool.roles[role], b.pool.roles[role]), role
    assert torch.equal(a.pool.steps, b.pool.steps)
    for (name, p), (_, q) in zip(a.model.named_parameters(), b.model.named_parameters()):
        assert p.dtype == q.dtype and torch.equal(p.detach(), q.detach()), name


def test_resume_from_an_epoch_end_step_state_is_bit_identical(env, run_a):
    """Design 4.7: resuming from the step state at the end of epoch 1 (taken after the window's drain) continues the
    run exactly: the same batches (the epoch shuffle and the noise from the restored RNG), the same LR and clipping
    counters, so master, moments, step counters and compute copies end bit-identical, and so do the logs."""
    loaded, logs = [], []
    out = env.root / "run_b"
    run(env, out, *ADAMW, "--resume", str(run_a.out / "tiny-step00000003-state"), loaded=loaded, logs=logs)
    model, pool, slots = loaded[0]
    resumed = types.SimpleNamespace(model=model, pool=pool, slots=slots)
    steps = by_step(logs)
    assert sorted(steps) == [4, 5, 6]
    for step in (4, 5, 6):
        for key in ("loss", "lr/decay", "grad_norm", "clip_coef"):
            assert steps[step][key] == run_a.steps[step][key], (step, key)
    _assert_same_training(run_a, resumed)
    final_a, _ = read(run_a.out / "tiny.safetensors")
    final_b, _ = read(out / "tiny.safetensors")
    assert all(torch.equal(final_a[k], final_b[k]) for k in final_a)
    last_a, last_b = train_state(run_a.out / "tiny-state"), train_state(out / "tiny-state")
    for key in ("global_step", "completed_epochs", "epoch_step", "last_grad_norm", "skip_streak", "slot_hash"):
        assert last_a[key] == last_b[key], key


def test_mid_epoch_resume_skips_the_trained_batches(env, run_a):
    """A step state inside epoch 1 (after 4 of its 5 batches): the resumed run skips those 4 batches of the epoch's
    new shuffle (kohya behavior, design 4.7), trains the partial window [4] as step 3 and epoch 2 as steps 4-6.
    Without save_every_n_epochs the end of LR phase 1 (epoch 1) still gets its pinned epoch state (design 4.6)."""
    logs = []
    out = env.root / "run_c"
    run(env, out, *ADAMW_RUN, "--resume", str(run_a.out / "tiny-step00000002-state"), logs=logs)
    steps = by_step(logs)
    assert sorted(steps) == [3, 4, 5, 6]
    for step, values in steps.items():
        assert math.isfinite(values["loss"])
        assert values["lr/decay"] == pytest.approx(EXPECTED_LR[step - 1], rel=1e-12), step
    last = train_state(out / "tiny-state")
    assert (last["global_step"], last["completed_epochs"], last["epoch_step"]) == (6, 2, 0)
    phase_end = train_state(out / "tiny-000001-state")
    assert (phase_end["global_step"], phase_end["completed_epochs"], phase_end["epoch_step"]) == (3, 1, 0)
    names = os.listdir(out)
    assert "tiny-000001.safetensors" not in names and not any("-step" in name for name in names)


COSINE = ("--mixed_precision", "no", "--gradient_accumulation_steps", str(GA), "--optimizer_type", "AdamW",
          "--optimizer_args", "weight_decay=0.01", "--learning_rate", "1e-3", "--lr_scheduler", "cosine",
          "--lr_warmup_steps", "2", "--max_train_epochs", "2", "--max_grad_norm", "1.0", "--save_state",
          "--save_every_n_steps", "1", "--save_last_n_steps_state", "100")


def test_resume_with_a_non_phase_scheduler_continues_its_lr(env):
    """Review fix (integration): without --moss_ft_lr_phases (kohya's cosine with a 2-step warmup here) the resumed
    run's first window used the freshly built scheduler's step-0 LR (0: the window only moved the moments). From the
    epoch-end step state the logged LR, loss, grad norm and the trained weights now continue bit for bit."""
    loaded_a, logs_a = [], []
    out_a = env.root / "run_cosine_a"
    run(env, out_a, *COSINE, loaded=loaded_a, logs=logs_a)
    steps_a = by_step(logs_a)
    assert sorted(steps_a) == list(range(1, 2 * STEPS_PER_EPOCH + 1))
    assert steps_a[1]["lr/decay"] == 0.0 and steps_a[4]["lr/decay"] > 0.0  # kohya's warmup starts at 0
    loaded_b, logs_b = [], []
    run(env, env.root / "run_cosine_b", *COSINE, "--resume", str(out_a / "tiny-step00000003-state"), loaded=loaded_b,
        logs=logs_b)
    steps_b = by_step(logs_b)
    assert sorted(steps_b) == [4, 5, 6]
    for step in (4, 5, 6):
        for key in ("loss", "lr/decay", "lr/no_decay", "grad_norm", "clip_coef"):
            assert steps_b[step][key] == steps_a[step][key], (step, key)
    a = types.SimpleNamespace(model=loaded_a[0][0], pool=loaded_a[0][1], slots=loaded_a[0][2])
    b = types.SimpleNamespace(model=loaded_b[0][0], pool=loaded_b[0][1], slots=loaded_b[0][2])
    _assert_same_training(a, b)


def test_an_incomplete_state_is_refused_before_anything_is_loaded(env, run_a):
    broken = env.root / "broken" / "tiny-step00000003-state"
    shutil.copytree(run_a.out / "tiny-step00000003-state", broken)
    os.remove(broken / ft_state.COMPLETE_FILE)
    loaded = []
    with pytest.raises(ValueError, match="COMPLETE.json") as refused:
        run(env, env.root / "run_d", *ADAMW, "--resume", str(broken), loaded=loaded)
    assert "no complete state" in str(refused.value)  # the hint about the newest complete state next to it
    assert not loaded and not (env.root / "run_d").exists()


# ---------------------------------------------------------------------------------------------------- mode (b)
ADAFACTOR = ("--mixed_precision", "bf16", "--moss_stream_fp32", "--optimizer_type", "Adafactor",
             "--fused_backward_pass", "--gradient_accumulation_steps", "1", "--learning_rate", "1e-3",
             "--max_train_epochs", "2", "--max_grad_norm", "1.0", "--save_state", "--save_every_n_steps", "5",
             "--save_last_n_steps_state", "10")


def test_mode_b_adafactor_fused_runs_and_resumes(env):
    """adafactor_fused (design 3.2): GA 1, every parameter stepped in its gradient hook (bf16 joint Linears written
    back with stochastic rounding, the fp32-stream parameters exactly), no host pool. 10 steps straight against 5 +
    a resume from the epoch-end state + 5: bit-identical (the Adafactor statistics keep their fp32 dtype through the
    state, the stochastic rounding draws from the restored RNG)."""
    loaded, logs = [], []
    out = env.root / "run_b_mode"
    run(env, out, *ADAFACTOR, loaded=loaded, logs=logs)
    model, pool, slots = loaded[0]
    assert pool is None
    steps = by_step(logs)
    assert sorted(steps) == list(range(1, 2 * IMAGES + 1))
    assert all(math.isfinite(v["loss"]) and v["lr/decay"] == pytest.approx(1e-3) for v in steps.values())
    final, meta = read(out / "tiny.safetensors")
    assert meta["ss_moss_ft_mode"] == "adafactor_fused"
    assert all(not torch.equal(final[k], env.base_state[k]) for k in final if k != "rope.bands")
    params = dict(model.named_parameters())
    assert params["layers.0.proj.q.weight"].dtype == torch.bfloat16
    assert params["shared_block_modulation.weight"].dtype == torch.float32  # the fp32 stream
    state = train_state(out / "tiny-step00000005-state")
    assert (state["global_step"], state["completed_epochs"], state["epoch_step"], state["mode"]) == (
        5, 1, 0, "adafactor_fused")

    loaded_r, logs_r = [], []
    out_r = env.root / "run_b_mode_resumed"
    run(env, out_r, *ADAFACTOR, "--resume", str(out / "tiny-step00000005-state"), loaded=loaded_r, logs=logs_r)
    assert sorted(by_step(logs_r)) == list(range(6, 2 * IMAGES + 1))
    resumed = dict(loaded_r[0][0].named_parameters())
    for name, p in params.items():
        assert p.dtype == resumed[name].dtype and torch.equal(p.detach(), resumed[name].detach()), name
