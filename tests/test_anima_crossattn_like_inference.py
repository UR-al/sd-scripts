"""CPU tests for --anima_crossattn_like_inference and --anima_sample_like_inference (no GPU, no model files).

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe tests/test_anima_crossattn_like_inference.py   (or with pytest)

Why (2026-11-09): eastown v6 samples looked unlike ComfyUI / Forge pictures of the same LoRA. Besides the sampler, the
trainer gave the DiT the LLM-adapter context zero-padded to --t5_max_token_length (620) where ComfyUI pads to 512, and its
sampler rounded the sigma table and the latent to bf16 at every step where ComfyUI keeps them in fp32.
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anima_train_network import AnimaNetworkTrainer, setup_parser
from library import anima_block_compile, anima_models, anima_train_utils
from library.anima38.sampling import ConditionedDiT

TOY = dict(max_img_h=32, max_img_w=32, max_frames=1, in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
           model_channels=64, num_blocks=2, num_heads=2, mlp_ratio=4.0, crossattn_emb_channels=32, use_adaln_lora=True,
           adaln_lora_dim=16, use_llm_adapter=False, attn_mode="torch", pos_emb_cls="rope3d")


def _mask(lengths, total):
    mask = torch.zeros(len(lengths), total, dtype=torch.long)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    return mask


def _context(lengths, total, dim=4):
    mask = _mask(lengths, total)
    context = torch.randn(len(lengths), total, dim)
    context[~mask.bool()] = 0
    return context, mask


# fit_crossattn_like_inference ------------------------------------------------------------------------------------------

def test_fit_cuts_trainer_padding_to_512():
    context, mask = _context([300], 620)
    out = anima_models.fit_crossattn_like_inference(context, mask)
    assert out.shape == (1, 512, 4)
    assert torch.equal(out[:, :300], context[:, :300])
    assert torch.count_nonzero(out[:, 300:]) == 0


def test_fit_never_cuts_a_caption_longer_than_512():
    context, mask = _context([600], 620)
    out = anima_models.fit_crossattn_like_inference(context, mask)
    assert out.shape == (1, 600, 4)
    assert torch.equal(out, context[:, :600])


def test_fit_pads_a_shorter_cache_up_to_512():
    context, mask = _context([100], 256)
    out = anima_models.fit_crossattn_like_inference(context, mask)
    assert out.shape == (1, 512, 4)
    assert torch.equal(out[:, :256], context)
    assert torch.count_nonzero(out[:, 256:]) == 0


def test_fit_batch_takes_the_longest_caption():
    context, mask = _context([100, 560], 620)
    out = anima_models.fit_crossattn_like_inference(context, mask)
    assert out.shape == (2, 560, 4)
    assert torch.count_nonzero(out[0, 100:]) == 0


def test_fit_without_mask_is_a_no_op():
    context = torch.randn(1, 620, 4)
    assert anima_models.fit_crossattn_like_inference(context, None) is context


# the model applies it only when switched on ------------------------------------------------------------------------------

def _toy_with_stub_adapter(total=620, real=300):
    torch.manual_seed(0)
    model = anima_models.Anima(**TOY).float().eval()
    model.use_llm_adapter = True
    calls = []

    def stub_adapter(source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        calls.append(target_input_ids.shape)
        return torch.ones(target_input_ids.shape[0], target_input_ids.shape[1], 32)

    model.llm_adapter = stub_adapter
    ids = torch.ones(1, total, dtype=torch.long)
    return model, ids, _mask([real], total), calls


def test_preprocess_text_embeds_off_keeps_trainer_length():
    model, ids, mask, _ = _toy_with_stub_adapter()
    out = model._preprocess_text_embeds(torch.randn(1, 540, 32), ids, mask, torch.ones(1, 540, dtype=torch.long))
    assert out.shape[1] == 620


def test_preprocess_text_embeds_on_gives_inference_length():
    model, ids, mask, _ = _toy_with_stub_adapter()
    model.crossattn_like_inference = True
    out = model._preprocess_text_embeds(torch.randn(1, 540, 32), ids, mask, torch.ones(1, 540, dtype=torch.long))
    assert out.shape[1] == 512
    assert torch.count_nonzero(out[:, 300:]) == 0 and torch.count_nonzero(out[:, :300]) == 300 * 32


def _block_context_lengths(model, ids, mask):
    seen = []
    hook = model.blocks[0].register_forward_pre_hook(lambda _m, args: seen.append(args[2].shape[1]))
    x = torch.randn(1, 16, 1, 8, 8)
    with torch.no_grad():
        model.forward_mini_train_dit(x, torch.tensor([0.5]), torch.randn(1, 540, 32), padding_mask=torch.zeros(1, 1, 8, 8),
                                     source_attention_mask=torch.ones(1, 540, dtype=torch.long), t5_input_ids=ids,
                                     t5_attn_mask=mask)
    hook.remove()
    return seen


def test_forward_inline_adapter_path_follows_the_switch():
    model, ids, mask, calls = _toy_with_stub_adapter()
    assert _block_context_lengths(model, ids, mask) == [620]
    model.crossattn_like_inference = True
    assert _block_context_lengths(model, ids, mask) == [512]
    assert len(calls) == 2


def test_trainer_switches_the_model_on_from_args():
    model = anima_models.Anima(**TOY)
    trainer = AnimaNetworkTrainer()
    trainer._use_unsloth_offload_checkpointing = False
    trainer.is_swapping_blocks = False
    accelerator = types.SimpleNamespace(prepare=lambda module, **_kwargs: module)
    args = types.SimpleNamespace(anima_stream_fp32=False, anima_crossattn_like_inference=True, gradient_checkpointing=False)
    trainer.prepare_unet_with_accelerator(args, accelerator, model)
    assert model.crossattn_like_inference is True
    other = anima_models.Anima(**TOY)
    trainer.prepare_unet_with_accelerator(types.SimpleNamespace(gradient_checkpointing=False), accelerator, other)
    assert other.crossattn_like_inference is False


# sampler ----------------------------------------------------------------------------------------------------------------

class _RecordingDiT(torch.nn.Module):
    """Stands in for the DiT: a bf16 weight (the compute dtype) behind an fp32 first parameter, as --anima_stream_fp32
    leaves Anima (x_embedder upcast, Block Linears bf16). Records what it is fed, returns a constant velocity."""

    def __init__(self, stream_fp32, velocity=0.25):
        super().__init__()
        self.first = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32 if stream_fp32 else torch.bfloat16))
        self.linear = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.stream_fp32 = stream_fp32
        self.velocity = velocity
        self.x_dtypes, self.t_dtypes, self.t_values = [], [], []

    def forward(self, x, t, context, padding_mask=None):
        self.x_dtypes.append(x.dtype)
        self.t_dtypes.append(t.dtype)
        self.t_values.append(float(t.float()))
        return torch.full_like(x, self.velocity)


def _comfy_simple(steps, shift):
    """ComfyUI, written out: ModelSamplingDiscreteFlow(shift, multiplier 1.0).sigmas + samplers.simple_scheduler."""
    t = torch.arange(1, 1001, 1) / 1000 * 1.0
    table = shift * (t / 1.0) / (1 + (shift - 1) * (t / 1.0))
    ss = len(table) / steps
    return [float(table[-(1 + int(x * ss))]) for x in range(steps)] + [0.0]


def test_sigma_table_is_comfyuis_simple_schedule():
    for steps in (20, 28, 30, 32, 50):
        ours = anima_train_utils.comfy_simple_flow_sigmas(steps, 3.0)
        assert ours.dtype == torch.float32
        assert ours.tolist() == _comfy_simple(steps, 3.0), steps
    # 20 steps divide 1000: the same values as the shifted linspace the original sampler uses
    lin = torch.linspace(1.0, 0.0, 21, dtype=torch.float32)
    assert torch.allclose(anima_train_utils.comfy_simple_flow_sigmas(20, 3.0), lin * 3.0 / (1 + 2.0 * lin), atol=1e-6)
    # 30 steps do not: ComfyUI truncates the timestep to the 1/1000 grid
    assert abs(float(anima_train_utils.comfy_simple_flow_sigmas(30, 3.0)[1]) - 0.988753) < 1e-5


def test_like_inference_feeds_bf16_x_and_fp32_t_and_keeps_the_latent_fp32():
    dit = _RecordingDiT(stream_fp32=True)  # dit.dtype is fp32 here, as on the trainer's fp32 stream
    out = anima_train_utils.do_sample(64, 64, 1, dit, torch.zeros(1, 512, 32), 30, torch.float32, torch.device("cpu"),
                                      guidance_scale=1.0, flow_shift=3.0, like_inference=True)
    assert set(dit.x_dtypes) == {torch.bfloat16}  # ComfyUI's xc.to(model compute dtype)
    assert set(dit.t_dtypes) == {torch.float32}
    assert dit.t_values == anima_train_utils.comfy_simple_flow_sigmas(30, 3.0).tolist()[:-1]
    assert out.dtype == torch.float32
    noise = torch.randn((1, 16, 1, 8, 8), dtype=torch.float32, generator=torch.manual_seed(1))
    assert torch.allclose(out, noise - 0.25, atol=1e-5)  # constant velocity over sigma 1 -> 0, accumulated in fp32


def test_like_inference_timestep_is_fp32_under_autocast_without_fp32_stream():
    dit = _RecordingDiT(stream_fp32=False)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        anima_train_utils.do_sample(64, 64, 1, dit, torch.zeros(1, 512, 32), 4, torch.bfloat16, torch.device("cpu"),
                                    like_inference=True)
    assert set(dit.t_dtypes) == {torch.float32}
    assert dit.t_values == anima_train_utils.comfy_simple_flow_sigmas(4, 3.0).tolist()[:-1]  # not bf16-rounded


def test_like_inference_timestep_stays_in_model_dtype_without_autocast_or_fp32_stream():
    dit = _RecordingDiT(stream_fp32=False)
    anima_train_utils.do_sample(64, 64, 1, dit, torch.zeros(1, 512, 32), 4, torch.bfloat16, torch.device("cpu"),
                                like_inference=True)
    assert set(dit.t_dtypes) == {torch.bfloat16}


def test_original_sampler_is_unchanged():
    dit = _RecordingDiT(stream_fp32=False)
    out = anima_train_utils.do_sample(64, 64, 1, dit, torch.zeros(1, 512, 32), 20, torch.bfloat16, torch.device("cpu"),
                                      flow_shift=3.0)
    assert set(dit.t_dtypes) == {torch.bfloat16} and set(dit.x_dtypes) == {torch.bfloat16}
    rounded = torch.linspace(1.0, 0.0, 21, dtype=torch.bfloat16)
    rounded = (rounded * 3.0) / (1 + (3.0 - 1) * rounded)
    assert dit.t_values == [float(v) for v in rounded[:-1]]
    x = torch.randn((1, 16, 1, 8, 8), dtype=torch.float32, generator=torch.manual_seed(1)).to(torch.bfloat16)
    for i in range(20):  # the original loop, written out
        x = (x + torch.full_like(x, 0.25).float() * (rounded[i + 1] - rounded[i])).to(torch.bfloat16)
    assert out.dtype == torch.bfloat16 and torch.equal(out, x)


def _cfg_dit():
    calls = []

    def dit(x, t, context, padding_mask=None):
        calls.append(float(context.flatten()[0]))
        return torch.full_like(x, 1.0 if float(context.flatten()[0]) > 0 else 0.0)

    dit.stream_fp32 = True
    return dit, calls


def test_cfg_on_denoised_like_comfyui():
    dit, calls = _cfg_dit()
    out = anima_train_utils.do_sample(64, 64, 1, dit, torch.ones(1, 512, 32), 1, torch.bfloat16, torch.device("cpu"),
                                      guidance_scale=3.0, flow_shift=1.0, neg_crossattn_emb=torch.zeros(1, 512, 32),
                                      like_inference=True)
    noise = torch.randn((1, 16, 1, 8, 8), dtype=torch.float32, generator=torch.manual_seed(1))
    assert calls == [1.0, 0.0]
    # sigma 1 -> 0 in one step: denoised = uncond + 3 * (cond - uncond) = x - 3 -> the result is x - 3
    assert torch.allclose(out, noise - 3.0, atol=1e-5)


def test_cfg_off_path_is_unchanged():
    dit, calls = _cfg_dit()
    out = anima_train_utils.do_sample(64, 64, 1, dit, torch.ones(1, 512, 32), 1, torch.bfloat16, torch.device("cpu"),
                                      guidance_scale=3.0, flow_shift=1.0, neg_crossattn_emb=torch.zeros(1, 512, 32))
    noise = torch.randn((1, 16, 1, 8, 8), dtype=torch.float32, generator=torch.manual_seed(1)).to(torch.bfloat16)
    assert calls == [1.0, 0.0]
    assert torch.equal(out, (noise + torch.full_like(noise, 3.0).float() * -1.0).to(torch.bfloat16))


def test_anima38_wrapper_exposes_compute_dtype_and_stream():
    dit = _RecordingDiT(stream_fp32=True)
    wrapped = ConditionedDiT(dit)
    assert wrapped.stream_fp32 is True
    assert anima_train_utils._model_input_dtype(wrapped, torch.float32) == torch.bfloat16


# prompt text ------------------------------------------------------------------------------------------------------------

def test_escapes_like_comfyui_and_forge():
    # only \( \) are escapes there; \[ \] keep the backslash
    text = r"painting \(object\), \[draft\], a\b"
    assert anima_train_utils.process_prompt_escapes_like_inference(text) == r"painting (object), \[draft\], a\b"
    assert anima_train_utils.process_prompt_escapes_like_inference(r"foo \[bar\]") == r"foo \[bar\]"


def test_emphasis_brackets_are_removed_and_reported():
    f = anima_train_utils.process_prompt_escapes_like_inference
    assert f("a (red:1.3) hat, (blue) sky") == "a red hat, blue sky"
    assert f("(a (b:0.5) c)") == "a b c"
    assert anima_train_utils.prompt_has_emphasis("a (red:1.3) hat")
    assert anima_train_utils.prompt_has_emphasis("(blue) sky")  # plain (x) is 1.1 in ComfyUI
    assert not anima_train_utils.prompt_has_emphasis(r"painting \(object\), plain text")
    assert anima_train_utils.comfy_prompt_segments("a (b:1.2) c") == [("a ", 1.0), ("b", 1.2), (" c", 1.0)]


def test_sample_prompt_text_follows_the_switch():
    text = r"painting \(object\)"
    assert anima_train_utils.sample_prompt_text(text, types.SimpleNamespace(anima_sample_like_inference=False)) == text
    on = types.SimpleNamespace(anima_sample_like_inference=True)
    assert anima_train_utils.sample_prompt_text(text, on) == "painting (object)"
    assert anima_train_utils.sample_prompt_text("", on) == ""


# arguments and metadata ---------------------------------------------------------------------------------------------------

def test_parser_and_metadata():
    args = setup_parser().parse_args(["--anima_crossattn_like_inference", "--anima_sample_like_inference"])
    assert args.anima_crossattn_like_inference and args.anima_sample_like_inference
    metadata = {}
    AnimaNetworkTrainer().update_metadata(metadata, args)
    assert metadata["ss_anima_crossattn"] == "like_inference_min512"
    assert metadata["ss_anima_sample_like_inference"] is True
    off = setup_parser().parse_args(["--t5_max_token_length", "620"])
    metadata = {}
    AnimaNetworkTrainer().update_metadata(metadata, off)
    assert metadata["ss_anima_crossattn"] == "pad_to_t5_max_620"
    assert metadata["ss_anima_sample_like_inference"] is False
    connector = setup_parser().parse_args(["--anima_crossattn_like_inference"])
    connector.anima38_connector = "connector.safetensors"
    metadata = {}
    AnimaNetworkTrainer().update_metadata(metadata, connector)
    assert metadata["ss_anima_crossattn"] == "semantic_connector"


def test_block_compile_constant_matches_the_model():
    assert anima_block_compile.INFERENCE_CROSSATTN_MIN_TOKENS == anima_models.INFERENCE_CROSSATTN_MIN_TOKENS


def test_vae_fp32_unnormalize_matches_the_formula():
    from library import qwen_image_autoencoder_kl as qk

    vae = types.SimpleNamespace(z_dim=2, latents_mean=[0.5, -1.0], latents_std=[2.0, 3.0], dtype=torch.bfloat16)
    seen = {}

    def fake_decode(latents, return_dict=False):
        seen["z"] = latents
        return (torch.zeros(1, 3, 1, 1, 1),)

    vae.decode = fake_decode
    latents = torch.tensor([0.1234567, 0.7654321]).view(1, 2, 1, 1, 1)
    qk.AutoencoderKLQwenImage.decode_to_pixels(vae, latents, fp32_unnormalize=True)
    expected = (latents * torch.tensor([2.0, 3.0]).view(1, 2, 1, 1, 1) + torch.tensor([0.5, -1.0]).view(1, 2, 1, 1, 1)).to(torch.bfloat16)
    assert torch.equal(seen["z"], expected)


if __name__ == "__main__":
    import sys

    failed = 0
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name)
        except Exception as exc:  # noqa: BLE001 - report every failure
            failed += 1
            print("FAIL", name, type(exc).__name__, exc)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
