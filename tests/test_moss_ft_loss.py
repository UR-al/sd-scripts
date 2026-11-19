"""CPU tests of library/moss/ft_loss.py moss_ft_noise_and_loss (docs/moss_full_finetune_design.md 4.4, section-5 row
test_loss_matches_lora_trainer_math).

The full fine-tune's micro-step loss must be the MOSS LoRA trainer's: the same noise, timesteps, target, weighting and
loss chain as MossNetworkTrainer.get_noise_pred_and_target + train_network.NetworkTrainer.process_batch (no network,
nothing LoRA-only). Both run on two copies of the tiny DiT from the same RNG state and batch; the losses and every
parameter gradient must be bit-identical, for the default recipe and each loss option the FT parser has.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_loss.py -q -p no:cacheprovider
"""
import contextlib
import copy
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

from library.moss import flow, ft_args, strategy
from library.moss.models import MossTransformer

TINY = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=48, freq_dim=16, latent_channels=128)


class _Accel:
    device = torch.device("cpu")

    def __init__(self, autocast_dtype=None):
        self.autocast_dtype = autocast_dtype

    def autocast(self):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast("cpu", dtype=self.autocast_dtype)

    def unwrap_model(self, model):
        return model


def _model(dtype=torch.float32, stream_fp32=False, ckpt=False):
    torch.manual_seed(0)
    model = MossTransformer(**TINY)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.02)
    model.to(dtype)
    if stream_fp32:
        model.enable_stream_fp32()
    if ckpt:
        model.enable_gradient_checkpointing()
    return model.train()


def _batch(b=2, hw=(4, 6), text=7, masks=False, seed=3):
    g = torch.Generator().manual_seed(seed)
    latents = torch.randn(b, TINY["latent_channels"], *hw, generator=g)
    embeds = torch.randn(b, text, TINY["text_dim"], generator=g).to(torch.bfloat16)  # cached TE outputs are bf16
    mask = torch.ones(b, text, dtype=torch.long)
    mask[-1, text - 2:] = 0  # a padded caption in the batch
    batch = {"latents": latents, "text_encoder_outputs_list": [embeds, mask, torch.zeros(b)],
             "loss_weights": torch.linspace(1.0, 0.5, b)}
    if masks:
        batch["alpha_masks"] = torch.rand(b, hw[0] * 16, hw[1] * 16, generator=g)
    return batch


def _lora_args(options):
    import moss_train_network as mtn

    args = mtn.setup_parser().parse_args([])
    for key, value in options.items():
        setattr(args, key, value)
    args._anima_model = True  # what prepare_moss_runtime sets: the rectified-flow CFM target (noise - latents)
    return args


def _ft_args(options):
    args = ft_args.setup_parser().parse_args([])
    for key, value in options.items():
        setattr(args, key, value)
    return args


def _clone_batch(batch):
    return {k: ([t.clone() for t in v] if isinstance(v, list) else v.clone()) for k, v in batch.items()}


def _lora_loss(model, batch, options, weight_dtype, accel, seed):
    import moss_train_network as mtn

    args = _lora_args(options)
    trainer = mtn.MossNetworkTrainer()
    trainer._sigma_sampler = flow.MossSigmaSampler(args)
    scheduler = trainer.get_noise_scheduler(args, "cpu")
    torch.manual_seed(seed)
    loss, _, _ = trainer.process_batch(_clone_batch(batch), [], model, None, None, scheduler, torch.float32, weight_dtype,
                                       accel, args, strategy.MossTextEncodingStrategy(), None, is_train=True,
                                       train_text_encoder=False, train_unet=True)
    return loss


def _ft_loss(model, batch, options, stream_dtype, accel, seed):
    from library import sd3_train_utils
    from library.moss import ft_loss

    args = _ft_args(options)
    sampler = flow.MossSigmaSampler(args)
    scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
    torch.manual_seed(seed)
    return ft_loss.moss_ft_noise_and_loss(args, model, batch, accel, sampler, scheduler, stream_dtype,
                                          strategy.MossTextEncodingStrategy())


CASES = {
    "default": {},
    "huber_constant": dict(loss_type="huber", huber_schedule="constant", huber_c=0.2),
    "smooth_l1_exponential": dict(loss_type="smooth_l1", huber_schedule="exponential"),
    "soft_welsch_loss_scale": dict(loss_type="soft_welsch", huber_schedule="constant", loss_scale=0.5),
    "sigma_sqrt_sigmoid": dict(weighting_scheme="sigma_sqrt", timestep_sampling="sigmoid", sigmoid_scale=1.0),
    "cosmap_min_snr": dict(weighting_scheme="cosmap", timestep_sampling="uniform", min_snr_gamma=5.0),
    "soft_min_snr_shift": dict(min_snr_gamma=2.0, min_snr_gamma_soft=True, timestep_sampling="shift",
                               discrete_flow_shift=3.0),
    "cfm": dict(contrastive_flow_matching=True, cfm_lambda=0.05),
    "ot": dict(flow_use_ot=True),
    "antithetic": dict(antithetic_noise_pairing=True),
    "masked": dict(masked_loss=True),
    "token_mining": dict(token_mining=True),
    "loss_multiplier": dict(loss_multiplier=2.0),
    "checkpointing": dict(gradient_checkpointing=True),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_loss_matches_lora_trainer_math(case):
    options = CASES[case]
    ckpt = bool(options.get("gradient_checkpointing", False))
    lora_model = _model(ckpt=ckpt)
    ft_model = copy.deepcopy(lora_model)
    batch = _batch(masks=options.get("masked_loss", False))
    original = _clone_batch(batch)
    lora = _lora_loss(lora_model, batch, options, torch.float32, _Accel(), seed=11)
    ft = _ft_loss(ft_model, batch, options, torch.float32, _Accel(), seed=11)
    assert ft.dim() == 0 and ft.dtype == lora.dtype == torch.float64
    assert torch.equal(ft, lora), (case, float(ft), float(lora))
    lora.backward()
    ft.backward()
    for (name, p), (_, q) in zip(lora_model.named_parameters(), ft_model.named_parameters()):
        assert p.grad is not None and q.grad is not None, name
        assert torch.equal(p.grad, q.grad), (case, name)
    for key, value in original.items():  # the FT function leaves the batch as it got it
        got = batch[key]
        if isinstance(value, list):
            assert all(torch.equal(a, b) for a, b in zip(value, got)), key
        else:
            assert torch.equal(value, got), key


def test_bf16_compute_with_fp32_stream_matches():
    """The run's own setting: bf16 joint Linears, fp32 stream (--moss_stream_fp32), bf16 autocast, checkpointing."""
    options = dict(moss_stream_fp32=True, gradient_checkpointing=True, mixed_precision="bf16")
    lora_model = _model(torch.bfloat16, stream_fp32=True, ckpt=True)
    ft_model = copy.deepcopy(lora_model)
    batch = _batch()
    lora = _lora_loss(lora_model, batch, options, torch.bfloat16, _Accel(torch.bfloat16), seed=5)
    ft = _ft_loss(ft_model, batch, options, torch.float32, _Accel(torch.bfloat16), seed=5)
    assert torch.equal(ft, lora), (float(ft), float(lora))
    lora.backward()
    ft.backward()
    for (name, p), (_, q) in zip(lora_model.named_parameters(), ft_model.named_parameters()):
        assert torch.equal(p.grad, q.grad), name


def test_batch_of_one_skips_the_pair_options():
    """CFM and OT need batch > 1 (as the LoRA trainer: they are skipped for a single sample)."""
    options = dict(contrastive_flow_matching=True, flow_use_ot=True, antithetic_noise_pairing=True)
    lora_model = _model()
    ft_model = copy.deepcopy(lora_model)
    batch = _batch(b=1)
    lora = _lora_loss(lora_model, batch, options, torch.float32, _Accel(), seed=2)
    ft = _ft_loss(ft_model, batch, options, torch.float32, _Accel(), seed=2)
    assert torch.equal(ft, lora)


def test_uncached_latents_are_refused():
    batch = _batch()
    del batch["latents"]
    with pytest.raises(ValueError, match="cache_latents"):
        _ft_loss(_model(), batch, {}, torch.float32, _Accel(), seed=0)


def test_caption_dropout_rates_are_refused_on_the_host():
    """drop_cached_text_encoder_outputs is called as the LoRA trainer does: a nonzero rate on the host is refused."""
    batch = _batch()
    batch["text_encoder_outputs_list"][2] = torch.tensor([0.0, 0.1])
    with pytest.raises(RuntimeError, match="caption dropout"):
        _ft_loss(_model(), batch, {}, torch.float32, _Accel(), seed=0)
