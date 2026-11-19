# MOSS-Image full fine-tune: one micro-step's noise, forward and flow-matching loss (docs/moss_full_finetune_design.md
# 4.4). The math is the MOSS LoRA trainer's - MossNetworkTrainer.get_noise_pred_and_target and the loss path of
# train_network.NetworkTrainer.process_batch - without what belongs to a network (T-LoRA masks, TSM grouped draws, the
# auxiliary losses' stashes, LoRA regularisers): the same RNG draws in the same order and the same loss chain, so a
# micro-step gives the LoRA trainer's loss and gradients bit for bit (tests/test_moss_ft_loss.py).

from typing import Optional

import torch

from library import anima_train_utils, custom_train_functions, train_util
from library.custom_train_functions import apply_masked_loss, apply_snr_weight_for_flow_matching
from library.moss import flow, ft_args
from library.utils import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)

_OT_LOGGED = False  # --flow_use_ot: the first batch's assignment is logged once per process (one trainer per process)


def _ot_reorder_noise(latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """--flow_use_ot: the batch's noise rows reassigned to the latents by cosine optimal transport (cost in fp32), as
    MossNetworkTrainer._ot_reorder_noise."""
    global _OT_LOGGED
    with torch.no_grad():
        b = latents.size(0)
        _, (_, col_indices) = train_util.cosine_optimal_transport(latents.reshape(b, -1).float(), noise.reshape(b, -1))
        noise = noise[col_indices.squeeze(0)]
    if not _OT_LOGGED:
        logger.info(f"[MOSS OT] First batch: noise reordered by cosine OT. New noise assignment indices:"
                    f" {col_indices.squeeze(0).tolist()}")
        _OT_LOGGED = True
    return noise


def _token_mining(loss: torch.Tensor, args, sigmas: torch.Tensor, noise_scheduler) -> torch.Tensor:
    """--token_mining as train_network.process_batch applies it (the sigma gate reads the flow sigmas)."""
    mining_sigmas = None
    if not getattr(args, "token_mining_no_sigma_gate", False):
        ts = sigmas.detach().float().reshape(-1)
        if ts.numel() > 0 and ts.max() > 1.5:  # discrete timesteps in [0, T] (not MOSS: kept for parity)
            mining_sigmas = ts / noise_scheduler.config.num_train_timesteps
        else:
            mining_sigmas = ts
    return custom_train_functions.apply_token_mining(
        loss,
        sigmas=mining_sigmas,
        alpha=float(getattr(args, "token_mining_alpha", 1.0)),
        min_weight=float(getattr(args, "token_mining_min_weight", 0.25)),
        max_weight=float(getattr(args, "token_mining_max_weight", 4.0)),
        sigma_gate=mining_sigmas is not None,
    )


def moss_ft_noise_and_loss(args, dit, batch, accelerator, sigma_sampler: Optional[flow.MossSigmaSampler],
                           noise_scheduler, stream_dtype: torch.dtype, text_encoding_strategy) -> torch.Tensor:
    """One micro-step's loss (a float64 0-dim tensor, the batch mean; the caller divides it by the window's n_micro).

    batch: the DataLoader's dict with cached latents ("latents"), cached text-encoder outputs
    ("text_encoder_outputs_list" = [text_embeds, text_mask, caption_dropout_rates]), "loss_weights" and optionally
    "alpha_masks". stream_dtype: fp32 with --moss_stream_fp32, else the compute dtype (the noisy input and the text go
    to the DiT in it). The batch dict is not modified.

    Steps (design 4.4): fp32 noise (randn_like) -> cosine OT (--flow_use_ot, batch > 1) -> antithetic pairing ->
    flow.get_noisy_model_input_and_timesteps (sigma sampler, fp32 timesteps = 1000 sigma) -> the noisy input requires
    grad under checkpointing -> caption-dropout check -> forward under accelerator.autocast() -> target = noise -
    latents in float64 -> conditional_loss (loss_type, Huber schedule, loss_scale) x the sigma weighting -> CFM (batch
    > 1) -> masked loss -> token mining -> mean over all but the batch -> x loss_weights -> min-SNR -> loss_multiplier ->
    mean. The order is the LoRA trainer's (min-SNR after the per-sample mean, as its post_process_loss)."""
    device = accelerator.device
    latents, kept = _latents_and_text(batch, device, text_encoding_strategy)

    # noise, timesteps (MossNetworkTrainer.get_noise_pred_and_target)
    noise = torch.randn_like(latents, dtype=torch.float32)
    if getattr(args, "flow_use_ot", False) and latents.size(0) > 1:
        noise = _ot_reorder_noise(latents, noise)  # before antithetic pairing, as Anima / train_util do
    noise = custom_train_functions.maybe_apply_antithetic_noise_pairing(args, noise, is_train=True)
    noisy, timesteps, sigmas = flow.get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device,
                                                                        stream_dtype, sigma_sampler)
    if args.gradient_checkpointing and not ft_args.carousel_enabled(args):
        # unsloth offload only returns grad-carrying outputs for grad-carrying inputs. Not in the carousel mode: it would
        # make the whole stream require grad (full-depth backward); its floor block takes care of unsloth itself
        noisy.requires_grad_(True)
    return _forward_loss_rows(args, dit, batch, accelerator, latents, noise, noisy, timesteps, sigmas, noise_scheduler,
                              stream_dtype, kept).mean()


def moss_ft_val_loss_rows(args, dit, batch, accelerator, noise: torch.Tensor, sigma: float, noise_scheduler,
                          stream_dtype: torch.dtype, text_encoding_strategy) -> torch.Tensor:
    """The validation loss of every sample of `batch` at the fixed noise level `sigma` (library/moss/ft_val.py): float64
    [B] on the device, the per-sample losses moss_ft_noise_and_loss averages (same target, weighting, loss chain and
    reduction; x loss_weights, min-SNR, loss_multiplier) - without its draws: `noise` is given (fp32, the latents'
    shape; no cosine OT, no antithetic pairing), every timestep is 1000 x sigma and the input-perturbation noise
    (--ip_noise_gamma) is left out (is_train=False). The caller runs it under torch.no_grad(); nothing here draws from
    an RNG or marks a tensor as grad-bearing."""
    device = accelerator.device
    latents, kept = _latents_and_text(batch, device, text_encoding_strategy)
    noise = noise.to(device=device, dtype=torch.float32)
    if noise.shape != latents.shape:
        raise ValueError(f"MOSS full FT validation: noise {tuple(noise.shape)} does not match the latents"
                         f" {tuple(latents.shape)}")
    fixed = torch.full((latents.shape[0],), float(sigma) * flow.NUM_TRAIN_TIMESTEPS, dtype=torch.float32)
    noisy, timesteps, sigmas = flow.get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device,
                                                                        stream_dtype, None, fixed_timesteps=fixed,
                                                                        is_train=False)
    return _forward_loss_rows(args, dit, batch, accelerator, latents, noise, noisy, timesteps, sigmas, noise_scheduler,
                              stream_dtype, kept)


def _latents_and_text(batch, device, text_encoding_strategy):
    """The batch's cached latents on the device and its text-encoder outputs through the caption-dropout check."""
    latents = batch.get("latents")
    if latents is None:
        raise ValueError("MOSS full FT: the batch has no cached latents: --cache_latents is required (design 2.4)")
    latents = latents.to(device=device)
    outputs = batch.get("text_encoder_outputs_list")
    if outputs is None:
        raise ValueError("MOSS full FT: the batch has no text-encoder outputs: --cache_text_encoder_outputs or"
                         " --moss_te_live is required (design 2.4)")
    kept = text_encoding_strategy.drop_cached_text_encoder_outputs(*outputs[:-1], caption_dropout_rates=outputs[-1])
    return latents, kept


def _forward_loss_rows(args, dit, batch, accelerator, latents, noise, noisy, timesteps, sigmas, noise_scheduler,
                       stream_dtype, kept) -> torch.Tensor:
    """The forward and the loss chain from the noisy input on: the per-sample losses (float64 [B]) before the batch
    mean (the training step takes their mean, validation sums them per noise level)."""
    device = accelerator.device
    text = kept[0].to(device, dtype=stream_dtype)
    text_mask = kept[1].to(device).bool()
    with accelerator.autocast():
        model_pred = dit(noisy, timesteps, text, text_mask)

    target = noise.to(torch.float64) - latents.to(torch.float64)  # velocity: eps - x0
    weighting = anima_train_utils.compute_loss_weighting_for_anima(args.weighting_scheme, sigmas)
    sigmas_flat = sigmas.flatten()  # what the LoRA trainer hands its loss path as "timesteps"
    noise = noise.to(torch.float64)

    # the loss chain (train_network.NetworkTrainer.process_batch, is_train, no network)
    model_pred = model_pred.to(dtype=torch.float64)
    huber_c = train_util.get_huber_threshold_if_needed(args, sigmas_flat, noise_scheduler)
    loss = train_util.conditional_loss(model_pred, target, args.loss_type, "none", huber_c,
                                       scale=float(getattr(args, "loss_scale", 1.0)))
    if weighting is not None:
        loss = loss * weighting
    if getattr(args, "contrastive_flow_matching", False) and latents.size(0) > 1:
        negative_latents = latents.roll(1, 0).detach()
        negative_noise = noise.roll(1, 0).detach()
        with torch.no_grad():
            target_negative = negative_noise - negative_latents  # rectified flow: noise - latents
        target_negative = target_negative.to(dtype=torch.float64)
        loss_contrastive = torch.nn.functional.mse_loss(model_pred, target_negative, reduction="none")
        loss = loss - float(args.cfm_lambda) * loss_contrastive
    if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
        loss = apply_masked_loss(loss, batch)
    if getattr(args, "token_mining", False):
        loss = _token_mining(loss, args, sigmas_flat, noise_scheduler)

    loss = loss.mean(dim=list(range(1, loss.ndim)))  # mean over all dims except batch
    loss = loss * batch["loss_weights"]
    if args.min_snr_gamma:
        loss = apply_snr_weight_for_flow_matching(loss, sigmas_flat, args.min_snr_gamma, soft=args.min_snr_gamma_soft)
    if getattr(args, "loss_multiplier", None):
        loss.mul_(float(args.loss_multiplier))
    return loss
