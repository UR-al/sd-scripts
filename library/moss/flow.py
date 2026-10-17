# MOSS-Image noise-level sampling for LoRA training.
#
# ``--timestep_sampling moss`` reproduces the final pretraining interval of the released Base (technical report,
# Appendix A.2, Eq. 4 and Table 6): on the 1000-step training grid sigma_j = j/1000, logit(sigma_raw) ~
# N(+1.09, 0.95^2) truncated at sigma_raw <= 0.95 and rescaled onto the full grid, mixed with 1% uniform. No shift
# (scheduler training_shift = 1.0). Median sigma ~0.78: the model was trained mostly at high noise.
# Every other mode is delegated to flux_train_utils.get_noisy_model_input_and_timesteps, but in fp32: that function
# casts its outputs to the given dtype, and bf16 would round 1000*sigma to steps of 2-4.

from typing import Optional

import torch

from library import flux_train_utils

NUM_TRAIN_TIMESTEPS = 1000
TIMESTEP_CHOICES = ["moss", "sigma", "uniform", "sigmoid", "shift"]


def moss_sigma_pmf(logit_mean: float = 1.09, logit_std: float = 0.95, cutoff: float = 0.95, uniform_mix: float = 0.01,
                   steps: int = NUM_TRAIN_TIMESTEPS) -> torch.Tensor:
    """Probability of sigma_j = j/steps (j = 1..steps), float64 on CPU."""
    normal = torch.distributions.Normal(torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))

    def cdf(x):  # P(sigma_raw <= x) with logit(sigma_raw) ~ N(logit_mean, logit_std^2)
        x = x.clamp(1e-15, 1 - 1e-15)
        return normal.cdf((torch.log(x / (1 - x)) - logit_mean) / logit_std)

    j = torch.arange(1, steps + 1, dtype=torch.float64)
    upper, lower = cdf(cutoff * j / steps), cdf(cutoff * (j - 1) / steps)
    lower[0] = 0.0
    pmf = (1 - uniform_mix) * (upper - lower) / cdf(torch.tensor(cutoff, dtype=torch.float64)) + uniform_mix / steps
    return pmf / pmf.sum()


class MossSigmaSampler:
    def __init__(self, args):
        self.pmf = moss_sigma_pmf(args.moss_logit_mean, args.moss_logit_std, args.moss_sigma_cutoff, args.moss_uniform_mix)

    def sample(self, batch: int, device) -> torch.Tensor:
        index = torch.multinomial(self.pmf.float(), batch, replacement=True)  # CPU RNG (seeded by the trainer)
        return ((index + 1).float() / NUM_TRAIN_TIMESTEPS).to(device)


def get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype, sampler: Optional[MossSigmaSampler],
                                        fixed_timesteps=None, is_train=True):
    """-> (noisy input in ``dtype``, timesteps = 1000*sigma fp32 [B], sigmas fp32 [B,1,1,1])."""
    if fixed_timesteps is None and is_train and args.timestep_sampling == "moss":
        fixed_timesteps = sampler.sample(latents.shape[0], device) * NUM_TRAIN_TIMESTEPS
    if fixed_timesteps is not None:
        fixed_timesteps = torch.as_tensor(fixed_timesteps, device=device).float()
    noisy, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
        args, noise_scheduler, latents.float(), noise.float(), device, torch.float32,
        fixed_timesteps=fixed_timesteps, is_train=is_train)
    return noisy.to(dtype), timesteps.float().reshape(-1), sigmas.float()


def describe_moss_sampling(args) -> str:
    if args.timestep_sampling != "moss":
        return f"timestep_sampling={args.timestep_sampling} (fp32 timesteps)"
    pmf = moss_sigma_pmf(args.moss_logit_mean, args.moss_logit_std, args.moss_sigma_cutoff, args.moss_uniform_mix)
    sigma = torch.arange(1, NUM_TRAIN_TIMESTEPS + 1, dtype=torch.float64) / NUM_TRAIN_TIMESTEPS
    cdf = pmf.cumsum(0)
    median = float(sigma[int(torch.searchsorted(cdf, torch.tensor(0.5, dtype=torch.float64)))])
    return (f"timestep_sampling=moss: logit(sigma)~N({args.moss_logit_mean}, {args.moss_logit_std}^2), cutoff "
            f"{args.moss_sigma_cutoff}, uniform {args.moss_uniform_mix}; mean sigma {float((pmf * sigma).sum()):.3f}, "
            f"median {median:.3f}, P(sigma<0.2) {float(pmf[sigma < 0.2].sum()):.3f}")
