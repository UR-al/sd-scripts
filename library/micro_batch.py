# Token-budget micro-batching (2026-10-25): a dataset batch too large for VRAM as one forward is run as a few chunks of
# rows, each with its own forward + backward, the chunk's mean loss weighted by rows / batch rows. The accumulated
# gradient is the gradient of the batch's mean loss (sum_c (n_c / n) * mean_c == mean over the batch) for every
# per-sample loss and per-sample weighting (loss_weights, min-SNR, EDM2, masks, FFL, hf, static patch topology), so the
# optimizer step is the one the whole batch would take; what changes is how many rows share one forward (speed: a
# 16k-token chunk saturates the GPU and amortises the per-forward host floor; memory: one chunk's activations at a time).
# Options that couple the rows of one forward see the chunk, not the batch, and are refused or warned about by
# anima_train_network.validate_micro_batch_args (contrastive flow matching, adaptive timestep sampling, dwa / gradnorm
# patch-topology weighting, anchor / adv SNR gates: refused; antithetic / stratified / QMC timestep sampling,
# antithetic noise pairing, OT noise assignment: warned, they pair / stratify within each chunk).
# Used by NetworkTrainer.train through NetworkTrainer.split_batch_for_backward, which anima_train_network.py implements
# with --anima_micro_batch_max_tokens (its token count = batch rows x DiT tokens per sample). Multi-GPU: the chunks run
# inside the same accelerate accumulate() context as one micro-step would.
# --batch_loss_norm_rows N (rows_norm): with bucketed datasets a batch_size of 8 is rarely full (bucket_no_upscale: one
# bucket per aspect ratio), and a mean loss per batch would give the samples of a 2-row batch 4x the weight of a full
# batch's. Scaling each micro-step's mean loss by rows x gradient_accumulation_steps / N (accelerate divides every
# backward by gradient_accumulation_steps) makes every sample weigh exactly 1 / N in the optimizer step: N is the number
# of samples one optimizer step stands for (batch_size x gradient_accumulation_steps when the batches are full), so
# N = 8 reproduces batch 1 x accumulation 8 at any accumulation setting.
from typing import Any, List, Optional, Sequence, Tuple

import torch


def chunk_sizes(n: int, max_rows: int) -> List[int]:
    """Row counts of the chunks of an n-row batch with at most max_rows rows each, as even as possible (8 rows at most 5
    per chunk -> [4, 4], not [5, 3]); [n] when it fits."""
    if n <= 0:
        return []
    max_rows = max(1, int(max_rows))
    chunks = -(-n // max_rows)
    base, extra = divmod(n, chunks)
    return [base + 1] * extra + [base] * (chunks - extra)


def _row_tensor(v: Any, n: int) -> bool:
    return isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == n


def _split_value(v: Any, n: int, bounds: Sequence[Tuple[int, int]]) -> List[Any]:
    """The per-chunk views of one batch entry: a tensor with n rows is sliced; a list / tuple of row tensors (or None)
    (text_encoder_outputs_list, input_ids_list) is sliced element-wise; a list / tuple of n items (captions, flippeds,
    custom_attributes, image_keys) is sliced; anything else (None, scalars, tensors of another leading size) is shared."""
    if _row_tensor(v, n):
        return [v[a:b] for a, b in bounds]
    if isinstance(v, (list, tuple)) and len(v):
        if all(x is None or _row_tensor(x, n) for x in v) and any(x is not None for x in v):
            return [type(v)(None if x is None else x[a:b] for x in v) for a, b in bounds]
        if len(v) == n:
            return [type(v)(v[a:b]) for a, b in bounds]
    return [v] * len(bounds)


def split_batch(batch: dict, sizes: Sequence[int]) -> List[dict]:
    """The batch dict split into len(sizes) dicts of sizes[i] rows (sum(sizes) == the batch's rows). Row tensors are
    sliced views (no copy); entries that are not per-row are shared. The dicts are new, so a trainer may add keys to a
    chunk (masked_latents, addift_conditioning_latents) without touching the others."""
    sizes = [int(s) for s in sizes]
    if any(s <= 0 for s in sizes):
        raise ValueError(f"micro_batch.split_batch: chunk sizes must be positive, got {sizes}")
    n = sum(sizes)
    bounds, start = [], 0
    for s in sizes:
        bounds.append((start, start + s))
        start += s
    if len(sizes) == 1:
        return [batch]
    split = {k: _split_value(v, n, bounds) for k, v in batch.items()}
    return [type(batch)((k, parts[i]) for k, parts in split.items()) for i in range(len(sizes))]


def batch_rows(batch: Any) -> int:
    """Rows (samples) of a collate_fn batch dict: latents, else images, else loss_weights, else captions; 0 if unknown."""
    if not isinstance(batch, dict):
        return 0
    for key in ("latents", "images", "loss_weights"):
        v = batch.get(key)
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            return int(v.shape[0])
    caps = batch.get("captions")
    return len(caps) if isinstance(caps, (list, tuple)) else 0


def rows_norm(args: Any, batch: Any) -> float:
    """--batch_loss_norm_rows N: the factor rows x gradient_accumulation_steps / N the micro-step's mean loss is scaled
    by before accelerator.backward (which divides by gradient_accumulation_steps; DeepSpeed's engine does the same), so
    every sample weighs exactly 1 / N in the optimizer step whatever the bucket batch's fill (bucket_no_upscale leaves
    many buckets with fewer images than batch_size; with plain means a 2-row batch's samples would weigh 4x a full 8-row
    batch's). N = the samples one optimizer step stands for. 1.0 when off (N <= 0) or when the rows are unknown. The
    logged losses stay the plain means; step_func (Polyak) optimizers get the same scaled value as the gradient."""
    n = int(getattr(args, "batch_loss_norm_rows", 0) or 0)
    if n <= 0:
        return 1.0
    rows = batch_rows(batch)
    if rows <= 0:
        return 1.0
    accumulation = max(1, int(getattr(args, "gradient_accumulation_steps", 1) or 1))
    return rows * accumulation / n


def weighted_sum(acc: Optional[torch.Tensor], value: Optional[torch.Tensor], weight: float) -> Optional[torch.Tensor]:
    """acc + weight * value for the per-chunk logging values (None stays None; weight 1.0 adds the value itself)."""
    if value is None:
        return acc
    v = value if weight == 1.0 else value * weight
    return v if acc is None else acc + v
