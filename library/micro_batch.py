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
# --step_rows N (2026-10-25 night): the optimizer step is N rows, not gradient_accumulation_steps bucket batches. Bucketed
# datasets give ragged batches (725 images: 184 batches of 1..8 rows), so batch 8 x accumulation 2 made steps of 2..16
# rows whose gradient magnitude swings with the fill (every row 1/8, so a 16-row step is twice a batch 1 x accumulation 8
# step, a 4-row step half). RowPacker wraps the prepared DataLoader and packs consecutive bucket batches into StepPacks of
# exactly N rows: a batch crossing the step boundary is cut there (slice_batch) and its tail opens the next pack, the
# epoch's last pack holds the remainder. The trainer runs a pack's sub-batches (each further chunked by the token budget)
# inside one accelerate step with gradient_accumulation_steps 1, every row weighted 1 / N (rows_norm reads N from
# step_rows): the step size and the per-sample weight of batch 1 x accumulation N whatever the buckets, and the same rows
# per step on every dataset without retuning the accumulation. Steps per epoch = ceil(rows per epoch / N), the rows
# counted from the datasets' bucket batch lists (dataset_rows_per_epoch), so max_train_steps and the LR schedule see the
# real count. Single process only (a shard's pack count would be an estimate); resolution jitter (the batch list is
# resampled every epoch) is refused. What still differs from batch 1 x accumulation N: the N rows come from 2-3 buckets
# (same tier and aspect ratio, repeats together) instead of N independent draws.
import logging
import math
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


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


def _split_value(v: Any, n: int, bounds: Sequence[Tuple[int, int]], copy: bool = False) -> List[Any]:
    """The per-chunk views of one batch entry: a tensor with n rows is sliced; a list / tuple of row tensors (or None)
    (text_encoder_outputs_list, input_ids_list) is sliced element-wise; a list / tuple of n items (captions, flippeds,
    custom_attributes, image_keys) is sliced; anything else (None, scalars, tensors of another leading size) is shared.
    copy=True clones the row-tensor slices, so the piece no longer pins the parent batch's storage."""
    if _row_tensor(v, n):
        return [v[a:b].clone() if copy else v[a:b] for a, b in bounds]
    if isinstance(v, (list, tuple)) and len(v):
        if all(x is None or _row_tensor(x, n) for x in v) and any(x is not None for x in v):
            return [type(v)(None if x is None else (x[a:b].clone() if copy else x[a:b]) for x in v) for a, b in bounds]
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
    """Rows (samples) of a collate_fn batch dict: latents, else images, else loss_weights, else captions; 0 if unknown.
    A StepPack: its rows."""
    if isinstance(batch, StepPack):
        return batch.rows
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
    logged losses stay the plain means; step_func (Polyak) optimizers get the same scaled value as the gradient.
    --step_rows N sets N too (gradient_accumulation_steps is 1 then): a full pack gives 1.0, the epoch's last, shorter
    pack rows / N - its rows still weigh 1 / N each."""
    n = step_rows(args) or int(getattr(args, "batch_loss_norm_rows", 0) or 0)
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


# ---------------------------------------------------------------------------------------------------------------------
# --step_rows: N-row optimizer steps packed from the bucket batches
# ---------------------------------------------------------------------------------------------------------------------


class StepPack:
    """One optimizer step's rows under --step_rows: the collate_fn batches (or row slices of them, slice_batch) whose rows
    add up to the step; the epoch's last pack may hold fewer. Deliberately not a dict: code that reads batch["latents"]
    has to go through sub_batches() / batch_rows(), so a pack can never be mistaken for one bucket batch."""

    __slots__ = ("sub_batches", "rows")

    def __init__(self, sub_batches: Iterable[dict]):
        self.sub_batches: List[dict] = list(sub_batches)
        if not self.sub_batches:
            raise ValueError("StepPack needs at least one sub-batch")
        self.rows = sum(batch_rows(b) for b in self.sub_batches)

    def __iter__(self) -> Iterator[dict]:
        return iter(self.sub_batches)

    def __len__(self) -> int:
        return len(self.sub_batches)

    def __repr__(self) -> str:
        return f"StepPack(rows={self.rows}, sub_batches={[batch_rows(b) for b in self.sub_batches]})"


def sub_batches(batch: Any) -> List[Any]:
    """The collate_fn batches of one training-loop item: [batch] for a plain batch, a StepPack's sub-batches."""
    return list(batch.sub_batches) if isinstance(batch, StepPack) else [batch]


def slice_batch(batch: dict, start: int, stop: int, copy: bool = False) -> dict:
    """Rows [start, stop) of a collate_fn batch as a new dict of sliced views (split_batch's rules for every entry); the
    batch itself when the range is all of its rows. copy=True clones the row tensors instead of viewing them (the
    RowPacker's cut pieces: a view would keep the whole bucket batch on the device for the next optimizer step)."""
    n = batch_rows(batch)
    if not (0 <= start < stop <= n):
        raise ValueError(f"micro_batch.slice_batch: rows [{start}, {stop}) outside a {n}-row batch")
    if start == 0 and stop == n:
        return batch
    bounds = [(start, stop)]
    return type(batch)((k, _split_value(v, n, bounds, copy)[0]) for k, v in batch.items())


def step_rows(args: Any) -> int:
    """--step_rows N, 0 when off or absent (other trainers' args)."""
    return int(getattr(args, "step_rows", 0) or 0)


def validate_step_rows(args: Any, num_processes: int = 1) -> int:
    """--step_rows N: N >= 0; with N > 0 gradient_accumulation_steps must be 1 (the N-row pack is the optimizer step),
    --batch_loss_norm_rows 0 or N (rows_norm reads N from step_rows) and a single process (the packs are formed from
    this process's batch stream; a shard's step count per epoch would be an estimate). Returns N (0 = off)."""
    n = step_rows(args)
    if n < 0:
        raise ValueError(f"--step_rows must be >= 0 (0 = off), got {n}")
    if n == 0:
        return 0
    accumulation = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
    if accumulation != 1:
        raise ValueError(f"--step_rows {n} needs --gradient_accumulation_steps 1: the {n}-row pack is the optimizer step"
                         f" (for more rows per step raise step_rows), got {accumulation}")
    norm = int(getattr(args, "batch_loss_norm_rows", 0) or 0)
    if norm not in (0, n):
        raise ValueError(f"--batch_loss_norm_rows {norm} contradicts --step_rows {n}: every row weighs 1 / step_rows;"
                         f" leave --batch_loss_norm_rows at 0")
    if int(num_processes) > 1:
        raise ValueError(f"--step_rows runs on a single process, got {num_processes}: the packs are formed from this"
                         f" process's batch stream and the steps per epoch of a shard would be an estimate")
    return n


# Options that couple the rows of one forward. When an optimizer step runs as several forwards (a token-budget chunk or a
# --step_rows pack is not the batch) the refused ones change the objective; the warned ones stay unbiased, but their
# pairing / strata / OT matching form within each forward (none in a 1-row one). Used by
# anima_train_network.validate_micro_batch_args (the budget) and step_rows_messages (--step_rows, every network trainer).
ROW_COUPLING_REFUSED = (
    ("contrastive_flow_matching", "contrastive flow matching rolls its negatives within the forward (a 1-row chunk has none)"),
    ("adaptive_timestep_sampling", "the adaptive sampler's Algorithm 2 / REINFORCE update would see the last chunk only"),
    ("anchor_snr_weighting", "the anchor SNR gate is normalised to batch-mean 1 over the forward"),
    ("adv_snr_weighting", "the adv SNR gate is normalised to batch-mean 1 over the forward"),
)
ROW_COUPLING_WARNED = ("antithetic_timestep_sampling", "antithetic_noise_pairing", "stratified_timestep_sampling",
                       "qmc_timestep_sampling", "flow_use_ot")


def row_coupling_messages(args: Any, cause: str) -> List[str]:
    """The row-coupling checks for an option that makes one optimizer step several forwards (`cause`: "--step_rows 8" or
    "--anima_micro_batch_max_tokens 16384"): ValueError for the refused options, a warning text per warned one."""
    flag = cause.split()[0]
    for name, why in ROW_COUPLING_REFUSED:
        if getattr(args, name, False):
            raise ValueError(f"--{name} cannot be combined with {cause}: {why}. Drop the option or switch {flag} off")
    dyn = str(getattr(args, "patch_topology_dynamic_weighting", "none") or "none").lower()
    if getattr(args, "patch_topology_loss", False) and dyn != "none":
        raise ValueError(f"--patch_topology_dynamic_weighting {dyn} cannot be combined with {cause}: the weight would be"
                         f" updated per chunk (forward), not per optimizer step. Use 'none' or switch {flag} off")
    return [f"--{name} with {cause}: the pairing / strata / matching are formed within each chunk (one forward; none in a"
            f" 1-row chunk); the marginal timestep / noise distribution is unchanged"
            for name in ROW_COUPLING_WARNED if getattr(args, name, None)]


def step_rows_messages(args: Any, num_processes: int = 1) -> List[str]:
    """validate_step_rows plus, with --step_rows on, the row-coupling checks (a pack runs as one forward per bucket
    batch or boundary piece): ValueError for the refused options, the warning texts for the others. [] when off."""
    n = validate_step_rows(args, num_processes)
    return row_coupling_messages(args, f"--step_rows {n}") if n else []


def steps_per_epoch(rows_per_epoch: int, rows_per_step: int) -> int:
    """ceil(rows / step_rows): the optimizer steps one epoch of packs takes (the last one may be shorter)."""
    if rows_per_step <= 0:
        raise ValueError(f"step_rows must be > 0, got {rows_per_step}")
    return -(-int(rows_per_epoch) // int(rows_per_step))


def dataset_rows_per_epoch(dataset_group: Any) -> int:
    """Rows (images x repeats) the DataLoader yields in one epoch: the sum over the datasets' bucket batch lists
    (BucketBatchIndex -> min(bucket_batch_size, the bucket's remaining images)), the stream the packs are cut from. A
    dataset without that structure counts its num_train_images. Resolution jitter resamples the batch list every epoch
    (train_util.shuffle_buckets: random.choices with weights), so the rows per epoch would vary: refused."""
    datasets = getattr(dataset_group, "datasets", None)
    if not datasets:
        datasets = [dataset_group]
    total = 0
    for ds in datasets:
        delegate = getattr(ds, "dreambooth_dataset_delegate", None)  # FineTuning / ControlNet datasets delegate
        src = ds if delegate is None else delegate
        if getattr(src, "has_resolution_jitter", False):
            raise ValueError("--step_rows cannot be combined with resolution jitter: its batch list is resampled every"
                             " epoch, so the rows (and steps) per epoch are not fixed")
        indices = getattr(src, "all_buckets_indices", None)
        if indices is None:
            indices = getattr(src, "buckets_indices", None)
        buckets = getattr(src, "batch_buckets", None)
        if indices is None or buckets is None:
            total += int(getattr(ds, "num_train_images", 0) or 0)
            continue
        for bi in indices:
            remaining = len(buckets[bi.bucket_index]) - bi.batch_index * bi.bucket_batch_size
            total += max(0, min(int(bi.bucket_batch_size), remaining))
    return total


class RowPacker:
    """Iterable of StepPacks of exactly rows_per_step rows over a (prepared) DataLoader of bucket batches: consecutive
    batches fill a pack, a batch crossing the step boundary is cut there (slice_batch) and its tail opens the next pack,
    the epoch's final pack holds the remainder. len() = ceil(rows_per_epoch / rows_per_step), the optimizer steps per
    epoch (a mismatch with the packs actually formed is logged as a warning at the end of the epoch). skip_first_packs(k)
    is the resume path (accelerate's skip_first_batches needs a DataLoader): the first k packs are formed from the
    loader's stream and dropped, so the epoch keeps the data order it would have had."""

    def __init__(self, dataloader: Any, rows_per_step: int, rows_per_epoch: int):
        if int(rows_per_step) <= 0:
            raise ValueError(f"RowPacker: rows_per_step must be > 0, got {rows_per_step}")
        self.dataloader = dataloader
        self.rows_per_step = int(rows_per_step)
        self.rows_per_epoch = int(rows_per_epoch)

    def __len__(self) -> int:
        return steps_per_epoch(self.rows_per_epoch, self.rows_per_step)

    def __iter__(self) -> Iterator[StepPack]:
        return self.packs(0)

    def skip_first_packs(self, k: int) -> "_SkippedPacks":
        return _SkippedPacks(self, int(k))

    def packs(self, skip: int) -> Iterator[StepPack]:
        pending: List[dict] = []
        filled = 0
        yielded = 0
        for batch in self.dataloader:
            n = batch_rows(batch)
            if n <= 0:
                raise ValueError("--step_rows: a batch without a known row count (no latents / images / loss_weights /"
                                 " captions entry)")
            start = 0
            while start < n:
                take = min(n - start, self.rows_per_step - filled)
                stop = start + take
                # a cut piece is a copy (copy=True): a view would keep the whole bucket batch (already on the device
                # after accelerate's placement) alive through the next optimizer step; a batch used whole is passed as is
                pending.append(batch if (start == 0 and stop == n) else slice_batch(batch, start, stop, copy=True))
                filled += take
                start = stop
                if start == n:
                    batch = None  # the generator frame must not pin the parent batch while the consumer runs the step
                if filled == self.rows_per_step:
                    pack = StepPack(pending)
                    pending, filled = [], 0
                    if yielded >= skip:
                        yield pack
                    yielded += 1
        if pending:
            pack = StepPack(pending)
            if yielded >= skip:
                yield pack
            yielded += 1
        expected = len(self)
        if yielded != expected:
            logger.warning(f"--step_rows: {yielded} optimizer steps this epoch, {expected} expected from"
                           f" {self.rows_per_epoch} rows / epoch; max_train_steps and the LR schedule assumed {expected}")


class _SkippedPacks:
    """RowPacker.skip_first_packs(k): the epoch's packs after the first k, with len() = the remaining packs - like the
    DataLoader accelerate's skip_first_batches returns (train_util.calculate_val_loss_check reads len() of what the loop
    iterates). Re-iterable; every iteration forms and drops the first k packs again."""

    def __init__(self, packer: RowPacker, k: int):
        self.packer = packer
        self.k = max(0, int(k))

    def __len__(self) -> int:
        return max(0, len(self.packer) - self.k)

    def __iter__(self) -> Iterator[StepPack]:
        return self.packer.packs(self.k)
