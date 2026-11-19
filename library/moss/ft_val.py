# MOSS-Image full fine-tune (moss_train.py): a deterministic validation loss on a held-out set.
#
# Flags (ft_args): --moss_ft_val_dataset_config (a kohya dataset TOML of the held-out images; no validation without it),
# --moss_ft_val_every_n_steps (0 = only before the first step and after the last), --moss_ft_val_sigmas (fixed noise
# levels; default the 8 bin midpoints k/16, k odd), --moss_ft_val_seed (default 1234).
#
# What is fixed, so that two validations of an unchanged model give the same numbers bit for bit:
#   * the images and batches: every image of the TOML once (num_repeats ignored), in its bucket (make_buckets), batched
#     by (bucket, the TOML's batch size) in sorted image-key order - no shuffle, no seed dependence;
#   * the latents: the cache the training set uses (cache_latents(_to_disk), written at start-up with the training
#     latents), variant 0, never flipped; flip_aug / color_aug / gamma_aug / random_crop / resolution jitter of the TOML
#     are switched off for it (logged);
#   * the captions: kohya's process_caption without its draws - the FIRST line of a multi-line caption (with or without
#     enable_wildcard), the first {a|b} option with enable_wildcard, prefix / suffix, the keep_tokens(_separator) split
#     joined with ", " as the shuffled captions are, no shuffle, no tag dropout, no caption dropout, no token warmup
#     (library/moss/captions.canonical_caption on the first line);
#   * their text-encoder outputs: encoded once at start-up, each caption alone (the cached path's encoding; the live
#     encoder runs its pass with batch_rows 1 even under --moss_te_live_batch N), kept on the CPU for the run and stored
#     next to the TOML in <toml stem>_moss_val_te/ (keyed by the TE signature and the text, as the sample prompts), so a
#     cached run and a live run read the same rows and a second run encodes nothing;
#   * the noise: for image i at sigma s, torch.randn on the CPU from a generator seeded with
#     sha256(seed, the image's file name, s) - independent of the batch, the training RNG and the device;
#   * the loss: ft_loss.moss_ft_val_loss_rows - the training loss chain (target, weighting, loss_type, CFM, masked loss,
#     token mining, loss weights, min-SNR, loss_multiplier) at timestep 1000 x s for every sample, without the training
#     draws (no sigma sampling, no OT / antithetic noise, no ip_noise_gamma). Per sigma the mean over the images; the
#     overall loss is the mean over the sigmas.
#
# What a validation does not touch: it runs under torch.no_grad() (no autograd graph: the adamw_host / carousel gradient
# hooks never fire, checkpointing and the carousel's no-grad wrappers run their plain forwards), between optimizer steps
# (the window is drained), with the training forward's autocast and stream dtype. Python's / numpy's / torch's (CPU and
# CUDA) RNG states are saved and restored around it, the noise comes from private generators, the DataLoader, the live
# TE feed, the optimizer, the loss scaler, the LR scheduler and the sample / save schedule are not used. dit.train() stays
# (the DiT has no dropout; under no_grad the blocks skip checkpointing).
#
# Output per validation: one JSON line in <output_dir>/val_log.jsonl
#   {"step": int, "images_seen": int, "loss": float, "loss_by_sigma": {"<sigma>": float}, "n": int, "seconds": float}
# (step 0 = before any update; images_seen = the training images consumed so far; a non-finite loss - an fp16 forward
# overflow - is written as null, never as the non-JSON NaN / Infinity), the TensorBoard scalars val/loss and
# val/loss_s<sigma>, and one console line "VAL step N: loss X (n images, s s)". A fresh run (not --resume) moves an
# existing non-empty val_log.jsonl aside to val_log.<time>.jsonl (kept); a resumed run appends.

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from library import config_util, strategy_base
from library.config_util import BlueprintGenerator, ConfigSanitizer
from library.moss import captions as moss_captions
from library.moss import ft_args, ft_loss, strategy
from library.moss import train_utils as moss_train_utils
from library.moss.te_live_feed import live_outputs
from library.utils import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)

PREFIX = ft_args.PREFIX
VAL_LOG_FILE = ft_args.VAL_LOG_FILE
VAL_TE_SUFFIX = "_moss_val_te"  # <val TOML stem>_moss_val_te/: the validation captions' text-encoder outputs
ENCODE_ROWS = 64  # captions per text-encoder call at start-up (a live pass streams the layers once for all of them)
LATENT_STRIDE = 16
# training-only augmentations switched off for the validation set (num_repeats -> 1, resolution jitter -> none)
OFF_SUBSET_FLAGS = ("flip_aug", "color_aug", "gamma_aug", "random_crop")
JITTER_PARAMS = ("resolution_jitter_resolutions", "resolution_jitter_batch_sizes", "resolution_jitter_weights")
# caption draws of training that validation replaces by the fixed caption (only logged)
RANDOM_CAPTION_FLAGS = ("shuffle_caption", "caption_dropout_rate", "caption_dropout_every_n_epochs",
                        "caption_tag_dropout_rate", "token_warmup_step")


def _refuse(message: str):
    raise ValueError(PREFIX + message)


# ---- RNG ------------------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def preserved_rng():
    """Python's, numpy's and torch's global RNG states (CPU, and CUDA when it is initialised) as they were before the
    block: whatever the block draws (kohya's dataset set-up seeds and draws from `random`), training continues as if it
    had not run."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def noise_seed(seed: int, name: str, sigma_key: str) -> int:
    """The seed of one image's noise at one noise level: a function of --moss_ft_val_seed, the image's file name and
    the sigma only (not of the batch, the order or the device)."""
    digest = hashlib.sha256(f"moss-ft-val\x00{int(seed)}\x00{name}\x00{sigma_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def sigma_key(sigma: float) -> str:
    """The key of a noise level in loss_by_sigma and the TensorBoard tag val/loss_s<key>."""
    return f"{float(sigma):g}"


# ---- the dataset ----------------------------------------------------------------------------------------------------
def _deterministic_blueprint(group_blueprint) -> List[str]:
    """Switch the training-only augmentations of the validation TOML off (in place); returns what was on."""
    switched = set()
    for d, dataset_blueprint in enumerate(group_blueprint.datasets):
        params = dataset_blueprint.params
        if float(getattr(params, "validation_split", 0.0) or 0.0) > 0.0:
            _refuse(f"validation_split in --moss_ft_val_dataset_config (datasets[{d}]): that TOML is the held-out set"
                    " itself; remove validation_split")
        for name in JITTER_PARAMS:
            if getattr(params, name, None) is not None:
                switched.add("resolution jitter")
                setattr(params, name, None)
        for subset_blueprint in dataset_blueprint.subsets:
            subset = subset_blueprint.params
            if float(getattr(subset, "validation_split", 0.0) or 0.0) > 0.0:
                _refuse(f"validation_split in --moss_ft_val_dataset_config (a subset of datasets[{d}]): that TOML is"
                        " the held-out set itself; remove validation_split")
            if getattr(subset, "is_val", False):
                subset.is_val = False  # every image of this TOML is a validation image
            if int(getattr(subset, "num_repeats", 1) or 1) != 1:
                switched.add("num_repeats")
                subset.num_repeats = 1
            for name in OFF_SUBSET_FLAGS:
                if getattr(subset, name, False):
                    switched.add(name)
                    setattr(subset, name, False)
            if hasattr(subset, "random_crop_padding_percent"):
                subset.random_crop_padding_percent = 0.0
            for name in JITTER_PARAMS:
                if getattr(subset, name, None) is not None:
                    switched.add("resolution jitter")
                    setattr(subset, name, None)
    return sorted(switched)


def make_val_dataset_group(args, train_dataset_group=None):
    """The validation dataset group from --moss_ft_val_dataset_config (start-up step 4, after the training group): kohya's
    blueprint of that TOML with the training-only augmentations off, its buckets made. The global RNGs are restored
    afterwards (kohya draws the dataset seed and reseeds `random` in make_buckets)."""
    path = args.moss_ft_val_dataset_config
    logger.info(f"{PREFIX}validation: loading the held-out dataset config {path}")
    with preserved_rng():
        user_config = config_util.load_user_config(path)
        blueprint = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True)).generate(user_config, args)
        switched = _deterministic_blueprint(blueprint.dataset_group)
        group, inner = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    if inner is not None and (getattr(inner, "datasets", None) or []):
        _refuse("--moss_ft_val_dataset_config has validation subsets of its own: that TOML is the held-out set itself")
    images = sum(len(getattr(dataset, "image_data", None) or {}) for dataset in group.datasets)
    if images == 0:
        _refuse(f"--moss_ft_val_dataset_config {path!r} has no images (check its image_dir / caption_extension)")
    try:
        group.verify_bucket_reso_steps(16)
    except AssertionError as error:
        _refuse(f"--moss_ft_val_dataset_config: {error} (MOSS latents need bucket sizes in steps of 16)")
    if not group.is_latent_cacheable():
        _refuse("--moss_ft_val_dataset_config: its latents cannot be cached (color_aug / random_crop)")
    if switched:
        logger.info(f"{PREFIX}validation: {', '.join(switched)} of the held-out config switched off (every image once,"
                    " its cached latents unaugmented)")
    random_captions = sorted({name for dataset in group.datasets for subset in dataset.subsets
                              for name in RANDOM_CAPTION_FLAGS if getattr(subset, name, 0)})
    multi = sum(1 for dataset in group.datasets for info in dataset.image_data.values() if "\n" in (info.caption or ""))
    if random_captions or multi:
        logger.info(f"{PREFIX}validation: fixed captions"
                    + (f" ({', '.join(random_captions)} of the held-out config not applied)" if random_captions else "")
                    + (f"; {multi} multi-line captions use their first line" if multi else ""))
    if train_dataset_group is not None:
        train_paths = {os.path.normcase(os.path.abspath(info.absolute_path))
                       for dataset in getattr(train_dataset_group, "datasets", None) or []
                       for info in (getattr(dataset, "image_data", None) or {}).values()
                       if getattr(info, "absolute_path", None)}
        shared = sum(1 for dataset in group.datasets for info in dataset.image_data.values()
                     if os.path.normcase(os.path.abspath(info.absolute_path)) in train_paths)
        if shared:
            logger.warning(f"{PREFIX}validation: {shared} of the {images} held-out images are also training images: the"
                           " validation loss is not a held-out loss for them")
    return group


def cache_val_latents(group, vae, accelerator) -> None:
    """The validation latents through the training set's caching strategy (start-up step 6, the VAE on the device):
    *_moss.npz next to the images with cache_latents_to_disk, else in memory. The global RNGs are restored."""
    with preserved_rng():
        group.new_cache_latents(vae, accelerator)


# ---- the plan: images, captions, batches ----------------------------------------------------------------------------
@dataclass
class ValItem:
    key: str  # kohya's image key
    name: str  # the noise key: the image's file name ("#k" appended when names repeat)
    caption: str  # the fixed caption
    bucket_reso: Tuple[int, int]
    batch_size: int
    loss_weight: float
    alpha_mask: bool
    info: Any  # kohya's ImageInfo: latents / latents_npz / alpha_mask after caching

    @property
    def image_tokens(self) -> int:
        return (int(self.bucket_reso[0]) // LATENT_STRIDE) * (int(self.bucket_reso[1]) // LATENT_STRIDE)


def val_caption(dataset, subset, caption: Optional[str]) -> str:
    """The fixed caption of a validation image: the first line, then process_caption without its draws."""
    first = (caption or "").split("\n")[0]
    return moss_captions.canonical_caption(dataset, subset, first)


def _batch_size(dataset, subset) -> int:
    getter = getattr(dataset, "get_subset_batch_size", None)
    if getter is not None:
        return int(getter(subset))
    return max(1, int(getattr(dataset, "batch_size", 1) or 1))


def val_plan(group) -> Tuple[List[ValItem], List[List[int]]]:
    """(items, batches): every image once, batches of the TOML's batch size within one bucket, in a fixed order
    (pools sorted by bucket, batch size and alpha mask; images by key)."""
    items: List[ValItem] = []
    for dataset in group.datasets:
        prior = float(getattr(dataset, "prior_loss_weight", 1.0) or 1.0)
        for key in sorted(dataset.image_data):
            info = dataset.image_data[key]
            subset = dataset.image_to_subset[key]
            items.append(ValItem(key=str(key), name=os.path.basename(str(info.absolute_path or key)),
                                 caption=val_caption(dataset, subset, info.caption),
                                 bucket_reso=(int(info.bucket_reso[0]), int(info.bucket_reso[1])),
                                 batch_size=_batch_size(dataset, subset),
                                 loss_weight=prior if getattr(info, "is_reg", False) else 1.0,
                                 alpha_mask=bool(getattr(subset, "alpha_mask", False)), info=info))
    seen: Dict[str, int] = {}
    for item in sorted(items, key=lambda it: it.key):  # file names that repeat (two folders): #1, #2, ... by key
        count = seen.get(item.name, 0)
        seen[item.name] = count + 1
        if count:
            item.name = f"{item.name}#{count}"
    pools: Dict[tuple, List[int]] = {}
    for index, item in enumerate(items):
        pools.setdefault((item.bucket_reso, item.batch_size, item.alpha_mask), []).append(index)
    batches: List[List[int]] = []
    for pool in sorted(pools):
        indices = sorted(pools[pool], key=lambda i: items[i].key)
        size = pool[1]
        batches.extend(indices[start:start + size] for start in range(0, len(indices), size))
    return items, batches


def val_batch_shapes(group) -> List[Tuple[int, int]]:
    """(image tokens, batch rows) of every validation batch (the budget's input)."""
    items, batches = val_plan(group)
    return sorted({(items[batch[0]].image_tokens, len(batch)) for batch in batches})


def val_texts(group) -> List[str]:
    return list(dict.fromkeys(item.caption for item in val_plan(group)[0]))


def check_val_latents(items: Sequence[ValItem]) -> None:
    """Every validation image has cached latents (in memory or an npz file); refused otherwise."""
    missing = [item.key for item in items
               if getattr(item.info, "latents", None) is None
               and not (getattr(item.info, "latents_npz", None) and os.path.isfile(item.info.latents_npz))]
    if missing:
        _refuse(f"validation: {len(missing)} held-out image(s) have no cached latents (e.g. {missing[0]}): the"
                " validation set uses the latent cache like the training set (cache_latents = true, cache_latents_to_disk"
                " = true writes *_moss.npz next to the images at start-up); check that their folder is writable")


# ---- the text-encoder outputs ---------------------------------------------------------------------------------------
def val_text_cache_dir(args) -> str:
    return os.path.splitext(os.path.abspath(args.moss_ft_val_dataset_config))[0] + VAL_TE_SUFFIX


@contextlib.contextmanager
def _rows_alone(encoder):
    """The live encoder runs every row alone during the block (--moss_te_live_batch N > 1 would pad and batch them:
    close to, not bit-identical with, the cached encoding)."""
    if not hasattr(encoder, "batch_rows") or getattr(encoder, "batch_rows", 1) == 1:
        yield
        return
    previous = encoder.batch_rows
    encoder.batch_rows = 1
    try:
        yield
    finally:
        encoder.batch_rows = previous


def encode_val_captions(args, texts: Iterable[str], encoder) -> Dict[str, torch.Tensor]:
    """{caption: hidden states [tokens, hidden] on the CPU} for the validation captions (start-up step 6). Each is read
    from <val TOML stem>_moss_val_te/ (keyed by the TE signature and the text) or encoded and written there: the missing
    ones in passes of up to ENCODE_ROWS captions, each caption alone and unpadded (the cached path's encoding, also
    through the live encoder). encoder: the live encoder (--moss_te_live) or the cached path's provider (it loads the
    9B only if a caption is missing). The global RNGs are restored."""
    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
    encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
    signature = strategy.moss_te_signature(args.moss_text_encoder, args.moss_max_token_length, args.moss_tokenizer_dir)
    cache_dir = val_text_cache_dir(args)
    rows: Dict[str, torch.Tensor] = {}
    missing: List[str] = []
    for text in dict.fromkeys(texts):
        cached = moss_train_utils.load_cached_sample_prompt(
            moss_train_utils.sample_prompt_cache_path(cache_dir, signature, text))
        if cached is None:
            missing.append(text)
            continue
        embeds, mask = cached
        rows[text] = embeds[0, : int(mask[0].sum())].contiguous()
    start = time.perf_counter()
    with preserved_rng(), torch.no_grad():
        for first in range(0, len(missing), ENCODE_ROWS):
            chunk = missing[first:first + ENCODE_ROWS]
            tokens = tokenize_strategy.tokenize(chunk)
            with _rows_alone(encoder):
                embeds, mask = encoding_strategy.encode_tokens(tokenize_strategy, [encoder], tokens)
            for i, text in enumerate(chunk):
                length = int(mask[i].sum())
                row = embeds[i, :length].to("cpu").contiguous()
                moss_train_utils.save_cached_sample_prompt(
                    moss_train_utils.sample_prompt_cache_path(cache_dir, signature, text),
                    [row.unsqueeze(0), mask[i:i + 1, :length].to("cpu")])
                rows[text] = row
    held = sum(row.numel() * row.element_size() for row in rows.values())
    logger.info(f"{PREFIX}validation: {len(rows)} captions ready ({len(rows) - len(missing)} from {cache_dir},"
                f" {len(missing)} encoded in {time.perf_counter() - start:.1f} s); {held / (1 << 20):.1f} MiB of text"
                " outputs kept on the CPU")
    return rows


# ---- the validation -------------------------------------------------------------------------------------------------
@dataclass
class ValRecord:
    step: int
    images_seen: int
    loss: float
    loss_by_sigma: Dict[str, float]
    n: int
    seconds: float

    def as_json(self) -> Dict[str, Any]:
        """The contract's keys, in its order; a non-finite float (an fp16 overflow: NaN / inf) becomes None (JSON null),
        since NaN / Infinity are not JSON and break strict readers (JS JSON.parse, jq)."""
        return {"step": self.step, "images_seen": self.images_seen, "loss": _finite_or_none(self.loss),
                "loss_by_sigma": {key: _finite_or_none(value) for key, value in self.loss_by_sigma.items()},
                "n": self.n, "seconds": _finite_or_none(self.seconds)}


def _finite_or_none(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


class Validator:
    """The validation of one run: the fixed plan, the text rows, the noise seeds, the log file."""

    def __init__(self, args, group, text_rows: Dict[str, torch.Tensor], *, resume: bool = False,
                 latents_strategy=None, text_encoding_strategy=None):
        self.args = args
        self.items, self.batches = val_plan(group)
        check_val_latents(self.items)
        missing = [item.caption for item in self.items if item.caption not in text_rows]
        if missing:
            raise ValueError(f"{PREFIX}validation: no text-encoder output for the caption {missing[0][:80]!r}")
        self.text_rows = text_rows
        self.sigmas = ft_args.parse_val_sigmas(getattr(args, "moss_ft_val_sigmas", None))
        self.keys = [sigma_key(sigma) for sigma in self.sigmas]
        self.seed = int(getattr(args, "moss_ft_val_seed", 1234))
        self.every = int(getattr(args, "moss_ft_val_every_n_steps", 0) or 0)
        self.latents_strategy = latents_strategy or strategy_base.LatentsCachingStrategy.get_strategy()
        self.text_encoding_strategy = text_encoding_strategy or strategy_base.TextEncodingStrategy.get_strategy()
        self.seeds = [[noise_seed(self.seed, item.name, key) for key in self.keys] for item in self.items]
        self.path = os.path.abspath(os.path.join(args.output_dir, VAL_LOG_FILE))
        self.last_step: Optional[int] = None
        self.records: List[ValRecord] = []
        self._prepare_log(resume)
        shapes = sorted({(self.items[batch[0]].bucket_reso, len(batch)) for batch in self.batches})
        logger.info(f"{PREFIX}validation: {len(self.items)} images in {len(self.batches)} batches"
                    f" ({', '.join(f'{w}x{h} b{b}' for (w, h), b in shapes)}) x {len(self.sigmas)} noise levels ="
                    f" {len(self.batches) * len(self.sigmas)} no-grad forwards per validation"
                    + (f" every {self.every} steps" if self.every else " at step 0 and after the last step")
                    + f"; log {self.path}")

    @property
    def n(self) -> int:
        return len(self.items)

    def _prepare_log(self, resume: bool) -> None:
        if resume or not os.path.isfile(self.path) or os.path.getsize(self.path) == 0:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(self.path)))
        kept = os.path.join(os.path.dirname(self.path), f"val_log.{stamp}.jsonl")
        suffix = 1
        while os.path.exists(kept):
            kept = os.path.join(os.path.dirname(self.path), f"val_log.{stamp}-{suffix}.jsonl")
            suffix += 1
        os.replace(self.path, kept)
        logger.info(f"{PREFIX}validation: the previous run's {VAL_LOG_FILE} was moved to {kept} (kept; this run starts a"
                    " new log, a resumed run appends)")

    def due(self, step: int, max_steps: int) -> bool:
        """After optimizer step `step`: every N steps, and after the last step."""
        step = int(step)
        return step >= int(max_steps) or (self.every > 0 and step % self.every == 0)

    def noise(self, index: int, k: int, shape) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seeds[index][k])
        return torch.randn(tuple(shape), generator=generator, dtype=torch.float32)

    def _latents(self, item: ValItem):
        info = item.info
        if getattr(info, "latents", None) is not None:
            return info.latents, getattr(info, "alpha_mask", None)
        latents, _, _, _, alpha, _ = self.latents_strategy.load_latents_from_disk(info.latents_npz, item.bucket_reso,
                                                                                  variant=0)
        return torch.FloatTensor(latents), (None if alpha is None else torch.FloatTensor(alpha))

    def batch(self, indices: Sequence[int], device) -> dict:
        """The batch dict moss_ft_val_loss_rows reads, in the training batches' form: latents (CPU), the text outputs as
        the live feed / the cached path give them (padded embeds, float mask, zero dropout rates), loss weights and
        alpha masks on the device."""
        latents, alphas, rows = [], [], []
        for index in indices:
            item = self.items[index]
            item_latents, alpha = self._latents(item)
            latents.append(item_latents)
            alphas.append(alpha)
            rows.append(self.text_rows[item.caption])
        alpha_masks = None
        if any(alpha is not None for alpha in alphas):
            like = next(alpha for alpha in alphas if alpha is not None)
            alpha_masks = torch.stack([torch.ones_like(like) if alpha is None else alpha for alpha in alphas]).to(device)
        weights = torch.tensor([self.items[i].loss_weight for i in indices], dtype=torch.float32, device=device)
        return {"latents": torch.stack(latents), "text_encoder_outputs_list": live_outputs(rows, device),
                "loss_weights": weights, "alpha_masks": alpha_masks}

    def run(self, accelerator, dit, step: int, images_seen: int, *, stream_dtype: torch.dtype,
            noise_scheduler) -> ValRecord:
        """One validation of the current weights (after an optimizer step's drain, or before the first)."""
        start = time.perf_counter()
        device = accelerator.device
        sums: List[Optional[torch.Tensor]] = [None] * len(self.sigmas)
        last_text_tokens = getattr(dit, "last_text_tokens", None)
        with preserved_rng(), torch.no_grad():
            try:
                for indices in self.batches:
                    batch = self.batch(indices, device)
                    shape = batch["latents"].shape[1:]
                    for k, sigma in enumerate(self.sigmas):
                        noise = torch.stack([self.noise(index, k, shape) for index in indices])
                        rows = ft_loss.moss_ft_val_loss_rows(self.args, dit, batch, accelerator, noise, sigma,
                                                             noise_scheduler, stream_dtype,
                                                             self.text_encoding_strategy)
                        total = rows.sum()
                        sums[k] = total if sums[k] is None else sums[k] + total
            finally:
                if hasattr(dit, "last_text_tokens"):
                    dit.last_text_tokens = last_text_tokens  # the training timer reads the last training forward's
        totals = torch.stack([total.to(torch.float64) for total in sums]).cpu().tolist()  # one device sync
        by_sigma = {key: float(total) / self.n for key, total in zip(self.keys, totals)}
        loss = float(sum(by_sigma.values()) / len(by_sigma))
        seconds = time.perf_counter() - start
        record = ValRecord(step=int(step), images_seen=int(images_seen), loss=loss, loss_by_sigma=by_sigma, n=self.n,
                           seconds=round(seconds, 3))
        self._write(record)
        accelerator.log({"val/loss": loss, **{f"val/loss_s{key}": value for key, value in by_sigma.items()}},
                        step=int(step))
        logger.info(f"VAL step {int(step)}: loss {loss:.6f} ({self.n} images, {seconds:.1f} s)")
        bad = [key for key, value in by_sigma.items() if not math.isfinite(value)]
        if bad:
            logger.warning(f"{PREFIX}validation at step {int(step)}: non-finite loss at sigma {', '.join(bad)} (an fp16"
                           " forward overflow, or diverged weights); training goes on")
        self.last_step = int(step)
        self.records.append(record)
        return record

    def _write(self, record: ValRecord) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.as_json(), allow_nan=False) + "\n")  # strict JSON: non-finite -> null
            f.flush()


def read_val_log(path: str) -> List[Dict[str, Any]]:
    """The records of a val_log.jsonl (the last one per step wins when a resumed run appended a step again)."""
    by_step: Dict[int, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                by_step[int(record["step"])] = record
    return [by_step[step] for step in sorted(by_step)]
