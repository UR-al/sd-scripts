# MOSS-Image full fine-tune ("algo=full"): every base weight of the 5.08B DiT is trained.
#
# Design: docs/moss_full_finetune_design.md (sections 3 and 4); user guide: docs/moss_train.md; deviations and details:
# docs/moss_ft_impl_notes.md. The skeleton mirrors anima_train.py; the parts live in library/moss/ft_*.py:
#   * optimizer modes (ft_args.optimizer_mode): adamw_host (default) and came_host keep the fp32 master, the moments
#     and, for GA > 1 or exact clipping, a gradient accumulator in host RAM; the GPU holds only the compute copy. The
#     gradients stream to the host inside the backward and the window's final micro-step steps every parameter on the
#     CPU as its gradient arrives (ft_host.GradPipeline, ft_optim). adafactor_fused (Adafactor + --fused_backward_pass)
#     steps every parameter in its gradient hook on the GPU (FusedAdafactorSteps below, anima_train.py's hooks).
#     carousel (--moss_carousel, ft_carousel.py, docs/moss_ft_carousel_plan.md): only K rotating blocks (+ the small
#     always group) are trainable, with fp32 master, accumulator and AdamW / AdamW8bit / CAME state on the GPU; the
#     host keeps every fp32 master (and the inactive units' states); every period steps a new set is switched in.
#   * start-up order (design 4.3): the 9B text encoder and the VAE are used for caching before the host pools and the
#     DiT exist, so their memory peaks never overlap.
#   * the trained DiT is saved in the Base layout (ft_model.save_moss_dit); training states in their own atomic format
#     (ft_state), resumable at an epoch end (exactly) or mid-epoch (the epoch is reshuffled, as kohya does).
#   * fp16 compute (mixed_precision "fp16" + --moss_stream_fp32, docs/moss_ft_te_live_fp16_plan.md part B): the loss is
#     multiplied by ft_scaler.DynamicLossScaler's power-of-two scale before backward() and the host step divides it out
#     (StepContext.inv_grad_scale). Accelerate builds its own GradScaler for fp16; it is never used: this script calls
#     loss.backward() itself (never accelerator.backward) and steps on the host, so accelerator.scaler stays idle.
#   * validation loss (--moss_ft_val_dataset_config, library/moss/ft_val.py): a held-out set at fixed noise levels with
#     fixed noise and captions, under no_grad between optimizer steps (step 0, every --moss_ft_val_every_n_steps, the
#     last step); <output_dir>/val_log.jsonl, val/loss and val/loss_s<sigma>.

import argparse
import gc
import math
import os
import time
from dataclasses import dataclass
from multiprocessing import Value
from typing import Any, Dict, List, Optional, Tuple

import toml
import torch
from tqdm import tqdm

from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

from library import (attention, config_util, cuda_memory, sd3_train_utils, strategy_base, tier_timing, timestep_presets,
                     train_util)
from library.config_util import BlueprintGenerator, ConfigSanitizer
from library.moss import flow, ft_args, ft_budget, ft_carousel, ft_host, ft_loss, ft_model, ft_scaler, ft_state, strategy
from library.moss import ft_val, te_cache
from library.moss import te_live, te_live_feed, te_live_weights
from library.moss import train_utils as moss_train_utils
from library.moss.ft_optim import HostAdamW, HostCAME
from library.moss.models import enable_elementwise_compile, set_ckpt_reuse
from library.moss.vae import load_moss_vae
from library.utils import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)

PREFIX = ft_args.PREFIX
MIB = 1 << 20
GIB = 1 << 30
LATENT_STRIDE = strategy.MossLatentsCachingStrategy.STRIDE  # one latent pixel = one DiT token (patch 1)
# kohya's CUDA allocator controls (library/cuda_memory.py): not applied here - no VRAM cap and no cache flush, an OOM
# must surface (design 2.1)
IGNORED_CUDA_MEMORY_OPTIONS = ("cuda_memory_fraction", "cuda_empty_cache_every_n_steps", "cuda_empty_cache_above_gib")


def setup_parser() -> argparse.ArgumentParser:
    """moss_train.py's parser: the composition of design 4.2 and the --moss_ft_* arguments of 4.5
    (library/moss/ft_args.setup_parser, with this script's defaults)."""
    return ft_args.setup_parser()


# ---- loop helpers ---------------------------------------------------------------------------------------------------
def window_of(index: int, num_batches: int, gradient_accumulation_steps: int) -> Tuple[int, int, int]:
    """(window w, micro-step j, window size n_micro) of the 0-based batch `index` of an epoch of `num_batches`
    batches: windows of GA batches, the epoch's last one possibly shorter (design 3.1 "Partial windows"). j ==
    n_micro - 1 is the window's final micro-step; every micro-step's loss is divided by its window's n_micro."""
    ga = int(gradient_accumulation_steps)
    w, j = divmod(int(index), ga)
    return w, j, min(ga, int(num_batches) - w * ga)


def progress_after_window(*, global_step: int, epoch: int, consumed: int, num_batches: int, steps_per_epoch: int,
                          gradient_accumulation_steps: int, lr_schedule: Optional[dict],
                          images_seen: Optional[int] = None) -> ft_state.TrainProgress:
    """Where training stands after the window that ended with the `consumed`-th batch of the 0-based `epoch`. An epoch's
    last window is that epoch's end (completed_epochs = epoch + 1, epoch_step 0), so a state written there - a step
    state too - resumes at the start of the next epoch instead of skipping every batch of a finished one. images_seen:
    the training images consumed so far (recorded so a resumed run's validation log continues it)."""
    if int(consumed) >= int(num_batches):
        epoch, consumed = int(epoch) + 1, 0
    return ft_state.TrainProgress(global_step=int(global_step), completed_epochs=int(epoch), epoch_step=int(consumed),
                                  steps_per_epoch=int(steps_per_epoch),
                                  gradient_accumulation_steps=int(gradient_accumulation_steps), lr_schedule=lr_schedule,
                                  images_seen=None if images_seen is None else int(images_seen))


def resumed_images_seen(progress: ft_state.TrainProgress, images_per_epoch: int, num_batches: int) -> int:
    """The training images consumed before a resumed run: the state's count, or (a state written before it was
    recorded) the epochs done x the images per epoch + the batches of the current epoch x the mean batch."""
    if progress.images_seen is not None:
        return int(progress.images_seen)
    estimate = int(progress.completed_epochs) * int(images_per_epoch)
    estimate += int(round(int(progress.epoch_step) * int(images_per_epoch) / max(1, int(num_batches))))
    logger.info(f"{PREFIX}the state does not record the images seen: the validation log continues from an estimate of"
                f" {estimate} ({progress.completed_epochs} epochs x {images_per_epoch} images + {progress.epoch_step}"
                " batches)")
    return estimate


def epoch_save_due(args, epoch_no: int, num_train_epochs: int) -> bool:
    """kohya's epoch save (train_util.save_sd_model_on_epoch_end_or_stepwise_common): every save_every_n_epochs
    epochs, never at the last epoch (the train-end save is the final model)."""
    every = getattr(args, "save_every_n_epochs", None)
    return bool(every) and epoch_no % int(every) == 0 and epoch_no < num_train_epochs


def step_save_due(args, global_step: int) -> bool:
    every = getattr(args, "save_every_n_steps", None)
    return bool(every) and global_step % int(every) == 0


def phase_state_due(args, epoch_no: int, num_train_epochs: int) -> bool:
    """The epoch state at the end of every LR phase but the last is pinned by the keep policy (design 4.6 / 4.7): it is
    written with save_state even when save_every_n_epochs misses that epoch, so phase 2 can be re-run from it. Not with
    --moss_ft_no_keep_phase_states; a fractional phase end has no epoch state."""
    phases = getattr(args, "moss_ft_lr_phases", None)
    if not (getattr(args, "save_state", False) and phases) or getattr(args, "moss_ft_no_keep_phase_states", False):
        return False
    return epoch_no < num_train_epochs and epoch_no in ft_state.pinned_phase_epochs(phases)


def window_context(optimizer, step_index: int, inv_grad_scale: float = 1.0) -> ft_host.StepContext:
    """The StepContext of a window: the host optimizers' hyperparameter snapshot (GradPipeline's worker reads only it)
    and 1 / the fp16 loss scale of the window; adafactor_fused reads its param groups in the hooks (never scaled), the
    carousel's optimizer its own groups (the scale is undone in its step)."""
    if hasattr(optimizer, "snapshot_group"):
        return ft_host.StepContext.snapshot(optimizer, step_index, inv_grad_scale=inv_grad_scale)
    return ft_host.StepContext(groups=(), step_index=int(step_index), inv_grad_scale=float(inv_grad_scale))


class FusedAdafactorSteps:
    """adafactor_fused (design 3.2) behind GradPipeline's interface: anima_train.py's fused Adafactor hooks. GA is 1, so
    every micro-step is a window. A parameter's post-accumulate-grad hook clips its gradient per tensor
    (--max_grad_norm, as anima_train.py), steps it (library.adafactor_fused: bf16 weights written back with stochastic
    rounding, fp32 ones exactly) and drops the gradient. grad_norm is the window's global norm from the per-tensor
    norms when clipping is on (the norms are computed then anyway), NaN otherwise."""

    last_grad_norm = None
    skip_streak = 0
    pinned_bytes = 0

    def __init__(self, optimizer, max_grad_norm: float):
        self.optimizer = optimizer
        self.max_grad_norm = float(max_grad_norm or 0.0)
        self._handles: List[Any] = []
        self._step_index = 0
        self._sumsq: Optional[torch.Tensor] = None
        self._stepped = 0

    def attach(self) -> None:
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.requires_grad:
                    self._handles.append(param.register_post_accumulate_grad_hook(self._make_hook(group)))

    def _make_hook(self, group):
        def hook(tensor: torch.Tensor):
            if self.max_grad_norm > 0.0:
                norm = torch.nn.utils.clip_grad_norm_(tensor, self.max_grad_norm)
                square = norm.float().square()
                self._sumsq = square if self._sumsq is None else self._sumsq + square
            self.optimizer.step_param(tensor, group)
            tensor.grad = None
            self._stepped += 1

        return hook

    def begin_window(self, n_micro: int, ctx: ft_host.StepContext) -> None:
        if int(n_micro) != 1:
            raise ValueError(f"{PREFIX}adafactor_fused steps in the backward: windows of 1 micro-step, got {n_micro}")
        self._step_index, self._sumsq, self._stepped = ctx.step_index, None, 0

    def begin_microstep(self, j: int) -> None:
        pass

    def end_microstep(self) -> ft_host.WindowStats:
        # updates nothing (adafactor_step skips parameters without a gradient, and the hooks dropped every one): the LR
        # scheduler sees optimizer.step() before its own step, as GradPipeline makes it see
        self.optimizer.step()
        norm = math.sqrt(float(self._sumsq)) if self._sumsq is not None else float("nan")
        return ft_host.WindowStats(step_index=self._step_index, n_micro=1, grad_norm=norm, clip_coef=1.0,
                                   clip_mode="per_tensor" if self.max_grad_norm > 0.0 else "none", skipped_params=0,
                                   skipped_names=[], skip_streak=0, stepped_params=self._stepped, tail_s=0.0,
                                   hook_wait_s=0.0, worker_busy_s=0.0)

    def check_skip_streak(self, stats) -> None:
        pass

    def restore_counters(self, last_grad_norm, skip_streak: int = 0) -> None:
        pass

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# ---- start-up -------------------------------------------------------------------------------------------------------
def prepare_runtime(args) -> None:
    """Process-wide settings and the checks of this script that ft_args does not make (design 4.3 step 1-2)."""
    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"
    if args.attn_mode == "flash" and not attention.is_flash_attn_supported():
        raise ValueError("MOSS-Image: attn_mode = \"flash\" needs the flash-attn package and an Ampere or newer GPU;"
                         " install flash-attn or set attn_mode = \"torch\"")
    if getattr(args, "moss_ft_lr_phases", None) and int(getattr(args, "scale_resume", 0) or 0) > 0:
        # before resolve, which would add the extra epochs to max_train_epochs (and refuse without one)
        raise ValueError(f"{PREFIX}--scale_resume is not supported with --moss_ft_lr_phases: the phases give every epoch"
                         " its LR and the run its length, so scale_resume's extra epochs would have no LR. Add them as a"
                         " last phase instead (e.g. \"2:5e-6,6:1e-6,2:1e-6\"); the low-noise preset of those epochs is"
                         " not available with phases")
    timestep_presets.resolve(args)  # sigmoid_scale presets (before the dataset-free validation reads the epochs)
    # the process-wide switches of the MOSS blocks (flags; nothing runs a DiT forward before step 7), set before
    # validate_moss_args reads them (design 4.3 step 7 had them after it: "--moss_ckpt_reuse_attn has no effect here")
    enable_elementwise_compile(not args.moss_no_compile)
    set_ckpt_reuse(False, args.moss_ckpt_reuse_attn, args.moss_ckpt_reuse_attn_budget_gib)
    # validate_moss_args reads the LoRA trainer's two attributes without getattr (design 4.2)
    args.network_train_unet_only = True
    args.network_train_text_encoder_only = False
    if not args.output_dir:
        raise ValueError(f"{PREFIX}--output_dir is required: the DiT saves, the training states (unless"
                         " --moss_ft_state_dir) and the sample images go there")
    if int(getattr(args, "cache_aug_refresh_epochs", 0) or 0) > 0:
        raise ValueError(f"{PREFIX}--cache_aug_refresh_epochs is not supported: it re-encodes variants between epochs,"
                         " while the VAE and the text encoder are released before the DiT is loaded (design 4.3)")
    if getattr(args, "save_model_as", None) not in (None, "safetensors"):
        raise ValueError(f"{PREFIX}save_model_as {args.save_model_as!r}: the trained DiT is saved as safetensors in the"
                         " Base layout (design 4.7)")
    for name in IGNORED_CUDA_MEMORY_OPTIONS:
        if getattr(args, name, None):
            logger.warning(f"{PREFIX}--{name} is ignored: no VRAM cap and no cache flush, an OOM must surface"
                           " (design 2.1)")
    if getattr(args, "huggingface_repo_id", None):
        logger.warning(f"{PREFIX}--huggingface_repo_id is ignored: moss_train.py does not upload")
    if not args.moss_text_encoder:
        raise ValueError("MOSS-Image: --moss_text_encoder (qwen3.5_9b_moss_image_sft_bf16.safetensors) is required")
    strategy.check_moss_text_encoder_path(args.moss_text_encoder)


def set_moss_strategies(args) -> strategy.MossTextEncodingStrategy:
    """The MOSS strategies (design 4.3 step 3; set before the dataset is made, which reads them). With --moss_te_live
    no TE-outputs caching strategy is set: the dataset then runs process_caption and the tokenizer at every fetch."""
    strategy_base.TokenizeStrategy.set_strategy(strategy.MossTokenizeStrategy(args.moss_tokenizer_dir,
                                                                              args.moss_max_token_length))
    strategy_base.LatentsCachingStrategy.set_strategy(strategy.MossLatentsCachingStrategy(
        args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check,
        cache_dtype=getattr(args, "cache_latents_dtype", "auto")))
    encoding = strategy.MossTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(encoding)
    if ft_args.te_live(args):
        return encoding
    strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(strategy.MossTextEncoderOutputsCachingStrategy(
        args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check,
        cache_dtype=getattr(args, "cache_text_encoder_outputs_dtype", "auto"),
        signature=strategy.moss_te_signature(args.moss_text_encoder, args.moss_max_token_length,
                                             args.moss_tokenizer_dir)))
    return encoding


def open_live_text_encoder(args, device) -> te_live.LiveTextEncoder:
    """--moss_te_live: the frozen 9B in host RAM (--moss_te_live_host) behind the streaming engine (plan 1.2)."""
    start = time.perf_counter()
    weights = te_live_weights.HostTextEncoderWeights.open(args.moss_text_encoder, host=args.moss_te_live_host)
    try:
        encoder = te_live.LiveTextEncoder(weights, device, ring_slots=args.moss_te_live_ring_slots,
                                          staging_threads=args.moss_te_live_threads,
                                          batch_rows=int(getattr(args, "moss_te_live_batch", 1) or 1))
    except BaseException:
        weights.close()
        raise
    logger.info(f"live text encoder ready in {time.perf_counter() - start:.1f} s: {weights.num_layers} decoder layers in"
                f" host RAM ({args.moss_te_live_host}: {weights.private_bytes / GIB:.2f} GiB private,"
                f" {weights.locked_bytes / GIB:.2f} GiB page-locked), {args.moss_te_live_ring_slots} device slots of"
                f" {encoder.slot_bytes / MIB:.1f} MiB" + (f", pinned staging {encoder.ring_slots * encoder.entry_bytes / GIB:.2f}"
                                                           " GiB" if encoder.staging else "")
                + (f"; batched layer calls of up to {encoder.batch_rows} captions (--moss_te_live_batch)"
                   if encoder.batch_rows > 1 else ""))
    return encoder


def live_prefetch_factor(ahead: int, n_workers: int) -> int:
    """--moss_te_live with DataLoader workers: batches each worker keeps ready, so the workers hold a whole pass
    (`ahead` batches) instead of torch's default 2 per worker - else most of a pass's loads, caption draws and
    tokenizations run while the pass waits, at worker parallelism, and the workers idle between passes."""
    return max(2, -(-int(ahead) // max(1, int(n_workers))))


def live_ahead(args) -> int:
    """--moss_te_live_ahead rounded up to a multiple of GA: a pass always starts a window (the previous window has
    drained, so the stager's memcpy does not compete with the host optimizer for memory bandwidth)."""
    ga = int(args.gradient_accumulation_steps)
    ahead = -(-int(args.moss_te_live_ahead) // ga) * ga
    if ahead != int(args.moss_te_live_ahead):
        logger.info(f"{PREFIX}--moss_te_live_ahead {args.moss_te_live_ahead} rounded up to {ahead} (a multiple of GA {ga})")
    return ahead


def make_dataset_groups(args):
    """(train group, validation group) as anima_train.py builds them: a dataset TOML, DreamBooth folders, a metadata
    file, or --dataset_class."""
    if args.dataset_class is not None:
        return train_util.load_arbitrary_dataset(args), None
    blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
    if args.dataset_config is not None:
        logger.info(f"Load dataset config from {args.dataset_config}")
        user_config = config_util.load_user_config(args.dataset_config)
        ignored = ["train_data_dir", "in_json"]
        if any(getattr(args, attr) is not None for attr in ignored):
            logger.warning("ignore following options because config file is found: {0}".format(", ".join(ignored)))
    elif args.in_json is None:
        logger.info("Using DreamBooth method.")
        user_config = {"datasets": [{"subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
            args.train_data_dir, args.reg_data_dir)}]}
    else:
        logger.info("Training with captions.")
        user_config = {"datasets": [{"subsets": [{"image_dir": args.train_data_dir, "metadata_file": args.in_json}]}]}
    blueprint = blueprint_generator.generate(user_config, args)
    return config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)


def configure_dataset_group(args, group) -> None:
    """anima_train.py's per-dataset switches, and the cached augmentation / caption variants (train_network.py)."""
    for dataset in group.datasets:
        if args.protected_tags_file:
            dataset.protected_tags_file = args.protected_tags_file
        if args.log_caption_tag_dropout:
            dataset.log_caption_tag_dropout = True
        if args.log_caption_dropout:
            dataset.log_caption_dropout = True
    latent_variants = int(getattr(args, "cache_aug_variants", 0) or 0)
    caption_variants = int(getattr(args, "cache_caption_variants", 0) or 0)
    if (latent_variants > 1 or caption_variants > 1) and hasattr(group, "set_aug_variant_config"):
        group.set_aug_variant_config(latent_variants, caption_variants)
        logger.info(f"cached variants per image: {max(latent_variants, 1)} latent, {max(caption_variants, 1)} caption")


def make_optimizer(args, mode: str, groups: List[dict], optimizer_kwargs: Dict[str, Any]):
    """The mode's optimizer over the "decay" / "no_decay" groups (design 4.4.1), at learning_rate (LR1 with phases)."""
    if mode == ft_args.ADAMW_HOST:
        return HostAdamW(groups, lr=args.learning_rate, **optimizer_kwargs)
    if mode == ft_args.CAME_HOST:
        return HostCAME(groups, lr=args.learning_rate, **optimizer_kwargs)
    import library.adafactor_fused

    _, _, optimizer = train_util.get_optimizer(args, trainable_params=groups, optimizer_kwargs=dict(optimizer_kwargs))
    library.adafactor_fused.patch_adafactor_fused(optimizer)  # step_param for the hooks (anima_train.py)
    return optimizer


def make_lr_scheduler(args, optimizer, steps_per_epoch: int, num_processes: int):
    """ft_args.make_lr_scheduler: the --moss_ft_lr_phases scheduler or train_util.get_scheduler_fix (the step-4
    validation built the same one on a stand-in optimizer already, ft_args.check_lr_scheduler)."""
    return ft_args.make_lr_scheduler(args, optimizer, steps_per_epoch, num_processes)


def make_pipeline(args, mode: str, slots, pool, optimizer, device):
    if mode == ft_args.ADAFACTOR_FUSED:
        return FusedAdafactorSteps(optimizer, args.max_grad_norm)
    return ft_host.GradPipeline(slots, pool, optimizer, device=device,
                                gradient_accumulation_steps=args.gradient_accumulation_steps,
                                max_grad_norm=args.max_grad_norm, clip_mode=args.moss_ft_clip_mode,
                                max_skipped_steps=args.moss_ft_max_skipped_steps,
                                piece_bytes=args.moss_ft_piece_mib * MIB, ring_bytes=args.moss_ft_ring_mib * MIB,
                                upload_ring_bytes=args.moss_ft_upload_ring_mib * MIB,
                                host_threads=args.moss_ft_host_threads,
                                overflow_policy=getattr(args, "moss_ft_overflow_policy", "partial"))


# ---- the run's objects, saving, logs --------------------------------------------------------------------------------
@dataclass
class Run:
    """What the loop, the saves and the samples share."""
    args: Any
    accelerator: Any
    mode: str
    optimizer_kwargs: Dict[str, Any]
    dit: Any
    pool: Optional[ft_host.HostPool]
    slots: List[ft_host.ParamSlot]
    optimizer: Any
    lr_scheduler: Any
    pipeline: Any
    vae: Any
    sample_prompts_te_outputs: Optional[dict]
    num_batches: int
    steps_per_epoch: int
    train_images: int
    te_feed: Optional[te_live_feed.TextEncodeAhead] = None  # --moss_te_live
    scaler: Optional[ft_scaler.DynamicLossScaler] = None  # mixed_precision "fp16"
    validator: Optional[ft_val.Validator] = None  # --moss_ft_val_dataset_config

    def lr_schedule(self) -> Optional[dict]:
        schedule = getattr(self.lr_scheduler, "schedule", None)
        return schedule() if callable(schedule) else None

    def flush(self) -> None:
        """carousel: the active masters into the host pool before a save reads it (no-op for the other modes)."""
        flush = getattr(self.pipeline, "flush_to_host", None)
        if flush is not None:
            flush()

    def save_dit(self, name: str, global_step: int, epoch: int) -> None:
        """A DiT file in the Base layout (after a window's drain): trained tensors from the fp32 master (from the
        model in adafactor_fused), save_precision."""
        self.flush()
        meta = ft_model.moss_dit_metadata(self.args, mode=self.mode, optimizer_kwargs=self.optimizer_kwargs,
                                          global_step=global_step, epoch=epoch, train_images=self.train_images)
        ft_model.save_moss_dit(os.path.join(self.args.output_dir, name), self.dit, self.pool, self.args.save_precision,
                               meta)

    def remove_dit(self, name: Optional[str]) -> None:
        if name is None:
            return
        path = os.path.join(self.args.output_dir, name)
        if os.path.exists(path):
            logger.info(f"removing old checkpoint: {path}")
            os.remove(path)

    def save_state(self, name: str, progress: ft_state.TrainProgress) -> str:
        """A training state (after a window's drain), then the keep policy of design 4.7."""
        state_dir = ft_state.state_dir_of(self.args)
        self.flush()
        path = ft_state.write_state(state_dir, name, model=self.dit, slots=self.slots, pool=self.pool,
                                    optimizer=self.optimizer, lr_scheduler=self.lr_scheduler, mode=self.mode,
                                    progress=progress, last_grad_norm=self.pipeline.last_grad_norm,
                                    skip_streak=self.pipeline.skip_streak, args=self.args, loss_scaler=self.scaler,
                                    extra_writer=getattr(self.pipeline, "state_writer", None))
        # never the state just written, nor another run's states past this run's position (a resume from an earlier
        # state into the same folder)
        ft_state.apply_keep_policy(state_dir, **ft_state.keep_policy_from_args(self.args), written=path)
        return path

    def sample(self, epoch: Optional[int], global_step: int) -> None:
        moss_train_utils.sample_images(self.accelerator, self.args, epoch, global_step, self.dit, self.vae,
                                       self.sample_prompts_te_outputs)

    def validate(self, global_step: int, images_seen: int, *, stream_dtype: torch.dtype, noise_scheduler) -> None:
        """The validation loss of the current weights (after a window's drain, or before the first step)."""
        if self.validator is not None and self.accelerator.is_main_process:
            self.validator.run(self.accelerator, self.dit, global_step, images_seen, stream_dtype=stream_dtype,
                               noise_scheduler=noise_scheduler)


class WindowMeter:
    """One window's loss (summed on the device: one .item() per window), times and token counts."""

    def __init__(self, start: float, lrs: Dict[str, float]):
        self.start, self.lrs = start, lrs
        self.loss: Optional[torch.Tensor] = None
        self.data_s = 0.0
        self.te_s: Optional[float] = None  # --moss_te_live: text-encoder pass seconds in this window (not in data_s)
        self.te_rows = 0  # --moss_te_live: captions those passes encoded (a pass serves ahead / GA windows)
        self.tokens = 0
        self.images = 0

    def add(self, loss: torch.Tensor, batch_size: int, image_tokens: int, text_tokens: Optional[int]) -> None:
        self.loss = loss.detach() if self.loss is None else self.loss + loss.detach()
        self.images += int(batch_size)
        self.tokens += int(batch_size) * (int(image_tokens) + int(text_tokens or 0))


def group_lrs(optimizer) -> Dict[str, float]:
    """The LR each param group steps with in this window (set by the scheduler before the window), by group name."""
    return {str(group.get("name", f"group{index}")): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)}


def window_logs(meter: WindowMeter, loss: float, stats: ft_host.WindowStats, step_s: float,
                te_feed=None, extra: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """The per-step scalars of design 4.9. te_feed (--moss_te_live): te/s_per_image is the run's pass time per encoded
    caption (GPU time from CUDA events once read, else host time), not this window's pass time over its images - a
    pass encodes the captions of ahead / GA windows and falls into one of them."""
    logs = {"loss": loss}
    logs.update({f"lr/{name}": lr for name, lr in meter.lrs.items()})
    if math.isfinite(stats.grad_norm):
        logs["grad_norm"] = stats.grad_norm
    logs.update({"clip_coef": stats.clip_coef, "skipped_params": float(stats.skipped_params), "time/step": step_s,
                 "time/tail": stats.tail_s, "time/hook_wait": stats.hook_wait_s,
                 "time/worker_busy": stats.worker_busy_s, "time/data": meter.data_s})
    if meter.te_s is not None:
        logs["te/pass_s"] = meter.te_s  # host seconds of the passes in this window (0 in the windows between passes)
        logs["te/pass_images"] = float(meter.te_rows)
        if te_feed is not None:
            gpu_s, gpu_rows = te_feed.device_seconds()
            logs["te/s_per_image"] = te_feed.s_per_image
            if gpu_rows:  # pragma: no cover - CUDA only
                logs["te/gpu_s_total"] = gpu_s
        else:
            logs["te/s_per_image"] = meter.te_s / max(1, meter.te_rows)
    if step_s > 0:
        logs["throughput/tok_per_s"] = meter.tokens / step_s
        logs["throughput/img_per_s"] = meter.images / step_s
    logs.update(getattr(stats, "extra", None) or {})  # the carousel's scalars
    logs.update(extra or {})  # a switch's scalars
    return logs


def memory_logs(device, pipeline) -> Dict[str, float]:
    """The every-moss_ft_log_every scalars of design 4.9: VRAM peaks (CUDA), host RSS / available / pinned."""
    logs: Dict[str, float] = {}
    stats = moss_train_utils.cuda_memory_stats(device, with_free=True)
    if stats:
        logs["vram/peak_allocated"] = stats["peak_allocated"] / GIB
        logs["vram/peak_reserved"] = stats["peak_reserved"] / GIB
        outside = moss_train_utils.outside_bytes(stats)
        if outside is not None:
            logs["vram/device_peak"] = (stats["peak_reserved"] + outside) / GIB
    try:
        import psutil

        logs["host/rss_gib"] = psutil.Process().memory_info().rss / GIB
        logs["host/available_gib"] = psutil.virtual_memory().available / GIB
    except Exception:  # noqa: BLE001 - logging only
        pass
    logs["host/pinned_gib"] = pipeline.pinned_bytes / GIB
    if torch.device(device).type == "cuda":
        # everything torch's pinned host allocator holds (rings, norm table, --unsloth_offload_checkpointing blocks):
        # the budget's power-of-two estimate of the unsloth blocks is checked against it in the smoke run
        try:
            logs["host/pinned_cache_gib"] = torch.cuda.host_memory_stats()["allocated_bytes.current"] / GIB
        except Exception:  # noqa: BLE001 - logging only
            pass
    return logs


def memory_line(logs: Dict[str, float]) -> str:
    parts = [f"host RSS {logs['host/rss_gib']:.1f} GiB, available {logs['host/available_gib']:.1f} GiB"
             if "host/rss_gib" in logs else "", f"pinned {logs['host/pinned_gib']:.2f} GiB"
             + (f" (pinned cache {logs['host/pinned_cache_gib']:.2f} GiB)" if "host/pinned_cache_gib" in logs else "")]
    if "vram/peak_reserved" in logs:
        parts.append(f"VRAM peak allocated {logs['vram/peak_allocated']:.1f} / reserved"
                     f" {logs['vram/peak_reserved']:.1f}"
                     + (f" / device {logs['vram/device_peak']:.1f}" if "vram/device_peak" in logs else "") + " GiB")
    return ", ".join(part for part in parts if part)


def init_trackers(accelerator, args) -> None:
    if not accelerator.is_main_process:
        return
    init_kwargs = {}
    if args.wandb_run_name:
        init_kwargs["wandb"] = {"name": args.wandb_run_name}
    if args.log_tracker_config is not None:
        init_kwargs = toml.load(args.log_tracker_config)
    accelerator.init_trackers("finetuning" if args.log_tracker_name is None else args.log_tracker_name,
                              config=train_util.get_sanitized_config_or_none(args), init_kwargs=init_kwargs)
    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        import wandb

        wandb.define_metric("epoch")
        wandb.define_metric("loss/epoch", step_metric="epoch")


# ---- train ----------------------------------------------------------------------------------------------------------
def train(args) -> None:
    # 1. arguments, logging, seed (design 4.3)
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    train_util.set_torch_cuda_reduced_precision(args)
    setup_logging(args, reset=True)
    cuda_memory.default_expandable_segments()  # torch >= 2.14; before the accelerator touches CUDA
    prepare_runtime(args)
    train_util.args_set_seed(args)

    # 2. what needs no dataset: the new arguments, the optimizer mode, the fixed handling of 4.5, the LR phases
    ft_args.validate_moss_ft_args(args)
    mode = ft_args.optimizer_mode(args)
    optimizer_kwargs = ft_args.parse_optimizer_args(args, mode, quiet=True)
    if args.resume:
        ft_state.verify_state(args.resume)  # an incomplete / truncated state is refused before anything is loaded

    # 3. strategies, 4. the dataset group
    text_encoding_strategy = set_moss_strategies(args)
    train_dataset_group, val_dataset_group = make_dataset_groups(args)
    configure_dataset_group(args, train_dataset_group)
    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)
    if args.debug_dataset:
        train_dataset_group.set_current_strategies()
        train_util.debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        logger.error("No data found. Please verify the metadata file and train_data_dir option.")
        raise SystemExit(1)  # a launcher must see the failure (exit code 0 = finished)
    assert train_dataset_group.is_latent_cacheable(), (
        "when caching latents, either color_aug or random_crop cannot be used (use --cache_aug_variants)")
    # incl. bucket steps of 16; the caption checks are ft_args.check_captions' (inside the next call)
    moss_train_utils.validate_moss_args(args, train_dataset_group, val_dataset_group, live_te=ft_args.te_live(args),
                                        check_captions=False)
    # the held-out set of --moss_ft_val_dataset_config (its own TOML; the global RNGs are left as they were)
    held_out_group = (ft_val.make_val_dataset_group(args, train_dataset_group) if ft_args.val_enabled(args) else None)
    ft_args.validate_moss_ft_args(args, train_dataset_group, val_dataset_group=val_dataset_group,
                                  moss_val_group=held_out_group)  # + budget checks
    sigma_sampler = flow.MossSigmaSampler(args)
    train_images = ft_budget.count_images(train_dataset_group)

    # 5. accelerator
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)
    ft_args.check_single_process(accelerator.num_processes)
    device = accelerator.device

    # 6. caches: latents (fp32 VAE on the device for caching, then back on the CPU for sample decodes), then the text
    # encoder outputs and sample prompts (the 9B encoder loads only if some caption needs encoding, then is released)
    vae = load_moss_vae(args.vae, device="cpu", dtype=torch.float32)
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)
    train_dataset_group.new_cache_latents(vae, accelerator)
    if held_out_group is not None:
        ft_val.cache_val_latents(held_out_group, vae, accelerator)  # the same strategy and VAE pass (RNG restored)
    vae.to("cpu")
    clean_memory_on_device(device)
    accelerator.wait_for_everyone()
    encoder = None
    val_text_rows = None
    if ft_args.te_live(args):
        # --moss_te_live: the encoder's layers stay in host RAM for the whole run (streamed per pass); the sample
        # prompts are encoded now, like the cached path does
        encoder = open_live_text_encoder(args, device)
        sample_prompts_te_outputs = (te_cache.cache_moss_sample_prompts(args, encoder) if args.sample_prompts is not None
                                     else None)
        if held_out_group is not None:  # one pass over the validation captions, each alone (the cached encoding)
            val_text_rows = ft_val.encode_val_captions(args, ft_val.val_texts(held_out_group), encoder)
    else:
        provider = strategy.MossTextEncoderProvider(args.moss_text_encoder, device, torch.bfloat16)
        if held_out_group is not None:  # before the TE cache step, which releases the encoder (loaded by either)
            val_text_rows = ft_val.encode_val_captions(args, ft_val.val_texts(held_out_group), provider)
        sample_prompts_te_outputs = te_cache.cache_moss_text_encoder_outputs(args, accelerator, vae,
                                                                             train_dataset_group, provider)
        del provider
    gc.collect()
    clean_memory_on_device(device)
    validator = None
    if held_out_group is not None:  # the plan, the latents present (refused otherwise), the log file - before the DiT
        validator = ft_val.Validator(args, held_out_group, val_text_rows, resume=bool(args.resume))

    try:
        # 7. only now: the host pool and the DiT (design 4.4.1), checkpointing (the process-wide switches of the
        # blocks were set by prepare_runtime)
        dit, pool, slots = ft_model.load_moss_for_full_ft(args.pretrained_model_name_or_path, args, device)
        if args.gradient_checkpointing:
            dit.enable_gradient_checkpointing(unsloth_offload=args.unsloth_offload_checkpointing)

        # 8. optimizer, LR scheduler, the gradient pipeline (modes a, c), the fused Adafactor hooks (mode b) or the
        # carousel engine (its optimizer steps handles of the device masters, built with it)
        pipeline = None
        if mode == ft_args.CAROUSEL:
            optimizer, pipeline = ft_carousel.build(args, dit, pool, slots, optimizer_kwargs, device)
        else:
            named = ft_model.trainable_named(slots)
            groups = ft_model.param_groups(named, float(optimizer_kwargs.get("weight_decay", 0.0) or 0.0))
            optimizer = make_optimizer(args, mode, groups, optimizer_kwargs)
            ft_host.bind_slots(slots, named, optimizer)
        ga = int(args.gradient_accumulation_steps)
        num_batches = len(train_dataset_group)
        steps_per_epoch = ft_args.steps_per_epoch(num_batches, ga)
        lr_scheduler = make_lr_scheduler(args, optimizer, steps_per_epoch, accelerator.num_processes)
        if pipeline is None:
            pipeline = make_pipeline(args, mode, slots, pool, optimizer, device)
        te_feed = None if encoder is None else te_live_feed.TextEncodeAhead(encoder, live_ahead(args))
        run = Run(args=args, accelerator=accelerator, mode=mode, optimizer_kwargs=optimizer_kwargs, dit=dit, pool=pool,
                  slots=slots, optimizer=optimizer, lr_scheduler=lr_scheduler, pipeline=pipeline, vae=vae,
                  sample_prompts_te_outputs=sample_prompts_te_outputs, num_batches=num_batches,
                  steps_per_epoch=steps_per_epoch, train_images=train_images, te_feed=te_feed,
                  scaler=ft_scaler.make(args), validator=validator)
        try:
            pipeline.attach()
            _train_loop(run, train_dataset_group, collator, current_epoch, current_step, sigma_sampler,
                        text_encoding_strategy)
        finally:
            pipeline.close()
    finally:
        if encoder is not None:
            encoder.close()
    accelerator.end_training()


def _train_loop(run: Run, train_dataset_group, collator, current_epoch, current_step, sigma_sampler,
                text_encoding_strategy) -> None:
    args, accelerator, dit, optimizer, pipeline = run.args, run.accelerator, run.dit, run.optimizer, run.pipeline
    device, ga, num_batches = accelerator.device, int(args.gradient_accumulation_steps), run.num_batches
    scaler = run.scaler  # fp16: the dynamic loss scale (None: bf16 / fp32 compute)
    max_steps = int(args.max_train_steps)

    # 9. resume (design 4.7), then the DataLoader
    progress = ft_state.TrainProgress(global_step=0, completed_epochs=0, steps_per_epoch=run.steps_per_epoch,
                                      gradient_accumulation_steps=ga, lr_schedule=run.lr_schedule())
    if args.resume:
        info = ft_state.load_state(args.resume, model=dit, slots=run.slots, pool=run.pool, optimizer=optimizer,
                                   lr_scheduler=run.lr_scheduler, mode=run.mode, args=args,
                                   steps_per_epoch=run.steps_per_epoch,
                                   extra_reader=getattr(pipeline, "state_reader", None))
        pipeline.restore_counters(info.last_grad_norm, info.skip_streak)
        ft_scaler.restore(scaler, info.loss_scaler)
        progress = info.progress
    global_step, batches_to_skip = int(progress.global_step), int(progress.epoch_step)
    # training images consumed (every micro-batch's rows): the validation log's x axis, kept in the states
    images_seen = (resumed_images_seen(progress, train_dataset_group.num_train_images, num_batches) if args.resume
                   else 0)
    train_dataset_group.set_current_strategies()
    if hasattr(train_dataset_group, "attach_epoch_value"):
        train_dataset_group.attach_epoch_value(current_epoch)  # each epoch's bucket shuffle before its first fetch
    train_dataset_group.set_max_train_steps(max_steps)
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count() or 1)
    loader_kwargs = {}
    if run.te_feed is not None and n_workers > 0:
        # the encode-ahead pulls `ahead` batches at once: let the workers keep that many ready (CPU batches; accelerate
        # moves each one to the device only when it is yielded, so VRAM does not change)
        loader_kwargs["prefetch_factor"] = live_prefetch_factor(run.te_feed.ahead, n_workers)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group, batch_size=1, shuffle=True, collate_fn=collator, num_workers=n_workers,
        persistent_workers=bool(args.persistent_data_loader_workers) and n_workers > 0, **loader_kwargs)
    train_dataloader = accelerator.prepare(train_dataloader)  # batches on the device

    # 10. trackers, the first sample, the loop
    num_train_epochs = math.ceil(max_steps / run.steps_per_epoch)
    if args.save_n_epoch_ratio is not None and args.save_n_epoch_ratio > 0:
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1
    accelerator.print("running training")
    accelerator.print(f"  mode: {run.mode}, {sum(s.numel for s in run.slots) / 1e9:.3f}B parameters trained")
    accelerator.print(f"  num examples: {train_dataset_group.num_train_images}, batches per epoch: {num_batches},"
                      f" gradient accumulation: {ga}, optimizer steps per epoch: {run.steps_per_epoch}")
    accelerator.print(f"  epochs: {num_train_epochs}, total optimizer steps: {max_steps}"
                      + (f", resuming at step {global_step} (epoch {progress.completed_epochs + 1},"
                         f" {batches_to_skip} batches into it)" if args.resume else ""))
    init_trackers(accelerator, args)

    noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000,
                                                                     shift=args.discrete_flow_shift)
    stream_dtype = torch.float32 if args.moss_stream_fp32 else ft_model.compute_dtype(args)
    timer = tier_timing.TierTimer(device.type)
    loss_recorder = train_util.EMARecorder()
    rate_tracker = train_util.RateTracker()
    log_every = int(args.moss_ft_log_every)
    progress_bar = tqdm(range(max_steps), initial=global_step, smoothing=0.1,
                        disable=not accelerator.is_local_main_process, desc="steps")

    if global_step == 0:
        run.sample(0, 0)  # --sample_at_first (a resumed run does not sample at its start)
        run.validate(0, 0, stream_dtype=stream_dtype, noise_scheduler=noise_scheduler)  # step 0: before any update
    for epoch in range(int(progress.completed_epochs), num_train_epochs):
        if global_step >= max_steps:
            break
        epoch_no = epoch + 1
        current_epoch.value = epoch_no
        timestep_presets.set_epoch(args, epoch_no)
        accelerator.print(f"\nepoch {epoch_no}/{num_train_epochs}")
        dit.train()
        offset, batches_to_skip = batches_to_skip, 0
        if offset % ga:  # the state was saved with another GA (ft_state warned): restart its unfinished window
            logger.warning(f"{PREFIX}the state's {offset} batches into epoch {epoch_no} are not a whole number of"
                           f" windows of GA {ga}: resuming at batch {offset - offset % ga}")
            offset -= offset % ga
        loader = train_dataloader
        if offset > 0:
            logger.info(f"resuming inside epoch {epoch_no}: skipping {offset} of its {num_batches} batches (a new"
                        " shuffle of the epoch, as kohya does; design 4.7)")
            loader = accelerator.skip_first_batches(train_dataloader, offset)
        if run.te_feed is not None:
            loader = run.te_feed.wrap(loader)  # after the skip: the feed only changes when a batch's text is encoded
        consumed, meter, mark = offset, None, time.perf_counter()
        for i, batch in enumerate(loader):
            index = offset + i
            arrived = time.perf_counter()
            te_s, te_rows = run.te_feed.take_pass() if run.te_feed is not None else (None, 0)
            w, j, n_micro = window_of(index, num_batches, ga)
            if j == 0:
                meter = WindowMeter(mark, group_lrs(optimizer))
                pipeline.begin_window(n_micro, window_context(optimizer, global_step,
                                                              1.0 if scaler is None else scaler.inv_scale))
            if te_s is not None:
                meter.te_s = (meter.te_s or 0.0) + te_s
                meter.te_rows += te_rows
            meter.data_s += arrived - mark - (te_s or 0.0)
            pipeline.begin_microstep(j)
            timer.start(tier_timing.sample_tokens(batch, patch=1, vae_downscale=LATENT_STRIDE))
            loss = ft_loss.moss_ft_noise_and_loss(args, dit, batch, accelerator, sigma_sampler, noise_scheduler,
                                                  stream_dtype, text_encoding_strategy) / n_micro
            # the hooks stream the gradients; the final micro-step steps inside the backward. fp16: the loss (a float64
            # 0-dim tensor) times the power-of-two scale - exact; the host step divides it out
            (loss if scaler is None else loss * scaler.scale).backward()
            stats = pipeline.end_microstep()  # final micro-step: the drain (tail), uploads ordered before the next use
            timer.stop(text_tokens=dit.last_text_tokens)
            latents = batch["latents"]
            meter.add(loss, latents.shape[0], latents.shape[-2] * latents.shape[-1], dit.last_text_tokens)
            images_seen += int(latents.shape[0])
            consumed = index + 1
            if j < n_micro - 1:
                mark = time.perf_counter()
                continue

            # the window is stepped and drained: the pool is stable for logs, samples and saves
            step_s = time.perf_counter() - meter.start
            stepped = stats.stepped_params > 0  # partial overflow still counts when any parameter was updated
            if stepped:
                run.lr_scheduler.step()
                global_step += 1
            current_step.value = global_step
            # carousel: a new active set at a period boundary (before the logs, samples and saves of this step; none
            # after the last step)
            switch_logs = (pipeline.after_step(global_step, max_steps)
                           if stepped and hasattr(pipeline, "after_step") else None)
            progress = progress_after_window(global_step=global_step, epoch=epoch, consumed=consumed,
                                             num_batches=num_batches, steps_per_epoch=run.steps_per_epoch,
                                             gradient_accumulation_steps=ga, lr_schedule=run.lr_schedule(),
                                             images_seen=images_seen)
            window_loss = float(meter.loss.item())
            if run.mode == ft_args.ADAFACTOR_FUSED and not math.isfinite(window_loss):
                raise RuntimeError(f"{PREFIX}non-finite loss at step {global_step} (adafactor_fused steps every"
                                   " parameter inside the backward, so the weights took it): resume from an earlier"
                                   " state, e.g. with a lower learning rate")
            logs = window_logs(meter, window_loss, stats, step_s, run.te_feed, switch_logs)
            if scaler is not None:
                event = scaler.update(stats, math.isfinite(window_loss))  # the scale for the next window
                logs.update({"train/loss_scale": stats.loss_scale, "train/overflow": float(stats.overflow),
                             "train/overflow_windows": float(scaler.overflow_windows)})
                if event != ft_scaler.KEEP:
                    logger.info(f"step {global_step}: fp16 loss scale {stats.loss_scale:g} -> {scaler.scale:g}"
                                f" ({'overflow' if event == ft_scaler.BACKOFF else 'clean steps'})")
            if not stepped:
                # Data progress and loss-scale recovery belong to attempted windows. LR, Carousel, step logs,
                # samples and saves belong to successful updates (also excludes stale clipping's measure window).
                if scaler is None:
                    pipeline.check_skip_streak(stats)
                else:
                    scaler.check(args.moss_ft_max_skipped_steps)
                logger.info(f"step {global_step}: window without a parameter update ({stats.clip_mode});"
                            " LR and step cadence unchanged")
                mark = time.perf_counter()
                continue
            if global_step % log_every == 0:
                memory = memory_logs(device, pipeline)
                tiers = timer.report()
                logs.update(memory)
                logs.update(timer.scalars(tiers))
                logger.info(f"step {global_step}: loss {window_loss:.4f}, {step_s:.2f} s/step (tail {stats.tail_s:.2f},"
                            f" hook wait {stats.hook_wait_s:.2f}, worker {stats.worker_busy_s:.2f}, data"
                            f" {meter.data_s:.2f} s"
                            + (f", text encoder {meter.te_s:.2f} s" if meter.te_s is not None else "")
                            + f"), {logs.get('throughput/tok_per_s', 0.0):,.0f} tok/s;"
                            f" {memory_line(memory)}" + ft_carousel.window_text(stats))
                logger.info(timer.format_line(tiers, what=f"step {global_step}"))
            accelerator.log(logs, step=global_step)
            rate_tracker.tick()
            if math.isfinite(window_loss):  # fp16 skips a non-finite window and goes on: keep the averages finite
                loss_recorder.add(epoch=epoch, step=index, loss=window_loss)
            progress_bar.update(1)
            progress_bar.set_postfix_str(f"{rate_tracker.display_rate}, avr_loss={loss_recorder.moving_average:.4f}")
            if scaler is None:
                pipeline.check_skip_streak(stats)  # --moss_ft_max_skipped_steps (after the window is logged)
            else:  # overflows are the scale's search; only the ones the scale cannot explain count
                scaler.check(args.moss_ft_max_skipped_steps)
            if run.validator is not None and run.validator.due(global_step, max_steps):
                # after the step and its logs, outside any accumulation window (the pipeline is drained)
                run.validate(global_step, images_seen, stream_dtype=stream_dtype, noise_scheduler=noise_scheduler)

            run.sample(None, global_step)  # sample_every_n_steps
            if step_save_due(args, global_step):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    run.save_dit(train_util.get_step_ckpt_name(args, ".safetensors", global_step), global_step, epoch_no)
                    remove_no = train_util.get_remove_step_no(args, global_step)
                    run.remove_dit(None if remove_no is None else
                                   train_util.get_step_ckpt_name(args, ".safetensors", remove_no))
                    if args.save_state:
                        run.save_state(ft_state.step_state_name(args.output_name, global_step), progress)
            mark = time.perf_counter()
            if global_step >= max_steps:
                break

        # the epoch's end (also when max_train_steps ended it early, as kohya)
        accelerator.log({"loss/epoch": loss_recorder.moving_average, "epoch": epoch_no}, step=global_step)
        accelerator.wait_for_everyone()
        if consumed >= num_batches and accelerator.is_main_process:
            if epoch_save_due(args, epoch_no, num_train_epochs):
                run.save_dit(train_util.get_epoch_ckpt_name(args, ".safetensors", epoch_no), global_step, epoch_no)
                remove_no = train_util.get_remove_epoch_no(args, epoch_no)
                run.remove_dit(None if remove_no is None else
                               train_util.get_epoch_ckpt_name(args, ".safetensors", remove_no))
            if (args.save_state and epoch_save_due(args, epoch_no, num_train_epochs)) or phase_state_due(
                    args, epoch_no, num_train_epochs):
                run.save_state(ft_state.epoch_state_name(args.output_name, epoch_no), progress)
        run.sample(epoch_no, global_step)
        logger.info(timer.format_line(timer.epoch_report(), what=f"epoch {epoch_no}"))

    # after the last step (when the loop's last step was not validated, e.g. a resume from the final state)
    if run.validator is not None and run.validator.last_step != global_step:
        run.validate(global_step, images_seen, stream_dtype=stream_dtype, noise_scheduler=noise_scheduler)
    # the train end: the last state (save_state / save_state_on_train_end), then the final DiT
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.save_state or args.save_state_on_train_end:
            run.save_state(ft_state.last_state_name(args.output_name), progress)
        save_epoch = progress.completed_epochs + int(progress.epoch_step > 0)
        run.save_dit(train_util.get_last_ckpt_name(args, ".safetensors"), global_step, save_epoch)
        logger.info("model saved.")


def main(argv: Optional[List[str]] = None) -> None:
    """The command line (as moss_train_network.py): --show_timesteps prints the timestep distribution on the CPU and
    exits (no model, no GPU checks); otherwise train(args)."""
    parser = setup_parser()
    args = parser.parse_args(argv)
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    if args.show_timesteps:
        moss_train_utils.show_timesteps(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
