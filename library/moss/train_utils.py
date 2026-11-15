# MOSS-Image training helpers: CLI arguments, option validation, metadata, sample images.

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from library import attention, cuda_memory, tier_timing, timestep_presets, train_util
from library.device_utils import clean_memory_on_device, synchronize_device
from library.utils import setup_logging

from .flow import TIMESTEP_CHOICES, describe_moss_sampling
from .models import (MOSS_DIT_CONFIG, ckpt_nograd_merged_enabled, ckpt_node_enabled, ckpt_reuse_attn_enabled,
                     ckpt_reuse_block_count, ckpt_stash_bytes_per_token, stream_scale_problem)
from library.lycoris_fast import cached_lokr_weights, ckpt_merged_modules, ckpt_merged_release
from .vae import moss_vae_decode

setup_logging()
import logging

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SHIFT = 5.0  # MOSSImageScheduler.shift (inference)
DEFAULT_UNCOND_SKIP_LAYERS = "22"  # official negative-branch skip


def add_moss_training_arguments(parser) -> None:
    parser.add_argument("--moss_text_encoder", type=str, default=None,
                        help="Qwen3.5-9B text encoder safetensors (qwen3.5_9b_moss_image_sft_bf16.safetensors; Base and SFT"
                             " share it). Loaded only while caching text-encoder outputs")
    parser.add_argument("--moss_tokenizer_dir", type=str, default=None,
                        help="folder with the Qwen3.5 tokenizer.json (MOSS-Image tokenizer/ or any Qwen3.5 copy). Default:"
                             " $MOSS_TOKENIZER_DIR")
    parser.add_argument("--moss_max_token_length", type=int, default=4096, help="text tokens per caption incl. chat template")
    parser.add_argument("--timestep_sampling", type=str, default="moss", choices=TIMESTEP_CHOICES,
                        help="moss = the released Base's last pretraining distribution (logit-normal, see --moss_logit_*);"
                             " sigma/uniform/sigmoid/shift = the usual flow-matching options (fp32 timesteps)")
    parser.add_argument("--sigmoid_scale", type=timestep_presets.parse_sigmoid_scale, default=1.0,
                        help=timestep_presets.SIGMOID_SCALE_HELP)
    timestep_presets.add_timestep_preset_arguments(parser)
    parser.add_argument("--discrete_flow_shift", type=float, default=1.0,
                        help="shift for timestep_sampling=shift and the scheduler of sigma mode (MOSS training_shift = 1.0)")
    parser.add_argument("--moss_logit_mean", type=float, default=1.09, help="moss sampling: mean of logit(sigma)")
    parser.add_argument("--moss_logit_std", type=float, default=0.95, help="moss sampling: std of logit(sigma)")
    parser.add_argument("--moss_sigma_cutoff", type=float, default=0.95, help="moss sampling: truncation before rescaling")
    parser.add_argument("--moss_uniform_mix", type=float, default=0.01, help="moss sampling: weight of the uniform mixture")
    parser.add_argument("--attn_mode", type=str, default=None, choices=["torch", "xformers", "flash", "sageattn", "sdpa"],
                        help="attention backend (padding is handled with masks / varlen); default torch SDPA."
                             " flash (flash-attn varlen) is ~1.6x faster per step on an RTX 5090 at 1536")
    parser.add_argument("--split_attn", action="store_true", help="per-sample attention (slower, lowest memory)")
    parser.add_argument("--moss_stream_fp32", action="store_true",
                        help="run the residual stream, modulation, norms, embedders and head in fp32 around the joint blocks'"
                             " 252 Linears, which stay in the training dtype (bf16 or fp16 mixed precision). Matches an fp32"
                             " stream at inference; about 0.5 GB more for weights plus the checkpointed stream")
    parser.add_argument("--moss_stream_scale", type=float, default=1.0,
                        help="carry the joint residual stream of the 42 blocks multiplied by C, a power of two in (0, 1]"
                             " (0.0625 = 1/16 recommended for fp16; 1.0 = off, the default). An exact fold, no weight"
                             " changes: what writes into the stream (video embedder and text refiner outputs, every"
                             " block's gate_a / gate_f) is multiplied by C and the RMSNorms that read it (norm1 / norm2,"
                             " final_norm_video) use eps 1e-6 * C^2, so RMSNorm(C x) = RMSNorm(x) and the LoRA / LyCORIS"
                             " inputs, outputs and gradients are the same (bit for bit in the CPU tests; a --resume state"
                             " or --network_weights continue unchanged). The stream's largest value shrinks by C, which"
                             " keeps an fp16 stream below 65504 without --moss_stream_fp32's extra VRAM (~4 GiB at the"
                             " 2048 tier); the stream's gradient grows by 1/C. Covers training, sample images,"
                             " checkpointing, --moss_block_compile and --moss_ckpt_reuse_attn")
    parser.add_argument("--moss_no_compile", action="store_true",
                        help="do not torch.compile the MossBlock elementwise chains (RMSNorm/RoPE/SiTU/residual) on CUDA")
    parser.add_argument("--moss_no_lokr_compile", action="store_true",
                        help="do not torch.compile the LyCORIS LoKr+DoRA weight rebuild of the fast forward (compiled results"
                             " differ by ~1 bf16 ulp)")
    parser.add_argument("--moss_stock_lycoris", action="store_true",
                        help="use LyCORIS's stock LoKr forward and per-call sample merges (slower; same results)")
    parser.add_argument("--moss_block_compile", action="store_true",
                        help="torch.compile every joint MossBlock's math with the LyCORIS / LoRA modules inside the graph"
                             " (library/moss/block_compile.py, the --anima_block_compile design): fewer Python module calls"
                             " and kernel launches per micro-step. The first steps compile (minutes on Windows). Not with"
                             " RamTorch, --blocks_to_swap, --torch_compile, stock LyCORIS or DyLoRA. Default off")
    parser.add_argument("--moss_block_compile_backend", choices=["inductor", "aot_eager", "eager"], default="inductor",
                        help="--moss_block_compile: torch.compile backend (default inductor)")
    parser.add_argument("--moss_block_compile_mode", type=str, default=None,
                        help="--moss_block_compile: torch.compile mode (default: torch's default)")
    parser.add_argument("--moss_block_compile_recompile_limit", type=int, default=64,
                        help="--moss_block_compile: dynamo recompile limit (>= 8; default 64)")
    parser.add_argument("--moss_ckpt_nograd_merged", action="store_true",
                        help="--anima_ckpt_nograd_merged for MOSS (library/moss/models.py, anima_models."
                             "NoGradFirstPassCheckpointer). Gradient checkpointing without offload: a checkpointed joint"
                             " block's first pass runs under no_grad on the GPU (the plain checkpoint runs it in grad mode)"
                             " and is recomputed + backpropagated in backward, and that first pass reads each merged"
                             " LoKr+DoRA weight from a tensor that is rebuilt only when the adapter changed (once per"
                             " optimizer step under gradient accumulation; invalidated by every optimizer.step(), at the end"
                             " of each optimizer step of the loop and after sample images) instead of on every call. Same"
                             " math: CPU tests give bit-identical outputs and gradients; under --moss_block_compile expect"
                             " the rounding-level difference block compile itself has (without it, that of the compiled"
                             " elementwise chains of a CUDA run: their no-grad and grad-mode graphs are separate"
                             " compiles). Costs one bf16/fp16 copy of the"
                             " adapted Linears of the joint blocks in VRAM while training (released for every sample round"
                             " and rebuilt at the next training step): ~8.65 GiB for MOSS-Image with 41 adapted blocks"
                             " (4.64B weight elements). A start-up line estimates the extra VRAM for the largest bucket of"
                             " the dataset against the card, the epoch-end lines report the measured peak, and a warning"
                             " follows when the peak reaches the card's size (Windows then spills into shared memory: slow"
                             " steps, no error). On its own it is not expected to be faster (Anima 2.9B: +-0 %%), and"
                             " --moss_ckpt_reuse_attn works without it (then nothing is cached, which is what fits a"
                             " 32 GB card at the 2048 tier). The two text refiner blocks keep the plain"
                             " checkpoint (preset moss-text trains as before), and so does a padded batch under"
                             " --moss_block_compile. Not with --unsloth_offload_checkpointing,"
                             " --cpu_offload_checkpointing, --blocks_to_swap, RamTorch or --moss_stock_lycoris. Default off")
    parser.add_argument("--moss_ckpt_reuse_attn", action="store_true",
                        help="--anima_ckpt_reuse_attn for MOSS (library/attention.py _FlashReuse); needs attn_mode flash."
                             " A checkpointed joint block's first pass runs under no_grad and keeps the block's joint"
                             " text+image attention output and softmax_lse, and the recompute in backward returns them"
                             " instead of running flash-attn's forward kernel again (only its backward kernel runs). Same"
                             " math. Works with or without --moss_ckpt_nograd_merged: without it (reuse-only) nothing is"
                             " cached and both passes rebuild the LoKr+DoRA weights per call like the plain checkpoint"
                             " (CPU tests: bit-identical outputs and gradients. On the GPU the no-grad first pass and the"
                             " grad-mode recompute are separate compiled graphs wherever something is compiled - the"
                             " block graphs of --moss_block_compile, and without it the elementwise chains and the"
                             " LoKr+DoRA rebuild a CUDA run compiles by default - so expect the rounding-level"
                             " difference those compiles themselves have; --moss_no_compile and --moss_no_lokr_compile"
                             " without --moss_block_compile leave both passes on the same eager kernels). The stored tensors live until the block's backward: ~0.10 GiB per block at a 2048^2"
                             " image, ~4.1 GiB over 42 blocks (1536^2: ~2.3 GiB, 1024^2: ~1.1 GiB); cap them with"
                             " --moss_ckpt_reuse_attn_budget_gib. Only the plain flash"
                             " call is reused: one sample, or a batch whose captions have equal token counts; a padded"
                             " batch (flash varlen) stores nothing and recomputes as before. Refused without"
                             " attn_mode flash, with --split_attn, or with a flash-attn that has no separate backward op."
                             " Not with --unsloth_offload_checkpointing, --cpu_offload_checkpointing, --blocks_to_swap,"
                             " RamTorch or --moss_stock_lycoris. Anima 2.9B measured -8.7 %% s/step with both options;"
                             " MOSS is not measured yet (tests/manual_test_moss_ckpt_reuse_gpu.py). Default off")
    parser.add_argument("--moss_ckpt_reuse_attn_budget_gib", type=float, default=None,
                        help="--moss_ckpt_reuse_attn: a cap in GiB on the stored joint attention of one micro-step. Every"
                             " reusing block stores (joint tokens x rows x 6,240 bytes) until its backward; per micro-step"
                             " n = floor(budget / that) is taken from the shape of the batch, the joint blocks 0..n-1"
                             " store and reuse and the others keep the plain gradient checkpoint (with"
                             " --moss_ckpt_nograd_merged: the no-grad first pass without a stored output). Small tiers"
                             " fit every block, the largest tier fewer: 3.0 lets about 30 of the 42 blocks reuse at a"
                             " 2048^2 image and all of them at 1536^2 and below. The start-up line lists the blocks and"
                             " the stored bytes per resolution tier of the dataset. Default: no limit (every block)")
    parser.add_argument("--moss_no_sample_cache", action="store_true",
                        help="sample images without the merged LoKr weight cache (~9 GB less VRAM while sampling; slower)")
    parser.add_argument("--moss_lora_exclude_blocks", type=moss_exclude_blocks_arg, default="",
                        help="comma-separated joint block indices (0-41) left out of the LoRA/LyCORIS network, e.g. \"22\":"
                             " the block the official sampler's negative branch skips, so CFG's skipped-block contrast stays"
                             " the base model's (ai-toolkit's MOSS preset does this). Changes the network: a --resume state or"
                             " --network_weights trained with those blocks will not match")
    parser.add_argument("--moss_sample_shift", type=float, default=DEFAULT_SAMPLE_SHIFT,
                        help="sample images: sigma shift (official 5.0); --fs in a prompt line overrides it")
    parser.add_argument("--moss_sample_uncond_skip_layers", type=str, default=DEFAULT_UNCOND_SKIP_LAYERS,
                        help="sample images: main blocks skipped in the negative CFG branch (official '22'; '' = off)")
    parser.add_argument("--moss_tier_timing", action="store_true",
                        help="--anima_tier_timing for MOSS (library/tier_timing.py): time every micro-step's forward +"
                             " backward and group the times by resolution tier of its image tokens per sample (one latent"
                             " pixel = one DiT token: 512^2 = 1024, 1024^2 = 4096, 1536^2 = 9216, 2048^2 = 16384); the text"
                             " tokens of the joint sequence are reported per tier next to it (tier_time/<tier>/text_tokens,"
                             " 'text N tok'). CUDA: event pairs read only once finished (no synchronization); CPU:"
                             " perf_counter. Every --moss_tier_timing_every optimizer steps the window goes to the tracker,"
                             " at the end of every epoch one INFO line. Off (default): one attribute check per micro-step")
    parser.add_argument("--moss_tier_timing_every", type=int, default=50,
                        help="optimizer steps between --moss_tier_timing reports (default 50)")
    # Accepted and ignored so Anima command lines / train.ps1 do not fail argparse.
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="ignored for MOSS-Image")
    parser.add_argument("--vae_disable_cache", action="store_true", help="ignored for MOSS-Image")
    parser.add_argument("--show_timesteps", type=str, default=None, choices=["console", "image"],
                        help="print (console) or plot (image, matplotlib) the timestep distribution and loss weighting the"
                             " current settings sample, then exit without training (CPU only, no model loaded)")
    parser.add_argument("--show_timesteps_epoch", type=int, default=None,
                        help="--show_timesteps: preview the distribution of this 1-based epoch (scale_resume switches to the"
                             " low-noise 'style' preset after max_train_epochs)")


def _datasets(groups: Sequence) -> List:
    out = []
    for group in groups or ():
        if group is not None:
            out.extend(getattr(group, "datasets", [group]))
    return out


def validate_moss_block_compile_args(args) -> None:
    """--moss_block_compile (library/moss/block_compile.py): refused with what the compiled block graphs cannot hold or
    would re-trace every step (the --anima_block_compile rules). Raises ValueError."""
    if not getattr(args, "moss_block_compile", False):
        return

    def refuse(why: str):
        raise ValueError(f"MOSS-Image: --moss_block_compile {why}")

    if getattr(args, "use_ramtorch", False) or getattr(args, "use_ramtorch_network", False):
        refuse("is not supported with RamTorch (use_ramtorch): the streamed weights change device inside the graph."
               " Turn one of them off (RamTorch for VRAM, block compile for speed)")
    if (getattr(args, "blocks_to_swap", None) or 0) > 0:
        refuse("is not supported with --blocks_to_swap: a swapped block's weights change device between calls")
    if getattr(args, "torch_compile", False):
        refuse("is not supported together with --torch_compile (the blocks are compiled here)")
    module = str(getattr(args, "network_module", "") or "")
    if module.startswith("lycoris") and (getattr(args, "moss_stock_lycoris", False) or getattr(args, "lycoris_stock", False)):
        refuse("needs the fork's LyCORIS paths (not --moss_stock_lycoris / --lycoris_stock): the stock forwards read"
               " org_module[0] through a bound method, which dynamo cannot trace")
    net_kwargs = dict(item.split("=", 1) for item in (getattr(args, "network_args", None) or []) if "=" in item)
    algo = str(net_kwargs.get("algo", "lora") or "lora").strip().lower()
    if (module.startswith("lycoris") and algo == "dylora") or module == "networks.dylora":
        refuse("is not supported with DyLoRA: it draws a random rank every forward (a graph per rank)")
    if int(getattr(args, "moss_block_compile_recompile_limit", 64) or 0) < 8:
        refuse("needs --moss_block_compile_recompile_limit >= 8")


LATENT_STRIDE = 16  # pixels per image token side (strategy.MossLatentsCachingStrategy.STRIDE: 8x VAE, 2x2 packed)


def dataset_group_limits(dataset_group) -> Tuple[int, int, Optional[Tuple[int, int]]]:
    """(the largest batch size, the image tokens of the largest bucket an image was assigned to, that bucket as (width,
    height)) over the datasets of a group. The batch size and the resolution tiers of a dataset TOML live per dataset,
    not in args (args.train_batch_size / max_bucket_reso say nothing about them). Read after make_buckets (the group is
    built with its buckets made); (1, 0, None) for whatever it cannot read: it only feeds warnings and the VRAM estimate,
    and runs for every MOSS training, so a dataset class it does not know must never stop one."""
    batch, tokens, bucket = 1, 0, None
    try:
        for dataset in getattr(dataset_group, "datasets", None) or []:
            batch = max(batch, int(getattr(dataset, "batch_size", 1) or 1))
            for info in (getattr(dataset, "image_data", None) or {}).values():
                resos = [getattr(info, "bucket_reso", None)]
                resos += [entry[0] for entry in (getattr(info, "jitter_bucket_info", None) or {}).values()]
                for reso in resos:
                    if reso is None:
                        continue
                    count = (int(reso[0]) // LATENT_STRIDE) * (int(reso[1]) // LATENT_STRIDE)
                    if count > tokens:
                        tokens, bucket = count, (int(reso[0]), int(reso[1]))
    except Exception as error:  # noqa: BLE001 - an unknown dataset layout: keep what was read
        logger.debug(f"MOSS-Image: dataset limits not fully read ({type(error).__name__}: {error})")
    return batch, tokens, bucket


def dataset_group_tiers(dataset_group) -> List[Tuple[str, int, Tuple[int, int]]]:
    """(tier label, image tokens, (width, height)) of the largest bucket an image was assigned to in every resolution
    tier of the datasets (library/tier_timing.tier_of_tokens: 512 / 1024 / 1536 / 2048 by image tokens), smallest tier
    first. For the per-tier lines of the stored-attention estimate; [] for whatever it cannot read (like
    dataset_group_limits, it must never stop a training)."""
    largest: Dict[str, Tuple[int, Tuple[int, int]]] = {}
    try:
        for dataset in getattr(dataset_group, "datasets", None) or []:
            for info in (getattr(dataset, "image_data", None) or {}).values():
                resos = [getattr(info, "bucket_reso", None)]
                resos += [entry[0] for entry in (getattr(info, "jitter_bucket_info", None) or {}).values()]
                for reso in resos:
                    if reso is None:
                        continue
                    count = (int(reso[0]) // LATENT_STRIDE) * (int(reso[1]) // LATENT_STRIDE)
                    tier = tier_timing.tier_of_tokens(count)
                    if count > 0 and count > largest.get(tier, (0, None))[0]:
                        largest[tier] = (count, (int(reso[0]), int(reso[1])))
    except Exception as error:  # noqa: BLE001 - an unknown dataset layout: keep what was read
        logger.debug(f"MOSS-Image: dataset tiers not fully read ({type(error).__name__}: {error})")
    order = [name for _, name in tier_timing.TIERS]
    return [(tier, *largest[tier]) for tier in order if tier in largest]


def largest_batch_size(args, dataset_limits=None) -> int:
    """The largest batch a training forward can get: --train_batch_size or a dataset's own batch_size (dataset TOML)."""
    return max(int(getattr(args, "train_batch_size", 1) or 1), int(dataset_limits[0]) if dataset_limits else 1)


CKPT_BUDGET_CACHED_RECOMPILE_LIMIT = 16  # cache + reuse + budget under block compile: 8 graphs, and as many to spare


def validate_moss_ckpt_reuse_args(args, train_dataset_group=None) -> None:
    """--moss_ckpt_nograd_merged / --moss_ckpt_reuse_attn / --moss_ckpt_reuse_attn_budget_gib (library/moss/models.py):
    refused where a switch cannot work or would silently do nothing; a warning where it has no effect. Raises ValueError.
    --moss_ckpt_reuse_attn alone is reuse-only: the no-grad first-pass node without the merged-weight cache. Call it
    after the checks that may switch gradient_checkpointing on (unsloth offload). train_dataset_group: for the batch
    sizes of a dataset TOML."""
    nograd = bool(getattr(args, "moss_ckpt_nograd_merged", False))
    reuse = bool(getattr(args, "moss_ckpt_reuse_attn", False))
    budget = getattr(args, "moss_ckpt_reuse_attn_budget_gib", None)

    def refuse(why: str):
        raise ValueError(f"MOSS-Image: {why}")

    if budget is not None:  # a TOML value bypasses argparse's type
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget < 0:
            refuse(f"--moss_ckpt_reuse_attn_budget_gib must be a number of GiB >= 0 (or left out: no limit), got {budget!r}")
        if not reuse:
            logger.warning("MOSS-Image: --moss_ckpt_reuse_attn_budget_gib changes nothing without --moss_ckpt_reuse_attn"
                           " (it caps what that option stores)")
    if not (nograd or reuse):
        return
    # the option the messages name: the node comes with either one
    given = " / ".join(name for name, on in (("--moss_ckpt_nograd_merged", nograd), ("--moss_ckpt_reuse_attn", reuse)) if on)

    # the offload checkpointers and block swap use other checkpoint paths in MossBlock._run: the switch would be ignored
    for flag in ("unsloth_offload_checkpointing", "cpu_offload_checkpointing"):
        if getattr(args, flag, False):
            refuse(f"{given} replaces the plain gradient checkpoint of a joint block; not with --{flag},"
                   " which uses another one")
    if (getattr(args, "blocks_to_swap", None) or 0) > 0:
        refuse(f"{given} is not supported with --blocks_to_swap: a swapped block's weights change device"
               " between calls" + (", and the cached merged weights would stay in VRAM next to them" if nograd else
                                   ", and the node's no-grad first pass and recompute expect them where they were"))
    if getattr(args, "use_ramtorch", False) or getattr(args, "use_ramtorch_network", False):
        refuse(f"{given} is not supported with RamTorch (use_ramtorch / use_ramtorch_network): "
               + ("the streamed base weights are not in VRAM to merge from, so nothing would be cached" if nograd else
                  "the no-grad first-pass checkpoint is only tested with the base weights resident in VRAM")
               + ". Turn one of them off")
    if getattr(args, "moss_stock_lycoris", False):
        # prepare_moss_runtime does not install the fast LoKr+DoRA forward then, and LyCORIS's stock forward never reads
        # the cache (lycoris_fast.ckpt_merged_modules returns nothing while that forward is not installed)
        refuse(f"{given} is built on the fork's fast LoKr+DoRA forward ("
               + ("it caches the merged weights for it" if nograd else "the only forward the node is tested with")
               + "); not with --moss_stock_lycoris, whose stock LyCORIS forward "
               + ("never reads that cache" if nograd else "is not covered") + ". Remove one of the two options")
    if reuse:
        if getattr(args, "attn_mode", None) != "flash":
            refuse("--moss_ckpt_reuse_attn needs attn_mode = \"flash\": it calls flash-attn's backward kernel with the output"
                   f" and softmax_lse the first pass stored (attn_mode is {getattr(args, 'attn_mode', None) or 'torch'!r})")
        if getattr(args, "split_attn", False):
            refuse("--moss_ckpt_reuse_attn is not supported with --split_attn (one flash call per row)")
        if not attention.flash_reuse_available():
            refuse("--moss_ckpt_reuse_attn cannot work with this flash-attn: it has no separate backward op"
                   " (flash_attn_interface._wrapped_flash_attn_backward; flash-attn 2.8.4 has it). Update flash-attn or"
                   " remove the option")
    if nograd and reuse and budget is not None and getattr(args, "moss_block_compile", False):
        # with the cache on, a block over the budget keeps the node but stores nothing (MossBlock._forward_no_stash):
        # the compiled entry the blocks share then holds the first pass with / without a stash and the recompute with /
        # without the stored tensors, each for the adapted blocks and for a block left out of the network - 8 graphs
        # where the other combinations need 4 (no budget) or 6 (reuse-only with a budget). The smallest limit
        # validate_moss_block_compile_args accepts (8) would leave no graph for anything else (a padded batch).
        limit = int(getattr(args, "moss_block_compile_recompile_limit", 64) or 0)
        if limit < CKPT_BUDGET_CACHED_RECOMPILE_LIMIT:
            refuse("--moss_ckpt_nograd_merged + --moss_ckpt_reuse_attn + --moss_ckpt_reuse_attn_budget_gib under"
                   " --moss_block_compile compiles 8 graphs on the shared block entry (first pass with / without the"
                   " stored attention, recompute with / without it, for adapted and left-out blocks): it needs"
                   f" --moss_block_compile_recompile_limit >= {CKPT_BUDGET_CACHED_RECOMPILE_LIMIT} (got {limit}; default 64)")
    if not getattr(args, "gradient_checkpointing", False):
        logger.warning(f"MOSS-Image: {given} changes nothing without --gradient_checkpointing")
    elif not ckpt_node_enabled() or (nograd and not ckpt_nograd_merged_enabled()):
        # a harness that builds MossNetworkTrainer without prepare_moss_runtime: the process-wide switch is still off
        logger.warning(f"MOSS-Image: {given} has no effect here: prepare_moss_runtime (moss_train_network.py) did not"
                       " switch it on; the joint blocks keep the plain gradient checkpoint")
    batch = largest_batch_size(args, dataset_group_limits(train_dataset_group))
    if batch > 1:
        # either option: under block compile a padded batch keeps the plain checkpoint (MossBlock._ckpt_keeps_plain)
        logger.warning(f"MOSS-Image: {given} with batch > 1 (largest batch size {batch}): a batch whose captions differ in"
                       " token count is padded (flash varlen) - nothing is stored or reused there, and under"
                       " --moss_block_compile such a micro-step keeps the plain gradient checkpoint (every padded length"
                       " still compiles a graph of its own, as without the option; the no-grad first pass would make it"
                       " two)")


def ckpt_reuse_outside_grad_note(network) -> Optional[str]:
    """The log line for --moss_ckpt_nograd_merged / --moss_ckpt_reuse_attn (switched on) with trained weights outside
    the joint blocks' six Linears
    and the text refiner (a LyCORIS preset / target list that reaches time_embedding or shared_block_modulation); None
    otherwise. The shared modulation enters every joint block from outside and the checkpoint node hands back one
    gradient per block for it. Each block reads it once, so on CPU the gradients are the plain checkpoint's bit for bit
    (tests/test_moss_ckpt_reuse.py), but that order is the autograd engine's, not something this code pins: the line
    tells the user which part of the run rests on it. (The text refiner needs no line: its output is part of joint block
    0's x, which has one consumer.)"""
    if network is None or not ckpt_node_enabled():
        return None
    trained = sorted({word for name, parameter in getattr(network, "named_parameters", lambda: [])()
                      if parameter.requires_grad
                      for word in ("time_embedding", "shared_block_modulation") if word in name})
    if not trained:
        return None
    return (f"MOSS-Image: {ckpt_reuse_options()} with {' and '.join(trained)} trained: their gradients reach the shared"
            " modulation through one checkpoint node per joint block (bit-identical to the plain checkpoint in the CPU"
            " tests; not measured on the GPU)")


def ckpt_reuse_options() -> str:
    """The options that are switched on, as the log lines name them."""
    names = [name for name, on in (("--moss_ckpt_nograd_merged", ckpt_nograd_merged_enabled()),
                                   ("--moss_ckpt_reuse_attn", ckpt_reuse_attn_enabled())) if on]
    return " / ".join(names) or "--moss_ckpt_nograd_merged / --moss_ckpt_reuse_attn"


CAPTION_TOKENS_LONG = 1000  # the "long caption" of the per-tier lines (the text tokens of the joint sequence)


def ckpt_reuse_vram_estimate(model, reuse_attn: bool, image_tokens: int, batch_size: int = 1, cached: bool = True,
                             stream_bytes: int = 2, budget_bytes: Optional[int] = None,
                             tiers: Optional[Sequence[Tuple[str, int, Tuple[int, int]]]] = None,
                             caption_tokens: int = 0, adapter_bytes: int = 0) -> Dict[str, object]:
    """Bytes --moss_ckpt_nograd_merged / --moss_ckpt_reuse_attn add to the peak of a training micro-step of `model` (a
    MossTransformer with the network applied), counted from the modules, not measured:
    cache   - the merged-weight cache: one autocast-dtype (2-byte) copy of every adapted Linear of the joint blocks, held
              while training (cached=False: reuse-only, or no bf16 / fp16 autocast - nothing is cached);
    stash   - the stored joint attention at `image_tokens` (+ `caption_tokens`) of the joint blocks that reuse (each
              block's out [L, dim] in the autocast dtype + softmax_lse [heads, L] fp32 lives until that block's backward,
              so at the start of backward every block's is alive), for `batch_size` samples of equal text length;
    blocks / total_blocks - how many joint blocks reuse there under `budget_bytes` (--moss_ckpt_reuse_attn_budget_gib;
              None: all of them), the count MossBlock._ckpt_reuses arrives at;
    stash_per_text_token - what each caption token adds to `stash` while the block count stays (a longer caption can
              only lower the count under a budget: the stash never passes the budget);
    budget  - budget_bytes as given;
    tiers   - for every (tier label, image tokens, bucket) of `tiers` (dataset_group_tiers): per_block bytes, the
              blocks that reuse without a caption (blocks) and with CAPTION_TOKENS_LONG text tokens (blocks_long), and
              the stash of the first;
    inputs  - the block inputs the checkpoint keeps for the recompute at `image_tokens` (the plain checkpoint keeps the
              same): a floor for what a micro-step needs next to the weights, in the stream's dtype (stream_bytes);
    adapter - adapter_bytes as given (adapter_vram_bytes(network): what the trained network still adds to the VRAM
              allocated at start-up; the plain run needs it too)."""
    blocks = list(model.layers)
    # the modules NoGradFirstPassCheckpointer caches for a MossBlock (anima_models._ckpt_cached_lokrs: every adapted
    # Linear of the block), looked up here without leaving that function's per-block list behind
    elements = 0 if not cached else sum(
        lokr.org_module[0].weight.numel() for block in blocks
        for lokr in ckpt_merged_modules(block, [m for m in block.modules() if isinstance(m, torch.nn.Linear)]))
    dim, heads = int(model.config["dim"]), int(model.config["num_heads"])
    rows, total = int(batch_size), len(blocks)
    per_token = ckpt_stash_bytes_per_token(dim, heads)  # one block, one row

    def reusing(tokens: int) -> int:
        return min(total, ckpt_reuse_block_count(budget_bytes, tokens, rows, per_token)) if reuse_attn else 0

    joint = int(image_tokens) + int(caption_tokens)
    count = reusing(joint)
    per_tier = []
    for tier, tokens, bucket in tiers or ():
        n = reusing(int(tokens))
        per_tier.append({"tier": tier, "bucket": bucket, "image_tokens": int(tokens), "per_block": int(tokens) * rows * per_token,
                         "blocks": n, "blocks_long": reusing(int(tokens) + CAPTION_TOKENS_LONG),
                         "stash": n * int(tokens) * rows * per_token})
    return {"cache": elements * 2, "stash": count * joint * rows * per_token, "stash_per_text_token": count * rows * per_token,
            "blocks": count, "total_blocks": total, "budget": budget_bytes, "tiers": per_tier, "reuse": bool(reuse_attn),
            "cached": bool(cached), "inputs": total * int(image_tokens) * rows * dim * int(stream_bytes),
            "adapter": int(adapter_bytes)}


def adapter_vram_bytes(network) -> int:
    """Bytes the trained network still adds to the VRAM that is allocated when the start-up estimate is made
    (after_network_loaded: before accelerator.prepare moves the network): one gradient per trained parameter, and the
    trained parameters themselves while they are not on a CUDA device yet. The optimizer's state comes on top and is not
    counted (it depends on the optimizer: CAME keeps about one more copy). 0 for whatever has no parameters."""
    total = 0
    try:
        for parameter in network.parameters():
            if parameter.requires_grad:
                size = parameter.numel() * parameter.element_size()
                total += size if parameter.device.type == "cuda" else 2 * size
    except Exception as error:  # noqa: BLE001 - it only feeds the estimate line: never stop a training
        logger.debug(f"MOSS-Image: adapter size not read ({type(error).__name__}: {error})")
        return 0
    return total


def cuda_memory_stats(device, with_free: bool = True) -> Optional[Dict[str, int]]:
    """Memory counters of a CUDA device in bytes (total, allocated, reserved, peak_allocated, peak_reserved of this
    process; free on the whole device with with_free): read-only driver / allocator counters, no synchronize. None for
    another device. The peaks are the run's: under CUDA_MEMLOG torch's peak counters are reset at every memlog row, and
    library/cuda_memory.run_peaks keeps what those resets cleared."""
    if device is None or torch.device(device).type != "cuda" or not torch.cuda.is_available():
        return None
    peaks = cuda_memory.run_peaks(device) or (int(torch.cuda.max_memory_allocated(device)),
                                              int(torch.cuda.max_memory_reserved(device)))
    stats = {"total": int(torch.cuda.get_device_properties(device).total_memory),
             "allocated": int(torch.cuda.memory_allocated(device)),
             "reserved": int(torch.cuda.memory_reserved(device)),
             "peak_allocated": int(peaks[0]),
             "peak_reserved": int(peaks[1])}
    if with_free:
        stats["free"] = int(torch.cuda.mem_get_info(device)[0])
    return stats


CKPT_VRAM_TIGHT = 0.80  # start-up: what can be counted (below) + the options' extra above this share of the card: warning
CKPT_VRAM_AT_LIMIT = 0.94  # run time: this process's peak reserved at / above this share of the card, or ...
CKPT_DEVICE_FREE_MIN = 0.02  # ... the device (peak reserved + what is held outside this process) within this share of it


def outside_bytes(stats: Optional[Dict[str, int]]) -> Optional[int]:
    """Bytes of the device held outside this process's allocator right now (other programs, the desktop, the CUDA
    context): device used - reserved. None when the counters do not say (no `free` / `reserved` in stats). The user's
    line is the device total - Windows spills into shared memory when the DEVICE is full, whoever holds it."""
    if not stats or "free" not in stats or "reserved" not in stats:
        return None
    return max(0, stats["total"] - stats["free"] - stats["reserved"])


def _gib(count) -> str:
    return f"{count / 2**30:.2f} GiB"


def ckpt_reuse_vram_note(estimate: Dict[str, int], bucket: Optional[Tuple[int, int]], stats: Optional[Dict[str, int]] = None
                         ) -> Tuple[str, bool]:
    """The start-up line for ckpt_reuse_vram_estimate's numbers and whether it is a warning: with the card's counters
    (cuda_memory_stats, read once the DiT is on the device), True when what can be counted - the weights allocated now,
    what other processes and the CUDA context hold on the device (outside_bytes), the block inputs the checkpoint
    keeps, the adapter's parameters and gradients (estimate["adapter"]) - plus the options' extra pass CKPT_VRAM_TIGHT
    of the card. What is left then has to hold the optimizer's state, one block's recompute and backward and the
    allocator's slack. The user's plain 2048-tier run (LoKr f4 + DoRA, CAME; 2026-11-14) peaked at 27.9 GiB on the
    device where ~21.8 GiB can be counted this way: ~6 GiB for those, about the 20 % of a 32 GB card the line leaves.
    An estimate from module sizes: the measured peak is the epoch-end line.
    Reuse-only (estimate["cached"] False) names no cache; with the block counts of the estimate (a budget, or the
    per-tier list) the line says how many blocks reuse at the largest bucket and in every tier."""
    extra = estimate["cache"] + estimate["stash"]
    where = f"the largest bucket {bucket[0]}x{bucket[1]}" if bucket else "an unknown largest bucket (0 image tokens)"
    stash = (f"{_gib(estimate['stash'])} stored joint attention at {where}"
             f" (+{_gib(1000 * estimate['stash_per_text_token'])} per 1000 caption tokens)")
    if "blocks" in estimate and estimate.get("reuse", True):
        budget = estimate.get("budget")
        stash += (f", {estimate['blocks']} of {estimate['total_blocks']} blocks reusing there ("
                  + (f"budget {_gib(budget)}: the stored attention never passes it" if budget is not None else "no budget")
                  + ")")
    if estimate.get("cached", True):
        line = (f"MOSS-Image: estimated extra VRAM of the no-grad first pass options: {_gib(estimate['cache'])} merged-weight"
                f" cache while training + {stash} = {_gib(extra)} on top of the plain run")
    else:
        line = f"MOSS-Image: estimated extra VRAM of --moss_ckpt_reuse_attn (nothing cached): {stash} on top of the plain run"
    tiers = [tier for tier in estimate.get("tiers") or () if estimate.get("reuse", True)]
    if tiers:
        def tier_text(tier):
            blocks = f"{tier['blocks']} of {estimate['total_blocks']} blocks"
            if tier["blocks_long"] != tier["blocks"]:
                blocks += f" ({tier['blocks_long']} with a {CAPTION_TOKENS_LONG}-token caption)"
            return f"{tier['tier']} ({tier['bucket'][0]}x{tier['bucket'][1]}): {blocks}, {_gib(tier['stash'])}"

        line += ". Stored attention per resolution tier (its largest bucket, no caption): " + "; ".join(map(tier_text, tiers))
    if not stats:
        return line + ".", False
    outside, adapter = outside_bytes(stats), int(estimate.get("adapter", 0) or 0)
    floor = stats["allocated"] + (outside or 0) + estimate["inputs"] + adapter
    left = stats["total"] - floor - extra
    line += f". This card: {_gib(stats['total'])}, {_gib(stats['allocated'])} allocated now"
    if outside is not None:
        line += f", {_gib(outside)} held outside this process (other programs, the CUDA context)"
    line += f", {_gib(estimate['inputs'])} of block inputs kept at that bucket"
    if adapter:
        line += f", {_gib(adapter)} for the adapter's parameters and gradients"
    line += (f": {_gib(left)} left for the optimizer's state, one block's recompute and backward, and the allocator"
             " (not counted here).")
    tight = floor + extra > CKPT_VRAM_TIGHT * stats["total"]
    if tight:
        line += (" That is tight. When VRAM runs out, Windows' NVIDIA driver can spill into shared memory instead of raising"
                 " an error (steps several times slower): watch the peak line at the epoch end, and if it reaches the"
                 " card's size train without --moss_ckpt_nograd_merged / --moss_ckpt_reuse_attn, with a (smaller)"
                 " --moss_ckpt_reuse_attn_budget_gib or without the largest resolution tier.")
    return line, tight


def ckpt_reuse_peak_note(stats: Dict[str, int], options: str = "--moss_ckpt_nograd_merged") -> Tuple[str, bool]:
    """The measured-peak line (cuda_memory_stats) and whether the run is at the card's limit: peak reserved at or above
    CKPT_VRAM_AT_LIMIT of the card, or (when the device was read) the DEVICE within CKPT_DEVICE_FREE_MIN of full - now,
    or at the peak: peak reserved + what is held outside this process (outside_bytes; a desktop with ~2.3 GiB in use
    fills a 31.84 GiB card at ~29.5 GiB reserved, under the 94 % line). On Windows that is where the driver's sysmem
    fallback starts: the step time, not an out-of-memory error, shows it. options: what the line names
    (ckpt_reuse_options(): the switches that are on)."""
    total = stats["total"]
    outside = outside_bytes(stats)
    device_peak = None if outside is None else stats["peak_reserved"] + outside
    at_limit = (stats["peak_reserved"] >= CKPT_VRAM_AT_LIMIT * total
                or stats.get("free", total) <= CKPT_DEVICE_FREE_MIN * total
                or (device_peak is not None and device_peak >= (1.0 - CKPT_DEVICE_FREE_MIN) * total))
    line = (f"MOSS-Image: {options} VRAM peak so far: {_gib(stats['peak_allocated'])} allocated,"
            f" {_gib(stats['peak_reserved'])} reserved of {_gib(total)}")
    if "free" in stats:
        line += f" ({_gib(stats['free'])} free on the device now)"
    if device_peak is not None:
        line += (f"; with the {_gib(outside)} held outside this process (other programs, the CUDA context) the device"
                 f" peak is {_gib(device_peak)}")
    if at_limit:
        line += (". VRAM is at the card's limit: on Windows the NVIDIA driver may be spilling into shared memory (no error,"
                 " steps several times slower). If s/step went up, train without --moss_ckpt_nograd_merged /"
                 " --moss_ckpt_reuse_attn (the start-up line has what they add), with a (smaller)"
                 " --moss_ckpt_reuse_attn_budget_gib or without the largest resolution tier.")
    return line, at_limit


def validate_moss_stream_scale_args(args) -> None:
    """--moss_stream_scale C (library/moss/models.py, MossTransformer.set_stream_scale): a power of two in (0, 1], else
    ValueError (a TOML value bypasses argparse's type: stored back as a float). On: the start-up line, and a warning
    where it adds nothing (with --moss_stream_fp32 the stream is already fp32; bf16 / fp32 have fp32's range)."""
    value = getattr(args, "moss_stream_scale", 1.0)
    problem = stream_scale_problem(value)
    if problem:
        raise ValueError(f"MOSS-Image: --moss_stream_scale {problem}")
    scale = float(value)
    args.moss_stream_scale = scale
    if scale == 1.0:
        return
    logger.info(f"MOSS-Image: --moss_stream_scale {scale:g} (1/{1 / scale:g}): the joint residual stream runs {1 / scale:g}x"
                f" smaller (its largest value x {scale:g}); norm1 / norm2 / final_norm_video eps 1e-6 x C^2 ="
                f" {1e-6 * scale * scale:.6g}, the weights and the adapter's inputs, outputs and gradients are unchanged")
    if getattr(args, "moss_stream_fp32", False):
        logger.warning("MOSS-Image: --moss_stream_scale with --moss_stream_fp32 is redundant: the fp32 stream does not"
                       " overflow (both stay exact; drop one - --moss_stream_fp32 costs the VRAM)")
    elif not (getattr(args, "mixed_precision", None) == "fp16" or getattr(args, "full_fp16", False)):
        logger.warning(f"MOSS-Image: --moss_stream_scale only matters for fp16 (mixed_precision is"
                       f" {getattr(args, 'mixed_precision', None)!r}): bf16 and fp32 have fp32's range, the stream cannot"
                       " overflow there")


def validate_moss_args(args, train_dataset_group, val_dataset_group) -> None:
    """Refuse option combinations the MOSS path does not support (ValueError with the reason)."""

    def fail(message):
        raise ValueError(f"MOSS-Image: {message}")

    if not args.pretrained_model_name_or_path or not os.path.isfile(args.pretrained_model_name_or_path):
        fail(f"pretrained_model_name_or_path (moss_image_base_bf16.safetensors) not found: {args.pretrained_model_name_or_path}")
    if not args.vae or not os.path.isfile(args.vae):
        fail(f"vae (moss_image_vae_fp32.safetensors) not found: {args.vae}")
    if args.timestep_sampling not in TIMESTEP_CHOICES:  # TOML values bypass argparse choices
        fail(f"timestep_sampling must be one of {TIMESTEP_CHOICES} (flux_shift assumes 8x latents), got {args.timestep_sampling}")
    if not 1 <= int(args.moss_max_token_length) <= 4096:
        fail("moss_max_token_length must be in 1..4096")
    if getattr(args, "moss_tier_timing", False) and int(getattr(args, "moss_tier_timing_every", 50) or 0) < 1:
        fail(f"--moss_tier_timing_every must be >= 1 (optimizer steps between reports), got {args.moss_tier_timing_every}")
    if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
        args.cache_text_encoder_outputs = True
    if getattr(args, "moss_stream_fp32", False):
        for flag in ("full_bf16", "full_fp16", "fp8_base", "fp8_base_unet", "deepspeed"):
            if getattr(args, flag, False):  # outside autocast needs fp32 network and base weights; deepspeed skips the switch
                fail(f"--moss_stream_fp32 is not supported with --{flag}")
        if args.mixed_precision not in ("bf16", "fp16"):
            logger.warning("MOSS-Image: --moss_stream_fp32 changes nothing without bf16 or fp16 mixed precision")
    validate_moss_stream_scale_args(args)
    if not args.cache_text_encoder_outputs:
        fail("--cache_text_encoder_outputs is required (the 16 GB Qwen3.5-9B is only loaded while caching; the user guide"
             " keeps the text encoder frozen)")
    if not args.network_train_unet_only:
        logger.warning("MOSS-Image: the text encoder is frozen; forcing --network_train_unet_only")
        args.network_train_unet_only = True
    for flag in ("weighted_captions", "adaptive_timestep_sampling", "network_train_text_encoder_only", "flow_model"):
        if getattr(args, flag, False):
            fail(f"--{flag} is not supported")
    if getattr(args, "fp8_base", False) or getattr(args, "fp8_base_unet", False):
        logger.warning("MOSS-Image: fp8_base is not supported, disabled")
        args.fp8_base = False
        args.fp8_base_unet = False
    args.fp8_scaled = False
    if (args.blocks_to_swap or 0) > 0 and (args.cpu_offload_checkpointing or args.unsloth_offload_checkpointing):
        fail("blocks_to_swap cannot be combined with cpu/unsloth offload checkpointing")
    if args.unsloth_offload_checkpointing:
        if args.cpu_offload_checkpointing:
            fail("use either --unsloth_offload_checkpointing or --cpu_offload_checkpointing")
        if not args.gradient_checkpointing:
            logger.warning("unsloth_offload_checkpointing is on, enabling gradient_checkpointing")
            args.gradient_checkpointing = True
    validate_moss_block_compile_args(args)
    # after the unsloth check above (it may switch gradient_checkpointing on)
    validate_moss_ckpt_reuse_args(args, train_dataset_group)
    if getattr(args, "hf_scale", 0.0) and getattr(args, "hf_patch", 1) != 1:
        logger.warning("MOSS latents are already 2x2 packed tokens: setting hf_patch=1")
        args.hf_patch = 1

    assert train_dataset_group.is_text_encoder_output_cacheable(cache_supports_dropout=True), (
        "when caching text-encoder outputs, shuffle_caption / caption_tag_dropout_rate need --cache_caption_variants N>1,"
        " and token_warmup_step cannot be used")
    for dataset in _datasets([train_dataset_group, val_dataset_group]):
        variants = int(getattr(dataset, "caption_aug_variant_count", 0) or 0)
        if variants > 1 and int(getattr(dataset, "aug_refresh_epochs", 0) or 0) > 0:
            fail("--cache_aug_refresh_epochs is not supported with caption variants (it re-encodes through the text encoder)")
        for subset in getattr(dataset, "subsets", []):
            if int(getattr(subset, "caption_dropout_every_n_epochs", 0) or 0) > 0:
                fail("caption_dropout_every_n_epochs is not supported (it zeroes cached outputs; MOSS needs >= 1 text"
                     " token). Use caption_dropout_rate with --cache_caption_variants N>1")
            if float(getattr(subset, "caption_dropout_rate", 0.0) or 0.0) > 0.0 and variants <= 1:
                fail("caption_dropout_rate > 0 needs --cache_caption_variants N>1 (a dropped caption is encoded as ''"
                     " through the chat template, the official unconditional prompt)")
    train_dataset_group.verify_bucket_reso_steps(16)
    if val_dataset_group is not None:
        val_dataset_group.verify_bucket_reso_steps(16)
    logger.info("MOSS-Image: " + describe_moss_sampling(args))


def show_timesteps(args, n_try: int = 100000) -> None:
    """--show_timesteps: sample the configured timestep distribution on the CPU and print / plot it (no model).

    Goes through the trainer's own sampling path (flow.get_noisy_model_input_and_timesteps with the moss pmf sampler,
    the sigmoid presets and scale_resume) and loss weighting, so the histogram is what training samples.
    library/timestep_visualization.py is upstream sd-scripts' renderer."""
    from library import sd3_train_utils, timestep_visualization
    from library.anima_train_utils import compute_loss_weighting_for_anima

    from .flow import NUM_TRAIN_TIMESTEPS, MossSigmaSampler, get_noisy_model_input_and_timesteps

    timestep_presets.resolve(args)
    epoch = getattr(args, "show_timesteps_epoch", None)
    if epoch is not None:
        timestep_presets.set_epoch(args, int(epoch))
    scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS,
                                                                shift=args.discrete_flow_shift)
    sampler = MossSigmaSampler(args) if args.timestep_sampling == "moss" else None
    device, dtype = torch.device("cpu"), torch.float32

    def sample_timesteps(batch_size: int) -> torch.Tensor:
        latents = torch.zeros(batch_size, MOSS_DIT_CONFIG["latent_channels"], 4, 4, dtype=dtype, device=device)
        _, timesteps, _ = get_noisy_model_input_and_timesteps(args, scheduler, latents, torch.ones_like(latents), device,
                                                              dtype, sampler)
        return timesteps

    def compute_weighting(timesteps: torch.Tensor) -> torch.Tensor:
        sigmas = timesteps / NUM_TRAIN_TIMESTEPS
        weighting = compute_loss_weighting_for_anima(args.weighting_scheme, sigmas)
        if getattr(args, "min_snr_gamma", None):  # the trainer applies it on top (post_process_loss)
            from library.custom_train_functions import apply_snr_weight_for_flow_matching

            weighting = apply_snr_weight_for_flow_matching(weighting, sigmas, args.min_snr_gamma,
                                                           soft=bool(getattr(args, "min_snr_gamma_soft", False)))
        return weighting

    header = "MOSS-Image timestep distribution: " + describe_moss_sampling(args)
    if epoch is not None and args.timestep_sampling == "sigmoid":
        header += f" (epoch {int(epoch)}: preset {timestep_presets.active_sigmoid_preset(args) or 'numeric scale'})"
    elif epoch is not None:
        header += f" (epoch {int(epoch)})"
    timestep_visualization.show_timestep_distribution(args.show_timesteps, sample_timesteps, compute_weighting,
                                                      num_train_timesteps=NUM_TRAIN_TIMESTEPS, n_try=n_try, header=header)


def parse_moss_exclude_blocks(value, num_layers: int = MOSS_DIT_CONFIG["num_layers"]) -> List[int]:
    """--moss_lora_exclude_blocks: "22", "22, 23", 22 or [22, 23] (TOML values arrive unconverted) -> sorted indices."""
    if value is None or value == "" or value == []:
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    blocks = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        if isinstance(item, bool) or not text.lstrip("-").isdigit():
            raise ValueError(f"MOSS-Image: moss_lora_exclude_blocks takes block indices, got {value!r}")
        index = int(text)
        if not 0 <= index < num_layers:
            raise ValueError(f"MOSS-Image: moss_lora_exclude_blocks index {index} is outside 0..{num_layers - 1}")
        blocks.add(index)
    return sorted(blocks)


def moss_exclude_blocks_arg(value):
    """argparse type of --moss_lora_exclude_blocks: checks the indices and keeps the value as given. config_validation
    converts a TOML string with a custom type (a bad "22,x" fails when the config is read, with the file and line) and
    passes an array or integer through unconverted to parse_moss_exclude_blocks; type=str refused [22, 23]."""
    try:
        parse_moss_exclude_blocks(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    return value


def moss_block_exclude_patterns(blocks: Sequence[int]) -> List[str]:
    """Regexes for network_args exclude_patterns. LyCORIS full-matches module paths ("layers.22" and its children),
    networks.lora_moss full-matches "layers.22.proj.q" etc."""
    return [rf"layers\.{index}(\..*)?" for index in blocks]


def apply_moss_block_exclusion(args) -> List[int]:
    """Merge the excluded blocks into network_args exclude_patterns (both network modules read it). Idempotent."""
    if getattr(args, "_moss_excluded_blocks", None) is not None:
        return args._moss_excluded_blocks
    blocks = parse_moss_exclude_blocks(getattr(args, "moss_lora_exclude_blocks", ""))
    args._moss_excluded_blocks = blocks
    if not blocks:
        return blocks
    patterns = moss_block_exclude_patterns(blocks)
    network_args = list(getattr(args, "network_args", None) or [])
    if "lycoris" in str(getattr(args, "network_module", "")):
        # LyCORIS ignores a preset's exclude_name once exclude_patterns is given; the MOSS presets have none.
        preset = next((str(e).partition("=")[2].strip() for e in network_args if str(e).partition("=")[0].strip() == "preset"),
                      "full")
        if preset not in ("moss", "moss-text"):
            raise ValueError(f"MOSS-Image: moss_lora_exclude_blocks needs LyCORIS preset=moss or moss-text, got {preset!r}")
    for i, entry in enumerate(network_args):
        key, _, value = str(entry).partition("=")
        if key.strip() == "exclude_patterns":
            existing = ast.literal_eval(value.strip())  # the network modules parse it the same way
            existing = list(existing) if isinstance(existing, (list, tuple)) else [existing]
            network_args[i] = f"exclude_patterns={existing + [p for p in patterns if p not in existing]!r}"
            break
    else:
        network_args.append(f"exclude_patterns={patterns!r}")
    args.network_args = network_args
    logger.info(f"MOSS-Image: joint blocks {blocks} are left out of the network (exclude_patterns {patterns})")
    return blocks


def moss_metadata(args) -> Dict[str, str]:
    excluded = getattr(args, "_moss_excluded_blocks", None) or []
    return {
        "ss_moss_lora_exclude_blocks": ",".join(str(index) for index in excluded),
        "ss_base_model_version": "moss_image",
        "ss_moss_text_encoder": os.path.basename(str(args.moss_text_encoder)),
        "ss_moss_max_token_length": str(args.moss_max_token_length),
        "ss_timestep_sampling": str(args.timestep_sampling),
        "ss_moss_logit_mean": str(args.moss_logit_mean),
        "ss_moss_logit_std": str(args.moss_logit_std),
        "ss_moss_sigma_cutoff": str(args.moss_sigma_cutoff),
        "ss_moss_uniform_mix": str(args.moss_uniform_mix),
        **timestep_presets.metadata(args),
        "ss_discrete_flow_shift": str(args.discrete_flow_shift),
        "ss_weighting_scheme": str(args.weighting_scheme),
        "ss_logit_mean": str(args.logit_mean),
        "ss_logit_std": str(args.logit_std),
    }


# ---- sample prompt text-encoder outputs: kept on disk so a run with complete caches never loads the 9B model -------
def sample_prompt_cache_dir(args) -> Optional[str]:
    if not args.sample_prompts:
        return None
    return os.path.splitext(os.path.abspath(args.sample_prompts))[0] + "_moss_te"


def sample_prompt_cache_path(cache_dir: str, signature: str, prompt: str) -> str:
    digest = hashlib.sha1((signature + "\n" + prompt).encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, digest + ".safetensors")


def load_cached_sample_prompt(path: str):
    if not os.path.isfile(path):
        return None
    state = load_file(path)
    return [state["text_embeds"], state["text_mask"]]


def save_cached_sample_prompt(path: str, outputs) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({"text_embeds": outputs[0].contiguous().cpu(), "text_mask": outputs[1].contiguous().cpu()}, path)


def sample_prompt_texts(args) -> List[str]:
    texts: List[str] = []
    for prompt_dict in train_util.load_prompts(args.sample_prompts):
        for text in (prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "") or ""):
            if text not in texts:
                texts.append(text)
    return texts


# ---- sampling (official MOSSImageScheduler grid + Euler + CFG with negative-branch layer skip) ----------------------
def moss_inference_sigmas(steps: int, shift: float = DEFAULT_SAMPLE_SHIFT, total: int = 1000, training_shift: float = 1.0):
    """MOSSImageScheduler.set_timesteps: shifted grid snapped to the nearest training sigma (ties -> lower), + 0."""

    def grid(count, s):
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(steps, shift)
    training = grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


@torch.no_grad()
def moss_sample(dit, cond, neg, height: int, width: int, steps: int, scale: float, shift: float, seed: Optional[int],
                device, skip_layers: Sequence[int] = (22,)) -> torch.Tensor:
    """Returns DiT latents [1,128,H/16,W/16] (fp32). cond/neg = (text_embeds [1,L,4096], text_mask [1,L])."""
    generator = torch.Generator("cpu").manual_seed(seed) if seed is not None else None
    latents = torch.randn((1, 128, height // 16, width // 16), generator=generator, dtype=torch.float32).to(device)
    sigmas = moss_inference_sigmas(steps, shift).to(device)
    text, mask = cond[0].to(device, dtype=dit.dtype), cond[1].to(device).bool()
    if neg is not None:
        neg_text, neg_mask = neg[0].to(device, dtype=dit.dtype), neg[1].to(device).bool()
    for i in range(len(sigmas) - 1):
        t = (sigmas[i] * 1000.0).reshape(1)
        pred = dit(latents, t, text, mask).float()
        if scale != 1.0 and neg is not None:
            uncond = dit(latents, t, neg_text, neg_mask, skip_layers=skip_layers).float()
            pred = uncond + scale * (pred - uncond)
        latents = latents + (sigmas[i + 1] - sigmas[i]) * pred
    return latents


def sample_images(accelerator, args, epoch, steps, dit, vae, sample_prompts_te_outputs: Dict[str, list]):
    if not train_util.sample_images_check(args, epoch, steps):
        return
    if not sample_prompts_te_outputs:
        logger.warning("MOSS sample images: no cached prompt embeddings, skipping")
        return
    if not accelerator.is_main_process:  # one process samples; the others would write the same file names
        accelerator.wait_for_everyone()
        return
    logger.info(f"Generating MOSS sample images at step {steps}")
    dit = accelerator.unwrap_model(dit)
    if ckpt_nograd_merged_enabled():
        # --moss_ckpt_nograd_merged: sampling never reads the checkpoint merged-weight cache (~8.65 GiB) and builds its
        # own merged weights below, so the cache is dropped for the round - only here, once a round really runs (the
        # trainer calls this function after every step). The next training first pass builds it again, which the
        # invalidation after a sample round (MossNetworkTrainer.after_sample_images) makes it do anyway.
        freed = ckpt_merged_release(dit)
        if freed:
            logger.info(f"  --moss_ckpt_nograd_merged: released {freed / 2**30:.2f} GiB of cached merged LoKr weights for"
                        " the sample round")
    dit.switch_block_swap_for_inference()
    prompts = train_util.load_prompts(args.sample_prompts)
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)
    skip = [int(x) for x in str(args.moss_sample_uncond_skip_layers or "").replace(",", " ").split()]

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    try:
        with torch.no_grad(), accelerator.autocast():
            device_type = accelerator.device.type
            # Merged LoKr weights built once per round (bit-identical to the per-call rebuild under autocast). Not with
            # block swap (weights move between CPU and GPU), without autocast, or when switched off. All prompts run
            # inside the cache; the VAE decodes after it is released so the two never hold VRAM at the same time.
            use_cache = (not getattr(args, "moss_stock_lycoris", False) and not getattr(args, "moss_no_sample_cache", False)
                         and not dit.blocks_to_swap and torch.is_autocast_enabled(device_type))
            cache = (cached_lokr_weights(dit, torch.get_autocast_dtype(device_type), accelerator.device) if use_cache
                     else contextlib.nullcontext())
            generated = []
            with cache:
                for prompt_dict in prompts:
                    dit.prepare_block_swap_before_forward()
                    latents = _generate_one(accelerator, args, dit, prompt_dict, sample_prompts_te_outputs, skip)
                    if latents is not None:
                        generated.append((prompt_dict, latents.cpu()))
            clean_memory_on_device(accelerator.device)
            for prompt_dict, latents in generated:
                _decode_and_save(accelerator, args, vae, save_dir, prompt_dict, epoch, steps, latents)
    finally:
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)
        dit.switch_block_swap_for_training()
        clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()


def _generate_one(accelerator, args, dit, prompt_dict, te_outputs, skip_layers) -> Optional[torch.Tensor]:
    """DiT sampling for one prompt line; returns latents [1,128,h,w] (fp32) or None when the prompt has no embedding."""
    prompt = prompt_dict.get("prompt", "")
    negative = prompt_dict.get("negative_prompt", "") or ""
    sample_steps = int(prompt_dict.get("sample_steps", 28))
    scale = float(prompt_dict.get("scale", 3.0))
    shift = float(prompt_dict.get("flow_shift", args.moss_sample_shift) or args.moss_sample_shift)
    seed = prompt_dict.get("seed")
    width = max(256, int(prompt_dict.get("width", 1024)) // 16 * 16)
    height = max(256, int(prompt_dict.get("height", 1024)) // 16 * 16)
    if prompt not in te_outputs or negative not in te_outputs:
        logger.warning(f"  no cached embedding for this prompt, skipped: {prompt[:80]}")
        return None
    logger.info(f"  prompt: {prompt}, size: {width}x{height}, steps: {sample_steps}, cfg: {scale}, shift: {shift}, seed: {seed}")
    clean_memory_on_device(accelerator.device)
    latents = moss_sample(dit, te_outputs[prompt], te_outputs[negative], height, width, sample_steps, scale, shift, seed,
                          accelerator.device, skip_layers)
    synchronize_device(accelerator.device)
    return latents


def _decode_and_save(accelerator, args, vae, save_dir, prompt_dict, epoch, steps, latents):
    seed = prompt_dict.get("seed")
    clean_memory_on_device(accelerator.device)
    org_device = vae.device
    vae.to(accelerator.device)
    with torch.autocast(device_type=accelerator.device.type, enabled=False):
        pixels = moss_vae_decode(vae, latents.to(accelerator.device))
    vae.to(org_device)
    clean_memory_on_device(accelerator.device)

    image = torch.clamp((pixels.float()[0] + 1.0) / 2.0, 0.0, 1.0)
    image = Image.fromarray((255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)).round().astype(np.uint8))
    ts = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    index = prompt_dict.get("enum", 0)
    name = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{index:02d}_{ts}{seed_suffix}.png"
    image.save(os.path.join(save_dir, name))
