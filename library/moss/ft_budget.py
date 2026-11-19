# MOSS-Image full fine-tune (moss_train.py): GPU / host RAM / disk / throughput estimates and the start-up checks.
#
# Design: docs/moss_full_finetune_design.md 2.1 (GPU), 2.2 (host RAM), 2.3 (disk), 3.5 (throughput) and 4.5 "Start-up
# checks". Every number here is an ESTIMATE (the GPU smoke run of design 7 measures them); tests/test_moss_ft_budget.py
# reads the design's 2.1 / 2.2 tables and checks these estimators against them, so changing one changes the other.
# Implementation notes: docs/moss_ft_impl_notes.md (T1).
#
# The estimators are pure functions of a few counts; the start-up checks take injectable callables (virtual_memory,
# disk_usage, exists) so tests can put them at their boundaries.

from __future__ import annotations

import functools
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

GIB = 1 << 30
MIB = 1 << 20

ADAMW_HOST, ADAFACTOR_FUSED, CAME_HOST = "adamw_host", "adafactor_fused", "came_host"
# The rotating-blocks mode (docs/moss_ft_carousel_plan.md). Its user-facing name lives HERE ONLY: the mode string, the
# --moss_<name>_* flags (ft_args.carousel_dest), the log prefix and the metadata derive from it, so a rename is this
# line (plus the user's TOML keys). The training-state files use the stable ROTATION_FORMAT id instead, so states
# written under an earlier name still resume.
CAROUSEL = "carousel"
ROTATION_FORMAT = "moss-ft-rotation/1"
CAROUSEL_DEFAULT_PERIOD = 50
CAROUSEL_GRADS_IN_FLIGHT_BYTES = GIB // 10  # one gradient tensor in flight (<= 100 MB), replaces the D2H ring bound
CAROUSEL_PINNED_LIMIT = 0.4  # --moss_carousel_host_pinned masters: refused past this share of physical RAM in pinned total

# ---- the model (design 1.1; counted on the meta device, test_model_constants_match_meta_model checks them) ----------
NUM_LAYERS = 42
HIDDEN = 3072
LATENT_CHANNELS = 128
LATENT_STRIDE = 16  # pixels per image token side (library/moss/train_utils.LATENT_STRIDE)
TEXT_HIDDEN = 4096  # Qwen3.5-9B hidden size: one cached text token = 4096 values
JOINT_LINEAR_PARAMS = 4_756_340_736  # 42 x (q, k, v, o, ffn_up_gate, ffn_down) = 42 x 113,246,208: bf16 compute copies
OTHER_PARAMS = 325_967_872  # text refiner, modulations, embedders, norms, head: fp32 under --moss_stream_fp32
TOTAL_PARAMS = JOINT_LINEAR_PARAMS + OTHER_PARAMS  # 5,082,308,608 in 462 tensors
PARAM_TENSORS = 462
REFINER_PARAMS = 226_568_704  # text_embedder.blocks.0/1 (text tokens only)
ROPE_BANDS = 16  # rope.bands: F32 [16], a buffer saved with the Base keys (463 keys in the Base file)
# fp32 CAME factors of every trainable tensor: rows + cols of exp_avg_sq and exp_avg_res for the 2-D ones, the
# full exp_avg_sq (+ exp_avg_res with non_factored_confidence) for the 1-D ones; Adafactor keeps half of that
CAME_FACTORED_BYTES = 22_925_312

# ---- GPU (design 2.1; conditions: bf16 compute copy, --moss_stream_fp32, every block checkpointed (plain)) ----------
DEFAULT_TEXT_TOKENS = 300  # text tokens per sample the design assumes (captions are not tokenized at start-up)
RECOMPUTE_BYTES_PER_TOKEN = 264_600  # measured (1.4): one joint block's saved activations in its recompute
RECOMPUTE_FACTOR = 1.4  # + that block's backward ("eager upper bound")
GRADS_IN_FLIGHT_BYTES = GIB // 2  # gradients waiting for the D2H ring (ring-bounded)
MISC_FIXED_BYTES = GIB // 4  # rope, modulation, head, fp64 loss, cuBLAS workspace: 0.25 GiB + 26 KiB per joint token
MISC_BYTES_PER_TOKEN = 26 * 1024  # (fits the 0.4-0.7 GiB row of 2.1 to its rounding; see the implementation notes)
REUSE_ATTN_BYTES_PER_TOKEN_BLOCK = 6_240  # --moss_ckpt_reuse_attn: stored output + softmax_lse per joint token
ALLOCATOR_SLACK = 0.07  # reserved over allocated
OUTSIDE_BYTES = int(2.3 * GIB)  # desktop, other programs, CUDA context (the user's measurement)
GPU_WARN_BYTES = int(30.5 * GIB)

# ---- host RAM (design 2.2) ------------------------------------------------------------------------------------------
PROCESS_BYTES = 8 * GIB  # process, CUDA host side, up to PROCESS_WORKERS DataLoader workers, CPU VAE (estimate)
PROCESS_WORKERS = 4  # DataLoader workers inside PROCESS_BYTES (design 2.2: "2-4 DataLoader workers")
WORKER_BYTES = GIB  # each worker above that: a spawned process importing torch + moss_train.py (0.81 GiB RSS measured
#                     on the CPU before any batch, a lower bound - review of 2026-11-17)
HOST_MARGIN_BYTES = 8 * GIB  # start-up refuses when available < total + this

# ---- disk (design 2.3) ----------------------------------------------------------------------------------------------
SAFETENSORS_HEADER_BYTES_PER_TENSOR = 128  # JSON header entry (name, dtype, shape, offsets), generous
STATE_SMALL_FILES_BYTES = MIB  # train_state.json, COMPLETE.json, scheduler.pt, rng.pt
TE_MASK_BYTES_PER_TOKEN = 4  # int32 text mask next to the embeddings
DTYPE_BYTES = {"fp32": 4, "float": 4, "float32": 4, "fp16": 2, "float16": 2, "bf16": 2, "bfloat16": 2}

# ---- throughput (design 3.5) ----------------------------------------------------------------------------------------
REFERENCE_IMAGE_TOKENS = 4096  # 1024^2
SECONDS_PER_IMAGE_1024 = 1.1  # GPU per image at 1024^2 (218.6 TFLOP per 4,396-token micro-step at ~195 TF/s)
ACCUMULATE_S = 1.1  # A: one accumulate pass over the whole DiT on the host (measured 1.0-1.15 s)
FINAL_S = 4.0  # F: convert + add + AdamW + cast at GA > 1 (measured 3.7-4.4 s)
FINAL_GA1_S = 3.6  # F at GA = 1 (measured 3.64 s)
CAME_FINAL_S = 20.5  # CAME core on the CPU (measured, 3.3)
BACKWARD_SHARE = 0.75  # the in-backward step overlaps the last micro-step's backward, ~0.75 of its GPU time
FLASH_BACKWARD_FACTOR = 2.5  # attention backward FLOPs over its forward

# ---- live text encoder (--moss_te_live, docs/moss_ft_te_live_fp16_plan.md 1.5; estimates for the smoke run) ---------
# the Qwen3.5-9B's decoder layers in the aligned bf16 layout of te_live_weights (meta model;
# test_te_constants_match_meta_model checks them): 24 GatedDeltaNet layers of 436,814,592 B and 8 full-attention
# layers of 419,447,808 B
TE_LAYER_BYTES_TOTAL = 13_839_132_672  # 12.889 GiB
TE_SLOT_BYTES = 436_814_592  # the largest layer: one device slot (416.6 MiB)
TE_STAGING_ENTRY_BYTES = 512 * MIB  # the slot rounded up to a power of two: one pinned staging entry
TE_EMBED_BYTES = 2_034_237_440  # 248,320 x 4,096 bf16: read from the mapped file, never copied (1.89 GiB)
TE_INTERMEDIATE = 12288  # MLP width
TE_LINEAR_CONV_DIM = 8192  # GatedDeltaNet q / k / v channels (2 x 2048 + 4096)
TE_ROW_FIXED_BYTES = 6 * MIB  # chunk matrices of the gated delta rule per row
TE_STAGE_BYTES_PER_S = 10e9  # memcpy host buffer -> pinned staging (copy / mmap host modes)
TE_DISK_BYTES_PER_S = 2e9  # mmap whose layers fell out of the OS file cache: page-fault reads from an NVMe (estimate)
PINNED_WARN_FRACTION = 0.5  # Windows: page-locked host memory past this share of physical RAM risks failed cudaHostAlloc
TE_H2D_BYTES_PER_S = 25e9  # PCIe 5 x16 H2D from pinned memory
TE_FLOPS_PER_TOKEN = 13.8e9  # 2 x the 6.9B decoder parameters
TE_GPU_FLOPS = 150e12  # bf16 dense throughput the encoder's GEMMs reach (estimate)
TE_LAUNCH_S_PER_ROW = 0.032  # python / kernel-launch overhead of one layer call through the 32 layers (~1 ms per layer):
# one caption, or one group of captions with --moss_te_live_batch (te_group_sizes)


def gib(count) -> float:
    return float(count) / GIB


def _gb(count) -> float:
    return float(count) / 1e9


# ---- model sizes ----------------------------------------------------------------------------------------------------
def is_joint_linear(name: str) -> bool:
    """The 252 joint-block Linears that stay in the compute dtype under --moss_stream_fp32 (the rule of
    MossTransformer.enable_stream_fp32)."""
    return name.startswith("layers.") and ".proj." in name


@functools.lru_cache(maxsize=1)
def _meta_shapes() -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
    import torch

    from .models import MOSS_DIT_CONFIG, MossTransformer

    with torch.device("meta"):
        model = MossTransformer(**MOSS_DIT_CONFIG)
    return tuple((name, tuple(parameter.shape)) for name, parameter in model.named_parameters())


def moss_param_shapes() -> "OrderedDict[str, Tuple[int, ...]]":
    """name -> shape of every MOSS-Image DiT parameter in registration order (built on the meta device: no memory)."""
    return OrderedDict(_meta_shapes())


def _factored_elements(shape: Tuple[int, ...]) -> int:
    """fp32 values CAME keeps for a parameter besides exp_avg: row + col factors of exp_avg_sq and exp_avg_res for
    2-D and larger tensors, a full exp_avg_sq (+ exp_avg_res, an upper bound) for 1-D ones."""
    if len(shape) >= 2:
        rows = math.prod(shape[:-1])
        cols = math.prod(shape[:-2]) * shape[-1]
        return 2 * (rows + cols)
    return 2 * math.prod(shape)


@dataclass(frozen=True)
class ModelSizes:
    """Parameter counts the estimators need. The defaults are the whole MOSS-Image DiT with nothing frozen."""

    total: int = TOTAL_PARAMS  # every parameter (the GPU compute copy, the master file of a state)
    joint_linear: int = JOINT_LINEAR_PARAMS  # of those, the joint Linears (compute dtype under --moss_stream_fp32)
    trainable: int = TOTAL_PARAMS  # host master / moments / accumulator, exp_avg files
    came_factored_bytes: int = CAME_FACTORED_BYTES  # of the trainable ones
    base_elements: int = TOTAL_PARAMS + ROPE_BANDS  # elements of a Base-layout file
    tensors: int = PARAM_TENSORS

    @property
    def other(self) -> int:
        return self.total - self.joint_linear

    @property
    def trainable_fraction(self) -> float:
        return self.trainable / self.total if self.total else 1.0

    @classmethod
    def from_shapes(cls, shapes: Mapping[str, Tuple[int, ...]], frozen: Iterable[str] = ()) -> "ModelSizes":
        frozen = set(frozen)
        total = joint = trainable = factored = 0
        for name, shape in shapes.items():
            count = math.prod(shape)
            total += count
            if is_joint_linear(name):
                joint += count
            if name in frozen:
                continue
            trainable += count
            factored += _factored_elements(tuple(shape))
        return cls(total=total, joint_linear=joint, trainable=trainable, came_factored_bytes=factored * 4,
                   base_elements=total + ROPE_BANDS, tensors=len(shapes))


# ---- GPU (2.1) ------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GpuBudget:
    """Allocated bytes on the GPU for one micro-step of `batch` samples at `image_tokens` (design 2.1)."""

    image_tokens: int
    text_tokens: int
    batch: int
    weights: int
    block_inputs: int
    recompute: int
    grads_in_flight: int
    misc: int
    stored_attention: int = 0
    reuse_blocks: int = 0
    optimizer_state: int = 0  # adafactor_fused only: its factored state on the GPU
    offloaded_block_inputs: int = 0  # --unsloth_offload_checkpointing: the block-input row, in pinned host RAM
    te_ring: int = 0  # --moss_te_live: the device slots of the layer ring (part of te_pass)
    te_pass: int = 0  # --moss_te_live: a pass's peak (ring + every row's layer input / output + one row's temporaries)
    te_held: int = 0  # --moss_te_live: encoded outputs and latents of the batches waiting for their micro-step
    # --moss_ft_val_dataset_config: a validation forward's activations (no grad) at the held-out set's largest batch;
    # it runs between optimizer steps, when the DiT's training activations are freed (val_forward_bytes)
    val_forward: int = 0
    val_batch: int = 0
    val_image_tokens: int = 0

    @property
    def joint_tokens(self) -> int:
        return self.image_tokens + self.text_tokens

    @property
    def dit_step(self) -> int:
        """A micro-step's peak (design 2.1)."""
        return (self.weights + self.block_inputs + self.recompute + self.grads_in_flight + self.misc
                + self.stored_attention + self.optimizer_state)

    @property
    def allocated(self) -> int:
        """The micro-step's peak, or the text encoder's pass when larger: the pass runs between micro-steps, when the
        DiT's activations are freed (only the weights and the fixed part of misc stay), plus the held batches. A
        validation forward (between optimizer steps, the held batches still waiting) counts only where it is larger."""
        return max(self.train_peak, self.val_step) + self.te_held

    @property
    def val_step(self) -> int:
        """A validation forward's peak (0 without validation): weights, the fixed part of misc and its no-grad
        activations; added to `allocated` only where it exceeds the training micro-step (and the live TE pass)."""
        if not self.val_forward:
            return 0
        return self.weights + MISC_FIXED_BYTES + self.optimizer_state + self.val_forward

    @property
    def train_peak(self) -> int:
        """The micro-step's peak or the live TE pass's, whichever is larger (allocated without validation and te_held)."""
        peak = self.dit_step
        if self.te_pass:
            peak = max(peak, self.weights + MISC_FIXED_BYTES + self.optimizer_state + self.te_pass)
        return peak

    @property
    def val_raises_peak(self) -> bool:
        return bool(self.val_forward) and self.val_step > self.train_peak

    @property
    def device(self) -> int:
        """The device-wide estimate: allocated + ~7 % allocator slack + 2.3 GiB used outside the trainer."""
        return int(round(self.allocated * (1.0 + ALLOCATOR_SLACK))) + OUTSIDE_BYTES


def gpu_budget(image_tokens: int, batch: int = 1, *, text_tokens: int = DEFAULT_TEXT_TOKENS,
               sizes: Optional[ModelSizes] = None, mixed_precision: str = "bf16", stream_fp32: bool = True,
               unsloth_offload: bool = False, reuse_attn: bool = False, reuse_budget_gib: Optional[float] = None,
               mode: str = ADAMW_HOST, te_live: Optional["TeLive"] = None, latent_dtype: Optional[str] = "auto",
               val_shapes: Optional[Iterable[Tuple[int, int]]] = None) -> GpuBudget:
    sizes = sizes or ModelSizes()
    tokens = (int(image_tokens) + int(text_tokens)) * int(batch)
    fp32_compute = mixed_precision == "no"  # fp16 (with the fp32 stream) costs what bf16 costs: 2-byte joint Linears
    if fp32_compute:  # a precision reference: every weight and the stream in fp32
        weights, stream_bytes, recompute_scale = sizes.total * 4, 4, 2.0  # fp32 saved activations: x2, upper bound
    else:
        weights = sizes.joint_linear * 2 + sizes.other * (4 if stream_fp32 else 2)
        stream_bytes, recompute_scale = (4 if stream_fp32 else 2), 1.0
    block_inputs = NUM_LAYERS * HIDDEN * stream_bytes * tokens
    recompute = int(round(RECOMPUTE_FACTOR * RECOMPUTE_BYTES_PER_TOKEN * tokens * recompute_scale))
    misc = MISC_FIXED_BYTES + MISC_BYTES_PER_TOKEN * tokens
    stored, blocks = 0, 0
    if reuse_attn:
        per_block = REUSE_ATTN_BYTES_PER_TOKEN_BLOCK * tokens
        blocks = NUM_LAYERS
        if reuse_budget_gib is not None:
            blocks = min(NUM_LAYERS, int(float(reuse_budget_gib) * GIB // per_block))
        stored = blocks * per_block
    offloaded = 0
    if unsloth_offload:
        offloaded, block_inputs = block_inputs, 0
    # modes a and c are the same on the GPU; b adds its small factored state (design 2.1)
    optimizer_state = sizes.came_factored_bytes // 2 if mode == ADAFACTOR_FUSED else 0
    te_ring = te_pass = te_held = 0
    if te_live is not None:
        rows = te_live.ahead * int(batch)
        te_ring, te_pass = te_live_gpu_bytes(rows, text_tokens, te_live.ring_slots, te_live.layer_batch)
        te_held = te_held_bytes(te_live.ahead, batch, text_tokens, image_tokens, latent_dtype)
    val_forward = val_batch = val_tokens = 0
    for shape_tokens, shape_batch in val_shapes or ():  # the validation batch with the largest no-grad forward
        candidate = val_forward_bytes(shape_tokens, shape_batch, text_tokens, mixed_precision=mixed_precision,
                                      stream_fp32=stream_fp32)
        if candidate > val_forward:
            val_forward, val_batch, val_tokens = candidate, int(shape_batch), int(shape_tokens)
    return GpuBudget(image_tokens=int(image_tokens), text_tokens=int(text_tokens), batch=int(batch), weights=weights,
                     block_inputs=block_inputs, recompute=recompute, grads_in_flight=GRADS_IN_FLIGHT_BYTES, misc=misc,
                     stored_attention=stored, reuse_blocks=blocks, optimizer_state=optimizer_state,
                     offloaded_block_inputs=offloaded, te_ring=te_ring, te_pass=te_pass, te_held=te_held,
                     val_forward=val_forward, val_batch=val_batch, val_image_tokens=val_tokens)


def val_forward_bytes(image_tokens: int, batch: int = 1, text_tokens: int = DEFAULT_TEXT_TOKENS, *,
                      mixed_precision: str = "bf16", stream_fp32: bool = True) -> int:
    """The activations of a validation forward (torch.no_grad(), ft_val): nothing is saved for a backward and no block
    is checkpointed, so per joint token it holds a block's input and output stream (2 x 3072 values in the stream
    dtype), one block's forward temporaries (at most what the checkpoint recompute saves: RECOMPUTE_BYTES_PER_TOKEN,
    without the backward's 1.4) and the per-token misc (the fp64 loss chain). The training micro-step holds the 42
    block inputs, the recompute + backward of one block and the gradients in flight: ~0.29 MB against ~0.91 MB per
    token (fp32 stream) + 0.5 GiB, so a validation forward only exceeds it when its batch has ~3x the tokens of the
    training batch (a larger validation bucket or batch size)."""
    tokens = (int(image_tokens) + int(text_tokens)) * int(batch)
    fp32_compute = mixed_precision == "no"
    stream_bytes = 4 if (fp32_compute or stream_fp32) else 2
    temporaries = RECOMPUTE_BYTES_PER_TOKEN * (2 if fp32_compute else 1)
    return tokens * (2 * HIDDEN * stream_bytes + temporaries + MISC_BYTES_PER_TOKEN)


# ---- live text encoder (--moss_te_live, plan 1.5) -------------------------------------------------------------------
@dataclass(frozen=True)
class TeLive:
    """The live text encoder's settings the estimates need."""
    host: str = "copy"
    ring_slots: int = 2
    ahead: int = 32  # micro-batches per pass (already a multiple of GA)
    layer_batch: int = 1  # --moss_te_live_batch: captions per batched layer call (1 = each caption alone)


def te_live_settings(args) -> Optional[TeLive]:
    """TeLive of a run with --moss_te_live (ahead rounded up to a multiple of GA, as moss_train.live_ahead), else None."""
    if not getattr(args, "moss_te_live", False):
        return None
    ga = max(1, int(getattr(args, "gradient_accumulation_steps", 1) or 1))
    ahead = -(-max(1, int(getattr(args, "moss_te_live_ahead", 32) or 32)) // ga) * ga
    layer_batch = max(1, int(getattr(args, "moss_te_live_batch", 1) or 1))
    return TeLive(host=str(getattr(args, "moss_te_live_host", "copy") or "copy"),
                  ring_slots=int(getattr(args, "moss_te_live_ring_slots", 2) or 2), ahead=ahead,
                  layer_batch=layer_batch)


def te_group_sizes(rows: int, layer_batch: int = 1) -> List[int]:
    """--moss_te_live_batch: the row counts of a pass's batched layer calls. ceil(rows / layer_batch) groups of as
    even a size as possible (largest first), so no group exceeds layer_batch and the last one is not a lone straggler
    (33 rows at 32 -> 17 + 16, not 32 + 1). layer_batch 1: one call per row (today's path). te_live.length_groups
    fills these sizes with the rows sorted by length."""
    rows, layer_batch = int(rows), max(1, int(layer_batch))
    if rows <= 0:
        return []
    count = -(-rows // layer_batch)
    base, extra = divmod(rows, count)
    return [base + 1] * extra + [base] * (count - extra)


def te_call_rows(rows: int, layer_batch: int = 1) -> int:
    """--moss_te_live_batch: the most captions one layer call holds in ANY pass of up to `rows` captions:
    min(layer_batch, rows). A full pass is split evenly (te_group_sizes: 33 at 32 -> 17 + 16), but a shorter one - the
    loader's last pass, bucket batches short of the batch size - can be one group of up to layer_batch (32 at 32 -> 32;
    7 at 8 -> 7 where 17 run as 6 + 6 + 5). te_live.length_groups also caps a call at this many x DEFAULT_TEXT_TOKENS
    padded tokens. layer_batch 1: 1."""
    return max(1, min(int(layer_batch), int(rows)))


def te_group_text(rows: int, layer_batch: int) -> str:
    """'4 x 8' / '17 + 16': the group sizes of a pass for the log lines."""
    sizes = te_group_sizes(rows, layer_batch)
    if len(set(sizes)) == 1:
        return f"{len(sizes)} x {sizes[0]}" if sizes else "0"
    return " + ".join(str(size) for size in sizes) if len(sizes) <= 4 else (
        f"{sizes.count(sizes[0])} x {sizes[0]} + {sizes.count(sizes[-1])} x {sizes[-1]}")


def te_live_host_bytes(host: str, ring_slots: int = 2) -> Tuple[int, int]:
    """(encoder, staging) host bytes: copy = the layer buffers (private) + the pinned staging ring; pinned = the layer
    buffers page-locked in place (cudaHostRegister: not rounded), no staging; mmap = no private copy (the OS file cache
    serves the layers) + the staging ring. The embedding table is a view of the mapped file in every mode."""
    staging = pinned_alloc_bytes(int(ring_slots) * TE_STAGING_ENTRY_BYTES)
    if host == "pinned":
        return TE_LAYER_BYTES_TOTAL, 0
    if host == "mmap":
        return 0, staging
    return TE_LAYER_BYTES_TOTAL, staging


def te_row_temp_bytes(tokens: int) -> int:
    """One row's temporaries inside a layer: the MLP's gate / up / product in bf16, the GatedDeltaNet's q / k / v /
    decay in fp32, the chunk matrices (an upper estimate, checked by the smoke run)."""
    return int(tokens) * (3 * TE_INTERMEDIATE * 2 + 4 * TE_LINEAR_CONV_DIM * 4) + TE_ROW_FIXED_BYTES


def te_live_gpu_bytes(rows: int, text_tokens: int = DEFAULT_TEXT_TOKENS, ring_slots: int = 2,
                      layer_batch: int = 1) -> Tuple[int, int]:
    """(ring, pass peak) of one pass over `rows` captions of `text_tokens`: R slots + the layer input and output of
    every row (bf16) + the temporaries of one layer call: one row's (layer_batch 1), or with --moss_te_live_batch those
    of the largest call any pass of up to `rows` captions can make: te_call_rows(rows, layer_batch) = G rows (a shorter
    pass can be one group of G, more than te_group_sizes gives the full pass).
    Why the batched pass stays within it (te_pass_gpu_bytes is the same model for a pass's actual call shapes): a call
    pads its rows to its longest, and te_live.length_groups gives every call at most G rows and at most G x
    DEFAULT_TEXT_TOKENS padded tokens - captions past DEFAULT_TEXT_TOKENS are grouped among themselves under that
    budget, the rest as before. With every caption at most text_tokens (the per-row path's assumption too) a pass of
    m <= rows captions holds at most m x text_tokens padded inputs + one call's output of at most G x text_tokens
    (<= the 2 x rows x text_tokens below) and one call's temporaries of at most G x te_row_temp_bytes(text_tokens).
    Longer captions exceed the activations on both paths alike; a call's temporaries stay within G rows' unless one
    caption past G x text_tokens runs alone (as one past text_tokens does on the per-row path). Reviewer 11-18: by row
    count alone the smoke subset's 32 captions (10-770 tokens) were one call of 32 x 770 at N = 32, ~6.1 GiB against
    this ~3.0 GiB; now [8 x 770, 24 x 284], ~2.4 GiB."""
    ring = int(ring_slots) * TE_SLOT_BYTES
    activations = 2 * int(rows) * int(text_tokens) * TEXT_HIDDEN * 2
    return ring, ring + activations + te_call_rows(rows, layer_batch) * te_row_temp_bytes(text_tokens)


def te_pass_gpu_bytes(calls: Iterable[Tuple[int, int]], ring_slots: int = 2) -> int:
    """te_live_gpu_bytes' pass peak for the layer calls a pass actually makes, [(rows, padded length)]
    (LiveTextEncoder.last_groups, or the shapes of te_live.length_groups before a pass; the per-row path is
    [(1, T_i)]): R slots + every call's padded layer input (all held through the pass) + the largest call's output +
    the largest call's temporaries (rows x te_row_temp_bytes(padded length)). The 9B's sizes, whatever encoder ran."""
    calls = [(int(rows), int(width)) for rows, width in calls]
    ring = int(ring_slots) * TE_SLOT_BYTES
    if not calls:
        return ring
    held = sum(rows * width for rows, width in calls) * TEXT_HIDDEN * 2
    output = max(rows * width for rows, width in calls) * TEXT_HIDDEN * 2
    return ring + held + output + max(rows * te_row_temp_bytes(width) for rows, width in calls)


def te_held_bytes(ahead: int, batch: int, text_tokens: int = DEFAULT_TEXT_TOKENS,
                  image_tokens: int = REFERENCE_IMAGE_TOKENS, latent_dtype: Optional[str] = "auto") -> int:
    """The batches a pass holds until their micro-step: their encoded text (bf16) and their latents (already on the
    device)."""
    samples = int(ahead) * int(batch)
    return samples * (int(text_tokens) * TEXT_HIDDEN * 2
                      + LATENT_CHANNELS * int(image_tokens) * _dtype_bytes(latent_dtype, 4))


def te_seconds_per_pass(rows: int, text_tokens: int = DEFAULT_TEXT_TOKENS, host: str = "copy",
                        from_disk: bool = False, layer_batch: int = 1) -> float:
    """max(staging memcpy, H2D, compute + per-call launch overhead): the three overlap across layers (plan 1.5).
    from_disk (mmap only): the layers are not in the OS file cache, so staging reads them from the disk at
    TE_DISK_BYTES_PER_S instead of copying them from RAM. layer_batch (--moss_te_live_batch): the launch overhead is
    paid per layer call, i.e. per group of rows (te_group_sizes), not per row; the FLOPs stay."""
    if host == "pinned":
        stage = 0.0
    elif host == "mmap" and from_disk:
        stage = TE_LAYER_BYTES_TOTAL / TE_DISK_BYTES_PER_S
    else:
        stage = TE_LAYER_BYTES_TOTAL / TE_STAGE_BYTES_PER_S
    h2d = TE_LAYER_BYTES_TOTAL / TE_H2D_BYTES_PER_S
    calls = len(te_group_sizes(rows, layer_batch))
    compute = int(rows) * int(text_tokens) * TE_FLOPS_PER_TOKEN / TE_GPU_FLOPS + calls * TE_LAUNCH_S_PER_ROW
    return max(stage, h2d, compute)


def te_seconds_per_image(te_live: TeLive, batch: int, text_tokens: int = DEFAULT_TEXT_TOKENS,
                         from_disk: bool = False) -> float:
    rows = max(1, int(te_live.ahead) * int(batch))
    return te_seconds_per_pass(rows, text_tokens, te_live.host, from_disk, te_live.layer_batch) / rows


# ---- host RAM (2.2) -------------------------------------------------------------------------------------------------
def pinned_alloc_bytes(nbytes: int) -> int:
    """What torch's pinned host allocator holds for a request of `nbytes`: CachingHostAllocator (torch 2.14,
    ATen/core/CachingHostAllocator.h) rounds every request up to the next power of two (pinned_max_round_threshold and
    pinned_max_cached_size default to SIZE_MAX), and a freed block stays cached in the free list of its power-of-two
    class (the run never empties the host cache)."""
    nbytes = int(nbytes)
    return 0 if nbytes <= 0 else 1 << (nbytes - 1).bit_length()


def unsloth_pinned_bytes(shapes: Iterable[Tuple[int, int]], *, text_tokens: int = DEFAULT_TEXT_TOKENS,
                         stream_bytes: int = 4) -> int:
    """Pinned host RAM of --unsloth_offload_checkpointing: every joint block saves its input
    ([b, image + text tokens, 3072] in the stream dtype) with .to("cpu", non_blocking=True), a block of torch's pinned
    allocator (pinned_alloc_bytes). All 42 are alive at once in a micro-step, and once freed they stay cached per
    power-of-two class, so every class the dataset reaches keeps 42 blocks for the rest of the run. shapes: (image
    tokens, batch size) of each bucket / tier; every b in 1 .. batch counts (a bucket's last batch is short)."""
    classes = set()
    for image_tokens, batch in shapes:
        per_sample = HIDDEN * int(stream_bytes) * (int(image_tokens) + int(text_tokens))
        classes.update(pinned_alloc_bytes(per_sample * b) for b in range(1, max(1, int(batch)) + 1))
    return NUM_LAYERS * sum(classes)


def dataloader_workers(args) -> int:
    """The DataLoader worker processes moss_train.py starts: min(max_data_loader_n_workers, cpu count)."""
    return max(0, min(int(getattr(args, "max_data_loader_n_workers", 0) or 0), os.cpu_count() or 1))


def needs_host_accumulator(mode: str, ga: int, clip_mode: str = "stale", max_grad_norm: float = 1.0) -> bool:
    """The fp32 grad accumulator of the host pool (design 2.2 / 3.1): GA > 1, or --moss_ft_clip_mode exact with
    clipping on (also at GA 1) - the rule of ft_host.needs_accumulator, so the budget counts what the pipeline
    allocates. Never in adafactor_fused (no host state). max_grad_norm defaults to kohya's 1.0."""
    return mode != ADAFACTOR_FUSED and (int(ga) > 1 or (clip_mode == "exact" and float(max_grad_norm or 0.0) > 0.0))


@dataclass(frozen=True)
class HostBudget:
    mode: str
    ga: int
    master: int
    exp_avg: int
    exp_avg_sq: int
    accumulator: int
    rings: int
    process: int
    pinned_checkpoints: int = 0
    memory_caches: int = 0
    text_encoder: int = 0  # --moss_te_live: the encoder's decoder layers (copy: private, pinned: page-locked)
    te_staging: int = 0  # --moss_te_live: the pinned staging ring (copy / mmap)
    te_host: str = ""  # --moss_te_live_host ("" without the live encoder)
    pinned_master: int = 0  # carousel --moss_carousel_host_pinned masters: the master pool cudaHostRegister'd

    @property
    def pinned(self) -> int:
        """Page-locked host memory of the run (part of total): the D2H / H2D rings, the unsloth checkpoint inputs, the
        TE staging ring and, with --moss_te_live_host pinned, the cudaHostRegister'd decoder layers."""
        return (self.rings + self.pinned_checkpoints + self.te_staging + self.pinned_master
                + (self.text_encoder if self.te_host == "pinned" else 0))

    @property
    def total(self) -> int:
        return (self.master + self.exp_avg + self.exp_avg_sq + self.accumulator + self.rings + self.process
                + self.pinned_checkpoints + self.memory_caches + self.text_encoder + self.te_staging)

    @property
    def required(self) -> int:
        """What the start-up check wants available: the total + 8 GiB (design 2.2)."""
        return self.total + HOST_MARGIN_BYTES


def host_budget(mode: str, ga: int = 1, *, clip_mode: str = "stale", max_grad_norm: float = 1.0,
                sizes: Optional[ModelSizes] = None, ring_mib: int = 1024, upload_ring_mib: int = 512,
                pinned_checkpoint_bytes: int = 0, memory_cache_bytes: int = 0, workers: int = 0,
                te_live: Optional[TeLive] = None) -> HostBudget:
    """Design 2.2. The pinned rings count as torch's pinned allocator holds them (pinned_alloc_bytes; validation
    requires powers of two). workers: DataLoader worker processes (dataloader_workers); PROCESS_BYTES covers
    PROCESS_WORKERS, each one more adds WORKER_BYTES. te_live: the live text encoder's host form and staging ring
    (te_live_host_bytes)."""
    sizes = sizes or ModelSizes()
    if mode == ADAFACTOR_FUSED:  # the model and its small optimizer state live on the GPU
        master = exp_avg = exp_avg_sq = accumulator = rings = 0
    else:
        master = exp_avg = sizes.trainable * 4
        exp_avg_sq = sizes.trainable * 4 if mode == ADAMW_HOST else sizes.came_factored_bytes
        accumulator = sizes.trainable * 4 if needs_host_accumulator(mode, ga, clip_mode, max_grad_norm) else 0
        rings = pinned_alloc_bytes(int(ring_mib) * MIB) + pinned_alloc_bytes(int(upload_ring_mib) * MIB)
    process = PROCESS_BYTES + WORKER_BYTES * max(0, int(workers) - PROCESS_WORKERS)
    text_encoder, te_staging = te_live_host_bytes(te_live.host, te_live.ring_slots) if te_live is not None else (0, 0)
    return HostBudget(mode=mode, ga=int(ga), master=master, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq,
                      accumulator=accumulator, rings=rings, process=process,
                      pinned_checkpoints=int(pinned_checkpoint_bytes), memory_caches=int(memory_cache_bytes),
                      text_encoder=text_encoder, te_staging=te_staging,
                      te_host=te_live.host if te_live is not None else "")


# ---- disk (2.3) -----------------------------------------------------------------------------------------------------
def _dtype_bytes(dtype: Optional[str], default: int) -> int:
    if dtype in (None, "", "auto"):
        return default
    try:
        return DTYPE_BYTES[str(dtype).lower()]
    except KeyError:
        raise ValueError(f"unknown dtype {dtype!r}") from None


def _header_bytes(tensors: int) -> int:
    return 8 + SAFETENSORS_HEADER_BYTES_PER_TENSOR * int(tensors)


def dit_save_bytes(sizes: Optional[ModelSizes] = None, save_precision: Optional[str] = "bf16") -> int:
    """A trained DiT in the Base layout: every parameter in save_precision, rope.bands F32 [16] (design 4.7)."""
    sizes = sizes or ModelSizes()
    per = _dtype_bytes(save_precision or "bf16", 2)
    return sizes.total * per + ROPE_BANDS * 4 + _header_bytes(sizes.tensors + 1)


def state_bytes(mode: str, sizes: Optional[ModelSizes] = None) -> int:
    """One training state directory (design 4.7): master.safetensors (fp32, every Base key incl. frozen parameters and
    rope.bands), the moments of the trained parameters, the per-slot step counters and the small files. carousel:
    the default configuration (adamw, keep) - carousel_state_bytes for another one."""
    sizes = sizes or ModelSizes()
    master = sizes.base_elements * 4 + _header_bytes(sizes.tensors + 1)
    small = 8 * sizes.tensors + _header_bytes(sizes.tensors) + STATE_SMALL_FILES_BYTES
    if mode == CAROUSEL:
        return master + small + _header_bytes(3 * sizes.tensors) + 8 * sizes.trainable
    if mode == ADAFACTOR_FUSED:  # master.safetensors holds the model values; Adafactor's factored state is small
        return master + sizes.came_factored_bytes // 2 + small
    exp_avg = sizes.trainable * 4 + _header_bytes(sizes.tensors)
    if mode == CAME_HOST:
        return master + exp_avg + sizes.came_factored_bytes + _header_bytes(2 * sizes.tensors) + small
    return master + 2 * exp_avg + small


def te_cache_bytes(images: int, variants: int = 1, text_tokens: int = DEFAULT_TEXT_TOKENS,
                   dtype: Optional[str] = "auto") -> int:
    """*_moss_te.npz: 8 KiB per text token (4096 bf16 values) + a 4-byte mask entry, per caption variant (design 2.3).
    "auto" keeps the encoder's bf16."""
    per_token = TEXT_HIDDEN * _dtype_bytes(dtype, 2) + TE_MASK_BYTES_PER_TOKEN
    return int(images) * max(1, int(variants or 0)) * int(text_tokens) * per_token


def latent_cache_bytes(images: int, image_tokens: int = REFERENCE_IMAGE_TOKENS, dtype: Optional[str] = "auto",
                       flip_aug: bool = False, variants: int = 1) -> int:
    """*_WxH_moss.npz: 128 channels per image token (2 MiB fp32 / 1 MiB bf16 at 1024^2), twice with flip_aug, once per
    cached augmentation variant (design 2.3). "auto" keeps the VAE's fp32."""
    per_image = LATENT_CHANNELS * int(image_tokens) * _dtype_bytes(dtype, 4)
    return int(images) * per_image * (2 if flip_aug else 1) * max(1, int(variants or 0))


@dataclass(frozen=True)
class SavePlan:
    """What a run keeps on disk at its peak (design 2.3 / 4.7): the DiT files in output_dir and the training states in
    the state directory, from the save cadence, kohya's removal of old DiT saves (save_last_n_epochs counts saves,
    save_last_n_steps counts steps) and the keep policy of ft_state (save_last_n_*_state, the pinned LR-phase states)."""
    epochs: int
    steps: int
    epoch_dit: int  # epoch DiT saves kept (save_every_n_epochs, never at the last epoch)
    step_dit: int  # step DiT saves kept (save_every_n_steps)
    epoch_states: int  # epoch states kept (the newest save_last_n_epochs_state)
    step_states: int  # step states kept (the newest save_last_n_steps_state)
    pinned_states: int  # LR-phase-boundary epoch states (never deleted)
    states_saved: bool  # save_state or save_state_on_train_end

    @property
    def dit_files(self) -> int:
        """At the peak: the kept saves + the final file (or the new save written before an old one is removed)."""
        return self.epoch_dit + self.step_dit + 1

    @property
    def state_dirs(self) -> int:
        """At the peak: the kept and pinned states + the one being written (the train-end state at the end)."""
        return self.epoch_states + self.step_states + self.pinned_states + 1 if self.states_saved else 0


def _count(value) -> Optional[int]:
    return None if value is None else int(value)


def save_plan(args, steps_per_epoch: int) -> SavePlan:
    """SavePlan of a run: max_train_steps (validated) at steps_per_epoch = ceil(batches / GA) - the cadence of
    moss_train.py's epoch_save_due / step_save_due / phase_state_due."""
    spe = max(1, int(steps_per_epoch))
    steps = int(getattr(args, "max_train_steps", None) or 0)
    if not steps and getattr(args, "max_train_epochs", None):
        steps = int(args.max_train_epochs) * spe
    epochs = -(-steps // spe) if steps else 0
    every_epochs = _count(getattr(args, "save_every_n_epochs", None))
    ratio = getattr(args, "save_n_epoch_ratio", None)
    if ratio is not None and ratio > 0:  # as moss_train.py sets it at the start of the loop
        every_epochs = math.floor(epochs / ratio) or 1
    epoch_saves = (epochs - 1) // every_epochs if every_epochs and epochs > 1 else 0
    every_steps = _count(getattr(args, "save_every_n_steps", None))
    step_saves = steps // every_steps if every_steps else 0
    last_epochs, last_steps = _count(getattr(args, "save_last_n_epochs", None)), _count(getattr(args, "save_last_n_steps", None))
    epoch_dit = min(epoch_saves, last_epochs) if last_epochs is not None and last_epochs >= 1 else epoch_saves
    step_dit = (min(step_saves, last_steps // every_steps + 1) if last_steps is not None and every_steps
                else step_saves)  # train_util.get_remove_step_no keeps the saves of the last save_last_n_steps steps
    save_state = bool(getattr(args, "save_state", False))
    keep_epochs = _count(getattr(args, "save_last_n_epochs_state", None))
    keep_steps = _count(getattr(args, "save_last_n_steps_state", None))
    epoch_states = step_states = pinned = 0
    if save_state:
        epoch_states = min(epoch_saves, keep_epochs) if keep_epochs is not None and keep_epochs >= 1 else epoch_saves
        step_states = min(step_saves, keep_steps) if keep_steps is not None and keep_steps >= 1 else step_saves
        phases = getattr(args, "moss_ft_lr_phases", None)
        if phases and not getattr(args, "moss_ft_no_keep_phase_states", False):
            from .ft_args import lr_phase_end_epochs

            ends = lr_phase_end_epochs(phases)[:-1]
            pinned = sum(1 for end in ends if abs(end - round(end)) < 1e-9 and round(end) < epochs)
    return SavePlan(epochs=epochs, steps=steps, epoch_dit=epoch_dit, step_dit=step_dit, epoch_states=epoch_states,
                    step_states=step_states, pinned_states=pinned,
                    states_saved=save_state or bool(getattr(args, "save_state_on_train_end", False)))


# ---- throughput (3.5) -----------------------------------------------------------------------------------------------
def micro_step_flops(image_tokens: int, text_tokens: int = DEFAULT_TEXT_TOKENS) -> float:
    """FLOPs of one sample's micro-step: forward + recompute + backward (2x) of the joint Linears and the refiner, and
    the joint attention (backward ~2.5x its forward). Only its shape matters: seconds_per_image pins 1024^2 to 1.1 s."""
    tokens = int(image_tokens) + int(text_tokens)
    linear = 8.0 * JOINT_LINEAR_PARAMS * tokens + 8.0 * REFINER_PARAMS * int(text_tokens)
    attention = NUM_LAYERS * 4.0 * HIDDEN * tokens * tokens * (2.0 + FLASH_BACKWARD_FACTOR)
    return linear + attention


def seconds_per_image(image_tokens: int, text_tokens: int = DEFAULT_TEXT_TOKENS) -> float:
    """GPU seconds per sample, 1.1 s at 1024^2 + 300 text tokens (design 3.5), scaled by micro_step_flops."""
    return SECONDS_PER_IMAGE_1024 * (micro_step_flops(image_tokens, text_tokens)
                                     / micro_step_flops(REFERENCE_IMAGE_TOKENS, DEFAULT_TEXT_TOKENS))


@dataclass(frozen=True)
class Throughput:
    batch: int
    ga: int
    tokens_per_image: int
    gpu: float  # GPU seconds of a window
    host: float  # host seconds of a window: (GA - 1) x A + F
    tail: float  # host time the GPU waits for at the window's end: max(0, F - 0.75 x b x t_img)
    step: float  # seconds per optimizer step
    host_paced: bool  # the accumulate pass is as slow as a micro-step: the micro-steps run at the host's pace
    te_s_per_image: float = 0.0  # --moss_te_live: text-encoder pass seconds per image (in step; the GPU runs it)

    @property
    def images_per_step(self) -> int:
        return self.batch * self.ga

    @property
    def images_per_s(self) -> float:
        return self.images_per_step / self.step

    @property
    def tokens_per_s(self) -> float:
        return self.images_per_s * self.tokens_per_image


def throughput(batch: int, ga: int, *, image_tokens: int = REFERENCE_IMAGE_TOKENS,
               text_tokens: int = DEFAULT_TEXT_TOKENS, mode: str = ADAMW_HOST,
               trainable_fraction: float = 1.0, te_live: Optional[TeLive] = None) -> Throughput:
    """Design 3.5: A overlaps the next micro-step's GPU time, F the last micro-step's backward. A micro-step takes
    max(b x t_img, A); the window ends F - 0.75 x b x t_img after the last micro-step's GPU work. The host passes scale
    with the trained share of the parameters. adafactor_fused has no host work (GPU-bound). te_live: the text
    encoder's passes run between micro-steps on the GPU, so their time adds to the step (te_seconds_per_image)."""
    batch, ga = int(batch), int(ga)
    micro = batch * seconds_per_image(image_tokens, text_tokens)
    tokens = int(image_tokens) + int(text_tokens)
    te = te_seconds_per_image(te_live, batch, text_tokens) if te_live is not None else 0.0
    te_step = te * batch * ga
    if mode == ADAFACTOR_FUSED:
        return Throughput(batch, ga, tokens, gpu=ga * micro, host=0.0, tail=0.0, step=ga * micro + te_step,
                          host_paced=False, te_s_per_image=te)
    accumulate = ACCUMULATE_S * trainable_fraction
    final = (CAME_FINAL_S if mode == CAME_HOST else (FINAL_S if ga > 1 else FINAL_GA1_S)) * trainable_fraction
    host = (ga - 1) * accumulate + final
    tail = max(0.0, final - BACKWARD_SHARE * micro)
    paced = ga > 1 and accumulate >= micro - 1e-9
    step = (ga - 1) * max(micro, accumulate) + micro + tail + te_step
    return Throughput(batch, ga, tokens, gpu=ga * micro, host=host, tail=tail, step=step, host_paced=paced,
                      te_s_per_image=te)


# ---- resource checks ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ResourceCheck:
    what: str
    required: int
    available: int
    where: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.available >= self.required


def existing_dir(path: str) -> str:
    """The path itself or its nearest existing parent directory (output folders are made later)."""
    current = os.path.abspath(str(path))
    while not os.path.isdir(current):
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return current


def same_filesystem(a: str, b: str) -> bool:
    try:
        return os.stat(existing_dir(a)).st_dev == os.stat(existing_dir(b)).st_dev
    except OSError:
        return os.path.splitdrive(os.path.abspath(a))[0].lower() == os.path.splitdrive(os.path.abspath(b))[0].lower()


def check_host_memory(required_bytes: int, *, virtual_memory: Optional[Callable] = None) -> ResourceCheck:
    """psutil.virtual_memory().available against required_bytes (injectable for tests)."""
    if virtual_memory is None:
        import psutil

        virtual_memory = psutil.virtual_memory
    return ResourceCheck("host RAM", int(required_bytes), int(virtual_memory().available))


def check_disk_space(path: str, required_bytes: int, *, disk_usage: Optional[Callable] = None) -> ResourceCheck:
    """shutil.disk_usage(<nearest existing directory of path>).free against required_bytes (injectable for tests)."""
    if disk_usage is None:
        import shutil

        disk_usage = shutil.disk_usage
    where = existing_dir(path)
    try:
        free = int(disk_usage(where).free)
    except OSError as error:  # e.g. a drive letter that is not mounted: nothing can be saved there
        return ResourceCheck("disk", int(required_bytes), 0, f"{where} (not readable: {error})")
    return ResourceCheck("disk", int(required_bytes), free, where)


# ---- the start-up report (4.5) --------------------------------------------------------------------------------------
@dataclass
class StartupReport:
    mode: str
    batch: int
    image_tokens: int
    bucket: Optional[Tuple[int, int]]
    sizes: ModelSizes
    gpu: GpuBudget
    host: HostBudget
    speed: Throughput
    dit_bytes: int
    state_bytes: int
    states_saved: bool
    state_dir: Optional[str]
    output_dir: Optional[str]
    images: int
    te_cache_bytes: int  # to be written to disk (existing files not counted)
    latent_cache_bytes: int
    cache_drives: Dict[str, Tuple[str, int]] = field(default_factory=dict)  # drive -> (a directory on it, bytes)
    plan: Optional[SavePlan] = None  # what the run keeps on disk at its peak
    workers: int = 0  # DataLoader worker processes
    lines: List[str] = field(default_factory=list)
    carousel: Optional["CarouselBudget"] = None  # mode carousel: K and its memory (plan 3.8)
    val_images: int = 0  # --moss_ft_val_dataset_config: held-out images (their caches are in the totals above)


def count_images(dataset_group) -> int:
    """Images (not repeats) over the datasets of a group."""
    return sum(len(getattr(dataset, "image_data", None) or {}) for dataset in getattr(dataset_group, "datasets", None) or [])


def _drive_key(path: str) -> str:
    drive = os.path.splitdrive(os.path.abspath(path))[0]
    return drive.upper() if drive else os.sep


def cache_estimate(args, dataset_group, exists: Optional[Callable[[str], bool]] = None):
    """(images, TE bytes to write, latent bytes to write, in-memory cache bytes, {drive: (dir, bytes)}) for the caches
    next to the images: *_moss_te.npz (per caption variant) and *_WxH_moss.npz (per bucket). A file that exists is
    already written; in-memory caches (no *_to_disk) count against host RAM instead of the disk. --moss_te_live writes
    and holds no TE cache (0 TE bytes)."""
    exists = exists or os.path.exists
    live = bool(getattr(args, "moss_te_live", False))
    te_disk = bool(getattr(args, "cache_text_encoder_outputs_to_disk", False))
    latent_disk = bool(getattr(args, "cache_latents_to_disk", False))
    te_dtype = getattr(args, "cache_text_encoder_outputs_dtype", "auto")
    latent_dtype = getattr(args, "cache_latents_dtype", "auto")
    images = te_bytes = latent_bytes = memory = 0
    drives: Dict[str, Tuple[str, int]] = {}
    for dataset in getattr(dataset_group, "datasets", None) or []:
        variants = int(getattr(dataset, "caption_aug_variant_count", 0) or 0)
        latent_variants = int(getattr(dataset, "latents_aug_variant_count", 0) or 0)
        flip = any(getattr(subset, "flip_aug", False) for subset in getattr(dataset, "subsets", None) or [])
        for info in (getattr(dataset, "image_data", None) or {}).values():
            images += 1
            path = getattr(info, "absolute_path", None)
            reso = getattr(info, "bucket_reso", None) or (1024, 1024)
            tokens = (int(reso[0]) // LATENT_STRIDE) * (int(reso[1]) // LATENT_STRIDE)
            te = te_cache_bytes(1, variants, DEFAULT_TEXT_TOKENS, te_dtype)
            latent = latent_cache_bytes(1, tokens, latent_dtype, flip, latent_variants)
            stem = os.path.splitext(path)[0] if path else None
            new = 0
            if live:
                pass
            elif te_disk:
                if not (stem and exists(stem + "_moss_te.npz")):
                    te_bytes += te
                    new += te
            else:
                memory += te
            if latent_disk:
                if not (stem and exists(f"{stem}_{int(reso[0]):04d}x{int(reso[1]):04d}_moss.npz")):
                    latent_bytes += latent
                    new += latent
            else:
                memory += latent
            if new and path:
                key = _drive_key(path)
                where, total = drives.get(key, (os.path.dirname(os.path.abspath(path)), 0))
                drives[key] = (where, total + new)
    return images, te_bytes, latent_bytes, memory, drives


def dataset_shapes(args, dataset_group) -> List[Tuple[int, int]]:
    """(image tokens, batch size) of every bucket an image was assigned to (bucket_reso and the jitter buckets), per
    dataset: the block-input shapes --unsloth_offload_checkpointing pins. [] when the layout is not readable."""
    shapes = set()
    default_batch = int(getattr(args, "train_batch_size", 1) or 1)
    try:
        for dataset in getattr(dataset_group, "datasets", None) or []:
            batch = int(getattr(dataset, "batch_size", 0) or default_batch)
            for info in (getattr(dataset, "image_data", None) or {}).values():
                resos = [getattr(info, "bucket_reso", None)]
                resos += [entry[0] for entry in (getattr(info, "jitter_bucket_info", None) or {}).values()]
                for reso in resos:
                    if reso is not None:
                        shapes.add(((int(reso[0]) // LATENT_STRIDE) * (int(reso[1]) // LATENT_STRIDE), batch))
    except Exception as error:  # noqa: BLE001 - an unknown dataset layout: keep what was read
        logger.debug(f"MOSS full FT budget: bucket shapes not fully read ({type(error).__name__}: {error})")
    return sorted(shapes)


def _image_tokens_of(args, dataset_group) -> Tuple[int, int, Optional[Tuple[int, int]]]:
    from . import train_utils as moss_train_utils

    limits = moss_train_utils.dataset_group_limits(dataset_group)
    batch = moss_train_utils.largest_batch_size(args, limits)
    tokens, bucket = int(limits[1] or 0), limits[2]
    if tokens <= 0:  # buckets not readable: the configured resolution, else 1024^2
        resolution = getattr(args, "resolution", None)
        if isinstance(resolution, str):
            resolution = tuple(int(r) for r in resolution.split(","))
        if isinstance(resolution, (tuple, list)) and resolution:
            width, height = int(resolution[0]), int(resolution[-1])
        else:
            width = height = 1024
        tokens, bucket = (width // LATENT_STRIDE) * (height // LATENT_STRIDE), (width, height)
    return batch, tokens, bucket


def val_cache_estimate(args, val_dataset_group, exists: Optional[Callable[[str], bool]] = None):
    """The held-out set of --moss_ft_val_dataset_config (ft_val): (images, latent bytes to write, host bytes, {drive:
    (dir, bytes)}). Its latents are cached like the training set's (disk or memory, never flipped); its captions'
    text-encoder outputs stay on the CPU for the run (one caption variant each)."""
    exists = exists or os.path.exists
    latent_disk = bool(getattr(args, "cache_latents_to_disk", False))
    latent_dtype = getattr(args, "cache_latents_dtype", "auto")
    images = latent_bytes = memory = 0
    drives: Dict[str, Tuple[str, int]] = {}
    for dataset in getattr(val_dataset_group, "datasets", None) or []:
        for info in (getattr(dataset, "image_data", None) or {}).values():
            images += 1
            path = getattr(info, "absolute_path", None)
            reso = getattr(info, "bucket_reso", None) or (1024, 1024)
            tokens = (int(reso[0]) // LATENT_STRIDE) * (int(reso[1]) // LATENT_STRIDE)
            latent = latent_cache_bytes(1, tokens, latent_dtype, False, 0)
            memory += te_cache_bytes(1, 1, DEFAULT_TEXT_TOKENS, "auto")
            if not latent_disk:
                memory += latent
            elif not (path and exists(f"{os.path.splitext(path)[0]}_{int(reso[0]):04d}x{int(reso[1]):04d}_moss.npz")):
                latent_bytes += latent
                if path:
                    key = _drive_key(path)
                    where, total = drives.get(key, (os.path.dirname(os.path.abspath(path)), 0))
                    drives[key] = (where, total + latent)
    return images, latent_bytes, memory, drives


def startup_report(args, dataset_group, mode: str, *, sizes: Optional[ModelSizes] = None,
                   exists: Optional[Callable[[str], bool]] = None, val_dataset_group=None) -> StartupReport:
    """The design-2.1-2.3 / 3.5 estimates for the largest bucket and batch size of the dataset (dataset_group_limits),
    with one log line per item. Checks nothing; run_startup_checks does. val_dataset_group: the held-out set of
    --moss_ft_val_dataset_config (its forward, latent cache and text outputs are counted)."""
    sizes = sizes or ModelSizes()
    batch, image_tokens, bucket = _image_tokens_of(args, dataset_group)
    ga = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
    clip_mode = getattr(args, "moss_ft_clip_mode", "stale")
    unsloth = bool(getattr(args, "unsloth_offload_checkpointing", False))
    te_live = te_live_settings(args)
    val_shapes = None
    if val_dataset_group is not None:
        from .ft_val import val_batch_shapes  # import here: ft_val imports ft_args, which imports this module

        val_shapes = val_batch_shapes(val_dataset_group)
    gpu = gpu_budget(image_tokens, batch, sizes=sizes, mixed_precision=getattr(args, "mixed_precision", "bf16"),
                     stream_fp32=bool(getattr(args, "moss_stream_fp32", False)), unsloth_offload=unsloth,
                     reuse_attn=bool(getattr(args, "moss_ckpt_reuse_attn", False)),
                     reuse_budget_gib=getattr(args, "moss_ckpt_reuse_attn_budget_gib", None), mode=mode,
                     te_live=te_live, latent_dtype=getattr(args, "cache_latents_dtype", "auto"), val_shapes=val_shapes)
    images, te_bytes, latent_bytes, memory, drives = cache_estimate(args, dataset_group, exists)
    val_images = 0
    if val_dataset_group is not None:
        val_images, val_latents, val_memory, val_drives = val_cache_estimate(args, val_dataset_group, exists)
        latent_bytes += val_latents
        memory += val_memory
        for key, (where, count) in val_drives.items():
            where, total = drives.get(key, (where, 0))
            drives[key] = (where, total + count)
    pinned_checkpoints = 0
    if unsloth:  # what torch's pinned allocator keeps: 42 blocks per power-of-two class the buckets reach
        fp32_stream = bool(getattr(args, "moss_stream_fp32", False)) or getattr(args, "mixed_precision", "bf16") == "no"
        shapes = dataset_shapes(args, dataset_group) or [(image_tokens, batch)]
        pinned_checkpoints = unsloth_pinned_bytes(shapes, stream_bytes=4 if fp32_stream else 2)
    workers = dataloader_workers(args)
    carousel = None
    if mode == CAROUSEL:
        carousel = carousel_budget_from_args(args, gpu)
        te_layers, te_staging = te_live_host_bytes(te_live.host, te_live.ring_slots) if te_live is not None else (0, 0)
        host = HostBudget(mode=mode, ga=ga, master=carousel.host_masters, exp_avg=carousel.host_states, exp_avg_sq=0,
                          accumulator=0, rings=0,
                          process=PROCESS_BYTES + WORKER_BYTES * max(0, int(workers) - PROCESS_WORKERS),
                          pinned_checkpoints=pinned_checkpoints, memory_caches=memory, text_encoder=te_layers,
                          te_staging=te_staging, te_host=te_live.host if te_live is not None else "",
                          pinned_master=carousel.pinned_master)
        speed = carousel_throughput(batch, ga, carousel, image_tokens=image_tokens, te_live=te_live,
                                    period=int(_carousel_arg(args, "period", CAROUSEL_DEFAULT_PERIOD) or 1))
    else:
        host = host_budget(mode, ga, clip_mode=clip_mode, max_grad_norm=getattr(args, "max_grad_norm", 1.0),
                           sizes=sizes, ring_mib=int(getattr(args, "moss_ft_ring_mib", 1024) or 1024),
                           upload_ring_mib=int(getattr(args, "moss_ft_upload_ring_mib", 512) or 512),
                           pinned_checkpoint_bytes=pinned_checkpoints, memory_cache_bytes=memory, workers=workers,
                           te_live=te_live)
        speed = throughput(batch, ga, image_tokens=image_tokens, mode=mode, trainable_fraction=sizes.trainable_fraction,
                           te_live=te_live)
    save_precision = getattr(args, "save_precision", None) or "bf16"
    output_dir = getattr(args, "output_dir", None)
    state_dir = getattr(args, "moss_ft_state_dir", None) or output_dir
    states_saved = bool(getattr(args, "save_state", False) or getattr(args, "save_state_on_train_end", False))
    try:
        batches = len(dataset_group)
    except TypeError:
        batches = 1
    plan = save_plan(args, -(-max(1, int(batches)) // max(1, ga)))
    one_state = state_bytes(mode, sizes)
    if carousel is not None:
        one_state = carousel_state_bytes(carousel.kind, state_policy=carousel.state_policy,
                                         state_dtype=carousel.state_dtype, k=carousel.k, always=carousel.always,
                                         frozen=_frozen_of(args), sizes=sizes)
    report = StartupReport(mode=mode, batch=batch, image_tokens=image_tokens, bucket=bucket, sizes=sizes, gpu=gpu,
                           host=host, speed=speed, dit_bytes=dit_save_bytes(sizes, save_precision),
                           state_bytes=one_state, states_saved=states_saved, state_dir=state_dir,
                           output_dir=output_dir, images=images, te_cache_bytes=te_bytes,
                           latent_cache_bytes=latent_bytes, cache_drives=drives, plan=plan, workers=workers,
                           carousel=carousel, val_images=val_images)
    report.lines = _report_lines(report, args, ga, clip_mode)
    return report


def _report_lines(report: StartupReport, args, ga: int, clip_mode: str) -> List[str]:
    gpu, host, speed, sizes = report.gpu, report.host, report.speed, report.sizes
    bucket = f"{report.bucket[0]}x{report.bucket[1]}" if report.bucket else "?"
    head = "MOSS full FT budget:"
    fp32_stream = getattr(args, "moss_stream_fp32", False) or getattr(args, "mixed_precision", "bf16") == "no"
    stream = "fp32" if fp32_stream else "bf16"
    lines = [
        f"{head} estimates (design 2.1-2.3 / 3.5) for the largest bucket {bucket} ({report.image_tokens:,} image +"
        f" {gpu.text_tokens} text tokens assumed) at batch {report.batch}, mode {report.mode}, GA {ga},"
        f" {sizes.trainable / 1e9:.2f}B of {sizes.total / 1e9:.2f}B parameters trained, {report.images:,} images",
        f"{head} GPU compute weights {gib(gpu.weights):.2f} GiB ({gib(sizes.joint_linear * 2):.2f} joint Linears in"
        f" {getattr(args, 'mixed_precision', 'bf16')} + {gib(gpu.weights - sizes.joint_linear * 2):.2f} the rest,"
        f" {stream} stream{'; fp16 costs the VRAM / host RAM of bf16' if getattr(args, 'mixed_precision', 'bf16') == 'fp16' else ''})"
        if getattr(args, "mixed_precision", "bf16") != "no" else
        f"{head} GPU compute weights {gib(gpu.weights):.2f} GiB (fp32 compute: every weight fp32)",
        f"{head} GPU checkpointed block inputs {gib(gpu.block_inputs):.2f} GiB ({NUM_LAYERS} blocks x"
        f" {HIDDEN * (4 if stream == 'fp32' else 2):,} B/token)"
        + (f"; {gib(gpu.offloaded_block_inputs):.2f} GiB of inputs moved to pinned host RAM"
           " (--unsloth_offload_checkpointing; what the pinned allocator keeps for them: the host line)"
           if gpu.offloaded_block_inputs else ""),
        f"{head} GPU recompute + backward of one block {gib(gpu.recompute):.2f} GiB (eager upper bound)",
        f"{head} GPU grads in flight to the host <= {gib(gpu.grads_in_flight):.2f} GiB (ring-bounded)",
        f"{head} GPU rope, modulation, head, fp64 loss, cuBLAS workspace {gib(gpu.misc):.2f} GiB",
    ]
    if gpu.stored_attention:
        lines.append(f"{head} GPU stored joint attention (--moss_ckpt_reuse_attn) {gib(gpu.stored_attention):.2f} GiB"
                     f" in {gpu.reuse_blocks} of {NUM_LAYERS} blocks")
    if gpu.optimizer_state:
        lines.append(f"{head} GPU Adafactor state {gib(gpu.optimizer_state):.2f} GiB")
    te_live = te_live_settings(args)
    if te_live is not None:
        rows = te_live.ahead * report.batch
        most = te_call_rows(rows, te_live.layer_batch)
        calls = (f" in layer calls of {te_group_text(rows, te_live.layer_batch)} rows (--moss_te_live_batch"
                 f" {te_live.layer_batch}; a call holds at most {most} captions and {most} x {DEFAULT_TEXT_TOKENS}"
                 f" padded tokens, captions past {DEFAULT_TEXT_TOKENS} tokens run in smaller groups)"
                 if te_live.layer_batch > 1 else "")
        lines.append(f"{head} GPU live text encoder: ring {te_live.ring_slots} x {TE_SLOT_BYTES / MIB:.1f} MiB ="
                     f" {gib(gpu.te_ring):.2f} GiB, pass peak {gib(gpu.te_pass):.2f} GiB for {rows} captions{calls}"
                     f" (between micro-steps: max(micro-step {gib(gpu.dit_step):.1f}, weights + fixed + pass"
                     f" {gib(gpu.weights + MISC_FIXED_BYTES + gpu.optimizer_state + gpu.te_pass):.1f}) GiB); held outputs"
                     f" + latents {gib(gpu.te_held):.2f} GiB")
    if gpu.val_forward:
        lines.append(f"{head} GPU validation forward (no grad, between steps) at its largest batch"
                     f" ({gpu.val_batch} x {gpu.val_image_tokens:,} image + {gpu.text_tokens} text tokens):"
                     f" activations {gib(gpu.val_forward):.2f} GiB, peak {gib(gpu.val_step):.1f} GiB"
                     + (f" - above the training peak {gib(gpu.train_peak):.1f} GiB: counted in the total"
                        if gpu.val_raises_peak else
                        f" - below the training peak {gib(gpu.train_peak):.1f} GiB: not added"))
    if report.carousel is not None:
        lines += _carousel_lines(report.carousel, head)
    else:
        lines.append(f"{head} GPU total allocated {gib(gpu.allocated):.1f} GiB; device estimate {gib(gpu.device):.1f}"
                     f" GiB (+{ALLOCATOR_SLACK:.0%} allocator slack, +{gib(OUTSIDE_BYTES):.1f} GiB outside the trainer;"
                     f" warning line {gib(GPU_WARN_BYTES):.1f} GiB)")
    if report.carousel is not None:
        cb = report.carousel
        lines += [
            f"{head} host fp32 masters of every trained tensor {gib(host.master):.2f} GiB"
            + (f" (page-locked: --moss_{CAROUSEL}_host_pinned masters)" if host.pinned_master else " (pageable)"),
            f"{head} host kept optimizer states of the inactive units {gib(host.exp_avg):.2f} GiB"
            + (f" (keep, every unit visited, {cb.state_dtype})" if cb.state_policy == "keep" else " (reset: none)"),
            f"{head} host gradient accumulator, pinned rings: none (the active units step on the GPU)",
        ]
    elif report.mode == ADAFACTOR_FUSED:
        lines.append(f"{head} host fp32 master, exp_avg, exp_avg_sq, accumulator, pinned rings: none"
                     " (adafactor_fused keeps the model and its state on the GPU)")
    else:
        lines += [
            f"{head} host fp32 master {gib(host.master):.2f} GiB",
            f"{head} host exp_avg {gib(host.exp_avg):.2f} GiB",
            f"{head} host exp_avg_sq {gib(host.exp_avg_sq):.2f} GiB"
            + (" (CAME: factored rows / cols)" if report.mode == CAME_HOST else ""),
            f"{head} host fp32 grad accumulator {gib(host.accumulator):.2f} GiB"
            + (" (GA > 1 or exact clipping)" if host.accumulator else f" (none: GA 1, clip mode {clip_mode})"),
            f"{head} host pinned rings {gib(host.rings):.2f} GiB (D2H {getattr(args, 'moss_ft_ring_mib', 1024)} MiB +"
            f" H2D {getattr(args, 'moss_ft_upload_ring_mib', 512)} MiB)",
        ]
    lines.append(f"{head} host process, CUDA host side, {report.workers} DataLoader workers, CPU VAE"
                 f" ~{gib(host.process):.0f} GiB")
    if host.pinned_checkpoints:
        lines.append(f"{head} host pinned checkpoint inputs (--unsloth_offload_checkpointing)"
                     f" {gib(host.pinned_checkpoints):.2f} GiB ({NUM_LAYERS} blocks in every power-of-two size class of"
                     " torch's pinned allocator the buckets and their short batches reach)")
    if host.memory_caches:
        lines.append(f"{head} host in-memory latent / TE caches {gib(host.memory_caches):.2f} GiB (no *_to_disk"
                     + (f"; with the {report.val_images} held-out images' text outputs" if report.val_images else "")
                     + ")")
    if te_live is not None:
        held = {"copy": "private", "pinned": "page-locked in place",
                "mmap": f"none private: the OS file cache serves them, ~{_gb(TE_LAYER_BYTES_TOTAL):.1f} GB of it wanted"
                }.get(te_live.host, "?")
        lines.append(f"{head} host live text encoder ({te_live.host}): decoder layers {gib(host.text_encoder):.2f} GiB"
                     f" ({held}), pinned staging {gib(host.te_staging):.2f} GiB; the embedding table"
                     f" ({gib(TE_EMBED_BYTES):.2f} GiB) is read from the mapped file")
    lines.append(f"{head} host total {gib(host.total):.1f} GiB; start-up needs {gib(host.required):.1f} GiB available"
                 f" (total + {gib(HOST_MARGIN_BYTES):.0f} GiB)")
    te_layers_locked = host.text_encoder if host.te_host == "pinned" else 0
    lines.append(f"{head} host page-locked (inside the total) {gib(host.pinned):.2f} GiB: rings {gib(host.rings):.2f}"
                 f" + checkpoint inputs {gib(host.pinned_checkpoints):.2f} + TE staging {gib(host.te_staging):.2f}"
                 f" + TE layers {gib(te_layers_locked):.2f} (Windows: warned past {PINNED_WARN_FRACTION:.0%} of"
                 " physical RAM)")
    precision = getattr(args, "save_precision", None) or "bf16"
    plan = report.plan or SavePlan(0, 0, 0, 0, 0, 0, 0, report.states_saved)
    lines.append(f"{head} disk trained DiT ({precision}, Base layout) {_gb(report.dit_bytes):.2f} GB per save; at the"
                 f" peak {plan.dit_files} files ({plan.epoch_dit} epoch + {plan.step_dit} step saves kept + the final"
                 f" one) = {_gb(plan.dit_files * report.dit_bytes):.1f} GB in {report.output_dir or '(output_dir not set)'}")
    if report.states_saved:
        lines.append(f"{head} disk training state ({report.mode}) {_gb(report.state_bytes):.1f} GB per state; at the"
                     f" peak {plan.state_dirs} states ({plan.epoch_states} epoch + {plan.step_states} step kept +"
                     f" {plan.pinned_states} pinned LR-phase + 1 being written) ="
                     f" {_gb(plan.state_dirs * report.state_bytes):.1f} GB at {report.state_dir or '(output_dir not set)'}")
    else:
        lines.append(f"{head} disk training state ({report.mode}) {_gb(report.state_bytes):.1f} GB per state - not"
                     " saved (save_state is off)")
    if te_live is not None:
        lines.append(f"{head} disk TE cache: none (--moss_te_live encodes the captions at train time)")
    else:
        lines.append(f"{head} disk TE cache to write {_gb(report.te_cache_bytes):.2f} GB ({DEFAULT_TEXT_TOKENS} tokens x"
                     f" {(TEXT_HIDDEN * 2) // 1024} KiB per caption variant per image; existing files not counted)")
    lines.append(f"{head} disk latent cache to write {_gb(report.latent_cache_bytes):.2f} GB (existing files not counted)")
    lines.append(f"{head} speed ~{speed.step:.1f} s/step (GPU {speed.gpu:.1f} s + tail {speed.tail:.1f} s"
                 + (f" + text encoder {speed.te_s_per_image * speed.images_per_step:.2f} s"
                    f" (~{speed.te_s_per_image:.3f} s/image, estimate)" if te_live is not None else "")
                 + f"; host {speed.host:.1f} s per window{', micro-steps host-paced' if speed.host_paced else ''}),"
                 f" {speed.images_per_s:.2f} img/s, {speed.tokens_per_s:,.0f} tok/s")
    return lines


def run_startup_checks(args, dataset_group, mode: str, *, sizes: Optional[ModelSizes] = None,
                       virtual_memory: Optional[Callable] = None, disk_usage: Optional[Callable] = None,
                       exists: Optional[Callable[[str], bool]] = None,
                       windows: Optional[bool] = None, val_dataset_group=None) -> StartupReport:
    """Design 4.5 "Start-up checks": log the estimates; refuse (ValueError) when host RAM available < host total
    + 8 GiB or when the free space at output_dir / the state directory is less than what the run keeps there at its
    peak (SavePlan: DiT files and states; design 4.5 said 2 x state + one DiT save, which a run with step and epoch saves
    outgrows); warn when the device estimate exceeds 30.5 GiB or the caches to write exceed the free space of their
    drive. --moss_ft_skip_resource_check turns the refusals into warnings (the estimates are still logged).
    Warnings for the live text encoder and the pinned memory: --moss_te_live_host mmap when the available RAM cannot
    also hold its ~13.8 GB of layers in the OS file cache (every pass then reads them from the disk), and - on Windows
    (windows=None: os.name == "nt") - page-locked host memory past PINNED_WARN_FRACTION of physical RAM."""
    report = startup_report(args, dataset_group, mode, sizes=sizes, exists=exists, val_dataset_group=val_dataset_group)
    for line in report.lines:
        logger.info(line)
    skip = bool(getattr(args, "moss_ft_skip_resource_check", False))
    problems: List[str] = []

    cb = report.carousel
    if cb is not None:
        if cb.k < 1:
            bucket = (f" at the largest bucket {report.bucket[0]}x{report.bucket[1]}" if report.bucket else "")
            raise ValueError(f"MOSS full FT: {CAROUSEL}: no unit fits the device budget of {gib(cb.vram_bytes):.1f} GiB"
                             f" (--moss_{CAROUSEL}_vram_gib) with batch {report.batch}{bucket}: weights"
                             f" {gib(cb.weights):.2f} + always group {gib(cb.always_bytes):.2f} + activations"
                             f" {gib(cb.activations):.2f} ({cb.activation_model} model) + optimizer temporaries"
                             f" {gib(cb.temp_bytes):.2f} GiB leave less than one unit ({gib(cb.unit_bytes):.3f} GiB with"
                             f" {cb.kind}). K is fixed for the run, so the largest bucket and batch decide it. Options:"
                             f" batch 1 with gradient_accumulation_steps (the {CAROUSEL} recommendation: more active"
                             " units at about the same images/s), --unsloth_offload_checkpointing (block inputs in host"
                             f" RAM; also for datasets that mix in 2048), adamw8bit / came, --moss_{CAROUSEL}_always"
                             f" head, or --moss_{CAROUSEL}_active N to set K yourself (the estimate may be high: then"
                             " only the 30.5 GiB warning is logged)")
        if cb.device > GPU_WARN_BYTES:
            logger.warning(f"MOSS full FT: {CAROUSEL}: the device estimate {gib(cb.device):.1f} GiB with {cb.k} active"
                           f" units is past the {gib(GPU_WARN_BYTES):.1f} GiB warning line: expect Windows to spill into"
                           f" shared memory (slow steps). Lower --moss_{CAROUSEL}_active or the batch size")
        if cb.pinned_master:
            try:
                total_ram = int(getattr((virtual_memory or _psutil_virtual_memory)(), "total", 0) or 0)
            except Exception:  # noqa: BLE001 - no total: no refusal
                total_ram = 0
            if total_ram and report.host.pinned > CAROUSEL_PINNED_LIMIT * total_ram:
                raise ValueError(f"MOSS full FT: --moss_{CAROUSEL}_host_pinned masters page-locks"
                                 f" {gib(cb.pinned_master):.1f} GiB; the run's page-locked total"
                                 f" {gib(report.host.pinned):.1f} GiB would pass {CAROUSEL_PINNED_LIMIT:.0%} of the"
                                 f" {gib(total_ram):.0f} GiB of physical RAM (Windows caps pinned memory near 50 %)."
                                 f" Use --moss_{CAROUSEL}_host_pinned none (pageable copies, measured ~8.5 GB/s)")
    elif report.gpu.device > GPU_WARN_BYTES:
        logger.warning(f"MOSS full FT: the device estimate {gib(report.gpu.device):.1f} GiB at batch {report.batch} on"
                       f" the {report.bucket[0] if report.bucket else '?'}x{report.bucket[1] if report.bucket else '?'}"
                       f" bucket is past the {gib(GPU_WARN_BYTES):.1f} GiB warning line (design 2.1): expect"
                       " Windows to spill into shared memory (slow steps, no error). Lower the batch size of that"
                       " bucket, or use --unsloth_offload_checkpointing")

    host = check_host_memory(report.host.required, virtual_memory=virtual_memory)
    logger.info(f"MOSS full FT budget: host RAM available {gib(host.available):.1f} GiB, needed"
                f" {gib(host.required):.1f} GiB")
    if not host.ok:
        problems.append(f"host RAM: {gib(host.available):.1f} GiB available, the estimate needs"
                        f" {gib(host.required):.1f} GiB ({gib(report.host.total):.1f} GiB for {mode} + 8 GiB margin,"
                        " design 2.2). Close RAM-heavy programs (ComfyUI with models loaded, browsers)"
                        + (", or use GA 1 / --moss_ft_clip_mode stale (no accumulator)" if report.host.accumulator else "")
                        + (", or --moss_te_live_host mmap (no private copy of the text encoder's"
                           f" {gib(report.host.text_encoder):.1f} GiB; it trades RAM for disk reads: the OS file cache"
                           " must still hold the layers, or every encoder pass reads"
                           f" ~{_gb(TE_LAYER_BYTES_TOTAL):.1f} GB from the disk)" if report.host.text_encoder
                           and getattr(args, "moss_te_live_host", "copy") == "copy" else ""))
    _te_mmap_warning(args, report, host)
    _pinned_warning(report, virtual_memory, windows)

    for check, needs in _disk_checks(report, disk_usage):
        logger.info(f"MOSS full FT budget: {check.where} has {_gb(check.available):.1f} GB free, needs"
                    f" {_gb(check.required):.1f} GB ({needs})")
        if not check.ok:
            problems.append(f"disk: {check.where} has {_gb(check.available):.1f} GB free, needs"
                            f" {_gb(check.required):.1f} GB ({needs}, design 2.3). Free space or put output_dir /"
                            " --moss_ft_state_dir on another drive")

    for drive, (where, count) in sorted(report.cache_drives.items()):
        check = check_disk_space(where, count, disk_usage=disk_usage)
        if not check.ok:
            logger.warning(f"MOSS full FT: the latent / TE caches still to write next to the images on {drive}"
                           f" ({_gb(count):.1f} GB) exceed its free space ({_gb(check.available):.1f} GB free at"
                           f" {check.where}). Use fewer caption variants (cache_caption_variants) or free space"
                           " (design 2.3)")

    if problems:
        if skip:
            for problem in problems:
                logger.warning(f"MOSS full FT: --moss_ft_skip_resource_check: not refusing: {problem}")
        else:
            raise ValueError("MOSS full FT: not enough resources for this run (--moss_ft_skip_resource_check skips"
                             " this check):\n  " + "\n  ".join(problems))
    return report


def _te_mmap_warning(args, report: StartupReport, host: ResourceCheck) -> None:
    """--moss_te_live_host mmap counts 0 GiB for the layers; they still have to sit in the OS file cache next to
    everything else, or each pass faults them in from the disk."""
    te_live = te_live_settings(args)
    if te_live is None or te_live.host != "mmap" or host.available >= host.required + TE_LAYER_BYTES_TOTAL:
        return
    rows = max(1, te_live.ahead * report.batch)
    logger.warning(f"MOSS full FT: --moss_te_live_host mmap: {gib(host.available):.1f} GiB available leaves no room for"
                   f" the text encoder's {gib(TE_LAYER_BYTES_TOTAL):.1f} GiB of layers in the OS file cache next to the"
                   f" {gib(host.required):.1f} GiB the run needs: each encoder pass will read"
                   f" ~{_gb(TE_LAYER_BYTES_TOTAL):.1f} GB from the disk"
                   f" (~{te_seconds_per_pass(rows, host='mmap', from_disk=True, layer_batch=te_live.layer_batch):.1f} s"
                   f" per pass of {rows} captions instead of"
                   f" ~{te_seconds_per_pass(rows, host='mmap', layer_batch=te_live.layer_batch):.1f} s;"
                   f" ~{te_seconds_per_image(te_live, report.batch, from_disk=True):.3f} s/image). Free RAM, or lower"
                   " the other host use (GA 1 / stale clipping: no accumulator)")


def _pinned_warning(report: StartupReport, virtual_memory: Optional[Callable], windows: Optional[bool]) -> None:
    """Windows: page-locked memory past PINNED_WARN_FRACTION of physical RAM (a cudaHostRegister failure is caught
    at start-up; a later cudaHostAlloc of the pinned allocator - unsloth checkpoint inputs - would fail mid-run)."""
    on_windows = (os.name == "nt") if windows is None else bool(windows)
    if not on_windows or not report.host.pinned:
        return
    if virtual_memory is None:
        import psutil

        virtual_memory = psutil.virtual_memory
    total = int(getattr(virtual_memory(), "total", 0) or 0)
    if total and report.host.pinned > PINNED_WARN_FRACTION * total:
        locked = ", TE layers" if report.host.te_host == "pinned" else ""
        logger.warning(f"MOSS full FT: {gib(report.host.pinned):.1f} GiB of page-locked host memory (rings, checkpoint"
                       f" inputs, TE staging{locked}) is past {PINNED_WARN_FRACTION:.0%} of the {gib(total):.0f} GiB of"
                       " physical RAM: on Windows a later cudaHostAlloc (e.g. the unsloth checkpoint inputs of a new"
                       " bucket) can fail mid-run. Use --moss_te_live_host copy, smaller rings, or no"
                       " --unsloth_offload_checkpointing")


def _disk_checks(report: StartupReport, disk_usage: Optional[Callable]) -> List[Tuple[ResourceCheck, str]]:
    """Free space for what the run keeps at its peak (SavePlan): the DiT files at output_dir and the training states
    at the state directory (only when states are saved), summed when both are on one filesystem."""
    if not report.output_dir and not report.state_dir:
        logger.info("MOSS full FT budget: no output_dir: the disk check is skipped")
        return []
    plan = report.plan or SavePlan(0, 0, 0, 0, 0, 0, 0, report.states_saved)
    targets: List[Tuple[str, int, str]] = []
    if report.output_dir:
        targets.append((report.output_dir, plan.dit_files * report.dit_bytes,
                        f"{plan.dit_files} DiT file{'s' if plan.dit_files != 1 else ''}"))
    if plan.state_dirs and report.state_dir:
        targets.append((report.state_dir, plan.state_dirs * report.state_bytes,
                        f"{plan.state_dirs} training state{'s' if plan.state_dirs != 1 else ''}"))
    merged: List[Tuple[str, int, List[str]]] = []
    for path, count, what in targets:
        for index, (other, total, names) in enumerate(merged):
            if same_filesystem(path, other):
                merged[index] = (other, total + count, names + [what])
                break
        else:
            merged.append((path, count, [what]))
    return [(check_disk_space(path, count, disk_usage=disk_usage), " + ".join(names)) for path, count, names in merged]


# ---- carousel (docs/moss_ft_carousel_plan.md 3.8) -------------------------------------------------------------------
CAROUSEL_PAGEABLE_BYTES_PER_S = 8.5e9  # measured 11-17: pageable copies each way
CAROUSEL_PINNED_H2D_BYTES_PER_S = 40e9  # measured 11-17: pinned H2D
CAROUSEL_PINNED_D2H_BYTES_PER_S = 24e9  # measured 11-17: pinned D2H
# plan 3.1, random K=8; rotate_stem: the stem unit active in 8/45 of the steps (134F), head-like otherwise (126F)
CAROUSEL_STEP_COST = {"small": 134.0 / 168.0, "head": 126.0 / 168.0, "none": 126.0 / 168.0,
                      "rotate_stem": 127.5 / 168.0}
# Measured activation model (11-17 GPU smoke, adamw_host, fp16 + --moss_stream_fp32, plain checkpointing, live TE):
# peak allocated minus the compute copies was 4.03 GiB at 1024^2 b1 (g3) and 11.63 GiB at b4 (s73); the design
# estimate (gpu_budget) was 4.49 / 15.71 GiB, 4.2 GiB over at b4. Linear fit over the joint tokens (300 text tokens
# assumed): 1.50 GiB + 618,800 B per token; the fixed part gets +0.5 GiB margin (longer captions, the bucket mix). It
# includes adamw_host's 0.5 GiB gradient ring (the carousel's single gradient in flight fits inside it). Used for K only
# in those conditions, and only when it is below the design estimate.
CAROUSEL_MEASURED_ACT_FIXED_BYTES = 2 * GIB
CAROUSEL_MEASURED_ACT_BYTES_PER_TOKEN = 620_000


@functools.lru_cache(maxsize=8)
def _carousel_layout(frozen: Tuple[str, ...], stem_unit: bool):
    from . import ft_carousel_units as cu

    shapes = moss_param_shapes()
    frozen_set = set(frozen)
    trained = [(n, s) for n, s in shapes.items() if n not in frozen_set]
    return tuple(cu.discover_units(trained, stem_unit=stem_unit)), tuple(trained)


def carousel_layout(frozen: Iterable[str] = (), stem_unit: bool = False):
    """(units, [(name, shape)] of the trained tensors) of the MOSS DiT with `frozen` tensors left out (meta model);
    stem_unit: --moss_carousel_always rotate_stem (the stem is a unit)."""
    return _carousel_layout(tuple(sorted(frozen)), bool(stem_unit))


def measured_activations(gpu: GpuBudget) -> Optional[int]:
    """The measured activation model (CAROUSEL_MEASURED_ACT_*) for gpu's micro-step, with the live TE pass / held
    batches as gpu_budget counts them; None outside the measured conditions (unsloth offload, reused attention, fp32
    compute, a bf16 stream)."""
    tokens = gpu.joint_tokens * gpu.batch
    if gpu.offloaded_block_inputs or gpu.stored_attention or gpu.optimizer_state:
        return None
    if gpu.block_inputs != NUM_LAYERS * HIDDEN * 4 * tokens:  # not the fp32 stream
        return None
    if gpu.recompute != int(round(RECOMPUTE_FACTOR * RECOMPUTE_BYTES_PER_TOKEN * tokens)):  # fp32 compute (x2)
        return None
    dit = CAROUSEL_MEASURED_ACT_FIXED_BYTES + CAROUSEL_MEASURED_ACT_BYTES_PER_TOKEN * tokens
    return max(dit, MISC_FIXED_BYTES + gpu.te_pass, MISC_FIXED_BYTES + gpu.val_forward) + gpu.te_held


@dataclass(frozen=True)
class CarouselBudget:
    """The Carousel estimates (plan 3.8). Device bytes are on top of the compute copies (weights)."""
    kind: str
    always: str
    state_policy: str
    state_dtype: str
    units: int
    unit_params: int  # the largest unit
    unit_bytes: int  # device bytes of the largest unit: fp32 master + fp32 accumulator + optimizer state
    unit_state_bytes: int  # the largest unit's optimizer state (what a keep switch moves besides the master)
    always_params: int
    always_bytes: int
    temp_bytes: int  # optimizer step temporaries
    weights: int  # the compute copies of every parameter
    activations: int  # gpu_budget's activations, live TE pass / held batches, one gradient in flight
    vram_bytes: int  # what the run may use on the device (--moss_carousel_vram_gib, default the 30.5 GiB line)
    k_budget: int  # the K that fits vram_bytes (may be < 1)
    k: int  # the K of the run: --moss_carousel_active, else k_budget clamped to the units
    k_given: bool
    host_masters: int
    host_states: int  # keep: the states of every unit (all visited); reset: 0
    pinned_master: int = 0
    trainable_params: int = 0
    activation_model: str = "design"  # "measured" when the 11-17 fit (measured_activations) gave the activations
    design_activations: int = 0  # gpu_budget's activations (what "design" would have used)

    @property
    def allocated(self) -> int:
        return self.weights + self.always_bytes + self.k * self.unit_bytes + self.temp_bytes + self.activations

    @property
    def device(self) -> int:
        return int(round(self.allocated * (1.0 + ALLOCATOR_SLACK))) + OUTSIDE_BYTES

    def switch_bytes(self, policy: Optional[str] = None) -> int:
        """Bytes one direction of a switch moves per unit: the fp32 master, plus the state with keep."""
        keep = (policy or self.state_policy) == "keep"
        return self.unit_params * 4 + (self.unit_state_bytes if keep else 0)

    def switch_seconds(self) -> float:
        """A full switch of k units (out and in), pageable copies; pinned masters at the measured pinned rates."""
        state = self.unit_state_bytes if self.state_policy == "keep" else 0
        master = self.unit_params * 4
        if self.pinned_master:
            one = master / CAROUSEL_PINNED_D2H_BYTES_PER_S + master / CAROUSEL_PINNED_H2D_BYTES_PER_S
        else:
            one = 2 * master / CAROUSEL_PAGEABLE_BYTES_PER_S
        return self.k * (one + 2 * state / CAROUSEL_PAGEABLE_BYTES_PER_S)


def carousel_budget(kind: str, gpu: GpuBudget, *, always: str = "small", state_policy: str = "keep",
                    state_dtype: str = "fp32", frozen: Iterable[str] = (), active: Optional[int] = None,
                    vram_bytes: Optional[int] = None, pinned_masters: bool = False) -> CarouselBudget:
    """K and the memory of a Carousel run (plan 3.8): K = floor((G_alloc - W - A_always - T_opt - act) / A_unit) with
    G_alloc = (G - 2.3 GiB) / 1.07; act = gpu's activations with one gradient in flight instead of the D2H ring, or the
    measured model (measured_activations) when it applies and is lower."""
    from . import ft_carousel_units as cu

    units, trained = carousel_layout(frozen, stem_unit=cu.stem_rotates(always))
    shapes = dict(trained)
    always_names = cu.always_names([n for n, _ in trained], always)
    unit_bytes = max([cu.group_device_bytes(kind, cu.unit_shapes(u, shapes)) for u in units] + [0])
    unit_state = max([sum(cu.state_bytes(kind, shapes[n], state_dtype) for n in u.names) for u in units] + [0])
    unit_params = max([u.numel for u in units] + [0])
    always_bytes = cu.group_device_bytes(kind, [shapes[n] for n in always_names])
    biggest_unit = max(units, key=lambda u: u.numel) if units else None
    temp_shapes = [shapes[n] for n in always_names] + (cu.unit_shapes(biggest_unit, shapes) if biggest_unit else [])
    temp = cu.optimizer_temp_bytes(kind, temp_shapes)
    design = gpu.train_peak + gpu.te_held - gpu.weights - gpu.grads_in_flight + CAROUSEL_GRADS_IN_FLIGHT_BYTES
    if gpu.val_forward:  # a validation forward between steps: the active units' memory stays, its activations replace
        design = max(design, gpu.val_step + gpu.te_held - gpu.weights)  # the micro-step's
    measured = measured_activations(gpu)
    model = "measured" if measured is not None and measured < design else "design"
    activations = measured if model == "measured" else design
    vram = int(GPU_WARN_BYTES if vram_bytes is None else vram_bytes)
    g_alloc = (vram - OUTSIDE_BYTES) / (1.0 + ALLOCATOR_SLACK)
    free = g_alloc - gpu.weights - always_bytes - temp - activations
    k_budget = int(math.floor(free / unit_bytes)) if unit_bytes else 0
    k_given = active is not None
    k = int(active) if k_given else max(0, min(k_budget, len(units)))
    host_states = (sum(cu.state_bytes(kind, shapes[n], state_dtype) for u in units for n in u.names)
                   if state_policy == "keep" else 0)
    masters = sum(math.prod(s) for _, s in trained) * 4
    return CarouselBudget(kind=kind, always=always, state_policy=state_policy, state_dtype=state_dtype,
                          units=len(units), unit_params=unit_params, unit_bytes=unit_bytes, unit_state_bytes=unit_state,
                          always_params=sum(math.prod(shapes[n]) for n in always_names), always_bytes=always_bytes,
                          temp_bytes=temp, weights=gpu.weights, activations=activations, vram_bytes=vram,
                          k_budget=k_budget, k=k, k_given=k_given, host_masters=masters, host_states=host_states,
                          pinned_master=masters if pinned_masters else 0,
                          trainable_params=sum(math.prod(s) for _, s in trained), activation_model=model,
                          design_activations=design)


def carousel_state_bytes(kind: str, *, state_policy: str = "keep", state_dtype: str = "fp32", k: int = 0,
                         always: str = "small", frozen: Iterable[str] = (), sizes: Optional[ModelSizes] = None) -> int:
    """One Carousel training state: master.safetensors, the optimizer states (keep: every unit + the always group;
    reset: k units + the always group; a unit stored in bf16 is counted so), the small files."""
    from . import ft_carousel_units as cu

    sizes = sizes or ModelSizes()
    units, trained = carousel_layout(frozen, stem_unit=cu.stem_rotates(always))
    shapes = dict(trained)
    master = sizes.base_elements * 4 + _header_bytes(sizes.tensors + 1)
    small = 8 * sizes.tensors + _header_bytes(sizes.tensors) + STATE_SMALL_FILES_BYTES
    per_unit = sorted((sum(cu.state_bytes(kind, shapes[n], state_dtype) for n in u.names) for u in units), reverse=True)
    states = sum(per_unit) if state_policy == "keep" else sum(per_unit[:max(0, int(k))])
    states += sum(cu.state_bytes(kind, shapes[n]) for n in cu.always_names([n for n, _ in trained], always))
    return master + small + states + _header_bytes(4 * len(trained))


def _psutil_virtual_memory():
    import psutil

    return psutil.virtual_memory()


def _carousel_arg(args, key: str, default=None):
    return getattr(args, f"moss_{CAROUSEL}_{key}", default)


def _frozen_of(args) -> Tuple[str, ...]:
    from .ft_args import frozen_parameter_names, split_freeze_patterns

    patterns = split_freeze_patterns(getattr(args, "moss_ft_freeze", ""))
    return tuple(frozen_parameter_names(moss_param_shapes(), patterns)) if patterns else ()


def carousel_budget_from_args(args, gpu: GpuBudget) -> CarouselBudget:
    """carousel_budget with the run's flags; also sets args._moss_rotation_k (the K the engine uses when
    --moss_carousel_active is not given)."""
    from .ft_args import carousel_kind

    vram = _carousel_arg(args, "vram_gib")
    cb = carousel_budget(carousel_kind(args), gpu, always=_carousel_arg(args, "always", "small"),
                         state_policy=_carousel_arg(args, "state", "keep"),
                         state_dtype=_carousel_arg(args, "state_dtype", "fp32"), frozen=_frozen_of(args),
                         active=_carousel_arg(args, "active"),
                         vram_bytes=None if vram is None else int(float(vram) * GIB),
                         pinned_masters=_carousel_arg(args, "host_pinned", "none") == "masters")
    args._moss_rotation_k = cb.k
    return cb


def carousel_throughput(batch: int, ga: int, cb: CarouselBudget, *, image_tokens: int = REFERENCE_IMAGE_TOKENS,
                        text_tokens: int = DEFAULT_TEXT_TOKENS, te_live: Optional[TeLive] = None,
                        period: int = CAROUSEL_DEFAULT_PERIOD) -> Throughput:
    """GPU-bound: the micro-steps at the plan 3.1 cost share of full FT (no recompute / dgrad below the floor), the
    optimizer step (~0.02 s) and the switch amortised over the period."""
    batch, ga = int(batch), int(ga)
    micro = batch * seconds_per_image(image_tokens, text_tokens) * CAROUSEL_STEP_COST.get(cb.always, 1.0)
    te = te_seconds_per_image(te_live, batch, text_tokens) if te_live is not None else 0.0
    switch = cb.switch_seconds() / max(1, int(period))
    step = ga * micro + 0.02 + switch + te * batch * ga
    return Throughput(batch, ga, int(image_tokens) + int(text_tokens), gpu=ga * micro, host=switch, tail=0.02,
                      step=step, host_paced=False, te_s_per_image=te)


def _carousel_lines(cb: CarouselBudget, head: str) -> List[str]:
    how = f"--moss_{CAROUSEL}_active" if cb.k_given else f"the {gib(cb.vram_bytes):.1f} GiB device budget"
    keep = f" ({cb.state_dtype} on the host)" if cb.state_policy == "keep" else ""
    stem = cb.always == "rotate_stem"
    act = (f"activations {gib(cb.activations):.2f} GiB (measured 11-17 model; the design estimate"
           f" {gib(cb.design_activations):.2f})" if cb.activation_model == "measured" else
           f"activations {gib(cb.activations):.2f} GiB (design estimate)")
    return [
        f"{head} {CAROUSEL}: {cb.units} units ({'the stem + ' if stem else ''}2 text refiner + 42 joint blocks),"
        f" optimizer {cb.kind}, state {cb.state_policy}{keep}, always {cb.always} ({cb.always_params / 1e6:.2f}M"
        " parameters)",
        f"{head} GPU per active unit {gib(cb.unit_bytes):.3f} GiB (fp32 master + fp32 accumulator + {cb.kind} state of"
        f" {cb.unit_params / 1e6:.2f}M parameters), always group {gib(cb.always_bytes):.3f} GiB, optimizer temporaries"
        f" {gib(cb.temp_bytes):.2f} GiB, {act}",
        f"{head} GPU K = {cb.k} active units (from {how}; the budget allows {cb.k_budget}): allocated"
        f" {gib(cb.allocated):.1f} GiB, device estimate {gib(cb.device):.1f} GiB (+{ALLOCATOR_SLACK:.0%} slack,"
        f" +{gib(OUTSIDE_BYTES):.1f} GiB outside the trainer)",
        f"{head} switch: {gib(cb.switch_bytes()):.2f} GiB per unit each way, ~{cb.switch_seconds():.1f} s for {cb.k}"
        f" units" + (" (pinned masters)" if cb.pinned_master else " (pageable)") + "; every unit trains in"
        f" {cb.k / max(1, cb.units):.0%} of the steps",
        f"{head} truncation: "
        + ("always small trains the embedders / time MLP / shared modulation, so every step backpropagates through all"
           " 42 blocks (weight gradients only for the active units)" if cb.always == "small" else
           "while the stem unit is inactive nothing below the lowest active block is recomputed or backpropagated"
           " (while it is active: all 42 blocks)" if stem else
           "nothing below the lowest active block is recomputed or backpropagated (the stem is never trained)"
           if cb.always == "head" else
           "nothing below the lowest active block is recomputed or backpropagated"),
    ]
