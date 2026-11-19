# MOSS-Image full fine-tune: loading the DiT for training, the trainable set and its parameter groups, and saving the
# trained DiT in the Base layout (docs/moss_full_finetune_design.md 4.4.1 and 4.7).
#
#   load_moss_for_full_ft(path, args, device)  -> (model, pool, slots)   the DiT built on the meta device, the host pool
#                                                 of the trainable set, the file read one tensor at a time
#   select_trainable(model, freeze)            --moss_ft_freeze (re.fullmatch), requires_grad set
#   param_groups(named, weight_decay)          "decay" (2-D weights) / "no_decay" (1-D, the two input embedders)
#   save_moss_dit(path, model, pool, dtype, metadata)   the Base key set, sorted, streamed (peak +1 tensor)
#   moss_dit_metadata(args, ...)               the DiT file's metadata
#
# Peak host memory while loading: the pools + one tensor (safetensors reads the file per tensor). The compute copies:
# bf16 joint-block Linears and, under --moss_stream_fp32, fp32 for every other parameter (fp32 for all with
# mixed_precision "no"); a trainable parameter's copy is always its master rounded to that dtype.

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from library.moss import ft_args, ft_host
from library.moss.ft_safetensors import TensorEntry, fsync_dir, write_safetensors
from library.moss.models import MOSS_DIT_CONFIG, MossTransformer, split_fused_qkv
from library.utils import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)

PREFIX = ft_args.PREFIX
FILE_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.", "model.")  # load_moss_dit's, in order
NO_DECAY_PREFIXES = ("video_embedder.", "text_embedder.proj.")  # the two input embedders (design 4.4.1 "Groups")
ROPE_KEY = "rope.bands"
SAVE_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "float": torch.float32, "fp32": torch.float32}
ARCHITECTURE = "moss-image-base"
IMPLEMENTATION = "https://github.com/OpenMOSS/MOSS-Image"


# ---- options ------------------------------------------------------------------------------------------------------
def compute_dtype(args) -> torch.dtype:
    """The compute dtype of mixed_precision (train_util.prepare_dtype's weight_dtype): bf16, fp16, else fp32."""
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(getattr(args, "mixed_precision", "bf16"), torch.float32)


def save_dtype(precision: Union[str, torch.dtype, None]) -> torch.dtype:
    """save_precision ("bf16" default, "fp16", "float") or a dtype -> the DiT file's dtype."""
    if isinstance(precision, torch.dtype):
        return precision
    try:
        return SAVE_DTYPES[str(precision or "bf16").lower()]
    except KeyError:
        raise ValueError(f"{PREFIX}save_precision must be one of bf16, fp16, float, got {precision!r}") from None


def attn_mode_of(args) -> str:
    """The DiT's attention mode as MossNetworkTrainer.load_unet_lazily picks it ("sdpa" is "torch")."""
    attn_mode = "torch"
    if getattr(args, "xformers", False):
        attn_mode = "xformers"
    if getattr(args, "attn_mode", None) is not None:
        attn_mode = "torch" if args.attn_mode == "sdpa" else args.attn_mode
    return attn_mode


def pool_roles(mode: str) -> Optional[Tuple[str, ...]]:
    """The HostPool roles of an optimizer mode; None for adafactor_fused (no host state)."""
    from library.moss.ft_optim import HostAdamW, HostCAME

    if mode == ft_args.ADAMW_HOST:
        return HostAdamW.POOL_ROLES
    if mode == ft_args.CAME_HOST:
        return HostCAME.POOL_ROLES
    if mode == ft_args.ADAFACTOR_FUSED:
        return None
    if mode == ft_args.CAROUSEL:
        return ("master",)  # only the fp32 masters live on the host; states move with their units (ft_carousel)
    raise ValueError(f"{PREFIX}unknown optimizer mode {mode!r}")


# ---- trainable set and groups -------------------------------------------------------------------------------------
def select_trainable(model: nn.Module, freeze="") -> List[Tuple[str, nn.Parameter]]:
    """[(name, parameter)] of the trained set in registration order (the order of the pool and the state files).
    `freeze`: --moss_ft_freeze, comma-separated regexes (or a list) matched with re.fullmatch on parameter names. Sets
    requires_grad (True: trained, False: frozen). ValueError when a pattern matches nothing or everything is frozen."""
    named = list(model.named_parameters())
    patterns = ft_args.split_freeze_patterns(freeze)
    frozen = set(ft_args.frozen_parameter_names([n for n, _ in named], patterns)) if patterns else set()
    if named and len(frozen) == len(named):
        raise ValueError(f"{PREFIX}--moss_ft_freeze {patterns} freezes every parameter: nothing would be trained")
    trainable = []
    for name, param in named:
        param.requires_grad_(name not in frozen)
        if name not in frozen:
            trainable.append((name, param))
    if frozen:
        count = sum(p.numel() for n, p in named if n in frozen)
        logger.info(f"{PREFIX}--moss_ft_freeze: {len(frozen)} of {len(named)} tensors frozen ({count / 1e9:.3f}B"
                    f" parameters); {sum(p.numel() for _, p in trainable) / 1e9:.3f}B trained")
    return trainable


def is_decay(name: str, param: torch.Tensor) -> bool:
    """Group 0 ("decay"): 2-D weights - joint blocks, refiner blocks, modulations, time MLP, final_linear_video. Group 1:
    every 1-D parameter (biases, norm weights) and the input embedders video_embedder.* / text_embedder.proj.*."""
    return param.dim() >= 2 and not name.startswith(NO_DECAY_PREFIXES)


def param_groups(named: Sequence[Tuple[str, torch.Tensor]], weight_decay: float = 0.0) -> List[dict]:
    """The optimizer's parameter groups: {"params", "weight_decay", "name"} for "decay" (weight_decay) and "no_decay"
    (0.0), in that order, each in the given order. Both use the optimizer's (scheduled) LR. An empty group is left out."""
    groups = []
    decay = [p for n, p in named if is_decay(n, p)]
    no_decay = [p for n, p in named if not is_decay(n, p)]
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay), "name": "decay"})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "name": "no_decay"})
    return groups


def trainable_named(slots: Sequence[ft_host.ParamSlot]) -> List[Tuple[str, nn.Parameter]]:
    """[(name, live parameter)] of bound slots (the input of param_groups and ft_host.bind_slots)."""
    return [(s.name, s.param) for s in slots]


# ---- loading ------------------------------------------------------------------------------------------------------
def _file_prefix(keys: Sequence[str]) -> str:
    for prefix in FILE_PREFIXES:
        if keys and all(k.startswith(prefix) for k in keys):
            return prefix
    return ""


def _read_dit_file(path: str, model: nn.Module, pool: Optional[ft_host.HostPool],
                   slots: Sequence[ft_host.ParamSlot]) -> None:
    """Fill the (materialised) model and the pool's master from the file, one tensor at a time. A fused joint-block
    q/k/v (heal layout) is split; strict: every parameter exactly once, nothing unknown, the model's shapes."""
    from safetensors import safe_open

    params = dict(model.named_parameters())
    slot_of = {s.name: s for s in slots} if pool is not None else {}
    loaded, unexpected, fused = set(), [], 0
    bands_shape = tuple(model.rope.bands.shape)
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        prefix = _file_prefix(keys)
        for file_key in keys:
            key = file_key[len(prefix):]
            items = {key: f.get_tensor(file_key)}
            if key.startswith("layers.") and key.endswith(".proj.qkv.weight"):
                split_fused_qkv(items)  # q, k, v rows of [3 dim, dim] (KBlueLeaf/MOSS-Image-heal's ComfyUI file)
                fused += 1
            for name, value in items.items():
                if name == ROPE_KEY:  # derived constants (MossRoPE re-derives them): the shape is checked only
                    if tuple(value.shape) != bands_shape:
                        raise ValueError(f"{PREFIX}{path}: {name} has shape {list(value.shape)}, the model"
                                         f" {list(bands_shape)}")
                    loaded.add(name)
                    continue
                param = params.get(name)
                if param is None:
                    unexpected.append(name)
                    continue
                if name in loaded:
                    raise ValueError(f"{PREFIX}{path}: {name} appears twice (a fused layers.N.proj.qkv.weight next to a"
                                     " separate q / k / v weight?)")
                if tuple(value.shape) != tuple(param.shape):
                    raise ValueError(f"{PREFIX}{path}: {name} has shape {list(value.shape)} in the file, the model"
                                     f" {list(param.shape)} (another MOSS config?)")
                with torch.no_grad():
                    slot = slot_of.get(name)
                    if slot is not None:
                        master = pool.view("master", slot).view(slot.shape)
                        master.copy_(value)  # exact for bf16 / fp16 / fp32 files
                        param.copy_(master)  # the compute copy is the master rounded to its dtype, always
                    else:
                        param.copy_(value)
                loaded.add(name)
            del items
    missing = [n for n in params if n not in loaded]
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing keys {missing[:8]}{' ...' if len(missing) > 8 else ''} ({len(missing)})")
        if unexpected:
            parts.append(f"unexpected keys {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''} ({len(unexpected)})")
        raise ValueError(f"{PREFIX}{path} is not a MOSS-Image DiT of this config: " + "; ".join(parts))
    if fused:
        logger.info(f"MOSS DiT: fused q/k/v split back into q / k / v in {fused} joint blocks (ComfyUI layout, e.g."
                    " heal)")


def load_moss_for_full_ft(path: str, args, device, trainable_fn: Optional[Callable] = None, *,
                          config: Optional[dict] = None) -> Tuple[MossTransformer, Optional[ft_host.HostPool],
                                                                  List[ft_host.ParamSlot]]:
    """The DiT for full fine-tuning, its HostPool and the trainable slots (design 4.4.1).

    1. MossTransformer(**config (default MOSS_DIT_CONFIG), attn_mode, split_attn) on the meta device, cast to the compute
       dtype (mixed_precision) and, with --moss_stream_fp32, the non-joint-Linear parameters to fp32.
    2. The trainable set: trainable_fn(model) -> [(name, param)] (default select_trainable with --moss_ft_freeze), its
       slots, and the HostPool of the optimizer mode (pool_roles; the fp32 accumulator when ft_host.needs_accumulator
       says so). adafactor_fused: no pool (None).
    3. model.to_empty(device); the slots bound to the live parameters (ft_host.bind_slots; the param groups are bound
       later with the optimizer); requires_grad restored.
    4. The file read one tensor at a time: prefixes stripped as load_moss_dit, a heal-layout qkv split; trained
       tensors into the master (fp32) and the compute copy from the master; frozen ones into the compute copy.
    5. --moss_stream_scale; model.train(); the compute copies checked against the masters.
    Gradient checkpointing, compile and checkpoint-reuse switches stay with the caller (design 4.3 step 7)."""
    config = dict(MOSS_DIT_CONFIG if config is None else config)
    device = torch.device(device)
    dtype = compute_dtype(args)
    mode = ft_args.optimizer_mode(args)
    roles = pool_roles(mode)
    with torch.device("meta"):
        model = MossTransformer(**config, attn_mode=attn_mode_of(args), split_attn=bool(getattr(args, "split_attn", False)))
    model.to(dtype)
    if getattr(args, "moss_stream_fp32", False):
        # on the meta device: the fp32-stream parameters are then allocated in fp32 and read from the file exactly (an
        # fp32 file keeps its values; enable_stream_fp32 after a bf16 copy would round them through bf16 first)
        model.enable_stream_fp32()
    trainable = (trainable_fn or (lambda m: select_trainable(m, getattr(args, "moss_ft_freeze", ""))))(model)
    slots = ft_host.make_slots(trainable)
    pool = None
    if roles is not None:
        with_acc = mode != ft_args.CAROUSEL and ft_host.needs_accumulator(
            int(getattr(args, "gradient_accumulation_steps", 1) or 1), getattr(args, "moss_ft_clip_mode", "stale"),
            float(getattr(args, "max_grad_norm", 0.0) or 0.0))
        pool = ft_host.HostPool(slots, roles, with_acc=with_acc)
    requires = {name: p.requires_grad for name, p in model.named_parameters()}

    model.to_empty(device=device)
    model.rope.bands = model.rope._default_bands(device)
    params = dict(model.named_parameters())
    for name, param in params.items():
        param.requires_grad_(requires[name])
    ft_host.bind_slots(slots, [(s.name, params[s.name]) for s in slots])

    _read_dit_file(str(path), model, pool, slots)
    stream_scale = float(getattr(args, "moss_stream_scale", 1.0))
    if stream_scale != 1.0:
        model.set_stream_scale(stream_scale)
    model.train()
    if dtype == torch.float16:
        fp16_range_report(model, pool, slots)
    if pool is not None:
        check_compute_copies(pool, slots)
    trained = sum(s.numel for s in slots)
    total = sum(p.numel() for p in params.values())
    pool_text = (f"host pool {'/'.join(pool.roles)}{' + acc' if pool.acc is not None else ''} "
                 f"{sum(pool.nbytes().values()) / 2**30:.2f} GiB" if pool is not None else "no host pool")
    logger.info(f"MOSS DiT loaded for full fine-tuning: {path} ({total / 1e9:.2f}B params, {trained / 1e9:.2f}B trained in"
                f" {len(slots)} tensors; compute {dtype}{' + fp32 stream' if model.stream_fp32 else ''}; {pool_text};"
                f" {device}, attn {model.attn_mode}{', split' if model.split_attn else ''})")
    return model, pool, slots


FP16_MAX = 65504.0
FP16_MIN_NORMAL = 2.0 ** -14  # 6.1e-5: below it fp16 is subnormal (fewer mantissa bits) or 0


@torch.no_grad()
def fp16_range_report(model: nn.Module, pool: Optional[ft_host.HostPool],
                      slots: Sequence[ft_host.ParamSlot]) -> Dict[str, int]:
    """fp16 compute (plan 2.1): refuse when a value of an fp16 compute copy is out of fp16's range - a trained
    parameter's fp32 master beyond 65504, or a frozen fp16 parameter that is not finite after the read - and log the
    share of the trained fp16 values below 6.1e-5 (subnormal in fp16) and the share the cast turns into exactly 0.
    Returns the counts {"values", "over", "subnormal", "zeroed"}."""
    counts = {"values": 0, "over": 0, "subnormal": 0, "zeroed": 0}
    over_names: List[str] = []
    trained = set()
    for slot in slots:
        trained.add(slot.name)
        if slot.compute_dtype != torch.float16:
            continue
        master = pool.view("master", slot) if pool is not None else slot.param.detach().to("cpu", torch.float32).view(-1)
        magnitude = master.abs()
        over = int((magnitude > FP16_MAX).sum())
        if over or not bool(torch.isfinite(master).all()):
            over_names.append(slot.name)
        nonzero = magnitude > 0
        counts["values"] += master.numel()
        counts["over"] += over
        counts["subnormal"] += int((nonzero & (magnitude < FP16_MIN_NORMAL)).sum())
        counts["zeroed"] += int((nonzero & (master.to(torch.float16) == 0)).sum())
    for name, param in model.named_parameters():
        if name in trained or param.dtype != torch.float16:
            continue
        if not bool(torch.isfinite(param.detach()).all()):
            over_names.append(name)
    if over_names:
        raise ValueError(f"{PREFIX}mixed_precision fp16: {len(over_names)} parameters have values beyond fp16's"
                         f" {FP16_MAX:g} ({counts['over']} trained values): {over_names[:8]}"
                         f"{' ...' if len(over_names) > 8 else ''}. Train with bf16 (the compute copy would be inf)")
    if counts["values"]:
        logger.info(f"{PREFIX}fp16 compute copies: {counts['subnormal'] / counts['values']:.3%} of the trained fp16"
                    f" values are below {FP16_MIN_NORMAL:.3g} (fp16 subnormal), {counts['zeroed'] / counts['values']:.3%}"
                    " become exactly 0 in the cast (the fp32 master keeps them)")
    return counts


@torch.no_grad()
def check_compute_copies(pool: ft_host.HostPool, slots: Sequence[ft_host.ParamSlot]) -> None:
    """RuntimeError unless every trained parameter == its master rounded to the parameter's dtype (compared on the CPU,
    one tensor at a time)."""
    bad = []
    for slot in slots:
        want = pool.view("master", slot).view(slot.shape).to(slot.param.dtype)
        if not torch.equal(slot.param.detach().to("cpu"), want):
            bad.append(slot.name)
    if bad:
        raise RuntimeError(f"{PREFIX}the compute copies of {len(bad)} parameters differ from their fp32 master rounded to"
                           f" their dtype: {bad[:8]}{' ...' if len(bad) > 8 else ''}")


# ---- saving -------------------------------------------------------------------------------------------------------
def dit_entries(model: MossTransformer, pool: Optional[ft_host.HostPool], dtype: torch.dtype) -> List[TensorEntry]:
    """The Base layout: every state-dict key of the model (its parameters and rope.bands), sorted. Trained parameters
    from the pool's master, the others from the model, cast to `dtype`; rope.bands = MossRoPE._default_bands, F32.
    The producers run one at a time (nothing is converted before it is written)."""
    slot_of = {s.name: s for s in pool.slots} if pool is not None else {}
    state = model.state_dict(keep_vars=True)
    entries = []
    for key in sorted(state):
        if key == ROPE_KEY:
            bands = model.rope._default_bands(torch.device("cpu"))
            entries.append(TensorEntry(key, torch.float32, tuple(bands.shape), lambda b=bands: b))
            continue
        tensor = state[key]
        slot = slot_of.get(key)
        if slot is not None:
            def produce(s=slot):
                return pool.view("master", s).view(s.shape).to(dtype)
        else:
            def produce(t=tensor):
                return t.detach().to(device="cpu", dtype=dtype)
        entries.append(TensorEntry(key, dtype, tuple(tensor.shape), produce))
    return entries


def save_moss_dit(path: str, model: MossTransformer, pool: Optional[ft_host.HostPool],
                  dtype: Union[torch.dtype, str, None] = torch.bfloat16, metadata: Optional[Dict[str, str]] = None) -> int:
    """Write the trained DiT in the Base layout (design 4.7): exactly the Base key set, sorted, bare names; `dtype`
    (bf16 default: the Base file's dtype) for every parameter, rope.bands F32; metadata with {"format": "pt"} first.
    Trained parameters come from the pool's fp32 master (pool None, adafactor_fused: from the model), frozen ones from
    the compute copy (exact for Base values). Streamed through `<path>.tmp` (fsync, rename): peak +1 tensor, and a crash
    leaves no truncated file under the name. Call it after a window's drain (the pool is stable then). Returns the file
    size."""
    dtype = save_dtype(dtype)
    entries = dit_entries(model, pool, dtype)
    meta = {"format": "pt"}
    meta.update({str(k): str(v) for k, v in (metadata or {}).items() if k != "format"})
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    try:
        size = write_safetensors(tmp, entries, meta, fsync=True)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    fsync_dir(folder)
    logger.info(f"saved the MOSS DiT: {path} ({len(entries)} tensors, {str(dtype).replace('torch.', '')},"
                f" {size / 1e9:.2f} GB)")
    return size


def moss_dit_metadata(args, *, mode: str, optimizer_kwargs: Optional[dict] = None, global_step: int = 0,
                      epoch: int = 0, train_images: Optional[int] = None) -> Dict[str, str]:
    """The DiT file's metadata (design 4.7): {"format": "pt"} as Base, the modelspec architecture / implementation /
    title, moss_train_utils.moss_metadata(args) without its LoRA key, and ss_moss_ft_*: the mode, betas, LR (phases),
    global step, epoch, base file name, train images and the frozen patterns."""
    from library.moss import train_utils as moss_train_utils

    meta = {"format": "pt", "modelspec.architecture": ARCHITECTURE, "modelspec.implementation": IMPLEMENTATION,
            "modelspec.title": str(getattr(args, "output_name", None) or "MOSS-Image full fine-tune")}
    moss = moss_train_utils.moss_metadata(args)
    moss.pop("ss_moss_lora_exclude_blocks", None)
    meta.update(moss)
    betas = (optimizer_kwargs or {}).get("betas")
    meta.update({
        "ss_moss_ft_mode": str(mode),
        "ss_moss_ft_betas": "" if betas is None else ",".join(f"{float(b):g}" for b in betas),
        "ss_moss_ft_lr_phases": str(getattr(args, "moss_ft_lr_phases", None) or ""),
        "ss_moss_ft_learning_rate": str(getattr(args, "learning_rate", "")),
        "ss_moss_ft_global_step": str(int(global_step)),
        "ss_moss_ft_epoch": str(int(epoch)),
        "ss_moss_ft_base_model": os.path.basename(str(getattr(args, "pretrained_model_name_or_path", "") or "")),
        "ss_moss_ft_train_images": "" if train_images is None else str(int(train_images)),
        "ss_moss_ft_freeze": ",".join(ft_args.split_freeze_patterns(getattr(args, "moss_ft_freeze", ""))),
        "ss_moss_ft_gradient_accumulation_steps": str(getattr(args, "gradient_accumulation_steps", 1)),
    })
    if mode == ft_args.CAROUSEL:
        import json

        meta[f"ss_moss_ft_{ft_args.CAROUSEL}"] = json.dumps({
            "optimizer": ft_args.carousel_kind(args),
            "active": getattr(args, "_moss_rotation_k", None) or ft_args.carousel_arg(args, "active"),
            **{key: ft_args.carousel_arg(args, key) for key in ("period", "select", "always", "state", "state_dtype")}},
            sort_keys=True)
    return meta
