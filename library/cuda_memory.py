# CUDA caching-allocator controls for any network / algo; all off by default.
# * --cuda_memory_fraction F (or env CUDA_MEMORY_FRACTION): torch.cuda.set_per_process_memory_fraction(F). Before the
#   allocator would reserve past F x total VRAM it frees its unused cache and retries, and only then raises OOM (instead
#   of growing into Windows' shared-memory fallback). torch applies PYTORCH_CUDA_ALLOC_CONF garbage_collection_threshold
#   only when a fraction is set (c10/cuda/CUDACachingAllocator.cpp: `set_fraction &&`), and expandable_segments is
#   compiled out of the Windows build (c10/cuda/CMakeLists.txt: `if(NOT WIN32)`), torch 2.7.1.
# * --cuda_empty_cache_every_n_steps N / --cuda_empty_cache_above_gib G: torch.cuda.empty_cache() at an optimizer step
#   (activations freed) every N steps / when reserved > G GiB.
# * CUDA_MEMLOG=N: allocated / reserved / window peaks / fragmentation / allocator retries every N optimizer steps and
#   after sampling / save_state / epoch end; CUDA_MEMLOG_FILE=path writes one JSON line per entry instead (N defaults to
#   50, CUDA_MEMLOG=0 switches it off). Peaks reset after each entry, so do not combine with TSM_LOKR_MEMLOG (its max_*
#   would become per-window too). Every process writes (multi-GPU: one file, rows tagged with device index and rank;
#   the save_state marks come from the main process only). Applied by NetworkTrainer.train (train_network.py and the
#   *_train_network.py scripts).
# * torch >= 2.14: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True unless the variable is already set (2026-11-04).
#   Windows builds ship expandable segments from 2.14 (2.13 and 2.7.1 compile them out). Without them every tensor of
#   10 MiB or more gets a segment of its own exact size and free segments never merge, so after each epoch-end
#   empty_cache the pool rebuilt in the shuffled bucket order up to the 32 GB ceiling, where WDDM paged over PCIe
#   (~130 W, 20-500 s/step). On a 2.9B LoKr run 2.14 kept reserved = max_allocated (19.4 GiB, device 21 GiB) across an
#   epoch boundary where 2.7.1 reached 30.5 GiB. Opt out: set PYTORCH_CUDA_ALLOC_CONF (or PYTORCH_ALLOC_CONF) yourself,
#   e.g. expandable_segments:False. Applied by NetworkTrainer.train and anima_train.py (full fine-tune).
import json
import os
import re
import sys
import time
from typing import Optional

import torch

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

GIB = 2**30
_CAP: dict = {}  # device index -> cap in bytes (fraction x total, as the allocator computes it)


def _rank() -> int:
    """The process rank under accelerate / torchrun (their env, so no accelerator handle is needed); 0 single-process."""
    for name in ("RANK", "LOCAL_RANK"):
        v = os.environ.get(name, "").strip()
        if v.isdigit():
            return int(v)
    return 0


def _cuda(device) -> Optional[torch.device]:
    if device is None:
        return None
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return device if device.index is not None else torch.device("cuda", torch.cuda.current_device())


def resolve_fraction(args) -> Optional[float]:
    """--cuda_memory_fraction, else env CUDA_MEMORY_FRACTION, else None."""
    frac = getattr(args, "cuda_memory_fraction", None)
    if frac is None:
        env = os.environ.get("CUDA_MEMORY_FRACTION", "").strip()
        frac = float(env) if env else None
    if frac is not None and not 0.0 < frac <= 1.0:
        raise ValueError(f"cuda_memory_fraction must be in (0, 1], got {frac}")
    return frac


EXPANDABLE_SEGMENTS_TORCH = (2, 14)  # first torch whose Windows build has expandable segments (c10_cuda.dll)


def torch_minor_version(version: Optional[str] = None) -> tuple:
    """(major, minor) of torch.__version__ (or of `version`), e.g. '2.14.0+cu130' -> (2, 14); (0, 0) if unreadable."""
    m = re.match(r"\s*(\d+)\.(\d+)", version if version is not None else torch.__version__)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


# the variables torch reads its allocator config from (2.14: PYTORCH_CUDA_ALLOC_CONF first, else PYTORCH_ALLOC_CONF)
ALLOC_CONF_VARS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")
EXPANDABLE_SEGMENTS_CONF = "expandable_segments:True"


def _set_allocator_settings(conf: str) -> None:
    """The runtime allocator-config setter (torch 2.14: torch._C._accelerator_setAllocatorSettings, older:
    torch.cuda.memory._set_allocator_settings). The allocator checks expandable_segments whenever it creates a segment,
    so this takes effect for every segment created afterwards, also after CUDA was initialized."""
    setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    if setter is None:
        setter = torch.cuda.memory._set_allocator_settings
    setter(conf)


def default_expandable_segments(env=None, version: Optional[str] = None) -> bool:
    """torch >= 2.14: turn expandable segments on unless an allocator config is already set (see the header). Sets
    PYTORCH_CUDA_ALLOC_CONF (child processes, an allocator not created yet) and, when CUDA is already initialized, the
    runtime setting too: importing flash-attn / triton queries the device (library/attention.py get_device_capability),
    which reads the allocator config before the trainer starts. Call before the models reach the GPU (the setting only
    applies to segments created afterwards). Returns True when it turned them on."""
    env = os.environ if env is None else env
    if torch_minor_version(version) < EXPANDABLE_SEGMENTS_TORCH:
        return False
    if any(env.get(name, "").strip() for name in ALLOC_CONF_VARS):
        return False  # the user's (or a launcher's) allocator config wins
    env["PYTORCH_CUDA_ALLOC_CONF"] = EXPANDABLE_SEGMENTS_CONF
    if env is os.environ and torch.cuda.is_available() and torch.cuda.is_initialized():
        try:
            _set_allocator_settings(EXPANDABLE_SEGMENTS_CONF)
        except Exception as e:  # an allocator backend without the setting (cudaMallocAsync) or a torch without the API
            logger.warning(f"could not turn expandable segments on at runtime ({type(e).__name__}: {e}); set"
                           f" PYTORCH_CUDA_ALLOC_CONF={EXPANDABLE_SEGMENTS_CONF} before starting the trainer")
            return False
        if torch.cuda.memory_reserved() > 0:
            logger.warning("expandable segments turned on after CUDA memory was reserved: only new segments expand")
    logger.info(f"PYTORCH_CUDA_ALLOC_CONF={EXPANDABLE_SEGMENTS_CONF} (torch >= 2.14: the CUDA cache grows in one mapped"
                " range instead of exact-size segments; set PYTORCH_CUDA_ALLOC_CONF yourself to override)")
    return True


def _alloc_conf_warnings(frac: Optional[float], version: Optional[str] = None) -> list:
    # the effective config, read the way torch does: PYTORCH_CUDA_ALLOC_CONF, else PYTORCH_ALLOC_CONF
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or os.environ.get("PYTORCH_ALLOC_CONF", "")
    out = []
    if "garbage_collection_threshold" in conf and frac is None:
        out.append("PYTORCH_CUDA_ALLOC_CONF garbage_collection_threshold does nothing without a memory fraction"
                   " (set --cuda_memory_fraction or CUDA_MEMORY_FRACTION)")
    if ("expandable_segments:true" in conf.replace(" ", "").lower() and sys.platform == "win32"
            and torch_minor_version(version) < EXPANDABLE_SEGMENTS_TORCH):
        out.append("PYTORCH_CUDA_ALLOC_CONF expandable_segments is not supported by Windows torch builds before 2.14"
                   " (no effect)")
    return out


def apply_memory_fraction(args, device) -> Optional[float]:
    """Call once, right after the accelerator is created (before the models reach the GPU)."""
    frac = resolve_fraction(args)
    d = _cuda(device)
    if d is None:
        return None
    for w in _alloc_conf_warnings(frac):
        logger.warning(w)
    if frac is None:
        return None
    torch.cuda.set_per_process_memory_fraction(frac, d)
    total = torch.cuda.get_device_properties(d).total_memory
    _CAP[d.index] = int(frac * total)
    logger.info(f"CUDA memory fraction {frac:.3f}: caching allocator capped at {frac * total / GIB:.2f} of"
                f" {total / GIB:.2f} GiB (CUDA context and other processes come on top)")
    return frac


def cap_bytes(device) -> Optional[int]:
    d = _cuda(device)
    return None if d is None else _CAP.get(d.index)


def maybe_empty_cache(args, device, global_step: int) -> bool:
    """At an optimizer-step boundary. True when the cache was released."""
    d = _cuda(device)
    if d is None:
        return False
    n = getattr(args, "cuda_empty_cache_every_n_steps", None) or 0
    above = getattr(args, "cuda_empty_cache_above_gib", None)
    due = n > 0 and global_step % n == 0
    if not due and above is not None:
        due = torch.cuda.memory_reserved(d) > above * GIB
    if due:
        torch.cuda.empty_cache()
    return due


class MemLog:
    """CUDA_MEMLOG / CUDA_MEMLOG_FILE: memory per window of optimizer steps (see the header)."""

    def __init__(self, device):
        self.device = _cuda(device)
        env = os.environ.get("CUDA_MEMLOG", "").strip()
        self.path = os.environ.get("CUDA_MEMLOG_FILE", "").strip() or None
        self.every = int(env) if env.isdigit() else 0
        off = env.isdigit() and self.every == 0  # CUDA_MEMLOG=0: off even with CUDA_MEMLOG_FILE
        self.enabled = self.device is not None and not off and (self.every > 0 or self.path is not None)
        if self.enabled and self.every == 0:
            self.every = 50
        if self.enabled and os.environ.get("TSM_LOKR_MEMLOG", "").strip():
            logger.warning("CUDA_MEMLOG resets the peak counters, so TSM_LOKR_MEMLOG max_* values become per-window")

    def step(self, global_step: int) -> None:
        if self.enabled and global_step % self.every == 0:
            self.mark("step", global_step)

    def mark(self, where: str, global_step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        d = self.device
        s = torch.cuda.memory_stats(d)
        free, total = torch.cuda.mem_get_info(d)
        row = {
            "where": where,
            "step": global_step,
            "time": round(time.time(), 1),
            "device": d.index,  # multi-GPU: every rank writes; with per-process CUDA_VISIBLE_DEVICES the rank tells them apart
            "rank": _rank(),
            "allocated": torch.cuda.memory_allocated(d) / GIB,
            "reserved": torch.cuda.memory_reserved(d) / GIB,
            "max_allocated": torch.cuda.max_memory_allocated(d) / GIB,  # since the previous entry
            "max_reserved": torch.cuda.max_memory_reserved(d) / GIB,
            "inactive_split": s.get("inactive_split_bytes.all.current", 0) / GIB,  # fragmentation the cap cannot free
            "inactive_split_peak": s.get("inactive_split_bytes.all.peak", 0) / GIB,
            "device_used": (total - free) / GIB,  # whole GPU, all processes (~ nvidia-smi)
            "cap": (_CAP.get(d.index) or 0) / GIB,
            "alloc_retries": s.get("num_alloc_retries", 0),  # cumulative: cache flushes forced by the cap / OOM
            "ooms": s.get("num_ooms", 0),
            "cuda_mallocs": s.get("num_device_alloc", 0),
            "cuda_frees": s.get("num_device_free", 0),
        }
        torch.cuda.reset_peak_memory_stats(d)
        if self.path:
            try:  # a log problem (folder gone, file locked by an editor) must not stop the training
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    print(json.dumps(row), file=f)
            except OSError as e:
                logger.warning(f"CUDA_MEMLOG_FILE {self.path}: {e}; memory log switched off")
                self.enabled = False
        else:
            logger.info(
                f"memlog {where} step {global_step} cuda:{d.index} rank {row['rank']}: allocated {row['allocated']:.2f},"
                f" reserved {row['reserved']:.2f},"
                f" window max allocated {row['max_allocated']:.2f} / reserved {row['max_reserved']:.2f}, inactive split"
                f" {row['inactive_split']:.2f} (peak {row['inactive_split_peak']:.2f}), device used"
                f" {row['device_used']:.2f} GiB; retries {row['alloc_retries']}, mallocs {row['cuda_mallocs']}"
            )
