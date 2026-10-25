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
import json
import os
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


def _alloc_conf_warnings(frac: Optional[float]) -> list:
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    out = []
    if "garbage_collection_threshold" in conf and frac is None:
        out.append("PYTORCH_CUDA_ALLOC_CONF garbage_collection_threshold does nothing without a memory fraction"
                   " (set --cuda_memory_fraction or CUDA_MEMORY_FRACTION)")
    if "expandable_segments:true" in conf.replace(" ", "").lower() and sys.platform == "win32":
        out.append("PYTORCH_CUDA_ALLOC_CONF expandable_segments is not supported by Windows torch builds (no effect)")
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
