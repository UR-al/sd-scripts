# --anima_tier_timing (default off; 2026-10-26): forward + backward time of every micro-step (one chunk: a bucket batch,
# or its token-budget / --step_rows piece) grouped by resolution tier, so a change aimed at the host-bound small tiers
# (--anima_block_compile) can be read per tier instead of from the mixed s/step of a multi-resolution set.
#
# * tier = the nearest of 512^2 = 1024, 1024^2 = 4096, 1536^2 = 9216, 2048^2 = 16384 DiT tokens PER SAMPLE on a log scale
#   (the aspect-ratio buckets of a tier have fewer tokens than its square: 704x320 = 880 tokens is the 512 tier; the
#   geometric midpoints 2048 / 6144 / 12288 are the boundaries, a tie goes to the lower tier).
# * CUDA: a pair of CUDA events recorded on the current stream around the micro-step; the elapsed time between them is
#   the GPU timeline span of forward + backward INCLUDING the gaps where the GPU waited for the host (that is the
#   host-bound time we want to see). The pairs wait in a list and are read only at a report, and only those whose end
#   event has completed (Event.query(), no synchronization) - the rest stay pending for the next report. CPU: perf_counter.
# * off: the trainer's hooks return after one attribute check per micro-step; no event is created.
# * reports: every --anima_tier_timing_every optimizer steps the window's per-tier count / mean / median go to the tracker
#   (tier_time/<tier>/count|mean_ms|median_ms, tier_time/pending); at the end of every epoch one INFO line with the
#   epoch's per-tier numbers.

from __future__ import annotations

import logging
import math
import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# (DiT tokens per sample of the tier's square bucket, label); 1024^2 pixels -> 128^2 latents -> 64^2 patches of 2
TIERS: Tuple[Tuple[int, str], ...] = ((1024, "512"), (4096, "1024"), (9216, "1536"), (16384, "2048"))
UNKNOWN = "unknown"
VAE_SPATIAL_DOWNSCALE = 8


def tier_of_tokens(tokens: int) -> str:
    """The tier label whose square token count is nearest to `tokens` on a log scale (a tie goes to the lower tier);
    UNKNOWN for a non-positive count."""
    tokens = int(tokens)
    if tokens <= 0:
        return UNKNOWN
    lt = math.log(tokens)
    best: Optional[Tuple[float, str]] = None
    for n, name in TIERS:
        d = abs(lt - math.log(n))
        if best is None or d < best[0] - 1e-12:
            best = (d, name)
    return best[1]


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def sample_tokens(batch: Any, patch: int = 2) -> int:
    """DiT tokens per sample of a collate batch (or chunk): from the cached latents (B, C, H, W) / (B, C, T, H, W) as
    ceil(H / patch) * ceil(W / patch), else from the pixels (B, C, Hpx, Wpx) through the VAE downscale; 0 when the batch
    carries neither."""
    if not isinstance(batch, dict):
        return 0
    latents = batch.get("latents")
    if isinstance(latents, torch.Tensor) and latents.ndim >= 3:
        return _ceil_div(latents.shape[-2], patch) * _ceil_div(latents.shape[-1], patch)
    images = batch.get("images")
    if isinstance(images, torch.Tensor) and images.ndim >= 3:
        h = int(images.shape[-2]) // VAE_SPATIAL_DOWNSCALE
        w = int(images.shape[-1]) // VAE_SPATIAL_DOWNSCALE
        return _ceil_div(h, patch) * _ceil_div(w, patch)
    return 0


def _cuda_event():
    return torch.cuda.Event(enable_timing=True)


class TierTimer:
    """Collects the forward + backward spans of micro-steps by tier. start(tokens) before the forward, stop() after the
    backward (both on the training thread, in order). report() harvests the finished spans of the window and returns the
    per-tier statistics; epoch_report() the same over everything harvested since the last epoch report.

    device_type "cuda" (with CUDA available) records event pairs on the current stream; anything else uses `clock`
    (time.perf_counter by default). `event_factory` exists for tests (an object with record() / query() / elapsed_time())."""

    def __init__(self, device_type: str = "cpu", clock: Optional[Callable[[], float]] = None,
                 event_factory: Optional[Callable[[], Any]] = None):
        self.cuda = event_factory is not None or (device_type == "cuda" and torch.cuda.is_available())
        self._event = event_factory or _cuda_event
        self._clock = clock or time.perf_counter
        self._open: Optional[Tuple[str, Any]] = None
        self._pending: List[Tuple[str, Any, Any]] = []
        self._window: Dict[str, List[float]] = {}
        self._epoch: Dict[str, List[float]] = {}
        self.dropped = 0  # start() called while a span was open (the previous span is dropped, not mis-measured)

    # ------------------------------------------------------------------------------------------------- recording
    def start(self, tokens: int) -> None:
        if self._open is not None:
            self.dropped += 1
        tier = tier_of_tokens(tokens)
        if self.cuda:
            ev = self._event()
            ev.record()
            self._open = (tier, ev)
        else:
            self._open = (tier, self._clock())

    def stop(self) -> None:
        if self._open is None:
            return
        tier, start = self._open
        self._open = None
        if self.cuda:
            end = self._event()
            end.record()
            self._pending.append((tier, start, end))
        else:
            self._pending.append((tier, start, self._clock()))

    # ------------------------------------------------------------------------------------------------- harvesting
    def harvest(self) -> int:
        """Move every finished pending span into the window (CUDA: only pairs whose end event completed - no sync); returns
        how many were moved."""
        keep: List[Tuple[str, Any, Any]] = []
        moved = 0
        for tier, start, end in self._pending:
            if self.cuda:
                if not end.query():
                    keep.append((tier, start, end))
                    continue
                ms = float(start.elapsed_time(end))
            else:
                ms = (end - start) * 1000.0
            self._window.setdefault(tier, []).append(ms)
            self._epoch.setdefault(tier, []).append(ms)
            moved += 1
        self._pending = keep
        return moved

    @property
    def pending(self) -> int:
        return len(self._pending)

    @staticmethod
    def _stats(samples: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for tier, ms in samples.items():
            if ms:
                out[tier] = {"count": float(len(ms)), "mean_ms": statistics.fmean(ms), "median_ms": statistics.median(ms)}
        return out

    def report(self) -> Dict[str, Dict[str, float]]:
        """Harvest, then the per-tier count / mean / median of the window since the previous report (the window is
        cleared)."""
        self.harvest()
        stats = self._stats(self._window)
        self._window = {}
        return stats

    def epoch_report(self) -> Dict[str, Dict[str, float]]:
        """Harvest, then the per-tier statistics over everything harvested since the previous epoch report."""
        self.harvest()
        stats = self._stats(self._epoch)
        self._epoch = {}
        return stats

    # ------------------------------------------------------------------------------------------------- formatting
    @staticmethod
    def tier_order(stats: Dict[str, Dict[str, float]]) -> List[str]:
        order = [name for _, name in TIERS] + [UNKNOWN]
        return [t for t in order if t in stats] + sorted(t for t in stats if t not in order)

    def scalars(self, stats: Dict[str, Dict[str, float]], prefix: str = "tier_time") -> Dict[str, float]:
        """Tracker scalars: <prefix>/<tier>/count|mean_ms|median_ms and <prefix>/pending."""
        out: Dict[str, float] = {}
        for tier in self.tier_order(stats):
            for k, v in stats[tier].items():
                out[f"{prefix}/{tier}/{k}"] = v
        out[f"{prefix}/pending"] = float(self.pending)
        return out

    def format_line(self, stats: Dict[str, Dict[str, float]], what: str = "epoch") -> str:
        parts = [f"{tier} n={int(s['count'])} mean {s['mean_ms']:.0f} ms median {s['median_ms']:.0f} ms"
                 for tier, s in ((t, stats[t]) for t in self.tier_order(stats))]
        body = " | ".join(parts) if parts else "no finished micro-steps"
        tail = f" | pending {self.pending}" if self.pending else ""
        drop = f" | dropped {self.dropped}" if self.dropped else ""
        return f"tier timing ({what}; fwd+bwd per micro-step by tokens-per-sample tier): {body}{tail}{drop}"
