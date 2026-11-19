# MOSS full fine-tune, live text encoder (--moss_te_live): encode-ahead over the DataLoader
# (docs/moss_ft_te_live_fp16_plan.md 1.2).
#
# The DataLoader yields kohya's tokenized batches (no TE-outputs caching strategy is set, so BaseDataset.__getitem__
# runs process_caption - wildcard line, dropout, shuffle, tag dropout - and the MOSS tokenizer at every fetch:
# "input_ids_list" = [ids [B, L], mask [B, L]]). TextEncodeAhead pulls `ahead` batches, encodes every caption of them
# in one LiveTextEncoder pass (one weight transfer for all of them), and yields the batches in their order with
# "text_encoder_outputs_list" in exactly the cached path's batch form: [embeds bf16 [B, T_max, H] zero-padded, mask
# float32 [B, T_max] (1 / 0), caption-dropout rates float32 zeros [B]]. It changes when a batch's text is encoded, not
# which batch is trained or in which order.
#
# Timing: take_pass() hands the training loop the host seconds of the passes since its last call and the captions they
# encoded (a pass serves ahead / GA windows, so its time is divided by its own captions, not by one window's images).
# On CUDA the host seconds are mostly kernel-launch time (nothing waits for the GPU); CUDA events around each pass give
# its GPU time, read lazily (device_seconds) once the events have completed - after the window's loss.item().

from __future__ import annotations

import time
from collections import deque
from typing import Iterable, Iterator, List, Tuple

import torch

from library.utils import setup_logging

from .ft_budget import te_pass_gpu_bytes
from .strategy import pad_rows

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


def batch_rows(batch: dict) -> List[torch.Tensor]:
    """The unpadded token ids [T_b] of every sample of a tokenized batch (on the CPU)."""
    tokens = batch.get("input_ids_list")
    if not tokens:
        raise ValueError("MOSS live text encoder: the batch has no input_ids_list (a TE-outputs caching strategy is set,"
                         " or the dataset did not tokenize)")
    ids, mask = tokens[0].to("cpu"), tokens[1].to("cpu")
    rows = []
    for b, length in enumerate(mask.sum(dim=1).tolist()):
        if int(length) < 1:
            raise ValueError("every MOSS prompt has at least the 10 chat-template tokens")
        rows.append(ids[b, : int(length)])
    return rows


def live_outputs(rows: List[torch.Tensor], device=None) -> List[torch.Tensor]:
    """[embeds, mask, rates] of one batch from its encoded rows [T_b, H]: what the cached path's __getitem__
    (none_or_stack_elements + convert_te_output_for_batch) gives - embeds zero-padded to the longest row, the mask in
    float32, zero caption-dropout rates (a dropped caption was encoded as '' already)."""
    embeds, mask = pad_rows(rows)
    device = embeds.device if device is None else torch.device(device)
    return [embeds.to(device), mask.to(device=device, dtype=torch.float32),
            torch.zeros(len(rows), dtype=torch.float32, device=device)]


class TextEncodeAhead:
    """wrap(loader) yields the loader's batches with their text encoded, `ahead` batches per encoder pass."""

    def __init__(self, encoder, ahead: int):
        if int(ahead) < 1:
            raise ValueError(f"ahead must be >= 1, got {ahead!r}")
        self.encoder = encoder
        self.ahead = int(ahead)
        self.pass_count = 0
        self.pass_s_total = 0.0  # host seconds of every pass
        self.rows_total = 0  # captions encoded by every pass
        self.gpu_s_total = 0.0  # CUDA: GPU seconds of the passes whose events have completed
        self.gpu_rows_total = 0  # ... and their captions
        self._pending_s = 0.0
        self._pending_rows = 0
        self._timed: deque = deque()  # CUDA: (start event, end event, rows) not read yet

    def take_pass(self) -> Tuple[float, int]:
        """(host seconds, captions encoded) of the passes since the last call (the training loop takes the seconds out
        of its data time)."""
        seconds, rows = self._pending_s, self._pending_rows
        self._pending_s, self._pending_rows = 0.0, 0
        return seconds, rows

    def take_pass_seconds(self) -> float:
        """Seconds spent in passes since the last call (take_pass without the caption count)."""
        return self.take_pass()[0]

    @property
    def s_per_image(self) -> float:
        """Pass seconds per encoded caption over the run: GPU time when CUDA events have been read, else host time."""
        if self.gpu_rows_total:
            return self.gpu_s_total / self.gpu_rows_total
        return self.pass_s_total / max(1, self.rows_total)

    def device_seconds(self) -> Tuple[float, int]:
        """CUDA: fold the completed passes' event times into gpu_s_total / gpu_rows_total (never blocks; a pass whose
        end event has not completed stays for a later call). Returns the totals."""
        while self._timed and self._timed[0][1].query():  # pragma: no cover - CUDA only
            begin, end, rows = self._timed.popleft()
            self.gpu_s_total += begin.elapsed_time(end) / 1000.0
            self.gpu_rows_total += rows
        return self.gpu_s_total, self.gpu_rows_total

    def _event(self):
        if not getattr(self.encoder, "cuda", False):
            return None
        event = torch.cuda.Event(enable_timing=True)  # pragma: no cover - CUDA only
        event.record()  # pragma: no cover - the current (compute) stream the pass runs on
        return event  # pragma: no cover

    def _encode(self, batches: List[dict]) -> None:
        start = time.perf_counter()
        begin = self._event()
        per_batch = [batch_rows(batch) for batch in batches]
        rows_in_pass = sum(len(rows) for rows in per_batch)
        encoded = self.encoder.encode_rows([row for rows in per_batch for row in rows])
        at = 0
        for batch, rows in zip(batches, per_batch):
            if batch.get("text_encoder_outputs_list") is not None:
                raise ValueError("MOSS live text encoder: the batch already has text-encoder outputs (a TE cache is"
                                 " set up next to --moss_te_live)")
            mine = encoded[at: at + len(rows)]
            at += len(rows)
            device = batch["latents"].device if isinstance(batch.get("latents"), torch.Tensor) else None
            batch["text_encoder_outputs_list"] = live_outputs(mine, device)
            batch.pop("input_ids_list", None)
        if begin is not None:  # pragma: no cover - CUDA only
            self._timed.append((begin, self._event(), rows_in_pass))
        seconds = time.perf_counter() - start
        if self.pass_count == 0:
            self._log_groups(rows_in_pass)
        self.pass_count += 1
        self.pass_s_total += seconds
        self.rows_total += rows_in_pass
        self._pending_s += seconds
        self._pending_rows += rows_in_pass

    def _log_groups(self, rows: int) -> None:
        """--moss_te_live_batch > 1: the first training pass's layer calls (group sizes, padded lengths, padding), once."""
        batch = int(getattr(self.encoder, "batch_rows", 1) or 1)
        groups = list(getattr(self.encoder, "last_groups", None) or [])
        if batch <= 1 or not groups:
            return
        stats = getattr(self.encoder, "stats", {}) or {}
        tokens, padded = int(stats.get("tokens", 0)), int(stats.get("padded_tokens", 0))
        share = 100.0 * (padded - tokens) / max(1, tokens)
        peak = te_pass_gpu_bytes(groups, int(getattr(self.encoder, "ring_slots", 2) or 2))
        logger.info(f"live text encoder, first pass (--moss_te_live_batch {batch}): {rows} captions in {len(groups)}"
                    f" layer calls of {[count for count, _ in groups]} rows, padded to {[width for _, width in groups]}"
                    f" tokens ({tokens} real + {padded - tokens} pad tokens, +{share:.1f} %); pass peak"
                    f" ~{peak / (1 << 30):.2f} GiB for these shapes (9B estimate; the start-up budget line's pass peak"
                    " covers any pass whose captions have at most 300 tokens)")

    def wrap(self, loader: Iterable[dict]) -> Iterator[dict]:
        queue: deque = deque()
        iterator = iter(loader)
        exhausted = False
        while True:
            if not queue:
                if exhausted:
                    return
                held = []
                for batch in iterator:
                    held.append(batch)
                    if len(held) == self.ahead:
                        break
                else:
                    exhausted = True
                if not held:
                    return
                self._encode(held)
                queue.extend(held)
            yield queue.popleft()
