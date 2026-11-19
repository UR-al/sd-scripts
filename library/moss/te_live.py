# MOSS full fine-tune, live text encoder (--moss_te_live): the streaming engine (docs/moss_ft_te_live_fp16_plan.md
# 1.2, 1.6).
#
# LiveTextEncoder encodes a pass of captions with the frozen Qwen3.5-9B whose decoder layers stay in host RAM
# (te_live_weights.HostTextEncoderWeights). A pass is layer-major: layer l is copied once into one of R device slots
# and applied to every caption of the pass, each caption alone and unpadded - the cached path's math
# (MossTextEncodingStrategy.encode_tokens runs one unpadded prompt per forward), so every row equals the cached row.
# The code that runs is qwen35.py's: a meta-built MossQwen35TextEncoder ("shell") whose layer parameters are bound to
# views of the slot for the duration of the layer (module._parameters[name] = view) and set back to meta after it.
#
# Schedule (CUDA): a stager thread copies layer l into a pinned staging entry (copy / mmap host modes; the pinned host
# mode is read directly), the main thread enqueues its H2D copy on a copy stream after layer l - R finished with that
# slot, and the compute stream waits for the copy before running layer l: layer l + 1 uploads while layer l computes
# (prefetch depth R - 1). The R device slots are separate allocations (no single R x 417 MiB block the caching
# allocator would have to find free again at every pass - Windows has no expandable segments). The first R layers of a
# pass are staged only after the previous pass's last H2D copy out of the staging entries has finished (an event on the
# copy stream; nothing else orders them). On the CPU the same steps run synchronously (the stager thread too when staging is forced, as
# the tests do). Everything runs under torch.no_grad() - not inference_mode: the DiT saves the text tensor for its
# backward, and inference tensors cannot be saved for backward.
#
# Batched layer calls (--moss_te_live_batch N > 1, opt-in; N = 1 is the per-row path above, unchanged): the rows of a
# pass are sorted by length and cut into groups of at most N (length_groups; captions past the 300 tokens of the VRAM
# estimate in smaller groups of their own, at most N x 300 padded tokens per call), each group right-padded to its
# longest row (pad_group) and run through every layer as ONE [G, L, H] call instead of G calls at batch 1 (24 GatedDeltaNet
# layers with a pure-torch chunked recurrence: ~a hundred small kernels per row and layer). Each output row is its
# group row sliced back to its own length, in the caller's order. Why right padding cannot change a real token
# (qwen35.py, read in full; tests/test_moss_ft_te_live_batch.py checks each point):
#   * per token, nothing across positions: the embedding, every RMSNorm (input / post-attention / final norm, q_norm /
#     k_norm and the gated norm reduce over the hidden or head dim only), the MLP, every in / out projection, the
#     GatedDeltaNet's beta = sigmoid(b) and g = -exp(A_log) * softplus(a + dt_bias), the l2norm of q / k (head dim).
#     Nothing reduces over the sequence.
#   * GatedDeltaNet's short conv: F.conv1d with K - 1 zeros on both sides, cut to [:length]: output t reads inputs
#     t - K + 1 .. t only (causal; the zeros on the left are the same as the per-row call's).
#   * chunk_gated_delta_rule: inside a chunk the decay matrix is tril()'d before and after exp() and the k.k matrix is
#     masked_fill(upper, 0)'d, the triangular solve reads columns < i only, and the recurrent state carries earlier
#     chunks only: output t is a sum over tokens <= t. A later (pad) token enters those sums only as an exact 0
#     coefficient times its value; 0 x finite = 0 and adding 0 changes no sum, but 0 x inf / nan = nan - so every pad
#     activation must stay finite (next point). The group's chunk padding (to a multiple of 64) is zeros, as per row.
#   * full attention: SDPA(is_causal = L > 1) builds its mask from the shape - the top-left lower triangle, the same for
#     every batch element - and right padding keeps each row's tokens at positions 0 .. T - 1, so a real query sees only
#     real keys at or before it. A row of length 1 in a group of L > 1 attends itself only, as is_causal=False on one
#     token does. No query row is fully masked (the diagonal), so SDPA's softmax never sees an all -inf row.
#   * rotary: cos / sin are elementwise in the position (arange(L) * inv_freq): the first T positions of the group's
#     table equal the row's own table bit for bit.
#   * the pad token: each row is padded with its OWN LAST TOKEN ID. A padded row is then a real token sequence (the
#     prompt with its last token repeated), so every pad activation is what the encoder computes for a real prompt -
#     finite as real tokens are (no zero / out-of-vocabulary embedding that could drift to inf in bf16). Pad positions
#     are cut off; nothing of them is returned.
# The rows are close to the per-row rows, not bit-identical: a [G, L, H] matmul / SDPA and the gated delta rule's
# reductions over more chunks may pick other kernels or accumulation orders than a [1, T, H] call (fp32 CPU tiny config:
# max abs ~1e-5 at |values| ~5; with other pad ids in the same shapes the rows are equal bit for bit). A group of one
# unpadded row runs the per-row shapes (bit-identical). Memory: a call's activations and temporaries scale with its
# rows x its longest row; length_groups keeps that within the call ft_budget.te_live_gpu_bytes charges (min(N, rows)
# rows x 300 tokens), ft_budget.te_pass_gpu_bytes models a pass's actual calls; te_live_feed logs the first pass's
# groups and that figure.

from __future__ import annotations

import concurrent.futures
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from library.utils import setup_logging

from .ft_budget import DEFAULT_TEXT_TOKENS, te_call_rows, te_group_sizes
from .qwen35 import MossQwen35TextEncoder
from .strategy import pad_rows
from .te_live_weights import HostTextEncoderWeights, LayerLayout, slot_bytes, staging_entry_bytes

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)

RING_SLOTS = (2, 4)


def length_groups(lengths: Sequence[int], batch: int, text_tokens: int = DEFAULT_TEXT_TOKENS) -> List[List[int]]:
    """--moss_te_live_batch: the row indices of every batched layer call of a pass, consecutive runs of the rows
    sorted by length (longest first; equal lengths keep their order). A call holds at most G = min(batch, n) rows
    (ft_budget.te_call_rows) and at most G x text_tokens padded tokens (rows x its longest), the call
    ft_budget.te_live_gpu_bytes charges:
    * captions past text_tokens are grouped among themselves, longest first, each call as many as fit that budget (one
      caption past it alone, as on the per-row path) - so a long caption never pads short ones, and the longest
      captions no longer fill the first and largest group (reviewer 11-18: N = 32 put the smoke subset's 32 captions,
      10-770 tokens, into one call of 32 x 770);
    * the others are cut into the groups of ft_budget.te_group_sizes (as even as possible, at most `batch` rows of at
      most text_tokens each): with no caption past text_tokens, exactly the groups of before.
    batch 1: one group per row."""
    if isinstance(batch, bool) or int(batch) < 1:
        raise ValueError(f"the layer batch must be an integer >= 1, got {batch!r}")
    if isinstance(text_tokens, bool) or int(text_tokens) < 1:
        raise ValueError(f"the text tokens of a caption in the estimate must be an integer >= 1, got {text_tokens!r}")
    batch, text_tokens = int(batch), int(text_tokens)
    order = sorted(range(len(lengths)), key=lambda i: (-int(lengths[i]), i))
    if not order:
        return []
    rows = te_call_rows(len(order), batch)
    budget = rows * text_tokens
    long = sum(1 for i in order if int(lengths[i]) > text_tokens)
    groups, at = [], 0
    while at < long:  # the longest first: a call's width is its first row's length
        take = min(rows, max(1, budget // int(lengths[order[at]])), long - at)
        groups.append(order[at: at + take])
        at += take
    for size in te_group_sizes(len(order) - at, batch):
        groups.append(order[at: at + size])
        at += size
    return groups


def pad_group(rows: Sequence[torch.Tensor]) -> torch.Tensor:
    """[T_i] token ids -> [G, max T] long on the CPU, each row right-padded with its own last id (a real token: the
    pad activations are the encoder's on a real prompt, finite; see the header)."""
    flat = [ids.reshape(-1).to(device="cpu", dtype=torch.long) for ids in rows]
    width = max(int(ids.numel()) for ids in flat)
    out = torch.empty(len(flat), width, dtype=torch.long)
    for j, ids in enumerate(flat):
        if ids.numel() < 1:
            raise ValueError("every MOSS prompt has at least the 10 chat-template tokens")
        out[j, : ids.numel()] = ids
        out[j, ids.numel():] = ids[-1]
    return out


class _Completed:
    """The CPU stand-in of a CUDA event: a synchronous copy has completed when it returns."""

    @staticmethod
    def synchronize() -> None:
        return None


_DONE = _Completed()


class LiveTextEncoder:
    """encode_rows(rows) / encode_tokens(tokens) with weights streamed from host RAM. Not thread-safe; one pass at a
    time. close() joins the stager and releases the staging buffer and the weights. batch_rows (--moss_te_live_batch):
    1 = every row alone (bit for bit the cached path); N > 1 = batched layer calls of up to N length-sorted,
    right-padded rows (see the header). It may be changed between passes (the benchmark does)."""

    def __init__(self, weights: HostTextEncoderWeights, device, ring_slots: int = 2, staging_threads: int = 8,
                 staging: Optional[bool] = None, batch_rows: int = 1):
        if ring_slots not in RING_SLOTS:
            raise ValueError(f"ring_slots must be one of {RING_SLOTS}, got {ring_slots!r}")
        self.batch_rows = batch_rows  # validated by the setter
        self.last_groups: List[Tuple[int, int]] = []  # (rows, padded length) of every layer call of the last pass
        self.weights = weights
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.ring_slots = int(ring_slots)
        self.staging_threads = max(1, int(staging_threads))
        self.layouts: List[LayerLayout] = list(weights.layouts)
        self.slot_bytes = slot_bytes(self.layouts)
        # staging: a pinned entry between the host form and the device slot. Needed on CUDA unless the host form is
        # page-locked itself; on the CPU only when a test forces it (the slot is then filled from the staging entry).
        self.staging = (self.cuda and weights.host != "pinned") if staging is None else bool(staging)
        self.entry_bytes = staging_entry_bytes(self.layouts) if self.staging else 0
        with torch.device("meta"):
            shell = MossQwen35TextEncoder(weights.config)
        shell.requires_grad_(False)
        shell.eval()
        shell.norm._parameters["weight"] = weights.final_norm_weight.to(self.device)
        self.shell = shell
        self._binds: List[List[Tuple[torch.nn.Module, str, object]]] = []
        self._meta: List[List[torch.Tensor]] = []
        for layer, layout in zip(shell.layers, self.layouts):
            binds, meta = [], []
            for entry in layout.entries:
                module_name, _, pname = entry.name.rpartition(".")
                module = layer.get_submodule(module_name) if module_name else layer
                binds.append((module, pname, entry))
                meta.append(module._parameters[pname])
            self._binds.append(binds)
            self._meta.append(meta)
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._staging_buffer: Optional[torch.Tensor] = None
        self._copy_stream = None
        self._uploads_done = _DONE  # CUDA: the previous pass's last H2D copy out of the staging entries
        self.stats: Dict[str, float] = {}
        self.passes = 0

    def __reduce__(self):
        raise TypeError("LiveTextEncoder is never pickled (DataLoader workers must not get the text encoder)")

    @property
    def config(self):
        return self.weights.config

    @property
    def batch_rows(self) -> int:
        return self._batch_rows

    @batch_rows.setter
    def batch_rows(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"batch_rows (--moss_te_live_batch) must be an integer >= 1, got {value!r}")
        self._batch_rows = int(value)

    # ---- binding the shell to a slot ----
    def _bind(self, index: int, slot: torch.Tensor) -> None:
        layout, dtype = self.layouts[index], self.weights.dtype
        for module, pname, entry in self._binds[index]:
            module._parameters[pname] = layout.view(slot, entry, dtype)

    def _unbind(self, index: int) -> None:
        for (module, pname, _), meta in zip(self._binds[index], self._meta[index]):
            module._parameters[pname] = meta

    def bound_layers(self) -> List[int]:
        """Layers whose parameters are not meta (none outside a pass)."""
        return [i for i, binds in enumerate(self._binds)
                if any(module._parameters[pname].device.type != "meta" for module, pname, _ in binds)]

    # ---- staging and uploads ----
    def _stager(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._executor is None:
            # intra-op threads of the memcpy into the staging entry, set in the stager thread itself (thread-local, as
            # the host optimizer worker does): the main thread keeps its own count
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="moss-te-stage", initializer=torch.set_num_threads,
                initargs=(self.staging_threads,))
        return self._executor

    def _staging_entry(self, s: int) -> torch.Tensor:
        if self._staging_buffer is None:
            self._staging_buffer = torch.empty(self.ring_slots * self.entry_bytes, dtype=torch.uint8,
                                               pin_memory=self.cuda)
        return self._staging_buffer[s * self.entry_bytes: (s + 1) * self.entry_bytes]

    def _fill(self, index: int, target: torch.Tensor) -> None:
        """Layer `index` in its layout into the uint8 `target` (a staging entry or a CPU slot)."""
        layout = self.layouts[index]
        flat = self.weights.layer_flat(index)
        if flat is not None:
            target[: layout.nbytes].copy_(flat)
            return
        for entry, source in self.weights.layer_sources(index):  # mmap: the file's tensors, cast to the layout dtype
            layout.view(target, entry, self.weights.dtype).copy_(source)

    def _stage(self, index: int, s: int, upload_done) -> None:
        upload_done.synchronize()  # the H2D copy of layer index - R from this entry has finished
        self._fill(index, self._staging_entry(s))

    # ---- encoding ----
    def encode_rows(self, rows: List[torch.Tensor]) -> List[torch.Tensor]:
        """[ids [T_i]] -> [hidden [T_i, hidden_size]] in the weights' dtype on the device: each row the cached
        encoder's MossQwen35TextEncoder.forward(ids[None])[0], bit for bit (batch_rows 1); with batch_rows > 1 the rows
        of the batched layer calls, close to it (see the header)."""
        if not rows:
            return []
        for ids in rows:
            if ids.numel() < 1:
                raise ValueError("every MOSS prompt has at least the 10 chat-template tokens")
        start = time.perf_counter()
        reserved = torch.cuda.memory_reserved(self.device) if self.cuda else 0
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
            if self.batch_rows > 1:
                out = self._encode_groups(rows)
            else:
                xs = [self.weights.embed_rows(ids).to(self.device).unsqueeze(0) for ids in rows]
                rotary = {}
                for x in xs:
                    if x.shape[1] not in rotary:
                        rotary[x.shape[1]] = self.shell._rotary(x.shape[1], x.device, x.dtype)
                self._run_layers(xs, rotary)
                out = [self.shell.norm(x)[0] for x in xs]
                self.last_groups = [(1, int(ids.numel())) for ids in rows]
        self.passes += 1
        self.stats = {"seconds": time.perf_counter() - start, "rows": float(len(rows)),
                      "tokens": float(sum(int(ids.numel()) for ids in rows)),
                      "calls": float(len(self.last_groups)),
                      "padded_tokens": float(sum(count * width for count, width in self.last_groups))}
        if self.cuda:  # pragma: no cover - CUDA only: did the pass grow the caching allocator (G2 smoke check)?
            self.stats.update(reserved_before=float(reserved),
                              reserved_after=float(torch.cuda.memory_reserved(self.device)))
        return out

    def _encode_groups(self, rows: List[torch.Tensor]) -> List[torch.Tensor]:
        """batch_rows > 1: the rows in length-sorted groups, each one [G, L, H] tensor through every layer (one call per
        layer and group), then every row cut back to its own length, in the caller's order."""
        lengths = [int(ids.numel()) for ids in rows]
        groups = length_groups(lengths, self.batch_rows)
        xs, rotary = [], {}
        for group in groups:
            ids = pad_group([rows[i] for i in group])  # each row padded with its own last id
            x = self.weights.embed_rows(ids).view(ids.shape[0], ids.shape[1], -1).to(self.device)
            xs.append(x)
            if x.shape[1] not in rotary:  # [1, L, rot]: broadcast over the group; its prefix = each row's own table
                rotary[x.shape[1]] = self.shell._rotary(x.shape[1], x.device, x.dtype)
        self._run_layers(xs, rotary)
        out: List[Optional[torch.Tensor]] = [None] * len(rows)
        for group, x in zip(groups, xs):
            hidden = self.shell.norm(x)  # per token: the pad positions do not touch the real ones
            for j, i in enumerate(group):
                out[i] = hidden[j, : lengths[i]]
        self.last_groups = [(len(group), int(x.shape[1])) for group, x in zip(groups, xs)]
        return out

    def _run_layers(self, xs: List[torch.Tensor], rotary) -> None:
        count, ring_slots, cuda = len(self.layouts), self.ring_slots, self.cuda
        # R separate slot blocks: smaller requests than one contiguous ring, easier to find free again next pass
        slots = [torch.empty(self.slot_bytes, dtype=torch.uint8, device=self.device) for _ in range(ring_slots)]
        staged: Dict[int, concurrent.futures.Future] = {}
        ready: List[object] = [None] * count
        freed: List[object] = [None] * count
        if cuda and self._copy_stream is None:  # pragma: no cover - CUDA only
            self._copy_stream = torch.cuda.Stream(device=self.device)
        compute = torch.cuda.current_stream(self.device) if cuda else None
        if cuda:  # pragma: no cover - CUDA only
            # the slots were allocated on the compute stream: their blocks may still be in use by kernels queued there
            # (the caching allocator only orders reuse within that stream), so the copy stream starts after them
            self._copy_stream.wait_stream(compute)

        def issue(k: int) -> None:
            s = k % ring_slots
            nbytes = self.layouts[k].nbytes
            if self.staging:
                staged.pop(k).result()  # raises the stager's error here, in the main thread
                source = self._staging_entry(s)[:nbytes]
            else:
                source = self.weights.layer_flat(k)
            if cuda:  # pragma: no cover - CUDA only
                if k >= ring_slots:
                    self._copy_stream.wait_event(freed[k - ring_slots])  # layer k - R is done with slot s
                with torch.cuda.stream(self._copy_stream):
                    if source is None:  # mmap without staging cannot happen on CUDA (staging is on)
                        raise RuntimeError("live text encoder: an mmap host form needs a staging entry on CUDA")
                    slots[s][:nbytes].copy_(source, non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(self._copy_stream)
                ready[k] = event
            else:
                if source is None:
                    self._fill(k, slots[s])
                else:
                    slots[s][:nbytes].copy_(source)
                ready[k] = _DONE
            if self.staging and k + ring_slots < count:
                staged[k + ring_slots] = self._stager().submit(self._stage, k + ring_slots, s, ready[k])

        try:
            if self.staging:
                # staging entry k was the source of the previous pass's last uploads: wait for them (on the CPU and in
                # the first pass: _DONE)
                for k in range(min(ring_slots, count)):
                    staged[k] = self._stager().submit(self._stage, k, k, self._uploads_done)
            issued = 0
            for index in range(count):
                while issued < min(count, index + ring_slots):  # layers up to index + R - 1 are uploading
                    issue(issued)
                    issued += 1
                if cuda:  # pragma: no cover - CUDA only
                    compute.wait_event(ready[index])
                self._bind(index, slots[index % ring_slots])
                try:
                    layer = self.shell.layers[index]
                    for i, x in enumerate(xs):
                        xs[i] = layer(x, *rotary[x.shape[1]])
                finally:
                    self._unbind(index)
                if cuda:  # pragma: no cover - CUDA only
                    event = torch.cuda.Event()
                    event.record(compute)
                    freed[index] = event
                else:
                    freed[index] = _DONE
        finally:
            for future in staged.values():
                future.cancel()
            for future in staged.values():
                if not future.cancelled():
                    try:
                        future.result()
                    except BaseException:  # noqa: BLE001 - the pass already failed or is being torn down
                        pass
            if cuda:  # pragma: no cover - CUDA only
                done = torch.cuda.Event()
                done.record(self._copy_stream)  # after every H2D copy of this pass: the staging entries are free
                self._uploads_done = done
                compute.wait_stream(self._copy_stream)  # every slot copy is ordered before the slots' reuse
            del slots

    def encode_tokens(self, tokens: List[torch.Tensor]) -> List[torch.Tensor]:
        """MossTextEncodingStrategy.encode_tokens' contract: [input_ids [B, L], mask [B, L]] -> [text_embeds [B, L,
        hidden] (zero at padding), text_mask [B, L] long]."""
        input_ids, mask = tokens[0], tokens[1]
        rows = []
        for i in range(input_ids.shape[0]):
            length = int(mask[i].sum())
            if length < 1:
                raise ValueError("every MOSS prompt has at least the 10 chat-template tokens")
            rows.append(input_ids[i, :length])
        return list(pad_rows(self.encode_rows(rows)))

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._staging_buffer = None
        self._copy_stream = None
        self._uploads_done = _DONE
        self.weights.close()
