# MOSS full fine-tune, live text encoder (--moss_te_live): the host form of the frozen Qwen3.5-9B
# (docs/moss_ft_te_live_fp16_plan.md 1.2, 1.5).
#
# The 32 decoder layers live in host RAM, each as one flat buffer in a fixed layout (LayerLayout: every tensor at a
# multiple of 256 bytes, the alignment class of a standalone tensor), so a layer crosses to the GPU as one copy and its
# tensors are views of the device slot. Host modes:
#   copy    one pageable flat buffer per layer (12.89 GiB private for the 9B), filled from the file
#   pinned  all layers in one pageable allocation registered with cudaHostRegister (page-locked, not rounded to a power
#           of two like torch's pinned allocator); the H2D copy reads it directly (CUDA only)
#   mmap    no private copy: the layer tensors are views of the memory-mapped file (needs the OS file cache)
# The embedding table (1.89 GiB) is never copied: rows are gathered on the CPU from the mapped file (a gather copies
# values, so the rows are exactly the embedding lookup of the loaded model). Values are cast to bf16 with the same
# round-to-nearest-even cast as load_moss_text_encoder's .to(dtype) (a tensor stored in fp32, e.g. A_log).
# The object is never pickled into a DataLoader worker (__reduce__ raises).

from __future__ import annotations

import functools
import math
import mmap
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .ft_safetensors import SafetensorsReader
from .qwen35 import (MossQwen35Config, MossQwen35TextEncoder, check_moss_text_encoder, detect_text_encoder_prefix,
                     is_language_model_key)

ALIGN = 256  # bytes: every tensor of a layer layout starts at a multiple of this
HOST_MODES = ("copy", "pinned", "mmap")
DEFAULT_CONFIG = MossQwen35Config()  # the 9B; tests patch a tiny one (and check_moss_text_encoder)


def _align(n: int) -> int:
    return -(-int(n) // ALIGN) * ALIGN


@dataclass(frozen=True)
class LayoutEntry:
    name: str  # module-relative to layers.{i} (e.g. "linear_attn.in_proj_qkv.weight")
    offset: int  # bytes from the start of the layer buffer, a multiple of ALIGN
    shape: Tuple[int, ...]

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class LayerLayout:
    kind: str  # "linear" (GatedDeltaNet) | "full" (gated attention)
    entries: Tuple[LayoutEntry, ...]
    nbytes: int  # the aligned buffer size

    def view(self, buffer: torch.Tensor, entry: LayoutEntry, dtype: torch.dtype) -> torch.Tensor:
        """`entry`'s tensor inside a uint8 `buffer` holding this layer (a view, no copy)."""
        size = entry.numel * torch.empty((), dtype=dtype).element_size()
        return buffer[entry.offset: entry.offset + size].view(dtype).view(entry.shape)


@functools.lru_cache(maxsize=8)
def _layouts(config: MossQwen35Config, element_size: int) -> Tuple[LayerLayout, ...]:
    with torch.device("meta"):
        model = MossQwen35TextEncoder(config)
    out = []
    for layer in model.layers:
        offset, entries = 0, []
        for name, param in layer.named_parameters():
            entries.append(LayoutEntry(name, offset, tuple(int(d) for d in param.shape)))
            offset = _align(offset + param.numel() * element_size)
        out.append(LayerLayout("full" if layer.full_attention else "linear", tuple(entries), offset))
    return tuple(out)


def layer_layouts(config: MossQwen35Config = DEFAULT_CONFIG, dtype: torch.dtype = torch.bfloat16) -> List[LayerLayout]:
    """The aligned layout of every decoder layer (from a meta-built model: no weights)."""
    return list(_layouts(config, torch.empty((), dtype=dtype).element_size()))


def slot_bytes(layouts: List[LayerLayout]) -> int:
    """One device slot: the largest layer buffer."""
    return max(layout.nbytes for layout in layouts)


def staging_entry_bytes(layouts: List[LayerLayout]) -> int:
    """A pinned staging entry: the slot rounded up to a power of two, so R entries (R = 2 or 4) are one power-of-two
    pinned allocation that torch's pinned allocator does not round further (512 MiB for the 9B)."""
    return 1 << (slot_bytes(layouts) - 1).bit_length()


def _file_view(buffer, dtype: torch.dtype, offset: int, shape: Tuple[int, ...]) -> torch.Tensor:
    """A zero-copy tensor over `buffer[offset:]`. Through numpy on purpose: its array keeps the buffer exported, so the
    mapping cannot be closed under a live view (mmap.close raises BufferError instead of unmapping)."""
    nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if nbytes == 0:
        return torch.empty(shape, dtype=dtype)
    with warnings.catch_warnings():  # a read-only mapping: torch warns that the array is not writable (never written)
        warnings.simplefilter("ignore")
        raw = torch.from_numpy(np.frombuffer(buffer, dtype=np.uint8, count=nbytes, offset=offset))
    return raw.view(dtype).view(shape)


class HostTextEncoderWeights:
    """The frozen text encoder in host RAM. Build with open(); close() releases the buffers and the mapping."""

    def __init__(self):  # use open()
        self.path = ""
        self.config: MossQwen35Config = DEFAULT_CONFIG
        self.host = "copy"
        self.dtype = torch.bfloat16
        self.layouts: List[LayerLayout] = []
        self._mm: Optional[mmap.mmap] = None
        self._embed: Optional[torch.Tensor] = None
        self._layer_files: List[List[torch.Tensor]] = []  # mmap mode: the file view of each entry, per layer
        self._flats: List[torch.Tensor] = []  # copy / pinned: one uint8 buffer per layer (views into one in pinned)
        self._pinned_base: Optional[torch.Tensor] = None
        self._registered = False
        self.final_norm_weight: Optional[torch.Tensor] = None

    def __reduce__(self):
        raise TypeError("HostTextEncoderWeights is never pickled (DataLoader workers must not get the text encoder)")

    @classmethod
    def open(cls, path: str, config: Optional[MossQwen35Config] = None, host: str = "copy",
             dtype: torch.dtype = torch.bfloat16, check: bool = True) -> "HostTextEncoderWeights":
        """Read the language model of `path` into the host form `host`. check: the 9B header check
        (check_moss_text_encoder); False (tests with a tiny config) only detects the key prefix. Every language-model
        key must be in the layout and of its shape (the strict load of load_moss_text_encoder)."""
        if host not in HOST_MODES:
            raise ValueError(f"text encoder host mode must be one of {HOST_MODES}, got {host!r}")
        if host == "pinned" and not torch.cuda.is_available():
            raise ValueError("text encoder host mode 'pinned' registers page-locked memory with CUDA: not available here")
        self = cls()
        self.path, self.config, self.host, self.dtype = str(path), config or DEFAULT_CONFIG, host, dtype
        self.layouts = layer_layouts(self.config, dtype)
        prefix = check_moss_text_encoder(self.path) if check else detect_text_encoder_prefix(self.path)
        with SafetensorsReader(self.path) as reader:
            infos, data_start = dict(reader.infos), reader.data_start
        expected = self._expected_shapes(prefix)
        present = {key for key in infos if is_language_model_key(prefix, key)}
        missing, unexpected = sorted(set(expected) - present), sorted(present - set(expected))
        if missing or unexpected:
            raise ValueError(f"{path}: not this text encoder's language model (missing {missing[:3]}"
                             f"{'...' if len(missing) > 3 else ''}, unexpected {unexpected[:3]}"
                             f"{'...' if len(unexpected) > 3 else ''})")
        for key, shape in expected.items():
            if tuple(infos[key].shape) != shape:
                raise ValueError(f"{path}: {key} is {list(infos[key].shape)}, expected {list(shape)}")
        with open(self.path, "rb") as f:
            self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            def view(key):
                info = infos[key]
                return _file_view(self._mm, info.dtype, data_start + info.begin, tuple(info.shape))

            self._embed = view(prefix + "embed_tokens.weight")
            self.final_norm_weight = view(prefix + "norm.weight").to(dtype).clone()
            files = [[view(f"{prefix}layers.{i}.{entry.name}") for entry in layout.entries]
                     for i, layout in enumerate(self.layouts)]
            if host == "mmap":
                self._layer_files = files
            else:
                self._fill(files)
        except BaseException:
            self.close()
            raise
        return self

    def _expected_shapes(self, prefix: str) -> Dict[str, Tuple[int, ...]]:
        c = self.config
        shapes = {prefix + "embed_tokens.weight": (c.vocab_size, c.hidden_size), prefix + "norm.weight": (c.hidden_size,)}
        for i, layout in enumerate(self.layouts):
            for entry in layout.entries:
                shapes[f"{prefix}layers.{i}.{entry.name}"] = entry.shape
        return shapes

    def _fill(self, files: List[List[torch.Tensor]]) -> None:
        total = sum(layout.nbytes for layout in self.layouts)
        if self.host == "pinned":
            base = torch.empty(total, dtype=torch.uint8)
            offset = 0
            for layout in self.layouts:
                self._flats.append(base[offset: offset + layout.nbytes])
                offset += layout.nbytes
        else:
            base = None
            self._flats = [torch.empty(layout.nbytes, dtype=torch.uint8) for layout in self.layouts]
        for flat, layout, sources in zip(self._flats, self.layouts, files):
            flat.zero_()  # the alignment gaps (never read; zeroed so a layer buffer is deterministic)
            for entry, source in zip(layout.entries, sources):
                layout.view(flat, entry, self.dtype).copy_(source)  # RNE cast of a stored fp32 tensor
        if base is not None:  # page-lock in place (pragma: CUDA only)
            result = torch.cuda.cudart().cudaHostRegister(base.data_ptr(), total, 0)
            if int(getattr(result, "value", result)) != 0:
                raise RuntimeError(f"cudaHostRegister of {total / (1 << 30):.2f} GiB failed (error {result}): use"
                                   " --moss_te_live_host copy")
            self._pinned_base, self._registered = base, True

    # ---- what the engine reads ----
    @property
    def num_layers(self) -> int:
        return len(self.layouts)

    def layer_flat(self, index: int) -> Optional[torch.Tensor]:
        """copy / pinned: the uint8 buffer of layer `index` in its layout (one copy to a device slot); mmap: None."""
        return self._flats[index] if self._flats else None

    def layer_sources(self, index: int) -> List[Tuple[LayoutEntry, torch.Tensor]]:
        """(entry, CPU tensor) of every tensor of layer `index`: views of its buffer (copy / pinned) or of the file
        (mmap; in the file's dtype, cast when copied into a slot)."""
        layout = self.layouts[index]
        if self._flats:
            flat = self._flats[index]
            return [(entry, layout.view(flat, entry, self.dtype)) for entry in layout.entries]
        return list(zip(layout.entries, self._layer_files[index]))

    def embed_rows(self, ids: torch.Tensor) -> torch.Tensor:
        """[L] token ids -> [L, hidden] rows of the embedding table in self.dtype, on the CPU."""
        ids = ids.reshape(-1).to(device="cpu", dtype=torch.long)
        return self._embed.index_select(0, ids).to(self.dtype)

    @property
    def private_bytes(self) -> int:
        """Pageable host RAM this object owns (copy mode: the layer buffers)."""
        return sum(int(flat.numel()) for flat in self._flats) if self.host == "copy" else 0

    @property
    def locked_bytes(self) -> int:
        """Page-locked host RAM (pinned mode)."""
        return int(self._pinned_base.numel()) if self._pinned_base is not None else 0

    def close(self) -> None:
        if self._registered and self._pinned_base is not None:  # pragma: no cover - CUDA only
            try:
                torch.cuda.cudart().cudaHostUnregister(self._pinned_base.data_ptr())
            except Exception:  # noqa: BLE001 - closing
                pass
            self._registered = False
        self._flats, self._pinned_base, self._layer_files, self._embed = [], None, [], None
        if self._mm is not None:
            try:
                self._mm.close()
            except BufferError:  # a caller still holds a file view: the mapping closes when it is collected
                pass
            self._mm = None


def te_layer_bytes(config: MossQwen35Config = DEFAULT_CONFIG) -> List[int]:
    """The aligned bf16 size of every decoder layer (ft_budget)."""
    return [layout.nbytes for layout in layer_layouts(config)]


def te_embed_bytes(config: MossQwen35Config = DEFAULT_CONFIG) -> int:
    return config.vocab_size * config.hidden_size * 2
