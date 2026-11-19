# MOSS full fine-tune: a streaming safetensors writer and an in-place reader (docs/moss_full_finetune_design.md 4.7).
#
# The writer takes the tensor list as (key, dtype, shape, producer) entries: the header is built from the shapes and
# dtypes alone, then each producer is called once, its bytes written and the tensor dropped - the peak is one converted
# tensor (<= 226 MB for MOSS), and a pool view (fp32, contiguous) is written without a copy. The header is compact JSON,
# "__metadata__" first, the tensors in entry order with no gaps, padded with spaces so that the data starts at a
# multiple of 256 bytes (library/safetensors_utils.mem_eff_save_file's alignment). Files are fsync'ed when asked.
#
# The reader parses and checks the header and reads a tensor's bytes straight into a given contiguous CPU tensor (the
# host pool's views: "load in place", no temporary) or into a new one. Plain file reads, no mmap: a state directory can
# be renamed or deleted on Windows right after it was read.

import json
import math
import os
import struct
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

HEADER_ALIGN = 256
DTYPE_NAMES = {torch.float64: "F64", torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
               torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8",
               torch.bool: "BOOL"}
NAME_DTYPES = {name: dtype for dtype, name in DTYPE_NAMES.items()}
_MAX_HEADER = 100 << 20  # a header larger than this is not a safetensors file
_READ_BLOCK = 1 << 28  # readinto in blocks: one raw read of several GB can come back short on Windows


class SafetensorsError(ValueError):
    """A safetensors file that is not what the reader expects (truncated, bad header, other dtype / shape)."""


@dataclass(frozen=True)
class TensorEntry:
    """One tensor to write: produce() returns a CPU tensor of exactly this dtype and shape (contiguous or not)."""
    key: str
    dtype: torch.dtype
    shape: Tuple[int, ...]
    produce: Callable[[], torch.Tensor]

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * torch.empty((), dtype=self.dtype).element_size()


@dataclass(frozen=True)
class TensorInfo:
    key: str
    dtype: torch.dtype
    shape: Tuple[int, ...]
    begin: int  # offsets in the data section
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


def header_bytes(entries: Sequence[TensorEntry], metadata: Optional[Dict[str, str]] = None,
                 align: int = HEADER_ALIGN) -> bytes:
    """The 8-byte little-endian length and the padded JSON header for `entries` laid out back to back."""
    header: Dict[str, object] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    offset, seen = 0, set()
    for entry in entries:
        if entry.key in seen or entry.key == "__metadata__":
            raise ValueError(f"safetensors: key {entry.key!r} twice (or reserved)")
        seen.add(entry.key)
        if entry.dtype not in DTYPE_NAMES:
            raise ValueError(f"safetensors: {entry.key}: dtype {entry.dtype} has no safetensors name")
        size = entry.nbytes
        header[entry.key] = {"dtype": DTYPE_NAMES[entry.dtype], "shape": [int(d) for d in entry.shape],
                             "data_offsets": [offset, offset + size]}
        offset += size
    text = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    text += b" " * (-(len(text) + 8) % align)
    return struct.pack("<Q", len(text)) + text


def fsync_dir(path: str) -> None:
    """Durable directory entries (POSIX). Windows cannot open a directory for fsync: NTFS journals the metadata."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_safetensors(path: str, entries: Sequence[TensorEntry], metadata: Optional[Dict[str, str]] = None, *,
                      fsync: bool = True, align: int = HEADER_ALIGN) -> int:
    """Write `entries` to `path` (overwritten), one produced tensor at a time. Returns the file size. A producer that
    returns another dtype, shape or a non-CPU tensor raises ValueError (the file is then incomplete: write to a
    temporary name and rename, as the callers do)."""
    head = header_bytes(entries, metadata, align)
    total = len(head) + sum(entry.nbytes for entry in entries)
    with open(path, "wb") as f:
        f.write(head)
        for entry in entries:
            tensor = entry.produce()
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != entry.dtype or \
                    tuple(tensor.shape) != tuple(entry.shape) or tensor.device.type != "cpu":
                got = (getattr(tensor, "dtype", None), tuple(getattr(tensor, "shape", ())), getattr(tensor, "device", None))
                raise ValueError(f"safetensors: {entry.key}: the producer returned {got}, the header says "
                                 f"({entry.dtype}, {tuple(entry.shape)}, cpu)")
            if tensor.numel():
                data = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
                f.write(memoryview(data))
            del tensor
        f.flush()
        if fsync:
            os.fsync(f.fileno())
        size = f.tell()
    if size != total:
        raise ValueError(f"safetensors: wrote {size} bytes to {path}, expected {total}")
    return size


class SafetensorsReader:
    """Header + plain reads of one safetensors file. Use as a context manager (the file stays open inside)."""

    def __init__(self, path: str):
        self.path = str(path)
        self.file_size = os.path.getsize(self.path)
        self._f = open(self.path, "rb")
        try:
            self.metadata, self.infos, self.data_start = self._parse()
        except BaseException:
            self._f.close()
            raise

    def __enter__(self) -> "SafetensorsReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._f.close()

    def _fail(self, why: str):
        raise SafetensorsError(f"{self.path}: {why}")

    def _parse(self):
        raw = self._f.read(8)
        if len(raw) != 8:
            self._fail("shorter than a safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if length > _MAX_HEADER or 8 + length > self.file_size:
            self._fail(f"header length {length} does not fit the file ({self.file_size} bytes): truncated or not"
                       " safetensors")
        try:
            header = json.loads(self._f.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            self._fail(f"unreadable header ({error})")
        if not isinstance(header, dict):
            self._fail("the header is not a JSON object")
        metadata = header.pop("__metadata__", None) or {}
        data_start = 8 + length
        data_size = self.file_size - data_start
        infos: Dict[str, TensorInfo] = {}
        for key, item in header.items():
            try:
                dtype = NAME_DTYPES[item["dtype"]]
                shape = tuple(int(d) for d in item["shape"])
                begin, end = (int(v) for v in item["data_offsets"])
            except (KeyError, TypeError, ValueError):
                self._fail(f"bad header entry for {key!r}: {item!r}")
            size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            if any(d < 0 for d in shape) or not 0 <= begin <= end or end - begin != size:
                self._fail(f"{key}: data_offsets {[begin, end]} do not hold {item['dtype']} {list(shape)}")
            if end > data_size:
                self._fail(f"{key} ends at byte {end} of the data, the file has {data_size}: truncated")
            infos[key] = TensorInfo(key, dtype, shape, begin, end)
        return metadata, infos, data_start

    def keys(self) -> List[str]:
        return list(self.infos)

    def info(self, key: str) -> TensorInfo:
        try:
            return self.infos[key]
        except KeyError:
            self._fail(f"has no tensor {key!r}")

    def read(self, key: str, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The tensor `key`. With `out` (a contiguous CPU tensor of the same dtype and element count, e.g. a flat pool
        view) its bytes are read straight into it and `out` is returned."""
        info = self.info(key)
        if out is None:
            out = torch.empty(info.shape, dtype=info.dtype)
        elif (out.dtype != info.dtype or out.numel() != info.numel or out.device.type != "cpu"
              or not out.is_contiguous()):
            self._fail(f"{key} is {DTYPE_NAMES[info.dtype]} {list(info.shape)}; it cannot be read into a "
                       f"{out.dtype} tensor of {out.numel()} elements on {out.device}"
                       f"{'' if out.is_contiguous() else ' (not contiguous)'}")
        if info.nbytes:
            view = memoryview(out.detach().reshape(-1).view(torch.uint8).numpy())
            self._f.seek(self.data_start + info.begin)
            done = 0
            while done < info.nbytes:
                n = self._f.readinto(view[done:min(info.nbytes, done + _READ_BLOCK)])
                if not n:
                    self._fail(f"{key}: read {done} of {info.nbytes} bytes: truncated")
                done += n
        return out


def read_header(path: str) -> Tuple[Dict[str, str], Dict[str, TensorInfo]]:
    """(metadata, {key: TensorInfo}) of a safetensors file; SafetensorsError when it is not readable."""
    with SafetensorsReader(path) as reader:
        return dict(reader.metadata), dict(reader.infos)

