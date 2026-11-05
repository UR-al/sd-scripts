"""Utilities shared by disk caches.

NPZ supports compression transparently through NumPy, but it does not have a
native bfloat16 dtype on all supported NumPy versions.  BF16 cache entries are
therefore stored as their uint16 bit representation with per-entry metadata.

"""

import json
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch


CACHE_DTYPE_CHOICES = ("auto", "fp16", "bf16", "fp32")
CACHE_DTYPE_METADATA_KEY = "__cache_dtypes__"


def normalize_cache_dtype(cache_dtype: str) -> str:
    """Validate and normalize a cache floating-point dtype policy."""
    # Some downstream callers construct ``argparse.Namespace``-like objects
    # with MagicMock attributes in tests. Treat absent/non-string optional
    # values as the backwards-compatible default rather than rejecting them.
    if not isinstance(cache_dtype, str):
        return "auto"
    normalized = str(cache_dtype).lower()
    if normalized not in CACHE_DTYPE_CHOICES:
        raise ValueError(f"cache dtype must be one of {CACHE_DTYPE_CHOICES}, got: {cache_dtype}")
    return normalized


def _to_numpy(value: Any) -> Tuple[np.ndarray, Any]:
    """Return a CPU NumPy array and the original torch dtype, when available."""
    if isinstance(value, torch.Tensor):
        source_dtype = value.dtype
        tensor = value.detach().to("cpu")
        if source_dtype == torch.bfloat16:
            return tensor.float().numpy(), source_dtype
        return tensor.numpy(), source_dtype
    return np.asarray(value), None


def _choose_auto_dtype(array: np.ndarray, source_dtype: Any) -> str:
    """Choose a compact safe dtype for an array.

    Auto preserves the source precision for ordinary FP32/FP64 arrays.  It
    still preserves compact FP16/BF16 sources, and therefore never changes
    the numerical behavior of existing caches unless a compact policy is
    explicitly requested.
    """
    if source_dtype == torch.bfloat16:
        return "bf16"
    if source_dtype == torch.float16 or array.dtype == np.float16:
        return "fp16"
    return "fp32"


def encode_cache_array(value: Any, cache_dtype: str, key: str) -> Tuple[np.ndarray, str | None]:
    """Convert one cache value and return its stored dtype metadata."""
    policy = normalize_cache_dtype(cache_dtype)
    array, source_dtype = _to_numpy(value)
    if not np.issubdtype(array.dtype, np.floating):
        return array, None

    target = _choose_auto_dtype(array, source_dtype) if policy == "auto" else policy
    if target == "fp16":
        finite = np.isfinite(array)
        if np.any(finite & (np.abs(array) > np.finfo(np.float16).max)):
            if policy == "auto":
                target = "fp32"
            else:
                raise ValueError(f"{key} contains values outside the finite FP16 range")

    if target == "fp16":
        return array.astype(np.float16), target
    if target == "fp32":
        return array.astype(np.float32), target

    # NumPy cannot consistently represent BF16. Store the exact BF16 bits and
    # reconstruct them on load. The metadata tells the reader which entries
    # contain encoded BF16 values.
    contiguous = np.ascontiguousarray(array.astype(np.float32, copy=False))
    bf16 = torch.from_numpy(contiguous).to(dtype=torch.bfloat16)
    raw_bits = bf16.view(dtype=torch.uint16).numpy().copy()
    return raw_bits, target


def save_npz(path: str, arrays: Mapping[str, Any], cache_dtype: str = "auto", compress: bool = True) -> None:
    """Save an NPZ cache (zlib-compressed unless ``compress=False``), encoding floating arrays per policy.

    ``load_npz`` reads both forms; the stored values are identical either way.
    """
    policy = normalize_cache_dtype(cache_dtype)
    encoded: Dict[str, np.ndarray] = {}
    dtype_metadata: Dict[str, str] = {}
    for key, value in arrays.items():
        if key == CACHE_DTYPE_METADATA_KEY:
            continue
        encoded_value, actual_dtype = encode_cache_array(value, policy, key)
        encoded[key] = encoded_value
        if actual_dtype is not None:
            dtype_metadata[key] = actual_dtype

    if dtype_metadata:
        encoded[CACHE_DTYPE_METADATA_KEY] = np.array(json.dumps(dtype_metadata, sort_keys=True))
    if compress:
        np.savez_compressed(path, **encoded)
    else:
        np.savez(path, **encoded)


def _decode_bf16(array: np.ndarray) -> np.ndarray:
    raw_bits = np.ascontiguousarray(array.astype(np.uint16, copy=False))
    return torch.from_numpy(raw_bits).view(dtype=torch.bfloat16).float().numpy()


class NpzReader:
    """Read-only view of an NPZ cache that decompresses only the entries it is asked for.

    ``load_npz`` decodes every array of the file; one latent cache holds all resolutions and augmentation variants of
    an image, so a reader that needs one of them should open the file with ``open_npz`` instead. ``files``, ``in`` and
    ``keys()`` list the entries without reading them; ``reader[key]`` reads one, decoded exactly as ``load_npz``
    returns it (BF16 entries as float32). The dtype metadata entry is hidden, as in ``load_npz``. Close it (or use it
    as a context manager) before the file is rewritten.
    """

    def __init__(self, path: str):
        self._archive = np.load(path, allow_pickle=False)
        try:
            names = list(self._archive.files)
            self._dtypes: Dict[str, str] = {}
            if CACHE_DTYPE_METADATA_KEY in names:
                self._dtypes = json.loads(str(self._archive[CACHE_DTYPE_METADATA_KEY].tolist()))
        except BaseException:
            self._archive.close()
            raise
        self.files = [name for name in names if name != CACHE_DTYPE_METADATA_KEY]
        self._names = set(self.files)

    def __contains__(self, key: object) -> bool:
        return key in self._names

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._names:
            raise KeyError(f"{key} is not a file in the archive")
        value = self._archive[key]
        if self._dtypes.get(key) == "bf16":
            value = _decode_bf16(value)
        return value

    def __iter__(self):
        return iter(self.files)

    def __len__(self) -> int:
        return len(self.files)

    def keys(self):
        return list(self.files)

    def close(self) -> None:
        self._archive.close()

    def __enter__(self) -> "NpzReader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def open_npz(path: str) -> NpzReader:
    """Open an NPZ cache for selective reading (see ``NpzReader``); ``with open_npz(path) as npz: npz[key]``."""
    return NpzReader(path)


def load_npz(path: str, return_dtypes: bool = False) -> Dict[str, np.ndarray]:
    """Load compressed or uncompressed NPZ and decode cache BF16 entries.

    Args:
        path: Path to the npz file.
        return_dtypes: If True, also return the per-key stored dtype metadata
            as a second dict. This lets callers reconstruct compact BF16
            tensors: NumPy has no native BF16, so decoded BF16 entries are
            returned as widened fp32 arrays, and the metadata is the only way
            to know they originated as BF16.

    Returns:
        ``(data, dtypes)`` when ``return_dtypes`` is True, otherwise ``data``.
    """
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files if key != CACHE_DTYPE_METADATA_KEY}
        if CACHE_DTYPE_METADATA_KEY not in archive:
            return (data, {}) if return_dtypes else data
        metadata = json.loads(str(archive[CACHE_DTYPE_METADATA_KEY].tolist()))

    for key, dtype in metadata.items():
        if key in data and dtype == "bf16":
            data[key] = _decode_bf16(data[key])
    return (data, metadata) if return_dtypes else data
