"""CPU tests: latent caches are read selectively (review 2026-11-05, #07).

One latent NPZ holds every resolution (and augmentation variant) of an image. The validity check and the loader used
``load_npz``, which decompresses every array before picking one key. They now open the file with
``cache_utils.open_npz`` and decode only the entries they use: the check reads the key list and small metadata, the
loader one resolution / variant. What they return must be exactly what the full read returned.

Only temporary folders and small fake arrays; no model, no GPU.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp=<scratch dir>
     tests/test_latent_npz_selective_read.py
"""
import contextlib
import itertools

import numpy as np
import pytest
import torch

from library import strategy_base
from library.cache_utils import CACHE_DTYPE_METADATA_KEY, load_npz, open_npz, save_npz
from library.strategy_anima import AnimaLatentsCachingStrategy
from library.strategy_base import compute_aug_config_hash, variant_key

RESOS = [(256, 256), (384, 256), (512, 512), (768, 512)]  # bucket (W, H); one file holds all of them
HASH = compute_aug_config_hash({"color_aug": True})


def _suffix(reso):
    return f"_{reso[1] // 8}x{reso[0] // 8}"


def _rand(shape, seed):
    return torch.from_numpy(np.random.default_rng(seed).standard_normal(shape).astype(np.float32))


def _latent(reso, seed):
    return _rand((16, 1, reso[1] // 8, reso[0] // 8), seed)


def _write_new_layout(strategy, path, image_path):
    """Variants at two resolutions (one with alpha masks), a flip cache and a plain cache at the others."""
    for i, reso in enumerate(RESOS):
        crop = (0, 0, reso[0], reso[1])
        if i % 2 == 0:
            variants = [
                {"latents": _latent(reso, 100 * i + k), "crop_ltrb": (k, 0, reso[0] - k, reso[1]), "flipped": bool(k % 2),
                 "alpha_mask": _rand((reso[1], reso[0]), 50 + k) if i == 2 else None}
                for k in range(1, 4)
            ]
            strategy.save_latents_to_disk(
                path, _latent(reso, i), reso, crop, None, _rand((reso[1], reso[0]), 7) if i == 2 else None, _suffix(reso),
                aug_variants=variants, aug_config_hash=HASH, source_path=image_path,
            )
        else:
            flipped = _latent(reso, 900 + i) if i == 1 else None
            strategy.save_latents_to_disk(path, _latent(reso, i), reso, crop, flipped, None, _suffix(reso), source_path=image_path)


def _write_old_layout(path, cache_dtype):
    """As the writer stored variant caches before 2026-11-05: one file-wide aug_variants / aug_config_hash pair."""
    arrays = {}
    for i, reso in enumerate(RESOS[:3]):
        s = _suffix(reso)
        arrays["latents" + s] = _latent(reso, i)
        arrays["original_size" + s] = np.array(reso)
        arrays["crop_ltrb" + s] = np.array((0, 0, reso[0], reso[1]))
        for k in range(1, 3):
            arrays[variant_key("latents", k) + s] = _latent(reso, 10 * i + k)
            arrays[variant_key("original_size", k) + s] = np.array(reso)
            arrays[variant_key("crop_ltrb", k) + s] = np.array((k, k, reso[0], reso[1]))
            arrays[variant_key("flipped", k) + s] = np.array(k == 1)
    arrays["aug_variants"] = np.array(3)
    arrays["aug_config_hash"] = np.array(HASH)
    save_npz(path, arrays, cache_dtype=cache_dtype)


def _write_unsuffixed(path, cache_dtype):
    """The oldest layout: one resolution, no suffix (the loader falls back to it)."""
    save_npz(path, {"latents": _latent(RESOS[0], 3), "latents_flipped": _latent(RESOS[0], 4), "original_size": np.array([256, 256]),
                    "crop_ltrb": np.array([0, 0, 256, 256]), "alpha_mask": _rand((256, 256), 5)}, cache_dtype=cache_dtype)


@pytest.fixture(params=["auto", "fp16", "bf16"])
def cache_files(request, tmp_path):
    cache_dtype = request.param
    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"not really a png")
    strategy = AnimaLatentsCachingStrategy(True, 1, False, cache_dtype=cache_dtype)
    new = str(tmp_path / "new_0768x0512_anima.npz")
    old = str(tmp_path / "old_0768x0512_anima.npz")
    unsuffixed = str(tmp_path / "unsuffixed_0256x0256_anima.npz")
    _write_new_layout(strategy, new, str(image_path))
    _write_old_layout(old, cache_dtype)
    _write_unsuffixed(unsuffixed, cache_dtype)
    return strategy, {"new": new, "old": old, "unsuffixed": unsuffixed}


@contextlib.contextmanager
def _full_read(path):
    yield load_npz(path)  # what both functions read before: every array decoded


def _same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
                and a.tobytes() == b.tobytes())
    return type(a) is type(b) and a == b


def _load_all(strategy, files):
    results = {}
    for name, path in files.items():
        for reso, variant in itertools.product(RESOS, range(4)):
            try:
                results[(name, reso, variant)] = strategy.load_latents_from_disk(path, reso, variant=variant)
            except (ValueError, KeyError) as e:  # a variant / resolution that is not in that file
                results[(name, reso, variant)] = type(e).__name__
    return results


def _check_all(strategy, files):
    results = {}
    for name, path in files.items():
        for reso, flip, alpha, count, config_hash in itertools.product(RESOS, (False, True), (False, True), (0, 3, 4), (HASH, "x")):
            results[(name, reso, flip, alpha, count, config_hash)] = strategy.is_disk_cached_latents_expected(
                reso, path, flip, alpha, num_aug_variants=count, aug_config_hash=config_hash
            )
    return results


def test_selective_load_returns_exactly_what_the_full_read_returned(cache_files, monkeypatch):
    strategy, files = cache_files
    selective = _load_all(strategy, files)
    monkeypatch.setattr(strategy_base, "open_npz", _full_read)
    full = _load_all(strategy, files)

    assert selective.keys() == full.keys()
    loaded = 0
    for key in full:
        if isinstance(full[key], str):
            assert selective[key] == full[key], key
            continue
        loaded += 1
        assert len(selective[key]) == len(full[key]) == 6
        for got, expected in zip(selective[key], full[key]):
            assert _same(got, expected), key
    assert loaded >= 20  # canonical and variant latents of every layout were compared


def test_selective_check_gives_the_same_verdicts_as_the_full_read(cache_files, monkeypatch):
    strategy, files = cache_files
    selective = _check_all(strategy, files)
    monkeypatch.setattr(strategy_base, "open_npz", _full_read)
    assert selective == _check_all(strategy, files)
    assert any(selective.values()) and not all(selective.values())


def test_open_npz_decodes_every_entry_like_load_npz(cache_files):
    _strategy, files = cache_files
    for path in files.values():
        full = load_npz(path)
        with open_npz(path) as npz:
            assert sorted(npz.files) == sorted(full)
            assert CACHE_DTYPE_METADATA_KEY not in npz
            for key in full:
                assert key in npz and _same(npz[key], full[key]), key
            with pytest.raises(KeyError):
                npz["no_such_key"]


@pytest.fixture
def read_keys(monkeypatch, tmp_path):
    """Records the members np.load decompresses."""
    probe = tmp_path / "probe.npz"
    np.savez(probe, a=np.zeros(1))
    with np.load(probe) as archive:
        npz_class = type(archive)
    original = npz_class.__getitem__
    keys = []

    def spy(self, key):
        keys.append(key)
        return original(self, key)

    monkeypatch.setattr(npz_class, "__getitem__", spy)
    return keys


def test_check_and_load_decompress_only_what_they_use(cache_files, read_keys):
    strategy, files = cache_files
    path = files["new"]
    reso = RESOS[2]  # variants with alpha masks
    assert strategy.is_disk_cached_latents_expected(reso, path, True, True, num_aug_variants=4, aug_config_hash=HASH)
    assert read_keys, "the check reads its metadata through np.load"
    assert not [k for k in read_keys if k.startswith(("latents", "alpha_mask", "crop_ltrb", "original_size"))], read_keys

    read_keys.clear()
    strategy.load_latents_from_disk(path, reso, variant=2)
    suffix = _suffix(reso)
    assert read_keys
    assert all(k.endswith(suffix) or k == CACHE_DTYPE_METADATA_KEY for k in read_keys), read_keys
    assert not [k for k in read_keys if k.startswith("latents") and k != variant_key("latents", 2) + suffix], read_keys
