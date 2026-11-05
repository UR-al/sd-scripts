"""CPU tests: the augmentation hash and variant count of a latent cache are per resolution (review 2026-11-05, #5).

One *_anima.npz (and every other multi-resolution latent cache) holds all bucket resolutions of an image, written at
different times. The augmentation config hash and the variant count used to be one pair for the whole file, so after
re-caching only 512 under augmentation config B, the 1024 latents (still built under A) passed the check under B.
Now each resolution carries its own ``aug_variants_<h>x<w>`` / ``aug_config_hash_<h>x<w>``; files with the old
file-wide pair are rebuilt once (never trusted, never a crash).

Only temporary folders and small fake arrays; no model, no GPU.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp=<scratch dir>
     tests/test_latent_cache_aug_per_resolution.py
"""
import numpy as np
import pytest
import torch

from library.cache_utils import load_npz, save_npz
from library.strategy_anima import AnimaLatentsCachingStrategy
from library.strategy_base import compute_aug_config_hash, variant_key

HASH_A = compute_aug_config_hash({"color_aug": True})
HASH_B = compute_aug_config_hash({"color_aug": True, "gamma_aug": True})
SMALL, LARGE = (256, 256), (512, 512)  # bucket resolutions (W, H); latents 32x32 and 64x64 (stride 8)


def _suffix(reso):
    return f"_{reso[1] // 8}x{reso[0] // 8}"


def _latent(reso, value):
    return torch.full((16, 1, reso[1] // 8, reso[0] // 8), float(value))


def _variants(reso, value, count):
    return [
        {"latents": _latent(reso, value + k), "crop_ltrb": (0, 0, reso[0], reso[1]), "flipped": bool(k % 2), "alpha_mask": None}
        for k in range(1, count)
    ]


def _save_variants(strategy, path, reso, value, count, config_hash):
    strategy.save_latents_to_disk(
        path, _latent(reso, value), reso, (0, 0, reso[0], reso[1]), key_reso_suffix=_suffix(reso),
        aug_variants=_variants(reso, value, count), aug_config_hash=config_hash,
    )


def _valid(strategy, path, reso, count, config_hash, flip_aug=False):
    return strategy.is_disk_cached_latents_expected(reso, path, flip_aug, False, num_aug_variants=count, aug_config_hash=config_hash)


@pytest.fixture
def strategy():
    return AnimaLatentsCachingStrategy(True, 1, False)


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "img_0800x0800_anima.npz")


def test_rewriting_one_resolution_does_not_validate_the_others(strategy, path):
    """The reviewer's case: 512 and 1024 under A, the config changes to B, only 512 is re-cached."""
    _save_variants(strategy, path, SMALL, 1.0, 2, HASH_A)
    _save_variants(strategy, path, LARGE, 1.0, 2, HASH_A)
    _save_variants(strategy, path, SMALL, 5.0, 2, HASH_B)

    assert _valid(strategy, path, SMALL, 2, HASH_B)
    assert not _valid(strategy, path, LARGE, 2, HASH_B), "the large latents were built under A"
    assert _valid(strategy, path, LARGE, 2, HASH_A)  # its own config is still recorded


def test_variant_count_is_per_resolution(strategy, path):
    _save_variants(strategy, path, SMALL, 1.0, 4, HASH_A)
    _save_variants(strategy, path, LARGE, 1.0, 2, HASH_A)
    assert _valid(strategy, path, SMALL, 4, HASH_A)
    assert not _valid(strategy, path, LARGE, 4, HASH_A), "only 2 variants were cached at the large resolution"
    assert _valid(strategy, path, LARGE, 2, HASH_A)

    # and the other way round: writing more variants at one resolution does not raise the count of another
    _save_variants(strategy, path, LARGE, 1.0, 6, HASH_A)
    assert _valid(strategy, path, LARGE, 6, HASH_A)
    assert not _valid(strategy, path, SMALL, 6, HASH_A)


def _old_layout_file(path, reso_values, count, config_hash, flip_legacy_reso=None):
    """A cache as the writer stored it before: per-resolution arrays, ONE aug_variants / aug_config_hash pair."""
    arrays = {}
    for reso, value in reso_values:
        s = _suffix(reso)
        arrays["latents" + s] = _latent(reso, value).numpy()
        arrays["original_size" + s] = np.array(reso)
        arrays["crop_ltrb" + s] = np.array((0, 0, reso[0], reso[1]))
        for k, var in enumerate(_variants(reso, value, count), start=1):
            arrays[variant_key("latents", k) + s] = var["latents"].numpy()
            arrays[variant_key("original_size", k) + s] = np.array(reso)
            arrays[variant_key("crop_ltrb", k) + s] = np.array(var["crop_ltrb"])
            arrays[variant_key("flipped", k) + s] = np.array(var["flipped"])
    if flip_legacy_reso is not None:  # a resolution cached without variants, with flipped latents
        s = _suffix(flip_legacy_reso)
        arrays["latents" + s] = _latent(flip_legacy_reso, 9.0).numpy()
        arrays["latents_flipped" + s] = _latent(flip_legacy_reso, -9.0).numpy()
        arrays["original_size" + s] = np.array(flip_legacy_reso)
        arrays["crop_ltrb" + s] = np.array((0, 0, flip_legacy_reso[0], flip_legacy_reso[1]))
    arrays["aug_variants"] = np.array(count)
    arrays["aug_config_hash"] = np.array(config_hash)
    save_npz(path, arrays)


def test_old_file_wide_hash_is_rebuilt_once(strategy, path):
    _old_layout_file(path, [(SMALL, 1.0), (LARGE, 1.0)], 2, HASH_B)

    # the pair may describe either resolution: neither is trusted, even with the matching hash
    assert not _valid(strategy, path, SMALL, 2, HASH_B)
    assert not _valid(strategy, path, LARGE, 2, HASH_B)
    # a check that does not ask for variants still accepts the canonical latents, and they still load
    assert strategy.is_disk_cached_latents_expected(SMALL, path, False, False)
    latents, original_size, crop, flipped_latents, alpha, variant_flipped = strategy.load_latents_from_disk(path, SMALL)
    assert latents.shape == (16, 1, 32, 32) and variant_flipped is False and flipped_latents is None

    # rebuilding one resolution moves the file to the per-resolution layout; the other resolution stays stale
    _save_variants(strategy, path, SMALL, 3.0, 2, HASH_B)
    stored = load_npz(path)
    assert "aug_variants" not in stored and "aug_config_hash" not in stored
    assert _valid(strategy, path, SMALL, 2, HASH_B)
    assert not _valid(strategy, path, LARGE, 2, HASH_B)
    # the stale resolution still loads (variant and canonical) until it is rebuilt
    assert strategy.load_latents_from_disk(path, LARGE, variant=1)[0].shape == (16, 1, 64, 64)
    assert strategy.load_latents_from_disk(path, LARGE)[0].shape == (16, 1, 64, 64)

    _save_variants(strategy, path, LARGE, 3.0, 2, HASH_B)
    assert _valid(strategy, path, LARGE, 2, HASH_B)
    assert _valid(strategy, path, SMALL, 2, HASH_B)  # once rebuilt, both stay valid


def test_old_single_resolution_file_without_suffix_still_loads(strategy, path):
    """Caches from before resolution suffixes (``latents`` only) keep loading through the fallback."""
    save_npz(path, {"latents": _latent(SMALL, 2.0).numpy(), "original_size": np.array(SMALL), "crop_ltrb": np.array((0, 0, 256, 256))})
    latents, original_size, crop, flipped_latents, alpha, variant_flipped = strategy.load_latents_from_disk(path, SMALL)
    assert float(latents.mean()) == 2.0 and original_size == [256, 256] and variant_flipped is None


def test_legacy_resolution_next_to_a_variant_resolution_keeps_its_flip(strategy, path):
    """Variant 0 of a variant cache is the unflipped canonical (flip is baked into the variants), but a resolution cached
    without variants keeps its flipped latents: the file-wide count made every resolution's variant 0 'unflipped'."""
    _save_variants(strategy, path, LARGE, 1.0, 2, HASH_A)
    strategy.save_latents_to_disk(
        path, _latent(SMALL, 7.0), SMALL, (0, 0, 256, 256), _latent(SMALL, -7.0), key_reso_suffix=_suffix(SMALL)
    )
    assert strategy.is_disk_cached_latents_expected(SMALL, path, True, False)
    *_, flipped_latents, _alpha, variant_flipped = strategy.load_latents_from_disk(path, SMALL)
    assert variant_flipped is None, "the legacy resolution must keep its random flip"
    assert float(flipped_latents.mean()) == -7.0
    assert strategy.load_latents_from_disk(path, LARGE)[5] is False  # the variant resolution's canonical is unflipped


def test_old_file_wide_count_marks_only_resolutions_with_variants(strategy, path):
    """Old writer: the large resolution cached with variants (file-wide pair), the small one later re-cached without
    variants and with flipped latents, and the file-wide pair stayed. flip_aug must still reach the small one."""
    _old_layout_file(path, [(LARGE, 1.0)], 2, HASH_A, flip_legacy_reso=SMALL)

    assert strategy.is_disk_cached_latents_expected(SMALL, path, True, False)  # a valid flip cache: not rebuilt
    *_, flipped_latents, _alpha, variant_flipped = strategy.load_latents_from_disk(path, SMALL)
    assert variant_flipped is None, "the file-wide count turned flip_aug off at a resolution without variants"
    assert float(flipped_latents.mean()) == -9.0
    assert strategy.load_latents_from_disk(path, LARGE)[5] is False  # the resolution with variants: canonical unflipped


@pytest.mark.parametrize("old_layout", [True, False], ids=["file-wide pair", "per-resolution pair"])
def test_flip_check_rebuilds_a_resolution_whose_variants_hide_its_flipped_latents(strategy, path, old_layout):
    """Flipped latents and variants at one resolution (a flip cache later cached with variants keeps its flipped
    latents): the loader takes variant 0 as unflipped, so a run with flip_aug and no variants rebuilds it once."""
    if old_layout:
        _old_layout_file(path, [(SMALL, 1.0)], 2, HASH_A, flip_legacy_reso=SMALL)
    else:
        strategy.save_latents_to_disk(
            path, _latent(SMALL, 9.0), SMALL, (0, 0, 256, 256), _latent(SMALL, -9.0), key_reso_suffix=_suffix(SMALL)
        )
        _save_variants(strategy, path, SMALL, 1.0, 2, HASH_A)
        assert _valid(strategy, path, SMALL, 2, HASH_A)  # a run with variants still uses it
    assert "latents_flipped" + _suffix(SMALL) in load_npz(path)
    assert strategy.load_latents_from_disk(path, SMALL)[5] is False

    assert not strategy.is_disk_cached_latents_expected(SMALL, path, True, False), "flip_aug would never be applied"
    assert strategy.is_disk_cached_latents_expected(SMALL, path, False, False)  # without flip_aug, variant 0 is right

    # the rebuild, a flip cache, drops this resolution's variant metadata, and the flip applies again
    strategy.save_latents_to_disk(
        path, _latent(SMALL, 9.0), SMALL, (0, 0, 256, 256), _latent(SMALL, -9.0), key_reso_suffix=_suffix(SMALL)
    )
    assert strategy.is_disk_cached_latents_expected(SMALL, path, True, False)
    *_, flipped_latents, _alpha, variant_flipped = strategy.load_latents_from_disk(path, SMALL)
    assert variant_flipped is None and float(flipped_latents.mean()) == -9.0


def _encode(img_tensor):  # (B, 3, H, W) -> (B, 16, H/8, W/8), like the Anima VAE wrapper's 4D output
    return torch.nn.functional.avg_pool2d(img_tensor.float(), 8).repeat(1, 6, 1, 1)[:, :16]


def test_the_caching_path_writes_and_checks_per_resolution(strategy, tmp_path):
    """Through _default_cache_batch_latents (the trainers' path): one image cached at two bucket resolutions."""
    from PIL import Image

    from library.train_util import ImageInfo

    image_path = str(tmp_path / "img.png")
    Image.new("RGB", (128, 128), (200, 10, 10)).save(image_path)
    infos = []
    for reso in ((64, 64), (128, 128)):
        info = ImageInfo("img", 1, "a cat", False, False, image_path)
        info.bucket_reso = info.resized_size = reso
        info.image_size = (128, 128)
        info.latents_npz = strategy.get_latents_npz_path(image_path, info.image_size)
        infos.append(info)
    small, large = infos

    def cache(info, config_hash):
        strategy._default_cache_batch_latents(
            _encode, torch.device("cpu"), torch.float32, [info], False, False, False, multi_resolution=True,
            num_aug_variants=2, aug_config_hash=config_hash,
        )

    cache(small, HASH_A)
    cache(large, HASH_A)
    cache(small, HASH_B)
    with np.load(small.latents_npz) as npz:
        assert npz["aug_config_hash_8x8"].tolist() == HASH_B and npz["aug_config_hash_16x16"].tolist() == HASH_A
        assert "aug_config_hash" not in npz.files
    assert _valid(strategy, small.latents_npz, (64, 64), 2, HASH_B)
    assert not _valid(strategy, large.latents_npz, (128, 128), 2, HASH_B)


def test_rewriting_a_resolution_without_variants_drops_its_variant_metadata(strategy, path):
    _save_variants(strategy, path, SMALL, 1.0, 3, HASH_A)
    _save_variants(strategy, path, LARGE, 1.0, 3, HASH_A)
    strategy.save_latents_to_disk(path, _latent(SMALL, 2.0), SMALL, (0, 0, 256, 256), key_reso_suffix=_suffix(SMALL))
    assert not _valid(strategy, path, SMALL, 3, HASH_A)  # now a plain cache at this resolution
    assert _valid(strategy, path, LARGE, 3, HASH_A)
    assert strategy.load_latents_from_disk(path, SMALL)[5] is None
