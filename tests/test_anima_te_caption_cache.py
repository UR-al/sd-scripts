"""CPU tests for the canonical caption variant and the plain Anima Text Encoder cache (*_anima_te.npz, the cache of
config_anima29.toml / config_anima_base.toml; review F5 and perf (a), 2026-10-21):

- BaseDataset.process_caption_canonical (variant 0 of --cache_caption_variants) is process_caption without
  randomness, keep_tokens_separator split and ", " join included: "fixed ||| tags ||| sentence" no longer reaches the
  text encoder with a literal "|||". Variants 1..n are drawn exactly as before.
- Anima-3.8B / MOSS: their canonical functions give byte-for-byte the same text (installing them changes nothing),
  their digests and cache checks never see the plain Anima digest, and the caption augmentation hash is unchanged, so
  their caches stay valid.
- An edited caption file (or a deterministic caption setting) rebuilds its *_anima_te.npz; a file written before the
  caption digest existed is rebuilt once.
- AnimaTextEncoderOutputsCachingStrategy.load_outputs_npz reads one variant and returns exactly what the old
  load_npz-based loader returned (type, dtype, shape, bits); the lazy validity check decides as the old one did.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp=<scratch dir>
     tests/test_anima_te_caption_cache.py
"""
import random
import types
from dataclasses import asdict

import numpy as np
import pytest
import torch
from PIL import Image

from library import train_util
from library.anima38 import captions as anima38_captions
from library.anima38 import strategy as anima38_strategy
from library.cache_utils import load_npz, save_npz
from library.config_util import DreamBoothDatasetParams, DreamBoothSubsetParams
from library.moss import captions as moss_captions
from library.moss import strategy as moss_strategy
from library.strategy_anima import (CAPTION_DIGEST_KEY, AnimaTextEncoderOutputsCachingStrategy, AnimaTextEncodingStrategy,
                                    cache_caption_digest)
from library.strategy_base import (AUG_VARIANT_SCHEMA_VERSION, TextEncoderOutputsCachingStrategy, TextEncodingStrategy,
                                   TokenizeStrategy, variant_key)
from library.train_util import DreamBoothDataset, DreamBoothSubset

USER_CAPTION = "@artist, 1boy ||| red hair, smile ||| A man smiles."
USER_CAPTION_JOINED = "@artist, 1boy, red hair, smile, A man smiles."
TE_KEYS = ("prompt_embeds", "attn_mask", "t5_input_ids", "t5_attn_mask")


def _subset(**overrides):
    values = dict(
        caption_prefix=None, caption_suffix=None, caption_dropout_rate=0.0, caption_dropout_every_n_epochs=0,
        enable_wildcard=False, shuffle_caption=True, token_warmup_step=0, token_warmup_min=1,
        caption_tag_dropout_rate=0.0, keep_tokens_separator="|||", keep_tokens=0, caption_separator=",",
        secondary_separator=None,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _bare_dataset(replacements=None):
    dataset = train_util.BaseDataset.__new__(train_util.BaseDataset)
    dataset.replacements = dict(replacements or {})
    dataset.log_caption_dropout = False
    dataset.log_caption_tag_dropout = False
    dataset.current_epoch = 1
    dataset.current_step = 1000  # past any token warmup used below
    dataset.max_train_steps = 1000
    return dataset


# ======================================================================================================
# canonical caption (variant 0)
# ======================================================================================================


def test_canonical_variant_has_no_keep_tokens_separator():
    dataset = _bare_dataset()
    assert dataset.process_caption_canonical(_subset(), "@a, 1boy ||| t1, t2 ||| A sentence.") == "@a, 1boy, t1, t2, A sentence."
    assert dataset.process_caption_canonical(_subset(), USER_CAPTION) == USER_CAPTION_JOINED
    assert dataset.process_caption_canonical(_subset(), "@a ||| t1,t2") == "@a, t1, t2"
    assert dataset.process_caption_canonical(_subset(caption_tag_dropout_rate=0.1, shuffle_caption=False), USER_CAPTION) == USER_CAPTION_JOINED
    # keep_tokens pins the first tags in place; without shuffling the order is the caption's own
    assert dataset.process_caption_canonical(_subset(keep_tokens=2, keep_tokens_separator=""), "a, b,c ,d") == "a, b, c, d"
    # nothing random: process_caption uses the caption as written, separator included, and so does variant 0
    assert dataset.process_caption_canonical(_subset(shuffle_caption=False), USER_CAPTION) == USER_CAPTION


CAPTIONS = [
    USER_CAPTION,
    "@a, 2boys ||| male focus,  towel, speech bubble,",
    "no separator,tags only , trailing ,",
    "first {x|y} line ||| tag\nsecond line, z",
    "a ;;; b, c ||| d",
    "",
]
SUBSET_CASES = [
    {},
    {"shuffle_caption": False},
    {"shuffle_caption": False, "caption_tag_dropout_rate": 0.3},
    {"shuffle_caption": False, "token_warmup_step": 5},
    {"caption_tag_dropout_rate": 0.3, "caption_dropout_rate": 0.5},
    {"keep_tokens": 2},
    {"keep_tokens": 2, "keep_tokens_separator": ""},
    {"keep_tokens_separator": None},
    {"caption_prefix": "masterpiece", "caption_suffix": "4k"},
    {"secondary_separator": ";;;"},
    {"enable_wildcard": True},
    {"enable_wildcard": True, "caption_prefix": "PRE"},
    {"caption_separator": ";"},
]
REPLACEMENTS = [{}, {"@a": "@artist"}, {"": ["whole caption", "other caption"]}]


@pytest.mark.parametrize("replacements", REPLACEMENTS)
@pytest.mark.parametrize("subset_kwargs", SUBSET_CASES)
def test_canonical_is_process_caption_without_randomness_and_matches_anima38_and_moss(monkeypatch, subset_kwargs, replacements):
    """Same text as process_caption with every random choice fixed, and byte-for-byte the text the Anima-3.8B / MOSS
    caching patches compute: installing those patches no longer changes anything, so their caches see the same text."""
    monkeypatch.setattr(train_util.random, "shuffle", lambda seq: None)  # tag order kept
    monkeypatch.setattr(train_util.random, "random", lambda: 1.0)  # no caption dropout, no tag dropout
    monkeypatch.setattr(train_util.random, "choice", lambda seq: seq[0])  # first line, first wildcard option
    dataset = _bare_dataset(replacements)
    for caption in CAPTIONS:
        subset = _subset(**subset_kwargs)
        canonical = dataset.process_caption_canonical(subset, caption)
        assert canonical == dataset.process_caption(_subset(**subset_kwargs), caption), caption
        assert anima38_captions.canonical_caption(dataset, subset, caption) == canonical, caption
        assert moss_captions.canonical_caption(dataset, subset, caption) == canonical, caption


def test_build_caption_variants_changes_only_variant_0():
    """Variants 1..n are the same process_caption draws as before (the canonical function draws no random numbers)."""
    dataset = _bare_dataset()
    subset = _subset(caption_tag_dropout_rate=0.2)
    info = types.SimpleNamespace(image_key="img", caption=USER_CAPTION, caption_variants=None)
    dataset.image_data = {"img": info}
    dataset.image_to_subset = {"img": subset}
    dataset.caption_aug_variant_count = 16
    random.seed(1234)
    dataset.build_caption_variants()
    random.seed(1234)
    expected = [dataset.process_caption(subset, USER_CAPTION) for _ in range(15)]
    assert info.caption_variants[1:] == expected
    assert info.caption_variants[0] == USER_CAPTION_JOINED
    assert all("|||" not in variant for variant in info.caption_variants)


# ======================================================================================================
# Anima-3.8B / MOSS caches are left alone
# ======================================================================================================


def test_caption_aug_hash_and_schema_are_unchanged():
    """Every *_anima38_te.npz / *_moss_te.npz stores this hash: another value would rebuild all of them."""
    assert AUG_VARIANT_SCHEMA_VERSION == 1
    dataset = _bare_dataset()
    assert dataset.get_caption_aug_config_hash(_subset()) == "456f78bb711c86f941a524d5df286d62c988ba77"
    assert dataset.get_caption_aug_config_hash(_subset(keep_tokens=3)) == "fb088d934d3faaff56c7bf1546d841185558b8c3"


def test_plain_anima_caption_digests_leave_anima38_and_moss_alone(tmp_path):
    cls = anima38_strategy.Anima38TextEncoderOutputsCachingStrategy
    for name in ("is_disk_cached_outputs_expected", "load_outputs_npz", "cache_batch_outputs"):
        assert name in vars(cls), name  # the 3.8B strategy never runs the plain Anima versions
    assert not hasattr(moss_strategy.MossTextEncoderOutputsCachingStrategy, "register_cache_captions")

    caching = cls(True, 1, False, False, qwen35_signature="sig")
    image = str(tmp_path / "img.png")
    npz_path = caching.get_outputs_npz_path(image)
    arrays = {key: np.zeros(2, dtype=np.float32) for key in caching.OUTPUT_KEYS}
    arrays["caption_dropout_rate"] = np.array(0.0, dtype=np.float32)
    arrays[anima38_strategy.SIGNATURE_KEY] = np.array("sig")
    arrays[anima38_strategy.CAPTION_DIGEST_KEY] = np.array(anima38_captions.caption_digest(USER_CAPTION))
    arrays[anima38_strategy.CACHE_CAPTION_DIGEST_KEY] = np.array(anima38_captions.caption_digest(USER_CAPTION_JOINED))
    np.savez(npz_path, **arrays)

    dataset = _bare_dataset()
    info = types.SimpleNamespace(absolute_path=image, image_key="k", caption=USER_CAPTION, caption_variants=None)
    dataset.image_data = {"k": info}
    dataset.image_to_subset = {"k": _subset()}
    anima38_captions.install_canonical_captions(dataset)
    anima38_captions.register_caption_digests(dataset, caching)
    anima38_captions.uninstall_canonical_captions(dataset)
    digests = (dict(caching.caption_digests), dict(caching.cache_caption_digests))
    assert caching.is_disk_cached_outputs_expected(npz_path, 0, None)

    # what BaseDataset.new_cache_text_encoder_outputs now calls (the 3.8B strategy inherits the hook), even with an
    # unrelated caption: only the plain dict changes, which the 3.8B check never reads
    caching.register_cache_captions([types.SimpleNamespace(absolute_path=image, caption="x", cache_caption="y", caption_variants=None)])
    assert (caching.caption_digests, caching.cache_caption_digests) == digests
    assert caching.is_disk_cached_outputs_expected(npz_path, 0, None)


# ======================================================================================================
# plain Anima cache: caption digest
# ======================================================================================================

SEQ = 96


class _CharTokenizer(TokenizeStrategy):
    """Stands in for Qwen3 / T5: one id per character, so cached ids decode back to the text that was encoded."""

    def tokenize(self, text):
        texts = [text] if isinstance(text, str) else text
        ids = torch.zeros(len(texts), SEQ, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, caption in enumerate(texts):
            codes = [ord(ch) for ch in caption[:SEQ]]
            ids[i, : len(codes)] = torch.tensor(codes, dtype=torch.long)
            mask[i, : len(codes)] = 1
        return [ids, mask, ids.clone(), mask.clone()]


class _FakeEncoding(AnimaTextEncodingStrategy):
    """bf16 prompt_embeds with varied bits (signs, exponents) and zeros on padding, as the Qwen3 encoder gives."""

    def encode_tokens(self, tokenize_strategy, models, tokens):
        ids, mask, t5_ids, t5_mask = tokens
        embeds = (ids.float()[..., None] * torch.linspace(-1.3, 2.7, 8)).sin() * 37.0 * mask[..., None]
        return [embeds.to(torch.bfloat16), mask, t5_ids, t5_mask]


class _RecordingCaching(AnimaTextEncoderOutputsCachingStrategy):
    def __init__(self, cache_dtype="auto"):
        super().__init__(cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False, cache_dtype=cache_dtype)
        self.encoded = []

    def cache_batch_outputs(self, tokenize_strategy, models, text_encoding_strategy, infos):
        self.encoded.extend(info.caption for info in infos)
        return super().cache_batch_outputs(tokenize_strategy, models, text_encoding_strategy, infos)


@pytest.fixture
def anima_caching():
    classes = (TokenizeStrategy, TextEncodingStrategy, TextEncoderOutputsCachingStrategy)
    saved = [cls._strategy for cls in classes]
    caching = _RecordingCaching()
    for cls, value in zip(classes, (_CharTokenizer(), _FakeEncoding(), caching)):
        cls._strategy = value
    try:
        yield caching
    finally:
        for cls, value in zip(classes, saved):
            cls._strategy = value


def _cache_run(image_dir, caption, caption_variants=16, **subset_kwargs):
    """One caching run as a trainer does it: a fresh dataset reads the caption file, then caches TE outputs."""
    image_dir.mkdir(exist_ok=True)
    if not (image_dir / "a.png").exists():
        Image.new("RGB", (64, 64), (10, 20, 30)).save(image_dir / "a.png")
    (image_dir / "a.txt").write_text(caption, encoding="utf-8")
    subset = DreamBoothSubset(**asdict(DreamBoothSubsetParams(image_dir=str(image_dir), caption_extension=".txt", **subset_kwargs)))
    dataset = DreamBoothDataset(subsets=[subset], is_training_dataset=True, **asdict(DreamBoothDatasetParams(resolution=(64, 64))))
    dataset.set_aug_variant_config(caption_aug_variants=caption_variants)
    dataset.new_cache_text_encoder_outputs([None], types.SimpleNamespace(num_processes=1, process_index=0))
    return list(dataset.image_data.values())[0]


def _decoded_t5(caching, info, variant=0):
    ids = np.asarray(caching.load_outputs_npz(info.text_encoder_outputs_npz, variant=variant)[2])
    return "".join(chr(int(x)) for x in ids if int(x))


USER_SUBSET = dict(shuffle_caption=True, keep_tokens_separator="|||")  # config_anima29 / config_anima_base


def test_edited_caption_rebuilds_the_plain_anima_cache(tmp_path, anima_caching):
    info = _cache_run(tmp_path, USER_CAPTION, **USER_SUBSET)
    assert anima_caching.encoded == [USER_CAPTION]
    assert _decoded_t5(anima_caching, info) == USER_CAPTION_JOINED  # variant 0 without the literal '|||'
    with np.load(info.text_encoder_outputs_npz) as npz:
        assert str(npz[CAPTION_DIGEST_KEY].tolist()) == cache_caption_digest(info)

    _cache_run(tmp_path, USER_CAPTION, **USER_SUBSET)
    assert anima_caching.encoded == [USER_CAPTION]  # same caption: the file is reused

    edited = "@artist, 1boy ||| BLUE hair, frown ||| A man frowns."
    info = _cache_run(tmp_path, edited, **USER_SUBSET)
    assert anima_caching.encoded == [USER_CAPTION, edited]  # only the caption file changed: rebuilt
    assert _decoded_t5(anima_caching, info) == "@artist, 1boy, BLUE hair, frown, A man frowns."
    assert sorted(_decoded_t5(anima_caching, info, 5).split(", ")) == sorted(["@artist", "1boy", "BLUE hair", "frown", "A man frowns."])
    assert anima_caching.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz, 16, info.caption_aug_hash)


def test_changed_caption_prefix_rebuilds_a_cache_without_variants(tmp_path, anima_caching):
    _cache_run(tmp_path, "1girl, smile", caption_variants=0)
    info = _cache_run(tmp_path, "1girl, smile", caption_variants=0, caption_prefix="masterpiece")
    assert anima_caching.encoded == ["1girl, smile", "1girl, smile"]
    assert _decoded_t5(anima_caching, info) == "masterpiece 1girl, smile"


def test_changed_caption_dropout_rate_rebuilds_a_cache_without_variants(tmp_path, anima_caching):
    # without variants the file stores the rate the trainer applies, so a new rate must not reuse the old file
    _cache_run(tmp_path, "1girl, smile", caption_variants=0)
    info = _cache_run(tmp_path, "1girl, smile", caption_variants=0, caption_dropout_rate=0.1)
    assert anima_caching.encoded == ["1girl, smile", "1girl, smile"]
    assert float(np.asarray(anima_caching.load_outputs_npz(info.text_encoder_outputs_npz)[-1])) == pytest.approx(0.1)
    _cache_run(tmp_path, "1girl, smile", caption_variants=0, caption_dropout_rate=0.1)
    assert len(anima_caching.encoded) == 2  # same rate: reused


def test_cache_without_a_caption_digest_is_rebuilt_once(tmp_path, anima_caching):
    info = _cache_run(tmp_path, USER_CAPTION, **USER_SUBSET)
    with np.load(info.text_encoder_outputs_npz) as npz:
        arrays = {key: npz[key] for key in npz.files if key != CAPTION_DIGEST_KEY}
    np.savez_compressed(info.text_encoder_outputs_npz, **arrays)  # what every run before the digest wrote
    _cache_run(tmp_path, USER_CAPTION, **USER_SUBSET)
    _cache_run(tmp_path, USER_CAPTION, **USER_SUBSET)
    assert anima_caching.encoded == [USER_CAPTION, USER_CAPTION]


def test_skip_cache_check_and_unregistered_files_are_not_digest_checked(tmp_path):
    info = types.SimpleNamespace(caption="a", cache_caption="a", caption_variants=None)
    path = str(tmp_path / "x_anima_te.npz")
    np.savez(path, **{key: np.zeros(2, dtype=np.float32) for key in TE_KEYS + ("caption_dropout_rate",)})
    assert AnimaTextEncoderOutputsCachingStrategy(True, 1, False).is_disk_cached_outputs_expected(path)  # nothing registered
    caching = AnimaTextEncoderOutputsCachingStrategy(True, 1, False)
    caching.register_cache_captions([types.SimpleNamespace(absolute_path=str(tmp_path / "x.png"), **vars(info))])
    assert not caching.is_disk_cached_outputs_expected(path)  # registered, file has no digest
    caching.skip_disk_cache_validity_check = True
    assert caching.is_disk_cached_outputs_expected(path)


# ======================================================================================================
# plain Anima cache: one-variant loader and lazy validity check == the old load_npz-based code
# ======================================================================================================


def _old_load_outputs_npz(npz_path, variant=0):
    """AnimaTextEncoderOutputsCachingStrategy.load_outputs_npz before perf (a): decode every entry, then pick one."""
    data, dtypes = load_npz(npz_path, return_dtypes=True)
    get = TextEncoderOutputsCachingStrategy._npz_get
    prompt_embeds = get(data, "prompt_embeds", variant)
    attn_mask = get(data, "attn_mask", variant)
    t5_input_ids = get(data, "t5_input_ids", variant)
    t5_attn_mask = get(data, "t5_attn_mask", variant)
    caption_dropout_rate = data["caption_dropout_rate"]
    if dtypes.get("prompt_embeds") == "bf16":
        prompt_embeds = torch.from_numpy(np.ascontiguousarray(prompt_embeds)).to(dtype=torch.bfloat16)
    return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, caption_dropout_rate]


def _old_is_disk_cached_outputs_expected(caching, npz_path, num_caption_variants=0, caption_aug_hash=None):
    """The validity check before perf (a) (load_npz of the whole file), without a caption digest."""
    npz = load_npz(npz_path)
    for key in TE_KEYS + ("caption_dropout_rate",):
        if key not in npz:
            return False
    return caching._check_cached_variant_keys(npz, set(TE_KEYS), num_caption_variants, caption_aug_hash)


def _assert_identical(new, old):
    assert len(new) == len(old) == 5
    for a, b in zip(new, old):
        assert type(a) is type(b)
        if isinstance(a, torch.Tensor):
            assert a.dtype == b.dtype and a.shape == b.shape and a.is_cpu and b.is_cpu
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)) if a.dtype == torch.bfloat16 else torch.equal(a, b)
        else:
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _variant_infos(tmp_path, n_infos=2, n_variants=16):
    infos = []
    for i in range(n_infos):
        variants = [f"canonical {i}, tag a, tag b"] + [f"variant {k} of {i}, " + ", ".join(["t"] * (k % 5)) for k in range(1, n_variants)]
        infos.append(types.SimpleNamespace(
            caption=f"raw {i}", cache_caption=variants[0], caption_variants=variants, caption_dropout_rate=0.0,
            caption_aug_hash="hash-a", absolute_path=str(tmp_path / f"img{i}.png"),
            text_encoder_outputs_npz=str(tmp_path / f"img{i}_anima_te.npz"),
        ))
    return infos


@pytest.mark.parametrize("cache_dtype", ["auto", "bf16", "fp16", "fp32"])
def test_one_variant_loader_returns_exactly_what_the_old_loader_returned(tmp_path, cache_dtype):
    caching = _RecordingCaching(cache_dtype=cache_dtype)
    infos = _variant_infos(tmp_path)
    caching.cache_batch_outputs(_CharTokenizer(), [None], _FakeEncoding(), infos)
    for info in infos:
        for variant in range(0, 18):  # 16 and 17 do not exist: both loaders fall back to the canonical entries
            new = caching.load_outputs_npz(info.text_encoder_outputs_npz, variant=variant)
            _assert_identical(new, _old_load_outputs_npz(info.text_encoder_outputs_npz, variant))
        assert new[0].dtype == (torch.bfloat16 if cache_dtype in ("auto", "bf16") else np.float16 if cache_dtype == "fp16" else np.float32)


def test_one_variant_loader_on_legacy_and_uncompressed_files(tmp_path):
    caching = AnimaTextEncoderOutputsCachingStrategy(True, 1, False)
    rng = np.random.RandomState(0)
    legacy = str(tmp_path / "legacy_anima_te.npz")
    np.savez(legacy, prompt_embeds=rng.randn(SEQ, 8).astype(np.float32), attn_mask=np.ones(SEQ, dtype=np.int64),
             t5_input_ids=np.arange(SEQ, dtype=np.int32), t5_attn_mask=np.ones(SEQ, dtype=np.int32),
             caption_dropout_rate=np.array(0.25, dtype=np.float32))  # an fp32 cache from before the bf16 metadata
    plain = str(tmp_path / "plain_anima_te.npz")
    arrays = {"prompt_embeds": torch.randn(SEQ, 8).to(torch.bfloat16), "attn_mask": np.ones(SEQ, dtype=np.int64),
              "t5_input_ids": np.arange(SEQ, dtype=np.int32), "t5_attn_mask": np.ones(SEQ, dtype=np.int32),
              "caption_dropout_rate": torch.tensor(0.1)}
    arrays.update({variant_key("prompt_embeds", 1): torch.randn(SEQ, 8).to(torch.bfloat16)})  # one variant, partial keys
    save_npz(plain, arrays, compress=False)
    for path in (legacy, plain):
        for variant in (0, 1, 3):
            _assert_identical(caching.load_outputs_npz(path, variant), _old_load_outputs_npz(path, variant))


def test_lazy_validity_check_decides_like_the_old_one(tmp_path):
    caching = _RecordingCaching()
    infos = _variant_infos(tmp_path, n_infos=1)
    caching.cache_batch_outputs(_CharTokenizer(), [None], _FakeEncoding(), infos)
    complete = infos[0].text_encoder_outputs_npz
    with np.load(complete) as npz:
        arrays = {key: npz[key] for key in npz.files}
    files = {"complete": complete}
    for name, drop in (("no_t5_mask", "t5_attn_mask"), ("no_rate", "caption_dropout_rate"), ("no_v5", "prompt_embeds_v5"),
                       ("no_hash", "caption_aug_hash"), ("legacy", None)):
        path = str(tmp_path / f"{name}_anima_te.npz")
        kept = {k: v for k, v in arrays.items() if k != drop}
        if name == "legacy":
            kept = {k: v for k, v in arrays.items() if "_v" not in k and k not in ("caption_variants", "caption_aug_hash")}
        np.savez_compressed(path, **kept)
        files[name] = path
    checks = [(0, None), (1, None), (8, "hash-a"), (16, "hash-a"), (17, "hash-a"), (16, "hash-b"), (16, None)]
    fresh = AnimaTextEncoderOutputsCachingStrategy(True, 1, False)  # no digests registered: the old decisions
    for name, path in files.items():
        for variants, aug_hash in checks:
            assert fresh.is_disk_cached_outputs_expected(path, variants, aug_hash) == _old_is_disk_cached_outputs_expected(
                fresh, path, variants, aug_hash), (name, variants, aug_hash)
    assert fresh.is_disk_cached_outputs_expected(complete, 16, "hash-a")
    assert not fresh.is_disk_cached_outputs_expected(files["no_v5"], 16, "hash-a")
