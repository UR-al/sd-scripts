# Caption handling for cached MOSS-Image training.
#
# 1. Canonical caption variant. BaseDataset.process_caption_canonical (variant 0 of --cache_caption_variants) ignores
#    keep_tokens_separator, so "fixed ||| tags ||| sentence" captions would reach the text encoder with a literal "|||"
#    in one of every N cached variants. canonical_caption() is BaseDataset.process_caption with every random choice
#    fixed (no caption dropout, no shuffle, no tag dropout; with enable_wildcard the first line and the first wildcard
#    option): the same joins as the shuffled variants. It is installed on each dataset INSTANCE only while caching (Windows DataLoader workers
#    pickle the dataset, and a bound foreign function does not pickle).
# 2. Caption digest. A cached *_moss_te.npz holds the encoding of one caption; the fork's validity check never compares
#    caption text, so the SHA-1 of every image's current caption is registered and an edited caption rebuilds its cache.

from __future__ import annotations

import hashlib
import os
import re
import types
from typing import Any, Dict, Optional


def caption_digest(caption: Optional[str]) -> str:
    return hashlib.sha1((caption or "").encode("utf-8")).hexdigest()


def npz_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _first_wildcard_options(caption: str) -> str:
    left, right = "⦅", "⦆"
    while left in caption or right in caption:
        left += "⦅"
        right += "⦆"
    caption = caption.replace("{{", left).replace("}}", right)
    caption = re.sub(r"\{([^}]+)\}", lambda m: m.group(1).split("|")[0], caption)
    return caption.replace(left, "{").replace(right, "}")


def canonical_caption(dataset: Any, subset: Any, caption: str) -> str:
    """train_util.BaseDataset.process_caption without randomness (same steps and joins, in the same order)."""
    if subset.enable_wildcard:  # one line per candidate: the first one, chosen before prefix/suffix like process_caption
        caption = caption.split("\n")[0]
    if subset.caption_prefix:
        caption = subset.caption_prefix + " " + caption
    if subset.caption_suffix:
        caption = caption + " " + subset.caption_suffix
    if subset.enable_wildcard:  # without it process_caption uses a multi-line caption whole
        caption = _first_wildcard_options(caption)

    if subset.shuffle_caption or subset.token_warmup_step > 0 or subset.caption_tag_dropout_rate > 0:
        separator = getattr(subset, "keep_tokens_separator", None)
        fixed, flex, suffix = [], [], []
        if separator and separator in caption:
            fixed_part, flex_part = caption.split(separator, 1)
            if separator in flex_part:
                flex_part, suffix_part = flex_part.split(separator, 1)
                suffix = [t.strip() for t in suffix_part.split(subset.caption_separator) if t.strip()]
            fixed = [t.strip() for t in fixed_part.split(subset.caption_separator) if t.strip()]
            flex = [t.strip() for t in flex_part.split(subset.caption_separator) if t.strip()]
        else:
            tokens = [t.strip() for t in caption.strip().split(subset.caption_separator)]
            flex = tokens[:]
            if subset.keep_tokens > 0:
                fixed, flex = flex[: subset.keep_tokens], tokens[subset.keep_tokens:]
        caption = ", ".join(fixed + flex + suffix)

    if subset.secondary_separator:
        caption = caption.replace(subset.secondary_separator, subset.caption_separator)
    for source, target in dataset.replacements.items():
        if source == "":
            caption = target[0] if isinstance(target, list) else target
        else:
            caption = caption.replace(source, target[0] if isinstance(target, list) else target)
    return caption


def _datasets(dataset_group: Any):
    if dataset_group is None:
        return []
    return list(getattr(dataset_group, "datasets", None) or [dataset_group])


def install_canonical_captions(dataset_group: Any) -> int:
    count = 0
    for dataset in _datasets(dataset_group):
        if hasattr(dataset, "process_caption_canonical"):
            dataset.process_caption_canonical = types.MethodType(canonical_caption, dataset)
            count += 1
    return count


def uninstall_canonical_captions(dataset_group: Any) -> int:
    count = 0
    for dataset in _datasets(dataset_group):
        method = vars(dataset).get("process_caption_canonical") if hasattr(dataset, "__dict__") else None
        if getattr(method, "__func__", None) is canonical_caption:
            del dataset.process_caption_canonical
            count += 1
    return count


def prepare_canonical_captions(dataset_group: Any) -> int:
    """Store the deterministic processed caption on every image info (``info.cache_caption``).

    BaseDataset.new_cache_text_encoder_outputs sets the same attribute for every model (with the installed canonical
    function this gives the same text); it is computed here first so register_caption_digests can hash it before the
    cache validity check runs."""
    count = 0
    for dataset in _datasets(dataset_group):
        image_to_subset = getattr(dataset, "image_to_subset", {})
        for info in getattr(dataset, "image_data", {}).values():
            subset = image_to_subset.get(info.image_key)
            if subset is None or info.caption is None:
                continue
            info.cache_caption = canonical_caption(dataset, subset, info.caption)
            count += 1
    return count


def register_caption_digests(dataset_group: Any, caching_strategy: Any) -> int:
    """Digest = raw caption + the processed canonical caption, so editing the caption file OR any deterministic
    caption-processing setting (prefix/suffix, separators, shuffle on/off, replacements) rebuilds the cache."""
    digests: Dict[str, str] = {}
    for dataset in _datasets(dataset_group):
        for info in getattr(dataset, "image_data", {}).values():
            text = (info.caption or "") + "\x00" + str(getattr(info, "cache_caption", info.caption) or "")
            digests[npz_key(caching_strategy.get_outputs_npz_path(info.absolute_path))] = caption_digest(text)
    caching_strategy.caption_digests.update(digests)
    return len(digests)


def info_caption_digest(info: Any) -> str:
    """The digest register_caption_digests computes for this info (stored in the cache file)."""
    return caption_digest((info.caption or "") + "\x00" + str(getattr(info, "cache_caption", info.caption) or ""))
