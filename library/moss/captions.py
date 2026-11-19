# Caption handling for cached MOSS-Image training.
#
# 1. Canonical caption variant. BaseDataset.process_caption_canonical (variant 0 of --cache_caption_variants) used to
#    ignore keep_tokens_separator, so "fixed ||| tags ||| sentence" captions reached the text encoder with a literal
#    "|||" in one of every N cached variants. It now returns the same text as canonical_caption() below
#    (tests/test_anima_te_caption_cache.py); this stays as MOSS's own definition. canonical_caption() is
#    BaseDataset.process_caption with every random choice
#    fixed (no caption dropout, no shuffle, no tag dropout; with enable_wildcard the first line and the first wildcard
#    option): the same joins as the shuffled variants. It is installed on each dataset INSTANCE only while caching (Windows DataLoader workers
#    pickle the dataset, and a bound foreign function does not pickle).
# 2. Caption digest. A cached *_moss_te.npz holds the encoding of one caption; the fork's validity check never compares
#    caption text, so the SHA-1 of every image's current caption is registered and an edited caption rebuilds its cache.
# 3. Live text encoder (moss_train.py --moss_te_live): kohya's process_caption runs at every fetch; separator_warnings
#    finds captions that would reach the encoder with a literal "|||", live_caption_summary logs what is drawn.

from __future__ import annotations

import hashlib
import os
import re
import types
from typing import Any, Dict, List, Optional


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


# ---- live text encoder (--moss_te_live): what process_caption draws per fetch -------------------------------------
SEPARATOR_CANDIDATES = ("|||",)  # the HEAD ||| TAGS ||| SENTENCES convention of the user's captions


def datasets_of(dataset_group: Any):
    return _datasets(dataset_group)


def _subset_captions(dataset: Any):
    """{subset id: (subset, [(image key, caption)])} of one dataset."""
    out: Dict[int, Any] = {}
    image_to_subset = getattr(dataset, "image_to_subset", None) or {}
    for key, info in (getattr(dataset, "image_data", None) or {}).items():
        subset = image_to_subset.get(key)
        if subset is None:
            continue
        out.setdefault(id(subset), (subset, []))[1].append((key, info.caption or ""))
    return out


def _splits_on_separator(subset: Any) -> bool:
    """process_caption splits at keep_tokens_separator only under shuffle, token warmup or tag dropout."""
    return bool(subset.shuffle_caption or subset.token_warmup_step > 0 or subset.caption_tag_dropout_rate > 0)


WILDCARD_RE = re.compile(r"\{([^}]+)\}")  # train_util.process_caption's wildcard pattern ({a|b|c})


def separator_warnings(dataset_group: Any) -> List[str]:
    """Captions that reach the text encoder in a form the user probably did not mean, one line per subset with the
    count and one example:
    - a literal separator: `|||` with keep_tokens_separator unset (or set to something else), or the subset's separator
      with none of shuffle / token warmup / tag dropout on (kohya's process_caption only splits there);
    - multi-line caption files with enable_wildcard off: every fetch encodes all the lines joined by newlines (long +
      medium + short variants as one prompt, far past the ~300 text tokens the VRAM budget assumes) instead of one line;
    - enable_wildcard on and a caption with `{...}`: process_caption replaces every {a|b} with one random option (and a
      `{word}` without `|` with the word), so braces in caption text do not reach the encoder."""
    lines = []
    for dataset in _datasets(dataset_group):
        for subset, items in _subset_captions(dataset).values():
            separator = getattr(subset, "keep_tokens_separator", None) or ""
            where = getattr(subset, "image_dir", None) or "?"
            for candidate in SEPARATOR_CANDIDATES:
                if candidate == separator:
                    continue
                hits = [key for key, caption in items if candidate in caption]
                if hits:
                    lines.append(f"{len(hits)} caption(s) in {where} contain {candidate!r} but keep_tokens_separator is"
                                 f" {separator or 'not set'!r}: the encoder sees a literal {candidate!r} (set"
                                 f" keep_tokens_separator = \"{candidate}\"), e.g. {hits[0]}")
            if separator and not _splits_on_separator(subset):
                hits = [key for key, caption in items if separator in caption]
                if hits:
                    lines.append(f"{len(hits)} caption(s) in {where} contain keep_tokens_separator {separator!r}, but"
                                 " kohya's process_caption splits there only with shuffle_caption, caption_tag_dropout_rate"
                                 f" or token_warmup_step: the encoder sees a literal {separator!r}, e.g. {hits[0]}")
            wildcard = bool(getattr(subset, "enable_wildcard", False))
            if not wildcard:
                hits = [key for key, caption in items if "\n" in caption.strip("\n")]
                if hits:
                    lines.append(f"{len(hits)} multi-line caption file(s) in {where} with enable_wildcard off: every"
                                 " fetch encodes all their lines joined (e.g. long + medium + short as one prompt, well"
                                 " past the ~300 text tokens the VRAM estimate assumes). Set enable_wildcard = true to"
                                 f" draw one line per fetch, e.g. {hits[0]}")
            else:
                hits = [key for key, caption in items if WILDCARD_RE.search(caption)]
                if hits:
                    lines.append(f"{len(hits)} caption(s) in {where} contain {{...}} and enable_wildcard is on:"
                                 " process_caption replaces each {a|b} with one random option (a {word} without '|'"
                                 f" with the word), so the braces never reach the encoder, e.g. {hits[0]}")
    return lines


def live_caption_summary(dataset_group: Any) -> List[str]:
    """One line per subset: what process_caption draws at every fetch in live mode."""
    lines = []
    for dataset in _datasets(dataset_group):
        for subset, items in _subset_captions(dataset).values():
            line_counts = [len(caption.split("\n")) for _, caption in items]
            multi = sum(1 for n in line_counts if n > 1)
            wildcard = (f"one of up to {max(line_counts, default=1)} lines per fetch ({multi} multi-line captions)"
                        if subset.enable_wildcard else
                        ("off" + (f" ({multi} multi-line captions used whole)" if multi else "")))
            protected = getattr(subset, "protected_tags_file", None) or getattr(dataset, "protected_tags_file", None)
            lines.append(
                f"{getattr(subset, 'image_dir', None) or '?'}: {len(items)} images, shuffle_caption"
                f" {bool(subset.shuffle_caption)}, keep_tokens {subset.keep_tokens}, keep_tokens_separator"
                f" {getattr(subset, 'keep_tokens_separator', None) or None!r}, wildcard {wildcard}, caption_dropout_rate"
                f" {subset.caption_dropout_rate:g}, caption_dropout_every_n_epochs {subset.caption_dropout_every_n_epochs},"
                f" caption_tag_dropout_rate {subset.caption_tag_dropout_rate:g}"
                + (f" (protected tags {protected})" if protected else "")
                + (f", token_warmup_step {subset.token_warmup_step:g} (min {subset.token_warmup_min})"
                   if subset.token_warmup_step else ""))
    return lines
