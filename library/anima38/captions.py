# Caption handling for cached Anima-3.8B training (--anima38_connector).
#
# 1. Canonical caption variant. BaseDataset.process_caption_canonical (variant 0 of --cache_caption_variants) never
#    handles keep_tokens_separator, so with the user's "fixed ||| tags ||| sentence" captions 1 of every 8 cached
#    variants fed all three encoders a literal "|||" that no Forge prompt contains. process_caption (variants 1..n,
#    and the online path the 2.9B config uses) splits on the separator and joins with ", ". canonical_caption() is
#    process_caption with every random choice fixed (no dropout, no shuffle, no tag dropout, first line, first
#    wildcard option), and install_canonical_captions() puts it on each dataset instance (the class is untouched)
#    for the duration of caching only: uninstall_canonical_captions() removes it again, because Windows DataLoader
#    workers pickle the dataset and a bound method of a foreign function does not pickle.
#
# 2. Caption digest. A cached *_anima38_te.npz holds features of the caption it was built from, and nothing in the
#    fork's validity check compares caption text. register_caption_digests() gives the caching strategy the SHA-1
#    of every image's current caption so an edited caption rebuilds its cache instead of silently reusing it.
#
# 3. Qwen3.5 text. Forge gives Qwen3.5 the raw prompt line, where a tag like "shrug (clothing)" is typed escaped
#    ("shrug \(clothing\)") so the prompt parser does not read it as emphasis; Qwen3 and T5 see the unescaped text.
#    forge_prompt_text() escapes training captions the same way for the Qwen3.5 input only.

from __future__ import annotations

import hashlib
import os
import re
import types
from typing import Any, Dict, Optional

_UNESCAPED_BRACKET = re.compile(r"(?<!\\)([()\[\]])")


def caption_digest(caption: Optional[str]) -> str:
    return hashlib.sha1((caption or "").encode("utf-8")).hexdigest()


def forge_prompt_text(caption: str) -> str:
    """The caption as it is typed into Forge: parentheses and brackets escaped (already escaped ones kept)."""
    return _UNESCAPED_BRACKET.sub(r"\\\1", caption)


def _first_wildcard_options(caption: str) -> str:
    replacer1, replacer2 = "⦅", "⦆"
    while replacer1 in caption or replacer2 in caption:
        replacer1 += "⦅"
        replacer2 += "⦆"
    caption = caption.replace("{{", replacer1).replace("}}", replacer2)
    caption = re.sub(r"\{([^}]+)\}", lambda m: m.group(1).split("|")[0], caption)
    return caption.replace(replacer1, "{").replace(replacer2, "}")


def canonical_caption(dataset: Any, subset: Any, caption: str) -> str:
    """BaseDataset.process_caption without randomness (same steps, same order, same joins)."""
    if subset.caption_prefix:
        caption = subset.caption_prefix + " " + caption
    if subset.caption_suffix:
        caption = caption + " " + subset.caption_suffix

    caption = caption.split("\n")[0]
    if subset.enable_wildcard:
        caption = _first_wildcard_options(caption)

    if subset.shuffle_caption or subset.token_warmup_step > 0 or subset.caption_tag_dropout_rate > 0:
        separator = getattr(subset, "keep_tokens_separator", None)
        fixed_tokens, flex_tokens, fixed_suffix_tokens = [], [], []
        if separator and separator in caption:
            fixed_part, flex_part = caption.split(separator, 1)
            if separator in flex_part:
                flex_part, fixed_suffix_part = flex_part.split(separator, 1)
                fixed_suffix_tokens = [t.strip() for t in fixed_suffix_part.split(subset.caption_separator) if t.strip()]
            fixed_tokens = [t.strip() for t in fixed_part.split(subset.caption_separator) if t.strip()]
            flex_tokens = [t.strip() for t in flex_part.split(subset.caption_separator) if t.strip()]
        else:
            flex_tokens = [t.strip() for t in caption.strip().split(subset.caption_separator)]
        caption = ", ".join(fixed_tokens + flex_tokens + fixed_suffix_tokens)

    if subset.secondary_separator:
        caption = caption.replace(subset.secondary_separator, subset.caption_separator)

    for str_from, str_to in dataset.replacements.items():
        if str_from == "":
            caption = str_to[0] if type(str_to) == list else str_to
        else:
            caption = caption.replace(str_from, str_to)
    return caption


def _datasets(dataset_group: Any):
    if dataset_group is None:
        return []
    return list(getattr(dataset_group, "datasets", None) or [dataset_group])


def install_canonical_captions(dataset_group: Any) -> int:
    """Replace process_caption_canonical on each dataset instance; returns how many datasets were patched."""
    count = 0
    for dataset in _datasets(dataset_group):
        if not hasattr(dataset, "process_caption_canonical"):
            continue
        dataset.process_caption_canonical = types.MethodType(canonical_caption, dataset)
        count += 1
    return count


def uninstall_canonical_captions(dataset_group: Any) -> int:
    """Undo install_canonical_captions (only instance attributes it set); returns how many were removed."""
    count = 0
    for dataset in _datasets(dataset_group):
        method = vars(dataset).get("process_caption_canonical") if hasattr(dataset, "__dict__") else None
        if getattr(method, "__func__", None) is canonical_caption:
            del dataset.process_caption_canonical
            count += 1
    return count


def register_caption_digests(dataset_group: Any, caching_strategy: Any) -> int:
    """Tell the caching strategy the digest of every image's current caption (keyed by its npz path)."""
    digests: Dict[str, str] = {}
    for dataset in _datasets(dataset_group):
        for info in getattr(dataset, "image_data", {}).values():
            npz_path = caching_strategy.get_outputs_npz_path(info.absolute_path)
            digests[npz_key(npz_path)] = caption_digest(info.caption)
    caching_strategy.caption_digests.update(digests)
    return len(digests)


def npz_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))
