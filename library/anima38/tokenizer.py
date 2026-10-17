# Qwen3.5 tokenizer wrapper for the Anima-3.8B v1.1 semantic path.
#
# Port of forge_sam3_extension/sam3ext/anima38/tokenizer.py (forge-anima-3.8B, MIT; see
# LICENSE-forge-anima-3.8B.txt). The tokenizer files themselves (Qwen/Qwen3.5-4B revision
# 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, Apache-2.0) are NOT copied into this package
# (tokenizer.json is 12.8 MB); they are read from a directory: the ``tokenizer_dir`` argument, else the
# ANIMA38_QWEN35_TOKENIZER_DIR environment variable, else the Forge extension's bundled copy.

from __future__ import annotations

import os
from typing import Optional

DEFAULT_TOKENIZER_DIR = r"C:\sd-webui-forge-classic\extensions\forge_sam3_extension\assets\qwen35_tokenizer"
TOKENIZER_DIR_ENV = "ANIMA38_QWEN35_TOKENIZER_DIR"


def resolve_tokenizer_dir(tokenizer_dir: Optional[str] = None) -> str:
    """Directory holding tokenizer.json. An explicit ``tokenizer_dir`` must be valid (no silent fallback)."""
    if tokenizer_dir:
        if not os.path.isfile(os.path.join(tokenizer_dir, "tokenizer.json")):
            raise FileNotFoundError(f"Qwen3.5 tokenizer.json not found in {tokenizer_dir}")
        return tokenizer_dir
    for candidate in (os.environ.get(TOKENIZER_DIR_ENV), DEFAULT_TOKENIZER_DIR):
        if candidate and os.path.isfile(os.path.join(candidate, "tokenizer.json")):
            return candidate
    raise FileNotFoundError(
        "Qwen3.5 tokenizer files not found. Pass tokenizer_dir, set "
        f"{TOKENIZER_DIR_ENV}, or keep {DEFAULT_TOKENIZER_DIR}"
    )


class Qwen35Tokenizer:
    """Raw prompt text -> Qwen3.5 ids: no special tokens, no truncation by HF, cut to 1024 tokens,
    empty text -> [pad_token_id] (same as the reference)."""

    pad_token_id = 151643
    max_length = 1024

    def __init__(self, tokenizer_dir: Optional[str] = None):
        from transformers import AutoTokenizer

        self.tokenizer_dir = resolve_tokenizer_dir(tokenizer_dir)
        self.inner = AutoTokenizer.from_pretrained(self.tokenizer_dir, trust_remote_code=False, local_files_only=True)

    def __call__(self, texts, **kwargs):
        kwargs.setdefault("truncation", False)
        kwargs.setdefault("add_special_tokens", False)
        result = self.inner(texts, **kwargs)
        result["input_ids"] = [(tokens[: self.max_length] or [self.pad_token_id]) for tokens in result["input_ids"]]
        return result
