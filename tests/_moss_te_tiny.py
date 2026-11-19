"""Shared helpers of tests/test_moss_ft_te_live_*.py (library/moss/te_live*.py): a tiny Qwen3.5-like text encoder
config, a random safetensors file of it in the 9B file's layout (``language_model.`` prefix, bf16 tensors, one fp32
tensor per GatedDeltaNet layer (A_log) to cover the load cast, an unrelated vision tensor that the loaders skip), the
reference model loaded from that file by the cached path's loader, and a character-level tokenizer with the real chat
template. Not collected by pytest (no test_ prefix)."""
from dataclasses import replace

import torch
from safetensors.torch import save_file

from library.moss import strategy, te_live, te_live_weights
from library.moss.qwen35 import MossQwen35Config, MossQwen35TextEncoder, format_moss_prompt, load_moss_text_encoder, \
    unescape_prompt

# hidden 64, intermediate 128, 8 layers (full attention at 3 and 7), 4 heads / 2 KV heads of 16, linear heads 2 K / 4 V
# of 16, rotary 4 dims, vocab 512
TINY_TE = replace(MossQwen35Config(), vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                  full_attention_interval=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                  linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16)
PREFIX = "language_model."
LENGTHS = (10, 37, 63, 64, 65, 130, 200)  # around the 64-token chunks of the gated delta rule


def write_tiny_te(path, seed: int = 0, config=TINY_TE):
    """A random tiny encoder file; returns {stored key: tensor as stored}."""
    torch.manual_seed(seed)
    with torch.device("meta"):
        model = MossQwen35TextEncoder(config)
    g = torch.Generator().manual_seed(seed + 1)
    state = {}
    for name, param in model.named_parameters():
        value = torch.randn(param.shape, generator=g) * (0.3 if name.endswith("norm.weight") else 0.08)
        if name.endswith("A_log"):
            state[PREFIX + name] = value.float()  # stored fp32 (not representable in bf16): the loaders cast it
        else:
            state[PREFIX + name] = value.to(torch.bfloat16)
    state["visual.blocks.0.attn.weight"] = torch.randn(8, 8, generator=g).to(torch.bfloat16)  # skipped by both loaders
    save_file({k: v.contiguous() for k, v in state.items()}, str(path), metadata={"format": "pt"})
    return state


def reference_model(path, config=TINY_TE):
    """The cached path's encoder: qwen35.load_moss_text_encoder (bf16 on the CPU)."""
    return load_moss_text_encoder(str(path), device="cpu", dtype=torch.bfloat16, config=config)


def open_weights(path, host: str = "copy", config=TINY_TE):
    return te_live_weights.HostTextEncoderWeights.open(str(path), config=config, host=host, check=False)


def live_encoder(path, host: str = "copy", ring_slots: int = 2, staging=None, config=TINY_TE):
    return te_live.LiveTextEncoder(open_weights(path, host, config), "cpu", ring_slots=ring_slots, staging_threads=2,
                                   staging=staging)


def random_ids(length: int, seed: int, vocab: int = TINY_TE.vocab_size) -> torch.Tensor:
    return torch.randint(0, vocab, (length,), generator=torch.Generator().manual_seed(seed), dtype=torch.long)


class CharTokenizer:
    """MossTokenizer's interface (tokenizer_dir, max_length, encode, __call__, pad_token_id), character level, with
    the real chat template and ComfyUI unescaping (ids < 251: inside the tiny vocab)."""

    pad_token_id = 248044

    def __init__(self, tokenizer_dir=None, max_length: int = 4096):
        self.tokenizer_dir = "<char tokenizer>"
        self.max_length = max_length

    def encode(self, prompt: str):
        return [ord(c) % 250 + 1 for c in format_moss_prompt(unescape_prompt(prompt))][: self.max_length]

    def __call__(self, prompts):
        return [self.encode(p) for p in prompts]


def tokenize_strategy(monkeypatch):
    """The real MossTokenizeStrategy over the character tokenizer."""
    monkeypatch.setattr(strategy, "MossTokenizer", CharTokenizer)
    return strategy.MossTokenizeStrategy()


def decode(ids) -> str:
    """CharTokenizer ids of ASCII text back to the text (the chat template included)."""
    return "".join(chr(int(i) - 1) for i in ids)
