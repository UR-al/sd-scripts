# Qwen3.5-9B text encoder (text-only) for MOSS-Image, pure torch.
# The venv's transformers 4.54 has no qwen3_5, so this re-implements the forward of transformers 5.x
# Qwen3_5TextModel (models/qwen3_5/modeling_qwen3_5.py, Apache License 2.0) for text-only input: the same op order
# and dtypes (bf16 activations, fp32 norms, fp32 chunked gated delta rule), so the cached embeddings match the
# official MOSS pipeline (AutoModel -> Qwen3_5Model.last_hidden_state). Interleaved mRoPE collapses to plain 1-D RoPE
# over the first 64 head dims for text-only positions. Checked against transformers 5.15.1 with a one-off script (fp32
# bit-exact on 4 layers, bf16 cosine >= 0.99999); tests/test_moss_models.py pins the tokenizer golden ids.

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MossQwen35Config:
    """text_encoder/config.json -> text_config of OpenMOSS-Team/MOSS-Image-Base (Qwen3.5-9B)."""

    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    full_attention_interval: int = 4
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    partial_rotary_factor: float = 0.25
    rope_theta: float = 1.0e7
    rms_norm_eps: float = 1e-6

    def is_full_attention(self, index: int) -> bool:
        return (index + 1) % self.full_attention_interval == 0  # layers 3, 7, ..., 31


class Qwen35RMSNorm(nn.Module):
    """``x * rsqrt(mean(x^2) + eps) * (1 + w)`` in fp32, cast back (HF Qwen3_5RMSNorm)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).type_as(x)


class Qwen35RMSNormGated(nn.Module):
    """Direct-scale norm gated by silu(z) (HF Qwen3_5RMSNormGated)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, gate):
        dtype = x.dtype
        y = x.to(torch.float32)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        y = self.weight * y.to(dtype)
        y = y * F.silu(gate.to(torch.float32))
        return y.to(dtype)


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64):
    """HF ``torch_chunk_gated_delta_rule`` (no cache, qk l2norm in kernel). Inputs [B,L,H,D], output [B,L,H,Dv]."""
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]
    batch, heads, length, k_dim = key.shape
    v_dim = value.shape[-1]
    pad = (chunk_size - length % chunk_size) % chunk_size
    query, key, value = (F.pad(t, (0, 0, 0, pad)) for t in (query, key, value))
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    total = length + pad
    query = query * (1 / (query.shape[-1] ** 0.5))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [t.reshape(t.shape[0], t.shape[1], -1, chunk_size, t.shape[-1])
                                         for t in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size).cumsum(dim=-1)
    upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    decay = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay).masked_fill(upper, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    state = torch.zeros(batch, heads, k_dim, v_dim, dtype=value.dtype, device=value.device)
    out = torch.zeros_like(value)
    for i in range(total // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        a = q_i @ k_i.transpose(-1, -2) * decay[:, :, i]
        v_new = v_i - k_cumdecay[:, :, i] @ state
        out[:, :, i] = (q_i * g[:, :, i, :, None].exp()) @ state + a @ v_new
        state = state * g[:, :, i, -1, None, None].exp() + (
            k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
    out = out.reshape(out.shape[0], out.shape[1], -1, out.shape[-1])[:, :, :length]
    return out.transpose(1, 2).contiguous().to(initial_dtype)


class GatedDeltaNet(nn.Module):
    """HF Qwen3_5GatedDeltaNet, prefill path without cache or padding (one unpadded prompt per forward)."""

    def __init__(self, c: MossQwen35Config):
        super().__init__()
        self.num_v_heads, self.num_k_heads = c.linear_num_value_heads, c.linear_num_key_heads
        self.head_k_dim, self.head_v_dim = c.linear_key_head_dim, c.linear_value_head_dim
        self.key_dim, self.value_dim = self.head_k_dim * self.num_k_heads, self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, bias=False, kernel_size=c.linear_conv_kernel_dim,
                                groups=self.conv_dim, padding=c.linear_conv_kernel_dim - 1)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = Qwen35RMSNormGated(self.head_v_dim, eps=c.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, c.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(c.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(c.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(c.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(c.hidden_size, self.num_v_heads, bias=False)

    def forward(self, x):
        batch, length, _ = x.shape
        mixed = self.in_proj_qkv(x).transpose(1, 2)
        z = self.in_proj_z(x).reshape(batch, length, -1, self.head_v_dim)
        b, a = self.in_proj_b(x), self.in_proj_a(x)
        weight = self.conv1d.weight  # [C, 1, K]
        mixed = F.conv1d(mixed.to(weight.dtype), weight, None, padding=weight.shape[-1] - 1,
                         groups=self.conv_dim)[:, :, :length]
        mixed = F.silu(mixed).to(x.dtype).transpose(1, 2)
        query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch, length, -1, self.head_k_dim)
        key = key.reshape(batch, length, -1, self.head_k_dim)
        value = value.reshape(batch, length, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)
        out = chunk_gated_delta_rule(query, key, value, g, beta)
        out = self.norm(out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(out.reshape(batch, length, -1))


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class GatedAttention(nn.Module):
    """HF Qwen3_5Attention: per-head [q | gate] from q_proj, (1+w) q/k norm, partial RoPE (64 of 256), causal GQA."""

    def __init__(self, c: MossQwen35Config):
        super().__init__()
        self.head_dim = c.head_dim
        self.num_heads, self.num_kv = c.num_attention_heads, c.num_key_value_heads
        self.q_proj = nn.Linear(c.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, self.num_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, self.num_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, c.hidden_size, bias=False)
        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=c.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=c.rms_norm_eps)

    def forward(self, x, cos, sin):
        shape = x.shape[:-1]
        query, gate = torch.chunk(self.q_proj(x).view(*shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*shape, -1)
        query = self.q_norm(query.view(*shape, -1, self.head_dim)).transpose(1, 2)
        key = self.k_norm(self.k_proj(x).view(*shape, -1, self.head_dim)).transpose(1, 2)
        value = self.v_proj(x).view(*shape, -1, self.head_dim).transpose(1, 2)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        rot = cos.shape[-1]
        q_rot, q_pass = query[..., :rot], query[..., rot:]
        k_rot, k_pass = key[..., :rot], key[..., rot:]
        query = torch.cat(((q_rot * cos) + (_rotate_half(q_rot) * sin), q_pass), dim=-1)
        key = torch.cat(((k_rot * cos) + (_rotate_half(k_rot) * sin), k_pass), dim=-1)
        groups = self.num_heads // self.num_kv
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        out = F.scaled_dot_product_attention(query, key, value, is_causal=query.shape[2] > 1,
                                             scale=self.head_dim ** -0.5)
        out = out.transpose(1, 2).reshape(*shape, -1).contiguous()
        return self.o_proj(out * torch.sigmoid(gate))


class MLP(nn.Module):
    def __init__(self, c: MossQwen35Config):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, c: MossQwen35Config, index: int):
        super().__init__()
        self.full_attention = c.is_full_attention(index)
        if self.full_attention:
            self.self_attn = GatedAttention(c)
        else:
            self.linear_attn = GatedDeltaNet(c)
        self.mlp = MLP(c)
        self.input_layernorm = Qwen35RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(c.hidden_size, eps=c.rms_norm_eps)

    def forward(self, x, cos, sin):
        h = self.input_layernorm(x)
        h = self.self_attn(h, cos, sin) if self.full_attention else self.linear_attn(h)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x))


class MossQwen35TextEncoder(nn.Module):
    """Qwen3.5-9B language model (keys = checkpoint ``language_model.*`` without the prefix)."""

    def __init__(self, config: MossQwen35Config = MossQwen35Config()):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        rotary = int(config.head_dim * config.partial_rotary_factor)
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, rotary, 2, dtype=torch.float32, device="cpu") / rotary))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @property
    def device(self):
        return self.embed_tokens.weight.device

    @property
    def dtype(self):
        return self.embed_tokens.weight.dtype

    def _rotary(self, length, device, dtype):
        inv_freq = self.inv_freq
        if inv_freq.device.type == "meta" or inv_freq.dtype != torch.float32:
            rotary = inv_freq.shape[0] * 2
            inv_freq = 1.0 / (self.config.rope_theta ** (
                torch.arange(0, rotary, 2, dtype=torch.float32, device="cpu") / rotary))
        positions = torch.arange(length, device=device, dtype=torch.float32)
        freqs = positions[:, None] * inv_freq.to(device)[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids [1, L] (one unpadded prompt) -> last_hidden_state [1, L, 4096] (after the final norm)."""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("MOSS text encoding runs one unpadded prompt per forward (official encode_prompt).")
        x = self.embed_tokens(input_ids.to(self.device))
        cos, sin = self._rotary(x.shape[1], x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


# ---- loading ----------------------------------------------------------------------------------------------------
_PREFIXES = ("language_model.", "model.language_model.", "model.", "")
_SKIP = ("visual.", "mtp.", "lm_head.")


def check_moss_text_encoder(path: str) -> str:
    """Header-only check that ``path`` is the Qwen3.5-9B text encoder; returns the key prefix."""
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for prefix in _PREFIXES:
            if prefix + "embed_tokens.weight" in keys:
                shape = list(f.get_slice(prefix + "embed_tokens.weight").get_shape())
                break
        else:
            raise ValueError(f"No embed_tokens.weight in {path}: not a Qwen3.5 text encoder")
    need = [prefix + "norm.weight", prefix + "layers.31.self_attn.q_proj.weight", prefix + "layers.31.mlp.down_proj.weight",
            prefix + "layers.0.linear_attn.in_proj_qkv.weight"]
    if shape != [248320, 4096] or any(k not in keys for k in need):
        raise ValueError(f"Not the Qwen3.5-9B (MOSS-Image) text encoder: embed {shape}, prefix {prefix!r}: {path}")
    return prefix


def load_moss_text_encoder(path: str, device="cpu", dtype: torch.dtype = torch.bfloat16) -> MossQwen35TextEncoder:
    """Strict load of the 426 language-model tensors (vision tower skipped), straight to ``device``."""
    from safetensors import safe_open

    prefix = check_moss_text_encoder(path)
    state = {}
    with safe_open(path, framework="pt", device=str(torch.device(device))) as f:
        for key in f.keys():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            if name.startswith(_SKIP):
                continue
            state[name] = f.get_tensor(key).to(dtype)
    with torch.device("meta"):
        model = MossQwen35TextEncoder()
    model.requires_grad_(False)
    model.load_state_dict(state, strict=True, assign=True)  # inv_freq is a non-persistent fp32 CPU buffer
    logger.info(f"MOSS text encoder (Qwen3.5-9B) loaded from {path}: {len(state)} tensors, {dtype}, {device}")
    return model.eval()


# ---- tokenizer + prompt template ---------------------------------------------------------------------------------
# transformers-5 Qwen2Tokenizer.PRETOKENIZE_REGEX: what the official pipeline effectively uses (it ignores the
# tokenizer.json pre-tokenizer). The Forge Qwen3.5 copy carries a \p{M} variant that differs on combining marks.
QWEN2_PRETOKENIZE_REGEX = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")
EXTRA_SPECIALS = ["<|audio_start|>", "<|audio_end|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>",
                  "<tts_text_bos_single>", "<|audio_pad|>"]  # ids 248070..248076 in the MOSS tokenizer.json
EXPECTED_IDS = {"<|endoftext|>": 248044, "<|im_start|>": 248045, "<|im_end|>": 248046, "<think>": 248068,
                "</think>": 248069, "<|audio_pad|>": 248076}
TOKENIZER_DIR_ENV = "MOSS_TOKENIZER_DIR"
FALLBACK_TOKENIZER_DIRS: tuple = ()  # no machine-specific default: pass --moss_tokenizer_dir or set the env var
MAX_TOKENS = 4096


def unescape_prompt(text: str) -> str:
    r"""``\(`` -> ``(`` and ``\)`` -> ``)`` like ComfyUI's tokenizer (MOSS node); the model was trained on raw text."""
    return text.replace("\\(", "(").replace("\\)", ")")


def format_moss_prompt(prompt: str) -> str:
    """chat_template.jinja with add_generation_prompt=True, enable_thinking=True for one user message."""
    return "<|im_start|>user\n" + (prompt or "").strip() + "<|im_end|>\n<|im_start|>assistant\n<think>\n"


def resolve_tokenizer_dir(tokenizer_dir: Optional[str] = None) -> str:
    if tokenizer_dir:
        if not os.path.isfile(os.path.join(tokenizer_dir, "tokenizer.json")):
            raise FileNotFoundError(f"tokenizer.json not found in --moss_tokenizer_dir {tokenizer_dir}")
        return tokenizer_dir
    for candidate in (os.environ.get(TOKENIZER_DIR_ENV), *FALLBACK_TOKENIZER_DIRS):
        if candidate and os.path.isfile(os.path.join(candidate, "tokenizer.json")):
            return candidate
    raise FileNotFoundError("No Qwen3.5 tokenizer.json found: pass --moss_tokenizer_dir (MOSS-Image tokenizer/ folder)")


class MossTokenizer:
    """Qwen3.5 BPE via the ``tokenizers`` library with the official (Qwen2) pre-tokenizer; no transformers needed."""

    pad_token_id = 248044

    def __init__(self, tokenizer_dir: Optional[str] = None, max_length: int = MAX_TOKENS):
        from tokenizers import AddedToken, Regex, Tokenizer, pre_tokenizers

        if not 1 <= max_length <= MAX_TOKENS:
            raise ValueError(f"max_length must be in 1..{MAX_TOKENS}")
        self.tokenizer_dir = resolve_tokenizer_dir(tokenizer_dir)
        tok = Tokenizer.from_file(os.path.join(self.tokenizer_dir, "tokenizer.json"))
        tok.no_truncation()
        tok.no_padding()
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(QWEN2_PRETOKENIZE_REGEX), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        missing = [t for t in EXTRA_SPECIALS if tok.token_to_id(t) is None]
        if missing:
            tok.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in missing])
        for token, expected in EXPECTED_IDS.items():
            if tok.token_to_id(token) != expected:
                raise ValueError(f"Not the Qwen3.5 / MOSS vocab: {token} -> {tok.token_to_id(token)} (expected {expected})")
        self.tok = tok
        self.max_length = max_length

    def encode(self, prompt: str) -> List[int]:
        """Prompt -> token ids of the chat-templated prompt, right-truncated at max_length (official encode_prompt)."""
        return self.tok.encode(format_moss_prompt(unescape_prompt(prompt)), add_special_tokens=True).ids[: self.max_length]

    def __call__(self, prompts: Sequence[str]) -> List[List[int]]:
        return [self.encode(p) for p in prompts]


@torch.no_grad()
def encode_prompts(model: MossQwen35TextEncoder, tokenizer: MossTokenizer, prompts: Sequence[str]) -> List[torch.Tensor]:
    """One unpadded forward per prompt (official). Returns a list of [T_i, 4096] tensors in the model dtype (CPU)."""
    rows = []
    with torch.autocast(device_type=model.device.type, enabled=False):
        for ids in tokenizer(prompts):
            hidden = model(torch.tensor([ids], dtype=torch.long, device=model.device))
            rows.append(hidden[0].to("cpu"))
    return rows
