"""CPU tests for the MOSS-Image port (no GPU, no model files needed except where noted).

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe tests/test_moss_models.py   (or with pytest if installed)
The parity tests compare our batched DiT with a per-sample reference. By default that is ReferenceMossImage below
(the official per-sample loop, copied from upstream sd-scripts' tests/test_moss_image_models.py, which copied it from
OpenMOSS pipeline_moss_image.py, Apache-2.0); set MOSS_REF_PIPELINE to a local copy of the official
pipeline_moss_image.py to run them against the official module itself.
"""
import contextlib
import importlib.util
import os

import torch
from torch import nn
from torch.nn import functional as F

try:
    import pytest
except ImportError:  # the training venv has no pytest: `python tests/test_moss_models.py` runs every test
    pytest = None

from library.moss.models import MossTransformer
from library.moss.qwen35 import MossTokenizer, format_moss_prompt, unescape_prompt

REF = os.environ.get("MOSS_REF_PIPELINE")


class Skip(Exception):
    pass


def skip(reason):
    if pytest is not None:
        pytest.skip(reason)
    raise Skip(reason)


@contextlib.contextmanager
def raises(exc):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"{exc.__name__} not raised")


TINY = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=48, freq_dim=16, latent_channels=128, rope_base=10000.0)


# region reference implementation (per-sample loop; upstream sd-scripts tests/test_moss_image_models.py, copied from the
# official pipeline_moss_image.py, OpenMOSS Team, Apache-2.0)


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.proj["q"].weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](_attention(q, k, self.proj["v"](y), rope, self.heads)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](_situ(hidden, gate)) * gate_f[:, None]


class _RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        exponent = torch.arange(self.num_bands, dtype=torch.float32) / self.num_bands
        self.register_buffer("bands", 1 / (self.base**exponent))

    def forward(self, height, width, text_length, device):
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class ReferenceMossImage(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, num_layers, text_dim, freq_dim, latent_channels, rope_base):
        super().__init__()
        self.freq_dim = freq_dim
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([_Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def forward(self, hidden_states, timestep, encoder_hidden_states, mask, skip_layers=()):
        skip_layers = frozenset(skip_layers)
        batch, _, height, width = hidden_states.shape
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1).expand(batch)
        half = self.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding).float()
        mod = self.shared_block_modulation(time).view(batch, 6, -1)
        head = self.video_head_modulation(time).view(batch, 2, -1)
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index : index + 1, mask[index].bool()]
            video = hidden_states[index : index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.contiguous())
            rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width :])
            x = torch.cat((video, text), 1)
            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip_layers:
                    x = layer(x, rope, mod[index : index + 1])
            x = self.final_norm_video(x[:, : height * width])
            x = x * (1 + head[index : index + 1, 1, None]) + head[index : index + 1, 0, None]
            x = self.final_linear_video(x)
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        return torch.cat(outputs)


# endregion


def _reference_class():
    if not REF:
        return ReferenceMossImage
    spec = importlib.util.spec_from_file_location("ref_pipeline_moss_image", REF)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MOSSImageTransformer


def _ref_forward(ref, x, t, text, mask, skip_layers=None):
    if REF:  # the official module returns a dataclass unless return_dict=False
        kwargs = {"skip_layers": list(skip_layers)} if skip_layers else {}
        return ref(x, t, text, mask, return_dict=False, **kwargs)[0]
    return ref(x, t, text, mask, skip_layers=skip_layers or ())


def _tiny_pair(seed=0, split_attn=False):
    torch.manual_seed(seed)
    ref = _reference_class()(**TINY).float()
    for p in ref.parameters():  # non-trivial norms / modulation
        p.data.normal_(0, 0.2)
    ours = MossTransformer(**TINY, attn_mode="torch", split_attn=split_attn).float()
    ours.load_state_dict(ref.state_dict(), strict=True)
    return ref, ours


def _batch(seed=1, lengths=(5, 9, 2), h=3, w=4):
    g = torch.Generator().manual_seed(seed)
    b, s = len(lengths), max(lengths) + 2
    x = torch.randn(b, 128, h, w, generator=g)
    t = torch.tensor([937.0, 12.0, 500.0])[:b]
    text = torch.randn(b, s, TINY["text_dim"], generator=g)
    mask = torch.zeros(b, s, dtype=torch.bool)
    for i, n in enumerate(lengths):
        mask[i, :n] = True
    text[~mask] = 100.0  # garbage in padding must not leak
    return x, t, text, mask


def test_batched_forward_matches_per_sample_reference():
    """Padded batch (garbage in the padding, extra padding beyond the batch max) == the official per-sample loop,
    with the batched masked attention and with split_attn (one sample at a time)."""
    for split_attn in (False, True):
        ref, ours = _tiny_pair(split_attn=split_attn)
        x, t, text, mask = _batch()
        want = _ref_forward(ref, x, t, text, mask)
        got = ours(x, t, text, mask)
        assert (want - got).abs().max().item() < 1e-5, split_attn


def test_gradients_match_reference():
    ref, ours = _tiny_pair()
    x, t, text, mask = _batch()
    _ref_forward(ref, x, t, text, mask).square().mean().backward()
    ours(x, t, text, mask).square().mean().backward()
    ref_grads = dict(ref.named_parameters())
    worst = max((p.grad - ref_grads[n].grad).abs().max().item() for n, p in ours.named_parameters() if p.grad is not None)
    assert worst < 1e-5


def test_skip_layers_and_unaligned_mask_match_reference():
    ref, ours = _tiny_pair()
    x, t, text, mask = _batch()
    mask = mask.flip(1)  # valid tokens at the end: the official forward accepts any position
    text = text.flip(1)
    want = _ref_forward(ref, x, t, text, mask, skip_layers=[1])
    got = ours(x, t, text, mask, skip_layers=[1])
    assert (want - got).abs().max().item() < 1e-5
    assert not torch.allclose(got, ours(x, t, text, mask))  # the skipped block does change the output


def test_forward_without_mask_is_full_length():
    ref, ours = _tiny_pair()
    x, t, text, mask = _batch()
    want = _ref_forward(ref, x, t, text, torch.ones_like(mask))
    got = ours(x, t, text, None)
    assert (want - got).abs().max().item() < 1e-5


def test_gradient_checkpointing_same_grads():
    _, a = _tiny_pair()
    _, b = _tiny_pair()
    b.enable_gradient_checkpointing()
    a.train(), b.train()
    x, t, text, mask = _batch()
    a(x, t, text, mask).square().mean().backward()
    b(x, t, text, mask).square().mean().backward()
    pa = dict(a.named_parameters())
    assert max((p.grad - pa[n].grad).abs().max().item() for n, p in b.named_parameters()) < 1e-6


def test_rope_bands_survive_bf16_cast():
    model = MossTransformer(**TINY)
    model.to(torch.bfloat16)
    assert model.rope.bands.dtype == torch.float32


def test_lora_target_names_match_checkpoint_keys():
    model = MossTransformer(**TINY)
    names = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    assert {"layers.0.proj.q", "layers.2.proj.ffn_up_gate", "layers.1.proj.ffn_down",
            "text_embedder.blocks.0.qkv", "text_embedder.proj"} <= names


def test_empty_text_rejected():
    model = MossTransformer(**TINY)
    x, t, text, mask = _batch()
    mask[1] = False
    with raises(ValueError):
        model(x, t, text, mask)


def test_prompt_template_and_unescape():
    assert format_moss_prompt("  a  ") == "<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n<think>\n"
    assert unescape_prompt(r"star \(symbol\)") == "star (symbol)"


GOLDEN = {
    "": [248045, 846, 198, 248046, 198, 248045, 74455, 198, 248068, 198],
    "  hello  ": [248045, 846, 198, 14556, 248046, 198, 248045, 74455, 198, 248068, 198],
    "Café noir, नमस्ते, สวัสดีครับ, が":
        [248045, 846, 198, 34, 2492, 933, 53695, 11, 149518, 84237, 75681, 196965, 33228, 11, 148340, 35982,
         22547, 149082, 27399, 148323, 80665, 11, 181509, 248046, 198, 248045, 74455, 198, 248068, 198],
}


def test_tokenizer_golden_ids():
    """Needs the real Qwen3.5 tokenizer (MOSS_TOKENIZER_DIR); skipped otherwise, never silently passed."""
    try:
        tok = MossTokenizer()
    except FileNotFoundError as e:
        skip(str(e))
    for prompt, ids in GOLDEN.items():
        assert tok.encode(prompt) == ids, prompt
    # escaped and raw parentheses give the same ids (ComfyUI strips the backslash)
    assert tok.encode(r"shrug \(clothing\)") == tok.encode("shrug (clothing)")


if __name__ == "__main__":
    import sys
    import traceback

    failed = 0
    skips = (Skip,) + ((pytest.skip.Exception,) if pytest is not None else ())
    for name, fn in [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]:
        try:
            fn()
            print(f"PASS {name}")
        except skips as e:
            print(f"SKIP {name}: {e}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print("failed:", failed)
    sys.exit(1 if failed else 0)
