"""CPU tests for the MOSS-Image port (no GPU, no model files needed except where noted).

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe tests/test_moss_models.py   (or with pytest if installed)
The official reference is OpenMOSS pipeline_moss_image.py; set MOSS_REF_PIPELINE to a local copy to run the parity test.
"""
import contextlib
import importlib.util
import os

import torch

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


def need_ref():
    if not REF:
        skip("MOSS_REF_PIPELINE not set")


@contextlib.contextmanager
def raises(exc):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"{exc.__name__} not raised")
TINY = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=48, freq_dim=16, latent_channels=128, rope_base=10000.0)


def _load_ref():
    spec = importlib.util.spec_from_file_location("ref_pipeline_moss_image", REF)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_pair(seed=0):
    torch.manual_seed(seed)
    ref = _load_ref().MOSSImageTransformer(**TINY).float()
    for p in ref.parameters():  # non-trivial norms / modulation
        p.data.normal_(0, 0.2)
    ours = MossTransformer(**TINY, attn_mode="torch").float()
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


def test_batched_forward_matches_official_per_sample_loop():
    need_ref()
    ref, ours = _tiny_pair()
    x, t, text, mask = _batch()
    want = ref(x, t, text, mask, return_dict=False)[0]
    got = ours(x, t, text, mask)
    assert (want - got).abs().max().item() < 1e-5


def test_gradients_match_official():
    need_ref()
    ref, ours = _tiny_pair()
    x, t, text, mask = _batch()
    ref(x, t, text, mask, return_dict=False)[0].square().mean().backward()
    ours(x, t, text, mask).square().mean().backward()
    ref_grads = dict(ref.named_parameters())
    worst = max((p.grad - ref_grads[n].grad).abs().max().item() for n, p in ours.named_parameters() if p.grad is not None)
    assert worst < 1e-5


def test_skip_layers_and_unaligned_mask_match_official():
    need_ref()
    ref, ours = _tiny_pair()
    x, t, text, mask = _batch()
    mask = mask.flip(1)  # valid tokens at the end: the official forward accepts any position
    text = text.flip(1)
    want = ref(x, t, text, mask, return_dict=False, skip_layers=[1])[0]
    got = ours(x, t, text, mask, skip_layers=[1])
    assert (want - got).abs().max().item() < 1e-5


def test_gradient_checkpointing_same_grads():
    need_ref()
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
    "Café noir, नमस्ते, สวัสดีครับ, が":
        [248045, 846, 198, 34, 2492, 933, 53695, 11, 149518, 84237, 75681, 196965, 33228, 11, 148340, 35982,
         22547, 149082, 27399, 148323, 80665, 11, 181509, 248046, 198, 248045, 74455, 198, 248068, 198],
}


def test_tokenizer_golden_ids():
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
    for name, fn in [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]:
        try:
            fn()
            print(f"PASS {name}")
        except Skip as e:
            print(f"SKIP {name}: {e}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print("failed:", failed)
    sys.exit(1 if failed else 0)
