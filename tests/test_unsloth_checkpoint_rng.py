"""CPU tests for the RNG handling of the Unsloth offloaded checkpoint (library/anima_models.py, also used by MOSS): the
recomputation in backward replays the forward's RNG states, so gradients through dropout are bit-identical to the
non-checkpointed ones, and the caller's RNG stream after backward is where plain autograd leaves it (as with
torch.utils.checkpoint).

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe tests/test_unsloth_checkpoint_rng.py   (or with pytest if installed)
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from library import anima_models, attention
from library.anima_models import Block, unsloth_checkpoint

# The three LyCORIS knobs that draw random numbers inside a block (all 0.0 in the user's configs)
LOKR_DROPOUTS = (dict(rank_dropout=0.3), dict(module_dropout=0.5), dict(dropout=0.3))
MOSS_TINY = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=48, freq_dim=16, latent_channels=128,
                 rope_base=10000.0)


def _grads(tensors):
    return [None if t.grad is None else t.grad.clone() for t in tensors]


def _assert_same(ref, other, what):
    (out_a, grads_a, rng_a), (out_b, grads_b, rng_b) = ref, other
    assert torch.equal(out_a, out_b), f"{what}: output"
    assert len(grads_a) == len(grads_b) and any(g is not None for g in grads_a), what
    for i, (a, b) in enumerate(zip(grads_a, grads_b)):
        assert (a is None) == (b is None) and (a is None or torch.equal(a, b)), f"{what}: grad {i}"
    assert torch.equal(rng_a, rng_b), f"{what}: CPU RNG state after backward"


def _dropout_run(mode):
    torch.manual_seed(0)
    lin = nn.Linear(8, 8)
    drop = nn.Dropout(0.5)

    def f(h):
        return drop(lin(h))

    x = torch.randn(4, 8, generator=torch.Generator().manual_seed(1)).requires_grad_(True)
    torch.manual_seed(123)
    if mode == "plain":
        y = f(x)
    elif mode == "unsloth":
        y = unsloth_checkpoint(f, x)
    else:
        y = torch_checkpoint(f, x, use_reentrant=False)
    y.square().sum().backward()
    return y.detach(), _grads([lin.weight, lin.bias, x]), torch.get_rng_state()


def test_dropout_grads_and_rng_stream_match_plain():
    """nn.Dropout(0.5): same output, same grads (the recomputation draws the forward's mask), and the RNG stream after
    backward is where plain autograd leaves it. torch.utils.checkpoint is the reference for both."""
    plain = _dropout_run("plain")
    _assert_same(plain, _dropout_run("torch"), "torch.utils.checkpoint")  # the behaviour being mirrored
    _assert_same(plain, _dropout_run("unsloth"), "unsloth_checkpoint")


def test_cpu_inputs_never_touch_cuda_rng():
    def boom(*a, **k):
        raise AssertionError("CUDA RNG touched for CPU inputs")

    saved = torch.cuda.get_rng_state, torch.cuda.set_rng_state
    torch.cuda.get_rng_state = torch.cuda.set_rng_state = boom
    try:
        assert anima_models._cuda_rng_devices(torch.zeros(1), (torch.zeros(1), torch.zeros(1)), None, 3) == []
        _dropout_run("unsloth")
    finally:
        torch.cuda.get_rng_state, torch.cuda.set_rng_state = saved


def _anima_lokr_run(drop, unsloth):
    """Anima Block + LyCORIS LoKr/DoRA with the network_args of config_anima38.toml, plus one dropout knob."""
    from lycoris import kohya

    torch.manual_seed(0)
    root = nn.Module()
    root.block = Block(x_dim=64, context_dim=48, num_heads=4)
    with torch.no_grad():
        for p in root.parameters():
            p.normal_(0, 0.05)
    root.requires_grad_(False)
    net = kohya.create_network(1.0, 10000, 10000, None, [], root, algo="lokr", preset="anima", factor=4,
                               full_matrix=True, dora_wd=True, wd_on_output=True, **drop)
    net.apply_to([], root, False, True)
    with torch.no_grad():
        g = torch.Generator().manual_seed(7)
        for p in net.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.02)
    root.train()
    if unsloth:
        root.block.enable_gradient_checkpointing(unsloth_offload=True)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 1, 3, 4, 64, generator=g).requires_grad_(True)
    emb = torch.randn(2, 1, 64, generator=g)
    crossattn = torch.randn(2, 7, 48, generator=g)
    rope = (torch.randn(12, 1, 1, 16, generator=g), torch.randn(12, 1, 1, 16, generator=g))
    params = attention.AttentionParams.create_attention_params("torch", False)
    torch.manual_seed(123)
    y = root.block(x, emb, crossattn, params, False, rope)
    y.square().mean().backward()
    return y.detach(), _grads(list(net.parameters()) + [x]), torch.get_rng_state()


def test_anima_block_lokr_dropouts_match_plain():
    for drop in LOKR_DROPOUTS:
        _assert_same(_anima_lokr_run(drop, False), _anima_lokr_run(drop, True), f"Anima Block LoKr {drop}")


def _moss_lokr_run(drop, unsloth):
    """Tiny MossTransformer + LyCORIS LoKr/DoRA preset=moss (config_moss.toml network_args): MOSS imports
    unsloth_checkpoint from anima_models, so it shares the fix."""
    from lycoris import kohya
    from library.moss.lycoris_presets import register_moss_lycoris_presets
    from library.moss.models import MossTransformer

    register_moss_lycoris_presets()
    torch.manual_seed(0)
    model = MossTransformer(**MOSS_TINY)
    model.requires_grad_(False)
    net = kohya.create_network(1.0, 10000, 10000, None, [], model, algo="lokr", preset="moss", factor=4,
                               full_matrix=True, dora_wd=True, wd_on_output=False, **drop)
    net.apply_to([], model, False, True)
    with torch.no_grad():
        g = torch.Generator().manual_seed(7)
        for p in net.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.02)
    model.train()
    if unsloth:
        model.enable_gradient_checkpointing(unsloth_offload=True)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 128, 2, 2, generator=g).requires_grad_(True)
    text = torch.randn(2, 4, MOSS_TINY["text_dim"], generator=g)
    torch.manual_seed(123)
    y = model(x, torch.tensor([10.0, 20.0]), text, torch.ones(2, 4, dtype=torch.bool))
    y.square().mean().backward()
    return y.detach(), _grads(list(net.parameters()) + [x]), torch.get_rng_state()


def test_moss_lokr_dropouts_match_plain():
    for drop in (dict(rank_dropout=0.3), dict(module_dropout=0.5)):
        _assert_same(_moss_lokr_run(drop, False), _moss_lokr_run(drop, True), f"MOSS LoKr {drop}")


if __name__ == "__main__":
    import sys
    import traceback

    failed = 0
    for name, fn in [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print("failed:", failed)
    sys.exit(1 if failed else 0)
