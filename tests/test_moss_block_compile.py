"""CPU tests for --moss_block_compile (library/moss/block_compile.py): per-block torch.compile of the MOSS-Image DiT with
the LyCORIS fixed paths inside the graphs - the --anima_block_compile design (tests/test_anima_block_compile.py) applied
to MossTransformer.

Checked on CPU with the aot_eager backend (the same dynamo tracing and AOTAutograd joint graph as inductor, no codegen) on
a 3-block toy MossTransformer with a LoKr + DoRA network built the way train_network.py builds it (lycoris.kohya +
library/lycoris_algo_fixes + library/lycoris_fast, preset moss):
* the mask-free joint attention params MossTransformer.forward uses under block compile (no padding) give the same output
  as the masked params;
* compiled forward + backward reproduce the eager output and every network gradient, with and without gradient
  checkpointing;
* different caption lengths and image sizes share the compiled graphs (soft token marks, no per-length ints);
* pause() and eval / no_grad run the eager blocks; restore_moss_blocks puts the class methods back;
* validate_moss_block_compile_args refuses RamTorch, block swap, --torch_compile, stock LyCORIS and DyLoRA.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -B -m pytest tests/test_moss_block_compile.py
"""
import argparse
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
import torch._dynamo
from torch._dynamo.backends.registry import lookup_backend

from library import lycoris_args, lycoris_fast
from library import lycoris_algo_fixes as laf
from library.moss import block_compile as mbc
from library.moss.lycoris_presets import register_moss_lycoris_presets
from library.moss.models import MossTransformer, enable_elementwise_compile
from library.moss.train_utils import validate_moss_block_compile_args

torch.set_num_threads(4)

TINY = dict(dim=192, ffn_dim=512, num_heads=4, num_layers=3, text_dim=64, freq_dim=32, latent_channels=128)
LOKR = ["algo=lokr", "preset=moss", "factor=4", "full_matrix=True", "decompose_both=False", "dora_wd=True",
        "wd_on_output=True"]


@pytest.fixture(autouse=True)
def _env():
    lycoris_fast.install_fast_lokr_forward()
    laf.apply_module_patches()
    register_moss_lycoris_presets()
    enable_elementwise_compile(False)  # production: off under block compile (prepare_moss_runtime)
    torch._dynamo.reset()
    yield
    laf.set_block_compile(False)
    torch._dynamo.reset()


def build(block_compile: bool = True, seed: int = 0):
    """Toy MOSS DiT (frozen bf16, train mode) + LoKr+DoRA applied as train_network.py does it. Returns (model, network)."""
    import lycoris.kohya as kohya

    torch.manual_seed(seed)
    model = MossTransformer(**TINY).to(torch.bfloat16)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.05)
    model.requires_grad_(False)
    model.train()
    args = argparse.Namespace(network_args=list(LOKR), network_dim=10000, network_alpha=10000,
                              network_module="lycoris.kohya", network_dropout=None, lycoris_stock=False,
                              lycoris_algo_fixes=False, network_weights=None)
    net_kwargs = lycoris_args.fix_lycoris_bool_kwargs(args.network_module, dict(i.split("=", 1) for i in args.network_args))
    create_kwargs = laf.prepare_network_kwargs("lokr", dict(net_kwargs))
    create_kwargs.setdefault("dropout", 0.0)
    laf.set_block_compile(block_compile)
    network = kohya.create_network(1.0, args.network_dim, args.network_alpha, None, None, model, neuron_dropout=None,
                                   **create_kwargs)
    if hasattr(network, "prepare_network"):
        network.prepare_network(args)
    network.apply_to(None, model, False, True)
    laf.after_network_created(network, args)
    network.requires_grad_(True)
    network.train()
    # LoKr w2 starts at zero (delta 0): seed it so the gradients of every factor depend on the forward
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for m in laf._all_loras(network):
            for name, p in m.named_parameters():
                if p.is_floating_point() and not bool(p.any()):
                    p.copy_(torch.randn(p.shape, generator=g).to(p.dtype) * 0.05)
    laf.prime_module_caches(network)
    return model, network


def inputs(hw=(4, 4), text_len=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 128, *hw, generator=g).to(torch.bfloat16)
    t = torch.tensor([500.0])
    text = torch.randn(1, text_len, TINY["text_dim"], generator=g).to(torch.bfloat16)
    return x, t, text, torch.ones(1, text_len, dtype=torch.bool)


def step(model, network, x, t, text, mask, checkpoint=False):
    network.zero_grad(set_to_none=True)
    if checkpoint:
        x = x.detach().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = model(x, t, text, mask)
    y.float().square().mean().backward()
    grads = {n: p.grad.detach().float().clone() for n, p in network.named_parameters() if p.grad is not None}
    assert grads, "the network got no gradients"
    return y.detach().float().clone(), grads


class Counter:
    """aot_eager backend that counts the graphs dynamo hands it (one per compile)."""

    def __init__(self):
        self.graphs = 0
        self.inner = lookup_backend("aot_eager")

    def __call__(self, gm, example_inputs):
        self.graphs += 1
        return self.inner(gm, example_inputs)


def assert_close(ref, new, what):
    (y0, g0), (y1, g1) = ref, new
    assert torch.allclose(y0, y1, rtol=2 ** -7, atol=1e-3), f"{what}: output max |d| {(y0 - y1).abs().max():.3e}"
    assert g0.keys() == g1.keys()
    for n in g0:
        scale = g0[n].abs().max().clamp_min(1e-6)
        assert ((g0[n] - g1[n]).abs().max() / scale) < 2e-2, f"{what}: grad {n} differs"


def test_mask_free_joint_params_match_the_masked_attention():
    model, _ = build(block_compile=False)
    model.eval()
    x, t, text, mask = inputs()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        masked = model(x, t, text, mask).float()
        model._block_compile = True  # forward's no-padding branch only
        free = model(x, t, text, mask).float()
        model._block_compile = False
    assert torch.equal(masked, free)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_compiled_blocks_match_eager(checkpoint):
    model, network = build()
    if checkpoint:
        model.enable_gradient_checkpointing()
    batch = inputs()
    ref = step(model, network, *batch, checkpoint=checkpoint)
    state = mbc.compile_moss_blocks(model, backend="aot_eager", isolate_cache=False)
    new = step(model, network, *batch, checkpoint=checkpoint)
    assert state.compiled_calls >= len(model.layers)
    assert_close(ref, new, f"checkpoint={checkpoint}")
    mbc.restore_moss_blocks(model)


def test_caption_lengths_and_sizes_share_the_graphs():
    model, network = build()
    counter = Counter()
    mbc.compile_moss_blocks(model, backend=counter, isolate_cache=False)
    step(model, network, *inputs((4, 4), 6))
    first = counter.graphs
    for hw, n in [((4, 4), 9), ((4, 6), 6), ((6, 6), 11), ((4, 4), 13)]:
        step(model, network, *inputs(hw, n))
    # soft marks: the first call may specialise, the second length makes the axis symbolic - then no more graphs
    assert counter.graphs <= first + 2, (first, counter.graphs)
    mbc.restore_moss_blocks(model)


def test_pause_eval_and_restore_run_the_eager_blocks():
    model, network = build()
    state = mbc.compile_moss_blocks(model, backend="aot_eager", isolate_cache=False)
    x, t, text, mask = inputs()
    with mbc.pause(), torch.no_grad():
        model(x, t, text, mask)
    assert state.compiled_calls == 0 and state.eager_calls == len(model.layers)
    model.eval()
    with torch.no_grad():
        model(x, t, text, mask)
    assert state.compiled_calls == 0
    mbc.restore_moss_blocks(model)
    assert "_forward" not in model.layers[0].__dict__ and not model._block_compile


def _args(**changes):
    base = dict(moss_block_compile=True, use_ramtorch=False, blocks_to_swap=None, torch_compile=False,
                network_module="lycoris.kohya", network_args=list(LOKR), moss_stock_lycoris=False, lycoris_stock=False,
                moss_block_compile_recompile_limit=64)
    base.update(changes)
    return types.SimpleNamespace(**base)


def test_validation_accepts_the_production_setup_and_off_is_unchecked():
    validate_moss_block_compile_args(_args())
    validate_moss_block_compile_args(_args(moss_block_compile=False, use_ramtorch=True, blocks_to_swap=4))


@pytest.mark.parametrize("change, word", [
    (dict(use_ramtorch=True), "RamTorch"),
    (dict(blocks_to_swap=4), "blocks_to_swap"),
    (dict(torch_compile=True), "torch_compile"),
    (dict(moss_stock_lycoris=True), "stock"),
    (dict(network_args=["algo=dylora"]), "DyLoRA"),
    (dict(moss_block_compile_recompile_limit=4), "recompile_limit"),
])
def test_validation_refuses(change, word):
    with pytest.raises(ValueError, match=word):
        validate_moss_block_compile_args(_args(**change))
