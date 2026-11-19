"""The block compilers (library/anima_block_compile.py, library/moss/block_compile.py) must pass their recompile limit per
region, not only as the global torch._dynamo.config value.

11-17 (GPU, library/elementwise_compile.py): torch 2.14 keeps torch._dynamo / torch._inductor config overrides per thread
(ContextVar). raise_recompile_limit sets the limit in the MAIN thread, but a checkpointed block's recompute runs in the
autograd engine's thread, which has its own thread count (a guard: the recompute compiles graphs of its own) and still
saw the default limit 8; past it the recompute ran eager beside a compiled first pass. On CPU the autograd engine runs
the backward in the calling thread, so the cross-thread tests call the compiled blocks from a fresh thread directly.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest -p no:cacheprovider tests/test_block_compile_region_limit.py
"""
import inspect
import os
import threading

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
import torch._dynamo
import torch._dynamo.config as dcfg

from library import anima_block_compile as bc
from library.moss import block_compile as mbc

HAS_REGION_LIMIT = "recompile_limit" in inspect.signature(torch.compile).parameters
VARIANTS = 12  # thread count (3) x grad mode (2) x input requires_grad (2): more graphs than the default limit 8


class _AnimaBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def _forward(self, x):
        return self.lin(x).sin()


class _Anima(torch.nn.Module):
    attn_mode = "torch"

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_AnimaBlock(), _AnimaBlock()])


class _MossBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def _forward(self, x, rope):
        return self.lin(x) * rope.cos()


class _Moss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_MossBlock(), _MossBlock()])


def _compile_anima(backend, recompile_limit, mode=None):
    model = _Anima().train()
    bc.compile_blocks(model, bands=[], backend=backend, mode=mode, dynamic="static", recompile_limit=recompile_limit,
                      isolate_cache=False)
    return model, model.blocks[0], lambda x: (x,)


def _compile_moss(backend, recompile_limit, mode=None):
    model = _Moss().train()
    mbc.compile_moss_blocks(model, backend=backend, mode=mode, recompile_limit=recompile_limit, isolate_cache=False)
    return model, model.layers[0], lambda x: (x, torch.ones(1, 4, 8))


COMPILERS = {"anima": (_compile_anima, bc.restore_blocks), "moss": (_compile_moss, mbc.restore_moss_blocks)}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # compile_blocks raises the global (this thread's) limits; put them back after each test
    for name in ("recompile_limit", "accumulated_recompile_limit"):
        monkeypatch.setattr(dcfg, name, getattr(dcfg, name))
    threads = torch.get_num_threads()
    torch._dynamo.reset()
    yield
    torch.set_num_threads(threads)
    torch._dynamo.reset()


def test_block_compile_kwargs():
    kwargs = bc.block_compile_kwargs("inductor", None, 128)
    assert kwargs["backend"] == "inductor" and kwargs["dynamic"] is False and "mode" not in kwargs
    assert "options" not in kwargs  # torch.compile takes mode or options, not both; the block compilers set no options
    assert bc.block_compile_kwargs("aot_eager", "max-autotune", 64)["mode"] == "max-autotune"
    if HAS_REGION_LIMIT:
        assert kwargs["recompile_limit"] == 128
    else:
        assert "recompile_limit" not in kwargs


@pytest.mark.parametrize("which", sorted(COMPILERS))
@pytest.mark.parametrize("global_limit, requested, expected", [(8, 128, 128), (200, 64, 200)])
def test_compile_passes_a_per_region_limit(which, global_limit, requested, expected, monkeypatch):
    """torch.compile gets the limit raise_recompile_limit settled on (never lower than the global value)."""
    seen = []

    def fake_compile(fn, **kwargs):
        seen.append(kwargs)
        return fn

    monkeypatch.setattr(dcfg, "recompile_limit", global_limit)
    monkeypatch.setattr(torch, "compile", fake_compile)
    compile_fn, restore = COMPILERS[which]
    model, _, _ = compile_fn("inductor", requested, mode="max-autotune")
    try:
        assert seen, "torch.compile was not called"
        for kwargs in seen:
            assert kwargs["backend"] == "inductor" and kwargs["dynamic"] is False
            assert kwargs["mode"] == "max-autotune" and "options" not in kwargs
            if HAS_REGION_LIMIT:
                assert kwargs.get("recompile_limit") == expected
            else:
                assert "recompile_limit" not in kwargs
        assert dcfg.recompile_limit == expected  # the global raise stays (torch without the per-region argument)
    finally:
        restore(model)


def _graphs_in_a_fresh_thread(block, args_of, calls):
    """Call the compiled block with VARIANTS guard variants in a new thread (no config override of its own, like the
    autograd engine's). Returns the number of graphs dynamo compiled there, or the exception the thread raised."""
    out = []

    def run():
        try:
            for i in range(VARIANTS):
                torch.set_num_threads(1 + i % 3)
                with torch.set_grad_enabled(i % 2 == 0):
                    block._forward(*args_of(torch.randn(1, 4, 8, requires_grad=(i // 6 == 0))))
            out.append(calls.count("fresh"))
        except Exception as error:  # noqa: BLE001 - reported by the test
            out.append(error)

    t = threading.Thread(target=run, name="fresh")
    t.start()
    t.join()
    return out[0]


def _counting_backend():
    calls = []

    def backend(gm, example_inputs, **kwargs):
        calls.append(threading.current_thread().name)
        return gm.forward

    return backend, calls


@pytest.mark.skipif(not HAS_REGION_LIMIT, reason="torch.compile has no per-region recompile_limit")
@pytest.mark.parametrize("which", sorted(COMPILERS))
def test_every_variant_compiles_in_another_thread(which):
    """What the fix is for: the main thread's raised limit does not reach a fresh thread, the per-region one does - all
    VARIANTS graphs compile there (with the limit 8 the last ones would run the eager _forward)."""
    backend, calls = _counting_backend()
    compile_fn, restore = COMPILERS[which]
    model, block, args_of = compile_fn(backend, 64)
    try:
        assert dcfg.recompile_limit >= 64  # raised in this (the main) thread
        graphs = _graphs_in_a_fresh_thread(block, args_of, calls)
        assert graphs == VARIANTS, graphs
    finally:
        restore(model)


@pytest.mark.skipif(not HAS_REGION_LIMIT, reason="torch.compile has no per-region recompile_limit")
@pytest.mark.parametrize("which", sorted(COMPILERS))
def test_without_the_region_limit_another_thread_stops_at_the_default(which, monkeypatch):
    """The control (documents torch's behaviour; if a torch release makes the config global again this test fails and
    the per-region argument is merely redundant): with only the global raise, the fresh thread stops at its own default
    limit and runs the remaining variants eager."""
    monkeypatch.setattr(bc, "REGION_RECOMPILE_LIMIT", False)
    backend, calls = _counting_backend()
    compile_fn, restore = COMPILERS[which]
    model, block, args_of = compile_fn(backend, 64)
    try:
        graphs = _graphs_in_a_fresh_thread(block, args_of, calls)
        assert isinstance(graphs, int) and graphs < VARIANTS, graphs
    finally:
        restore(model)
