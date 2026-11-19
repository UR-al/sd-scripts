"""library/elementwise_compile: the compiled chains must carry their recompile limit and inductor options with them.

11-17 GPU smoke (Carousel rotate_stem + window): torch 2.14 keeps torch._dynamo / torch._inductor config overrides per
thread (ContextVar). ElementwiseChains.get raised torch._dynamo.config.recompile_limit to 64 in the MAIN thread, but
gradient-checkpoint recomputes run in the autograd engine's thread, which still saw the default 8 (and its own
num_threads, so every chain compiled again there). Past 8 graphs the recompute hit FailOnRecompileLimitHit, the chains
fell back to eager mid-recompute and torch.utils.checkpoint raised "Recomputed values ... different metadata"."""
import inspect
import threading
import types

import pytest
import torch
import torch._dynamo
import torch._dynamo.exc as dynamo_exc

from library import elementwise_compile as ec

HAS_REGION_LIMIT = "recompile_limit" in inspect.signature(torch.compile).parameters


def _variants(fn, out):
    """Call fn with 12 guard variants (thread count x grad mode x requires_grad) in the CURRENT thread."""
    try:
        for i in range(12):
            torch.set_num_threads(1 + i % 3)
            with torch.set_grad_enabled(i % 2 == 0):
                fn(torch.randn(4, requires_grad=(i // 6 == 0)), torch.randn(4))
        out.append("ok")
    except dynamo_exc.FailOnRecompileLimitHit as error:  # pragma: no cover - the failure being tested
        out.append(f"limit: {error}")


def _in_new_thread(fn):
    out = []
    t = threading.Thread(target=_variants, args=(fn, out))
    t.start()
    t.join()
    return out[0]


def test_get_passes_a_per_region_limit_and_the_precision_option(monkeypatch):
    seen = []

    def fake_compile(fn, **kwargs):
        seen.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    chains = ec.ElementwiseChains("T", f=lambda a, b: a + b)
    chains.enable(True)
    chains.get(types.SimpleNamespace(is_cuda=True))  # get() only reads x.is_cuda
    assert seen, "torch.compile was not called"
    kwargs = seen[0]
    assert kwargs["dynamic"] is True and kwargs["fullgraph"] is True
    if HAS_REGION_LIMIT:
        assert kwargs.get("recompile_limit", 0) >= ec.RECOMPILE_LIMIT >= 64
    assert kwargs.get("options", {}).get("emulate_precision_casts") is True


@pytest.mark.skipif(not HAS_REGION_LIMIT, reason="torch.compile has no per-region recompile_limit")
def test_region_limit_holds_in_another_thread():
    """What the fix relies on: the per-region limit applies in a thread that never saw a config override."""
    torch._dynamo.reset()
    threads = torch.get_num_threads()
    try:
        fn = torch.compile(lambda a, b: (a * b).sin(), backend="eager", dynamic=True, fullgraph=True,
                           recompile_limit=ec.RECOMPILE_LIMIT)
        assert _in_new_thread(fn) == "ok"
    finally:
        torch.set_num_threads(threads)
        torch._dynamo.reset()


@pytest.mark.skipif(not HAS_REGION_LIMIT, reason="torch.compile has no per-region recompile_limit")
def test_main_thread_config_override_is_not_seen_in_another_thread():
    """The cause (documents torch's behaviour; if a torch release makes the config global again this test fails and
    the per-region argument is merely redundant)."""
    import torch._dynamo.config as dcfg

    torch._dynamo.reset()
    threads = torch.get_num_threads()
    old = dcfg.recompile_limit
    try:
        dcfg.recompile_limit = 64
        fn = torch.compile(lambda a, b: (a * b).cos(), backend="eager", dynamic=True, fullgraph=True)
        assert _in_new_thread(fn).startswith("limit")
    finally:
        dcfg.recompile_limit = old
        torch.set_num_threads(threads)
        torch._dynamo.reset()
