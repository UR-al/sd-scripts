# Per-block torch.compile for the MOSS-Image DiT: --moss_block_compile (default off; 2026-11-14).
#
# The --anima_block_compile design (library/anima_block_compile.py, read its header first) applied to MossTransformer:
# every joint MossBlock's `_forward` (attention + MLP math, NOT the checkpointing wrapper `forward` / `_run`) is compiled
# with the LyCORIS-adapted Linears inside the graph, so a micro-step runs a few graph calls per block instead of the
# Python-level module calls of the 6 adapted Linears and the elementwise chains. Shared with Anima, imported from
# anima_block_compile: the pause() switch (sampling / validation run eager), the recompile limit, the per-signature
# inductor cache dir, CompiledBlocks / restore, and the autocast-state-as-graph-argument entry.
#
# What differs from Anima:
# * the block input is already flat: x [B, L, D] with L = image tokens + text tokens of the joint sequence, rope
#   [B, L, 128]. The text length changes with every caption, so the token axis (dim 1 of x and rope) is marked dynamic on
#   every call with SOFT marks (maybe_mark_dynamic): no planned band can hold the caption lengths, and soft marks let
#   dynamo keep any guard it needs (one extra graph at worst) instead of failing on a strict range.
# * no AttentionParams ints inside the graph: with one sample, or several of equal text length, there is no padding and
#   MossTransformer.forward hands the joint blocks mask-free attention params (library/moss/models.py) - the same
#   attention, but no per-length Python ints (seqlens_host, max_seqlen, img_len) for dynamo to guard on, which would
#   compile a graph per caption length. A padded batch (different text lengths in one batch) keeps the mask and its
#   ints: those calls specialise per length (see the warning at compile time for batch > 1).
# * only the 42 joint blocks: the two text refiner blocks run once per forward on the text only and are frozen by the
#   default "moss" LyCORIS preset.
#
# Refused combinations (library/moss/train_utils.validate_moss_args): RamTorch, block swap, accelerate's --torch_compile,
# stock LyCORIS (--moss_stock_lycoris / --lycoris_stock), DyLoRA - the same reasons as for Anima.

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import torch

from library import anima_block_compile as abc

logger = logging.getLogger(__name__)

pause = abc.pause  # one switch for both models (sampling / validation run eager)


def compile_moss_blocks(
    model: torch.nn.Module,
    *,
    backend: str = "inductor",
    mode: Optional[str] = None,
    recompile_limit: int = 64,
    isolate_cache: bool = True,
    stream_fp32: bool = False,
) -> abc.CompiledBlocks:
    """torch.compile every joint MossBlock's `_forward` of `model` (a MossTransformer). Call AFTER the network's apply_to
    and weight loading (the graphs trace the patched Linear forwards) and with the elementwise-chain / LoKr-rebuild
    compiles off (nested torch.compile inside a compiled frame exhausts the recompile budget)."""
    limit = abc.raise_recompile_limit(recompile_limit)
    if isolate_cache and backend == "inductor":
        signature = abc.compile_signature([], backend, mode, "moss-soft", stream_fp32)
        logger.info(f"MOSS block compile: inductor cache dir {abc.isolate_inductor_cache(signature)}")
    kwargs: Dict[str, Any] = {"backend": backend, "dynamic": False}
    if mode:
        kwargs["mode"] = mode
    state = abc.CompiledBlocks()
    entries: Dict[Any, Callable] = {}

    def shared_entry(fn: Callable) -> Callable:
        # one compiled entry per class function, the block passed as an argument: all 42 blocks share the cache entries
        # (dynamo keys on the code object; the parameters are graph inputs) - see anima_block_compile.shared_entry
        compiled = entries.get(fn)
        if compiled is None:

            def entry(autocast_enabled: bool, autocast_dtype: torch.dtype, block_: torch.nn.Module, x, *args, **kw):
                with torch.autocast(device_type=x.device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                    return fn(block_, x, *args, **kw)

            compiled = entries[fn] = torch.compile(entry, **kwargs)
        return compiled

    def make(block: torch.nn.Module):
        orig = block._forward  # the bound class method (the eager calls)
        fn = getattr(orig, "__func__", None)
        if fn is None or getattr(orig, "__self__", None) is not block:
            raise TypeError("MOSS block compile: MossBlock._forward is not a bound method of the block (already wrapped?)")
        compiled = shared_entry(fn)

        def wrapped(x, rope, *args, **kw):
            # a training forward: train mode (gradient checkpointing) or grad enabled; sampling (pause()) and validation
            # (eval, no grad) run the eager _forward
            if not (abc.is_active() and (block.training or torch.is_grad_enabled())):
                state.eager_calls += 1
                return orig(x, rope, *args, **kw)
            torch._dynamo.maybe_mark_dynamic(x, 1)
            if isinstance(rope, torch.Tensor):
                torch._dynamo.maybe_mark_dynamic(rope, 1)
            state.compiled_calls += 1
            dev = x.device.type
            # autocast OFF around the compiled call, the caller's state handed to the graph (anima_block_compile.make)
            enabled, dtype = torch.is_autocast_enabled(dev), torch.get_autocast_dtype(dev)
            with torch.autocast(device_type=dev, enabled=False):
                return compiled(enabled, dtype, block, x, rope, *args, **kw)

        wrapped.__name__ = "compiled__forward"
        wrapped.__wrapped__ = orig
        return wrapped

    for block in model.layers:
        state.originals.append((block, "_forward", block._forward))
        block._forward = make(block)
    model._block_compile = True
    model._block_compile_state = state
    logger.info(f"MOSS block compile: {len(model.layers)} joint blocks, backend={backend}, mode={mode or 'default'},"
                f" soft token marks, recompile_limit={limit}")
    return state


def restore_moss_blocks(model: torch.nn.Module) -> None:
    state = getattr(model, "_block_compile_state", None)
    if state is not None:
        state.restore()
    model._block_compile = False
    model._block_compile_state = None
