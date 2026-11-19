# Per-block torch.compile for the MOSS-Image DiT: --moss_block_compile (default off; 2026-11-14).
#
# The --anima_block_compile design (library/anima_block_compile.py, read its header first) applied to MossTransformer:
# every joint MossBlock's `_forward` (attention + MLP math, NOT the checkpointing wrapper `forward` / `_run`) is compiled
# with the LyCORIS-adapted Linears inside the graph, so a micro-step runs a few graph calls per block instead of the
# Python-level module calls of the 6 adapted Linears and the elementwise chains. Shared with Anima, imported from
# anima_block_compile: the pause() switch (sampling / validation run eager), the recompile limit (global and per region,
# block_compile_kwargs), the per-signature inductor cache dir, CompiledBlocks / restore, and the
# autocast-state-as-graph-argument entry.
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
# * --moss_ckpt_nograd_merged (library/moss/models.py, anima_models.NoGradFirstPassCheckpointer) replaces the plain
#   checkpoint's "grad forward twice" by a no-grad forward that reads the cached merged LoKr weights (graph inputs;
#   lycoris_fast._CKPT_PASS == 1) and a grad forward for the recompute (_CKPT_PASS == 2): two more entries. With
#   --moss_ckpt_reuse_attn the first-pass graph returns the joint attention's output and softmax_lse (a list argument
#   it appends to) and the recompute graph takes them as inputs, their token axis soft-marked like x's. A padded batch
#   (masked params, a graph per padded length) keeps torch.utils.checkpoint there, so the option adds no graphs per
#   padded length (MossBlock._ckpt_keeps_plain).
# * --moss_ckpt_reuse_attn without --moss_ckpt_nograd_merged (reuse-only) and --moss_ckpt_reuse_attn_budget_gib: the
#   same two entries (first pass with a stash list, recompute with the stored tensors) with lycoris_fast._CKPT_PASS == 0
#   and no cached weight among the graph inputs; both passes rebuild the merged weights inside their graph, so the two
#   graphs may round that rebuild differently (the cached mode gives the recompute the first pass's weight bits). Which
#   blocks reuse under the budget is decided per call outside the graphs (MossBlock._ckpt_reuses, shape ints): all blocks
#   share the entries, so a block changing between "plain", "first pass" and "recompute" calls compiles nothing new.
#   Entries on the shared block (adapted blocks + a block left out of the network): 4 without a budget, 6 reuse-only
#   with a budget (the plain grad graph next to the node's two), 8 with the cache and a budget (the node with and
#   without a stored output): that last combination is refused below a recompile limit of 16.
# * --moss_stream_scale (library/moss/models.py): the scaled input and gates come from outside the graphs; inside, only
#   the eps of norm1 / norm2 differs (read from the block, one value for the run): no extra graph (compile_moss_blocks).
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
    stream_scale: float = 1.0,
) -> abc.CompiledBlocks:
    """torch.compile every joint MossBlock's `_forward` of `model` (a MossTransformer). Call AFTER the network's apply_to
    and weight loading (the graphs trace the patched Linear forwards) and with the elementwise-chain / LoKr-rebuild
    compiles off (nested torch.compile inside a compiled frame exhausts the recompile budget).

    --moss_stream_scale (`stream_scale`, MossTransformer.set_stream_scale): nothing changes in the graphs but the eps of
    norm1 / norm2, which the traced `_forward` reads from the block (`block.norm1.eps`; None when off: the graphs of
    before). It is one value for the whole run and for all 42 blocks, so the shared entry compiles no extra graph; a C
    other than 1.0 gets an inductor cache dir of its own (the signature)."""
    limit = abc.raise_recompile_limit(recompile_limit)
    if isolate_cache and backend == "inductor":
        signature = abc.compile_signature([], backend, mode, "moss-soft", stream_fp32)
        if float(stream_scale) != 1.0:
            signature += f";stream_scale={float(stream_scale)!r}"
        logger.info(f"MOSS block compile: inductor cache dir {abc.isolate_inductor_cache(signature)}")
    # the limit passed per region as well: the recomputes compile in the autograd engine's thread, which does not see
    # raise_recompile_limit's global value on torch 2.14 (anima_block_compile.block_compile_kwargs)
    kwargs = abc.block_compile_kwargs(backend, mode, limit)
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
            if kw:  # --moss_ckpt_reuse_attn: the stored joint attention (out [B, L, heads, head_dim], softmax_lse
                # [B, heads, L]) handed to a recompute - the same token axis as x's. On torch 2.14 a tensor that left
                # the first-pass graph with a symbolic axis is already treated as dynamic here (measured: the graph
                # count is the same without these marks); they keep the axis symbolic should the stored tensors ever
                # come from an eager first pass, where an unmarked size compiles a graph per caption length.
                reuse = kw.get("attn_reuse")
                if reuse is not None:
                    torch._dynamo.maybe_mark_dynamic(reuse[0], 1)
                    torch._dynamo.maybe_mark_dynamic(reuse[1], 2)
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
