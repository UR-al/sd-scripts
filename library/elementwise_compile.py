# torch.compile for the elementwise chains of DiT blocks (MOSS-Image, Anima).
#
# Between two matmuls a block runs strings of small memory-bound kernels (fp32 norms, RoPE, gated residuals,
# activations). Compiled, each string becomes one or two fused kernels: less memory traffic and fewer saved
# intermediates. Only these pure functions are compiled; LoRA-wrapped Linears and attention stay eager.
# Off by default; trainers call ``enable(True)``. CUDA tensors only, so CPU tests keep the eager path.

import inspect
import logging

import torch

logger = logging.getLogger(__name__)

# Grad mode, batch 1 vs >1, requires_grad, view offsets and the thread count each add a graph per function (7-8 seen
# for one MOSS chain in the main thread; the autograd thread compiles its own, see compile_kwargs); with fullgraph the
# default limit of 8 raises instead of falling back.
RECOMPILE_LIMIT = 64
_REGION_LIMIT = "recompile_limit" in inspect.signature(torch.compile).parameters  # torch >= 2.14


def compile_kwargs() -> dict:
    """torch.compile arguments of the chains.

    torch 2.14 keeps torch._dynamo / torch._inductor config overrides per thread (ContextVar): a value set in the main
    thread is not seen by the autograd engine's thread, where gradient-checkpoint recomputes run - those compiled with
    the default recompile limit 8 and without emulate_precision_casts (11-17 GPU smoke: FailOnRecompileLimitHit in the
    recompute, then torch.utils.checkpoint "Recomputed values ... different metadata"). The per-region limit and the
    inductor options travel with the compiled function into every thread."""
    kwargs = dict(dynamic=True, fullgraph=True, options={"emulate_precision_casts": True})
    if _REGION_LIMIT:
        kwargs["recompile_limit"] = RECOMPILE_LIMIT
    return kwargs


class ElementwiseChains:
    """A named set of pure functions, used eagerly or as torch.compile(dynamic=True, fullgraph=True) versions.

    emulate_precision_casts keeps eager's intermediate roundings (differences stay at ~1 ulp). A Dynamo/Inductor
    failure switches the whole set back to eager once, with a warning, instead of stopping training."""

    def __init__(self, label: str, **functions):
        self.label = label
        self.eager = dict(functions)
        self.compiled = None
        self.enabled = False

    def enable(self, enabled: bool = True):
        self.enabled = bool(enabled)

    def get(self, x: torch.Tensor) -> dict:
        if not (self.enabled and x.is_cuda):
            return self.eager
        if self.compiled is None:
            import torch._dynamo.config as dynamo_config
            import torch._inductor.config as inductor_config

            # the global (this thread's) settings stay for torch without per-region arguments; compile_kwargs() is what
            # reaches the autograd thread on torch 2.14
            inductor_config.emulate_precision_casts = True
            dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, RECOMPILE_LIMIT)
            kwargs = compile_kwargs()
            self.compiled = {name: self.guard(name, torch.compile(fn, **kwargs), fn)
                             for name, fn in self.eager.items()}
            logger.info(f"torch.compile for the {self.label} elementwise chains (first steps compile)")
        return self.compiled

    def guard(self, name, compiled, eager):
        import torch._dynamo.exc as dynamo_exc

        failures = (dynamo_exc.TorchDynamoException, dynamo_exc.FailOnRecompileLimitHit)

        def call(*args):
            if self.enabled:
                try:
                    return compiled(*args)
                except failures as error:
                    self.enabled = False
                    first_line = (str(error).strip().splitlines() or [""])[0][:300]
                    logger.warning(f"torch.compile of {self.label}.{name} failed ({type(error).__name__}: {first_line});"
                                   " continuing with the eager elementwise chains")
            return eager(*args)

        return call
