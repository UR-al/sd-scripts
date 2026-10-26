"""count_device_reads(): records every read of a tensor value by the host inside a block - what synchronizes with the GPU
on CUDA, counted on the CPU: aten._local_scalar_dense (.item(), bool(), int(), a tensor as a slice end), aten.equal /
aten.is_nonzero, and Tensor.tolist / .cpu / .numpy. No pytest import (test files that run without pytest use it)."""
import contextlib

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class _Reads(TorchDispatchMode):
    OPS = {torch.ops.aten._local_scalar_dense.default, torch.ops.aten.equal.default, torch.ops.aten.is_nonzero.default}

    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in self.OPS:
            self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


@contextlib.contextmanager
def count_device_reads():
    rec = _Reads()
    patched = {name: getattr(torch.Tensor, name) for name in ("tolist", "cpu", "numpy")}

    def spy(name, orig):
        def f(self, *a, **k):
            rec.ops.append(name)
            return orig(self, *a, **k)

        return f

    for name, orig in patched.items():
        setattr(torch.Tensor, name, spy(name, orig))
    try:
        with rec:
            yield rec
    finally:
        for name, orig in patched.items():
            setattr(torch.Tensor, name, orig)
