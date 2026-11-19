"""Shared helpers of tests/test_moss_ft_*.py for the host optimizer pipeline (library/moss/ft_host.py, ft_optim.py):
the tiny MOSS DiT, its batches, the two parameter groups, the pipeline set-up and the two training loops (the pipeline
and the torch reference that it must reproduce).

Sizes are tiny on purpose: pieces of 2 KiB (512 fp32 / 1024 bf16 elements) and 128-element chunks, so every 2-D
parameter of the toy model spans several pieces and chunks, and the rings are a few entries deep.
"""
import copy
import re
from typing import Callable, List, Optional, Sequence

import torch

from library import anima_models
from library.moss import ft_host
from library.moss.models import MossTransformer

TINY = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=48, freq_dim=16, latent_channels=128)
PIECE_BYTES = 2048
RING_BYTES = 8 * PIECE_BYTES
UPLOAD_RING_BYTES = 4 * PIECE_BYTES
CHUNK = 128  # a multiple of ft_host.ALIGN_ELEMENTS
NO_DECAY_PREFIXES = ("video_embedder.", "text_embedder.proj.")
CKPT_MODES = ("none", "plain", "unsloth", "nograd", "reuse")


def build_model(seed: int = 0, dtype=torch.float32, stream_fp32: bool = False, ckpt: str = "plain",
                attn_mode: str = "torch") -> MossTransformer:
    """The toy DiT in train mode, every parameter moved off its init (so each one matters to the loss)."""
    torch.manual_seed(seed)
    model = MossTransformer(**TINY, attn_mode=attn_mode)
    g = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.02)
    model.to(dtype)
    if stream_fp32:
        model.enable_stream_fp32()
    if ckpt != "none":
        model.enable_gradient_checkpointing(unsloth_offload=(ckpt == "unsloth"))
    model.train()
    return model


def set_ckpt_flags(monkeypatch, ckpt: str) -> None:
    """The process-wide switches of the no-grad first-pass node (set_ckpt_reuse) for one test."""
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", ckpt == "nograd")
    monkeypatch.setattr(anima_models, "CKPT_REUSE_ATTN", ckpt == "reuse")
    monkeypatch.setattr(anima_models, "CKPT_RECOMPUTE_DIFFS", None)


def make_batch(seed: int, dtype=torch.float32, rows: int = 1, hw=(4, 4), text: int = 6):
    g = torch.Generator().manual_seed(1000 + seed)
    x = torch.randn(rows, TINY["latent_channels"], *hw, generator=g).to(dtype)
    t = torch.full((rows,), 100.0 + 70.0 * (seed % 11))
    emb = torch.randn(rows, text, TINY["text_dim"], generator=g).to(dtype)
    mask = torch.ones(rows, text, dtype=torch.bool)
    return x, t, emb, mask


def loss_of(model, batch, autocast: Optional[torch.dtype] = None) -> torch.Tensor:
    """One micro-step's loss as the trainer forms it: the noisy input marked grad-bearing (checkpointing), optional
    bf16 autocast, an fp32 mean."""
    x, t, emb, mask = batch
    x = x.detach().requires_grad_(True)
    with torch.autocast("cpu", dtype=autocast or torch.bfloat16, enabled=autocast is not None):
        y = model(x, t, emb, mask)
    return y.float().square().mean()


def trainable(model, frozen: Optional[str] = None):
    """[(name, parameter)] of the trainable set in registration order; `frozen` is a regex (re.fullmatch)."""
    out = []
    for name, p in model.named_parameters():
        if frozen is not None and re.fullmatch(frozen, name):
            p.requires_grad_(False)
            continue
        out.append((name, p))
    return out


def is_decay(name: str, p) -> bool:
    return p.dim() >= 2 and not name.startswith(NO_DECAY_PREFIXES)


def param_groups(named, wd: float = 0.1):
    """Group 0 (decay): 2-D weights; group 1 (no decay): 1-D parameters and the input embedders."""
    decay = [p for n, p in named if is_decay(n, p)]
    rest = [p for n, p in named if not is_decay(n, p)]
    return [{"params": decay, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}]


def make_pipeline(named, optimizer, ga: int, *, inline: bool = False, with_acc: Optional[bool] = None,
                  piece_bytes: int = PIECE_BYTES, ring_bytes: int = RING_BYTES,
                  upload_ring_bytes: int = UPLOAD_RING_BYTES, **kw):
    """Slots, pool (master filled from the parameters) and an attached CPU pipeline."""
    slots = ft_host.make_slots(named, optimizer)
    if with_acc is None:
        with_acc = ft_host.needs_accumulator(ga, kw.get("clip_mode", "stale"), kw.get("max_grad_norm", 0.0),
                                             overlap=kw.get("overlap", True))
    pool = ft_host.HostPool(slots, type(optimizer).POOL_ROLES, with_acc=with_acc)
    pool.copy_master_from_params()
    pipe = ft_host.GradPipeline(slots, pool, optimizer, device="cpu", gradient_accumulation_steps=ga, inline=inline,
                                piece_bytes=piece_bytes, ring_bytes=ring_bytes, upload_ring_bytes=upload_ring_bytes,
                                acc_chunk=CHUNK, step_chunk=CHUNK, **kw)
    pipe.attach()
    return pipe, pool, slots


def window_sizes(batches: int, ga: int) -> List[int]:
    """n_micro of each window when an epoch of `batches` micro-batches is cut into windows of `ga` (the last short)."""
    return [min(ga, batches - w * ga) for w in range((batches + ga - 1) // ga)]


def train_pipeline(pipe, optimizer, windows: Sequence[Sequence[Callable[[], torch.Tensor]]], scheduler=None,
                   after_window=None):
    """The trainer's loop (design 4.4): begin_window / begin_microstep / backward of loss / n_micro / end_microstep.
    windows[w][j]() returns micro-step j's loss. Returns the WindowStats of every window."""
    stats = []
    for w, micro in enumerate(windows):
        n = len(micro)
        pipe.begin_window(n, ft_host.StepContext.snapshot(optimizer, step_index=w))
        result = None
        for j, fn in enumerate(micro):
            pipe.begin_microstep(j)
            (fn() / n).backward()
            result = pipe.end_microstep()
            assert (result is None) == (j < n - 1)
        stats.append(result)
        if scheduler is not None:
            scheduler.step()
        if after_window is not None:
            after_window(w, result)
    return stats


def train_reference(optimizer, params, windows, scheduler=None, max_grad_norm: float = 0.0, coef_for=None,
                    before_step=None):
    """The same windows with plain autograd accumulation and a torch optimizer. max_grad_norm: clip_grad_norm_ (exact);
    coef_for(w) -> a 0-dim coefficient tensor (or None) multiplied into the grads (the stale reference); before_step(w)
    may edit the grads. Returns clip_grad_norm_'s total norms."""
    norms = []
    for w, micro in enumerate(windows):
        n = len(micro)
        for fn in micro:
            (fn() / n).backward()
        with_grad = [p for p in params if p.grad is not None]
        if before_step is not None:
            before_step(w)
            with_grad = [p for p in params if p.grad is not None]
        if max_grad_norm > 0:
            norms.append(torch.nn.utils.clip_grad_norm_(with_grad, max_grad_norm))
        elif coef_for is not None:
            coef = coef_for(w)
            if coef is not None:
                torch._foreach_mul_([p.grad for p in with_grad], coef)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
    return norms


def model_windows(model, seeds_per_window, autocast=None, extra=None):
    """Micro-step closures for `model`: seeds_per_window[w] = batch seeds; extra(w, j) -> an added loss term."""
    out = []
    for w, seeds in enumerate(seeds_per_window):
        micro = []
        for j, seed in enumerate(seeds):
            def fn(seed=seed, w=w, j=j):
                loss = loss_of(model, make_batch(seed, dtype=_input_dtype(model)), autocast=autocast)
                if extra is not None:
                    term = extra(w, j)
                    if term is not None:
                        loss = loss + term
                return loss
            micro.append(fn)
        out.append(micro)
    return out


def _input_dtype(model):
    return torch.float32 if model.stream_fp32 else model.video_embedder.weight.dtype


def seeds(sizes: Sequence[int], start: int = 0) -> List[List[int]]:
    out, k = [], start
    for n in sizes:
        out.append(list(range(k, k + n)))
        k += n
    return out


def assert_pool_equals_torch(pool, slots, ref_named, ref_opt, what: str = "") -> None:
    """master == the reference parameters, exp_avg / exp_avg_sq / step == torch AdamW's state, bit for bit."""
    ref = dict(ref_named)
    for s in slots:
        p = ref[s.name]
        master = pool.view("master", s)
        assert torch.equal(master, p.detach().reshape(-1)), (
            f"{what}: master of {s.name} differs, max |d| = {(master - p.detach().reshape(-1)).abs().max().item():.3e}")
        state = ref_opt.state.get(p) or {}
        if not state:
            assert float(pool.steps[s.index]) == 0.0, f"{what}: {s.name} stepped, the reference did not"
            continue
        for role in ("exp_avg", "exp_avg_sq"):
            mine, theirs = pool.view(role, s), state[role].reshape(-1)
            assert torch.equal(mine, theirs), (
                f"{what}: {role} of {s.name} differs, max |d| = {(mine - theirs).abs().max().item():.3e}")
        assert float(pool.steps[s.index]) == float(state["step"]), f"{what}: step of {s.name}"


def assert_compute_copies(pool, slots, what: str = "") -> None:
    """Every compute copy == master rounded to its dtype (round to nearest even)."""
    for s in slots:
        expected = pool.view("master", s).to(s.compute_dtype)
        assert torch.equal(s.param.detach().reshape(-1), expected), f"{what}: compute copy of {s.name} != master"


def clone_model(model):
    return copy.deepcopy(model)
