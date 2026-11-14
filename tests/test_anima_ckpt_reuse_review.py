"""Reviewer tests (2026-11-14) for --anima_ckpt_nograd_merged / --anima_ckpt_reuse_attn: what tests/test_anima_ckpt_reuse.py
left unproven. CPU only (flash runs on tests/_fake_flash.py); nothing here touches the implementation.

Why each group exists:
* the user's optimizer is LoraEasyCustomOptimizer's CAME, which writes parameters through `p.data` (no version bump, same
  storage): the cache keys cannot see its steps, so a stale cache would freeze the first pass's weights while training
  goes on (the failure mode of today's first CAME hotfix: a cache compared against something updated in place). The
  implementer's accumulation test steps with torch.optim.SGD, which the keys DO see. Here the real CAME drives several
  optimizer steps the way train_network.py's loop does (mixed tiers: no / half / full checkpointing, gradient clipping,
  optimizer.step, zero_grad, then the trainer hook), once per invalidation layer, eager and compiled, and a negative
  control shows the run goes stale when both layers are missing (so the comparison has teeth);
* the user's mixed precision is fp16 with the fp32 stream; the implementer's tests are bf16 only;
* accelerate's AcceleratedOptimizer + GradScaler (the fp16 path) patch optimizer.step at instance level: the global
  optimizer-step hook must still fire, exactly once per real step, never for a skipped / non-sync step;
* resume-like load_state_dict, a sampling round with lycoris_fast.cached_lokr_weights, a validation forward and the
  trainer marking the text conditioning as grad-bearing - all between micro-steps of one accumulation;
* a second backward (retain_graph), backward called under autocast, training without autocast, an exception inside a
  first pass, graph counts across optimizer steps, tensors alive across steps, state_dict keys.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_anima_ckpt_reuse_review.py
"""
import argparse
import gc
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
import torch._dynamo

from library import anima_block_compile as bc
from library import anima_models, lycoris_args, lycoris_fast
from library import lycoris_algo_fixes as laf
from tests import _fake_flash as fake_flash
from tests import test_anima_block_compile as T

torch.set_num_threads(4)

GRIDS = {16: (8, 8), 24: (8, 12), 36: (12, 12), 64: (16, 16)}  # toy tokens -> latent grid (patch 2)
# one accumulation = 3 micro-steps; the tiers of 3 optimizer steps (<= 16 tokens: no checkpointing, <= 36: every other
# Block, above: every Block - the trainer's --anima_no_checkpoint_max_tokens / --anima_half_checkpoint_max_tokens)
TIERS = (64, 16, 36, 24, 64, 16, 36, 64, 64)
CKPT_ARGS = argparse.Namespace(gradient_checkpointing=True, anima_no_checkpoint_max_tokens=16,
                               anima_half_checkpoint_max_tokens=36, cpu_offload_checkpointing=False)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    lycoris_fast.install_fast_lokr_forward()
    laf.apply_module_patches()
    torch._dynamo.reset()
    bc._warned_oob.clear()
    fake_flash.install(monkeypatch)
    monkeypatch.setitem(T.TOY, "attn_mode", "flash")
    for name, value in (("CKPT_NOGRAD_MERGED", False), ("CKPT_REUSE_ATTN", False), ("CKPT_RECOMPUTE_DIFFS", None),
                        ("ADALN_HOIST", False)):
        monkeypatch.setattr(anima_models, name, value)
    import anima_train_network as atn

    if atn._CKPT_STEP_HOOK is not None:  # another test file left the global optimizer-step hook registered: every
        atn._CKPT_STEP_HOOK.remove()  # run here chooses its invalidation layers itself
        atn._CKPT_STEP_HOOK = None
    with torch.backends.mkldnn.flags(enabled=False):  # oneDNN has no bf16 backward on some CPUs
        yield
    if atn._CKPT_STEP_HOOK is not None:
        atn._CKPT_STEP_HOOK.remove()
        atn._CKPT_STEP_HOOK = None
    laf.set_block_compile(False)
    assert lycoris_fast._CKPT_PASS == 0, "a checkpoint pass left lycoris_fast._CKPT_PASS set"
    torch._dynamo.reset()


@pytest.fixture
def came_cls(monkeypatch):
    """The user's optimizer class. Its step assumes a CUDA build with a visible GPU even for CPU parameters (pin_memory,
    torch.cuda.current_device as the compute device, a final synchronize); the three calls are pointed at the CPU so the
    real write path (p.data.copy_) runs here."""
    try:
        from LoraEasyCustomOptimizer.came import CAME
    except Exception as error:  # noqa: BLE001 - the package is an editable install outside this repository
        pytest.skip(f"LoraEasyCustomOptimizer.came is not importable: {error}")
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: self)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    return CAME


@pytest.fixture
def accelerate_state():
    """An initialised accelerate state (AcceleratedOptimizer refuses to exist without an Accelerator), reset afterwards so
    the singletons do not leak into other test files."""
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState, GradientState

    AcceleratorState._reset_state(True)
    GradientState._reset_state()
    accelerator = Accelerator(cpu=True)
    yield accelerator
    AcceleratorState._reset_state(True)
    GradientState._reset_state()


def make_came(came_cls, params, lr=2e-3):
    # config_anima29.toml's optimizer_args; the state on the CPU, compile_step off (inductor needs a compiler here)
    return came_cls(params, lr=lr, state_storage_device="cpu", state_storage_dtype="float32", weight_decay=0.1,
                    weight_decouple=True, fixed_decay=False, update_strategy="cautious", clip_threshold=1.0, eps1=1e-30,
                    eps2=1e-16, betas=(0.9, 0.999, 0.9999))


# ------------------------------------------------------------------------------------------------------------- helpers


def set_flags(monkeypatch, nograd: bool, reuse: bool = False, diffs: bool = True):
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", nograd)
    monkeypatch.setattr(anima_models, "CKPT_REUSE_ATTN", reuse)
    monkeypatch.setattr(anima_models, "CKPT_RECOMPUTE_DIFFS", [] if (nograd and diffs) else None)


def perturb(net, seed: int = 7, scale: float = 0.02) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            if p.requires_grad and p.is_floating_point():
                p.add_(torch.randn(p.shape, generator=g).to(p.dtype) * scale)


def build_model(blocks: int = 4, compiled: bool = False, stream: bool = True, dtype=torch.bfloat16, backend="aot_eager"):
    """tests/test_anima_block_compile.build with the base dtype as a parameter (fp16 = the user's mixed precision): the
    frozen toy DiT in train mode, LoKr factor 4 full matrix + DoRA (preset anima-exp) applied the way train_network.py
    does it, every trained factor moved off its init, checkpointing on."""
    import lycoris.kohya as kohya

    torch.manual_seed(0)
    unet = anima_models.Anima(**{**T.TOY, "num_blocks": blocks}).to(dtype)
    unet.requires_grad_(False)
    unet.train()
    p = T.PRESETS["lokr_dora"]
    args = T.make_args(p["args"], p["dim"], p["alpha"])
    net_kwargs = lycoris_args.fix_lycoris_bool_kwargs(args.network_module, dict(i.split("=", 1) for i in args.network_args))
    create_kwargs = laf.prepare_network_kwargs("lokr", dict(net_kwargs))
    create_kwargs.setdefault("dropout", 0.0)
    laf.set_block_compile(compiled)
    net = kohya.create_network(1.0, args.network_dim, args.network_alpha, None, None, unet, neuron_dropout=None,
                               **create_kwargs)
    if hasattr(net, "prepare_network"):
        net.prepare_network(args)
    net.apply_to(None, unet, False, True)
    laf.after_network_created(net, args)
    net.requires_grad_(True)
    net.train()
    perturb(net)
    if stream:
        unet.enable_stream_fp32()
    unet.enable_gradient_checkpointing()
    state = None
    if compiled:
        state = T.compile_toy(unet, net, backend=backend, **({"stream_fp32": True} if stream else {}))
    return unet, net, state


def micro_inputs(tokens: int, seed: int, stream: bool, dtype):
    x, t, ctx, pad = T.inputs(latent_hw=GRIDS[tokens], t=[0.1 + 0.07 * (seed % 11)], seed=seed)
    x, t, ctx, pad = x.to(dtype), t.to(dtype), ctx.to(dtype), pad.to(dtype)
    if stream:  # --anima_stream_fp32: the trainer keeps the noisy input and the timestep in fp32
        x, t = x.float(), t.float()
    return x, t, ctx, pad


def forward(unet, net, data, autocast_dtype, mark_text: bool = True):
    """The trainer's forward of one micro-step: the noisy input and the text conditioning marked as grad-bearing
    (anima_train_network.get_noise_pred_and_target does both under --gradient_checkpointing)."""
    x, t, ctx, pad = data
    laf.set_timestep(net, t.float() * 1000.0)
    x = x.detach().requires_grad_(True)
    if mark_text:
        ctx = ctx.detach().requires_grad_(True)
    if autocast_dtype is None:
        return unet(x, t, ctx, padding_mask=pad), ctx
    with torch.autocast("cpu", dtype=autocast_dtype):
        return unet(x, t, ctx, padding_mask=pad), ctx


def simulate(monkeypatch, flags, *, make_opt, invalidate=("hook", "trainer"), compiled=False, stream=True,
             dtype=torch.bfloat16, tiers=TIERS, accum=3, blocks=4, between=None, bypass_step_hooks=False, scaler=False,
             backend="aot_eager"):
    """A short training run driven like train_network.py's loop: per micro-step the checkpointing mode of its token count
    (anima_train_network.apply_checkpointing_for_tokens), forward, loss / accumulation, backward; every `accum` micro-steps
    gradient clipping (max_grad_norm 1.0), optimizer.step, zero_grad and then the trainer's on_optimizer_step_end.

    invalidate: which of the two invalidation layers exist - "hook" (the global optimizer-step hook prepare_anima_runtime
    registers) and / or "trainer" (AnimaNetworkTrainer.on_optimizer_step_end).
    bypass_step_hooks: step through the undecorated optimizer.step (no torch.optim step hooks run), like step_func or an
    optimizer whose step is replaced at instance level.
    scaler: accelerate's AcceleratedOptimizer around the optimizer with a GradScaler (the fp16 path).
    between(i, unet, net): called after micro-step i (0-based) and its optimizer step, if any.
    Returns (outputs of every micro-step, the trained parameters, extras)."""
    import anima_train_network as atn

    nograd, reuse = flags
    unet, net, state = build_model(blocks, compiled, stream, dtype, backend)
    set_flags(monkeypatch, nograd, reuse)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = raw = make_opt(params)
    grad_scaler = None
    if scaler:
        from accelerate.optimizer import AcceleratedOptimizer

        grad_scaler = torch.amp.GradScaler("cpu")
        opt = AcceleratedOptimizer(raw, device_placement=False, scaler=grad_scaler)
    if "hook" in invalidate and nograd:
        atn.invalidate_ckpt_merged_on_optimizer_steps()
    trainer = atn.AnimaNetworkTrainer() if "trainer" in invalidate else None
    autocast_dtype = torch.float16 if dtype == torch.float16 else torch.bfloat16
    outputs, modes, rebuilds = [], [], []
    spy = {"nograd": 0}
    inner = lycoris_fast._REBUILD.eager["rebuild"]

    def counting_rebuild(*args):
        spy["nograd"] += not torch.is_grad_enabled()
        return inner(*args)

    if not compiled:  # traced into a compiled Block graph, the counter would be guarded (a recompile per count)
        monkeypatch.setitem(lycoris_fast._REBUILD.eager, "rebuild", counting_rebuild)
    for i, tokens in enumerate(tiers):
        modes.append(atn.apply_checkpointing_for_tokens(unet, CKPT_ARGS, tokens, False, False))
        y, _ = forward(unet, net, micro_inputs(tokens, i + 1, stream, dtype), autocast_dtype)
        loss = y.float().square().mean() / accum
        (grad_scaler.scale(loss) if grad_scaler else loss).backward()
        outputs.append(y.detach().float().clone())
        del y, loss
        if (i + 1) % accum == 0:
            if grad_scaler:
                grad_scaler.unscale_(raw)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            if bypass_step_hooks:
                type(raw).step.__wrapped__(raw)
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            if trainer is not None:
                trainer.on_optimizer_step_end(argparse.Namespace(), None, (i + 1) // accum, False)
            rebuilds.append(spy["nograd"])
            spy["nograd"] = 0
        if between is not None:
            between(i, unet, net)
    monkeypatch.setitem(lycoris_fast._REBUILD.eager, "rebuild", inner)
    if atn._CKPT_STEP_HOOK is not None:  # one run's hook must not serve the next run of the same test
        atn._CKPT_STEP_HOOK.remove()
        atn._CKPT_STEP_HOOK = None
    extras = {"unet": unet, "net": net, "state": state, "modes": modes, "nograd_rebuilds": rebuilds,
              "diffs": anima_models.CKPT_RECOMPUTE_DIFFS}
    return outputs, [p.detach().clone() for p in params], extras


def assert_same_run(ref, new, what: str):
    (y0, p0, _), (y1, p1, _) = ref, new
    for i, (a, b) in enumerate(zip(y0, y1)):
        assert torch.equal(a, b), f"{what}: output of micro-step {i} differs, max |d| = {(a - b).abs().max().item():.3e}"
    for i, (a, b) in enumerate(zip(p0, p1)):
        assert torch.equal(a, b), f"{what}: trained parameter {i} differs, max |d| = {(a - b).abs().max().item():.3e}"


# ----------------------------------------------------------------------------------------- CAME: the cache must refresh


def test_came_steps_are_invisible_to_the_cache_keys(came_cls):
    """The premise of the invalidation hooks, with the real optimizer: a CAME step changes every trained parameter but
    neither its version counter nor its storage, so the keys still call the cached weights current."""
    unet, net, _ = build_model(blocks=1)
    lokrs = anima_models._ckpt_cached_lokrs(unet.blocks[0])
    keys = lycoris_fast.ckpt_merged_refresh(lokrs, torch.bfloat16)
    params = [p for p in net.parameters() if p.requires_grad]
    before = [(p._version, p.data_ptr(), p.detach().clone()) for p in params]
    opt = make_came(came_cls, params)
    for p in params:
        p.grad = torch.randn(p.shape, generator=torch.Generator().manual_seed(1))
    opt.step()
    assert all(p._version == v and p.data_ptr() == ptr for p, (v, ptr, _) in zip(params, before))
    assert all(not torch.equal(p, old) for p, (_, _, old) in zip(params, before))
    assert lycoris_fast.ckpt_merged_current(lokrs, keys, torch.bfloat16)  # stale, and the keys cannot tell
    lokr = lokrs[0]
    args = (lokr.org_module[0].weight.detach(), lokr.lokr_w1, lokr.lokr_w2, float(lokr.scale), lokr.dora_scale,
            lokr.multiplier_buf, lokr._dora_eps, lokr._dora_norm_dims, torch.bfloat16)
    with torch.no_grad():
        assert not torch.equal(lokr.__dict__["_ckpt_merged"], lycoris_fast._rebuild_weight(*args))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
@pytest.mark.parametrize("layers", [("hook",), ("trainer",), ("hook", "trainer")], ids=["hook", "trainer", "both"])
def test_came_training_matches_the_plain_checkpoint(layers, compiled, came_cls, monkeypatch):
    """3 optimizer steps x 3 accumulated micro-steps over mixed tiers with CAME: every micro-step's output and the trained
    parameters are bit-identical to the flags-off run, with either invalidation layer alone and with both. One no-grad
    rebuild per cached module and optimizer step at most (the cache is not rebuilt per micro-step)."""
    opt = lambda params: make_came(came_cls, params)  # noqa: E731
    ref = simulate(monkeypatch, (False, False), make_opt=opt, compiled=compiled)
    assert not torch.equal(ref[0][0], ref[0][3]) and ref[2]["modes"][:3] == ["all", "none", "half"], ref[2]["modes"]
    new = simulate(monkeypatch, (True, True), make_opt=opt, invalidate=layers, compiled=compiled)
    assert_same_run(ref, new, f"CAME, invalidation by {layers}, compiled={compiled}")
    extras = new[2]
    assert set(extras["diffs"]) == {0.0}, extras["diffs"]
    if not compiled:
        n = len([l for b in extras["unet"].blocks for l in anima_models._ckpt_cached_lokrs(b)])
        assert n == 40 and all(0 < r <= n for r in extras["nograd_rebuilds"]), extras["nograd_rebuilds"]


def test_came_training_without_any_invalidation_goes_stale(came_cls, monkeypatch):
    """Negative control for the test above: with neither layer the first pass keeps reading the weights of step 0 while
    CAME moves the parameters - the outputs leave the flags-off run at the first checkpointed micro-step after the first
    optimizer step. (Nothing in the trainer runs like this; it shows what the two layers are holding up, and that the
    comparison above would catch a cache that stopped refreshing.)"""
    opt = lambda params: make_came(came_cls, params)  # noqa: E731
    ref = simulate(monkeypatch, (False, False), make_opt=opt)
    stale = simulate(monkeypatch, (True, True), make_opt=opt, invalidate=())
    same = [torch.equal(a, b) for a, b in zip(ref[0], stale[0])]
    assert same[:3] == [True, True, True], same  # before the first optimizer step nothing is stale
    assert not all(same[3:]), "a cache nobody invalidated still followed CAME's steps?"
    assert stale[2]["nograd_rebuilds"][1:] == [0, 0]  # nothing was rebuilt after the first optimizer step


def test_optimizer_stepping_past_the_step_hooks_is_covered_by_the_trainer(came_cls, monkeypatch):
    """step_func (train_network.py: Polyak-step optimizers) and any optimizer whose step does not run torch.optim's step
    hooks: the global hook never fires, AnimaNetworkTrainer.on_optimizer_step_end alone keeps the cache right."""
    opt = lambda params: make_came(came_cls, params)  # noqa: E731
    ref = simulate(monkeypatch, (False, False), make_opt=opt, bypass_step_hooks=True)
    new = simulate(monkeypatch, (True, True), make_opt=opt, bypass_step_hooks=True)
    assert_same_run(ref, new, "optimizer stepped without step hooks")


def test_accelerate_optimizer_with_grad_scaler(came_cls, accelerate_state, monkeypatch):
    """The fp16 path of the trainer: accelerate's AcceleratedOptimizer swaps optimizer.step for a patched instance method
    around GradScaler.step. The global hook still fires - once per real step, not for a step the scaler skips (inf
    gradient) nor for a non-sync call - and a run stepped through it with the hook as the only invalidation layer matches
    the flags-off run."""
    from accelerate.optimizer import AcceleratedOptimizer
    from torch.optim.optimizer import register_optimizer_step_post_hook

    fired = []
    handle = register_optimizer_step_post_hook(lambda o, a, k: fired.append(type(o).__name__))
    try:
        p = torch.nn.Parameter(torch.ones(4, 4))
        raw = make_came(came_cls, [p])
        # the trainer builds the LR scheduler before accelerator.prepare: it replaces raw.step with an instance-level
        # wrapper of its own (step-order tracking), which accelerate's patched step then wraps again
        scheduler = torch.optim.lr_scheduler.LambdaLR(raw, lambda step: 1.0)
        assert "step" in raw.__dict__ and scheduler.get_last_lr() == [2e-3]
        scaler = torch.amp.GradScaler("cpu")
        opt = AcceleratedOptimizer(raw, device_placement=False, scaler=scaler)
        scaler.scale(torch.zeros(()))  # as accelerator.backward does for the loss: the scale tensor exists from here
        p.grad = torch.full((4, 4), 0.5) * scaler.get_scale()
        opt.step()
        assert fired == ["CAME"] and not opt.step_was_skipped and not torch.equal(p, torch.ones(4, 4))
        p.grad = torch.full((4, 4), float("inf"))
        value = p.detach().clone()
        opt.step()
        assert fired == ["CAME"] and opt.step_was_skipped and torch.equal(p, value)  # skipped: no hook, nothing changed
        opt.gradient_state._set_sync_gradients(False)
        try:
            p.grad = torch.full((4, 4), 0.5)
            opt.step()
        finally:
            opt.gradient_state._set_sync_gradients(True)
        assert fired == ["CAME"] and torch.equal(p, value)  # mid-accumulation: accelerate does not step
    finally:
        handle.remove()

    make = lambda params: make_came(came_cls, params)  # noqa: E731
    ref = simulate(monkeypatch, (False, False), make_opt=make, scaler=True)
    new = simulate(monkeypatch, (True, True), make_opt=make, scaler=True, invalidate=("hook",))
    assert_same_run(ref, new, "AcceleratedOptimizer + GradScaler, hook only")
    plain = simulate(monkeypatch, (False, False), make_opt=make)
    assert not torch.equal(plain[1][0], torch.zeros_like(plain[1][0]))  # the scaled run trained too
    assert all(torch.isfinite(p).all() for p in new[1])


# --------------------------------------------------------------------------------------------- fp16 (the user's setup)


def _fp16_probe() -> None:
    try:
        with torch.autocast("cpu", dtype=torch.float16):
            w = torch.randn(8, 8, dtype=torch.float16, requires_grad=True)
            y = torch.nn.functional.gelu(torch.nn.functional.linear(torch.randn(2, 3, 8), w))
        y.float().sum().backward()
    except RuntimeError as error:
        pytest.skip(f"no fp16 autocast on this CPU build: {error}")


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "aot_eager"])
@pytest.mark.parametrize("stream", [True, False], ids=["fp32_stream", "fp16_stream"])
def test_fp16_training_matches_the_plain_checkpoint(stream, compiled, came_cls, monkeypatch):
    """config_anima29.toml: mixed_precision fp16 (an fp16 base, fp16 autocast) with --anima_stream_fp32. The same run as
    above in fp16: bit-identical to the flags-off run, the cache is fp16. Without the fp32 stream the Blocks get an fp16
    input (use_fp32: the adaLN region under an fp32 autocast)."""
    _fp16_probe()
    opt = lambda params: make_came(came_cls, params)  # noqa: E731
    ref = simulate(monkeypatch, (False, False), make_opt=opt, dtype=torch.float16, stream=stream, compiled=compiled)
    new = simulate(monkeypatch, (True, True), make_opt=opt, dtype=torch.float16, stream=stream, compiled=compiled)
    assert all(torch.isfinite(y).all() for y in ref[0])
    assert_same_run(ref, new, f"fp16, stream={stream}, compiled={compiled}")
    lokrs = [l for b in new[2]["unet"].blocks for l in anima_models._ckpt_cached_lokrs(b)]
    assert {l.__dict__["_ckpt_merged"].dtype for l in lokrs} == {torch.float16}
    assert set(new[2]["diffs"]) == {0.0}


def test_training_without_autocast_matches_the_plain_checkpoint(monkeypatch):
    """mixed_precision "no": the node has no autocast dtype to cache for (the LoKr forward is LyCORIS's stock one). The
    no-grad first pass alone must still give the plain checkpoint's run, with nothing cached."""
    opt = lambda params: torch.optim.SGD(params, lr=1e-2)  # noqa: E731

    def run(flags):
        unet, net, _ = build_model(blocks=2, stream=False, dtype=torch.float32)
        set_flags(monkeypatch, *flags)
        params = [p for p in net.parameters() if p.requires_grad]
        o = opt(params)
        outs = []
        for i in range(3):
            y, _ = forward(unet, net, micro_inputs(16, i + 1, False, torch.float32), None)
            y.square().mean().backward()
            outs.append(y.detach().clone())
            o.step()
            o.zero_grad(set_to_none=True)
        return outs, [p.detach().clone() for p in params], {"unet": unet}

    ref, new = run((False, False)), run((True, True))
    assert_same_run(ref, new, "no autocast")
    assert lycoris_fast.ckpt_merged_bytes(new[2]["unet"]) == 0


# ------------------------------------------------------------------- things that happen between micro-steps of a run


def test_resume_like_load_state_dict_between_steps(came_cls, monkeypatch):
    """accelerator.load_state (--resume) and --network_weights copy into the parameters (load_state_dict): the keys see
    it. Here the weights jump back to a snapshot in the middle of an accumulation, without any invalidation call."""
    snapshot = {}

    def between(i, unet, net):
        if i == 0:
            snapshot.update({k: v.detach().clone() for k, v in net.state_dict().items()})
        if i == 3:  # mid-accumulation of the second optimizer step (the next micro-step checkpoints every Block): the
            net.load_state_dict(snapshot)  # even-indexed Blocks' cached weights were built for other values

    opt = lambda params: make_came(came_cls, params)  # noqa: E731
    ref = simulate(monkeypatch, (False, False), make_opt=opt, between=between)
    snapshot.clear()
    new = simulate(monkeypatch, (True, True), make_opt=opt, between=between, invalidate=("hook",))
    assert_same_run(ref, new, "load_state_dict mid-accumulation")


def test_sampling_and_validation_between_micro_steps(came_cls, monkeypatch):
    """A sample round (eval mode, no_grad, block compile paused, lycoris_fast.cached_lokr_weights patching the Linears'
    forwards) and a validation forward (train-mode DiT under no_grad, the network in eval mode) between micro-steps:
    neither reads the checkpoint cache nor leaves it wrong, and both give the flags-off values."""
    seen = {}

    def between(i, unet, net):
        if i not in (1, 4, 6):
            return
        x, t, ctx, pad = micro_inputs(24, 99, True, torch.bfloat16)
        laf.set_timestep(net, t.float() * 1000.0)
        net.eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            val = unet(x, t, ctx, padding_mask=pad).float().clone()  # validation: the DiT stays in train mode
        unet.eval()
        with torch.no_grad(), bc.pause(), torch.autocast("cpu", dtype=torch.bfloat16), \
                lycoris_fast.cached_lokr_weights(unet, torch.bfloat16, park_base=False) as cached:
            sample = unet(x, t, ctx, padding_mask=pad).float().clone()
        assert cached > 0
        unet.train()
        net.train()
        seen.setdefault(i, []).append((val, sample))

    opt = lambda params: make_came(came_cls, params)  # noqa: E731
    ref = simulate(monkeypatch, (False, False), make_opt=opt, between=between)
    new = simulate(monkeypatch, (True, True), make_opt=opt, between=between)
    assert_same_run(ref, new, "sampling / validation between micro-steps")
    for i, ((val0, sample0), (val1, sample1)) in seen.items():
        assert torch.equal(val0, val1) and torch.equal(sample0, sample1), f"validation / sample after micro-step {i}"


def test_text_conditioning_gradient_through_many_blocks(monkeypatch):
    """The trainer marks the text conditioning as grad-bearing, so every Block's node hands back a gradient for the
    cross-attention context. The node sums a Block's contributions (k_proj, v_proj) before they join the other Blocks',
    the plain checkpoint adds them one by one: the LoKr / DoRA gradients must be bit-identical (they do not depend on that
    sum); the context's own gradient is summed in another order, in the context's dtype (bf16 here, fp16 in the user's
    run), so it differs by half-precision rounding of the partial sums. It feeds nothing trainable with
    train_llm_adapter=False; with a trained llm_adapter / text encoder those gradients are NOT bit-identical to the plain
    checkpoint's (docs/anima_train_network.md says so since the review; the trainer warns at start-up:
    tests/test_anima_ckpt_reuse.py::test_text_training_and_unapplied_flags_are_warned_about)."""
    def run(flags):
        unet, net, _ = build_model(blocks=4)
        set_flags(monkeypatch, *flags)
        net.zero_grad(set_to_none=True)
        y, ctx = forward(unet, net, micro_inputs(64, 5, True, torch.bfloat16), torch.bfloat16)
        y.float().square().mean().backward()
        return y.detach().float(), {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}, ctx.grad

    (y0, g0, c0), (y1, g1, c1) = run((False, False)), run((True, True))
    assert torch.equal(y0, y1) and g0.keys() == g1.keys()
    assert all(torch.equal(g0[n], g1[n]) for n in g0), [n for n in g0 if not torch.equal(g0[n], g1[n])][:3]
    assert c0 is not None and c1 is not None and bool(c0.float().abs().max() > 0)
    d, top = (c0.float() - c1.float()).abs().max().item(), c0.float().abs().max().item()
    # recorded for the report: how far the context gradient moves (a bf16 leaf here: the per-Block partial sums are
    # rounded to bf16, so the difference is a few bf16 ulps of the SUMMANDS - large relative to an element that cancels)
    print(f"\ncontext grad: max |d| = {d:.3e}, max |g| = {top:.3e}, differing {(c0 != c1).sum().item()} of {c0.numel()}")
    assert d <= 2 ** -5 * top, (d, top)


# ----------------------------------------------------------------------------------------------- autograd corner cases


def test_second_backward_with_retain_graph(monkeypatch):
    """backward(retain_graph=True) twice: the node's second backward has no stored attention left (dropped when the first
    one started) and recomputes plainly; the accumulated gradients equal the plain checkpoint's."""
    def run(flags):
        unet, net, _ = build_model(blocks=2)
        set_flags(monkeypatch, *flags, diffs=False)
        net.zero_grad(set_to_none=True)
        y, _ = forward(unet, net, micro_inputs(36, 3, True, torch.bfloat16), torch.bfloat16)
        loss = y.float().square().mean()
        loss.backward(retain_graph=True)
        first = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
        loss.backward()
        return first, {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}

    (a0, b0), (a1, b1) = run((False, False)), run((True, True))
    assert all(torch.equal(a0[n], a1[n]) for n in a0), "first backward"
    assert all(torch.equal(b0[n], b1[n]) for n in b0), "second backward"
    assert any(not torch.equal(a0[n], b0[n]) for n in a0)  # the second pass did accumulate


def test_backward_called_under_autocast(monkeypatch):
    """The recompute re-enters the forward's autocast and the inner backward runs in the caller's state, as the plain
    checkpoint's nodes do: a backward() issued inside an autocast block gives the plain checkpoint's gradients too."""
    def run(flags):
        unet, net, _ = build_model(blocks=2)
        set_flags(monkeypatch, *flags)
        net.zero_grad(set_to_none=True)
        y, _ = forward(unet, net, micro_inputs(36, 3, True, torch.bfloat16), torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y.float().square().mean().backward()
        return {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}

    g0, g1 = run((False, False)), run((True, True))
    assert g0.keys() == g1.keys() and all(torch.equal(g0[n], g1[n]) for n in g0)


def test_exception_inside_a_first_pass_leaves_the_switches_clean(monkeypatch):
    """A first pass that raises (an OOM in the trainer): lycoris_fast._CKPT_PASS is back to 0 and the next micro-step
    gives the plain checkpoint's result."""
    unet, net, _ = build_model(blocks=2)
    set_flags(monkeypatch, True, True)
    data = micro_inputs(36, 3, True, torch.bfloat16)
    ref_y, _ = forward(unet, net, data, torch.bfloat16)
    ref_y.float().square().mean().backward()
    ref = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
    net.zero_grad(set_to_none=True)

    block = unet.blocks[1]
    block.cross_attn.forward = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("out of memory (test)"))
    with pytest.raises(RuntimeError, match="out of memory"):
        forward(unet, net, data, torch.bfloat16)
    del block.cross_attn.__dict__["forward"]  # the class method shows through again
    assert lycoris_fast._CKPT_PASS == 0
    gc.collect()
    y, _ = forward(unet, net, data, torch.bfloat16)
    y.float().square().mean().backward()
    assert torch.equal(y, ref_y)
    assert all(torch.equal(ref[n], p.grad) for n, p in net.named_parameters() if p.grad is not None)


# --------------------------------------------------------------------------------------- compile / memory / state files


def test_graph_count_is_flat_across_optimizer_steps(came_cls, monkeypatch):
    """Compiled, mixed tiers, 3 optimizer steps of CAME: after the first optimizer step's micro-steps have compiled the
    graphs of every (checkpoint pass, tier mode), the cache refreshes of the later steps compile nothing - the cached
    weights are graph inputs, their keys are not guarded."""
    counts = {}
    for name, flags in (("off", (False, False)), ("on", (True, True))):
        backend, calls = T.counting_backend("aot_eager")
        seen = []
        simulate(monkeypatch, flags, make_opt=lambda params: make_came(came_cls, params), compiled=True,
                 backend=backend, between=lambda i, unet, net: seen.append(len(calls)))
        counts[name] = seen
    print(f"\ngraphs after each micro-step: off {counts['off']}, on {counts['on']}")
    for name, seen in counts.items():
        assert seen[2] == seen[-1], f"flags {name}: graphs kept compiling after the first optimizer step: {seen}"
    # one dtype regime (fp32 stream): flags off 2 graphs here, flags on 3 (the no-grad first pass, the recompute, and the
    # plain grad graph of the un-checkpointed Blocks)
    assert counts["off"][-1] <= 2 and counts["on"][-1] <= 3, counts


def test_no_tensors_pile_up_across_optimizer_steps(came_cls, monkeypatch):
    """Flags on, eager: the number of tensors created by the run and still alive after each optimizer step is flat (no
    stash, recompute graph, cache copy or node context survives its step). Tensors that existed before the run (other
    tests' leftovers, collected whenever) are not counted."""
    import weakref

    gc.collect()
    before = {id(o): weakref.ref(o) for o in gc.get_objects() if isinstance(o, torch.Tensor)}  # by identity
    live = []

    def old(o) -> bool:
        ref = before.get(id(o))
        return ref is not None and ref() is o

    def between(i, unet, net):
        if (i + 1) % 3 == 0:
            gc.collect()
            live.append(sum(1 for o in gc.get_objects() if isinstance(o, torch.Tensor) and not old(o)))

    simulate(monkeypatch, (True, True), make_opt=lambda params: make_came(came_cls, params), between=between,
             tiers=TIERS + (64, 36, 64, 64, 24, 64))
    # the first steps allocate the cache, CAME's state and the RoPE tables of each grid; after that only simulate()'s
    # own record grows (one output tensor per micro-step)
    assert len(live) == 5 and live[3] - live[2] == 3 and live[4] - live[3] == 3, live


def test_state_dicts_do_not_see_the_cache(monkeypatch):
    """The cache lives in the modules' __dict__, not in buffers: saving the LoRA / a training state is unchanged."""
    unet, net, _ = build_model(blocks=2)
    before = (list(net.state_dict().keys()), list(unet.state_dict().keys()))
    set_flags(monkeypatch, True, True)
    y, _ = forward(unet, net, micro_inputs(36, 3, True, torch.bfloat16), torch.bfloat16)
    y.float().square().mean().backward()
    assert lycoris_fast.ckpt_merged_bytes(unet) > 0
    assert (list(net.state_dict().keys()), list(unet.state_dict().keys())) == before
    assert not any("_ckpt" in k for k in before[0] + before[1])


# ------------------------------------------------------------------------------------------ multiplier / two forwards


def test_network_multiplier_other_than_one(monkeypatch):
    """iLECO / ADDifT / differential output preservation set the LyCORIS multiplier per forward. The fixed-path LoKr
    forward leaves the fast path for m != 1 (lycoris_algo_fixes._lokr_forward: W0 + m (W1 - W0)), so the cached weight -
    built by the stock formula - must not be what a checkpointed Block computes with: m = 1, 0.5, 0, 1 again, each
    micro-step equal to the plain checkpoint's."""
    def run(flags):
        unet, net, _ = build_model(blocks=2)
        set_flags(monkeypatch, *flags)
        results = []
        for i, m in enumerate((1.0, 0.5, 0.0, 1.0)):
            net.set_multiplier(m)
            net.zero_grad(set_to_none=True)
            y, _ = forward(unet, net, micro_inputs(36, i + 1, True, torch.bfloat16), torch.bfloat16)
            y.float().square().mean().backward()
            results.append((y.detach().float().clone(),
                            {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}))
        return results

    ref, new = run((False, False)), run((True, True))
    assert not torch.equal(ref[0][0], ref[1][0])  # same weights, another multiplier: another output
    for i, ((y0, g0), (y1, g1)) in enumerate(zip(ref, new)):
        assert torch.equal(y0, y1), f"multiplier step {i}: output"
        assert g0.keys() == g1.keys() and all(torch.equal(g0[n], g1[n]) for n in g0), f"multiplier step {i}: grads"


def test_two_forwards_before_one_backward(monkeypatch):
    """Two training forwards (different inputs and tiers) whose losses are summed into one backward: each Block has two
    nodes alive at once, with their own stored attention and the same cached weights; the gradients are the plain
    checkpoint's."""
    def run(flags):
        unet, net, _ = build_model(blocks=3)
        set_flags(monkeypatch, *flags)
        net.zero_grad(set_to_none=True)
        y1, _ = forward(unet, net, micro_inputs(36, 3, True, torch.bfloat16), torch.bfloat16)
        y2, _ = forward(unet, net, micro_inputs(64, 4, True, torch.bfloat16), torch.bfloat16)
        (y1.float().square().mean() + 0.5 * y2.float().square().mean()).backward()
        return y1.detach().float(), y2.detach().float(), {n: p.grad.clone() for n, p in net.named_parameters()
                                                          if p.grad is not None}

    (a0, b0, g0), (a1, b1, g1) = run((False, False)), run((True, True))
    assert torch.equal(a0, a1) and torch.equal(b0, b1)
    assert g0.keys() == g1.keys() and all(torch.equal(g0[n], g1[n]) for n in g0)
    assert set(anima_models.CKPT_RECOMPUTE_DIFFS) == {0.0} and len(anima_models.CKPT_RECOMPUTE_DIFFS) == 6


def test_backward_from_another_thread(monkeypatch):
    """On CUDA the autograd engine runs the node's backward in a device worker thread, not in the thread that ran the
    forward. A CPU backward issued from a second Python thread is the nearest thing here: that thread starts with its own
    thread-local state (no autocast), so the recompute only sees the forward's autocast because the node re-enters it,
    and lycoris_fast._CKPT_PASS (a module global) is set from that thread. Gradients equal the plain checkpoint's."""
    import threading

    def run(flags):
        unet, net, _ = build_model(blocks=2)
        set_flags(monkeypatch, *flags)
        net.zero_grad(set_to_none=True)
        y, _ = forward(unet, net, micro_inputs(36, 3, True, torch.bfloat16), torch.bfloat16)
        loss = y.float().square().mean()
        errors = []

        def work():
            try:
                assert not torch.is_autocast_enabled("cpu")
                loss.backward()
            except BaseException as error:  # noqa: BLE001 - reported in the main thread
                errors.append(error)

        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
        assert not errors, errors
        return {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}

    g0, g1 = run((False, False)), run((True, True))
    assert g0.keys() == g1.keys() and all(torch.equal(g0[n], g1[n]) for n in g0)
    assert set(anima_models.CKPT_RECOMPUTE_DIFFS) == {0.0}


# ----------------------------------------------- findings of the 2026-11-14 review (strict xfail then; fixed the same day)


_STOCK_LYCORIS_PROBE = r"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.getcwd())
import torch
torch.set_num_threads(2)
from library import anima_models, lycoris_fast
from library import lycoris_algo_fixes as laf
from tests import test_anima_block_compile as T

laf.apply_module_patches()  # the fixed paths; install_fast_lokr_forward is NOT called (--anima_stock_lycoris)
assert lycoris_fast._STOCK_FORWARD is None
anima_models.CKPT_NOGRAD_MERGED = True
anima_models.CKPT_RECOMPUTE_DIFFS = []  # one entry per NoGradFirstPassCheckpointer backward: the node really ran
unet, net, _ = T.build("lokr_dora", num_blocks=1, block_compile=False)
unet.enable_gradient_checkpointing()
with torch.backends.mkldnn.flags(enabled=False):
    T.step(unet, net, *T.inputs(), checkpoint=True)
print("NODES", len(anima_models.CKPT_RECOMPUTE_DIFFS), "MAXDIFF", max(anima_models.CKPT_RECOMPUTE_DIFFS, default=-1.0))
print("CACHE_BYTES", lycoris_fast.ckpt_merged_bytes(unet))
"""


def test_stock_lycoris_forward_builds_no_cache(monkeypatch):
    """The defensive half of the fix. --anima_ckpt_nograd_merged with --anima_stock_lycoris: prepare_anima_runtime does
    not install the fast LoKr+DoRA forward, so nothing would ever read lycoris_fast's checkpoint cache (Anima 2.9B: 4.69
    GiB and ~400 eager rebuilds per optimizer step for nothing; 122,880 bytes on this 1-Block toy before the fix).
    lycoris_fast.ckpt_merged_modules now returns nothing while that forward is not installed: the no-grad first pass
    still runs (the node's backward is counted), nothing is cached. A fresh interpreter: the fast forward is a
    class-level patch this process already carries. The trainer refuses the combination before it gets here
    (test_stock_lycoris_is_refused_at_startup)."""
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    done = subprocess.run([sys.executable, "-B", "-c", _STOCK_LYCORIS_PROBE], cwd=root, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    nodes = done.stdout.rsplit("NODES", 1)[1].split()
    assert int(nodes[0]) == 1 and float(nodes[2]) == 0.0, nodes  # the no-grad checkpoint ran, recompute == first pass
    held = int(done.stdout.rsplit("CACHE_BYTES", 1)[1].split()[0])
    assert held == 0, f"{held} bytes of merged weights cached that no forward can read"

    # the same rule in this process, where the fast forward IS installed (_env): the Block's modules are found, and
    # are not once lycoris_fast forgets its stock forward (= never installed)
    unet, net, _ = build_model(blocks=1)
    block = unet.blocks[0]
    linears = [m for name, m in block.named_modules()
               if isinstance(m, torch.nn.Linear) and not name.startswith("adaln_modulation")]
    assert lycoris_fast._STOCK_FORWARD is not None and len(lycoris_fast.ckpt_merged_modules(block, linears)) > 0
    monkeypatch.setattr(lycoris_fast, "_STOCK_FORWARD", None)
    assert lycoris_fast.ckpt_merged_modules(block, linears) == []
    assert anima_models._ckpt_cached_lokrs(block) == [] and lycoris_fast.ckpt_merged_bytes(unet) == 0


def test_stock_lycoris_is_refused_at_startup():
    """The other half: validate_ckpt_reuse_args (AnimaNetworkTrainer.assert_extra_args) refuses --anima_ckpt_nograd_merged
    with --anima_stock_lycoris, on the real parser's attribute names; either option alone passes."""
    import anima_train_network as atn

    parser = atn.setup_parser()
    base = ["--timestep_sampling", "sigmoid", "--gradient_checkpointing"]
    both = parser.parse_args(base + ["--anima_ckpt_nograd_merged", "--anima_stock_lycoris"])
    assert both.anima_ckpt_nograd_merged is True and both.anima_stock_lycoris is True
    with pytest.raises(ValueError, match="not with --anima_stock_lycoris"):
        atn.validate_ckpt_reuse_args(both)
    atn.validate_ckpt_reuse_args(parser.parse_args(base + ["--anima_ckpt_nograd_merged"]))
    atn.validate_ckpt_reuse_args(parser.parse_args(base + ["--anima_stock_lycoris"]))
    assert "validate_ckpt_reuse_args(args)" in __import__("inspect").getsource(atn.AnimaNetworkTrainer.assert_extra_args)


def test_anima38_trainer_invalidates_after_sample_images(monkeypatch):
    """The docs name three invalidation points: every optimizer.step(), the end of the trainer's optimizer step, after
    sample images. The Anima-3.8B trainer (--anima38_connector: Anima38TrainerMixin in front of AnimaNetworkTrainer)
    overrides sample_images without calling the Anima one and used to miss the third (harmless then: sampling does not
    change adapter values and every sample round follows an optimizer step whose invalidation is still pending). Both
    sample_images now end in AnimaNetworkTrainer.after_sample_images: one invalidation per sample round with the switch
    on, none with it off."""
    import anima_train_network as atn
    from library.anima38 import sampling
    from library.anima38.trainer import make_anima38_trainer

    trainer = make_anima38_trainer(atn.AnimaNetworkTrainer)
    sampled = []
    monkeypatch.setattr(sampling, "sample_images", lambda *a, **k: sampled.append(lycoris_fast._CKPT_GENERATION))
    monkeypatch.setattr(type(trainer), "_anima38_update_grad_policy", lambda self: None)
    args = argparse.Namespace(anima38_sample_negative=None)
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", True)
    before = lycoris_fast._CKPT_GENERATION
    trainer.sample_images(None, args, 0, 1, "cpu", None, None, [None], None)
    assert lycoris_fast._CKPT_GENERATION == before + 1
    assert sampled == [before], "the invalidation comes after the sampler ran, once"
    monkeypatch.setattr(anima_models, "CKPT_NOGRAD_MERGED", False)
    trainer.sample_images(None, args, 0, 1, "cpu", None, None, [None], None)
    assert lycoris_fast._CKPT_GENERATION == before + 1 and len(sampled) == 2
