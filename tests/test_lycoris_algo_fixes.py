"""CPU tests for the LyCORIS fixed paths (library/lycoris_algo_fixes.py; default on, off with --lycoris_stock).

Tiny Linear/Conv models built like Anima where it matters: the base weights are bf16, training runs under bf16 autocast,
and one Linear (an "adaLN" layer) runs with autocast off on a bf16 input. For every covered algo: the patched
forward/backward works there, every trainable tensor gets a gradient and is in the optimizer, the saved file loads in
Forge's own weight adapters (C:/sd-webui-forge-classic/modules_forge/packages/comfy/weight_adapter, imported with a
two-function backend stub) and reproduces the trained weight; DoRA on (both axes) and off where supported.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -B -m pytest tests/test_lycoris_algo_fixes.py
"""
import argparse
import copy
import importlib
import json
import logging
import math
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from library import lycoris_algo_fixes as laf

torch.set_num_threads(4)
torch.backends.mkldnn.enabled = False  # this CPU's oneDNN refuses some bf16 backward kernels; only picks other kernels

FORGE = r"C:/sd-webui-forge-classic"
LOG = logging.getLogger("test_lycoris_algo_fixes")

PRESET = {
    "enable_conv": True,
    "unet_target_module": ["ToyBlock"],
    "unet_target_name": [],
    "text_encoder_target_module": [],
    "text_encoder_target_name": [],
    "exclude_name": [],
    "module_algo_map": {},
    "name_algo_map": {},
}


# ---------------------------------------------------------------------------------------------------------------- model


class ToyBlock(nn.Module):
    def __init__(self, dim=32, hidden=64, cond=16, conv=False):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.mlp1 = nn.Linear(dim, hidden)
        self.mlp2 = nn.Linear(hidden, dim)
        self.ada = nn.Linear(cond, 2 * dim)  # Anima adaLN: autocast off, bf16 input, bf16 weight
        self.conv = nn.Conv2d(8, 16, 3, padding=1) if conv else None

    def forward(self, x, c, img=None):
        with torch.autocast("cpu", enabled=False):
            mod = self.ada(c)
        shift, scale = mod.chunk(2, -1)
        h = x * (1 + scale[:, None, None, :].to(x.dtype)) + shift[:, None, None, :].to(x.dtype)
        h = h + self.q(h)
        h = h + self.mlp2(F.gelu(self.mlp1(h)))
        out_img = self.conv(img) if (self.conv is not None and img is not None) else None
        return h, out_img


class ToyNet(nn.Module):
    def __init__(self, conv=False, blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([ToyBlock(conv=conv) for _ in range(blocks)])

    def forward(self, x, c, img=None, checkpoint=False):
        outs = []
        for b in self.blocks:
            if checkpoint:
                x, oi = torch.utils.checkpoint.checkpoint(b, x, c, img, use_reentrant=False)
            else:
                x, oi = b(x, c, img)
            if oi is not None:
                outs.append(oi)
        return x, outs


def make_model(conv=False, seed=0, dtype=torch.bfloat16):
    """Base weights are always the bf16 values (as trained); dtype=float32 gives an exact fp32 copy of them."""
    torch.manual_seed(seed)
    m = ToyNet(conv=conv)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(torch.bfloat16).to(dtype)


def make_batch(seed, B=2, conv=False):
    g = torch.Generator().manual_seed(seed)
    b = {
        "x": torch.randn(B, 1, 5, 32, generator=g).bfloat16(),  # (B, frames=1, tokens, C): >3-D like Anima
        "c": torch.randn(B, 16, generator=g).bfloat16(),
        "y": torch.randn(B, 1, 5, 32, generator=g),
    }
    if conv:
        b["img"] = torch.randn(B, 8, 6, 6, generator=g).bfloat16()
        b["img_y"] = torch.randn(B, 16, 6, 6, generator=g)
    return b


def loss_fn(model, batch, checkpoint=False):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        h, outs = model(batch["x"], batch["c"], batch.get("img"), checkpoint=checkpoint)
    loss = F.mse_loss(h.float(), batch["y"])
    for o in outs:
        loss = loss + F.mse_loss(o.float(), batch["img_y"])
    return loss


# -------------------------------------------------------------------------------------------------------------- network


def make_net(model, algo, dim, alpha, **kwargs):
    from lycoris import kohya

    raw = {"algo": algo, **{k: str(v) for k, v in kwargs.items()}}
    kw = laf.prepare_network_kwargs(algo, raw, LOG)
    net = kohya.create_network(1.0, dim, alpha, None, [], model, preset=dict(PRESET), **kw)
    net.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    args = argparse.Namespace(network_args=[f"{k}={v}" for k, v in raw.items()], network_dim=dim, network_alpha=alpha,
                              lycoris_algo_fixes=True)
    laf.after_network_created(net, args, LOG)
    net.prepare_grad_etc(None, model)
    return net


def build_optimizer(net, lr):
    res = net.prepare_optimizer_params(text_encoder_lr=None, unet_lr=lr, learning_rate=lr)
    groups = res[0] if isinstance(res, tuple) else res
    opt = torch.optim.AdamW(groups, lr=lr)
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    names = {id(p): n for n, p in net.named_parameters()}
    missing = [names[id(p)] for p in net.parameters() if p.requires_grad and id(p) not in in_opt]
    assert not missing, f"trainable but not in the optimizer: {missing[:5]}"
    return opt


def train(model, net, steps=3, lr=5e-3, conv=False, timesteps=None, checkpoint=False, B=2):
    net.train()
    opt = build_optimizer(net, lr)
    trainable = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    got = {n: False for n, _ in trainable}
    before = {n: p.detach().clone() for n, p in trainable}
    for step in range(steps):
        batch = make_batch(100 + step, B=B, conv=conv)
        if timesteps is not None:
            laf.set_timestep(net, timesteps)
        loss = loss_fn(model, batch, checkpoint=checkpoint)
        loss.backward()
        if timesteps is not None:
            laf.clear_timestep(net)  # after backward (checkpointing recomputes the forward inside it)
        for n, p in trainable:
            if p.grad is not None and bool(torch.any(p.grad != 0)):
                got[n] = True
        opt.step()
        opt.zero_grad(set_to_none=True)
    no_grad = [n for n, ok in got.items() if not ok]
    unchanged = [n for n, p in trainable if torch.equal(before[n], p.detach())]
    return no_grad, unchanged


def loras(net):
    return laf._all_loras(net)


def to_fp32_eval(model, net):
    model.float()
    net.float()
    net.eval()


# ---------------------------------------------------------------------------------------------------------------- Forge


_FORGE = None


def load_forge():
    global _FORGE
    if _FORGE is not None:
        return _FORGE
    if not os.path.isdir(os.path.join(FORGE, "modules_forge", "packages", "comfy", "weight_adapter")):
        pytest.skip("Forge not found")
    existing = sys.modules.get("backend")
    if existing is not None and not getattr(existing, "_laf_stub", False):
        pytest.skip("a real 'backend' package is imported")
    backend = types.ModuleType("backend")
    backend.__path__ = []
    backend._laf_stub = True
    mm = types.ModuleType("backend.memory_management")

    def cast_to_device(tensor, device, dtype, copy=False):  # backend/memory_management.py cast_to_device on CPU
        if device is None or tensor.device == torch.device(device):
            if not copy and (dtype is None or tensor.dtype == dtype):
                return tensor
            return tensor.to(dtype=dtype, copy=copy)
        return tensor.to(dtype=dtype, device=device, copy=copy)

    mm.cast_to_device = cast_to_device
    mm.is_oom = lambda e: isinstance(e, torch.OutOfMemoryError)
    backend.memory_management = mm
    sys.modules["backend"] = backend
    sys.modules["backend.memory_management"] = mm
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # nothing written into the Forge tree
    sys.path.insert(0, FORGE)
    try:
        lora = importlib.import_module("modules_forge.packages.comfy.lora")
    finally:
        sys.path.remove(FORGE)
        sys.dont_write_bytecode = old
    _FORGE = lora
    return lora


def forge_patches(net, sd):
    lora = load_forge()
    key_map = {m.lora_name: f"diffusion_model.{m.original_name}.weight" for m in loras(net)}
    patch, remaining = lora.load_lora(dict(sd), key_map)
    return key_map, patch, remaining


def rel(a, b, ref):
    d = (ref).norm()
    return float((a - b).norm() / d) if d > 0 else float((a - b).norm())


def merged_reference(m):
    """The trained weight: get_merged_weight, after checking it is what the module's forward computes."""
    W, _ = m.get_merged_weight(multiplier=1)
    W = W.detach().float().reshape(m.org_module[0].weight.shape)
    org = m.org_module[0]
    with torch.no_grad():
        if isinstance(org, nn.Linear):
            x = torch.randn(3, org.in_features)
            y_mod = m(x).float()
            y_ref = F.linear(x, W, org.bias)
        else:
            x = torch.randn(2, org.in_channels, 6, 6)
            y_mod = m(x).float()
            y_ref = F.conv2d(x, W, org.bias, org.stride, org.padding, org.dilation, org.groups)
    err = float((y_mod - y_ref).norm() / y_ref.norm())
    assert err < 1e-5, f"{m.lora_name}: get_merged_weight differs from the forward ({err:.3g})"
    return W


def check_forge(net, path, dora_axis=None, tol=1e-4, allow_remaining=()):
    """Forge's adapters on the saved file vs the trained weights (fp32 merge, strength 1)."""
    from safetensors.torch import load_file

    sd = load_file(path)
    key_map, patch, remaining = forge_patches(net, sd)
    left = [k for k in remaining if not k.endswith(tuple(allow_remaining))]
    assert not left, f"keys Forge did not use: {left[:5]}"
    not_patched = [k for k, v in key_map.items() if v not in patch]
    assert not not_patched, f"modules Forge skipped: {not_patched[:5]}"
    raw_patch = None
    if dora_axis == "output":
        _, raw_patch, _ = forge_patches(net, {k: v for k, v in sd.items() if not k.endswith(".dora_scale")})
    worst, stock_dora = 0.0, 0.0
    for m in loras(net):
        key = key_map[m.lora_name]
        W0 = m.org_module[0].weight.detach().float()
        ref = merged_reference(m)
        res = patch[key].calculate_weight(W0.clone(), key, 1.0, 1.0, None, lambda a: a, torch.float32)
        if dora_axis == "output":
            # Stock Forge divides by ||W0|| on the output axis (weight_adapter/base.py:45-47), LyCORIS by ||W0+dW||: a
            # Forge property, not the file. The file must give the trained weight with the LyCORIS norm (sam-extra
            # "lycoris" DoRA mode); stock Forge must at least pick the output axis.
            ds = sd[m.lora_name + ".dora_scale"].float()
            assert ds.shape[0] == W0.shape[0] and ds[0].numel() == 1
            stock_dora = max(stock_dora, rel(res, ref, ref - W0))
            raw = raw_patch[key].calculate_weight(W0.clone(), key, 1.0, 1.0, None, lambda a: a, torch.float32)
            norm = torch.linalg.vector_norm(raw, dim=tuple(range(1, raw.dim())), keepdim=True)
            res = raw * (ds / (norm + torch.finfo(torch.float32).eps))
        worst = max(worst, rel(res, ref, ref - W0))
    print(f"[forge] {os.path.basename(path)} dora={dora_axis}: max |Forge - trained| / |trained change| = {worst:.3g}"
          + (f", stock Forge output-axis DoRA = {stock_dora:.3g}" if dora_axis == "output" else ""))
    assert worst < tol, f"Forge differs from training by {worst:.3g} of the trained change"
    return worst, stock_dora


def save_both(net, tmp_path, name):
    p32 = str(tmp_path / f"{name}_fp32.safetensors")
    p16 = str(tmp_path / f"{name}_fp16.safetensors")
    net.save_weights(p32, torch.float32, {"ss_network_dim": "x"})
    net.save_weights(p16, torch.float16, {})
    return p32, p16


def metadata(path):
    from safetensors import safe_open

    with safe_open(path, "pt") as f:
        return f.metadata() or {}


def model_out(model, seed=7, conv=False):
    b = make_batch(seed, conv=conv)
    b = {k: v.float() for k, v in b.items()}
    with torch.no_grad():
        h, outs = model(b["x"], b["c"], b.get("img"))
    return torch.cat([h.reshape(-1)] + [o.reshape(-1) for o in outs])


# -------------------------------------------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _patched():
    laf.apply_module_patches()
    from lycoris.modules.locon import GoRAModule, RaLoRAModule
    from lycoris.modules.lora2 import LoRA2Module

    GoRAModule.reset_gora_registry()
    RaLoRAModule.reset_ralora_registry()
    LoRA2Module.reset_lora2_registry()
    yield
    laf.clear_timestep(argparse.Namespace())  # no loras: no-op, keeps the API exercised
    from lycoris.modules.tlora import clear_timestep_mask
    from lycoris.modules.tsm import clear_tsm_timestep

    clear_timestep_mask()
    clear_tsm_timestep()
    laf._TSM_SAMPLE_T.clear()
    laf._revert_module_patches()  # other test files in the same process see stock LyCORIS


@pytest.fixture(scope="module", autouse=True)
def _forget_forge_stub():
    yield
    global _FORGE
    if _FORGE is not None:
        for name in [n for n in sys.modules if n == "backend" or n.startswith(("backend.", "modules_forge"))]:
            if name.startswith("modules_forge") or getattr(sys.modules[name], "_laf_stub", False) or name.startswith("backend."):
                del sys.modules[name]
        _FORGE = None


# ------------------------------------------------------------------------------------------------------------ flag / API


def test_flag_off_patches_nothing():
    from lycoris.modules.loha import LohaModule
    from lycoris.modules import gora_utils

    laf._revert_module_patches()
    assert not getattr(LohaModule.__dict__["get_diff_weight"], "_lycoris_algo_fix", False)
    assert not getattr(gora_utils.allocate_ranks, "_lycoris_algo_fix", False)
    # default ON for LyCORIS (user 2026-10-24: "flag ON -> original"); --lycoris_stock = LyCORIS as installed
    ns = argparse.Namespace
    assert laf.enabled(ns()) is False
    assert laf.enabled(ns(network_module="lycoris.kohya")) is True
    assert laf.enabled(ns(network_module="lycoris.kohya", lycoris_algo_fixes=False)) is True
    assert laf.enabled(ns(network_module="lycoris.kohya", lycoris_algo_fixes=True)) is True
    assert laf.enabled(ns(network_module="lycoris.kohya", lycoris_stock=True)) is False
    assert laf.enabled(ns(network_module="lycoris.kohya", lycoris_stock=True, lycoris_algo_fixes=True)) is False
    assert laf.enabled(ns(network_module="networks.lora")) is False
    assert laf.enabled(ns(network_module="networks.lora", lycoris_algo_fixes=True)) is False
    laf.apply_module_patches()
    n = len(laf.patched_targets())
    laf.apply_module_patches()  # idempotent
    assert len(laf.patched_targets()) == n
    assert getattr(LohaModule.__dict__["get_diff_weight"], "_lycoris_algo_fix", False)


@pytest.mark.parametrize("algo", sorted(laf.DORA_UNSUPPORTED))
def test_dora_unsupported_algos_raise(algo):
    with pytest.raises(ValueError, match="DoRA"):
        laf.prepare_network_kwargs(algo, {"algo": algo, "dora_wd": "True"}, LOG)
    laf.prepare_network_kwargs(algo, {"algo": algo, "dora_wd": "False"}, LOG)


def test_prepare_kwargs_casts_numbers_only():
    kw = laf.prepare_network_kwargs("gora", {"gora_min_rank": "2", "gora_max_rank": "8.0", "gora_adaptive_n": "True"}, LOG)
    assert kw == {"gora_min_rank": 2, "gora_max_rank": 8, "gora_adaptive_n": "True"}
    assert laf.prepare_network_kwargs("lora2", {"lora2_nu_target": "4"}, LOG)["lora2_nu_target"] == 4
    assert laf.prepare_network_kwargs("ralora", {"ralora_min_rank": "1", "ralora_pro": "True"}, LOG) == {
        "ralora_min_rank": 1, "ralora_pro": "True"}
    with pytest.raises(ValueError):
        laf.prepare_network_kwargs("gora", {"gora_min_rank": "9", "gora_max_rank": "4"}, LOG)
    with pytest.raises(ValueError):
        laf.prepare_network_kwargs("lora2", {"lora2_nu_target": "abc"}, LOG)
    assert laf.prepare_network_kwargs("lokr", {"factor": "4"}, LOG) == {"factor": "4"}  # not covered: untouched


# --------------------------------------------------------------------------------------------- stock bugs still present


def _with_stock(fn):
    laf._revert_module_patches()
    try:
        return fn()
    finally:
        laf.apply_module_patches()


@pytest.mark.parametrize("algo", ["diag-oft", "boft", "tlora"])
def test_stock_bf16_dtype_crash_and_fix(algo):
    def run():
        model = make_model()
        net = make_net(model, algo, 8, 8)
        return loss_fn(model, make_batch(1))

    with pytest.raises(RuntimeError):
        _with_stock(run)
    model = make_model()
    net = make_net(model, algo, 8, 8)
    loss_fn(model, make_batch(1)).backward()


def test_stock_loha_merge_scale_bug_and_fix():
    def merged_err():
        model = make_model()
        net = make_net(model, "loha", 8, 4)
        for m in loras(net):
            with torch.no_grad():
                m.hada_w2_a.normal_(std=0.1)
        to_fp32_eval(model, net)
        try:
            for m in loras(net):
                merged_reference(m)
        except AssertionError:
            return "mismatch"
        return "ok"

    assert _with_stock(merged_err) == "mismatch"
    assert merged_err() == "ok"


# --------------------------------------------------------------------------------------------------------------- LoHa


@pytest.mark.parametrize("dora", [None, "output", "input"])
def test_loha(tmp_path, dora):
    kw = {"dora_wd": True, "wd_on_output": dora == "output"} if dora else {}
    model = make_model(conv=True)
    net = make_net(model, "loha", 8, 4, **kw)  # alpha != dim: the merge bug's case
    no_grad, unchanged = train(model, net, conv=True)
    assert not no_grad and not unchanged, (no_grad, unchanged)
    to_fp32_eval(model, net)
    p32, p16 = save_both(net, tmp_path, "loha")
    md = metadata(p32)
    assert md[laf.METADATA_FLAG_KEY] == "1" and md[laf.METADATA_FORMAT_KEY] == "loha"
    check_forge(net, p32, dora_axis=dora)
    check_forge(net, p16, dora_axis=dora, tol=5e-2)
    # create_network_from_weights (--dim_from_weights / --base_weights / merge tools) incl. DoRA on the input axis
    from lycoris import kohya

    ref = model_out(model, conv=True)
    model2 = make_model(conv=True, dtype=torch.float32)
    net2, _ = kohya.create_network_from_weights(1.0, p32, None, None, model2)
    net2.apply_to(None, model2, False, True)
    net2.eval()
    assert rel(model_out(model2, conv=True), ref, ref) < 1e-5


# --------------------------------------------------------------------------------------------------------------- ABBA


@pytest.mark.parametrize("dora", [None, "output", "input"])
def test_abba(tmp_path, dora):
    kw = {"dora_wd": True, "wd_on_output": dora == "output"} if dora else {}
    model = make_model(conv=True)
    net = make_net(model, "abba", 7, 4, **kw)  # odd dim: r1=3, r2=4
    no_grad, unchanged = train(model, net, conv=True)
    assert not no_grad and not unchanged, (no_grad, unchanged)
    to_fp32_eval(model, net)
    p32, p16 = save_both(net, tmp_path, "abba")
    assert metadata(p32)[laf.METADATA_FORMAT_KEY] == "loha"
    check_forge(net, p32, dora_axis=dora)
    check_forge(net, p16, dora_axis=dora, tol=5e-2)
    ref = model_out(model, conv=True)
    # LyCORIS reads the converted file back as LoHa (pair 1 padded to r2)
    from lycoris import kohya

    model2 = make_model(conv=True, dtype=torch.float32)
    net2, _ = kohya.create_network_from_weights(1.0, p32, None, None, model2)
    net2.apply_to(None, model2, False, True)
    net2.eval()
    assert {type(m).__name__ for m in net2.unet_loras} == {"LohaModule"}
    assert rel(model_out(model2, conv=True), ref, ref) < 1e-5
    # --network_weights into an ABBA network: converted back to ABBA tensors
    model3 = make_model(conv=True)
    net3 = make_net(model3, "abba", 7, 4, **kw)
    info = net3.load_weights(p32)
    assert not info, info
    to_fp32_eval(model3, net3)
    assert rel(model_out(model3, conv=True), ref, ref) < 1e-5


def test_abba_rank_above_layer_width():
    def build():
        model = make_model()
        return make_net(model, "abba", 80, 80)  # r1 = 40 > 16 (ada in) and 32

    with pytest.raises(Exception):
        _with_stock(build)
    net = build()
    m = [m for m in loras(net) if m.original_name.endswith("ada")][0]
    assert m.lora_up1.weight.shape[1] == 40


def test_abba_merge_strength():
    model = make_model()
    net = make_net(model, "abba", 8, 4)
    train(model, net, steps=2)
    to_fp32_eval(model, net)
    for m in loras(net):
        m.multiplier = 0.5
        W, _ = m.get_merged_weight(multiplier=0.5)  # abba.py:371 applied the 0.5 twice
        x = torch.randn(3, m.org_module[0].in_features)
        with torch.no_grad():
            y = m(x)
        assert rel(F.linear(x, W.float(), m.org_module[0].bias), y, y) < 1e-5
        m.multiplier = 1.0


def test_abba_bypass_trains():
    model = make_model()
    net = make_net(model, "abba", 8, 4, bypass_mode=True)
    with torch.no_grad():  # like sample_at_first: a no-grad forward before training
        loss_fn(model, make_batch(0))
    no_grad, unchanged = train(model, net, steps=3)
    assert not no_grad and not unchanged, (no_grad, unchanged)
    to_fp32_eval(model, net)
    for m in loras(net):  # the bypass forward uses the current (updated) weights
        x = torch.randn(3, m.org_module[0].in_features)
        W, _ = m.get_merged_weight(1)
        with torch.no_grad():
            y = m(x)
        assert rel(F.linear(x, W.float(), m.org_module[0].bias), y, y) < 1e-5


# --------------------------------------------------------------------------------------------------------------- GoRA


def _prepass_batches(n=3):
    return [make_batch(10 + i) for i in range(n)]


def _base_grad_mean(model, net, batches):
    """Mean base-weight gradient over the batches, measured independently of the pre-pass."""
    ws = [laf._org_weight(m) for m in loras(net)]
    for w in ws:
        w.requires_grad_(True)
    sums = [torch.zeros_like(w, dtype=torch.float32) for w in ws]
    for b in batches:
        gs = torch.autograd.grad(loss_fn(model, b), ws)
        for s, g in zip(sums, gs):
            s += g.float()
    for w in ws:
        w.requires_grad_(False)
    return [s / len(batches) for s in sums]


@pytest.mark.parametrize("dora", [False, True])
def test_gora(tmp_path, dora):
    kw = {"dora_wd": True, "wd_on_output": True} if dora else {}
    model = make_model()
    net = make_net(model, "gora", 4, 4, gora_min_rank="2", gora_max_rank="8", **kw)  # strings: kohya.py:304-305
    batches = _prepass_batches()
    G = _base_grad_mean(model, net, batches)
    net.prepare_gora(dataloader=batches, forward_fn=lambda b: loss_fn(model, b), max_steps=3, adaptive_n=False)
    assert not net._gora_needs_init
    ranks = set()
    for m, g in zip(loras(net), G):
        w = laf._org_weight(m)
        assert w.grad is None and not w.requires_grad  # gora_utils.py:58/:785 kept the base .grad
        A0 = m.lora_up.weight.detach().double()
        PA = A0 @ torch.linalg.pinv(A0)
        xi = 0.05 * math.sqrt(A0.shape[0]) / 4.0
        ratio = float((A0 @ m.lora_down.weight.detach().double()).norm() / (xi * (PA @ g.double()).norm()))
        assert abs(ratio - 1.0) < 0.05, f"{m.lora_name}: init {ratio:.3f} of the intended size (1/N bug: 0.333)"
        ranks.add(A0.shape[1])
    assert len(ranks) > 1  # ranks were reallocated: the optimizer must be built after the pre-pass
    no_grad, unchanged = train(model, net)
    assert not no_grad and not unchanged, (no_grad, unchanged)
    to_fp32_eval(model, net)
    p32, p16 = save_both(net, tmp_path, "gora")
    md = metadata(p32)
    alloc = json.loads(md[laf.METADATA_ALLOC_KEY])
    assert alloc["algo"] == "gora" and len(alloc["modules"]) == len(loras(net))
    check_forge(net, p32, dora_axis="output" if dora else None)
    check_forge(net, p16, dora_axis="output" if dora else None, tol=5e-2)
    # --network_weights: a fresh network takes the per-layer ranks from the file
    ref = model_out(model)
    model2 = make_model()
    net2 = make_net(model2, "gora", 4, 4, gora_min_rank="2", gora_max_rank="8", **kw)
    info = net2.load_weights(p32)
    assert not info and not net2._gora_needs_init, info
    to_fp32_eval(model2, net2)
    assert rel(model_out(model2), ref, ref) < 1e-5
    # create_network_from_weights: LoCon loader takes all 8 weight_list entries (locon.py:491-493)
    from lycoris import kohya

    model3 = make_model(dtype=torch.float32)
    net3, _ = kohya.create_network_from_weights(1.0, p32, None, None, model3)
    net3.apply_to(None, model3, False, True)
    net3.eval()
    assert rel(model_out(model3), ref, ref) < 1e-5


# ------------------------------------------------------------------------------------------------------------- RaLoRA


def _ralora_prepass(model, net):
    from lycoris.modules.locon import RaLoRAModule

    # the trainer's forward function takes (batch) only (train_network.py:2674); ralora_utils.py:542 passes (model, batch)
    RaLoRAModule.precompute_and_init(model=net, dataloader=_prepass_batches(), forward_fn=lambda b: loss_fn(model, b),
                                     max_steps=3, device=torch.device("cpu"))


def test_ralora_stock_prepass_crashes():
    def run():
        model = make_model()
        net = make_net(model, "ralora", 2, 2)
        _ralora_prepass(model, net)

    with pytest.raises(TypeError):
        _with_stock(run)


@pytest.mark.parametrize("pro,dora", [(False, False), (False, True), (True, False), (True, True)])
def test_ralora(tmp_path, pro, dora):
    kw = {"dora_wd": True, "wd_on_output": True} if dora else {}
    if pro:
        kw.update(ralora_pro=True, ralora_min_rank="1", ralora_max_rank="4")
    model = make_model()
    net = make_net(model, "ralora", 2, 2, **kw)
    _ralora_prepass(model, net)
    mods = loras(net)
    for m in mods:
        w = laf._org_weight(m)
        assert w.grad is None and not w.requires_grad  # ralora_utils.py:516
        for p in list(m._mini_lora_A) + list(m._mini_lora_B):
            assert p.dtype == m.lora_up.weight.dtype and p.device == m.lora_up.weight.device
    assert any(m.n_split > 1 for m in mods), [m.n_split for m in mods]
    if pro:
        assert len({(m.mini_lora_rank if m.n_split > 1 else m.lora_down.weight.shape[0]) for m in mods}) > 1
    state = laf.gora_ralora_allocation_state(net)
    net._ralora_needs_init = False
    net.prepare_grad_etc(None, model)  # the trainer calls it again after the pre-pass (TN:2792): requires_grad_(True)
    no_grad, unchanged = train(model, net)
    assert not no_grad and not unchanged, (no_grad, unchanged)
    # block layers keep the dense lora_up/lora_down they no longer use (never saved): frozen, so not in the optimizer
    for m in mods:
        if m.n_split > 1:
            assert not m.lora_up.weight.requires_grad and not m.lora_down.weight.requires_grad
    to_fp32_eval(model, net)
    p32, p16 = save_both(net, tmp_path, "ralora")
    assert metadata(p32)[laf.METADATA_FORMAT_KEY] == "lora"
    check_forge(net, p32, dora_axis="output" if dora else None)
    check_forge(net, p16, dora_axis="output" if dora else None, tol=5e-2)
    ref = model_out(model)

    # --resume: restore the allocation on a fresh network, then load LyCORIS' own state in place
    model2 = make_model()
    net2 = make_net(model2, "ralora", 2, 2, **kw)
    laf.restore_gora_ralora_allocation(net2, json.loads(json.dumps(state)))
    assert not net2._ralora_needs_init
    ids = {id(p) for p in net2.parameters()}
    res = net2.load_state_dict(net.state_dict(), strict=True)
    assert ids == {id(p) for p in net2.parameters()}  # loaded in place: an optimizer built before still matches
    to_fp32_eval(model2, net2)
    assert rel(model_out(model2), ref, ref) < 1e-5

    # --network_weights with the Forge file: dense per-layer ranks
    model3 = make_model()
    net3 = make_net(model3, "ralora", 2, 2, **kw)
    assert not net3.load_weights(p32)
    to_fp32_eval(model3, net3)
    assert rel(model_out(model3), ref, ref) < 1e-5
    # the trainer restores the file's ss_lycoris_alloc first (block structure), then loads it
    model4 = make_model()
    net4 = make_net(model4, "ralora", 2, 2, **kw)
    laf.restore_gora_ralora_allocation(net4, json.loads(metadata(p32)[laf.METADATA_ALLOC_KEY]))
    assert not net4.load_weights(p32)
    to_fp32_eval(model4, net4)
    assert rel(model_out(model4), ref, ref) < 1e-5


# --------------------------------------------------------------------------------------------------------------- LoRA²


@pytest.mark.parametrize("dora", [None, "output", "input"])
def test_lora2(tmp_path, dora):
    kw = {"dora_wd": True, "wd_on_output": dora == "output"} if dora else {}
    model = make_model(conv=True)
    net = make_net(model, "lora2", 8, 8, lora2_nu_init="0.4", lora2_nu_target="6", **kw)  # D = 6 < 8; string target
    assert all(m.compute_effective_rank() < m.lora_dim for m in loras(net))
    no_grad, unchanged = train(model, net, conv=True)
    assert not no_grad and not unchanged, (no_grad, unchanged)
    to_fp32_eval(model, net)
    p32, p16 = save_both(net, tmp_path, "lora2")
    check_forge(net, p32, dora_axis=dora)
    check_forge(net, p16, dora_axis=dora, tol=5e-2)
    ref = model_out(model, conv=True)
    # the state (resume) is full rank now: it loads back into the same network exactly (lora2.py:372-373 cut it)
    model2 = make_model(conv=True)
    net2 = make_net(model2, "lora2", 8, 8, lora2_nu_init="0.4", lora2_nu_target="6", **kw)
    net2.load_state_dict(net.state_dict(), strict=True)
    to_fp32_eval(model2, net2)
    assert rel(model_out(model2, conv=True), ref, ref) < 1e-6
    # a Forge export cannot continue LoRA² training: clear error
    model3 = make_model(conv=True)
    net3 = make_net(model3, "lora2", 8, 8, **kw)
    with pytest.raises(ValueError, match="LoRA"):
        net3.load_weights(p32)


# ---------------------------------------------------------------------------------------------------------- Diag-OFT/BOFT


@pytest.mark.parametrize("algo", ["diag-oft", "boft"])
def test_oft(tmp_path, algo):
    model = make_model(conv=True)
    net = make_net(model, algo, 8, 8)
    no_grad, unchanged = train(model, net, conv=True)
    assert not no_grad and not unchanged, (no_grad, unchanged)
    to_fp32_eval(model, net)
    p32, p16 = save_both(net, tmp_path, algo)
    check_forge(net, p32, allow_remaining=(".I",))  # LyCORIS saves its identity buffer; Forge ignores it
    check_forge(net, p16, allow_remaining=(".I",), tol=5e-2)


def _oft_classes():
    from lycoris.modules.boft import ButterflyOFTModule
    from lycoris.modules.diag_oft import DiagOFTModule

    return {"diag-oft": DiagOFTModule, "boft": ButterflyOFTModule}


def _get_r64(self):
    """Test-only Cayley transform in fp64 (stock casts r to fp32, so stock BOFT cannot run an fp64 module at all)."""
    q = self.oft_blocks - self.oft_blocks.transpose(-1, -2)
    if self.constraint > 0:
        n = torch.norm(q) + 1e-8
        q = torch.where(n > self.constraint, q * self.constraint / n, q)
    I = self.I.double()
    return (I + q.double()) @ torch.inverse(I - q.double())


class _OftSpeedOff:
    """Inside: the diag-oft/boft classes use the previous flag code (stock forward + the bf16 rebuild fix, stock get_r)."""

    def __enter__(self):
        self.saved = {}
        for cls in _oft_classes().values():
            for name in ("forward", "get_r"):
                self.saved[(cls, name)] = cls.__dict__[name]
                setattr(cls, name, laf._original(cls, name))
        return self

    def __exit__(self, *exc):
        for (cls, name), f in self.saved.items():
            setattr(cls, name, f)


def _oft_module(algo, i, o, dtype, bias=True, bypass=False, mult=1.0, rescaled=False, constraint=0.0, seed=0):
    torch.manual_seed(seed)
    org = nn.Linear(i, o, bias=bias)
    org.requires_grad_(False)
    with torch.no_grad():
        for p in org.parameters():
            p.copy_(p.bfloat16().float())
    m = _oft_classes()[algo]("o", org, mult, lora_dim=8, alpha=8, bypass_mode=bypass, rescaled=rescaled,
                             constraint=constraint)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        m.oft_blocks.copy_(torch.randn(m.oft_blocks.shape, generator=g) * 0.3)
        if rescaled:
            m.rescale.copy_(1 + 0.2 * torch.randn(m.rescale.shape, generator=g))
    m.float()
    if dtype == torch.bfloat16:  # Anima: fp32 network, bf16 base weights
        org.to(torch.bfloat16)
    else:
        m.to(dtype)
        org.to(dtype)
    m.train()
    return m


def _oft_run(m, x, checkpoint=False, autocast=False):
    """Output and the gradients of the input and every parameter for sum(out * w)."""
    x = x.detach().clone().requires_grad_(True)
    for p in m.parameters():
        p.grad = None
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        out = torch.utils.checkpoint.checkpoint(m, x, use_reentrant=False) if checkpoint else m(x)
    w = torch.randn(out.shape, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    (out.double() * w).sum().backward()
    res = {"out": out.detach(), "x": x.grad}
    res.update({n: p.grad.clone() for n, p in m.named_parameters()})
    return res


_OFT_CASES = [  # (algo, bypass, x shape, out features, bias, multiplier, rescaled, constraint)
    ("diag-oft", False, (6, 32), 32, True, 1.0, False, 0.0),
    ("diag-oft", False, (2, 5, 64), 128, False, 0.7, True, 0.0),
    ("diag-oft", False, (1, 1, 2, 3, 64), 48, True, 1.0, False, 1e-3),  # Anima MLP inputs are 5-D
    ("diag-oft", True, (2, 300, 32), 64, True, 0.7, False, 0.0),
    ("boft", False, (6, 32), 32, True, 1.0, False, 0.0),
    ("boft", False, (2, 5, 64), 128, False, 0.7, True, 1e-3),
    ("boft", False, (1, 1, 4, 80, 64), 128, True, 1.0, True, 0.0),  # 320 rows: block path (not the few-rows path)
    ("boft", False, (300, 32), 192, False, 0.7, False, 0.0),  # b = 12
    ("boft", True, (2, 150, 32), 128, True, 0.7, True, 0.0),
]


@pytest.mark.parametrize("max_block", [8, 256])
@pytest.mark.parametrize("case", _OFT_CASES)
def test_oft_output_rotation_matches_rebuilt_weight(case, max_block, monkeypatch):
    """The flag's Linear path (base GEMM + blockwise rotation of the output; BOFT stages merged into dense blocks,
    max_block=8 forces one permuted chunk per stage) == the previous flag code (rebuilt W' / stock bypass): output, the
    input gradient and every parameter gradient, fp64 (both with an fp64 Cayley transform) and fp32, checkpointed or
    not; no dW'-sized tensor needed for the gradient."""
    algo, bypass, shape, o, bias, mult, rescaled, constraint = case
    monkeypatch.setattr(laf, "_OFT_MAX_BLOCK", max_block)
    for dtype, tol in ((torch.float64, 1e-12), (torch.float32, 1e-5)):
        x = torch.randn(shape, generator=torch.Generator().manual_seed(5), dtype=dtype)
        for ckpt in (False, True):
            cls = _oft_classes()[algo]
            if dtype == torch.float64:
                monkeypatch.setattr(cls, "get_r", _get_r64)
            got = _oft_run(_oft_module(algo, shape[-1], o, dtype, bias, bypass, mult, rescaled, constraint), x, ckpt)
            with _OftSpeedOff():
                if dtype == torch.float64:
                    cls.get_r = _get_r64
                ref = _oft_run(_oft_module(algo, shape[-1], o, dtype, bias, bypass, mult, rescaled, constraint), x, ckpt)
            monkeypatch.undo()
            monkeypatch.setattr(laf, "_OFT_MAX_BLOCK", max_block)
            assert got.keys() == ref.keys()
            for k in got:  # relative to the tensor's norm (fp32: summation order of a different association)
                assert rel(got[k], ref[k], ref[k]) < tol, (k, dtype, ckpt, rel(got[k], ref[k], ref[k]))


@pytest.mark.parametrize("algo", ["diag-oft", "boft"])
def test_oft_bf16_autocast_error_not_worse(algo):
    """bf16 autocast (fp32 network, bf16 base) vs the fp64 result: the output rotation keeps R - I in bf16 instead of
    rounding R^T W, so the mean relative error of the output and of every gradient is not larger than the old path's."""
    errs = {"new": [], "old": []}
    for k, (shape, o) in enumerate((((2, 5, 64), 128), ((1, 1, 4, 80, 64), 128), ((6, 32), 64))):
        x = torch.randn(shape, generator=torch.Generator().manual_seed(k)).bfloat16()
        cls = _oft_classes()[algo]
        saved = cls.__dict__["get_r"]
        cls.get_r = _get_r64
        try:
            truth = _oft_run(_oft_module(algo, shape[-1], o, torch.float64, seed=k), x.double())
        finally:
            cls.get_r = saved
        for name in ("new", "old"):
            if name == "old":
                with _OftSpeedOff():
                    got = _oft_run(_oft_module(algo, shape[-1], o, torch.bfloat16, seed=k), x, autocast=True)
            else:
                got = _oft_run(_oft_module(algo, shape[-1], o, torch.bfloat16, seed=k), x, autocast=True)
            assert got["out"].dtype == torch.bfloat16
            errs[name].append([rel(got[n].double(), truth[n], truth[n]) for n in truth])
    new = torch.tensor(errs["new"]).mean(0)
    old = torch.tensor(errs["old"]).mean(0)
    print(f"[{algo}] bf16 rel. error (out, x, params...): new {new.tolist()} old {old.tolist()}")
    assert bool((new <= old * 1.02).all()), (new, old)


@pytest.mark.parametrize("algo", ["diag-oft", "boft"])
def test_oft_get_r_bit_identical_and_no_host_sync(algo):
    """The patched Cayley transform gives stock's bits (constraint off, inactive and active); stock reads the GPU on the
    host (linalg_inv's error check, `if q_norm > constraint`), the flag's forward + backward does not."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Watch(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.hits = []

        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            if func.overloadpacket.__name__ in ("_local_scalar_dense", "lift_fresh", "nonzero", "is_nonzero", "item",
                                                "_linalg_check_errors"):
                self.hits.append(func.overloadpacket.__name__)
            return func(*args, **(kwargs or {}))

    cls = _oft_classes()[algo]
    for constraint in (0.0, 1e-4, 10.0):  # off, active (norm above), inactive (norm below)
        m = _oft_module(algo, 32, 64, torch.float32, constraint=constraint)
        stock = laf._original(cls, "get_r")(m)
        assert torch.equal(m.get_r(), stock), constraint
        x = torch.randn(2, 300, 32)
        for speed_on in (True, False):
            if speed_on:
                w = Watch()
                with w:
                    torch.utils.checkpoint.checkpoint(m, x.requires_grad_(True), use_reentrant=False).sum().backward()
                assert not w.hits, (constraint, w.hits)
            else:
                with _OftSpeedOff():
                    w = Watch()
                    with w:
                        m.forward(x).sum().backward()
                assert "_linalg_check_errors" in w.hits, w.hits  # stock: torch.inverse reads its info on the host
                assert ("_local_scalar_dense" in w.hits) == (constraint > 0), w.hits  # `if q_norm > constraint`


@pytest.mark.parametrize("algo", ["diag-oft", "boft"])
def test_oft_training_and_saved_file_unchanged(tmp_path, algo):
    """Real network path (create_network + after_network_created), fp32, AdamW: the same trajectory with the output
    rotation as with the previous flag code; same state_dict names/shapes/order (--resume), and the saved file is
    identical for the same weights (the save does not go through forward or get_r)."""
    from safetensors.torch import load_file

    def run(speed_on):
        model = make_model(dtype=torch.float32)
        ctx = _OftSpeedOff() if not speed_on else None
        if ctx:
            ctx.__enter__()
        try:
            net = make_net(model, algo, 8, 8)
            net.train()
            opt = build_optimizer(net, 5e-3)
            for step in range(4):
                b = make_batch(100 + step)
                h, _ = model(b["x"].float(), b["c"].float(), checkpoint=step % 2 == 1)
                F.mse_loss(h, b["y"]).backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            sd = net.state_dict()
            path = str(tmp_path / f"{algo}_{speed_on}.safetensors")
            net.save_weights(path, torch.float32, {})
            return sd, path, net
        finally:
            if ctx:
                ctx.__exit__(None, None, None)

    sd_new, p_new, net_new = run(True)
    sd_old, p_old, _ = run(False)
    assert list(sd_new) == list(sd_old) and all(sd_new[k].shape == sd_old[k].shape for k in sd_new)
    for k in sd_new:
        assert torch.allclose(sd_new[k], sd_old[k], rtol=1e-4, atol=1e-6), k
    with _OftSpeedOff():
        p_same = str(tmp_path / f"{algo}_same.safetensors")
        net_new.save_weights(p_same, torch.float32, {})
    a, b = load_file(p_new), load_file(p_same)  # same tensors bit for bit, same metadata (its key order varies)
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in a)
    assert metadata(p_new) == metadata(p_same)
    assert list(load_file(p_old)) == list(a)


# -------------------------------------------------------------------------------------------------------------- T-LoRA


def test_tlora_mask_survives_checkpointing():
    model = make_model()
    net = make_net(model, "tlora", 8, 8)
    ts = torch.tensor([100.0, 900.0], dtype=torch.bfloat16)  # sigma * 1000, per sample -> ranks 7 and 4 of 8
    laf.set_timestep(net, ts)
    loss_fn(model, make_batch(3), checkpoint=True).backward()
    laf.clear_timestep(net)
    for m in loras(net):
        g = m.lambda_layer.grad.reshape(-1)
        assert float(g[7].abs()) == 0.0 and float(g[:4].abs().sum()) > 0  # rank 8 is masked for both samples


def test_tlora(tmp_path):
    model = make_model(conv=True)
    net = make_net(model, "tlora", 8, 8, conv_dim=4, conv_alpha=4)
    no_grad, unchanged = train(model, net, conv=True, timesteps=torch.tensor([200.0, 700.0]), checkpoint=True)
    assert not no_grad and not unchanged, (no_grad, unchanged)
    to_fp32_eval(model, net)
    p32, p16 = save_both(net, tmp_path, "tlora")
    assert metadata(p32)[laf.METADATA_FORMAT_KEY] == "lora"
    check_forge(net, p32)  # full mask: all ranks
    worst16, _ = check_forge(net, p16, tol=5e-3)  # balanced factors: the fp16 file keeps the small trained change
    model3 = make_model(conv=True)
    net3 = make_net(model3, "tlora", 8, 8, conv_dim=4, conv_alpha=4)
    with pytest.raises(ValueError, match="T-LoRA"):
        net3.load_weights(p32)


# ---------------------------------------------------------------------------------------------------------------- TSM


def test_tsm_stage1_routes_per_sample_and_stage2_trains_router(tmp_path):
    model = make_model()
    net = make_net(model, "tsm", 4, 4, tsm_n_scales="4,1")
    ts = torch.tensor([100.0, 350.0, 600.0, 900.0])  # one per expert interval of the 4-expert scale
    no_grad, unchanged = train(model, net, steps=3, timesteps=ts, B=4, checkpoint=True)
    assert not no_grad and not unchanged, (no_grad, unchanged)
    for m in loras(net):
        assert not m.router_fc.weight.requires_grad  # stage 1 keeps the router frozen
    p32 = str(tmp_path / "tsm_stage1.safetensors")
    net.save_weights(p32, torch.float32, {})
    assert metadata(p32)[laf.METADATA_FORMAT_KEY] == "tsm-native"

    # stage 2 from the stage-1 file (--network_weights)
    model2 = make_model()
    net2 = make_net(model2, "tsm", 4, 4, tsm_n_scales="4,1", tsm_stage=2)
    assert not net2.load_weights(p32)
    experts_before = {n: p.detach().clone() for n, p in net2.named_parameters() if ".experts." in n}
    opt = build_optimizer(net2, 5e-3)  # prepare_optimizer_params -> requires_grad_(True): stage re-applied
    assert all(not p.requires_grad for n, p in net2.named_parameters() if ".experts." in n)
    assert all(p.requires_grad for g in opt.param_groups for p in g["params"])  # frozen experts left out
    net2.train()
    got = set()
    for step in range(2):
        laf.set_timestep(net2, ts)
        loss_fn(model2, make_batch(200 + step, B=4), checkpoint=True).backward()
        laf.clear_timestep(net2)
        got |= {n for n, p in net2.named_parameters() if p.grad is not None and bool(torch.any(p.grad != 0))}
        opt.step()
        opt.zero_grad(set_to_none=True)
    routers = [n for n, _ in net2.named_parameters() if n.endswith("router_fc.weight")]
    assert routers and all(n in got for n in routers), sorted(set(routers) - got)[:3]
    assert all(torch.equal(experts_before[n], p) for n, p in net2.named_parameters() if n in experts_before)


def _tsm_module(stage, bypass=False, scales=(8, 1), router="input", dtype=torch.float64, seed=0, rank=4,
                use_scalar=False):
    """One TSMModule on a Linear(24 -> 20) with random experts/router/embedding (up is zero-initialised otherwise)."""
    from lycoris.modules.tsm import TSMModule

    torch.manual_seed(seed)
    org = nn.Linear(24, 20).to(dtype)
    org.requires_grad_(False)
    m = TSMModule("t", org, multiplier=0.7, lora_dim=rank, alpha=rank / 2, bypass_mode=bypass,
                  tsm_n_scales=list(scales), tsm_stage=stage, tsm_router_input=router, use_scalar=use_scalar)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    m.to(dtype)
    m.set_stage(stage)
    return m


def _tsm_run(m, x, ts, checkpoint=False):
    """Output and every gradient (input + parameters; None where none) of sum(out * w) with the given timesteps."""
    net = argparse.Namespace(loras=[m])
    x = x.detach().clone().requires_grad_(True)
    for p in m.parameters():
        p.grad = None
    laf.set_timestep(net, torch.tensor(ts))
    out = torch.utils.checkpoint.checkpoint(m, x, use_reentrant=False) if checkpoint else m(x)
    w = torch.randn(out.shape, generator=torch.Generator().manual_seed(3), dtype=out.dtype)
    (out * w).sum().backward()
    laf.clear_timestep(net)
    res = {"out": out.detach(), "x": x.grad}
    res.update({n: (None if p.grad is None else p.grad.clone()) for n, p in m.named_parameters()})
    return res


def _assert_same(a, b, rtol=1e-9, atol=1e-12):
    assert a.keys() == b.keys()
    for k in a:
        assert (a[k] is None) == (b[k] is None), k
        if a[k] is not None:
            assert torch.allclose(a[k], b[k], rtol=rtol, atol=atol), (k, (a[k] - b[k]).abs().max())


def _tsm_mixed(stage, router="input"):
    """fp64 experts / router / scalar Parameter over an fp32 base Linear, for an fp32 input without autocast: the fused
    plan's cdt is the input dtype (fp32) and ld = promote(fp64, fp32) = fp64 != cdt, so tsm_lowrank_fp32=True really
    changes the path (_TSMLinear, or the fp32-mode branch of _TSMExpertLinear) - in an all-fp64 module ld == cdt and
    both modes take _LowRankLinear. The base GEMM and the input gradient are fp32 on every path, so comparisons are at
    fp32 level."""
    m = _tsm_module(stage, router=router, use_scalar=True)
    m.org_module[0].to(torch.float32)
    return m


@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("scales", [(8, 1), (4, 2, 1)])
@pytest.mark.parametrize("t", [1.0, 125.0, 126.0, 999.0, 1000.0])
def test_tsm_fused_stage1_matches_stock(bypass, scales, t):
    """The fused path (one down + one up matmul over the concatenated active experts) == stock LyCORIS rebuild
    (dense delta weight) / bypass (one matmul pair per expert): output, input gradient, each expert's gradient, and no
    gradient for the inactive experts."""
    x = torch.randn(1, 1, 3, 4, 24, generator=torch.Generator().manual_seed(0), dtype=torch.float64)

    def stock():
        from lycoris.modules.tsm import clear_tsm_timestep, set_tsm_timestep

        m = _tsm_module(1, bypass, scales)
        set_tsm_timestep(int(t))
        try:
            xs = x.clone().requires_grad_(True)
            out = m(xs)
            (out * torch.randn(out.shape, generator=torch.Generator().manual_seed(3), dtype=out.dtype)).sum().backward()
        finally:
            clear_tsm_timestep()
        res = {"out": out.detach(), "x": xs.grad}
        res.update({n: (None if p.grad is None else p.grad.clone()) for n, p in m.named_parameters()})
        return res

    ref = _with_stock(stock)
    for ckpt in (False, True):
        _assert_same(_tsm_run(_tsm_module(1, bypass, scales), x, [t], checkpoint=ckpt), ref)
    active = [n for n, g in ref.items() if ".experts." in n or n.startswith("experts.")]
    assert sum(ref[n] is not None for n in active) == 2 * len(scales)  # down + up of one expert per scale


@pytest.mark.parametrize("router", ["input", "bottleneck"])
@pytest.mark.parametrize("scales", [(8, 1), (4, 2, 1)])
@pytest.mark.parametrize("shape", [(6, 24), (2, 5, 24), (1, 1, 3, 4, 24)])
def test_tsm_fused_stage2_matches_explicit_math(router, scales, shape):
    """Stage 2 == B_core A_core x + sum_j G_j B_j A_j x with G = router_fc(pool(z)) + embed[t-1], pool over every
    leading axis, written out per expert here (fp64; the module pools in promote(activation dtype, fp32) = fp64 here,
    fp32 on the GPU's bf16 / fp32 activations). Experts frozen: no expert gradient; router and embedding get theirs;
    the scalar folding keeps scalar * scale * multiplier."""
    x = torch.randn(shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    t = 700
    m = _tsm_module(2, False, scales, router, use_scalar=True)
    got = _tsm_run(m, x, [float(t)], checkpoint=True)

    ref_m = _tsm_module(2, False, scales, router, use_scalar=True)
    xr = x.clone().requires_grad_(True)
    idx = [ref_m._get_expert_index(t, s) for s in range(len(scales))]
    ex = [ref_m.experts[s][e] for s, e in enumerate(idx)]
    core_mid = xr @ ex[0]["down"].weight.T
    src = core_mid if router == "bottleneck" else xr
    z = src.detach().reshape(-1, src.shape[-1]).mean(0)  # the router pools over every leading axis (tsm.py:362)
    gates = ref_m.router_fc(z) + ref_m.timestep_embed.weight[t - 1]
    delta = core_mid @ ex[0]["up"].weight.T
    for j in range(1, len(scales)):
        delta = delta + gates[j - 1] * ((xr @ ex[j]["down"].weight.T) @ ex[j]["up"].weight.T)
    # ref_m.multiplier, not the literal 0.7: LyCORIS keeps the multiplier as an fp32-rounded float (0.699999988...)
    out = ref_m.org_module[0](xr) + delta * ref_m.scalar * ref_m.scale * ref_m.multiplier
    (out * torch.randn(out.shape, generator=torch.Generator().manual_seed(3), dtype=out.dtype)).sum().backward()
    ref = {"out": out.detach(), "x": xr.grad}
    ref.update({n: (None if p.grad is None else p.grad.clone()) for n, p in ref_m.named_parameters()})
    _assert_same(got, ref)
    assert all(got[n] is None for n in got if n.startswith("experts."))
    assert all(got[n] is not None for n in ("router_fc.weight", "router_fc.bias", "timestep_embed.weight", "scalar"))


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("ts", [[100.0, 900.0], [900.0, 100.0, 900.0], [5.0, 5.0, 990.0, 5.0], [125.0, 126.0],
                                [120.0, 124.0]])  # the last: different timesteps, the same experts (stage 2: gates differ)
def test_tsm_grouped_batch_equals_per_sample_calls(stage, ts):
    """A batch with different timesteps == each sample run alone at its own timestep (stage 2: the gate pools over the
    samples that share a timestep, so those run together), under gradient checkpointing; same gradients."""
    B = len(ts)
    x = torch.randn(B, 1, 3, 4, 24, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    got = _tsm_run(_tsm_module(stage), x, ts, checkpoint=True)

    m = _tsm_module(stage)
    groups = {}
    for i, t in enumerate(ts):
        key = tuple(m._get_expert_index(int(t), s) for s in range(2)) if stage == 1 else t
        groups.setdefault(key, (t, []))[1].append(i)
    xr = x.clone().requires_grad_(True)
    net = argparse.Namespace(loras=[m])
    outs = [None] * B
    for t, idx in groups.values():
        laf.set_timestep(net, torch.tensor([t]))
        o = m(xr[idx])
        laf.clear_timestep(net)
        for k, i in enumerate(idx):
            outs[i] = o[k:k + 1]
    out = torch.cat(outs, 0)
    (out * torch.randn(out.shape, generator=torch.Generator().manual_seed(3), dtype=out.dtype)).sum().backward()
    ref = {"out": out.detach(), "x": xr.grad}
    ref.update({n: (None if p.grad is None else p.grad.clone()) for n, p in m.named_parameters()})
    _assert_same(got, ref)


def test_tsm_expert_cache_and_no_host_sync():
    """The cached expert indices equal tsm.py's _get_expert_index for every t; a TSM forward + backward makes no tensor
    from Python data (a blocking host-to-device copy on CUDA) and no .item() (old stage 2: two of each per call), and a
    mixed-timestep batch builds its index tensors once per step, not per module."""
    from torch.utils._python_dispatch import TorchDispatchMode

    for scales in ((8, 1), (4, 2, 1), (3, 1)):
        m = _tsm_module(1, scales=scales, dtype=torch.float32)
        for t in range(0, 1001):
            assert laf._tsm_experts(m, t) == tuple(m._get_expert_index(t, s) for s in range(len(scales))), (scales, t)

    class Watch(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.hits = []

        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            if func.overloadpacket.__name__ in ("lift_fresh", "_local_scalar_dense", "nonzero"):
                self.hits.append(func.overloadpacket.__name__)
            return func(*args, **(kwargs or {}))

    for stage in (1, 2):
        # first_ok: the step's plan tensors (group plan; mixed plan: block mask, core positions, arange, stage 2 four
        # more), built by the first module and cached for the others
        for ts, shape, first_ok in (([500.0], (1, 6, 24), 0), ([900.0, 100.0, 900.0], (3, 6, 24), 8)):
            mods = [_tsm_module(stage, dtype=torch.float32, seed=s) for s in range(2)]
            net = argparse.Namespace(loras=mods)
            laf.set_timestep(net, torch.tensor(ts))
            for k, m in enumerate(mods):
                x = torch.randn(shape).requires_grad_(True)
                w = Watch()
                with w:
                    torch.utils.checkpoint.checkpoint(m, x, use_reentrant=False).sum().backward()
                assert len(w.hits) <= (first_ok if k == 0 else 0), (stage, ts, k, w.hits)
            laf.clear_timestep(net)


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("lowrank_fp32", [False, True])
def test_tsm_factor_path_is_the_default_and_matches_the_expert_function(monkeypatch, stage, lowrank_fp32):
    """The default (tsm_expert_fn not given) is the factor path: A/B as autograd ops, then _LowRankLinear (low-rank part
    in the compute dtype) or _TSMLinear (fp32 low-rank); network_args tsm_expert_fn=True switches to _TSMExpertLinear.
    Same values and gradients (mixed-timestep batch, use_scalar Parameter, both stages). tsm_lowrank_fp32 goes through
    network_args (prepare_network_kwargs resets the global): False = all fp64, exact; True = _tsm_mixed (fp64 experts
    over an fp32 base and input: ld = fp64 != cdt = fp32, so the call lands in _TSMLinear, counted), fp32-level
    agreement."""
    monkeypatch.setattr(laf, "_TSM_FUSED", True)
    monkeypatch.setattr(laf, "_TSM_EXPERT_FN", False)
    monkeypatch.setattr(laf, "_TSM_MIXED", False)  # the per-group factor path is what this test dispatches to
    monkeypatch.setattr(laf, "_TSM_LOWRANK_FP32", None)  # set through network_args below; restored after the test
    xdt = torch.float32 if lowrank_fp32 else torch.float64
    x = torch.randn(2, 1, 3, 4, 24, generator=torch.Generator().manual_seed(6), dtype=xdt)
    make = (lambda: _tsm_mixed(stage)) if lowrank_fp32 else (lambda: _tsm_module(stage, use_scalar=True))
    tol = dict(rtol=1e-5, atol=1e-6) if lowrank_fp32 else {}
    calls = {"lr": [], "tl": [], "efn": []}
    for name, cls in (("lr", laf._LowRankLinear), ("tl", laf._TSMLinear), ("efn", laf._TSMExpertLinear)):
        orig = cls.apply
        monkeypatch.setattr(cls, "apply", (lambda n, o: lambda *a: calls[n].append(1) or o(*a))(name, orig))
    kw = laf.prepare_network_kwargs("tsm", {"algo": "tsm", "tsm_lowrank_fp32": str(lowrank_fp32), "tsm_mixed": "False"}, LOG)
    assert "tsm_mixed" not in kw and laf._TSM_EXPERT_FN is False and laf._TSM_LOWRANK_FP32 is lowrank_fp32
    assert laf._TSM_MIXED is False  # the per-group loop, so the factor path's Functions are what gets counted
    factor = _tsm_run(make(), x, [100.0, 900.0], checkpoint=True)
    assert not calls["efn"]
    assert len(calls["tl" if lowrank_fp32 else "lr"]) == 4 and not calls["lr" if lowrank_fp32 else "tl"]  # 2 groups x 2
    kw = laf.prepare_network_kwargs("tsm", {"algo": "tsm", "tsm_expert_fn": "True", "tsm_mixed": "False",
                                            "tsm_lowrank_fp32": str(lowrank_fp32)}, LOG)
    assert "tsm_expert_fn" not in kw and laf._TSM_EXPERT_FN is True and laf._TSM_LOWRANK_FP32 is lowrank_fp32
    for v in calls.values():
        v.clear()
    fn = _tsm_run(make(), x, [100.0, 900.0], checkpoint=True)
    assert len(calls["efn"]) == 4 and not calls["lr"] and not calls["tl"]
    _assert_same(factor, fn, **tol)


def test_tsm_scalar_buffer_written_in_place_then_copied_is_read(monkeypatch):
    """_tsm_scale: the non-trainable scalar buffer is assumed 1.0 only while untouched; after an in-place write
    (apply_max_norm) and after a module.to() copy that follows it, the fused path reads the new value (== split path),
    with one float() (aten _local_scalar_dense, counted over the fused forward + recompute + backward) per change and
    none per call. A scalar Parameter always goes in as a tensor, frozen or not, with no float() at all."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Syncs(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.n = 0

        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            if func.overloadpacket.__name__ == "_local_scalar_dense":
                self.n += 1
            return func(*args, **(kwargs or {}))

    x = torch.randn(1, 1, 3, 4, 24, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    monkeypatch.setattr(laf, "_TSM_FUSED", True)

    def both(m, syncs):
        """fused == split; the fused run reads the scalar with float() exactly `syncs` times."""
        s = Syncs()
        with s:
            fused = _tsm_run(m, x, [250.0], checkpoint=True)
        assert s.n == syncs, (s.n, syncs)
        monkeypatch.setattr(laf, "_TSM_FUSED", False)
        split = _tsm_run(m, x, [250.0], checkpoint=True)
        monkeypatch.setattr(laf, "_TSM_FUSED", True)
        _assert_same(fused, split)
        return fused

    m = _tsm_module(1)
    out1 = both(m, 0)["out"]  # untouched: 1.0 without a read
    both(m, 0)
    with torch.no_grad():
        m.scalar.fill_(0.5)  # apply_max_norm style in-place write
    out2 = both(m, 1)["out"]  # one float() for the new value, shared by forward and checkpoint recompute
    both(m, 0)
    assert not torch.allclose(out1, out2)
    m.to(torch.float64)  # a copy of the written buffer: storage changes, version back to 0
    m.scalar.data = m.scalar.data.clone()  # force a new storage even where .to() is a no-op
    out3 = both(m, 1)["out"]
    both(m, 0)
    assert torch.allclose(out2, out3, rtol=1e-12, atol=1e-12)
    p = _tsm_module(1, use_scalar=True)
    p.scalar.requires_grad_(False)  # a frozen scalar Parameter keeps its value
    assert isinstance(laf._tsm_scale(p)[0], torch.Tensor)
    both(p, 0)


def _count_ops(fn) -> int:
    """aten ops dispatched while fn() runs (a proxy for kernel launches)."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Count(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.n = 0

        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            self.n += 1
            return func(*args, **(kwargs or {}))

    c = Count()
    with c:
        fn()
    return c.n


# For the record (CPU, torch 2.7.1, 2026-10-25): aten ops over forward + checkpoint recompute + backward of _tsm_run for
# the fp64 cases of the test below - the factor path (the default) 239 / 302 / 310 (stage 1) and 273 / 368 / 376
# (stage 2), _TSMExpertLinear 233 / 292 / 300 and 258 / 328 / 336. The test compares counts measured in the same process
# (torch versions split ops differently), not against these numbers.


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("ts", [[300.0], [100.0, 900.0], [900.0, 100.0, 900.0]])
def test_tsm_fused_forward_is_taken_and_dispatches_less(monkeypatch, stage, ts):
    """The fused forward (not org_forward + delta): by default a mixed-timestep batch is ONE _TSMMaskedLinear call per
    module call (forward + checkpoint recompute), a single-timestep batch one factor call; tsm_mixed=False restores the
    per-expert-group factor loop (one _tsm_factor_linear call per group of _tsm_group_plan), tsm_expert_fn=True the
    _TSMExpertLinear Function. Each dispatches fewer aten ops over forward + checkpoint recompute + backward than the
    next: mixed < factor loop < split, the Function fewer still on the CPU count (the GPU step did not follow, see
    _TSM_EXPERT_FN); all give the same result (fp64). _TSM_FUSED=False takes the split forward for every call."""
    x = torch.randn(len(ts), 1, 3, 4, 24, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    fused_calls, split_calls, factor_calls, fn_calls = [], [], [], []
    orig_fused, orig_split = laf._tsm_fused_linear, laf._tsm_split_forward
    orig_factor, orig_fn = laf._tsm_factor_linear, laf._TSMExpertLinear.apply
    monkeypatch.setattr(laf, "_tsm_fused_linear", lambda *a: fused_calls.append(1) or orig_fused(*a))
    monkeypatch.setattr(laf, "_tsm_split_forward", lambda *a, **k: split_calls.append(1) or orig_split(*a, **k))
    monkeypatch.setattr(laf, "_tsm_factor_linear", lambda *a: factor_calls.append(1) or orig_factor(*a))
    monkeypatch.setattr(laf._TSMExpertLinear, "apply", lambda *a: fn_calls.append(1) or orig_fn(*a))
    mixed_calls = []
    orig_mixed = laf._TSMMaskedLinear.apply
    monkeypatch.setattr(laf._TSMMaskedLinear, "apply", lambda *a: mixed_calls.append(1) or orig_mixed(*a))
    monkeypatch.setattr(laf, "_TSM_FUSED", True)
    monkeypatch.setattr(laf, "_TSM_EXPERT_FN", False)
    groups = len({tuple(laf._tsm_experts(_tsm_module(stage), int(t))) if stage == 1 else t for t in ts})
    # the default: a mixed-timestep batch as one masked call per module call (forward + checkpoint recompute)
    monkeypatch.setattr(laf, "_TSM_MIXED", True)
    got_mixed = _tsm_run(_tsm_module(stage), x, ts, checkpoint=True)
    if groups > 1:
        assert len(mixed_calls) == 2 and not fused_calls and not factor_calls and not split_calls, (len(mixed_calls), len(fused_calls))
    else:
        assert not mixed_calls and len(fused_calls) == len(factor_calls) == 2
    n_mixed = _count_ops(lambda: _tsm_run(_tsm_module(stage), x, ts, checkpoint=True))
    # the group loop (tsm_mixed=False): one factor call per expert group
    monkeypatch.setattr(laf, "_TSM_MIXED", False)
    fused_calls.clear()
    factor_calls.clear()
    mixed_calls.clear()
    got = _tsm_run(_tsm_module(stage), x, ts, checkpoint=True)
    assert len(fused_calls) == len(factor_calls) == 2 * groups  # forward + checkpoint recompute
    assert not split_calls and not fn_calls and not mixed_calls
    n_factor = _count_ops(lambda: _tsm_run(_tsm_module(stage), x, ts, checkpoint=True))
    _assert_same(got_mixed, got)
    if groups > 1:
        assert n_mixed < n_factor, (n_mixed, n_factor)
    else:
        assert n_mixed == n_factor

    monkeypatch.setattr(laf, "_TSM_EXPERT_FN", True)
    fused_calls.clear()
    factor_calls.clear()
    got_fn = _tsm_run(_tsm_module(stage), x, ts, checkpoint=True)
    assert len(fused_calls) == len(fn_calls) == 2 * groups and not split_calls and not factor_calls
    n_fn = _count_ops(lambda: _tsm_run(_tsm_module(stage), x, ts, checkpoint=True))

    monkeypatch.setattr(laf, "_TSM_FUSED", False)
    fused_calls.clear()
    fn_calls.clear()
    ref = _tsm_run(_tsm_module(stage), x, ts, checkpoint=True)
    assert split_calls and not fused_calls and not fn_calls and not factor_calls
    n_split = _count_ops(lambda: _tsm_run(_tsm_module(stage), x, ts, checkpoint=True))
    _assert_same(got, ref)
    _assert_same(got_fn, ref)
    assert n_factor < n_split, (n_factor, n_split)
    assert n_fn < n_factor, (n_fn, n_factor)


def test_tsm_fused_network_arg_and_fallbacks(monkeypatch):
    """network_args tsm_fused=False is popped before TSMModule sees it and switches the flag; the fused path steps aside
    for conv modules, delta dropout while training, RamTorch base weights and a dtype mix outside autocast (all keep
    the split forward == stock)."""
    from lycoris.modules.tsm import TSMModule

    monkeypatch.setattr(laf, "_TSM_FUSED", True)
    kw = laf.prepare_network_kwargs("tsm", {"algo": "tsm", "tsm_fused": "False", "tsm_n_scales": "4,1"}, LOG)
    assert "tsm_fused" not in kw and laf._TSM_FUSED is False
    kw = laf.prepare_network_kwargs("tsm", {"algo": "tsm", "tsm_n_scales": "4,1"}, LOG)
    assert laf._TSM_FUSED is True
    kw = laf.prepare_network_kwargs("tsm", {"algo": "tsm", "tsm_fused": "1"}, LOG)
    assert laf._TSM_FUSED is True
    kw = laf.prepare_network_kwargs("tsm", {"algo": "tsm", "tsm_fused": ""}, LOG)  # an empty value = not given
    assert "tsm_fused" not in kw and laf._TSM_FUSED is True
    monkeypatch.setattr(laf, "_TSM_LOWRANK_FP32", True)
    kw = laf.prepare_network_kwargs("tsm", {"algo": "tsm", "tsm_lowrank_fp32": "False"}, LOG)
    assert "tsm_lowrank_fp32" not in kw and laf._TSM_LOWRANK_FP32 is False
    assert not laf._tsm_lowrank_fp32(_tsm_module(1)) and not laf._tsm_lowrank_fp32(_tsm_module(2))
    kw = laf.prepare_network_kwargs("tsm", {"algo": "tsm"}, LOG)  # not given: fp32 low-rank part in stage 2 only
    assert laf._TSM_LOWRANK_FP32 is None
    assert not laf._tsm_lowrank_fp32(_tsm_module(1)) and laf._tsm_lowrank_fp32(_tsm_module(2))
    kw = laf.prepare_network_kwargs("tsm", {"algo": "tsm", "tsm_lowrank_fp32": "True"}, LOG)
    assert laf._TSM_LOWRANK_FP32 is True and laf._tsm_lowrank_fp32(_tsm_module(1))

    m = _tsm_module(1)
    x = torch.randn(2, 6, 24, dtype=torch.float64)
    assert laf._tsm_fused_ok(m, x)
    assert not laf._tsm_fused_ok(m, x.float())  # module fp64, input fp32, no autocast: the split path's dtype mix
    m.dropout = 0.1
    assert laf._tsm_fused_ok(m, x)  # stage-1 rebuild: no delta dropout anywhere, the fused path stays
    for md in (_tsm_module(2), _tsm_module(1, bypass=True)):
        md.dropout = 0.1
        assert not laf._tsm_fused_ok(md, x)  # self.drop(delta) in training
        md.eval()
        assert laf._tsm_fused_ok(md, x)
    m.dropout = 0.0
    m.is_ramtorch_org = True
    assert not laf._tsm_fused_ok(m, x)
    m.is_ramtorch_org = False
    m.org_module[0].weight.requires_grad_(True)
    assert not laf._tsm_fused_ok(m, x)
    m.org_module[0].weight.requires_grad_(False)
    assert laf._tsm_fused_ok(m, x)

    torch.manual_seed(0)
    conv = nn.Conv2d(6, 8, 3, padding=1).to(torch.float64)
    conv.requires_grad_(False)
    mc = TSMModule("c", conv, multiplier=0.7, lora_dim=4, alpha=2, tsm_n_scales=[4, 1], tsm_stage=1)
    with torch.no_grad():
        for p in mc.parameters():
            p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(5)) * 0.2)
    mc.to(torch.float64)
    xi = torch.randn(2, 6, 5, 5, dtype=torch.float64)
    assert not laf._tsm_fused_ok(mc, xi)
    got = _tsm_run(mc, xi, [400.0], checkpoint=True)

    def stock():
        from lycoris.modules.tsm import clear_tsm_timestep, set_tsm_timestep

        torch.manual_seed(0)
        conv2 = nn.Conv2d(6, 8, 3, padding=1).to(torch.float64)
        conv2.requires_grad_(False)
        ms = TSMModule("c", conv2, multiplier=0.7, lora_dim=4, alpha=2, tsm_n_scales=[4, 1], tsm_stage=1)
        with torch.no_grad():
            for p in ms.parameters():
                p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(5)) * 0.2)
        ms.to(torch.float64)
        set_tsm_timestep(400)
        try:
            xs = xi.clone().requires_grad_(True)
            out = ms(xs)
            (out * torch.randn(out.shape, generator=torch.Generator().manual_seed(3), dtype=out.dtype)).sum().backward()
        finally:
            clear_tsm_timestep()
        res = {"out": out.detach(), "x": xs.grad}
        res.update({n: (None if p.grad is None else p.grad.clone()) for n, p in ms.named_parameters()})
        return res

    _assert_same(got, _with_stock(stock))


@pytest.mark.parametrize("mixed", [True, False])
@pytest.mark.parametrize("expert_fn", [False, True])
@pytest.mark.parametrize("stage", [1, 2])
def test_tsm_fused_bf16_autocast_no_worse_than_split(monkeypatch, stage, expert_fn, mixed):
    """Under bf16 autocast (fp32 experts, bf16 base, Anima-like >3-D input) the fused forward is as close to the fp64
    result as the split forward - loss error within 1e-4 absolute, gradient error within 1.1x - with absolute ceilings
    on the gradient error (the output is rounded once instead of base and delta each; the two paths round at different
    places, so they are not compared with each other). The exact checks are the fp64 tests above; this one also makes
    sure the bf16 paths really run (tsm_fused / tsm_expert_fn / tsm_lowrank_fp32 / tsm_mixed through network_args; the
    low-rank Function each call lands in, with its dtypes): _TSMMaskedLinear - the mixed default for this two-timestep
    batch, one masked call per module, carrying both low-rank dtype modes (ld fp32 / cdt bf16 with tsm_lowrank_fp32,
    ld == cdt == bf16 without) - and with mixed=False the per-group factor path (_LowRankLinear for the bf16 low-rank
    part, _TSMLinear for fp32) and _TSMExpertLinear."""
    monkeypatch.setattr(laf, "_TSM_FUSED", True)  # prepare_network_kwargs sets all four below; restored after the test
    monkeypatch.setattr(laf, "_TSM_LOWRANK_FP32", True)
    monkeypatch.setattr(laf, "_TSM_EXPERT_FN", expert_fn)
    monkeypatch.setattr(laf, "_TSM_MIXED", mixed)
    calls = []
    orig = laf._tsm_fused_linear
    monkeypatch.setattr(laf, "_tsm_fused_linear", lambda *a: calls.append(1) or orig(*a))
    mixed_calls = []
    orig_mixed = laf._tsm_mixed_linear
    monkeypatch.setattr(laf, "_tsm_mixed_linear", lambda *a: mixed_calls.append(1) or orig_mixed(*a))
    fn_calls = []  # (Function, ld, cdt) of every low-rank Function call
    for name, cls, pick in (("lr", laf._LowRankLinear, lambda a: (a[6], a[6])),  # x, W0, bias, A, B, None, cdt, ..
                            ("tl", laf._TSMLinear, lambda a: (a[6], a[5])),  # x, W0, bias, A, B, cdt, ld
                            ("efn", laf._TSMExpertLinear, lambda a: (a[11], a[10])),  # .., cdt, ld, *weights
                            ("mx", laf._TSMMaskedLinear, lambda a: (a[8], a[7]))):  # .., mask, tok, cdt, ld
        monkeypatch.setattr(cls, "apply",
                            (lambda n, o, p: lambda *a: fn_calls.append((n, *p(a))) or o(*a))(name, cls.apply, pick))

    def run(fused, dtype, lowrank_fp32=True):
        calls.clear()
        mixed_calls.clear()
        fn_calls.clear()
        model = make_model(dtype=dtype)
        net = make_net(model, "tsm", 4, 4, tsm_n_scales="4,1", tsm_stage=stage, tsm_fused=fused,
                       tsm_lowrank_fp32=lowrank_fp32, tsm_expert_fn=expert_fn, tsm_mixed=mixed)
        assert laf._TSM_FUSED is fused and laf._TSM_LOWRANK_FP32 is lowrank_fp32 and laf._TSM_EXPERT_FN is expert_fn
        assert laf._TSM_MIXED is mixed
        with torch.no_grad():
            for m in net.unet_loras:
                for n, p in m.named_parameters():
                    if stage == 2 and ("router" in n or "timestep_embed" in n or "experts" in n):
                        p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(9)) * 0.2)
        if dtype == torch.float64:
            net.to(torch.float64)
        b = make_batch(1)
        laf.set_timestep(net, torch.tensor([300.0, 800.0]))
        if dtype == torch.float64:  # the reference: no autocast, everything fp64
            with torch.autocast("cpu", enabled=False):
                h, _ = model(b["x"].double(), b["c"].double(), None, checkpoint=True)
            loss = F.mse_loss(h, b["y"].double())
        else:
            loss = loss_fn(model, b, checkpoint=True)
        loss.backward()
        laf.clear_timestep(net)
        assert bool(fn_calls) is fused
        # the two timesteps of the batch differ: the mixed path (one masked call per module) unless tsm_mixed=False or
        # the expert Function is on; the group loop calls _tsm_fused_linear per group
        use_mixed = fused and mixed and not expert_fn
        assert bool(mixed_calls) is use_mixed and bool(calls) is (fused and not use_mixed)
        if fused:  # fp64 without autocast: ld == cdt, so the factor path takes _LowRankLinear in both low-rank modes
            want = "mx" if use_mixed else "efn" if expert_fn else ("tl" if lowrank_fp32 and dtype != torch.float64 else "lr")
            assert {n for n, _, _ in fn_calls} == {want}, fn_calls[:3]
        if fused and lowrank_fp32:  # the fp32 low-rank variant: ld = promote(expert dtype, fp32), not the bf16 cdt
            assert all(ld in (torch.float32, torch.float64) for _, ld, _ in fn_calls), fn_calls[:3]
            assert dtype == torch.float64 or any(cdt == torch.bfloat16 and ld == torch.float32 for _, ld, cdt in fn_calls)
        elif fused:  # the bf16 variant: every low-rank GEMM in the compute dtype itself
            assert all(ld == cdt for _, ld, cdt in fn_calls) and any(cdt == torch.bfloat16 for _, _, cdt in fn_calls)
        g = torch.cat([p.grad.reshape(-1).double() for m in net.unet_loras for _, p in sorted(m.named_parameters())
                       if p.grad is not None])
        return float(loss), g

    lr, gr = run(True, torch.float64)
    lf, gf = run(True, torch.bfloat16)  # fused, low-rank part in fp32 (tsm_lowrank_fp32=True forced; default: stage 2 only)
    lb, gb = run(True, torch.bfloat16, lowrank_fp32=False)  # fused, low-rank part in bf16 (_LowRankLinear)
    ls, gs = run(False, torch.bfloat16)
    assert gf.shape == gb.shape == gs.shape == gr.shape and gf.numel()
    err = {k: (abs(l - lr) / abs(lr), float((g - gr).norm() / gr.norm())) for k, (l, g) in
           {"fused": (lf, gf), "fused_bf16": (lb, gb), "split": (ls, gs)}.items()}
    for k in ("fused", "fused_bf16"):  # no worse than the split forward: loss within 1e-4 absolute, gradients 1.1x
        assert err[k][0] <= err["split"][0] + 1e-4, (k, err)
        assert err[k][1] <= 1.1 * err["split"][1] + 1e-4, (k, err)
    # absolute ceilings, ~2x over the measured values (CPU, torch 2.7.1): fp32 low-rank gradient error 8.3e-3 (stage 1),
    # 6.5e-3 factor / 1.0e-2 Function (stage 2); bf16 low-rank 8.6e-3 (stage 1) / 4.7e-2 (stage 2); split 1.0e-1 (stage 2)
    assert err["fused"][1] < 2e-2 and err["fused_bf16"][1] < (2e-2 if stage == 1 else 1e-1), err
    # the fp32 low-rank part is at least as close to fp64 as the bf16 one in the gradients (its purpose); the loss is one
    # scalar whose bf16 roundings elsewhere can cancel either way, so only its split bound above applies to it
    assert err["fused"][1] <= err["fused_bf16"][1] + 1e-4, err


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("core", [False, True])
@pytest.mark.parametrize("cdt", [torch.float64, torch.float32])
@pytest.mark.parametrize("scalar", ["float", "tensor"])
def test_tsm_expert_function_matches_autograd_reference(stage, core, cdt, scalar):
    """_TSMExpertLinear against the same math written in plain autograd ops (fp64 parameters): output, input gradient,
    the experts' down / up gradients - only the ones asked for (needs_input_grad), each contiguous with its parameter's
    strides - the scalar's (a Parameter; a float scale gets none) and the router's (stage 2) gradients, with the router
    pooling the input or the core bottleneck. ld == cdt == fp64 is exact; cdt = fp32 under ld = fp64 takes the fp32-variant
    branch (two dtypes, the delta rounded once into the fp32 base GEMM) and is compared at fp32 precision."""
    t, inn, out, r, m, ti = 7, 24, 20, 4, 3, 699
    k = 0.35

    def params():
        g = torch.Generator().manual_seed(4)

        def mk(*s):
            return (torch.randn(*s, generator=g, dtype=torch.float64) * 0.3).requires_grad_(True)

        downs, ups = [mk(r, inn) for _ in range(m)], [mk(out, r) for _ in range(m)]
        if stage == 1:
            downs[1].requires_grad_(False)
            ups[2].requires_grad_(False)
        else:
            for p in downs + ups:
                p.requires_grad_(False)
        rw, rb, ew = mk(m - 1, r if core else inn), mk(m - 1), mk(1000, m - 1)  # the router reads z of that width
        for p in (rw, rb, ew):
            p.requires_grad_(stage == 2)
        s = torch.tensor(0.8, dtype=torch.float64, requires_grad=True) if scalar == "tensor" else None
        return mk(t, inn), mk(out, inn).detach(), mk(out).detach(), downs, ups, rw, rb, ew, s

    w = torch.randn(t, out, generator=torch.Generator().manual_seed(3), dtype=cdt)

    x, W0, b, downs, ups, rw, rb, ew, s = params()
    weights = [p for pair in zip(downs, ups) for p in pair]
    if scalar == "tensor":
        y = laf._TSMExpertLinear.apply(x, W0, b, s, k, *((rw, rb, ew) if stage == 2 else (None, None, None)), ti, core,
                                       cdt, torch.float64, *weights)
    else:
        y = laf._TSMExpertLinear.apply(x, W0, b, 0.8 * k, None, *((rw, rb, ew) if stage == 2 else (None, None, None)),
                                       ti, core, cdt, torch.float64, *weights)
    assert y.dtype == cdt
    (y * w).sum().backward()
    got = {"y": y.detach(), "x": x.grad, "s": None if s is None else s.grad, "rw": rw.grad, "rb": rb.grad, "ew": ew.grad}
    got.update({f"d{j}": p.grad for j, p in enumerate(downs)})
    got.update({f"u{j}": p.grad for j, p in enumerate(ups)})
    for j, p in enumerate(ups):
        if p.grad is not None:
            assert p.grad.is_contiguous() and p.grad.stride() == p.stride() and p.grad.dtype == p.dtype, j
    for j, p in enumerate(downs):
        if p.grad is not None:
            assert p.grad.is_contiguous() and p.grad.stride() == p.stride(), j

    x, W0, b, downs, ups, rw, rb, ew, s = params()
    sk = (s if s is not None else 0.8) * k
    A = torch.cat(downs, 0)
    B = torch.cat(ups, 1) * sk
    if stage == 2:
        src = x @ downs[0].T if core else x
        z = src.detach().mean(0)  # the router pools in promote(activation dtype, fp32): fp64 here
        gates = rw @ z + rb + ew[ti]
        B = (B.view(out, m, r) * torch.cat([torch.ones(1, dtype=torch.float64), gates]).view(1, m, 1)).view(out, m * r)
    y = (x @ W0.T + (x @ A.T) @ B.T + b).to(cdt)
    (y * w).sum().backward()
    ref = {"y": y.detach(), "x": x.grad, "s": None if s is None else s.grad, "rw": rw.grad, "rb": rb.grad, "ew": ew.grad}
    ref.update({f"d{j}": p.grad for j, p in enumerate(downs)})
    ref.update({f"u{j}": p.grad for j, p in enumerate(ups)})
    tol = dict(rtol=1e-9, atol=1e-12) if cdt == torch.float64 else dict(rtol=2e-4, atol=1e-5)
    _assert_same(got, ref, **tol)
    assert (got["s"] is not None) is (scalar == "tensor")
    assert (got["rw"] is not None) is (stage == 2) and (got["ew"] is not None) is (stage == 2)
    if stage == 1:
        assert got["d1"] is None and got["u2"] is None and all(got[n] is not None for n in ("d0", "d2", "u0", "u1"))
    else:
        assert all(got[f"d{j}"] is None and got[f"u{j}"] is None for j in range(m))
        nz = got["ew"].abs().sum(1).nonzero().flatten().tolist()
        assert nz == [ti]  # one embedding row, as select's backward gave


def test_tsm_mixed_plan_and_switch(monkeypatch):
    """_tsm_mixed_plan: the experts active anywhere in the chunk in (scale, expert) order, the per-sample block mask,
    each sample's core block, stage 2 the groups by timestep; cached per step and shared by the modules (one plan for
    one ints tuple); 'single' in stage 1 when the timesteps differ but every sample picks the same experts (the plain
    fused call then, no masked Function), never in stage 2 (the gate depends on each sample's timestep: the same chunk
    runs masked and equals the per-group loop). network_args tsm_mixed=False switches the group loop back on; '' = not
    given."""
    monkeypatch.setattr(laf, "_TSM_FUSED", True)
    monkeypatch.setattr(laf, "_TSM_EXPERT_FN", False)
    monkeypatch.setattr(laf, "_TSM_MIXED", True)
    m1 = _tsm_module(1)
    laf._TSM_PLANS.clear()
    ints = [900, 100, 900, 5]
    plan = _tsm_mixed_plan_for(m1, ints)
    experts = [laf._tsm_experts(m1, t) for t in ints]
    want_u = sorted({(s, e) for row in experts for s, e in enumerate(row)})
    assert plan["U"] == want_u and plan["nb"] == len(want_u) and not plan["single"]
    assert plan["block_mask"].shape == (4, len(want_u)) and plan["block_mask"].sum(1).tolist() == [2.0] * 4
    for b, row in enumerate(experts):
        on = {want_u[i] for i in plan["block_mask"][b].nonzero().flatten().tolist()}
        assert on == set(enumerate(row)) and want_u[int(plan["core_pos"][b])] == (0, row[0])
    assert "grp_idx" not in plan
    assert _tsm_mixed_plan_for(m1, ints) is plan  # cached
    assert _tsm_mixed_plan_for(_tsm_module(1, seed=3), ints) is plan  # shared by every module with the same shape
    same = _tsm_mixed_plan_for(m1, [120, 124])  # different timesteps, the same core expert (0): nothing to mask
    assert same["single"] and same["nb"] == 2
    m2 = _tsm_module(2)
    assert not _tsm_mixed_plan_for(m2, [120, 124])["single"]  # stage 2: gates per timestep, so the masked path
    # the dispatch: stage 1 single -> the plain fused call, no masked Function; stage 2 same chunk -> masked, == group loop
    fused_calls, masked_calls = [], []
    orig_fused, orig_masked = laf._tsm_fused_linear, laf._TSMMaskedLinear.apply
    monkeypatch.setattr(laf, "_tsm_fused_linear", lambda *a: fused_calls.append(1) or orig_fused(*a))
    monkeypatch.setattr(laf._TSMMaskedLinear, "apply", lambda *a: masked_calls.append(1) or orig_masked(*a))
    x = torch.randn(2, 1, 3, 4, 24, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    _tsm_run(_tsm_module(1), x, [120.0, 124.0], checkpoint=True)
    assert len(fused_calls) == 2 and not masked_calls  # forward + checkpoint recompute, one fused call each
    fused_calls.clear()
    got2 = _tsm_run(_tsm_module(2), x, [120.0, 124.0], checkpoint=True)
    assert len(masked_calls) == 2 and not fused_calls
    monkeypatch.setattr(laf, "_TSM_MIXED", False)
    _assert_same(got2, _tsm_run(_tsm_module(2), x, [120.0, 124.0], checkpoint=True))
    rows = got2["timestep_embed.weight"].abs().sum(1).nonzero().flatten().tolist()
    assert rows == [119, 123]  # one embedding row per distinct timestep got its gradient
    monkeypatch.setattr(laf, "_TSM_MIXED", True)
    p2 = _tsm_mixed_plan_for(m2, ints)
    assert p2["n_groups"] == 3 and p2["grp_idx"].tolist() == [0, 1, 0, 2] and p2["counts"].flatten().tolist() == [2.0, 1.0, 1.0]
    assert p2["t_idx"].tolist() == [899, 99, 4] and p2["ctx_pos"].shape == (4, 1)
    assert all(want_u[int(p)][0] == 1 for p in p2["ctx_pos"].flatten().tolist())
    laf.set_timestep(argparse.Namespace(loras=[m1]), torch.tensor([900.0, 100.0]))  # a new step clears the plans
    assert not any(k[0] == "mixed" for k in laf._TSM_PLANS)
    laf.clear_timestep(argparse.Namespace(loras=[m1]))

    monkeypatch.setattr(laf, "_TSM_MIXED", True)
    kw = laf.prepare_network_kwargs("tsm", {"algo": "tsm", "tsm_mixed": "False"}, LOG)
    assert "tsm_mixed" not in kw and laf._TSM_MIXED is False
    laf.prepare_network_kwargs("tsm", {"algo": "tsm", "tsm_mixed": ""}, LOG)
    assert laf._TSM_MIXED is True
    laf.prepare_network_kwargs("tsm", {"algo": "tsm"}, LOG)
    assert laf._TSM_MIXED is True


def _tsm_mixed_plan_for(m, ints):
    return laf._tsm_mixed_plan(m, [int(t) for t in ints], torch.device("cpu"))


@pytest.mark.parametrize("mixed", [True, False])
@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("lowrank_fp32", [False, True])
@pytest.mark.parametrize("router", ["input", "bottleneck"])
def test_tsm_fused_both_lowrank_modes_match_split_fp64(monkeypatch, stage, lowrank_fp32, router, mixed):
    """Both low-rank dtype modes of the fused path == the split forward (== stock, the tests above) on a mixed-timestep
    batch with use_scalar=True: output, input gradient, expert / router gradients and the scalar's gradient; the stage-2
    router pooling the input or the core bottleneck; the default one masked call per module (_TSMMaskedLinear, mixed)
    and the per-group loop (tsm_mixed=False). tsm_lowrank_fp32 off: everything fp64, exact. On: _tsm_mixed (fp64
    experts over an fp32 base and input), so the calls really land in the fp32-low-rank Functions (_TSMLinear per group,
    or _TSMMaskedLinear with ld fp64 != cdt fp32) and the agreement is at fp32 level."""
    ts = [900.0, 100.0, 900.0, 5.0]
    xdt = torch.float32 if lowrank_fp32 else torch.float64
    x = torch.randn(len(ts), 1, 3, 4, 24, generator=torch.Generator().manual_seed(6), dtype=xdt)
    make = ((lambda: _tsm_mixed(stage, router)) if lowrank_fp32
            else (lambda: _tsm_module(stage, router=router, use_scalar=True)))
    monkeypatch.setattr(laf, "_TSM_LOWRANK_FP32", lowrank_fp32)
    monkeypatch.setattr(laf, "_TSM_FUSED", True)
    monkeypatch.setattr(laf, "_TSM_EXPERT_FN", False)
    monkeypatch.setattr(laf, "_TSM_MIXED", mixed)
    calls = {"lr": [], "tl": [], "mx": []}
    for name, cls in (("lr", laf._LowRankLinear), ("tl", laf._TSMLinear), ("mx", laf._TSMMaskedLinear)):
        orig = cls.apply
        monkeypatch.setattr(cls, "apply", (lambda n, o: lambda *a: calls[n].append(a) or o(*a))(name, orig))
    got = _tsm_run(make(), x, ts, checkpoint=True)
    groups = len({tuple(laf._tsm_experts(make(), int(t))) if stage == 1 else t for t in ts})
    assert groups > 1
    if mixed:  # one call per module call: forward + checkpoint recompute
        assert len(calls["mx"]) == 2 and not calls["lr"] and not calls["tl"]
        ld, cdt = calls["mx"][0][8], calls["mx"][0][7]
        assert (ld, cdt) == ((torch.float64, torch.float32) if lowrank_fp32 else (torch.float64, torch.float64))
    else:
        assert len(calls["tl" if lowrank_fp32 else "lr"]) == 2 * groups and not calls["lr" if lowrank_fp32 else "tl"]
        assert not calls["mx"]
    monkeypatch.setattr(laf, "_TSM_FUSED", False)
    ref = _tsm_run(make(), x, ts, checkpoint=True)
    _assert_same(got, ref, **(dict(rtol=2e-4, atol=1e-5) if lowrank_fp32 else {}))
    assert got["scalar"] is not None and bool(torch.any(got["scalar"] != 0))
    # an expert no sample of the chunk uses is not in the graph (grad None, the optimizer skips it), as on every path
    active = {(s, e) for t in ts for s, e in enumerate(laf._tsm_experts(make(), int(t)))}
    for s, scale_experts in enumerate(make().experts):
        for e in range(len(scale_experts)):
            key = f"experts.{s}.{e}.down.weight"
            assert (got[key] is None) == (stage == 2 or (s, e) not in active), (key, got[key] is None)


@pytest.mark.parametrize("expert_fn", [False, True])
@pytest.mark.parametrize("stage", [1, 2])
def test_tsm_expert_grads_dtype_layout_active_only(monkeypatch, stage, expert_fn):
    """Anima-like (bf16 base, fp32 experts, fp32 input, bf16 autocast, checkpointing), the factor path and
    _TSMExpertLinear: every gradient lands in its parameter's dtype, contiguous with the parameter's strides, only for
    the experts active at the batch's timestep (stage 1) or for the router (stage 2: experts frozen; one embedding row
    non-zero); the input gradient comes back in the input's dtype. The raw gradients are contiguous in the parameter's
    layout only from _TSMExpertLinear (read from what its backward returns: a tensor hook on a leaf the Function saved
    for backward does not fire under the non-reentrant checkpoint, torch 2.7.1) - the factor path hands AccumulateGrad
    a column slice of cat(ups)' gradient (tensor hooks), which it clones."""
    monkeypatch.setattr(laf, "_TSM_FUSED", True)
    monkeypatch.setattr(laf, "_TSM_EXPERT_FN", expert_fn)
    t = 650
    m = _tsm_module(stage, dtype=torch.float32)
    m.org_module[0].to(torch.bfloat16)
    net = argparse.Namespace(loras=[m])
    raw = {}  # parameter -> (contiguous, same strides) of the gradient as the autograd graph delivered it
    for p in m.parameters():
        if p.requires_grad:
            p.register_hook(lambda g, p=p: raw.__setitem__(p, (g.is_contiguous(), g.stride() == p.stride())))
    applied, returned = [], []  # the Function's inputs (last apply) and what its backward returned
    orig_apply, orig_bwd = laf._TSMExpertLinear.apply, laf._TSMExpertLinear.backward
    monkeypatch.setattr(laf._TSMExpertLinear, "apply", lambda *a: applied.append(a) or orig_apply(*a))
    monkeypatch.setattr(laf._TSMExpertLinear, "backward",
                        staticmethod(lambda ctx, g: returned.append(orig_bwd(ctx, g)) or returned[-1]))
    x = torch.randn(2, 5, 24, requires_grad=True)
    laf.set_timestep(net, torch.tensor([float(t)]))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = torch.utils.checkpoint.checkpoint(m, x, use_reentrant=False)
    assert out.dtype == torch.bfloat16
    (out.float() * torch.randn(out.shape, generator=torch.Generator().manual_seed(3))).sum().backward()
    laf.clear_timestep(net)
    assert x.grad is not None and x.grad.dtype == torch.float32 and x.grad.shape == x.shape
    fn_raw = {}  # id(parameter) -> (contiguous, same strides, dtype) of the gradient _TSMExpertLinear returned for it
    if expert_fn:
        assert len(applied) == 2 and len(returned) == 1  # forward + checkpoint recompute, one backward
        for a, g in zip(applied[-1], returned[-1]):
            if isinstance(a, torch.Tensor) and a.requires_grad and g is not None:
                fn_raw[id(a)] = (g.is_contiguous(), g.stride() == a.stride(), g.dtype)
    else:
        assert not applied and not returned
    active = {(si, e) for si, e in enumerate(laf._tsm_experts(m, t))}
    for si, scale in enumerate(m.experts):
        for e, ex in enumerate(scale):
            for p in (ex["down"].weight, ex["up"].weight):
                if stage == 1 and (si, e) in active:
                    assert p.grad is not None and p.grad.dtype == p.dtype == torch.float32, (si, e)
                    assert p.grad.is_contiguous() and p.grad.stride() == p.stride(), (si, e)
                    assert bool(torch.any(p.grad != 0)), (si, e)
                    if expert_fn:  # the Function's own layout and dtype, nothing for AccumulateGrad to clone or cast
                        assert fn_raw[id(p)] == (True, True, p.dtype), (si, e, fn_raw[id(p)])
                else:
                    assert p.grad is None, (stage, si, e)
    router = (m.router_fc.weight, m.router_fc.bias, m.timestep_embed.weight)
    if stage == 1:
        assert all(p.grad is None for p in router)
    else:
        for p in router:
            assert p.grad is not None and p.grad.dtype == p.dtype == torch.float32 and p.grad.is_contiguous()
            if expert_fn:
                assert fn_raw[id(p)] == (True, True, p.dtype), fn_raw[id(p)]
        rows = m.timestep_embed.weight.grad.abs().sum(1).nonzero().flatten().tolist()
        assert rows == [t - 1]
    if stage == 1 and not expert_fn:  # the factor path: a column slice (ups) reaches AccumulateGrad, which clones it
        ups = [m.experts[si][e]["up"].weight for si, e in active]
        assert any(raw[p] != (True, True) for p in ups), {p.shape: raw[p] for p in ups}


# ------------------------------------------------------------------------------- LoRA family: factored forward (speed)


def _lf_module(algo, dora=None, n_split=1, bypass=False, dtype=torch.float64, seed=0, rank=4, bias=True, mult=0.7,
               nu=0.4, **kw):
    """One GoRA / RaLoRA / LoRA² / T-LoRA module on a Linear(24 -> 20) with random factors (up is zero otherwise),
    DoRA magnitudes off the base norms, and a non-trivial multiplier."""
    from lycoris.modules.locon import GoRAModule, RaLoRAModule
    from lycoris.modules.lora2 import LoRA2Module
    from lycoris.modules.tlora import TLoraModule

    torch.manual_seed(seed)
    org = nn.Linear(24, 20, bias=bias).to(torch.bfloat16).to(dtype)  # bf16 values, as Anima's base weights
    org.requires_grad_(False)
    kwargs = dict(multiplier=mult, lora_dim=rank, alpha=rank / 2, bypass_mode=bypass, **kw)
    if dora:
        kwargs.update(weight_decompose=True, wd_on_output=dora == "output")
    cls = {"gora": GoRAModule, "ralora": RaLoRAModule, "lora2": LoRA2Module, "tlora": TLoraModule}[algo]
    if algo == "lora2":
        kwargs.update(lora2_nu_target=rank // 2)
    m = cls("t", org, **kwargs)
    if algo == "ralora" and n_split > 1:
        m.dynamic_init(rank, rank, n_split)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n == "lora2_nu":
                p.fill_(nu)
            elif n == "dora_scale":
                p.mul_(1 + 0.05 * torch.randn(p.shape, generator=g))
            elif algo == "tlora":  # away from the init (current == base factors)
                p.add_(torch.randn(p.shape, generator=g) * 0.02 * p.abs().mean())
            elif n != "scalar":
                p.copy_(torch.randn(p.shape, generator=g) * 0.2 / p.shape[-1] ** 0.5)
    m.to(dtype)
    m.apply_to()
    m.train()
    return m


def _lf_run(m, x, checkpoint=False, ts=None):
    """Output and every gradient (input + parameters) of sum(gelu(out) * w); gelu saves the output, so checkpointing
    recomputes the module as in a real block."""
    net = argparse.Namespace(loras=[m])
    x = x.detach().clone().requires_grad_(True)
    for p in m.parameters():
        p.grad = None
    if ts is not None:
        laf.set_timestep(net, torch.tensor(ts))
    fn = lambda t: F.gelu(m(t))  # noqa: E731
    out = torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False) if checkpoint else fn(x)
    w = torch.randn(out.shape, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    (out.to(w.dtype) * w).sum().backward()
    if ts is not None:
        laf.clear_timestep(net)
    res = {"out": out.detach(), "x": x.grad}
    res.update({n: (None if p.grad is None else p.grad.clone()) for n, p in m.named_parameters()})
    return res


_LF_SHAPES = [(6, 24), (2, 5, 24), (2, 1, 3, 2, 24)]  # 2-D, 3-D, Anima's 5-D MLP input


@pytest.fixture(params=["factored", "dense"])
def dora_path(request, monkeypatch):
    """Both functions for DoRA per output row (n == 1): _LowRankDoRA (factored) and _DenseDoRAOut (dense rebuild with a
    hand-written backward); on CUDA _lf_dora_dense picks one per call, on the CPU the factored one."""
    monkeypatch.setattr(laf, "_LF_DORA_PATH", request.param)
    return request.param


@pytest.mark.parametrize("algo", ["gora", "ralora", "lora2"])
@pytest.mark.parametrize("dora", [None, "output", "input"])
@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("shape", _LF_SHAPES)
def test_lowrank_factored_forward_matches_stock(algo, dora, bypass, shape, dora_path):
    """GoRA / RaLoRA / LoRA² factored forward == stock LyCORIS rebuild (dense delta, merged weight, DoRA norm over it) /
    bypass: output, input gradient and every parameter gradient (fp64), with and without checkpointing. DoRA per input
    column keeps the stock dense path (bit-identical)."""
    x = torch.randn(shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    ref = _with_stock(lambda: _lf_run(_lf_module(algo, dora, bypass=bypass), x))
    for ckpt in (False, True):
        _assert_same(_lf_run(_lf_module(algo, dora, bypass=bypass), x, checkpoint=ckpt), ref)


@pytest.mark.parametrize("dora", [None, "output", "input"])
@pytest.mark.parametrize("scalar_type", ["scalar", "row", "column", "row_column"])
def test_gora_factored_forward_scalars(dora, scalar_type, dora_path):
    """use_scalar (every scalar_type) folded into the factors == stock _apply_scalar on the dense delta."""
    x = torch.randn(2, 5, 24, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    kw = dict(use_scalar=True, scalar_type=scalar_type)
    ref = _with_stock(lambda: _lf_run(_lf_module("gora", dora, **kw), x))
    _assert_same(_lf_run(_lf_module("gora", dora, **kw), x, checkpoint=True), ref)


@pytest.mark.parametrize("nu", [0.6, 0.25, 3.0, -0.1])
@pytest.mark.parametrize("dora", [None, "output"])
def test_lora2_factored_forward_effective_rank(nu, dora, dora_path):
    """LoRA²: Lambda masked past D on the device == make_weight's slice to D (D = 4, 8, 1, and lora_dim for nu <= 0 at
    rank 8): same output and gradients; the ranks past D get zero gradients, as the slice gives them."""
    x = torch.randn(2, 5, 24, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    ref = _with_stock(lambda: _lf_run(_lf_module("lora2", dora, rank=8, nu=nu), x))
    got = _lf_run(_lf_module("lora2", dora, rank=8, nu=nu), x, checkpoint=True)
    _assert_same(got, ref)
    d = _lf_module("lora2", dora, rank=8, nu=nu).compute_effective_rank()
    if d < 8:
        assert float(got["lora_up.weight"][:, d:].abs().max()) == 0.0
        assert float(got["lora_down.weight"][d:].abs().max()) == 0.0


def test_lora2_lambda_mask_equals_compute_effective_rank():
    """The device-side D equals lora2_utils.compute_effective_rank (Python float64 from nu.item()) for many nu,
    including values where -ln(1-q)/nu lands on an integer and nu <= 0; Lambda equals compute_lambda_diag up to D."""
    from lycoris.modules.lora2_utils import compute_effective_rank, compute_lambda_diag

    m = _lf_module("lora2", rank=16, dtype=torch.float32)
    c = -math.log(1.0 - m.lora2_quantile)
    nus = [c / k for k in range(1, 20)] + [0.01, 0.1, 0.13, 0.5, 1.0, 2.5, 50.0, 0.0, -0.3, 1e-9]
    for k in (3, 7, 16):
        v = torch.tensor(c / k, dtype=torch.float32)
        nus += [float(torch.nextafter(v, torch.tensor(1.0))), float(torch.nextafter(v, torch.tensor(0.0)))]
    for nu in nus:
        with torch.no_grad():
            m.lora2_nu.fill_(nu)
        d = compute_effective_rank(m.lora2_nu, m.lora2_quantile, m.lora_dim)
        lam = laf._lora2_lam(m).detach()
        assert float(lam[d:].abs().sum()) == 0.0, (nu, d)
        assert torch.equal(lam[:d], compute_lambda_diag(m.lora2_nu.detach(), d)), (nu, d)


@pytest.mark.parametrize("dora", [None, "output"])
def test_lora2_up_factor_gradient_from_fp32_mid(dora, dora_path):
    """LoRA²: nu's gradient sums the up-factor gradient over the output rows (through Lambda). From mid = x A^T rounded
    to bf16 in the forward (one rounding shared by every row) that sum lost the averaging of the stock dense weight
    gradient: nu up to ~3x the stock bf16 error on 256-2048 wide layers. The up-factor gradient now comes from x A^T
    recomputed in fp32. bf16 autocast error vs fp64 (same bf16 base and input), a 256-wide layer, 128 tokens, 8 seeds:
    the up-factor gradient (mean) well below stock; nu by its median (its error is heavy-tailed: a near-zero true
    gradient in a seed makes either code look 10x off) not above stock."""
    from lycoris.modules.lora2 import LoRA2Module

    def build(seed, dtype):
        torch.manual_seed(seed)
        org = nn.Linear(256, 256, bias=False)
        with torch.no_grad():
            org.weight.normal_(0, 0.05)
        org = org.to(torch.bfloat16).to(dtype).requires_grad_(False)
        kw = dict(weight_decompose=True, wd_on_output=True) if dora else {}
        m = LoRA2Module("t", org, multiplier=1.0, lora_dim=32, alpha=32, lora2_nu_target=16, **kw)
        g = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            m.lora2_nu.fill_(0.15)
            for n, p in m.named_parameters():
                if n in ("lora_down.weight", "lora_up.weight"):
                    p.copy_(torch.randn(p.shape, generator=g) * 0.09 / p.shape[-1] ** 0.5)
                elif n == "dora_scale":
                    p.mul_(1 + 0.05 * torch.randn(p.shape, generator=g))
        m.to(dtype)
        m.apply_to()
        m.train()
        return m

    def errs():
        e = {}
        for s in range(8):
            x = torch.randn(1, 128, 256, generator=torch.Generator().manual_seed(s), dtype=torch.float64).bfloat16()
            ref = _lf_run(build(s, torch.float64), x.double())
            m = build(s, torch.float32)
            m.org_module[0].to(torch.bfloat16)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                got = _lf_run(m, x, checkpoint=True)
            for k in ("lora_up.weight", "lora2_nu"):
                e.setdefault(k, []).append(float((got[k].double() - ref[k]).norm() / ref[k].norm()))
        return {"up": sum(e["lora_up.weight"]) / 8, "nu": sorted(e["lora2_nu"])[4]}

    old = _with_stock(errs)
    new = errs()
    print(f"[bf16] lora2 dora={dora}: stock {old} factored {new}")
    assert new["up"] <= 0.8 * old["up"] and new["nu"] <= old["nu"], (old, new)


def _ralora_block_reference(m, x):
    """RaLoRA n_split > 1 forward written out: W0 + scale * blockdiag(B_i A_i) * scalar, DoRA per output row (the stock
    builds the block weight with torch.zeros in the default dtype: fp32 even for an fp64 module)."""
    dw = torch.block_diag(*[b @ a for a, b in zip(m._mini_lora_A, m._mini_lora_B)]) * m.scalar
    W0 = m.org_module[0].weight
    if m.wd:
        W = W0 + dw * m.scale
        norm = torch.linalg.vector_norm(W, dim=1, keepdim=True) + m._dora_eps
        W = W * (m.multiplier * (m.dora_scale / norm - 1) + 1)
    else:
        W = W0 + dw * m.scale * m.multiplier
    return F.linear(x, W, m.org_module[0].bias)


@pytest.mark.parametrize("n_split", [2, 4])
@pytest.mark.parametrize("dora", [None, "output"])
@pytest.mark.parametrize("shape", _LF_SHAPES)
def test_ralora_blocks_factored_forward(n_split, dora, shape):
    """RaLoRA block-diagonal layers: batched-matmul factored forward == the written-out fp64 math (tight) and the stock
    rebuild up to its fp32 block weight; bypass == stock bypass (one matmul pair per block)."""
    x = torch.randn(shape, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    got = _lf_run(_lf_module("ralora", dora, n_split, use_scalar=True), x, checkpoint=True)

    m = _lf_module("ralora", dora, n_split, use_scalar=True)
    xr = x.clone().requires_grad_(True)
    out = F.gelu(_ralora_block_reference(m, xr))
    (out * torch.randn(out.shape, generator=torch.Generator().manual_seed(3), dtype=out.dtype)).sum().backward()
    ref = {"out": out.detach(), "x": xr.grad}
    ref.update({n: (None if p.grad is None else p.grad.clone()) for n, p in m.named_parameters()})
    _assert_same(got, ref, rtol=1e-10, atol=1e-12)
    assert got["lora_up.weight"] is None and got["lora_down.weight"] is None  # unused dense factors of block layers
    stock = _with_stock(lambda: _lf_run(_lf_module("ralora", dora, n_split, use_scalar=True), x))
    _assert_same(got, stock, rtol=1e-5, atol=1e-7)  # stock: fp32 block weight
    ref_b = _with_stock(lambda: _lf_run(_lf_module("ralora", None, n_split, bypass=True), x))
    _assert_same(_lf_run(_lf_module("ralora", None, n_split, bypass=True), x, checkpoint=True), ref_b)


@pytest.mark.parametrize("ts", [None, [300.0], [100.0, 900.0]])
@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("use_scalar", [False, True])
def test_tlora_factored_forward_matches_stock(ts, bypass, use_scalar):
    """T-LoRA [P lam - Pb lamb, Pb lamb] diag(m, m) [Q; Q - Qb] == stock rebuild (dense delta, second GEMM) / bypass /
    per-sample masks: output and every gradient (fp64), no mask / one mask / one per sample, checkpointing; the masked
    ranks get no lambda gradient."""
    B = 2 if ts is None or len(ts) == 1 else len(ts)
    x = torch.randn(B, 1, 3, 2, 24, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    kw = dict(rank=6, use_scalar=use_scalar)
    ref = _with_stock(lambda: _lf_run(_lf_module("tlora", None, bypass=bypass, **kw), x, ts=ts))
    for ckpt in (False, True):
        got = _lf_run(_lf_module("tlora", None, bypass=bypass, **kw), x, checkpoint=ckpt, ts=ts)
        _assert_same(got, ref)
    if ts == [300.0]:  # 6 ranks, min 3: int(0.7 * 3) + 3 = 5 active
        assert float(got["lambda_layer"].reshape(-1)[5:].abs().sum()) == 0.0


def test_tlora_cpu_mask_moved_once():
    """train_network.apply_tlora_mask stores compute_timestep_mask's CPU tensor: the first module call moves it to the
    compute device and stores the copy back, so the other modules and the checkpoint recompute make no blocking copy
    (a host sync per call on CUDA). Same values; a mask already on the device is left alone. ("meta" stands in for
    CUDA on this CPU-only test.)"""
    from lycoris.modules.tlora import clear_timestep_mask, compute_timestep_mask, get_timestep_mask, set_timestep_mask

    m = _lf_module("tlora", dtype=torch.float32)
    cpu_mask = compute_timestep_mask(300, 1000, 4, 2, 1.0)
    try:
        set_timestep_mask(cpu_mask)
        got = laf._tlora_mask(m, torch.device("meta"))
        stored = get_timestep_mask()
        assert stored.device.type == "meta" and got.shape == cpu_mask.shape
        assert laf._tlora_mask(m, torch.device("meta")) is stored  # no second copy
        set_timestep_mask(cpu_mask)
        assert torch.equal(laf._tlora_mask(m, torch.device("cpu")), cpu_mask) and get_timestep_mask() is cpu_mask
    finally:
        clear_timestep_mask()


def test_lowrank_fallbacks_keep_the_stock_path():
    """Configurations the factored forward does not cover run LoConModule.forward: a base weight that needs its gradient
    (the GoRA/RaLoRA pre-pass) and rank dropout while training (eval: covered)."""
    from lycoris.modules.locon import LoConModule

    x = torch.randn(2, 5, 24, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    calls = []
    orig = LoConModule.__dict__["forward"]

    def spy(self, *a, **k):
        calls.append(1)
        return orig(self, *a, **k)

    LoConModule.forward = spy
    try:
        m = _lf_module("gora", "output")
        m(x)
        assert not calls  # covered
        m.org_module[0].weight.requires_grad_(True)
        m(x)
        assert len(calls) == 1  # pre-pass: stock (gives W0 its gradient)
        m2 = _lf_module("ralora", None, rank_dropout=0.5)
        m2(x)
        assert len(calls) == 2
        m2.eval()
        m2(x)
        assert len(calls) == 2  # rank dropout is a no-op in eval: covered
    finally:
        LoConModule.forward = orig


def test_dora_factored_norm_zero_row(dora_path):
    """A zero base row with a zero up row: |W'_i| = 0 -> norm 0 and zero gradients for that row (as vector_norm's
    backward gives), no NaN; the rest == stock."""
    def run():
        m = _lf_module("gora", "output")
        with torch.no_grad():
            m.org_module[0].weight[3].zero_()
            m.lora_up.weight[3].zero_()
        return _lf_run(m, torch.randn(2, 5, 24, generator=torch.Generator().manual_seed(9), dtype=torch.float64))

    ref = _with_stock(run)
    got = run()
    assert all(v is None or bool(torch.isfinite(v).all()) for v in got.values())
    _assert_same(got, ref)


@pytest.mark.parametrize("algo,n", [("gora", 1), ("ralora", 4), ("lora2", 1)])
def test_dora_factored_zero_magnitude_row(algo, n, dora_path):
    """A base row that is zero when the network is built gets dora_scale 0; at multiplier 1 its DoRA factor k is 0 (the
    row of k * W0 and of the up factor vanish). grad_e = sum g (y - C) / k was 0/0 there: NaN in dora_scale and in every
    lora_down / lora_up (and nu) gradient. Now finite and == stock, also through checkpointing."""
    from lycoris.modules.locon import GoRAModule, RaLoRAModule
    from lycoris.modules.lora2 import LoRA2Module

    def run(checkpoint=False):
        torch.manual_seed(0)
        org = nn.Linear(24, 20).to(torch.bfloat16).to(torch.float64)
        with torch.no_grad():
            org.weight[3].zero_()
        org.requires_grad_(False)
        cls = {"gora": GoRAModule, "ralora": RaLoRAModule, "lora2": LoRA2Module}[algo]
        kw = {"lora2_nu_target": 2} if algo == "lora2" else {}
        m = cls("t", org, multiplier=1.0, lora_dim=4, alpha=2, weight_decompose=True, wd_on_output=True, **kw)
        if n > 1:
            m.dynamic_init(4, 4, n)
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for name, p in m.named_parameters():
                if name not in ("dora_scale", "lora2_nu"):
                    p.copy_(torch.randn(p.shape, generator=g) * 0.2 / p.shape[-1] ** 0.5)
        m.to(torch.float64)
        m.apply_to()
        m.train()
        assert float(m.dora_scale.reshape(-1)[3]) == 0.0
        x = torch.randn(2, 5, 24, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
        return _lf_run(m, x, checkpoint=checkpoint)

    ref = _with_stock(run)
    assert all(v is None or bool(torch.isfinite(v).all()) for v in ref.values())
    for ckpt in (False, True):
        got = run(ckpt)
        assert all(v is None or bool(torch.isfinite(v).all()) for v in got.values())
        tol = dict(rtol=1e-6, atol=1e-7) if n > 1 else {}  # n > 1: stock assembles the blocks in fp32 zeros
        _assert_same(got, ref, **tol)


@pytest.mark.parametrize("algo,n", [("gora", 1), ("ralora", 4)])
def test_dora_zero_magnitude_on_a_nonzero_base_row(algo, n, dora_path):
    """dora_scale exactly 0 on a NONZERO base row (LyCORIS' init never makes it, dora_scale = ||W0_i||; a file trained
    on another base can): k = 0 at multiplier 1 zeroes that row of k * W0, so the factored functions (_LowRankDoRA,
    _LowRankLinear) cannot get x W0_i^T back from y - C and give that row's dora_scale gradient without its base term
    (documented limitation; recovering it needs the dense g^T x). Everything else - output, input gradient, the
    factors, every other dora_scale entry - stays finite and == stock; _DenseDoRAOut (n == 1, dense) is exact there."""
    from lycoris.modules.locon import GoRAModule, RaLoRAModule

    row = 5

    def run():
        torch.manual_seed(0)
        org = nn.Linear(24, 20).to(torch.bfloat16).to(torch.float64)
        org.requires_grad_(False)
        cls = {"gora": GoRAModule, "ralora": RaLoRAModule}[algo]
        m = cls("t", org, multiplier=1.0, lora_dim=4, alpha=2, weight_decompose=True, wd_on_output=True)
        if n > 1:
            m.dynamic_init(4, 4, n)
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for name, p in m.named_parameters():
                if name == "dora_scale":
                    p.view(-1)[row] = 0.0
                else:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.2 / p.shape[-1] ** 0.5)
        m.to(torch.float64)
        m.apply_to()
        m.train()
        assert bool(m.org_module[0].weight[row].abs().sum() > 0)
        x = torch.randn(2, 5, 24, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
        return _lf_run(m, x, checkpoint=True)

    ref = _with_stock(run)
    got = run()
    assert all(v is None or bool(torch.isfinite(v).all()) for v in got.values())
    tol = dict(rtol=1e-6, atol=1e-7) if n > 1 else {}
    exact = n == 1 and dora_path == "dense"
    keep = torch.ones(ref["dora_scale"].numel(), dtype=torch.bool)
    if not exact:
        keep[row] = False  # the documented gap: base term missing from this row's magnitude gradient
    assert got.keys() == ref.keys()
    for k in ref:
        a, b = got[k], ref[k]
        assert (a is None) == (b is None), k  # RaLoRA blocks: the unused dense factors get no gradient either way
        if a is None:
            continue
        if k == "dora_scale":
            a, b = a.reshape(-1)[keep], b.reshape(-1)[keep]
        assert torch.allclose(a, b, **(tol or dict(rtol=1e-9, atol=1e-12))), (k, (a - b).abs().max())


def test_lowrank_patches_only_with_the_flag():
    """Without apply_module_patches (flag off) the LoRA-family classes keep LyCORIS' own forward / rank loss."""
    from lycoris.modules.locon import GoRAModule, LoConModule, RaLoRAModule
    from lycoris.modules.lora2 import LoRA2Module
    from lycoris.modules.tlora import TLoraModule

    laf._revert_module_patches()
    try:
        assert "forward" not in GoRAModule.__dict__ and "forward" not in RaLoRAModule.__dict__
        assert "forward" not in LoRA2Module.__dict__ and GoRAModule.forward is LoConModule.forward
        assert isinstance(LoRA2Module.__dict__["get_total_rank_reg_loss"], staticmethod)
        assert not getattr(TLoraModule.__dict__["forward"], "_lycoris_algo_fix", False)
    finally:
        laf.apply_module_patches()
    assert GoRAModule.__dict__["forward"] is laf._lf_forward
    assert TLoraModule.__dict__["forward"] is laf._tlora_fast_forward


@pytest.mark.parametrize("algo,dora,n", [("gora", None, 1), ("ralora", None, 4), ("lora2", None, 1), ("gora", "output", 1),
                                         ("lora2", "output", 1)])
def test_lowrank_backward_inside_autocast(algo, dora, n, dora_path):
    """The factored GEMMs' backward does its own dtype handling: calling backward inside the autocast region gives the
    same gradients as outside it (the trainer calls it outside; with DoRA, the norm's ordinary autograd ops would be
    autocast there, as the stock path's are)."""
    x = torch.randn(2, 5, 24, generator=torch.Generator().manual_seed(12), dtype=torch.float64).bfloat16()
    res = []
    for inside in (False, True):
        m = _lf_module(algo, dora, n, dtype=torch.float32)
        m.org_module[0].to(torch.bfloat16)
        xx = x.clone().requires_grad_(True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = torch.utils.checkpoint.checkpoint(m, xx, use_reentrant=False)
            loss = (out.float() ** 2).sum()
            if inside:
                loss.backward()
        if not inside:
            loss.backward()
        res.append([xx.grad] + [p.grad for p in m.parameters() if p.grad is not None])
    assert len(res[0]) == len(res[1]) and all(torch.equal(a, b) for a, b in zip(*res))


def _stock_rank_reg(cls):
    total = torch.tensor(0.0)
    for mod in cls._lora2_modules:
        total = total + mod.get_rank_reg_loss()
    return total


def test_lowrank_no_host_sync(dora_path):
    """GoRA / RaLoRA / LoRA² / T-LoRA forward + backward: no .item() (LoRA²'s compute_effective_rank synced per call)
    and no tensor made from Python data (a blocking host-to-device copy on CUDA); LoRA²'s rank regularisation is one
    stacked op without either after its first call, same value and gradients as the stock loop."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from lycoris.modules.lora2 import LoRA2Module

    class Watch(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.hits = []

        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            if func.overloadpacket.__name__ in ("lift_fresh", "_local_scalar_dense", "nonzero"):
                self.hits.append(func.overloadpacket.__name__)
            return func(*args, **(kwargs or {}))

    x = torch.randn(2, 5, 24, dtype=torch.float32)
    f32 = torch.float32
    mods = [_lf_module("gora", "output", dtype=f32), _lf_module("ralora", None, 4, dtype=f32),
            _lf_module("lora2", "output", dtype=f32, seed=1), _lf_module("lora2", None, dtype=f32, seed=2),
            _lf_module("tlora", dtype=f32)]
    net = argparse.Namespace(loras=mods)
    laf.set_timestep(net, torch.tensor([400.0]))
    for m in mods:
        m(x).sum().backward()  # warm-up: one-time caches (arange, ones mask)
        w = Watch()
        with w:
            torch.utils.checkpoint.checkpoint(m, x.clone().requires_grad_(), use_reentrant=False).sum().backward()
        assert not w.hits, (type(m).__name__, w.hits)
    laf.clear_timestep(net)
    l2 = [m for m in mods if isinstance(m, LoRA2Module)]
    for m in mods:
        m.zero_grad(set_to_none=True)
    LoRA2Module.get_total_rank_reg_loss()
    w = Watch()
    with w:
        reg = LoRA2Module.get_total_rank_reg_loss()
    assert not w.hits, w.hits
    reg.backward()
    got = [m.lora2_nu.grad.clone() for m in l2]
    for m in mods:
        m.zero_grad(set_to_none=True)
    ref = _stock_rank_reg(LoRA2Module)
    ref.backward()
    assert torch.allclose(reg, ref, rtol=1e-6)
    assert all(torch.equal(g, m.lora2_nu.grad) for g, m in zip(got, l2))


def _bf16_errors(build, x64, seeds, ts=None):
    """Per output/gradient, the mean over seeds of the relative error vs the fp64 module, for bf16 base weights, bf16
    input, fp32 adapter and bf16 autocast (the Anima training setup)."""
    errs = {}
    for s in range(seeds):
        ref = _lf_run(build(s, torch.float64), x64, ts=ts)
        m = build(s, torch.float32)
        m.org_module[0].to(torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            got = _lf_run(m, x64.bfloat16(), checkpoint=True, ts=ts)
        for k, r in ref.items():
            if r is not None and r.norm() > 0:
                errs.setdefault(k, []).append(float((got[k].double() - r).norm() / r.norm()))
    return {k: sum(v) / len(v) for k, v in errs.items()}


@pytest.mark.parametrize("case", ["gora", "gora_dora", "ralora_blocks_dora", "lora2_dora", "tlora"])
def test_lowrank_bf16_error_not_worse_than_stock(case, dora_path):
    """Under bf16 autocast the factored forward is not less accurate than the stock path (relative error vs fp64 of the
    output and of all gradients, mean over fixed seeds): the delta is no longer rounded into the bf16 merged weight, and
    T-LoRA no longer subtracts two rounded full products. The output and the input gradient (what the rest of the
    network sees) and the mean over everything stay within 3%; DoRA's magnitude gradient is recovered from the bf16
    output (sum over tokens of g * x W0^T) and has about one bf16 rounding more than the stock's, printed."""
    algo, dora, n = {"gora": ("gora", None, 1), "gora_dora": ("gora", "output", 1),
                     "ralora_blocks_dora": ("ralora", "output", 4), "lora2_dora": ("lora2", "output", 1),
                     "tlora": ("tlora", None, 1)}[case]
    ts = [300.0] if algo == "tlora" else None
    x = torch.randn(2, 1, 4, 2, 24, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    x = x.bfloat16().double()  # bf16 values, as the activations

    def build(s, dt):
        return _lf_module(algo, dora, n, dtype=dt, seed=s, rank=6 if algo == "tlora" else 4)

    old = _with_stock(lambda: _bf16_errors(build, x, 16, ts))
    new = _bf16_errors(build, x, 16, ts)
    mo, mn = sum(old.values()) / len(old), sum(new.values()) / len(new)
    print(f"[bf16] {case}: mean rel err stock {mo:.3e} factored {mn:.3e} | "
          + " ".join(f"{k}: {old[k]:.2e} -> {new[k]:.2e}" for k in old))
    assert mn <= mo * 1.03 and new["out"] <= old["out"] * 1.03 and new["x"] <= old["x"] * 1.03, (old, new)


@pytest.mark.parametrize("case", ["gora_dora", "ralora_blocks_dora", "lora2_dora"])
def test_lowrank_dora_bf16_parameters_not_worse_than_stock(case, dora_path):
    """--full_bf16 (network parameters, dora_scale included, in bf16), bf16 autocast: k - 1 = m / |W'| - 1 is a small
    difference, so the DoRA norm from the factors runs in fp32 (in bf16 each term of |W0|^2 + 2s cross + s^2 quad was
    rounded on its own: GoRA / RaLoRA blocks mean error 7-14% above stock, dora_scale ~40%). The reference is the fp64
    module with the same bf16-rounded parameters; mean over output and every gradient within 3% of stock, 8 seeds."""
    algo, n = {"gora_dora": ("gora", 1), "ralora_blocks_dora": ("ralora", 4), "lora2_dora": ("lora2", 1)}[case]
    x = torch.randn(2, 1, 4, 2, 24, generator=torch.Generator().manual_seed(11), dtype=torch.float64).bfloat16()

    def errs():
        e = {}
        for s in range(8):
            ref_m = _lf_module(algo, "output", n, dtype=torch.bfloat16, seed=s).to(torch.float64)
            ref_m.org_module[0].to(torch.float64)
            ref = _lf_run(ref_m, x.double())
            with torch.autocast("cpu", dtype=torch.bfloat16):
                got = _lf_run(_lf_module(algo, "output", n, dtype=torch.bfloat16, seed=s), x, checkpoint=True)
            for k, r in ref.items():
                if r is not None and r.norm() > 0:
                    e.setdefault(k, []).append(float((got[k].double() - r).norm() / r.norm()))
        return {k: sum(v) / len(v) for k, v in e.items()}

    old = _with_stock(errs)
    new = errs()
    mo, mn = sum(old.values()) / len(old), sum(new.values()) / len(new)
    print(f"[full_bf16] {case}: mean rel err stock {mo:.3e} factored {mn:.3e} | "
          + " ".join(f"{k}: {old[k]:.2e} -> {new[k]:.2e}" for k in old))
    assert mn <= mo * 1.03, (old, new)


# -------------------------------------------------------------------------------------------------- LoHa / ABBA speed


def _hada_module(algo, *, dora=None, bypass=False, dim=4, alpha=2.0, mult=0.7, scalar=None, dtype=torch.float64,
                 base_dtype=None, inn=24, out=20, seed=0):
    """One LoHa / ABBA module on a Linear(inn -> out) with random non-zero factors (stock init has one zero factor)."""
    from lycoris.modules.abba import AbbaModule
    from lycoris.modules.loha import LohaModule

    torch.manual_seed(seed)
    org = nn.Linear(inn, out)
    with torch.no_grad():
        org.weight.normal_(0, 0.02)
        org.bias.normal_(0, 0.02)
    org = org.to(base_dtype or dtype).requires_grad_(False)
    kw = {"weight_decompose": dora is not None, "wd_on_output": dora == "output", "bypass_mode": bypass}
    if scalar:
        kw.update(use_scalar=True, **({"scalar_type": scalar} if algo == "loha" else {}))
    cls = LohaModule if algo == "loha" else AbbaModule
    m = cls("t", org, multiplier=mult, lora_dim=dim, alpha=alpha, **kw)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n == "dora_scale":
                p.mul_(1 + 0.05 * torch.randn(p.shape, generator=g))
            elif "scalar" in n:
                p.copy_(0.5 + torch.rand(p.shape, generator=g))
            else:
                p.copy_(torch.randn(p.shape, generator=g) * (0.5 / p.shape[-1] ** 0.5))
    m.to(dtype)
    m.train()
    return m


def _hada_run(m, x, checkpoint=False, autocast=False):
    """Output and every gradient (input + parameters) of sum(out * w)."""
    x = x.detach().clone().requires_grad_(True)
    for p in m.parameters():
        p.grad = None
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        out = torch.utils.checkpoint.checkpoint(m, x, use_reentrant=False) if checkpoint else m(x)
    w = torch.randn(out.shape, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    (out.double() * w).sum().backward()
    res = {"out": out.detach(), "x": x.grad}
    res.update({n: (None if p.grad is None else p.grad.clone()) for n, p in m.named_parameters()})
    return res


@pytest.fixture
def hada_path(monkeypatch):
    """hada_path(True / False) forces the factored / dense path; returns the list of calls that reached the choice."""
    calls = []

    def force(value):
        def choose(*args, **kwargs):
            calls.append(kwargs.get("bypass", False))
            return value

        monkeypatch.setattr(laf, "_hada_factored_cheaper", choose)
        monkeypatch.setattr(laf, "_ABBA_DORA_FUSED", False)  # ABBA DoRA per output row: these _hada_* paths too
        return calls

    return force


_HADA_CASES = [
    ("loha", {}), ("loha", {"dora": "output"}), ("loha", {"dora": "input", "mult": 1.3}),
    ("loha", {"scalar": "scalar"}), ("loha", {"scalar": "row_column", "dora": "output"}),
    ("loha", {"scalar": "column", "dora": "input"}), ("loha", {"scalar": "row", "alpha": 4.0, "mult": 1.0}),
    ("abba", {"dim": 6}), ("abba", {"dim": 7, "dora": "output"}), ("abba", {"dim": 6, "dora": "input", "mult": 1.3}),
    ("abba", {"dim": 7, "scalar": True, "dora": "output"}),
]


@pytest.mark.parametrize("algo,kw", _HADA_CASES)
@pytest.mark.parametrize("factored", [True, False])
@pytest.mark.parametrize("shape", [(6, 24), (2, 5, 24), (1, 1, 3, 4, 24)])
def test_hada_rebuild_paths_match_stock(algo, kw, factored, shape, hada_path):
    """Rebuild mode (bypass_mode=False): the factored path (x W0^T + (x KA^T) KB^T, DoRA norm from the factors) equals the
    stock forward (dense delta) in fp64 incl. every gradient, with and without gradient checkpointing; the dense path is
    bit-identical to it."""
    x = torch.randn(shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    ref = _with_stock(lambda: _hada_run(_hada_module(algo, **kw), x))
    calls = hada_path(factored)
    for ckpt in (False, True):
        got = _hada_run(_hada_module(algo, **kw), x, checkpoint=ckpt)
        if factored:
            _assert_same(got, ref)
        else:
            assert ref.keys() == got.keys() and all(
                (ref[k] is None and got[k] is None) or torch.equal(ref[k], got[k]) for k in ref)
    assert calls, "the flag's forward was not used"


@pytest.mark.parametrize("algo,kw", [("loha", {}), ("loha", {"scalar": "row_column"}), ("abba", {"dim": 6}),
                                     ("abba", {"dim": 7, "scalar": True})])
def test_hada_bypass_factored_matches_dense(algo, kw, hada_path):
    """Bypass mode: (x KA^T) KB^T == the old flag's dense diff weight (LoHa) / output scaling (ABBA), fp64."""
    x = torch.randn(2, 5, 24, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    if algo == "abba":
        ref = _with_stock(lambda: _hada_run(_hada_module(algo, bypass=True, **kw), x))  # stock: one call is correct
    else:
        hada_path(False)
        ref = _hada_run(_hada_module(algo, bypass=True, **kw), x)
    calls = hada_path(True)
    n0 = len(calls)
    got = _hada_run(_hada_module(algo, bypass=True, **kw), x, checkpoint=True)
    _assert_same(got, ref)
    assert algo == "abba" or calls[n0:] == [True, True]  # forward + checkpoint recompute, bypass cost rule


def test_hada_fallbacks_use_stock_forward(hada_path):
    """Conv layers and rank dropout in training keep the stock forward."""
    calls = hada_path(True)
    model = make_model(conv=True)
    net = make_net(model, "loha", 4, 4)
    convs = [m for m in loras(net) if m.module_type != "linear"]
    assert convs
    with torch.no_grad():
        convs[0](torch.randn(1, 8, 6, 6, dtype=torch.bfloat16))
    assert not calls
    m = _hada_module("loha")
    m.rank_dropout = 0.1
    m(torch.randn(3, 24, dtype=torch.float64))
    assert not calls
    m.eval()  # rank dropout only acts in training
    m(torch.randn(3, 24, dtype=torch.float64))
    assert calls == [False]


@pytest.mark.parametrize("algo,dora", [("loha", None), ("loha", "output"), ("abba", None), ("abba", "output"),
                                       ("gora", "output"), ("ralora", "output"), ("lora2", "output")])
def test_fp8_base_keeps_the_stock_path(algo, dora):
    """train_network.py --fp8_base (the other trainers; Anima turns it off): float8 base weights and bias. The stock
    rebuild forward casts them with .to(); the factored / dense hada paths and the DoRA norm from the factors promoted
    float8 (RuntimeError under bf16 autocast). They fall back to stock: the same result bit for bit."""
    def run():
        if algo in ("loha", "abba"):
            m = _hada_module(algo, dora=dora, dtype=torch.float32, base_dtype=torch.float8_e4m3fn)
        else:
            m = _lf_module(algo, dora, n_split=4 if algo == "ralora" else 1, dtype=torch.float32)
            m.org_module[0].to(torch.float8_e4m3fn)
        x = torch.randn(2, 5, 24, generator=torch.Generator().manual_seed(2)).bfloat16()
        return _hada_run(m, x, checkpoint=True, autocast=True)

    ref = _with_stock(run)
    _assert_same(run(), ref, rtol=0, atol=0)


def test_hada_bf16_not_less_accurate(hada_path):
    """bf16 autocast, bf16 base, fp32 network (the trainer's setup) vs an fp64 reference of the same module: the
    factored path's error is no larger than the stock dense path's (5% slack for the sample) for the output, the input
    gradient and each adapter gradient (LoHa's adapter gradients come from fp32 products like its HadaWeight backward)."""
    worse = []
    for algo, kw in (("loha", {}), ("loha", {"dora": "output"}), ("abba", {"dim": 8, "dora": "output"}),
                     ("loha", {"dora": "input"})):
        errs = {"stock": {}, "new": {}}
        for seed in range(3):
            x = torch.randn(2, 32, 96, generator=torch.Generator().manual_seed(seed + 10)).bfloat16()
            extra = {k: v for k, v in kw.items() if k != "dim"}

            def mk(dtype):
                return _hada_module(algo, inn=96, out=80, dim=kw.get("dim", 8), alpha=4.0, seed=seed, dtype=dtype,
                                    base_dtype=torch.bfloat16, **extra)

            ref_m = mk(torch.float32)
            ref_m.org_module[0].to(torch.float64)  # the bf16 base values, in fp64
            ref_m.to(torch.float64)
            ref = _hada_run(ref_m, x.double())
            stock = _with_stock(lambda: _hada_run(mk(torch.float32), x, checkpoint=True, autocast=True))
            hada_path(True)
            new = _hada_run(mk(torch.float32), x, checkpoint=True, autocast=True)
            for k in ref:
                if ref[k] is not None:
                    for name, res in (("stock", stock), ("new", new)):
                        errs[name].setdefault(k, []).append(rel(res[k].double(), ref[k], ref[k]))
        for k in errs["stock"]:
            e_old, e_new = sum(errs["stock"][k]) / 3, sum(errs["new"][k]) / 3
            if e_new > 1.05 * e_old:
                worse.append((algo, kw, k, e_old, e_new))
    assert not worse, worse


def test_hada_no_host_sync_and_no_merged_weight_matmul(hada_path):
    """LoHa's stock get_weight makes torch.tensor(scale, device=cuda) per call (a blocking host-to-device copy): the
    flag's paths never call it, the factored path creates no tensor from Python data at all. Gradient checkpointing with
    the module alone in the region: the factored backward runs one base-sized matmul (the input gradient), no merged-weight
    gradient, and the recompute skips the base product (early stop) except for DoRA on the output axis (its scale
    gradient needs the unscaled product, as the stock path needs the merged weight's gradient)."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Watch(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.lift = 0
            self.big = 0

        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            name = func.overloadpacket.__name__
            self.lift += name == "lift_fresh"
            if name in ("mm", "addmm"):
                a, b = (args[0], args[1]) if name == "mm" else (args[1], args[2])
                self.big += a.shape[0] * a.shape[1] * b.shape[1] == 30 * 24 * 20
            return func(*args, **(kwargs or {}))

    def no_get_weight(self, shape):
        raise AssertionError("stock get_weight called")

    for factored in (True, False):
        hada_path(factored)
        for dora in (None, "output"):
            m = _hada_module("loha", dora=dora, dtype=torch.float32)
            m.get_weight = types.MethodType(no_get_weight, m)
            x = torch.randn(30, 24, requires_grad=True)
            w = Watch()
            with w:
                y = torch.utils.checkpoint.checkpoint(m, x, use_reentrant=False)
                n_fwd = w.big
                y.sum().backward()
            if factored:
                assert w.lift == 0, (dora, w.lift)
                assert (n_fwd, w.big - n_fwd) == (1, 2 if dora else 1), (dora, n_fwd, w.big)


@pytest.mark.parametrize("algo,wd,shape,tokens,want", [
    ("abba", True, (2048, 2048), 4096, True), ("abba", False, (2048, 8192), 4096, True),
    ("loha", True, (2048, 2048), 4096, False), ("loha", True, (2048, 8192), 4096, True),
    ("loha", True, (2048, 2048), 512, True), ("loha", False, (2048, 2048), 16384, False),
])
def test_hada_cost_rule(algo, wd, shape, tokens, want):
    """The path choice at Anima shapes (dim 16: LoHa rank 256 with fp32 adapter gradients, ABBA rank 64) by the RTX 5090
    roofline estimate: ABBA factored; LoHa factored at few tokens and on MLP layers, dense on square layers at ~4k
    tokens with DoRA and at very many tokens."""
    out_f, in_f = shape
    m = types.SimpleNamespace(shape=(out_f, in_f), wd_on_output=True)
    x = torch.empty(1, tokens, in_f, device="meta")
    rank = 256 if algo == "loha" else 64
    assert laf._hada_factored_cheaper(m, x, algo, rank, wd, algo == "loha") is want


@pytest.mark.parametrize("algo,kw", [("loha", {"dora_wd": True, "wd_on_output": True}), ("loha", {}),
                                     ("abba", {"dora_wd": True, "wd_on_output": True}), ("loha", {"bypass_mode": True})])
def test_hada_network_training_and_saves_match_dense(tmp_path, algo, kw, hada_path):
    """Through the real network path (create_network, after_network_created, prepare_optimizer_params): 3 AdamW steps
    with gradient checkpointing, fp32 base and network, dense vs factored: same parameters (fp32 rounding), same
    state_dict layout (--resume), and the same weights give the same saved files (fp32 and fp16)."""
    from safetensors import safe_open

    def trained(factored):
        hada_path(factored)
        model = make_model(dtype=torch.float32)
        torch.manual_seed(1)
        net = make_net(model, algo, 8, 4, **kw)
        for m in loras(net):
            with torch.no_grad():
                for n, p in m.named_parameters():
                    if n in ("hada_w2_a", "lora_up2.weight"):
                        p.normal_(0, 0.05, generator=torch.Generator().manual_seed(len(n)))
        net.train()
        opt = build_optimizer(net, 1e-3)
        for step in range(3):
            b = {k: v.float() for k, v in make_batch(100 + step).items()}
            h, _ = model(b["x"], b["c"], None, checkpoint=True)
            F.mse_loss(h, b["y"]).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        return net

    dense, fact = trained(False), trained(True)
    assert [(k, v.shape) for k, v in dense.state_dict().items()] == [(k, v.shape) for k, v in fact.state_dict().items()]
    pd, pf = dict(dense.named_parameters()), dict(fact.named_parameters())
    for n in pd:
        assert rel(pf[n], pd[n], pd[n]) < 1e-4, n
    with torch.no_grad():
        for n, p in fact.named_parameters():
            p.copy_(pd[n])
    for dtype in (torch.float32, torch.float16):
        paths = []
        for name, net in (("dense", dense), ("fact", fact)):
            paths.append(str(tmp_path / f"{name}_{str(dtype)[6:]}.safetensors"))
            net.save_weights(paths[-1], dtype, {"ss_x": "1"})
        a, b = (safe_open(p, "pt") for p in paths)
        assert a.metadata() == b.metadata() and list(a.keys()) == list(b.keys())
        assert all(torch.equal(a.get_tensor(k), b.get_tensor(k)) for k in a.keys())


# --------------------------------------------------------------- DoRA per output row: the two fused functions, path rule


def _spy_apply(monkeypatch, cls, used):
    orig = cls.apply

    def spy(*args):
        used.append(cls.__name__)
        return orig(*args)

    monkeypatch.setattr(cls, "apply", spy)


@pytest.mark.parametrize("kw", [{"dim": 7}, {"dim": 6, "scalar": True, "mult": 1.3}, {"dim": 7, "mult": 1.0}])
@pytest.mark.parametrize("shape", [(6, 24), (1, 1, 3, 4, 24)])
def test_abba_dora_dense_fused_matches_stock(kw, shape, monkeypatch):
    """ABBA with DoRA per output row, where _lf_dora_dense picks the dense form (CUDA, not very wide inputs), runs
    _DenseDoRAOut (B = KB with scalar, scale and multiplier in it, A = KA, s = 1): == the stock rebuild
    (abba.py:416-427) in fp64, output and every gradient, with and without gradient checkpointing. The factored choice
    keeps the _hada_* paths (test_hada_rebuild_paths_match_stock)."""
    monkeypatch.setattr(laf, "_LF_DORA_PATH", "dense")
    used = []
    _spy_apply(monkeypatch, laf._DenseDoRAOut, used)
    x = torch.randn(shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    ref = _with_stock(lambda: _hada_run(_hada_module("abba", dora="output", **kw), x))
    assert not used
    for ckpt in (False, True):
        _assert_same(_hada_run(_hada_module("abba", dora="output", **kw), x, checkpoint=ckpt), ref)
    assert len(used) == 3  # forward, forward + checkpoint recompute


@pytest.mark.parametrize("base", [torch.bfloat16, torch.float32])
def test_abba_dora_dense_fused_bf16_not_less_accurate(base, monkeypatch):
    """bf16 autocast, bf16 (Anima) or fp32 base, fp32 network: ABBA DoRA per output row through _DenseDoRAOut is no
    less accurate than the stock rebuild vs an fp64 reference of the same module (5% slack), output, input and every
    gradient. The base goes in as the stock merge sees it: an fp32 base rounded to bf16 before W' (rounded once more
    inside) made the output ~20% and dora_scale ~25% less accurate. (Through _LowRankDoRA, the dora_scale gradient from
    sum g * (y - C) was ~16% less accurate here than the stock's: that is why the factored choice for ABBA stays
    _HadaDoRAOut, which keeps the unscaled product.)"""
    monkeypatch.setattr(laf, "_LF_DORA_PATH", "dense")
    used = []
    _spy_apply(monkeypatch, laf._DenseDoRAOut, used)
    errs = {"stock": {}, "new": {}}
    for seed in range(3):
        x = torch.randn(2, 32, 96, generator=torch.Generator().manual_seed(seed + 10)).bfloat16()

        def mk(dtype):
            return _hada_module("abba", inn=96, out=80, dim=8, alpha=4.0, seed=seed, dtype=dtype, dora="output",
                                base_dtype=base)

        ref_m = mk(torch.float32)
        ref_m.org_module[0].to(torch.float64)
        ref_m.to(torch.float64)
        ref = _hada_run(ref_m, x.double())
        stock = _with_stock(lambda: _hada_run(mk(torch.float32), x, checkpoint=True, autocast=True))
        n0 = len(used)
        new = _hada_run(mk(torch.float32), x, checkpoint=True, autocast=True)
        assert len(used) == n0 + 2  # this route: forward + checkpoint recompute
        for k in ref:
            if ref[k] is not None:
                for name, res in (("stock", stock), ("new", new)):
                    errs[name].setdefault(k, []).append(rel(res[k].double(), ref[k], ref[k]))
    worse = [(k, sum(errs["stock"][k]) / 3, sum(errs["new"][k]) / 3) for k in errs["stock"]
             if sum(errs["new"][k]) > 1.05 * sum(errs["stock"][k])]
    assert not worse, worse


@pytest.mark.parametrize("algo", ["gora", "lora2", "ralora_blocks", "abba"])
@pytest.mark.parametrize("dora", [None, "output", "input"])
@pytest.mark.parametrize("path", [None, "dense", "factored"])
def test_dora_output_routes_to_the_chosen_function(algo, dora, path, monkeypatch):
    """Which function a rebuild-mode call reaches: DoRA per output row (GoRA / LoRA² / RaLoRA n == 1) the one
    _lf_dora_dense picks (CPU default: factored), ABBA _DenseDoRAOut when that picks dense and its _hada_* paths
    otherwise, RaLoRA blocks with DoRA the split _LowRankLinear form, DoRA per input column neither (the stock dense
    rebuild), no DoRA the plain factored GEMMs."""
    monkeypatch.setattr(laf, "_LF_DORA_PATH", path)
    used = []
    for cls in (laf._LowRankDoRA, laf._DenseDoRAOut, laf._LowRankLinear, laf._HadaDoRAOut):
        _spy_apply(monkeypatch, cls, used)
    if algo == "abba":
        m = _hada_module("abba", dora=dora, dim=6)
    else:
        m = _lf_module(algo.split("_")[0], dora, n_split=4 if algo == "ralora_blocks" else 1)
    m(torch.randn(2, 5, 24, dtype=torch.float64)).sum().backward()
    fused = "_DenseDoRAOut" if path == "dense" else "_LowRankDoRA"
    if algo == "abba":
        assert "_LowRankDoRA" not in used and "_LowRankLinear" not in used
        assert ("_DenseDoRAOut" in used) == (dora == "output" and path == "dense"), used
        return
    if dora == "output":
        want = ["_LowRankLinear"] if algo == "ralora_blocks" else [fused]
    elif dora == "input":
        want = []
    else:
        want = ["_LowRankLinear"]
    assert used == want, used


@pytest.mark.parametrize("tokens,out_f,in_f,rank,want", [
    (4096, 2048, 2048, 16, True), (4096, 8192, 2048, 16, True), (4096, 2048, 8192, 16, False),
    (512, 2048, 1024, 16, True), (1, 256, 2048, 16, True), (1, 6144, 256, 16, True),
    (4096, 2048, 2048, 64, True), (4096, 2048, 8192, 64, False), (4096, 2048, 8192, 8, False),
])
def test_lf_dora_cost_rule(tokens, out_f, in_f, rank, want, monkeypatch):
    """The DoRA path choice at Anima's shapes under bf16 autocast on CUDA (x [1, 4096, 2048] self / cross attention,
    MLP 2048 -> 8192 -> 2048 on 4096 tokens, cross-attention k / v from 512 text tokens, adaLN 2048 -> 256 -> 6144 on 1
    token; GoRA ranks 8-64): dense everywhere but the MLP's second layer, where the dense path's merged-weight-gradient
    GEMM and fp32 weight passes (8192 wide input) cost more than the factored path's extra launches. On the CPU always
    factored; _LF_DORA_PATH overrides both."""
    cuda, cpu = torch.device("cuda"), torch.device("cpu")  # only the device type is read
    assert laf._lf_dora_dense(tokens, out_f, in_f, rank, torch.bfloat16, cuda) is want
    assert laf._lf_dora_dense(tokens, out_f, in_f, rank, torch.bfloat16, cpu) is False
    monkeypatch.setattr(laf, "_LF_DORA_PATH", "dense")
    assert laf._lf_dora_dense(tokens, out_f, in_f, rank, torch.bfloat16, cpu) is True
    monkeypatch.setattr(laf, "_LF_DORA_PATH", "factored")
    assert laf._lf_dora_dense(tokens, out_f, in_f, rank, torch.bfloat16, cuda) is False


# ------------------------------------------------------------------------------------------------- more stock reproductions


def test_stock_gora_init_and_grad_leak():
    def run():
        model = make_model()
        net = make_net(model, "gora", 4, 4, gora_min_rank="2", gora_max_rank="8")
        batches = _prepass_batches()
        G = _base_grad_mean(model, net, batches)
        net.prepare_gora(dataloader=batches, forward_fn=lambda b: loss_fn(model, b), max_steps=3, adaptive_n=False)
        m, g = loras(net)[0], G[0]
        A0 = m.lora_up.weight.detach().double()
        xi = 0.05 * math.sqrt(A0.shape[0]) / 4.0
        ratio = float((A0 @ m.lora_down.weight.detach().double()).norm()
                      / (xi * (A0 @ torch.linalg.pinv(A0) @ g.double()).norm()))
        return ratio, laf._org_weight(m).grad is not None

    ratio, leaked = _with_stock(run)
    assert abs(ratio - 1 / 3) < 0.02 and leaked  # init N times too small, base .grad kept


def test_stock_lora2_string_target_crash():
    from lycoris import kohya

    def build():  # the TOML string goes to LyCORIS as is, without prepare_network_kwargs
        return kohya.create_network(1.0, 8, 8, None, [], make_model(), preset=dict(PRESET), algo="lora2",
                                    lora2_nu_target="6")

    with pytest.raises(TypeError):
        _with_stock(build)


# ------------------------------------------------------------------------------- gradient checkpointing: same gradients


@pytest.mark.parametrize("algo,extra", [("tlora", {}), ("tsm", {"tsm_n_scales": "4,1"}), ("tsm", {"tsm_n_scales": "4,1", "tsm_stage": 2})])
def test_checkpointing_gives_the_same_gradients(algo, extra):
    ts = torch.tensor([100.0, 350.0, 600.0, 900.0])
    grads = []
    for ckpt in (False, True):
        model = make_model()
        net = make_net(model, algo, 4, 4, **extra)
        net.train()
        for m in loras(net):  # non-zero experts/up so every path carries signal
            for n, p in m.named_parameters():
                if n.endswith("up.weight") or "router_fc" in n or "timestep_embed" in n:
                    with torch.no_grad():
                        p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(len(n))) * 0.05)
        laf.set_timestep(net, ts)
        loss_fn(model, make_batch(5, B=4), checkpoint=ckpt).backward()
        laf.clear_timestep(net)
        grads.append({n: p.grad.detach().clone() for n, p in net.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys() and grads[0]
    for n in grads[0]:
        assert torch.allclose(grads[0][n], grads[1][n], rtol=1e-3, atol=1e-6), n


# ===================================================================================== LoRA/LoCon, OrthoLoRA, DyLoRA, PiSSA


def _forge_strength(net, path, strength, base=None, lycoris_dora=False):
    """Forge at `strength` (fp32 merge) vs base + strength * (trained - base): max over modules, relative to the change.
    base: {lora_name: weight the file applies to} (PiSSA changed the base in place). lycoris_dora: Forge's LoRA delta,
    then LyCORIS' output-axis DoRA norm (sam-extra "lycoris" mode; stock Forge divides by ||W0||), strength 1 only."""
    from safetensors.torch import load_file

    sd = load_file(path)
    key_map, patch, remaining = forge_patches(net, sd)
    assert not remaining, remaining[:5]
    if lycoris_dora:
        _, patch, _ = forge_patches(net, {k: v for k, v in sd.items() if not k.endswith(".dora_scale")})
    worst = 0.0
    for m in loras(net):
        key = key_map[m.lora_name]
        W0 = (base or {}).get(m.lora_name, m.org_module[0].weight).detach().float()
        trained = m.get_merged_weight(multiplier=1)[0].detach().float().reshape(W0.shape)
        res = patch[key].calculate_weight(W0.clone(), key, strength, 1.0, None, lambda a: a, torch.float32)
        if lycoris_dora:
            ds = sd[m.lora_name + ".dora_scale"].float()
            res = res * (ds / (torch.linalg.vector_norm(res, dim=1, keepdim=True) + torch.finfo(torch.float32).eps))
        worst = max(worst, rel(res, W0 + strength * (trained - W0), strength * (trained - W0)))
    return worst


def _fill(net, std=0.05, seed=0):
    """Non-zero factors so every parameter carries signal (LoRA up / DyLoRA up blocks / OrthoLoRA S, lambda)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in net.named_parameters():
            if n.endswith(("lora_up.weight", "S_p", "S_q", "lambda_layer")) or ".up_list." in n:
                p.copy_(torch.randn(p.shape, generator=g) * std)


def _grads(net):
    return {n: p.grad.detach().clone() for n, p in net.named_parameters() if p.grad is not None}


def _same_grads(a, b, rtol, atol):
    assert a.keys() == b.keys() and a
    for n in a:
        assert torch.allclose(a[n], b[n], rtol=rtol, atol=atol), (n, float((a[n] - b[n]).abs().max()))


_LORA_INPUTS = {"2d": (6, 32), "3d": (2, 5, 32), "5d": (2, 1, 3, 4, 32)}


def _single(cls, dtype=torch.float64, out_f=48, in_f=32, dim=8, alpha=4, seed=1, **kw):
    """One module on a Linear whose base weight holds bf16 values (in `dtype`), fp32/fp64 module, non-zero factors."""
    torch.manual_seed(seed)
    lin = nn.Linear(in_f, out_f)
    lin.weight.data = lin.weight.data.bfloat16().to(dtype)
    lin.bias.data = lin.bias.data.bfloat16().to(dtype)
    lin.requires_grad_(False)
    m = cls("t", lin, 1.0, dim, alpha, **kw)
    if dtype == torch.float64:
        m.double()
    m.apply_to()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("lora_up.weight") or n.startswith("up_list"):
                p.copy_((torch.randn(p.shape, generator=g) * 0.05).to(p.dtype))
            elif n in ("S_p", "S_q"):
                p.copy_((torch.randn(p.shape, generator=g) * 0.1).to(p.dtype))
            elif n == "lambda_layer":
                p.copy_((torch.randn(p.shape, generator=g) * 0.5).to(p.dtype))
            elif n == "dora_scale":
                p.mul_(1 + 0.05 * torch.randn(p.shape, generator=g).to(p.dtype))
            elif n == "scalar":
                p.fill_(0.7)
    return m, lin


def _fwd_bwd(m, fwd, x, gc, autocast=False):
    for p in m.parameters():
        p.grad = None
    x = x.detach().clone().requires_grad_(True)

    def f(inp):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            return fwd(m, inp)

    y = torch.utils.checkpoint.checkpoint(f, x, use_reentrant=False) if gc else f(x)
    g = torch.randn(y.shape, generator=torch.Generator().manual_seed(5), dtype=torch.float64).to(y.dtype)
    (y.float() * g.float()).sum().backward()
    out = {n: p.grad.detach().double().clone() for n, p in m.named_parameters() if p.grad is not None}
    out["<y>"] = y.detach().double()
    out["<x>"] = x.grad.detach().double()
    return out


def _rel_all(a, b):
    assert a.keys() == b.keys()
    return {k: float((a[k] - b[k]).norm() / b[k].norm()) for k in a}


def _cls(name):
    from lycoris.modules.locon import LoConModule
    from lycoris.modules.ortholora import OrthoLoRAModule

    return {"locon": LoConModule, "ortholora": OrthoLoRAModule}[name]


_FAST_CASES = {
    "lora": ("locon", {}),
    "lora_scalar": ("locon", {"use_scalar": True}),
    "lora_nora": ("locon", {"nora": True}),
    "lora_orthogonalize": ("locon", {"orthogonalize": True}),
    "ortholora": ("ortholora", {}),
}
_BF16_CASES = ["lora", "ortholora"]


@pytest.mark.parametrize("case", sorted(_FAST_CASES))
@pytest.mark.parametrize("shape", sorted(_LORA_INPUTS))
@pytest.mark.parametrize("gc", [False, True])
def test_factored_forward_matches_stock_fp64(case, shape, gc):
    """The flag's LoRA/LoCon and OrthoLoRA training forward is the stock rebuild math: fp64 output and every gradient
    (LoRA parameters, scalar, input) agree to ~1e-12, with checkpointing on and off, 2/3/5-D inputs."""
    cls_name, kw = _FAST_CASES[case]
    cls = _cls(cls_name)
    m, _ = _single(cls, **kw)
    m.train()
    assert laf._original(cls, "forward") is not None and type(m).forward is not laf._original(cls, "forward")
    x = torch.randn(*_LORA_INPUTS[shape], dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    stock = _fwd_bwd(m, lambda mm, i: laf._original(cls, "forward")(mm, i), x, gc)
    fast = _fwd_bwd(m, lambda mm, i: mm(i), x, gc)
    errs = _rel_all(fast, stock)
    assert max(errs.values()) < 1e-12, errs


@pytest.mark.parametrize("case", _BF16_CASES)
def test_factored_forward_bf16_not_worse(case):
    """Under bf16 autocast (fp32 LoRA, bf16 base and input) the flag's forward is at least as close to fp64 as the stock
    forward, averaged over the output and all gradients (6 seeds), and for the output alone. Without autocast (Anima's
    adaLN layers) both compute in the fp32 module dtype."""
    cls_name, kw = _FAST_CASES[case]
    cls = _cls(cls_name)
    stock_fwd = laf._original(cls, "forward")
    es, ef = {}, {}
    for seed in range(6):
        m, _ = _single(cls, dtype=torch.bfloat16, out_f=256, in_f=128, dim=16, alpha=8, seed=seed, **kw)
        m.train()
        x = torch.randn(2, 64, 128, generator=torch.Generator().manual_seed(seed + 10)).bfloat16()
        ref_m = copy.deepcopy(m)
        ref_m.double()
        ref_m.org_module[0].double()
        ref = _fwd_bwd(ref_m, lambda mm, i: stock_fwd(mm, i), x.double(), False)
        stock = _fwd_bwd(m, lambda mm, i: stock_fwd(mm, i), x, True, autocast=True)
        fast = _fwd_bwd(m, lambda mm, i: mm(i), x, True, autocast=True)
        assert stock["<y>"].shape == fast["<y>"].shape
        for k, v in _rel_all(stock, ref).items():
            es[k] = es.get(k, 0.0) + v / 6
        for k, v in _rel_all(fast, ref).items():
            ef[k] = ef.get(k, 0.0) + v / 6
    mean_s, mean_f = sum(es.values()) / len(es), sum(ef.values()) / len(ef)
    print(f"[bf16 vs fp64] {case}: mean stock {mean_s:.3e} flag {mean_f:.3e} | "
          + " ".join(f"{k} {es[k]:.2e}/{ef[k]:.2e}" for k in es))
    assert mean_f <= mean_s and ef["<y>"] <= es["<y>"], (es, ef)
    m, _ = _single(cls, dtype=torch.bfloat16, seed=4, **kw)  # adaLN: autocast off, bf16 input
    m.train()
    x = torch.randn(3, 32, generator=torch.Generator().manual_seed(1)).bfloat16()
    with torch.no_grad():
        assert m(x).dtype == stock_fwd(m, x).dtype == torch.float32
    stock = _fwd_bwd(m, lambda mm, i: stock_fwd(mm, i), x, True)
    fast = _fwd_bwd(m, lambda mm, i: mm(i), x, True)
    tol = 2e-2 if cls_name == "ortholora" else 1e-4  # OrthoLoRA's stock delta is a bf16 product (ortholora.py:217)
    assert max(_rel_all(fast, stock).values()) < tol


def test_factored_forward_fallbacks():
    """DoRA (both axes; its magnitude gradient would need the base matmul again under checkpointing), GoRA/RaLoRA
    (subclasses) and conv keep the stock forward."""
    from lycoris.modules.locon import GoRAModule, LoConModule, RaLoRAModule
    from lycoris.modules.ortholora import OrthoLoRAModule

    for cls, kw in [(LoConModule, {"wd_on_output": True}), (LoConModule, {"wd_on_output": False}), (OrthoLoRAModule, {})]:
        m, _ = _single(cls, dtype=torch.float32, weight_decompose=True, **kw)
        m.train()
        x = torch.randn(2, 5, 32)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert torch.equal(m(x), laf._original(cls, "forward")(m, x))
        assert m.__dict__["_laf_fast"] is False
    for cls in (GoRAModule, RaLoRAModule):
        assert laf._locon_fast_ok(cls("t", nn.Linear(32, 48).requires_grad_(False), 1.0, 4, 4)) is False
    assert laf._locon_fast_ok(LoConModule("t", nn.Conv2d(8, 16, 3).requires_grad_(False), 1.0, 4, 4)) is False


@pytest.mark.parametrize("algo", ["lora", "locon"])
@pytest.mark.parametrize("dora", [None, "output", "input"])
def test_lora_locon(tmp_path, algo, dora):
    kw = {"dora_wd": True, "wd_on_output": dora == "output"} if dora else {}
    model = make_model(conv=True)
    net = make_net(model, algo, 8, 4, conv_dim=4, conv_alpha=2, **kw)
    _fill(net)
    no_grad, unchanged = train(model, net, conv=True, checkpoint=True)
    assert not no_grad and not unchanged, (no_grad, unchanged)
    to_fp32_eval(model, net)
    p32, p16 = save_both(net, tmp_path, algo)
    assert metadata(p32)[laf.METADATA_FORMAT_KEY] == "lora"
    check_forge(net, p32, dora_axis=dora)
    check_forge(net, p16, dora_axis=dora, tol=5e-2)
    if not dora:
        assert _forge_strength(net, p32, 0.5) < 1e-5
    ref = model_out(model, conv=True)
    for load in ("state", "weights"):  # --resume (accelerate, strict) and --network_weights
        model2 = make_model(conv=True)
        net2 = make_net(model2, algo, 8, 4, conv_dim=4, conv_alpha=2, **kw)
        if load == "state":
            net2.load_state_dict(net.state_dict(), strict=True)
        else:
            assert not net2.load_weights(p32)
        to_fp32_eval(model2, net2)
        assert rel(model_out(model2, conv=True), ref, ref) < 1e-6, load


def test_dylora_stock_bugs():
    def run_bf16():
        model = make_model()
        make_net(model, "dylora", 8, 4)
        return loss_fn(model, make_batch(1))

    with pytest.raises(RuntimeError, match="dtype"):  # adaLN: bf16 input, fp32 merged weight (dylora.py:162-168)
        _with_stock(run_bf16)

    def run_ckpt():
        import random

        model = make_model(dtype=torch.float32)
        make_net(model, "dylora", 16, 4)
        random.seed(0)
        for i in range(8):  # a forward and its recompute draw two random ranks (Python random)
            loss_fn(model, {k: v.float() for k, v in make_batch(i).items()}, checkpoint=True).backward()

    with pytest.raises(torch.utils.checkpoint.CheckpointError):
        _with_stock(run_ckpt)


def test_dylora_checkpoint_recompute_uses_the_forward_rank():
    import random

    grads = []
    for ckpt in (False, True):
        model = make_model()
        net = make_net(model, "dylora", 16, 4)  # 4 blocks
        _fill(net)
        net.train()
        random.seed(123)
        loss_fn(model, make_batch(5), checkpoint=ckpt).backward()
        grads.append(_grads(net))
    _same_grads(grads[0], grads[1], rtol=1e-3, atol=1e-6)
    for m in loras(net):  # only block b of each module trains this step; blocks below are frozen, above unused
        b = m.__dict__["_laf_dylora_b"]
        with_grad = sorted(n for n, p in m.named_parameters() if p.grad is not None)
        assert with_grad == [f"down_list.{b}", f"up_list.{b}"], (m.lora_name, b, with_grad)


def test_dylora(tmp_path):
    from lycoris import kohya

    model = make_model(conv=True)
    net = make_net(model, "dylora", 8, 4, conv_dim=4, conv_alpha=4)
    _fill(net)
    no_grad, unchanged = set(), set()
    for i in range(4):  # one random block per step: over several runs every block trains
        ng, uc = train(model, net, conv=True, checkpoint=True, steps=3)
        no_grad = set(ng) if i == 0 else no_grad & set(ng)
        unchanged = set(uc) if i == 0 else unchanged & set(uc)
    assert not no_grad and not unchanged, (no_grad, unchanged)
    to_fp32_eval(model, net)
    assert torch.equal(model_out(model, conv=True), model_out(model, conv=True))  # not training: all blocks
    p32, p16 = save_both(net, tmp_path, "dylora")
    assert metadata(p32)[laf.METADATA_FORMAT_KEY] == "lora"
    from safetensors.torch import load_file

    m0 = loras(net)[0]
    assert float(load_file(p32)[m0.lora_name + ".alpha"]) == float(m0.alpha) * m0.block_size
    check_forge(net, p32)
    check_forge(net, p16, tol=5e-2)
    assert _forge_strength(net, p32, 0.5) < 1e-5
    ref = model_out(model, conv=True)
    for load in ("state", "weights", "sd-scripts"):
        if load == "sd-scripts":  # create_network_from_weights (merge tools) reads the Forge file as LoCon
            model2 = make_model(conv=True, dtype=torch.float32)
            net2, _ = kohya.create_network_from_weights(1.0, p32, None, None, model2)
            net2.apply_to(None, model2, False, True)
        else:
            model2 = make_model(conv=True)
            net2 = make_net(model2, "dylora", 8, 4, conv_dim=4, conv_alpha=4)
            if load == "state":
                net2.load_state_dict(net.state_dict(), strict=True)
            else:
                assert not net2.load_weights(p32)
                assert float(loras(net2)[0].alpha) == float(m0.alpha)
        to_fp32_eval(model2, net2)
        assert rel(model_out(model2, conv=True), ref, ref) < 1e-6, load


def test_dylora_bypass_applies_scale():
    model = make_model()
    net = make_net(model, "dylora", 8, 4, bypass_mode=True)
    _fill(net)
    assert all(m.bypass_mode for m in loras(net))
    no_grad, unchanged = train(model, net, checkpoint=True, steps=2)
    to_fp32_eval(model, net)
    for m in loras(net):  # bypass delta == rebuild delta (alpha / blocks * multiplier)
        org = m.org_module[0]
        x = torch.randn(3, org.in_features)
        with torch.no_grad():
            W = m.get_merged_weight(multiplier=1)[0]
            assert torch.allclose(m(x), F.linear(x, W, org.bias), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dora", [False, True])
def test_ortholora(tmp_path, dora):
    kw = {"dora_wd": True} if dora else {}
    model = make_model()
    net = make_net(model, "ortholora", 8, 4, **kw)
    _fill(net, std=0.1)
    no_grad, unchanged = train(model, net, checkpoint=True)
    assert not no_grad and not unchanged, (no_grad, unchanged)
    to_fp32_eval(model, net)
    p32, p16 = save_both(net, tmp_path, "ortholora")
    from safetensors.torch import load_file

    sd = load_file(p32)
    assert metadata(p32)[laf.METADATA_FORMAT_KEY] == "lora"
    assert sorted({k.split(".", 1)[1] for k in sd}) == sorted(
        ["alpha", "lora_down.weight", "lora_up.weight"] + (["dora_scale"] if dora else []))
    check_forge(net, p32, dora_axis="output" if dora else None)
    check_forge(net, p16, dora_axis="output" if dora else None, tol=5e-2)
    if not dora:
        assert _forge_strength(net, p32, 0.5) < 1e-5
    ref = model_out(model)
    model2 = make_model()
    net2 = make_net(model2, "ortholora", 8, 4, **kw)  # new random SVD bases: the state carries the trained ones
    net2.load_state_dict(net.state_dict(), strict=True)
    to_fp32_eval(model2, net2)
    assert rel(model_out(model2), ref, ref) < 1e-6
    model3 = make_model()
    net3 = make_net(model3, "ortholora", 8, 4, **kw)
    with pytest.raises(ValueError, match="OrthoLoRA"):
        net3.load_weights(p32)


@pytest.mark.parametrize("dora", [False, True])
def test_ortholora_cayley_solve_no_host_sync(dora):
    """The effective bases come from linalg.solve_ex: bit for bit the stock linalg.solve result and its gradients, and
    the flag's forward + backward (checkpointed; with DoRA the stock forward over the same bases) make no host sync,
    where linalg.solve's error check read the LU info back on every call (two blocking syncs per module and step)."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from lycoris.modules.ortholora import OrthoLoRAModule

    class Watch(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.hits = []

        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            if func.overloadpacket.__name__ in ("_local_scalar_dense", "_linalg_check_errors", "nonzero", "lift_fresh"):
                self.hits.append(func.overloadpacket.__name__)
            return func(*args, **(kwargs or {}))

    m, _ = _single(OrthoLoRAModule, dtype=torch.float32, **({"weight_decompose": True} if dora else {}))
    m.train()
    stock = laf._original(OrthoLoRAModule, "_compute_effective_bases")
    assert type(m)._compute_effective_bases is laf._ortho_effective_bases
    res = []
    for fn in (stock, laf._ortho_effective_bases):
        m.zero_grad(set_to_none=True)
        q, p = fn(m)
        ((q.float() * 1.3).sum() + (p.float() ** 2).sum()).backward()
        res.append((q, p, m.S_q.grad.clone(), m.S_p.grad.clone()))
    assert all(torch.equal(a, b) for a, b in zip(*res))
    w = Watch()
    with w:
        stock(m)
    assert w.hits  # the stock solve's error check (on CUDA it reads the info back: a sync)
    x = torch.randn(2, 5, 32)
    m(x)  # warm-up
    w = Watch()
    with w:
        torch.utils.checkpoint.checkpoint(m, x.clone().requires_grad_(), use_reentrant=False).sum().backward()
    assert not w.hits, w.hits


def test_ortholora_input_axis_dora_raises():
    with pytest.raises(ValueError, match="wd_on_output"):
        laf.prepare_network_kwargs("ortholora", {"algo": "ortholora", "dora_wd": "True", "wd_on_output": "False"}, LOG)


@pytest.mark.parametrize("alpha,dora", [(8, False), (4, False), (8, True)])
def test_pissa(tmp_path, alpha, dora):
    kw = {"dora_wd": True} if dora else {}
    model = make_model()
    original = {mod: mod.weight.detach().clone() for mod in model.modules() if isinstance(mod, nn.Linear)}
    net = make_net(model, "lora", 8, alpha, init_weights="pissa", **kw)
    base = {m.lora_name: original[m.org_module[0]].float() for m in loras(net)}
    # the model is unchanged at step 0 up to the bf16 residual (stock with alpha != dim: 0.4 in the 10-24 Anima run)
    with torch.no_grad():
        b = make_batch(7)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            h0, _ = make_model()(b["x"], b["c"])
            h1, _ = model(b["x"], b["c"])
    assert rel(h1.float(), h0.float(), h0.float()) < 5e-2
    no_grad, unchanged = train(model, net, checkpoint=True)
    assert not no_grad and not unchanged, (no_grad, unchanged)
    to_fp32_eval(model, net)
    p32, p16 = save_both(net, tmp_path, "pissa")
    from safetensors.torch import load_file

    sd = load_file(p32)
    assert not any(k.endswith(("pissa_converted", "pissa_A_init", "pissa_B_init")) for k in sd)
    assert all(v.shape[0] == 16 for k, v in sd.items() if k.endswith("lora_down.weight"))
    # The file applies to the original base: on residual + removed component it gives the trained weights exactly; on
    # the original bf16 weights it differs by the bf16 rounding of the stored residual (printed)
    removed = {m.lora_name: m.org_module[0].weight.detach().float()
               + float(m.scale) * (m.pissa_A_init.float() @ m.pissa_B_init.float()).reshape(m.shape) for m in loras(net)}
    for strength in ((1.0,) if dora else (1.0, 0.5)):
        assert _forge_strength(net, p32, strength, base=removed, lycoris_dora=dora) < 1e-5, strength
    removed_vs_original = max(rel(removed[k], base[k], base[k]) for k in base)
    print(f"[pissa] alpha={alpha} dora={dora}: Forge on the original base vs trained "
          f"{_forge_strength(net, p32, 1.0, base=base, lycoris_dora=dora):.3g} of the change "
          f"(|residual + removed - original| / |original| = {removed_vs_original:.2e})")
    assert removed_vs_original < 1e-2
    ref = model_out(model)
    model2 = make_model()
    net2 = make_net(model2, "lora", 8, alpha, init_weights="pissa", **kw)
    net2.load_state_dict(net.state_dict(), strict=True)  # --resume: native trained + initial factors
    to_fp32_eval(model2, net2)
    assert rel(model_out(model2), ref, ref) < 1e-6
    model3 = make_model()
    net3 = make_net(model3, "lora", 8, alpha, init_weights="pissa", **kw)
    with pytest.raises(ValueError, match="PiSSA"):
        net3.load_weights(p32)


# --------------------------------------------------------------------------------------------- LoKr / GLoRA / IA3 helpers


def _lke_single(algo, out=48, inp=32, dtype=torch.float64, base_dtype=None, seed=0, bias=False, **kw):
    """One LyCORIS module on a Linear (not applied): base weights are bf16 values in base_dtype, random factors."""
    from lycoris.modules.glora import GLoRAModule
    from lycoris.modules.ia3 import IA3Module
    from lycoris.modules.lokr import LokrModule

    torch.manual_seed(seed)
    lin = nn.Linear(inp, out, bias=bias).to(torch.bfloat16).to(base_dtype or dtype)
    for p in lin.parameters():
        p.requires_grad_(False)
    mult = kw.pop("multiplier", 1.0)
    if algo == "glora":
        m = GLoRAModule("m", lin, mult, lora_dim=kw.pop("dim", 8), alpha=kw.pop("alpha", 4), **kw)
    elif algo == "ia3":
        m = IA3Module("m", lin, mult, **kw)
    else:
        m = LokrModule("m", lin, mult, lora_dim=10000, alpha=10000, factor=4, full_matrix=True, weight_decompose=True,
                       wd_on_output=True, **kw)
    m.to(dtype)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n == "scalar":
                p.fill_(0.7)
            elif n == "dora_scale":
                p.mul_(1 + 0.1 * torch.randn(p.shape, generator=g, dtype=torch.float64).to(p.dtype))
            else:
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64).to(p.dtype) * 0.05)
    return m, lin


def _lke_fwd_bwd(m, x, gout, ckpt=False, autocast=False):
    x = x.detach().clone().requires_grad_(True)
    for p in m.parameters():
        p.grad = None
    ctx = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else torch.autocast("cpu", enabled=False)
    with ctx:
        y = torch.utils.checkpoint.checkpoint(m.forward, x, use_reentrant=False) if ckpt else m.forward(x)
    (y.double() * gout).sum().backward()
    grads = {n: p.grad.detach().double().clone() for n, p in m.named_parameters() if p.requires_grad}
    return y.detach(), grads, x.grad.detach()


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / max(float(b.norm()), 1e-300))


# --------------------------------------------------------------------------------------------------------------- GLoRA


@pytest.mark.parametrize("shape", [(6, 32), (2, 5, 32), (2, 1, 3, 2, 32)])
@pytest.mark.parametrize("ckpt", [False, True])
@pytest.mark.parametrize("variant", [{}, {"use_scalar": True}, {"multiplier": 0.5, "bias": True}, {"rs_lora": True},
                                     {"out": 32, "alpha": 8}])
def test_glora_dense_forward_matches_stock_exact_math(shape, ckpt, variant):
    """fp64 / fp32: same output, same input and factor gradients as stock (factor gradients by other matmuls)."""
    for dtype, tol in ((torch.float64, 1e-12), (torch.float32, 1e-5)):
        kw = dict(variant)
        out = kw.pop("out", 48)
        m, _ = _lke_single("glora", out=out, dtype=dtype, **kw)
        m.train()
        g = torch.Generator().manual_seed(5)
        x = torch.randn(shape, generator=g, dtype=torch.float64).to(dtype)
        gout = torch.randn(*shape[:-1], out, generator=g, dtype=torch.float64)
        new = _lke_fwd_bwd(m, x, gout, ckpt)
        stock = _with_stock(lambda: _lke_fwd_bwd(m, x, gout, ckpt))
        assert torch.equal(new[0], stock[0])  # the forward is the stock forward
        assert _rel(new[2], stock[2]) < tol
        assert new[1].keys() == stock[1].keys()
        for n in stock[1]:
            assert _rel(new[1][n], stock[1][n]) < tol, (dtype, n)


@pytest.mark.parametrize("adaln", [False, True])
def test_glora_dense_forward_bf16(adaln):
    """bf16 autocast (and autocast off with a bf16 input, Anima adaLN): output and input gradient bit-identical to
    stock; under autocast the factor gradients are not less accurate than stock against fp64 (stock rounds G = g^T x
    to bf16, the thin matmuls run in fp32)."""
    err_stock, err_new = [], []
    for seed in range(4):
        m32, _ = _lke_single("glora", out=96, inp=64, dtype=torch.float32, base_dtype=torch.bfloat16, seed=seed,
                         use_scalar=True)
        m64, _ = _lke_single("glora", out=96, inp=64, dtype=torch.float64, seed=seed, use_scalar=True)
        g = torch.Generator().manual_seed(100 + seed)
        x = torch.randn(2, 16, 64, generator=g).bfloat16()
        gout = torch.randn(2, 16, 96, generator=g, dtype=torch.float64)
        for m in (m32, m64):
            m.train()
        ref = _lke_fwd_bwd(m64, x.double(), gout)
        new = _lke_fwd_bwd(m32, x, gout, ckpt=True, autocast=not adaln)
        stock = _with_stock(lambda: _lke_fwd_bwd(m32, x, gout, ckpt=True, autocast=not adaln))
        assert torch.equal(new[0], stock[0]) and torch.equal(new[2], stock[2])
        if adaln:
            for n in stock[1]:
                assert _rel(new[1][n], stock[1][n]) < 1e-5, n
        err_stock.append(sum(_rel(stock[1][n], ref[1][n]) for n in ref[1]))
        err_new.append(sum(_rel(new[1][n], ref[1][n]) for n in ref[1]))
    print(f"[glora bf16 adaln={adaln}] factor-gradient error vs fp64: stock {sum(err_stock):.3e}, flag {sum(err_new):.3e}")
    if not adaln:  # with autocast off both are fp32 (~1e-7 of fp64), checked against stock at 1e-5 above
        assert sum(err_new) <= sum(err_stock)


def test_glora_stock_bypass_bugs_and_fix():
    """Stock bypass applies alpha/rank twice (glora.py:298+:350-354), ignores the learnable scalar and puts the base bias
    into bypass_forward_diff; with the flag bypass == rebuild."""
    def outputs():
        reb, lin = _lke_single("glora", bias=True, use_scalar=True)
        byp, _ = _lke_single("glora", bias=True, use_scalar=True, bypass_mode=True)
        byp.load_state_dict(reb.state_dict())  # scalar folded into a2/b2 (glora.py:126-134), byp.scalar reset to 1
        x = torch.randn(5, 32, dtype=torch.float64)
        d = reb.get_diff_weight()[0]
        return reb.forward(x), byp.forward(x), byp.bypass_forward_diff(x), x @ d.T, byp

    y_r, y_b, diff, ref_diff, _ = _with_stock(outputs)
    assert _rel(y_b, y_r) > 1e-2 and _rel(diff, ref_diff) > 1e-2
    y_r, y_b, diff, ref_diff, byp = outputs()
    assert _rel(y_b, y_r) < 1e-12 and _rel(diff, ref_diff) < 1e-12
    byp.train()
    byp.forward(torch.randn(5, 32, dtype=torch.float64)).sum().backward()
    assert byp.scalar.grad is not None and float(byp.scalar.grad.abs()) > 0


@pytest.mark.parametrize("rs_lora", [False, True])
def test_glora(tmp_path, rs_lora):
    kw = {"rs_lora": True} if rs_lora else {}
    model = make_model()
    net = make_net(model, "glora", 8, 4, **kw)
    no_grad, unchanged = train(model, net, checkpoint=True)
    assert not no_grad and not unchanged, (no_grad, unchanged)
    to_fp32_eval(model, net)
    p32, p16 = save_both(net, tmp_path, "glora")
    md = metadata(p32)
    assert md[laf.METADATA_FORMAT_KEY] == "glora"
    from safetensors.torch import load_file

    sd = load_file(p32)
    m0 = loras(net)[0]
    assert math.isclose(float(sd[m0.lora_name + ".alpha"]), float(m0.scale) * 8, rel_tol=1e-6)
    if rs_lora:  # stock saves alpha (4): Forge would apply alpha / rank = 0.5 instead of alpha / sqrt(rank) = 1.41
        assert not math.isclose(float(m0.alpha), float(m0.scale) * 8, rel_tol=1e-3)
    check_forge(net, p32)  # strength 1
    check_forge(net, p16, tol=5e-2)
    key_map, patch, _ = forge_patches(net, sd)
    for m in loras(net):  # strength 0.5: Forge's strength * alpha / rank * delta == LyCORIS multiplier 0.5
        key = key_map[m.lora_name]
        W0 = m.org_module[0].weight.detach().float()
        ref = m.get_merged_weight(multiplier=0.5)[0].float()
        res = patch[key].calculate_weight(W0.clone(), key, 0.5, 1.0, None, lambda a: a, torch.float32)
        assert rel(res, ref, ref - W0) < 1e-5
    from lycoris import kohya

    ref = model_out(model)
    model2 = make_model(dtype=torch.float32)
    net2, _ = kohya.create_network_from_weights(1.0, p32, None, None, model2)
    net2.apply_to(None, model2, False, True)
    net2.eval()
    assert rel(model_out(model2), ref, ref) < 1e-5


# ----------------------------------------------------------------------------------------------------------------- IA3


@pytest.mark.parametrize("on_input", [False, True])
def test_ia3_stock_bypass_scales_bias_and_fix(on_input):
    def outputs():
        reb, _ = _lke_single("ia3", bias=True, train_on_input=on_input)
        byp, _ = _lke_single("ia3", bias=True, train_on_input=on_input, bypass_mode=True)
        byp.load_state_dict(reb.state_dict())
        x = torch.randn(5, 32, dtype=torch.float64)
        d = reb.get_diff_weight()[0]
        return reb.forward(x), byp.forward(x), byp.bypass_forward_diff(x), x @ d.T

    y_r, y_b, diff, ref_diff = _with_stock(outputs)
    if not on_input:
        assert _rel(y_b, y_r) > 1e-4  # (W x + b) (1 + w): the bias is scaled
    assert _rel(diff, ref_diff) > 1e-2
    y_r, y_b, diff, ref_diff = outputs()
    assert _rel(y_b, y_r) < 1e-12 and _rel(diff, ref_diff) < 1e-12


def test_ia3_no_bias_bypass_unchanged():
    reb, _ = _lke_single("ia3", train_on_input=False)
    byp, _ = _lke_single("ia3", train_on_input=False, bypass_mode=True)
    byp.load_state_dict(reb.state_dict())
    x = torch.randn(5, 32, dtype=torch.float64)
    assert torch.equal(byp.forward(x), _with_stock(lambda: byp.forward(x)))


def test_ia3(tmp_path):
    model = make_model()
    net = make_net(model, "ia3", 8, 8)
    no_grad, unchanged = train(model, net, checkpoint=True)
    assert not no_grad and not unchanged, (no_grad, unchanged)
    to_fp32_eval(model, net)
    p32, _ = save_both(net, tmp_path, "ia3")
    assert metadata(p32)[laf.METADATA_FORMAT_KEY] == "ia3-native"
    from safetensors.torch import load_file

    _, patch, remaining = forge_patches(net, load_file(p32))
    assert not patch and remaining  # no IA3 adapter in Forge: kept native, the trainer warns
    # create_network_from_weights passes (weight, on_input): stock make_module_from_state_dict takes only weight
    from lycoris import kohya

    def from_weights():
        model2 = make_model(dtype=torch.float32)
        net2, _ = kohya.create_network_from_weights(1.0, p32, None, None, model2)
        net2.apply_to(None, model2, False, True)
        net2.eval()
        return model2

    with pytest.raises(TypeError):
        _with_stock(from_weights)
    ref = model_out(model)
    assert rel(model_out(from_weights()), ref, ref) < 1e-6


def test_ia3_on_input_from_weights(tmp_path):
    from lycoris.modules.ia3 import IA3Module

    m, lin = _lke_single("ia3", dtype=torch.float32, train_on_input=True)
    m2 = IA3Module.make_module_from_state_dict("m", lin, m.weight.detach(), m.on_input)
    assert m2.train_input and torch.equal(m2.weight, m.weight)
    x = torch.randn(4, 32)
    assert torch.equal(m2.forward(x), m.forward(x))


def test_ia3_preset_matching_nothing_raises():
    from lycoris import kohya

    model = make_model()
    net = kohya.create_network(1.0, 8, 8, None, [], model, preset="ia3", algo="ia3")
    net.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    args = argparse.Namespace(network_args=["algo=ia3", "preset=ia3"], network_dim=8, network_alpha=8,
                              lycoris_algo_fixes=True)
    with pytest.raises(ValueError, match="preset"):
        laf.after_network_created(net, args, LOG)


# ---------------------------------------------------------------------------------------------------------------- LoKr


def test_lokr_dora_multiplier():
    """Stock DoRA LoKr keeps the whole dW at multiplier 0 (lokr.py:851): DOP / iLECO / ADDifT "base" targets are not
    the base model. With the flag: m=0 is the base layer exactly, m in (0, 1) interpolates W0 -> W1, m=1 unchanged."""
    m, lin = _lke_single("lokr", dtype=torch.float32, base_dtype=torch.bfloat16)
    m.eval()
    x = torch.randn(2, 5, 32).bfloat16()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        base = lin(x)
        y1_stock = _with_stock(lambda: m.forward(x))
        y1 = m.forward(x)
        m.multiplier = 0.0
        y0_stock = _with_stock(lambda: m.forward(x))
        y0 = m.forward(x)
        m.multiplier = 0.5
        yh = m.forward(x)
    assert torch.equal(y1, y1_stock)
    assert _rel(y0_stock, base) > 1e-2 and torch.equal(y0, base)
    W0 = lin.weight.double()
    W1 = m.get_merged_weight(multiplier=1)[0].double()
    Wh = m.get_merged_weight(multiplier=0.5)[0].double()
    assert _rel(Wh, W0 + 0.5 * (W1 - W0)) < 1e-6
    assert torch.equal(m.get_merged_weight(multiplier=0)[0].double(), W0)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        ref = F.linear(x, Wh.float())
    assert _rel(yh, ref) < 1e-2


def test_lokr_flag_keeps_fast_forward_at_multiplier_1():
    """With the fork's fast LoKr+DoRA forward installed first (prepare_anima_runtime), the flag's forward hands every
    multiplier-1 call to it unchanged (bit-identical output and gradients)."""
    from library import lycoris_fast
    from lycoris.modules.lokr import LokrModule

    laf._revert_module_patches()
    saved = (LokrModule.__dict__["forward"], lycoris_fast._STOCK_FORWARD)
    try:
        lycoris_fast.install_fast_lokr_forward()
        m, _ = _lke_single("lokr", dtype=torch.float32, base_dtype=torch.bfloat16)
        m.train()
        x = torch.randn(2, 5, 32).bfloat16()
        gout = torch.randn(2, 5, 48, dtype=torch.float64)
        fast = _lke_fwd_bwd(m, x, gout, ckpt=True, autocast=True)
        laf.apply_module_patches()
        assert LokrModule.__dict__["forward"] is not lycoris_fast._fast_forward
        flag = _lke_fwd_bwd(m, x, gout, ckpt=True, autocast=True)
        assert torch.equal(fast[0], flag[0]) and torch.equal(fast[2], flag[2])
        for n in fast[1]:
            assert torch.equal(fast[1][n], flag[1][n]), n
    finally:
        laf._revert_module_patches()
        LokrModule.forward = saved[0]
        lycoris_fast._STOCK_FORWARD = saved[1]
        laf.apply_module_patches()


def test_lokr_fixed_path_still_runs_the_fast_rebuild(monkeypatch):
    """Production order (prepare_anima_runtime / prepare_moss_runtime install the fast forward, then train() applies the
    fixed paths): every multiplier-1 call still reaches lycoris_fast's fused rebuild (the speed of the production
    LoKr+DoRA runs; its numbers equal stock, so bit-identity alone would not notice a fallback to the stock forward),
    and multiplier != 1 takes the interpolating path instead."""
    from library import lycoris_fast
    from lycoris.modules.lokr import LokrModule

    laf._revert_module_patches()
    saved = (LokrModule.__dict__["forward"], lycoris_fast._STOCK_FORWARD)
    calls = []
    orig_get = lycoris_fast._REBUILD.get
    monkeypatch.setattr(lycoris_fast._REBUILD, "get", lambda *a, **k: calls.append(1) or orig_get(*a, **k))
    try:
        lycoris_fast.install_fast_lokr_forward()
        laf.apply_module_patches()
        assert LokrModule.__dict__["forward"] is laf._lokr_forward
        assert laf._original(LokrModule, "forward") is lycoris_fast._fast_forward
        assert laf._LOKR_STOCK["forward"] is lycoris_fast._fast_forward  # what _lokr_forward calls at multiplier 1
        m, _ = _lke_single("lokr", dtype=torch.float32, base_dtype=torch.bfloat16)
        m.apply_to()
        m.train()
        x = torch.randn(2, 5, 32).bfloat16()
        gout = torch.randn(2, 5, 48, dtype=torch.float64)
        _lke_fwd_bwd(m, x, gout, ckpt=True, autocast=True)
        assert len(calls) == 2  # forward + checkpoint recompute
        m.multiplier = 0.5
        _lke_fwd_bwd(m, x, gout, ckpt=True, autocast=True)
        assert len(calls) == 2
    finally:
        laf._revert_module_patches()
        LokrModule.forward = saved[0]
        lycoris_fast._STOCK_FORWARD = saved[1]
        laf.apply_module_patches()


def test_lokr_dora_network(tmp_path):
    model = make_model()
    net = make_net(model, "lokr", 10000, 10000, factor=4, full_matrix=True, dora_wd=True, wd_on_output=True)
    no_grad, unchanged = train(model, net, checkpoint=True)
    assert not no_grad and not unchanged, (no_grad, unchanged)
    to_fp32_eval(model, net)
    p32, p16 = save_both(net, tmp_path, "lokr")
    assert metadata(p32)[laf.METADATA_FORMAT_KEY] == "lokr"
    check_forge(net, p32, dora_axis="output")
    net.set_multiplier(0.0)
    ref0 = model_out(model)
    for m in loras(net):
        m.restore()
    base = model_out(model)
    assert torch.equal(ref0, base)


# --------------------------------------------------------------------------------------------------------------- full


def _full_module(out_f=24, in_f=16, bias=True, seed=0, delta=0.05):
    """A FullModule on a bf16 Linear after apply_to, trained weight W0 + delta (bf16-representable values)."""
    from lycoris.modules.full import FullModule

    torch.manual_seed(seed)
    org = nn.Linear(in_f, out_f, bias=bias).bfloat16().requires_grad_(False)
    m = FullModule(f"lora_full_{seed}", org, 1.0)
    m.apply_to()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        m.weight.copy_((m._org_weight[0].float() + delta * torch.randn(m.weight.shape, generator=g)).bfloat16())
        if m.bias is not None:
            m.bias.copy_((m.org_bias[0].float() + delta * torch.randn(m.bias.shape, generator=g)).bfloat16())
    return m


def _full_delta(m):
    dw = m.weight.detach().double() - m._org_weight[0].double()
    db = None if m.bias is None else m.bias.detach().double() - m.org_bias[0].double()
    return dw, db


def test_full_stock_bugs():
    def run():
        out = {}
        model = make_model()
        net = make_net(model, "full", 8, 8)
        out["dtypes"] = {p.dtype for p in net.parameters()}
        m = loras(net)[0]
        try:
            m.get_diff_weight(multiplier=0.5)
            out["bias_diff"] = "ok"
        except AttributeError:
            out["bias_diff"] = "crash"
        nb = _full_module(bias=False)
        out["diff_at_half"] = float(nb.get_diff_weight(multiplier=0.5)[0].double().norm() / _full_delta(nb)[0].norm())
        nb.train()
        for attr in ("rank_dropout", "module_dropout"):
            setattr(nb, attr, 0.5 if attr == "rank_dropout" else 1.0)
            try:
                nb(torch.randn(2, 16).bfloat16())
                out[attr] = "ok"
            except (RuntimeError, AttributeError):
                out[attr] = "crash"
            setattr(nb, attr, 0.0)
        # one AdamW step of 1e-5 on bf16 W0 + dW: steps under half an ulp of |W| are rounded away
        opt = torch.optim.AdamW([nb.weight], lr=1e-5, weight_decay=0.0)
        before = nb.weight.detach().clone()
        nb.weight.grad = torch.ones_like(nb.weight)
        opt.step()
        out["moved"] = float((nb.weight.detach() != before).float().mean())
        # a second load adds the file's dW to the current weight
        sd = {k: v.clone() for k, v in nb.state_dict().items()}
        nb.load_state_dict(sd)
        out["reload_twice"] = float(_full_delta(nb)[0].norm() / sd["diff"].double().norm())
        return out

    stock = _with_stock(run)
    assert stock["dtypes"] == {torch.bfloat16}
    assert stock["bias_diff"] == "crash"
    assert abs(stock["diff_at_half"] - 0.25) < 1e-3  # multiplier applied twice
    assert stock["rank_dropout"] == "crash" and stock["module_dropout"] == "crash"
    assert stock["moved"] < 0.5
    assert abs(stock["reload_twice"] - 2.0) < 1e-2
    fixed = run()
    assert fixed["dtypes"] == {torch.float32}
    assert fixed["bias_diff"] == "ok"
    assert abs(fixed["diff_at_half"] - 0.5) < 1e-6
    assert fixed["rank_dropout"] == "ok" and fixed["module_dropout"] == "ok"
    assert fixed["moved"] == 1.0
    assert abs(fixed["reload_twice"] - 1.0) < 1e-6


@pytest.mark.parametrize("checkpoint", [False, True])
def test_full(tmp_path, checkpoint):
    model = make_model(conv=True)
    net = make_net(model, "full", 8, 8)
    assert {p.dtype for p in net.parameters()} == {torch.float32}
    no_grad, unchanged = train(model, net, conv=True, checkpoint=checkpoint)
    assert not no_grad and not unchanged, (no_grad, unchanged)
    to_fp32_eval(model, net)
    ref = model_out(model, conv=True)
    p32, p16 = save_both(net, tmp_path, "full")
    md = metadata(p32)
    assert md[laf.METADATA_FLAG_KEY] == "1" and md[laf.METADATA_FORMAT_KEY] == "diff"
    from safetensors.torch import load_file

    for path, tol in ((p32, 1e-6), (p16, 5e-3)):
        sd = load_file(path)
        key_map, patch, remaining = forge_patches(net, sd)
        assert not remaining, remaining[:5]
        for m in loras(net):
            key = key_map[m.lora_name]
            dw, db = _full_delta(m)
            kind, (diff,) = patch[key]
            kind_b, (diff_b,) = patch[key[: -len(".weight")] + ".bias"]
            assert kind == kind_b == "diff"
            for s in (1.0, 0.5):  # backend/patcher/lora.py "diff": weight += strength * diff
                got = m._org_weight[0].float() + s * diff.float()
                assert rel(got.double() - m._org_weight[0].double(), s * dw, s * dw) < tol
                assert rel(s * diff_b.double(), s * db, s * db) < tol
    # --network_weights (load_weights), loading twice, --resume (state_dict, strict), create_network_from_weights
    model2 = make_model(conv=True)
    net2 = make_net(model2, "full", 8, 8)
    net2.load_weights(p32)
    net2.load_weights(p32)
    to_fp32_eval(model2, net2)
    assert rel(model_out(model2, conv=True), ref, ref) < 1e-6
    model3 = make_model(conv=True)
    net3 = make_net(model3, "full", 8, 8)
    net3.load_weights(p32)  # --network_weights kept in the config, then the resume state
    net3.load_state_dict(net.state_dict(), strict=True)
    to_fp32_eval(model3, net3)
    assert rel(model_out(model3, conv=True), ref, ref) < 1e-6
    from lycoris import kohya

    model4 = make_model(conv=True, dtype=torch.float32)
    net4, _ = kohya.create_network_from_weights(1.0, p32, None, None, model4)
    net4.apply_to(None, model4, False, True)
    net4.eval()
    assert rel(model_out(model4, conv=True), ref, ref) < 1e-6


@pytest.mark.parametrize("s", [0.0, 0.5, -1.0])
def test_full_multiplier(s):
    """DOP / iLECO / ADDifT set the multiplier: the forward must use W0 + s dW (stock: W0 + s^2 dW, crash with a bias)."""
    model = make_model()
    net = make_net(model, "full", 8, 8)
    train(model, net)
    to_fp32_eval(model, net)
    plain = make_model(dtype=torch.float32)
    mods = dict(plain.named_modules())
    with torch.no_grad():
        for m in loras(net):
            dw, db = _full_delta(m)
            org = mods[m.original_name]
            org.weight.copy_(m._org_weight[0].double() + s * dw)
            org.bias.copy_(m.org_bias[0].double() + s * db)
            W, b = m.get_merged_weight(multiplier=s)
            assert torch.allclose(W.double(), org.weight.double(), atol=1e-7, rtol=0)
            assert torch.allclose(b.double(), org.bias.double(), atol=1e-7, rtol=0)
    net.set_multiplier(s)
    got = model_out(model)
    net.set_multiplier(1.0)
    want = model_out(plain)
    assert rel(got, want, want) < 1e-6


@pytest.mark.parametrize("shape", [(3, 16), (2, 5, 16), (2, 1, 3, 4, 16)])
@pytest.mark.parametrize("autocast", [True, False])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_full_forward_matches_stock(shape, autocast, checkpoint):
    """fp32 master weights holding the stock bf16 values: same output, same input/weight gradients, bit for bit
    (autocast on: the op's own bf16 cast; off: bf16 input as Anima's adaLN, cast to the input dtype)."""
    stock = _with_stock(lambda: _full_module())
    fixed = _full_module()
    assert fixed.weight.dtype == torch.float32 and torch.equal(stock.weight.float(), fixed.weight)
    g = torch.Generator().manual_seed(3)
    x0 = torch.randn(shape, generator=g).bfloat16()
    outs = []
    for m in (stock, fixed):
        m.train()
        x = x0.clone().requires_grad_(True)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            y = torch.utils.checkpoint.checkpoint(m, x, use_reentrant=False) if checkpoint else m(x)
        y.float().square().sum().backward()
        outs.append((y.detach(), x.grad, m.weight.grad.float(), m.bias.grad.float()))
    for a, b in zip(*outs):
        assert a.dtype == b.dtype and torch.equal(a, b)


def test_full_dropouts():
    m = _full_module()
    m.train()
    x = torch.randn(4, 16)
    dw, db = _full_delta(m)
    w0, b0 = m._org_weight[0].double(), m.org_bias[0].double()
    m.module_dropout = 1.0
    assert torch.allclose(m(x).double(), F.linear(x.double(), w0, b0), atol=1e-6)
    m.module_dropout = 0.0
    m.rank_dropout = 0.5
    torch.manual_seed(11)
    y = m(x)
    torch.manual_seed(11)
    keep = (torch.rand(dw.shape[0]) > 0.5).double()
    want = F.linear(x.double(), w0 + keep[:, None] * dw, b0 + keep * db)
    assert 0 < keep.sum() < keep.numel()
    assert torch.allclose(y.double(), want, atol=1e-6)


def test_tsm_gate_init_starts_stage2_at_the_stage1_model():
    """algo=tsm stage 2 with tsm_gate_init (user 10-24): 1.0 over a zero router = the stage-1 output; 0 keeps LyCORIS'
    gate-0 start; a trained router is never overwritten."""
    x = torch.randn(2, 5, 24, dtype=torch.float64, generator=torch.Generator().manual_seed(11))
    ts = [700, 700]
    m1 = _tsm_module(1)
    m2 = _tsm_module(2)
    m2.load_state_dict(m1.state_dict())  # same experts
    with torch.no_grad():
        m2.router_fc.weight.zero_()
        m2.router_fc.bias.zero_()
        m2.timestep_embed.weight.zero_()
    net = argparse.Namespace(loras=[m2])
    assert laf._tsm_apply_gate_init(net, 0.0) == 0 and not bool(torch.any(m2.router_fc.bias != 0))
    gate0 = _tsm_run(m2, x, ts)["out"]
    assert laf._tsm_apply_gate_init(net, 1.0) == 1
    assert torch.allclose(_tsm_run(m2, x, ts)["out"], _tsm_run(m1, x, ts)["out"], rtol=1e-12, atol=1e-12)
    assert not torch.allclose(gate0, _tsm_run(m1, x, ts)["out"])
    with torch.no_grad():
        m2.router_fc.weight[0, 0] = 0.5  # a trained router: left alone
        m2.router_fc.bias.fill_(0.25)
    assert laf._tsm_apply_gate_init(net, 1.0) == 0 and bool(torch.all(m2.router_fc.bias == 0.25))
