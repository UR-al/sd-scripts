"""CPU tests for TSM-LoKr (network_args algo=tsm-lokr, library/tsm_lokr.py) under --lycoris_algo_fixes.

Math against explicit dense references (fp64 / fp32 / bf16 autocast), stage 1 and 2, DoRA on/off, 3-D and 5-D inputs,
per-sample timesteps in a batch, gradient checkpointing, registration/dispatch, validation messages, save/load round
trips and the stage-1 -> stage-2 hand-over.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -B -m pytest tests/test_tsm_lokr.py
"""
import argparse
import logging
import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from library import lycoris_algo_fixes as laf
from library import tsm_lokr

torch.set_num_threads(4)
torch.backends.mkldnn.enabled = False

LOG = logging.getLogger("test_tsm_lokr")

from tests.test_lycoris_algo_fixes import PRESET, loss_fn, make_batch, make_model  # noqa: E402

BASE_ARGS = {"factor": 4, "full_matrix": True, "dora_wd": True, "wd_on_output": True, "tsm_n_scales": "4,1"}


@pytest.fixture(autouse=True)
def _patches():
    laf.apply_module_patches()
    yield
    laf._TSM_SAMPLE_T.clear()
    from lycoris.modules.tsm import clear_tsm_timestep

    clear_tsm_timestep(0)


# ------------------------------------------------------------------------------------------------------------- helpers


def make_module(out_dim=32, in_dim=24, dtype=torch.float32, seed=0, base_dtype=None, **kw):
    """One TSMLokrModule on a Linear, random factors (LoKr starts at w2 = 0), applied to the Linear."""
    raw = dict(BASE_ARGS)
    raw.update({k: str(v) for k, v in kw.items()})
    raw = {k: str(v) for k, v in raw.items()}
    tsm_lokr.set_active_config(tsm_lokr.parse_config(raw))
    torch.manual_seed(seed)
    lin = nn.Linear(in_dim, out_dim, bias=False)
    if base_dtype is not None:
        lin = lin.to(base_dtype)
    for p in lin.parameters():
        p.requires_grad_(False)
    cls = tsm_lokr.get_module_class()
    m = cls("lora_test", lin, 1.0, 10000, 10000)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for sc in m.experts:
            for e in sc:
                for p in e.parameters():
                    p.copy_(torch.randn(p.shape, generator=g) * 0.3)
        if m.dora_scale is not None:
            m.dora_scale.mul_(1 + 0.1 * torch.randn(m.dora_scale.shape, generator=g))
        m.router_fc.weight.copy_(torch.randn(m.router_fc.weight.shape, generator=g) * 0.1)
        m.router_fc.bias.copy_(torch.randn(m.router_fc.bias.shape, generator=g))
        m.timestep_embed.weight.copy_(torch.randn(m.timestep_embed.weight.shape, generator=g) * 0.5)
    m.apply_to()
    if dtype != torch.float32:
        m = m.to(dtype)
        lin.to(dtype)
    return m, lin


def set_t(ints):
    from lycoris.modules.tsm import set_tsm_timestep

    ints = [int(t) for t in ints]
    set_tsm_timestep(ints[0], 0)
    if len(ints) > 1:
        laf._TSM_SAMPLE_T[0] = ints
    else:
        laf._TSM_SAMPLE_T.pop(0, None)


def expert_index(m, t):
    T = m.tsm_num_timesteps
    return [max(1, min(math.ceil(t / T * n), n)) - 1 for n in m.tsm_n_scales]


def ref_weight(m, W0, t, x_row=None, mu=1.0, leaves=None):
    """Dense fp64 reference weight for one sample (stage 1, or stage 2 with the gates of x_row)."""
    idx = expert_index(m, t)
    V = W0.double()
    for s, e in enumerate(idx):
        ex = m.experts[s][e]
        w1, w2 = leaves[(s, e)] if leaves else (p.double() for p in ex.factors())
        K = torch.kron(w1, w2) * ex.scale
        if m.tsm_stage == 2 and s > 0:
            z = x_row.double().reshape(-1, m.in_dim).mean(0)
            rw = leaves["router_w"] if leaves else m.router_fc.weight.double()
            rb = leaves["router_b"] if leaves else m.router_fc.bias.double()
            emb = leaves["embed"] if leaves else m.timestep_embed.weight.double()
            gate = (rw @ z + rb + emb[min(max(t - 1, 0), m.tsm_num_timesteps - 1)])[s - 1]
            K = gate * K
        V = V + K
    if m.wd:
        mag = leaves["m"] if leaves else m.dora_scale.double()
        D = V * (mag / (torch.linalg.vector_norm(V, dim=1, keepdim=True) + torch.finfo(torch.float32).eps))
    else:
        D = V
    return W0.double() + mu * (D - W0.double())


def ref_forward(m, W0, x, ints, mu=1.0, leaves=None):
    rows = []
    for b in range(x.shape[0]):
        t = ints[b] if len(ints) > 1 else ints[0]
        W = ref_weight(m, W0, t, x[b], mu, leaves)
        rows.append(x[b].double() @ W.t())
    return torch.stack(rows)


def make_leaves(m):
    leaves = {}
    for s, sc in enumerate(m.experts):
        for e, ex in enumerate(sc):
            leaves[(s, e)] = tuple(p.detach().double().clone().requires_grad_(True) for p in ex.factors())
    if m.wd:
        leaves["m"] = m.dora_scale.detach().double().clone().requires_grad_(True)
    leaves["router_w"] = m.router_fc.weight.detach().double().clone().requires_grad_(True)
    leaves["router_b"] = m.router_fc.bias.detach().double().clone().requires_grad_(True)
    leaves["embed"] = m.timestep_embed.weight.detach().double().clone().requires_grad_(True)
    return leaves


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


# ---------------------------------------------------------------------------------------------------------- T1 math


def test_kron4_and_kron_mv_match_torch_kron():
    g = torch.Generator().manual_seed(0)
    for (a, c), (b, d) in [((4, 4), (8, 6)), ((2, 3), (5, 7)), ((4, 4), (64, 128))]:
        w1 = torch.randn(a, c, generator=g)
        w2 = torch.randn(b, d, generator=g)
        assert torch.equal(tsm_lokr.kron4(w1, w2), torch.kron(w1, w2))
        x = torch.randn(3, 5, c * d, generator=g, dtype=torch.float64)
        dense = x @ torch.kron(w1.double(), w2.double()).t() * 0.7
        assert rel(tsm_lokr.kron_mv(x, w1.double(), w2.double(), 0.7), dense) < 1e-13


def test_row_norm_terms_three_vector_formula():
    g = torch.Generator().manual_seed(1)
    W0 = torch.randn(64, 48, generator=g, dtype=torch.float64)
    C = torch.randn(64, 48, generator=g, dtype=torch.float64) * 0.1
    X1 = torch.randn(64, 48, generator=g, dtype=torch.float64) * 0.1
    X2 = torch.randn(64, 48, generator=g, dtype=torch.float64) * 0.1
    a, b, c = tsm_lokr.row_norm_terms(W0, C, [X1, X2])
    for gate in ([0.3, -1.2], [2.5, 0.0], [-3.0, 1.7]):
        V = W0 + C + gate[0] * X1 + gate[1] * X2
        gt = torch.tensor(gate, dtype=torch.float64)
        n2 = a + 2 * (gt @ b) + torch.einsum("j,k,jko->o", gt, gt, c)
        assert rel(n2, (V * V).sum(1)) < 1e-13
    a32, b32, c32 = tsm_lokr.row_norm_terms(W0.float(), C.float(), [X1.float(), X2.float()])
    assert a32.dtype == torch.float32 and rel(a32, a) < 1e-6


# ------------------------------------------------------------------------------------------------- T2 stage 1 parity


@pytest.mark.parametrize("dora", [True, False])
@pytest.mark.parametrize("shape", [(3, 24), (2, 7, 24), (2, 1, 3, 4, 24)])
def test_stage1_forward_and_grads_fp64(dora, shape):
    m, lin = make_module(dtype=torch.float64, dora_wd=dora)
    W0 = lin.weight.detach()
    x = torch.randn(*shape, dtype=torch.float64)
    t = 600
    set_t([t])
    y = lin(x)
    leaves = make_leaves(m)
    y_ref = ref_forward(m, W0, x, [t], leaves=leaves)
    assert rel(y, y_ref) < 1e-12
    R = torch.randn_like(y)
    (y * R).sum().backward()
    (y_ref * R).sum().backward()
    idx = expert_index(m, t)
    for s, e in enumerate(idx):
        got = [p.grad for p in (m.experts[s][e].lokr_w1, m.experts[s][e].lokr_w2)]
        for gg, ref in zip(got, leaves[(s, e)]):
            assert rel(gg, ref.grad) < 1e-10
    if dora:
        assert rel(m.dora_scale.grad, leaves["m"].grad) < 1e-10
    others = [p for s, sc in enumerate(m.experts) for e, ex in enumerate(sc) if e != idx[s] for p in ex.parameters()]
    assert all(p.grad is None for p in others)  # only the active experts are in the graph


def test_stage1_fp32_close_to_fp64_reference():
    m, lin = make_module(out_dim=64, in_dim=48)
    x = torch.randn(2, 9, 48)
    set_t([140])
    y = lin(x)
    y_ref = ref_forward(m, lin.weight.detach(), x, [140])
    assert rel(y, y_ref) < 1e-5


def test_stage1_bf16_autocast_not_worse_than_lokr():
    """With the context expert at zero TSM-LoKr is one LoKr+DoRA: under bf16 autocast its error against the fp64
    reference is no worse than LyCORIS' own LokrModule with the same factors."""
    from lycoris.modules.lokr import LokrModule

    errs_tsm, errs_lokr = [], []
    for seed in range(4):
        m, lin = make_module(out_dim=64, in_dim=48, seed=seed, base_dtype=torch.bfloat16)
        with torch.no_grad():
            m.experts[1][0].lokr_w2.zero_()
        W0 = lin.weight.detach()
        lin2 = nn.Linear(48, 64, bias=False).to(torch.bfloat16)
        lin2.weight.data.copy_(W0)
        lk = LokrModule("lk", lin2, 1.0, 10000, 10000, factor=4, full_matrix=True, weight_decompose=True)
        t = 300
        core = m.experts[0][expert_index(m, t)[0]]
        with torch.no_grad():
            lk.lokr_w1.copy_(core.lokr_w1)
            lk.lokr_w2.copy_(core.lokr_w2)
            lk.dora_scale.copy_(m.dora_scale)
        lk.apply_to()
        x = torch.randn(2, 11, 48).bfloat16()
        set_t([t])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = lin(x)
            y_lk = lin2(x)
        y_ref = ref_forward(m, W0, x.double(), [t])
        errs_tsm.append(rel(y, y_ref))
        errs_lokr.append(rel(y_lk, y_ref))
    assert max(errs_tsm) <= max(errs_lokr) * 1.05 + 1e-6, (errs_tsm, errs_lokr)
    assert max(errs_tsm) < 2e-2


def test_one_expert_parity_with_stock_lokr_fp32():
    """T3: context w2 = 0 -> the same output as LyCORIS LokrModule (fp32, no autocast) within a few ulps."""
    from lycoris.modules.lokr import LokrModule

    m, lin = make_module(out_dim=64, in_dim=48)
    with torch.no_grad():
        m.experts[1][0].lokr_w2.zero_()
    lin2 = nn.Linear(48, 64, bias=False)
    lin2.weight.data.copy_(lin.weight)
    lk = LokrModule("lk", lin2, 1.0, 10000, 10000, factor=4, full_matrix=True, weight_decompose=True)
    core = m.experts[0][expert_index(m, 999)[0]]
    with torch.no_grad():
        lk.lokr_w1.copy_(core.lokr_w1)
        lk.lokr_w2.copy_(core.lokr_w2)
        lk.dora_scale.copy_(m.dora_scale)
    lk.apply_to()
    x = torch.randn(3, 48)
    set_t([999])
    y, y_lk = lin(x), lin2(x)
    assert rel(y, y_lk) < 4e-7


# ------------------------------------------------------------------------------------------------------- T4 stage 2


@pytest.mark.parametrize("dora", [True, False])
@pytest.mark.parametrize("scales", ["4,1", "4,2,1"])
def test_stage2_three_vector_matches_dense_and_router_grads(dora, scales):
    m, lin = make_module(dtype=torch.float64, dora_wd=dora, tsm_n_scales=scales, tsm_stage=2)
    W0 = lin.weight.detach()
    x = torch.randn(3, 1, 2, 5, 24, dtype=torch.float64)
    ints = [100, 620, 980]
    set_t(ints)
    y = lin(x)
    leaves = make_leaves(m)
    y_ref = ref_forward(m, W0, x, ints, leaves=leaves)
    assert rel(y, y_ref) < 1e-10
    R = torch.randn_like(y)
    (y * R).sum().backward()
    (y_ref * R).sum().backward()
    assert rel(m.router_fc.weight.grad, leaves["router_w"].grad) < 1e-9
    assert rel(m.router_fc.bias.grad, leaves["router_b"].grad) < 1e-9
    assert rel(m.timestep_embed.weight.grad, leaves["embed"].grad) < 1e-9
    assert all(p.grad is None for sc in m.experts for e in sc for p in e.parameters())
    if dora:
        assert m.dora_scale.grad is None


@pytest.mark.parametrize("stage", [1, 2])
def test_multiplier_zero_is_base_and_half_is_forge_blend(stage):
    m, lin = make_module(dtype=torch.float64, tsm_stage=stage)
    W0 = lin.weight.detach()
    x = torch.randn(2, 4, 24, dtype=torch.float64)
    set_t([420, 77])
    m.multiplier = 0.0
    assert torch.equal(lin(x), F.linear(x, W0))
    m.multiplier = 0.5
    y = lin(x)
    m.multiplier = 1.0
    assert rel(y, ref_forward(m, W0, x, [420, 77], mu=0.5)) < 1e-11


# ------------------------------------------------------------------------------------------------ T5 mixed batches


@pytest.mark.parametrize("stage", [1, 2])
def test_mixed_timestep_batch_equals_per_sample_calls(stage):
    m, lin = make_module(tsm_stage=stage)
    x = torch.randn(4, 6, 24)
    ints = [900, 10, 905, 10]
    set_t(ints)
    y = lin(x)
    for b, t in enumerate(ints):
        set_t([t])
        assert torch.allclose(y[b:b + 1], lin(x[b:b + 1]), rtol=0, atol=1e-6)
    set_t([5, 6, 7])
    with pytest.raises(RuntimeError, match="timesteps"):
        lin(x)


def test_training_forward_without_timestep_raises():
    m, lin = make_module()
    laf._TSM_SAMPLE_T.clear()
    from lycoris.modules.tsm import clear_tsm_timestep

    clear_tsm_timestep(0)
    m.train()
    with pytest.raises(RuntimeError, match="without a timestep"):
        lin(torch.randn(2, 24))
    with torch.no_grad():
        lin(torch.randn(2, 24))  # sampling without a hook: t = T with a warning


# ---------------------------------------------------------------------------------------- network-level helpers


def make_net(model, stage=1, **kw):
    from lycoris import kohya

    raw = {"algo": "tsm-lokr", **{k: str(v) for k, v in {**BASE_ARGS, "tsm_stage": stage, **kw}.items()}}
    ckw = laf.prepare_network_kwargs("tsm-lokr", raw, LOG)
    net = kohya.create_network(1.0, 10000, 10000, None, [], model, preset=dict(PRESET), **ckw)
    net.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    args = argparse.Namespace(network_args=[f"{k}={v}" for k, v in raw.items()], network_dim=10000, network_alpha=10000,
                              lycoris_algo_fixes=True, mixed_precision="bf16")
    laf.after_network_created(net, args, LOG)
    net.prepare_grad_etc(None, model)
    return net


def randomize(net, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in net.named_parameters():
            if "lokr_w" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def grads_of(net):
    return {n: (p.grad.clone() if p.grad is not None else None) for n, p in net.named_parameters()}


# --------------------------------------------------------------------------------------------- T6 checkpointing


def test_gradient_checkpointing_gives_identical_grads():
    out = []
    for ckpt in (False, True):
        model = make_model(seed=0)
        net = make_net(model)
        randomize(net)
        batch = make_batch(5)
        laf.set_timestep(net, torch.tensor([250.0, 760.0]))
        loss = loss_fn(model, batch, checkpoint=ckpt)
        loss.backward()
        laf.clear_timestep(net)
        out.append(grads_of(net))
    a, b = out
    assert a.keys() == b.keys()
    for k in a:
        assert (a[k] is None) == (b[k] is None), k
        if a[k] is not None:
            assert torch.equal(a[k], b[k]), k


# ------------------------------------------------------------------------------------------------- T7 dispatch


def test_dispatch_and_revert():
    from lycoris.wrapper import network_module_dict

    assert network_module_dict["tsm-lokr"].__name__ == "TSMLokrModule"
    model = make_model()
    net = make_net(model)
    assert {type(m).__name__ for m in net.loras} == {"TSMLokrModule"}
    from tests.test_lycoris_algo_fixes import make_net as make_net_other

    for algo, cls in (("tsm", "TSMModule"), ("lokr", "LokrModule")):
        n2 = make_net_other(make_model(), algo, 4, 4)
        assert {type(m).__name__ for m in n2.loras} == {cls}
    laf._revert_module_patches()
    assert "tsm-lokr" not in network_module_dict
    laf.apply_module_patches()
    assert "tsm-lokr" in network_module_dict


def test_module_outside_the_flag_is_refused():
    tsm_lokr.set_active_config(None)
    cls = tsm_lokr.get_module_class()
    with pytest.raises(ValueError, match="lycoris_algo_fixes"):
        cls("x", nn.Linear(8, 8), 1.0, 4, 4)


# ---------------------------------------------------------------------------------------------- T8 validation


@pytest.mark.parametrize("bad", [
    {"tsm_router_input": "bottleneck"}, {"dora_wd": "True", "wd_on_output": "False"}, {"tsm_stage": "3"},
    {"tsm_n_scales": "8"}, {"tsm_n_scales": "8,0"}, {"bypass_mode": "True"}, {"use_scalar": "True"},
    {"rank_dropout": "0.1"}, {"dropout": "0.05"}, {"module_dropout": "0.1"}, {"rs_lora": "True"},
    {"unbalanced_factorization": "True"}, {"svd_segment": "top"}, {"ggpo_beta": "0.01"}, {"nora": "True"},
    {"torch_compile": "True"}, {"tsm_resident_cores": "9"}, {"tsm_resident_cores": "0"}, {"factor": "0"},
    {"tsm_n_scales": "a,b"}, {"weight_noise_sigma": "0.1"},
])
def test_validation_errors_are_bilingual(bad):
    raw = {"algo": "tsm-lokr", "factor": "4", "full_matrix": "True", "dora_wd": "True", "tsm_n_scales": "8,1", **bad}
    with pytest.raises(ValueError) as e:
        laf.prepare_network_kwargs("tsm-lokr", raw, LOG)
    msg = str(e.value)
    assert msg.startswith("tsm-lokr:") and " / tsm-lokr: " in msg
    assert any("가" <= ch <= "힣" for ch in msg)  # Korean half present


def test_defaults_and_trainer_checks():
    cfg = tsm_lokr.parse_config({"tsm_n_scales": "8,1", "factor": "4"})
    assert cfg.n_scales == (8, 1) and cfg.resident_cores == 8 and cfg.offload_states and not cfg.grouped_accumulation
    assert [cfg.core_index(t) for t in (0, 1, 125, 126, 1000)] == [0, 0, 0, 1, 7]
    assert cfg.core_index_from_timestep(124.9) == 0 and cfg.core_index_from_timestep(125.6) == 1
    tsm_lokr.set_active_config(cfg)
    ns = argparse.Namespace
    for args in (ns(deepspeed=True), ns(use_ramtorch_network=True), ns(scale_weight_norms=1.0),
                 ns(dim_from_weights=True)):
        with pytest.raises(ValueError, match=" / tsm-lokr: "):
            laf.check_tsm_lokr_args(args)
    tsm_lokr.set_active_config(tsm_lokr.parse_config({"tsm_stage": "2"}))
    with pytest.raises(ValueError, match="network_weights"):
        laf.check_tsm_lokr_args(ns(network_weights=None, resume=None))
    tsm_lokr.set_active_config(tsm_lokr.parse_config({"tsm_grouped_accumulation": "True"}))
    with pytest.raises(ValueError, match="qmc"):
        laf.check_tsm_lokr_args(ns(qmc_timestep_sampling="sobol"))
    laf.check_tsm_lokr_args(ns())


# ----------------------------------------------------------------------------------------------- T9 save / load


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_save_load_round_trip_and_metadata(tmp_path, dtype):
    from safetensors import safe_open

    model = make_model()
    net = make_net(model)
    randomize(net)
    laf.set_timestep(net, torch.tensor([300.0], dtype=torch.bfloat16))  # training call: captures the dtype
    laf.clear_timestep(net)
    f = str(tmp_path / "x.safetensors")
    net.save_weights(f, dtype, {"ss_network_args": "{}"})
    with safe_open(f, framework="pt") as h:
        md = h.metadata()
        keys = set(h.keys())
    assert md["ss_lycoris_algo"] == "tsm-lokr" and md["ss_saved_format"] == "tsm-lokr-native"
    assert md["ss_tsm_expert"] == "lokr" and md["ss_lycoris_algo_fixes"] == "1"
    cfg = __import__("json").loads(md["ss_tsm_lokr"])
    assert cfg == {"version": 1, "n_scales": [4, 1], "num_timesteps": 1000, "stage": 1, "router_input": "input",
                   "factors": [4, 4], "full_matrix": True, "decompose_both": False, "dora": "out",
                   "timestep_dtype": "bf16", "timestep_scale": 1000, "pooling": "sample", "multiplier": "forge-blend",
                   "gate_init": 0.0}
    assert "lora_unet_blocks_0_q.experts.0.3.lokr_w2" in keys and "lora_unet_blocks_0_q.dora_scale" in keys
    assert not any(k.endswith((".up.weight", ".down.weight")) for k in keys)  # never mistaken for algo=tsm
    net2 = make_net(make_model())
    info = net2.load_weights(f)
    assert not info.get("missing keys") and not info.get("unexpected keys")
    sd1, sd2 = net.state_dict(), net2.state_dict()
    for k in sd1:
        if dtype == torch.float32:
            assert torch.equal(sd1[k], sd2[k]), k
        else:
            assert torch.equal(sd1[k].to(dtype).float(), sd2[k].float()), k


def test_stage2_from_stage1_file(tmp_path):
    model = make_model()
    net1 = make_net(model)
    randomize(net1)
    f = str(tmp_path / "s1.safetensors")
    net1.save_weights(f, torch.float32, {})
    net2 = make_net(make_model(), stage=2, tsm_gate_init=1.0)
    net2.load_weights(f)
    m1, m2 = net1.loras[0], net2.loras[0]
    assert torch.equal(m1.experts[0][2].lokr_w1, m2.experts[0][2].lokr_w1)
    assert torch.all(m2.router_fc.bias == 1.0) and torch.all(m2.router_fc.weight == 0)
    assert torch.equal(m1.dora_scale, m2.dora_scale)
    # missing experts in a stage-2 load are an error
    from safetensors.torch import load_file, save_file

    sd = {k: v for k, v in load_file(f).items() if ".experts.0.1." not in k}
    f2 = str(tmp_path / "s1_cut.safetensors")
    save_file(sd, f2)
    net3 = make_net(make_model(), stage=2)
    with pytest.raises(ValueError, match="missing"):
        net3.load_weights(f2)


def test_stage2_gate_one_reproduces_stage1_output():
    torch.manual_seed(0)
    m1, lin1 = make_module(dtype=torch.float64, tsm_stage=1)
    m2, lin2 = make_module(dtype=torch.float64, tsm_stage=2)
    m2.load_state_dict(m1.state_dict())
    with torch.no_grad():
        m2.router_fc.weight.zero_()
        m2.timestep_embed.weight.zero_()
        m2.router_fc.bias.fill_(1.0)
    lin2.weight.data.copy_(lin1.weight)
    x = torch.randn(2, 5, 24, dtype=torch.float64)
    set_t([111, 777])
    assert rel(lin2(x), lin1(x)) < 1e-12


# --------------------------------------------------------------------------------------------- T10 stage 2 train


def test_stage2_freezes_experts_and_trains_router():
    from tests.test_lycoris_algo_fixes import build_optimizer

    model = make_model()
    net = make_net(model, stage=2)
    randomize(net)
    opt = build_optimizer(net, 1e-3)
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    names = {id(p): n for n, p in net.named_parameters()}
    assert all(("router_fc" in names[i] or "timestep_embed" in names[i]) for i in in_opt)
    laf.set_timestep(net, torch.tensor([500.0, 20.0]))
    loss_fn(model, make_batch(1)).backward()
    laf.clear_timestep(net)
    m = net.loras[0]
    assert m.router_fc.weight.grad is not None and bool(torch.any(m.router_fc.weight.grad != 0))
    assert all(p.grad is None for sc in m.experts for e in sc for p in e.parameters())
    assert m.dora_scale.grad is None



def test_timestep_dtype_is_recorded_only_for_tsm_lokr():
    """set_timestep records the trainer's timestep dtype (saved as timestep_dtype) for TSM-LoKr networks only; algo=tsm
    keeps its old path."""
    from tests.test_lycoris_algo_fixes import make_net as make_net_other

    tsm_net = make_net_other(make_model(), "tsm", 4, 4)
    laf.set_timestep(tsm_net, torch.tensor([300.0]).bfloat16())
    laf.clear_timestep(tsm_net)
    assert "tsm_timestep_dtype" not in laf._network_cfg(tsm_net)
    net = make_net(make_model())
    with torch.no_grad():  # sampling calls do not decide it
        laf.set_timestep(net, torch.tensor([300.0]))
    laf.clear_timestep(net)
    assert "tsm_timestep_dtype" not in laf._network_cfg(net)
    laf.set_timestep(net, torch.tensor([300.0]).bfloat16())
    laf.clear_timestep(net)
    assert laf._network_cfg(net)["tsm_timestep_dtype"] == "bf16"
