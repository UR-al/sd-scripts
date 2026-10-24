"""CPU tests for the TSM-LoKr core-expert offload manager (library/tsm_offload.py).

The manager is run with device=cpu: its "device" buffers are separate CPU tensors and every copy is synchronous, so
these tests check the bookkeeping (which tensors move when, what the optimizer sees, bit-identity of the updates,
resume) but not CUDA stream ordering, pinned memory, or VRAM; those are the GPU checks in
scratchpad/tsm_lokr/gpu/run_tsm_lokr_gpu.py.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -B -m pytest tests/test_tsm_offload.py
"""
import argparse
import io
import os
import random

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

from library import lycoris_algo_fixes as laf
from library import tsm_lokr
from library.tsm_offload import ExpertOffloadManager, SlabLayout

torch.set_num_threads(4)
torch.backends.mkldnn.enabled = False

from tests.test_lycoris_algo_fixes import loss_fn, make_batch, make_model  # noqa: E402
from tests.test_tsm_lokr import make_net, randomize  # noqa: E402


@pytest.fixture(autouse=True)
def _patches():
    laf.apply_module_patches()
    yield
    laf._TSM_SAMPLE_T.clear()


def build(seed=0, accum=4, offload=None, **kw):
    model = make_model(seed=0)
    net = make_net(model, **kw)
    randomize(net, seed=3)
    res = net.prepare_optimizer_params(text_encoder_lr=None, unet_lr=1e-2, learning_rate=1e-2)
    groups = res[0] if isinstance(res, tuple) else res
    opt = torch.optim.AdamW(groups, lr=1e-2, betas=(0.9, 0.99), weight_decay=0.01, eps=1e-8, fused=True)
    mgr = None
    if offload:
        args = argparse.Namespace(gradient_accumulation_steps=accum)
        mgr = laf.attach_tsm_offload(net, opt, args, torch.device("cpu"), batch_size=2)
        assert mgr is not None
    return model, net, opt, mgr


def run(model, net, opt, steps, accum, seed=11, start=0, check=None):
    rng = random.Random(seed)
    for _ in range(start):  # resume: consume the same random stream
        for _ in range(accum):
            rng.random(), rng.random()
    params = [p for g in opt.param_groups for p in g["params"]]
    for step in range(start, steps):
        for micro in range(accum):
            ts = torch.tensor([rng.random() * 1000, rng.random() * 1000]).bfloat16()
            laf.set_timestep(net, ts)
            loss = loss_fn(model, make_batch(1000 * step + micro))
            (loss / accum).backward()
            laf.clear_timestep(net)
        torch.nn.utils.clip_grad_norm_(params, 0.05)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if check is not None:
            check(step)


def snapshot(net):
    return {n: p.detach().clone() for n, p in net.named_parameters()}


def assert_same(a, b):
    assert a.keys() == b.keys()
    diff = [k for k in a if not torch.equal(a[k], b[k])]
    assert not diff, diff[:5]


def in_slabs(t, layout):
    ptr = t.data_ptr()
    return any(s.data_ptr() <= ptr < s.data_ptr() + s.numel() * s.element_size() for s in layout.slabs)


# --------------------------------------------------------------------------------------------- slab layout


def test_slab_layout_first_fit_and_views():
    lay = SlabLayout([10, 7, 3, 20, 5], slab_numel=16, dtype=torch.float32, pin=False)
    assert [len(lay.slabs), lay.slabs[0].numel()] == [3, 16]
    assert lay.places == [(0, 0), (1, 0), (0, 10), (2, 0), (1, 7)]  # 20 > slab: own slab of 20
    buf = torch.arange(lay.total, dtype=torch.float32)
    v = lay.dev_view(buf, 4, (5,))
    assert v.tolist() == [lay.base[1] + 7 + i for i in range(5)]


# --------------------------------------------------------------------------------------- T11 state streaming


@pytest.mark.parametrize("accum", [1, 4])
def test_state_offload_is_bitwise_identical(accum):
    ref = build(accum=accum)
    run(*ref[:3], steps=12, accum=accum)
    off = build(accum=accum, offload=True)
    model, net, opt, mgr = off
    managed = {id(p) for g in mgr.groups for p in g.params}

    def check(step):
        for g in mgr.groups:  # between steps every existing core state lives in the host slabs
            for j, p in enumerate(g.params):
                st = opt.state.get(p)
                if st:
                    assert g.has_state[j]
                    assert in_slabs(st["exp_avg"], g.state) and in_slabs(st["exp_avg_sq"], g.state)
                    assert st["step"].data_ptr() == g.steps_host[j].data_ptr()
        ctx = [p for p in net.parameters() if id(p) not in managed and opt.state.get(p)]
        assert ctx and not any(in_slabs(opt.state[p]["exp_avg"], g.state) for p in ctx[:3] for g in mgr.groups)

    run(model, net, opt, steps=12, accum=accum, check=check)
    assert_same(snapshot(ref[1]), snapshot(net))
    # same optimizer state values too
    for (pa, pb) in zip([p for g in ref[2].param_groups for p in g["params"]],
                        [p for g in opt.param_groups for p in g["params"]]):
        sa, sb = ref[2].state.get(pa, {}), opt.state.get(pb, {})
        assert sa.keys() == sb.keys()
        for k in sa:
            assert torch.equal(sa[k], sb[k])


def test_offload_refuses_non_adam_and_small_residency():
    model = make_model()
    net = make_net(model)
    sgd = torch.optim.SGD(net.parameters(), lr=0.1)
    args = argparse.Namespace(gradient_accumulation_steps=4)
    with pytest.raises(ValueError, match="AdamW"):
        laf.attach_tsm_offload(net, sgd, args, torch.device("cpu"))
    net2 = make_net(make_model(), tsm_resident_cores=2)
    opt = torch.optim.AdamW(net2.parameters(), lr=0.1, fused=True)
    with pytest.raises(ValueError, match="tsm_resident_cores=2"):
        laf.attach_tsm_offload(net2, opt, args, torch.device("cpu"), batch_size=1)
    # grouped accumulation: one core per step is enough, but only where the trainer implements it
    net3 = make_net(make_model(), tsm_resident_cores=1, tsm_grouped_accumulation=True)
    opt3 = torch.optim.AdamW(net3.parameters(), lr=0.1, fused=True)
    with pytest.raises(ValueError, match="Anima"):
        laf.attach_tsm_offload(net3, opt3, args, torch.device("cpu"), batch_size=1)
    assert laf.attach_tsm_offload(net3, opt3, args, torch.device("cpu"), batch_size=1, grouped_supported=True)


def test_nothing_to_manage_returns_none():
    net = make_net(make_model(), tsm_offload_states=False)
    opt = torch.optim.AdamW(net.parameters(), lr=0.1, fused=True)
    assert laf.attach_tsm_offload(net, opt, argparse.Namespace(gradient_accumulation_steps=4), torch.device("cpu")) is None


# ------------------------------------------------------------------------------------------------- T12 LRU


@pytest.mark.parametrize("offload_states", [True, False])
def test_lru_weights_bitwise_identical(offload_states):
    accum = 1  # batch 2 x accumulation 1: at most 2 cores per step
    ref = build(accum=accum)
    run(*ref[:3], steps=15, accum=accum)
    model, net, opt, mgr = build(accum=accum, offload=True, tsm_resident_cores=2, tsm_offload_states=offload_states)
    assert len(mgr.resident_groups()) == 2

    def check(step):
        assert len(mgr.resident_groups()) <= 2
        for g in mgr.groups:
            if not g.resident:
                assert all(in_slabs(p.data, g.weights) for p in g.params)

    run(model, net, opt, steps=15, accum=accum, check=check)
    assert mgr.evictions > 2 and mgr.uploads > 0
    mgr.flush()
    assert_same(snapshot(ref[1]), snapshot(net))


def test_lru_never_evicts_experts_with_grads():
    # attach allows 2 for batch 2 x accumulation 1; three intervals inside one window must stop, not evict a core
    # that holds gradients
    model, net, opt, mgr = build(accum=1, offload=True, tsm_resident_cores=2)
    cfg = net.loras[0].cfg
    ts = [50.0, 300.0, 600.0]  # three different core intervals of 4
    assert len({cfg.core_index_from_timestep(t) for t in ts}) == 3
    laf.set_timestep(net, torch.tensor([ts[0]]))
    loss_fn(model, make_batch(1)).backward()
    laf.clear_timestep(net)
    laf.set_timestep(net, torch.tensor([ts[1]]))
    loss_fn(model, make_batch(2)).backward()
    laf.clear_timestep(net)
    with pytest.raises(RuntimeError, match="tsm_resident_cores=2"):
        laf.set_timestep(net, torch.tensor([ts[2]]))


def test_stage2_eviction_needs_no_write_back(tmp_path):
    net1 = make_net(make_model())
    randomize(net1)
    f = str(tmp_path / "s1.safetensors")
    net1.save_weights(f, torch.float32, {})
    model = make_model()
    net = make_net(model, stage=2, tsm_resident_cores=2)
    net.load_weights(f)
    res = net.prepare_optimizer_params(text_encoder_lr=None, unet_lr=1e-2, learning_rate=1e-2)
    opt = torch.optim.AdamW(res[0] if isinstance(res, tuple) else res, lr=1e-2, fused=True)
    mgr = laf.attach_tsm_offload(net, opt, argparse.Namespace(gradient_accumulation_steps=2), torch.device("cpu"),
                                 batch_size=2)  # stage 2: experts frozen, so 2 resident cores do for any accumulation
    run(model, net, opt, steps=4, accum=2)
    assert mgr.evictions > 1 and not any(g.dirty for g in mgr.groups)
    for a, b in zip(net1.loras, net.loras):  # frozen experts unchanged after the swaps
        for sa, sb in zip(a.experts, b.experts):
            for ea, eb in zip(sa, sb):
                for pa, pb in zip(ea.parameters(), eb.parameters()):
                    assert torch.equal(pa, pb)


# ---------------------------------------------------------------------------------------------- T13 resume


@pytest.mark.parametrize("resident", [4, 2])
def test_resume_round_trip_is_bitwise(resident):
    accum = 1 if resident == 2 else 4
    ref = build(accum=accum, offload=True, tsm_resident_cores=resident)
    run(*ref[:3], steps=14, accum=accum)

    model, net, opt, mgr = build(accum=accum, offload=True, tsm_resident_cores=resident)
    run(model, net, opt, steps=7, accum=accum)
    buf_o, buf_n = io.BytesIO(), io.BytesIO()
    torch.save(opt.state_dict(), buf_o)  # state_dict pre-hook flushes; host views are saved as CPU tensors
    torch.save(net.state_dict(), buf_n)

    model2, net2, opt2, mgr2 = build(accum=accum, offload=True, tsm_resident_cores=resident)
    net2.load_state_dict(torch.load(io.BytesIO(buf_n.getvalue()), weights_only=False))
    opt2.load_state_dict(torch.load(io.BytesIO(buf_o.getvalue()), weights_only=False))
    for g in mgr2.groups:
        for j, p in enumerate(g.params):
            st = opt2.state.get(p)
            if st:
                assert in_slabs(st["exp_avg"], g.state) and st["step"].data_ptr() == g.steps_host[j].data_ptr()
    run(model2, net2, opt2, steps=14, accum=accum, start=7)
    assert_same(snapshot(ref[1]), snapshot(net2))


# ----------------------------------------------------------------------------------------- grouped accumulation


def test_grouped_draw_keeps_window_interval_and_marginal():
    cfg = tsm_lokr.parse_config({"tsm_n_scales": "8,1", "tsm_grouped_accumulation": "True", "tsm_resident_cores": "1"})
    mgr = ExpertOffloadManager([[torch.nn.Parameter(torch.zeros(2))] for _ in range(8)], None, "cpu",
                               core_index=cfg.core_index, offload_states=False, resident_cores=8, grouped=True,
                               core_index_t=cfg.core_index_from_timestep)
    g = torch.Generator().manual_seed(0)

    def draw(B=3):
        sig = torch.sigmoid(torch.randn(B, generator=g) * 1.3)  # like timestep_sampling=sigmoid
        t = (sig * 1000).bfloat16()
        return torch.randn(B, 4, generator=g), t, sig.view(-1, 1)

    W = 12000
    pooled = []
    for w in range(W):
        mgr.window_core = None  # what the optimizer step post-hook does
        for micro in range(2):
            nmi, t, s = mgr.grouped_draw(lambda: draw(2))
            cores = {cfg.core_index_from_timestep(v) for v in t.float().tolist()}
            assert cores == {mgr.window_core}
            assert torch.equal((s.view(-1) * 1000).bfloat16(), t)  # rows stay consistent (same draw)
            pooled.extend(t.float().tolist())
    ref = (torch.sigmoid(torch.randn(200000, generator=torch.Generator().manual_seed(9)) * 1.3) * 1000).bfloat16().float()
    hist = lambda v: torch.histc(torch.tensor(v), bins=8, min=0, max=1000) / len(v)  # noqa: E731
    p_ref = hist(ref.tolist())
    sd = (p_ref * (1 - p_ref) / W).sqrt()  # one interval per window: W independent interval draws
    assert bool(torch.all((hist(pooled) - p_ref).abs() < 4.5 * sd + 0.002)), (hist(pooled), p_ref)


def test_swapped_out_experts_save_with_safetensors(tmp_path):
    """accelerate save_state writes the network state with safetensors' save_file, which refuses shared storage: the
    swapped-out experts (views into host slabs) must come out of state_dict() as their own tensors."""
    from safetensors.torch import load_file, save_file

    model, net, opt, mgr = build(accum=1, offload=True, tsm_resident_cores=2)
    run(model, net, opt, steps=3, accum=1)
    assert any(not g.resident for g in mgr.groups)
    sd = net.state_dict()
    f = str(tmp_path / "model.safetensors")
    save_file({k: v.contiguous() for k, v in sd.items()}, f)
    back = load_file(f)
    for n, p in net.named_parameters():
        assert torch.equal(back[n], p.detach()), n
    net.save_weights(str(tmp_path / "lora.safetensors"), None, {})  # dtype None: LyCORIS does not clone


# ------------------------------------------------------------------------------------------------ review additions


def test_accelerate_save_state_keeps_swapped_out_experts():
    """accelerate's save_state runs clean_state_dict_for_safetensors, which drops every tensor sharing storage with
    another one: without the state_dict post-hook the swapped-out experts (views into one host slab) would be lost."""
    import collections

    from accelerate import PartialState
    from accelerate.utils.other import clean_state_dict_for_safetensors

    PartialState()
    model, net, opt, mgr = build(accum=1, offload=True, tsm_resident_cores=2)
    run(model, net, opt, steps=3, accum=1)
    assert any(not g.resident for g in mgr.groups)
    sd = collections.OrderedDict(net.state_dict())
    kept = clean_state_dict_for_safetensors(collections.OrderedDict(sd))
    assert list(kept) == list(sd)
    for n, p in net.named_parameters():
        assert torch.equal(kept[n], p.detach()), n


def test_state_buffers_are_one_per_slab():
    """The step uploads each used 64 MiB slab into a GPU buffer of its own (pieces the caching allocator can carve out
    of the freed activations), not one buffer per core expert."""
    model, net, opt, mgr = build(accum=1, offload=True)
    seen = []

    def spy(o, a, k):
        for g in mgr._stepped:
            if g.dev_buf is not None:
                assert isinstance(g.dev_buf, list) and len(g.dev_buf) == len(g.state.slabs)
                for buf, used in zip(g.dev_buf, g.state.used):
                    assert (buf is None) == (used == 0) and (buf is None or buf.numel() == used)
                seen.append(g.index)

    opt.register_step_pre_hook(spy)  # runs after the manager's pre-hook
    run(model, net, opt, steps=3, accum=1)
    assert seen  # step 2 and 3 upload the states created at step 1


def test_step_hook_dependent_settings_refuse_other_update_paths():
    net = make_net(make_model(), tsm_resident_cores=2, tsm_offload_states=False)
    args = argparse.Namespace(gradient_accumulation_steps=1)

    class StepFunc(torch.optim.SGD):
        def step_func(self, loss):
            pass

    class ScheduleFree(torch.optim.AdamW):
        def train(self):
            pass

        def eval(self):
            pass

    with pytest.raises(ValueError, match="step_func"):
        laf.attach_tsm_offload(net, StepFunc(net.parameters(), lr=0.1), args, torch.device("cpu"), batch_size=1)
    with pytest.raises(ValueError, match="schedule-free"):
        laf.attach_tsm_offload(net, ScheduleFree(net.parameters(), lr=0.1, fused=True), args, torch.device("cpu"),
                               batch_size=1)
    net2 = make_net(make_model(), tsm_resident_cores=1, tsm_grouped_accumulation=True, tsm_offload_states=False)
    with pytest.raises(ValueError, match="step_func"):
        laf.attach_tsm_offload(net2, StepFunc(net2.parameters(), lr=0.1), args, torch.device("cpu"), batch_size=1,
                               grouped_supported=True)
    net3 = make_net(make_model())
    with pytest.raises(ValueError, match="amsgrad"):
        laf.attach_tsm_offload(net3, torch.optim.AdamW(net3.parameters(), lr=0.1, amsgrad=True), args,
                               torch.device("cpu"), batch_size=1)
    # all cores resident and the states on the GPU: nothing follows optimizer.step(), any optimizer works
    net4 = make_net(make_model(), tsm_offload_states=False)
    assert laf.attach_tsm_offload(net4, StepFunc(net4.parameters(), lr=0.1), args, torch.device("cpu")) is None
