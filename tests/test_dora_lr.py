"""CPU tests for --learning_rate_DoRA (library/dora_lr.py): DoRA magnitude parameters get their own optimizer group and lr.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m pytest tests/test_dora_lr.py
"""
import argparse
import os

import pytest
import torch

from library import dora_lr
from library.moss.lycoris_presets import register_moss_lycoris_presets
from library.moss.models import MossTransformer

TINY = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=48, freq_dim=16, latent_channels=128, rope_base=10000.0)


def _network(dora=True):
    register_moss_lycoris_presets()
    from lycoris import kohya

    torch.manual_seed(0)
    dit = MossTransformer(**TINY)
    net = kohya.create_network(1.0, 10000, 10000, None, [], dit, algo="lokr", preset="moss", factor=4,
                               full_matrix=True, dora_wd=dora, wd_on_output=False)
    net.apply_to([], dit, apply_text_encoder=False, apply_unet=True)  # registers the modules, as train_network does
    results = net.prepare_optimizer_params(text_encoder_lr=None, unet_lr=None, learning_rate=5e-5)
    params, desc = results if isinstance(results, tuple) else (results, None)
    return net, params, desc


def test_flag_parsing_and_validation():
    import train_network

    parser = train_network.setup_parser()
    assert parser.parse_args(["--learning_rate_DoRA", "1e-4"]).learning_rate_DoRA == 1e-4
    assert parser.parse_args(["--learning_rate_dora", "2e-4"]).learning_rate_DoRA == 2e-4
    assert parser.parse_args([]).learning_rate_DoRA is None
    assert dora_lr.dora_learning_rate(argparse.Namespace(learning_rate_DoRA=None, learning_rate_dora="3e-4")) == 3e-4  # TOML
    with pytest.raises(ValueError):
        dora_lr.dora_learning_rate(argparse.Namespace(learning_rate_DoRA=0.0))


def test_dora_params_get_their_own_group_and_lr():
    net, params, desc = _network()
    before = [p for g in params for p in g["params"]]
    groups, descriptions = dora_lr.split_dora_param_groups(net, params, desc, 1e-4, default_lr=5e-5)
    after = [p for g in groups for p in g["params"]]
    assert len(after) == len(before) and {id(p) for p in after} == {id(p) for p in before}  # same params, no duplicates
    assert len(groups) == len(params) + 1 and len(descriptions) == len(groups)
    dora_groups = [g for g in groups if all(getattr(p, "_is_dora_scale", False) for p in g["params"])]
    assert len(dora_groups) == 1 and len(dora_groups[0]["params"]) == 18  # 3 blocks x 6 Linears
    assert float(dora_groups[0]["lr"]) == pytest.approx(1e-4) and "initial_lr" not in dora_groups[0]
    assert all(float(g["lr"]) == pytest.approx(5e-5) for g in groups if g is not dora_groups[0])
    assert descriptions[-1].endswith("_dora")

    # effective lr: the first AdamW step moves every element by ~lr of its own group
    optimizer = torch.optim.AdamW(groups, lr=5e-5, weight_decay=0.0)
    snapshot = {id(p): p.detach().clone() for p in after}
    for p in after:
        p.grad = torch.randn_like(p)
    optimizer.step()
    for g in groups:
        expected = float(g["lr"])
        for p in g["params"]:
            step = (p.detach() - snapshot[id(p)]).abs()
            assert float(step.max()) == pytest.approx(expected, rel=1e-3), (g.get("name"), expected)


def test_scheduler_scales_the_dora_group_too():
    net, params, desc = _network()
    groups, _ = dora_lr.split_dora_param_groups(net, params, desc, 1e-4, default_lr=5e-5)
    optimizer = torch.optim.AdamW(groups, lr=5e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 10))  # warmup
    lrs = scheduler.get_last_lr()
    assert sorted(lrs) == pytest.approx(sorted([5e-6] * (len(groups) - 1) + [1e-5]))


def test_no_flag_or_no_dora_is_a_no_op():
    net, params, desc = _network()
    assert dora_lr.split_dora_param_groups(net, params, desc, None) == (params, desc)
    plain, plain_params, plain_desc = _network(dora=False)
    assert dora_lr.split_dora_param_groups(plain, plain_params, plain_desc, 1e-4) == (plain_params, plain_desc)
    assert dora_lr.metadata(argparse.Namespace(learning_rate_DoRA=None)) == {"ss_weight_decay_DoRA": "0.0"}
    assert dora_lr.metadata(argparse.Namespace(learning_rate_DoRA=1e-4, weight_decay_DoRA=1e-3)) == {
        "ss_learning_rate_DoRA": "0.0001", "ss_weight_decay_DoRA": "0.001"}


def test_dora_magnitudes_get_no_weight_decay_by_default():
    import train_network

    args = train_network.setup_parser().parse_args([])
    assert dora_lr.dora_weight_decay(args) == 0.0
    with pytest.raises(ValueError):
        dora_lr.dora_weight_decay(argparse.Namespace(weight_decay_DoRA=-1.0))
    net, params, desc = _network()
    groups, _ = dora_lr.split_dora_param_groups(net, params, desc, None, default_lr=5e-5,
                                                weight_decay=dora_lr.dora_weight_decay(args), decay_keys=("weight_decay",))
    dora_group = [g for g in groups if all(getattr(p, "_is_dora_scale", False) for p in g["params"])][0]
    assert dora_group["weight_decay"] == 0.0 and float(dora_group["lr"]) == pytest.approx(5e-5)  # lr unchanged without the flag
    optimizer = torch.optim.AdamW(groups, lr=5e-5, weight_decay=0.1)
    every = [p for g in groups for p in g["params"]]
    snapshot = {id(p): p.detach().clone() for p in every}
    for p in every:
        p.grad = torch.zeros_like(p)
    optimizer.step()  # zero gradients: only weight decay moves parameters
    for g in groups:
        for p in g["params"]:
            moved = not torch.equal(p.detach(), snapshot[id(p)])
            if g is dora_group:
                assert not moved
            elif snapshot[id(p)].abs().sum() > 0:
                assert moved


def test_lycoris_string_booleans_become_bools():
    from library.lycoris_args import fix_lycoris_bool_kwargs

    fixed = fix_lycoris_bool_kwargs("lycoris.kohya", {"decompose_both": "False", "ggpo_conv": " true ", "algo": "lokr"})
    assert fixed == {"decompose_both": False, "ggpo_conv": True, "algo": "lokr"}
    assert fix_lycoris_bool_kwargs("networks.lora_moss", {"decompose_both": "False"}) == {"decompose_both": "False"}
    with pytest.raises(ValueError):
        fix_lycoris_bool_kwargs("lycoris.kohya", {"decompose_both": "maybe"})

    register_moss_lycoris_presets()
    from lycoris import kohya

    def w1_decomposed(**kw):
        torch.manual_seed(0)
        dit = MossTransformer(**TINY)
        net = kohya.create_network(1.0, 2, 2, None, [], dit, algo="lokr", preset="moss", factor=-1, full_matrix="False", **kw)
        return any(hasattr(lora, "lokr_w1_a") for lora in net.unet_loras)

    assert w1_decomposed(decompose_both="False")  # the LyCORIS bug: the string "False" is truthy
    assert not w1_decomposed(**fix_lycoris_bool_kwargs("lycoris.kohya", {"decompose_both": "False"}))


def test_prepare_optimizer_applies_the_split():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "library", "train_util.py"), encoding="utf-8") as f:
        source = f.read()
    hook = source.index("dora_lr.split_dora_param_groups(")
    assert hook < source.index("get_optimizer(args, trainable_params, optimizer_kwargs)", hook)


# --- prepare_optimizer reads the signature of the optimizer get_optimizer builds (AdamW8bit, adamw, ... not only torch.optim)

def _args(*extra):
    import train_network

    return train_network.setup_parser().parse_args(["--network_module", "lycoris.kohya", "--learning_rate", "5e-5", *extra])


def _prepare(args):
    import types

    from library import train_util

    net = _network()[0]
    optimizer = train_util.prepare_optimizer(args, net, types.SimpleNamespace(device=torch.device("cpu")))[2]
    return net, optimizer


def _dora_group(optimizer):
    groups = [g for g in optimizer.param_groups if all(getattr(p, "_is_dora_scale", False) for p in g["params"])]
    return groups[0] if len(groups) == 1 else None


def test_resolve_optimizer_class_is_the_class_get_optimizer_builds():
    import inspect
    import re

    from library import train_util

    # every name get_optimizer handles itself is in the table resolve_optimizer_class follows
    names = {n.lower() for n in re.findall(r'optimizer_type == "(\w+)"\.lower\(\)', inspect.getsource(train_util.get_optimizer))}
    assert names and names <= set(train_util._OPTIMIZER_TYPE_CLASSES), names - set(train_util._OPTIMIZER_TYPE_CLASSES)

    checked = []
    for name in ["", "AdamW", "ADAMW", "RMSprop", "torch.optim.RMSprop", *train_util._OPTIMIZER_TYPE_CLASSES]:
        args = _args("--optimizer_type", name)
        try:
            resolved = train_util.resolve_optimizer_class(args)
        except ImportError:  # package not installed: get_optimizer cannot build it either
            continue
        built = train_util.get_optimizer(args, [torch.nn.Parameter(torch.zeros(4, 4))], {})[2]
        assert type(built) is resolved, (name, type(built), resolved)
        checked.append(name)
    assert {"adamw", "sgdnesterov"} <= set(checked)
    assert train_util.resolve_optimizer_class(_args("--optimizer_type", "adamw")) is torch.optim.AdamW
    with pytest.raises(AttributeError):
        train_util.resolve_optimizer_class(_args("--optimizer_type", "NoSuchOptimizer"))  # get_optimizer fails the same way


def test_wrapper_optimizer_reads_the_base_optimizer_signature(monkeypatch):
    import sys
    import types

    from library import train_util

    class ScheduleFreeWrapper(torch.optim.Optimizer):
        def __init__(self, params, base_optimizer, **kwargs):  # passes kwargs on to base_optimizer
            super().__init__(params, {})

    monkeypatch.setitem(sys.modules, "fake_wrappers", types.SimpleNamespace(ScheduleFreeWrapper=ScheduleFreeWrapper))
    args = _args("--optimizer_type", "fake_wrappers.ScheduleFreeWrapper")
    assert train_util.resolve_optimizer_class(args) is ScheduleFreeWrapper
    parameters = train_util.optimizer_init_parameters(args, {"base_optimizer_type": "torch.optim.AdamW"})
    assert "weight_decay" in parameters and "amsgrad" in parameters


@pytest.mark.parametrize("optimizer_type", ["adamw", "AdamW8bit"])
def test_dora_group_gets_weight_decay_DoRA_with_optimizer_aliases(optimizer_type):
    if optimizer_type == "AdamW8bit":
        pytest.importorskip("bitsandbytes")
    # no weight_decay in optimizer_args: before the fix the alias was not found in torch.optim, so the DoRA magnitudes
    # got the optimizer's default 0.01 (AdamW / AdamW8bit)
    for extra, lr, wd in (((), 5e-5, 0.0), (("--learning_rate_DoRA", "1e-4"), 1e-4, 0.0),
                          (("--weight_decay_DoRA", "0.05"), 5e-5, 0.05)):
        net, optimizer = _prepare(_args("--optimizer_type", optimizer_type, *extra))
        dora_group = _dora_group(optimizer)
        assert dora_group is not None, extra
        assert dora_group["weight_decay"] == wd and float(dora_group["lr"]) == pytest.approx(lr), extra
        assert all(g["weight_decay"] == 0.01 for g in optimizer.param_groups if g is not dora_group)


def test_unreadable_optimizer_signature_warns(monkeypatch, caplog):
    import logging

    from library import train_util

    def unreadable(args, optimizer_kwargs):
        raise RuntimeError("no signature")

    monkeypatch.setattr(train_util, "optimizer_init_parameters", unreadable)
    args = _args("--optimizer_type", "AdamW", "--learning_rate_DoRA", "1e-4")
    with caplog.at_level(logging.INFO):
        net, optimizer = _prepare(args)
    messages = [r.getMessage() for r in caplog.records if r.name == dora_lr.logger.name]
    assert any("could not read the optimizer's arguments" in m for m in messages)
    assert not any("optimizer has none" in m for m in messages)
    assert _dora_group(optimizer)["weight_decay"] == 0.01  # not set: AdamW's own default, as the warning says
    assert dora_lr.metadata(args)["ss_weight_decay_DoRA"] == "optimizer default"  # not the unapplied weight_decay_DoRA

    caplog.clear()
    with caplog.at_level(logging.INFO):
        net, optimizer = _prepare(_args("--optimizer_type", "AdamW"))  # no DoRA lr: nothing to split on
    assert len(optimizer.param_groups) == 1
    assert any("could not read the optimizer's arguments" in r.getMessage() for r in caplog.records)


# --- --resume of a state saved before the DoRA group existed keeps that state's layout

@pytest.fixture
def cpu_accelerator():
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState

    yield lambda: Accelerator(cpu=True)
    AcceleratorState._reset_state(True)


def _train_and_save(accelerator, args, path):
    net, optimizer = _prepare(args)
    prepared = accelerator.prepare(optimizer)
    for p in net.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)
    prepared.step()
    accelerator.save_state(path)
    return optimizer


USER_OPTIMIZER = ("--optimizer_type", "AdamW", "--optimizer_args", "weight_decay=0.01", "betas=(0.9, 0.99)", "eps=1e-8")


def test_group_layout_helpers(tmp_path):
    assert dora_lr.keep_pre_split_layout([54], [36, 18], [54])
    assert not dora_lr.keep_pre_split_layout([54], [36, 18], [36, 18])  # saved with the DoRA group: split as usual
    assert not dora_lr.keep_pre_split_layout([54], [36, 18], [50])  # another network: load_state reports it
    assert not dora_lr.keep_pre_split_layout([54], [36, 18], None)
    assert dora_lr.group_sizes([torch.zeros(1), torch.zeros(1)]) == [2]
    assert dora_lr.group_sizes([{"params": [torch.zeros(1)]}, {"params": [torch.zeros(1)] * 3}]) == [1, 3]
    assert dora_lr.group_sizes([{"params": iter([torch.zeros(1)])}]) is None
    assert dora_lr.saved_group_sizes(str(tmp_path)) is None  # no optimizer.bin


def test_resume_from_a_state_saved_without_the_dora_group(tmp_path, monkeypatch, cpu_accelerator, caplog):
    import logging

    old = str(tmp_path / "old-state")
    with monkeypatch.context() as m:  # the code before the DoRA group: one group with the magnitudes in it
        m.setattr(dora_lr, "split_dora_param_groups", lambda network, params, descriptions, *a, **k: (params, descriptions))
        saved_optimizer = _train_and_save(cpu_accelerator(), _args(*USER_OPTIMIZER), old)
    assert dora_lr.saved_group_sizes(old) == [54]

    args = _args(*USER_OPTIMIZER, "--resume", old, "--learning_rate_DoRA", "1e-4")
    with caplog.at_level(logging.WARNING):
        net, optimizer = _prepare(args)
    assert [len(g["params"]) for g in optimizer.param_groups] == [54]  # the split would give [36, 18]
    assert any("predates the DoRA optimizer group" in r.getMessage() for r in caplog.records)
    assert args.learning_rate_DoRA is None and dora_lr.metadata(args) == {}  # not applied, not recorded

    accelerator = cpu_accelerator()
    accelerator.prepare(optimizer)
    accelerator.load_state(old)  # fails ("different number of parameter groups") with the split layout
    saved = torch.load(os.path.join(old, "optimizer.bin"), map_location="cpu", weights_only=True)["state"]
    params = optimizer.param_groups[0]["params"]
    assert all(torch.equal(saved[i]["exp_avg"], optimizer.state[p]["exp_avg"]) for i, p in enumerate(params))
    assert optimizer.param_groups[0]["weight_decay"] == saved_optimizer.param_groups[0]["weight_decay"] == 0.01


def test_resume_from_a_state_saved_with_the_dora_group_is_unchanged(tmp_path, cpu_accelerator):
    new = str(tmp_path / "new-state")
    saved_optimizer = _train_and_save(cpu_accelerator(), _args(*USER_OPTIMIZER), new)
    assert dora_lr.saved_group_sizes(new) == [36, 18]

    args = _args(*USER_OPTIMIZER, "--resume", new)
    net, optimizer = _prepare(args)
    assert [len(g["params"]) for g in optimizer.param_groups] == [36, 18] and _dora_group(optimizer)["weight_decay"] == 0.0
    assert dora_lr.metadata(args) == {"ss_weight_decay_DoRA": "0.0"}
    _dora_group(optimizer)["weight_decay"] = 0.5  # load_state restores the saved group values
    accelerator = cpu_accelerator()
    accelerator.prepare(optimizer)
    accelerator.load_state(new)
    assert _dora_group(optimizer)["weight_decay"] == 0.0 and _dora_group(saved_optimizer)["weight_decay"] == 0.0
