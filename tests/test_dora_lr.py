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
