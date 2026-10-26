"""T-LoRA mask exponent: tlora_mask_alpha and tlora_alpha are the same setting on both T-LoRA paths.

Before, algo=tlora read only tlora_mask_alpha and the LoCon-flag path only tlora_alpha, so a config written with the
other spelling (config_anima29_tlora.toml used tlora_alpha with algo=tlora) was silently ignored.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m pytest tests/test_tlora_mask_alpha_args.py -q
"""
import types

import pytest


def _setup(net_kwargs):
    import train_network

    trainer = train_network.NetworkTrainer()
    scheduler = types.SimpleNamespace(config=types.SimpleNamespace(num_train_timesteps=1000))
    trainer.setup_tlora_masking(net_kwargs, 64, scheduler)
    return trainer


@pytest.mark.parametrize("key", ["tlora_mask_alpha", "tlora_alpha"])
@pytest.mark.parametrize("algo_kwargs", [{"algo": "tlora"}, {"algo": "locon", "use_timestep_mask": "True"}])
def test_both_spellings_reach_the_mask(key, algo_kwargs):
    trainer = _setup({**algo_kwargs, "tlora_min_rank": "16", key: "2.5"})
    assert trainer.tlora_enabled
    assert trainer.tlora_mask_alpha == 2.5
    assert trainer.tlora_min_rank == 16


def test_default_is_linear():
    assert _setup({"algo": "tlora"}).tlora_mask_alpha == 1.0


def test_conflicting_spellings_are_rejected():
    with pytest.raises(ValueError):
        _setup({"algo": "tlora", "tlora_mask_alpha": "1.0", "tlora_alpha": "2.0"})


def test_same_value_under_both_spellings_is_fine():
    assert _setup({"algo": "tlora", "tlora_mask_alpha": "2", "tlora_alpha": "2.0"}).tlora_mask_alpha == 2.0
