"""Shared helpers of tests/test_moss_ft_model.py, test_moss_ft_save_dit.py and test_moss_ft_state.py
(library/moss/ft_model.py, ft_state.py): a tiny MOSS DiT written as a Base-layout safetensors file (bare keys, BF16 and
rope.bands F32, sorted), the arguments the loader reads, and the heal layout (fused joint-block q/k/v). Not collected
by pytest (no test_ prefix)."""
import types
from typing import Dict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from library.moss.models import MossTransformer

# ffn_dim 96 and dim 64: every 2-D parameter spans several 2 KiB pieces of the tests' pipelines
TINY = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=48, freq_dim=16, latent_channels=128)


def base_state(seed: int = 0, dtype=torch.bfloat16) -> Dict[str, torch.Tensor]:
    """The tiny DiT's state dict as a Base file holds it: every parameter moved off its init (values not representable
    in bf16 when dtype is fp32), in `dtype`; rope.bands F32."""
    torch.manual_seed(seed)
    model = MossTransformer(**TINY)
    g = torch.Generator().manual_seed(seed + 100)
    state = {}
    for name, tensor in model.state_dict().items():
        if name == "rope.bands":
            state[name] = tensor.float().clone()
            continue
        value = tensor.float() + torch.randn(tensor.shape, generator=g) * 0.05
        state[name] = value.to(dtype)
    return state


def write_base(tmp_path, seed: int = 0, dtype=torch.bfloat16, name: str = "base.safetensors"):
    """(path, state) of a tiny Base-layout file."""
    state = base_state(seed, dtype)
    path = tmp_path / name
    save_file({k: state[k].contiguous() for k in sorted(state)}, str(path), metadata={"format": "pt"})
    return path, state


def load(path) -> Dict[str, torch.Tensor]:
    out = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            out[key] = f.get_tensor(key).clone()
    return out


def fuse_qkv(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """The heal file's layout: each joint block's q, k, v rows concatenated into layers.N.proj.qkv.weight."""
    fused = dict(state)
    for key in [k for k in state if k.startswith("layers.") and k.endswith(".proj.q.weight")]:
        prefix = key[:-len("q.weight")]
        fused[prefix + "qkv.weight"] = torch.cat([fused.pop(prefix + n + ".weight") for n in ("q", "k", "v")], 0)
    return fused


def args(**kw):
    """What load_moss_for_full_ft and the state code read (the defaults of moss_train.py's parser for the rest)."""
    values = dict(mixed_precision="bf16", moss_stream_fp32=True, moss_stream_scale=1.0, attn_mode=None, xformers=False,
                  split_attn=False, moss_ft_freeze="", optimizer_type="AdamW", optimizer_args=[],
                  fused_backward_pass=False, gradient_accumulation_steps=1, moss_ft_clip_mode="stale",
                  max_grad_norm=0.0, output_name="tiny", save_last_n_epochs_state=None, save_last_n_steps_state=None,
                  moss_ft_lr_phases=None, moss_ft_no_keep_phase_states=False)
    values.update(kw)
    return types.SimpleNamespace(**values)
