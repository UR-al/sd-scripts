# LoRA network for MOSS-Image (networks.lora_anima with MOSS target modules).
# Targets every Linear inside MossBlock (layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}) -> keys
# lora_unet_layers_N_proj_q.lora_down.weight etc., which ComfyUI maps onto the MOSS model. The two text refiner
# blocks (text_embedder.blocks.*, keys lora_unet_text_embedder_blocks_*) are added with network_args
# train_llm_adapter=true (alias: train_text_embedder=true, the upstream sd-scripts name). exclude_patterns works as
# usual (--moss_lora_exclude_blocks writes it). Note that in this fork lora_anima treats include_patterns as a
# whitelist: a list naming only the refiner blocks would drop every joint block, so do not use it for that.

import contextlib
from typing import List, Optional

from networks import lora_anima
from networks.lora_anima import LoRAInfModule, LoRAModule  # noqa: F401 (re-exported for tools importing this module)

MOSS_TARGET_REPLACE_MODULE = ["MossBlock"]
MOSS_TEXT_TARGET_REPLACE_MODULE = ["MossTextBlock"]
# lora_anima.create_network appends this pattern to every network. "_embedder" full-matches the refiner Linears too
# (text_embedder.blocks.N.qkv ...), so train_llm_adapter=true used to add nothing. The MOSS targets only contain the
# MossBlock / MossTextBlock Linears, so the term is not needed to keep the embedders out of the network.
ANIMA_DEFAULT_EXCLUDE = r".*(_modulation|_norm|_embedder|final_layer).*"
MOSS_DEFAULT_EXCLUDE = r".*(_modulation|_norm|final_layer).*"


class MossLoRANetwork(lora_anima.LoRANetwork):
    """lora_anima.LoRANetwork whose default exclude pattern leaves the text refiner blocks selectable."""

    def __init__(self, text_encoders, unet, *args, exclude_patterns: Optional[List[str]] = None, **kwargs):
        if exclude_patterns:
            exclude_patterns = [MOSS_DEFAULT_EXCLUDE if p == ANIMA_DEFAULT_EXCLUDE else p for p in exclude_patterns]
        super().__init__(text_encoders, unet, *args, exclude_patterns=exclude_patterns, **kwargs)


@contextlib.contextmanager
def _moss_targets():
    cls = lora_anima.LoRANetwork
    saved = (cls.ANIMA_TARGET_REPLACE_MODULE, cls.ANIMA_ADAPTER_TARGET_REPLACE_MODULE, lora_anima.LoRANetwork)
    cls.ANIMA_TARGET_REPLACE_MODULE = list(MOSS_TARGET_REPLACE_MODULE)
    cls.ANIMA_ADAPTER_TARGET_REPLACE_MODULE = list(MOSS_TEXT_TARGET_REPLACE_MODULE)  # train_llm_adapter=true
    lora_anima.LoRANetwork = MossLoRANetwork  # lora_anima.create_network builds the network through this module global
    try:
        yield
    finally:
        cls.ANIMA_TARGET_REPLACE_MODULE, cls.ANIMA_ADAPTER_TARGET_REPLACE_MODULE = saved[0], saved[1]
        lora_anima.LoRANetwork = saved[2]


def _normalize_adapter_flag(kwargs: dict) -> None:
    """train_llm_adapter (lora_anima) and train_text_embedder (upstream lora_generic) both turn the refiner on."""
    values = [kwargs.pop(key) for key in ("train_llm_adapter", "train_text_embedder") if key in kwargs]
    if values:
        kwargs["train_llm_adapter"] = "true" if any(str(v).strip().lower() == "true" for v in values) else "false"


def create_network(multiplier, network_dim, network_alpha, vae, text_encoders, unet, neuron_dropout=None, **kwargs):
    _normalize_adapter_flag(kwargs)
    with _moss_targets():
        return lora_anima.create_network(multiplier, network_dim, network_alpha, vae, text_encoders, unet,
                                         neuron_dropout=neuron_dropout, **kwargs)


def create_network_from_weights(multiplier, file, ae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
    """lora_anima.create_network_from_weights, but text-refiner adapters are recognised by their MOSS names
    (lora_unet_text_embedder_blocks_*), not by 'llm_adapter'."""
    import os

    import torch

    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
    modules_dim, modules_alpha, train_text = {}, {}, False
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            modules_dim[lora_name] = value.size()[0]
        if "text_embedder_blocks" in lora_name:
            train_text = True
    module_class = lora_anima.LoRAInfModule if for_inference else lora_anima.LoRAModule
    with _moss_targets():
        network = MossLoRANetwork(text_encoders, unet, multiplier=multiplier, modules_dim=modules_dim,
                                  modules_alpha=modules_alpha, module_class=module_class, train_llm_adapter=train_text)
    return network, weights_sd
