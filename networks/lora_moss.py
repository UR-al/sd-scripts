# LoRA network for MOSS-Image (networks.lora_anima with MOSS target modules).
# Targets every Linear inside MossBlock (layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}) -> keys
# lora_unet_layers_N_proj_q.lora_down.weight etc., which ComfyUI maps onto the MOSS model. The text refiner blocks
# (text_embedder.blocks.*) are excluded by lora_anima's default exclude ("_embedder"); add them with
# network_args include_patterns=['text_embedder\.blocks\..*'] if wanted.

import contextlib

from networks import lora_anima
from networks.lora_anima import LoRAInfModule, LoRAModule  # noqa: F401 (re-exported for tools importing this module)

MOSS_TARGET_REPLACE_MODULE = ["MossBlock"]
MOSS_TEXT_TARGET_REPLACE_MODULE = ["MossTextBlock"]


@contextlib.contextmanager
def _moss_targets():
    cls = lora_anima.LoRANetwork
    saved = (cls.ANIMA_TARGET_REPLACE_MODULE, cls.ANIMA_ADAPTER_TARGET_REPLACE_MODULE)
    cls.ANIMA_TARGET_REPLACE_MODULE = list(MOSS_TARGET_REPLACE_MODULE)
    cls.ANIMA_ADAPTER_TARGET_REPLACE_MODULE = list(MOSS_TEXT_TARGET_REPLACE_MODULE)  # train_llm_adapter=true
    try:
        yield
    finally:
        cls.ANIMA_TARGET_REPLACE_MODULE, cls.ANIMA_ADAPTER_TARGET_REPLACE_MODULE = saved


def create_network(multiplier, network_dim, network_alpha, vae, text_encoders, unet, neuron_dropout=None, **kwargs):
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
        network = lora_anima.LoRANetwork(text_encoders, unet, multiplier=multiplier, modules_dim=modules_dim,
                                         modules_alpha=modules_alpha, module_class=module_class,
                                         train_llm_adapter=train_text)
    return network, weights_sd
