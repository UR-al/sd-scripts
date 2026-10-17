# LyCORIS presets for MOSS-Image, registered at runtime (site-packages untouched).
# kohya.py imports PRESET from lycoris.config (the same dict object), so entries added before the network module is
# imported by train_network are visible to ``network_args = ["preset=moss", ...]``.
# Targets the six Linears of every joint block (layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}); saved keys are
# lora_unet_layers_N_proj_q... which ComfyUI maps onto the MOSS model. "moss-text" adds the two text refiner blocks.
# exclude_name is always explicit: apply_preset mutates class attributes, so a previous preset's excludes must not leak.

MOSS_LYCORIS_PRESETS = {
    "moss": {"enable_conv": False, "unet_target_module": ["MossBlock"], "unet_target_name": [],
             "text_encoder_target_module": [], "text_encoder_target_name": [], "exclude_name": []},
    "moss-text": {"enable_conv": False, "unet_target_module": ["MossBlock", "MossTextBlock"], "unet_target_name": [],
                  "text_encoder_target_module": [], "text_encoder_target_name": [], "exclude_name": []},
}


def register_moss_lycoris_presets() -> bool:
    try:
        from lycoris import config as lycoris_config
    except ImportError:
        return False
    for name, preset in MOSS_LYCORIS_PRESETS.items():
        lycoris_config.PRESET.setdefault(name, preset)
    return True
