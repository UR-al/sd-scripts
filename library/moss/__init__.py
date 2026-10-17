"""MOSS-Image (OpenMOSS) LoRA / LyCORIS training support for this sd-scripts fork.

Model: OpenMOSS-Team/MOSS-Image-Base (Apache-2.0). The DiT port in ``models.py`` keeps the official module names
(``layers.N.proj.q`` ...) so checkpoints load strictly and saved LoRA keys are ``lora_unet_layers_N_proj_q`` (ComfyUI
maps them through its generic ``lora_unet_`` rule). Portions of ``models.py`` and ``vae.py`` are adapted from the
official ``pipeline_moss_image.py`` (Copyright OpenMOSS Team, Apache License 2.0).
"""
