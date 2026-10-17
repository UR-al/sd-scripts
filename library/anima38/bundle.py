# Anima-3.8B v1.1 bundle helpers: detection, DiT/connector split, connector config and building.
#
# Reproduces forge_sam3_extension/sam3ext/anima38/files.py ``bundle_metadata`` and runtime.py
# ``_load_v2_models`` (config parsing from ``anima_v2_adapter_*`` metadata, parameter_bank renames,
# strict load) without Forge (forge-anima-3.8B, MIT; see LICENSE-forge-anima-3.8B.txt).
# Model files are only ever opened read-only (safetensors ``safe_open``).

from __future__ import annotations

import json
import logging
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch

from .semantic_v2 import DEFAULT_LAYER_INDICES, V2_ARCHITECTURE, QualityAnchoredSemanticConnectorV2

logger = logging.getLogger(__name__)

BUNDLE_ARCHITECTURE = "anima_3_8b_semantic_connector_v2_bundle"
BUNDLE_FORMAT = "1"
CONNECTOR_PREFIX = "net.anima_v2_connector."
ADAPTER_METADATA_PREFIX = "anima_v2_adapter_"

# runtime.py: checkpoint name -> module name (the bare parameters live in small "parameter_bank" modules)
PARAMETER_BANK_RENAMES = {
    "quality_anchor.layer_mix_logits": "quality_anchor.parameter_bank.layer_mix_logits",
    "semantic_resampler.query_tokens": "semantic_resampler.parameter_bank.query_tokens",
    "semantic_resampler.layer_embeddings": "semantic_resampler.parameter_bank.layer_embeddings",
}


def bundle_metadata_matches(metadata: Optional[Mapping[str, str]]) -> bool:
    """files.bundle_metadata: architecture == anima_3_8b_semantic_connector_v2_bundle and format == "1"."""
    if not metadata:
        return False
    return metadata.get("architecture") == BUNDLE_ARCHITECTURE and metadata.get("anima_v2_bundle_format") == BUNDLE_FORMAT


def connector_prefixes(metadata: Optional[Mapping[str, str]] = None) -> Tuple[str, ...]:
    """Key prefixes the connector can have: the bundle's (``net.anima_v2_connector.``), and the forms
    left by the trainer's ``_rename_hook`` (``net.`` / ``model.diffusion_model.`` stripped) or a ComfyUI
    style ``model.diffusion_model.`` prefix."""
    prefix = (metadata or {}).get("anima_v2_connector_prefix", CONNECTOR_PREFIX) or CONNECTOR_PREFIX
    bare = prefix[len("net.") :] if prefix.startswith("net.") else prefix
    candidates = (prefix, "net." + bare, "model.diffusion_model." + bare, bare)
    return tuple(dict.fromkeys(candidates))


def _connector_prefix_of(key: str, prefixes: Iterable[str]) -> Optional[str]:
    for prefix in prefixes:
        if key.startswith(prefix):
            return prefix
    return None


def is_anima38_v2_bundle(keys: Optional[Iterable[str]] = None, metadata: Optional[Mapping[str, str]] = None) -> bool:
    """True if the safetensors metadata marks an Anima-3.8B v1.1 connector bundle (Forge's test), or if the
    keys contain Semantic Connector v2 tensors (with or without the ``net.`` prefix)."""
    if bundle_metadata_matches(metadata):
        return True
    if keys is None:
        return False
    prefixes = connector_prefixes(metadata)
    return any(_connector_prefix_of(key, prefixes) is not None for key in keys)


def read_bundle_metadata(path: str) -> Optional[Dict[str, str]]:
    """The file's metadata if it is a v2 bundle (files.bundle_metadata), else None. Read-only."""
    from safetensors import safe_open

    try:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
    except Exception:
        return None
    return dict(metadata) if bundle_metadata_matches(metadata) else None


def adapter_metadata(metadata: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """runtime.py _adapter_metadata: ``anima_v2_adapter_*`` keys with the prefix removed."""
    return {name[len(ADAPTER_METADATA_PREFIX) :]: value for name, value in (metadata or {}).items() if name.startswith(ADAPTER_METADATA_PREFIX)}


def connector_config_from_metadata(metadata: Optional[Mapping[str, str]], require_architecture: bool = True) -> Dict[str, int]:
    """Constructor kwargs for QualityAnchoredSemanticConnectorV2, parsed like runtime.py ``_load_v2_models``.

    Raises if the declared adapter architecture is not Semantic Connector v2 (unless
    ``require_architecture`` is False, e.g. for a keys-only bundle without metadata; then the reference
    defaults are used for missing values).
    """
    values = adapter_metadata(metadata)
    if require_architecture and values.get("architecture") != V2_ARCHITECTURE:
        raise RuntimeError("The bundled connector is not Semantic Connector v2.")
    return {
        "num_queries": int(values.get("semantic_query_tokens", "64")),
        "resampler_blocks": int(values.get("semantic_resampler_blocks", "6")),
        "resampler_dim": int(values.get("semantic_resampler_dim", "2048")),
        "resampler_heads": int(values.get("semantic_resampler_heads", "16")),
        "mlp_hidden_dim": int(values.get("semantic_resampler_mlp_hidden_dim", "5632")),
    }


def declared_layer_indices(metadata: Optional[Mapping[str, str]]) -> Optional[Tuple[int, ...]]:
    """``anima_v2_adapter_layer_indices`` from the metadata, or None. Forge ignores it and always uses
    (7, 15, 23, 31); callers should follow Forge and only warn on a difference."""
    raw = adapter_metadata(metadata).get("layer_indices")
    if raw is None:
        return None
    try:
        return tuple(int(index) for index in json.loads(raw))
    except (TypeError, ValueError):
        return None


def rename_connector_state(connector_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Apply runtime.py's parameter_bank renames in place (KeyError if a renamed tensor is missing, as in
    the reference) and return the dict."""
    for source, target in PARAMETER_BANK_RENAMES.items():
        if source not in connector_state:
            raise KeyError(f"connector tensor '{source}' missing from the bundle")
        connector_state[target] = connector_state.pop(source)
    return connector_state


def _connector_info(metadata: Optional[Mapping[str, str]], prefix: Optional[str]) -> Dict[str, object]:
    is_bundle = bundle_metadata_matches(metadata)
    declared = declared_layer_indices(metadata)
    if declared is not None and declared != tuple(DEFAULT_LAYER_INDICES):
        logger.warning(
            f"bundle declares Qwen3.5 layers {declared}; using {tuple(DEFAULT_LAYER_INDICES)} like the Forge runtime"
        )
    return {
        "is_bundle": is_bundle,
        "config": connector_config_from_metadata(metadata, require_architecture=is_bundle),
        "layer_indices": tuple(DEFAULT_LAYER_INDICES),
        "declared_layer_indices": declared,
        "prefix": prefix,
        "metadata": dict(metadata or {}),
    }


def split_anima38_bundle(
    state_dict: Mapping[str, torch.Tensor], metadata: Optional[Mapping[str, str]] = None
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, object]]:
    """Split an Anima-3.8B v1.1 bundle state dict into (dit_state_dict, connector_state_dict, info).

    - ``dit_state_dict``: every other tensor, same objects (no copies), keys unchanged.
    - ``connector_state_dict``: connector tensors with the prefix removed and the parameter_bank renames
      applied, ready for ``QualityAnchoredSemanticConnectorV2.load_state_dict(..., strict=True)``.
    - ``info``: ``config`` (constructor kwargs from the metadata), ``layer_indices`` (7, 15, 23, 31),
      ``declared_layer_indices``, ``prefix`` found, ``is_bundle`` (metadata test), raw ``metadata``.

    Works on the raw file keys (``net.anima_v2_connector.*``) and on keys already renamed by the trainer's
    ``_rename_hook`` (``anima_v2_connector.*``). Raises ValueError if no connector tensors are present.
    """
    prefixes = connector_prefixes(metadata)
    dit_state: Dict[str, torch.Tensor] = {}
    connector_state: Dict[str, torch.Tensor] = {}
    found = set()
    for key, value in state_dict.items():
        prefix = _connector_prefix_of(key, prefixes)
        if prefix is None:
            dit_state[key] = value
        else:
            found.add(prefix)
            connector_state[key[len(prefix) :]] = value
    if not connector_state:
        raise ValueError("no Semantic Connector v2 tensors (anima_v2_connector.*) in this state dict")
    if len(found) > 1:
        raise ValueError(f"connector tensors use several prefixes: {sorted(found)}")
    rename_connector_state(connector_state)
    return dit_state, connector_state, _connector_info(metadata, next(iter(found)))


def load_anima38_connector_state(bundle_path: str, dtype: Optional[torch.dtype] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """Read only the connector tensors from a bundle file (read-only ``safe_open``), like runtime.py.

    Returns ``(connector_state_dict, info)`` with the same meaning as ``split_anima38_bundle``. ``dtype``
    optionally converts the tensors (they are BF16 in the file).
    """
    from safetensors import safe_open

    with safe_open(bundle_path, framework="pt", device="cpu") as checkpoint:
        metadata = dict(checkpoint.metadata() or {})
        prefixes = connector_prefixes(metadata)
        state = {}
        for key in checkpoint.keys():
            if _connector_prefix_of(key, prefixes) is not None:
                tensor = checkpoint.get_tensor(key)
                state[key] = tensor if dtype is None else tensor.to(dtype)
    if not state:
        raise RuntimeError("The selected v2 bundle contains no connector tensors.")
    _, connector_state, info = split_anima38_bundle(state, metadata)
    return connector_state, info


def build_semantic_connector(
    native_adapter: torch.nn.Module,
    connector_state_dict: Mapping[str, torch.Tensor],
    config: Optional[Mapping[str, int]] = None,
    layer_indices=DEFAULT_LAYER_INDICES,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> QualityAnchoredSemanticConnectorV2:
    """Build the frozen connector around ``native_adapter`` (the DiT's own trainer ``LLMAdapter``, shared,
    not copied) and load ``connector_state_dict`` strictly (runtime.py raises on any mismatch too).

    The connector is built on the meta device and takes the state tensors by assignment; ``dtype``/``device``
    then convert it (storage is BF16; Forge computes it in the native adapter's embedding dtype). Returns
    the connector in eval mode with ``requires_grad`` False everywhere.
    """
    with torch.device("meta"):
        connector = QualityAnchoredSemanticConnectorV2(native_adapter=native_adapter, layer_indices=layer_indices, **dict(config or {}))
    connector.requires_grad_(False)
    incompatible = connector.load_state_dict(dict(connector_state_dict), strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Bundled v2 connector mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    if device is not None or dtype is not None:
        connector.to(device=device, dtype=dtype)
    return connector.eval()
