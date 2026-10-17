# Anima-3.8B v1.1 conditioning (Qwen3.5-4B + Semantic Connector v2) for sd-scripts.
#
# Ported from https://github.com/GumGum10/forge-anima-3.8B (MIT, (c) 2026 GumGum10 contributors;
# commit 59c27e5 as vendored in forge_sam3_extension/sam3ext/anima38). Full license text:
# LICENSE-forge-anima-3.8B.txt in this directory. The Qwen3.5 tokenizer (Apache-2.0) is read from a
# directory at runtime and not copied; no Forge (AGPL) code is included.
#
# Modules:
#   qwen35       Qwen3.5-4B hybrid encoder (FP8 storage, per-layer manual cast), semantic-layer capture
#   tokenizer    Qwen3.5 tokenizer wrapper (raw text, no special tokens, max 1024 tokens)
#   semantic_v2  frozen, timestep-aware Semantic Connector v2 built on the trainer's LLMAdapter
#   bundle       bundle detection, DiT/connector split, metadata config, connector building
#
# Training integration (anima_train_network.py --anima38_connector; imported on demand, not re-exported here):
#   training     options, validation, DiT conditioner hook (Anima.forward), Qwen3.5 layer packing
#   strategy     tokenize / encode / per-caption-variant TE caching with Qwen3.5 (*_anima38_te.npz)
#   sampling     training-time samples with per-step connector conditioning
#   trainer      AnimaNetworkTrainer mixin (make_anima38_trainer)
#   captions     canonical caption variant, caption digests (cache invalidation), Forge-style Qwen3.5 text

from .bundle import (
    BUNDLE_ARCHITECTURE,
    CONNECTOR_PREFIX,
    build_semantic_connector,
    bundle_metadata_matches,
    connector_config_from_metadata,
    is_anima38_v2_bundle,
    load_anima38_connector_state,
    read_bundle_metadata,
    split_anima38_bundle,
)
from .qwen35 import LAYER_INDICES, Qwen35HybridModel, encode_semantic_layers, load_qwen35_model
from .semantic_v2 import V2_ARCHITECTURE, QualityAnchoredSemanticConnectorV2
from .tokenizer import DEFAULT_TOKENIZER_DIR, Qwen35Tokenizer

__all__ = [
    "BUNDLE_ARCHITECTURE",
    "CONNECTOR_PREFIX",
    "DEFAULT_TOKENIZER_DIR",
    "LAYER_INDICES",
    "V2_ARCHITECTURE",
    "QualityAnchoredSemanticConnectorV2",
    "Qwen35HybridModel",
    "Qwen35Tokenizer",
    "build_semantic_connector",
    "bundle_metadata_matches",
    "connector_config_from_metadata",
    "encode_semantic_layers",
    "is_anima38_v2_bundle",
    "load_anima38_connector_state",
    "load_qwen35_model",
    "read_bundle_metadata",
    "split_anima38_bundle",
]
