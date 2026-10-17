# Anima-3.8B v1.1 training integration: options, validation, and the DiT conditioner hook.
#
# With --anima38_connector the frozen Semantic Connector v2 (semantic_v2.py) is attached to the Anima DiT as a
# plain attribute (``dit.anima38_conditioner``, not a registered submodule, so it stays out of state_dict(),
# parameters(), LyCORIS module search and ``.to()``). ``Anima.forward`` then calls the conditioner instead of
# ``_preprocess_text_embeds``, so the connector runs at every DiT call with that call's timesteps, exactly where
# Forge's runtime replaces the llm_adapter output (sam3ext/anima38/runtime.py ``patched_forward`` ->
# ``_expand_v2_context``). Without the option nothing here is used and the Anima path is unchanged.

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Optional, Sequence

import torch

from .bundle import CONNECTOR_PREFIX, connector_prefixes, is_anima38_v2_bundle
from .semantic_v2 import DEFAULT_LAYER_INDICES, QualityAnchoredSemanticConnectorV2

logger = logging.getLogger(__name__)

SAMPLE_NEGATIVE_CHOICES = ("connector", "native")
QWEN35_HIDDEN_SIZE = 2560
CONDITIONER_ATTRIBUTE = "anima38_conditioner"


# region options


def add_anima38_arguments(parser: argparse.ArgumentParser) -> None:
    """Anima-3.8B v1.1 options (anima_train_network.py). All are inert unless --anima38_connector is given."""
    parser.add_argument(
        "--anima38_connector",
        action="store_true",
        help="train on an Anima-3.8B v1.1 bundle with its frozen Semantic Connector v2 conditioning (Qwen3.5-4B features,"
        " timestep-aware; same as the Forge anima38 extension). Requires --cache_text_encoder_outputs and --qwen35_path",
    )
    parser.add_argument(
        "--qwen35_path",
        type=str,
        default=None,
        help="Qwen3.5-4B text encoder for --anima38_connector (qwen35_4b.safetensors). Loaded only while caching",
    )
    parser.add_argument(
        "--qwen35_tokenizer_dir",
        type=str,
        default=None,
        help="directory with the Qwen3.5 tokenizer.json (default: ANIMA38_QWEN35_TOKENIZER_DIR or the Forge extension's"
        " assets/qwen35_tokenizer)",
    )
    parser.add_argument(
        "--anima38_sample_negative",
        type=str,
        default="connector",
        choices=list(SAMPLE_NEGATIVE_CHOICES),
        help="conditioning of the negative prompt in training samples: 'connector' (Forge 'Anima38 negative' on) or"
        " 'native' (plain llm_adapter, Forge default)",
    )
    parser.add_argument(
        "--anima38_qwen35_batch_size",
        type=int,
        default=1,
        help="captions per Qwen3.5 forward while caching. 1 = one unpadded caption per call, exactly like Forge;"
        " larger is faster but pads the batch (bf16-level differences)",
    )


def _dataset_list(dataset_groups) -> List:
    datasets = []
    for group in dataset_groups or ():
        if group is None:
            continue
        datasets.extend(getattr(group, "datasets", [group]))
    return datasets


def validate_anima38_args(args, dataset_groups: Sequence = ()) -> None:
    """Refuse option combinations the Anima-3.8B path does not support. Raises ValueError with the reason.

    Called after the base Anima checks (which already refuse shuffle_caption / tag dropout without
    --cache_caption_variants when TE outputs are cached, and token_warmup_step).
    """
    if not getattr(args, "anima38_connector", False):
        return

    def fail(message: str):
        raise ValueError(f"--anima38_connector: {message}")

    if not getattr(args, "qwen35_path", None):
        fail("--qwen35_path (qwen35_4b.safetensors) is required")
    if not os.path.isfile(args.qwen35_path):
        fail(f"--qwen35_path not found: {args.qwen35_path}")
    if int(getattr(args, "anima38_qwen35_batch_size", 1) or 0) < 1:
        fail("--anima38_qwen35_batch_size must be >= 1")
    if getattr(args, "anima38_sample_negative", "connector") not in SAMPLE_NEGATIVE_CHOICES:
        fail(f"--anima38_sample_negative must be one of {SAMPLE_NEGATIVE_CHOICES}")
    if not getattr(args, "cache_text_encoder_outputs", False):
        fail(
            "Qwen3.5-4B features are cached per caption (the model is only loaded while caching), so"
            " --cache_text_encoder_outputs is required (add --cache_text_encoder_outputs_to_disk to keep them;"
            " with shuffle_caption / tag dropout also set --cache_caption_variants N>1)"
        )
    if getattr(args, "weighted_captions", False):
        fail("--weighted_captions is not supported (captions would be re-tokenized per step without Qwen3.5 features)")
    if getattr(args, "ileco", False) or getattr(args, "addift", False):
        fail("--ileco / --addift are not supported (their DiT calls have no Qwen3.5 features)")
    if getattr(args, "adaptive_timestep_sampling", False):
        fail("--adaptive_timestep_sampling is not supported (its model_fn calls the DiT without Qwen3.5 features)")
    if getattr(args, "network_train_text_encoder_only", False):
        fail("--network_train_text_encoder_only is not supported")

    datasets = _dataset_list(dataset_groups)
    for dataset in datasets:
        variants = int(getattr(dataset, "caption_aug_variant_count", 0) or 0)
        if variants > 1 and int(getattr(dataset, "aug_refresh_epochs", 0) or 0) > 0:
            fail("--cache_aug_refresh_epochs is not supported with caption variants (refresh re-encodes without Qwen3.5)")
        for subset in getattr(dataset, "subsets", []):
            if int(getattr(subset, "caption_dropout_every_n_epochs", 0) or 0) > 0:
                fail(
                    "caption_dropout_every_n_epochs is not supported (it zeroes the cached outputs, which leaves the"
                    " connector without any valid Qwen3.5 token); use caption_dropout_rate with --cache_caption_variants"
                )
            if float(getattr(subset, "caption_dropout_rate", 0.0) or 0.0) > 0.0 and variants <= 1:
                fail(
                    "caption_dropout_rate > 0 needs --cache_caption_variants N>1, so a dropped caption is encoded as ''"
                    " by both Qwen3 and Qwen3.5 (zeroing cached outputs is not a valid Qwen3.5 encoding)"
                )

    t5_max = getattr(args, "t5_max_token_length", 512)
    if t5_max is not None and int(t5_max) != 512:
        logger.warning(
            f"--anima38_connector: t5_max_token_length={t5_max}; Forge feeds at most 512 T5 ids to the connector"
            " (longer ids are cut, shorter limits truncate prompts Forge would keep)"
        )


def inspect_checkpoint(path: str) -> Dict[str, object]:
    """Read-only header scan of a DiT checkpoint: {'is_bundle', 'metadata', 'connector_keys'} (no tensors are read)."""
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = dict(checkpoint.metadata() or {})
        keys = list(checkpoint.keys())
    prefixes = connector_prefixes(metadata)
    connector_keys = [key for key in keys if key.startswith(prefixes)]
    return {"is_bundle": is_anima38_v2_bundle(keys, metadata), "metadata": metadata, "connector_keys": connector_keys}


def require_anima38_bundle(path: str) -> Dict[str, object]:
    info = inspect_checkpoint(path)
    if not info["is_bundle"] or not info["connector_keys"]:
        raise ValueError(
            f"--anima38_connector needs an Anima-3.8B v1.1 bundle with {CONNECTOR_PREFIX}* tensors; {path} has none"
        )
    return info


def check_qwen35_checkpoint(path: str) -> None:
    """Read-only header check that ``path`` is the Qwen3.5-4B encoder the connector was trained with."""
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        shape = checkpoint.get_slice("embed_tokens.weight").get_shape() if "embed_tokens.weight" in keys else None
    if shape is None or list(shape)[1:] != [QWEN35_HIDDEN_SIZE] or "layers.31.self_attn.q_proj.weight" not in keys:
        raise ValueError(f"--qwen35_path is not a Qwen3.5-4B checkpoint (embed_tokens {shape}): {path}")


def anima38_metadata(args) -> Dict[str, object]:
    """Training metadata entries for the saved LoRA."""
    return {
        "ss_anima38_connector": True,
        "ss_anima38_bundle": os.path.basename(str(getattr(args, "pretrained_model_name_or_path", "") or "")),
        "ss_anima38_qwen35": os.path.basename(str(getattr(args, "qwen35_path", "") or "")),
        "ss_anima38_sample_negative": getattr(args, "anima38_sample_negative", "connector"),
        "ss_anima38_qwen35_batch_size": int(getattr(args, "anima38_qwen35_batch_size", 1) or 1),
    }


# endregion

# region Qwen3.5 layer packing (one [L, n_layers * 2560] tensor per caption, so the dataloader pads rows)


def pack_semantic_layers(states: Sequence[torch.Tensor]) -> torch.Tensor:
    """n_layers x [B, L, D] -> [B, L, n_layers * D] (layer-major along the last dim)."""
    return torch.cat(list(states), dim=-1)


def unpack_semantic_layers(packed, num_layers: int = len(DEFAULT_LAYER_INDICES)) -> List[torch.Tensor]:
    """[B, L, n_layers * D] -> n_layers x [B, L, D]. A list/tuple of layer tensors is returned as a list."""
    if isinstance(packed, (list, tuple)):
        return list(packed)
    if packed.ndim != 3 or packed.shape[-1] % num_layers:
        raise ValueError(f"packed Qwen3.5 states must be [B, L, {num_layers} * D], got {tuple(packed.shape)}")
    batch, length, width = packed.shape
    return list(packed.reshape(batch, length, num_layers, width // num_layers).unbind(2))


def semantic_dit_kwargs(text_encoder_conds: Sequence, device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """DiT keyword arguments carrying the cached Qwen3.5 features of a batch.

    ``text_encoder_conds`` is the Anima-3.8B layout: [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask,
    qwen35_states, qwen35_mask, (caption_dropout_rate)].
    """
    if len(text_encoder_conds) < 6 or text_encoder_conds[4] is None or text_encoder_conds[5] is None:
        raise RuntimeError(
            "--anima38_connector: the batch has no Qwen3.5 features. The text encoder outputs must come from the"
            " Anima-3.8B cache (*_anima38_te.npz / in-memory cache built with --anima38_connector)"
        )
    states = text_encoder_conds[4]
    mask = text_encoder_conds[5]
    if not isinstance(states, torch.Tensor):
        states = torch.as_tensor(states)
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    return {
        "semantic_hidden_states": states.to(device=device, dtype=dtype),
        "semantic_attention_mask": mask.to(device=device),
    }


# endregion

# region DiT conditioner


class Anima38Conditioner:
    """Callable used by ``Anima.forward`` in place of ``_preprocess_text_embeds`` when attached.

    Plain object on purpose: assigning it to the DiT does not register the connector as a submodule.
    """

    def __init__(self, connector: QualityAnchoredSemanticConnectorV2, grad_through_connector: bool = False):
        self.connector = connector
        # True only when the network trains llm_adapter modules (their LoRA acts inside the connector's native
        # blocks, which are the DiT's own llm_adapter). The connector's parameters stay frozen either way.
        self.grad_through_connector = grad_through_connector

    @property
    def num_layers(self) -> int:
        return len(self.connector.layer_indices)

    def __call__(
        self,
        context: torch.Tensor,
        timesteps: torch.Tensor,
        target_input_ids: Optional[torch.Tensor],
        target_attention_mask: Optional[torch.Tensor],
        source_attention_mask: Optional[torch.Tensor],
        semantic_hidden_states=None,
        semantic_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if target_input_ids is None:
            if semantic_hidden_states is not None:
                raise ValueError("Anima-3.8B: Qwen3.5 features were given without T5 ids")
            # already a cross-attention context (e.g. a precomputed native negative prompt in sampling)
            return context
        if semantic_hidden_states is None:
            raise RuntimeError(
                "Anima-3.8B: this DiT call has T5 ids but no Qwen3.5 features (semantic_hidden_states); the llm_adapter"
                " output must not be used with the v1.1 bundle. This code path is not supported with --anima38_connector"
            )
        semantic = [state.detach() for state in unpack_semantic_layers(semantic_hidden_states, self.num_layers)]
        batch = target_input_ids.shape[0]
        steps = timesteps.reshape(-1)
        if steps.numel() == 1 and batch > 1:
            steps = steps.expand(batch)
        elif steps.numel() != batch:
            raise ValueError(f"Anima-3.8B: {steps.numel()} timesteps for a batch of {batch}")
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.grad_through_connector):
            return self.connector.forward_context(
                context,
                target_input_ids,
                semantic,
                steps,
                target_attention_mask=target_attention_mask,
                native_source_mask=source_attention_mask,
                semantic_source_mask=semantic_attention_mask,
            )


def attach_anima38_conditioner(dit: torch.nn.Module, connector: QualityAnchoredSemanticConnectorV2) -> Anima38Conditioner:
    """Attach the connector to the DiT. It must wrap this DiT's own llm_adapter (shared, not copied)."""
    if connector.native_adapter is not getattr(dit, "llm_adapter", None):
        raise ValueError("the connector must be built around this DiT's llm_adapter")
    conditioner = Anima38Conditioner(connector)
    object.__setattr__(dit, CONDITIONER_ATTRIBUTE, conditioner)
    return conditioner


def get_anima38_conditioner(dit) -> Optional[Anima38Conditioner]:
    """The conditioner attached to ``dit`` (unwraps DDP ``.module`` / torch.compile ``._orig_mod``), or None."""
    module = dit
    for _ in range(4):
        conditioner = getattr(module, "__dict__", {}).get(CONDITIONER_ATTRIBUTE)
        if conditioner is not None:
            return conditioner
        inner = getattr(module, "__dict__", {}).get("_modules", {})
        module = inner.get("module", inner.get("_orig_mod"))
        if module is None:
            return None
    return None


def network_trains_llm_adapter(network) -> bool:
    """True if the network (LyCORIS or the fork's LoRA/LoKr/LoHa) has trainable modules on the llm_adapter."""
    if network is None:
        return False
    if bool(getattr(network, "train_llm_adapter", False)):
        return True
    for name, parameter in getattr(network, "named_parameters", lambda: [])():
        if parameter.requires_grad and "llm_adapter" in name:
            return True
    return False


# endregion
