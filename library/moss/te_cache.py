# MOSS-Image text-encoder output caching, shared by the LoRA trainer (moss_train_network.py) and the full fine-tune
# (moss_train.py): the two caching methods of MossNetworkTrainer moved here unchanged
# (docs/moss_full_finetune_design.md 4.10). The 9B Qwen3.5 encoder (MossTextEncoderProvider) loads on the first caption
# or sample prompt that actually needs encoding and is released at the end, so a run with complete caches never loads it.

from typing import Callable, Dict, List, Optional

import torch

from library import strategy_base
from library.device_utils import clean_memory_on_device
from library.moss import captions, strategy
from library.moss import train_utils as moss_train_utils
from library.utils import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


def cache_moss_sample_prompts(args, provider) -> Dict[str, List[torch.Tensor]]:
    """{prompt text: [text_embeds, text_mask] on the CPU} for every prompt and negative prompt of --sample_prompts. Each
    is read from `<sample_prompts>_moss_te/` (keyed by the TE signature and the text) or encoded once and written there.
    The tokenize / text-encoding strategies are the ones set for the run."""
    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
    encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
    signature = strategy.moss_te_signature(args.moss_text_encoder, args.moss_max_token_length, args.moss_tokenizer_dir)
    cache_dir = moss_train_utils.sample_prompt_cache_dir(args)
    outputs = {}
    for text in moss_train_utils.sample_prompt_texts(args):
        path = moss_train_utils.sample_prompt_cache_path(cache_dir, signature, text)
        cached = moss_train_utils.load_cached_sample_prompt(path)
        if cached is None:
            logger.info(f"  encode sample prompt: {text[:100]}")
            with torch.no_grad():
                cached = encoding_strategy.encode_tokens(tokenize_strategy, [provider], tokenize_strategy.tokenize(text))
            cached = [cached[0].to("cpu"), cached[1].to("cpu")]
            moss_train_utils.save_cached_sample_prompt(path, cached)
        outputs[text] = cached
    logger.info(f"sample prompt embeddings ready: {len(outputs)} prompts ({cache_dir})")
    return outputs


def cache_moss_text_encoder_outputs(args, accelerator, vae, dataset, provider,
                                    sample_prompts_te_outputs: Optional[dict] = None, *,
                                    cache_sample_prompts: Optional[Callable] = None) -> Optional[dict]:
    """Cache the dataset captions (every caption variant, canonical captions) and the sample prompts, then release the
    text encoder. Returns the sample prompt outputs: `sample_prompts_te_outputs` when it is already set, else the
    freshly cached ones (None without --sample_prompts).

    The VAE (if any) goes to the CPU while the encoder may be on the GPU (unless --lowram) and back afterwards. The
    canonical-caption hooks are removed in any case: DataLoader workers pickle the dataset. cache_sample_prompts: the
    sample-prompt function (default cache_moss_sample_prompts; MossNetworkTrainer passes its own method)."""
    cache_sample_prompts = cache_sample_prompts or cache_moss_sample_prompts
    caching = strategy_base.TextEncoderOutputsCachingStrategy.get_strategy()
    captions.install_canonical_captions(dataset)
    captions.prepare_canonical_captions(dataset)
    captions.register_caption_digests(dataset, caching)
    org_vae_device = vae.device if vae is not None else None
    if not args.lowram:
        if vae is not None:
            vae.to("cpu")
        clean_memory_on_device(accelerator.device)
    try:
        dataset.new_cache_text_encoder_outputs([provider], accelerator)
        if args.sample_prompts is not None and sample_prompts_te_outputs is None:
            sample_prompts_te_outputs = cache_sample_prompts(args, provider)
        accelerator.wait_for_everyone()
    finally:
        captions.uninstall_canonical_captions(dataset)  # DataLoader workers pickle the dataset
        if provider.loaded:
            provider.release()
            logger.info("released the MOSS text encoder after caching")
        clean_memory_on_device(accelerator.device)
        if not args.lowram and vae is not None:
            vae.to(org_vae_device)
    return sample_prompts_te_outputs
