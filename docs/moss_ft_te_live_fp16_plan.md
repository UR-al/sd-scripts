# moss_train.py: live text encoder (A) and fp16 compute (B) - implementation plan

Status: part A (live text encoder) and part B (fp16 compute + dynamic loss scaling) implemented 2026-11-17 and tested
on the CPU (deviations: `moss_ft_impl_notes.md` T5, T6); GPU smoke items G1-G3 not run. Worktree `backend/sd_scripts_moss_full`, branch `feat/moss-full-ft`
(uncommitted). Read with `moss_full_finetune_design.md` (sections 1.5, 2, 3.1, 4.3, 8.3), `moss_train.md` and
`moss_ft_impl_notes.md`. KohakuMossTrainer (KMT, Apache-2.0 + naming clause) was read for ideas only (TEXT_ENCODE
`ring` / `ahead`, `docs/internals/text-encoder.md`, `fp16.md`); no code is copied.

Test interpreter: the task text names `LoRA_Easy_Training_Scripts\venv_t214`, which does not exist. The torch 2.14 venv
is `<LoRA_Easy_Training_Scripts>\backend\sd_scripts\venv_t214\Scripts\python.exe` (torch
2.14.0+cu130). Every test runs with `CUDA_VISIBLE_DEVICES=-1`.

## 한국어 요약

- **(A) 라이브 TE:** Qwen3.5-9B 를 캐시 없이 학습 중에 돌립니다.
  - 디코더 32층(12.89 GiB bf16)은 호스트 RAM 에 두고, 한 번에 1층씩 GPU 링(기본 2칸, 833 MiB)으로 보냅니다.
    임베딩 표(1.89 GiB)는 파일 mmap 에서 CPU 로 바로 뽑습니다(GPU 로 안 보냄).
  - 다음 N 마이크로배치(기본 32, GA 배수로 올림)의 캡션을 한 패스에 모아 인코딩합니다. 가중치 전송 1회를 N 배치가 나눠 씁니다.
  - 각 캡션은 지금 캐시와 똑같이 **혼자, 패딩 없이** 돌립니다(층 순서만 layer-major). 그래서 CPU 에서 캐시 경로와
    비트 단위로 같고, 시험이 그것을 확인합니다.
  - 캡션 셔플(+keep_tokens / `|||`), 여러 줄 = 변형(`enable_wildcard`), caption dropout, tag dropout, token warmup 이
    전부 매 스텝 무작위로 됩니다. 캐시 경로는 그대로 두고, 플래그로 고릅니다.
  - RAM: +13.9 GiB(가중치 12.89 + 핀 스테이징 1). adamw_host GA>1 이면 합계 약 99 GiB + 여유 8 → 128 GB 에서 빠듯합니다.
    `--moss_te_live_host mmap` 이면 사적 RAM 거의 0(파일 페이지 캐시 사용).
  - 12.5만 장 x 변형 3개의 TE 캐시는 디스크 약 0.9 TB 라서, 라이브 쪽이 디스크도 아낍니다.
  - 비용: 이미지당 TE 약 0.05-0.1 s(추정, 1024² b1 의 1.1 s/이미지 대비 5-9 %). 스모크 런에서 잽니다.
- **(B) fp16 계산:** `mixed_precision = "fp16"` 은 `--moss_stream_fp32` 필수.
  - 동적 loss scale S(2^15 시작, 2의 거듭제곱만): 오버플로면 1/2, 2,000 정상 스텝마다 x2.
  - 호스트 AdamW 안에서 클립 계수에 1/S 를 곱해 언스케일합니다(2의 거듭제곱이라 정확).
  - in-backward 스텝이라 오버플로 창은 '그 파라미터만' 건너뜁니다(기본). `exact` 클립이면 창 전체를 건너뛰는 정책도 고를 수 있습니다.
  - S 와 카운터는 state 에 저장되어 이어하기가 정확합니다. bf16 은 그대로 기본값입니다.

---

## 0. What exists today (read for this plan)

- `library/moss/qwen35.py`: `MossQwen35TextEncoder.forward(input_ids [1, L])` = embed -> `_rotary(L)` -> 32
  `DecoderLayer(x, cos, sin)` -> final `norm`; `@torch.no_grad`. GatedDeltaNet runs one unpadded prompt
  (`chunk_gated_delta_rule`, python loop over 64-token chunks, fp32); full attention is causal SDPA.
  `load_moss_text_encoder` reads the 426 language-model tensors (`.to(dtype)`) and `load_state_dict(assign=True)` on a
  meta-built model.
- `library/moss/strategy.py`: `MossTokenizeStrategy.tokenize(texts) -> [ids [B, L] (pad 248044), mask [B, L] long]`;
  `MossTextEncodingStrategy.encode_tokens` runs **each row alone** (`input_ids[i:i+1, :len]`) and `pad_rows` the
  results; `MossTextEncoderOutputsCachingStrategy` writes unpadded bf16 rows + int32 ones mask + rate;
  `drop_cached_text_encoder_outputs` refuses non-zero rates.
- What the DiT gets on the cached path (`train_util.BaseDataset.__getitem__` + `none_or_stack_elements` +
  `convert_te_output_for_batch`): `text_encoder_outputs_list = [embeds bf16 [B, T_max, 4096] zero-padded (F.pad),
  mask float32 [B, T_max] (1 / 0; int32 -> torch.FloatTensor), rates float32 [B]]`; `ft_loss` casts embeds to the
  stream dtype and the mask to bool.
- Without a TE-outputs caching strategy, `__getitem__` calls `process_caption(subset, info.caption)` in the DataLoader
  worker (wildcard line pick, prefix/suffix, dropout, keep_tokens / keep_tokens_separator split, shuffle, tag dropout
  with protected tags, token warmup, replacements) and `tokenize_strategy.tokenize(caption)`;
  `input_ids_list = [ids [B, L] (0-padded by F.pad), mask [B, L] long]`.
- `moss_train.py` step 6 caches TE outputs through `te_cache.cache_moss_text_encoder_outputs` (provider loads the 9B
  onto the GPU, released before the DiT exists). `ft_args._check_caches` and `train_utils.validate_moss_args` require
  `cache_text_encoder_outputs`; `validate_moss_args` asserts `is_text_encoder_output_cacheable` (shuffle / tag dropout
  need `--cache_caption_variants`, token warmup refused).
- `ft_args._check_precision` refuses `fp16` ("phase 1b"); `_check_stream` requires `--moss_stream_fp32` with bf16.
- `ft_host.GradPipeline`: per-parameter post-accumulate-grad hooks; per-slot fp32 GPU norm (`norm_host[s, j]`) is the
  finiteness check (`w.poisoned[s]`); the final micro-step steps each piece in the worker as it arrives
  (`_step_range`: `_add_sumsq(t)` then `t.mul_(w.coef)`), unless `plan.after_backward` (exact clip, `overlap=False`)
  where `step_from_accumulator(coef)` runs after the drain. `last_grad_norm` / `skip_streak` are saved in states.
- Real sizes (meta model, computed for this plan): decoder layers 12.889 GiB bf16 in total; a GatedDeltaNet layer
  436,814,208 B (416.6 MiB, 14 tensors), a full-attention layer 419,447,808 B (400.0 MiB, 11 tensors); embed
  2,034,237,440 B (1.89 GiB); final norm 8 KiB; whole encoder 14.78 GiB.

## 1. (A) Live frozen text encoder

### 1.1 Shape of the solution

- One flag `--moss_te_live` chooses the path. Off (default) = today's cached path, unchanged.
- The DataLoader workers do the caption randomness and tokenization (kohya's existing non-cached branch). The main
  process encodes in **passes**: one pass encodes every caption of the next `ahead` micro-batches.
- A pass streams the 32 decoder layers host -> GPU through a ring of `R` device slots, **layer-major**: layer l is
  uploaded once and applied to every caption of the pass, each caption alone and unpadded (the cached path's math, so
  the outputs are the cached path's values; see 1.6).
- Embedding rows are gathered on the CPU from the memory-mapped file (exact: a gather copies values), so the 1.89 GiB
  table never crosses PCIe and is not a private copy.
- The final norm (8 KiB) and `inv_freq` stay resident on the device.
- Everything runs under `torch.no_grad()` (**not** `inference_mode`: the DiT saves the text tensor for its backward,
  and inference tensors cannot be saved). TE tensors never have `requires_grad`, are never in `ft_model.trainable_named`
  / the host pool / the optimizer, and never reach the dataset (DataLoader workers pickle it).

### 1.2 New files (small, single-purpose)

**`library/moss/te_live_weights.py`** - the host form of the encoder.
- `LayerLayout` (frozen dataclass): `kind` ("linear" | "full"), `entries: Tuple[(name, offset, shape, numel)]` with
  every offset a multiple of 256 B (same pointer alignment class as a standalone tensor, so cuBLAS picks the same
  kernels), `nbytes`.
- `layer_layouts(config) -> List[LayerLayout]`: from a meta-built `MossQwen35TextEncoder(config)`; names are the
  module-relative parameter names of `layers.{i}` (`linear_attn.in_proj_qkv.weight`, `mlp.down_proj.weight`, ...).
  `slot_bytes(layouts)` = max nbytes (416.6 MiB for the real config).
- `HostTextEncoderWeights`:
  - `open(path, config=MossQwen35Config(), host="copy" | "pinned" | "mmap", dtype=torch.bfloat16, check=True)`.
    `check` runs `qwen35.check_moss_text_encoder` (prefix, 9B shapes); tests pass a tiny config with `check=False`
    and an explicit prefix.
  - The file is opened with `ft_safetensors.SafetensorsReader` (offsets, dtypes) and `mmap.mmap(..., ACCESS_READ)`;
    `torch.frombuffer` views give zero-copy tensors of each stored tensor.
  - `copy`: per layer one pageable flat bf16 buffer (`torch.empty(nbytes // 2, dtype=bf16)`), filled tensor by tensor
    with `view.copy_(file_tensor)` (a stored fp32 tensor, e.g. `A_log`, is rounded to bf16 by the same RNE cast as
    `get_tensor(key).to(dtype)` in `load_moss_text_encoder`). Peak during load = total + nothing (no dense model).
  - `pinned`: like `copy` but all 32 layers in **one** pageable allocation of exactly Σ nbytes, then
    `torch.cuda.cudart().cudaHostRegister(ptr, nbytes, 0)` (no power-of-two rounding of torch's pinned allocator;
    `cudaHostUnregister` in `close()`); the ring's H2D reads it directly, no staging. CUDA only (refused on the CPU).
  - `mmap`: no private copy; layer tensors are the file views. Needs the layer pages in the OS file cache (~13 GB) or
    each pass reads them from disk.
  - `embed_rows(ids: LongTensor [L]) -> bf16 [L, 4096]` on the CPU: `index_select` on the file view of
    `embed_tokens.weight`, `.to(bf16)` after the gather (same values as gathering a bf16 table).
  - `final_norm_weight`, `layer_sources(i) -> List[(entry, cpu tensor)]` (copy/pinned: views into the flat buffer;
    mmap: file views), `private_bytes`, `locked_bytes`, `close()`.
  - `__reduce__` raises (never pickled into a worker).

**`library/moss/te_live.py`** - the streaming engine (`LiveTextEncoder`).
- `LiveTextEncoder(weights, device, ring_slots=2, staging_threads=8)`:
  - `shell = MossQwen35TextEncoder(config)` built on `meta`, `requires_grad_(False)`, `eval()`. Its modules are the
    forward code: per layer, `bind(layer, slot_views)` sets `module._parameters[name] = view` (plain tensors) for each
    entry, runs, then `unbind` restores the meta parameters. `shell.norm.weight` and `shell.inv_freq` are real
    resident tensors on the device. So the code that runs is `DecoderLayer.forward` / `Qwen35RMSNorm.forward` /
    `MossQwen35TextEncoder._rotary` of `qwen35.py`, unchanged.
  - Device ring: `R` slots of `slot_bytes` (one `torch.empty(R * slot_bytes, uint8, device)`), allocated per pass and
    freed after it (the caching allocator keeps the blocks reserved; no `empty_cache`, design 2.1). The DiT's
    activations can reuse those blocks between passes.
  - Host staging (copy / mmap): `ft_host.PinnedRing(R * 512 MiB, 512 MiB, pin=cuda)`; the slot is padded to 512 MiB so
    the ring is one power-of-two pinned allocation for R in {2, 4} (`R` restricted to 2 or 4). `pinned` host mode needs
    no staging ring.
  - `encode_rows(rows: List[LongTensor]) -> List[bf16 Tensor [T_i, 4096]]` (on the device):
    1. `x_i = weights.embed_rows(ids_i).to(device)`; `cos/sin` per distinct length from `shell._rotary(T, device,
       bf16)` (cached in a dict for the pass).
    2. Schedule (CUDA; the CPU path runs the same steps synchronously):
       - a stager thread (`torch.set_num_threads(staging_threads)` thread-locally, like the host worker) copies layer
         l's sources into a pinned staging entry (`copy`/`mmap`), then enqueues `dev_slot[l % R].copy_(staging,
         non_blocking=True)` on a dedicated copy stream after waiting the event `free[l % R]` (layer l - R finished on
         the compute stream), records `ready[l]`, and releases the staging entry once `ready[l]` completes;
       - the main thread, on the compute (default) stream, for l = 0..31: `wait_event(ready[l])`, bind layer l to
         `dev_slot[l % R]`, `for i: x_i = layer(x_i, cos[T_i], sin[T_i])`, unbind, `free[l % R].record()`;
       - prefetch depth = R - 1 layers (R = 2: layer l+1 uploads while layer l computes).
    3. `out_i = shell.norm(x_i)`.
    4. Under `torch.no_grad()` and `torch.autocast(device.type, enabled=False)` exactly like `encode_tokens`.
  - `encode_tokens(tokens) -> [embeds bf16 [B, L, 4096], mask long [B, L]]` = `pad_rows(encode_rows(...))`: what
    `MossTextEncodingStrategy.encode_tokens` returns (used for the sample prompts).
  - `stats`: last pass seconds (stage / H2D wait / compute), rows, tokens, peak device bytes (`max_memory_allocated`
    delta on CUDA); logged per pass at DEBUG and summarised every `--moss_ft_log_every` steps.
  - `close()`: joins the stager, frees the ring, closes the weights.

**`library/moss/te_live_feed.py`** - encode-ahead over the DataLoader.
- `TextEncodeAhead(encoder, ahead: int)`:
  - `wrap(loader_iterable)` is a generator: it pulls up to `ahead` batches, collects every row of their
    `input_ids_list` (`ids[b, :mask[b].sum()]`, moved to the CPU once per pass: one sync per pass), runs one
    `encode_rows`, splits the results back per batch and yields each batch in order with
    `batch["text_encoder_outputs_list"] = live_outputs(rows)` and the `input_ids_list` removed. The epoch's tail pass
    takes what is left (fewer batches).
  - `live_outputs(rows) -> [embeds, mask, rates]` builds the cached path's exact batch form: embeds bf16 zero-padded to
    the batch's longest row, mask **float32** 1/0, rates float32 zeros `[B]` (`convert_te_output_for_batch` makes the
    cached mask float32).
  - The pass runs when the queue is empty at a micro-step boundary. `ahead` is rounded up to a multiple of GA (log line)
    so a pass always starts at j = 0 of a window: the previous window has drained (no host worker traffic, uploads
    issued), so the stager's memcpy does not compete with the host AdamW for memory bandwidth.
  - Holds at most `ahead` batches (their latents are already on the device: accounted in 1.5).
  - `last_pass_s` / `pass_count` for the window logs (`te/pass_s`, `te/s_per_micro`).
- Order, resume and RNG: the feed only changes WHEN a batch's text is encoded, not which batch is trained or in which
  order; `accelerator.skip_first_batches` is applied to the inner loader before `wrap`. With
  `max_data_loader_n_workers = 0` the dataset's own random draws happen in the main process, so pulling batches ahead
  interleaves them differently with the training RNG (valid, but not bit-identical to `ahead = 1`); with workers the
  draws happen in the workers and are unaffected. Documented; the bit-identity test uses a dataset that draws nothing.

### 1.3 Changes to existing files

- `library/moss/strategy.py`
  - `resolve_text_encoder`: a `LiveTextEncoder` is returned as is (no `.get()`).
  - `MossTextEncodingStrategy.encode_tokens`: if `models[0]` is a `LiveTextEncoder` -> `models[0].encode_tokens(tokens)`
    (same contract). Import inside the function (no import cycle).
  - `CAPTION_DROPOUT_ERROR` text: mention `--moss_te_live` as the other way to get caption dropout.
- `library/moss/te_cache.py`: no logic change; `cache_moss_sample_prompts(args, provider)` works with the live encoder
  through the strategy branch (the sample prompt cache `<sample_prompts>_moss_te/` is reused, keyed by the same
  signature).
- `library/moss/captions.py`: add `live_caption_summary(dataset_group) -> List[str]` (per subset: shuffle, keep_tokens /
  separator, wildcard lines (max lines seen), dropout rates, tag dropout, protected tags, token warmup) for the start-up
  log, and `separator_warnings(dataset_group)`: a caption containing `|||` (or the subset's separator candidates) while
  `keep_tokens_separator` is unset reaches the encoder with a literal `|||` - warn with the count and one example path.
- `library/moss/train_utils.py` `validate_moss_args(args, train, val, *, live_te=False)`: with `live_te` it skips the
  `cache_text_encoder_outputs` requirement, the `is_text_encoder_output_cacheable` assert and the
  caption-dropout-needs-variants checks (the LoRA trainer passes nothing: unchanged).
- `library/moss/ft_args.py`
  - New arguments (in `add_moss_ft_arguments`): see 1.7.
  - `_check_caches` splits into `_check_latent_cache` (unchanged rule) and `_check_text_path`: live vs cached (1.7).
  - `_validate_dataset`: the 2,000-image rule asks only for `cache_latents_to_disk` in live mode; new
    `_check_live_captions(dataset_group)` / `_check_cached_captions(dataset_group)` (moved from the assert in
    `validate_moss_args`, now `_refuse` with a reason pointing to `--moss_te_live`).
- `library/moss/ft_budget.py`: 1.5.
- `moss_train.py`
  - `set_moss_strategies(args)`: in live mode the TE-outputs caching strategy is **not** set (the dataset must take the
    tokenize branch); the tokenize strategy is set as today.
  - step 6: cached path unchanged. Live path: `weights = HostTextEncoderWeights.open(...)`, `encoder =
    LiveTextEncoder(weights, device, ...)`, sample prompts via `te_cache.cache_moss_sample_prompts(args, encoder)` when
    `--sample_prompts`; the start-up caption summary is logged. The VAE goes back to the CPU first (as today).
  - `Run` gets `te_feed: Optional[TextEncodeAhead]` and closes the encoder in `train()`'s `finally`.
  - `_train_loop`: `loader = run.te_feed.wrap(loader)` after `skip_first_batches`; `meter.data_s` excludes the pass
    time, which is logged as `te/pass_s` and `te/s_per_step`.
- `library/moss/ft_loss.py`: only the error message (`"... --cache_text_encoder_outputs or --moss_te_live"`).

### 1.4 Captions in live mode (all existing kohya mechanisms, now random per fetch)

- `HEAD ||| TAGS ||| SENTENCES`: `keep_tokens_separator = "|||"`. HEAD (trigger, count, character, series) is fixed in
  front, TAGS are shuffled (`shuffle_caption`) and tag-dropped (`caption_tag_dropout_rate`, protected tags kept),
  SENTENCES stay at the end; the parts are joined with `", "` (kohya). Plain tag lists: `keep_tokens = N`.
- Multi-line caption file = variants: `enable_wildcard = true` picks one line uniformly per fetch (each repeat of each
  epoch draws again), before prefix / suffix. `{a|b}` is also expanded under it; literal braces are written `{{ }}`.
- `caption_dropout_rate`: the caption becomes `""` -> chat template with an empty user turn = the official
  unconditional prompt (same as the cached variants' dropped row). `caption_dropout_every_n_epochs` also works (it is
  refused on the cached path because it zeroes cached rows).
- `token_warmup_step` / `token_warmup_min` work (step-dependent; refused on the cached path).
- `cache_caption_variants` is refused in live mode (nothing is cached; every fetch is a new draw).

### 1.5 Memory (formulas and the real numbers)

Notation: `L_b` = bytes of decoder layer b (aligned layout); `S` = `slot_bytes` = max L_b = 416.6 MiB; `R` = ring slots;
`P` = 512 MiB (staging entry); rows of a pass: `N = ahead x B`, lengths `T_i`; `H` = 4096.

Host RAM (new `HostBudget.text_encoder` and `.te_staging`):
- `copy`: Σ L_b = 12.89 GiB private + staging `pinned_alloc_bytes(R x P)` = 1 GiB (R 2) / 2 GiB (R 4). Total 13.9 GiB.
- `pinned`: Σ L_b = 12.89 GiB page-locked (registered, not rounded), no staging.
- `mmap`: 0 private + staging 1 GiB; the report says the run wants ~12.9 GB of file cache (else ~7 s of disk reads per
  pass at 2 GB/s; review fix: a start-up warning when available < required + 12.9 GiB, priced at the disk rate).
- Embedding table: file view in every mode, 0 private (touched pages only).
- With the host budget of design 2.2 (adamw_host: master + m + v = 56.8 GiB, accumulator 18.9 GiB at GA > 1, rings
  1.5 GiB, process 8 GiB): GA 1 66.3 -> 80.2 GiB, GA > 1 85.2 -> 99.1 GiB in `copy` mode; the start-up check wants
  available >= total + 8 GiB (107.1 GiB at GA > 1). On 128 GB (119.2 GiB) with Windows and the desktop that is tight:
  the report suggests `mmap` or GA 1 (stale clipping, no accumulator) when the check fails.

VRAM (new `GpuBudget.te_ring`, `.te_pass`, `.te_held`):
- `te_ring = R x S` = 833 MiB (R 2).
- `te_pass = te_ring + 2 x Σ_i T_i H 2 B` (layer inputs and outputs of all rows) `+ te_row_temp(T_max)` where
  `te_row_temp(T) = T x (3 x 12288 x 2 + 4 x 8192 x 4) B + 6 MiB` (MLP gate/up/product in bf16; GatedDeltaNet q/k/v
  /decay in fp32 per row; chunk matrices), about 22 + 128 MiB per 1k tokens - an upper estimate checked by the smoke run.
  N = 32 rows x 300 tokens: 0.81 + 0.15 + 0.05 = ~1.0 GiB.
- `te_held = ahead x B x T_max x H x 2 B` (outputs waiting) `+ ahead x B x latent bytes` (the held batches' latents,
  already on the device): 32 x 1 x 300 x 8 KiB = 75 MiB + 32 x 2 MiB = 64 MiB at 1024 b1.
- `allocated = max(dit_step, weights + MISC_FIXED + te_pass) + te_held`, where `dit_step` is today's sum (the pass runs
  between micro-steps when the DiT's activations are freed; the ring blocks are reused by the next micro-step). At 1024
  b1: max(14.6, 10.07 + 0.25 + 1.0) + 0.14 = 14.74 GiB. The pass never adds to a micro-step's peak.

Disk: live mode writes no `*_moss_te.npz` (`cache_estimate` counts 0 TE bytes). For scale: 125k images x 3 variants x
300 tokens x 8 KiB = ~0.92 TB of TE cache that live mode does not need.

Throughput (new `Throughput.te_s_per_image`, estimates to be measured):
- per pass `t = max(t_stage, t_h2d, t_compute) + t_launch`; `t_stage = Σ L_b / 10 GB/s` (~1.4 s, `copy`/`mmap`),
  `t_h2d = Σ L_b / 25 GB/s` (~0.55 s), `t_compute = tokens x 13.8 GFLOP / 150 TFLOP/s` (~0.9 s for 9,600 tokens),
  `t_launch ~ rows x 32 x 30 us` (per-row python / launch overhead, ~1 s for 32 rows).
- ~2-3 s per 32 captions = 0.06-0.1 s per image, 5-9 % of 1.1 s/image at 1024 b1. `pinned` removes `t_stage`.
- Not in this plan (later, after the smoke run): a padded batched layer forward (KMT shows right padding is exact for
  real tokens because every layer is causal, but GEMM shapes change, so it is not bit-identical to the cache).

### 1.6 Proof that the outputs equal the cached path

Argument: for each caption the live path runs the same ops in the same order on the same device and dtypes - embed
gather (copy), `_rotary(T)` with the same `inv_freq`, the 32 `DecoderLayer.forward` calls on a `[1, T, 4096]` input
whose weights are equal values at the same alignment class, the final norm - so each row is bitwise the cached row; the
batch assembly then reproduces `F.pad` zeros, the float32 mask and zero rates. Layer-major order only changes when each
row's layer runs, not what it computes.

Tests (CPU, tiny random Qwen-like config: hidden 64, intermediate 128, 8 layers with full attention every 4th, 4 heads
/ 2 KV heads of 16, linear heads 2 K / 4 V of 16, rotary 4 dims, vocab 512; weights random bf16 written to a
safetensors file with the `language_model.` prefix and one fp32 tensor (`A_log`) to cover the cast):
- `encode_rows` vs `MossQwen35TextEncoder` loaded by `load_moss_text_encoder` from the same file (`load_moss_text_encoder`
  gets the tiny config through a test-only `config=` parameter added to it): `torch.equal` per row for lengths
  {10, 37, 63, 64, 65, 130, 200} (chunk boundaries of the gated delta rule), R in {2, 4}, host in {copy, mmap}.
- Through the strategies: `MossTextEncodingStrategy.encode_tokens(tok, [live], tokens)` equals
  `encode_tokens(tok, [model], tokens)` (embeds and mask).
- Through the real cache: `MossTextEncoderOutputsCachingStrategy.cache_batch_outputs` writes npz files,
  `load_outputs_npz` + `convert_te_output_for_batch` + the `F.pad` stacking vs `live_outputs(encode_rows(...))`: equal
  embeds, equal float32 mask, equal rates.
- End to end (extends the harness of `tests/test_moss_ft_train_e2e.py`: tiny DiT, real dataset TOML, real strategies):
  run A = cached TE, run B = `--moss_te_live` (`ahead` 1 and 3, workers 0, captions without randomness): every logged
  loss and the saved DiT bitwise equal.
- On CUDA (smoke run, needs the user's OK): 64 real captions, cached npz rows vs live rows; expected `torch.equal`
  (same kernels); if a kernel choice differs, report max abs / rel error next to the bf16 rounding floor.

### 1.7 New flags and refusals (A)

| flag | default | meaning |
|---|---|---|
| `--moss_te_live` | off | encode captions at train time; no TE cache |
| `--moss_te_live_ahead` | 32 | micro-batches per pass, rounded up to a multiple of GA |
| `--moss_te_live_ring_slots` | 2 | device / staging slots: 2 or 4 |
| `--moss_te_live_host` | `copy` | `copy` / `pinned` (CUDA only) / `mmap` |
| `--moss_te_live_threads` | 8 | stager intra-op threads (memcpy into pinned staging) |

Refusals (`ft_args`, each `_refuse` with its reason):
- live + `cache_text_encoder_outputs` or `_to_disk`: "choose one: --moss_te_live encodes every caption at train time".
- live + `cache_caption_variants > 1`: variants are a cache feature; live mode draws a new caption every fetch.
- live + `--moss_te_live_host pinned` without CUDA.
- ring slots not in {2, 4}; ahead < 1; threads not in [1, cpu count].
- live flags (non-default) without `--moss_te_live`.
- cached path (unchanged rules, now explicit refusals instead of the assert): `shuffle_caption`,
  `caption_tag_dropout_rate > 0`, `caption_dropout_rate > 0` without `cache_caption_variants > 1`;
  `token_warmup_step > 0`; `caption_dropout_every_n_epochs > 0` - each message ends with "or use --moss_te_live".
- Warnings: `enable_wildcard` on the cached path without variants (today's warning); `|||` in captions without
  `keep_tokens_separator` (both paths); workers > 0 with `ahead` below workers x prefetch_factor is fine (no warning).

## 2. (B) fp16 compute with dynamic loss scaling (phase 1b)

### 2.1 Precision setting

- `mixed_precision = "fp16"` is allowed only with `--moss_stream_fp32` (joint Linears fp16 under autocast, every other
  parameter and the residual stream fp32). bf16 stays the default and keeps its rules.
- New `ft_args._check_precision` messages:
  - fp16 without the fp32 stream: "mixed_precision fp16 needs --moss_stream_fp32: Base's residual stream exceeds fp16's
    65504 and the stream must be fp32 for full fine-tuning (--moss_stream_scale is for LoRA); <STREAM_PRECISION_NOTE>".
  - fp16 with `adafactor_fused`: "fp16 needs a host optimizer mode (adamw_host / came_host): the loss scale is undone in
    the host step; adafactor_fused steps on the GPU inside the hooks".
  - `full_fp16` stays refused (host master).
  - other values: "mixed_precision must be bf16 (default), fp16 (with --moss_stream_fp32) or no".
- `ft_model.load_moss_for_full_ft` under fp16: `fp16_range_report(pool, slots)` - refuse if any master value of a
  joint Linear is beyond 65504; log the fraction below 6.1e-5 (fp16 subnormal) and exactly zeroed by the cast.
- Accelerate builds its own GradScaler for fp16; it is unused (moss_train calls `loss.backward()` itself and never
  `accelerator.backward`). A comment and a test assert that the run's backward path does not touch
  `accelerator.scaler`.
- `models.py` keeps the text refiner's flash attention in bf16 under the fp32 stream (line 162); that is the stream
  path, unchanged by fp16 (noted in the impl notes).

### 2.2 Loss scaler (`library/moss/ft_scaler.py`, new)

- `DynamicLossScaler(init_scale=2**15, growth_interval=2000, min_scale=1.0, max_scale=2**24)`; growth x2, backoff x0.5
  (fixed: powers of two keep the unscale exact). `growth_interval = 0` = never grows (an overflow still halves it).
- `scale` (python float, power of two), `inv_scale`, `good_windows`, `overflow_windows`, `overflow_streak`.
- `update(stats: WindowStats, loss_finite: bool) -> str` ("grow" | "backoff" | "keep"): a window with
  `stats.skipped_params > 0` is an overflow -> scale x0.5 (not below `min_scale`), `good_windows = 0`; else
  `good_windows += 1` and at `growth_interval` scale x2 (not above `max_scale`).
- `check(max_skipped_steps)`: raises (like `check_skip_streak`) after `max_skipped_steps` consecutive windows that
  overflowed **with a non-finite forward loss or at `min_scale`** - a divergence the scale cannot fix. Overflow windows
  with a finite loss above `min_scale` are the scaler's normal search and do not count.
- `state_dict()` / `load_state_dict()`: all fields + the constants (a resume with other constants logs and uses the
  saved scale with the new constants).

### 2.3 Where the scale enters (moss_train.py and ft_host.py)

- `moss_train._train_loop`: `scaler = ft_scaler.make(args)` (None unless fp16); per micro-step
  `(loss * scaler.scale).backward()` (loss is a float64 0-dim tensor; x 2^k is exact); `meter.add` gets the unscaled
  loss. At window start `window_context(optimizer, global_step, inv_grad_scale=scaler.inv_scale)`. After the final
  micro-step: `event = scaler.update(stats, math.isfinite(window_loss))`, logged (`train/loss_scale`, overflow count);
  `scaler.check(...)` replaces `pipeline.check_skip_streak` when a scaler exists.
- `ft_host.StepContext` gains `inv_grad_scale: float = 1.0` (power of two; validated in `begin_window`).
- `_Window.__init__`: `self.inv_scale = ctx.inv_grad_scale`; `self.coef = _coefficient(_combine(plan.coef, inv_scale))`
  where `_combine(c, s)` = `c * s` as an fp32 0-dim tensor (None when both are 1). Exact: `s` is a power of two, so
  `round(g * c * s) == round(g * c) * s` for every normal value - the worker's `t.mul_(coef)` unscales and clips in one
  multiply ("unscale fused into the host convert", design 8.3).
- `step_from_accumulator(clip_coef)`: `w.coef = _coefficient(_combine(clip_coef, w.inv_scale))`.
- Norms are scaled where measured and unscaled where used:
  - `_finish_window`: `grad_norm = sqrt(w.sumsq) * w.inv_scale` (sumsq is taken before the multiply; exact).
  - exact mode: `total = vector_norm(stack(slot norms)) * w.inv_scale` before `_coef_from_norm`.
  - `last_grad_norm` is always the unscaled norm (stale clipping and states unchanged in meaning).
- Overflow detection: unchanged - the per-slot fp32 GPU norm of the fp16 gradient is inf / nan -> `w.poisoned[s]`.
- Overflow policy `--moss_ft_overflow_policy`:
  - `partial` (default; the only one possible with the in-backward step): slots whose gradient is finite step, poisoned
    slots keep master / moments / step / compute copy (today's per-parameter skip, design 3.1); the scale halves. A
    finite gradient under loss scaling is the true gradient x S, so the slots that step get correct updates; with
    growth every 2,000 windows about one window in 2,000 is partial.
  - `window`: no slot steps in a window with any poisoned slot (torch GradScaler semantics). Needs every window to step
    after the drain (`plan.after_backward`: `--moss_ft_clip_mode exact` with `max_grad_norm > 0`); refused otherwise.
    Implementation: in `_finish_window`, when `after_backward` and any `w.poisoned`, skip `step_from_accumulator`,
    count every slot as skipped, `stepped_params = 0`.
- `WindowStats` gains `loss_scale: float` and `overflow: bool` for the logs.
- `came_host` uses the same `w.coef` in `_step_param`: covered by the same change.

### 2.4 State and resume

- `ft_state.write_state(..., loss_scaler=None)`: `train_state.json["loss_scaler"] = scaler.state_dict()` (or null).
- `load_state` -> `ResumeInfo.loss_scaler`; `moss_train` restores it. Cases: fp16 state -> fp16 run: exact; bf16 state
  -> fp16 run: start at `init_scale` (log line); fp16 state -> bf16 run: ignored (log line). `STATE_FORMAT` stays
  `moss-ft-state/1` (the key is optional).
- The fp16 compute copies are rebuilt from the fp32 master at load (as the bf16 ones are), so a resume needs nothing
  else.

### 2.5 New flags and refusals (B)

| flag | default | meaning |
|---|---|---|
| `--moss_ft_loss_scale` | 32768 | initial scale (power of two in [1, 2^24]) |
| `--moss_ft_loss_scale_growth_interval` | 2000 | clean windows before x2; 0 = never grows (still halves) |
| `--moss_ft_loss_scale_min` | 1.0 | power of two; reaching it counts toward the divergence streak |
| `--moss_ft_overflow_policy` | `partial` | `partial` / `window` (needs exact clipping) |

Refusals: non-power-of-two scales; min > initial; loss-scale flags (non-default) without fp16; `window` without
`--moss_ft_clip_mode exact` and `max_grad_norm > 0`; fp16 without `--moss_stream_fp32`; fp16 with `adafactor_fused`.

## 3. Budget, docs

- `ft_budget.py`: `te_live_host_bytes(config, host, ring_slots)`, `te_live_gpu_bytes(rows, t_max, ring_slots)`,
  `te_layer_bytes(config)` (meta model, cached; `test_te_constants_match_meta_model` pins 12.889 GiB / 416.6 MiB),
  `GpuBudget.te_ring/te_pass/te_held` with the `allocated` formula of 1.5, `HostBudget.text_encoder/te_staging`,
  `cache_estimate` without TE bytes in live mode, report lines (`TE live: host copy 12.9 GiB + staging 1.0 GiB, ring
  0.8 GiB, ~0.07 s/image est.`), fp16 in the precision line. fp16 costs the same VRAM / host RAM as bf16.
- `docs/moss_train.md`: Korean summary bullets for live TE and fp16; example config gets `mixed_precision = "fp16"`,
  `moss_stream_fp32 = true`, `moss_te_live = true`, `moss_te_live_ahead = 32`, the caption keys in the dataset TOML
  comment (`keep_tokens_separator = "|||"`, `shuffle_caption`, `enable_wildcard`, `caption_dropout_rate = 0.05`,
  `caption_tag_dropout_rate = 0.1`), cache_text_encoder_* lines removed; sections 3 (memory: the 1.5 table), 6 (logs:
  `te/*`, `train/loss_scale`), 8 (limits: fp16 is no longer refused).
- `docs/moss_full_finetune_design.md`: 2.1/2.2 rows for live TE, 4.3 step 6 live branch, 8.3 marked built.
- `docs/moss_ft_impl_notes.md`: the deviations (layer-major per-row pass, no inference_mode, partial overflow policy).

## 4. Test list (all CPU, `CUDA_VISIBLE_DEVICES=-1`)

New files:
- `tests/_moss_te_tiny.py`: tiny config, random safetensors writer, `CharTokenizer` (shared with the e2e harness).
- `tests/test_moss_ft_te_live_weights.py`: layouts and alignment; copy / mmap give the same values as
  `load_moss_text_encoder`; fp32-stored tensor cast; `embed_rows` = embedding lookup; `__reduce__` refuses; pinned mode
  refused on the CPU.
- `tests/test_moss_ft_te_live_engine.py`: `encode_rows` bitwise vs the cached encoder (lengths, R, host modes);
  layer-major order and binding restored (shell parameters are meta again after a pass); no `requires_grad` anywhere;
  outputs usable in an autograd graph (a Linear on them backprops: catches inference_mode); a failing stager raises
  in the main thread and the ring is released.
- `tests/test_moss_ft_te_live_feed.py`: `ahead` grouping and order, tail pass, rounding to a multiple of GA, one pass
  per `ahead` batches (counted), `live_outputs` equals the real cache's batch form (npz written by
  `MossTextEncoderOutputsCachingStrategy`), `skip_first_batches` before `wrap`.
- `tests/test_moss_ft_te_live_captions.py` (real `DreamBoothDataset` of tiny images, live strategies, workers 0):
  `HEAD ||| TAGS ||| SENTENCES` keeps HEAD first and SENTENCES last while TAGS permute across fetches; `keep_tokens`;
  multi-line + `enable_wildcard` draws every line; `caption_dropout_rate = 1.0` encodes exactly the cached `""` row;
  tag dropout keeps protected tags; token warmup; `|||` warning; the dataset pickles without the encoder (size bound).
- `tests/test_moss_ft_te_live_args.py`: every refusal and warning of 1.7, and the cached-path refusals that replaced the
  assert (message text pinned).
- `tests/test_moss_ft_te_live_e2e.py`: cached vs live runs bitwise equal (losses, DiT save) with `ahead` 1 and 3; a live
  run with shuffle + dropout trains and logs `te/*`; resume mid-epoch in live mode; sample prompts through the live
  encoder equal the provider path's cache files.
- `tests/test_moss_ft_loss_scaler.py`: update rules, bounds, static mode, divergence check (finite-loss overflows do not
  count; min-scale ones do), state round trip.
- `tests/test_moss_ft_fp16_pipeline.py` (fp32 tiny DiT, scaled loss): with S = 2^10 the masters, moments, steps and
  `grad_norm` equal the unscaled run bitwise for GA 1 / 3, clip none / stale / exact, adamw_host and came_host; an inf
  injected into one slot: `partial` steps the others and halves S, `window` steps nothing; growth after N windows.
- `tests/test_moss_ft_fp16_state.py`: scaler in `train_state.json`, exact resume, bf16 -> fp16 and fp16 -> bf16 resumes.

Extended files:
- `test_moss_ft_precision.py`: fp16 accepted with the fp32 stream, refused without it / with adafactor_fused; the new
  messages; `fp16_range_report` (overflow refused, subnormal fraction logged); compute copies = `master.to(fp16)`.
- `test_moss_ft_budget.py`: TE live host / GPU formulas, the 1.5 numbers, `allocated` max rule, live `cache_estimate`,
  host check boundary with live on (GA 1 / GA > 1, copy / mmap), design-table cross check for the new rows.
- `test_moss_ft_args.py`: new flags parse from TOML; defaults.
- `test_moss_ft_train_e2e.py`: a run with a forced scaler under fp32 compute (accelerate refuses fp16 on the CPU, so
  `ft_scaler.make` is monkeypatched) is bitwise equal to the run without it, and its resume is exact.
- `test_moss_ft_te_cache.py`: `encode_tokens` / `resolve_text_encoder` branch for the live encoder.

GPU smoke items to add to design section 7 (each needs the user's OK before running): G1 live vs cached rows on real
weights (1.6); G2 pass time split (stage / H2D / compute) and peak VRAM for ahead 8 / 32 and host copy / pinned / mmap;
G3 200 fp16 steps at 1024 b1: scale trajectory, overflow windows, loss vs a bf16 run, VRAM equal to bf16.

## 5. Order of work

1. `te_live_weights.py` + tests; 2. `te_live.py` + engine identity tests; 3. `te_live_feed.py`, strategy branch,
moss_train wiring, args / refusals, captions tests, e2e identity; 4. budget + report; 5. `ft_scaler.py`, ft_host
changes, fp16 precision path, state; 6. docs; 7. full `tests/test_moss_ft_*.py` run on the CPU. GPU smoke only with the
user's permission.
