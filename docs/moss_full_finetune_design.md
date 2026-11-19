# MOSS-Image full fine-tune (`moss_train.py`, "algo=full") - design

Status: design only, 2026-11-17. Worktree `backend/sd_scripts_moss_full`, branch `feat/moss-full-ft`, base `62bc699f`.
Nothing here is implemented yet. Line numbers refer to the base commit.

How to read the numbers:
- **measured**: measured on this PC on the CPU only (torch 2.14.0+cu130 from `venv_t214`, `CUDA_VISIBLE_DEVICES=-1`,
  24 threads, Core Ultra 9 285K, DDR5-4400 4 x 32 GB). The snippets are in Appendix A.
- **estimate**: computed, not measured. The GPU smoke run (section 7) must confirm or replace it.
- No GPU was used for this document.

---

## 한국어 요약

- **CAME 질문의 답: MOSS 에서도 똑같이 망가집니다.** 원인은 모델이 아니라 CAME 계산식입니다. beta3 EMA 에 편향 보정이 없습니다.
  - 사용자 came.py 의 실제 core 로 CPU 시뮬레이션했습니다. beta3=0.9999 이면 첫 스텝 업데이트가 정상 상태의 x32.7, 71스텝째 x11.5 입니다.
  - 지금 `config_moss.toml` 도 betas `(0.9, 0.999, 0.9999)` 입니다.
- **고치는 법(검증함):** EMA 상태는 그대로 두고 읽을 때만 나눕니다.
  - 2차 모멘트 열 인자를 `1-b2^t` 로, 잔차 열 인자를 `1-b3^t` 로, 모멘텀을 `1-b1^t` 로 나눕니다.
  - 이렇게 하면 초반 프로파일이 AdamW 와 같아집니다(t1 x3.4, t24 x1.0). 정상 상태 값은 변하지 않습니다.
- **기본 경로: AdamW + 호스트 상주 상태.**
  - fp32 master, m, v 는 일반 호스트 RAM 에 둡니다.
  - GPU 에는 bf16 계산 가중치만 상주합니다. fp32 스트림을 켜면 10.07 GiB 이고, 5.08B 전부 올라갑니다.
  - 마이크로스텝마다 bf16 기울기를 고정 메모리 링(1 GiB)으로 내리고, 호스트에서 fp32 로 누적합니다.
  - 마지막 마이크로스텝의 backward 안에서 64 MiB 조각 단위로 CPU fused AdamW 를 돌리고, 새 가중치를 바로 GPU 로 올립니다.
- **호스트 RAM**(추정): GA>1 이면 약 85 GiB(master·m·v·누적 각 18.93), GA=1 이면 약 66 GiB 입니다.
- **VRAM**(추정, 1024², fp32 스트림): batch 1 약 14.6, batch 2 약 18.3, batch 4 약 25.8 GiB(할당 기준)입니다.
- **속도**(추정): 1024 에서 batch 4 x GA 2(스텝당 8장)가 약 9.5~9.7 s/step(약 0.83 장/s)입니다.
  - batch 1 x GA 8 은 이 PC 의 메모리 대역폭에 막혀 약 13 s/step 입니다.
  - 측정값: CPU AdamW 2.8 s, 누적 1.1 s, 마지막 스텝 3.7~4.4 s.
- **대안 경로:**
  - (b) Adafactor fused: VRAM·RAM 은 가볍습니다. 대신 GA=1 만 되고, lr 1e-6 에서 확률적 반올림 잡음이 의도한 이동의 2~9배입니다(측정).
  - (c) CAME 호스트판: 편향 보정을 켜야 하는 옵션입니다. 지금 CPU 로는 약 20 s/step 이라 느립니다.
- **LR 일정:** `--moss_ft_lr_phases "2:5e-6,6:1e-6"` 와 짧은 warmup 으로 줍니다.
- **저장:**
  - 학습된 DiT 는 Base 와 같은 키 463개의 bf16 파일로 저장합니다. ComfyUI 에서 그대로 읽힙니다.
  - 학습 state 는 61 GB 입니다. 원자적으로 쓰고, 마지막 N개와 단계 경계 state 를 남깁니다.
- **주의할 디스크 용량:**
  - TE 캐시는 캡션 변형 수에 비례합니다. 지금 설정(변형 18)으로는 1.6만 장에 0.7 TB 가 듭니다.
  - 데이터셋과 출력은 D: 에 두어야 합니다.

---

## 0. Decisions at a glance

1. **New script `moss_train.py`.** The new code lives in small modules `library/moss/ft_*.py`.
   - `moss_train_network.py` with LyCORIS `algo=full` stays out of scope and is refused with a pointer to the new script.
     It crashes at `library/moss/models.py:476`, because LyCORIS deletes `weight`.
   - Even if that crash were fixed, it needs about 70.9 GiB of VRAM for `preset=moss`
     (`library/lycoris_algo_fixes.py:4007-4014`).
2. **Default optimizer mode `adamw_host`** (section 3.1):
   - **Host RAM (pageable):** fp32 master, exp_avg and exp_avg_sq.
   - **GPU:** the bf16 compute copy, fully resident.
   - **Gradients:** every micro-step's bf16 gradients stream to the host through a small pinned ring and are summed there in fp32.
   - **Step:** the AdamW step runs on the CPU (torch fused kernel). For the window's last micro-step it runs piece by piece inside the backward.
   - **Upload:** updated weights go back per piece on a copy stream.
3. **Fallbacks:**
   - `adafactor_fused` (b): the fork's existing fused Adafactor with stochastic rounding. GA=1 only.
   - `came_host` (c): CAME math on the host, with an opt-in, state-compatible bias correction. β3 > 0.99 is refused without it.
4. **Precision:**
   - bf16 compute.
   - `--moss_stream_fp32` is required. Base's residual stream needs it; the user's measurement (c) had bf16-stream LoKr gradients at cos 0.58.
   - fp16 compute (phase 1b) is built (2026-11-17, plan `moss_ft_te_live_fp16_plan.md` part B): `mixed_precision = "fp16"` with `--moss_stream_fp32` and a dynamic loss scale; the example config of `moss_train.md` uses it, bf16 stays available. 7.2 / G3 compare them on the GPU.
5. **Run shape** at the 1024 tier:
   - Recommended: batch 4 x GA 2 (8 images per step), if the smoke run confirms its VRAM. Otherwise batch 2 x GA 4.
   - batch 1 x GA 8 works, but on this PC it is host-memory-bandwidth bound (section 3.5).
6. **LR:** `--moss_ft_lr_phases "2:5e-6,6:1e-6"` and `--lr_warmup_steps` (section 4.6).
7. **Outputs:**
   - Trained DiT: bf16 safetensors in the Base layout. These are the same 463 keys, and ComfyUI's MOSS detection reads them.
   - Training states: in an own atomic format, 61 GB each, keep-last plus states pinned at LR-phase boundaries.
8. **CAME on MOSS:** it fails the same way as on Anima (section 3.3). The default here is AdamW, which is bias-corrected.

## 1. Facts this design rests on

### 1.1 Model (counted on the meta device by the code reader)

| Group | Params | Notes |
|---|---|---|
| Joint blocks: 42 x (q, k, v, o 3072x3072, ffn_up_gate 16384x3072, ffn_down 3072x8192) | 4.7563B (113.25M per block) | `layers.N.proj.*`, bf16 compute copies |
| Text refiner, 2 blocks | 226.6M | `text_embedder.blocks.*`; with biases; fp32 under `--moss_stream_fp32` |
| shared_block_modulation | 56.6M | the largest non-block tensor: 18432 x 3072 |
| video_head_modulation | 18.9M | |
| text_embedder.proj | 12.6M | |
| time_embedding | 10.2M | |
| video_embedder + final_linear_video | 0.8M | |
| norms (all) | 0.27M | |
| **Total** | **5.0823B** | non-block part 326.0M |

Sizes:
- fp32 all: 18.93 GiB.
- bf16 all: 9.47 GiB.
- **Compute copy with the fp32 stream:** 8.86 GiB (bf16 joint Linears) + 1.21 GiB (fp32 for the rest) = **10.07 GiB**.
- AdamW m + v in fp32: 37.87 GiB.

### 1.2 The Base file and ComfyUI (read from the files)

- **Base file** `moss_image_base_bf16.safetensors`:
  - 463 tensors: 462 BF16 and `rope.bands` F32 [16]. 5,082,308,624 elements.
  - Bare keys (`layers.0.proj.q.weight`, `shared_block_modulation.weight`, ...), with q / k / v kept split.
  - Header metadata `{"format": "pt"}`.
  - Keys are written in sorted order.
- **ComfyUI** (`comfy/model_detection.py:1050-1058`):
  - It detects MOSS from `shared_block_modulation.weight` + `layers.0.proj.ffn_up_gate.weight`.
  - It sets `fused_qkv` only when `layers.0.proj.qkv.weight` exists.
  - `supported_models.MOSSImage.process_unet_state_dict` pops `rope.bands`.
- **So a trained DiT saved with the Base key set and dtypes loads in ComfyUI unchanged.**

### 1.3 What the fork already has (from the three code readers)

**MOSS LoRA path, reusable:**
- `library/moss/models.py`: model, gradient checkpointing (plain / cpu-offload / unsloth / no-grad node), `enable_stream_fp32`, `set_stream_scale`.
- `strategy.py`, `captions.py`, `flow.py`, `vae.py`, `qwen35.py`.
- From `train_utils.py`: `sample_images`, the sample-prompt cache, `validate_moss_*`, `dataset_group_limits/tiers`, `moss_metadata`.
- `load_moss_dit` (`moss_train_network.py:43-64`), which also splits fused qkv.

**Full-FT pieces:**
- `anima_train.py`: the skeleton. It has no step-count resume, and block swap is broken with a normal step.
- `library/adafactor_fused.py`: `copy_stochastic_`, `adafactor_step_param`.
- `library/tsm_offload.py`: pinned slab and copy-stream patterns.
- `library/safetensors_utils.py`: `mem_eff_save_file` (one tensor at a time) and `MemoryEfficientSafeOpen`.

**Nothing in the fork:**
- keeps a separate fp32 master of a whole model;
- accumulates grads on the host;
- saves a full MOSS DiT.

**Why not LyCORIS `algo=full` (LoRA trainer):**
- It crashes at `models.py:476` (`self.proj["q"].weight.dtype` after LyCORIS deletes `weight`). The same pattern is at 387, 671-673, 723 and 758.
- It keeps the trained tensors fp32 on the GPU: 17.72 GiB weights + 17.72 grads + 35.4 AdamW = 70.9 GiB.

### 1.4 CPU measurements on this PC (measured)

Unit: one joint block's worth of fp32 tensors = 113.25M elements. "Whole DiT" = x44.9.

| Operation | ms / block | Whole DiT |
|---|---|---|
| `torch._fused_adamw_` fp32 (param, grad, m, v) | 62.5 | 2.80 s |
| `acc.add_(bf16 grad)` (mixed dtype, one call) | 50.1 | 2.25 s |
| chunked `tmp.copy_(bf16) ; acc.add_(tmp)`, 4M chunks | 25.5 (22-25 at 16-24 threads) | 1.0-1.15 s |
| `acc.copy_(bf16 grad)` (first micro-step of a window) | 20.8 | 0.93 s |
| `out_bf16.copy_(master)` (cast for upload) | 18.7 | 0.84 s |
| final step, chunked (convert + add acc + AdamW + cast), 1M chunks | 83-97 | 3.7-4.4 s |
| GA=1 step, chunked (convert + AdamW + cast) | 81 | 3.64 s |
| CAME `_core_factored_full_fp32` (came.py) on CPU | 457 | 20.5 s |

Thread scaling of the final step:
- 8 threads: 4.81 s.
- 12 threads: 4.23 s.
- 16 threads: 3.87 s.
- 20 threads: 3.73 s.
- 24 threads: 3.93 s.

It is DRAM-bound above about 16 threads, so **the worker uses 16 threads by default**. `torch.set_num_threads` called in the worker thread is thread-local: the worker saw 8, the main thread still 24 (measured).

API facts checked in torch 2.14 on the CPU:
- `torch._fused_adamw_` refuses bf16 grads with fp32 params ("expected scalar type Float but found BFloat16"), so grads are converted to fp32 chunks first.
- `grad_scale=s` gives the same bits as dividing the grads by s.
- `found_inf=1` skips the update and leaves the step counter unchanged.
- **Stepping a tensor in pieces with the raw kernel**, with the step counter incremented once beforehand, is **bit-identical** to one `torch.optim.adamw.adamw(..., fused=True)` call on the whole tensor.

Autograd checks (tiny `MossTransformer` on the CPU):
- `Tensor.register_post_accumulate_grad_hook` fires **exactly once per parameter per backward** in four modes: no checkpoint, plain checkpoint, `--unsloth_offload_checkpointing`, and the no-grad first-pass node (`set_ckpt_reuse(True, False)`).
- The grads seen in the hook are bit-identical to the reference.
- Setting `p.grad = None` inside the hook works.
- Order in which the hooks fire:
  `final_linear_video > final_norm_video > layers.41 ... layers.0 > text_embedder.blocks.1 > .blocks.0 > text_embedder.proj > video_embedder > video_head_modulation > shared_block_modulation > time_embedding`.

Saved activations of one full-size joint block in its recompute:
- Conditions: eager, CPU bf16 autocast, fp32 stream input.
- **264.6 KB per token** (129.2 MiB at 512 tokens, linear in tokens).
- On CUDA the compiled elementwise chains save less (not measured).

### 1.5 Precision facts

**bf16 weights lose small updates.** Share of AdamW steps lost to bf16 round-to-nearest (code reader, real Base weights):

| Part | lr 5e-6 | lr 1e-6 |
|---|---|---|
| Joint Linears | 96.2% | 99.0% |
| Text refiner | 96.2% | 99.1% |
| Embedders and modulation | 71.4% | 87.4% |

**Stochastic rounding (SR) keeps the mean but adds noise** (measured: 1M weights ~N(0, 0.02), 2000 steps of +-lr):

| lr | gradient | RTN error / intended move | SR error / intended move |
|---|---|---|---|
| 5e-6 | pure noise | 0.97 | 4.10 |
| 5e-6 | weak drift (55 % one sign) | 0.96 | 0.90 |
| 1e-6 | pure noise | 0.99 | 9.41 |
| 1e-6 | weak drift | 0.99 | 2.05 |

So SR noise is 2-9x the intended movement at 1e-6. For pure noise this is consistent with sqrt(ulp/lr): about 8-11 for |w| between 0.01 and 0.02.

**An fp32 master keeps every AdamW step.** Per-element steps of about lr x (m/√v) ≥ 1e-8 absolute are ≥ 5e-7 relative to |w| ≈ 0.02. fp32 resolves about 6e-8 relative.

**Weight decay at these LRs is below fp32 resolution.**
- lr · wd = 1e-6 x 0.01 = 1e-8 relative per step. It rounds away even in fp32 (the user's lesson: "wd·lr below about 3e-8 is a no-op").
- `adamw_host` therefore defaults to `weight_decay=0.0`. It logs one line when `lr x wd < 6e-8`.

**Residual stream.** The user's precision run (`UR\ANIMA\_chains\20261115_moss_fp16_fold`, 11-16) measured LoKr gradient error against an fp32 reference:

| Arm | Error | cos |
|---|---|---|
| (a) fp16 + stream fp32 | 18.6 % | 0.989 |
| (c) bf16 with a **bf16** stream | 82 % | 0.58 (blocks 0-13: 0.11) |

bf16 compute with an fp32 stream was not measured. That arm is the first item of the smoke run (7.2).

## 2. Memory budget

### 2.1 GPU (RTX 5090: 31.84 GiB on the device)

What the budget must leave room for:
- The user's measurement has about 2.3 GiB in use outside the trainer: desktop, other programs, CUDA context.
- The user's warning line is 98 % of the device (31.2 GiB).
- So the trainer's reserved peak should stay **≤ about 28.5 GiB**.
- The user forbids a VRAM cap and per-step `empty_cache()`: an OOM must surface.

**Estimate**, allocated GiB. Conditions: bf16 compute copy, `--moss_stream_fp32`, every block checkpointed (plain), text T = 300 tokens per sample. Modes (a) and (c) are the same on the GPU; (b) adds about 0.03 GiB of Adafactor state.

| | 1024² b1 | 1024² b2 | 1024² b4 | 1536² b1 | 1536² b2 | 2048² b1 |
|---|---|---|---|---|---|---|
| joint tokens per sample (image + text) | 4,396 | 4,396 | 4,396 | 9,516 | 9,516 | 16,684 |
| compute weights (8.86 bf16 + 1.21 fp32) | 10.07 | 10.07 | 10.07 | 10.07 | 10.07 | 10.07 |
| checkpointed block inputs (42 x 12,288 B/token, fp32) | 2.11 | 4.23 | 8.45 | 4.57 | 9.15 | 8.02 |
| one block's recompute + backward (1.4 x 264.6 KB/token, eager upper bound) | 1.52 | 3.03 | 6.07 | 3.28 | 6.57 | 5.76 |
| grads in flight to the host (ring-bounded) | ≤0.5 | ≤0.5 | ≤0.5 | ≤0.5 | ≤0.5 | ≤0.5 |
| rope, modulation, head, fp64 loss, cuBLAS workspace | 0.4 | 0.5 | 0.7 | 0.5 | 0.7 | 0.7 |
| **total allocated** | **14.6** | **18.3** | **25.8** | **18.9** | **27.0** | **25.0** |
| device with ~7 % allocator slack + 2.3 outside | 17.9 | 21.9 | 29.9 | 22.5 | 31.2 | 29.1 |

Reading the table:
- 1024 b4 and 2048 b1 fit but are tight.
- 1536 b2 is at the warning line: do not use it.
- The compiled elementwise chains (default on CUDA) should lower the recompute row. The smoke run measures it.

Options and their effect:
- `--moss_ckpt_reuse_attn` (flash, reuse-only) adds 6,240 B/token/block = 262 KB/token: +1.07 GiB at 1024 b1, +4.3 GiB at 1024 b4. Use its budget option, or leave it off at b4.
- `--unsloth_offload_checkpointing` moves the "block inputs" row to pinned host RAM. It costs about 12 % time (user's Anima measurement).
- **Without `--moss_stream_fp32`** the inputs row halves, but see 1.5 (refused by default).
- fp32 compute (`mixed_precision = "no"`) is about 24.7 GiB at 1024 b1. It is allowed only as a precision reference.

### 2.2 Host RAM (128 GB = 119.2 GiB)

**Estimate**, GiB:

| | adamw_host, GA>1 | adamw_host, GA=1 | came_host, GA>1 | adafactor_fused |
|---|---|---|---|---|
| fp32 master (trainable params) | 18.93 | 18.93 | 18.93 | - |
| exp_avg | 18.93 | 18.93 | 18.93 | - (beta1=None) |
| exp_avg_sq | 18.93 | 18.93 | ~0.01 (factored rows/cols) | - (GPU, tiny) |
| fp32 grad accumulator (only GA>1 or `--moss_ft_clip_mode exact`) | 18.93 | - | 18.93 | - |
| pinned rings (D2H 1.0 + H2D 0.5) | 1.5 | 1.5 | 1.5 | - |
| process, CUDA host side, 2-4 DataLoader workers, CPU VAE (estimate) | ~8 | ~8 | ~8 | ~8 |
| **total** | **~85** | **~66** | **~66** | **~8** |

- Only the rings are pinned. Master, m, v and the accumulator are ordinary pageable memory, because the step runs on the CPU and never DMAs them.
  - This avoids page-locking ~60 GiB on Windows (untested) and torch's power-of-two rounding of pinned allocations.
  - Both rings are single power-of-two allocations, so nothing is wasted.
- The 9B text encoder is used only before these pools are allocated (section 4.3), so it never adds to this table.
- `--unsloth_offload_checkpointing` adds the block-input row of 2.1 as pinned host memory (8.45 GiB at 1024 b4).
- **The user must close RAM-heavy programs for `adamw_host` GA>1.** ComfyUI with models in RAM is an example.
  - The startup check (4.5) refuses when `psutil.virtual_memory().available` < total + 8 GiB.

### 2.3 Disk

| Item | Size | Notes |
|---|---|---|
| Trained DiT, bf16 (Base layout) | 10.16 GB per save | `--save_precision float` gives 20.3 GB |
| Training state, `adamw_host` | 61.0 GB | master, m, v fp32 (5.0823e9 x 12 B) + small files |
| Training state, `came_host` | 40.7 GB | master + exp_avg + factored |
| While a state is written | +1 state | the new one is complete before the old one is deleted |
| TE cache (`*_moss_te.npz`) | 8 KiB per text token per caption variant | T = 300: 2.46 MB per image per variant |
| | 16k images | 39 GB x variants (**18 variants = 0.7 TB**) |
| | 56k images | 138 GB x variants (**18 variants = 2.5 TB**) |
| Latents at 1024² (`*_1024x1024_moss.npz`, 128x64x64) | 2 MiB fp32, 1 MiB with `--cache_latents_dtype bf16` | double that with flip_aug |
| | 16k images | 16-32 GB |
| | 56k images | 56-112 GB |

**Consequences:**
- Use 1-3 caption variants for a 16k-56k set, not the LoRA config's 18.
- Keep the dataset (caches are written next to the images), `output_dir` and `--moss_ft_state_dir` on **D:** (4 TB). C: had 65-90 GB free per the notes.
- Plan state retention as (kept states + 1 + pinned phase states) x 61 GB. One kept + one pinned + one being written is about 183 GB at the peak.

### 2.4 Text encoder and latents

- The TE (Qwen3.5-9B bf16, 16.8 GB) runs on the GPU only while caching, before the DiT exists. Then `provider.release()`. The existing `MossNetworkTrainer.cache_text_encoder_outputs_if_needed` logic is reused as is (4.2).
- With complete caches the TE is never loaded.
- `--cache_text_encoder_outputs` and `--cache_latents` are required.
- Both `*_to_disk` flags are required when the training set has more than 2,000 images. In-memory caches of 16k images would take tens of GB of RAM next to the 85 GiB of 2.2.
- The disk caches are read per item by the DataLoader workers. About 4 MB per image at 0.8 images/s is trivial I/O.
- The VAE (fp32) stays on the CPU and moves to the GPU only for sample decoding (`train_utils._decode_and_save`).
- **Live text encoder (`--moss_te_live`, built 2026-11-17, plan `moss_ft_te_live_fp16_plan.md` part A):** the
  alternative to the TE cache. The 32 decoder layers stay in host RAM (copy: +12.89 GiB private + 1 GiB pinned staging;
  pinned: 12.89 GiB page-locked; mmap: the OS file cache) and are streamed per pass through a 2-slot device ring
  (0.81 GiB); a pass encodes the captions of the next 32 micro-batches (~1.0 GiB at its peak, between micro-steps, so
  the 2.1 peak is max(micro-step, weights + 0.25 GiB + pass) + the held outputs and latents, ~0.14 GiB at 1024² b1).
  Host totals of 2.2 with it in copy mode: `adamw_host` GA 1 80.2 GiB, GA > 1 99.1 GiB (+8 GiB margin). No TE cache
  on disk (125k images x 3 variants would be ~0.92 TB). Then only `--cache_latents` is required, and above 2,000
  images only `--cache_latents_to_disk`. The estimates are `ft_budget.te_live_*`; the 2.1 / 2.2 tables above stay the
  cached path's.

## 3. Optimizer modes

### 3.1 (a) `adamw_host` - the default

#### Data flow per micro-step

```
GPU compute stream:  forward (bf16 weights, fp32 stream)  ->  backward (checkpointed blocks, 41 ... 0, then refiner, embedders)
                                                              |  post-accumulate-grad hook per parameter (autograd thread):
                                                              |   event R(p) on the compute stream
GPU d2h stream:                                               +-> wait R(p); fp32 norm of grad(p); copy grad in <=64 MiB pieces
                                                                  into pinned ring entries; event D(piece); grad freed
host worker thread (16 torch threads):                            wait D(piece); then
   non-final micro-step:  acc[p][piece] = piece (first contribution in the window) or += piece   (fp32, chunked convert)
   final micro-step:      g = acc[p][piece] + piece (or piece when GA = 1); AdamW on master/m/v[p][piece] (fused kernel,
                          fp32); cast master[p][piece] -> compute dtype into a pinned upload entry
GPU h2d stream:           wait R(p); copy the upload entry into p.data[piece]
end of the final micro-step: main thread waits until the worker is idle (the "tail"); compute stream waits for h2d
```

#### Components

Code lives in `library/moss/ft_host.py`; the optimizer math in `library/moss/ft_optim.py`.

**`ParamSlot`** holds `index, name, param, numel, offset, compute_dtype, group`.
- Order: `model.named_parameters()` order of the trainable set. This is also the order in state files.

**`HostPool`**: one contiguous fp32 CPU tensor per role (`master`, `exp_avg`, `exp_avg_sq`, `acc`), with per-slot flat views.
- Pageable memory.
- `acc` is allocated only for GA > 1 or `--moss_ft_clip_mode exact`.
- Per-slot step counters: one float32 CPU scalar per slot, as `torch._fused_adamw_` wants.

**`PinnedRing(total_bytes, piece_bytes)`**: one pinned `torch.empty(total_bytes, dtype=uint8, pin_memory=cuda)`.
- Split into `total/piece` entries. Defaults: D2H 1024 MiB / 64 MiB = 16 entries; H2D 512 MiB / 64 MiB = 8 entries.
- `acquire()` blocks on a semaphore and wakes every 0.5 s to re-raise a worker error. `release(entry)`.
- An entry is viewed as `torch.bfloat16` (32M elements) or `torch.float32` (16M elements).
- A grad larger than one piece goes in several entries: ffn_up_gate is 2 pieces; shared_block_modulation (fp32, 226 MB) is 4.

**`GradPipeline`** has these methods:
- `attach()`: registers `register_post_accumulate_grad_hook` on every trainable parameter.
- `begin_window(n_micro, ctx)`: takes a `StepContext`. This is a snapshot of each group's lr / betas / eps / weight_decay, the clip coefficient and the step index. The worker never reads `optimizer.param_groups` itself.
- `begin_microstep(j)`.
- `end_microstep()`: non-final micro-steps return at once; the final one drains and returns `WindowStats`.
- `step_from_accumulator(clip_coef)`: for exact clipping.
- `close()`.

**The hook** (runs in the autograd device thread):
1. Record `R(p)` on the current stream.
2. On the d2h stream: wait `R(p)`; `torch.linalg.vector_norm(g, dtype=torch.float32)` into `norm_dev[slot]`; copy it non-blocking into pinned `norm_host[slot, j]`.
3. For each piece: acquire an entry, copy it non-blocking, record `D`, and enqueue an `Item(kind, slot, start, numel, entry, D, R(p), j)`.
4. `g.record_stream(d2h)`, then `p.grad = None`.

Grads of Linear weights are contiguous. Assert it, and fall back to `reshape(-1)`.

**The worker**: one daemon thread with a FIFO `queue.Queue`. FIFO order makes every earlier micro-step of a parameter land in `acc` before its final piece.
- **Accumulation:**
  - The first contribution of a parameter in a window is `copy_`. No zero fill is needed.
  - Later ones are `tmp.copy_(piece_chunk); acc_chunk.add_(tmp)` with 4M-element chunks.
  - Per-parameter bookkeeping: "micro-steps seen in this window".
- **Final pieces (AdamW):** for each 1M-element chunk:
  1. `t.copy_(piece_chunk)`; `t.add_(acc_chunk)` if the window has earlier contributions.
  2. `sumsq += torch.dot(t, t)` (Python float).
  3. `torch._fused_adamw_([master_chunk], [t], [m_chunk], [v_chunk], [], [step[slot]], lr=..., beta1, beta2, weight_decay=group wd, eps, amsgrad=False, maximize=False, grad_scale=1/c or None)`.
  4. `upload_chunk.copy_(master_chunk)`.
- **Step counter:** `step[slot] += 1` once, before the first final piece of that slot.
- **Upload:** after a piece, an H2D copy on the h2d stream that waits `R(p)`. Its event releases the upload entry.
- **Skipping (non-finite gradients):** a slot is "poisoned" when any of its window's `norm_host[slot, 0..j]` is not finite. A poisoned slot keeps its master, m, v and step, gets no upload, and is counted.
  - This is per parameter, like KMT. `torch.optim` with GradScaler would skip the whole step instead (see Risks).
- **Pieces in a window are never processed out of order:** one worker thread, FIFO.

**Drain**: `queue.join()` (the worker calls `task_done`), then `torch.cuda.current_stream().wait_stream(h2d)`. The CPU wait is the step tail. The upload completion is a stream dependency.

**Exceptions** in the worker are stored. The next `acquire()` or the drain re-raises them as `RuntimeError` with the worker traceback. The trainer then exits non-zero.

#### Why the in-backward step is safe

- A parameter's hook fires after its `AccumulateGrad`. The node that read the weight (dX = dY·W and dW) was already enqueued.
- Each block's weights are read only inside that block's forward, recompute and backward.
- So once `R(p)` completes on the compute stream, no later kernel of this backward reads `p`. Overwriting it from h2d after `R(p)` is ordered.
- Blocks not yet backpropagated still hold their old weights, as their recompute needs.
- This also holds for the no-grad node and stored attention (`--moss_ckpt_reuse_attn`). The upload of block i comes after block i's node backward.

#### Clipping (`--max_grad_norm` > 0)

- **Default `--moss_ft_clip_mode stale`** (KMT's idea): the in-backward step cannot know this window's global norm.
  - The coefficient is `c_t = min(1, max_norm / (G_{t-1} + 1e-6))`, with G the previous window's norm. The worker sums `sumsq` over the final pass.
  - A window with no known previous G (the first after start or after a resume without a stored G) runs in exact mode.
- **`exact`**: the final micro-step only accumulates; its kind adds `sumsq`.
  - After the drain the main thread computes `c`, then `step_from_accumulator(c)` steps every slot from `acc` and uploads.
  - The tail becomes the whole host step (3.7-4.4 s).
  - It needs `acc` even at GA = 1 (+18.93 GiB host).
- `max_grad_norm = 0`: no clipping and no staleness question.
- Log `grad_norm` (the window's G) and `clip_coef` every step. Recommendation for the first smoke run: `max_grad_norm = 0`. Read the natural norm, then set it to about 2-3x the median so it only catches spikes.

#### Equivalence target

With fp32 compute (`mixed_precision = "no"`, the CPU tests), `adamw_host` must be **bit-identical** to the reference:
- `torch.optim.AdamW(fused=True)`;
- on an fp32 model with `loss / n_micro` and the same backward order;
- with `torch.nn.utils.clip_grad_norm_` in exact mode.

The accumulation order (g1, then +g2, then +g3, in fp32) and the elementwise kernel are the same (1.4).

#### Partial windows

- The last window of an epoch has `n_micro = len(dataloader) - w*GA` < GA, as accelerate syncs at the end of the dataloader.
- Each micro-step's loss is divided by its window's `n_micro`.

### 3.2 (b) `adafactor_fused` - memory-light fallback

- Selected by `optimizer_type = "Adafactor"` + `--fused_backward_pass`. It uses `train_util.get_optimizer` and `library.adafactor_fused.patch_adafactor_fused` exactly as `anima_train.py:409-428`.
- The hook steps the parameter and sets `grad = None`.
- **Weights:**
  - Joint Linears are bf16 and are written back with `copy_stochastic_`.
  - The fp32-stream parameters are fp32 and get exact fp32 updates (`adafactor_step_param` skips SR for fp32).
  - No host state at all.
- **Required `optimizer_args`:** `scale_parameter=False, relative_step=False, warmup_init=False`, `beta1` unset. A beta1 would allocate a full fp32 first moment on the GPU.
- **Limits:**
  - Gradient accumulation must be 1 (`train_util.py:6776-6782`), so a step's batch is the micro-batch: at most 4 at 1024.
  - `max_grad_norm` is applied per tensor in the hook, as in `anima_train.py`, not globally.
  - SR noise at lr 1e-6 is 2-9x the intended movement (1.5). It is acceptable only for the 5e-6 phase or short experiments.
  - Not AdamW, so the optimizer differs from the user's recipe.
- **Speed:** no host work, so GPU-bound (about 4.5 s per 4 images at 1024, estimate).
- **State:** the GPU bf16 / fp32 model and the small Adafactor state. It is saved in the same state directory format (4.7) with `master.safetensors` holding the model values (no fp32 master exists in this mode).

### 3.3 (c) `came_host` - CAME with opt-in bias correction

#### The failure, measured

Setup:
- The reference math below, which is bit-identical to came.py's core when given came.py's fp32 betas.
- Synthetic gradients 0.2·μ + N(0,1), 128x128, 30,000 steps.
- Values are RMS(Δp)/lr at step t over the same optimizer's late steady value.
- The uncorrected rows agree with the code reader's run of came.py's real `_core_factored_full_fp32` (x33 / x19 / x11.6 / x7.3 / x3.2 / x1.9 / x1.2).

| Optimizer | t1 | t2 | t5 | t10 | t24 | t71 | t180 | t1000 | t3000 | t10000 |
|---|---|---|---|---|---|---|---|---|---|---|
| CAME β3=0.9999 | x32.7 | x31.7 | x29.0 | x25.6 | x19.2 | x11.5 | x7.3 | x3.2 | x1.9 | x1.2 |
| CAME β3=0.9999 + correction | x3.4 | x2.4 | x1.6 | x1.3 | x1.0 | x1.0 | x1.0 | x1.0 | x1.0 | x1.0 |
| CAME β3=0.999 | x10.6 | x10.3 | x9.4 | x8.4 | x6.3 | x3.8 | x2.5 | x1.3 | x1.0 | x1.0 |
| CAME β3=0.999 + correction | x3.4 | x2.4 | x1.6 | x1.3 | x1.0 | x1.0 | x1.0 | x1.0 | x1.0 | x1.0 |
| torch AdamW (bias-corrected) | x3.4 | x2.4 | x1.6 | x1.3 | x1.0 | x1.0 | x1.0 | x1.0 | x1.0 | x1.0 |

This is model-independent. MOSS breaks like Anima: the Anima notes say β3 = 0.9999 broke a full fine-tune in 24 steps.

#### The correction (state-compatible "read-out" correction)

- Keep every EMA update exactly as came.py.
- Divide only where the EMAs are read. With bc_k = 1 - β_k^t, and t the parameter's step count after its increment (1-based):
  - **factored path:**
    - `c_factor = (exp_avg_sq_col / bc2).rsqrt()`
    - `c_factor_res = (exp_avg_res_col / bc3).rsqrt()`
    - final update uses `exp_avg / bc1`
    - the row factors are ratios (`row / mean(row)`) and need nothing
  - **unfactored path (1-D parameters):**
    - `rsqrt(exp_avg_sq / bc2)`
    - final update `exp_avg / bc1`
    - with `non_factored_confidence`: `sqrt(exp_avg_res / bc3)`
  - Everything else is unchanged: residual `(update - exp_avg)^2` with the uncorrected `exp_avg`, RMS clip, cautious mask, weight decay, alphas `1 - β` in fp32 as came.py passes them.
  - `ams_bound` with the correction is refused.
- Because the EMA buffers are unchanged, a state saved with or without the correction loads into either.
- At t = 1 the corrected update is about lr x 1.11·sign(update). Later it decays like Adam's.

Reference math, used in the simulation above:

```python
def came_factored_step(grad, m, sq_r, sq_c, rs_r, rs_c, b1, b2, b3, t, lr, p, eps1=1e-30, eps2=1e-16, clip=1.0,
                       bias_correction=True):
    bc1, bc2, bc3 = ((1 - b1 ** t), (1 - b2 ** t), (1 - b3 ** t)) if bias_correction else (1.0, 1.0, 1.0)
    u = grad * grad + eps1
    sq_r.mul_(b2).add_(u.mean(-1), alpha=1 - b2); sq_c.mul_(b2).add_(u.mean(-2), alpha=1 - b2)
    r = (sq_r / sq_r.mean(-1, keepdim=True)).rsqrt().unsqueeze(-1)
    c = (sq_c / bc2).unsqueeze(-2).rsqrt()
    u = r * c * grad
    u = u / ((u.norm() / math.sqrt(u.numel())) / clip).clamp(min=1.0)
    m.mul_(b1).add_(u, alpha=1 - b1)
    res = (u - m).pow(2) + eps2
    rs_r.mul_(b3).add_(res.mean(-1), alpha=1 - b3); rs_c.mul_(b3).add_(res.mean(-2), alpha=1 - b3)
    rr = (rs_r / rs_r.mean(-1, keepdim=True)).rsqrt().unsqueeze(-1)
    cr = (rs_c / bc3).unsqueeze(-2).rsqrt()
    p.add_(rr * cr * (m / bc1), alpha=-lr)   # + weight decay / cautious exactly as came.py
```

With `bias_correction=False`, the same fp32 betas and alphas (`1 - fp32(β)`, as `CAME.step` passes them) and came.py's operation order, this reimplementation was **bit-identical** to came.py's `_core_factored_full_fp32` over 5 steps (measured). With Python-float betas it differs by 1.3e-5 after 5 steps. The implementation must take the betas and alphas as came.py does.

#### Where the code goes

- **`library/moss/ft_optim.py`**: `came_factored_core(...)` and `came_unfactored_core(...)`.
  - These are the fork's own implementation of the paper algorithm plus the correction.
  - They cover `update_strategy` unmodified / cautious and `weight_decouple`, `fixed_decay` and `cautious_weight_decay` as came.py does.
  - They are device-agnostic.
- `backend/custom_scheduler/LoraEasyCustomOptimizer/came.py` is in another repository with the user's uncommitted edits. **It is not edited by this work.**
  - Porting the same opt-in flag into it (for the LoRA trainer, where warmup currently masks the effect) is a separate follow-up that needs the user's go-ahead.

#### `came_host` in the pipeline

- The same hooks and accumulation as 3.1.
- CAME needs whole matrices (row/col means), so the final step is **per parameter**: after the last piece of a parameter is accumulated, the CAME core runs on `(master[p], acc[p], exp_avg[p], factored states)` on the CPU. Then all its pieces are cast and uploaded.
- Host RAM about 66 GiB (2.2).
- **Speed:** 20.5 s CPU per optimizer step (measured core), about 2.5x the AdamW step time. Phase 2 can move this core to a GPU-assisted step (section 8).
- **Startup rules:**
  - β3 > 0.99 without `bias_correction=True` is **refused**, quoting the table above. With β3 ≤ 0.99 the uncorrected CAME is allowed; the user's Anima `config_anima_fft_came.toml` ran clean with 0.99.
  - GPU-path options in `optimizer_args` are ignored with one warning: `state_storage_device/dtype`, `compile_step`, `group_shapes*`, `foreach`, `sync_chunk_size`.
  - `ams_bound=True`, `kahan_sum=True` and `update_strategy` grams / both are refused.

### 3.4 Why `adamw_host` is the default

- It is the user's recipe (AdamW 5e-6 then 1e-6).
- It has bias correction (no β3 amplification).
- It keeps 1e-6 updates exactly (fp32 master).
- It supports gradient accumulation with fp32 accumulated grads.
- It costs one CPU step per window, mostly hidden at batch ≥ 2.
- The fallback (b) has no accumulation and SR noise.
- (c) is slower in phase 1, and its β3 default is the known failure unless corrected.

### 3.5 Throughput model (estimate)

**GPU time per image at 1024²**, t_img ≈ 1.1 s:
- 218.6 TFLOP per 4,396-token micro-step, counting forward + recompute + backward and attention.
- At KMT's 5090 rate of 195 TF/s including recompute.
- Measure it in 7.3.

**Host per window:**
- (GA - 1) x A + F.
- A = accumulate pass, 1.0-1.15 s (measured).
- F = final step, 3.7-4.4 s at GA > 1, or 3.64 s at GA = 1 (measured).

**Overlap:**
- A overlaps the next micro-step's GPU time.
- F overlaps the last micro-step's backward, about 0.75 x b x t_img.
- Tail ≈ max(0, F - 0.75·b·t_img).

| Run shape (8 images per step) | GPU | Host | Tail | s/step | images/s | tok/s |
|---|---|---|---|---|---|---|
| batch 1 x GA 8 | 8.8 | 7 x 1.1 + 4.0 | 3.2 (micro-steps host-paced) | ~12-13 | ~0.63 | ~2,750 |
| batch 2 x GA 4 | 8.8 | 3 x 1.1 + 4.0 | 2.3 | ~11.1 | ~0.72 | ~3,150 |
| batch 4 x GA 2 | 8.8 | 1.1 + 4.0 | 0.7 | ~9.5-9.7 | ~0.83 | ~3,650 |
| batch 8 x GA 1 (phase 2: needs activation offload) | 8.8 | 3.6 | ~0 | ~8.9 | ~0.9 | ~3,950 |

Notes:
- KMT reported 2,432 tok/s for GA 8 and 4,088 tok/s for GA 1 on Linux. The model agrees.
- Wall time for 8 epochs at batch 4 x GA 2: **16k images ≈ 43 h, 56k images ≈ 150 h** (6.3 days). At batch 1 x GA 8: 57 h / 200 h.
- Other speed options that are allowed but measured only later (default off):
  - `--moss_ckpt_reuse_attn` (−6 % on the user's MOSS LoRA ABAB).
  - Block compile (−6 % on the LoRA), refused in phase 1 (8).

## 4. `moss_train.py` and its modules

### 4.1 Files (all new unless marked)

| File | Contents |
|---|---|
| `moss_train.py` | script: parser, `train(args)`, loop, saves, samples (mirrors `anima_train.py`) |
| `library/moss/ft_args.py` | `add_moss_ft_arguments`, `validate_moss_ft_args`, `optimizer_mode(args)`, `parse_lr_phases`, `make_lr_phase_scheduler` |
| `library/moss/ft_model.py` | `load_moss_for_full_ft`, `select_trainable`, `param_groups`, `save_moss_dit` |
| `library/moss/ft_host.py` | `ParamSlot`, `HostPool`, `PinnedRing`, `GradPipeline`, `StepContext`, `WindowStats`, the CPU-device fallbacks |
| `library/moss/ft_optim.py` | `HostAdamW`, `HostCAME` (`torch.optim.Optimizer` subclasses: param_groups for schedulers + per-piece / per-param math), `came_factored_core`, `came_unfactored_core` |
| `library/moss/ft_state.py` | state directory write / verify / load, keep policy |
| `library/moss/ft_loss.py` | `moss_ft_noise_and_loss(...)`: noise, sigmas, forward, flow-matching loss for one micro-step |
| `library/moss/ft_budget.py` | GPU / host / disk / throughput estimates of sections 2 and 3.5, start-up log and checks |
| `library/moss/te_cache.py` | `cache_moss_text_encoder_outputs`, `cache_moss_sample_prompts`, moved out of `MossNetworkTrainer` |
| `moss_train_network.py` (**changed**) | its two caching methods call `te_cache` (behavior-preserving), and `algo=full` with LyCORIS is refused with a pointer to `moss_train.py` |
| `docs/moss_train.md` | user guide, after implementation |

### 4.2 Reused code (unchanged)

**Model and loading:**
- `library/moss/models.py`: `MossTransformer`, `MOSS_DIT_CONFIG`, `split_fused_qkv`, `enable_gradient_checkpointing(unsloth_offload=...)`, `enable_stream_fp32()`, `set_stream_scale()`, `enable_elementwise_compile()`, `set_ckpt_reuse()`.

**Caching strategies:**
- `strategy.py`: `MossTokenizeStrategy`, `MossTextEncodingStrategy` (incl. `drop_cached_text_encoder_outputs`), `MossTextEncoderOutputsCachingStrategy`, `MossLatentsCachingStrategy`, `MossTextEncoderProvider`, `moss_te_signature`, `check_moss_text_encoder_path`.
- `captions.py` (canonical captions); `vae.py` (`load_moss_vae`).

**Noise and timesteps:**
- `flow.py`: `MossSigmaSampler`, `get_noisy_model_input_and_timesteps`, `describe_moss_sampling`.

**`train_utils.py`:**
- `add_moss_training_arguments`, `validate_moss_args`, `validate_moss_ckpt_reuse_args`.
- `dataset_group_limits/tiers`, `sample_images`, `sample_prompt_*`, `moss_metadata`.
- `cuda_memory_stats`, `outside_bytes`, `show_timesteps`.

**Common trainer code:**
- `library/train_util.py`: `prepare_accelerator`, `prepare_dtype`, `get_scheduler_fix`, `get_optimizer` (mode b only), `conditional_loss`, `get_huber_threshold_if_needed`, `EMARecorder`, `RateTracker`, `load_prompts`, `sample_images_check`, `get_epoch_ckpt_name`, `get_step_ckpt_name`, `get_remove_epoch_no`, `get_remove_step_no`, `cosine_optimal_transport`, constants `EPOCH_STATE_NAME` / `STEP_STATE_NAME` / `LAST_STATE_NAME`.
- `library/anima_train_utils.compute_loss_weighting_for_anima`.
- `library/custom_train_functions`: `apply_snr_weight_for_flow_matching`, `maybe_apply_antithetic_noise_pairing`, `apply_masked_loss`.
- `library/adafactor_fused.py` (mode b); `library/safetensors_utils.mem_eff_save_file` (state files).
- `library/cuda_memory.default_expandable_segments`; `library/tier_timing.TierTimer`; `library/sd3_train_utils.FlowMatchEulerDiscreteScheduler`.

**Parser composition**, checked on the CPU to build without argparse conflicts:
- `add_logging_arguments`, `add_sd_models_arguments`, `add_dataset_arguments(p, True, True, True)`, `add_training_arguments(p, False)`, `add_masked_loss_arguments`, `deepspeed_utils.add_deepspeed_arguments`, `add_sd_saving_arguments`, `add_optimizer_arguments`, `config_util.add_config_arguments`, `add_custom_train_arguments`, `add_dit_training_arguments`, `moss_train_utils.add_moss_training_arguments`, `sai_model_spec.add_model_spec_arguments`.
- Plus what `anima_train.setup_parser` adds itself: `--cpu_offload_checkpointing`, `--unsloth_offload_checkpointing`, `--flow_use_ot`, `--contrastive_flow_matching`, `--cfm_lambda` (default 0.02 as the MOSS LoRA trainer).
- Plus `add_moss_ft_arguments`.

**`validate_moss_args`** reads `args.network_train_unet_only` without `getattr` (`train_utils.py:630`). `moss_train.py` sets these before calling it instead of changing that function:
- `args.network_train_unet_only = True`
- `args.network_train_text_encoder_only = False`

### 4.3 Start-up order (host memory and VRAM peaks never overlap)

1. `verify_training_args`, `prepare_dataset_args(args, True)`, `setup_logging`, `cuda_memory.default_expandable_segments()`, seed.
2. `validate_moss_ft_args(args)` for the parts that need no dataset; `optimizer_mode(args)`.
3. MOSS strategies: tokenize, text encoding, TE-outputs caching (with `moss_te_signature`; not set with
   `--moss_te_live`, so the dataset tokenizes at every fetch), latents caching.
4. Dataset group (`BlueprintGenerator`, as in `anima_train.py:97-137`), `verify_bucket_reso_steps(16)`. Then:
   - `validate_moss_args(args, group, None)`;
   - `validate_moss_ft_args(args, group)`: the dataset-dependent parts and the budget checks of `ft_budget`.
5. `accelerator = train_util.prepare_accelerator(args)`. `num_processes > 1` is refused. `weight_dtype, save_dtype = prepare_dtype(args)`.
6. Latent caching:
   - VAE loaded with `load_moss_vae(args.vae, "cpu", fp32)`; moved to the GPU for caching, back to the CPU after.
   - Then `te_cache.cache_moss_text_encoder_outputs(...)`: canonical captions, dataset caching and sample prompts. It releases the TE.
   - With `--moss_te_live` instead: `HostTextEncoderWeights.open` (the layers into host RAM), `LiveTextEncoder` (the
     streaming engine) and `te_cache.cache_moss_sample_prompts(args, encoder)`; the encoder lives until `train()`
     ends and `TextEncodeAhead.wrap` encodes the captions per pass in the loop (after `skip_first_batches`).
7. **Only now:**
   - the `HostPool` is allocated;
   - the DiT is loaded by `load_moss_for_full_ft` (4.4.1);
   - checkpointing is switched on;
   - `enable_elementwise_compile(not args.moss_no_compile)`;
   - `set_ckpt_reuse(False, args.moss_ckpt_reuse_attn, args.moss_ckpt_reuse_attn_budget_gib)`.
8. Trainable set and groups, optimizer, LR scheduler, `GradPipeline.attach()` (modes a, c) or the Adafactor hooks (mode b).
9. Resume (4.7). Then the DataLoader (`batch_size=1`, `shuffle=True`, the collator, `max_data_loader_n_workers`, persistent), as `anima_train.py:336-344`.
10. Trackers, `sample_at_first`, loop.

#### 4.4.1 `load_moss_for_full_ft(path, args, device, trainable_fn) -> (model, pool, slots)`

1. Build `MossTransformer(**MOSS_DIT_CONFIG, attn_mode, split_attn)` under `torch.device("meta")`.
2. Pick the trainable set with `select_trainable(model, args.moss_ft_freeze)` and make the slots (registration order). Allocate `pool.master` and the other roles of 2.2.
3. Read the file **one tensor at a time** with `safetensors.safe_open(path, "pt", "cpu")`.
   - Strip the same prefixes as `load_moss_dit`.
   - A `layers.N.proj.qkv.weight` (heal layout) is split into q / k / v with `split_fused_qkv` on a one-key dict.
   - Per tensor:
     - Trainable: `master_view.copy_(t.float())`.
     - Always: keep `t` (in its file dtype) for the GPU copy.
4. `model = model.to_empty(device=device)`. Copy each parameter in the compute dtype (bf16 for every parameter at this point).
   - `model.rope.bands = model.rope._default_bands(device)`.
   - If `--moss_stream_scale` != 1: `model.set_stream_scale(...)`.
5. If `--moss_stream_fp32`: `model.enable_stream_fp32()`. The bf16 → fp32 cast of Base values is exact. The `Parameter` objects keep their identity.
6. `requires_grad_(True)` on trainable parameters, `False` on frozen ones. `model.train()`.
7. **Peak host memory:** the pools + one tensor (the file is mmapped).
8. **Assert:** for every trainable parameter, the GPU copy == `master.to(compute_dtype)`.

**Groups:**
- Group 0 ("decay"): 2-D weights of the joint blocks, the refiner blocks, the modulations, the time MLP, `final_linear_video`.
- Group 1 ("no_decay"): every 1-D parameter (biases, norm weights) and the two input embedders `video_embedder.*` and `text_embedder.proj.*`.
- Both groups use the same LR.
- `weight_decay` from `optimizer_args` applies to group 0 only.
- `--moss_ft_freeze` takes regexes matched with `re.fullmatch` on parameter names. Default: nothing frozen, all 5.08B trained.

### 4.4 Training loop

```python
for epoch in range(start_epoch, num_train_epochs):
    current_epoch.value = epoch + 1; dit.train()
    batches = len(train_dataloader)
    for step, batch in enumerate(dataloader_skipping(epoch_step_to_skip)):
        w, j = divmod(step, GA); n_micro = min(GA, batches - w * GA); final = (j == n_micro - 1)
        if j == 0:
            pipeline.begin_window(n_micro, StepContext.snapshot(optimizer, lr_scheduler, clip_coef(), global_step))
        pipeline.begin_microstep(j)                       # mode b: no-op
        timer.start(tokens_of(batch))
        loss = ft_loss.moss_ft_noise_and_loss(args, dit, batch, accelerator, sigma_sampler, noise_scheduler,
                                              stream_dtype, text_encoding_strategy) / n_micro
        loss.backward()                                   # hooks stream grads; final micro-step steps in backward
        stats = pipeline.end_microstep()                  # final: drain (tail) + compute stream waits for uploads
        timer.stop(text_tokens=dit.last_text_tokens)
        window_loss += loss.detach()                      # summed on the device, one .item() per window
        if final:
            lr_scheduler.step(); global_step += 1; current_step.value = global_step
            log_window(stats, window_loss.item(), lr, grad_norm, clip_coef, skipped, timings, memory)
            check_skip_streak(stats)                      # --moss_ft_max_skipped_steps
            maybe_sample(steps=global_step); maybe_save(on_epoch_end=False)
            if global_step >= args.max_train_steps: break
    maybe_save(on_epoch_end=True); maybe_sample(epoch=epoch + 1)
save_final(); save_state_on_train_end_if_asked()
```

**`moss_ft_noise_and_loss`** follows `MossNetworkTrainer.get_noise_pred_and_target` and `anima_train.py:607-651` without the LoRA-only hooks:
1. `stream_dtype` = fp32 with `--moss_stream_fp32`.
2. Noise: `noise = randn_like(latents, fp32)`. Optional OT (`train_util.cosine_optimal_transport`, batch > 1). Antithetic pairing.
3. `noisy, timesteps, sigmas = flow.get_noisy_model_input_and_timesteps(args, scheduler, latents, noise, device, stream_dtype, sampler)`.
4. With checkpointing: `noisy.requires_grad_(True)`.
5. Text: `drop_cached_text_encoder_outputs(*outputs[:-1], caption_dropout_rates=rates)`; text to `stream_dtype`; mask to bool.
6. `with accelerator.autocast(): pred = dit(noisy, timesteps, text, mask)`.
7. `target = noise.double() - latents.double()`.
8. Loss chain:
   - `loss = conditional_loss(pred.double(), target, args.loss_type, "none", huber_c)`;
   - x `compute_loss_weighting_for_anima(args.weighting_scheme, sigmas)`;
   - min-SNR;
   - CFM (batch > 1);
   - masked loss;
   - `.mean([1,2,3])`;
   - x `batch["loss_weights"]`;
   - `.mean()`.

**`dataloader_skipping(n)`**: `accelerate.skip_first_batches(train_dataloader, n)` for the first resumed epoch when `n > 0`, otherwise the plain loader.

### 4.5 Arguments

#### New (`add_moss_ft_arguments`)

| Argument | Default | Validation / meaning |
|---|---|---|
| `--moss_ft_lr_phases` | none | `"E1:LR1,E2:LR2[,...]"`, epochs float > 0, LR float > 0. Sum of E must equal `max_train_epochs` if given; else it sets it. `--learning_rate` unset or equal to LR1 (it is set to LR1). `--max_train_steps` refused. `--lr_scheduler` must be `constant` or `constant_with_warmup` (any other is refused: one schedule source). See 4.6 |
| `--moss_ft_freeze` | `""` | comma-separated regexes, `re.fullmatch` on parameter names. Refused when a regex matches nothing or when everything is frozen; logs the frozen count |
| `--moss_ft_clip_mode` | `stale` | `stale` or `exact` (3.1). `exact` allocates the accumulator even at GA = 1 |
| `--moss_ft_host_threads` | 16 | int in [1, os.cpu_count()]; intra-op threads of the host worker (thread-local) |
| `--moss_ft_piece_mib` | 64 | power of two in [16, 256] |
| `--moss_ft_ring_mib` | 1024 | D2H pinned ring; multiple of the piece size; ≥ 4 pieces |
| `--moss_ft_upload_ring_mib` | 512 | H2D pinned ring; multiple of the piece size; ≥ 2 pieces |
| `--moss_ft_max_skipped_steps` | 8 | stop (RuntimeError, after logging) after this many consecutive steps that skipped a parameter for non-finite grads; 0 = never stop |
| `--moss_ft_state_dir` | none (= `output_dir`) | directory for training states |
| `--moss_ft_no_keep_phase_states` | off | by default the epoch state at the end of every LR phase but the last is never deleted |
| `--moss_ft_allow_bf16_stream` | off | allow training without `--moss_stream_fp32` (e.g. a heal-type model); a warning quotes 1.5 |
| `--moss_ft_skip_resource_check` | off | skip the start-up host-RAM and disk checks (still logs the estimates) |
| `--moss_ft_log_every` | 10 | optimizer steps between pipeline timing / memory lines |

#### `optimizer_type` → mode (no new flag)

| optimizer_type | Mode |
|---|---|
| `""` or `AdamW` (any case) | `adamw_host` |
| `Adafactor` with `--fused_backward_pass` | `adafactor_fused` |
| `LoraEasyCustomOptimizer.came.CAME` or `CAME` | `came_host` |
| anything else (8-bit, Prodigy, schedule-free, ...) | refused with this table |

`optimizer_args` per mode (unknown keys are refused):
- **`adamw_host`:** `betas` (default `(0.9, 0.999)`; the tech report pretrained with `(0.9, 0.95)`), `eps` (`1e-8`), `weight_decay` (`0.0`, 1.5).
- **`came_host`:** `betas` (default `(0.9, 0.999, 0.99)`), `bias_correction` (False), `weight_decay`, `weight_decouple`, `fixed_decay`, `cautious_weight_decay`, `eps1`, `eps2`, `clip_threshold`, `update_strategy` (unmodified / cautious), `non_factored_confidence`.
  - GPU-only tuning keys are ignored with a warning (3.3).
- **`adafactor_fused`:** as 3.2.

#### Existing arguments with fixed handling

| Argument | Handling |
|---|---|
| `mixed_precision` | `bf16` (default). `fp16` with `--moss_stream_fp32` and a host optimizer mode (dynamic loss scale, plan part B). `no` = fp32 compute (reference / CPU tests; VRAM warning) |
| `full_bf16`, `full_fp16`, `fp8_base`, `deepspeed`, `blocks_to_swap > 0`, `use_ramtorch`, `cpu_offload_checkpointing`, `moss_block_compile`, `torch_compile`, `fused_optimizer_groups`, `blockwise_fused_optimizers`, `validation_split > 0` / validation datasets, `num_processes > 1` | refused, each with its reason (the host master replaces full_*; block swap and RamTorch move weights between devices; block compile is not validated with trainable base weights) |
| `moss_ckpt_nograd_merged`, `moss_lora_exclude_blocks`, `moss_stock_lycoris`, `moss_no_lokr_compile`, `moss_no_sample_cache` | LoRA-only: must stay at their defaults |
| `gradient_checkpointing` | required when the device is CUDA |
| `unsloth_offload_checkpointing` | allowed (hooks verified), costs time and pinned host RAM |
| `moss_ckpt_reuse_attn` (+ budget) | allowed (reuse-only), needs `attn_mode = "flash"`; off by default |
| `moss_stream_fp32` | required unless `--moss_ft_allow_bf16_stream` |
| `cache_latents`, `cache_text_encoder_outputs` | required; the `*_to_disk` flags are required above 2,000 images |
| `gradient_accumulation_steps` | ≥ 1; must be 1 with `adafactor_fused` |
| `max_grad_norm` | kohya default 1.0, applied as 3.1 (stale by default) |
| `lr_scheduler` / `lr_warmup_steps` | any `get_scheduler_fix` scheduler when `--moss_ft_lr_phases` is not given; with phases see 4.6 |
| `save_precision` | `bf16` default; `fp16`, `float` allowed |
| `save_state` | recommended `true`. `save_last_n_epochs_state` / `save_last_n_steps_state` unset with `save_state` → 1 (logged) |
| `resume` | only this script's state format (4.7); an accelerate state directory (has `optimizer.bin`, no `COMPLETE.json`) is refused |
| `sample_*`, `seed`, `output_dir`, `output_name`, `logging_dir`, `log_with` | as usual |

#### Start-up checks (`ft_budget`)

- Log one line per item of 2.1-2.3 for the largest bucket and the batch size (`dataset_group_limits`).
- **Refuse:**
  - when host `available` < host total + 8 GiB;
  - when free space at the state directory < 2 x state size + one DiT save.
- **Warn:**
  - when the GPU estimate with 2.3 GiB outside exceeds 30.5 GiB;
  - when the TE cache estimate exceeds the free space of the dataset drive.

### 4.6 Two-phase LR: `--moss_ft_lr_phases "2:5e-6,6:1e-6"`

- `steps_per_epoch = ceil(len(train_dataloader) / GA)`. This is kohya's formula; the last window of an epoch may be short.
- Boundaries: `b_i = round(cumsum(E)_i x steps_per_epoch)`.
- With W = `lr_warmup_steps` (int, or a float fraction of the total steps as kohya does):

```python
def lr_lambda(s):                      # s = 0-based optimizer step
    phase = next((i for i, b in enumerate(bounds) if s < b), len(bounds) - 1)
    warm = min(1.0, (s + 1) / W) if W > 0 else 1.0
    return warm * lrs[phase] / lrs[0]
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)    # optimizer.param_groups lr = lrs[0]
```

- Step 0 uses lr0/W, not 0. This is deliberate; the test pins it.
- W must end inside phase 1.
- **Example:** 16k images at batch 4 x GA 2:
  - 4,000 batches per epoch, so 2,000 steps per epoch;
  - 5e-6 for steps [0, 4,000), 1e-6 for [4,000, 16,000).
- **Recommended warmup:** 100-200 steps. Bias-corrected AdamW still makes x3.4 larger steps at t = 1 (3.3); 24+ steps of warmup absorb it.
- **Resume:**
  - `scheduler.state_dict()` restores `last_epoch`; the lambda is rebuilt from the config.
  - The state stores `steps_per_epoch` and the boundaries. A changed dataset or GA logs both versions as a warning.
  - The end-of-phase-1 state is pinned (4.7), so phase 2 can be re-run from it with other settings. This is the WSD lesson in the user's notes.

### 4.7 Checkpoints, states, resume

#### Trained DiT: `ft_model.save_moss_dit(path, model, pool, dtype, metadata)`

- **Keys:** exactly the Base set, sorted, bare names.
  - Trainable parameters come from `master` cast to `dtype`.
  - Frozen ones come from the GPU copy cast to `dtype` (exact for Base values).
  - `rope.bands` = `MossRoPE._default_bands` in F32 [16].
- **Streaming writer:** header first from shapes and dtypes, then one converted tensor at a time, same 256-byte header alignment as `mem_eff_save_file`. Peak +1 tensor (≤ 226 MB).
- **Metadata:**
  - `{"format": "pt"}` (as Base);
  - `modelspec.architecture = "moss-image-base"`;
  - `modelspec.implementation = "https://github.com/OpenMOSS/MOSS-Image"`;
  - `modelspec.title = output_name`;
  - `moss_train_utils.moss_metadata(args)` minus the LoRA key;
  - `ss_moss_ft_*`: mode, betas, lr phases, global step, epoch, base file name, train images.
- **Names:**
  - `get_epoch_ckpt_name` / `get_step_ckpt_name` / `<output_name>.safetensors`;
  - removal by kohya's `save_last_n_epochs` / `save_last_n_steps`.

#### Training state directory

The name is `EPOCH_STATE_NAME` / `STEP_STATE_NAME` / `LAST_STATE_NAME` under `--moss_ft_state_dir`.

```
<name>/
  train_state.json        global_step, completed epochs, epoch_step (batches done in the current epoch), steps_per_epoch,
                          GA, lr phase boundaries, optimizer mode + hyperparameters, slot list hash (name, shape, offset),
                          last window grad norm, skip streak, torch version, args digest
  master.safetensors      fp32, every Base key incl. frozen parameters and rope.bands -> itself a loadable fp32 DiT (20.3 GB)
  exp_avg.safetensors     fp32, trainable parameter names            (20.3 GB)
  exp_avg_sq.safetensors  fp32 (adamw_host)                          (20.3 GB) | came_factored.safetensors (came_host)
  optimizer_steps.safetensors   int64 per slot
  scheduler.pt            lr_scheduler.state_dict()
  rng.pt                  torch CPU + CUDA, python random, numpy states
  COMPLETE.json           written last: format "moss-ft-state/1", every file's byte size
```

**Writing:**
- When: only right after a window's drain (worker idle), so the host pools are stable.
- How: `mem_eff_save_file` over the pool views (zero-copy) into `<name>.tmp/`; `os.fsync` every file; `COMPLETE.json` + fsync; `os.replace(<name>.tmp, <name>)`.
- An existing `<name>` is moved to `<name>.old` first and deleted after the replace.
- Then the keep policy deletes older states:
  - the newest `save_last_n_epochs_state` epoch states;
  - the newest `save_last_n_steps_state` step states;
  - the pinned phase-boundary states;
  - the train-end state.
- **Expected time** (estimate): 61 GB at about 2-3 GB/s, so about 20-30 s per state.
- **Cadence:** epochs are long (16k images: 5.4 h, 56k: 19 h). Use `save_every_n_steps` of about 1,000 (2.7 h) with `save_last_n_steps_state = 1`.
  - The PC has had a BSOD and a GPU drop-out during long runs (notes).

**Resume:**
1. Require `COMPLETE.json` and the recorded sizes. A missing marker or a size mismatch is refused, naming the newest complete state. This covers the truncated-state lesson.
2. Check the slot list hash and the optimizer mode. Hyperparameter changes are warnings.
3. Load `master` per tensor into the pool; upload the compute copies (cast) into the GPU parameters.
4. Load `exp_avg`, `exp_avg_sq` / CAME states and the steps in place. Peak is +1 tensor; nothing is double-buffered.
5. Build the scheduler from the config and `load_state_dict`. Load the RNG states.
6. Continue at `global_step`, epoch `completed + 1`, and skip `epoch_step` batches if the state was taken mid-epoch.
   - Mid-epoch resumes see a new shuffle of the epoch, as kohya does. Epoch-end states resume exactly at the boundary.

### 4.8 Sample images

- `moss_train_utils.sample_images(accelerator, args, epoch, global_step, dit, vae, sample_prompts_te_outputs)` as is:
  - official scheduler grid, shift 5, CFG with block 22 skipped in the negative branch;
  - RNG saved and restored;
  - fp32 VAE decode.
- Called only after a drain, so the GPU weights are the stepped ones.
- `accelerator.unwrap_model` of an unprepared module is the module itself.
- With no LoKr modules, `cached_lokr_weights` holds nothing.
- VRAM during sampling: weights + one no-grad forward + the VAE decode, below the training peak (estimate).
- Prompts need cached embeddings (`<sample_prompts>_moss_te/`). Missing ones are skipped by the existing code.

### 4.9 Logging (TensorBoard via `accelerator.log`, every step unless noted)

**Every step:**
- `loss` (window mean), `lr/group0`, `grad_norm`, `clip_coef`, `skipped_params`.
- `time/step`, `time/tail` (CPU wait at the drain), `time/hook_wait` (autograd thread blocked on the ring), `time/worker_busy`, `time/data`.
- `throughput/tok_per_s`, `throughput/img_per_s`.

**Every `--moss_ft_log_every` steps:**
- `vram/peak_allocated`, `vram/peak_reserved`, `vram/device_peak` (with `outside_bytes`).
- `host/rss_gib`, `host/available_gib`, `host/pinned_gib`.
- `TierTimer` lines per tier.

**At start:** the 4.5 budget lines.

### 4.10 Changes to existing files (behavior-preserving)

- **`moss_train_network.py`:**
  - `cache_text_encoder_outputs_if_needed` and `_cache_sample_prompts` become thin calls into `library/moss/te_cache.py`. The existing MOSS LoRA tests must stay green.
  - `assert_extra_args` refuses `network_module` LyCORIS with `algo=full` ("MOSS full fine-tuning is moss_train.py; LyCORIS full needs ~71 GiB of VRAM").
- **Nothing else** in shared modules changes. `validate_moss_args` is satisfied by setting the two `network_*` attributes (4.2).

## 5. CPU test plan

**Run from the worktree root:**

```
CUDA_VISIBLE_DEVICES=-1 <LoRA_Easy_Training_Scripts>\backend\sd_scripts\venv_t214\Scripts\python.exe -B -m pytest tests/test_moss_ft_*.py -q -p no:cacheprovider
```

**Common setup:**
- An empty `CUDA_VISIBLE_DEVICES` does not hide the GPU on this machine: use -1.
- Tiny config, as `tests/test_moss_models.py`: `TINY = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=48, freq_dim=16, latent_channels=128)`.
- The pipeline runs with `device="cpu"`: no streams, events are no-ops, rings are not pinned, copies are synchronous.
- Both `inline=True` (the hook calls the worker function) and the threaded worker are tested.
- Pieces and chunks are made tiny (e.g. 256 / 64 elements) so that every parameter spans several pieces.

| File | Test | Assertion |
|---|---|---|
| `test_moss_ft_host_adamw.py` | `test_adamw_host_equals_torch_adamw` (GA 1 and 3, two groups with different wd, a frozen regex, inline and threaded) | fp32 compute: master, m, v after 5 steps **bit-identical** to `torch.optim.AdamW(fused=True)` on an fp32 copy with `loss / n_micro`; compute copies == master |
| | `test_exact_clip_equals_clip_grad_norm` | exact mode, `max_grad_norm` small enough to clip: bit-identical to `clip_grad_norm_` + AdamW |
| | `test_stale_clip_uses_previous_norm` | window 2's applied coefficient == min(1, max/G1); window 1 ran exact |
| | `test_in_backward_step_equals_after_backward_step` | overlap on vs exact mode with clipping off: bit-identical |
| | `test_partial_last_window` | `len(dataloader) % GA != 0`: last window divides by its own n_micro; equals reference |
| | `test_nonfinite_grad_skips_only_that_param` | NaN injected into one parameter's grad in micro-step 1: its master / m / v / step unchanged, others stepped, `skipped_params == 1`; `--moss_ft_max_skipped_steps 2` raises on the 2nd consecutive |
| | `test_param_without_final_grad_is_flushed` | a parameter with a grad only in micro-step 0 is still stepped at the window end |
| | `test_worker_error_surfaces` | an exception in the worker raises in `backward()` / drain with the worker traceback |
| `test_moss_ft_hooks.py` | `test_hooks_fire_once_per_param` for none / plain / unsloth / no-grad node / reuse-only | count == 1 per parameter per backward; pipeline grads == reference (the check of 1.4 kept as a test) |
| `test_moss_ft_precision.py` | `test_fp32_master_keeps_1e-6_updates` | bf16 compute + fp32 master, lr 1e-6, 100 steps of fixed gradients: master movement within 1e-3 relative of an fp32 AdamW reference |
| | `test_pure_bf16_loses_them` | torch AdamW on bf16 parameters, same setup: ≥ 90 % of the movement lost (documents why the master exists) |
| | `test_compute_copy_is_rtn_of_master` | after every step: GPU-copy == `master.to(bf16)` |
| `test_moss_ft_came_bc.py` | `test_core_matches_came_py_without_correction` | skipped if `LoraEasyCustomOptimizer` is not importable; same fp32 alphas: bit-identical (≤ 1 ulp) to `CAME._core_factored_full_fp32` and `_core_unfactored_full_fp32` for 5 steps, unmodified and cautious |
| | `test_bias_correction_removes_amplification` | 64x64, β3 0.999, T 3,000: corrected ratio at t1 ≤ 4.0 and t24 ≤ 1.2; uncorrected t1 ≥ 8 (same synthetic gradients as 3.3) |
| | `test_correction_is_state_compatible` | EMA buffers identical with / without the correction after N steps |
| | `test_refuses_high_beta3_without_correction` | `came_host` with β3 0.9999 and no `bias_correction`: ValueError |
| `test_moss_ft_lr_phases.py` | `test_values` | S = 100, `"2:5e-6,6:1e-6"`, W = 10: lr(0) = 5e-7, lr(9) = 5e-6, lr(199) = 5e-6, lr(200) = 1e-6, lr(799) = 1e-6, total 800 |
| | `test_fractional_epochs`, `test_parse_errors`, `test_conflicts` | `"0.5:1e-5,1:1e-6"` boundary 50; bad strings, non-positive values, sum ≠ `max_train_epochs`, other `lr_scheduler`, `max_train_steps`, warmup past phase 1: ValueError |
| | `test_resume_keeps_schedule` | scheduler state_dict round trip continues the same values |
| `test_moss_ft_state.py` | `test_save_resume_equivalence` | run 4 windows straight vs 2 windows + save + new process objects + load + 2 windows (fixed batches, RNG restored): master, m, v, steps, GPU copies bit-identical; fp32 and bf16 compute |
| | `test_incomplete_state_refused` | missing `COMPLETE.json`, a truncated file, a changed slot list: refused with the reason |
| | `test_keep_policy` | keep-last 1 + pinned phase state + atomic replace of an existing name |
| | `test_master_file_is_a_loadable_dit` | `master.safetensors` loads with `load_moss_dit` (tiny) |
| `test_moss_ft_save_dit.py` | `test_base_layout` | tiny model: keys == Base-layout key set (incl. `rope.bands` F32 [16]), bf16 elsewhere, `{"format": "pt"}` present, `shared_block_modulation.weight` + `layers.0.proj.ffn_up_gate.weight` present, no `qkv` key |
| | `test_round_trip` | `save_moss_dit` → `load_moss_dit` → tensors == `master.to(bf16)`; frozen parameters keep the base values |
| | `test_heal_layout_input` | loading a fused-qkv file for full FT splits it and saves the split layout |
| `test_moss_ft_args.py` | `test_parser_builds` | the 4.2 composition + `add_moss_ft_arguments` builds; an example TOML (8.1) reads |
| | `test_refusals` | every refused row of 4.5, one parametrized case each |
| | `test_optimizer_mode_mapping`, `test_optimizer_args_per_mode` | the 4.5 tables |
| | `test_resource_checks` | mocked `psutil.virtual_memory` / `shutil.disk_usage`: refuse / pass at the boundaries |
| `test_moss_ft_budget.py` | `test_numbers_match_design_tables` | `ft_budget` reproduces the tables of 2.1-2.3 and 3.5 to ±0.05 GiB / ±0.1 s; a change to the estimator then also changes this document |
| `test_moss_ft_loss.py` | `test_loss_matches_lora_trainer_math` | same seed and batch: `moss_ft_noise_and_loss` == the target / weighting / loss of `MossNetworkTrainer.get_noise_pred_and_target` + `train_network` loss path (LoRA disabled) |
| existing | `tests/test_moss_*.py` | still pass after the `te_cache` move |

## 6. Risks

1. **Host memory bandwidth decides speed at small batches.**
   - Measured: AdamW 2.8 s, accumulate 1.0-1.15 s, final step 3.7-4.4 s; it saturates at about 16 threads.
   - DMA traffic (about 10 GiB D2H per micro-step and 10 GiB H2D per step) competes for the same DRAM.
   - Mitigation: batch ≥ 2; measure `time/hook_wait` and `time/tail`.
2. **Windows WDDM transfers.**
   - The D2H of grads and the H2D of weights overlap in the final backward.
   - KMT measured no bidirectional gain on a 4090 under WDDM (25.7 GB/s total). The 5090's link speed here is unknown.
   - At about 10 GiB each per step, even 20 GB/s is under 1 s, but measure it (7.1).
3. **bf16 compute precision with Base's massive activations.**
   - The fp32 stream is required.
   - bf16 + fp32 stream gradient error is unmeasured (7.2). The fallback plan is fp16 compute (phase 1b).
4. **Host RAM 85 GiB of 119 GiB.**
   - Other programs can push Windows into paging, which makes the CPU step crawl.
   - Mitigation: the start-up check; log `host/available_gib`; GA = 1 needs 66 GiB.
5. **Stale clipping is not torch's clipping.**
   - Documented; exact mode exists; `max_grad_norm = 0` has no difference.
6. **Per-parameter non-finite skip differs from GradScaler's whole-step skip.**
   - With bf16 a non-finite grad means divergence.
   - Mitigation: the skip streak stops the run after 8 steps.
7. **Hook backpressure.**
   - A slow worker blocks the autograd thread. That is intended: it is not a deadlock, because the worker never waits on autograd and H2D events wait only on already-recorded compute-stream events.
   - A worker crash would block forever without the 0.5 s error poll in `acquire()`. A test covers it.
8. **Elementwise compile with trainable norm weights.**
   - The compiled chains now produce gradients for `norm1/norm2/q_norm/k_norm` weights. AccumulateGrad stays outside the graph, so the hooks still fire.
   - Compare one micro-step's grads compiled vs `--moss_no_compile` on the GPU (7.3). It is worth keeping: about 2 s of a 9 s step at 1536 per the code comment.
9. **Private torch API.** `torch._fused_adamw_` (pinned torch 2.14); the equivalence test catches a change.
10. **Long runs (43-150 h).** Power loss, BSOD and GPU drop-out (the notes record one). Use step-wise states every ~1,000 steps with atomic writes, and resume tested on the GPU (7.4).
11. **Disk.** 61 GB states, 10 GB DiT saves, TE caches up to TB scale with many caption variants (2.3). Mitigations: D:, the start-up check, 1-3 variants.
12. **Data order on mid-epoch resume** is not reproduced (kohya behavior). Epoch-end states are exact.
13. **The training recipe itself.**
    - Catastrophic forgetting or style drift is not a code risk but the main outcome risk.
    - KMT saw 1.7 % relative weight change after 10k steps at 2e-5.
    - Watch samples every epoch. Phase-boundary states allow re-running phase 2.

## 7. GPU smoke plan (not run; needs the user's go-ahead)

**Preconditions:**
- The user says the GPU is free.
- ComfyUI 8188 is closed: it held up to 27 GB of VRAM before and holds RAM.
- No other training is running.
- **Expected peak:** ~30 GiB VRAM device-wide at batch 4, ~85 GiB host RAM.
- Launched the user's way (new PowerShell window via WMI). The log is read from files.
- No VRAM cap and no `empty_cache()` per step (user rule).

**Total:** about 60-70 min.

| # | What | How | Pass / decision |
|---|---|---|---|
| 7.1 | PCIe probe (2 min) | pinned 1 GiB H2D, D2H, both at once on two streams, 10 repeats each | record GB/s; if the bidirectional sum < 15 GB/s, the final-step overlap is copy-bound: prefer batch 4 x GA 2 |
| 7.2 | Precision probe (10 min) | 2 images at 1024, σ 0.1 / 0.5 / 0.9; dW of all joint Linears vs an fp32 reference (fp32 weights, no autocast, SDPA fp32) for (i) bf16 + fp32 stream, (ii) fp16 + fp32 stream; per-block relative error and cos (the method of `UR\ANIMA\_chains\20261115_moss_fp16_fold\measure_error.py`, arms "a" style, weight grads instead of LoKr grads) | if (i) has cos < 0.98 in any block range where (ii) has cos ≥ 0.99, build phase 1b (fp16) before the long run |
| 7.3 | Speed / memory (35 min) | 256 images at 1024; batch 1 x GA 8, batch 2 x GA 4, batch 4 x GA 2; 14 optimizer steps each (first 2 excluded); also one micro-step's grads with and without `--moss_no_compile` | per config: s/step (median), tok/s, `time/tail`, `time/hook_wait`, `time/worker_busy`, D2H / H2D GB/s, per-micro-step fwd + bwd (TierTimer), `vram/device_peak`, host RSS / available, loss finite (~0.4), `grad_norm`. Pick the largest batch with device peak ≤ 30.5 GiB. Compiled vs eager grads: max relative difference ≤ bf16 level (1e-2) |
| 7.4 | State and resume (10 min) | best config, `save_every_n_steps = 5`; stop after step 7; resume from step 5; save again right after loading; run 2 steps | the re-saved state equals the step-5 files bit for bit; GPU copies == master cast; global_step, LR and phase boundaries continue; loss finite. Trained weights after a resume are not compared bitwise on the GPU: flash-attn's backward is nondeterministic, and the user's ABAB runs diverged that way in 65 steps. State size 61.0 GB; write and load times recorded |
| 7.5 | DiT save sanity (5 min) | save at step 7, load with `load_moss_dit`, generate one 1024² sample with the trainer's sampler; the user can load the file in ComfyUI later | keys / dtypes as 4.7; image not broken |

Report it as one table: config x (s/step, tok/s, tail, hook wait, peak VRAM, host RAM, transfer GB/s), with the decisions above.

## 8. Phase 2 (not in the first implementation)

1. **Activation offload** (KMT idea 7, own implementation): checkpoint inputs to pinned host slabs on a side stream with one-block prefetch.
   - Enables batch 8 x GA 1 at 1024: est. −8 % s/step, +17 GiB pinned host.
2. **GPU-assisted step**: per parameter, upload master / exp_avg (/ exp_avg_sq) and the accumulated grad, step on the GPU, download.
   - Needs the big host buffers page-locked (`cudaHostRegister` on the pool allocations; untested on Windows at ~60 GiB) or a CPU staging copy.
   - Worth it for `came_host` (20.5 s → est. 4-5 s per step). For AdamW only if 7.1 shows ≥ 40 GB/s each way.
3. **fp16 compute (phase 1b; built 2026-11-17 as `moss_ft_te_live_fp16_plan.md` part B, `library/moss/ft_scaler.py`):**
   - dynamic loss scale S (2^15 start, halve on overflow, double after 2,000 clean steps);
   - unscale fused into the host convert (`t.mul_(1/S)`);
   - overflow found by the per-parameter GPU norms;
   - per-parameter skip as in 3.1 and S halved.
4. **Block compile** (`--moss_block_compile`) with trainable base weights: test the hook path and the gradients, then allow it.
5. **bf16 optimizer states / √v storage** (AdamW16-like, own code) to cut host RAM by about 19 GiB.
6. **EMA of weights on the host:** about 61 GB of memory traffic per step, about 1.2 s. As an option.
7. **Validation loss** (built 2026-11-18: `--moss_ft_val_dataset_config`, `library/moss/ft_val.py`; deterministic held-out
   loss at fixed sigmas and noise, user guide `moss_train.md` section 10).
8. **Port the CAME read-out correction into `came.py`** behind a default-off flag. Parent repository; needs the user's OK.

### 8.1 Example config for the first real run (illustrative; values to confirm after 7)

```toml
pretrained_model_name_or_path = "<forge>/models/Stable-diffusion/moss_image_base_bf16.safetensors"
vae = "<forge>/models/VAE/moss_image_vae_fp32.safetensors"
moss_text_encoder = "<forge>/models/text_encoder/qwen3.5_9b_moss_image_sft_bf16.safetensors"
moss_tokenizer_dir = "<forge>/extensions/forge_sam3_extension/assets/qwen35_tokenizer"
dataset_config = "D:/<dataset>/moss_fft_1024.toml"   # resolution 1024, batch_size 4, bucket_reso_steps 64, cache next to images on D:
mixed_precision = "bf16"
moss_stream_fp32 = true
attn_mode = "flash"
gradient_checkpointing = true
gradient_accumulation_steps = 2
optimizer_type = "AdamW"                               # -> adamw_host
optimizer_args = ["betas=(0.9, 0.999)", "eps=1e-8", "weight_decay=0.0"]
moss_ft_lr_phases = "2:5e-6,6:1e-6"
lr_warmup_steps = 100
max_grad_norm = 0.0                                    # first run: read grad_norm, then set ~2-3x its median
cache_latents = true
cache_latents_to_disk = true
cache_text_encoder_outputs = true
cache_text_encoder_outputs_to_disk = true
cache_caption_variants = 3
save_precision = "bf16"
save_every_n_epochs = 1
save_state = true
save_last_n_epochs_state = 1
save_every_n_steps = 1000
save_last_n_steps_state = 1
output_dir = "<output_root>/<run>"
sample_prompts = "path/to/sample.txt"
sample_at_first = true
sample_every_n_epochs = 1
max_data_loader_n_workers = 3
persistent_data_loader_workers = true
seed = 1234
```

The timestep distribution (`timestep_sampling = "moss"`, the Base's pretraining distribution, or the user's LoRA choice `sigmoid` 1.0) is a recipe decision left to the user.

## Appendix A. Measurement snippets (CPU, `CUDA_VISIBLE_DEVICES=-1`, run through stdin, not saved)

**Host step components** (1.4):

```python
N = 113_246_208; master = torch.randn(N) * 0.02; m = torch.zeros(N); v = torch.zeros(N); acc = torch.randn(N) * 1e-3
g16 = (torch.randn(N) * 1e-3).to(torch.bfloat16); out16 = torch.empty(N, dtype=torch.bfloat16); step = torch.tensor(1.0)
def final_step(C=1 << 20, tmp=torch.empty(1 << 20)):
    for s in range(0, N, C):
        e = min(N, s + C); t = tmp[:e - s]; t.copy_(g16[s:e]); t.add_(acc[s:e])
        torch._fused_adamw_([master[s:e]], [t], [m[s:e]], [v[s:e]], [], [step], lr=5e-6, beta1=0.9, beta2=0.999,
                            weight_decay=0.01, eps=1e-8, amsgrad=False, maximize=False)
        out16[s:e].copy_(master[s:e])
# timed with time.perf_counter, best of 3, scaled x 5_082_308_608 / N; torch.set_num_threads(4..24)
```

**Hook check** (1.4):
- Tiny `MossTransformer(dim=64, ffn_dim=128, num_heads=2, num_layers=3, text_dim=32, freq_dim=16, latent_channels=8)`.
- `p.register_post_accumulate_grad_hook(hook)` on every parameter, where the hook records the call and clones `p.grad`, then sets `p.grad = None`.
- One backward for each of: no checkpoint, `enable_gradient_checkpointing()`, `enable_gradient_checkpointing(unsloth_offload=True)`, `enable_gradient_checkpointing()` + `set_ckpt_reuse(True, False)`.
- Results: 72 / 72 parameters, every call count 1, max |grad difference| 0.

**Saved activations** (1.4):
- `MossBlock(3072, 8192, 24).to(bf16)`, input `[1, L, 3072]` fp32, `torch.autocast("cpu", bfloat16)`.
- `torch.autograd.graph.saved_tensors_hooks` sums the storage bytes of saved tensors that are not parameters.
- L = 512: 129.2 MiB; L = 1024: 258.4 MiB. That is 264.6 KB per token.

**CAME** (3.3):
- The reader's `came_amp_check.py` setup (`mu = 0.2·randn`, `grad = mu + randn`).
- `came_factored_step` of 3.3 with `bias_correction` False / True, β3 0.9999 / 0.999, N = 128, T = 30,000.
- The ratio is to the mean RMS(Δp) of the last 2,000 steps.

**SR / RTN** (1.5):
- `w0 = (0.02·randn(2^20)).bf16`.
- 2,000 steps of ±lr with P(+) = 0.5 or 0.55.
- Reference in fp64; RTN = `(w.float() + s).bf16()`; SR = `copy_stochastic_` of `library/adafactor_fused.py`.
- Error metric: ‖w − ref‖ / ‖ref − w0‖.
