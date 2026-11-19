# MOSS-Image full fine-tune: "Carousel" (rotating active blocks) - implementation plan

Status: implemented 2026-11-17 (CPU-tested; GPU smoke C1-C4 pending), see docs/moss_ft_impl_notes.md T8 and docs/moss_train.md section 9. Worktree `backend/sd_scripts_moss_full`, branch `feat/moss-full-ft`.
Working name "Carousel": the user may rename it. Every user-visible name is derived from ONE constant (section 4.1).
Read with: `docs/moss_full_finetune_design.md` (design of the existing modes), `docs/moss_ft_impl_notes.md`,
`docs/moss_ft_te_live_fp16_plan.md` (fp16 + loss scale), `docs/moss_train.md` (user guide). GPU numbers marked
"measured" come from the 11-17 smoke run (`UR\ANIMA\_chains\20261117_moss_ft_smoke\results.md`); everything else is an
estimate that the GPU smoke items of section 8 check.

---

## 한국어 요약

- **무엇:** LISA / BAdam 처럼 학습 단위(unit) 44개 중 K개만 한 번에 학습합니다. P 옵티마이저 스텝마다 K개를 새로 고릅니다.
  - unit = joint 블록 42개 + 텍스트 refiner 블록 2개(크기가 joint 블록과 거의 같음, 113.28M).
  - 고르는 방법: `random`(기본, 한 바퀴 안에서 중복 없이) / `round_robin`(깊이 전체에 흩어진 고정 순서) / `window`(연속 K개).
  - 런 전체로 보면 모든 블록이 진짜 전체 가중치 업데이트를 받습니다.
- **GPU 에 두는 것:**
  - 모든 블록의 fp16 계산 사본(지금과 같음, 10.07 GiB).
  - 활성 unit 만 fp32 master·fp32 기울기 누적·옵티마이저 상태를 GPU 에 둡니다. 스텝은 GPU 에서 돕니다.
  - 기존 adamw_host 의 병목(호스트 AdamW 4.7 s/스텝)이 사라집니다.
- **호스트에 두는 것:**
  - 모든 블록의 fp32 master(18.93 GiB).
  - `keep` 정책일 때 비활성 unit 의 옵티마이저 상태. AdamW 37.1 / AdamW8bit 9.4 / CAME 18.6 GiB(bf16 모멘텀이면 9.3).
- **unit 당 GPU 추가 바이트(정확한 값):**
  - adamw 1.688 GiB, adamw8bit 1.058 GiB, came 1.266 GiB.
- **K 예상치:** 측정한 활성화 메모리 기준, 1024², 31.84 GiB 카드.

  | | adamw | adamw8bit | came |
  |---|---|---|---|
  | b1 | 7 | 12 | 9 |
  | b4 | 2 | 5 | 3 |

- **속도(추정):** 1024 b1 기준 약 0.7 s/스텝입니다. 지금 adamw_host 는 4.84 s 라서 약 7배 빠릅니다.
  - 대신 각 unit 은 스텝의 K/44 만큼만 학습됩니다.
  - 전환 1회는 adamw keep, K=7 기준 약 2.2 s 입니다. P=50 이면 약 6 % 손해입니다.
- **CAME 질문:** GPU 에서는 CAME 이 충분히 씁니다.
  - 상태가 4 B/param 이라 AdamW 8 B/param 보다 작습니다. K 가 7에서 9로 늘어납니다.
  - 느렸던 건 CPU 에서 돌린 탓(20.5 s)이었습니다. GPU 에서는 스텝당 수십 ms 로 추정합니다.
  - 단, came.py 를 그대로 쓰면 안 됩니다. bias_correction 인자가 없어서 그냥 무시되고, step 이 그룹 단위라서 `keep` 과 맞지 않습니다.
  - 그래서 ft_optim 의 CAME core(came.py 와 비트 동일, 보정 포함)를 GPU 에서 돌리고, step 은 파라미터별로 둡니다.
- **OCGOptV2:** 상태가 value/centralized momentum 2개라 8 B/param 입니다. AdamW 와 같아서 메모리 이득이 없습니다.
  - 2-D 텐서마다 Newton-Schulz 행렬곱이 붙고, step 도 그룹 단위입니다. 1차 구현에서는 제외합니다.
- **결정이 필요한 충돌 1개:** "작은 그룹 항상 학습(기본 켬)"과 "가장 낮은 활성 블록 아래로 역전파 안 함"은 같이 될 수 없습니다.
  - 임베더·time·shared modulation 이 학습되면 42블록 전부를 통과하는 역전파가 필요합니다. 가중치 기울기는 여전히 활성 unit 만 계산합니다.
  - 비용은 random 선택이면 약 +6 %(어차피 가장 낮은 활성 블록이 평균 4번 근처), window 선택이면 약 +34 % 입니다.
  - 기본값은 요청대로 `small`(항상 학습)로 두고, `head` 를 고르면 잘라내기가 실제로 작동합니다.
- **정확성:**
  - 전환은 fp32 그대로 왕복해서 비트 단위로 같습니다. fp16 계산 사본은 늘 "master 를 반올림한 값"이라 다시 만들 필요가 없습니다.
  - K=전체이고 전환이 없으면 평범한 GPU AdamW 전체 FT 와 스텝마다 비트 단위로 같아야 합니다. 이걸 테스트로 지킵니다.
  - K<전체일 때는 "requires_grad 를 같은 순서로 바꾸는 평범한 FT + torch AdamW" 와 비트 단위로 같아야 합니다. 이것도 테스트로 지킵니다.
- **기존 adamw_host / came_host / adafactor_fused 는 그대로 둡니다.** models.py 도 고치지 않습니다(인스턴스 forward 래퍼로 해결).

---

## 1. Decisions at a glance

| Topic | Decision |
|---|---|
| Name | One constant `ft_budget.CAROUSEL = "carousel"`: mode string, flag prefix `--moss_carousel_*`, log prefix. State files use a stable internal id (`ROTATION_FORMAT = "moss-ft-rotation/1"`), so a rename does not orphan saved states |
| Units | 44: `text_embedder.blocks.0/1` (113,284,352 params each) and `layers.0..41` (113,252,608 each). A unit holds every trainable tensor of its block, its 4 norm vectors included |
| Always-trainable | `--moss_carousel_always small` (default, as requested): head + small stem, 99,130,368 params. `head`: 19,276,800 params (truncation works). `none` |
| Device layout | Active unit: fp16 (bf16 / fp32) compute copy = the module parameter, requires_grad True; fp32 master, fp32 grad accumulator and optimizer state in flat device buffers. Inactive: compute copy only, requires_grad False |
| Forward weights | The existing compute copies (not autocast from the fp32 master): bit-identical forward to the frozen blocks and to the existing modes, no dtype switch of module parameters (section 3.2) |
| Host | `ft_host.HostPool(slots, roles=("master",))`: one contiguous pageable fp32 master of every trainable tensor (reuses the loader, `dit_entries`, `_load_master`, `check_compute_copies`). Kept states: per-unit flat host buffers |
| Step | After the window's last backward (not in-backward): unscale + non-finite check, exact global clip over the trainable set, one optimizer step on the GPU, cast master -> compute copy (fp16 range check) |
| Overflow | Whole window skipped (GradScaler semantics); `ft_scaler.DynamicLossScaler` reused unchanged |
| Optimizers | `adamw` (torch AdamW fused=True), `adamw8bit` (bitsandbytes AdamW8bit, not paged), `came` (new GPU `CarouselCAME` on `ft_optim` cores, read-out bias correction on) |
| State policy | `keep` (default, exact fp32 / native 8-bit; `--moss_carousel_state_dtype bf16` = lossy bf16 CAME momentum) or `reset` (LISA) |
| Selection | Pure function of (seed, switch index): no RNG state, no draws from the training RNG; resumes from `global_step` alone |
| Switch | Synchronous, between windows, outgoing units first (D2H), then incoming (H2D); compute copies untouched; an equality check per incoming tensor |
| Truncation | Blocks below the lowest unit that needs gradients run under `torch.no_grad()` (instance `forward` wrapper, models.py unchanged); the floor block's input gets `requires_grad` under unsloth offload |
| Saves | DiT and training state after a flush of the active masters / states to the host; the existing atomic state directory with two carousel files added |

---

## 2. Facts from the code (read 11-17)

### 2.1 Parameters (meta model, `ft_budget.moss_param_shapes`)

| Group | Tensors | Params | Compute dtype (fp16 + `--moss_stream_fp32`) |
|---|---|---|---|
| joint block `layers.i` (x42): q, k, v, o [3072,3072], ffn_up_gate [16384,3072], ffn_down [3072,8192] | 6 | 113,246,208 | fp16 |
| ... its norms norm1, norm2 [3072], q_norm, k_norm [128] | 4 | 6,400 | fp32 |
| refiner `text_embedder.blocks.j` (x2): qkv, proj, ffn_gate, ffn_up, ffn_down (+biases), 4 norms | 14 | 113,284,352 | fp32 (not `layers.*.proj.*`) |
| `video_embedder` | 2 | 396,288 | fp32 |
| `text_embedder.proj` | 2 | 12,585,984 | fp32 |
| `time_embedding.0/.2` | 4 | 10,229,760 | fp32 |
| `shared_block_modulation` [18432,3072] + bias | 2 | 56,641,536 | fp32 |
| `final_norm_video`, `final_linear_video`, `video_head_modulation` | 4 | 19,276,800 | fp32 |
| total | 462 | 5,082,308,608 | 10,816,552,960 B = 10.07 GiB compute copies |

"small" = the last five rows = 99,130,368 params (stem part 79,853,568: video_embedder, text_embedder.proj,
time_embedding, shared_block_modulation; head part 19,276,800). Check: other params 325,967,872 - refiners 226,568,704 -
42 x 6,400 block norms = 99,130,368.

### 2.2 Forward / backward structure (`library/moss/models.py`)

- `MossTransformer.forward`: `time_conditioning` (time MLP -> `mod` [B,6,D] for every block, `head` [B,2,D]), video
  embedder, `text_embedder` (proj + 2 refiner blocks), concat -> 42 `MossBlock(hidden, rope, mod, params)` -> final norm,
  head modulation, final linear. `mod` enters every block: if `shared_block_modulation` or `time_embedding` is trained,
  its gradient needs the activation gradient at every block (full-depth backprop).
- `_CheckpointMixin._run`: plain `torch_checkpoint(use_reentrant=False)`, `unsloth_checkpoint`
  (`anima_models.UnslothOffloadedGradientCheckpointer`, a custom `autograd.Function`), or the LoRA-only no-grad node
  (refused in full FT). The unsloth Function's output requires grad only if a TENSOR INPUT requires grad: block
  parameters are not inputs, so a block whose input has no grad loses its weight gradients silently. Plain
  non-reentrant checkpoint has no such problem.
- `MossTextProjection.forward` sets `x.requires_grad_(True)` under checkpointing when the input has none (a LoRA fix for
  unsloth): in a carousel step with a frozen text path this would drag the text path and every joint block into the
  backward.
- `ft_loss.moss_ft_noise_and_loss` sets `noisy.requires_grad_(True)` under gradient checkpointing (same reason): it makes
  the whole stream require grad from the video embedder on.
- `--moss_block_compile`, `--cpu_offload_checkpointing`, block swap are refused in full FT (`ft_args._check_refused_options`);
  that stays.

### 2.3 Optimizer facts

- torch 2.14 `AdamW(fused=True)` works on CPU (checked 11-17, CPU only), so the CPU tests run the real kernel family.
  Per-parameter `state["step"]`: a skipped parameter (grad None) keeps its step - exactly the LISA "keep" semantics.
- bitsandbytes 0.49.2 `AdamW8bit`: CUDA only (CPU raises "All input tensors need to be on the same GPU", checked 11-17).
  State per tensor with numel >= 4096 (`min_8bit_size`): `state1`, `state2` uint8, `absmax1`, `absmax2` fp32 per 256-value
  block, `qmap1` / `qmap2` (shared `name2qmap` tensors, NOT per parameter), `step` (Python int, per parameter). Below
  4096: fp32 `state1` / `state2`. 2.03125 B/param for the big tensors.
- `LoraEasyCustomOptimizer/came.py` `CAME`: no `bias_correction` parameter (an unknown kwarg is logged and ignored), step
  counter per GROUP (`group['step']`), state storage defaults to CPU bf16 with staging buffers keyed by `id(p)`. All three
  conflict with Carousel (keep needs a per-parameter t; the beta3 amplification needs the correction).
  `ft_optim.came_factored_core` / `came_unfactored_core` are plain torch ops (device-agnostic), bit-identical to came.py
  without the correction (`tests/test_moss_ft_came_bc.py`), with the read-out correction (design 3.3). Temporaries: 2
  full-size fp32 tensors of the parameter (`update`, `res`).
- `ocgoptv2.py` (OCGOptV2): `value_momentum` + `centralized_momentum` (fp32, `denom` only for 0-D) = 8 B/param like AdamW,
  a Newton-Schulz spectral clip per 2-D tensor per step, step per group. No memory gain over AdamW; not in v1.

### 2.4 Measured on the 5090 (smoke 11-17)

- PCIe pinned H2D ~40 GB/s, D2H ~24 GB/s, pageable ~8.4-8.7 GB/s; copies overlap compute.
- adamw_host 1024 b1 GA1: 4.84 s/step, hook wait 4.03 s -> the GPU computes ~0.8 s of it. Allocated peak 14.1 GiB.
- b4 x GA2: 9.71 s/step, hook wait 7.09 s -> GPU ~2.6 s. Allocated peak 21.7 GiB, device 24.6 GiB, host RSS 94.5 GiB.
- Non-weight allocation (peak - 10.07 GiB compute copies - 0.5 GiB gradient ring): b1 ~3.5 GiB, b4 ~11.1 GiB (live TE
  included). The ft_budget estimate for b4 is lower (~7.3 GiB): section 3.8 uses the measured values.

---

## 3. Design

### 3.1 Units, always-trainable groups and the truncation conflict

- Units in forward order: `refiner.0`, `refiner.1`, `layers.0` ... `layers.41` (unit index 0..43). A unit = every
  trainable tensor whose name starts with the block prefix (`--moss_ft_freeze` still applies; a unit with nothing
  trainable left is dropped from the rotation).
- `--moss_carousel_always` (one of):
  - `small` (default): head + stem tiny groups (99.13M) always trainable. Every step then needs the activation gradient
    down to the input (embedders, `mod`, the text projection), so there is NO truncation: all 42 joint blocks run
    recompute + dgrad. Weight gradients are still computed only for the active units. The refiners run recompute +
    dgrad too (text tokens only).
  - `head`: final norm / final linear / head modulation (19.28M) always trainable; stem frozen. The truncation of 3.5
    applies: nothing below the lowest active unit is recomputed or backpropagated.
  - `none`: only the units train.
- Per-block norms move with their block (if they were always on, every block would need dgrad: same cost as `small`).
- Cost model per micro-step in units of one block forward F (checkpointed block: forward F, recompute F, dgrad F,
  wgrad F): below the floor F; inactive above the floor 3F; active 4F. Full FT = 168F.
  - random selection, K=8 of 42: E[lowest active] = (42-8)/(8+1) = 3.8.
    - `head`: 3.8 + 30.2 x 3 + 32 = 126F (-25 % vs full FT).
    - `small`: 34 x 3 + 32 = 134F (-20 %). So `small` costs ~6 % over `head`.
  - window selection: E[floor] ~17.
    - `head`: 17 + 17 x 3 + 32 = 100F (-40 %).
    - `small`: 134F. Here `small` costs ~34 % over `head`.
  - Recommendation for the user guide: `small` with `random`; `head` with `window` when speed matters most.

### 3.2 Per-tensor layout

For every trainable tensor (slot, `ft_host.ParamSlot`, registration order):

| | Active unit / always group | Inactive unit |
|---|---|---|
| compute copy (module parameter) | fp16 joint Linear / fp32 otherwise (unchanged dtype), `requires_grad=True`, post-accumulate-grad hook | same tensor, `requires_grad=False` |
| fp32 master | device: a view of the unit's flat `master` buffer, wrapped in a persistent leaf `handle` (`nn.Parameter`, `requires_grad=False`) that the optimizer steps | host: the `HostPool` master view (authoritative) |
| fp32 grad accumulator | device: view of the unit's flat `acc`; `handle.grad = acc view` at step time | none |
| optimizer state | device: `optimizer.state[handle]` (AdamW: views of flat `exp_avg` / `exp_avg_sq`; CAME: flat `exp_avg` + factored stats; bnb: its own tensors) | host store (keep) or nothing (reset) |

Why the compute copy and not "autocast from the fp32 master": both are exact and need about the same memory (18 B/param
for AdamW: 2 compute + 4 master + 4 acc + 8 state, vs 4 param + 4 grad + 8 state + 2 autocast cache). The compute-copy
form wins on:
- The active block's forward is the same Parameter objects and dtypes as an inactive block's. Under autocast, `MossBlock._forward`
  casts its stream to `self.proj["q"].weight.dtype`; fp32 weights would put the compiled elementwise chains on fp32
  inputs (recompiles, different chain numerics).
- No `param.data` dtype swap at every switch.
- The existing loader, `check_compute_copies`, `fp16_range_report`, `dit_entries`, `_load_master` work unchanged.

The cost is one cast master -> compute copy per step for the active tensors (~1 GB of traffic for K=8, <1 ms).

Invariant (I1): for every trainable tensor, compute copy == master rounded to the compute dtype, at every window
boundary. Kept by the loader, by the step's copy-back, and untouched by switches.

### 3.3 Window protocol (`CarouselEngine`, GradPipeline's interface)

`attach()`, `begin_window(n_micro, ctx)`, `begin_microstep(j)`, `end_microstep() -> WindowStats | None`,
`check_skip_streak`, `restore_counters`, `close()`, attributes `last_grad_norm`, `skip_streak`, `pinned_bytes`.

1. Hook (every unit / always tensor, registered once; fires only while `requires_grad`): `j == 0`:
   `acc.copy_(p.grad)`, else `acc.add_(p.grad)` (fp16 / fp32 grad promoted to fp32 exactly), then `p.grad = None`.
   One fp16 gradient tensor alive at a time (<= 100 MB). Plain autograd accumulation in fp32 gives the same bits:
   `g1 + g2`, so the fp32 reference test can be bitwise.
2. After the final micro-step's backward (`end_microstep` with j = n_micro - 1):
   1. `found_inf = zeros(1)`; `torch._amp_foreach_non_finite_check_and_unscale_(accs, found_inf, inv_scale)`
      (inv_scale from `StepContext.inv_grad_scale`, 1.0 without fp16; power of two, so exact).
   2. One host sync: `found_inf.item()`. Non-finite -> skip the whole window. Masters, states and step counters are
      unchanged. `WindowStats(skipped_params = number of trainable tensors, overflow=True)`. `skip_streak` + 1.
   3. Clip: `torch.nn.utils.clip_grad_norm_(handles, max_grad_norm, foreach=True)` over the active + always handles in
      registration order (exact; `--moss_ft_clip_mode` is ignored with one info line). grad_norm logged (unscaled).
   4. `handle.grad = acc view` for every trainable handle; `optimizer.step()`; `handle.grad = None`.
   5. Copy-back: `compute.copy_(master)` per active tensor; fp16 range check over the new compute copies, one fused
      `isinf().any()` folded into the next sync. A master past 65504 stops the run naming the tensor (as `ft_host`).
   6. `WindowStats`: `clip_mode "exact" | "none"`, `stepped_params`. `tail_s` = optimizer step + copy-back seconds.
      `worker_busy_s` = the same (no worker). `hook_wait_s = 0`. `extra` = the carousel scalars of 3.11.
3. Non-final micro-steps return None (as GradPipeline).
- `--moss_ft_overflow_policy`: carousel always behaves as `window` (info line when left at `partial`).
- Piece / ring / host-thread flags are unused (info line).

### 3.4 Switch protocol (exact)

`engine.after_step(global_step)` is called by the loop right after `global_step += 1` (before logs, samples, saves).
When `global_step % P == 0`: `new = select(global_step // P)`; if `new == current`: nothing.

1. Precondition: the window was stepped (accs unused), no autograd graph alive. `torch.cuda.synchronize(device)`.
2. Outgoing units (`current - new`, ascending):
   1. `pool.master[unit range].copy_(dev_master_flat)`: one contiguous D2H, since a unit's slots are contiguous in
      registration order.
   2. keep:
      - AdamW / CAME: `host_state[unit].copy_(dev flat)` (fp32 exact; `state_dtype bf16`: CAME exp_avg rounded once to
        bf16, the factored stats stay fp32).
      - bnb: each tensor's `state1/state2/absmax*` moved; `qmap*` left out (shared), the int `step` kept.
      - reset: nothing.
   3. `requires_grad_(False)` on the unit's compute tensors.
   4. Drop the unit's handles from the param groups, `optimizer.state.pop(handle)`, `handle.data = empty(0)`.
   5. Free the device flats.
3. Incoming units (`new - current`, ascending):
   1. Allocate the device flats: equal sizes, so the caching allocator reuses the freed blocks.
   2. H2D master.
   3. Rebuild optimizer state:
      - keep with a saved state: H2D the state (bf16 -> fp32 on the device), `optimizer.state[handle]` rebuilt as views.
        bnb: its tensors + the shared `qmap1/qmap2` re-attached + the int `step`.
      - otherwise: empty state (torch and bnb init lazily at the first step; CarouselCAME inits zeros, t = 0).
   4. `handle.data = master view`.
   5. Check I1 for each incoming tensor: `torch.equal(compute, master.to(compute.dtype))` on the device (one 113M compare
      per unit, ~1 ms). A failure is a RuntimeError naming the tensor ("no drift" guard).
   6. `requires_grad_(True)`.
4. Param groups rebuilt in registration order (decay / no_decay as `ft_model.param_groups`). Floor recomputed and the
   grad-floor wrappers re-applied (3.5).
5. Log one line: step, out ids, in ids, active ids, floor, D2H / H2D GB and seconds.

Peak VRAM during a switch = max(old set, new set): outgoing memory is freed before incoming is allocated.

Exactness: fp32 copies are bitwise; compute copies never leave the GPU; the host master of an inactive unit is never
written while it is inactive. The only lossy option is the opt-in bf16 CAME momentum.

Not in v1: prefetching the next set during the last steps of a period (needs VRAM for a second set, phase 2).

### 3.5 Backward truncation guarantee

Floor f = index of the lowest joint block whose output needs a gradient:
- 0 when the stem is trainable (`small`) or any refiner unit is active;
- else the lowest active joint unit's index.

`apply_grad_floor(model, f, text_grad, unsloth)` (`ft_carousel_units.py`) sets instance attributes (removed on the
next call; `models.py` unchanged):
- `layers[i].forward` for i < f: a wrapper running `MossBlock.forward` under `torch.no_grad()`. Nothing is saved, no
  checkpoint node, no unsloth D2H copy of the block input, no recompute.
- `layers[f].forward` under `--unsloth_offload_checkpointing`: `x = x.detach().requires_grad_(True)` when x has no grad
  (else the unsloth Function's output carries no grad and the floor block's weight gradients are lost, 2.2). Cost: one
  block's dgrad (~1/134 of the step). Plain checkpoint needs nothing.
- `text_embedder.forward` when the text path is frozen (always != small and no refiner unit active): a `torch.no_grad()`
  wrapper. This also disables the `requires_grad_` hack of 2.2.
- `ft_loss`: `noisy.requires_grad_(True)` only when not carousel (one guarded line). With the no-grad prefix it would be
  cut anyway, but at f = 0 it would cost block 0's input dgrad.

Why this is a guarantee: below f, every op runs with grad disabled, so no graph exists to backprop into. At f, the
input is a leaf: the graph starts there. Above f, frozen blocks get activation gradients only, since autograd computes no
weight gradient for a `requires_grad=False` weight (the mm backward checks `needs_input_grad`). Switches happen only
between windows, so a recompute never sees other flags than its forward.

### 3.6 Optimizers and state policies

`--moss_carousel` with `optimizer_type`:

- `""` / `AdamW` -> `adamw`: `torch.optim.AdamW(groups, fused=True)`. Fused on CUDA and on the CPU tests. Same kernel
  family as `HostAdamW.step_chunk` (`torch._fused_adamw_`). Hyperparameters from `ft_args.ADAMW_ARGS`.
- `AdamW8bit` (or `bitsandbytes.optim.AdamW8bit`, `--use_8bit_adam`) -> `adamw8bit`: `bnb.optim.AdamW8bit(groups,
  betas, eps, weight_decay=<parsed, default 0.0 - bnb's own default is 1e-2>)`. `PagedAdamW8bit` / `is_paged` refused:
  unified memory would fight the explicit switch. CUDA only (CPU start refused with that reason).
- `CAME` -> `came`: new `CarouselCAME(torch.optim.Optimizer)` in `ft_carousel_optim.py`.
  - The `ft_optim` cores on the device's tensors; scalars from `came_scalars(..., device=device)`.
  - Per-parameter `state["step"]` (int), and `came_bias_correction(t)` per tensor.
  - `bias_correction` defaults to True. An explicit False is refused when beta3 > 0.99 (`ft_args.CAME_BETA3_LIMIT`,
    message with `CAME_AMPLIFICATION`) and warned otherwise.
  - Options: the came_host set (`ft_args.CAME_ARGS`).
  - Bitwise check against `HostCAME` math on the CPU.

State policy `--moss_carousel_state {keep,reset}`:
- `keep` (default): the state follows the unit to the host and back. AdamW's per-parameter step makes the bias
  correction count only the unit's own updates. Weight decay is applied only on steps where the unit is active (LISA
  semantics).
- `reset`: an outgoing unit's state is dropped; an incoming one starts fresh (t = 0).
- `--moss_carousel_state_dtype {fp32,bf16}`: host storage of kept states. bf16 only for `came` (exp_avg; lossy, opt-in);
  refused for `adamw` (v in bf16) and meaningless for `adamw8bit` (already 8-bit, stored as-is = exact).

User question answered here: on the GPU, CAME's per-parameter eager step (~20 kernels per tensor, ~10 tensors per unit)
is estimated at tens of ms for K=9. The 20.5 s was the CPU host step (design 3.3). CAME's state is 4 B/param + factored
(AdamW 8 B), which buys ~2 more active units at b1. Fused AdamW stays the fastest step, but on the GPU every step is
small next to the ~0.6 s backward. OCGOptV2 saves no memory (2.3).

### 3.7 Selection (`--moss_carousel_select`)

A pure function `select(i)` of the switch index i = global_step // P and the config (U units, K, seed). No RNG state,
no draws from torch / numpy / python global RNGs: the training noise stream is unchanged, and a resume needs only
`global_step`.

- `random` (default): a stream of shuffles. Cycle c = all U units shuffled by `random.Random(f"{seed}:{c}")` (Python's
  Mersenne Twister with a string seed: stable across versions). Set i = the next K distinct units of the stream: an
  element already in the set (at a cycle boundary) is consumed and skipped. Computed by simulating from 0 (O(i K),
  cached).
- `round_robin`: M = ceil(U / K) sets per cycle, set i = {u : u mod M == i mod M}. Spread across depth, each unit
  exactly once per cycle; a set may have K-1 units.
- `window`: contiguous: set i = the next K distinct units of the stream 0, 1, ..., U-1, 0, 1, ... (stride K, wrapping).
  The highest floor on average.
- seed: `--moss_carousel_seed` (default `args.seed`, 0 when unset).
- `K = U`: every set is all units, no switching. This is the equivalence-test configuration.

### 3.8 Memory formulas

Per-unit bytes on the device, on top of the compute copy (exact counts from the meta model):

| | per param | joint unit | refiner unit | small (99.13M) | head (19.28M) |
|---|---|---|---|---|---|
| adamw: master 4 + acc 4 + m 4 + v 4 | 16 B | 1,812,041,728 B = 1.6876 GiB | 1.6881 GiB | 1.4772 GiB | 0.2872 GiB |
| adamw8bit: 8 + bnb state | 10.03 B | 1,136,103,424 B = 1.0581 GiB | 1.0584 GiB | 0.9262 GiB | 0.1801 GiB |
| came: 8 + exp_avg 4 + factored | 12 B + 0.47 MiB | 1,359,524,864 B = 1.2662 GiB | 1.2667 GiB | 1.1086 GiB | 0.2156 GiB |
| optimizer temporaries | | adamw / 8bit: 0. came: 2 x the largest active tensor fp32 = 0.375 GiB (0.42 GiB with `small`: shared_block_modulation) | | | |

VRAM (allocated):

```
alloc = W + A_always + K * A_unit + T_opt + act + te
  W        = 10.07 GiB  every compute copy (fp16 joint Linears + fp32 rest; mixed_precision "no": x2)
  A_always = the small / head / none column above
  A_unit   = the joint-unit column of the chosen optimizer (refiner units are 0.03 % larger: the budget takes the max)
  T_opt    = the optimizer temporaries row
  act      = the activations of ft_budget.gpu_budget (block inputs [only blocks >= floor; the estimate keeps all 42],
             recompute, misc) + one gradient in flight (0.1 GiB, replaces GRADS_IN_FLIGHT_BYTES 0.5 GiB) + the live TE
             pass / held batches as gpu_budget counts them
device = alloc * 1.07 + 2.3 GiB  (ALLOCATOR_SLACK, OUTSIDE_BYTES)  <= 31.84 GiB
```

K by budget: K = floor((G_alloc - W - A_always - T_opt - act) / A_unit), G_alloc = (G - 2.3) / 1.07.
G = `--moss_carousel_vram_gib`, default 31.84 - 1.34 = 30.5 GiB (`GPU_WARN_BYTES`), i.e. G_alloc = 26.4 GiB. At the
full 31.84 GiB card, G_alloc = 27.6 GiB.

Worked K at 1024², fp16 + fp32 stream, live TE, `act` = the measured non-weight peaks of 2.4 (b1 3.5 GiB, b4 11.1
GiB), full card (27.6 GiB allocatable). Free for the optimizer: b1 14.0 GiB, b4 6.4 GiB.

| always | optimizer | K at b1 | K at b4 |
|---|---|---|---|
| small | adamw | 7 | 2 |
| small | adamw8bit | 12 | 5 |
| small | came | 9 | 3 |
| head | adamw | 8 | 3 |
| head | adamw8bit | 13 | 5 |
| head | came | 10 | 4 |

With the ft_budget activation estimate instead of the measured one, the b4 numbers come out about 2-4 units higher.
That estimate is known low at b4 (2.4), so the start-up line prints both the estimate-based K and the reason. After the
first window the engine logs the measured peak and the K that peak would allow.

Host RAM:

| item | bytes |
|---|---|
| fp32 masters of every trainable tensor (HostPool, pageable) | 18.93 GiB |
| kept states, all 44 units visited (4,983,178,240 params) | adamw 37.13, adamw8bit 9.43, came 18.58 (bf16 momentum 9.30), reset 0 GiB |
| always-group / active states at a save (temporary host copies) | <= K x unit state + always state, e.g. adamw K=7: 6.7 GiB |
| process + workers | `PROCESS_BYTES` 8 GiB (+1 GiB per worker past 4) |
| live TE | as now: copy 12.9 GiB + 1 GiB pinned staging; mmap 1 GiB staging (+ ~13 GB OS file cache wanted) |
| no gradient rings, no host accumulator | 0 |

Example adamw keep, live TE mmap: 18.93 + 37.13 + 8 + 1 = 65.1 GiB, required with margin 73.1 GiB. The adamw_host b4
GA2 smoke peaked at 94.5 GiB RSS.

Pinned memory: none by default (pageable copies measured at 8.4-8.7 GB/s). `--moss_carousel_host_pinned masters` pins
the 18.93 GiB master pool; refused if the run's total pinned memory would pass 40 % of physical RAM (Windows caps it near
50 %). It only speeds the master half of the switch copy (section 3.9); states stay pageable.

Disk per training state:
- master.safetensors 20.33 GB, plus the kept states.
- adamw keep: + 39.87 + 0.79 (always) = ~61.0 GB.
- adamw8bit: ~30.7 GB.
- came fp32: ~40.7 GB (bf16: ~30.7 GB).
- reset: 20.33 + the active and always states (adamw K=7: +7.1 GB).

### 3.9 Time model (estimate; the smoke measures it)

- Step (b1, 1024², K=7 adamw, random, `small`): GPU ~0.8 s x 134/168 = 0.64 s, + optimizer step ~20 ms (0.9B params x
  28 B at ~1.5 TB/s), + switch amortised, = ~0.69 s/step. adamw_host: 4.84 s.
- b4 x GA2: ~2.6 s x 0.8 = ~2.1 s/step = ~0.26 s/image. adamw_host: 1.21 s/image. But K is only 2 with adamw at b4;
  adamw8bit (K=5) or came (K=3) suit b4 better.
- Switch (pageable, the whole set changes; random and window sets are disjoint within a cycle):

  | optimizer | per unit (D2H + H2D) | full switch |
  |---|---|---|
  | adamw keep | 1.36 GB each way, 0.32 s | K=7: 2.2 s |
  | adamw reset | 0.45 GB each way, 0.11 s | K=7: 0.75 s |
  | adamw8bit keep | 0.68 GB each way, 0.16 s | K=12: 1.9 s |
  | came keep | 0.91 GB each way, 0.21 s | K=9: 1.9 s (bf16 momentum: 1.4 s) |

  At P = 50 (default) and ~0.7 s/step: 2.2 / 35 = ~6 %. P = 100: ~3 %.
- Trade-off to state in the guide: each unit trains in K/44 of the steps (K=7: 16 %). Per hour that is about the same
  number of block-updates as adamw_host at b1, with fresher gradients and a much faster small group. Whether quality
  per hour is better is the empirical question (LISA / BAdam report parity on LLMs; not measured for DiTs here).

### 3.10 Save, state format, resume

- `engine.flush_to_host()` before any save (the window is drained): active masters D2H into the pool; with `keep` the
  active units' states into the host store; the always group's states into temporary host tensors. Nothing on the
  device changes.
- DiT save: `ft_model.save_moss_dit(path, model, pool, ...)` unchanged. `dit_entries` reads every trained tensor from the
  pool master: active (just flushed) and inactive alike. bf16 / fp16 / float as now.
- Training state: the existing atomic directory of `ft_state.write_state`, mode = `ft_budget.CAROUSEL`:
  - `master.safetensors`: unchanged (every Base key, fp32): a loadable fp32 DiT.
  - `carousel_optimizer.safetensors` (new): sparse. Every tensor that has optimizer state (keep: every unit ever
    activated + the always group; reset: the active ones + always) as `<param name>::<state key>` in its stored dtype.
    - adamw: `exp_avg`, `exp_avg_sq` fp32, `step` fp32 0-dim (torch's).
    - adamw8bit: `state1`, `state2` uint8 (fp32 below 4096), `absmax1`, `absmax2` fp32, `step` int64; never `qmap*`.
    - came: `exp_avg` fp32 (or bf16 as stored), the factored stats fp32, `step` int64.
    - `ft_safetensors.TensorEntry` already takes a dtype per entry.
  - `carousel.json` (new):
    - `format: ROTATION_FORMAT`, `name: CAROUSEL`, `optimizer` (kind), `state_policy`, `state_dtype`, `select`, `K`,
      `P`, `seed`, `always`, `units` (names, in order), `switch_index`, `active` (unit ids);
    - `has_state` (unit ids), `torch_version`, `bitsandbytes_version`.
  - `param_groups.pt`, `scheduler.pt`, `rng.pt`, `train_state.json` (loss scaler, last grad norm, skip streak,
    progress): unchanged. No `optimizer_steps.safetensors` / `exp_avg*.safetensors` of the host modes.
  - `ft_state` gets two generic hooks instead of carousel code: `write_state(..., extra_writer=None)` -> extra files +
    `extra_train_state`, and `load_state(..., extra_reader=None)`. The host modes pass None (unchanged).
- Resume:
  1. `load_state`: masters into the pool and the compute copies (`_load_master`, unchanged).
  2. `extra_reader`:
     - refuses another `ROTATION_FORMAT`, another optimizer kind, a unit list that differs (another freeze set);
     - fills the host store (all kept states) and the always group's state.
  3. Changed K / P / select / seed / always: a warning listing them. The set is recomputed from `global_step` with the
     new config; kept states stay valid.
  4. `engine.activate_initial(select(global_step // P))`: every unit of the set is "incoming" (3.4 step 3; I1 checked).
  5. Loss scale, LR, RNG as now.
  6. A mid-period state resumes in the same set and period position, since the set is a function of global_step.
- An adamw_host / came_host state cannot resume a carousel run (mode check, existing message). Its `master.safetensors`
  is a valid start model (`pretrained_model_name_or_path`). Importing its m / v is phase 2.

### 3.11 Logs

- Per step (`accelerator.log`):
  - `carousel/lowest_active_block` (the lowest active joint block) and `carousel/grad_floor` (the joint floor f: 0 means
    full depth) - split by the 11-17 review (it logged f under the first name);
  - `carousel/grad_depth` (42 - f);
  - `carousel/active_units`, `carousel/trainable_params`;
  - `time/optimizer_step`;
  - existing `grad_norm`, `clip_coef`, `train/loss_scale`, `train/overflow`.
- On switch steps: `carousel/switch_s`, `carousel/switch_d2h_gib`, `carousel/switch_h2d_gib`.
  - info line: `carousel switch at step N (cycle c): out [..] in [..] active [..], floor f, D2H x GB y s, H2D x GB y s`.
- Start-up: per-unit bytes, K (and how it was chosen), the VRAM / host breakdown, switch volume and time estimate,
  the truncation note (`small` => full depth).
- After the first window: measured peak and the K it would allow.

---

## 4. Flags and refusals

### 4.1 The name constant

`library/moss/ft_budget.py`: `CAROUSEL = "carousel"` next to `ADAMW_HOST` etc. (re-exported by ft_args like the other
mode names).
- Flags are built as `f"--moss_{CAROUSEL}_..."`, dests `f"moss_{CAROUSEL}_..."`. A helper `ft_args.carousel_arg(args, key)`
  reads them.
- The log prefix is `CAROUSEL.title()`.
- Renaming = one line + the user's TOML keys. `ROTATION_FORMAT` (state files) does not change with it.
- A test asserts that every carousel dest is built from the constant.

### 4.2 Flags

| flag | default | meaning |
|---|---|---|
| `--moss_carousel` | off | enable (selects the mode; optimizer_type picks the kind) |
| `--moss_carousel_active N` | - | active units, 1 .. U (44); U = no switching |
| `--moss_carousel_vram_gib G` | 30.5 | device GiB the run may use; K from 3.8 (only when `_active` is not given) |
| `--moss_carousel_period P` | 50 | optimizer steps between switches, >= 1 |
| `--moss_carousel_select` | random | random / round_robin / window |
| `--moss_carousel_seed` | `seed` | selection seed |
| `--moss_carousel_always` | small | small / head / none / rotate_stem (11-17 review: the stem as a 45th unit) |
| `--moss_carousel_state` | keep | keep / reset |
| `--moss_carousel_state_dtype` | fp32 | fp32 / bf16 (came only) |
| `--moss_carousel_host_pinned` | none | none / masters |

### 4.3 Refusals (each a `ValueError("MOSS full FT: ...")` with the reason)

- Optimizer:
  - `--fused_backward_pass`, Adafactor, Lion, Prodigy, any other optimizer type;
  - Paged 8-bit / `is_paged`;
  - `adamw8bit` without CUDA;
  - came `bias_correction=False` with beta3 > 0.99; came grams / both / ams_bound / kahan_sum (as came_host).
- Flags:
  - `_active` outside 1..U, or both `_active` and `_vram_gib` given;
  - K from the budget < 1, naming the shortfall;
  - `_period` < 1;
  - `state_dtype bf16` with adamw / adamw8bit;
  - `host_pinned masters` past 40 % of physical RAM in pinned total;
  - every unit frozen by `--moss_ft_freeze`.
- Unchanged from today and also refused here: block compile, block swap, cpu_offload checkpointing, deepspeed,
  full_bf16 / fp16, fp8, multi-process, gradient checkpointing missing on CUDA.
- Ignored with one info line: `--moss_ft_clip_mode` (always exact), `--moss_ft_overflow_policy` (always window), the
  piece / ring / host-thread flags.

---

## 5. Files and functions

New (small single-purpose modules, `library/moss/`):

- `ft_carousel_units.py` (pure, no device):
  - `discover_units(names, frozen) -> List[Unit]` (name, slot indices, params);
  - `always_groups(names, policy) -> List[str]`;
  - `select(i, units, k, kind, seed) -> Tuple[int, ...]` + `Selector` (cached simulation);
  - `grad_floor(active, always, units) -> (f, text_grad)`;
  - `apply_grad_floor(model, f, text_grad, unsloth) / clear_grad_floor(model)`;
  - `unit_bytes(kind, shapes) / always_bytes(...)` (the 3.8 table, used by ft_budget).
- `ft_carousel_optim.py`:
  - `make_optimizer(kind, groups, kwargs, device)`;
  - `CarouselCAME`;
  - per-kind state codecs: `state_keys(kind)`, `shared_keys` (bnb qmaps), `to_host(state, policy, dtype)`,
    `to_device(...)`, `entries(...)` / `load_entries(...)` for the state file.
- `ft_carousel.py`:
  - `CarouselEngine(model, pool, slots, units, optimizer, args, device)`: device flats, handles, hooks, the window
    protocol (3.3), `after_step`, `switch(new)` (3.4), `activate_initial`, `flush_to_host`, `state_writer` /
    `state_reader` (3.10), logs (3.11), `check_invariant()`;
  - `build(args, dit, pool, slots, kwargs, device) -> (optimizer, engine)`.

Changed (behaviour of the existing modes preserved; the 530 existing tests keep passing):

- `ft_budget.py`: `CAROUSEL`; `carousel_gpu_budget(...)`, `carousel_host_budget(...)`, `carousel_active_from_budget(...)`;
  `state_bytes` branch; `startup_report` / `run_startup_checks` branch (report lines, the host refusal as today, a K < 1
  refusal).
- `ft_args.py`: the flags (built from the constant); `optimizer_mode` returns `CAROUSEL` when `--moss_carousel` (8-bit
  accepted only there); `carousel_kind(args)`; `parse_optimizer_args` for the kinds; validation and refusals of 4.3;
  `_check_saving` uses the carousel state size.
- `ft_model.py`: `pool_roles(CAROUSEL) -> ("master",)`; no accumulator for it.
- `ft_state.py`: `extra_writer` / `extra_reader` hooks (3.10); `ft_state._extra_entries` keeps the entry's own dtype.
- `ft_host.py`: `WindowStats.extra: Dict[str, float] = field(default_factory=dict)`.
- `ft_loss.py`: the `noisy.requires_grad_` guard (3.5).
- `moss_train.py`:
  - `make_optimizer` / `make_pipeline` branches;
  - `window_context` passes `inv_grad_scale` for every mode (today it drops it when the optimizer has no
    `snapshot_group`);
  - after `global_step += 1`: `pipeline.after_step(global_step)` when present;
  - `Run.save_dit` / `save_state` call `pipeline.flush_to_host()` when present;
  - `window_logs` merges `stats.extra`;
  - the resume path calls the engine's reader / `activate_initial`.
- `docs/moss_train.md`: a Carousel section (Korean summary + English) after implementation.
- `models.py`, `ft_host.GradPipeline`, `ft_optim`: unchanged.

---

## 6. CPU test list (`CUDA_VISIBLE_DEVICES=-1`, tiny model `tests/_moss_ft_common.TINY`: 3 joint + 2 refiner = 5 units)

`tests/test_moss_ft_carousel_units.py`
1. Units on the real meta model: 44 units in forward order, joint 113,252,608 / refiner 113,284,352 params, norms inside
   their block; `small` = 99,130,368 and `head` = 19,276,800 with the exact tensor names; `--moss_ft_freeze` drops
   tensors / units.
2. Selection:
   - `random` covers every unit once per cycle, K constant, no duplicates in a set at cycle boundaries, same output
     for the same seed, independent of the global RNGs (draws before / after unchanged);
   - `round_robin` and `window` sets as specified;
   - `select(i)` equal from a cold start and from the cache;
   - K = U is always the full set.
3. Floor: values for each always policy and for refiner-active sets.

`tests/test_moss_ft_carousel_budget.py`
4. Per-unit bytes equal the 3.8 table (1,812,041,728 / 1,136,103,424 / 1,359,524,864; small / head columns).
5. K from a VRAM budget at the table's boundaries (K, K+1 GiB edges); refusal when K < 1.
6. Host budget for keep / reset / bf16, and the state size per kind.
7. The start-up report lines.

`tests/test_moss_ft_carousel_args.py`
8. Every refusal / ignore of 4.3.
9. All dests derive from `ft_budget.CAROUSEL`: rebuilding the parser with a monkeypatched constant renames every flag.

`tests/test_moss_ft_carousel_equivalence.py` (the core)

10. **K = U, no switching == plain GPU-optimizer full FT, step for step, bitwise** (fp32 compute).
    - Reference: the same model with plain autograd accumulation, `clip_grad_norm_` over all params in registration
      order, `torch.optim.AdamW(fused=True)` over `ft_model.param_groups` (the `train_reference` helper).
    - GA 1 and 3 with a partial last window; clip on / off; weight decay on; LR scheduler.
    - Compared after every window: masters, compute copies, exp_avg, exp_avg_sq, steps.
11. Same for `came` against a reference loop of `HostCAME.step_param` math on fp32 tensors (bias correction on).
12. bf16 compute copies, K = U vs the existing `adamw_host` GradPipeline (`moss_ft_clip_mode exact`, no clipping): masters
    and compute copies bitwise. With clipping: within 1e-6 relative (the host norm sums in another order).
13. **K < U with switching == "masked plain FT"**:
    - The reference model toggles `requires_grad` per the same `select(i)` every P steps.
    - One `torch.optim.AdamW(fused=True)` over all params with `zero_grad(set_to_none=True)`: a param without grad is
      skipped, its state and step untouched = keep. For reset the reference pops `optimizer.state[p]` for deactivated
      params.
    - Bitwise per window over >= 3 switches, random / round_robin / window, GA 2.
14. Loss scale:
    - a power-of-two scale gives the unscaled run's bits;
    - an injected inf skips the whole window (masters, states, steps unchanged), halves the scale, and the next
      clean window steps;
    - `skip_streak` / `--moss_ft_max_skipped_steps`.

`tests/test_moss_ft_carousel_switch.py`
15. **Exact round trips**: activate -> train -> deactivate -> activate for every kind.
    - adamw and came for real.
    - adamw8bit through a CPU stand-in optimizer with bnb's exact state layout: uint8 states, absmax, a shared qmap
      object that must stay the same object, int step.
    - Every tensor bitwise after N round trips; the host master of an inactive unit never written.
16. Invariant I1 after every switch; the I1 guard raises on a corrupted compute copy.
17. reset zeroes state and step; bf16 momentum rounds exactly once per deactivation (equals `x.bfloat16().float()`),
    factored stats stay fp32.
18. Memory hygiene: after a switch no device tensor of an outgoing unit is referenced (optimizer.state, groups, handles).

`tests/test_moss_ft_carousel_backward.py`
19. **Only active params get grads**: after a micro-step, `.grad` is None on every compute tensor, `acc` is filled for
    active + always only, and inactive `requires_grad` is False. The accumulated grads equal the same model's grads with
    every param trainable (fp32: bitwise).
20. **No backprop below the lowest active block**, for no checkpoint, plain checkpoint and unsloth checkpoint:
    - each block's `_forward` is wrapped with a counter of grad / no-grad calls;
    - blocks < f run once under no_grad (no recompute, no unsloth CPU copy: `torch.Tensor.to` spy);
    - blocks >= f run forward + recompute;
    - output hooks show no gradient reaches outputs below f - 1;
    - the floor block input has `grad_fn is None`.
21. The unsloth pitfall: the floor block's weight grads exist and equal the no-checkpoint run's.
22. Text path: frozen -> the refiner `_forward` runs once without grad; with a refiner unit or `small` -> grads on the
    text path and f = 0.

`tests/test_moss_ft_carousel_state.py`
23. **Resume exactness**:
    - Uninterrupted run vs a run saved and resumed into fresh objects, at a mid-period step and exactly at a switch
      boundary.
    - Equal: masters, compute copies, all optimizer states incl. steps (keep and reset; adamw and came), active set,
      loss scale, LR, the next noise draws.
24. Refusals: another kind, another unit list, a host-mode state. Warnings: changed K / P / select / seed.
25. The DiT save while units are active on the device equals the assembled masters; Base key set and dtypes as the
    existing save tests.

`tests/test_moss_ft_carousel_train_e2e.py`
26. `moss_train.main` on the CPU, tiny model, adamw and came, GA 2, save + `--resume` in the middle: same final DiT as
    the uninterrupted run; the carousel log scalars present.
27. The existing 530 `tests/test_moss_ft_*.py` stay green (run with the venv_t214 python).

---

## 7. Order of work

1. Add `ft_carousel_units.py` and `ft_budget` formulas with tests 1-7 (pure, fast).
2. Add `ft_carousel_optim.py` (CarouselCAME, codecs) with tests 11, 15, 17.
3. Add the `ft_carousel.py` engine with tests 10, 12, 13, 14, 16, 18-22.
4. Add the `ft_state` hooks, resume and save with tests 23-25.
5. Wire up `ft_args` / `moss_train.py` with tests 8, 9, 26, 27; then the guide section.

---

## 8. GPU smoke items (later; GPU use needs the user's go-ahead)

- C1: adamw8bit real round trip (bnb on CUDA) and its first steps against bnb without switching (`K = U`).
- C2: VRAM peak vs the 3.8 formula at b1 / b4 for each kind (K from the budget); fp16 overflow behaviour; I1 guard
  silent.
- C3: switch time vs 3.9 (pageable and `host_pinned masters`); step time vs 3.9; the truncation speed-up with `head`
  + `window` vs `small` + `random`.
- C4: 200-step stability (fp16 + fp32 stream + loss scale) for adamw and came.

## 9. Open points for the user

- Defaults: `always small` (requested; full-depth backprop) vs `head` (real truncation). `P = 50`. `K` from the VRAM
  budget.
- The final name (one constant).
- Phase 2 candidates:
  - prefetch the next set while the current period ends;
  - import m / v from an adamw_host state;
  - an OCGOptV2 kind with per-parameter step;
  - `window` direction descending.
