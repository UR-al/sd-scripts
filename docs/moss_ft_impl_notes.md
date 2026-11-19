# MOSS full fine-tune - implementation notes

Deviations from docs/moss_full_finetune_design.md and details it leaves open, one section per task.

## T1 - arguments, LR phases, budget (`library/moss/ft_args.py`, `library/moss/ft_budget.py`)

Tests: `tests/test_moss_ft_args.py`, `tests/test_moss_ft_lr_phases.py`, `tests/test_moss_ft_budget.py`
(the 2.1 and 2.2 tables and the 8.1 example TOML are read from the design document itself).

**Parser (4.2).** `ft_args.setup_parser()` is the 4.2 composition + `add_moss_ft_arguments`, with moss_train.py's own
defaults: `mixed_precision` and `save_precision` "bf16", `cfm_lambda` 0.02, and `learning_rate` / `max_train_steps`
None. kohya's defaults (2e-6 / 1600) made "unset" undetectable, which the phase rules need; `validate_moss_ft_args`
fills kohya's values in when nothing else sets them.

**Validation entry points.** `validate_moss_ft_args(args)` runs the dataset-free part (4.3 step 2) and marks
`args._moss_ft_static_checked`; `validate_moss_ft_args(args, train_group, val_dataset_group=...)` (step 4) then runs
the dataset part only (or both, if step 2 was skipped). Extra keyword-only arguments: `val_dataset_group` (validation
datasets are refused), `cuda` (default `torch.cuda.is_available()`; gradient checkpointing is required there), and the
injectables `virtual_memory`, `disk_usage`, `exists`. `num_processes > 1` is refused through `WORLD_SIZE` here and by
`check_single_process(accelerator.num_processes)`, which moss_train.py should call after `prepare_accelerator`.

**LR phases (4.6).**
- `max_train_epochs` is set from the phases only when their sum is whole. The dataset part always sets
  `args.max_train_steps` = the last boundary, which is kohya's `max_train_epochs x steps_per_epoch` for a whole sum. A
  fractional sum ("0.5:1e-5,1:1e-6") leaves `max_train_epochs` None, so moss_train.py must take `args.max_train_steps`
  as validated and not recompute it from `max_train_epochs`.
- Boundaries round half up after rounding float noise to 6 decimals. Python's `round()` would round 2.5 to 2.
- One phase is accepted (a constant LR with warmup). The warmup may end exactly at b_1 (W <= b_1).
- "One schedule source" also refuses `lr_scheduler_type`, `lr_scheduler_args`, a non-zero `lr_decay_steps` and
  `zero_lr_warmup` together with phases: they would be silently ignored. `lr_scheduler` "adafactor..." is refused in
  every mode. (`zero_lr_warmup` is now refused in every mode: "Review fixes" 3.)
- `make_lr_phase_scheduler` returns `LrPhaseScheduler(LambdaLR)`. It sets every group's `lr` / `initial_lr` to LR1.
  `state_dict()` keeps the step count and records the schedule under `moss_ft_lr_phases`.
  `load_state_dict()` rebuilds the base LR and the lambda from the current config, writes the resumed step's LR into
  the param groups at once, and warns with both versions when the schedule changed. A bare LambdaLR load would leave
  the constructor's LR in the groups until the next `step()`, so the first resumed window would read the wrong LR.
- In modes a and c `optimizer.step()` is never called, so torch warns once ("lr_scheduler.step() before
  optimizer.step()") at the first scheduler step. It is harmless (pytest.ini ignores UserWarning).

**Optimizer (4.5).**
- `--fused_backward_pass` with AdamW / CAME is refused (the flag selects mode b).
- `parse_optimizer_args(args, mode, quiet=False)` returns the full keyword dict. For adamw_host and came_host it is
  exactly `ft_optim.HostAdamW` / `HostCAME`'s keywords (checked by a test); for adafactor_fused it is
  `get_optimizer(..., optimizer_kwargs=)`.
- adafactor_fused: missing `scale_parameter` / `relative_step` / `warmup_init` are filled in as False (explicit True
  is refused; kohya would default `relative_step` to True). `beta1=None` is accepted.
- came_host: `ams_bound` / `kahan_sum` are accepted only as False. came.py's deprecated `cautious` is refused (use
  `update_strategy=cautious`), as `HostCAME` does. `update_strategy=None` means "unmodified".
- Duplicate optimizer_args keys are refused.

**Fixed handling (4.5).**
- `--moss_stream_fp32` is required with bf16 compute only. `mixed_precision "no"` runs the stream in fp32 anyway.
- Validation: the per-dataset `validation_split`, a non-empty validation group and an args-level `validation_split`
  are refused.
- `--resume` is refused when it is not a directory, or when it holds accelerate markers (`optimizer.bin` /
  `random_states_0.pkl`) without `COMPLETE.json`. `--resume_from_huggingface` is refused. The remaining checks are
  ft_state's.
- `--moss_ft_freeze` splits only on top-level commas: a comma inside `()`, `{}`, `[]` or escaped stays in its regex.
  A TOML array is accepted. ft_model's `select_trainable` can reuse `frozen_parameter_names(names, value)`.

**Budget (2.1-2.3, 3.5).**
- GPU "rope, modulation, head, fp64 loss, cuBLAS workspace" row: 0.25 GiB + 26 KiB per joint token. This fits that
  row and every total and device value of the 2.1 table within +-0.05 GiB. The tightest is 1536 b1 at 0.041.
- fp32 compute: all weights in fp32 and the recompute row x2, giving 24.9 GiB at 1024 b1 (the design says "about
  24.7").
- Mode b on the GPU = mode a + its Adafactor state: 0.011 GiB (the design says "about 0.03").
- The CAME factored state is 0.021 GiB (the design says "~0.01"). It includes the 1-D `exp_avg_res` as an upper bound.
- The host accumulator rule matches `ft_host.needs_accumulator` (a test compares the two): GA > 1, or exact clipping
  with `max_grad_norm > 0`. Design 2.2's "or `--moss_ft_clip_mode exact`" is read with clipping on.
- Disk: the 2 x state requirement applies only when states are saved (`save_state` or `save_state_on_train_end`). One
  DiT save is checked at `output_dir`, summed with the state requirement when both directories are on one filesystem.
  An unreadable drive refuses with the reason. (Superseded by "Review fixes" 17: the check counts the run's SavePlan.)
- The cache warning counts the TE and latent caches that are not yet on disk (one existence check per image) against
  the free space of each image drive. In-memory caches (no `*_to_disk`) are added to host RAM instead. The TE cache is
  8 KiB + a 4 B mask per token. Latent "auto" = fp32.
- The latent rows of 2.3 ("16-32 GB", "56-112 GB") count a MiB as a MB. The exact values are 16.8-33.6 GB and
  58.7-117.4 GB.
- Throughput uses the design's model as written: a micro-step takes max(b x t_img, A), and tail =
  max(0, F - 0.75 b t_img). t_img at other resolutions scales by a FLOP model pinned to 1.1 s at 1024^2.
  - batch 1 x GA 8 gives 12.0 s/step (0.67 img/s, 2,937 tok/s). The design's ~0.63 img/s, ~2,750 tok/s and 57 h
    assume about 12.7 s, inside its own "~12-13".
  - batch 8 x GA 1 gives 8.8 s against ~8.9.
  - The tests hold the time columns to +-0.1 s (s/step against the stated ranges) and the rates to +-7 %.

**Open issues for moss_train.py.**
- Call `check_single_process(accelerator.num_processes)`.
- Use `args.max_train_steps` as validated.
- With phases, build the scheduler with `make_lr_phase_scheduler(optimizer, args.moss_ft_lr_phases,
  steps_per_epoch(len(train_dataloader), GA), args.lr_warmup_steps)`. Without phases use `get_scheduler_fix`.
- ft_state can store `LrPhaseScheduler.schedule()` in train_state.json and use `lr_phase_end_epochs()` for the
  pinned phase states (a fractional phase end has no epoch state).
- Every estimate still waits for the GPU smoke run (design 7).

## T2 - host optimizer pipeline (`library/moss/ft_host.py`, `library/moss/ft_optim.py`; design 3.1, 3.3)

Tests: `tests/test_moss_ft_host_adamw.py`, `tests/test_moss_ft_hooks.py`, `tests/test_moss_ft_precision.py`,
`tests/test_moss_ft_came_bc.py`; shared helpers in `tests/_moss_ft_common.py` (not collected).

**API for the next tasks.**
- Slots and pool: `make_slots(named_params, optimizer=None)` and `bind_slots(slots, named_params, optimizer=None)`.
  Slots can be made from the meta model and bound to the live parameters (and the groups) after `to_empty` /
  `enable_stream_fp32`. `slot_list_hash(slots)` = sha256 of `[name, shape, offset]` in slot order.
  `needs_accumulator(ga, clip_mode, max_grad_norm, overlap=True)`.
- `HostPool(slots, roles=<Optimizer>.POOL_ROLES, with_acc)`: `roles` (dict of flat fp32 tensors, `acc` not in it),
  `acc`, `view(role, slot)`, `steps` (float32 [n_slots]) and `step(slot)`, `layout()`, `slot_hash()`, `nbytes()`,
  `copy_master_from_params()`, `copy_params_from_master()`.
- `GradPipeline(slots, pool, optimizer, device=, gradient_accumulation_steps=, max_grad_norm=, clip_mode=,
  max_skipped_steps=, piece_bytes=, ring_bytes=, upload_ring_bytes=, acc_chunk=, step_chunk=, host_threads=, inline=,
  overlap=True)`: `attach()`, `begin_window(n_micro, StepContext.snapshot(optimizer, step_index))`,
  `begin_microstep(j)`, `end_microstep()` (None, or `WindowStats` for the final micro-step), `step_from_accumulator(c)`,
  `sync()`, `check_skip_streak(stats)`, `restore_counters(last_grad_norm, skip_streak)`, `close()`. For states:
  `last_grad_norm`, `skip_streak` (write a state only after `end_microstep` of a final micro-step or `sync()`).
  For logs: `pinned_bytes`, `WindowStats.tail_s / hook_wait_s / worker_busy_s`.
- `HostAdamW` / `HostCAME` take `lr` plus exactly the dict of `ft_args.parse_optimizer_args` for their mode.
  `HostCAME.extra_state()` = the factored statistics as `"<param>.<came.py key>"` (live views, load in place).
  `HostCAME.bind` (run by `GradPipeline.__init__`) keeps statistics already present, so a resume may load them first.

**Decisions and deviations.**
1. Clipping multiplies the fp32 chunk (`t.mul_(c)`) before the fused kernel instead of passing `grad_scale=1/c`. The
   kernel divides by `grad_scale`, and `g / (1/c)` rounds twice, so it is not `clip_grad_norm_`'s `g * c`. With the
   multiply, exact mode is bit-identical to `clip_grad_norm_` + `AdamW(fused=True)` (tested).
2. Exact mode's G is torch's: `torch.linalg.vector_norm` of each slot's accumulated gradient (the worker computes it as
   a slot's last piece lands, the rest after the drain), then the norm of the stacked norms, then
   `clamp(max_norm / (G + 1e-6), max=1)` as an fp32 0-dim tensor. The chunk `sumsq` (`torch.dot`, summed as a Python
   float) is G only for the in-backward windows ("none" / "stale"). Stale uses the same formula on `fp32(G_prev)`.
3. Stale clipping at GA = 1 has no accumulator (2.2), so a window without a previous G cannot run exact. It runs
   unclipped (`WindowStats.clip_mode` "none", one log line); AdamW's and CAME's first step do not depend on the
   gradient scale beyond eps. At GA > 1 such a window runs exact as designed. A resume should pass the stored G to
   `restore_counters`. (Superseded by "Review fixes" 4: the unclipped gradient still enters the moments; that window
   now only measures G.)
4. `needs_accumulator`: exact clipping allocates `acc` only with `max_grad_norm > 0` (T1's budget uses the same rule).
5. came_host at GA = 1 has no `acc` either. One fp32 scratch the size of the largest parameter (MOSS: 56.6M elements,
   0.21 GiB host) assembles each parameter's gradient: one hook enqueues all pieces of a parameter and the FIFO keeps
   them together (interleaving raises). The 2.2 table / `ft_budget` may add 0.21 GiB for came_host at GA = 1.
6. `StepContext` holds the groups' hyperparameters (`optimizer.snapshot_group`) and the step index. The clip
   coefficient is decided by the pipeline at `begin_window` (it knows G_prev) and reported in `WindowStats.clip_coef`
   and `clip_mode`. Design 4.4's `StepContext.snapshot(optimizer, lr_scheduler, clip_coef(), global_step)` is
   `StepContext.snapshot(optimizer, global_step)`.
7. The worker queue is a deque + Condition (`_Fifo`) instead of `queue.Queue`: its join wakes every 0.5 s to re-raise
   a worker error or notice a dead worker thread. After an error the worker drops queued items (returning their ring
   entries). `acquire()`, the drain and every later call raise `HostPipelineError` (a RuntimeError) with the worker
   traceback; in inline mode the hook / drain raise the same error.
8. Alignment: `acc_chunk` / `step_chunk` must be multiples of 64 elements and `piece_bytes` of 256 bytes. torch's CPU
   fused AdamW splits a tensor into 64-byte units with a vector body and a scalar tail. Chunks at other boundaries are
   not bit-identical to one whole-tensor call (measured: 7e-9 differences with 7-element chunks).
9. `end_microstep()` of the final micro-step does the whole step: the drain, then the exact norm and
   `step_from_accumulator`, or the flush of slots whose gradient came only in earlier micro-steps. It then calls
   `optimizer.step()`, which updates nothing (`HostAdamW.step` / `HostCAME.step` are markers), so an LR scheduler sees
   the call order it expects. The "lr_scheduler.step() before optimizer.step()" warning of T1's note does not appear.
10. A post-accumulate-grad hook that fires twice for one parameter in one micro-step raises. A non-finite value that
    appears only after fp32 accumulation (every micro-step's norm finite) raises instead of skipping, because the
    slot's earlier chunks are already stepped.
11. The CUDA primitives (`_CudaOps`) cannot run here. Their docstring walks the ordering argument line by line. It
    states one assumption: no backward node reads a parameter without feeding its gradient (e.g. `p.detach()` in a
    forward). The inline CPU test of in-backward vs after-backward stepping (no / plain checkpoint) uploads inside the
    hook and would catch a violation.
12. CAME read-out correction: `came_bias_correction(b1, b2, b3, t)` = `1 - fp32(beta)^t` in float64, rounded to fp32
    (the value an fp32 division uses). A divisor of exactly 1.0 is skipped, which is bit-identical. Late in training the
    corrected step is the uncorrected one bit for bit.
13. Design 5's `test_correction_is_state_compatible` ("EMA buffers identical with / without the correction") holds only
    for the second-moment EMAs (exp_avg_sq_row / _col, exp_avg_sq). exp_avg and the residual EMAs follow the same
    update rule but are fed the corrected update, so they differ in the first steps. The test checks the same state
    tensors (names, shapes, dtypes), bit-identical second-moment EMAs, and identical steps from either state once
    1 - beta^t rounds to 1.
14. With `weight_decouple=False`, HostCAME and came.py add `weight_decay * master` to the gradient before computing
    the moments. `cautious_weight_decay=True` takes precedence and applies masked decay directly to the master.
    HostCAME refuses came.py's deprecated `cautious=` and applies the refusal rules to every param group too.
15. `test_bias_correction_removes_amplification` also runs beta3 = 0.9999 (the failing default): with beta2 = beta3 a
    swapped bc2 / bc3 goes unnoticed (checked by a mutation run). Measured (64x64, ratio to the mean of steps
    1,001-3,000): corrected t1 / t2 / t5 / t10 / t24 = 3.37 / 2.45 / 1.62 / 1.27 / 1.04 for both beta3; AdamW
    3.40 / 2.44 / 1.61 / 1.25 / 1.03; uncorrected t1 9.7 (beta3 0.999) and 13.9 (0.9999).

**Open.**
- GPU smoke (7.1, 7.3): `torch.linalg.vector_norm(bf16 grad, dtype=fp32, out=norm_dev[slot])` on the d2h stream,
  `time/hook_wait`, `time/tail`, the ring sizes. Every 1-D parameter takes a whole 64 MiB ring entry (about 200 tiny
  pieces per micro-step): harmless, could be packed later.
- `torch.set_num_threads(host_threads)` in the worker is thread-local for threads that already run, but a thread whose
  OpenMP state is created later starts from the last value set anywhere (ATen's global default). Check thread counts in
  the smoke run if CPU work outside the worker matters.

## T3 - model load / save, states, loss, TE-cache move (`ft_model.py`, `ft_state.py`, `ft_loss.py`, `te_cache.py`, `ft_safetensors.py`; design 4.4, 4.4.1, 4.7, 4.10)

Tests: `tests/test_moss_ft_model.py`, `test_moss_ft_save_dit.py`, `test_moss_ft_state.py`, `test_moss_ft_loss.py`,
`test_moss_ft_te_cache.py`; shared helpers in `tests/_moss_ft_model_common.py` (not collected).

**API for moss_train.py.**
- `model, pool, slots = ft_model.load_moss_for_full_ft(path, args, device, trainable_fn=None, config=None)`. pool is
  None for adafactor_fused. The slots are bound to the live parameters but not to param groups: after building the
  optimizer over `ft_model.param_groups(ft_model.trainable_named(slots), weight_decay)` call
  `ft_host.bind_slots(slots, ft_model.trainable_named(slots), optimizer)`, then the GradPipeline. Checkpointing,
  `enable_elementwise_compile` and `set_ckpt_reuse` stay with the caller (4.3 step 7).
- `ft_model.save_moss_dit(path, model, pool, dtype, ft_model.moss_dit_metadata(args, mode=, optimizer_kwargs=,
  global_step=, epoch=, train_images=))`, after a drain. dtype: a torch dtype or the save_precision string.
- States: `ft_state.write_state(state_dir, name, model=, slots=, pool=, optimizer=, lr_scheduler=, mode=,
  progress=ft_state.TrainProgress(global_step, completed_epochs, epoch_step, steps_per_epoch, GA, lr_schedule),
  last_grad_norm=pipeline.last_grad_norm, skip_streak=pipeline.skip_streak, args=args)` after a drain, then
  `ft_state.apply_keep_policy(state_dir, **ft_state.keep_policy_from_args(args))`. Names:
  `ft_state.epoch_state_name / step_state_name / last_state_name(args.output_name, n)` (kohya's names and defaults);
  directory `ft_state.state_dir_of(args)`.
- Resume: `info = ft_state.load_state(args.resume, model=, slots=, pool=, optimizer=, lr_scheduler=, mode=, args=,
  steps_per_epoch=)` with the objects made as for a fresh start (scheduler from the current config), then
  `pipeline.restore_counters(info.last_grad_norm, info.skip_streak)`, `global_step = info.progress.global_step`, start
  in epoch `info.start_epoch` (0-based) and skip `info.batches_to_skip` batches.
- `ft_loss.moss_ft_noise_and_loss(args, dit, batch, accelerator, sigma_sampler, noise_scheduler, stream_dtype,
  text_encoding_strategy)` -> float64 0-dim loss of the micro-step (the caller divides by n_micro).
- `te_cache.cache_moss_text_encoder_outputs(args, accelerator, vae, dataset, provider, sample_prompts_te_outputs=None)`
  -> the sample-prompt outputs (vae may be None); `te_cache.cache_moss_sample_prompts(args, provider)`.

**Decisions and deviations.**
1. Loader order (4.4.1 steps 4-5): the meta model is cast to the compute dtype and `enable_stream_fp32()` runs on the
   meta model, so the fp32-stream parameters are allocated in fp32 and read exactly from any file dtype. The design's
   order (all bf16 first, then the bf16 -> fp32 cast) is exact only for a bf16 file: an fp32 file (a state's
   master.safetensors, an fp32 DiT) would have its fp32-stream values rounded through bf16 (a test covers it).
2. Per tensor: a trained tensor goes `master.copy_(t)`, then `param.copy_(master)`; a frozen one `param.copy_(t)`.
   Nothing is kept for a second pass (peak: the pools + one tensor). The compute copy is the master rounded to the
   parameter's dtype for every file dtype; `check_compute_copies` asserts it at the end (on CUDA it moves each
   parameter to the CPU for the compare: about 10 GB D2H once at start, not measured).
3. Strict like `load_moss_dit` (missing / unexpected keys, shapes; a fused qkv next to a separate q/k/v is refused),
   except `rope.bands`: optional in the file (derived constants; only its shape is checked). Prefixes as
   `load_moss_dit`. `trainable_fn(model) -> [(name, param)]` replaces the default `select_trainable(model,
   args.moss_ft_freeze)`; `config=` (default MOSS_DIT_CONFIG) is for the tiny test model. The pool roles follow the
   mode, the accumulator `ft_host.needs_accumulator(GA, clip_mode, max_grad_norm)`.
4. `param_groups` adds a "name" ("decay" / "no_decay") and leaves out an empty group (group indices then shift: log by
   name).
5. Streaming writer: own module `library/moss/ft_safetensors.py` (not in the design's file list) instead of
   `mem_eff_save_file`, which needs every tensor converted up front (frozen GPU tensors, master casts) and does not
   fsync. Same format and 256-byte header alignment; compact JSON, `__metadata__` first. The reader reads into the pool
   views (`readinto`, no temporary) without mmap, so a state can be renamed or deleted on Windows right after a load.
6. DiT layout: sorted by key name (the Base file is sorted by name with the F32 `rope.bands` among the BF16 tensors,
   not dtype-first). Checked against the real Base header (`test_full_size_plan_matches_the_base_file`, skipped
   without the file): keys, dtypes, shapes and every data offset are identical; only the header padding differs (256 vs
   8). `save_moss_dit` writes `<path>.tmp`, fsyncs and renames (a crash leaves no truncated file under the name).
7. master.safetensors is fp32 in every mode, also adafactor_fused (model values cast; exact), as T1's `state_bytes`
   counts it. Mode b keeps the torch optimizer's `state_dict()` in `optimizer.pt`. Moments and steps are keyed by
   parameter name in slot order; steps: one int64 0-dim tensor per name (T1 counts 8 B per tensor).
8. train_state.json has no timestamps (a state re-saved right after a load is byte-identical, design 7.4; tested). A
   non-finite last grad norm is stored as null (the pipeline treats both as "no previous norm"). "args": vars(args)
   without kohya's secrets (wandb_api_key, huggingface_token) and private `_*` attributes, plus its sha256; a resume
   logs the changed keys. Hyperparameters = each group without params / lr / initial_lr. rng.pt stores numpy's state as
   tensors and ints so every .pt of a state loads with `torch.load(weights_only=True)`.
9. Atomic replace: a stale `<name>.tmp` is deleted before writing; a stale `<name>.old` only after the new tmp is
   complete; if the final rename fails, `.old` is renamed back. Refusals (`ft_state.StateError`, a ValueError): not a
   directory, an accelerate state (optimizer.bin / random_states_0.pkl / scheduler.bin / pytorch_model.bin without
   COMPLETE.json), no COMPLETE.json, another format, a missing file, a size mismatch, an unreadable safetensors header -
   each names the newest complete state in the same directory (global step, then mtime) - plus another optimizer mode
   or slot list hash. Changed GA, steps per epoch (when passed), hyperparameters, arguments: warnings.
10. Keep policy counts states, as design 4.7 says ("the newest N"). kohya's `save_last_n_steps_state` counts steps
    (the states of the last N steps); the two agree for N = 1 (T1's default). Only directories named like this run's
    epoch / step states AND holding a complete state of this format are deleted: the train-end state, pinned states,
    incomplete or foreign directories (e.g. an accelerate state of a LoRA run with the same output_name) never are.
    Pinned phase states do not count against N. A fractional phase end has no pinned state. (Also never the state just
    written, nor states past the run's position: "Review fixes" 10.)
11. Loss: the LoRA trainer's order, not the design list's: min-SNR comes after the per-sample mean and loss_weights
    (train_network's post_process_loss). Also kept because the FT parser has them: `loss_scale` (conditional_loss's
    scale), `--token_mining`, `--loss_multiplier`. Huber / min-SNR / token mining get `sigmas.flatten()` as the LoRA
    trainer's "timesteps"; the CFM negative target is noise - latents. The batch dict is not modified (the LoRA path
    rewrites text_encoder_outputs_list in it). Left out (LoRA-only): T-LoRA masks, TSM grouped draws, HF / anchor / adv
    stashes, wavelet masking; differential guidance and EDM2 are not in the FT parser. Missing latents / TE outputs:
    ValueError. Bit-identical to `MossNetworkTrainer.process_batch` (loss and every gradient) for 16 option sets incl.
    bf16 + fp32 stream + checkpointing.
12. te_cache: the function returns the sample-prompt outputs instead of setting an attribute and accepts `vae=None`.
    MossNetworkTrainer assigns the result and passes its own `_cache_sample_prompts` (still overridable). The only
    behavior difference: if `accelerator.wait_for_everyone()` raised after the sample prompts were encoded, the
    trainer's attribute stays unset (the run ends with that exception anyway); the three log lines now come from the
    `library.moss.te_cache` logger. The characterization tests give the same event order before and after the move,
    also when the caching raises.
13. `moss_train_network.refuse_lycoris_full(args)` runs first in `assert_extra_args`: network_module lycoris* with
    `algo=full` (case, spaces and quotes ignored) raises with the pointer to moss_train.py and the ~71 GiB figure. A
    Namespace without network fields passes (existing tests call it so). The now unused imports of the trainer
    (captions, strategy_base, clean_memory_on_device) are kept. The file keeps its CRLF line endings.

**Open.**
- CUDA paths not run: `to_empty` on the GPU and per-tensor copies, `check_compute_copies` (D2H compare), frozen tensors
  saved from the GPU, a state loaded into GPU parameters. Load / save times of the 61 GB state (design: 20-30 s) and
  the 10 GB DiT are for the smoke run (7.4, 7.5).
- moss_train.py should save an epoch state at every phase-end epoch even when `save_every_n_epochs` misses it,
  otherwise the pinned state does not exist; and should prepare the DataLoader with accelerate (batches on the device),
  as anima_train.py: the loss uses `loss_weights` / `alpha_masks` as they come, like the LoRA path.
- A kohya-style `save_last_n_steps_state` > 1 (a step count) would keep that many 61 GB states here: T1's validation
  could warn above a few states.

## T4 - the script (`moss_train.py`; design 3.2, 4.3-4.9) and the user guide (`docs/moss_train.md`)

Tests: `tests/test_moss_ft_train_e2e.py` (the script end to end on the CPU), `tests/test_moss_ft_train_unit.py` (the
loop helpers, the start-up refusals, mode b's hooks, the mode b state dtypes).

**Structure.** `train(args)` follows 4.3 step by step; `_train_loop` is 4.4 (resume and DataLoader first, then the
epochs). Small pure helpers carry the loop arithmetic: `window_of(index, num_batches, GA) -> (w, j, n_micro)`,
`progress_after_window(...)`, `epoch_save_due` / `step_save_due` / `phase_state_due`, `window_context`. `Run` holds the
objects the saves and samples share. Mode b is `FusedAdafactorSteps`: anima_train.py's hooks behind GradPipeline's
interface, so the loop has one path.

**Decisions and deviations.**
1. A state written after an epoch's last window - a step state too - records that epoch as completed
   (`completed_epochs = epoch + 1, epoch_step = 0`). Otherwise a resume from it would build the finished epoch's
   iterator only to skip all of it, which draws an extra shuffle from the RNG and loses the exact continuation.
2. Exact epoch-boundary resume (tested bit for bit: master, m, v, step counters, compute copies, the logged loss / LR /
   grad norm / clip coefficient, the final DiT) rests on:
   - the DataLoader has no generator of its own (as anima_train.py): each epoch's permutation is drawn from the global
     torch RNG, which the state restores;
   - `DatasetGroup.attach_epoch_value(current_epoch)`: each epoch's bucket shuffle runs before its first fetch, seeded
     with seed + epoch, so a fresh process reaches the same order through the same chain of shuffles;
   - sampling restores the torch RNG (moss train_utils), nothing else draws between the state and the next epoch.
   It is tested with `max_data_loader_n_workers = 0`. Worker processes seed their RNG at the first iterator of a
   process, so with workers the data order / augmentation draws after a resume differ (the optimizer state is still
   exact). A mid-epoch resume draws a new permutation and skips the trained batch count (design 4.7, kohya).
3. The early refusal: `ft_state.verify_state(args.resume)` runs at step 2, before caching and loading (without it an
   incomplete state would fail only after the caches and the DiT load); `load_state` verifies again.
4. `lr/<group name>` (decay / no_decay, ft_model.param_groups' names) instead of `lr/group0`, and the value is the LR
   the window's update used (read when the window starts). kohya logs `get_last_lr()` after the scheduler step, which
   is the next step's LR.
5. `sample_at_first` samples only on a fresh start (global step 0). kohya's check would also sample at a resumed
   run's start when sample_every_n_epochs is set (epoch 0); here a resume does not.
6. Epoch-end DiT / state saves happen only when the epoch ran to its end. An epoch cut short by max_train_steps (a
   fractional phase sum) is the run's end; its saves are the train-end ones. The pinned LR-phase states (T3's open
   issue) are forced: with save_state, the epoch state at the end of every phase but the last is written even when
   save_every_n_epochs misses it (`phase_state_due`), unless --moss_ft_no_keep_phase_states.
7. Script-level refusals and notes (prepare_runtime): `--output_dir` is required (a missing one would fail only at
   the first save); `--cache_aug_refresh_epochs > 0` is refused (it re-encodes between epochs, the VAE / text
   encoder are released); `save_model_as` other than safetensors is refused; `--cuda_memory_fraction` /
   `--cuda_empty_cache_*` are ignored with a warning (design 2.1: no VRAM cap or cache flush); `--huggingface_repo_id`
   is ignored with a warning.
   `--moss_tier_timing*` (LoRA flags) are unused: the TierTimer always runs and reports every --moss_ft_log_every steps.
8. A mid-epoch state saved with another GA (ft_state warns) whose batch count is not a whole number of windows resumes
   at the start of that window.
9. Mode b: `FusedAdafactorSteps.end_microstep()` calls `optimizer.step()` (a no-op: the hooks dropped every gradient)
   so the LR scheduler sees the expected order (no "lr_scheduler.step() before optimizer.step()" warning).
   `grad_norm` is the global norm of the per-tensor clip norms when max_grad_norm > 0 (computed then anyway), else not
   logged. A non-finite window loss stops the run: the weights already took that step (no per-parameter skip, as
   anima_train.py).
10. ft_state fix (T3's module): `torch.optim.Optimizer.load_state_dict` casts every floating state tensor but "step"
    to its parameter's dtype, so a mode b resume continued Adafactor's fp32 statistics of the bf16 joint Linears in
    bf16 (T3's mode b test ran fp32 parameters). `load_state` now puts the saved dtypes back
    (`_restore_state_dtypes`); the e2e mode b resume (bf16 + fp32 stream) is bit-identical with it.
11. `time/step` runs from the request of the window's first batch to the end of its final micro-step (drain
    included); the previous window's saves and samples are not counted. `throughput/tok_per_s` counts image tokens +
    the padded text length (`MossTransformer.last_text_tokens`) per sample.

**E2E test stand-ins** (monkeypatched; the caches are written by the real code paths: the dataset's
`new_cache_latents` -> MossLatentsCachingStrategy -> moss_vae_encode, te_cache -> MossTextEncoderOutputsCachingStrategy,
the sample-prompt cache): `ft_model.MOSS_DIT_CONFIG` -> a tiny config (text_dim 64); `moss_train.load_moss_vae` -> a
randomly initialised diffusers AutoencoderKLFlux2 with 32 channels; `strategy.load_moss_text_encoder` -> a 4-layer
width-64 MossQwen35TextEncoder; `strategy.MossTokenizer` -> a character tokenizer with the chat template;
`strategy.check_moss_text_encoder_path` -> no-op. The test resets accelerate's shared state and kohya's strategy
singletons between runs (one train() per process otherwise). Planted bugs caught: no pipeline counter restore, no RNG
restore, no epoch-end normalisation, no forced phase state, no early verify, no partial windows, no refresh refusal,
no ignored-option warning.

**Open.**
- No CUDA path ran: the pinned rings and streams under the real loop, the VRAM logs, sampling on the GPU, the
  start-up memory checks against the real machine. The smoke runs of design 7 decide the run shape.
- With DataLoader workers an epoch-boundary resume is exact for the optimizer state only (2.).
- The budget lines and resource checks always use the full-size MOSS numbers (the e2e test passes
  --moss_ft_skip_resource_check).

## Review fixes (after T1-T4; 18 findings, 2 of them duplicates)

Each finding was reproduced first (a failing test, or for --show_timesteps the CLI itself); the fixes below changed
behavior. Where a fix existed before its test could fail for the right reason, a planted bug was run instead: 14 planted
bugs (keep policy position / protection, zero_lr_warmup, the resumability guard, the step-4 scheduler build, unsloth
classes, workers, the disk plan x2, param-group restore in the state test and in the e2e, the skipped-window norm, the
hyperparameter comparison, the run-folder refusal) - every one caught.

**Changed behavior, by finding.**
1. LR after a resume, non-phase schedulers (numerics + integration findings, the same bug). `LambdaLR.load_state_dict`
   never touches the param groups and modes a / c do not load them through `Optimizer.load_state_dict`, so the first
   resumed window ran at the fresh scheduler's step-0 LR (0 with a warmup), and recursive (ExponentialLR) or
   group-keeping schedulers (the user's LoraEasyCustomOptimizer restart schedulers) continued from wrong values.
   - A state of modes a / c now has `param_groups.pt`: every param group's values but its parameters
     (`ft_args.param_group_values`; read with `weights_only=True`). Mode b's `optimizer.pt` already holds them.
   - `load_state` restores `lr`, `initial_lr` and every key that is not in `optimizer.defaults` (what a scheduler keeps
     there) before `lr_scheduler.load_state_dict`. The optimizer's own settings (betas, eps, weight_decay, ...) stay as
     this run configures them; a change is still warned about.
   - `hyperparameters()` (train_state.json and that warning) lists only the optimizer's own keys + "name": a restart
     scheduler's counters no longer raise a spurious "hyperparameters changed".
   - A state without the file (none exist) gets `get_last_lr()` in the groups, with a warning.
   - The format name stays `moss-ft-state/1` (COMPLETE.json lists the new file).
   - Tests: `test_moss_ft_state.py::test_resume_continues_a_non_phase_scheduler` (cosine + warmup, linear,
     constant_with_warmup, ExponentialLR, CosineAnnealingWarmRestarts: every window's LR and the pool bit-identical,
     no warning); `test_moss_ft_train_e2e.py::test_resume_with_a_non_phase_scheduler_continues_its_lr` (the script).
2. Skipped windows and stale clipping. A window that skipped parameters left them out of its norm: a fully skipped one
   stored 0.0, so the next stale coefficient was 1 (a spike right after a divergence went through unclipped), and a
   partly skipped one understated G. Now such a window reports `grad_norm` NaN (not logged; the warning gives the
   stepped part's norm) and does not replace `last_grad_norm`: the next window clips with the last clean window's norm,
   and that is what a state stores. Test: `test_moss_ft_host_adamw.py::test_skipped_window_keeps_the_previous_norm`.
3. `--zero_lr_warmup` (numerics + integration, the same bug). kohya's ZeroLRWarmupScheduler keeps its inner scheduler
   object (with the optimizer and every parameter) in its state_dict: scheduler.pt could be written but never read with
   `weights_only=True`. Refused in every mode (it was refused only with phases). Behind it a generic guard:
   `ft_args.check_scheduler_resumable` round-trips the scheduler's state and the param groups through `torch.save` /
   `torch.load(weights_only=True)` in the step-4 build (13.), so any other scheduler with objects in its state is refused
   before the run. Tests: `test_refusals[zero_lr_warmup]`, `test_unresumable_scheduler_is_refused`,
   `test_allowed_schedulers_build_and_are_resumable` (9 kohya schedulers).
4. First window at GA 1 (stale clipping, max_grad_norm > 0, no accumulator, no known norm). T2 ran it unclipped
   (deviation 3 above); the reviewer measured exp_avg_sq x2.92 after 40 windows and 21.5 % less movement. Design 3.1
   wants it exact, which needs the 18.93 GiB accumulator. It is now a "measure" window: its pieces only add to the sum
   of squares, nothing is stepped (no moment, step counter or upload), `WindowStats.clip_mode` "measure",
   `stepped_params` 0; the next windows clip with its G. The loop still counts it as an optimizer step (global step,
   LR scheduler): one batch without an update at the start and after a resume from a state without a norm. Also
   came_host (per-parameter step). Tests: `test_stale_clip_without_accumulator_measures_the_first_window` (window 1
   changes nothing; then bit-identical to torch AdamW with the same coefficients),
   `test_came_host_measures_the_first_window_without_accumulator`.
5. + 13. The LR scheduler at start-up. `get_scheduler_fix` reads `args.validation_split` for `first_cycle_max_steps`;
   the FT parser now defaults it to 0.0 (`set_defaults`, no argument: validation stays refused). The scheduler factory
   moved to `ft_args.make_lr_scheduler` (moss_train.py's delegates), and `ft_args.check_lr_scheduler` builds it in the
   dataset part of the validation (design 4.3 step 4, once max_train_steps is known, before any caching) on a stand-in
   SGD with the two named groups and steps it twice: kohya's refusals (a warmup for "constant", reduce_lr_on_plateau
   whose step needs a metric, cosine_with_min_lr without min_lr, piecewise_constant without rules, a wrong
   lr_scheduler_args key) now stop the run at step 4 with kohya's reason. Its [SCHEDULER] info lines are muted there
   (they come again at step 8). Tests: `test_scheduler_problems_surface_at_step_4`,
   `test_restart_scheduler_with_first_cycle_max_steps`.
6. --unsloth_offload_checkpointing pinned memory. `.to("cpu", non_blocking=True)` of a CUDA tensor allocates from
   torch's CachingHostAllocator, which (torch 2.14 headers: `PowerOf2Ceil`, thresholds SIZE_MAX) rounds every request
   up to a power of two and keeps freed blocks in per-class free lists that this run never empties. All 42 block inputs
   are alive in a micro-step, so every class the run reaches keeps 42 blocks. `ft_budget.unsloth_pinned_bytes` counts
   42 x the sum of the distinct classes over every bucket (bucket_reso and jitter buckets, per dataset batch size) and
   every b in 1..batch (a bucket's last batch is short): 18.4 GiB at 1024^2 batch 4 instead of design 2.2's 8.45 GiB
   (that row is superseded). On CUDA the log adds `host/pinned_cache_gib` (`torch.cuda.host_memory_stats()`
   allocated_bytes.current) so the smoke run can check the estimate. Test:
   `test_unsloth_pinned_checkpoints_follow_the_pinned_allocator`, `test_host_budget_counts_workers_and_unsloth_classes`.
7. Ring sizes. The same allocator rounds the rings: `--moss_ft_ring_mib` / `--moss_ft_upload_ring_mib` must be powers
   of two (design 2.2 says they are), and `ft_budget.host_budget` / `GradPipeline.pinned_bytes` count the rounded sizes
   (`ft_budget.pinned_alloc_bytes`). Tests: `test_refusals[ring_not_a_power_of_two]` (and upload),
   `test_pinned_allocations_round_up_to_a_power_of_two`, `test_pinned_bytes_count_the_allocator_rounding`.
8. DataLoader workers. Each worker re-imports torch and moss_train.py (0.81 GiB RSS on the CPU per the reviewer, a lower
   bound). Design 2.2's ~8 GiB process row now covers up to 4 workers (its "2-4"); each one above 4 adds 1 GiB
   (`PROCESS_WORKERS`, `WORKER_BYTES`; workers = min(max_data_loader_n_workers, cpu count) as the script starts them),
   and the validation warns above 4 (kohya's default 8), mentioning the per-epoch respawn without persistent workers.
   The design table still matches (its runs have <= 4). Tests: `test_dataloader_workers_add_host_ram`,
   `test_many_dataloader_workers_warn`.
10. Keep policy after a resume from an earlier state into the same folder. `apply_keep_policy(..., written=path)`
    (moss_train.py passes the state it just wrote): that state is never deleted, and its train_state.json gives the run's
    position - epoch states numbered past its completed epochs and step states past its global step belong to the
    abandoned run and are neither deleted nor counted until this run passes them (logged once). `complete_states` (and
    so every refusal hint) ranks by write time (COMPLETE.json's mtime), then step: the "newest" is the last written,
    not the abandoned run's higher step. Tests: `test_keep_policy_after_resuming_an_earlier_state`,
    `test_the_state_just_written_is_never_deleted`.
11. `--show_timesteps`. moss_train.py's command line is `main(argv)`: it prints the distribution
    (`moss_train_utils.show_timesteps`) and exits, as moss_train_network.py. Test:
    `test_show_timesteps_prints_and_does_not_train` (and the CLI was run before / after).
14. `--moss_ckpt_reuse_attn` log. `enable_elementwise_compile` and `set_ckpt_reuse` moved from start-up step 7 to
    `prepare_runtime` (step 1), as `prepare_moss_runtime` does: they are process-wide flags and nothing runs a DiT
    forward before step 7, so `validate_moss_args` (step 4) no longer logs "has no effect here". Test:
    `test_checkpoint_switches_are_on_before_the_validation`.
15. `--scale_resume` with `--moss_ft_lr_phases`: refused in `prepare_runtime` before `timestep_presets.resolve` (which
    added the extra epochs to max_train_epochs, giving two contradictory refusals). The phases give every epoch its LR,
    so a last phase is the replacement. Test: `test_scale_resume_with_lr_phases_is_refused_once_clearly`.
16. docs/moss_train.md: an exact resume needs 0 DataLoader workers for the data order (the optimizer state and LR are
    exact with workers too, now for any scheduler); the pinned phase state needs save_state and a whole-epoch phase end.
17. Disk. `ft_budget.SavePlan` / `save_plan(args, steps_per_epoch)`: the DiT files and states the run keeps at its peak,
    from max_train_steps, the cadence of moss_train.py (epoch saves never at the last epoch, step saves every n steps),
    kohya's removal rules (save_last_n_epochs counts saves; save_last_n_steps counts steps: n // every + 1 files), the
    keep policy and the pinned phase states, + the final DiT / the state being written. The start-up refusal compares
    the free space with it (design 4.5 said 2 states + one DiT save: the guide's example keeps 4 states and, without
    save_last_n_steps, 24 DiT files, 488 GB). The start-up lines show the counts. The guide's example now has
    `save_last_n_steps = 2000` (355.8 GB). Test: `test_disk_check_counts_what_the_run_keeps`.
18. `--resume <output_dir>`: `verify_state` says it is a folder of training states and names the newest one inside it,
    instead of "not finished (interrupted?)". Test: `test_resume_pointing_at_the_run_folder_names_its_states`.

**Design deviations added here** (beyond T1-T4's): design 4.3 step 7's `set_ckpt_reuse` / `enable_elementwise_compile`
run at step 1; the step-4 validation builds the scheduler; 3.1's "first window exact" is "measure" at GA 1; 2.2's unsloth
row and its process row (workers) are replaced by the estimators above; 4.5's disk refusal counts the SavePlan; 4.7's
state directory gains `param_groups.pt` and the keep policy its "never past the run's position" rule; `zero_lr_warmup`
is refused.

**Open for the GPU smoke run** (nothing of this ran on CUDA): `host/pinned_cache_gib` against the unsloth and ring
estimates (7.3 with --unsloth_offload_checkpointing, if used); host RSS per DataLoader worker with CUDA visible (each
spawned worker imports `library.attention`, which queries `torch.cuda.get_device_capability()` at import - whether that
creates a CUDA context, and so VRAM, per worker is unmeasured); the measure window's tail at GA 1 (no step, only the
sum of squares).

## T5 - live text encoder (`te_live_weights.py`, `te_live.py`, `te_live_feed.py`; plan `moss_ft_te_live_fp16_plan.md` part A)

Built 2026-11-17, CPU only. Part B (fp16 compute + loss scaler) followed the same day: T6.

- **Files.** `te_live_weights.HostTextEncoderWeights` (host form: copy / pinned / mmap; aligned `LayerLayout` per layer,
  every tensor at a multiple of 256 B; embedding rows gathered from the mapped file), `te_live.LiveTextEncoder` (the
  layer-major streamed pass), `te_live_feed.TextEncodeAhead` (encode-ahead over the DataLoader, the cached batch form).
  `qwen35.load_moss_text_encoder` gained `config=` (tests: a tiny config skips the 9B header check, the prefix is only
  detected: `detect_text_encoder_prefix`); `is_language_model_key` is the one key filter of both loaders.
- **Deviations from the plan.**
  - The tiny test encoder has 8 layers (plan); the e2e harness reads the same tiny file in both paths (the cached
    provider's loader is `load_moss_text_encoder(config=TINY_TE)`, the live one patches `te_live_weights.DEFAULT_CONFIG`
    and the 9B check), so the identity runs compare one set of weights.
  - The file views go through numpy (`np.frombuffer` + `torch.from_numpy`), not `torch.frombuffer`: numpy keeps the
    mmap's buffer exported, so `mmap.close()` raises `BufferError` under a live view instead of unmapping it.
  - The staging entry is `staging_entry_bytes` = the largest layer rounded up to a power of two (512 MiB for the 9B)
    without `ft_host.PinnedRing`: one `torch.empty(R x entry, pin_memory=True)` kept for the run, the stager a
    one-thread `ThreadPoolExecutor` whose initializer sets its intra-op threads. On the CPU the stager runs only when
    a test forces `staging=True` (the scheduling, failure propagation and unbinding are tested that way); CUDA streams
    and events are not exercised on the CPU (smoke item G1 / G2).
  - Caption checks moved out of `validate_moss_args` for moss_train.py: it passes `live_te` and `check_captions=False`;
    `ft_args.check_captions` (inside the dataset step) refuses on the cached path with "or use --moss_te_live" and the
    subset, warns about literal `|||` on both paths and logs a per-subset summary in live mode. The LoRA trainer's call
    is unchanged.
  - `ft_budget`: the `TE_*` constants (24 GatedDeltaNet layers of 436,814,592 B, 8 full-attention layers of
    419,447,808 B, 12.889 GiB in all) are pinned against the meta model by
    `test_real_encoder_layout_matches_the_budget_constants`. The pass time uses max(stage, H2D, compute + 32 ms per row
    of launch overhead) = ~0.06 s/image at 32 rows of 300 tokens (an estimate; G2 measures it) and is added to the step.
  - Not built: a padded batched layer forward (faster, not bit-identical to the cache), peak-VRAM stats per pass (the
    trainer's peak counters are not reset by the pass).
- **Tests.** `test_moss_ft_te_live_{weights,engine,feed,captions,args,budget,e2e}.py` + `_moss_te_tiny.py`. The e2e file
  checks a cached run against live runs with ahead 1 and 3 (-> 4 at GA 2): every step's loss, grad norm, clip
  coefficient and the saved DiT bit for bit; a live run with shuffle / wildcard lines / dropouts trains and resumes
  mid-epoch; the sample prompts through the live encoder equal the provider path's.

## T6 - fp16 compute with dynamic loss scaling (`ft_scaler.py`, `ft_host.py`; plan `moss_ft_te_live_fp16_plan.md` part B)

Built 2026-11-17, CPU only (accelerate refuses fp16 mixed precision on the CPU; smoke item G3 runs it on the GPU).

- **Files.** `library/moss/ft_scaler.py` (new): `DynamicLossScaler` (power-of-two scale, x0.5 on an overflow window,
  x2 after `growth_interval` clean windows, `min_scale` / `max_scale` 2^24 bounds, `growth_interval` 0 = never grows),
  `check(max_skipped_steps)`, `state_dict` / `load_state_dict`, `make(args)`, `restore(scaler, saved)`.
- **Where the scale enters.** `moss_train._train_loop`: `(loss * scaler.scale).backward()` (the loss already divided by
  n_micro; float64, exact), the window's `StepContext.inv_grad_scale = 1 / scale`, `scaler.update(stats,
  isfinite(window_loss))` after the drain, `scaler.check` instead of `pipeline.check_skip_streak` with fp16.
  `ft_host`: `_Window.coef = clip x inv_scale` (`_combine`, an fp32 0-dim tensor; exact as inv_scale is a power of two),
  the same in `step_from_accumulator`; `grad_norm`, the exact-clip total and `last_grad_norm` are multiplied by
  inv_scale where they are used. `_Window.clip` keeps the clip coefficient alone so `WindowStats.clip_coef` stays the
  unscaled one. `WindowStats` gained `loss_scale` and `overflow` (defaults 1.0 / False, so `FusedAdafactorSteps` is
  unchanged). `begin_window` refuses an `inv_grad_scale` that is not a power of two.
- **Overflow policy.** `partial` (default): the existing per-slot non-finite skip; the slots that step get exactly the
  unscaled update. `window` (`GradPipeline(overflow_policy="window")`, `--moss_ft_overflow_policy window`): in a window
  with a poisoned slot, `step_from_accumulator` is not called, every slot with a gradient counts as skipped. Allowed
  only when every window steps after the backward (exact clipping with `max_grad_norm > 0`, or `overlap=False`).
  With fp16 an overflow window logs an INFO line (the scaler's search), not the warning of a bf16 skip.
- **Deviations from the plan.** `--moss_ft_overflow_policy` default and the loss-scale flags are refused (not ignored)
  when changed without fp16. `load_state_dict` clamps the saved scale into this run's bounds and the clean-window count
  below this run's `growth_interval`. `fp16_range_report` also refuses a frozen fp16 parameter that is not finite after
  the read, and is called for fp16 compute before `check_compute_copies`. The budget is unchanged (fp16 costs what
  bf16 costs); the GPU weights line names the compute dtype. Accelerate's own GradScaler is never used (moss_train never
  calls `accelerator.backward`; pinned by a source test). `models.py` keeps the text refiner's flash attention in bf16
  under the fp32 stream - the stream path, unchanged by fp16.
- **State.** `train_state.json["loss_scaler"]` = the scaler's `state_dict` or null (`STATE_FORMAT` unchanged);
  `ResumeInfo.loss_scaler`; fp16 -> fp16 exact, bf16 state -> fp16 run starts at `moss_ft_loss_scale`, fp16 state ->
  bf16 run ignores it (log lines).
- **Tests.** `test_moss_ft_loss_scaler.py` (update rules, bounds, static scale, divergence check, state round trip,
  restore cases), `test_moss_ft_fp16_pipeline.py` (scaled == unscaled bit for bit for adamw_host / came_host, GA 1 / 3,
  clip none / stale / exact, inline / threaded, with the scale growing during the run; partial and window overflow; the
  production fp16 + fp32-stream model under fp16 autocast keeps compute copies == master rounded),
  `test_moss_ft_fp16_args.py` (flags, refusals, messages, `fp16_range_report`, the GradScaler source check),
  `test_moss_ft_train_fp16_e2e.py` (moss_train with a forced scaler under fp32 compute == the unscaled run bit for bit, the
  scaler in every state, exact resume, cross resumes). `test_moss_ft_args.py`'s fp16 case now expects the stream
  refusal.

## T7 - review fixes of T5 / T6 (2026-11-17, CPU only)

Numerics:
- **A non-finite loss keeps the scale** (`ft_scaler.update`): the loss is formed (float64) before it is multiplied by
  S, so a non-finite window loss is a forward overflow of fp16 activations; halving S cannot help and only pushes the
  gradients towards underflow (growth_interval clean windows per lost factor 2). It returns KEEP, leaves the
  clean-window count as it is, and counts in `overflow_streak` (the divergence check).
- **Exact clipping skips the whole window under `partial`** (`ft_host._finish_window`): a window whose plan is "exact"
  (clip_mode exact, or stale without a previous norm) steps nothing in the backward, so with a poisoned slot every slot
  is skipped instead of clipping the finite slots with their norm alone (the true norm is not finite). bf16 too.
- **`growth_interval` 0 = never grows** (help, refusal text, docs): an overflow still halves the scale (it was
  documented as "static").
- **fp16 range on the upload** (`ft_host._check_fp16_copy`): every fp16 upload entry gets one `aminmax` after the cast
  from the master (in `_step_range` and `_upload_all`); a non-finite value raises `HostPipelineError` naming the
  parameter, the element and |w| (the ring entry is released first). `fp16_range_report` still checks the load.
- **The loss EMA skips non-finite windows** (`moss_train`: `loss_recorder.add` only for a finite window loss), so
  `avr_loss` / `loss/epoch` stay finite after an fp16 skip.
- **Resume reset** (`DynamicLossScaler.load_state_dict`): when the saved `init_scale` differs from this run's
  `--moss_ft_loss_scale`, the scale restarts at the new value with fresh clean / streak counts (the overflow total
  stays). The same value continues exactly, as before.

Live text encoder / budget:
- **mmap honesty** (`ft_budget`): the refusal hint says mmap trades RAM for disk reads; `_te_mmap_warning` warns when
  available < required + 12.9 GiB and prices that case at `TE_DISK_BYTES_PER_S` (2 GB/s, estimate:
  `te_seconds_per_pass(..., from_disk=True)`, ~6.9 s per pass of 32 captions vs ~1.9 s).
- **Page-locked total** (`HostBudget.pinned`, `te_host`): rings + unsloth checkpoint inputs + TE staging + the TE layers
  in `pinned` mode, a report line, and `_pinned_warning` on Windows past `PINNED_WARN_FRACTION` (50 %) of physical RAM.
- **TE timing** (`te_live_feed.TextEncodeAhead.take_pass` / `s_per_image` / `device_seconds`, `moss_train.window_logs`):
  `te/s_per_image` is the run's pass seconds per encoded caption (it divided one pass by one window's images, ~ahead /
  GA times too high); CUDA events around each pass give GPU time, read lazily once completed; `te/pass_images` added.
  `LiveTextEncoder.stats` gains `reserved_before` / `reserved_after` on CUDA (G2).
- **Caption warnings** (`captions.separator_warnings`): multi-line caption files with `enable_wildcard` off (all lines
  joined into one prompt), and `{...}` in captions with `enable_wildcard` on.
- **DataLoader prefetch** (`moss_train.live_prefetch_factor`): with workers in live mode `prefetch_factor =
  max(2, ceil(ahead / workers))`, so a pass does not load ~ahead - 2 x workers batches on its critical path.
- **Device slots** (`te_live._run_layers`): R separate slot tensors instead of one R x 417 MiB block.
- **Staging order across passes**: an event recorded on the copy stream after a pass's last H2D copy
  (`_uploads_done`) is what the next pass's first R `_stage` calls wait for (it relied on implicit syncs before).
- **Example config**: `moss_te_live_ahead = 8` at batch 4 (32 captions per pass; `ahead` counts micro-batches).

Tests added: `test_moss_ft_loss_scaler` (non-finite loss keeps the scale, growth 0, init_scale reset),
`test_moss_ft_fp16_pipeline` (exact clipping + partial skips the window, fp16 and unscaled; a master past the fp16
range raises at GA 1 / 2, inline / threaded), `test_moss_ft_te_live_budget` (mmap disk pricing, page-locked sum and
warning), `test_moss_ft_te_live_args` (mmap hint and warning), `test_moss_ft_te_live_captions` (multi-line and brace
warnings), `test_moss_ft_te_live_feed` (per-caption timing, window_logs, prefetch factor), `test_moss_ft_te_live_engine`
(staging waits for the previous pass's uploads), `test_moss_ft_te_live_e2e` (live run with one spawned DataLoader
worker == the cached run). The CUDA-only paths (copy stream, events, pinned staging, cudaHostRegister) still have no
CPU test; they are G1 / G2.

## T8 - Carousel: rotating blocks (`ft_carousel_units.py`, `ft_carousel_optim.py`, `ft_carousel.py`; plan `moss_ft_carousel_plan.md`)

Tests (CPU only): `tests/test_moss_ft_carousel_{units,budget,args,equivalence,switch,backward,state,train_e2e}.py`
(helpers `tests/_moss_ft_carousel_common.py`). GPU smoke C1-C4 not run (needs the user's go-ahead).

Deviations from the plan and details it left open:
- **Name.** `ft_budget.CAROUSEL` is the only place the name lives: flags `--moss_<name>_*` (`ft_args.carousel_dest`),
  the mode string, log keys (`<name>/...`), the metadata key `ss_moss_ft_<name>`, the log prefix (`<Name>`).
  `test_every_name_derives_from_the_constant` rebuilds the parser with another value and scans the string constants of
  `library/moss/ft_*.py` and `moss_train.py` for a hard-coded name. State files: `rotation.json`,
  `rotation_optimizer.safetensors`, format `moss-ft-rotation/1`; `ft_state.load_state` accepts a state of another
  mode name when it has `rotation.json` (a renamed mode still resumes).
- **Optimizer state is not in flat buffers.** The plan had per-unit flat device buffers for the states too; the
  optimizers allocate their own state lazily (torch AdamW, bitsandbytes, CarouselCAME) and the codec
  (`ft_carousel_optim.state_to_host / state_to_device`) moves the state dict of each tensor. Masters and accumulators
  are flat per group (unit or always group); a group's pool range is copied in one piece when its slots are
  consecutive (every unit), per slot otherwise (the always group: stem + head).
- **Handles.** One persistent `nn.Parameter` per slot (`handles[i]`) is in the optimizer's two groups for the whole
  run; its `.data` is the device master view while active and an empty tensor otherwise. An inactive handle has no
  `.grad`, so torch AdamW / bitsandbytes / CarouselCAME skip it: its state, step count and weight decay are untouched
  (exactly the "masked plain FT" reference of the tests).
- **Kept states on the host** are allocated once per tensor and reused at every later deactivation (the host budget
  counts every unit visited). Saving reads the active / always states from the device one tensor at a time
  (producers), so a save needs no host copy of them; only the active masters are flushed into the pool
  (`flush_to_host`, called by `moss_train.Run` before every DiT / state save).
- **Activation is lazy:** `attach()` registers the hooks and switches every compute copy off; the first `begin_window`
  activates `select(step // period)`. A resume reads its states (`state_reader`) before that, so nothing stale is on
  the device when `load_state` replaces the masters and compute copies.
- **Resume exactness** holds for epoch-end states (incl. a switch boundary): CPU-tested end to end. A mid-epoch state
  reshuffles that epoch (kohya's behaviour, as in the other modes) - the active set and period position still follow
  the saved step.
- **Truncation** uses instance `forward` wrappers (`ft_carousel_units.apply_grad_floor`): `torch.no_grad()` below the
  floor and for the text embedder when the text path is not trained; under unsloth offload the floor block's input is
  made a grad-requiring leaf. `ft_loss` no longer marks the noisy input grad-requiring in this mode. A deep copy of a
  model with the wrappers would keep calling the original module: tests clone before attaching.
- **Overflow / non-finite gradients** always skip the whole step (`torch._amp_foreach_non_finite_check_and_unscale_`,
  one host sync per step), also without fp16; `--moss_ft_max_skipped_steps` counts these steps.
- **K** comes from `--moss_carousel_active`, else from the start-up report (`args._moss_rotation_k`, set by
  `ft_budget.carousel_budget_from_args`), else the budget at 1024² b1; it is clamped to the run's units.
- **adamw8bit** has no CPU kernels: its codec is tested through a stand-in optimizer with bitsandbytes' exact state
  layout (uint8 states, absmax, the shared qmap objects of `name2qmap`, an int step); smoke C1 checks the real one.
- **OCGOptV2** is not included (8 B/param state, same as AdamW; no memory gain).

### T8 review fixes (2026-11-17 numerics + integration reviews; `tests/test_moss_ft_carousel_review.py`)

- **Selection fairness.** `Selector` random draws each unit exactly once per cycle, also across a boundary: the new
  cycle's order leads with k - r units that are neither carried nor in the previous set, then k uncarried ones (the
  old stream consumed and skipped a carry unit drawn twice, and could keep a unit active two periods in a row). With
  K <= U/2 no unit is active in two consecutive sets.
- **`--moss_carousel_always rotate_stem`** (new, default stays `small`): the stem tensors (`STEM_PREFIXES`, 79.8M) are
  unit `stem` (index 0, `discover_units(..., stem_unit=True)`), the head is always trained. `grad_floor` needs no
  change: an inactive stem leaves only joint / refiner names, so the floor is the lowest active block. The budget
  (`carousel_layout(frozen, stem_unit)`) counts 45 units; the stem's state is a kept state.
- **No switch after the last step:** `after_step(global_step, max_steps)`; `moss_train` passes `max_train_steps`.
- **Seed on resume:** `CarouselEngine(seed_given=...)`; without `--moss_carousel_seed` `state_reader` adopts the saved
  seed and rebuilds the selector before activation (`args_set_seed` draws a new `--seed` per run when it is unset).
- **Log keys:** `<name>/lowest_active_block` is now the lowest active joint block (num_layers when none);
  the gradient floor moved to `<name>/grad_floor`.
- **CAME `reset` + `bias_correction=False` is refused** in `parse_optimizer_args` (the t = 1 amplification would recur
  at every switch).
- **Measured activation model** (`ft_budget.measured_activations`, `CAROUSEL_MEASURED_ACT_*`): 2.0 GiB + 620,000 B per
  joint token, the 11-17 smoke peaks over the compute copies (b1 4.03, b4 11.63 GiB; adamw_host, fp16 + fp32 stream,
  plain checkpointing, live TE) fitted linearly with +0.5 GiB. Used only under those conditions and only when below the
  design estimate (b1 keeps the design estimate; b4 drops from 15.4 to 12.3 GiB). `CarouselBudget.activation_model`
  says which; the K < 1 refusal names the bucket / batch and the ways out (`--moss_carousel_active`, unsloth, b1 + GA).
- **Rejected / deferred:** the fp32 compute copy as its own master for tensors already fp32 under the fp32 stream
  (0.37 GiB always + 0.42 GiB per active refiner: below one unit, and it would bypass the I1 check of the switch);
  skipping the fp32 accumulator at GA 1 without clipping (fp16 joint-Linear gradients still need it for the exact
  unscale and the fp32 optimizer). Doc-only: weight decay / clip strength under rotation, `host_pinned masters` gain,
  the live TE cost (example config now b1, GA 4, ahead 32).
