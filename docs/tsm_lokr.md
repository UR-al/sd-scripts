# TSM-LoKr: timestep experts with LoKr (+DoRA)

`network_args = ["algo=tsm-lokr", ...]` with `network_module = "lycoris.kohya"` (the LyCORIS fixed paths are on by
default; `algo=tsm-lokr` is refused with `lycoris_stock = true`, the original-LyCORIS switch).
TimeStep Master routing (LyCORIS `algo=tsm`: core experts per noise interval plus context experts, two stages) where
every expert is a LoKr (Kronecker) factor pair instead of a low-rank LoRA, with one DoRA magnitude per layer shared by
all experts. `algo=tsm` and `algo=lokr` are unchanged; nothing here runs unless `algo=tsm-lokr` is selected.
Code: `library/tsm_lokr.py` (module, math, metadata), `library/tsm_offload.py` (expert offload), registration and
trainer hooks in `library/lycoris_algo_fixes.py`.

## Example (Anima-2.9B, production LoKr settings)

```toml
network_module = "lycoris.kohya"
network_dim = 10000      # full_matrix LoKr: scale 1, as with algo=lokr
network_alpha = 10000
network_train_unet_only = true
optimizer_type = "AdamW" # torch AdamW, optimizer_args fused=True
network_args = [
  "algo=tsm-lokr", "preset=anima-exp", "tsm_n_scales=8,1", "tsm_stage=1",
  "factor=4", "full_matrix=True", "decompose_both=False", "dora_wd=True", "wd_on_output=True",
  "tsm_offload_states=True", "tsm_resident_cores=8", "tsm_grouped_accumulation=False",
]
```

Stage 2 (router only): `tsm_stage=2` and `--network_weights <stage-1 file>`. With `tsm_gate_init=1.0` stage 2 starts
exactly at the stage-1 model; the default 0.0 is LyCORIS' behaviour (the context expert is switched off at the first
stage-2 step and the router learns it back).

## Settings

| network_arg | default | meaning |
|---|---|---|
| `tsm_n_scales` | `8,1` | experts per scale; scale 0 = core (ungated), the others = context (gated in stage 2) |
| `tsm_num_timesteps` | 1000 | T; expert of scale j for timestep t: `clamp(ceil(t/T*n_j),1,n_j)-1` |
| `tsm_stage` | 1 | 1: experts + DoRA magnitude train; 2: router only |
| `tsm_gate_init` | 0.0 | stage 2: router bias when the loaded router is all zero |
| `factor` / `tsm_core_factor` / `tsm_context_factor` | -1 | LoKr factor (all / core / context experts) |
| `full_matrix`, `decompose_both` | False | LyCORIS LoKr rules |
| `dora_wd` | False | DoRA magnitude (output axis; `wd_on_output=False` is refused) |
| `tsm_offload_states` | True | core experts' AdamW states in pinned RAM between optimizer steps |
| `tsm_resident_cores` | n_core | core experts whose fp32 weights stay on the GPU (least recently used are swapped out) |
| `tsm_grouped_accumulation` | False | all samples of one optimizer step from one core interval (Anima trainer only) |

Nothing is switched on, off or changed at runtime: no memory guard, no fallback. If memory runs out the run stops
with an out-of-memory error and the settings are changed by hand. Inconsistent settings stop at start-up with an
error (English / Korean), e.g. `tsm_resident_cores` below what one optimizer step can touch
(`batch x gradient_accumulation_steps`, unless grouped), offload with a non-AdamW optimizer, `--deepspeed`,
`--use_ramtorch_network`, `--scale_weight_norms`, `--dim_from_weights`, dropouts, `bypass_mode`, `use_scalar`.
`tsm_resident_cores` below n_core and `tsm_grouped_accumulation` follow `optimizer.step()` (the swapped-out cores are
written back, and a grouped window ends, in its post-hook), so they are refused with an optimizer that updates through
`step_func` (schedule-free plus) and, for swapped-out cores, with schedule-free `train()`/`eval()` (which rewrite every
parameter outside the step); `tsm_offload_states` is refused with `amsgrad=True` (it does not move `max_exp_avg_sq`).

## Memory (Anima-2.9B, preset anima-exp, factor 4 full matrix + DoRA)

One expert: 173.0 M parameters (0.645 GiB fp32), AdamW state 1.29 GiB per expert. 9 experts (8 core + 1 context).

Measured 2026-10-24 (RTX 5090, `config_anima29.toml` with the network args above, tb_1018 dataset, 30 optimizer steps
of batch 1 x accumulation 4, stage 1; CUDA numbers from `TSM_LOKR_MEMLOG`, nvidia-smi = whole GPU incl. ~0.7 GB desktop):

| run | s/step | CUDA allocated peak (GiB) | CUDA reserved peak (GiB) | nvidia-smi peak (GB) |
|---|---|---|---|---|
| plain `algo=lokr` (same args, for reference) | 5.69 | - | - | 19.0 |
| `tsm-lokr` default (`tsm_resident_cores=8`) | 6.13 | 22.2 | 28.5 | 30.5 |
| `tsm_resident_cores=4` | 6.25 | 19.6 | 26.4 | 28.3 |
| `tsm_grouped_accumulation=True` + `tsm_resident_cores=1` | 6.05 | 17.0 | 19.5 | 21.2 |

The reserved - allocated gap is the caching allocator's cache (block sizes differ between the forward and the step);
the training process reuses it, but other programs see the whole reserved amount as used. `expandable_segments` is not
supported on Windows (no change measured). The table below is the earlier estimate, kept for the per-item breakdown.
Estimates for the 2048 bucket, batch 1 x accumulation 4, gradient checkpointing.

| what | GPU (GiB) |
|---|---|
| CUDA context + workspaces | ~1.0 |
| DiT bf16 | 5.4 |
| expert weights, 9 resident | 5.8 |
| grads: context + up to 4 cores | 3.2 |
| AdamW state, context expert | 1.3 |
| activations, one 2048^2 sample | ~7.6 |
| allocator slack | ~1.3 |
| **peak during backward** | **~25.6 (~27.5 GB in nvidia-smi with the desktop)** |
| at the optimizer step | ~23.2 (core states of the step's <= 4 cores uploaded, activations freed) |

| setting change | GPU | other cost |
|---|---|---|
| `tsm_offload_states=False` | +10.3 GiB (all core states resident) | no state copies |
| `tsm_resident_cores=K` | -0.645 GiB per core below 8 | swap ~28 ms per miss |
| `tsm_grouped_accumulation=True` + `tsm_resident_cores=1` | about -6.4 GiB | each core gets ~3.2-3.6x fewer AdamW updates per step count (more epochs) |
| `tsm_core_factor=8` | peak -5.8 GiB (weights -3.9, grads -1.9); step-time states -3.9 GiB | core experts of 43 M instead of 173 M params |
| `use_ramtorch=true` | -4.7 GiB | ~+5 % s/step |
| `unsloth_offload_checkpointing=true` | -5.0 GiB | ~10 GiB PCIe per micro-step |
| 1536 bucket instead of 2048 | -3.3 GiB | |
| batch 2 x accumulation 2 | +7.6 GiB | |

Pinned RAM: core states 8 x 1.29 = 10.3 GiB (+ up to 0.5 GiB slab slack); swapped-out core weights +0.645 GiB each.
Disk: one fp16 LoRA file ~2.9 GiB; one `--save_state` ~17.4 GiB (weights + AdamW).

How the step-time states take the activations' place: the optimizer step pre-hook copies each used core's states into
GPU buffers of one 64 MiB host slab each (pieces the caching allocator can cut out of the freed activation blocks; one
1.29 GiB buffer per core would need new segments next to them), and the next micro-step waits for the write-backs of
the post-hook before its forward allocates (~0.2 s per optimizer step: the copy back no longer overlaps the next
forward). A swapped-out core is uploaded only after the evicted core's write-back finished. `TSM_LOKR_MEMLOG=1` in the
environment logs allocated / reserved / peak CUDA memory before each forward and at each optimizer step (GPU check);
any other value is a file that gets one JSON line per entry.

## Grouped accumulation

With `tsm_grouped_accumulation=True` the first sample of each optimizer step is drawn by the trainer's own timestep
sampler; that fixes the step's core interval, and the other samples are redrawn until they fall in it. Each sample's
timestep distribution is unchanged, but each core expert is updated on fewer optimizer steps (with more samples per
update): train longer for the same core training. Not combinable with antithetic / stratified / QMC / adaptive timestep
sampling, iLECO or ADDifT.

## Saved file

Keys per layer: `experts.<scale>.<i>.lokr_w1`, `.lokr_w2` (or `_a`/`_b` when low-rank), `alpha`, `dora_scale`,
`router_fc.weight`, `router_fc.bias`, `timestep_embed.weight`. Metadata: `ss_lycoris_algo=tsm-lokr`,
`ss_saved_format=tsm-lokr-native`, `ss_tsm_expert=lokr`, `ss_tsm_lokr` (JSON: version, n_scales, num_timesteps, stage,
factors, full_matrix, decompose_both, dora, timestep_dtype, timestep_scale, pooling, multiplier, gate_init). Strength
mu is applied as `W0 + mu * (W' - W0)` (Forge's blend), in training and in the runtime.
