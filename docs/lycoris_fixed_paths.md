# LyCORIS fixed paths (default) and `--lycoris_stock`

For `network_module = "lycoris.kohya"` every trainer of this fork built on `train_network.py`'s `NetworkTrainer`
(Anima incl. `anima_train_addift.py`, MOSS, SDXL, FLUX, SD3, Lumina, HunyuanImage) runs LyCORIS through
`library/lycoris_algo_fixes.py`: in-memory fixes for algorithms that are broken on these trainers, faster forwards that
compute the same function, and saves that Forge reads correctly. LyCORIS in site-packages is not modified.

Scripts that build or merge LyCORIS networks themselves keep stock LyCORIS: the LECO trainers (`anima_train_leco.py`,
`sdxl_train_leco.py`, `train_leco.py`), `gen_img*.py` and the merge tools, and `train_network.py`'s own
`--base_weights` merge (it runs before the patches are applied).

| setting | effect |
|---|---|
| nothing (default) | fixed paths on for LyCORIS networks |
| `lycoris_stock = true` / `--lycoris_stock` | LyCORIS exactly as installed; `library/lycoris_algo_fixes.py` is not imported; `algo=tsm-lokr` is refused |
| `lycoris_algo_fixes = true` / `--lycoris_algo_fixes` | the old opt-in switch: accepted for old configs, changes nothing (one log line) |
| both | `--lycoris_stock` wins (warning) |

Other network modules (`networks.lora`, ...) are never touched. `--lycoris_stock` is not `--anima_stock_lycoris`
(that one only turns off the fast LoKr+DoRA forward of `library/lycoris_fast.py`).

Saved files of the fixed paths carry three extra metadata keys: `ss_lycoris_algo_fixes = "1"`, `ss_saved_format`
(`lora`, `loha`, `lokr`, `oft`, `boft`, `glora`, `diff`, `tsm-native`, `tsm-lokr-native`, `ia3-native`) and
`ss_lycoris_algo`.

## What changes per algo

The module header of `library/lycoris_algo_fixes.py` lists, per algo, what stock LyCORIS does wrong on these trainers
(with the LyCORIS source lines) and what the fixed path does; every patched function is listed in
`apply_module_patches()` and logged at start-up. In short:

- `lokr` (the production setting): training at multiplier 1 is bit-identical to stock (the fast LoKr+DoRA forward of
  `library/lycoris_fast.py` still does every multiplier-1 call); only DoRA at multiplier != 1 (the teacher / target
  passes of differential output preservation, iLECO and ADDifT) now interpolates `W0 + m (W1 - W0)` like Forge instead
  of keeping the whole update. Sample images during training always use multiplier 1.
- `lora`, `locon`, `ortholora`, `dylora`, `gora`, `ralora`, `lora2`, `tlora`, `loha`, `abba`, `diag-oft`, `boft`,
  `tsm`, `glora`: faster forwards (no dense delta / merged weight per call where that is cheaper), same math.
- `ralora`, `gora`, `lora2`, `tlora`, `abba`, `dylora`, `ortholora`, PiSSA, `full`, `glora`, `ia3`, `tsm`: training,
  resume and save fixes (see the module header).
- `dora_wd=True` on an algo whose LyCORIS module ignores it (diag-oft, boft, tlora, tsm, dylora, glora, ia3, full)
  stops with an error instead of training without DoRA.
- `full`: the trained weights are fp32 master weights (stock kept them in bf16 and lost small AdamW steps): with AdamW
  that is 16 bytes per adapted base parameter on the GPU, twice the stock footprint (the size is logged at start-up).
- `algo=tsm-lokr`: see `docs/tsm_lokr.md`.
- `algo=tsm` (2026-10-25): the training forward is the LoRA family's fused one (base GEMM with the active experts'
  delta as its C input, hand-written backward) instead of `org_forward` + a separate delta; same math, the output
  rounded once (under bf16 autocast at least as close to fp64 as the split forward; stage 2 ~2x closer). Measured on
  the GPU (Anima 2.9B, 4-tier batch 1 x accumulation 8, 16 steps) it is speed-neutral against the split forward
  (12.6 vs 12.5 s/step; stock LyCORIS 17.6), so expect no step-time change from it. Conv modules, RamTorch base weights (`use_ramtorch`), delta dropout (bypass mode / stage 2 while
  training), `--fp8_base`, a base weight that needs its gradient and a dtype mix outside autocast keep the split
  forward, and `network_args` `tsm_fused=False` switches it off everywhere. Stage 2 with `tsm_router_input=bottleneck`
  computes the core bottleneck once more (no grad) for the gate pooling. The expert projections `x A^T`, `mid B^T` and
  the expert gradients can run in fp32 with autocast off (`tsm_lowrank_fp32`; not given = stage 2 only, since on the
  GPU it cost +11% step time and +3 GiB for a stage-1 gain AdamW normalises away; `True` / `False` force it): the base
  GEMM and the input gradient stay bf16, the delta is rounded once into the base GEMM. `tsm_expert_fn=True` (opt-in,
  default off) uses one autograd Function per call that takes the expert weights directly (`_TSMExpertLinear`); on the
  GPU it held ~1.3 GiB more at the 2048-tier peak and was not faster, so the default stays the factor path.
  A chunk whose samples have different timesteps (batch > 1 with per-sample timesteps) runs as ONE masked call per
  module (`_TSMMaskedLinear`): A / B hold every expert active anywhere in the chunk and a per-sample mask over their
  blocks (the router gate on the context block in stage 2) zeroes the other experts' columns of the bottleneck, so the
  per-module cost no longer grows with the number of distinct timesteps in the chunk (the previous per-group loop made
  a 4-row chunk cost 4 calls per module, which is why micro-batching bought TSM nothing). Same math, same gradients, an
  expert no sample uses stays out of the graph (grad None); `tsm_mixed=False` restores the per-group loop.

Known limitation: GoRA / RaLoRA / LoRA² with DoRA on the factored path (`_LowRankDoRA`, RaLoRA blocks): a `dora_scale`
entry that is exactly 0 on a nonzero base row (LyCORIS never initialises one, it starts at `||W0_i||`; only a file
trained on another base can bring one) gets that entry's gradient without its base term; everything else stays exact.

## Under `--anima_block_compile`

`anima_train_network.py --anima_block_compile` (`library/anima_block_compile.py`, `docs/anima_train_network.md`) traces
each DiT Block with the fixed paths' forwards inside the compiled graph. Three things change for LyCORIS networks, none
of them for the eager math:

- after `apply_to`, every adapted Linear's instance `forward` (LyCORIS binds `org_module.forward = module.forward`) is
  re-bound to a closure over the module (`rebind_forwards`): dynamo 2.7 cannot read `org_module[0]` through a bound
  method (every adapted Linear would be a graph break); the closure carries `__self__`, so the sample-time merged LoKr
  weight cache of `library/lycoris_fast.py` still finds its modules. Stock LyCORIS forwards are not re-bound, which is
  why `--lycoris_stock` is refused together with block compile.
- `set_timestep` hoists the TSM step data onto the modules (`_hoist_tsm_step`): the expert tuple (`_hx_experts`), in
  stage 2 the embedding row as an index tensor (`_hx_tidx`), for `tsm-lokr` the expert indices (`_hx_idx`). While
  tracing, `_tsm_experts`, `_get_timestep` (patched: a constant instead of the global integer timestep, so the graph
  guards on the expert choice and not on the timestep value), `_tsm_compute_gates` (`index_select` on the hoisted row)
  and `_tsm_scale` (the 0-d scale buffer handed to the graph instead of the data_ptr cache) read them: one expert choice
  is one set of graphs, and a timestep inside the same expert interval never recompiles. Outside a compiled graph the
  hoisted data is ignored.
- a chunk whose rows carry different timesteps (`mixed_timestep_chunk()`: batch > 1 with per-sample timesteps) runs the
  eager masked path (`_TSMMaskedLinear`): its expert union changes with the chunk, so a compiled graph would be re-traced
  per combination. With `batch_size 1` (also under `--step_rows`) every chunk has one timestep and runs compiled.

Covered by `tests/test_anima_block_compile.py` (lokr+DoRA, lora, tsm stage 1 / 2, tsm-lokr: one graph per Block, no
graph break, compiled forward + backward identical to eager with and without checkpointing - plain lora's adaLN
up-weight gradients within a bf16 ulp or two - and no recompile inside an expert interval or across the bucket resolutions of
the union band).
