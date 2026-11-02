"""CPU tests for --anima_stream_fp32 (no GPU, no model files).

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe tests/test_anima_stream_fp32.py   (or with pytest if installed)
CPU autocast stands in for accelerate's bf16 autocast: the same ops are cast, on the same rule.
"""
import argparse
import types

import torch
import torch._dynamo
from torch._dynamo.backends.registry import lookup_backend

from anima_train_network import AnimaNetworkTrainer, validate_stream_args
from library import anima_block_compile, anima_models, anima_train_utils, attention, lycoris_args, lycoris_fast
from library import lycoris_algo_fixes as laf

TOY = dict(max_img_h=32, max_img_w=32, max_frames=1, in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
           model_channels=64, num_blocks=3, num_heads=2, mlp_ratio=4.0, crossattn_emb_channels=32, use_adaln_lora=True,
           adaln_lora_dim=16, use_llm_adapter=False, attn_mode="torch", pos_emb_cls="rope3d")
BLOCK_LINEARS = 10  # q, k, v, output of both attentions, the two MLP layers
LOKR = ["factor=4", "full_matrix=True", "decompose_both=False"]
DORA = ["dora_wd=True", "wd_on_output=True"]
ALGOS = {
    "lokr+dora": (["algo=lokr", *LOKR, *DORA], 10000, 10000),
    "lokr": (["algo=lokr", *LOKR], 10000, 10000),
    "lora": (["algo=lora"], 8, 8),
    "lora+dora": (["algo=lora", *DORA], 8, 8),
    "locon": (["algo=locon"], 8, 8),
    "loha": (["algo=loha"], 8, 8),
    "loha+dora": (["algo=loha", *DORA], 8, 8),
    "tsm stage 1": (["algo=tsm", "tsm_n_scales=8,1", "tsm_stage=1"], 8, 8),
    "tsm stage 2": (["algo=tsm", "tsm_n_scales=8,1", "tsm_stage=2"], 8, 8),
    "tsm-lokr stage 1": (["algo=tsm-lokr", "tsm_n_scales=8,1", "tsm_stage=1", *LOKR, *DORA, "tsm_offload_states=False"], 10000, 10000),
    "tsm-lokr stage 2": (["algo=tsm-lokr", "tsm_n_scales=8,1", "tsm_stage=2", *LOKR, *DORA, "tsm_offload_states=False"], 10000, 10000),
    "tlora": (["algo=tlora"], 8, 8),
    "full": (["algo=full"], 8, 8),
    "abba": (["algo=abba"], 8, 8),
    "abba+dora": (["algo=abba", *DORA], 8, 8),
    "diag-oft": (["algo=diag-oft"], 4, 4),
    "boft": (["algo=boft"], 6, 6),
    "glora": (["algo=glora"], 8, 8),
    "lora2": (["algo=lora2"], 8, 8),
    "ia3": (["algo=ia3"], 8, 8),
}


def _model(dtype=torch.bfloat16, **changes):
    torch.manual_seed(0)
    model = anima_models.Anima(**{**TOY, **changes})
    for parameter in model.parameters():
        if parameter.dim() >= 2:
            torch.nn.init.normal_(parameter, std=0.05)
    return model.to(dtype).eval()


def _inputs(dtype, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 16, 1, 8, 8, generator=generator).to(dtype), torch.tensor([0.3]).to(dtype),
            torch.randn(1, 8, 32, generator=generator).to(torch.bfloat16), torch.zeros(1, 1, 8, 8, dtype=torch.bfloat16))


def _stream(model):
    seen = []
    hooks = [block.register_forward_pre_hook(lambda _m, inputs: seen.append(inputs[0].dtype)) for block in model.blocks]
    hooks.append(model.final_layer.register_forward_pre_hook(lambda _m, inputs: seen.append(inputs[0].dtype)))
    return seen, hooks


def _run(model, dtype, autocast=True):
    seen, hooks = _stream(model)
    x, timesteps, text, pad = _inputs(dtype)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        out = model(x, timesteps, text.to(dtype if not autocast else torch.bfloat16), padding_mask=pad.to(dtype))
    for hook in hooks:
        hook.remove()
    return out, seen


def _stream_model(**changes):
    model = _model(**changes)
    model.enable_stream_fp32()
    return model


def test_only_the_block_linears_stay_in_the_training_dtype():
    model = _stream_model()
    kept = [name for name, parameter in model.named_parameters() if parameter.dtype == torch.bfloat16]
    assert len(kept) == TOY["num_blocks"] * BLOCK_LINEARS, kept
    assert all(name.startswith("blocks.") and "adaln" not in name and "_norm" not in name for name in kept), kept
    assert model.dtype == torch.float32  # what sampling builds its latent and sigmas from


def test_the_stream_stays_fp32_under_bf16_autocast():
    with torch.no_grad():
        out, seen = _run(_stream_model(), torch.float32)
    assert seen == [torch.float32] * (TOY["num_blocks"] + 1)
    assert out.dtype == torch.float32


def test_the_stream_is_closer_to_an_fp32_forward():
    with torch.no_grad():
        truth, _ = _run(_model(torch.float32), torch.float32, autocast=False)
        plain, _ = _run(_model(), torch.bfloat16)
        stream, _ = _run(_stream_model(), torch.float32)
    error = lambda out: float((out.float() - truth).norm() / truth.norm())
    assert error(stream) < error(plain) / 1.5, (error(stream), error(plain))


def test_a_training_dtype_input_is_accepted():
    """iLECO, ADDifT and the adaptive sampler feed the DiT from their own code paths."""
    with torch.no_grad():
        out, seen = _run(_stream_model(), torch.bfloat16)
    assert out.dtype == torch.float32 and seen[0] == torch.float32


def _spy_on_attention(monkeyed):
    original = attention.attention

    def spy(tensors, attn_params):
        monkeyed.append([t.dtype for t in tensors])
        return torch.zeros_like(tensors[0]).flatten(2)  # only the dtypes matter here

    return original, spy


def test_the_attention_kernel_gets_16_bit_tensors():
    for mode in ("flash", "torch"):
        model = _stream_model()
        model.attn_mode = mode
        seen = []
        original, spy = _spy_on_attention(seen)
        attention.attention = spy
        try:
            with torch.no_grad():
                out, stream = _run(model, torch.float32)
        finally:
            attention.attention = original
        assert len(seen) == 2 * TOY["num_blocks"]
        assert all(dtypes == [torch.bfloat16] * 3 for dtypes in seen), seen
        assert stream[-1] == torch.float32 and out.dtype == torch.float32


def test_the_q_k_norms_and_rope_run_in_fp32():
    seen = []
    original = anima_models._rope_rotate

    def spy(t, cos, sin):
        seen.append((t.dtype, cos.dtype, sin.dtype))
        return original(t, cos, sin)

    anima_models._rope_rotate = spy
    try:
        with torch.no_grad():
            _run(_stream_model(), torch.float32)
    finally:
        anima_models._rope_rotate = original
    assert len(seen) == 2 * TOY["num_blocks"]  # q and k of each self attention
    assert all(dtypes == (torch.float32,) * 3 for dtypes in seen), seen


def test_off_keeps_the_rounding_of_q_and_k():
    seen = []
    original = anima_models._rope_rotate

    def spy(t, cos, sin):
        seen.append(t.dtype)
        return original(t, cos, sin)

    anima_models._rope_rotate = spy
    try:
        with torch.no_grad():
            _, stream = _run(_model(), torch.bfloat16)
    finally:
        anima_models._rope_rotate = original
    assert set(seen) == {torch.bfloat16} and set(stream) == {torch.bfloat16}


def _env():
    # production order (anima_train_network.prepare_anima_runtime, then train())
    lycoris_fast.install_fast_lokr_forward()
    laf.apply_module_patches()
    laf.set_block_compile(False)
    torch._dynamo.reset()


def _network(network_args, dim, alpha, blocks=3, block_compile=False, switch=True):
    """Toy Anima (frozen bf16, train mode) + LyCORIS the way train_network.py does it, then the switch in the trainer's order."""
    import lycoris.kohya as kohya

    torch.manual_seed(0)
    unet = anima_models.Anima(**{**TOY, "num_blocks": blocks}).to(torch.bfloat16)
    unet.requires_grad_(False)
    unet.train()
    args = argparse.Namespace(network_args=list(network_args), network_dim=dim, network_alpha=alpha,
                              network_module="lycoris.kohya", network_dropout=None, lycoris_stock=False,
                              lycoris_algo_fixes=False, network_weights=None)
    kwargs = lycoris_args.fix_lycoris_bool_kwargs(args.network_module, dict(item.split("=", 1) for item in args.network_args))
    algo = (kwargs.get("algo", "lora") or "lora").lower()
    create = laf.prepare_network_kwargs(algo, dict(kwargs))
    create.setdefault("dropout", 0.0)
    laf.set_block_compile(block_compile)
    network = kohya.create_network(1.0, dim, alpha, None, None, unet, neuron_dropout=None, **create)
    if hasattr(network, "prepare_network"):
        network.prepare_network(args)
    network.apply_to(None, unet, False, True)
    laf.after_network_created(network, args)
    network.requires_grad_(True)
    network.train()
    torch.manual_seed(3)
    with torch.no_grad():
        for parameter in network.parameters():  # a zero-init factor must not hide a path
            if parameter.is_floating_point() and parameter.abs().sum() == 0:
                parameter.add_(torch.randn_like(parameter) * 0.01)
    if switch:
        unet.enable_stream_fp32()
    return unet, network


def _step(unet, network, x, timesteps, text, pad, checkpoint=False, backward=True):
    network.zero_grad(set_to_none=True)
    if checkpoint:
        x = x.detach().requires_grad_(True)
    laf.set_timestep(network, timesteps.float() * 1000.0)
    with torch.backends.mkldnn.flags(enabled=False):  # oneDNN has no bf16 backward on some CPUs
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = unet(x, timesteps, text, padding_mask=pad)
        if not backward:
            return out.detach().clone(), {}
        out.float().square().mean().backward()
    grads = {name: parameter.grad.detach().clone() for name, parameter in network.named_parameters() if parameter.grad is not None}
    return out.detach().clone(), grads


def test_every_network_keeps_the_fp32_stream_and_trains():
    _env()
    for algo, (network_args, dim, alpha) in ALGOS.items():
        for preset in ("anima-exp", "anima"):
            what = f"{algo} {preset}"
            unet, network = _network([*network_args, f"preset={preset}"], dim, alpha)
            seen, hooks = _stream(unet)
            out, grads = _step(unet, network, *_inputs(torch.float32))
            for hook in hooks:
                hook.remove()
            assert seen == [torch.float32] * (TOY["num_blocks"] + 1), (what, seen)
            assert out.dtype == torch.float32 and torch.isfinite(out).all(), what
            assert grads and any(float(g.abs().sum()) > 0 for g in grads.values()), what
            assert all(torch.isfinite(g).all() for g in grads.values()), what
            unet.eval()
            with torch.no_grad():
                sampled, _ = _step(unet, network, *_inputs(torch.float32), backward=False)
            assert sampled.dtype == torch.float32, what


def test_checkpointed_blocks_train_on_the_fp32_stream():
    _env()
    network_args, dim, alpha = ALGOS["lokr+dora"]
    plain, network = _network([*network_args, "preset=anima-exp"], dim, alpha)
    _, reference = _step(plain, network, *_inputs(torch.float32))
    checkpointed, network = _network([*network_args, "preset=anima-exp"], dim, alpha)
    checkpointed.enable_gradient_checkpointing()
    _, grads = _step(checkpointed, network, *_inputs(torch.float32), checkpoint=True)
    assert grads.keys() == reference.keys()
    for name in grads:
        torch.testing.assert_close(grads[name], reference[name], rtol=1e-3, atol=1e-6, msg=lambda m: f"{name}: {m}")


def test_the_adaln_hoist_runs_on_the_fp32_stream():
    _env()
    network_args, dim, alpha = ALGOS["lokr+dora"]
    unet, network = _network([*network_args, "preset=anima-exp"], dim, alpha)
    unet.enable_gradient_checkpointing()
    anima_models.ADALN_HOIST = True
    try:
        out, grads = _step(unet, network, *_inputs(torch.float32), checkpoint=True)
    finally:
        anima_models.ADALN_HOIST = False
    assert out.dtype == torch.float32 and grads


def test_merged_sample_weights_give_the_per_call_result():
    """The cache's adaLN merges keep the wider base: a bf16 merge would meet the fp32 stream outside autocast."""
    _env()
    network_args, dim, alpha = ALGOS["lokr+dora"]
    unet, network = _network([*network_args, "preset=anima-exp"], dim, alpha)
    unet.eval()
    x, timesteps, text, pad = _inputs(torch.float32)
    with torch.no_grad():
        per_call, _ = _step(unet, network, x, timesteps, text, pad, backward=False)
        seen, hooks = _stream(unet)
        with lycoris_fast.cached_lokr_weights(unet, torch.bfloat16, torch.device("cpu"), park_base=False) as cached:
            merged, _ = _step(unet, network, x, timesteps, text, pad, backward=False)
        for hook in hooks:
            hook.remove()
    assert cached > 0 and seen == [torch.float32] * (TOY["num_blocks"] + 1)
    assert merged.dtype == torch.float32
    assert float((merged - per_call).norm() / per_call.norm()) < 1e-2


def test_the_sampler_state_stays_fp32():
    model = _stream_model()
    states = []
    original = model.forward

    def spy(x, t, *args, **kwargs):
        states.append((x.dtype, t.dtype))
        return original(x, t, *args, **kwargs)

    model.forward = spy
    text = _inputs(torch.float32)[2]
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        latent = anima_train_utils.do_sample(64, 64, 1, model, text, 2, model.dtype, torch.device("cpu"), 2.0, 3.0, text)
    assert latent.dtype == torch.float32 and torch.isfinite(latent).all()
    assert states == [(torch.float32, torch.float32)] * 4


def test_the_llm_adapter_stays_in_the_training_dtype():
    model = _stream_model(use_llm_adapter=True, crossattn_emb_channels=1024, num_blocks=1)
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.llm_adapter.parameters())
    x, timesteps, _, pad = _inputs(torch.float32)
    source = torch.randn(1, 8, 1024)
    ids, mask = torch.ones(1, 8, dtype=torch.long), torch.ones(1, 8, dtype=torch.long)
    for dtype in (torch.bfloat16, torch.float32):  # training passes the training dtype, sampling model.dtype
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = model(x, timesteps, source.to(dtype), padding_mask=pad, target_input_ids=ids, target_attention_mask=mask,
                        source_attention_mask=mask)
        assert out.dtype == torch.float32 and torch.isfinite(out).all()


def _counting_backend():
    wrapped = lookup_backend("aot_eager")
    calls = []

    def backend(graph, example_inputs, **kwargs):
        calls.append(len(example_inputs))
        return wrapped(graph, example_inputs, **kwargs)

    return backend, calls


def _compiled(unet, network, backend="aot_eager"):
    bands = anima_block_compile.plan_bands([16, 24, 36, 64], "union")
    laf.prime_module_caches(network)
    laf.plan_block_compile_rows(network, unet, max(hi for _, hi in bands))
    return anima_block_compile.compile_blocks(unet, bands=bands, backend=backend, dynamic="union", recompile_limit=64,
                                              isolate_cache=False, stream_fp32=True)


def test_compiled_blocks_match_the_eager_stream():
    for algo in ("lokr+dora", "lora", "tsm stage 1", "loha+dora"):
        _env()
        network_args, dim, alpha = ALGOS[algo]
        unet, network = _network([*network_args, "preset=anima-exp"], dim, alpha, block_compile=True)
        inputs = _inputs(torch.float32)
        reference, reference_grads = _step(unet, network, *inputs)
        state = _compiled(unet, network)
        out, grads = _step(unet, network, *inputs)
        assert state.compiled_calls >= TOY["num_blocks"] and state.eager_calls == 0, algo
        assert torch.equal(out, reference), (algo, float((out - reference).abs().max()))
        assert grads.keys() == reference_grads.keys()
        for name in grads:
            torch.testing.assert_close(grads[name], reference_grads[name], rtol=2 ** -5, atol=1e-5, msg=lambda m: f"{algo} {name}: {m}")
        anima_block_compile.restore_blocks(unet)
    _env()


def test_every_block_shares_one_compiled_graph():
    """Off, anima-exp runs 2 (Block 0's bf16 input, the fp32 residual of Blocks 1+): here Block 0 sees fp32 too."""
    _env()
    network_args, dim, alpha = ALGOS["lora"]
    unet, network = _network([*network_args, "preset=anima-exp"], dim, alpha, blocks=4, block_compile=True)
    backend, calls = _counting_backend()
    state = _compiled(unet, network, backend)
    _step(unet, network, *_inputs(torch.float32), checkpoint=True, backward=False)  # as training: the input carries grad
    assert state.compiled_calls == 4 and state.eager_calls == 0
    assert len(calls) == 1, f"{len(calls)} graphs for 4 Blocks"
    anima_block_compile.restore_blocks(unet)
    _env()


def test_the_stream_has_a_compile_cache_of_its_own():
    off = anima_block_compile.compile_signature([(16, 64)], "inductor", None, "union")
    assert anima_block_compile.compile_signature([(16, 64)], "inductor", None, "union", False) == off
    assert ";stream" not in off
    assert anima_block_compile.compile_signature([(16, 64)], "inductor", None, "union", True) == f"{off};stream=fp32"


def _args(**changes):
    values = dict(anima_stream_fp32=True, mixed_precision="bf16")
    return types.SimpleNamespace(**{**values, **changes})


def test_unsupported_flags_are_refused():
    for flag in ("full_bf16", "full_fp16", "deepspeed", "network_train_text_encoder_only"):
        try:
            validate_stream_args(_args(**{flag: True}))
        except ValueError as error:
            assert flag in str(error)
        else:
            raise AssertionError(f"--{flag} was accepted")
    validate_stream_args(_args())
    validate_stream_args(_args(anima_stream_fp32=False, full_bf16=True))


def test_the_trainer_switches_it_on_after_the_whole_model_cast():
    trainer = AnimaNetworkTrainer()
    trainer.is_swapping_blocks = False
    trainer._use_unsloth_offload_checkpointing = False
    model = _model()
    accelerator = types.SimpleNamespace(prepare=lambda module, **_kwargs: module)
    args = types.SimpleNamespace(anima_stream_fp32=True, gradient_checkpointing=False)
    assert trainer.prepare_unet_with_accelerator(args, accelerator, model) is model
    assert model.stream_fp32 and model.dtype == torch.float32


def test_the_trainer_leaves_it_off_by_default():
    trainer = AnimaNetworkTrainer()
    trainer.is_swapping_blocks = False
    trainer._use_unsloth_offload_checkpointing = False
    model = _model()
    accelerator = types.SimpleNamespace(prepare=lambda module, **_kwargs: module)
    trainer.prepare_unet_with_accelerator(types.SimpleNamespace(gradient_checkpointing=False), accelerator, model)
    assert not model.stream_fp32 and model.dtype == torch.bfloat16


def _switch(trainer, unet):
    trainer.is_swapping_blocks = False
    trainer._use_unsloth_offload_checkpointing = False
    args = types.SimpleNamespace(anima_stream_fp32=True, gradient_checkpointing=False)
    trainer.prepare_unet_with_accelerator(args, types.SimpleNamespace(prepare=lambda module, **_kwargs: module), unet)


def test_the_model_input_dtype_follows_the_switch_and_not_the_option():
    trainer = AnimaNetworkTrainer()
    assert trainer.model_input_dtype(torch.bfloat16) == torch.bfloat16
    _switch(trainer, _model())
    assert trainer.model_input_dtype(torch.bfloat16) == torch.float32


def _trainer_step(trainer, args, accelerator, scheduler, unet, network, seen=None):
    """One call of the trainer's own get_noise_pred_and_target, what the GoRA / RaLoRA pre-pass and every step run."""
    hook = None
    if seen is not None:
        hook = unet.register_forward_pre_hook(lambda _m, inputs: seen.extend(t.dtype for t in inputs[:2]))
    latents = torch.randn(1, 16, 8, 8)
    text = [torch.randn(1, 8, 32), torch.ones(1, 8, dtype=torch.long), torch.zeros(1, 8, dtype=torch.long),
            torch.ones(1, 8, dtype=torch.long)]
    with torch.backends.mkldnn.flags(enabled=False):
        out = trainer.get_noise_pred_and_target(args, accelerator, scheduler, latents, {}, text, [], unet, network,
                                                torch.bfloat16, True)[0]
        out.float().square().mean().backward()
    if hook is not None:
        hook.remove()
    return out


def _trainer_network(algo, preset):
    """A real trainer and accelerator on a toy DiT, the option on and the DiT not switched yet."""
    from accelerate import Accelerator

    from anima_train_network import setup_parser

    args = setup_parser().parse_args(["--pretrained_model_name_or_path", "x", "--output_dir", "x", "--output_name", "x"])
    args.anima_stream_fp32, args.gradient_checkpointing = True, False
    args.timestep_sampling, args.discrete_flow_shift, args.sigmoid_scale = "sigmoid", 1.0, 1.0
    unet, network = _network([f"algo={algo}", f"preset={preset}"], 8, 8, switch=False)
    unet._preprocess_text_embeds = lambda context, *_args, **_kwargs: context  # the toy has no llm_adapter
    trainer = AnimaNetworkTrainer()
    return trainer, args, Accelerator(mixed_precision="bf16", cpu=True), trainer.get_noise_scheduler(args, torch.device("cpu")), unet, network


def test_a_pre_pass_before_the_switch_runs_like_the_option_off():
    """GoRA and RaLoRA measure their ranks through the trainer's forward before prepare_unet_with_accelerator."""
    _env()
    for algo in ("gora", "ralora"):
        for preset in ("anima-exp", "anima"):
            trainer, args, accelerator, scheduler, unet, network = _trainer_network(algo, preset)
            seen = []
            before = _trainer_step(trainer, args, accelerator, scheduler, unet, network, seen)
            assert seen == [torch.bfloat16, torch.bfloat16] and before.dtype == torch.bfloat16, (algo, preset, seen)
            network.zero_grad(set_to_none=True)
            _switch(trainer, unet)
            seen.clear()
            after = _trainer_step(trainer, args, accelerator, scheduler, unet, network, seen)
            assert seen == [torch.float32, torch.float32] and after.dtype == torch.float32, (algo, preset, seen)


def test_the_routing_timestep_stays_in_the_training_dtype():
    """Forge replays TSM from ss_mixed_precision: routing on an fp32 timestep would pick other experts than the file says."""
    _env()
    trainer, args, accelerator, scheduler, unet, network = _trainer_network("lora", "anima-exp")
    _switch(trainer, unet)
    routed, seen = [], []
    trainer.apply_tlora_mask = lambda timesteps: routed.append(timesteps.dtype)
    _trainer_step(trainer, args, accelerator, scheduler, unet, network, seen)
    assert routed == [torch.bfloat16]
    assert seen == [torch.float32, torch.float32]  # the DiT itself gets the fp32 timestep


def test_a_grouped_draw_groups_on_the_routed_timestep():
    """874.5 lies in core 6 of 8, its bf16 rounding 876 in core 7: the window must hold the expert the row is routed to."""
    from library import tsm_lokr
    from library.tsm_offload import ExpertOffloadManager

    cfg = tsm_lokr.parse_config({"tsm_n_scales": "8,1", "tsm_grouped_accumulation": "True", "tsm_resident_cores": "1"})
    manager = ExpertOffloadManager([[torch.nn.Parameter(torch.zeros(2))] for _ in range(8)], None, "cpu",
                                   core_index=cfg.core_index, offload_states=False, resident_cores=8, grouped=True,
                                   core_index_t=cfg.core_index_from_timestep)
    t = torch.tensor([874.5])
    draw = lambda: (torch.zeros(1, 4), t, torch.zeros(1, 1))
    manager.window_core = None
    manager.grouped_draw(draw)
    assert manager.window_core == cfg.core_index_from_timestep(874.5)
    manager.window_core = None
    manager.grouped_draw(draw, routing_dtype=torch.bfloat16)
    assert manager.window_core == cfg.core_index_from_timestep(float(t.to(torch.bfloat16).float()))
    assert cfg.core_index_from_timestep(874.5) != manager.window_core


if __name__ == "__main__":
    failed = 0
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                test()
                print("ok", name)
            except Exception as error:  # every test reports, the exit code keeps the failure
                failed += 1
                print("FAILED", name, type(error).__name__, str(error)[:300])
    raise SystemExit(1 if failed else 0)
