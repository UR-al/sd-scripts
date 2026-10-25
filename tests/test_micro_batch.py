"""CPU tests for library/micro_batch.py (token-budget micro-batching) and its use by the trainers: chunk sizes, the batch
dict split (collate_fn's entry kinds), the weighted backward == the batch-mean backward, the logging sums, and
anima_train_network's split_batch_for_backward (budget in DiT tokens, announced once per tier).

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m pytest tests/test_micro_batch.py -q
"""
import argparse
import logging
import types

import pytest
import torch

from library import micro_batch


def test_chunk_sizes_are_even_and_minimal():
    assert micro_batch.chunk_sizes(8, 8) == [8]
    assert micro_batch.chunk_sizes(8, 9) == [8]
    assert micro_batch.chunk_sizes(8, 4) == [4, 4]
    assert micro_batch.chunk_sizes(8, 5) == [4, 4]  # not [5, 3]
    assert micro_batch.chunk_sizes(8, 2) == [2, 2, 2, 2]
    assert micro_batch.chunk_sizes(8, 1) == [1] * 8
    assert micro_batch.chunk_sizes(7, 4) == [4, 3]
    assert micro_batch.chunk_sizes(3, 2) == [2, 1]
    assert micro_batch.chunk_sizes(5, 2) == [2, 2, 1]
    assert micro_batch.chunk_sizes(1, 1) == [1]
    assert micro_batch.chunk_sizes(0, 4) == []
    assert micro_batch.chunk_sizes(6, 0) == [1] * 6  # a budget below one row still runs one row per chunk
    for n in range(1, 20):
        for k in range(1, 12):
            sizes = micro_batch.chunk_sizes(n, k)
            assert sum(sizes) == n and max(sizes) <= k and max(sizes) - min(sizes) <= 1 and len(sizes) == -(-n // k)


def _collated(n=8):
    """A batch like train_util's collate_fn builds it: row tensors, lists of row tensors (text encoder outputs with a
    None entry), per-row lists, shared None / scalars."""
    g = torch.Generator().manual_seed(0)
    return {
        "latents": torch.randn(n, 16, 8, 8, generator=g),
        "loss_weights": torch.arange(n, dtype=torch.float32),
        "text_encoder_outputs_list": [torch.randn(n, 7, 4, generator=g), None, torch.randn(n, 7, generator=g)],
        "input_ids_list": (torch.arange(n * 3).view(n, 3),),
        "captions": [f"cap {i}" for i in range(n)],
        "custom_attributes": [{"i": i} for i in range(n)],
        "flippeds": [bool(i % 2) for i in range(n)],
        "original_sizes_hw": torch.tensor([[512, 512]] * n),
        "network_multipliers": torch.ones(n),
        "images": None,
        "masks": None,
        "alpha_masks": None,
        "shared_scalar": 3,
        "shared_tensor": torch.zeros(3),  # not n rows: shared, not sliced
    }


def test_split_batch_slices_rows_and_shares_the_rest():
    b = _collated(8)
    parts = micro_batch.split_batch(b, [3, 3, 2])
    assert len(parts) == 3 and all(set(p) == set(b) for p in parts)
    assert [p["latents"].shape[0] for p in parts] == [3, 3, 2]
    assert torch.equal(torch.cat([p["latents"] for p in parts]), b["latents"])
    assert parts[2]["loss_weights"].tolist() == [6.0, 7.0]
    assert parts[1]["text_encoder_outputs_list"][1] is None
    assert torch.equal(parts[1]["text_encoder_outputs_list"][0], b["text_encoder_outputs_list"][0][3:6])
    assert torch.equal(parts[2]["text_encoder_outputs_list"][2], b["text_encoder_outputs_list"][2][6:8])
    assert isinstance(parts[0]["input_ids_list"], tuple) and parts[0]["input_ids_list"][0].shape == (3, 3)
    assert parts[1]["captions"] == ["cap 3", "cap 4", "cap 5"] and parts[2]["custom_attributes"] == [{"i": 6}, {"i": 7}]
    assert parts[0]["flippeds"] == [False, True, False]
    assert parts[2]["original_sizes_hw"].shape == (2, 2) and parts[0]["network_multipliers"].shape == (3,)
    assert all(p["images"] is None and p["masks"] is None and p["alpha_masks"] is None for p in parts)
    assert all(p["shared_scalar"] == 3 and p["shared_tensor"] is b["shared_tensor"] for p in parts)
    assert parts[0]["latents"].data_ptr() == b["latents"].data_ptr()  # views, no copy
    parts[0]["masked_latents"] = torch.zeros(1)  # a chunk may grow keys without touching the others
    assert "masked_latents" not in parts[1] and "masked_latents" not in b
    assert micro_batch.split_batch(b, [8]) == [b]  # one chunk: the batch itself
    with pytest.raises(ValueError):
        micro_batch.split_batch(b, [4, 0, 4])


def test_weighted_chunk_backward_equals_batch_mean_backward():
    """sum_c (n_c / n) * mean-loss_c backward == mean-loss backward of the whole batch (fp64), for any chunking."""
    torch.manual_seed(0)
    x, y = torch.randn(8, 5, dtype=torch.float64), torch.randn(8, 3, dtype=torch.float64)
    w = torch.rand(8, dtype=torch.float64) + 0.5  # per-sample loss weights, as the trainers use

    def model_and_grads(sizes):
        torch.manual_seed(1)
        lin = torch.nn.Linear(5, 3, dtype=torch.float64)
        batch = {"x": x, "y": y, "loss_weights": w}
        n = x.shape[0]
        total = None
        for sub, s in zip(micro_batch.split_batch(batch, sizes), sizes):
            loss = ((lin(sub["x"]) - sub["y"]) ** 2).mean(1) * sub["loss_weights"]
            loss = loss.mean()
            weight = s / n
            (loss if weight == 1.0 else loss * weight).backward()
            total = micro_batch.weighted_sum(total, loss.detach(), weight)
        return total, lin.weight.grad.clone(), lin.bias.grad.clone()

    ref = model_and_grads([8])
    for sizes in ([4, 4], [3, 3, 2], [1] * 8, [5, 3], [2, 2, 2, 2]):
        got = model_and_grads(sizes)
        for a, b in zip(got, ref):
            assert torch.allclose(a, b, rtol=1e-12, atol=1e-12), sizes


def test_batch_rows_and_rows_norm():
    b = _collated(8)
    assert micro_batch.batch_rows(b) == 8
    assert micro_batch.batch_rows({"images": torch.zeros(3, 3, 8, 8), "captions": ["a"] * 3}) == 3
    assert micro_batch.batch_rows({"loss_weights": torch.ones(5)}) == 5
    assert micro_batch.batch_rows({"captions": ["a", "b"]}) == 2
    assert micro_batch.batch_rows({}) == 0 and micro_batch.batch_rows(None) == 0
    ns = lambda **kw: argparse.Namespace(**kw)
    assert micro_batch.rows_norm(ns(batch_loss_norm_rows=8), b) == 1.0  # a full batch, accumulation 1: unchanged
    assert micro_batch.rows_norm(ns(batch_loss_norm_rows=8), _collated(2)) == 0.25
    assert micro_batch.rows_norm(ns(batch_loss_norm_rows=8), _collated(12)) == 1.5
    # accelerate divides every backward by gradient_accumulation_steps: the factor carries it so a sample is 1/N per step
    assert micro_batch.rows_norm(ns(batch_loss_norm_rows=8, gradient_accumulation_steps=2), b) == 2.0
    assert micro_batch.rows_norm(ns(batch_loss_norm_rows=8, gradient_accumulation_steps=2), _collated(2)) == 0.5
    assert micro_batch.rows_norm(ns(batch_loss_norm_rows=8, gradient_accumulation_steps=None), _collated(4)) == 0.5
    assert micro_batch.rows_norm(ns(batch_loss_norm_rows=0, gradient_accumulation_steps=2), _collated(2)) == 1.0  # off
    assert micro_batch.rows_norm(ns(), _collated(2)) == 1.0  # arg absent (other trainers' args)
    assert micro_batch.rows_norm(ns(batch_loss_norm_rows=8), {"x": 1}) == 1.0  # rows unknown


@pytest.mark.parametrize("accumulation", [1, 2, 3])
def test_rows_norm_makes_ragged_batches_weigh_per_sample(accumulation):
    """Ragged bucket batches of 8, 3 and 1 rows in one optimizer step with --batch_loss_norm_rows 8 (chunking inside
    the batches, accelerate's division of every backward by gradient_accumulation_steps modelled): the accumulated
    gradient equals the gradient of (sum of the 12 per-sample losses) / 8 - every sample exactly 1/8, as batch 1 x
    accumulation 8 gives - where plain batch means would weigh the lone sample 8x the full batch's samples."""
    torch.manual_seed(0)
    x, y = torch.randn(12, 5, dtype=torch.float64), torch.randn(12, 3, dtype=torch.float64)
    args = argparse.Namespace(batch_loss_norm_rows=8, gradient_accumulation_steps=accumulation)

    def grads(batches, chunking, norm):
        torch.manual_seed(1)
        lin = torch.nn.Linear(5, 3, dtype=torch.float64)
        for rows in batches:
            batch = {"x": x[rows], "y": y[rows]}
            rn = micro_batch.rows_norm(args, {"loss_weights": torch.ones(len(rows))}) if norm else 1.0
            sizes = chunking.get(len(rows), [len(rows)])
            for sub, s in zip(micro_batch.split_batch(batch, sizes), sizes):
                loss = ((lin(sub["x"]) - sub["y"]) ** 2).mean()
                (loss * (s / len(rows)) * rn / accumulation).backward()  # / accumulation: what accelerator.backward does
        return lin.weight.grad.clone(), lin.bias.grad.clone()

    batches = [list(range(0, 8)), list(range(8, 11)), [11]]
    got = grads(batches, {8: [4, 4], 3: [2, 1]}, norm=True)
    torch.manual_seed(1)
    ref_lin = torch.nn.Linear(5, 3, dtype=torch.float64)
    (((ref_lin(x) - y) ** 2).mean(1).sum() / 8).backward()  # every sample 1/8
    assert torch.allclose(got[0], ref_lin.weight.grad, rtol=1e-12, atol=1e-12)
    assert torch.allclose(got[1], ref_lin.bias.grad, rtol=1e-12, atol=1e-12)
    plain = grads(batches, {}, norm=False)  # batch means: the 1-row batch weighs as much as the 8-row one
    assert not torch.allclose(plain[0], ref_lin.weight.grad, rtol=1e-6, atol=1e-6)
    # the old setting itself, batch 1 x accumulation 8 over 8 samples, through the same formula: 1/8 per sample
    args8 = argparse.Namespace(batch_loss_norm_rows=8, gradient_accumulation_steps=8)
    torch.manual_seed(1)
    lin8 = torch.nn.Linear(5, 3, dtype=torch.float64)
    for i in range(8):
        rn = micro_batch.rows_norm(args8, {"loss_weights": torch.ones(1)})
        (((lin8(x[i:i + 1]) - y[i:i + 1]) ** 2).mean() * rn / 8).backward()
    torch.manual_seed(1)
    ref8 = torch.nn.Linear(5, 3, dtype=torch.float64)
    (((ref8(x[:8]) - y[:8]) ** 2).mean(1).sum() / 8).backward()
    assert torch.allclose(lin8.weight.grad, ref8.weight.grad, rtol=1e-12, atol=1e-12)


def test_weighted_sum_handles_none_and_unit_weight():
    t = torch.tensor(2.0)
    assert micro_batch.weighted_sum(None, None, 0.5) is None
    assert micro_batch.weighted_sum(t, None, 0.5) is t
    assert micro_batch.weighted_sum(None, t, 1.0) is t
    assert float(micro_batch.weighted_sum(None, t, 0.25)) == 0.5
    assert float(micro_batch.weighted_sum(torch.tensor(1.0), t, 0.25)) == 1.5


def _anima_split(args, batch, patch=2, logged=None):
    from anima_train_network import AnimaNetworkTrainer

    fake_self = types.SimpleNamespace(_micro_batch_logged=set() if logged is None else logged)
    accel = types.SimpleNamespace(unwrap_model=lambda m: m)
    unet = types.SimpleNamespace(patch_spatial=patch)
    return AnimaNetworkTrainer.split_batch_for_backward(fake_self, args, accel, batch, unet)


def _latents_batch(n, px):
    return {"latents": torch.zeros(n, 16, px // 8, px // 8), "captions": [str(i) for i in range(n)],
            "loss_weights": torch.ones(n)}


def test_anima_split_batch_for_backward_by_token_budget(caplog):
    """--anima_micro_batch_max_tokens 20000 with batch 8: 2048 tier 8 x 1 row, 1536 4 x 2, 1024 2 x 4, 512 one chunk of 8;
    weights rows / 8; ragged last batches; 0 = off; one INFO line in total (every aspect-ratio bucket has its own token
    count, so a line per count would flood the progress bar), the other distinct counts at DEBUG."""
    from anima_train_network import micro_step_tokens, tokens_of_latents

    assert tokens_of_latents(torch.zeros(1, 16, 64, 64), 2) == 1024 and micro_step_tokens(torch.zeros(8, 16, 64, 64), 2) == 8192
    args = argparse.Namespace(anima_micro_batch_max_tokens=20000)
    want = {2048: [1] * 8, 1536: [2, 2, 2, 2], 1024: [4, 4], 512: [8]}
    logged = set()
    with caplog.at_level(logging.INFO, logger="anima_train_network"):
        for px, sizes in want.items():
            chunks = _anima_split(args, _latents_batch(8, px), logged=logged)
            assert [c["latents"].shape[0] for c, _ in chunks] == sizes, px
            assert [w for _, w in chunks] == [s / 8 for s in sizes], px
            assert all(c["captions"] == [str(i) for i in range(a, a + s)]
                       for (c, _), a, s in zip(chunks, [sum(sizes[:i]) for i in range(len(sizes))], sizes)), px
            _anima_split(args, _latents_batch(8, px), logged=logged)  # the same tier again: no second log line
    lines = [r for r in caplog.records if "micro-batching" in r.getMessage()]
    assert [r.levelname for r in lines] == ["INFO"] and "20000" in lines[0].getMessage(), lines  # the first chunked batch only
    assert len(logged) == 3  # 2048 / 1536 / 1024 seen; the 512 batch fits whole
    with caplog.at_level(logging.DEBUG, logger="anima_train_network"):
        caplog.clear()
        _anima_split(args, _latents_batch(8, 1024), logged=set(logged))  # a known count: nothing, not even at DEBUG
        assert not [r for r in caplog.records if "micro-batching" in r.getMessage()]
        _anima_split(args, _latents_batch(8, 1536), logged={1})  # a new count after the first: DEBUG only
        assert [r.levelname for r in caplog.records if "micro-batching" in r.getMessage()] == ["DEBUG"]
    # ragged last batch of a bucket, and a batch of one row
    chunks = _anima_split(args, _latents_batch(3, 1536), logged=logged)
    assert [c["latents"].shape[0] for c, _ in chunks] == [2, 1] and [w for _, w in chunks] == [2 / 3, 1 / 3]
    one = _latents_batch(1, 2048)
    assert _anima_split(args, one, logged=logged) == [(one, 1.0)]
    # off, or no latents in the batch: the batch itself
    full = _latents_batch(8, 2048)
    assert _anima_split(argparse.Namespace(anima_micro_batch_max_tokens=0), full) == [(full, 1.0)]
    assert _anima_split(argparse.Namespace(), full) == [(full, 1.0)]
    nolat = {"captions": ["a"] * 8}
    assert _anima_split(args, nolat) == [(nolat, 1.0)]
    # a 4-patch model halves the token count per side: 2048 fits 1 row at 20000 either way, 1024 fits all 8
    assert [c["latents"].shape[0] for c, _ in _anima_split(args, _latents_batch(8, 1024), patch=4)] == [8]


def test_validate_micro_batch_args():
    """validate_micro_batch_args (called by assert_extra_args): a negative budget is refused; with a budget the options
    whose objective would change with the chunking are refused, the variance-reduction ones only warned about; budget 0
    accepts anything."""
    import inspect

    from anima_train_network import AnimaNetworkTrainer, validate_micro_batch_args

    ns = argparse.Namespace
    assert validate_micro_batch_args(ns(anima_micro_batch_max_tokens=0, contrastive_flow_matching=True)) == []
    assert validate_micro_batch_args(ns(anima_micro_batch_max_tokens=20000)) == []
    assert validate_micro_batch_args(ns()) == []
    with pytest.raises(ValueError, match="must be >= 0"):
        validate_micro_batch_args(ns(anima_micro_batch_max_tokens=-1))
    for flag in ("contrastive_flow_matching", "adaptive_timestep_sampling", "anchor_snr_weighting", "adv_snr_weighting"):
        with pytest.raises(ValueError, match=flag):
            validate_micro_batch_args(ns(anima_micro_batch_max_tokens=20000, **{flag: True}))
        assert validate_micro_batch_args(ns(anima_micro_batch_max_tokens=20000, **{flag: False})) == []
    with pytest.raises(ValueError, match="dwa"):
        validate_micro_batch_args(ns(anima_micro_batch_max_tokens=20000, patch_topology_loss=True,
                                     patch_topology_dynamic_weighting="dwa"))
    assert validate_micro_batch_args(ns(anima_micro_batch_max_tokens=20000, patch_topology_loss=True,
                                        patch_topology_dynamic_weighting="none")) == []
    assert validate_micro_batch_args(ns(anima_micro_batch_max_tokens=20000, patch_topology_loss=False,
                                        patch_topology_dynamic_weighting="gradnorm")) == []
    warns = validate_micro_batch_args(ns(anima_micro_batch_max_tokens=20000, antithetic_timestep_sampling=True,
                                         flow_use_ot=True, qmc_timestep_sampling="sobol", stratified_timestep_sampling=False))
    assert len(warns) == 3 and all("within each chunk" in w for w in warns), warns
    assert "validate_micro_batch_args(args)" in inspect.getsource(AnimaNetworkTrainer.assert_extra_args)


def test_checkpoint_call_site_uses_micro_step_tokens():
    """get_noise_pred_and_target compares --anima_no_checkpoint_max_tokens against the micro-step's rows x tokens per
    sample (what the activation memory scales with), not against one sample's tokens."""
    import inspect

    from anima_train_network import AnimaNetworkTrainer

    src = inspect.getsource(AnimaNetworkTrainer.get_noise_pred_and_target)
    assert "micro_step_tokens(latents, anima_patch_spatial(m))" in src
    assert "tokens_of_latents(latents" not in src


def test_trainer_loop_runs_backward_per_chunk():
    """train_network.NetworkTrainer.train (structure, from its AST): the chunk loop `for sub_batch, chunk_weight in
    chunks` holds the process_batch call, the backward of the chunk's loss scaled by chunk_weight x rows_norm and the
    three weighted_sum logging sums - nothing of that sits outside the loop - and after the loop the step_func value is
    scaled by rows_norm; the default split_batch_for_backward is the batch itself."""
    import ast
    import inspect
    import textwrap

    import train_network

    tree = ast.parse(textwrap.dedent(inspect.getsource(train_network.NetworkTrainer.train)))
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple)
             and [ast.unparse(e) for e in n.target.elts] == ["sub_batch", "chunk_weight"] and ast.unparse(n.iter) == "chunks"]
    assert len(loops) == 1
    loop = loops[0]
    inside = [ast.unparse(n) for n in ast.walk(loop) if isinstance(n, ast.Call)]
    assert sum(c.startswith("self.process_batch(sub_batch,") for c in inside) == 1
    assert inside.count("accelerator.backward(c_loss if backward_scale == 1.0 else c_loss * backward_scale)") == 1
    assert sum(c.startswith("micro_batch.weighted_sum(") for c in inside) == 3
    assigns = [ast.unparse(n) for n in ast.walk(loop) if isinstance(n, ast.Assign)]
    assert "backward_scale = chunk_weight * rows_norm" in assigns
    # outside the loop: no training backward and no process_batch on a chunk (the validation path calls process_batch
    # with batch=... keywords; that is not the training step)
    loop_calls = {id(n) for n in ast.walk(loop)}
    outside = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call) and id(n) not in loop_calls]
    assert not any(c.startswith("accelerator.backward(") or c.startswith("self.process_batch(sub_batch") for c in outside), \
        [c for c in outside if c.startswith("accelerator.backward(") or c.startswith("self.process_batch(")]
    top_assigns = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Assign)]
    assert "chunks = self.split_batch_for_backward(args, accelerator, batch, unet)" in top_assigns
    assert "rows_norm = micro_batch.rows_norm(args, batch)" in top_assigns
    aug = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.AugAssign)]
    assert "_step_func_loss_accum += loss.detach().item() * rows_norm" in aug
    trainer = types.SimpleNamespace()
    batch = {"latents": torch.zeros(2, 1, 1, 1)}
    assert train_network.NetworkTrainer.split_batch_for_backward(trainer, None, None, batch, None) == [(batch, 1.0)]


def test_anima_split_uses_images_when_latents_are_not_cached():
    """cache_latents off: the collated batch carries pixels (n, 3, H, W) and latents None; the token count comes from the
    8x-downscaled grid, the chunks slice the images (process_batch encodes each chunk)."""
    args = argparse.Namespace(anima_micro_batch_max_tokens=3000)
    imgs = {"images": torch.zeros(8, 3, 512, 512), "latents": None, "captions": [str(i) for i in range(8)]}
    chunks = _anima_split(args, imgs)  # 512^2 -> 64 x 64 latents -> 1024 tokens per row -> 2 rows per 3000-token chunk
    assert [c["images"].shape[0] for c, _ in chunks] == [2, 2, 2, 2] and [w for _, w in chunks] == [0.25] * 4
    assert chunks[1][0]["captions"] == ["2", "3"] and chunks[0][0]["latents"] is None
    one = {"images": torch.zeros(1, 3, 512, 512), "latents": None}
    assert _anima_split(args, one) == [(one, 1.0)]
