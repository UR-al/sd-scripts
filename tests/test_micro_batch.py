"""CPU tests for library/micro_batch.py (token-budget micro-batching) and its use by the trainers: chunk sizes, the batch
dict split (collate_fn's entry kinds), the weighted backward == the batch-mean backward, the logging sums, and
anima_train_network's split_batch_for_backward (budget in DiT tokens, announced once per tier).

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m pytest tests/test_micro_batch.py -q
"""
import argparse
import logging
import types
from typing import NamedTuple

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
    fake_self._token_budget_chunks = types.MethodType(AnimaNetworkTrainer._token_budget_chunks, fake_self)
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


# ---------------------------------------------------------------------------------------------------------------------
# --step_rows: N-row optimizer steps packed from the bucket batches
# ---------------------------------------------------------------------------------------------------------------------


class _BBI(NamedTuple):
    """train_util.BucketBatchIndex's fields (dataset_rows_per_epoch reads them by name)."""

    bucket_index: int
    bucket_batch_size: int
    batch_index: int


def _fake_dataset(bucket_sizes, batch_size=8, jitter=False, delegate=False):
    """A dataset as train_util.BaseDataset.make_buckets leaves it: batch_buckets (image keys per bucket) and
    all_buckets_indices (ceil(len / batch_size) BucketBatchIndex per bucket); delegate=True wraps it the way the
    FineTuning / ControlNet datasets hold a DreamBooth delegate."""
    buckets = [[f"b{b}_{i}" for i in range(n)] for b, n in enumerate(bucket_sizes)]
    indices = [_BBI(b, batch_size, k) for b, bucket in enumerate(buckets) for k in range(-(-len(bucket) // batch_size))]
    ds = types.SimpleNamespace(batch_buckets=buckets, all_buckets_indices=indices, buckets_indices=list(indices),
                               has_resolution_jitter=jitter, num_train_images=sum(bucket_sizes))
    if delegate:
        return types.SimpleNamespace(dreambooth_dataset_delegate=ds, num_train_images=ds.num_train_images)
    return ds


def test_dataset_rows_per_epoch_counts_the_bucket_batch_rows():
    ds = _fake_dataset([20, 3, 1, 8, 9])  # 20 -> 8, 8, 4; 3; 1; 8; 9 -> 8, 1: 41 rows in 8 batches
    assert len(ds.all_buckets_indices) == 8
    assert micro_batch.dataset_rows_per_epoch(ds) == 41
    group = types.SimpleNamespace(datasets=[ds, _fake_dataset([5], delegate=True)], num_train_images=46)
    assert micro_batch.dataset_rows_per_epoch(group) == 46
    # a dataset without the bucket structure counts its num_train_images: no attributes at all, or the empty lists every
    # train_util.BaseDataset starts with (a --dataset_class MinimalDataset never makes buckets: before the fix it counted 0
    # rows, so max_train_steps became 0 and the trainer divided by zero)
    plain = types.SimpleNamespace(datasets=[types.SimpleNamespace(num_train_images=7)])
    assert micro_batch.dataset_rows_per_epoch(plain) == 7
    empty = types.SimpleNamespace(datasets=[types.SimpleNamespace(all_buckets_indices=[], buckets_indices=[],
                                                                  batch_buckets=[], num_train_images=12)])
    assert micro_batch.dataset_rows_per_epoch(empty) == 12
    from library import train_util

    class _Arbitrary(train_util.MinimalDataset):  # the --dataset_class shape (train_util.load_arbitrary_dataset)
        def __init__(self):
            super().__init__((512, 512), 1.0)
            self.num_train_images = 40

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return {"latents": torch.zeros(4, 1)}

    arbitrary = _Arbitrary()
    assert arbitrary.datasets == [arbitrary] and arbitrary.all_buckets_indices == [] and arbitrary.batch_buckets == []
    assert micro_batch.dataset_rows_per_epoch(arbitrary) == 40 and micro_batch.steps_per_epoch(40, 8) == 5
    with pytest.raises(ValueError, match="0 rows per epoch"):
        micro_batch.dataset_rows_per_epoch(types.SimpleNamespace(datasets=[types.SimpleNamespace(num_train_images=0)]))
    with pytest.raises(ValueError, match="resolution jitter"):
        micro_batch.dataset_rows_per_epoch(types.SimpleNamespace(datasets=[_fake_dataset([4], jitter=True)]))
    assert micro_batch.steps_per_epoch(725, 8) == 91 and micro_batch.steps_per_epoch(728, 8) == 91
    assert micro_batch.steps_per_epoch(729, 8) == 92 and micro_batch.steps_per_epoch(0, 8) == 0
    with pytest.raises(ValueError):
        micro_batch.steps_per_epoch(10, 0)


def test_validate_step_rows():
    ns = argparse.Namespace
    assert micro_batch.validate_step_rows(ns()) == 0
    assert micro_batch.validate_step_rows(ns(step_rows=0, gradient_accumulation_steps=4)) == 0
    assert micro_batch.validate_step_rows(ns(step_rows=8, gradient_accumulation_steps=1, batch_loss_norm_rows=0)) == 8
    assert micro_batch.validate_step_rows(ns(step_rows=8, gradient_accumulation_steps=1, batch_loss_norm_rows=8)) == 8
    assert micro_batch.validate_step_rows(ns(step_rows=8)) == 8  # accumulation absent = 1
    with pytest.raises(ValueError, match="gradient_accumulation_steps 1"):
        micro_batch.validate_step_rows(ns(step_rows=8, gradient_accumulation_steps=2))
    with pytest.raises(ValueError, match="batch_loss_norm_rows 4 contradicts"):
        micro_batch.validate_step_rows(ns(step_rows=8, gradient_accumulation_steps=1, batch_loss_norm_rows=4))
    with pytest.raises(ValueError, match="single process"):
        micro_batch.validate_step_rows(ns(step_rows=8, gradient_accumulation_steps=1), num_processes=2)
    with pytest.raises(ValueError, match=">= 0"):
        micro_batch.validate_step_rows(ns(step_rows=-1))
    # the Anima trainer validates at assert_extra_args (before the models load); train() again with the process count
    import inspect

    from anima_train_network import AnimaNetworkTrainer

    assert "micro_batch.validate_step_rows(args)" in inspect.getsource(AnimaNetworkTrainer.assert_extra_args)
    # a pack is several forwards: the row-coupling options are refused / warned about as under the token budget, with
    # the budget off, for every trainer (step_rows_messages = validate_step_rows + row_coupling_messages)
    assert micro_batch.step_rows_messages(ns(step_rows=8, anima_micro_batch_max_tokens=0)) == []
    assert micro_batch.step_rows_messages(ns(step_rows=0, contrastive_flow_matching=True, adaptive_timestep_sampling=True)) == []
    for flag in ("contrastive_flow_matching", "adaptive_timestep_sampling", "anchor_snr_weighting", "adv_snr_weighting"):
        with pytest.raises(ValueError, match=f"--{flag} cannot be combined with --step_rows 8"):
            micro_batch.step_rows_messages(ns(step_rows=8, anima_micro_batch_max_tokens=0, **{flag: True}))
    with pytest.raises(ValueError, match="gradnorm cannot be combined with --step_rows 8"):
        micro_batch.step_rows_messages(ns(step_rows=8, patch_topology_loss=True, patch_topology_dynamic_weighting="gradnorm"))
    warns = micro_batch.step_rows_messages(ns(step_rows=8, antithetic_noise_pairing=True, flow_use_ot=True))
    assert len(warns) == 2 and all("with --step_rows 8" in w and "within each chunk" in w for w in warns), warns
    with pytest.raises(ValueError, match="gradient_accumulation_steps 1"):  # the step_rows checks come first
        micro_batch.step_rows_messages(ns(step_rows=8, gradient_accumulation_steps=2, contrastive_flow_matching=True))


def test_slice_batch_step_pack_and_rows_norm():
    b = _collated(8)
    part = micro_batch.slice_batch(b, 3, 7)
    assert micro_batch.batch_rows(part) == 4 and part["captions"] == ["cap 3", "cap 4", "cap 5", "cap 6"]
    assert torch.equal(part["latents"], b["latents"][3:7]) and part["text_encoder_outputs_list"][1] is None
    assert torch.equal(part["text_encoder_outputs_list"][2], b["text_encoder_outputs_list"][2][3:7])
    assert part["shared_scalar"] == 3 and part["shared_tensor"] is b["shared_tensor"] and part["images"] is None
    assert micro_batch.slice_batch(b, 0, 8) is b
    assert part["latents"].data_ptr() == b["latents"][3:7].data_ptr()  # a view by default ...
    cp = micro_batch.slice_batch(b, 3, 7, copy=True)  # ... a copy on request (the packer's cut pieces)
    assert torch.equal(cp["latents"], part["latents"]) and cp["latents"].data_ptr() != b["latents"][3:7].data_ptr()
    assert cp["text_encoder_outputs_list"][2].data_ptr() != b["text_encoder_outputs_list"][2][3:7].data_ptr()
    assert cp["captions"] == part["captions"] and cp["shared_tensor"] is b["shared_tensor"]
    for bad in ((0, 0), (5, 3), (-1, 2), (2, 9)):
        with pytest.raises(ValueError):
            micro_batch.slice_batch(b, *bad)
    pack = micro_batch.StepPack([micro_batch.slice_batch(b, 0, 3), _collated(5)])
    assert pack.rows == 8 and len(pack) == 2 and micro_batch.batch_rows(pack) == 8
    assert micro_batch.sub_batches(pack) == pack.sub_batches and micro_batch.sub_batches(b) == [b]
    assert [micro_batch.batch_rows(sb) for sb in pack] == [3, 5] and "rows=8" in repr(pack)
    with pytest.raises(ValueError):
        micro_batch.StepPack([])
    ns = argparse.Namespace
    assert micro_batch.rows_norm(ns(step_rows=8, gradient_accumulation_steps=1), pack) == 1.0  # a full pack
    short = micro_batch.StepPack([_collated(5)])
    assert micro_batch.rows_norm(ns(step_rows=8, gradient_accumulation_steps=1), short) == 5 / 8  # the epoch's last pack
    assert micro_batch.rows_norm(ns(step_rows=8, batch_loss_norm_rows=8), short) == 5 / 8  # the same N given twice
    assert micro_batch.rows_norm(ns(step_rows=0, batch_loss_norm_rows=4), _collated(2)) == 0.5  # step_rows off: the old flag


def _loader(sizes):
    return [_collated(n) for n in sizes]


def test_row_packer_packs_exact_rows_and_cuts_at_the_boundary(caplog):
    """Ragged bucket batches 3, 8, 1, 2, 6, 4 (24 rows) -> three 8-row packs: [3, 5 of the 8] [3 rest, 1, 2, 2 of the 6]
    [4 rest, 4]; the cut pieces of one batch carry consecutive rows; len() is the step count; 21 rows -> 8, 8, 5;
    skip_first_packs drops whole packs; a wrong rows_per_epoch is warned about at the end of the epoch."""
    loader = _loader([3, 8, 1, 2, 6, 4])
    packer = micro_batch.RowPacker(loader, 8, 24)
    assert len(packer) == 3
    packs = list(packer)
    assert [p.rows for p in packs] == [8, 8, 8]
    assert [[micro_batch.batch_rows(sb) for sb in p] for p in packs] == [[3, 5], [3, 1, 2, 2], [4, 4]]
    assert packs[0].sub_batches[0] is loader[0]  # a batch that fits whole is passed through, not copied
    assert packs[0].sub_batches[1]["latents"].data_ptr() != loader[1]["latents"].data_ptr()  # a cut piece is a copy ...
    assert packs[1].sub_batches[0]["latents"].data_ptr() != loader[1]["latents"][5:].data_ptr()  # ... both of them
    assert packs[0].sub_batches[1]["captions"] == [f"cap {i}" for i in range(5)]  # the 8-row batch: rows 0..4 ...
    assert packs[1].sub_batches[0]["captions"] == [f"cap {i}" for i in range(5, 8)]  # ... and 5..7 open the next pack
    assert packs[1].sub_batches[3]["captions"] == ["cap 0", "cap 1"]
    assert packs[2].sub_batches[0]["captions"] == [f"cap {i}" for i in range(2, 6)]
    assert torch.equal(torch.cat([packs[1].sub_batches[3]["latents"], packs[2].sub_batches[0]["latents"]]), loader[4]["latents"])
    assert [p.rows for p in packer] == [8, 8, 8]  # re-iterable: one epoch per iteration
    # the epoch's remainder, and the step count
    short = micro_batch.RowPacker(_loader([8, 3, 8, 2]), 8, 21)
    assert len(short) == 3 and [p.rows for p in short] == [8, 8, 5]
    assert [[micro_batch.batch_rows(sb) for sb in p] for p in short] == [[8], [3, 5], [3, 2]]
    skipped = short.skip_first_packs(1)  # the resume path: len() = the remaining packs (calculate_val_loss_check reads it)
    assert len(skipped) == 2 and [p.rows for p in skipped] == [8, 5] and [p.rows for p in skipped] == [8, 5]
    assert len(short.skip_first_packs(3)) == 0 and [p.rows for p in short.skip_first_packs(3)] == []
    # a mismatching rows_per_epoch: every pack is still yielded, the mismatch is logged once per epoch
    with caplog.at_level(logging.WARNING, logger="library.micro_batch"):
        right = micro_batch.RowPacker(_loader([8, 3]), 8, 11)
        assert [p.rows for p in right] == [8, 3] and len(right) == 2
        assert not [r for r in caplog.records if "step_rows" in r.getMessage()]
        wrong = micro_batch.RowPacker(_loader([8, 3, 8]), 8, 16)  # 19 rows -> 3 packs, 2 expected
        assert [p.rows for p in wrong] == [8, 8, 3]
    assert [r.levelname for r in caplog.records if "step_rows" in r.getMessage()] == ["WARNING"]
    with pytest.raises(ValueError, match="row count"):
        list(micro_batch.RowPacker([{"x": 1}], 8, 8))
    with pytest.raises(ValueError):
        micro_batch.RowPacker(loader, 0, 24)


def test_step_pack_gradient_is_one_over_n_per_row():
    """A StepPack through the trainer's formula (chunk_weight x rows_norm, gradient_accumulation_steps 1, Anima's token
    chunking inside each bucket batch): a full 8-row pack of bucket batches [3, 5] (the 5 chunked [3, 2] by the budget)
    accumulates the gradient of (sum of the 8 per-sample losses) / 8; the epoch's last 5-row pack that of (sum of its 5)
    / 8 - every row 1/8 either way, as batch 1 x accumulation 8 gives."""
    torch.manual_seed(0)
    x, y = torch.randn(13, 5, dtype=torch.float64), torch.randn(13, 3, dtype=torch.float64)
    args = argparse.Namespace(step_rows=8, gradient_accumulation_steps=1, anima_micro_batch_max_tokens=3100)

    def pack_of(row_groups):  # bucket batches with 512^2 latents: 1024 tokens a row, 3 rows per 3100-token chunk
        subs = []
        for rows in row_groups:
            sb = _latents_batch(len(rows), 512)
            sb["x"], sb["y"] = x[rows], y[rows]
            subs.append(sb)
        return micro_batch.StepPack(subs)

    def grads(pack):
        torch.manual_seed(1)
        lin = torch.nn.Linear(5, 3, dtype=torch.float64)
        rn = micro_batch.rows_norm(args, pack)
        chunks = _anima_split(args, pack)
        for chunk, w in chunks:
            (((lin(chunk["x"]) - chunk["y"]) ** 2).mean() * w * rn).backward()  # accumulation 1: no division
        return chunks, lin.weight.grad.clone()

    def ref(rows):
        torch.manual_seed(1)
        lin = torch.nn.Linear(5, 3, dtype=torch.float64)
        (((lin(x[rows]) - y[rows]) ** 2).mean(1).sum() / 8).backward()
        return lin.weight.grad.clone()

    chunks, g = grads(pack_of([list(range(0, 3)), list(range(3, 8))]))
    assert [c["x"].shape[0] for c, _ in chunks] == [3, 3, 2] and [w for _, w in chunks] == [3 / 8, 3 / 8, 2 / 8]
    assert torch.allclose(g, ref(list(range(8))), rtol=1e-12, atol=1e-12)
    last = pack_of([list(range(8, 11)), list(range(11, 13))])  # the epoch's last pack: 5 rows
    assert micro_batch.rows_norm(args, last) == 5 / 8
    chunks, g = grads(last)
    assert [w for _, w in chunks] == [3 / 5, 2 / 5]
    assert torch.allclose(g, ref(list(range(8, 13))), rtol=1e-12, atol=1e-12)
    # the base trainer (no token budget): the pack's bucket batches themselves, weights rows / pack rows
    import train_network

    pack = pack_of([[0, 1, 2], [3, 4, 5, 6, 7]])
    base = train_network.NetworkTrainer.split_batch_for_backward(types.SimpleNamespace(), args, None, pack, None)
    assert [b is sb for (b, _), sb in zip(base, pack.sub_batches)] == [True, True] and [w for _, w in base] == [3 / 8, 5 / 8]


def test_trainer_wires_step_rows():
    """train_network.NetworkTrainer.train (source): validates --step_rows with the process count, counts the epoch's rows
    from the datasets, takes max_train_steps from ceil(rows / N), wraps the prepared loader in a RowPacker (after
    accelerator.prepare, before the loop), reads the step's rows through micro_batch.batch_rows, calls on_step_start per
    bucket batch, hands the validation path one bucket batch, and resumes through skip_first_packs instead of
    accelerate's skip_first_batches. --step_rows is a train_util argument."""
    import inspect

    import train_network
    from library import train_util

    src = inspect.getsource(train_network.NetworkTrainer.train)
    for needle in (
        "for message in micro_batch.step_rows_messages(args):",  # every trainer, right after assert_extra_args
        "step_rows = micro_batch.validate_step_rows(args, accelerator.num_processes)",
        '"ss_step_rows": step_rows,',
        "total_batch_size = step_rows * accelerator.num_processes",
        "rows_per_epoch = micro_batch.dataset_rows_per_epoch(train_dataset_group) if step_rows else 0",
        "args.max_train_steps = args.max_train_epochs * pack_steps_per_epoch",
        "train_dataloader = micro_batch.RowPacker(train_dataloader, step_rows, rows_per_epoch)",
        "_actual_bs = micro_batch.batch_rows(batch) or args.train_batch_size",
        "for sb in micro_batch.sub_batches(batch):",
        "micro_batch.sub_batches(batch)[0],",
        "skipped_dataloader = train_dataloader.skip_first_packs(initial_step - 1)",
        "train_dataset_group.attach_epoch_value(current_epoch)",  # the epoch's shuffle before its first fetch
    ):
        assert needle in src, needle
    assert src.index("train_dataset_group.attach_epoch_value(current_epoch)") \
        < src.index("train_dataloader = torch.utils.data.DataLoader(")
    assert src.index("self.assert_extra_args(args, train_dataset_group, val_dataset_group)") \
        < src.index("micro_batch.step_rows_messages(args)") < src.index("train_util.prepare_accelerator(args)")
    assert src.index("training_model = network") < src.index("micro_batch.RowPacker(train_dataloader") \
        < src.index("for step, batch in enumerate(skipped_dataloader or train_dataloader)")
    assert '"--step_rows"' in inspect.getsource(train_util)


def test_step_rows_messages_warn_for_a_large_batch_and_dataset_class():
    """--step_rows N with train_batch_size > N (a bucket batch cut over several steps stays on the device through them)
    and with --dataset_class (rows per epoch from num_train_images) are warned about, not refused."""
    base = dict(step_rows=8, gradient_accumulation_steps=1, batch_loss_norm_rows=0, train_batch_size=8)
    assert micro_batch.step_rows_messages(argparse.Namespace(**base)) == []
    msgs = micro_batch.step_rows_messages(argparse.Namespace(**dict(base, train_batch_size=13)))
    assert len(msgs) == 1 and "train_batch_size 13" in msgs[0] and "<= step_rows" in msgs[0]
    msgs = micro_batch.step_rows_messages(argparse.Namespace(**dict(base, dataset_class="my.module.Dataset")))
    assert len(msgs) == 1 and "dataset_class" in msgs[0] and "num_train_images" in msgs[0]
    assert micro_batch.step_rows_messages(argparse.Namespace(**dict(base, step_rows=0, train_batch_size=13))) == []


def test_row_packer_releases_a_cut_batch_before_yielding():
    """A batch cut at a step boundary: its pieces are copies made before the pack is yielded and the packer drops the
    batch, so while the consumer holds the pack the parent is gone (only the loader could still hold it); a batch larger
    than the step fills whole packs on its own. Same packs as before (the boundary test above covers the layout)."""
    import gc
    import weakref

    refs = []

    def make(n):
        b = _collated(n)
        refs.append(weakref.ref(b["latents"]))
        return b

    def loader(sizes):  # yields without keeping the batch in its own frame (a loader that does, like accelerate's
        for n in sizes:  # DataLoaderShard, holds the parent until the next pull - the packer cannot change that)
            yield make(n)

    packer = micro_batch.RowPacker(loader([3, 13, 16, 4]), 8, 36)
    assert len(packer) == 5
    it = iter(packer)
    pack0 = next(it)  # [3 | 5 of the 13]
    gc.collect()
    assert [micro_batch.batch_rows(sb) for sb in pack0] == [3, 5] and pack0.sub_batches[0]["latents"] is not None
    assert refs[0]() is not None  # the whole 3-row batch is the pack's own sub-batch ...
    assert refs[1]() is None  # ... the cut 13-row batch is released: the pack holds its copies only
    pack1 = next(it)  # [8 of the 13]
    gc.collect()
    assert [micro_batch.batch_rows(sb) for sb in pack1] == [8] and refs[1]() is None
    assert pack1.sub_batches[0]["captions"] == [f"cap {i}" for i in range(5, 13)]
    pack2, pack3 = next(it), next(it)  # the 16-row batch: two whole packs, released
    gc.collect()
    assert [p.rows for p in (pack2, pack3)] == [8, 8] and refs[2]() is None
    assert pack3.sub_batches[0]["captions"] == [f"cap {i}" for i in range(8, 16)]
    pack4 = next(it)
    assert [micro_batch.batch_rows(sb) for sb in pack4] == [4] and pack4.sub_batches[0]["captions"] == ["cap 0", "cap 1", "cap 2", "cap 3"]
    with pytest.raises(StopIteration):
        next(it)


class _RowCountDataset:
    """A train_util.BaseDataset with the bucket structure make_buckets leaves (real set_current_epoch / shuffle_buckets /
    sync_epoch) and a stub payload with the rows BaseDataset.__getitem__ would give (bucket[image_index : + batch size])."""

    def __new__(cls, bucket_sizes, batch_size, seed):
        import math

        from library import train_util

        class RowCountDataset(train_util.BaseDataset):
            def __init__(self):
                super().__init__((512, 512), 1.0, False, False)
                self.image_data = {}
                self.num_train_images = sum(bucket_sizes)
                self.num_reg_images = self.num_val_images = 0
                self.batch_buckets = [[f"b{b}_{i}" for i in range(n)] for b, n in enumerate(bucket_sizes)]
                self.batch_bucket_jitter_resos = [None] * len(self.batch_buckets)
                self.batch_bucket_weights = [None] * len(self.batch_buckets)
                self.buckets_indices = [train_util.BucketBatchIndex(b, batch_size, k)
                                        for b, bucket in enumerate(self.batch_buckets)
                                        for k in range(math.ceil(len(bucket) / batch_size))]
                self.all_buckets_indices = list(self.buckets_indices)  # as make_buckets
                self.set_seed(seed)
                self.shuffle_buckets()
                self._length = len(self.buckets_indices)
                self.epoch_at_fetch = []  # (epoch the trainer is in, the dataset's epoch when the item was read)

            def __getitem__(self, index):
                self.sync_epoch()  # the first statement of the real __getitem__ (asserted below)
                self.epoch_at_fetch.append(self.current_epoch)
                bbi = self.buckets_indices[index]
                bucket = self.batch_buckets[bbi.bucket_index]
                keys = bucket[bbi.batch_index * bbi.bucket_batch_size:(bbi.batch_index + 1) * bbi.bucket_batch_size]
                return {"latents": torch.zeros(len(keys), 1), "captions": list(keys),
                        "bbi": [(bbi.bucket_index, bbi.batch_index)] * len(keys)}

        return RowCountDataset()


@pytest.mark.parametrize("attached", [True, False])
def test_epoch_shuffle_precedes_the_first_fetch(attached):
    """train_util.collator_class sets the dataset's epoch after the item was fetched (torch collates after dataset[idx]),
    so without the shared epoch attached the first bucket batch of an epoch is read under the previous epoch's bucket
    permutation (one batch duplicated, another missed: the rows per epoch, and with --step_rows the packs, vary). With
    attach_epoch_value (train_network.py, before the DataLoader is built) __getitem__ shuffles first: every bucket batch
    exactly once per epoch, the pack count equals the formula in every epoch."""
    import inspect
    from collections import Counter
    from multiprocessing import Value

    from library import train_util

    for cls in (train_util.BaseDataset, train_util.ControlNetDataset):  # the real __getitem__s start with the sync
        body = inspect.getsource(cls.__getitem__).split("\n", 1)[1].lstrip()
        assert body.startswith("self.sync_epoch()"), cls
    sizes = [3, 8, 1, 2, 6, 4, 5, 9, 2, 7, 1, 1, 8, 3, 6, 5, 2, 4, 7, 1]  # 85 rows, 26 ragged bucket batches
    ds = _RowCountDataset(sizes, batch_size=4, seed=7)
    group = train_util.DatasetGroup([ds])
    rows_formula = micro_batch.dataset_rows_per_epoch(group)
    assert rows_formula == 85
    current_epoch, current_step = Value("i", 0), Value("i", 0)
    if attached:
        group.attach_epoch_value(current_epoch)
    collator = train_util.collator_class(current_epoch, current_step, group)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)
    loader = torch.utils.data.DataLoader(group, batch_size=1, shuffle=True, collate_fn=collator, num_workers=0, generator=gen)
    packer = micro_batch.RowPacker(loader, 8, rows_formula)
    expected = {(bi.bucket_index, bi.batch_index): min(4, len(ds.batch_buckets[bi.bucket_index]) - bi.batch_index * 4)
                for bi in ds.all_buckets_indices}
    stale_first_fetch, mismatched_epochs = 0, 0
    for epoch in range(1, 7):
        current_epoch.value = epoch  # as the trainer does before iterating the loader
        ds.epoch_at_fetch.clear()
        packs = list(packer)
        got = Counter(b for p in packs for sb in p for b in sb["bbi"])
        if ds.epoch_at_fetch[0] != epoch:
            stale_first_fetch += 1
        if got != Counter(expected) or sum(p.rows for p in packs) != rows_formula or len(packs) != len(packer):
            mismatched_epochs += 1
        assert all(e == epoch for e in ds.epoch_at_fetch[1:]), (epoch, ds.epoch_at_fetch[:3])
    if attached:
        assert stale_first_fetch == 0 and mismatched_epochs == 0
    else:
        assert stale_first_fetch == 6 and mismatched_epochs >= 1  # every epoch's first batch: the previous permutation
