"""CPU tests for the budget estimators of the MOSS full fine-tune (library/moss/ft_budget.py; design
docs/moss_full_finetune_design.md 2.1 GPU, 2.2 host RAM, 2.3 disk, 3.5 throughput; test plan 5 "test_moss_ft_budget.py").

test_numbers_match_design_tables: ft_budget reproduces the tables to +-0.05 GiB (GB where the table says GB) / +-0.1 s.
The 2.1 and 2.2 tables are read from the design document itself, so a change to the estimator also changes the document
(and the other way round); 2.3 and 3.5 are irregular text and are copied here with their section named.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_budget.py -q -p no:cacheprovider
"""
import math
import os
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest

from library.moss import ft_budget

DESIGN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "moss_full_finetune_design.md")
GIB, MIB = 1 << 30, 1 << 20


def design_section(title: str) -> str:
    with open(DESIGN, "r", encoding="utf-8") as handle:
        text = handle.read()
    start = text.index(title)
    end = text.find("\n### ", start + len(title))
    return text[start:end if end > 0 else len(text)]


def design_table(section: str):
    """(header cells, {row label: cells}) of the first markdown table in a section."""
    lines = [line.strip() for line in section.splitlines() if line.strip().startswith("|")]
    split = lambda line: [cell.strip() for cell in line.strip("|").split("|")]  # noqa: E731
    header = split(lines[0])
    rows = {}
    for line in lines[2:]:
        cells = split(line)
        rows[cells[0].strip("*").strip()] = cells[1:]
    return header, rows


def number(cell: str):
    """(value, tolerance) of a table cell: '**14.6**', '≤0.5', '~85', '18.93', '-' (0), '~0.01 (factored rows/cols)'.
    A '~' integer is good to +-0.5, everything else to +-0.05 (the tables round to 0.1 or 0.01 GiB)."""
    text = cell.replace("*", "").replace(",", "").strip()
    if text.startswith("-"):
        return 0.0, 0.0
    approx = text.startswith("~")
    match = re.match(r"[~≤]?\s*([0-9]+(?:\.[0-9]+)?)", text)
    assert match, cell
    value = float(match.group(1))
    return value, (0.5 if approx and "." not in match.group(1) else 0.05)


def row(rows, prefix):
    """The row whose label is `prefix`, else the one row whose label starts with it."""
    found = [cells for label, cells in rows.items() if label == prefix]
    found = found or [cells for label, cells in rows.items() if label.startswith(prefix)]
    assert len(found) == 1, (prefix, list(rows))
    return found[0]


# ---- 1.1: the constants are the model's ----------------------------------------------------------------------------
def test_model_constants_match_meta_model():
    shapes = ft_budget.moss_param_shapes()
    assert len(shapes) == 462
    total = sum(math.prod(shape) for shape in shapes.values())
    joint = sum(math.prod(shape) for name, shape in shapes.items() if ft_budget.is_joint_linear(name))
    assert (total, joint, total - joint) == (ft_budget.TOTAL_PARAMS, ft_budget.JOINT_LINEAR_PARAMS, ft_budget.OTHER_PARAMS)
    assert ft_budget.TOTAL_PARAMS == 5_082_308_608 and ft_budget.JOINT_LINEAR_PARAMS == 42 * 113_246_208
    sizes = ft_budget.ModelSizes.from_shapes(shapes)
    assert sizes == ft_budget.ModelSizes()  # the defaults are the full model, nothing frozen
    assert sizes.came_factored_bytes == ft_budget.CAME_FACTORED_BYTES


# ---- 2.1 GPU ---------------------------------------------------------------------------------------------------------
def test_gpu_table_matches_design():
    header, rows = design_table(design_section("### 2.1"))
    columns = header[1:]
    assert len(columns) == 6
    checked = 0
    for index, column in enumerate(columns):
        match = re.match(r"(\d+)²\s*b(\d+)", column)
        resolution, batch = int(match.group(1)), int(match.group(2))
        image_tokens = (resolution // 16) ** 2
        gpu = ft_budget.gpu_budget(image_tokens, batch)
        assert gpu.joint_tokens == number(row(rows, "joint tokens")[index])[0]
        for prefix, got in (("compute weights", gpu.weights), ("checkpointed block inputs", gpu.block_inputs),
                            ("one block's recompute", gpu.recompute), ("grads in flight", gpu.grads_in_flight),
                            ("rope, modulation", gpu.misc), ("total allocated", gpu.allocated),
                            ("device with", gpu.device)):
            want, tolerance = number(row(rows, prefix)[index])
            assert got / GIB == pytest.approx(want, abs=tolerance), (column, prefix, got / GIB, want)
            checked += 1
    assert checked == 42


def test_gpu_options():
    plain = ft_budget.gpu_budget(4096, 1)
    # --moss_ckpt_reuse_attn: 6,240 B / token / block, +1.07 GiB at 1024 b1 and +4.3 GiB at 1024 b4 (design 2.1)
    reuse = ft_budget.gpu_budget(4096, 1, reuse_attn=True)
    assert reuse.stored_attention == 42 * 6240 * 4396 and reuse.stored_attention / GIB == pytest.approx(1.07, abs=0.005)
    assert ft_budget.gpu_budget(4096, 4, reuse_attn=True).stored_attention / GIB == pytest.approx(4.3, abs=0.05)
    assert reuse.allocated == plain.allocated + reuse.stored_attention
    # its budget: n = floor(budget / per-block bytes) blocks store
    per_block = 6240 * 4396 * 4
    capped = ft_budget.gpu_budget(4096, 4, reuse_attn=True, reuse_budget_gib=1.0)
    assert capped.stored_attention == (GIB // per_block) * per_block
    assert ft_budget.gpu_budget(4096, 1, reuse_attn=True, reuse_budget_gib=100.0).stored_attention == reuse.stored_attention
    # --unsloth_offload_checkpointing: the block-input row moves to pinned host RAM
    unsloth = ft_budget.gpu_budget(4096, 4, unsloth_offload=True)
    assert unsloth.block_inputs == 0 and unsloth.offloaded_block_inputs == ft_budget.gpu_budget(4096, 4).block_inputs
    # without --moss_stream_fp32 the block inputs halve and every weight is bf16 (9.47 GiB)
    bf16_stream = ft_budget.gpu_budget(4096, 1, stream_fp32=False)
    assert bf16_stream.block_inputs * 2 == plain.block_inputs and bf16_stream.weights / GIB == pytest.approx(9.47, abs=0.005)
    # fp32 compute (mixed_precision "no"): all weights fp32, about 24.7 GiB at 1024 b1 (design 2.1)
    fp32 = ft_budget.gpu_budget(4096, 1, mixed_precision="no")
    assert fp32.weights / GIB == pytest.approx(18.93, abs=0.005) and fp32.allocated / GIB == pytest.approx(24.7, abs=0.5)
    # modes a and c are the same on the GPU; b adds its small Adafactor state (design 2.1: "about 0.03 GiB")
    fused = ft_budget.gpu_budget(4096, 4, mode="adafactor_fused")
    assert ft_budget.gpu_budget(4096, 4, mode="came_host") == ft_budget.gpu_budget(4096, 4)
    assert fused.allocated - ft_budget.gpu_budget(4096, 4).allocated == fused.optimizer_state
    assert 0 < fused.optimizer_state / GIB < 0.05


# ---- 2.2 host RAM ----------------------------------------------------------------------------------------------------
def test_host_table_matches_design():
    header, rows = design_table(design_section("### 2.2"))
    assert header[1:] == ["adamw_host, GA>1", "adamw_host, GA=1", "came_host, GA>1", "adafactor_fused"]
    budgets = [ft_budget.host_budget("adamw_host", 2), ft_budget.host_budget("adamw_host", 1),
               ft_budget.host_budget("came_host", 2), ft_budget.host_budget("adafactor_fused", 1)]
    for index, host in enumerate(budgets):
        for prefix, got in (("fp32 master", host.master), ("exp_avg", host.exp_avg), ("exp_avg_sq", host.exp_avg_sq),
                            ("fp32 grad accumulator", host.accumulator), ("pinned rings", host.rings),
                            ("process", host.process), ("total", host.total)):
            want, tolerance = number(row(rows, prefix)[index])
            assert got / GIB == pytest.approx(want, abs=tolerance), (header[index + 1], prefix, got / GIB, want)
        assert host.required == host.total + 8 * GIB  # refused below total + 8 GiB available (2.2, 4.5)
    # the accumulator: GA > 1 or --moss_ft_clip_mode exact with clipping on (design 2.2 / 3.1; the rule of
    # ft_host.needs_accumulator, so the budget counts what the pipeline allocates). max_grad_norm defaults to kohya's 1.0
    assert ft_budget.host_budget("adamw_host", 1, clip_mode="exact").accumulator == budgets[0].accumulator
    assert ft_budget.host_budget("adamw_host", 1, clip_mode="exact", max_grad_norm=0.0).accumulator == 0
    assert ft_budget.needs_host_accumulator("adamw_host", 1, "stale") is False
    assert ft_budget.needs_host_accumulator("adamw_host", 2, "stale", max_grad_norm=0.0) is True
    assert ft_budget.needs_host_accumulator("came_host", 1, "exact") is True
    assert ft_budget.needs_host_accumulator("came_host", 1, "exact", max_grad_norm=0.0) is False
    assert ft_budget.needs_host_accumulator("adafactor_fused", 1, "exact") is False
    from library.moss import ft_host  # T2's pipeline: the same rule (overlap on, as moss_train.py runs it)

    for ga in (1, 2):
        for clip_mode in ("stale", "exact"):
            for norm in (0.0, 1.0):
                assert ft_budget.needs_host_accumulator("adamw_host", ga, clip_mode, norm) == ft_host.needs_accumulator(
                    ga, clip_mode, norm)
    # pinned checkpoint inputs (unsloth) and in-memory caches add to the host
    extra = ft_budget.host_budget("adamw_host", 2, pinned_checkpoint_bytes=3 * GIB, memory_cache_bytes=GIB)
    assert extra.total == budgets[0].total + 4 * GIB
    rings = ft_budget.host_budget("adamw_host", 2, ring_mib=2048, upload_ring_mib=256)
    assert rings.rings == (2048 + 256) * MIB


# ---- review fixes: torch's pinned host allocator, DataLoader workers ---------------------------------------------------
def test_pinned_allocations_round_up_to_a_power_of_two():
    """torch 2.14's CachingHostAllocator (ATen/core/CachingHostAllocator.h) rounds every pinned request up to the next
    power of two (pinned_max_round_threshold / pinned_max_cached_size default to SIZE_MAX), so a 1.5 GiB ring
    page-locks 2 GiB. The budget counts what the allocator holds."""
    assert ft_budget.pinned_alloc_bytes(1024 * MIB) == 1024 * MIB
    assert ft_budget.pinned_alloc_bytes(1536 * MIB) == 2048 * MIB
    assert ft_budget.pinned_alloc_bytes(216_072_192) == 256 * MIB
    assert ft_budget.pinned_alloc_bytes(1) == 1 and ft_budget.pinned_alloc_bytes(0) == 0
    rings = ft_budget.host_budget("adamw_host", 2, ring_mib=1536, upload_ring_mib=768)
    assert rings.rings == (2048 + 1024) * MIB


def test_unsloth_pinned_checkpoints_follow_the_pinned_allocator():
    """Review fix: --unsloth_offload_checkpointing saves each joint block's input with .to("cpu", non_blocking=True):
    a pinned block from torch's caching host allocator, rounded up to a power of two and, once freed, kept in the free
    list of its power-of-two class for the rest of the run. Every class the dataset's block inputs reach keeps 42
    blocks: a short last batch of a bucket (b = 1 .. batch - 1) adds its classes. Design 2.2 counted one exact-size copy
    of the largest bucket (8.45 GiB at 1024^2 batch 4)."""
    # 1024^2, T 300, fp32 stream: 12,288 B per joint token; b4 = 216,072,192 B and b3 = 162 MB -> the 256 MiB class,
    # b2 = 108 MB -> 128 MiB, b1 = 54 MB -> 64 MiB: 42 x 448 MiB = 18.4 GiB
    assert ft_budget.unsloth_pinned_bytes([(4096, 4)]) == 42 * (256 + 128 + 64) * MIB
    assert ft_budget.unsloth_pinned_bytes([(4096, 4)]) / GIB == pytest.approx(18.4, abs=0.05)
    assert ft_budget.unsloth_pinned_bytes([(4096, 1)]) == 42 * 64 * MIB
    assert ft_budget.unsloth_pinned_bytes([(4096, 1)], stream_bytes=2) == 42 * 32 * MIB  # bf16 stream
    # a second tier: 512^2 (1,324 joint tokens) at batch 4 reaches the 16 / 32 / 64 MiB classes; 64 MiB is shared
    assert ft_budget.unsloth_pinned_bytes([(4096, 1), (1024, 4)]) == 42 * (16 + 32 + 64) * MIB
    assert ft_budget.unsloth_pinned_bytes([]) == 0


def test_dataloader_workers_add_host_ram():
    """Review fix: the ~8 GiB "process" row of design 2.2 covers the main process, the CUDA host side, the CPU VAE and up
    to 4 DataLoader workers; each worker above 4 (a spawned process importing torch and moss_train.py again: 0.81 GiB
    RSS measured on the CPU, a lower bound) adds 1 GiB. kohya's default is 8 workers."""
    base = ft_budget.host_budget("adamw_host", 2)
    assert base.process == ft_budget.PROCESS_BYTES == 8 * GIB
    assert ft_budget.host_budget("adamw_host", 2, workers=4).process == 8 * GIB
    assert ft_budget.host_budget("adamw_host", 2, workers=8).process == 12 * GIB
    assert ft_budget.host_budget("adamw_host", 2, workers=8).total == base.total + 4 * GIB
    assert ft_budget.host_budget("adafactor_fused", 1, workers=6).process == 10 * GIB


def test_frozen_parameters_shrink_the_host_pools():
    shapes = ft_budget.moss_param_shapes()
    frozen = {name for name in shapes if name.startswith("layers.0.")}
    sizes = ft_budget.ModelSizes.from_shapes(shapes, frozen)
    block = 113_246_208 + 2 * 3072 + 2 * 128
    assert sizes.trainable == ft_budget.TOTAL_PARAMS - block and sizes.total == ft_budget.TOTAL_PARAMS
    host = ft_budget.host_budget("adamw_host", 2, sizes=sizes)
    assert host.master == host.exp_avg == host.exp_avg_sq == host.accumulator == sizes.trainable * 4
    # the state's master file keeps every Base key (frozen ones too), the moments only the trained ones
    full = ft_budget.state_bytes("adamw_host")
    assert full - ft_budget.state_bytes("adamw_host", sizes) == 2 * 4 * block


# ---- 2.3 disk --------------------------------------------------------------------------------------------------------
def test_disk_table_matches_design():
    # "Trained DiT, bf16 (Base layout) | 10.16 GB per save | --save_precision float gives 20.3 GB"
    assert ft_budget.dit_save_bytes(save_precision="bf16") / 1e9 == pytest.approx(10.16, abs=0.05)
    assert ft_budget.dit_save_bytes(save_precision="fp16") == ft_budget.dit_save_bytes(save_precision="bf16")
    assert ft_budget.dit_save_bytes(save_precision="float") / 1e9 == pytest.approx(20.3, abs=0.05)
    # "Training state, adamw_host | 61.0 GB" / "Training state, came_host | 40.7 GB"
    assert ft_budget.state_bytes("adamw_host") / 1e9 == pytest.approx(61.0, abs=0.05)
    assert ft_budget.state_bytes("came_host") / 1e9 == pytest.approx(40.7, abs=0.05)
    assert ft_budget.state_bytes("adafactor_fused") < ft_budget.state_bytes("came_host")
    # "TE cache | 8 KiB per text token per caption variant | T = 300: 2.46 MB per image per variant"
    assert ft_budget.te_cache_bytes(1, 1, text_tokens=1) // 1024 == 8
    assert ft_budget.te_cache_bytes(1, 1) / 1e6 == pytest.approx(2.46, abs=0.005)
    # "16k images | 39 GB x variants (18 variants = 0.7 TB)", "56k images | 138 GB x variants (18 variants = 2.5 TB)"
    assert ft_budget.te_cache_bytes(16_000, 1) / 1e9 == pytest.approx(39, abs=0.5)
    assert ft_budget.te_cache_bytes(16_000, 18) / 1e12 == pytest.approx(0.7, abs=0.05)
    assert ft_budget.te_cache_bytes(56_000, 1) / 1e9 == pytest.approx(138, abs=0.5)
    assert ft_budget.te_cache_bytes(56_000, 18) / 1e12 == pytest.approx(2.5, abs=0.05)
    assert ft_budget.te_cache_bytes(10, 18) == ft_budget.te_cache_bytes(10, 1) * 18
    assert ft_budget.te_cache_bytes(10, 0) == ft_budget.te_cache_bytes(10, 1)  # no variants = the canonical caption
    assert ft_budget.te_cache_bytes(10, 1, dtype="fp32") > ft_budget.te_cache_bytes(10, 1) * 1.99
    # "Latents at 1024² (128x64x64) | 2 MiB fp32, 1 MiB with --cache_latents_dtype bf16 | double that with flip_aug"
    assert ft_budget.latent_cache_bytes(1, 4096, dtype="fp32") == 2 * MIB
    assert ft_budget.latent_cache_bytes(1, 4096, dtype="auto") == 2 * MIB
    assert ft_budget.latent_cache_bytes(1, 4096, dtype="bf16") == MIB
    assert ft_budget.latent_cache_bytes(1, 4096, dtype="bf16", flip_aug=True) == 2 * MIB
    # "16k images | 16-32 GB", "56k images | 56-112 GB": 1-2 MiB per image (the table counts a MiB as a MB)
    assert ft_budget.latent_cache_bytes(16_000, 4096, dtype="bf16") == 16_000 * MIB
    assert ft_budget.latent_cache_bytes(56_000, 4096, dtype="fp32") == 112_000 * MIB


# ---- 3.5 throughput --------------------------------------------------------------------------------------------------
# design 3.5: run shape -> (GPU s, host s, tail s, s/step (low, high), images/s, tok/s); every row is 8 images per step
THROUGHPUT_TABLE = {
    (1, 8): (8.8, 7 * 1.1 + 4.0, 3.2, (12.0, 13.0), 0.63, 2750),  # micro-steps host-paced
    (2, 4): (8.8, 3 * 1.1 + 4.0, 2.3, (11.1, 11.1), 0.72, 3150),
    (4, 2): (8.8, 1.1 + 4.0, 0.7, (9.5, 9.7), 0.83, 3650),
    (8, 1): (8.8, 3.6, 0.0, (8.9, 8.9), 0.9, 3950),  # phase 2 (activation offload)
}


@pytest.mark.parametrize("shape", list(THROUGHPUT_TABLE))
def test_throughput_table_matches_design(shape):
    batch, ga = shape
    gpu, host, tail, (low, high), images_per_s, tokens_per_s = THROUGHPUT_TABLE[shape]
    got = ft_budget.throughput(batch, ga)
    assert got.gpu == pytest.approx(gpu, abs=0.1) and got.host == pytest.approx(host, abs=0.1)
    assert got.tail == pytest.approx(tail, abs=0.1)
    assert low - 0.1 - 1e-9 <= got.step <= high + 0.1 + 1e-9
    assert got.images_per_s == pytest.approx(batch * ga / got.step) and got.tokens_per_s == pytest.approx(
        batch * ga * 4396 / got.step)
    # the rates are "~" values in the design; batch 1 x GA 8 is the loosest (its s/step is the range 12-13)
    assert got.images_per_s == pytest.approx(images_per_s, rel=0.07)
    assert got.tokens_per_s == pytest.approx(tokens_per_s, rel=0.07)
    assert got.host_paced == (batch == 1)


def test_throughput_details():
    # GPU time per image is pinned to 1.1 s at 1024² (4,396 joint tokens) and grows with tokens (attention ~ T²)
    assert ft_budget.seconds_per_image(4096) == pytest.approx(1.1)
    t1536, t2048 = ft_budget.seconds_per_image(9216), ft_budget.seconds_per_image(16384)
    assert 1.1 * 9516 / 4396 < t1536 < t2048 and t2048 > 1.1 * 16684 / 4396
    # adafactor_fused: no host work, GPU-bound, about 4.5 s per 4 images at 1024 (design 3.2)
    fused = ft_budget.throughput(4, 1, mode="adafactor_fused")
    assert fused.step == pytest.approx(4.5, abs=0.1) and fused.host == 0 and fused.tail == 0
    # frozen parameters shorten the host passes in proportion
    half = ft_budget.throughput(4, 2, trainable_fraction=0.5)
    assert half.host == pytest.approx((1.1 + 4.0) / 2)
    # the 8-epoch wall time at batch 4 x GA 2 (design 3.5: 16k images ~43 h, 56k ~150 h)
    hours = lambda images: images * 8 / ft_budget.throughput(4, 2).images_per_s / 3600  # noqa: E731
    assert hours(16_000) == pytest.approx(43, rel=0.05) and hours(56_000) == pytest.approx(150, rel=0.05)
