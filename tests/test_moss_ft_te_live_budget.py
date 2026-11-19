"""CPU tests of the live text encoder's estimates in library/moss/ft_budget.py (docs/moss_ft_te_live_fp16_plan.md 1.5):
host RAM per host mode, the device ring / pass / held bytes, the allocated max rule (the pass runs between
micro-steps), the plan's numbers at 1024^2 batch 1, the host totals of adamw_host at GA 1 / GA > 1, the pass time
estimate in the step, and no TE cache bytes in live mode. The constants against the real encoder's layout:
test_moss_ft_te_live_weights.test_real_encoder_layout_matches_the_budget_constants.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_te_live_budget.py -q -p no:cacheprovider
"""
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest

from library.moss import ft_budget as B

GIB, MIB = 1 << 30, 1 << 20


def test_host_bytes_per_mode():
    assert B.te_live_host_bytes("copy", 2) == (B.TE_LAYER_BYTES_TOTAL, GIB)
    assert B.te_live_host_bytes("copy", 4) == (B.TE_LAYER_BYTES_TOTAL, 2 * GIB)
    assert B.te_live_host_bytes("pinned", 2) == (B.TE_LAYER_BYTES_TOTAL, 0)  # registered in place, no staging
    assert B.te_live_host_bytes("mmap", 2) == (0, GIB)  # the OS file cache serves the layers
    copy = sum(B.te_live_host_bytes("copy", 2)) / GIB
    assert copy == pytest.approx(13.889, abs=1e-3)  # plan 1.5: +13.9 GiB in copy mode


def test_device_ring_pass_and_held():
    ring, peak = B.te_live_gpu_bytes(32, 300, 2)
    assert ring == 2 * B.TE_SLOT_BYTES and ring / MIB == pytest.approx(833.2, abs=0.1)
    assert peak - ring == 2 * 32 * 300 * 4096 * 2 + B.te_row_temp_bytes(300)
    assert peak / GIB == pytest.approx(1.02, abs=0.02)  # plan 1.5: ~1.0 GiB for 32 rows of 300 tokens
    assert B.te_live_gpu_bytes(32, 300, 4)[0] == 4 * B.TE_SLOT_BYTES
    held = B.te_held_bytes(32, 1, 300, 4096)
    assert held == 32 * 300 * 8192 + 32 * 128 * 4096 * 4  # 75 MiB of text + 64 MiB of fp32 latents
    assert B.te_held_bytes(32, 1, 300, 4096, "bf16") == 32 * 300 * 8192 + 32 * 128 * 4096 * 2


def test_allocated_takes_the_larger_of_step_and_pass():
    plain = B.gpu_budget(4096, 1)
    live = B.gpu_budget(4096, 1, te_live=B.TeLive(ahead=32))
    assert plain.te_pass == plain.te_held == 0 and plain.allocated == plain.dit_step
    assert live.dit_step == plain.dit_step
    assert live.allocated == plain.allocated + live.te_held  # the pass never adds to a micro-step's peak
    assert live.weights + B.MISC_FIXED_BYTES + live.te_pass < live.dit_step
    assert live.allocated / GIB == pytest.approx(plain.allocated / GIB + 0.136, abs=0.01)
    huge = B.gpu_budget(4096, 1, te_live=B.TeLive(ahead=2000))  # a pass larger than a micro-step
    assert huge.allocated == huge.weights + B.MISC_FIXED_BYTES + huge.te_pass + huge.te_held
    assert huge.allocated > huge.dit_step


def test_host_totals_of_adamw_host():
    live = B.TeLive(host="copy")
    ga1, ga1_live = B.host_budget("adamw_host", 1), B.host_budget("adamw_host", 1, te_live=live)
    ga2, ga2_live = B.host_budget("adamw_host", 2), B.host_budget("adamw_host", 2, te_live=live)
    assert ga1.total / GIB == pytest.approx(66.3, abs=0.1) and ga1_live.total / GIB == pytest.approx(80.2, abs=0.1)
    assert ga2.total / GIB == pytest.approx(85.2, abs=0.1) and ga2_live.total / GIB == pytest.approx(99.1, abs=0.1)
    assert ga2_live.required / GIB == pytest.approx(107.1, abs=0.1)
    mmap = B.host_budget("adamw_host", 2, te_live=B.TeLive(host="mmap"))
    assert mmap.total - ga2.total == GIB and mmap.text_encoder == 0
    assert B.host_budget("adafactor_fused", 1, te_live=live).text_encoder == B.TE_LAYER_BYTES_TOTAL


def test_pass_time_estimate_and_step():
    per_image = B.te_seconds_per_image(B.TeLive(ahead=32), 1)
    assert 0.05 <= per_image <= 0.1  # plan 1.5: ~0.06-0.1 s per image (estimate)
    assert B.te_seconds_per_pass(32, 300, "pinned") <= B.te_seconds_per_pass(32, 300, "copy")
    plain = B.throughput(1, 2)
    live = B.throughput(1, 2, te_live=B.TeLive(ahead=32))
    assert live.te_s_per_image == pytest.approx(per_image)
    assert live.step == pytest.approx(plain.step + 2 * per_image)
    assert B.throughput(1, 1, mode="adafactor_fused", te_live=B.TeLive()).step > B.throughput(1, 1,
                                                                                            mode="adafactor_fused").step


def test_settings_from_args():
    assert B.te_live_settings(types.SimpleNamespace(moss_te_live=False)) is None
    args = types.SimpleNamespace(moss_te_live=True, moss_te_live_ahead=5, gradient_accumulation_steps=2,
                                 moss_te_live_host="mmap", moss_te_live_ring_slots=4)
    assert B.te_live_settings(args) == B.TeLive(host="mmap", ring_slots=4, ahead=6)


def test_no_te_cache_in_live_mode(tmp_path):
    infos = {f"img{i}": types.SimpleNamespace(bucket_reso=(1024, 1024), jitter_bucket_info=None,
                                              absolute_path=str(tmp_path / f"img{i}.png")) for i in range(4)}
    group = types.SimpleNamespace(datasets=[types.SimpleNamespace(image_data=infos, caption_aug_variant_count=0,
                                                                  latents_aug_variant_count=0, subsets=[])])
    cached = types.SimpleNamespace(cache_text_encoder_outputs_to_disk=True, cache_latents_to_disk=True)
    live = types.SimpleNamespace(moss_te_live=True, cache_latents_to_disk=True)
    in_memory = types.SimpleNamespace(moss_te_live=True, cache_latents_to_disk=False)
    images, te, latent, memory, _ = B.cache_estimate(cached, group, exists=lambda p: False)
    assert te == 4 * B.te_cache_bytes(1) and memory == 0
    images, te, latent_live, memory, _ = B.cache_estimate(live, group, exists=lambda p: False)
    assert te == 0 and memory == 0 and latent_live == latent
    _, te, _, memory, _ = B.cache_estimate(in_memory, group, exists=lambda p: False)
    assert te == 0 and memory == latent  # only the latents are held in RAM


def test_mmap_from_disk_is_priced_at_the_disk_rate():
    """Review fix: mmap with the layers out of the file cache reads 13.8 GB per pass from the disk."""
    live = B.TeLive(host="mmap", ahead=32)
    hot, cold = B.te_seconds_per_pass(32, 300, "mmap"), B.te_seconds_per_pass(32, 300, "mmap", from_disk=True)
    assert hot == B.te_seconds_per_pass(32, 300, "copy")
    assert cold == pytest.approx(B.TE_LAYER_BYTES_TOTAL / B.TE_DISK_BYTES_PER_S) and cold > 3 * hot
    assert B.te_seconds_per_image(live, 1, from_disk=True) == pytest.approx(cold / 32)
    assert B.te_seconds_per_pass(32, 300, "copy", from_disk=True) == hot  # only mmap reads the file


def test_page_locked_total_and_warning(monkeypatch):
    """Review fix: the page-locked parts (rings, unsloth checkpoint inputs, TE staging, pinned TE layers) are summed
    and warned about on Windows past half of physical RAM."""
    unsloth = 18 * GIB
    copy = B.host_budget("adamw_host", 1, pinned_checkpoint_bytes=unsloth, te_live=B.TeLive(host="copy"))
    pinned = B.host_budget("adamw_host", 1, pinned_checkpoint_bytes=unsloth, te_live=B.TeLive(host="pinned"))
    assert copy.pinned == copy.rings + unsloth + GIB and copy.te_host == "copy"
    assert pinned.pinned == pinned.rings + unsloth + B.TE_LAYER_BYTES_TOTAL
    assert B.host_budget("adamw_host", 1).pinned == B.host_budget("adamw_host", 1).rings
    warnings = []
    monkeypatch.setattr(B.logger, "warning", lambda m, *a, **k: warnings.append(str(m)))
    memory = lambda total: (lambda: types.SimpleNamespace(available=total, total=total))  # noqa: E731
    B._pinned_warning(types.SimpleNamespace(host=pinned), memory(64 * GIB), windows=True)  # 33.4 GiB > 32
    assert len(warnings) == 1 and "page-locked" in warnings[0] and "TE layers" in warnings[0]
    B._pinned_warning(types.SimpleNamespace(host=pinned), memory(128 * GIB), windows=True)
    B._pinned_warning(types.SimpleNamespace(host=pinned), memory(64 * GIB), windows=False)
    assert len(warnings) == 1
