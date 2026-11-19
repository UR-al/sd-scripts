"""CPU trainer loop with the fp16 pipeline helpers; no model files or GPU."""
import types

import pytest
import torch

import moss_train
import test_moss_ft_fp16_pipeline as F
from _moss_ft_carousel_common import build_model, make_engine, carousel_loss
from library.moss import ft_args, ft_scaler


def loop(monkeypatch, kind, poison=True, max_steps=4):
    if kind == "carousel":
        model = build_model()
        pipe = make_engine(model, k=2, period=1, select="window")
        opt, pool, slots = pipe.optimizer, pipe.pool, pipe.slots
        loss_fn = lambda: carousel_loss(model, 7)
    else:
        model = F.C.build_model(seed=7)
        opt = F.make_optimizer("adamw", F.C.trainable(model))
        kw = dict(max_grad_norm=0.05, clip_mode="exact") if kind == "window" else {}
        pipe, pool, slots = F.C.make_pipeline(F.C.trainable(model), opt, 2, inline=True, **kw)
        loss_fn = F.C.model_windows(model, [[7]])[0][0]
    model.last_text_tokens = 0
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.5)
    scaler = ft_scaler.DynamicLossScaler(init_scale=1024)
    logs, samples, saves, states, windows = [], [], [], [], []
    args = moss_train.setup_parser().parse_args([])
    for k, v in dict(max_train_steps=max_steps, gradient_accumulation_steps=2, max_data_loader_n_workers=0,
                     mixed_precision="no", moss_ft_log_every=100, output_name="tiny", save_state=True,
                     save_every_n_steps=1, save_every_n_epochs=1, moss_ft_max_skipped_steps=8).items():
        setattr(args, k, v)

    class Dataset(torch.utils.data.Dataset):
        num_train_images = 4

        def __len__(self):
            return 4

        def __getitem__(self, i):
            return {"latents": torch.zeros(1, 1, 2, 2)}

        def set_current_strategies(self):
            pass

        def set_max_train_steps(self, n):
            pass

    accelerator = types.SimpleNamespace(device=torch.device("cpu"), is_main_process=True,
                                        is_local_main_process=False, prepare=lambda x: x, print=lambda *a: None,
                                        wait_for_everyone=lambda: None,
                                        log=lambda values, step: logs.append((step, values)))
    run = types.SimpleNamespace(args=args, accelerator=accelerator, dit=model, optimizer=opt, pipeline=pipe,
                                scaler=scaler, num_batches=4, steps_per_epoch=2, slots=slots, pool=pool,
                                mode=ft_args.CAROUSEL if kind == "carousel" else ft_args.ADAMW_HOST,
                                lr_scheduler=scheduler, lr_schedule=lambda: None, te_feed=None, validator=None,
                                sample=lambda epoch, step: samples.append((epoch, step)), validate=lambda *a, **k: None,
                                save_dit=lambda name, step, epoch: saves.append((name, step, epoch)),
                                remove_dit=lambda name: None, save_state=lambda name, p: states.append((name, p)))
    calls = 0
    handles = []

    def loss(*a, **kw):
        nonlocal calls
        for h in handles:
            h.remove()
        handles.clear()
        if calls % 2 == 0:
            windows.append((scheduler.last_epoch, [g["lr"] for g in opt.param_groups],
                            getattr(pipe, "switches", 0), getattr(pipe, "active", ())))
        if poison and calls == 0:
            targets = [p for p in model.parameters() if p.requires_grad]
            if kind == "partial":
                targets = targets[:1]
            for p in targets:
                handles.append(p.register_hook(lambda g: torch.full_like(g, float("inf"))))
        calls += 1
        return loss_fn()

    monkeypatch.setattr(moss_train.ft_loss, "moss_ft_noise_and_loss", loss)
    monkeypatch.setattr(moss_train, "init_trackers", lambda *a: None)
    try:
        moss_train._train_loop(run, Dataset(), lambda rows: rows[0], types.SimpleNamespace(value=0),
                               types.SimpleNamespace(value=0), None, None)
    finally:
        for h in handles:
            h.remove()
        pipe.close()
    return types.SimpleNamespace(logs=logs, samples=samples, saves=saves, states=states, windows=windows,
                                 scheduler=scheduler, scaler=scaler)


@pytest.mark.parametrize("kind", ["window", "all", "carousel", "partial"])
def test_only_successful_updates_advance_step_lr_and_carousel(monkeypatch, kind):
    result = loop(monkeypatch, kind)
    expected = 4 if kind == "partial" else 3
    if kind != "partial":
        assert result.windows[1] == result.windows[0]  # LR, counter and active set survived the skipped window
    assert result.scheduler.last_epoch == expected
    assert result.scaler.overflow_windows == 1  # backoff still happens on skipped windows
    assert [s for s, values in result.logs if "loss" in values] == list(range(1, expected + 1))
    assert [s for e, s in result.samples if e is None] == list(range(1, expected + 1))
    assert [s for name, s, e in result.saves if "step" in name] == list(range(1, expected + 1))
    final = result.states[-1][1]
    assert (final.global_step, final.completed_epochs, final.epoch_step, final.images_seen) == (expected, 2, 0, 8)


@pytest.mark.parametrize("max_steps", [3, 4])
def test_dit_epoch_metadata_uses_one_based_epoch_for_every_save(monkeypatch, max_steps):
    result = loop(monkeypatch, "window", poison=False, max_steps=max_steps)
    expected = [("tiny-step00000001.safetensors", 1, 1),
                ("tiny-step00000002.safetensors", 2, 1),
                ("tiny-000001.safetensors", 2, 1),
                ("tiny-step00000003.safetensors", 3, 2)]
    if max_steps == 4:
        expected.append(("tiny-step00000004.safetensors", 4, 2))
    assert result.saves == expected + [("tiny.safetensors", max_steps, 2)]
