"""CPU tests: the validation loader is cycled without keeping its batches (review 2026-11-05, #08).

The trainers wrapped the accelerate-prepared validation DataLoader in itertools.cycle, which stores every item of its
first pass to replay it: every validation batch (moved to the training device by accelerate) stayed referenced for the
whole run. train_util.DataLoaderCycle starts a new pass over the loader instead and stores nothing.

No model, no GPU.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp=<scratch dir>
     tests/test_val_dataloader_cycle.py
"""
import gc
import os
import weakref

import pytest
import torch

from library import train_util

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Batch:
    def __init__(self, index):
        self.index = index


class _FreshBatches:
    """Like a DataLoader: every pass builds new batch objects and keeps none of them."""

    def __init__(self, n):
        self.n = n
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        for i in range(self.n):
            yield _Batch(i)

    def __len__(self):
        return self.n


def test_cycle_restarts_the_loader_instead_of_keeping_its_batches():
    loader = _FreshBatches(5)
    cycle = train_util.DataLoaderCycle(loader)
    refs, order = [], []
    for _ in range(13):
        batch = next(cycle)
        refs.append(weakref.ref(batch))
        order.append(batch.index)
        del batch
    gc.collect()

    assert order == [0, 1, 2, 3, 4] * 2 + [0, 1, 2]
    assert loader.passes == 3
    assert [ref for ref in refs if ref() is not None] == [], "the cycle must not keep yielded batches alive"


def test_cycle_over_a_torch_dataloader():
    loader = torch.utils.data.DataLoader(list(range(3)), batch_size=1, shuffle=False)
    cycle = train_util.DataLoaderCycle(loader)
    assert [int(next(cycle)) for _ in range(7)] == [0, 1, 2, 0, 1, 2, 0]
    assert iter(cycle) is cycle


def test_cycle_over_an_empty_loader_stops_like_itertools_cycle():
    cycle = train_util.DataLoaderCycle([])
    with pytest.raises(StopIteration):
        next(cycle)
    with pytest.raises(StopIteration):
        next(cycle)


@pytest.mark.parametrize("script", ["train_network.py", "sdxl_train.py"])
def test_trainers_do_not_cycle_the_validation_loader_with_itertools(script):
    with open(os.path.join(REPO, script), encoding="utf-8") as f:
        source = f.read()
    assert "itertools.cycle(val_dataloader)" not in source
    assert "train_util.DataLoaderCycle(val_dataloader)" in source


# --- an accelerate-prepared validation loader must not change the training loop's gradient sync


@pytest.fixture
def fresh_accelerate_state():
    from accelerate.state import AcceleratorState, GradientState

    AcceleratorState._reset_state(True)
    GradientState._reset_state()
    yield
    AcceleratorState._reset_state(True)
    GradientState._reset_state()


def _sync_pattern(validation_steps, n_val, accumulation=4, n_train=22, epochs=2, validate_every=2):
    """train_network's loop shape on CPU: a step-0 validation, `accelerator.accumulate` per micro-batch, and a validation
    of `validation_steps` batches every `validate_every` optimizer steps (no validation when validation_steps is None).

    Returns one character per micro-batch, 'S' when it was an optimizer step and '.' when it accumulated, then 'V' per
    validation and '|' per epoch end. 22 micro-batches per epoch with accumulation 4 end on a partial step, which only
    the training loader's end of epoch syncs.
    """
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=accumulation)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_loader = torch.utils.data.DataLoader(torch.arange(n_train, dtype=torch.float32).view(-1, 1), batch_size=1)
    val_loader = torch.utils.data.DataLoader(torch.zeros(n_val, 1), batch_size=1)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    val_loader = accelerator.prepare(val_loader)
    cycle = train_util.DataLoaderCycle(val_loader)

    def validate():
        with torch.no_grad():
            for _ in range(validation_steps):
                next(cycle)
        registered = accelerator.gradient_state.dataloader_references
        assert not any(ref is val_loader for ref in registered), "the validation loader stayed in accelerate's GradientState"

    if validation_steps is not None:
        validate()
    pattern, global_step = [], 0
    for _ in range(epochs):
        for batch in train_loader:
            with accelerator.accumulate(model):
                accelerator.backward(model(batch).sum())
                optimizer.step()
                optimizer.zero_grad()
            if not accelerator.sync_gradients:
                pattern.append(".")
                continue
            pattern.append("S")
            global_step += 1
            if validation_steps is not None and global_step % validate_every == 0:
                validate()
                pattern.append("V")
        pattern.append("|")
    return "".join(pattern)


EPOCH_WITHOUT_VALIDATION = "...S...S...S...S...S.S|"


@pytest.mark.parametrize(
    "validation_steps, n_val",
    [
        (3, 3),  # the default, len(val_dataloader) batches: each pass ends inside a validation
        (2, 5),  # --max_validation_steps below the loader length: passes stop mid-way and end at a later validation
    ],
)
def test_validation_keeps_the_training_gradient_sync(fresh_accelerate_state, validation_steps, n_val):
    assert _sync_pattern(None, n_val) == EPOCH_WITHOUT_VALIDATION * 2

    pattern = _sync_pattern(validation_steps, n_val)

    assert pattern.replace("V", "") == EPOCH_WITHOUT_VALIDATION * 2
    assert pattern == "...S...SV...S...SV...S.SV|" * 2
