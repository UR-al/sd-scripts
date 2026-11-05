"""CPU tests: --cache_aug_refresh_epochs and persistent DataLoader workers (review 2026-11-05, #09).

The refresh regenerates augmentation variants in the main process's dataset at epoch boundaries. A persistent worker
keeps the dataset copy it received at the first epoch, so with --persistent_data_loader_workers the refreshed variants
never reached training. train_network.py now keeps the training loader's workers persistent only when nothing refreshes
its dataset (Anima-3.8B and MOSS refuse the caption-variant refresh before the loader exists; their latent refresh goes
through the same path).

Only small in-memory datasets; one real DataLoader worker process at a time. No model, no GPU.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp=<scratch dir>
     tests/test_aug_refresh_workers.py
"""
import os

import torch

# no library import at module level: the DataLoader worker imports this module to unpickle the dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _RefreshedDataset(torch.utils.data.Dataset):
    """Stands for a training dataset whose variants the main process regenerates between epochs."""

    def __init__(self):
        self.version = 1

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return self.version


def _versions_per_epoch(persistent_workers):
    dataset = _RefreshedDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, persistent_workers=persistent_workers)
    seen = []
    for epoch in range(2):
        if epoch > 0:
            dataset.version += 1  # what refresh_latent_variants / refresh_caption_te_variants do in the main process
        seen.append(sorted({int(batch) for batch in loader}))
    return seen


def test_refresh_reaches_the_workers_only_when_they_are_not_persistent():
    from library import train_util

    assert _versions_per_epoch(True) == [[1], [1]]  # the persistent worker never sees the refresh
    persistent = train_util.dataloader_persistent_workers(True, 1, refreshes_dataset=True)
    assert persistent is False
    assert _versions_per_epoch(persistent) == [[1], [2]]


def test_persistent_workers_are_kept_without_a_refresh():
    from library import train_util

    assert train_util.dataloader_persistent_workers(True, 4, refreshes_dataset=False) is True
    assert train_util.dataloader_persistent_workers(False, 4, refreshes_dataset=True) is False
    assert train_util.dataloader_persistent_workers(False, 4, refreshes_dataset=False) is False
    # in-process loading (0 workers) sees the refresh anyway: the option is passed on unchanged
    assert train_util.dataloader_persistent_workers(True, 0, refreshes_dataset=True) is True


def test_train_network_decides_persistent_workers_with_the_refresh():
    with open(os.path.join(REPO, "train_network.py"), encoding="utf-8") as f:
        source = f.read()
    train_loader = source[source.index("train_dataloader = torch.utils.data.DataLoader(") :]
    train_loader = train_loader[: train_loader.index(")\n\n")]
    assert "persistent_workers=train_util.dataloader_persistent_workers(" in train_loader
    assert "refreshes_train_dataset" in train_loader
