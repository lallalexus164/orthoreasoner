from collections.abc import Sized
from typing import cast

from orthoreasoner_rc.core.config import DataConfig
from orthoreasoner_rc.data.synthetic import (
    SyntheticRotatorCuffDataset,
    SyntheticSpec,
    collate_batch,
)
from torch.utils.data import DataLoader, Dataset, random_split


def make_dataset(config: DataConfig, seed: int) -> Dataset[object]:
    if config.mode == "synthetic":
        return SyntheticRotatorCuffDataset(SyntheticSpec(config.records, config.image_size, seed))
    raise ValueError(f"unsupported data mode: {config.mode}")


def make_loaders(
    config: DataConfig, seed: int
) -> tuple[DataLoader[object], DataLoader[object], DataLoader[object]]:
    dataset = make_dataset(config, seed)
    n = len(cast(Sized, dataset))
    train_n = max(1, int(n * 0.7))
    valid_n = max(1, int(n * 0.15))
    test_n = n - train_n - valid_n
    if test_n < 1:
        test_n = 1
        train_n = n - valid_n - test_n
    train, valid, test = random_split(dataset, [train_n, valid_n, test_n])
    return (
        DataLoader(
            train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
            collate_fn=collate_batch,
        ),
        DataLoader(
            valid,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            collate_fn=collate_batch,
        ),
        DataLoader(
            test,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            collate_fn=collate_batch,
        ),
    )
