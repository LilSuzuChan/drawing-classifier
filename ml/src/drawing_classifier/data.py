from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor


def load_mnist(root: str = "data", train: bool = True, download: bool = True) -> Dataset:
    return datasets.MNIST(
        root=root,
        train=train,
        download=download,
        transform=ToTensor(),
    )


def split_train_validation(
    dataset: Dataset,
    validation_size: int | float,
    seed: int | None = None,
) -> tuple[Subset, Subset]:
    total = len(dataset)
    if isinstance(validation_size, float):
        if not 0.0 < validation_size < 1.0:
            raise ValueError("validation_size as a fraction must be between 0 and 1")
        validation_count = int(round(total * validation_size))
    else:
        validation_count = validation_size
    if not 0 < validation_count < total:
        raise ValueError("validation_size must leave at least one sample in each split")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    train_subset, validation_subset = random_split(
        dataset,
        [total - validation_count, validation_count],
        generator=generator,
    )
    return train_subset, validation_subset


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
