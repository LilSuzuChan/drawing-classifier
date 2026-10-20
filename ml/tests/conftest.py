import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))


def synthetic_mnist(count: int, seed: int = 0) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(count) % 10
    images = torch.rand(count, 1, 28, 28, generator=generator) * 0.1
    for index, label in enumerate(labels.tolist()):
        images[index, 0, label * 2 : label * 2 + 3, :] = 1.0
    return TensorDataset(images, labels)


@pytest.fixture
def small_dataset() -> TensorDataset:
    return synthetic_mnist(200)


mps_available = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is only available on Apple silicon")
