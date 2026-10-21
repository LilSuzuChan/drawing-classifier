from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from drawing_classifier.evaluation import EvaluationResult, evaluate


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int | None = None


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    validation: EvaluationResult | None = None


@dataclass
class TrainingHistory:
    epochs: list[EpochResult] = field(default_factory=list)

    @property
    def final_train_loss(self) -> float:
        return self.epochs[-1].train_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def select_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    validation_loader: DataLoader | None = None,
    on_epoch_end: Callable[[EpochResult], None] | None = None,
) -> TrainingHistory:
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = TrainingHistory()

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        validation = evaluate(model, validation_loader, device) if validation_loader is not None else None
        result = EpochResult(epoch=epoch, train_loss=train_loss, validation=validation)
        history.epochs.append(result)
        if on_epoch_end is not None:
            on_epoch_end(result)

    return history
