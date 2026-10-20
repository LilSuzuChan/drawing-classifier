from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from drawing_classifier.model import NUM_CLASSES


@dataclass
class EvaluationResult:
    loss: float
    accuracy: float
    correct: int
    total: int
    confusion_matrix: list[list[int]] = field(default_factory=list)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> EvaluationResult:
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    total_loss = 0.0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item()
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            pairs = labels.cpu() * num_classes + predicted.cpu()
            confusion += torch.bincount(pairs, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    if total == 0:
        raise ValueError("cannot evaluate an empty dataset")
    return EvaluationResult(
        loss=total_loss / total,
        accuracy=correct / total,
        correct=correct,
        total=total,
        confusion_matrix=confusion.tolist(),
    )


def meets_threshold(result: EvaluationResult, min_accuracy: float) -> bool:
    return result.accuracy >= min_accuracy
