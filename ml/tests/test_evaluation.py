import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from drawing_classifier.data import make_loader
from drawing_classifier.evaluation import EvaluationResult, evaluate, meets_threshold


class Oracle(nn.Module):
    def __init__(self, wrong_for_label=None):
        super().__init__()
        self.wrong_for_label = wrong_for_label

    def forward(self, x):
        labels = x[:, 0, 0, 0].long()
        if self.wrong_for_label is not None:
            labels = torch.where(labels == self.wrong_for_label, (labels + 1) % 10, labels)
        return nn.functional.one_hot(labels, 10).float() * 10


def labelled_images(labels):
    images = torch.zeros(len(labels), 1, 28, 28)
    images[:, 0, 0, 0] = torch.tensor(labels, dtype=torch.float32)
    return TensorDataset(images, torch.tensor(labels))


def test_perfect_predictions():
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 3]
    result = evaluate(Oracle(), make_loader(labelled_images(labels), 5, shuffle=False), torch.device("cpu"))
    assert result.accuracy == 1.0
    assert (result.correct, result.total) == (12, 12)
    assert result.confusion_matrix[3][3] == 3
    assert sum(map(sum, result.confusion_matrix)) == 12
    assert result.loss < 0.01


def test_confusion_matrix_rows_are_true_labels():
    labels = [2, 2, 5]
    result = evaluate(Oracle(wrong_for_label=2), make_loader(labelled_images(labels), 2, shuffle=False), torch.device("cpu"))
    assert result.accuracy == pytest.approx(1 / 3)
    assert result.confusion_matrix[2][3] == 2
    assert result.confusion_matrix[5][5] == 1


def test_empty_dataset_is_rejected():
    empty = TensorDataset(torch.zeros(0, 1, 28, 28), torch.zeros(0, dtype=torch.long))
    with pytest.raises(ValueError):
        evaluate(Oracle(), make_loader(empty, 4, shuffle=False), torch.device("cpu"))


def test_threshold_is_inclusive_and_separate_from_metrics():
    result = EvaluationResult(loss=0.1, accuracy=0.95, correct=95, total=100)
    assert meets_threshold(result, 0.95)
    assert not meets_threshold(result, 0.951)
    assert result.accuracy == 0.95
