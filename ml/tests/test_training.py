import math

import torch

from drawing_classifier.data import make_loader, split_train_validation
from drawing_classifier.model import MnistModel
from drawing_classifier.training import TrainConfig, fit, select_device, set_seed
from tests.conftest import mps_available


def run(dataset, config, device="cpu", with_validation=True):
    set_seed(config.seed)
    train, validation = split_train_validation(dataset, 0.25, seed=config.seed)
    model = MnistModel().to(device)
    history = fit(
        model,
        make_loader(train, config.batch_size, shuffle=True, seed=config.seed),
        config,
        torch.device(device),
        validation_loader=make_loader(validation, config.batch_size, shuffle=False) if with_validation else None,
    )
    return model, history


def test_history_returns_losses_and_validation_metrics(small_dataset):
    epochs_seen = []
    config = TrainConfig(epochs=3, batch_size=16, seed=0)
    set_seed(0)
    model = MnistModel()
    history = fit(model, make_loader(small_dataset, 16, shuffle=True, seed=0), config, torch.device("cpu"),
                  validation_loader=make_loader(small_dataset, 16, shuffle=False), on_epoch_end=lambda r: epochs_seen.append(r.epoch))
    assert epochs_seen == [1, 2, 3]
    assert [r.epoch for r in history.epochs] == [1, 2, 3]
    assert all(math.isfinite(r.train_loss) for r in history.epochs)
    assert history.final_train_loss == history.epochs[-1].train_loss
    assert 0.0 <= history.epochs[-1].validation.accuracy <= 1.0
    assert history.epochs[-1].validation.accuracy > 0.5


def test_without_validation_loader_has_no_validation(small_dataset):
    _, history = run(small_dataset, TrainConfig(epochs=1, batch_size=32, seed=1), with_validation=False)
    assert history.epochs[0].validation is None


def test_same_seed_gives_identical_results(small_dataset):
    config = TrainConfig(epochs=2, batch_size=32, seed=123)
    model_a, history_a = run(small_dataset, config)
    model_b, history_b = run(small_dataset, config)
    assert [r.train_loss for r in history_a.epochs] == [r.train_loss for r in history_b.epochs]
    for a, b in zip(model_a.state_dict().values(), model_b.state_dict().values()):
        assert torch.equal(a, b)


def test_different_seed_changes_results(small_dataset):
    _, history_a = run(small_dataset, TrainConfig(epochs=1, batch_size=32, seed=1))
    _, history_b = run(small_dataset, TrainConfig(epochs=1, batch_size=32, seed=2))
    assert history_a.epochs[0].train_loss != history_b.epochs[0].train_loss


def test_select_device():
    assert select_device("cpu").type == "cpu"
    expected = "mps" if torch.backends.mps.is_available() else "cpu"
    assert select_device().type == expected


@mps_available
def test_training_and_evaluation_on_mps(small_dataset):
    _, history = run(small_dataset, TrainConfig(epochs=1, batch_size=32, seed=5), device="mps")
    assert math.isfinite(history.epochs[0].train_loss)
    assert 0.0 <= history.epochs[0].validation.accuracy <= 1.0
