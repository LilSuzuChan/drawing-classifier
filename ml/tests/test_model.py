import torch

from drawing_classifier.model import INPUT_SHAPE, NUM_CLASSES, MnistModel


def test_output_has_ten_logits_per_image():
    model = MnistModel()
    assert model(torch.zeros(1, *INPUT_SHAPE)).shape == (1, NUM_CLASSES)
    assert model(torch.zeros(5, *INPUT_SHAPE)).shape == (5, 10)


def test_cli_shim_exports_the_same_class():
    import model as cli_model

    assert cli_model.MnistModel is MnistModel
