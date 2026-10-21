import onnx
import torch

import export as export_cli
import train as train_cli
from tests.conftest import synthetic_mnist


def fake_load_mnist(root="data", train=True, download=True):
    return synthetic_mnist(120 if train else 40, seed=0 if train else 1)


def test_train_cli_default_output_and_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train_cli, "load_mnist", fake_load_mnist)
    model_path = tmp_path / "models" / "mnist_model.pth"
    train_cli.main(["--epochs", "2", "--model-path", str(model_path), "--device", "cpu"])
    out = capsys.readouterr().out
    assert "Device: cpu" in out
    assert "Epoch 1/2" in out and "Epoch 2/2" in out
    assert out.count("Test Accuracy:") == 2
    assert "Loss: " in out
    assert f"Model saved to {model_path}" in out
    state = torch.load(model_path, map_location="cpu")
    assert set(state) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}


def test_train_cli_with_validation_split_reports_test_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train_cli, "load_mnist", fake_load_mnist)
    train_cli.main(["--epochs", "2", "--validation-size", "0.25", "--seed", "1",
                    "--model-path", str(tmp_path / "m.pth"), "--device", "cpu"])
    out = capsys.readouterr().out
    assert out.count("Validation Accuracy:") == 2
    assert out.count("Test Accuracy:") == 1


def test_export_cli_produces_the_same_contract(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train_cli, "load_mnist", fake_load_mnist)
    model_path = tmp_path / "m.pth"
    onnx_path = tmp_path / "m.onnx"
    train_cli.main(["--epochs", "1", "--model-path", str(model_path), "--device", "cpu"])
    export_cli.main(["--model-path", str(model_path), "--onnx-path", str(onnx_path)])
    assert f"Model exported to {onnx_path}" in capsys.readouterr().out
    graph = onnx.load(str(onnx_path)).graph
    assert [i.name for i in graph.input] == ["input"]
    assert [o.name for o in graph.output] == ["output"]
