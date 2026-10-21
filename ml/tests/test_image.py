import importlib
import os
import socket
import tempfile
from pathlib import Path

import onnx
import pytest
import torch
from torch.utils.data import Subset

import drawing_classifier
from drawing_classifier.data import load_mnist, make_loader
from drawing_classifier.evaluation import evaluate
from drawing_classifier.model import INPUT_SHAPE, MnistModel, NUM_CLASSES
from drawing_classifier.onnx_export import INPUT_NAME, OUTPUT_NAME, export_onnx
from drawing_classifier.training import TrainConfig, fit, set_seed

pytestmark = pytest.mark.skipif(
    os.environ.get("DRAWING_CLASSIFIER_IMAGE_TEST") != "1",
    reason="runs only inside the training image",
)

MNIST_DIR = Path(os.environ.get("MNIST_DIR", "/opt/mnist"))
VENV_DIR = Path("/opt/venv")


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("network access is not allowed in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_runs_as_uid_and_gid_1000():
    assert os.getuid() == 1000
    assert os.getgid() == 1000


def test_package_is_installed_into_the_image_environment():
    assert Path(drawing_classifier.__file__).resolve().is_relative_to(VENV_DIR)


@pytest.mark.parametrize("directory", [VENV_DIR, MNIST_DIR])
def test_installed_files_are_readable_but_not_writable(directory):
    files = [path for path in directory.rglob("*") if path.is_file()]
    assert files
    assert all(os.access(path, os.R_OK) for path in files)
    assert not os.access(directory, os.W_OK)
    assert not any(os.access(path, os.W_OK) for path in files)


def test_temporary_directory_and_home_are_writable():
    assert Path(tempfile.gettempdir()) == Path("/tmp")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "check.bin"
        path.write_bytes(b"ok")
        assert path.read_bytes() == b"ok"
    assert os.access(os.path.expanduser("~"), os.W_OK)


def test_mnist_is_baked_in_and_loads_offline(no_network):
    raw = sorted(path.name for path in (MNIST_DIR / "MNIST" / "raw").iterdir())
    assert raw == [
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
    ]
    train = load_mnist(str(MNIST_DIR), train=True, download=False)
    test = load_mnist(str(MNIST_DIR), train=False, download=False)
    assert (len(train), len(test)) == (60000, 10000)
    image, label = train[0]
    assert tuple(image.shape) == INPUT_SHAPE
    assert image.dtype == torch.float32
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    assert 0 <= label < NUM_CLASSES


def test_trains_evaluates_and_exports_on_baked_mnist_offline(tmp_path, no_network):
    set_seed(0)
    train = Subset(load_mnist(str(MNIST_DIR), train=True, download=False), range(2048))
    test = Subset(load_mnist(str(MNIST_DIR), train=False, download=False), range(512))
    model = MnistModel()
    device = torch.device("cpu")
    history = fit(model, make_loader(train, 64, shuffle=True, seed=0), TrainConfig(epochs=1, seed=0), device)
    result = evaluate(model, make_loader(test, 256, shuffle=False), device)
    assert history.final_train_loss > 0
    assert result.total == 512
    assert result.accuracy > 0.5

    path = tmp_path / "model.onnx"
    export_onnx(model, path)
    graph = onnx.load(path).graph
    assert [value.name for value in graph.input] == [INPUT_NAME]
    assert [value.name for value in graph.output] == [OUTPUT_NAME]


def test_pipeline_dependencies_are_cpu_only_and_include_kfp():
    import google.protobuf
    import kfp
    importlib.import_module("kfp.dsl.executor_main")

    assert kfp.__version__ == "2.17.0"
    assert google.protobuf.__version__.startswith("6.")
    assert torch.version.cuda is None
    assert torch.__version__.endswith("+cpu")
