from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from drawing_classifier.model import MnistModel
from drawing_classifier.onnx_export import INPUT_NAME, OUTPUT_NAME, export_onnx
from tests.conftest import mps_available

WEB_MODEL = Path(__file__).resolve().parents[2] / "web" / "public" / "models" / "mnist_model.onnx"


def signature(model_proto):
    def dims(value):
        return [d.dim_param or d.dim_value for d in value.type.tensor_type.shape.dim]

    graph = model_proto.graph
    return (
        [(i.name, dims(i), i.type.tensor_type.elem_type) for i in graph.input],
        [(o.name, dims(o), o.type.tensor_type.elem_type) for o in graph.output],
    )


def exported(tmp_path, model):
    path = tmp_path / "model.onnx"
    export_onnx(model, str(path))
    proto = onnx.load(str(path))
    onnx.checker.check_model(proto)
    return path, proto


def test_contract_is_input_1x28x28_and_output_10_with_dynamic_batch(tmp_path):
    _, proto = exported(tmp_path, MnistModel())
    inputs, outputs = signature(proto)
    assert inputs == [(INPUT_NAME, ["batch", 1, 28, 28], onnx.TensorProto.FLOAT)]
    assert outputs == [(OUTPUT_NAME, ["batch", 10], onnx.TensorProto.FLOAT)]
    assert (INPUT_NAME, OUTPUT_NAME) == ("input", "output")


@pytest.mark.skipif(not WEB_MODEL.exists(), reason="web model is not present")
def test_contract_matches_the_model_used_by_the_web_app(tmp_path):
    _, proto = exported(tmp_path, MnistModel())
    assert signature(proto) == signature(onnx.load(str(WEB_MODEL)))


def test_onnx_matches_pytorch_for_several_batch_sizes(tmp_path):
    torch.manual_seed(0)
    model = MnistModel().eval()
    path, _ = exported(tmp_path, model)
    session = ReferenceEvaluator(str(path))
    for batch in (1, 3):
        images = torch.rand(batch, 1, 28, 28)
        expected = model(images).detach().numpy()
        actual = session.run(None, {INPUT_NAME: images.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


@mps_available
def test_export_of_a_model_trained_on_mps(tmp_path):
    model = MnistModel().to("mps")
    _, proto = exported(tmp_path, model)
    assert signature(proto)[0][0][0] == "input"
