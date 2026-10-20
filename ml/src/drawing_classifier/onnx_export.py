from __future__ import annotations

import torch
import torch.nn as nn

from drawing_classifier.model import INPUT_SHAPE

INPUT_NAME = "input"
OUTPUT_NAME = "output"


def export_onnx(model: nn.Module, path: str) -> None:
    model = model.to("cpu")
    model.eval()
    dummy_input = torch.randn(1, *INPUT_SHAPE)

    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={
            INPUT_NAME: {0: "batch"},
            OUTPUT_NAME: {0: "batch"},
        },
        dynamo=False,
    )
