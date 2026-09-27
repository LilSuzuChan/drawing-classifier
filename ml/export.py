import torch

from model import MnistModel

MODEL_PATH = "models/mnist_model.pth"
ONNX_PATH = "../web/public/models/mnist_model.onnx"

model = MnistModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 1, 28, 28)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"},
    },
    dynamo=False,
)

print(f"Model exported to {ONNX_PATH}")
