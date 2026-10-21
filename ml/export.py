import argparse

import torch

from drawing_classifier.onnx_export import export_onnx
from model import MnistModel

MODEL_PATH = "models/mnist_model.pth"
ONNX_PATH = "../web/public/models/mnist_model.onnx"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export the trained MNIST classifier to ONNX")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--onnx-path", default=ONNX_PATH)
    args = parser.parse_args(argv)

    model = MnistModel()
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    export_onnx(model, args.onnx_path)

    print(f"Model exported to {args.onnx_path}")


if __name__ == "__main__":
    main()
