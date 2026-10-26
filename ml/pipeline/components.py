import os
import re
from kfp import dsl
from kfp.dsl import Artifact, ClassificationMetrics, Input, Metrics, Model, Output

TRAINING_IMAGE = os.environ.get("DRAWING_TRAINING_IMAGE") or (
    "ghcr.io/lilsuzuchan/drawing-classifier-train"
    "@sha256:ad8c4d2939e73aebd53d2f3e093ddb48c5e9061adf0c1a3071bc00ec2132b8f2"
)

if not re.fullmatch(r"ghcr\.io/[a-z0-9./-]+@sha256:[0-9a-f]{64}", TRAINING_IMAGE):
    raise ValueError("DRAWING_TRAINING_IMAGE must be an immutable GHCR digest reference")


@dsl.component(base_image=TRAINING_IMAGE, install_kfp_package=False)
def train(
    mnist_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    validation_size: int,
    num_threads: int,
    image: str,
    run_id: str,
    model: Output[Model],
    training_config: Output[Artifact],
    metrics: Output[Metrics],
    feedback: Input[Artifact] = None,
):
    import hashlib
    import json
    import os
    import platform

    import torch

    from drawing_classifier.data import load_mnist, make_loader, split_train_validation
    from drawing_classifier.model import MnistModel
    from drawing_classifier.training import TrainConfig, fit, set_seed

    def sha256_of(path):
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for block in iter(lambda: file.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    torch.set_num_threads(num_threads)
    set_seed(seed)
    device = torch.device("cpu")
    config = TrainConfig(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, seed=seed)

    dataset = load_mnist(mnist_dir, train=True, download=False)
    train_set, validation_set = split_train_validation(dataset, validation_size, seed=seed)
    feedback_count = 0
    feedback_sha256 = None
    if feedback is not None:
        from torch.utils.data import ConcatDataset
        from drawing_classifier.continuous import feedback_dataset, split_feedback
        with open(feedback.path) as file:
            records = json.load(file)
        feedback_train, _ = split_feedback(records)
        feedback_count = len(feedback_train)
        feedback_sha256 = sha256_of(feedback.path)
        if feedback_train:
            train_set = ConcatDataset([train_set, feedback_dataset(feedback_train)])
    raw_dir = os.path.join(mnist_dir, "MNIST", "raw")
    dataset_files = {
        name: sha256_of(os.path.join(raw_dir, name))
        for name in ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    }

    def report(result):
        print(
            f"epoch {result.epoch}/{epochs} train_loss={result.train_loss:.4f} "
            f"validation_loss={result.validation.loss:.4f} validation_accuracy={result.validation.accuracy:.4f}",
            flush=True,
        )

    net = MnistModel()
    history = fit(
        net,
        make_loader(train_set, batch_size, shuffle=True, seed=seed),
        config,
        device,
        validation_loader=make_loader(validation_set, batch_size, shuffle=False),
        on_epoch_end=report,
    )
    final = history.epochs[-1]

    torch.save(net.state_dict(), model.path)
    model_sha256 = sha256_of(model.path)

    conditions = {
        "feedback_samples": feedback_count,
        "feedback_sha256": feedback_sha256,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "validation_size": validation_size,
        "train_samples": len(train_set),
        "validation_samples": len(validation_set),
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "device": device.type,
        "num_threads": num_threads,
        "image": image,
        "run_id": run_id,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "mnist_dir": mnist_dir,
        "dataset_sha256": dataset_files,
        "model_sha256": model_sha256,
        "history": [
            {
                "epoch": result.epoch,
                "train_loss": result.train_loss,
                "validation_loss": result.validation.loss,
                "validation_accuracy": result.validation.accuracy,
            }
            for result in history.epochs
        ],
    }
    with open(training_config.path, "w") as file:
        json.dump(conditions, file, indent=2, sort_keys=True)

    scalar_conditions = {key: value for key, value in conditions.items() if isinstance(value, (str, int, float))}
    training_config.metadata.update(scalar_conditions)
    model.metadata.update(
        {
            "framework": "pytorch",
            "format": "state_dict",
            "model_class": "drawing_classifier.model.MnistModel",
            "sha256": model_sha256,
            **{key: scalar_conditions[key] for key in ("epochs", "batch_size", "learning_rate", "seed", "image", "run_id", "torch_version")},
        }
    )

    metrics.log_metric("train_loss", round(final.train_loss, 6))
    metrics.log_metric("validation_loss", round(final.validation.loss, 6))
    metrics.log_metric("validation_accuracy", round(final.validation.accuracy, 6))
    metrics.log_metric("epochs", epochs)


@dsl.component(base_image=TRAINING_IMAGE, install_kfp_package=False)
def evaluate(
    mnist_dir: str,
    min_accuracy: float,
    batch_size: int,
    num_threads: int,
    model: Input[Model],
    training_config: Input[Artifact],
    metrics: Output[Metrics],
    confusion_matrix: Output[ClassificationMetrics],
    evaluation: Output[Artifact],
) -> bool:
    import hashlib
    import json

    import torch

    from drawing_classifier.data import load_mnist, make_loader
    from drawing_classifier.evaluation import evaluate as evaluate_model
    from drawing_classifier.evaluation import meets_threshold
    from drawing_classifier.model import MnistModel

    with open(model.path, "rb") as file:
        model_sha256 = hashlib.sha256(file.read()).hexdigest()
    with open(training_config.path) as file:
        conditions = json.load(file)
    if model_sha256 != conditions["model_sha256"]:
        raise ValueError(f"model sha256 {model_sha256} does not match the training record {conditions['model_sha256']}")

    torch.set_num_threads(num_threads)
    device = torch.device("cpu")
    net = MnistModel()
    net.load_state_dict(torch.load(model.path, map_location=device, weights_only=True))
    test_set = load_mnist(mnist_dir, train=False, download=False)
    result = evaluate_model(net, make_loader(test_set, batch_size, shuffle=False), device)
    passed = meets_threshold(result, min_accuracy)
    print(f"test_loss={result.loss:.4f} test_accuracy={result.accuracy:.4f} ({result.correct}/{result.total}) passed={passed}", flush=True)

    metrics.log_metric("test_accuracy", round(result.accuracy, 6))
    metrics.log_metric("test_loss", round(result.loss, 6))
    metrics.log_metric("test_samples", result.total)
    metrics.log_metric("min_accuracy", min_accuracy)
    confusion_matrix.log_confusion_matrix([str(label) for label in range(len(result.confusion_matrix))], result.confusion_matrix)

    record = {
        "model_sha256": model_sha256,
        "test_loss": result.loss,
        "test_accuracy": result.accuracy,
        "test_correct": result.correct,
        "test_samples": result.total,
        "confusion_matrix": result.confusion_matrix,
        "min_accuracy": min_accuracy,
        "meets_threshold": passed,
        "training": {key: conditions[key] for key in ("epochs", "batch_size", "learning_rate", "seed", "image", "run_id")},
    }
    with open(evaluation.path, "w") as file:
        json.dump(record, file, indent=2, sort_keys=True)
    evaluation.metadata.update(
        {
            "model_sha256": model_sha256,
            "test_accuracy": result.accuracy,
            "test_loss": result.loss,
            "min_accuracy": min_accuracy,
            "meets_threshold": passed,
        }
    )
    return passed


@dsl.component(base_image=TRAINING_IMAGE, install_kfp_package=False)
def export(
    mnist_dir: str,
    parity_samples: int,
    parity_tolerance: float,
    model: Input[Model],
    evaluation: Input[Artifact],
    onnx_model: Output[Model],
    metrics: Output[Metrics],
):
    import hashlib
    import json

    import numpy as np
    import onnx
    import torch
    from onnx.reference import ReferenceEvaluator

    from drawing_classifier.data import load_mnist
    from drawing_classifier.model import MnistModel
    from drawing_classifier.onnx_export import INPUT_NAME, OUTPUT_NAME, export_onnx

    expected_contract = {
        "inputs": [["input", ["batch", 1, 28, 28], onnx.TensorProto.FLOAT]],
        "outputs": [["output", ["batch", 10], onnx.TensorProto.FLOAT]],
    }

    def sha256_of(path):
        with open(path, "rb") as file:
            return hashlib.sha256(file.read()).hexdigest()

    def signature(values):
        return [
            [value.name, [dim.dim_param or dim.dim_value for dim in value.type.tensor_type.shape.dim], value.type.tensor_type.elem_type]
            for value in values
        ]

    with open(evaluation.path) as file:
        evaluated = json.load(file)
    model_sha256 = sha256_of(model.path)
    if model_sha256 != evaluated["model_sha256"]:
        raise ValueError(f"model sha256 {model_sha256} is not the evaluated model {evaluated['model_sha256']}")

    net = MnistModel()
    net.load_state_dict(torch.load(model.path, map_location="cpu", weights_only=True))
    net.eval()
    export_onnx(net, onnx_model.path)

    proto = onnx.load(onnx_model.path)
    onnx.checker.check_model(proto)
    contract = {"inputs": signature(proto.graph.input), "outputs": signature(proto.graph.output)}
    if contract != expected_contract or (INPUT_NAME, OUTPUT_NAME) != ("input", "output"):
        raise ValueError(f"ONNX contract {contract} differs from the web app contract {expected_contract}")

    test_set = load_mnist(mnist_dir, train=False, download=False)
    images = torch.stack([test_set[index][0] for index in range(parity_samples)])
    with torch.no_grad():
        expected = net(images).numpy()
    actual = ReferenceEvaluator(proto).run(None, {INPUT_NAME: images.numpy()})[0]
    max_abs_diff = float(np.abs(actual - expected).max())
    argmax_agreement = float((actual.argmax(axis=1) == expected.argmax(axis=1)).mean())
    print(f"onnx parity: samples={parity_samples} max_abs_diff={max_abs_diff:.3e} argmax_agreement={argmax_agreement}", flush=True)
    if max_abs_diff > parity_tolerance:
        raise ValueError(f"ONNX output differs from PyTorch by {max_abs_diff}, above {parity_tolerance}")

    metrics.log_metric("onnx_max_abs_diff", max_abs_diff)
    metrics.log_metric("onnx_argmax_agreement", argmax_agreement)
    metrics.log_metric("onnx_parity_samples", parity_samples)

    onnx_model.metadata.update(
        {
            "framework": "onnx",
            "opset": max(entry.version for entry in proto.opset_import if entry.domain in ("", "ai.onnx")),
            "sha256": sha256_of(onnx_model.path),
            "source_model_sha256": model_sha256,
            "input": "input[batch,1,28,28] float32",
            "output": "output[batch,10] float32",
            "test_accuracy": evaluated["test_accuracy"],
            "meets_threshold": evaluated["meets_threshold"],
            "max_abs_diff": max_abs_diff,
            **{f"train_{key}": value for key, value in evaluated["training"].items()},
        }
    )
