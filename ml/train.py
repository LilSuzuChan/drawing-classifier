import argparse
import os

import torch

from drawing_classifier.data import load_mnist, make_loader, split_train_validation
from drawing_classifier.evaluation import evaluate
from drawing_classifier.training import EpochResult, TrainConfig, fit, select_device, set_seed
from model import MnistModel

EPOCHS = 5
BATCH_SIZE = 64
MODEL_PATH = "models/mnist_model.pth"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the MNIST classifier")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--validation-size", type=float, default=0.0,
                        help="fraction (0-1) or count of training samples held out for validation; 0 keeps the original behaviour")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--device", default=None, help="cpu or mps; defaults to mps when available")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed)
    if config.seed is not None:
        set_seed(config.seed)

    device = select_device(args.device)
    print(f"Device: {device}")

    train_dataset = load_mnist(args.data_dir, train=True)
    test_dataset = load_mnist(args.data_dir, train=False)
    test_loader = make_loader(test_dataset, config.batch_size, shuffle=False)

    if args.validation_size > 0:
        size = int(args.validation_size) if args.validation_size >= 1 else args.validation_size
        train_dataset, validation_dataset = split_train_validation(train_dataset, size, seed=config.seed)
        validation_loader = make_loader(validation_dataset, config.batch_size, shuffle=False)
        label = "Validation"
    else:
        validation_loader = test_loader
        label = "Test"

    train_loader = make_loader(train_dataset, config.batch_size, shuffle=True, seed=config.seed)
    model = MnistModel().to(device)

    def report(result: EpochResult) -> None:
        print(f"\nEpoch {result.epoch}/{config.epochs}")
        print(f"Loss: {result.train_loss:.4f}")
        print(f"{label} Accuracy: {100 * result.validation.accuracy:.2f}%")

    fit(model, train_loader, config, device, validation_loader=validation_loader, on_epoch_end=report)

    if validation_loader is not test_loader:
        test = evaluate(model, test_loader, device)
        print(f"\nTest Accuracy: {100 * test.accuracy:.2f}%")

    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.model_path)
    print(f"\nModel saved to {args.model_path}")


if __name__ == "__main__":
    main()
