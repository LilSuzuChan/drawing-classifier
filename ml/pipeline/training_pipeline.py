import argparse
from pathlib import Path

from kfp import compiler, dsl

from pipeline.components import TRAINING_IMAGE, evaluate, export, train

PIPELINE_NAME = "drawing-classifier-training"
PIPELINE_PACKAGE = Path(__file__).resolve().with_name(f"{PIPELINE_NAME}.yaml")
MNIST_DIR = "/opt/mnist"
ACTIVE_DEADLINE_SECONDS = 3600

RESOURCES = {
    "train": {"cpu_request": "1", "cpu_limit": "2", "memory_request": "1Gi", "memory_limit": "2Gi", "threads": 2},
    "evaluate": {"cpu_request": "500m", "cpu_limit": "1", "memory_request": "768Mi", "memory_limit": "1536Mi", "threads": 1},
    "export": {"cpu_request": "500m", "cpu_limit": "1", "memory_request": "768Mi", "memory_limit": "1536Mi", "threads": 1},
}


def configure(task: dsl.PipelineTask, step: str, display_name: str) -> dsl.PipelineTask:
    resources = RESOURCES[step]
    return (
        task.set_display_name(display_name)
        .set_caching_options(False)
        .set_cpu_request(resources["cpu_request"])
        .set_cpu_limit(resources["cpu_limit"])
        .set_memory_request(resources["memory_request"])
        .set_memory_limit(resources["memory_limit"])
    )


def build_pipeline(mnist_dir: str = MNIST_DIR):
    @dsl.pipeline(
        name=PIPELINE_NAME,
        description="Train, evaluate and export the MNIST drawing classifier on CPU with the pinned training image",
        pipeline_config=dsl.PipelineConfig(active_deadline_seconds=ACTIVE_DEADLINE_SECONDS),
    )
    def training_pipeline(
        epochs: int = 1,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        seed: int = 42,
        validation_size: int = 5000,
        min_accuracy: float = 0.9,
        parity_samples: int = 256,
        parity_tolerance: float = 1e-4,
    ):
        trained = configure(
            train(
                mnist_dir=mnist_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                validation_size=validation_size,
                num_threads=RESOURCES["train"]["threads"],
                image=TRAINING_IMAGE,
                run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
            ),
            "train",
            "Train",
        )
        evaluated = configure(
            evaluate(
                mnist_dir=mnist_dir,
                min_accuracy=min_accuracy,
                batch_size=1000,
                num_threads=RESOURCES["evaluate"]["threads"],
                model=trained.outputs["model"],
                training_config=trained.outputs["training_config"],
            ),
            "evaluate",
            "Evaluate",
        )
        configure(
            export(
                mnist_dir=mnist_dir,
                parity_samples=parity_samples,
                parity_tolerance=parity_tolerance,
                model=trained.outputs["model"],
                evaluation=evaluated.outputs["evaluation"],
            ),
            "export",
            "Export",
        )

    return training_pipeline


training_pipeline = build_pipeline()


def compile_pipeline(path: Path = PIPELINE_PACKAGE) -> Path:
    compiler.Compiler().compile(training_pipeline, str(path))
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Compile the drawing-classifier KFP training pipeline")
    parser.add_argument("--output", type=Path, default=PIPELINE_PACKAGE)
    args = parser.parse_args(argv)
    print(f"Pipeline compiled to {compile_pipeline(args.output)}")


if __name__ == "__main__":
    main()
