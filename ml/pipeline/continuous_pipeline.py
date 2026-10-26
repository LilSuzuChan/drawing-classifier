"""Scheduled feedback -> train -> evaluate -> ONNX parity -> incumbent gate -> release."""
import argparse
from pathlib import Path

from kfp import compiler, dsl
from kfp import kubernetes
from kfp.dsl import Artifact, Input, Model, Output

from pipeline.components import TRAINING_IMAGE, train, evaluate, export
from pipeline.training_pipeline import configure as base_configure, MNIST_DIR


def configure(task, step, display_name):
    kubernetes.set_image_pull_secrets(task, secret_names=["ghcr-pull"])
    return base_configure(task, step, display_name)


@dsl.component(base_image=TRAINING_IMAGE, install_kfp_package=False)
def snapshot(api: str, feedback: Output[Artifact]):
    from drawing_classifier.continuous import request
    with request(api, '/internal/snapshot') as response, open(feedback.path, 'wb') as file:
        file.write(response.read())


@dsl.component(base_image=TRAINING_IMAGE, install_kfp_package=False)
def promote(api: str, mnist_dir: str, min_accuracy: float, min_holdout: int,
            feedback: Input[Artifact], model: Input[Model], report: Output[Artifact]):
    import json
    from drawing_classifier.continuous import compare_and_promote
    with open(feedback.path) as file:
        records = json.load(file)
    result = compare_and_promote(api, model.path, records, mnist_dir, min_accuracy, min_holdout)
    with open(report.path, 'w') as file:
        json.dump(result, file, indent=2)
    report.metadata.update(result)
    print(json.dumps(result), flush=True)


def authenticate(task):
    kubernetes.use_secret_as_env(task, secret_name='drawing-feedback', secret_key_to_env={'token': 'FEEDBACK_TOKEN'})
    task.set_caching_options(False)
    return task


@dsl.pipeline(name='drawing-classifier-continuous',
              pipeline_config=dsl.PipelineConfig(active_deadline_seconds=7200))
def continuous_pipeline(api: str = 'http://feedback-api.drawing-classifier.svc.cluster.local:8080',
                        epochs: int = 3, min_accuracy: float = 0.95, min_holdout: int = 20):
    captured = authenticate(configure(snapshot(api=api), 'evaluate', 'Snapshot feedback'))
    trained = configure(train(mnist_dir=MNIST_DIR, epochs=epochs, batch_size=64,
                              learning_rate=0.001, seed=42, validation_size=5000, num_threads=2,
                              image=TRAINING_IMAGE, run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
                              feedback=captured.outputs['feedback']), 'train', 'Train with feedback')
    evaluated = configure(evaluate(mnist_dir=MNIST_DIR, min_accuracy=min_accuracy, batch_size=1000,
                                  num_threads=1, model=trained.outputs['model'],
                                  training_config=trained.outputs['training_config']), 'evaluate', 'Evaluate MNIST')
    with dsl.If(evaluated.outputs["Output"] == True):
        exported = configure(export(mnist_dir=MNIST_DIR, parity_samples=256, parity_tolerance=1e-4,
                                    model=trained.outputs['model'], evaluation=evaluated.outputs['evaluation']),
                             'export', 'Export and check ONNX')
        authenticate(configure(promote(api=api, mnist_dir=MNIST_DIR, min_accuracy=min_accuracy,
                                       min_holdout=min_holdout, feedback=captured.outputs['feedback'],
                                       model=exported.outputs['onnx_model']), 'evaluate', 'Compare and release'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(Path(__file__).with_name('drawing-classifier-continuous.yaml')))
    args = parser.parse_args()
    compiler.Compiler().compile(continuous_pipeline, args.output)
