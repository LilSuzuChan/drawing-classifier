import hashlib
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

kfp = pytest.importorskip("kfp")
yaml = pytest.importorskip("yaml")
onnx = pytest.importorskip("onnx")

from kfp import dsl, local

from pipeline import run as run_cli
from pipeline.components import TRAINING_IMAGE, evaluate, export, train
from pipeline.training_pipeline import (
    ACTIVE_DEADLINE_SECONDS,
    MNIST_DIR,
    PIPELINE_NAME,
    PIPELINE_PACKAGE,
    RESOURCES,
    build_pipeline,
    compile_pipeline,
)

ML_DIR = Path(__file__).resolve().parents[1]
WEB_MODEL = ML_DIR.parent / "web" / "public" / "models" / "mnist_model.onnx"
F2_IMAGE_DIGEST = "sha256:ad8c4d2939e73aebd53d2f3e093ddb48c5e9061adf0c1a3071bc00ec2132b8f2"
LIMIT_RANGE_MAX = {"cpu": 3.0, "memory": 6 * 1024**3}


def find_mnist_dir():
    for candidate in (os.environ.get("MNIST_DIR"), str(ML_DIR / "data")):
        if candidate and all(
            (Path(candidate) / "MNIST" / "raw" / name).is_file()
            for name in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
        ):
            return candidate
    return None


MNIST = find_mnist_dir()
needs_mnist = pytest.mark.skipif(MNIST is None, reason="uncompressed MNIST is not available locally")


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    path = compile_pipeline(tmp_path_factory.mktemp("compiled") / "pipeline.yaml")
    with open(path) as file:
        pipeline_spec, platform_spec = yaml.safe_load_all(file)
    return {**pipeline_spec, **platform_spec}


def executors(spec):
    return spec["deploymentSpec"]["executors"]


def quantity(value: str) -> float:
    units = {"m": 1e-3, "Ki": 1024, "Mi": 1024**2, "Gi": 1024**3}
    for suffix, factor in units.items():
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * factor
    return float(value)


def signature(values):
    return [
        (value.name, [dim.dim_param or dim.dim_value for dim in value.type.tensor_type.shape.dim], value.type.tensor_type.elem_type)
        for value in values
    ]


def test_committed_pipeline_package_is_up_to_date(tmp_path):
    fresh = compile_pipeline(tmp_path / "pipeline.yaml")
    assert PIPELINE_PACKAGE.read_text() == fresh.read_text()


def test_every_step_uses_the_pinned_training_image(spec):
    assert TRAINING_IMAGE.endswith("@" + F2_IMAGE_DIGEST)
    images = {name: executor["container"]["image"] for name, executor in executors(spec).items()}
    assert set(images) == {"exec-train", "exec-evaluate", "exec-export"}
    assert set(images.values()) == {TRAINING_IMAGE}
    assert all(re.fullmatch(r"ghcr\.io/[a-z0-9./-]+@sha256:[0-9a-f]{64}", image) for image in images.values())


def test_no_step_installs_packages_at_runtime(spec):
    for executor in executors(spec).values():
        container = executor["container"]
        script = " ".join(container["command"])
        assert "pip" not in script and "uv " not in script and "ensurepip" not in script
        assert "python3 -m kfp.dsl.executor_main" in script


def test_resources_are_explicit_and_within_the_limit_range(spec):
    for step, resources in RESOURCES.items():
        container = executors(spec)[f"exec-{step}"]["container"]["resources"]
        assert container["resourceCpuRequest"] == resources["cpu_request"]
        assert container["resourceCpuLimit"] == resources["cpu_limit"]
        assert container["resourceMemoryRequest"] == resources["memory_request"]
        assert container["resourceMemoryLimit"] == resources["memory_limit"]
        assert quantity(resources["cpu_request"]) <= quantity(resources["cpu_limit"]) <= LIMIT_RANGE_MAX["cpu"]
        assert quantity(resources["memory_request"]) <= quantity(resources["memory_limit"]) <= LIMIT_RANGE_MAX["memory"]
        assert resources["threads"] == int(quantity(resources["cpu_limit"]))
        assert "accelerator" not in container


def test_steps_run_in_sequence_without_cache_and_with_a_deadline(spec):
    tasks = spec["root"]["dag"]["tasks"]
    assert set(tasks) == {"train", "evaluate", "export"}
    assert tasks["train"].get("dependentTasks", []) == []
    assert tasks["evaluate"]["dependentTasks"] == ["train"]
    assert sorted(tasks["export"]["dependentTasks"]) == ["evaluate", "train"]
    assert all(task["cachingOptions"].get("enableCache", False) is False for task in tasks.values())
    assert spec["platforms"]["kubernetes"]["pipelineConfig"]["activeDeadlineSeconds"] == ACTIVE_DEADLINE_SECONDS
    assert spec["pipelineInfo"]["name"] == PIPELINE_NAME


def test_mnist_comes_from_the_image_and_defaults_train_one_epoch(spec):
    tasks = spec["root"]["dag"]["tasks"]
    for name in ("train", "evaluate", "export"):
        assert tasks[name]["inputs"]["parameters"]["mnist_dir"]["runtimeValue"]["constant"] == MNIST_DIR == "/opt/mnist"
    assert tasks["train"]["inputs"]["parameters"]["image"]["runtimeValue"]["constant"] == TRAINING_IMAGE
    assert tasks["train"]["inputs"]["parameters"]["run_id"]["runtimeValue"]["constant"] == "{{$.pipeline_job_uuid}}"
    defaults = {name: value["defaultValue"] for name, value in spec["root"]["inputDefinitions"]["parameters"].items()}
    assert defaults == {
        "epochs": 1,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "seed": 42,
        "validation_size": 5000,
        "min_accuracy": 0.9,
        "parity_samples": 256,
        "parity_tolerance": 1e-4,
    }
    assert set(defaults) == set(run_cli.PARAMETERS)


def test_artifacts_flow_from_train_to_evaluate_to_export(spec):
    components = spec["components"]
    outputs = {
        name: {key: value["artifactType"]["schemaTitle"] for key, value in component["outputDefinitions"]["artifacts"].items()}
        for name, component in components.items()
    }
    assert outputs["comp-train"] == {"model": "system.Model", "training_config": "system.Artifact", "metrics": "system.Metrics"}
    assert outputs["comp-evaluate"] == {
        "metrics": "system.Metrics",
        "confusion_matrix": "system.ClassificationMetrics",
        "evaluation": "system.Artifact",
    }
    assert outputs["comp-export"] == {"onnx_model": "system.Model", "metrics": "system.Metrics"}
    assert components["comp-evaluate"]["outputDefinitions"]["parameters"]["Output"]["parameterType"] == "BOOLEAN"

    tasks = spec["root"]["dag"]["tasks"]
    evaluate_inputs = tasks["evaluate"]["inputs"]["artifacts"]
    export_inputs = tasks["export"]["inputs"]["artifacts"]
    assert evaluate_inputs["model"]["taskOutputArtifact"] == {"outputArtifactKey": "model", "producerTask": "train"}
    assert evaluate_inputs["training_config"]["taskOutputArtifact"] == {"outputArtifactKey": "training_config", "producerTask": "train"}
    assert export_inputs["model"]["taskOutputArtifact"] == {"outputArtifactKey": "model", "producerTask": "train"}
    assert export_inputs["evaluation"]["taskOutputArtifact"] == {"outputArtifactKey": "evaluation", "producerTask": "evaluate"}


def test_pipeline_does_not_override_the_pipeline_root_or_service_account(spec):
    assert "defaultPipelineRoot" not in spec
    assert run_cli.SERVICE_ACCOUNT == "pipeline-runner"


@needs_mnist
def test_pipeline_runs_locally_and_is_reproducible(tmp_path):
    local.init(runner=local.SubprocessRunner(use_venv=False), pipeline_root=str(tmp_path))
    pipeline = build_pipeline(MNIST)
    pipeline(seed=7, validation_size=5000)
    pipeline(seed=7, validation_size=5000)

    runs = sorted(path for path in tmp_path.iterdir() if path.is_dir())
    assert len(runs) == 2
    records = []
    for run_dir in runs:
        training = json.loads((run_dir / "train" / "training_config").read_text())
        evaluation = json.loads((run_dir / "evaluate" / "evaluation").read_text())
        model_sha256 = hashlib.sha256((run_dir / "train" / "model").read_bytes()).hexdigest()
        onnx_path = run_dir / "export" / "onnx_model"
        records.append((training, evaluation, model_sha256, onnx_path))

        assert training["image"] == TRAINING_IMAGE
        assert (training["epochs"], training["batch_size"], training["learning_rate"], training["seed"]) == (1, 64, 1e-3, 7)
        assert (training["train_samples"], training["validation_samples"]) == (55000, 5000)
        assert training["model_sha256"] == model_sha256 == evaluation["model_sha256"]
        assert set(training["dataset_sha256"]) == {"t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "train-images-idx3-ubyte", "train-labels-idx1-ubyte"}
        assert evaluation["test_samples"] == 10000
        assert evaluation["meets_threshold"] is (evaluation["test_accuracy"] >= 0.9)
        assert sum(map(sum, evaluation["confusion_matrix"])) == 10000

        proto = onnx.load(str(onnx_path))
        assert signature(proto.graph.input) == [("input", ["batch", 1, 28, 28], onnx.TensorProto.FLOAT)]
        assert signature(proto.graph.output) == [("output", ["batch", 10], onnx.TensorProto.FLOAT)]
        if WEB_MODEL.exists():
            web = onnx.load(str(WEB_MODEL))
            assert signature(proto.graph.input) == signature(web.graph.input)
            assert signature(proto.graph.output) == signature(web.graph.output)

    (first_training, first_evaluation, first_sha, _), (second_training, second_evaluation, second_sha, _) = records
    assert first_sha == second_sha
    assert first_training["history"] == second_training["history"]
    assert first_evaluation["test_accuracy"] == second_evaluation["test_accuracy"]
    assert first_evaluation["test_loss"] == second_evaluation["test_loss"]


def artifact(kind, path, metadata=None):
    return kind(name=Path(path).name, uri=str(path), metadata=dict(metadata or {}))


@needs_mnist
def test_evaluate_rejects_a_model_that_does_not_match_the_training_record(tmp_path):
    model_path = tmp_path / "model"
    model_path.write_bytes(b"not the trained model")
    config_path = tmp_path / "training_config"
    config_path.write_text(json.dumps({"model_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="does not match the training record"):
        evaluate.python_func(
            mnist_dir=MNIST,
            min_accuracy=0.9,
            batch_size=1000,
            num_threads=1,
            model=artifact(dsl.Model, model_path),
            training_config=artifact(dsl.Artifact, config_path),
            metrics=artifact(dsl.Metrics, tmp_path / "metrics"),
            confusion_matrix=artifact(dsl.ClassificationMetrics, tmp_path / "confusion"),
            evaluation=artifact(dsl.Artifact, tmp_path / "evaluation"),
        )


@needs_mnist
def test_export_rejects_a_model_that_was_not_evaluated(tmp_path):
    model_path = tmp_path / "model"
    model_path.write_bytes(b"not the evaluated model")
    evaluation_path = tmp_path / "evaluation"
    evaluation_path.write_text(json.dumps({"model_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="is not the evaluated model"):
        export.python_func(
            mnist_dir=MNIST,
            parity_samples=8,
            parity_tolerance=1e-4,
            model=artifact(dsl.Model, model_path),
            evaluation=artifact(dsl.Artifact, evaluation_path),
            onnx_model=artifact(dsl.Model, tmp_path / "onnx_model"),
            metrics=artifact(dsl.Metrics, tmp_path / "metrics"),
        )


def test_train_component_is_a_thin_wrapper_over_the_package():
    source = train.component_spec.implementation.container.command[-1]
    for name in ("load_mnist", "split_train_validation", "make_loader", "MnistModel", "TrainConfig", "fit", "set_seed"):
        assert name in source
    assert "optimizer.step" not in source and "loss.backward" not in source


class FakeClient:
    def __init__(self, runs=(), pipeline_id=None, versions=()):
        self.runs = list(runs)
        self.pipeline_id = pipeline_id
        self.versions = list(versions)
        self.uploaded = []

    def list_runs(self, **kwargs):
        return SimpleNamespace(runs=self.runs)

    def get_pipeline_id(self, name):
        return self.pipeline_id

    def upload_pipeline(self, path, pipeline_name, description=None):
        self.uploaded.append(("pipeline", pipeline_name))
        self.pipeline_id = "pipeline-1"
        self.versions = [SimpleNamespace(display_name=pipeline_name, pipeline_version_id="version-0")]
        return SimpleNamespace(pipeline_id=self.pipeline_id)

    def list_pipeline_versions(self, pipeline_id, page_size):
        return SimpleNamespace(pipeline_versions=self.versions)

    def upload_pipeline_version(self, path, name, pipeline_id):
        self.uploaded.append(("version", name))
        version = SimpleNamespace(display_name=name, pipeline_version_id=f"version-{len(self.versions)}")
        self.versions.append(version)
        return version


def test_active_runs_block_a_new_run():
    client = FakeClient(runs=[SimpleNamespace(run_id="a", state="RUNNING"), SimpleNamespace(run_id="b", state="SUCCEEDED")])
    assert [run.run_id for run in run_cli.active_runs(client, "experiment")] == ["a"]
    assert run_cli.active_runs(FakeClient(runs=[SimpleNamespace(run_id="c", state="FAILED")]), "experiment") == []


def test_register_reuses_the_version_of_an_identical_package(tmp_path):
    package = tmp_path / "pipeline.yaml"
    package.write_text("pipeline: one")
    client = FakeClient()
    first = run_cli.register(client, package)
    second = run_cli.register(client, package)
    assert first == second
    assert client.uploaded == [("pipeline", PIPELINE_NAME), ("version", run_cli.package_version_name(package))]

    package.write_text("pipeline: two")
    third = run_cli.register(client, package)
    assert third != first
    assert client.uploaded[-1] == ("version", run_cli.package_version_name(package))


def test_cli_only_forwards_the_parameters_that_were_given():
    args = run_cli.parse_args(["--epochs", "3", "--learning-rate", "0.01"])
    assert {name: getattr(args, name) for name in run_cli.PARAMETERS if getattr(args, name) is not None} == {
        "epochs": 3,
        "learning_rate": 0.01,
    }
