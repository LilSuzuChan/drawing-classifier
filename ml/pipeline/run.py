from __future__ import annotations

import argparse
import hashlib
import os
import socket
import subprocess
import sys
import time

from kfp import Client

from pipeline.training_pipeline import PIPELINE_NAME, PIPELINE_PACKAGE, compile_pipeline

EXPERIMENT_NAME = "drawing-classifier"
SERVICE_ACCOUNT = "pipeline-runner"
ACTIVE_STATES = {"PENDING", "RUNNING", "CANCELING", "PAUSED"}
PARAMETERS = {
    "epochs": int,
    "batch_size": int,
    "learning_rate": float,
    "seed": int,
    "validation_size": int,
    "min_accuracy": float,
    "parity_samples": int,
    "parity_tolerance": float,
}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_port_forward():
    port = free_port()
    proc = subprocess.Popen(
        ["kubectl", "-n", "kubeflow", "port-forward", "svc/ml-pipeline-ui", f"{port}:80"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return proc, f"http://127.0.0.1:{port}"
        except OSError:
            time.sleep(1)
    proc.terminate()
    sys.exit("port-forward to ml-pipeline-ui did not become ready")


def package_version_name(path) -> str:
    with open(path, "rb") as file:
        return f"sha256-{hashlib.sha256(file.read()).hexdigest()[:16]}"


def register(client: Client, path) -> tuple[str, str]:
    version_name = package_version_name(path)
    pipeline_id = client.get_pipeline_id(PIPELINE_NAME)
    if pipeline_id is None:
        pipeline_id = client.upload_pipeline(
            str(path), pipeline_name=PIPELINE_NAME, description="MNIST drawing classifier training"
        ).pipeline_id
    versions = client.list_pipeline_versions(pipeline_id, page_size=100).pipeline_versions or []
    for version in versions:
        if version.display_name == version_name:
            return pipeline_id, version.pipeline_version_id
    version = client.upload_pipeline_version(str(path), version_name, pipeline_id=pipeline_id)
    return pipeline_id, version.pipeline_version_id


def active_runs(client: Client, experiment_id: str):
    runs = client.list_runs(experiment_id=experiment_id, page_size=100, sort_by="created_at desc").runs or []
    return [run for run in runs if run.state in ACTIVE_STATES]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Register and run the drawing-classifier training pipeline on KFP")
    parser.add_argument("--host", default=os.environ.get("KFP_HOST"), help="KFP UI/API URL; a port-forward is started when omitted")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--no-wait", action="store_true")
    for name, kind in PARAMETERS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=kind)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    arguments = {name: getattr(args, name) for name in PARAMETERS if getattr(args, name) is not None}
    package = compile_pipeline(PIPELINE_PACKAGE)

    port_forward = None
    host = args.host
    if not host:
        port_forward, host = start_port_forward()
    try:
        client = Client(host=host)
        experiment = client.create_experiment(EXPERIMENT_NAME)
        running = active_runs(client, experiment.experiment_id)
        if running:
            sys.exit(f"{len(running)} run(s) of {EXPERIMENT_NAME} are still active: {', '.join(run.run_id for run in running)}")
        pipeline_id, version_id = register(client, package)
        run = client.run_pipeline(
            experiment_id=experiment.experiment_id,
            job_name=f"{PIPELINE_NAME}-{time.strftime('%Y%m%d-%H%M%S')}",
            params=arguments,
            pipeline_id=pipeline_id,
            version_id=version_id,
            enable_caching=False,
            service_account=SERVICE_ACCOUNT,
        )
        print(f"run_id={run.run_id} pipeline_id={pipeline_id} version_id={version_id} arguments={arguments}")
        if args.no_wait:
            return
        result = client.wait_for_run_completion(run.run_id, timeout=args.timeout)
        print(f"state={result.state}")
        if result.state != "SUCCEEDED":
            sys.exit(1)
    finally:
        if port_forward:
            port_forward.terminate()


if __name__ == "__main__":
    main()
