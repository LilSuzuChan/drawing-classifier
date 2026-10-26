#!/usr/bin/env python3
"""One-time bootstrap after publishing immutable web/API/training images."""
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def kubectl(*args, data=None, capture=False):
    return subprocess.run(['kubectl', *args], input=data, text=True, check=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('web', 'api', 'training'):
        parser.add_argument(f'--{name}-image', required=True)
    parser.add_argument('--lab-root', type=Path, default=ROOT.parent/'K8s-home-lab')
    parser.add_argument('--kfp-host', required=True, help='Reachable KFP API/UI URL')
    args = parser.parse_args()
    for image in (args.web_image, args.api_image, args.training_image):
        if not re.fullmatch(r'ghcr\.io/[a-z0-9./-]+@sha256:[0-9a-f]{64}', image):
            parser.error('all images must be immutable GHCR digest references')
    network_policy = args.lab_root/'platform/kubeflow-pipelines/app/networkpolicy.yaml'
    if not network_policy.is_file():
        parser.error('--lab-root must contain the K8s-home-lab repository')
    kubectl('apply', '-f', str(network_policy))
    # Preserve the existing token on re-runs. Never emit credentials to stdout.
    kubectl('apply', '-f', str(ROOT/'k8s/namespace.yaml'))
    secret = json.loads(kubectl('-n', 'drawing-classifier', 'get', 'secret', 'drawing-feedback',
                               '--ignore-not-found', '-o', 'json', capture=True) or '{}')
    token_data = secret.get('data', {}).get('token')
    if token_data is None:
        import base64
        token_data = base64.b64encode(secrets.token_urlsafe(48).encode()).decode()
    for namespace in ('drawing-classifier', 'kubeflow'):
        body = {'apiVersion': 'v1', 'kind': 'Secret', 'metadata': {'name': 'drawing-feedback', 'namespace': namespace},
                'type': 'Opaque', 'data': {'token': token_data}}
        kubectl('apply', '-f', '-', data=json.dumps(body))
    # Reuse the registry credential already maintained for the app, without printing it.
    pull = json.loads(kubectl('-n', 'drawing-classifier', 'get', 'secret', 'ghcr-pull', '-o', 'json', capture=True))
    registry_secret = {'apiVersion': 'v1', 'kind': 'Secret',
                       'metadata': {'name': 'ghcr-pull', 'namespace': 'kubeflow'},
                       'type': pull['type'], 'data': pull['data']}
    kubectl('apply', '-f', '-', data=json.dumps(registry_secret))
    # Render locally so the working tree never acquires environment-specific credentials or tags.
    import yaml
    rendered = kubectl('kustomize', str(ROOT/'k8s'), capture=True)
    manifests = list(yaml.safe_load_all(rendered))
    for manifest in manifests:
        if manifest.get('kind') == 'Deployment':
            containers = manifest['spec']['template']['spec']['containers']
            containers[0]['image'] = args.api_image if manifest['metadata']['name'] == 'feedback-api' else args.web_image
    web = [m for m in manifests if m.get('kind') == 'Deployment' and m['metadata']['name'] == 'drawing-classifier']
    api_and_resources = [m for m in manifests if m not in web]
    kubectl('apply', '-f', '-', data=yaml.safe_dump_all(api_and_resources))
    kubectl('-n', 'drawing-classifier', 'rollout', 'status', 'deployment/feedback-api', '--timeout=180s')
    kubectl('apply', '-f', '-', data=yaml.safe_dump_all(web))
    kubectl('-n', 'drawing-classifier', 'rollout', 'status', 'deployment/drawing-classifier', '--timeout=180s')
    env = dict(os.environ, DRAWING_TRAINING_IMAGE=args.training_image)
    subprocess.run([str(ROOT/'ml/.venv/bin/python'), '-m', 'pipeline.schedule', '--host', args.kfp_host],
                   cwd=ROOT/'ml', env=env, check=True)


if __name__ == '__main__':
    main()
