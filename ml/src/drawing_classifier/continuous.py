"""Fixed feedback split, incumbent comparison and authenticated release client."""
import hashlib
import json
import os
from urllib.request import Request, urlopen


def request(base, path, data=None):
    body = None if data is None else json.dumps(data, allow_nan=False).encode()
    req = Request(base.rstrip('/') + path, data=body, headers={
        'Authorization': 'Bearer ' + os.environ['FEEDBACK_TOKEN'],
        'Content-Type': 'application/json',
    })
    return urlopen(req, timeout=120)


def split_feedback(records):
    # Content-based membership never changes as new examples arrive.
    from drawing_classifier.feedback import validate
    train, holdout = [], []
    seen = set()
    for row in records:
        row = validate(row)
        identity = hashlib.sha256(bytes(round(x * 255) for x in row['pixels'])).hexdigest()
        if identity in seen:
            continue
        seen.add(identity)
        (holdout if int(identity[:8], 16) % 5 == 0 else train).append(row)
    return train, holdout


def feedback_dataset(records):
    import torch
    from torch.utils.data import TensorDataset
    return TensorDataset(torch.tensor([r['pixels'] for r in records], dtype=torch.float32).reshape(-1, 1, 28, 28),
                         torch.tensor([r['label'] for r in records], dtype=torch.long))


def should_promote(candidate, incumbent, feedback_candidate, feedback_incumbent, min_accuracy):
    return (candidate >= min_accuracy and candidate >= incumbent
            and feedback_candidate > feedback_incumbent)


def compare_and_promote(base, candidate_path, records, mnist_dir, min_accuracy=0.9, min_holdout=20):
    import base64
    import numpy as np
    import onnx
    import torch
    from onnx.reference import ReferenceEvaluator
    from drawing_classifier.data import load_mnist, make_loader

    if min_holdout < 20:
        raise ValueError('at least 20 holdout drawings are required')
    _, heldout = split_feedback(records)
    if len(heldout) < min_holdout:
        return {'promoted': False, 'reason': 'insufficient feedback holdout', 'holdout_samples': len(heldout)}
    with request(base, '/internal/model.onnx') as response:
        incumbent_bytes = response.read()
        version = response.headers['X-Model-Version']
    candidate = ReferenceEvaluator(onnx.load(candidate_path))
    incumbent = ReferenceEvaluator(onnx.load_model_from_string(incumbent_bytes))

    def accuracy(evaluator, dataset):
        correct = total = 0
        for images, labels in make_loader(dataset, 256, shuffle=False):
            output = evaluator.run(None, {'input': images.numpy()})[0]
            if not np.isfinite(output).all():
                raise ValueError('non-finite model output')
            correct += int((torch.from_numpy(output.argmax(1)) == labels).sum())
            total += len(labels)
        return correct / total

    test = load_mnist(mnist_dir, train=False, download=False)
    feedback = feedback_dataset(heldout)
    report = {
        'candidate_mnist': accuracy(candidate, test), 'incumbent_mnist': accuracy(incumbent, test),
        'candidate_feedback': accuracy(candidate, feedback), 'incumbent_feedback': accuracy(incumbent, feedback),
        'holdout_samples': len(heldout), 'expected_version': version,
        'snapshot_sha256': hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
    }
    report['promoted'] = should_promote(report['candidate_mnist'], report['incumbent_mnist'],
                                        report['candidate_feedback'], report['incumbent_feedback'], min_accuracy)
    if report['promoted']:
        from pathlib import Path
        with request(base, '/internal/promote', {
            'model': base64.b64encode(Path(candidate_path).read_bytes()).decode(),
            'evaluation': report, 'expected_version': version,
        }) as response:
            report['version'] = json.load(response)['version']
    return report
