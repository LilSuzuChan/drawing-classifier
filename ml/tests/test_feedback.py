import hashlib
import json
from pathlib import Path

import pytest
from drawing_classifier.feedback import Store, validate
from drawing_classifier.continuous import split_feedback, should_promote


def record(value=1):
    return dict(pixels=[value] + [0] * 783, label=3, prediction=8, model_version='bundled')


@pytest.mark.parametrize('change', [dict(pixels=[0]*784), dict(pixels=[float('nan')]*784),
    dict(pixels=[1]*783), dict(pixels=[True]*784), dict(pixels=[10**1000]*784), dict(label=True), dict(label=10), dict(model_version='')])
def test_bad_feedback(change):
    with pytest.raises(ValueError):
        validate(dict(record(), **change))


def test_persistent_dedup_and_atomic_release(tmp_path):
    store = Store(tmp_path/'feedback.db')
    identity = store.add(record())
    assert store.add(dict(record(), label=2)) == identity
    assert len(Store(tmp_path/'feedback.db').snapshot()) == 1
    assert store.snapshot()[0]['label'] == 3
    first = store.promote(b'model1', {'accuracy': 0.97}, 'bundled')
    with pytest.raises(ValueError, match='changed'):
        store.promote(b'model2', {}, 'bundled')
    assert store.current() == (first, b'model1', {'accuracy': 0.97})
    second = store.promote(b'model2', {}, first)
    assert second == hashlib.sha256(b'model2').hexdigest()


def test_fixed_disjoint_split():
    records = [record(i/255) for i in range(1, 255)]
    train, holdout = split_feedback(records)
    assert train and holdout
    keys = lambda rows: {json.dumps(r['pixels']) for r in rows}
    assert not keys(train) & keys(holdout)
    assert split_feedback(records + records) == (train, holdout)
    later_train, later_holdout = split_feedback(records[:100])
    assert keys(later_train) <= keys(train)
    assert keys(later_holdout) <= keys(holdout)


@pytest.mark.parametrize('candidate,incumbent,fc,fi,passed', [
    (0.97,0.96,0.9,0.8,True), (0.94,0.93,0.9,0.8,False),
    (0.96,0.97,0.9,0.8,False), (0.97,0.96,0.8,0.8,False),
    (0.97,0.96,0.7,0.8,False),
])
def test_release_gate(candidate, incumbent, fc, fi, passed):
    assert should_promote(candidate, incumbent, fc, fi, 0.95) is passed


def test_compare_and_promote_uses_actual_onnx_and_incumbent(tmp_path, monkeypatch):
    import io
    import numpy as np
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    import torch
    from torch.utils.data import TensorDataset
    from drawing_classifier import continuous, data

    def model(digit):
        weights = np.zeros((784, 10), dtype=np.float32)
        bias = np.zeros(10, dtype=np.float32)
        bias[digit] = 1
        graph = helper.make_graph([
            helper.make_node('Flatten', ['input'], ['flat'], axis=1),
            helper.make_node('MatMul', ['flat', 'weights'], ['logits']),
            helper.make_node('Add', ['logits', 'bias'], ['output']),
        ], 'constant-digit', [helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch',1,28,28])],
           [helper.make_tensor_value_info('output', TensorProto.FLOAT, ['batch',10])],
           initializer=[numpy_helper.from_array(weights, 'weights'), numpy_helper.from_array(bias, 'bias')])
        return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)]).SerializeToString()

    candidate = tmp_path/'candidate.onnx'
    candidate.write_bytes(model(1))
    records = [dict(record(i/255), label=1) for i in range(1,255)]
    dataset = TensorDataset(torch.zeros(8,1,28,28), torch.ones(8, dtype=torch.long))
    monkeypatch.setattr(data, 'load_mnist', lambda *a, **k: dataset)
    baseline = [model(0)]
    promoted = []

    def request(base, path, payload=None):
        if path == '/internal/model.onnx':
            response = io.BytesIO(baseline[0])
            response.headers = {'X-Model-Version': 'baseline'}
            return response
        assert path == '/internal/promote'
        assert payload['expected_version'] == 'baseline'
        assert payload['evaluation']['candidate_mnist'] == 1.0
        promoted.append(payload)
        return io.BytesIO(b'{"version":"new-model"}')

    monkeypatch.setattr(continuous, 'request', request)
    result = continuous.compare_and_promote('http://api', str(candidate), records, '/mnist')
    assert result['promoted'] is True
    assert result['version'] == 'new-model'
    assert len(promoted) == 1
    baseline[0] = model(1)
    result = continuous.compare_and_promote('http://api', str(candidate), records, '/mnist')
    assert result['promoted'] is False
    assert len(promoted) == 1
    result = continuous.compare_and_promote('http://api', str(candidate), [], '/mnist')
    assert result['promoted'] is False
    assert result['reason'] == 'insufficient feedback holdout'


def test_training_merges_only_feedback_train_partition(tmp_path, monkeypatch):
    from types import SimpleNamespace as Obj
    import torch
    from torch.utils.data import TensorDataset
    from kfp import dsl
    from drawing_classifier import data, training
    from pipeline.components import train
    from drawing_classifier.continuous import split_feedback

    raw = tmp_path/'MNIST/raw'
    raw.mkdir(parents=True)
    for name in ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte',
                 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte'):
        (raw/name).write_bytes(b'synthetic')
    monkeypatch.setattr(data, 'load_mnist', lambda *a, **k: TensorDataset(torch.zeros(20,1,28,28), torch.zeros(20, dtype=torch.long)))
    records = [record(i/255) for i in range(1,255)]
    feedback_file = tmp_path/'feedback.json'
    feedback_file.write_text(json.dumps(records))
    expected_train, expected_holdout = split_feedback(records)
    def fit(net, loader, config, device, validation_loader, on_epoch_end):
        assert len(loader.dataset) == 15 + len(expected_train)
        feedback_data = loader.dataset.datasets[1]
        assert [float(x[0].flatten()[0]) for x in feedback_data] == pytest.approx([r['pixels'][0] for r in expected_train])
        assert len(validation_loader.dataset) == 5
        return Obj(epochs=[Obj(epoch=1, train_loss=1.0, validation=Obj(loss=1.0, accuracy=0.9))])
    monkeypatch.setattr(training, 'fit', fit)
    artifact = lambda kind, name: kind(name=name, uri=str(tmp_path/name))
    config = artifact(dsl.Artifact, 'config.json')
    train.python_func(mnist_dir=str(tmp_path), epochs=1, batch_size=8, learning_rate=0.001,
                      seed=42, validation_size=5, num_threads=1, image='test', run_id='run',
                      model=artifact(dsl.Model, 'model'), training_config=config,
                      metrics=artifact(dsl.Metrics, 'metrics'),
                      feedback=dsl.Artifact(name='feedback', uri=str(feedback_file)))
    result = json.loads(Path(config.path).read_text())
    assert result['feedback_samples'] == len(expected_train)
    assert result['feedback_sha256'] == hashlib.sha256(feedback_file.read_bytes()).hexdigest()
