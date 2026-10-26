import importlib.util
import io
import json
from pathlib import Path

import pytest


@pytest.fixture
def api(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv('FEEDBACK_TOKEN', 't'*32)
    monkeypatch.setenv('FEEDBACK_DB', str(tmp_path/'feedback.db'))
    monkeypatch.setenv('BUNDLED_MODEL', str(root/'web/public/models/mnist_model.onnx'))
    spec = importlib.util.spec_from_file_location('feedback_server', root/'api/server.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call(api, method, path, body=None, token=None):
    handler = object.__new__(api.Handler)
    handler.path = path
    raw = b'' if body is None else json.dumps(body).encode()
    handler.headers = {'Content-Length': str(len(raw)), 'Content-Type': 'application/json'}
    if token:
        handler.headers['Authorization'] = 'Bearer '+token
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()
    response = {}
    handler.send_response = lambda status: response.update(status=status)
    handler.send_header = lambda key, value: response.update({key:value})
    handler.end_headers = lambda: None
    getattr(handler, 'do_'+method)()
    response['body'] = handler.wfile.getvalue()
    return response


def test_routes_and_authorization(api):
    assert call(api, 'GET', '/internal/snapshot')['status'] == 401
    assert call(api, 'POST', '/internal/promote', {})['status'] == 401
    assert json.loads(call(api, 'GET', '/internal/snapshot', token='t'*32)['body']) == []
    response = call(api, 'GET', '/api/model.onnx')
    assert response['X-Model-Version'] == 'bundled'
    assert response['body'] == api.BUNDLED
    assert call(api, 'POST', '/api/feedback', [])['status'] == 400
    assert call(api, 'POST', '/api/feedback', {'pixels': []})['status'] == 400
    record = dict(pixels=[1]+[0]*783, label=1, prediction=2, model_version='bundled')
    assert call(api, 'POST', '/api/feedback', record)['status'] == 201
    assert call(api, 'POST', '/api/feedback', record)['status'] == 201
    assert len(json.loads(call(api, 'GET', '/internal/snapshot', token='t'*32)['body'])) == 1
