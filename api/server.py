"""Same-origin feedback API. Internal routes require a shared Kubernetes Secret."""
import base64
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from drawing_classifier.feedback import Store

STORE = Store(os.environ.get('FEEDBACK_DB', '/data/feedback.sqlite3'))
TOKEN = os.environ['FEEDBACK_TOKEN']
if len(TOKEN) < 32:
    raise ValueError('FEEDBACK_TOKEN must have at least 32 characters')
BUNDLED = Path(os.environ.get('BUNDLED_MODEL', '/app/bundled.onnx')).read_bytes()


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data, content_type='application/json'):
        body = json.dumps(data, allow_nan=False).encode() if content_type == 'application/json' else data
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + TOKEN)

    def do_GET(self):
        if self.path == '/healthz':
            return self.reply(200, {'ok': True})
        if self.path.startswith('/internal/') and not self.authorized():
            return self.reply(401, {'error': 'unauthorized'})
        if self.path == '/internal/snapshot':
            return self.reply(200, STORE.snapshot())
        current = STORE.current()
        version, model = current[:2] if current else ('bundled', BUNDLED)
        if self.path in ('/api/model', '/internal/model'):
            return self.reply(200, {'version': version})
        if self.path in ('/api/model.onnx', '/internal/model.onnx'):
            # Version and bytes come from the same SQLite snapshot.
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('X-Model-Version', version)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(model)))
            self.end_headers()
            return self.wfile.write(model)
        return self.reply(404, {'error': 'not found'})

    def do_POST(self):
        internal = self.path == '/internal/promote'
        if internal and not self.authorized():
            return self.reply(401, {'error': 'unauthorized'})
        if self.path not in ('/api/feedback', '/internal/promote'):
            return self.reply(404, {'error': 'not found'})
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            return self.reply(415, {'error': 'application/json required'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= (8_000_000 if internal else 32_000):
                return self.reply(413, {'error': 'invalid request size'})
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError('expected JSON object')
            if internal:
                model = base64.b64decode(data['model'], validate=True)
                version = STORE.promote(model, data['evaluation'], data['expected_version'])
                return self.reply(200, {'version': version})
            return self.reply(201, {'id': STORE.add(data)})
        except (ValueError, KeyError, TypeError) as error:
            return self.reply(400, {'error': str(error)})


if __name__ == '__main__':
    ThreadingHTTPServer(('0.0.0.0', 8080), Handler).serve_forever()
