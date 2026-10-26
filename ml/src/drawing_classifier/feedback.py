"""Validated, deduplicated feedback and atomic model releases on a persistent SQLite DB."""
import hashlib
import json
import math
import sqlite3
from pathlib import Path


def validate(record):
    pixels = record.get('pixels')
    if not isinstance(pixels, list) or len(pixels) != 784:
        raise ValueError('pixels must contain 784 values')
    if any(type(x) not in (float, int) or not 0 <= x <= 1 or not math.isfinite(x) for x in pixels):
        raise ValueError('pixels must be finite numbers between 0 and 1')
    if not any(pixels):
        raise ValueError('blank drawing')
    for key in ('label', 'prediction'):
        if type(record.get(key)) is not int or not 0 <= record[key] <= 9:
            raise ValueError(f'{key} must be a digit')
    version = record.get('model_version')
    if not isinstance(version, str) or not 1 <= len(version) <= 128:
        raise ValueError('model_version is required')
    return {key: record[key] for key in ('pixels', 'label', 'prediction', 'model_version')}


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY, record TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS releases (
                    version TEXT PRIMARY KEY, model BLOB NOT NULL, record TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS current (id INTEGER PRIMARY KEY CHECK(id=1), version TEXT);
            ''')

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def add(self, record):
        record = validate(record)
        # A drawing can only contribute once, even when relabeled or retried.
        identity = hashlib.sha256(bytes(round(x * 255) for x in record['pixels'])).hexdigest()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT count(*) FROM feedback').fetchone()[0] >= 2000:
                raise ValueError('feedback capacity reached (2000 drawings)')
            db.execute('INSERT OR IGNORE INTO feedback(id,record) VALUES (?,?)',
                       (identity, json.dumps(record, allow_nan=False)))
        return identity

    def snapshot(self):
        with self.connect() as db:
            return [dict(id=row[0], **json.loads(row[1])) for row in
                    db.execute('SELECT id,record FROM feedback ORDER BY id')]

    def current(self):
        with self.connect() as db:
            row = db.execute('SELECT r.version,r.model,r.record FROM releases r JOIN current c ON r.version=c.version WHERE c.id=1').fetchone()
        return None if row is None else (row[0], row[1], json.loads(row[2]))

    def promote(self, model, record, expected):
        version = hashlib.sha256(model).hexdigest()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT version FROM current WHERE id=1').fetchone()
            current = row[0] if row else 'bundled'
            if current != expected:
                raise ValueError('current model changed during evaluation; retry next run')
            db.execute('INSERT OR IGNORE INTO releases(version,model,record) VALUES (?,?,?)',
                       (version, model, json.dumps(record, allow_nan=False)))
            db.execute('INSERT OR REPLACE INTO current(id,version) VALUES (1,?)', (version,))
        return version
