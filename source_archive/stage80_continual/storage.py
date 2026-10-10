"""Cycle commits: immutable checkpoint first, one atomic active pointer last."""
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class CycleStore:
    def __init__(self, root, manifest):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        p = self.root / 'manifest.json'
        if p.exists():
            if json.loads(p.read_text()) != manifest:
                raise RuntimeError('Resume rejected: effective experiment manifest changed')
        else:
            atomic_json(p, manifest)

    def active(self):
        p = self.root / 'active.json'
        if not p.exists():
            return None
        pointer = json.loads(p.read_text())
        checkpoint = self.root / pointer['checkpoint']
        record = json.loads((checkpoint / 'record.json').read_text())
        if record['cycle'] != pointer['cycle'] or digest(record) != pointer['record_hash']:
            raise RuntimeError('Checkpoint/pointer mismatch')
        return checkpoint, record

    def _remove_owned(self, path):
        path = Path(path).resolve()
        if path.parent != self.root or not path.name.startswith(('pending_', 'cycle_')):
            raise RuntimeError('Refusing cleanup outside owned checkpoint directories')
        if path.exists():
            shutil.rmtree(path)

    def pending(self, cycle):
        active = self.active()
        if cycle != (active[1]['cycle'] + 1 if active else 1):
            raise RuntimeError('Cycles must commit in order')
        p = self.root / f'pending_{cycle:04d}'
        self._remove_owned(p)
        p.mkdir()
        return p

    def commit(self, pending, record):
        pending = Path(pending).resolve()
        cycle = record['cycle']
        if pending != self.root / f'pending_{cycle:04d}':
            raise RuntimeError('Invalid pending checkpoint')
        if not record.get('reload_parity', {}).get('passed'):
            raise RuntimeError('Reload verification must pass before commit')
        atomic_json(pending / 'record.json', record)
        destination = self.root / f'cycle_{cycle:04d}'
        # A crash after rename but before pointer update leaves an orphan only.
        self._remove_owned(destination)
        os.replace(pending, destination)
        atomic_json(self.root / 'active.json', {
            'cycle': cycle, 'checkpoint': destination.name,
            'record_hash': digest(record), 'manifest_hash': digest(self.manifest),
        })
        # Keep the previous completed cycle available for rollback and debugging.
        for old in self.root.glob('cycle_*'):
            if old.is_dir() and int(old.name.split('_')[1]) < cycle - 1:
                self._remove_owned(old)
        return destination


def exact_score(response, expected):
    return ' '.join(response.lower().split()).strip('.! ') == ' '.join(expected.lower().split()).strip('.! ')
