"""Fail-closed lock for future standalone GPU experiments.

The existing campaign keeps its own lock. Do not alter its active protocol.
Standalone jobs refuse that campaign lock and share this additional lease.
Stale locks require explicit PID inspection; they are never removed here.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path


@contextmanager
def gpu_lease(data_root, role):
    data_root=Path(data_root)
    data_root.mkdir(parents=True,exist_ok=True)
    campaign=data_root/'campaign_seed42.lock'
    benchmark=data_root/'gpu_optimizer_benchmark.lock'
    if campaign.exists() or benchmark.exists():
        raise RuntimeError('Existing campaign/benchmark GPU lock: inspect PID and wait for completion')
    path=data_root/'standalone_gpu.lock'
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    payload=json.dumps({'pid':os.getpid(),'role':role})
    with os.fdopen(fd,'w') as f:
        f.write(payload)
    try:
        if campaign.exists() or benchmark.exists():
            raise RuntimeError('Campaign/benchmark started; standalone GPU experiment refused')
        yield
    finally:
        # Remove only the exact lease acquired by this process.
        if path.exists() and path.read_text()==payload:
            path.unlink()
