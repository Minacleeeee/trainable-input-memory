"""Cooperative wall-clock limit; incomplete work never becomes a resume point."""
from datetime import datetime, timezone


class DeadlineReached(RuntimeError):
    def __init__(self, progress):
        super().__init__('Authorized execution window ended')
        self.progress=progress or {}


def check_deadline(stop_at, progress=None, now=None):
    if stop_at is None:
        return
    end=datetime.fromisoformat(stop_at)
    if end.tzinfo is None:
        raise ValueError('Deadline must include an explicit timezone')
    current=now or datetime.now(timezone.utc)
    if current>=end:
        raise DeadlineReached(progress)
