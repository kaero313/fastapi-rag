from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from time import time
from typing import Any
from uuid import uuid4


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


@dataclass
class JobRecord:
    id: str
    status: JobStatus
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


_jobs: dict[str, JobRecord] = {}
_lock = Lock()


def create_job() -> JobRecord:
    job = JobRecord(id=uuid4().hex, status=JobStatus.queued, created_at=time())
    with _lock:
        _jobs[job.id] = job
    return job


def get_job(job_id: str) -> JobRecord | None:
    with _lock:
        return _jobs.get(job_id)


def mark_running(job_id: str) -> None:
    _update(job_id, status=JobStatus.running, started_at=time())


def mark_completed(job_id: str, result: dict[str, Any]) -> None:
    _update(
        job_id,
        status=JobStatus.completed,
        finished_at=time(),
        result=result,
    )


def mark_failed(job_id: str, error: str) -> None:
    _update(job_id, status=JobStatus.failed, finished_at=time(), error=error)


def serialize_job(job: JobRecord) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "status": job.status.value,
        "created_at": _format_ts(job.created_at),
        "started_at": _format_ts(job.started_at),
        "finished_at": _format_ts(job.finished_at),
        "error": job.error,
        "result": job.result,
    }


def _update(job_id: str, **updates: Any) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        for key, value in updates.items():
            setattr(job, key, value)


def _format_ts(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
