import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Literal

JOB_TTL_SECONDS = 7 * 24 * 60 * 60


@dataclass(frozen=True)
class Job:
    id: str
    user_id: str | None
    model_id: str
    tier: Literal["free", "pro"]
    config: dict
    status: Literal["queued", "running", "done", "failed"]
    result: dict | None
    error: str | None
    created_at: str
    completed_at: str | None


def _key(job_id: str) -> str:
    return f"job:{job_id}"


class JobStore:
    def __init__(self, redis_client: Any):
        self._redis = redis_client

    def create(
        self,
        model_id: str,
        tier: Literal["free", "pro"],
        config: dict,
        user_id: str | None,
    ) -> Job:
        job = Job(
            id=str(uuid.uuid4()),
            user_id=user_id,
            model_id=model_id,
            tier=tier,
            config=config,
            status="queued",
            result=None,
            error=None,
            created_at=datetime.now(timezone.utc).isoformat(),
            completed_at=None,
        )
        self._write(job)
        return job

    def get(self, job_id: str) -> Job | None:
        raw = self._redis.get(_key(job_id))
        if raw is None:
            return None
        data = json.loads(raw)
        return Job(**data)

    def set_status(self, job_id: str, status: Literal["queued", "running", "done", "failed"]) -> None:
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"job {job_id} not found")
        updated = Job(**{**asdict(job), "status": status})
        self._write(updated)

    def set_result(
        self,
        job_id: str,
        status: Literal["done", "failed"],
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"job {job_id} not found")
        updated = Job(
            **{
                **asdict(job),
                "status": status,
                "result": result,
                "error": error,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._write(updated)

    def _write(self, job: Job) -> None:
        self._redis.set(_key(job.id), json.dumps(asdict(job)), ex=JOB_TTL_SECONDS)
