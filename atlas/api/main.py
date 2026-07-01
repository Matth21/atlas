import os

import redis

from atlas.api.app import create_app
from atlas.api.auth import get_jwks_client
from atlas.api.jobs import JobStore
from atlas.api.tasks import compress_task

REDIS_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CLERK_JWKS_URL = os.environ.get("CLERK_JWKS_URL", "")
CLERK_ISSUER = os.environ.get("CLERK_ISSUER", "")

_redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
_job_store = JobStore(_redis_client)
_jwks_client = get_jwks_client(CLERK_JWKS_URL) if CLERK_JWKS_URL else None


def _enqueue(job_id: str) -> None:
    compress_task.delay(job_id)


app = create_app(_job_store, _jwks_client, CLERK_ISSUER, _enqueue)
