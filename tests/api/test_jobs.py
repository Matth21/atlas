import fakeredis
import pytest

from atlas.api.jobs import Job, JobStore


@pytest.fixture
def store():
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    return JobStore(client)


def test_create_returns_queued_job(store):
    job = store.create(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tier="free",
        config={},
        user_id=None,
    )
    assert job.status == "queued"
    assert job.model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert job.tier == "free"
    assert job.user_id is None
    assert job.result is None
    assert job.error is None
    assert job.completed_at is None
    assert len(job.id) > 0


def test_get_returns_created_job(store):
    created = store.create(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tier="pro",
        config={"quality": 0.8},
        user_id="user_123",
    )
    fetched = store.get(created.id)
    assert fetched == created


def test_get_missing_job_returns_none(store):
    assert store.get("does-not-exist") is None


def test_set_status_updates_job(store):
    job = store.create(model_id="m", tier="free", config={}, user_id=None)
    store.set_status(job.id, "running")
    fetched = store.get(job.id)
    assert fetched.status == "running"


def test_set_result_success_sets_completed_at(store):
    job = store.create(model_id="m", tier="free", config={}, user_id=None)
    store.set_result(job.id, "done", result={"bits": 4.0})
    fetched = store.get(job.id)
    assert fetched.status == "done"
    assert fetched.result == {"bits": 4.0}
    assert fetched.error is None
    assert fetched.completed_at is not None


def test_set_result_failure_sets_error(store):
    job = store.create(model_id="m", tier="free", config={}, user_id=None)
    store.set_result(job.id, "failed", error="model too large")
    fetched = store.get(job.id)
    assert fetched.status == "failed"
    assert fetched.error == "model too large"
    assert fetched.result is None
