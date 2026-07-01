from unittest.mock import MagicMock

from atlas.api.jobs import JobStore
from atlas.api.tasks import run_compress_job
import fakeredis


def _store():
    return JobStore(fakeredis.FakeStrictRedis(decode_responses=True))


def test_run_compress_job_success_free_tier(monkeypatch):
    store = _store()
    job = store.create(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", tier="free", config={"quality": 0.8}, user_id=None)

    pipeline = MagicMock()
    fake_result = MagicMock()
    pipeline.run.return_value = fake_result

    stripe_client = MagicMock()

    import atlas.api.tasks as tasks_module
    monkeypatch.setattr(tasks_module, "serialize_compression_result", lambda r: {"ok": True})

    run_compress_job(job.id, store, pipeline, stripe_client=stripe_client)

    updated = store.get(job.id)
    assert updated.status == "done"
    assert updated.result == {"ok": True}
    pipeline.run.assert_called_once_with(quality=0.8, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    stripe_client.billing.MeterEvent.create.assert_not_called()


def test_run_compress_job_success_pro_tier_bills_usage(monkeypatch):
    store = _store()
    job = store.create(model_id="meta-llama/Llama-3.1-8B", tier="pro", config={"quality": 0.9}, user_id="user_123")

    pipeline = MagicMock()
    pipeline.run.return_value = MagicMock()

    stripe_client = MagicMock()
    stripe_client.billing.MeterEvent.create.return_value = MagicMock(id="evt_1")

    import atlas.api.tasks as tasks_module
    monkeypatch.setattr(tasks_module, "serialize_compression_result", lambda r: {"ok": True})

    run_compress_job(job.id, store, pipeline, stripe_client=stripe_client)

    updated = store.get(job.id)
    assert updated.status == "done"
    stripe_client.billing.MeterEvent.create.assert_called_once()


def test_run_compress_job_pipeline_failure_sets_failed():
    store = _store()
    job = store.create(model_id="m", tier="free", config={}, user_id=None)

    pipeline = MagicMock()
    pipeline.run.side_effect = RuntimeError("model too large for available hardware")

    run_compress_job(job.id, store, pipeline, stripe_client=None)

    updated = store.get(job.id)
    assert updated.status == "failed"
    assert "model too large" in updated.error


def test_run_compress_job_billing_failure_does_not_flip_done_to_failed(monkeypatch):
    store = _store()
    job = store.create(model_id="meta-llama/Llama-3.1-8B", tier="pro", config={"quality": 0.9}, user_id="user_123")

    pipeline = MagicMock()
    pipeline.run.return_value = MagicMock()

    stripe_client = MagicMock()
    stripe_client.billing.MeterEvent.create.side_effect = RuntimeError("stripe down")

    import atlas.api.tasks as tasks_module
    monkeypatch.setattr(tasks_module, "serialize_compression_result", lambda r: {"ok": True})

    run_compress_job(job.id, store, pipeline, stripe_client=stripe_client)

    updated = store.get(job.id)
    assert updated.status == "done"
    assert updated.result == {"ok": True}


def test_run_compress_job_sets_running_before_calling_pipeline(monkeypatch):
    store = _store()
    job = store.create(model_id="m", tier="free", config={}, user_id=None)

    seen_status = {}

    def _capture_status(*args, **kwargs):
        seen_status["value"] = store.get(job.id).status
        return MagicMock()

    pipeline = MagicMock()
    pipeline.run.side_effect = _capture_status

    import atlas.api.tasks as tasks_module
    monkeypatch.setattr(tasks_module, "serialize_compression_result", lambda r: {"ok": True})

    run_compress_job(job.id, store, pipeline, stripe_client=None)

    assert seen_status["value"] == "running"
