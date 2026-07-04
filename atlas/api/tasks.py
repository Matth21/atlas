import os

from celery import Celery

from atlas.api.billing import create_usage_record
from atlas.api.jobs import JobStore
from atlas.api.serialize import serialize_compression_result
from atlas.core.pipeline import Pipeline

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")

celery_app = Celery("atlas_api", broker=CELERY_BROKER_URL, backend=CELERY_BROKER_URL)


def run_compress_job(job_id: str, job_store: JobStore, pipeline: Pipeline, stripe_client=None) -> None:
    job = job_store.get(job_id)
    if job is None:
        raise KeyError(f"job {job_id} not found")

    job_store.set_status(job_id, "running")

    try:
        config = dict(job.config)
        budget_gb = config.pop("budget_gb", None)
        if budget_gb is not None:
            # Percorso SGSR-2: budget in GB, costo misurato, piano ottimo.
            from atlas.core.model import ModelLoader
            from atlas.core.sgsr2_flow import compress_to_budget

            mi = ModelLoader().load_metadata(job.model_id)
            r = compress_to_budget(job.model_id, float(budget_gb), mi.num_params)
            serialized = {
                "engine": "sgsr2",
                "output_path": str(r.output_path),
                "budget_gb": r.budget_gb,
                "plan_bits": r.plan_bits,
                "quantized_size_mb": r.quantized_size_mb,
                "original_size_mb": r.original_size_mb,
                "assignment_summary": r.assignment_summary,
            }
        else:
            result = pipeline.run(model_id=job.model_id, **config)
            serialized = serialize_compression_result(result)
        job_store.set_result(job_id, "done", result=serialized)
    except Exception as e:
        job_store.set_result(job_id, "failed", error=str(e))
        return

    # Pro tier is only reachable via a valid Clerk JWT in the FastAPI layer,
    # which always yields a user_id, so `job.user_id is None` here is an
    # expected-impossible defensive guard, not a silent bug.
    if job.tier == "pro" and stripe_client is not None and job.user_id is not None:
        try:
            create_usage_record(stripe_client, customer_id=job.user_id, job_id=job_id)
        except Exception:
            # Billing failures must not flip a successful job back to "failed"
            # and must not overwrite its result. Fase A has no retry queue and
            # no billing-failure alerting yet, so we swallow this here.
            pass


@celery_app.task(name="compress")
def compress_task(job_id: str) -> None:
    import redis
    import stripe

    redis_client = redis.Redis.from_url(CELERY_BROKER_URL, decode_responses=True)
    job_store = JobStore(redis_client)
    pipeline = Pipeline()
    stripe_client = stripe if os.environ.get("STRIPE_API_KEY") else None
    if stripe_client is not None:
        stripe_client.api_key = os.environ["STRIPE_API_KEY"]

    run_compress_job(job_id, job_store, pipeline, stripe_client=stripe_client)
