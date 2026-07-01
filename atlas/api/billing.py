class BillingError(RuntimeError):
    pass


def create_usage_record(
    stripe_client,
    customer_id: str,
    job_id: str,
    event_name: str = "atlas_compress_job",
) -> str:
    try:
        event = stripe_client.billing.MeterEvent.create(
            event_name=event_name,
            payload={"stripe_customer_id": customer_id, "value": "1"},
            identifier=job_id,
        )
    except Exception as e:
        raise BillingError(f"failed to record usage for job {job_id}: {e}") from e
    return event.id
