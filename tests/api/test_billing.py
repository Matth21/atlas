from unittest.mock import MagicMock

import pytest

from atlas.api.billing import BillingError, create_usage_record


def test_create_usage_record_calls_stripe_and_returns_id():
    stripe_client = MagicMock()
    stripe_client.billing.MeterEvent.create.return_value = MagicMock(id="evt_123")

    result = create_usage_record(stripe_client, customer_id="cus_abc", job_id="job_xyz")

    assert result == "evt_123"
    stripe_client.billing.MeterEvent.create.assert_called_once_with(
        event_name="atlas_compress_job",
        payload={"stripe_customer_id": "cus_abc", "value": "1"},
        identifier="job_xyz",
    )


def test_create_usage_record_wraps_stripe_errors():
    stripe_client = MagicMock()
    stripe_client.billing.MeterEvent.create.side_effect = RuntimeError("stripe down")

    with pytest.raises(BillingError):
        create_usage_record(stripe_client, customer_id="cus_abc", job_id="job_xyz")
