import time
from unittest.mock import MagicMock

import fakeredis
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from atlas.api.app import create_app
from atlas.api.free_tier import FREE_TIER_CONFIG, FREE_TIER_MODELS
from atlas.api.jobs import JobStore

ISSUER = "https://clerk.example.com"


class _FakeSigningKey:
    def __init__(self, key):
        self.key = key


class _FakeJWKSClient:
    def __init__(self, public_key):
        self._public_key = public_key

    def get_signing_key_from_jwt(self, token):
        return _FakeSigningKey(self._public_key)


@pytest.fixture(scope="module")
def keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def _token(private_key, subject="user_123"):
    payload = {"iss": ISSUER, "sub": subject, "exp": int(time.time()) + 3600}
    return jwt.encode(payload, private_key, algorithm="RS256")


@pytest.fixture
def client(keypair):
    _, public_key = keypair
    job_store = JobStore(fakeredis.FakeStrictRedis(decode_responses=True))
    jwks_client = _FakeJWKSClient(public_key)
    enqueue = MagicMock()
    app = create_app(job_store, jwks_client, ISSUER, enqueue)
    test_client = TestClient(app)
    test_client.job_store = job_store
    test_client.enqueue = enqueue
    return test_client


def test_free_tier_compress_accepted(client):
    model_id = next(iter(FREE_TIER_MODELS))
    response = client.post("/compress", json={"model_id": model_id, "config": dict(FREE_TIER_CONFIG)})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert "job_id" in body
    client.enqueue.assert_called_once_with(body["job_id"])


def test_free_tier_non_curated_model_rejected(client):
    response = client.post("/compress", json={"model_id": "meta-llama/Llama-3.1-70B", "config": dict(FREE_TIER_CONFIG)})

    assert response.status_code == 403
    client.enqueue.assert_not_called()


def test_pro_tier_requires_auth_header(client):
    response = client.post(
        "/compress",
        json={"model_id": "meta-llama/Llama-3.1-8B", "config": {"quality": 0.9, "target": "quality", "output_format": "mlx", "mode": "mixed"}},
    )

    assert response.status_code == 402


def test_pro_tier_with_valid_token_accepted(client, keypair):
    private_key, _ = keypair
    token = _token(private_key)

    response = client.post(
        "/compress",
        json={"model_id": "meta-llama/Llama-3.1-8B", "config": {"quality": 0.9, "target": "quality", "output_format": "mlx", "mode": "mixed"}},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 202
    client.enqueue.assert_called_once()


def test_pro_tier_with_invalid_token_rejected(client):
    response = client.post(
        "/compress",
        json={"model_id": "meta-llama/Llama-3.1-8B", "config": {"quality": 0.9, "target": "quality", "output_format": "mlx", "mode": "mixed"}},
        headers={"Authorization": "Bearer garbage-token"},
    )

    assert response.status_code == 402


def test_get_compress_returns_job_status(client):
    model_id = next(iter(FREE_TIER_MODELS))
    post_response = client.post("/compress", json={"model_id": model_id, "config": dict(FREE_TIER_CONFIG)})
    job_id = post_response.json()["job_id"]

    get_response = client.get(f"/compress/{job_id}")

    assert get_response.status_code == 200
    body = get_response.json()
    assert body["job_id"] == job_id
    assert body["status"] == "queued"
    assert body["result"] is None
    assert body["error"] is None


def test_get_compress_unknown_job_returns_404(client):
    response = client.get("/compress/does-not-exist")
    assert response.status_code == 404
