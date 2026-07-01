import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from atlas.api.auth import AuthError, verify_clerk_jwt

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
    public_key = private_key.public_key()
    return private_key, public_key


def _make_token(private_key, issuer=ISSUER, subject="user_123", exp_offset=3600):
    payload = {
        "iss": issuer,
        "sub": subject,
        "exp": int(time.time()) + exp_offset,
    }
    return jwt.encode(payload, private_key, algorithm="RS256")


def test_valid_token_returns_user_id(keypair):
    private_key, public_key = keypair
    token = _make_token(private_key)
    jwks_client = _FakeJWKSClient(public_key)

    user_id = verify_clerk_jwt(token, jwks_client, issuer=ISSUER)

    assert user_id == "user_123"


def test_wrong_issuer_raises(keypair):
    private_key, public_key = keypair
    token = _make_token(private_key, issuer="https://evil.example.com")
    jwks_client = _FakeJWKSClient(public_key)

    with pytest.raises(AuthError):
        verify_clerk_jwt(token, jwks_client, issuer=ISSUER)


def test_expired_token_raises(keypair):
    private_key, public_key = keypair
    token = _make_token(private_key, exp_offset=-3600)
    jwks_client = _FakeJWKSClient(public_key)

    with pytest.raises(AuthError):
        verify_clerk_jwt(token, jwks_client, issuer=ISSUER)


def test_wrong_signing_key_raises(keypair):
    _, _ = keypair
    other_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = _make_token(other_private_key)
    _, real_public_key = keypair
    jwks_client = _FakeJWKSClient(real_public_key)

    with pytest.raises(AuthError):
        verify_clerk_jwt(token, jwks_client, issuer=ISSUER)
