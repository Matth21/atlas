import jwt


class AuthError(ValueError):
    pass


def get_jwks_client(jwks_url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(jwks_url)


def verify_clerk_jwt(token: str, jwks_client, issuer: str) -> str:
    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=issuer,
        )
    except jwt.PyJWTError as e:
        raise AuthError(f"invalid Clerk token: {e}") from e

    sub = payload.get("sub")
    if not sub:
        raise AuthError("token missing 'sub' claim")
    return sub
