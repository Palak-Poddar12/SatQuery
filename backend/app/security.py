import hashlib
import os

from fastapi import Header, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address


MASTER_API_KEY = os.getenv(
    "MASTER_API_KEY",
    "satquery-demo-secret"
)

RATE_LIMIT_PER_MINUTE = int(
    os.getenv("RATE_LIMIT_PER_MINUTE", "60")
)


limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{RATE_LIMIT_PER_MINUTE}/minute"]
)


def api_key_fingerprint(api_key: str) -> str:
    """
    Never log the actual API key.

    Instead create a short SHA-256 fingerprint.
    """

    return hashlib.sha256(
        api_key.encode()
    ).hexdigest()[:12]


async def verify_api_key(
    x_api_key: str | None = Header(default=None)
):
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
        )

    if x_api_key != MASTER_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return x_api_key


def verify_jwt(token: str):
    """
    JWT verification placeholder.

    Future implementation:
        python-jose
        OAuth2 / SSO
    """

    # TODO:
    # from jose import jwt
    # jwt.decode(...)

    return {
        "status": "not_implemented",
        "token_present": bool(token),
    }