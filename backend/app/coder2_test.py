from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from .security import verify_api_key
from .uploads import save_upload
from .config import settings


app = FastAPI(
    title="SatQuery AI - Coder 2 Security Test",
    version="1.0.0"
)


# -----------------------------
# RATE LIMITER
# -----------------------------

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[
        f"{settings.RATE_LIMIT_PER_MINUTE}/minute"
    ]
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(
    request: Request,
    exc: RateLimitExceeded
):
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please try again later."
        }
    )


# -----------------------------
# HEALTH CHECK
# -----------------------------

@app.get("/")
async def root():
    return {
        "project": "SatQuery AI",
        "module": "Coder 2 - Cybersecurity",
        "status": "Security layer running"
    }


# -----------------------------
# AUTHENTICATION TEST
# -----------------------------

@app.get("/api/protected")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def protected_endpoint(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    return {
        "success": True,
        "message": "Authentication successful",
        "security": "API key verified"
    }


# -----------------------------
# SECURE UPLOAD TEST
# -----------------------------

@app.post("/api/upload")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def upload_endpoint(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    try:
        metadata = await save_upload(file)

        return {
            "success": True,
            "message": "Secure upload successful",
            "metadata": metadata
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Upload processing failed"
        ) from exc