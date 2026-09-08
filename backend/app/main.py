from fastapi import FastAPI, UploadFile, File, Depends, Request
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from .security import (
    verify_api_key,
    limiter,
    api_key_fingerprint,
)

from .uploads import save_upload
from .logging_config import log_event


app = FastAPI(
    title="SatQuery AI Secure API",
    version="1.0.0",
)


# ============================================================
# RATE LIMITER
# ============================================================

app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(
    request: Request,
    exc: RateLimitExceeded
):
    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "detail": "Rate limit exceeded. Try again later."
        },
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "SatQuery AI",
        "status": "running",
        "security": "enabled",
    }


# ============================================================
# PROTECTED TEST ENDPOINT
# ============================================================

@app.get("/api/protected")
@limiter.limit("60/minute")
async def protected_endpoint(
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    return {
        "success": True,
        "message": "API authentication successful",
    }


# ============================================================
# SECURE UPLOAD
# ============================================================

@app.post("/api/upload")
@limiter.limit("60/minute")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
):

    fingerprint = api_key_fingerprint(api_key)

    client_ip = (
        request.client.host
        if request.client
        else "unknown"
    )

    try:

        # IMPORTANT:
        # This matches your existing uploads.py
        metadata = await save_upload(file)

        log_event(
            "upload_success",
            api_key=fingerprint,
            client_ip=client_ip,
            image_id=metadata["image_id"],
            original_filename=metadata[
                "original_filename"
            ],
            size=metadata["size"],
            sha256=metadata["sha256"],
            mime_type=metadata["mime_type"],
        )

        return {
            "success": True,
            "message": "Secure upload successful",
            "metadata": metadata,
        }

    except HTTPException as exc:

    reason = exc.detail

    if exc.status_code == 413:

        log_event(
            "upload_rejected_size",
            api_key=fingerprint,
            client_ip=client_ip,
            original_filename=file.filename,
            reason=reason,
        )

    else:

        log_event(
            "upload_rejected_invalid",
            api_key=fingerprint,
            client_ip=client_ip,
            original_filename=file.filename,
            reason=reason,
        )

    raise

except Exception as exc:

    log_event(
        "upload_rejected_invalid",
        api_key=fingerprint,
        client_ip=client_ip,
        original_filename=file.filename,
        reason=str(exc),
    )

    raise


# ============================================================
# QUERY ENDPOINT
# ============================================================

@app.post("/api/query")
@limiter.limit("60/minute")
async def query_image(
    request: Request,
    api_key: str = Depends(verify_api_key),
):

    fingerprint = api_key_fingerprint(api_key)

    client_ip = (
        request.client.host
        if request.client
        else "unknown"
    )

    # ----------------------------------------
    # Audit: query received
    # ----------------------------------------

    log_event(
        "query_received",
        api_key=fingerprint,
        client_ip=client_ip,
    )

    # Temporary response.
    # Coder 3's real agent will replace this.
    execution_summary = {
        "task_type": "security_test",
        "models_used": [],
    }

    # ----------------------------------------
    # Audit: query completed
    # ----------------------------------------

    log_event(
        "query_completed",
        api_key=fingerprint,
        client_ip=client_ip,
        task_type=execution_summary[
            "task_type"
        ],
        models_used=execution_summary[
            "models_used"
        ],
    )

    return {
        "answer": "SatQuery agent integration pending.",
        "confidence": 0.0,
        "visuals": [],
        "execution_summary": execution_summary,
    }