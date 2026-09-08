import hashlib
import io
import uuid

import magic

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    MAX_UPLOAD_SIZE_BYTES,
    UPLOAD_DIR,
)


CHUNK_SIZE = 1024 * 1024  # 1 MB


async def save_upload(
    file: UploadFile
) -> dict:

    # ============================================================
    # 1. Read file safely in chunks
    # ============================================================

    chunks = []
    total_size = 0

    while True:

        chunk = await file.read(CHUNK_SIZE)

        if not chunk:
            break

        total_size += len(chunk)

        # Reject immediately if size exceeds configured limit.
        if total_size > MAX_UPLOAD_SIZE_BYTES:

            raise HTTPException(
                status_code=413,
                detail="File exceeds maximum upload size",
            )

        chunks.append(chunk)

    content = b"".join(chunks)

    file_size = len(content)

    # ============================================================
    # 2. Empty file protection
    # ============================================================

    if file_size == 0:

        raise HTTPException(
            status_code=400,
            detail="Empty file is not allowed",
        )

    # ============================================================
    # 3. Content-based MIME detection
    # ============================================================

    try:

        detected_mime = magic.from_buffer(
            content,
            mime=True,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail="Unable to determine file type",
        ) from exc

    # ============================================================
    # 4. MIME allowlist
    # ============================================================

    if detected_mime not in ALLOWED_MIME_TYPES:

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {detected_mime}",
        )

    # ============================================================
    # 5. Verify actual image
    # ============================================================

    try:

        image = Image.open(
            io.BytesIO(content)
        )

        image.verify()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:

        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image",
        ) from exc

    # ============================================================
    # 6. SHA-256 integrity hash
    # ============================================================

    sha256 = hashlib.sha256(
        content
    ).hexdigest()

    # ============================================================
    # 7. Generate random UUID
    # ============================================================

    image_id = str(
        uuid.uuid4()
    )

    extension = ALLOWED_EXTENSIONS[
        detected_mime
    ]

    safe_filename = (
        image_id + extension
    )

    destination = (
        UPLOAD_DIR / safe_filename
    )

    # ============================================================
    # 8. Safe storage
    # ============================================================

    destination.write_bytes(
        content
    )

    # ============================================================
    # 9. Return metadata
    # ============================================================

    return {
        "image_id": image_id,
        "filename": safe_filename,
        "original_filename": file.filename,
        "size": file_size,
        "mime_type": detected_mime,
        "sha256": sha256,
        "path": str(destination),
    }