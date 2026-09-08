import base64
import io
import time

from PIL import Image


def run_grounding(
    image_b64: str, query: str, model_path: str = None
) -> dict:
    t0 = time.time()

    # Stub - GeoChat integration baad mein.
    img = Image.open(
        io.BytesIO(base64.b64decode(image_b64))
    ).convert("RGB")
    w, h = img.size

    # Mock bounding box - center region.
    mock_bbox = {
        "x1": int(w * 0.2),
        "y1": int(h * 0.2),
        "x2": int(w * 0.8),
        "y2": int(h * 0.8),
        "label": query,
        "score": 0.75,
    }

    return {
        "query": query,
        "bboxes": [mock_bbox],
        "mask_b64": None,
        "model_used": "GeoChat-stub-v0",
        "inference_time_ms": int((time.time() - t0) * 1000),
        "task_type": "grounding",
    }
