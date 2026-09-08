import time

from models.base_inference import infer
from models.captioning import run_caption


def run_change_vqa(
    img_t1_b64: str,
    img_t2_b64: str,
    question: str,
) -> dict:
    t0 = time.time()

    # Fast path - caption both images.
    cap_t1 = run_caption(img_t1_b64, style="brief")
    cap_t2 = run_caption(img_t2_b64, style="brief")

    # Combined question.
    combined_q = (
        f"Before: {cap_t1['caption']}. "
        f"After: {cap_t2['caption']}. "
        f"{question}"
    )

    result = infer(img_t1_b64, combined_q)

    return {
        "change_description": result["answer"],
        "change_mask_b64": None,
        "changed_regions": [],
        "confidence": result["confidence"],
        "temporal_metadata": {
            "t1_desc": cap_t1["caption"],
            "t2_desc": cap_t2["caption"],
        },
        "inference_time_ms": int((time.time() - t0) * 1000),
        "model_used": result["model_used"],
        "task_type": "change_vqa",
    }
