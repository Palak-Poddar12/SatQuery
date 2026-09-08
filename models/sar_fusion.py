import time

from models.base_inference import infer


def run_sar_fusion(
    optical_b64: str,
    sar_b64: str,
    question: str,
) -> dict:
    t0 = time.time()

    # Stub - optical model use kar raha hai abhi.
    result = infer(optical_b64, question)

    return {
        "answer": result["answer"],
        "detected_classes": result.get("top_classes", []),
        "fusion_confidence": result["confidence"],
        "optical_only_confidence": result["confidence"],
        "sar_contribution_score": 0.0,
        "sar_changed_answer": False,
        "inference_time_ms": int((time.time() - t0) * 1000),
        "model_used": "DualStream-stub-v0",
        "task_type": "sar_fusion",
    }
