from models.base_inference import infer as base_infer
import hashlib, time


def run_vqa(image_b64: str, question: str) -> dict:
    result = base_infer(image_b64, question)
    result["task_type"] = "vqa"
    result["image_hash"] = hashlib.sha256(
        image_b64.encode()
    ).hexdigest()[:16]
    return result
