# ml/pipeline.py
# GeoAI Analyst - Complete ML Pipeline
# Coder 1 deliverable - Coder 3 (Backend) yahi import karega

import os, io, re, time, uuid, datetime, base64, json
import numpy as np
import torch
from PIL import Image, ImageChops
from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering,
    BlipForConditionalGeneration,
)

try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("WARNING: open_clip not installed. run_grounding will be disabled.")
    print("Fix: pip install open-clip-torch")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
audit_log = []
_call_timestamps = []


class ModelRegistry:
    def __init__(self):
        self._vqa_proc = self._vqa_model = None
        self._cap_proc = self._cap_model = None
        self._clip_model = self._clip_pre = None
        self._clip_tok = None

    def load_vqa(self):
        if self._vqa_model is None:
            print("Loading VQA model...")
            self._vqa_proc = BlipProcessor.from_pretrained(
                "Salesforce/blip-vqa-base"
            )
            self._vqa_model = BlipForQuestionAnswering.from_pretrained(
                "Salesforce/blip-vqa-base"
            ).to(DEVICE).eval()
            print("VQA ready")
        return self._vqa_proc, self._vqa_model

    def load_caption(self):
        if self._cap_model is None:
            print("Loading Caption model...")
            self._cap_proc = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            )
            self._cap_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            ).to(DEVICE).eval()
            print("Caption ready")
        return self._cap_proc, self._cap_model

    def load_clip(self):
        if not CLIP_AVAILABLE:
            raise ImportError("pip install open-clip-torch")
        if self._clip_model is None:
            print("Loading CLIP model...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self._clip_model = model.to(DEVICE).eval()
            self._clip_pre = preprocess
            self._clip_tok = open_clip.get_tokenizer("ViT-B-32")
            print("CLIP ready")
        return self._clip_model, self._clip_pre, self._clip_tok

    def model_health_check(self):
        results = {}
        dummy = Image.new("RGB", (224, 224), (60, 120, 40))

        for name, callback in (("vqa", lambda: run_vqa(dummy, "test")),
                               ("caption", lambda: run_caption(dummy))):
            try:
                started = time.time()
                callback()
                results[name] = {
                    "status": "ok",
                    "latency_ms": round((time.time() - started) * 1000),
                    "gpu_mb": round(torch.cuda.memory_allocated() / 1e6, 1)
                    if DEVICE == "cuda" else 0,
                }
            except Exception as exc:
                results[name] = {"status": "error", "detail": str(exc)}
        return results


registry = ModelRegistry()


def load_image(source):
    """Accept PIL, bytes, base64 string, or file path."""
    try:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, bytes):
            with Image.open(io.BytesIO(source)) as image:
                return image.convert("RGB")
        if isinstance(source, str):
            if source.startswith("data:image"):
                source = source.split(",", 1)[1]
                raw = base64.b64decode(source, validate=True)
                with Image.open(io.BytesIO(raw)) as image:
                    return image.convert("RGB")
            with Image.open(source) as image:
                return image.convert("RGB")
    except Exception as exc:
        raise ValueError("Invalid image input") from exc
    raise ValueError(f"Unsupported image type: {type(source)}")


def run_vqa(image_input, question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty")

    image = load_image(image_input)
    if min(image.size) < 32:
        raise ValueError("Image too small (minimum 32x32)")

    processor, model = registry.load_vqa()
    inputs = processor(
        images=image,
        text=question.strip(),
        return_tensors="pt",
    ).to(DEVICE)

    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=5,
            output_scores=True,
            return_dict_in_generate=True,
        )
    inference_ms = round((time.perf_counter() - started) * 1000)

    answer = processor.decode(
        outputs.sequences[0],
        skip_special_tokens=True,
    ).strip()

    if outputs.sequences_scores is not None:
        raw_score = outputs.sequences_scores[0].item()
        confidence = round(
            min(0.99, max(0.3, 1 / (1 + np.exp(-raw_score / 5))),),
            3,
        )
    else:
        confidence = 0.82

    answer_lower = answer.lower()
    if answer_lower in {"yes", "no"}:
        category = "yes_no"
    elif answer_lower.replace(".", "").isdigit():
        category = "count"
    elif any(word in answer_lower for word in (
        "forest", "urban", "water", "farmland", "road", "crop"
    )):
        category = "land_cover"
    else:
        category = "descriptive"

    return {
        "answer": answer,
        "confidence": confidence,
        "answer_category": category,
        "model": "BLIP-VQA-Base",
        "inference_ms": inference_ms,
        "task": "single_image_vqa",
    }


def run_caption(image_input):
    image = load_image(image_input)
    if min(image.size) < 32:
        raise ValueError("Image too small (minimum 32x32)")

    processor, model = registry.load_caption()
    results = []

    for prompt in ("A satellite image showing", "Aerial view of"):
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(DEVICE)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=5,
                repetition_penalty=1.3,
            )
        inference_ms = round((time.perf_counter() - started) * 1000)
        caption = processor.decode(
            output[0],
            skip_special_tokens=True,
        ).strip()
        results.append((caption, inference_ms))

    caption, inference_ms = max(
        results,
        key=lambda item: len(item[0].split()),
    )

    tags = (
        "forest", "urban", "water", "farmland", "road", "crop",
        "vegetation", "industrial", "residential", "river", "lake",
    )
    detected = [tag for tag in tags if tag in caption.lower()]

    return {
        "caption": caption,
        "detected_classes": detected,
        "confidence": 0.85,
        "model": "BLIP-Caption-Large",
        "inference_ms": inference_ms,
        "task": "captioning",
    }


def _nms(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        current = order[0]
        keep.append(int(current))
        rest = order[1:]

        if rest.size == 0:
            break

        xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        width = np.maximum(0, xx2 - xx1)
        height = np.maximum(0, yy2 - yy1)
        intersection = width * height

        area_current = (
            boxes[current, 2] - boxes[current, 0]
        ) * (boxes[current, 3] - boxes[current, 1])
        area_rest = (
            boxes[rest, 2] - boxes[rest, 0]
        ) * (boxes[rest, 3] - boxes[rest, 1])
        overlap = intersection / (
            area_current + area_rest - intersection + 1e-8
        )
        order = rest[overlap < iou_threshold]

    return keep


def run_grounding(image_input, query):
    if not CLIP_AVAILABLE:
        return {"error": "open_clip not installed", "task": "grounding"}
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must not be empty")

    image = load_image(image_input).resize((448, 448))
    clip_model, preprocess, tokenizer = registry.load_clip()

    synonyms = {
        "water": ["water", "lake", "river", "reservoir", "pond"],
        "road": ["road", "highway", "street", "path"],
        "urban": ["urban", "city", "building", "settlement"],
    }
    queries = [query]
    for key, expanded in synonyms.items():
        if key in query.lower():
            queries = expanded
            break

    text_tokens = tokenizer(queries).to(DEVICE)
    with torch.inference_mode():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    width, height = image.size
    all_boxes = []
    all_scores = []
    started = time.perf_counter()

    for grid in (4, 3):
        cell_width = width // grid
        cell_height = height // grid
        for row in range(grid):
            for column in range(grid):
                x1 = column * cell_width
                y1 = row * cell_height
                x2 = min(x1 + cell_width * 2, width)
                y2 = min(y1 + cell_height * 2, height)
                patch = preprocess(image.crop((x1, y1, x2, y2)))

                with torch.inference_mode():
                    image_feature = clip_model.encode_image(
                        patch.unsqueeze(0).to(DEVICE)
                    )
                    image_feature = image_feature / image_feature.norm(
                        dim=-1,
                        keepdim=True,
                    )
                    score = (image_feature @ text_features.T).max().item()

                all_boxes.append([x1 / width, y1 / height, x2 / width, y2 / height])
                all_scores.append(float(score))

    scores = np.asarray(all_scores)
    threshold = scores.mean() + 0.3 * scores.std()
    candidates = [
        (box, score)
        for box, score in zip(all_boxes, all_scores)
        if score >= threshold
    ]
    if not candidates:
        best = int(np.argmax(scores))
        candidates = [(all_boxes[best], all_scores[best])]

    candidate_boxes = [item[0] for item in candidates]
    candidate_scores = [item[1] for item in candidates]
    keep = _nms(candidate_boxes, candidate_scores)

    regions = [
        {
            "label": query,
            "bbox": [round(value, 3) for value in candidate_boxes[index]],
            "confidence": round(float(candidate_scores[index]), 3),
        }
        for index in keep
    ]
    regions.sort(key=lambda item: item["confidence"], reverse=True)

    return {
        "regions": regions[:4],
        "query": query,
        "heatmap_scores": scores[:16].reshape(4, 4).tolist(),
        "model": "CLIP-ViT-B32 (multi-scale NMS)",
        "inference_ms": round((time.perf_counter() - started) * 1000),
        "task": "grounding",
    }


def run_change(image_t1_input, image_t2_input, question=""):
    image_t1 = load_image(image_t1_input).resize((448, 448))
    image_t2 = load_image(image_t2_input).resize((448, 448))

    started = time.perf_counter()
    difference = np.asarray(
        ImageChops.difference(image_t1, image_t2).convert("L"),
        dtype=np.float32,
    )
    threshold = difference.mean() + 1.5 * difference.std()
    mask = (difference > threshold).astype(np.uint8)
    change_pct = round(float(mask.mean() * 100), 2)

    array_t1 = np.asarray(image_t1, dtype=np.float32)
    array_t2 = np.asarray(image_t2, dtype=np.float32)
    delta_red = (array_t2[:, :, 0] - array_t1[:, :, 0]).mean()
    delta_blue = (array_t2[:, :, 2] - array_t1[:, :, 2]).mean()

    hints = []
    if delta_blue > 15:
        hints.append("possible flood (blue channel increase)")
    if delta_red < -15:
        hints.append("vegetation loss (red channel decrease)")
    if not hints:
        hints.append("general land-use change")

    if change_pct < 5:
        severity = "Minimal - seasonal variation likely"
    elif change_pct < 15:
        severity = "Moderate - land use transition detected"
    elif change_pct < 35:
        severity = "Significant - urban expansion or deforestation"
    else:
        severity = "Major - flood, fire, or construction event"

    ys, xs = np.where(mask)
    if len(ys):
        padding = 20
        y1 = max(0, int(ys.min()) - padding)
        y2 = min(448, int(ys.max()) + padding)
        x1 = max(0, int(xs.min()) - padding)
        x2 = min(448, int(xs.max()) + padding)
        crop = image_t2.crop((x1, y1, x2, y2))
        bbox = [round(x1 / 448, 3), round(y1 / 448, 3),
                round(x2 / 448, 3), round(y2 / 448, 3)]
    else:
        crop, bbox = image_t2, None

    vqa = run_vqa(crop, question or "What has changed in this region?")

    return {
        "description": vqa["answer"],
        "change_pct": change_pct,
        "change_severity": severity,
        "spectral_change_hints": hints,
        "confidence": round(vqa["confidence"] * 0.95, 3),
        "change_bbox": bbox,
        "model": "PixelDiff + BLIP-VQA",
        "inference_ms": round((time.perf_counter() - started) * 1000),
        "task": "change_detection",
    }


def _lee_filter(image_array, size=3):
    image_array = image_array.astype(np.float64)
    padding = size // 2
    padded = np.pad(image_array, padding, mode="reflect")
    height, width = image_array.shape
    means = np.zeros((height, width))
    variances = np.zeros((height, width))

    for row in range(height):
        for column in range(width):
            patch = padded[row:row + size, column:column + size]
            means[row, column] = patch.mean()
            variances[row, column] = patch.var()

    noise_variance = variances.mean()
    weight = variances / (variances + noise_variance + 1e-10)
    filtered = means + weight * (image_array - means)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def run_sar_fusion(optical_input, sar_input):
    optical = load_image(optical_input).resize((448, 448))
    sar_image = load_image(sar_input).resize((448, 448))
    optical_array = np.asarray(optical)
    sar_array = np.asarray(sar_image.convert("L"), dtype=np.float32)

    sar_filtered = _lee_filter(sar_array)
    sar_db = 10 * np.log10(sar_filtered.astype(np.float32) + 1e-10)
    sar_range = sar_db.max() - sar_db.min()
    sar_normalized = (
        (sar_db - sar_db.min()) / (sar_range + 1e-8) * 255
    ).astype(np.uint8)

    strategies = {
        "standard": np.stack(
            [optical_array[:, :, 0], optical_array[:, :, 1], sar_normalized],
            axis=-1,
        ),
        "sar_emphasis": np.stack(
            [optical_array[:, :, 0], sar_normalized, optical_array[:, :, 2]],
            axis=-1,
        ),
        "sar_dual": np.stack(
            [optical_array[:, :, 0], sar_normalized, sar_normalized],
            axis=-1,
        ),
    }

    best_caption = ""
    best_ms = 0
    best_strategy = "standard"

    for name, array in strategies.items():
        caption_result = run_caption(Image.fromarray(array.astype(np.uint8)))
        if len(caption_result["caption"]) > len(best_caption):
            best_caption = caption_result["caption"]
            best_ms = caption_result["inference_ms"]
            best_strategy = name

    fused = Image.fromarray(strategies[best_strategy].astype(np.uint8))
    vqa = run_vqa(
        fused,
        "Identify water bodies, urban areas, and vegetation using both optical and radar data.",
    )

    conflict = float(sar_normalized.mean()) > float(optical_array.mean()) * 1.4

    return {
        "analysis": vqa["answer"],
        "caption": best_caption,
        "confidence": round((vqa["confidence"] + 0.85) / 2, 3),
        "fusion_strategy_used": best_strategy,
        "sar_preprocessing_steps": ["lee_filter", "db_scaling", "normalize"],
        "modality_conflict_detected": conflict,
        "model": "ChannelFusion + BLIP",
        "inference_ms": vqa["inference_ms"] + best_ms,
        "task": "sar_fusion",
    }


TOOL_REGISTRY = {
    "single_image_vqa": run_vqa,
    "captioning": run_caption,
    "visual_grounding": run_grounding,
    "change_detection": run_change,
    "sar_fusion": run_sar_fusion,
}


def _rate_limit_check(max_per_min: int = 30) -> bool:
    """Allow at most ``max_per_min`` calls in the rolling 60-second window."""
    now = time.time()
    _call_timestamps[:] = [
        timestamp
        for timestamp in _call_timestamps
        if now - timestamp < 60
    ]
    if len(_call_timestamps) >= max_per_min:
        return False
    _call_timestamps.append(now)
    return True


def _classify_task(question, image_count, has_sar):
    query = question.lower()
    matched = []

    if has_sar:
        matched.append("sar_fusion")
    if image_count == 2:
        matched.append("change_detection")
    if any(word in query for word in ("where", "locate", "find", "highlight", "show")):
        matched.append("visual_grounding")
    if not question.strip():
        matched.append("captioning")
    if not matched:
        matched.append("single_image_vqa")

    priority = (
        "sar_fusion",
        "change_detection",
        "visual_grounding",
        "captioning",
        "single_image_vqa",
    )
    chosen = next(task for task in priority if task in matched)

    reasons = {
        "sar_fusion": "SAR image detected -> dual-stream fusion selected",
        "change_detection": "2 images detected -> bi-temporal change analysis",
        "visual_grounding": "Localization keyword detected -> CLIP grounding",
        "captioning": "No question -> auto-caption mode",
        "single_image_vqa": "Single image + question -> VQA pipeline",
    }
    return chosen, matched, reasons[chosen]


def run_agent(images, question="", has_sar=False):
    if not isinstance(images, (list, tuple)) or not 1 <= len(images) <= 2:
        raise ValueError("1 or 2 images are required")
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    if has_sar and len(images) != 2:
        raise ValueError("has_sar=True requires two images")

    started = time.perf_counter()
    trace = []

    def step(name, detail, status="done"):
        trace.append({
            "step": name,
            "status": status,
            "detail": detail,
            "timestamp_ms": round((time.perf_counter() - started) * 1000),
        })

    for index, image in enumerate(images):
        load_image(image)
        step("input_validation", f"image {index + 1} valid")

    if not _rate_limit_check():
        step("rate_limit", "30 requests per minute exceeded", "error")
        raise RuntimeError("Rate limit exceeded (30 requests per minute)")
    step("rate_limit", "within limit")

    task, ambiguous, reasoning = _classify_task(
        question,
        len(images),
        has_sar,
    )
    step("task_classification", f"{task}; ambiguous={ambiguous}")
    step("model_routing", f"tool={TOOL_REGISTRY[task].__name__}")

    request_id = f"rq-{uuid.uuid4().hex[:8]}"
    audit_entry = {
        "request_id": request_id,
        "task": task,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "image_count": len(images),
        "question_length": len(question),
    }
    audit_log.append(audit_entry)

    try:
        step("inference", "running", "running")
        if task == "single_image_vqa":
            result = run_vqa(images[0], question)
        elif task == "captioning":
            result = run_caption(images[0])
        elif task == "visual_grounding":
            result = run_grounding(images[0], question)
        elif task == "change_detection":
            result = run_change(images[0], images[1], question)
        else:
            result = run_sar_fusion(images[0], images[1])
        trace[-1]["status"] = "done"
        trace[-1]["detail"] = "inference completed"
    except Exception as exc:
        step("inference", str(exc), "error")
        return {
            "request_id": request_id,
            "task": task,
            "agent_reasoning": reasoning,
            "ambiguous_tasks": ambiguous,
            "result": None,
            "execution_trace": trace,
            "error": str(exc),
        }

    low_confidence_fallback = False
    if task == "single_image_vqa" and result.get("confidence", 1) < 0.5:
        caption_result = run_caption(images[0])
        result["caption_fallback"] = caption_result["caption"]
        low_confidence_fallback = True
        step("confidence_reroute", "VQA confidence below 0.5")

    audit_entry["confidence"] = result.get("confidence")
    step("audit_log", f"{request_id} logged")

    return {
        "request_id": request_id,
        "task": task,
        "agent_reasoning": reasoning,
        "ambiguous_tasks": ambiguous,
        "low_confidence_fallback": low_confidence_fallback,
        "result": result,
        "execution_trace": trace,
        "total_ms": round((time.perf_counter() - started) * 1000),
    }


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    dummy = Image.new("RGB", (224, 224), (60, 120, 40))
    output = run_agent([dummy], "What land cover is visible?")
    print("Task:", output["task"])
    print("Answer:", output["result"]["answer"])
    print("Trace:", [item["step"] for item in output["execution_trace"]])
