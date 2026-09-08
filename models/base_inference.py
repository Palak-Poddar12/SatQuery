import torch
import open_clip
from PIL import Image
import base64, io, time

# RS land cover classes
RS_CLASSES = [
    "urban area with buildings and roads",
    "forest and dense vegetation",
    "agricultural farmland and crops",
    "water body river or lake",
    "bare soil or desert",
    "residential area",
    "industrial area",
    "wetland or marsh",
]

_model = None
_preprocess = None
_tokenizer = None


def load_model():
    global _model, _preprocess, _tokenizer
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _model.eval()
    return _model, _preprocess, _tokenizer


def infer(image_b64: str, question: str) -> dict:
    t0 = time.time()
    model, preprocess, tokenizer = load_model()

    img = Image.open(
        io.BytesIO(base64.b64decode(image_b64))
    ).convert("RGB")

    img_tensor = preprocess(img).unsqueeze(0)
    text_tokens = tokenizer(RS_CLASSES)

    with torch.no_grad():
        img_feat = model.encode_image(img_tensor)
        txt_feat = model.encode_text(text_tokens)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        probs = (img_feat @ txt_feat.T).squeeze().softmax(dim=-1)

    top3_idx = probs.topk(3).indices.tolist()
    top3 = [RS_CLASSES[i] for i in top3_idx]

    return {
        "answer": f"This image contains: {', '.join(top3)}",
        "confidence": round(probs[top3_idx[0]].item(), 3),
        "model_used": "CLIP-ViT-B-32",
        "inference_time_ms": int((time.time() - t0) * 1000),
        "task_type": "base",
    }
