# GeoAI Analyst - Remote Sensing AI Platform

VQA | Caption | Grounding | Change Detection | Optical-SAR Fusion

## Team
- Coder 1 (ML): VLM inference, model pipeline
- Coder 2 (Security): Auth, rate limiting
- Coder 3 (Backend): FastAPI, agent routing
- Coder 4 (Frontend): React dashboard

---

## Coder 1 - ML Setup (Google Colab)

1. Colab kholo: https://colab.research.google.com
2. Runtime -> Change runtime type -> **GPU (T4)**
3. Repo clone karo:

  git clone https://github.com/TUMHARA_USERNAME/geoai-analyst.git
  cd geoai-analyst/ml

4. Run karo:

  !pip install -r ml/requirements.txt
  # Phir ml/cell_6_agent.py run karo

---

## Coder 3 - Backend Setup (Local)

  git clone https://github.com/TUMHARA_USERNAME/geoai-analyst.git
  cd geoai-analyst/backend
  pip install -r backend/requirements.txt
  uvicorn main:app --reload

---

## Coder 4 - Frontend Setup (Local)

  git clone https://github.com/TUMHARA_USERNAME/geoai-analyst.git
  cd geoai-analyst/frontend
  pnpm install
  pnpm dev
  # http://localhost:5173

---

# RS-LLaVA BigEarthNet VQA

A reproducible pipeline for turning BigEarthNet patch labels into single-image VQA conversations and fine-tuning an LLaVA-compatible model with QLoRA.

## Setup

Use a CUDA-enabled environment for practical training. On Windows, install a CUDA-compatible PyTorch wheel first, then install this package. Use the same interpreter for both commands:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

`bitsandbytes` is commonly easiest on Linux/WSL. The training command fails early with a clear message if 4-bit loading is unavailable.

## 1. Build VQA records

Prepare a manifest as JSONL or CSV with `image` and `labels` fields. `labels` can be a JSON list or a semicolon/comma-separated string.

```json
{"image":"/data/BigEarthNet-S2_0001.jpg","labels":["Broad-leaved forest","Water bodies"]}
```

Then generate two records per patch:

```powershell
python -m rs_llava_ben.prepare_data --input manifest.jsonl --output data/ben_vqa.jsonl
```

The generated records contain the exact requested questions, an image path, normalized labels, and a `task` field (`multilabel` or `presence`).

## 2. Fine-tune

The default base model is `liuhaotian/llava-v1.5-7b`; pass an RS-adapted LLaVA-compatible model with `--model-name` when available.

```powershell
python -m rs_llava_ben.train `
  --data data/ben_vqa.jsonl `
  --output-dir checkpoints/rs-llava-bigearthnet `
  --model-name liuhaotian/llava-v1.5-7b `
  --epochs 3
```

QLoRA settings are fixed to rank 16 and alpha 32 by default. Images are read from the `image` field; no image bytes are embedded in the JSONL.

## 3. Evaluate OA and AA

For a prediction JSONL containing `true_labels` and `pred_labels` lists:

```powershell
python -m rs_llava_ben.evaluate --predictions predictions.jsonl --output metrics.json
```

`overall_accuracy` is exact multilabel-set accuracy. `average_accuracy` is macro per-class accuracy, computed over the union of labels. The evaluator also reports presence-question accuracy when those fields are supplied.

## 4. Inference output

```powershell
python -m rs_llava_ben.infer `
  --adapter checkpoints/rs-llava-bigearthnet `
  --image /data/example.jpg `
  --question "What land cover types are present in this satellite image?"
```

The command prints:

```json
{"answer":"...","confidence":0.87,"model":"RS-LLaVA-BigEarthNet","task":"vqa"}
```

Confidence is a reproducible heuristic based on generated token scores, not a calibrated probability.

## VQA API

The FastAPI service exposes `POST /api/v1/vqa` with JSON input:

```json
{"image_b64":"<base64 JPEG, PNG, or GeoTIFF>","question":"What land cover types are present?","top_k":1}
```

Start it with:

```powershell
python -m rs_llava_ben.api
```

The endpoint converts accepted images to RGB, rejects payloads over 10MB with `400`, applies a 30-second inference timeout, and writes every request to `audit_log.sqlite3`. `create_app(backend=..., audit_log=...)` is the injection point for a loaded RS-LLaVA backend; the backend returns `BackendResult` so the API can include the answer, confidence, model name, and optional evidence bounding boxes.
