"""
ml/multilingual.py
===================
Multilingual support layer for GeoAI Analyst.

Adds language detection + translation around the existing ML pipeline
(ml.pipeline.run_vqa / ml.pipeline.run_agent) without touching that
pipeline's code.

Python 3.13 SAFE:
  - Uses `langdetect` for language identification (pure Python, no cgi).
  - Uses `deep-translator` for translation. `googletrans` is NEVER used
    because it depends on an old httpx/httpcore stack and (transitively,
    in some forks) the `cgi` module, which was removed in Python 3.13.

NOTE ON ASSUMED UPSTREAM INTERFACE:
  This file assumes `ml/pipeline.py` exposes:
      run_vqa(image, question: str) -> dict
      run_agent(images: list, question: str = "", has_sar: bool = False) -> dict
  with `run_vqa` returning at least an "answer" key (and optionally
  "confidence" / "inference_ms"), and `run_agent` returning at least an
  "execution_trace" and "request_id". If your actual signatures differ,
  adjust the two call sites marked "# >>> ADAPT HERE" below — everything
  else (detection, translation, output shape) is independent of that.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TypedDict

from langdetect import detect as _langdetect_detect
from langdetect import DetectorFactory, LangDetectException
from deep_translator import GoogleTranslator

# Make langdetect deterministic across calls/runs.
DetectorFactory.seed = 0

# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: Dict[str, str] = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "en": "English",
}


# ---------------------------------------------------------------------------
# Output typing
# ---------------------------------------------------------------------------

class MultilingualVQAResult(TypedDict):
    answer: str
    answer_english: str
    question_english: str
    question_original: str
    detected_language: str
    language_code: str
    was_translated: bool
    confidence: float
    inference_ms: float
    task: str


class MultilingualAgentResult(TypedDict, total=False):
    # Mirrors run_agent's own result shape, plus translation metadata.
    execution_trace: Any
    request_id: str
    detected_language: str
    language_code: str
    question_original: str
    question_english: str
    was_translated: bool
    task: str


# ---------------------------------------------------------------------------
# 1. Language detection
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """
    Detect the ISO 639-1 language code of `text`.

    Returns "en" as a safe fallback if detection fails (e.g. empty string,
    text too short/ambiguous for langdetect to classify).
    """
    if not text or not text.strip():
        return "en"
    try:
        code = _langdetect_detect(text)
    except LangDetectException:
        return "en"

    # langdetect can return regional variants (e.g. "zh-cn"); normalize to
    # the base ISO 639-1 code used throughout this module.
    code = code.split("-")[0].lower()
    return code


# ---------------------------------------------------------------------------
# 2. Translation
# ---------------------------------------------------------------------------

def translate_to_english(text: str, src_lang: str = "auto") -> dict:
    """
    Translate `text` into English.

    Returns:
        {
          "text_english": str,
          "detected_language": str,   # ISO 639-1 code actually used as source
          "was_translated": bool,     # False if source was already English
        }
    """
    if not text or not text.strip():
        return {"text_english": text, "detected_language": "en", "was_translated": False}

    detected = detect_language(text) if src_lang == "auto" else src_lang

    if detected == "en":
        return {"text_english": text, "detected_language": "en", "was_translated": False}

    try:
        translated = GoogleTranslator(source=detected, target="en").translate(text)
    except Exception:
        # Fall back to explicit "auto" source if the detected code isn't
        # one deep-translator recognizes.
        translated = GoogleTranslator(source="auto", target="en").translate(text)

    return {
        "text_english": translated if translated else text,
        "detected_language": detected,
        "was_translated": True,
    }


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate English `text` into `target_lang` (ISO 639-1 code).

    If target_lang is "en" or unset, returns `text` unchanged.
    """
    if not text or not text.strip():
        return text
    if not target_lang or target_lang == "en":
        return text

    try:
        translated = GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception:
        # If the target code isn't supported by the translator backend,
        # fail soft and return the English text rather than raising.
        return text

    return translated if translated else text


# ---------------------------------------------------------------------------
# 3. Multilingual VQA
# ---------------------------------------------------------------------------

def multilingual_vqa(image_input: Any, question: str, user_lang: str = "auto") -> MultilingualVQAResult:
    """
    Full pipeline: detect language -> translate question to English ->
    run_vqa (existing ML pipeline) -> translate answer back to user's
    language.

    Args:
        image_input: whatever ml.pipeline.run_vqa expects (PIL.Image, path, etc.)
        question: user's question, in any supported language (or English).
        user_lang: "auto" to detect from `question`, or an explicit ISO
                   639-1 code (e.g. "hi") to force both detection and the
                   language the answer is translated back into.

    Returns:
        MultilingualVQAResult
    """
    start = time.perf_counter()

    # >>> ADAPT HERE if ml.pipeline.run_vqa has a different import path.
    from ml.pipeline import run_vqa

    question_original = question

    translation = translate_to_english(question, src_lang=user_lang)
    question_english = translation["text_english"]
    detected_language = translation["detected_language"]
    was_translated = translation["was_translated"]

    # If the caller pinned a target language explicitly, honor that for the
    # *answer* language even if detection would have said something else
    # (useful when the question was itself in English but the user still
    # wants the answer spoken/shown in e.g. Hindi).
    answer_lang = user_lang if user_lang != "auto" else detected_language

    # >>> ADAPT HERE if run_vqa's return shape differs.
    vqa_result = run_vqa(image_input, question_english)

    if isinstance(vqa_result, dict):
        answer_english = str(vqa_result.get("answer", ""))
        confidence = float(vqa_result.get("confidence", 0.0) or 0.0)
    else:
        # Defensive fallback if run_vqa returns a bare string.
        answer_english = str(vqa_result)
        confidence = 0.0

    answer = translate_from_english(answer_english, answer_lang)

    inference_ms = (time.perf_counter() - start) * 1000.0

    return {
        "answer": answer,
        "answer_english": answer_english,
        "question_english": question_english,
        "question_original": question_original,
        "detected_language": SUPPORTED_LANGUAGES.get(detected_language, detected_language),
        "language_code": detected_language,
        "was_translated": was_translated,
        "confidence": confidence,
        "inference_ms": inference_ms,
        "task": "multilingual_vqa",
    }


# ---------------------------------------------------------------------------
# 4. Multilingual Agent
# ---------------------------------------------------------------------------

def multilingual_agent(
    images: List[Any],
    question: str = "",
    has_sar: bool = False,
    user_lang: str = "auto",
) -> MultilingualAgentResult:
    """
    Wraps ml.pipeline.run_agent with the same detect -> translate ->
    run -> translate-back layer used by multilingual_vqa, while preserving
    run_agent's own execution_trace / request_id untouched.
    """
    # >>> ADAPT HERE if ml.pipeline.run_agent has a different import path.
    from ml.pipeline import run_agent

    question_original = question
    detected_language = "en"
    question_english = question
    was_translated = False

    if question and question.strip():
        translation = translate_to_english(question, src_lang=user_lang)
        question_english = translation["text_english"]
        detected_language = translation["detected_language"]
        was_translated = translation["was_translated"]

    answer_lang = user_lang if user_lang != "auto" else detected_language

    # >>> ADAPT HERE if run_agent's signature differs.
    agent_result = run_agent(images, question_english, has_sar=has_sar)

    result: MultilingualAgentResult = dict(agent_result) if isinstance(agent_result, dict) else {}

    # Translate any top-level "answer" / "summary"-style text fields back
    # into the user's language, without disturbing execution_trace/request_id.
    for text_key in ("answer", "summary", "narrative"):
        if text_key in result and isinstance(result[text_key], str) and result[text_key]:
            english_key = f"{text_key}_english"
            result[english_key] = result[text_key]
            result[text_key] = translate_from_english(result[text_key], answer_lang)

    result.update({
        "detected_language": SUPPORTED_LANGUAGES.get(detected_language, detected_language),
        "language_code": detected_language,
        "question_original": question_original,
        "question_english": question_english,
        "was_translated": was_translated,
        "task": "multilingual_agent",
    })

    return result