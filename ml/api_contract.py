"""Typed API contracts for multilingual and voice analysis endpoints."""

from __future__ import annotations

from typing import Any, TypedDict


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


class VoiceVQAResult(TypedDict):
    transcribed_question: str
    detected_language: str
    answer: str
    answer_english: str
    confidence: float
    stt_ms: float
    vqa_ms: float
    task: str


class MultilingualAgentResult(TypedDict, total=False):
    execution_trace: Any
    request_id: str
    detected_language: str
    language_code: str
    question_original: str
    question_english: str
    was_translated: bool
    task: str


# Backend endpoint owners can use these wrappers after validating uploads.
def analyze_multilingual(image: Any, question: str = "", language: str = "auto") -> MultilingualVQAResult:
    from ml.multilingual import multilingual_vqa

    return multilingual_vqa(image, question, user_lang=language)


def analyze_voice(image: Any, audio: Any) -> VoiceVQAResult:
    from ml.voice_assistant import voice_vqa_pipeline

    return voice_vqa_pipeline(image, audio)
