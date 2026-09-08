"""
ml/voice_assistant.py
======================
Voice I/O layer for GeoAI Analyst: Whisper speech-to-text + gTTS
text-to-speech, plus a full audio -> multilingual VQA -> audio pipeline.

Python 3.13 SAFE:
  - openai-whisper for STT (lazy-loaded, cached globally).
  - gTTS for TTS.
  - No use of `aifc`, `chunk`, `cgi`, or other stdlib modules removed in
    Python 3.13; audio bytes are written to a NamedTemporaryFile and
    handed to whisper/ffmpeg as a path, never decoded with stdlib audio
    modules.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
from typing import Any, Dict, Optional, TypedDict, Union

import whisper
from gtts import gTTS

from ml.multilingual import multilingual_vqa

# ---------------------------------------------------------------------------
# Whisper: lazy-loaded, cached model
# ---------------------------------------------------------------------------

_whisper_model = None
_whisper_model_size: Optional[str] = None


def load_whisper(size: str = "base"):
    """
    Load (or reuse) a Whisper model of the given size.

    Valid sizes: tiny, base, small, medium, large (and their .en variants).
    Loading is cached globally so repeated calls are cheap after the first.
    """
    global _whisper_model, _whisper_model_size

    if _whisper_model is not None and _whisper_model_size == size:
        return _whisper_model

    _whisper_model = whisper.load_model(size)
    _whisper_model_size = size
    return _whisper_model


class SpeechToTextResult(TypedDict):
    text: str
    language: str
    inference_ms: float


def speech_to_text(audio_input: Union[str, bytes], language: Optional[str] = None) -> SpeechToTextResult:
    """
    Transcribe audio to text using Whisper.

    Args:
        audio_input: path to an audio file (str), or raw audio bytes.
        language: optional ISO 639-1 code to force transcription language.
                  If None, Whisper auto-detects.

    Returns:
        {"text": str, "language": str, "inference_ms": float}
    """
    start = time.perf_counter()
    model = load_whisper("base")

    tmp_path: Optional[str] = None
    audio_path = audio_input

    if isinstance(audio_input, (bytes, bytearray)):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            tmp.write(audio_input)
        finally:
            tmp.close()
        tmp_path = tmp.name
        audio_path = tmp_path

    try:
        transcribe_kwargs: Dict[str, Any] = {}
        if language:
            transcribe_kwargs["language"] = language

        result = model.transcribe(audio_path, **transcribe_kwargs)
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    inference_ms = (time.perf_counter() - start) * 1000.0

    return {
        "text": (result.get("text") or "").strip(),
        "language": result.get("language", language or "en"),
        "inference_ms": inference_ms,
    }


# ---------------------------------------------------------------------------
# gTTS: text-to-speech
# ---------------------------------------------------------------------------

SUPPORTED_TTS_LANGUAGES = {"hi", "en", "fr", "de", "es", "ta", "te", "bn", "mr", "gu", "ur", "pa"}


def _running_in_colab() -> bool:
    return "google.colab" in sys.modules


def text_to_speech(text: str, lang: str = "en", play: bool = True) -> bytes:
    """
    Convert `text` to speech audio (MP3 bytes) using gTTS.

    - In Google Colab, if `play` is True, plays the audio inline via
      IPython.display.Audio(autoplay=True).
    - In VSCode / any non-Colab server context, playback is skipped
      regardless of `play` — only the bytes are returned, since there is
      no notebook display surface to render audio into.

    Returns:
        Raw MP3 bytes of the synthesized speech.
    """
    if not text or not text.strip():
        raise ValueError("text_to_speech: `text` must be non-empty")

    tts_lang = lang if lang in SUPPORTED_TTS_LANGUAGES else "en"

    buf = io.BytesIO()
    gTTS(text=text, lang=tts_lang).write_to_fp(buf)
    audio_bytes = buf.getvalue()

    if play and _running_in_colab():
        from IPython.display import Audio, display
        display(Audio(audio_bytes, autoplay=True))

    return audio_bytes


# ---------------------------------------------------------------------------
# Full voice pipeline: audio -> STT -> multilingual VQA -> TTS
# ---------------------------------------------------------------------------

class VoiceVQAResult(TypedDict):
    transcribed_question: str
    detected_language: str
    answer: str
    answer_english: str
    confidence: float
    stt_ms: float
    vqa_ms: float
    task: str


def voice_vqa_pipeline(
    image_input: Any,
    audio_input: Union[str, bytes],
    speak_answer: bool = True,
) -> VoiceVQAResult:
    """
    End-to-end voice pipeline:
      1. Transcribe `audio_input` with Whisper.
      2. Run the transcribed question through multilingual_vqa (which
         detects language, translates to English, runs run_vqa, and
         translates the answer back).
      3. Optionally speak the answer aloud with gTTS.

    Prints step-by-step progress to stdout.
    """
    print("[voice_vqa_pipeline] Step 1/3: transcribing audio (Whisper)...")
    stt_start = time.perf_counter()
    stt_result = speech_to_text(audio_input)
    stt_ms = (time.perf_counter() - stt_start) * 1000.0
    print(f"[voice_vqa_pipeline]   -> transcribed: {stt_result['text']!r} "
          f"(lang={stt_result['language']}, {stt_ms:.0f}ms)")

    print("[voice_vqa_pipeline] Step 2/3: running multilingual VQA...")
    vqa_start = time.perf_counter()
    vqa_result = multilingual_vqa(
        image_input,
        stt_result["text"],
        user_lang=stt_result["language"] or "auto",
    )
    vqa_ms = (time.perf_counter() - vqa_start) * 1000.0
    print(f"[voice_vqa_pipeline]   -> answer: {vqa_result['answer']!r} ({vqa_ms:.0f}ms)")

    if speak_answer:
        print("[voice_vqa_pipeline] Step 3/3: synthesizing speech (gTTS)...")
        text_to_speech(vqa_result["answer"], lang=vqa_result["language_code"], play=True)
    else:
        print("[voice_vqa_pipeline] Step 3/3: skipped (speak_answer=False)")

    return {
        "transcribed_question": stt_result["text"],
        "detected_language": vqa_result["detected_language"],
        "answer": vqa_result["answer"],
        "answer_english": vqa_result["answer_english"],
        "confidence": vqa_result["confidence"],
        "stt_ms": stt_ms,
        "vqa_ms": vqa_ms,
        "task": "voice_vqa",
    }