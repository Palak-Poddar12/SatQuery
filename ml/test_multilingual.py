"""
ml/test_multilingual.py
========================
Test suite for ml/multilingual.py and ml/voice_assistant.py.

These tests mock ml.pipeline.run_vqa / run_agent and the network-calling
GoogleTranslator so they run offline and deterministically in CI / CPU-only
environments (no GPU, no internet required).
"""

import sys
import types
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Stub out ml.pipeline before importing ml.multilingual, since the real
# pipeline module depends on torch/transformers models we don't want to
# load in unit tests.
# ---------------------------------------------------------------------------

def _make_fake_pipeline_module():
    fake = types.ModuleType("ml.pipeline")

    def run_vqa(image, question):
        return {"answer": f"echo: {question}", "confidence": 0.87, "inference_ms": 12.3}

    def run_agent(images, question="", has_sar=False):
        return {
            "answer": f"agent-echo: {question}",
            "execution_trace": ["step1", "step2"],
            "request_id": "req-123",
        }

    fake.run_vqa = run_vqa
    fake.run_agent = run_agent
    return fake


@pytest.fixture(autouse=True)
def fake_ml_pipeline(monkeypatch):
    fake_module = _make_fake_pipeline_module()
    monkeypatch.setitem(sys.modules, "ml.pipeline", fake_module)
    yield


@pytest.fixture
def no_network_translate(monkeypatch):
    """Patch GoogleTranslator so tests never hit the network."""
    import ml.multilingual as ml_multi

    class FakeTranslator:
        def __init__(self, source, target):
            self.source = source
            self.target = target

        def translate(self, text):
            return f"[{self.target}]{text}"

    monkeypatch.setattr(ml_multi, "GoogleTranslator", FakeTranslator)
    yield


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------

def test_detect_language_empty_string_returns_en():
    from ml.multilingual import detect_language
    assert detect_language("") == "en"
    assert detect_language("   ") == "en"


def test_detect_language_english():
    from ml.multilingual import detect_language
    assert detect_language("What is visible in this satellite image?") == "en"


def test_detect_language_hindi():
    from ml.multilingual import detect_language
    assert detect_language("इस उपग्रह छवि में क्या दिखता है?") == "hi"


# ---------------------------------------------------------------------------
# translate_to_english / translate_from_english
# ---------------------------------------------------------------------------

def test_translate_to_english_already_english_skips_translation(no_network_translate):
    from ml.multilingual import translate_to_english
    result = translate_to_english("Hello there", src_lang="en")
    assert result["was_translated"] is False
    assert result["text_english"] == "Hello there"
    assert result["detected_language"] == "en"


def test_translate_to_english_non_english_translates(no_network_translate):
    from ml.multilingual import translate_to_english
    result = translate_to_english("Bonjour", src_lang="fr")
    assert result["was_translated"] is True
    assert result["detected_language"] == "fr"
    assert result["text_english"] == "[en]Bonjour"


def test_translate_from_english_to_hindi(no_network_translate):
    from ml.multilingual import translate_from_english
    out = translate_from_english("hello", "hi")
    assert out == "[hi]hello"


def test_translate_from_english_target_english_is_noop(no_network_translate):
    from ml.multilingual import translate_from_english
    out = translate_from_english("hello", "en")
    assert out == "hello"


def test_translate_from_english_empty_target_is_noop(no_network_translate):
    from ml.multilingual import translate_from_english
    out = translate_from_english("hello", "")
    assert out == "hello"


# ---------------------------------------------------------------------------
# multilingual_vqa
# ---------------------------------------------------------------------------

def test_multilingual_vqa_english_question_no_translation(no_network_translate):
    from ml.multilingual import multilingual_vqa
    result = multilingual_vqa("fake_image", "What do you see?", user_lang="en")
    assert result["was_translated"] is False
    assert result["language_code"] == "en"
    assert result["answer_english"] == "echo: What do you see?"
    assert result["answer"] == "echo: What do you see?"
    assert result["task"] == "multilingual_vqa"
    assert result["confidence"] == 0.87


def test_multilingual_vqa_hindi_question_translates_both_ways(no_network_translate):
    from ml.multilingual import multilingual_vqa
    result = multilingual_vqa("fake_image", "यह क्या है?", user_lang="hi")
    assert result["was_translated"] is True
    assert result["language_code"] == "hi"
    assert result["question_english"] == "[en]यह क्या है?"
    assert result["answer_english"] == "echo: [en]यह क्या है?"
    assert result["answer"] == "[hi]echo: [en]यह क्या है?"


def test_multilingual_vqa_result_has_all_required_keys(no_network_translate):
    from ml.multilingual import multilingual_vqa
    result = multilingual_vqa("fake_image", "Hello", user_lang="en")
    expected_keys = {
        "answer", "answer_english", "question_english", "question_original",
        "detected_language", "language_code", "was_translated",
        "confidence", "inference_ms", "task",
    }
    assert expected_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# multilingual_agent
# ---------------------------------------------------------------------------

def test_multilingual_agent_preserves_execution_trace_and_request_id(no_network_translate):
    from ml.multilingual import multilingual_agent
    result = multilingual_agent(["img1"], "What is this?", user_lang="en")
    assert result["execution_trace"] == ["step1", "step2"]
    assert result["request_id"] == "req-123"
    assert result["task"] == "multilingual_agent"


def test_multilingual_agent_empty_question_skips_translation(no_network_translate):
    from ml.multilingual import multilingual_agent
    result = multilingual_agent(["img1"], "", user_lang="auto")
    assert result["was_translated"] is False
    assert result["language_code"] == "en"


# ---------------------------------------------------------------------------
# voice_assistant: text_to_speech environment detection
# ---------------------------------------------------------------------------

def test_text_to_speech_empty_text_raises():
    from ml.voice_assistant import text_to_speech
    with pytest.raises(ValueError):
        text_to_speech("", lang="en")


def test_text_to_speech_returns_bytes(monkeypatch):
    import ml.voice_assistant as va

    class FakeGTTS:
        def __init__(self, text, lang):
            self.text = text
            self.lang = lang

        def write_to_fp(self, fp):
            fp.write(b"FAKE_MP3_BYTES")

    monkeypatch.setattr(va, "gTTS", FakeGTTS)
    monkeypatch.setattr(va, "_running_in_colab", lambda: False)

    result = va.text_to_speech("नमस्ते", lang="hi", play=True)
    assert result == b"FAKE_MP3_BYTES"


def test_text_to_speech_unsupported_lang_falls_back_to_english(monkeypatch):
    import ml.voice_assistant as va

    captured = {}

    class FakeGTTS:
        def __init__(self, text, lang):
            captured["lang"] = lang

        def write_to_fp(self, fp):
            fp.write(b"X")

    monkeypatch.setattr(va, "gTTS", FakeGTTS)
    monkeypatch.setattr(va, "_running_in_colab", lambda: False)

    va.text_to_speech("hello", lang="zz")
    assert captured["lang"] == "en"


# ---------------------------------------------------------------------------
# voice_assistant: whisper lazy loading
# ---------------------------------------------------------------------------

def test_load_whisper_caches_model(monkeypatch):
    import ml.voice_assistant as va

    va._whisper_model = None
    va._whisper_model_size = None

    call_count = {"n": 0}

    def fake_load_model(size):
        call_count["n"] += 1
        return f"model-{size}"

    monkeypatch.setattr(va.whisper, "load_model", fake_load_model)

    m1 = va.load_whisper("base")
    m2 = va.load_whisper("base")

    assert m1 == "model-base"
    assert m2 == "model-base"
    assert call_count["n"] == 1  # only loaded once


def test_speech_to_text_from_bytes(monkeypatch, tmp_path):
    import ml.voice_assistant as va

    class FakeModel:
        def transcribe(self, path, **kwargs):
            assert path.endswith(".wav")
            return {"text": "hello world", "language": "en"}

    va._whisper_model = FakeModel()
    va._whisper_model_size = "base"

    result = va.speech_to_text(b"FAKE_AUDIO_BYTES")
    assert result["text"] == "hello world"
    assert result["language"] == "en"
    assert result["inference_ms"] >= 0