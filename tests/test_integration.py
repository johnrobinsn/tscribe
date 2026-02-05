"""Integration tests — require whisper.cpp to be installed.

Run with: uv run pytest tests/test_integration.py -v -m slow
"""

import json
import wave

import numpy as np
import pytest

from tscribe.transcriber import Transcriber
from tscribe.whisper_manager import WhisperManager


@pytest.fixture
def whisper_dir(tmp_path):
    """Create a whisper directory and check if whisper.cpp is available."""
    wm = WhisperManager(tmp_path / "whisper")
    binary = wm.find_binary()
    if binary is None:
        pytest.skip("whisper.cpp not installed — skipping integration tests")
    return tmp_path / "whisper"


@pytest.fixture
def model_path(whisper_dir):
    """Ensure the tiny model is available for fast testing."""
    wm = WhisperManager(whisper_dir)
    if not wm.is_model_available("tiny"):
        pytest.skip("tiny model not available — run 'tscribe setup --model tiny'")
    return wm.model_path("tiny")


@pytest.fixture
def speech_wav(tmp_path):
    """Create a WAV with silence (for basic smoke testing)."""
    path = tmp_path / "test_speech.wav"
    samples = np.zeros(16000 * 3, dtype=np.int16)  # 3 seconds silence
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.tobytes())
    return path


@pytest.mark.slow
def test_transcribe_silence(whisper_dir, model_path, speech_wav):
    """Smoke test: transcribing silence should succeed without crashing."""
    wm = WhisperManager(whisper_dir)
    binary = wm.find_binary()

    t = Transcriber(binary, model_path)
    result = t.transcribe(speech_wav, output_formats=["txt", "json"])

    # Should succeed, may produce empty or minimal segments
    assert result.file == speech_wav.name
    assert result.language is not None

    # Output files should exist
    assert speech_wav.with_suffix(".txt").exists()
    assert speech_wav.with_suffix(".json").exists()


@pytest.mark.slow
def test_transcribe_output_json_structure(whisper_dir, model_path, speech_wav):
    """Verify the JSON output has the expected structure."""
    wm = WhisperManager(whisper_dir)
    binary = wm.find_binary()

    t = Transcriber(binary, model_path)
    result = t.transcribe(speech_wav, output_formats=["json"])

    json_path = speech_wav.with_suffix(".json")
    data = json.loads(json_path.read_text())

    assert "file" in data
    assert "model" in data
    assert "language" in data
    assert "segments" in data
    assert isinstance(data["segments"], list)
