"""Shared test fixtures."""

import wave
from pathlib import Path

import numpy as np
import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary tscribe data directory structure."""
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    whisper = tmp_path / "whisper" / "models"
    whisper.mkdir(parents=True)
    return tmp_path


def generate_silence_wav(path: Path, duration: float = 2.0, sample_rate: int = 16000):
    """Generate a WAV file with silence."""
    samples = np.zeros(int(duration * sample_rate), dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def generate_tone_wav(path: Path, freq: float = 440.0, duration: float = 2.0, sample_rate: int = 16000):
    """Generate a WAV file with a sine tone."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


# Generate fixture files if they don't exist
def pytest_configure(config):
    """Generate test fixture WAV files."""
    FIXTURES_DIR.mkdir(exist_ok=True)

    silence_path = FIXTURES_DIR / "silence.wav"
    if not silence_path.exists():
        generate_silence_wav(silence_path, duration=2.0)

    tone_path = FIXTURES_DIR / "tone.wav"
    if not tone_path.exists():
        generate_tone_wav(tone_path, duration=2.0)
