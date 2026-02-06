"""Tests for WASAPI loopback recorder."""

import sys
import threading
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tscribe.recorder import RecordingConfig, RecordingResult


def _make_mock_pyaudiowpatch():
    """Create a mock pyaudiowpatch module."""
    mock_mod = MagicMock()
    mock_mod.paInt16 = 8  # pyaudio.paInt16 constant
    mock_mod.paContinue = 0

    mock_pyaudio_instance = MagicMock()

    # Default loopback device info
    mock_loopback_device = {
        "index": 5,
        "name": "Speakers (Realtek) [Loopback]",
        "maxInputChannels": 2,
        "maxOutputChannels": 0,
        "defaultSampleRate": 48000.0,
        "isLoopbackDevice": True,
    }
    mock_pyaudio_instance.get_default_wasapi_loopback.return_value = mock_loopback_device
    mock_pyaudio_instance.get_device_info_by_index.return_value = mock_loopback_device

    # Mock stream
    mock_stream = MagicMock()
    mock_pyaudio_instance.open.return_value = mock_stream

    mock_mod.PyAudio.return_value = mock_pyaudio_instance

    return mock_mod, mock_pyaudio_instance, mock_stream, mock_loopback_device


@pytest.fixture
def mock_pyaudio(monkeypatch):
    """Inject a mock pyaudiowpatch module."""
    mock_mod, mock_instance, mock_stream, mock_device = _make_mock_pyaudiowpatch()
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", mock_mod)
    return mock_mod, mock_instance, mock_stream, mock_device


@pytest.fixture
def recorder(mock_pyaudio):
    from tscribe.recorder.wasapi_recorder import WasapiRecorder

    return WasapiRecorder()


@pytest.fixture
def config():
    return RecordingConfig(loopback=True)


# ──── start ────


def test_start_opens_stream(recorder, config, tmp_path, mock_pyaudio):
    mock_mod, mock_instance, mock_stream, mock_device = mock_pyaudio
    wav_path = tmp_path / "test.wav"

    recorder.start(wav_path, config)

    assert recorder.is_recording()
    mock_instance.get_default_wasapi_loopback.assert_called_once()
    mock_instance.open.assert_called_once()

    call_kwargs = mock_instance.open.call_args[1]
    assert call_kwargs["format"] == mock_mod.paInt16
    assert call_kwargs["channels"] == 2
    assert call_kwargs["rate"] == 48000
    assert call_kwargs["input"] is True
    assert call_kwargs["input_device_index"] == 5
    assert call_kwargs["stream_callback"] is not None


def test_start_with_explicit_device(recorder, config, tmp_path, mock_pyaudio):
    mock_mod, mock_instance, mock_stream, mock_device = mock_pyaudio
    wav_path = tmp_path / "test.wav"

    config_with_device = RecordingConfig(loopback=True, device=7)
    recorder.start(wav_path, config_with_device)

    mock_instance.get_device_info_by_index.assert_called_with(7)
    mock_instance.get_default_wasapi_loopback.assert_not_called()


def test_start_double_raises(recorder, config, tmp_path, mock_pyaudio):
    wav_path = tmp_path / "test.wav"
    recorder.start(wav_path, config)

    with pytest.raises(RuntimeError, match="Already recording"):
        recorder.start(wav_path, config)


def test_start_no_loopback_device_raises(recorder, config, tmp_path, mock_pyaudio):
    mock_mod, mock_instance, mock_stream, mock_device = mock_pyaudio
    mock_instance.get_default_wasapi_loopback.side_effect = LookupError("No device")
    wav_path = tmp_path / "test.wav"

    with pytest.raises(RuntimeError, match="No WASAPI loopback device"):
        recorder.start(wav_path, config)


# ──── stop ────


def test_stop_returns_result(recorder, config, tmp_path, mock_pyaudio):
    wav_path = tmp_path / "test.wav"
    recorder.start(wav_path, config)

    result = recorder.stop()

    assert not recorder.is_recording()
    assert isinstance(result, RecordingResult)
    assert result.path == wav_path
    assert result.source_type == "loopback"
    assert result.device_name == "WASAPI Loopback"
    assert result.sample_rate == 48000
    assert result.channels == 2


def test_stop_without_start_raises(recorder):
    with pytest.raises(RuntimeError, match="Not recording"):
        recorder.stop()


def test_stop_closes_stream(recorder, config, tmp_path, mock_pyaudio):
    mock_mod, mock_instance, mock_stream, mock_device = mock_pyaudio
    wav_path = tmp_path / "test.wav"
    recorder.start(wav_path, config)

    recorder.stop()

    mock_stream.stop_stream.assert_called_once()
    mock_stream.close.assert_called_once()
    mock_instance.terminate.assert_called_once()


# ──── callback ────


def test_callback_writes_frames_and_updates_level(recorder, config, tmp_path, mock_pyaudio):
    mock_mod, mock_instance, mock_stream, mock_device = mock_pyaudio
    wav_path = tmp_path / "test.wav"
    recorder.start(wav_path, config)

    # Get the callback function from the open() call
    call_kwargs = mock_instance.open.call_args[1]
    callback = call_kwargs["stream_callback"]

    # Create some audio data (stereo, 512 frames)
    audio = (np.sin(np.linspace(0, np.pi, 1024)) * 16000).astype(np.int16)
    in_data = audio.tobytes()

    result = callback(in_data, 512, {}, 0)

    assert result == (in_data, mock_mod.paContinue)
    assert recorder.level > 0.0
    assert recorder._frames_written == 512


# ──── elapsed ────


def test_elapsed_seconds(recorder, config, tmp_path, mock_pyaudio):
    assert recorder.elapsed_seconds == 0.0

    wav_path = tmp_path / "test.wav"
    recorder.start(wav_path, config)

    assert recorder.elapsed_seconds >= 0.0


# ──── level ────


def test_level_default_zero(recorder):
    assert recorder.level == 0.0
