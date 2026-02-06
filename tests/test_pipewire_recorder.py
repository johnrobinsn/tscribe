"""Tests for PipeWire recorder."""

import signal
import subprocess
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tscribe.recorder import RecordingConfig, RecordingResult
from tscribe.recorder.pipewire_recorder import PipewireRecorder


def _make_wav(path, duration=1.0, sample_rate=16000, channels=1):
    """Create a valid WAV file for testing."""
    samples = np.zeros(int(duration * sample_rate * channels), dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def _make_wav_with_audio(path, sample_rate=16000, channels=1):
    """Create a WAV file with non-silent audio for level testing."""
    t = np.linspace(0, 0.5, int(0.5 * sample_rate), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


@pytest.fixture
def recorder():
    return PipewireRecorder()


@pytest.fixture
def config():
    return RecordingConfig(sample_rate=16000, channels=1)


@pytest.fixture
def mock_popen(tmp_path):
    """Create a mock Popen that simulates pw-record writing a WAV file."""

    def _create_mock(wav_path, duration=1.0, sample_rate=16000, channels=1):
        _make_wav(wav_path, duration, sample_rate, channels)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Running
        mock_proc.wait.return_value = 0
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = b""
        return mock_proc

    return _create_mock


# ──── start ────


def test_start_launches_pw_record(recorder, config, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen(wav_path)

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc) as popen_mock, \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)
        recorder._stop_event.set()  # Stop level thread

    popen_mock.assert_called_once()
    cmd = popen_mock.call_args[0][0]
    assert cmd[0] == "pw-record"
    assert "--format" in cmd
    assert "s16" in cmd
    assert "--rate" in cmd
    assert "16000" in cmd
    assert "--channels" in cmd
    assert "1" in cmd
    assert str(wav_path) in cmd
    assert "--target" not in cmd
    assert "-P" not in cmd  # No capture.sink for mic recording


def test_start_with_target(recorder, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen(wav_path)
    config = RecordingConfig(sample_rate=16000, channels=1, loopback=True)

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc) as popen_mock, \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value="alsa_output.usb-device"), \
         patch("tscribe.pipewire_devices.get_node_audio_info", return_value={"channels": 2}):
        recorder.start(wav_path, config)
        recorder._stop_event.set()

    cmd = popen_mock.call_args[0][0]
    assert "--target" in cmd
    idx = cmd.index("--target")
    assert cmd[idx + 1] == "alsa_output.usb-device"
    # Loopback should use 48kHz and 2 channels (from sink info)
    assert "48000" in cmd
    assert "2" in cmd
    # Loopback must set stream.capture.sink property
    assert "-P" in cmd
    p_idx = cmd.index("-P")
    assert "stream.capture.sink=true" in cmd[p_idx + 1]


def test_start_with_device(recorder, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen(wav_path)
    config = RecordingConfig(sample_rate=48000, channels=2, device="my_mic")

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc) as popen_mock, \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value="my_mic"):
        recorder.start(wav_path, config)
        recorder._stop_event.set()

    cmd = popen_mock.call_args[0][0]
    assert "--target" in cmd
    assert "my_mic" in cmd
    assert "48000" in cmd
    assert "2" in cmd
    assert "-P" not in cmd  # No capture.sink for device recording


def test_start_double_raises(recorder, config, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen(wav_path)

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)

    with pytest.raises(RuntimeError, match="Already recording"):
        recorder.start(wav_path, config)

    recorder._stop_event.set()


# ──── stop ────


def test_stop_sends_sigint(recorder, config, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen(wav_path)

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)

    result = recorder.stop()
    mock_proc.send_signal.assert_called_with(signal.SIGINT)
    assert isinstance(result, RecordingResult)
    assert result.path == wav_path
    assert result.source_type == "microphone"


def test_stop_loopback_source_type(recorder, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    config = RecordingConfig(sample_rate=16000, channels=1, loopback=True)
    mock_proc = mock_popen(wav_path)

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value="sink_name"), \
         patch("tscribe.pipewire_devices.get_node_audio_info", return_value={"channels": 2}):
        recorder.start(wav_path, config)

    result = recorder.stop()
    assert result.source_type == "loopback"


def test_stop_reads_wav_duration(recorder, config, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen(wav_path, duration=2.5)

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)

    result = recorder.stop()
    assert abs(result.duration_seconds - 2.5) < 0.01


def test_stop_escalates_to_terminate(recorder, config, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen(wav_path)
    mock_proc.wait.side_effect = [subprocess.TimeoutExpired("pw-record", 3), 0, 0]

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)

    recorder.stop()
    mock_proc.send_signal.assert_called_with(signal.SIGINT)
    mock_proc.terminate.assert_called_once()


def test_stop_without_start_raises(recorder):
    with pytest.raises(RuntimeError, match="Not recording"):
        recorder.stop()


# ──── is_recording / elapsed ────


def test_is_recording(recorder, config, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen(wav_path)

    assert recorder.is_recording() is False

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)

    assert recorder.is_recording() is True
    recorder.stop()
    assert recorder.is_recording() is False


def test_elapsed_seconds(recorder, config, tmp_path, mock_popen):
    assert recorder.elapsed_seconds == 0.0

    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen(wav_path)

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)

    assert recorder.elapsed_seconds > 0.0
    recorder._stop_event.set()


# ──── level ────


def test_level_default_zero(recorder):
    assert recorder.level == 0.0


def test_level_reads_from_wav(recorder, config, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    # Create WAV with actual audio data
    _make_wav_with_audio(wav_path)

    # Directly create a mock process (the wav is already created by _make_wav_with_audio)
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.wait.return_value = 0
    mock_proc.stderr = MagicMock()

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)

    # Wait for at least one level poll cycle
    import time
    time.sleep(0.3)

    assert recorder.level > 0.0
    recorder.stop()


def test_recording_result_metadata(recorder, config, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen(wav_path)

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)

    result = recorder.stop()
    assert result.sample_rate == 16000
    assert result.channels == 1
    assert result.device_name == "default"
