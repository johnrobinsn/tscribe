"""Tests for PipeWire recorder (raw PCM pipe mode)."""

import io
import signal
import subprocess
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tscribe.recorder import RecordingConfig, RecordingResult
from tscribe.recorder.pipewire_recorder import PipewireRecorder


def _make_raw_pcm(duration=1.0, sample_rate=16000, channels=1):
    """Create raw PCM bytes (s16le) for simulated pw-record stdout."""
    n_samples = int(duration * sample_rate * channels)
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def _make_pcm_with_audio(duration=0.5, sample_rate=16000, channels=1):
    """Create raw PCM with a 440 Hz sine tone."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    if channels > 1:
        tone = np.column_stack([tone] * channels).ravel()
    return tone.tobytes()


@pytest.fixture
def recorder():
    return PipewireRecorder()


@pytest.fixture
def config():
    return RecordingConfig(sample_rate=16000, channels=1)


@pytest.fixture
def mock_popen():
    """Create a mock Popen whose stdout yields raw PCM then EOF."""

    def _create(pcm_data=None, duration=1.0, sample_rate=16000, channels=1):
        if pcm_data is None:
            pcm_data = _make_raw_pcm(duration, sample_rate, channels)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = 0
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = b""
        # stdout is a BytesIO that the pipe reader will read from
        mock_proc.stdout = io.BytesIO(pcm_data)
        return mock_proc

    return _create


# ──── start ────


def test_start_launches_pw_record(recorder, config, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen()

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc) as popen_mock, \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)
        recorder._stop_event.set()
        if recorder._reader_thread:
            recorder._reader_thread.join(timeout=2)

    popen_mock.assert_called_once()
    cmd = popen_mock.call_args[0][0]
    assert cmd[0] == "pw-record"
    assert "--format" in cmd
    assert "s16" in cmd
    assert "--rate" in cmd
    assert "16000" in cmd
    assert "--channels" in cmd
    assert "1" in cmd
    assert cmd[-1] == "-"  # raw PCM to stdout
    assert "--target" not in cmd
    assert "-P" not in cmd


def test_start_with_target(recorder, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen(sample_rate=48000, channels=2)
    config = RecordingConfig(sample_rate=16000, channels=1, loopback=True)

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc) as popen_mock, \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value="alsa_output.usb-device"), \
         patch("tscribe.pipewire_devices.get_node_audio_info", return_value={"channels": 2}):
        recorder.start(wav_path, config)
        recorder._stop_event.set()
        if recorder._reader_thread:
            recorder._reader_thread.join(timeout=2)

    cmd = popen_mock.call_args[0][0]
    assert "--target" in cmd
    idx = cmd.index("--target")
    assert cmd[idx + 1] == "alsa_output.usb-device"
    assert "48000" in cmd
    assert "2" in cmd
    assert "-P" in cmd
    p_idx = cmd.index("-P")
    assert "stream.capture.sink=true" in cmd[p_idx + 1]


def test_start_with_device(recorder, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen(sample_rate=48000, channels=2)
    config = RecordingConfig(sample_rate=48000, channels=2, device="my_mic")

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc) as popen_mock, \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value="my_mic"):
        recorder.start(wav_path, config)
        recorder._stop_event.set()
        if recorder._reader_thread:
            recorder._reader_thread.join(timeout=2)

    cmd = popen_mock.call_args[0][0]
    assert "--target" in cmd
    assert "my_mic" in cmd
    assert "48000" in cmd
    assert "2" in cmd
    assert "-P" not in cmd


def test_start_double_raises(recorder, config, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen()

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)

    with pytest.raises(RuntimeError, match="Already recording"):
        recorder.start(wav_path, config)

    recorder._stop_event.set()
    if recorder._reader_thread:
        recorder._reader_thread.join(timeout=2)


# ──── stop ────


def test_stop_sends_sigint(recorder, config, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen(duration=1.0)

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)

    # Let reader thread consume the data
    import time
    time.sleep(0.2)

    result = recorder.stop()
    mock_proc.send_signal.assert_called_with(signal.SIGINT)
    assert isinstance(result, RecordingResult)
    assert result.path == wav_path
    assert result.source_type == "microphone"


def test_stop_writes_valid_wav(recorder, config, tmp_path, mock_popen):
    """Verify that stop produces a valid WAV file written by Python."""
    wav_path = tmp_path / "test.wav"
    pcm = _make_raw_pcm(duration=0.5, sample_rate=16000, channels=1)
    mock_proc = mock_popen(pcm_data=pcm)

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)

    import time
    time.sleep(0.2)

    result = recorder.stop()

    # Verify the WAV file
    assert wav_path.exists()
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() > 0


def test_stop_loopback_source_type(recorder, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    config = RecordingConfig(sample_rate=16000, channels=1, loopback=True)
    mock_proc = mock_popen(sample_rate=48000, channels=2)

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value="sink_name"), \
         patch("tscribe.pipewire_devices.get_node_audio_info", return_value={"channels": 2}):
        recorder.start(wav_path, config)

    import time
    time.sleep(0.2)

    result = recorder.stop()
    assert result.source_type == "loopback"


def test_stop_duration_from_frames(recorder, config, tmp_path, mock_popen):
    """Duration is computed from frames written, not WAV file reading."""
    wav_path = tmp_path / "test.wav"
    pcm = _make_raw_pcm(duration=2.0, sample_rate=16000, channels=1)
    mock_proc = mock_popen(pcm_data=pcm)

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)

    import time
    time.sleep(0.3)

    result = recorder.stop()
    assert abs(result.duration_seconds - 2.0) < 0.1


def test_stop_escalates_to_terminate(recorder, config, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen()
    mock_proc.wait.side_effect = [subprocess.TimeoutExpired("pw-record", 3), 0, 0]

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)

    import time
    time.sleep(0.1)

    recorder.stop()
    mock_proc.send_signal.assert_called_with(signal.SIGINT)
    mock_proc.terminate.assert_called_once()


def test_stop_without_start_raises(recorder):
    with pytest.raises(RuntimeError, match="Not recording"):
        recorder.stop()


# ──── is_recording / elapsed ────


def test_is_recording(recorder, config, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen()

    assert recorder.is_recording() is False

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)

    assert recorder.is_recording() is True

    import time
    time.sleep(0.1)

    recorder.stop()
    assert recorder.is_recording() is False


def test_elapsed_seconds(recorder, config, tmp_path, mock_popen):
    assert recorder.elapsed_seconds == 0.0

    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen()

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)

    assert recorder.elapsed_seconds > 0.0
    recorder._stop_event.set()
    if recorder._reader_thread:
        recorder._reader_thread.join(timeout=2)


# ──── level ────


def test_level_default_zero(recorder):
    assert recorder.level == 0.0


def test_level_computed_from_pcm(recorder, config, tmp_path, mock_popen):
    """Level is computed from raw PCM data in the pipe reader."""
    wav_path = tmp_path / "test.wav"
    pcm = _make_pcm_with_audio(duration=0.5, sample_rate=16000, channels=1)
    mock_proc = mock_popen(pcm_data=pcm)

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)

    import time
    time.sleep(0.3)

    assert recorder.level > 0.0
    recorder.stop()


# ──── frame callback ────


def test_frame_callback_invoked(recorder, config, tmp_path, mock_popen):
    """Frame callback receives audio with sample_rate and channels."""
    wav_path = tmp_path / "test.wav"
    pcm = _make_raw_pcm(duration=0.5, sample_rate=16000, channels=1)
    mock_proc = mock_popen(pcm_data=pcm)

    received = []

    def on_frames(frames, sr, ch):
        received.append((len(frames), sr, ch))

    recorder.set_frame_callback(on_frames)

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)

    import time
    time.sleep(0.3)

    recorder.stop()

    assert len(received) > 0
    # Check that sample_rate and channels were passed
    _, sr, ch = received[0]
    assert sr == 16000
    assert ch == 1


# ──── metadata ────


def test_recording_result_metadata(recorder, config, tmp_path, mock_popen):
    wav_path = tmp_path / "test.wav"
    mock_proc = mock_popen()

    with patch("tscribe.recorder.pipewire_recorder.subprocess.Popen", return_value=mock_proc), \
         patch("tscribe.pipewire_devices.resolve_pipewire_target", return_value=None):
        recorder.start(wav_path, config)

    import time
    time.sleep(0.1)

    result = recorder.stop()
    assert result.sample_rate == 16000
    assert result.channels == 1
    assert result.device_name == "default"
