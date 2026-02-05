"""Tests for the CLI entry point."""

import json
import sys
import threading
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from tscribe.cli import main
from tscribe.recorder import MockRecorder


def _make_wav(path, duration=1.0):
    samples = np.zeros(int(duration * 16000), dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.tobytes())


def test_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Record audio and transcribe" in result.output


def test_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_subcommands_registered():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    for cmd in ("record", "transcribe", "list", "devices", "config", "setup"):
        assert cmd in result.output, f"Subcommand '{cmd}' not found in help output"


# ──── record ────


def _run_record_with_mock(monkeypatch, tmp_path, extra_args=None):
    """Helper to run record with a mock recorder and simulated Ctrl+C."""
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    mock_recorder = MockRecorder(duration=1.0)

    def fake_create(cfg):
        def trigger_stop():
            import time, signal
            time.sleep(0.1)
            signal.raise_signal(signal.SIGINT)
        threading.Thread(target=trigger_stop, daemon=True).start()
        return mock_recorder

    monkeypatch.setattr("tscribe.cli._create_recorder", fake_create)
    runner = CliRunner()
    args = ["record", "--no-transcribe"] + (extra_args or [])
    return runner.invoke(main, args)


def test_record_with_mock_recorder(monkeypatch, tmp_path):
    result = _run_record_with_mock(monkeypatch, tmp_path)
    assert result.exit_code == 0
    assert "Recording saved" in result.output

    recordings_dir = tmp_path / "recordings"
    assert len(list(recordings_dir.glob("*.wav"))) == 1
    assert len(list(recordings_dir.glob("*.meta"))) == 1


def test_record_no_transcribe_flag(monkeypatch, tmp_path):
    result = _run_record_with_mock(monkeypatch, tmp_path)
    assert result.exit_code == 0
    assert "Auto-transcription" not in result.output


def test_record_custom_output(monkeypatch, tmp_path):
    out = tmp_path / "custom.wav"
    result = _run_record_with_mock(monkeypatch, tmp_path, ["-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()


def test_record_with_auto_transcribe_no_whisper(monkeypatch, tmp_path):
    """Auto-transcribe should gracefully handle missing whisper.cpp."""
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    mock_recorder = MockRecorder(duration=0.5)

    def fake_create(cfg):
        def trigger_stop():
            import time, signal
            time.sleep(0.1)
            signal.raise_signal(signal.SIGINT)
        threading.Thread(target=trigger_stop, daemon=True).start()
        return mock_recorder

    monkeypatch.setattr("tscribe.cli._create_recorder", fake_create)
    runner = CliRunner()
    result = runner.invoke(main, ["record"])
    assert result.exit_code == 0
    assert "whisper.cpp not found" in result.output or "Recording saved" in result.output


# ──── transcribe ────


def test_transcribe_help():
    runner = CliRunner()
    result = runner.invoke(main, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "FILE" in result.output


def test_transcribe_missing_file():
    runner = CliRunner()
    result = runner.invoke(main, ["transcribe"])
    assert result.exit_code != 0


def test_transcribe_with_mock(monkeypatch, tmp_path):
    """Test transcribe command with mocked whisper.cpp."""
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))

    audio_file = tmp_path / "test.wav"
    _make_wav(audio_file)

    # Create a fake whisper binary
    whisper_dir = tmp_path / "whisper"
    whisper_dir.mkdir(parents=True, exist_ok=True)
    binary = whisper_dir / "main"
    binary.write_text("#!/bin/sh\necho 'fake'")
    binary.chmod(0o755)

    models_dir = whisper_dir / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "ggml-base.bin").write_bytes(b"fake model")

    mock_stdout = "[00:00:00.000 --> 00:00:01.000]  Hello.\n"
    mock_proc = type("Proc", (), {"returncode": 0, "stdout": mock_stdout, "stderr": ""})()

    with patch("tscribe.transcriber.subprocess.run", return_value=mock_proc):
        runner = CliRunner()
        result = runner.invoke(main, ["transcribe", str(audio_file)])

    assert result.exit_code == 0
    assert "Transcription complete" in result.output


def test_transcribe_format_all(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))

    audio_file = tmp_path / "test.wav"
    _make_wav(audio_file)

    whisper_dir = tmp_path / "whisper"
    whisper_dir.mkdir(parents=True, exist_ok=True)
    (whisper_dir / "main").write_text("fake")
    (whisper_dir / "main").chmod(0o755)
    (whisper_dir / "models").mkdir(exist_ok=True)
    (whisper_dir / "models" / "ggml-base.bin").write_bytes(b"fake")

    mock_proc = type("Proc", (), {"returncode": 0, "stdout": "[00:00:00.000 --> 00:00:01.000]  Hi.\n", "stderr": ""})()

    with patch("tscribe.transcriber.subprocess.run", return_value=mock_proc):
        runner = CliRunner()
        result = runner.invoke(main, ["transcribe", str(audio_file), "--format", "all"])

    assert result.exit_code == 0


# ──── list ────


def test_list_no_recordings(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(main, ["list"])
    assert result.exit_code == 0
    assert "No recordings found" in result.output


def test_list_with_recordings(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    rec_dir = tmp_path / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    _make_wav(rec_dir / "2025-01-15-143022.wav")
    (rec_dir / "2025-01-15-143022.meta").write_text(json.dumps({"duration_seconds": 10.0}))

    runner = CliRunner()
    result = runner.invoke(main, ["list"])
    assert result.exit_code == 0
    assert "2025-01-15-143022" in result.output


def test_list_with_search(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    rec_dir = tmp_path / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    _make_wav(rec_dir / "2025-01-15-143022.wav")
    (rec_dir / "2025-01-15-143022.txt").write_text("hello meeting world")

    runner = CliRunner()
    result = runner.invoke(main, ["list", "--search", "meeting"])
    assert result.exit_code == 0
    assert "2025-01-15-143022" in result.output


def test_list_no_header(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    rec_dir = tmp_path / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    _make_wav(rec_dir / "2025-01-15-143022.wav")

    runner = CliRunner()
    result = runner.invoke(main, ["list", "--no-header"])
    assert result.exit_code == 0
    assert "Date" not in result.output


# ──── devices ────


def test_devices_with_mock(monkeypatch):
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = []
    mock_sd.query_hostapis.return_value = []
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)
    runner = CliRunner()
    result = runner.invoke(main, ["devices"])
    assert result.exit_code == 0
    assert "No audio input devices found" in result.output


def test_devices_with_results(monkeypatch):
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = [
        {"name": "Mic", "max_input_channels": 2, "max_output_channels": 0,
         "default_samplerate": 44100.0, "hostapi": 0},
    ]
    mock_sd.query_hostapis.return_value = [{"name": "ALSA"}]
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)
    runner = CliRunner()
    result = runner.invoke(main, ["devices"])
    assert result.exit_code == 0
    assert "Mic" in result.output


def test_devices_loopback_guidance(monkeypatch):
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = []
    mock_sd.query_hostapis.return_value = []
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)
    runner = CliRunner()
    result = runner.invoke(main, ["devices", "--loopback"])
    assert result.exit_code == 0


# ──── config ────


def test_config_list():
    runner = CliRunner()
    result = runner.invoke(main, ["config"])
    assert result.exit_code == 0
    assert "recording.sample_rate" in result.output


def test_config_get(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(main, ["config", "recording.sample_rate"])
    assert result.exit_code == 0
    assert "16000" in result.output


def test_config_set(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(main, ["config", "recording.sample_rate", "44100"])
    assert result.exit_code == 0
    assert "44100" in result.output


def test_config_get_invalid(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(main, ["config", "invalid"])
    assert result.exit_code != 0


def test_config_set_invalid(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(main, ["config", "transcription.model", "nonexistent"])
    assert result.exit_code != 0


# ──── setup ────


def test_setup_help():
    runner = CliRunner()
    result = runner.invoke(main, ["setup", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
