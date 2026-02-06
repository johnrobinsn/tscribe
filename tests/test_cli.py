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
    for cmd in ("record", "transcribe", "list", "devices", "config"):
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
    args = ["record", "--mic", "--no-transcribe"] + (extra_args or [])
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


def test_record_with_auto_transcribe_failure(monkeypatch, tmp_path):
    """Auto-transcribe should gracefully handle transcription failures."""
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

    with patch("faster_whisper.WhisperModel", side_effect=RuntimeError("Model not available")):
        runner = CliRunner()
        result = runner.invoke(main, ["record", "--mic"])

    assert result.exit_code == 0
    assert "Transcription failed" in result.output or "Recording saved" in result.output


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


def _mock_whisper_for_cli():
    """Create mock WhisperModel class for CLI transcribe tests."""
    mock_seg = MagicMock()
    mock_seg.start = 0.0
    mock_seg.end = 1.0
    mock_seg.text = " Hello."
    mock_seg.avg_logprob = -0.1

    mock_info = MagicMock()
    mock_info.language = "en"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_seg], mock_info)
    return mock_model


def test_transcribe_with_mock(monkeypatch, tmp_path):
    """Test transcribe command with mocked faster-whisper."""
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))

    audio_file = tmp_path / "test.wav"
    _make_wav(audio_file)

    mock_model = _mock_whisper_for_cli()

    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        runner = CliRunner()
        result = runner.invoke(main, ["transcribe", str(audio_file)])

    assert result.exit_code == 0
    assert "Transcription complete" in result.output


def test_transcribe_format_all(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))

    audio_file = tmp_path / "test.wav"
    _make_wav(audio_file)

    mock_model = _mock_whisper_for_cli()

    with patch("faster_whisper.WhisperModel", return_value=mock_model):
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


# ──── play ────


def _setup_recordings(tmp_path, count=2):
    """Create test recordings with sequential timestamps."""
    import time
    rec_dir = tmp_path / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    stems = []
    for i in range(count):
        stem = f"2025-01-1{5+i}-143022"
        _make_wav(rec_dir / f"{stem}.wav")
        (rec_dir / f"{stem}.meta").write_text(
            json.dumps({"duration_seconds": 10.0 + i})
        )
        stems.append(stem)
    return stems


def test_play_help():
    runner = CliRunner()
    result = runner.invoke(main, ["play", "--help"])
    assert result.exit_code == 0
    assert "HEAD" in result.output


def test_play_head(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    stems = _setup_recordings(tmp_path)
    monkeypatch.setattr("tscribe.cli._find_player", lambda: ["echo"])
    runner = CliRunner()
    result = runner.invoke(main, ["play"])
    assert result.exit_code == 0
    assert "Playing:" in result.output
    # HEAD should be the most recent (2025-01-16)
    assert stems[-1] in result.output


def test_play_head_offset(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    stems = _setup_recordings(tmp_path, count=3)
    monkeypatch.setattr("tscribe.cli._find_player", lambda: ["echo"])
    runner = CliRunner()
    result = runner.invoke(main, ["play", "HEAD~1"])
    assert result.exit_code == 0
    # HEAD~1 = second most recent (sorted descending by date)
    assert stems[-2] in result.output


def test_play_by_stem(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    stems = _setup_recordings(tmp_path)
    monkeypatch.setattr("tscribe.cli._find_player", lambda: ["echo"])
    runner = CliRunner()
    result = runner.invoke(main, ["play", stems[0]])
    assert result.exit_code == 0
    assert stems[0] in result.output


def test_play_invalid_ref(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    _setup_recordings(tmp_path)
    monkeypatch.setattr("tscribe.cli._find_player", lambda: ["echo"])
    runner = CliRunner()
    result = runner.invoke(main, ["play", "HEAD~abc"])
    assert result.exit_code != 0
    assert "Invalid ref" in result.output


def test_play_no_recordings(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("tscribe.cli._find_player", lambda: ["echo"])
    runner = CliRunner()
    result = runner.invoke(main, ["play"])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_play_out_of_range(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    _setup_recordings(tmp_path, count=2)
    monkeypatch.setattr("tscribe.cli._find_player", lambda: ["echo"])
    runner = CliRunner()
    result = runner.invoke(main, ["play", "HEAD~10"])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_play_no_player(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    _setup_recordings(tmp_path)
    monkeypatch.setattr("tscribe.cli._find_player", lambda: [])
    runner = CliRunner()
    result = runner.invoke(main, ["play"])
    assert result.exit_code != 0
    assert "No audio player found" in result.output


# ──── open ────


def test_open_help():
    runner = CliRunner()
    result = runner.invoke(main, ["open", "--help"])
    assert result.exit_code == 0
    assert "Open a transcription file" in result.output


def test_open_head_txt(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    stems = _setup_recordings(tmp_path)
    # Create a .txt transcript for the most recent
    rec_dir = tmp_path / "recordings"
    (rec_dir / f"{stems[-1]}.txt").write_text("hello world")
    monkeypatch.setattr("tscribe.cli._open_file", lambda p: None)
    runner = CliRunner()
    result = runner.invoke(main, ["open"])
    assert result.exit_code == 0
    assert "Opening:" in result.output
    assert f"{stems[-1]}.txt" in result.output


def test_open_with_format(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    stems = _setup_recordings(tmp_path)
    rec_dir = tmp_path / "recordings"
    (rec_dir / f"{stems[-1]}.txt").write_text("hello")
    (rec_dir / f"{stems[-1]}.json").write_text("{}")
    monkeypatch.setattr("tscribe.cli._open_file", lambda p: None)
    runner = CliRunner()
    result = runner.invoke(main, ["open", "--format", "json"])
    assert result.exit_code == 0
    assert f"{stems[-1]}.json" in result.output


def test_open_no_transcript(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    _setup_recordings(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["open"])
    assert result.exit_code != 0
    assert "No transcript found" in result.output


def test_open_missing_format(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    stems = _setup_recordings(tmp_path)
    rec_dir = tmp_path / "recordings"
    (rec_dir / f"{stems[-1]}.txt").write_text("hello")
    runner = CliRunner()
    result = runner.invoke(main, ["open", "--format", "srt"])
    assert result.exit_code != 0
    assert "File not found" in result.output


def test_open_wav(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    stems = _setup_recordings(tmp_path)
    monkeypatch.setattr("tscribe.cli._open_file", lambda p: None)
    runner = CliRunner()
    result = runner.invoke(main, ["open", "--format", "wav"])
    assert result.exit_code == 0
    assert f"{stems[-1]}.wav" in result.output


# ──── dump ────


def test_dump_help():
    runner = CliRunner()
    result = runner.invoke(main, ["dump", "--help"])
    assert result.exit_code == 0
    assert "Print a transcription" in result.output


def test_dump_head_txt(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    stems = _setup_recordings(tmp_path)
    rec_dir = tmp_path / "recordings"
    (rec_dir / f"{stems[-1]}.txt").write_text("hello world")
    runner = CliRunner()
    result = runner.invoke(main, ["dump"])
    assert result.exit_code == 0
    assert "hello world" in result.output


def test_dump_with_format(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    stems = _setup_recordings(tmp_path)
    rec_dir = tmp_path / "recordings"
    (rec_dir / f"{stems[-1]}.txt").write_text("text version")
    (rec_dir / f"{stems[-1]}.json").write_text('{"segments": []}')
    runner = CliRunner()
    result = runner.invoke(main, ["dump", "--format", "json"])
    assert result.exit_code == 0
    assert '"segments"' in result.output


def test_dump_no_transcript(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    _setup_recordings(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["dump"])
    assert result.exit_code != 0
    assert "No transcript found" in result.output


def test_dump_head_offset(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    stems = _setup_recordings(tmp_path, count=3)
    rec_dir = tmp_path / "recordings"
    (rec_dir / f"{stems[-1]}.txt").write_text("latest")
    (rec_dir / f"{stems[-2]}.txt").write_text("previous")
    runner = CliRunner()
    result = runner.invoke(main, ["dump", "HEAD~1"])
    assert result.exit_code == 0
    assert "previous" in result.output


# ──── search ────


def test_search_help():
    runner = CliRunner()
    result = runner.invoke(main, ["search", "--help"])
    assert result.exit_code == 0
    assert "Search transcript text" in result.output


def test_search_matches(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    stems = _setup_recordings(tmp_path)
    rec_dir = tmp_path / "recordings"
    (rec_dir / f"{stems[-1]}.txt").write_text("discussed the budget today")
    runner = CliRunner()
    result = runner.invoke(main, ["search", "budget"])
    assert result.exit_code == 0
    assert stems[-1] in result.output
    assert "discussed the budget today" in result.output
    assert "1 match found." in result.output


def test_search_no_matches(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    _setup_recordings(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["search", "nonexistent"])
    assert result.exit_code == 0
    assert "No matches found." in result.output


def test_search_multiple_sessions(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    stems = _setup_recordings(tmp_path, count=3)
    rec_dir = tmp_path / "recordings"
    (rec_dir / f"{stems[0]}.txt").write_text("meeting notes here")
    (rec_dir / f"{stems[2]}.txt").write_text("another meeting recap")
    runner = CliRunner()
    result = runner.invoke(main, ["search", "meeting"])
    assert result.exit_code == 0
    assert stems[0] in result.output
    assert stems[2] in result.output
    assert "2 matches found." in result.output


def test_search_case_insensitive(monkeypatch, tmp_path):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path))
    stems = _setup_recordings(tmp_path)
    rec_dir = tmp_path / "recordings"
    (rec_dir / f"{stems[-1]}.txt").write_text("Important ACTION ITEMS below")
    runner = CliRunner()
    result = runner.invoke(main, ["search", "action items"])
    assert result.exit_code == 0
    assert stems[-1] in result.output


# ──── devices ────


def test_devices_with_mock(monkeypatch):
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = []
    mock_sd.query_hostapis.return_value = []
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)
    monkeypatch.setattr("tscribe.cli.sys.platform", "darwin")  # Skip PipeWire path
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
    monkeypatch.setattr("tscribe.cli.sys.platform", "darwin")  # Skip PipeWire path
    runner = CliRunner()
    result = runner.invoke(main, ["devices"])
    assert result.exit_code == 0
    assert "Mic" in result.output


def test_devices_loopback_guidance(monkeypatch):
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = []
    mock_sd.query_hostapis.return_value = []
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)
    monkeypatch.setattr("tscribe.cli.sys.platform", "darwin")  # Skip PipeWire path
    runner = CliRunner()
    result = runner.invoke(main, ["devices", "--loopback"])
    assert result.exit_code == 0


def test_devices_pipewire(monkeypatch):
    """On Linux with PipeWire, devices command shows PipeWire nodes."""
    from tscribe.pipewire_devices import PipewireNode

    nodes = [
        PipewireNode(serial=57, name="alsa_output.usb", description="USB Headset",
                     nick="USB Headset", media_class="Audio/Sink", is_monitor=True),
        PipewireNode(serial=60, name="alsa_input.builtin", description="Built-in Mic",
                     nick="Built-in", media_class="Audio/Source", is_monitor=False),
    ]
    monkeypatch.setattr("tscribe.cli.sys.platform", "linux")
    monkeypatch.setattr("tscribe.pipewire_devices.is_pipewire_available", lambda: True)
    monkeypatch.setattr("tscribe.pipewire_devices.list_pipewire_nodes", lambda loopback_only=False: nodes)
    runner = CliRunner()
    result = runner.invoke(main, ["devices"])
    assert result.exit_code == 0
    assert "USB Headset" in result.output
    assert "Built-in" in result.output
    assert "Monitor" in result.output
    assert "Input" in result.output


def test_devices_pipewire_loopback_only(monkeypatch):
    from tscribe.pipewire_devices import PipewireNode

    nodes = [
        PipewireNode(serial=57, name="alsa_output.usb", description="USB Headset",
                     nick="USB Headset", media_class="Audio/Sink", is_monitor=True),
    ]
    monkeypatch.setattr("tscribe.cli.sys.platform", "linux")
    monkeypatch.setattr("tscribe.pipewire_devices.is_pipewire_available", lambda: True)
    monkeypatch.setattr("tscribe.pipewire_devices.list_pipewire_nodes", lambda loopback_only=False: nodes)
    runner = CliRunner()
    result = runner.invoke(main, ["devices", "--loopback"])
    assert result.exit_code == 0
    assert "USB Headset" in result.output


# ──── recorder factory ────


def test_create_recorder_pipewire_on_linux(monkeypatch):
    """On Linux with PipeWire, factory returns PipewireRecorder."""
    from tscribe.cli import _create_recorder

    monkeypatch.setattr("tscribe.cli.sys.platform", "linux")
    monkeypatch.setattr("tscribe.pipewire_devices.is_pipewire_available", lambda: True)
    recorder = _create_recorder(None)
    from tscribe.recorder.pipewire_recorder import PipewireRecorder
    assert isinstance(recorder, PipewireRecorder)


def test_create_recorder_sounddevice_no_pipewire(monkeypatch):
    """When PipeWire is unavailable, factory returns SounddeviceRecorder."""
    from tscribe.cli import _create_recorder
    from tscribe.recorder.sounddevice_recorder import SounddeviceRecorder

    monkeypatch.setattr("tscribe.cli.sys.platform", "linux")
    monkeypatch.setattr("tscribe.pipewire_devices.is_pipewire_available", lambda: False)
    # Mock sounddevice import since it may fail on this system
    mock_sd = MagicMock()
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)
    recorder = _create_recorder(None)
    assert isinstance(recorder, SounddeviceRecorder)


def test_create_recorder_sounddevice_on_non_linux(monkeypatch):
    """On macOS/Windows, factory always returns SounddeviceRecorder."""
    from tscribe.cli import _create_recorder
    from tscribe.recorder.sounddevice_recorder import SounddeviceRecorder

    monkeypatch.setattr("tscribe.cli.sys.platform", "darwin")
    mock_sd = MagicMock()
    monkeypatch.setitem(sys.modules, "sounddevice", mock_sd)
    recorder = _create_recorder(None)
    assert isinstance(recorder, SounddeviceRecorder)


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
