"""Tests for the transcriber module."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tscribe.transcriber import (
    TranscriptResult,
    TranscriptSegment,
    Transcriber,
    _format_srt_time,
    _format_vtt_time,
    _parse_timestamp,
)


class TestParseTimestamp:
    def test_hhmmss(self):
        assert _parse_timestamp("00:01:23.456") == pytest.approx(83.456)

    def test_hhmmss_comma(self):
        assert _parse_timestamp("00:01:23,456") == pytest.approx(83.456)

    def test_mmss(self):
        assert _parse_timestamp("01:23.456") == pytest.approx(83.456)

    def test_zero(self):
        assert _parse_timestamp("00:00:00.000") == 0.0


class TestFormatTimestamps:
    def test_srt_format(self):
        assert _format_srt_time(83.456) == "00:01:23,456"

    def test_vtt_format(self):
        assert _format_vtt_time(83.456) == "00:01:23.456"

    def test_zero(self):
        assert _format_srt_time(0.0) == "00:00:00,000"
        assert _format_vtt_time(0.0) == "00:00:00.000"


class TestTranscriptResult:
    @pytest.fixture
    def result(self):
        return TranscriptResult(
            file="test.wav",
            model="base",
            language="en",
            segments=[
                TranscriptSegment(start=0.0, end=2.5, text=" Hello world. "),
                TranscriptSegment(start=2.5, end=5.0, text=" This is a test. "),
            ],
        )

    def test_to_txt(self, result):
        txt = result.to_txt()
        assert "Hello world." in txt
        assert "This is a test." in txt
        # Should be stripped
        assert txt == "Hello world.\nThis is a test."

    def test_to_json(self, result):
        data = result.to_json()
        assert data["file"] == "test.wav"
        assert data["model"] == "base"
        assert len(data["segments"]) == 2
        assert data["segments"][0]["text"] == "Hello world."
        assert data["segments"][0]["start"] == 0.0

    def test_to_srt(self, result):
        srt = result.to_srt()
        assert "1\n00:00:00,000 --> 00:00:02,500\nHello world." in srt
        assert "2\n00:00:02,500 --> 00:00:05,000\nThis is a test." in srt

    def test_to_vtt(self, result):
        vtt = result.to_vtt()
        assert vtt.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:02.500" in vtt


class TestTranscriberBuildCommand:
    def test_basic_command(self, tmp_path):
        t = Transcriber(
            whisper_binary=tmp_path / "whisper",
            model_path=tmp_path / "ggml-base.bin",
        )
        cmd = t._build_command(tmp_path / "audio.wav", language="auto", gpu=False)
        assert str(tmp_path / "whisper") in cmd
        assert "-m" in cmd
        assert "-f" in cmd
        assert "--output-json" in cmd

    def test_language_flag(self, tmp_path):
        t = Transcriber(
            whisper_binary=tmp_path / "whisper",
            model_path=tmp_path / "ggml-base.bin",
        )
        cmd = t._build_command(tmp_path / "audio.wav", language="en", gpu=False)
        assert "-l" in cmd
        assert "en" in cmd

    def test_auto_language_no_flag(self, tmp_path):
        t = Transcriber(
            whisper_binary=tmp_path / "whisper",
            model_path=tmp_path / "ggml-base.bin",
        )
        cmd = t._build_command(tmp_path / "audio.wav", language="auto", gpu=False)
        assert "-l" not in cmd


class TestTranscriberParseStdout:
    def test_parse_timestamped_lines(self, tmp_path):
        t = Transcriber(
            whisper_binary=tmp_path / "whisper",
            model_path=tmp_path / "ggml-base.bin",
        )
        stdout = (
            "[00:00:00.000 --> 00:00:02.500]  Hello world.\n"
            "[00:00:02.500 --> 00:00:05.000]  This is a test.\n"
        )
        result = t._parse_stdout(stdout, tmp_path / "audio.wav")
        assert len(result.segments) == 2
        assert result.segments[0].text == "Hello world."
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 2.5

    def test_parse_empty_output(self, tmp_path):
        t = Transcriber(
            whisper_binary=tmp_path / "whisper",
            model_path=tmp_path / "ggml-base.bin",
        )
        result = t._parse_stdout("", tmp_path / "audio.wav")
        assert result.segments == []


class TestTranscriberWriteOutputs:
    def test_write_txt_and_json(self, tmp_path):
        t = Transcriber(
            whisper_binary=tmp_path / "whisper",
            model_path=tmp_path / "ggml-base.bin",
        )
        result = TranscriptResult(
            file="test.wav",
            model="base",
            language="en",
            segments=[TranscriptSegment(start=0.0, end=2.0, text="Hello")],
        )
        base = tmp_path / "test"
        t._write_outputs(result, base, ["txt", "json"])

        assert (tmp_path / "test.txt").exists()
        assert (tmp_path / "test.json").exists()

        txt = (tmp_path / "test.txt").read_text()
        assert "Hello" in txt

        data = json.loads((tmp_path / "test.json").read_text())
        assert data["segments"][0]["text"] == "Hello"

    def test_write_srt_and_vtt(self, tmp_path):
        t = Transcriber(
            whisper_binary=tmp_path / "whisper",
            model_path=tmp_path / "ggml-base.bin",
        )
        result = TranscriptResult(
            file="test.wav",
            model="base",
            language="en",
            segments=[TranscriptSegment(start=0.0, end=2.0, text="Hello")],
        )
        base = tmp_path / "test"
        t._write_outputs(result, base, ["srt", "vtt"])

        assert (tmp_path / "test.srt").exists()
        assert (tmp_path / "test.vtt").exists()


class TestTranscriberTranscribe:
    def test_transcribe_success(self, tmp_path):
        t = Transcriber(
            whisper_binary=tmp_path / "whisper",
            model_path=tmp_path / "ggml-base.bin",
        )
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake wav")

        mock_stdout = "[00:00:00.000 --> 00:00:02.500]  Hello world.\n"
        mock_proc = type("Proc", (), {
            "returncode": 0,
            "stdout": mock_stdout,
            "stderr": "",
        })()

        with patch("tscribe.transcriber.subprocess.run", return_value=mock_proc):
            result = t.transcribe(audio_path, output_formats=["txt"])

        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello world."
        assert (tmp_path / "audio.txt").exists()

    def test_transcribe_failure(self, tmp_path):
        t = Transcriber(
            whisper_binary=tmp_path / "whisper",
            model_path=tmp_path / "ggml-base.bin",
        )
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake wav")

        mock_proc = type("Proc", (), {
            "returncode": 1,
            "stdout": "",
            "stderr": "error: invalid file",
        })()

        with patch("tscribe.transcriber.subprocess.run", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="whisper.cpp failed"):
                t.transcribe(audio_path)
