"""Tests for the transcriber module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tscribe.transcriber import (
    TranscriptResult,
    TranscriptSegment,
    Transcriber,
    _format_srt_time,
    _format_vtt_time,
)


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


class TestTranscriberInit:
    def test_default_cpu(self):
        t = Transcriber(model_name="base")
        assert t._model_name == "base"
        assert t._device == "cpu"
        assert t._compute_type == "int8"

    def test_gpu_defaults_to_float16(self):
        t = Transcriber(model_name="base", device="cuda")
        assert t._compute_type == "float16"

    def test_custom_compute_type(self):
        t = Transcriber(model_name="small", compute_type="float32")
        assert t._compute_type == "float32"


def _mock_whisper_model():
    """Create a mock WhisperModel with a transcribe method."""
    mock_seg = MagicMock()
    mock_seg.start = 0.0
    mock_seg.end = 2.5
    mock_seg.text = " Hello world."
    mock_seg.avg_log_prob = -0.1

    mock_info = MagicMock()
    mock_info.language = "en"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_seg], mock_info)
    return mock_model


class TestTranscriberTranscribe:
    def test_transcribe_success(self, tmp_path):
        mock_model = _mock_whisper_model()

        t = Transcriber(model_name="base")
        t._model = mock_model  # inject mock directly

        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake wav")

        result = t.transcribe(audio_path, output_formats=["txt"])

        assert len(result.segments) == 1
        assert result.segments[0].text == " Hello world."
        assert result.model == "base"
        assert result.language == "en"
        assert (tmp_path / "audio.txt").exists()
        mock_model.transcribe.assert_called_once()

    def test_transcribe_with_language(self, tmp_path):
        mock_model = _mock_whisper_model()

        t = Transcriber(model_name="base")
        t._model = mock_model

        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake wav")

        t.transcribe(audio_path, language="en")
        mock_model.transcribe.assert_called_once_with(str(audio_path), language="en")

    def test_transcribe_auto_language(self, tmp_path):
        mock_model = _mock_whisper_model()

        t = Transcriber(model_name="base")
        t._model = mock_model

        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake wav")

        t.transcribe(audio_path, language="auto")
        mock_model.transcribe.assert_called_once_with(str(audio_path), language=None)

    def test_lazy_model_loading(self):
        with patch("faster_whisper.WhisperModel") as mock_cls:
            mock_cls.return_value = _mock_whisper_model()
            t = Transcriber(model_name="tiny")
            assert t._model is None
            model = t._get_model()
            assert model is not None
            mock_cls.assert_called_once_with("tiny", device="cpu", compute_type="int8")


class TestTranscriberWriteOutputs:
    def test_write_txt_and_json(self, tmp_path):
        t = Transcriber(model_name="base")
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
        t = Transcriber(model_name="base")
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
