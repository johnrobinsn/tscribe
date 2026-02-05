"""Tests for the recorder module."""

import wave

import numpy as np
import pytest

from tscribe.recorder import MockRecorder, RecordingConfig, RecordingResult


class TestRecordingConfig:
    def test_defaults(self):
        config = RecordingConfig()
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.dtype == "int16"
        assert config.device is None
        assert config.loopback is False

    def test_custom(self):
        config = RecordingConfig(sample_rate=44100, channels=2, loopback=True)
        assert config.sample_rate == 44100
        assert config.channels == 2
        assert config.loopback is True


class TestMockRecorder:
    def test_records_silence_by_default(self, tmp_path):
        recorder = MockRecorder(duration=1.0)
        out = tmp_path / "test.wav"
        config = RecordingConfig(sample_rate=16000, channels=1)

        recorder.start(out, config)
        assert recorder.is_recording()

        result = recorder.stop()
        assert not recorder.is_recording()
        assert result.path == out
        assert result.duration_seconds == pytest.approx(1.0)
        assert result.sample_rate == 16000
        assert result.channels == 1
        assert result.device_name == "mock"
        assert result.source_type == "microphone"

        # Verify WAV file is valid
        with wave.open(str(out), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2
            assert wf.getnframes() == 16000  # 1 second

    def test_records_provided_audio_data(self, tmp_path):
        audio = np.array([100, -100, 200, -200] * 4000, dtype=np.int16)
        recorder = MockRecorder(audio_data=audio)
        out = tmp_path / "test.wav"
        config = RecordingConfig(sample_rate=16000, channels=1)

        recorder.start(out, config)
        result = recorder.stop()

        assert result.duration_seconds == pytest.approx(1.0)

        with wave.open(str(out), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            data = np.frombuffer(frames, dtype=np.int16)
            np.testing.assert_array_equal(data, audio)

    def test_loopback_source_type(self, tmp_path):
        recorder = MockRecorder(duration=0.5)
        out = tmp_path / "test.wav"
        config = RecordingConfig(loopback=True)

        recorder.start(out, config)
        result = recorder.stop()
        assert result.source_type == "loopback"

    def test_stereo(self, tmp_path):
        recorder = MockRecorder(duration=0.5)
        out = tmp_path / "test.wav"
        config = RecordingConfig(channels=2)

        recorder.start(out, config)
        result = recorder.stop()

        with wave.open(str(out), "rb") as wf:
            assert wf.getnchannels() == 2

    def test_double_start_raises(self, tmp_path):
        recorder = MockRecorder()
        config = RecordingConfig()
        recorder.start(tmp_path / "a.wav", config)

        with pytest.raises(RuntimeError, match="Already recording"):
            recorder.start(tmp_path / "b.wav", config)

        recorder.stop()

    def test_stop_without_start_raises(self):
        recorder = MockRecorder()
        with pytest.raises(RuntimeError, match="Not recording"):
            recorder.stop()

    def test_elapsed_seconds(self, tmp_path):
        recorder = MockRecorder()
        assert recorder.elapsed_seconds == 0.0

        recorder.start(tmp_path / "test.wav", RecordingConfig())
        assert recorder.elapsed_seconds >= 0.0

        recorder.stop()
