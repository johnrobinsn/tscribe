"""Tests for the VAD detector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tscribe.vad import VadDetector, _ensure_model


# ── helpers ──


def _make_mock_session(version=4):
    """Create a mock ONNX InferenceSession for Silero VAD."""
    session = MagicMock()

    if version == 5:
        inp_input = MagicMock()
        inp_input.name = "input"
        inp_state = MagicMock()
        inp_state.name = "state"
        inp_sr = MagicMock()
        inp_sr.name = "sr"
        session.get_inputs.return_value = [inp_input, inp_state, inp_sr]
    else:
        inp_input = MagicMock()
        inp_input.name = "input"
        inp_h = MagicMock()
        inp_h.name = "h"
        inp_c = MagicMock()
        inp_c.name = "c"
        inp_sr = MagicMock()
        inp_sr.name = "sr"
        session.get_inputs.return_value = [inp_input, inp_h, inp_c, inp_sr]

    return session


def _make_vad_with_mock(probs, version=4, **kwargs):
    """Create a VadDetector with a mock ONNX model returning *probs* in order.

    Returns (vad, mock_session).
    """
    vad = VadDetector(**kwargs)

    session = _make_mock_session(version)
    prob_iter = iter(probs)

    def mock_run(output_names, inputs):
        p = next(prob_iter, 0.0)
        if version == 5:
            return [np.array([[p]], dtype=np.float32), inputs["state"]]
        else:
            return [np.array([p], dtype=np.float32), inputs["h"], inputs["c"]]

    session.run.side_effect = mock_run

    # Inject the mock session (bypass _load_model)
    vad._session = session
    if version == 5:
        vad._model_version = 5
        vad._state = np.zeros((2, 1, 128), dtype=np.float32)
    else:
        vad._model_version = 4
        vad._h = np.zeros((2, 1, 64), dtype=np.float32)
        vad._c = np.zeros((2, 1, 64), dtype=np.float32)

    return vad, session


# ── tests ──


class TestVadDetector:
    def test_no_speech(self):
        """All silence → no speech regions, no callback."""
        n_windows = 20
        probs = [0.1] * n_windows
        vad, _ = _make_vad_with_mock(probs)

        callback = MagicMock()
        vad.set_chunk_callback(callback)

        # Feed 20 windows of silence (each 512 samples at 16 kHz)
        silence = np.zeros(512 * n_windows, dtype=np.int16)
        vad.process_frames(silence, sample_rate=16000, channels=1)
        vad.flush()

        callback.assert_not_called()
        assert vad.speech_regions == []

    def test_speech_then_silence_emits_chunk(self):
        """Speech followed by long silence → one chunk emitted."""
        # 10 speech windows, then 40 silence windows (enough for 1s silence)
        speech_windows = 10
        silence_windows = 40  # 40 * 512 / 16000 ≈ 1.28 s > 1 s threshold
        probs = [0.9] * speech_windows + [0.1] * silence_windows
        vad, _ = _make_vad_with_mock(probs, min_silence_ms=1000)

        chunks = []

        def on_chunk(audio, sr, offset):
            chunks.append((audio.copy(), sr, offset))

        vad.set_chunk_callback(on_chunk)

        audio = np.zeros(512 * (speech_windows + silence_windows), dtype=np.int16)
        vad.process_frames(audio, sample_rate=16000, channels=1)

        assert len(chunks) == 1
        assert chunks[0][1] == 16000  # sample rate
        assert chunks[0][2] == 0.0  # offset (speech started at beginning)
        assert len(vad.speech_regions) == 1

    def test_flush_emits_remaining_speech(self):
        """Speech without trailing silence → flush emits the chunk."""
        probs = [0.9] * 20
        vad, _ = _make_vad_with_mock(probs)

        chunks = []
        vad.set_chunk_callback(lambda a, sr, o: chunks.append((a, sr, o)))

        audio = np.zeros(512 * 20, dtype=np.int16)
        vad.process_frames(audio, sample_rate=16000, channels=1)

        assert len(chunks) == 0  # no silence yet → no emission
        vad.flush()
        assert len(chunks) == 1

    def test_multiple_speech_segments(self):
        """Two speech regions separated by silence → two chunks."""
        speech1 = [0.9] * 10
        gap = [0.1] * 40  # ~1.28 s silence
        speech2 = [0.9] * 10
        trailing = [0.1] * 40
        probs = speech1 + gap + speech2 + trailing
        vad, _ = _make_vad_with_mock(probs, min_silence_ms=1000)

        chunks = []
        vad.set_chunk_callback(lambda a, sr, o: chunks.append((a, sr, o)))

        audio = np.zeros(512 * len(probs), dtype=np.int16)
        vad.process_frames(audio, sample_rate=16000, channels=1)

        assert len(chunks) == 2
        assert chunks[0][2] < chunks[1][2]  # second has later offset

    def test_max_speech_forces_boundary(self):
        """Continuous speech exceeding max_speech_s forces a chunk boundary."""
        # At 16 kHz / 512 samples per window = 31.25 windows/sec
        # 2 seconds of speech ≈ 63 windows
        n_windows = 70
        probs = [0.9] * n_windows
        vad, _ = _make_vad_with_mock(probs, max_speech_s=2.0, overlap_s=0.5)

        chunks = []
        vad.set_chunk_callback(lambda a, sr, o: chunks.append((a, sr, o)))

        audio = np.zeros(512 * n_windows, dtype=np.int16)
        vad.process_frames(audio, sample_rate=16000, channels=1)
        vad.flush()

        # Should have at least 2 chunks (one forced at ~2s, one flushed)
        assert len(chunks) >= 2

    def test_resample_48k_to_16k(self):
        """48 kHz audio is resampled to 16 kHz internally."""
        # 1 window at 16kHz = 512 samples.  At 48kHz that's 1536 samples.
        n_windows = 10
        probs = [0.9] * n_windows
        vad, session = _make_vad_with_mock(probs)

        chunks = []
        vad.set_chunk_callback(lambda a, sr, o: chunks.append((a, sr, o)))

        audio_48k = np.zeros(1536 * n_windows, dtype=np.int16)
        vad.process_frames(audio_48k, sample_rate=48000, channels=1)
        vad.flush()

        # The model should have been called ~10 times
        assert session.run.call_count >= n_windows - 1
        assert len(chunks) == 1

    def test_stereo_downmix(self):
        """Stereo audio is downmixed to mono."""
        n_windows = 10
        probs = [0.9] * n_windows
        vad, session = _make_vad_with_mock(probs)

        chunks = []
        vad.set_chunk_callback(lambda a, sr, o: chunks.append((a, sr, o)))

        # Interleaved stereo: L R L R ... at 16 kHz
        # 512 mono samples = 1024 interleaved samples
        audio_stereo = np.zeros(1024 * n_windows, dtype=np.int16)
        vad.process_frames(audio_stereo, sample_rate=16000, channels=2)
        vad.flush()

        assert session.run.call_count >= n_windows - 1
        assert len(chunks) == 1

    def test_v5_model_detection(self):
        """VadDetector auto-detects v5 model (state-based)."""
        probs = [0.9] * 5 + [0.1] * 40
        vad, _ = _make_vad_with_mock(probs, version=5, min_silence_ms=1000)

        chunks = []
        vad.set_chunk_callback(lambda a, sr, o: chunks.append(o))

        audio = np.zeros(512 * len(probs), dtype=np.int16)
        vad.process_frames(audio, sample_rate=16000, channels=1)

        assert len(chunks) == 1
        assert vad._model_version == 5

    def test_reset_clears_state(self):
        """reset() clears all accumulated state."""
        probs = [0.9] * 10
        vad, _ = _make_vad_with_mock(probs)

        audio = np.zeros(512 * 10, dtype=np.int16)
        vad.process_frames(audio, sample_rate=16000, channels=1)
        vad.flush()
        assert len(vad.speech_regions) == 1

        vad.reset()
        assert vad.speech_regions == []
        assert vad._session is None

    def test_threshold_property(self):
        vad = VadDetector(threshold=0.6)
        assert vad.threshold == 0.6

    def test_short_silence_does_not_emit(self):
        """Brief silence during speech does not trigger chunk emission."""
        # Speech, short silence (< 1s), more speech
        probs = [0.9] * 10 + [0.1] * 10 + [0.9] * 10  # 10*32ms = 320ms silence
        vad, _ = _make_vad_with_mock(probs, min_silence_ms=1000)

        chunks = []
        vad.set_chunk_callback(lambda a, sr, o: chunks.append(o))

        audio = np.zeros(512 * len(probs), dtype=np.int16)
        vad.process_frames(audio, sample_rate=16000, channels=1)

        # No chunk yet — silence was too short
        assert len(chunks) == 0
        vad.flush()
        # Now the single combined chunk is emitted
        assert len(chunks) == 1


class TestEnsureModel:
    def test_custom_path_returned_directly(self, tmp_path):
        model = tmp_path / "custom.onnx"
        model.write_bytes(b"fake")
        assert _ensure_model(model) == model

    def test_downloads_when_missing(self, tmp_path):
        from pathlib import Path as _Path

        with patch("tscribe.vad._get_models_dir", return_value=tmp_path), \
             patch("tscribe.vad.urllib.request.urlretrieve") as mock_dl:
            mock_dl.side_effect = lambda url, p: _Path(p).write_bytes(b"model-data")
            path = _ensure_model()
            mock_dl.assert_called_once()
            assert path == tmp_path / "silero_vad.onnx"

    def test_uses_cache(self, tmp_path):
        cached = tmp_path / "silero_vad.onnx"
        cached.write_bytes(b"cached")
        with patch("tscribe.vad._get_models_dir", return_value=tmp_path):
            path = _ensure_model()
            assert path == cached
