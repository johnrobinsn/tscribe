"""Tests for the streaming (overlapped) transcriber."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tscribe.streaming import StreamingTranscriber
from tscribe.transcriber import TranscriptResult, TranscriptSegment


# ── helpers ──


def _mock_transcriber(segments_per_call=None):
    """Create a mock Transcriber that returns canned results.

    *segments_per_call* is a list of lists of (start, end, text) tuples,
    one per call to transcribe().
    """
    if segments_per_call is None:
        segments_per_call = [[(0.0, 2.0, "hello world")]]

    t = MagicMock()
    call_idx = [0]

    def fake_transcribe(audio_path, language="auto", output_formats=None, progress_callback=None):
        idx = min(call_idx[0], len(segments_per_call) - 1)
        segs = segments_per_call[idx]
        call_idx[0] += 1
        return TranscriptResult(
            file=audio_path.name,
            model="tiny",
            language="en",
            segments=[
                TranscriptSegment(start=s, end=e, text=txt)
                for s, e, txt in segs
            ],
        )

    t.transcribe.side_effect = fake_transcribe
    return t


# ── tests ──


class TestStreamingTranscriber:
    def test_single_chunk(self):
        """One chunk → result with segments offset-adjusted."""
        mock_t = _mock_transcriber([[(0.0, 1.5, "hello")]])
        st = StreamingTranscriber(mock_t)
        st.start()

        audio = np.zeros(16000, dtype=np.int16)  # 1s of silence
        st.submit_chunk(audio, 16000, offset_secs=5.0)

        result = st.finish()
        assert len(result.segments) == 1
        assert result.segments[0].start == pytest.approx(5.0)
        assert result.segments[0].end == pytest.approx(6.5)
        assert result.segments[0].text == "hello"
        assert result.model == "tiny"
        assert result.language == "en"

    def test_multiple_chunks_merged_in_order(self):
        """Multiple chunks → segments merged and ordered by offset."""
        mock_t = _mock_transcriber([
            [(0.0, 1.0, "first")],
            [(0.0, 2.0, "second")],
        ])
        st = StreamingTranscriber(mock_t)
        st.start()

        audio = np.zeros(16000, dtype=np.int16)
        st.submit_chunk(audio, 16000, offset_secs=10.0)
        st.submit_chunk(audio, 16000, offset_secs=3.0)

        result = st.finish()
        assert len(result.segments) == 2
        # Segments should be sorted by offset
        assert result.segments[0].start == pytest.approx(3.0)
        assert result.segments[0].text == "second"
        assert result.segments[1].start == pytest.approx(10.0)
        assert result.segments[1].text == "first"

    def test_empty_finish(self):
        """No chunks submitted → empty result."""
        mock_t = _mock_transcriber()
        st = StreamingTranscriber(mock_t)
        st.start()

        result = st.finish()
        assert result.segments == []
        assert result.file == ""

    def test_segments_done_counter(self):
        """segments_done tracks total segments across processed chunks."""
        mock_t = _mock_transcriber([
            [(0.0, 1.0, "a")],
            [(0.0, 0.5, "b"), (0.5, 1.0, "c")],
        ])
        st = StreamingTranscriber(mock_t)
        st.start()

        audio = np.zeros(16000, dtype=np.int16)
        st.submit_chunk(audio, 16000, offset_secs=0.0)
        st.submit_chunk(audio, 16000, offset_secs=1.0)

        result = st.finish()
        assert st.segments_done == 3

    def test_worker_error_propagated(self):
        """If transcription fails, finish() raises the error."""
        mock_t = MagicMock()
        mock_t.transcribe.side_effect = RuntimeError("model failed")

        st = StreamingTranscriber(mock_t)
        st.start()

        audio = np.zeros(16000, dtype=np.int16)
        st.submit_chunk(audio, 16000, offset_secs=0.0)

        with pytest.raises(RuntimeError, match="model failed"):
            st.finish()

    def test_language_passed_through(self):
        """Language parameter is forwarded to the transcriber."""
        mock_t = _mock_transcriber([[(0.0, 1.0, "hola")]])
        st = StreamingTranscriber(mock_t, language="es")
        st.start()

        audio = np.zeros(16000, dtype=np.int16)
        st.submit_chunk(audio, 16000, offset_secs=0.0)
        st.finish()

        _, kwargs = mock_t.transcribe.call_args
        assert kwargs["language"] == "es"

    def test_output_formats_empty(self):
        """Streaming chunks pass output_formats=[] to avoid file writes."""
        mock_t = _mock_transcriber([[(0.0, 1.0, "test")]])
        st = StreamingTranscriber(mock_t)
        st.start()

        audio = np.zeros(16000, dtype=np.int16)
        st.submit_chunk(audio, 16000, offset_secs=0.0)
        st.finish()

        _, kwargs = mock_t.transcribe.call_args
        assert kwargs["output_formats"] == []

    def test_temp_wav_cleaned_up(self, tmp_path):
        """Temp WAV files are deleted after transcription."""
        import os

        created_files = []

        mock_t = MagicMock()

        def track_transcribe(audio_path, **kwargs):
            created_files.append(str(audio_path))
            return TranscriptResult(
                file=audio_path.name, model="tiny", language="en",
                segments=[TranscriptSegment(start=0, end=1, text="x")],
            )

        mock_t.transcribe.side_effect = track_transcribe

        st = StreamingTranscriber(mock_t)
        st.start()
        st.submit_chunk(np.zeros(16000, dtype=np.int16), 16000, 0.0)
        st.finish()

        # The temp file should have been cleaned up
        for f in created_files:
            assert not os.path.exists(f)
