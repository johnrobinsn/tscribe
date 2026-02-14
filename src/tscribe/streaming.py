"""Streaming (overlapped) transcription — transcribe speech chunks in background."""

from __future__ import annotations

import queue
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np

from tscribe.transcriber import TranscriptResult, TranscriptSegment


class StreamingTranscriber:
    """Background transcription worker that processes speech chunks.

    Usage::

        st = StreamingTranscriber(transcriber)
        st.start()
        # ... during recording, VAD submits chunks ...
        st.submit_chunk(audio_int16, 16000, offset_secs=3.5)
        # ... recording ends ...
        result = st.finish()  # blocks until all chunks processed

    Parameters
    ----------
    transcriber : tscribe.transcriber.Transcriber
        A (shared) Transcriber instance — model is loaded lazily on first chunk.
    language : str
        Language code for transcription (default ``"auto"``).
    """

    def __init__(self, transcriber, language: str = "auto"):
        self._transcriber = transcriber
        self._language = language
        self._results: list[tuple[float, TranscriptResult]] = []
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._error: Exception | None = None
        self._segments_done = 0
        self._lock = threading.Lock()

    # -- public API --

    def start(self) -> None:
        """Start the background worker thread."""
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def submit_chunk(
        self, audio_int16: np.ndarray, sample_rate: int, offset_secs: float
    ) -> None:
        """Enqueue a speech chunk for transcription."""
        self._queue.put((audio_int16, sample_rate, offset_secs))

    @property
    def segments_done(self) -> int:
        with self._lock:
            return self._segments_done

    def finish(self, timeout: float = 300) -> TranscriptResult:
        """Send sentinel, wait for worker, return merged result.

        Raises the first error encountered by the worker, if any.
        """
        self._queue.put(None)  # sentinel
        if self._worker is not None:
            self._worker.join(timeout=timeout)
        if self._error is not None:
            raise self._error
        return self._merge_results()

    # -- internal --

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            audio, sr, offset = item
            try:
                self._transcribe_chunk(audio, sr, offset)
            except Exception as exc:
                self._error = exc

    def _transcribe_chunk(
        self, audio: np.ndarray, sample_rate: int, offset: float
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            with wave.open(str(tmp_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio.tobytes())

            result = self._transcriber.transcribe(
                tmp_path,
                language=self._language,
                output_formats=[],  # don't write files for chunks
            )
            self._results.append((offset, result))
            with self._lock:
                self._segments_done += len(result.segments)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _merge_results(self) -> TranscriptResult:
        if not self._results:
            return TranscriptResult(file="", model="", language="", segments=[])

        self._results.sort(key=lambda x: x[0])

        all_segments: list[TranscriptSegment] = []
        for offset, result in self._results:
            for seg in result.segments:
                all_segments.append(
                    TranscriptSegment(
                        start=seg.start + offset,
                        end=seg.end + offset,
                        text=seg.text,
                        confidence=seg.confidence,
                    )
                )

        _, first = self._results[0]
        return TranscriptResult(
            file=first.file,
            model=first.model,
            language=first.language,
            segments=all_segments,
        )
