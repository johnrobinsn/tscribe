"""Transcription via faster-whisper."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    confidence: float = 0.0


@dataclass
class TranscriptResult:
    file: str
    model: str
    language: str
    segments: list[TranscriptSegment] = field(default_factory=list)

    def to_txt(self) -> str:
        """Plain text, one segment per line."""
        return "\n".join(seg.text.strip() for seg in self.segments if seg.text.strip())

    def to_json(self) -> dict:
        return {
            "file": self.file,
            "model": self.model,
            "language": self.language,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "confidence": seg.confidence,
                }
                for seg in self.segments
            ],
        }

    def to_srt(self) -> str:
        """SRT subtitle format."""
        lines = []
        for i, seg in enumerate(self.segments, 1):
            start_ts = _format_srt_time(seg.start)
            end_ts = _format_srt_time(seg.end)
            lines.append(f"{i}")
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(seg.text.strip())
            lines.append("")
        return "\n".join(lines)

    def to_vtt(self) -> str:
        """WebVTT subtitle format."""
        lines = ["WEBVTT", ""]
        for seg in self.segments:
            start_ts = _format_vtt_time(seg.start)
            end_ts = _format_vtt_time(seg.end)
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(seg.text.strip())
            lines.append("")
        return "\n".join(lines)


class Transcriber:
    """Transcribe audio files using faster-whisper (CTranslate2 backend)."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str | None = None,
    ):
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        self._model = None

    def _get_model(self):
        """Lazy-load the WhisperModel (downloads on first use)."""
        if self._model is None:
            import os

            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        language: str = "auto",
        output_formats: list[str] | None = None,
        progress_callback: Callable[[float, float], None] | None = None,
    ) -> TranscriptResult:
        """Transcribe an audio file and write output files.

        If *progress_callback* is provided it is called after each segment
        with ``(segment_end_seconds, audio_duration_seconds)``.
        """
        if output_formats is None:
            output_formats = ["txt", "json"]

        model = self._get_model()
        lang = None if language == "auto" else language
        segments_iter, info = model.transcribe(str(audio_path), language=lang)

        duration = getattr(info, "duration", 0.0) or 0.0

        segments = []
        for seg in segments_iter:
            segments.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    confidence=seg.avg_logprob,
                )
            )
            if progress_callback and duration > 0:
                progress_callback(seg.end, duration)

        if progress_callback and duration > 0:
            progress_callback(duration, duration)

        result = TranscriptResult(
            file=audio_path.name,
            model=self._model_name,
            language=info.language,
            segments=segments,
        )

        base_path = audio_path.with_suffix("")
        self._write_outputs(result, base_path, output_formats)

        return result

    def _write_outputs(
        self, result: TranscriptResult, base_path: Path, formats: list[str]
    ) -> None:
        """Write output files in the requested formats."""
        for fmt in formats:
            if fmt == "txt":
                base_path.with_suffix(".txt").write_text(result.to_txt())
            elif fmt == "json":
                base_path.with_suffix(".json").write_text(
                    json.dumps(result.to_json(), indent=2)
                )
            elif fmt == "srt":
                base_path.with_suffix(".srt").write_text(result.to_srt())
            elif fmt == "vtt":
                base_path.with_suffix(".vtt").write_text(result.to_vtt())


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def _format_vtt_time(seconds: float) -> str:
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
