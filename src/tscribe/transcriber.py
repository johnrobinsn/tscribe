"""Transcription via whisper.cpp subprocess."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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
    def __init__(self, whisper_binary: Path, model_path: Path):
        self.whisper_binary = whisper_binary
        self.model_path = model_path

    def transcribe(
        self,
        audio_path: Path,
        language: str = "auto",
        output_formats: list[str] | None = None,
        gpu: bool = False,
    ) -> TranscriptResult:
        """Run whisper.cpp on the audio file and return parsed results."""
        if output_formats is None:
            output_formats = ["txt", "json"]

        cmd = self._build_command(audio_path, language, gpu)

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"whisper.cpp failed (exit code {proc.returncode}):\n{proc.stderr}"
            )

        result = self._parse_output(proc.stdout, audio_path)

        # Write output files alongside the audio
        base_path = audio_path.with_suffix("")
        self._write_outputs(result, base_path, output_formats)

        return result

    def _build_command(self, audio_path: Path, language: str, gpu: bool) -> list[str]:
        """Construct the whisper.cpp CLI arguments."""
        cmd = [str(self.whisper_binary)]
        cmd.extend(["-m", str(self.model_path)])
        cmd.extend(["-f", str(audio_path)])
        cmd.extend(["--output-json"])
        cmd.extend(["--print-progress", "false"])
        if language != "auto":
            cmd.extend(["-l", language])
        return cmd

    def _parse_output(self, stdout: str, audio_path: Path) -> TranscriptResult:
        """Parse whisper.cpp output into structured segments.

        whisper.cpp with --output-json writes a .json file next to the input.
        It also outputs timestamped text to stdout.
        """
        # Try to read the JSON file that whisper.cpp produces
        json_output_path = audio_path.with_suffix(".wav.json")
        if json_output_path.exists():
            return self._parse_json_file(json_output_path, audio_path)

        # Fallback: parse stdout timestamps like [00:00:00.000 --> 00:00:04.520]  text
        return self._parse_stdout(stdout, audio_path)

    def _parse_json_file(self, json_path: Path, audio_path: Path) -> TranscriptResult:
        """Parse whisper.cpp's JSON output file."""
        with open(json_path) as f:
            data = json.load(f)

        segments = []
        for item in data.get("transcription", []):
            timestamps = item.get("timestamps", {})
            start = _parse_timestamp(timestamps.get("from", "00:00:00.000"))
            end = _parse_timestamp(timestamps.get("to", "00:00:00.000"))
            text = item.get("text", "")

            segments.append(TranscriptSegment(
                start=start,
                end=end,
                text=text,
            ))

        # Clean up the whisper-generated json file since we write our own
        json_path.unlink(missing_ok=True)

        return TranscriptResult(
            file=audio_path.name,
            model=self.model_path.stem.replace("ggml-", ""),
            language=data.get("result", {}).get("language", "unknown"),
            segments=segments,
        )

    def _parse_stdout(self, stdout: str, audio_path: Path) -> TranscriptResult:
        """Parse whisper.cpp stdout as timestamped text."""
        segments = []
        pattern = re.compile(
            r"\[(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.*)"
        )
        for line in stdout.splitlines():
            m = pattern.match(line.strip())
            if m:
                start = _parse_timestamp(m.group(1))
                end = _parse_timestamp(m.group(2))
                text = m.group(3)
                segments.append(TranscriptSegment(start=start, end=end, text=text))

        return TranscriptResult(
            file=audio_path.name,
            model=self.model_path.stem.replace("ggml-", ""),
            language="unknown",
            segments=segments,
        )

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


def _parse_timestamp(ts: str) -> float:
    """Parse a timestamp like '00:01:23.456' or '00:01:23,456' to seconds."""
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(ts)


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
