"""Mock recorder for testing without audio hardware."""

from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np

from tscribe.recorder.base import Recorder, RecordingConfig, RecordingResult


class MockRecorder(Recorder):
    """A recorder that writes pre-loaded audio data for deterministic testing."""

    def __init__(self, audio_data: np.ndarray | None = None, duration: float = 2.0):
        """Initialize with audio data or generate silence.

        Args:
            audio_data: Pre-loaded PCM data (int16). If None, generates silence.
            duration: Duration in seconds if generating silence.
        """
        self._audio_data = audio_data
        self._default_duration = duration
        self._output_path: Path | None = None
        self._config: RecordingConfig | None = None
        self._start_time: float | None = None
        self._recording = False
        self._duration_written: float = 0.0

    def start(self, output_path: Path, config: RecordingConfig) -> None:
        if self._recording:
            raise RuntimeError("Already recording")

        self._config = config
        self._output_path = output_path
        self._recording = True
        self._start_time = time.monotonic()

        # Generate or use provided audio data
        if self._audio_data is not None:
            audio = self._audio_data
        else:
            num_samples = int(self._default_duration * config.sample_rate)
            audio = np.zeros(num_samples * config.channels, dtype=np.int16)

        # Write to WAV immediately
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(config.channels)
            wf.setsampwidth(2)
            wf.setframerate(config.sample_rate)
            wf.writeframes(audio.tobytes())

        self._duration_written = len(audio) / (config.sample_rate * config.channels)

    def stop(self) -> RecordingResult:
        if not self._recording:
            raise RuntimeError("Not recording")

        self._recording = False

        return RecordingResult(
            path=self._output_path,
            duration_seconds=self._duration_written,
            sample_rate=self._config.sample_rate,
            channels=self._config.channels,
            device_name="mock",
            source_type="loopback" if self._config.loopback else "microphone",
        )

    def is_recording(self) -> bool:
        return self._recording

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def level(self) -> float:
        return 0.0
