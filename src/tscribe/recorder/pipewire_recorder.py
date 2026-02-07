"""PipeWire-native audio recorder using pw-record subprocess."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np

from tscribe.recorder.base import Recorder, RecordingConfig, RecordingResult


class PipewireRecorder(Recorder):
    """Records audio via pw-record subprocess.

    Supports both microphone input and system audio (loopback) capture
    by targeting PipeWire sinks with --target.
    """

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._config: RecordingConfig | None = None
        self._output_path: Path | None = None
        self._start_time: float | None = None
        self._recording = False
        self._level: float = 0.0
        self._lock = threading.Lock()
        self._level_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._actual_sample_rate: int = 16000
        self._actual_channels: int = 1

    def start(self, output_path: Path, config: RecordingConfig) -> None:
        if self._recording:
            raise RuntimeError("Already recording")

        from tscribe.pipewire_devices import (
            get_node_audio_info,
            resolve_pipewire_target,
        )

        self._config = config
        self._output_path = output_path

        target = resolve_pipewire_target(
            device=config.device,
            loopback=config.loopback,
        )

        # For loopback, use PipeWire default rate (48kHz) and match
        # the sink's channel count to avoid volume loss from resampling
        sample_rate = config.sample_rate
        channels = config.channels
        if config.loopback:
            sample_rate = 48000
            if target:
                info = get_node_audio_info(target)
                if info:
                    channels = info["channels"]

        self._actual_sample_rate = sample_rate
        self._actual_channels = channels

        cmd = [
            "pw-record",
            "--format", "s16",
            "--rate", str(sample_rate),
            "--channels", str(channels),
        ]
        if config.loopback:
            cmd.extend(["-P", "{ stream.capture.sink=true }"])
        if target is not None:
            cmd.extend(["--target", str(target)])
        cmd.append(str(output_path))

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        self._start_time = time.monotonic()
        self._recording = True
        self._stop_event.clear()

        # Start level monitoring thread
        self._level_thread = threading.Thread(
            target=self._level_loop, daemon=True
        )
        self._level_thread.start()

    def stop(self) -> RecordingResult:
        if not self._recording:
            raise RuntimeError("Not recording")

        self._recording = False
        self._stop_event.set()

        # Gracefully stop pw-record with SIGINT
        if self._process and self._process.poll() is None:
            self._process.send_signal(signal.SIGINT)
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()

        # Wait for level thread to finish
        if self._level_thread is not None:
            self._level_thread.join(timeout=2)

        # Read final WAV to get accurate duration
        duration = 0.0
        sample_rate = self._config.sample_rate
        try:
            with wave.open(str(self._output_path), "rb") as wf:
                sample_rate = wf.getframerate()
                duration = wf.getnframes() / sample_rate
        except Exception:
            # Fall back to wall clock if WAV can't be read
            if self._start_time is not None:
                duration = time.monotonic() - self._start_time

        device_name = str(self._config.device or "default")
        source_type = "loopback" if self._config.loopback else "microphone"

        return RecordingResult(
            path=self._output_path,
            duration_seconds=duration,
            sample_rate=sample_rate,
            channels=self._actual_channels,
            device_name=device_name,
            source_type=source_type,
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
        return self._level

    def _level_loop(self) -> None:
        """Poll the growing WAV file for audio level data."""
        wav_header_size = 44  # Standard PCM WAV header
        bytes_per_sample = 2  # s16
        chunk_samples = 2048

        while not self._stop_event.is_set():
            self._stop_event.wait(0.1)
            try:
                file_size = os.path.getsize(self._output_path)
                audio_bytes = file_size - wav_header_size
                if audio_bytes < bytes_per_sample * chunk_samples:
                    continue

                # Read the last chunk of audio data
                read_bytes = chunk_samples * bytes_per_sample
                with open(self._output_path, "rb") as f:
                    f.seek(file_size - read_bytes)
                    data = f.read(read_bytes)

                if len(data) >= bytes_per_sample:
                    samples = np.frombuffer(data, dtype=np.int16)
                    peak = float(np.max(np.abs(samples.astype(np.float32)))) / 32768.0
                    with self._lock:
                        self._level = peak
            except (OSError, ValueError):
                pass  # File not ready or being written
