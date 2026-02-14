"""PipeWire-native audio recorder using pw-record subprocess.

Uses raw PCM pipe mode (stdout) for real-time audio frame access,
enabling overlapped transcription via frame callbacks.
"""

from __future__ import annotations

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

    Uses raw PCM pipe mode: pw-record outputs to stdout, Python handles
    WAV writing and provides real-time frame callbacks for overlapped
    transcription.
    """

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._config: RecordingConfig | None = None
        self._output_path: Path | None = None
        self._start_time: float | None = None
        self._recording = False
        self._level: float = 0.0
        self._lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._actual_sample_rate: int = 16000
        self._actual_channels: int = 1
        self._wav_file: wave.Wave_write | None = None
        self._frames_written: int = 0
        self._frame_callback = None

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
        self._frames_written = 0

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
        cmd.append("-")  # raw PCM to stdout

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Open WAV file — we handle the WAV format ourselves
        self._wav_file = wave.open(str(output_path), "wb")
        self._wav_file.setnchannels(channels)
        self._wav_file.setsampwidth(2)  # 16-bit
        self._wav_file.setframerate(sample_rate)

        self._start_time = time.monotonic()
        self._recording = True
        self._stop_event.clear()

        # Start pipe reader thread
        self._reader_thread = threading.Thread(
            target=self._pipe_reader, daemon=True
        )
        self._reader_thread.start()

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

        # Wait for reader thread to finish
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)

        # Close WAV file
        with self._lock:
            if self._wav_file is not None:
                self._wav_file.close()
                self._wav_file = None

        # Calculate duration from frames written
        duration = self._frames_written / self._actual_sample_rate
        if duration == 0.0 and self._start_time is not None:
            duration = time.monotonic() - self._start_time

        device_name = str(self._config.device or "default")
        source_type = "loopback" if self._config.loopback else "microphone"

        return RecordingResult(
            path=self._output_path,
            duration_seconds=duration,
            sample_rate=self._actual_sample_rate,
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

    def _pipe_reader(self) -> None:
        """Read raw PCM from pw-record stdout, write to WAV, compute levels."""
        # Read in chunks: 4096 bytes = 2048 mono samples or 1024 stereo samples
        chunk_bytes = 4096
        bytes_per_frame = 2 * self._actual_channels  # 2 bytes per sample * channels

        while not self._stop_event.is_set():
            try:
                data = self._process.stdout.read(chunk_bytes)
            except (OSError, ValueError):
                break
            if not data:
                break

            # Ensure we have complete frames
            remainder = len(data) % bytes_per_frame
            if remainder:
                data = data[:-remainder]
            if not data:
                continue

            # Write to WAV file
            with self._lock:
                if self._wav_file is not None:
                    self._wav_file.writeframes(data)
                    self._frames_written += len(data) // bytes_per_frame

            # Compute level from raw PCM
            samples = np.frombuffer(data, dtype=np.int16)
            if len(samples) > 0:
                peak = float(np.max(np.abs(samples.astype(np.float32)))) / 32768.0
                self._level = peak

            # Call frame callback for overlapped transcription
            if self._frame_callback is not None:
                self._frame_callback(
                    samples.copy(), self._actual_sample_rate, self._actual_channels
                )
