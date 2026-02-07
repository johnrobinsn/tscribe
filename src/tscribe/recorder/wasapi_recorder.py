"""WASAPI loopback recorder for Windows via PyAudioWPatch."""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path

from tscribe.recorder.base import Recorder, RecordingConfig, RecordingResult


class WasapiRecorder(Recorder):
    """Records system audio via WASAPI loopback on Windows."""

    def __init__(self):
        self._pyaudio = None
        self._stream = None
        self._wav_file: wave.Wave_write | None = None
        self._config: RecordingConfig | None = None
        self._output_path: Path | None = None
        self._frames_written = 0
        self._start_time: float | None = None
        self._recording = False
        self._level: float = 0.0
        self._lock = threading.Lock()
        self._actual_sample_rate: int = 48000
        self._actual_channels: int = 2

    def _resolve_loopback_device(self, config: RecordingConfig):
        """Find the WASAPI loopback device to record from."""
        import pyaudiowpatch as pyaudio

        p = pyaudio.PyAudio()
        self._pyaudio = p

        if config.device is not None:
            return p.get_device_info_by_index(int(config.device))

        try:
            return p.get_default_wasapi_loopback()
        except (OSError, LookupError) as exc:
            p.terminate()
            self._pyaudio = None
            raise RuntimeError(
                "No WASAPI loopback device found.\n"
                "Ensure your system has an active audio output device."
            ) from exc

    def start(self, output_path: Path, config: RecordingConfig) -> None:
        import pyaudiowpatch as pyaudio

        if self._recording:
            raise RuntimeError("Already recording")

        device_info = self._resolve_loopback_device(config)

        self._actual_sample_rate = int(device_info["defaultSampleRate"])
        self._actual_channels = device_info["maxInputChannels"]
        self._config = config
        self._output_path = output_path
        self._frames_written = 0

        self._wav_file = wave.open(str(output_path), "wb")
        self._wav_file.setnchannels(self._actual_channels)
        self._wav_file.setsampwidth(2)  # 16-bit
        self._wav_file.setframerate(self._actual_sample_rate)

        import numpy as np

        def callback(in_data, frame_count, time_info, status):
            with self._lock:
                if self._wav_file is not None:
                    self._wav_file.writeframes(in_data)
                    self._frames_written += frame_count
                    audio = np.frombuffer(in_data, dtype=np.int16)
                    peak = float(np.max(np.abs(audio.astype(np.float32)))) / 32768.0
                    self._level = peak
            return (in_data, pyaudio.paContinue)

        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=self._actual_channels,
            rate=self._actual_sample_rate,
            frames_per_buffer=512,
            input=True,
            input_device_index=device_info["index"],
            stream_callback=callback,
        )

        self._start_time = time.monotonic()
        self._recording = True

    def stop(self) -> RecordingResult:
        if not self._recording:
            raise RuntimeError("Not recording")

        self._recording = False
        self._stream.stop_stream()
        self._stream.close()
        self._stream = None

        with self._lock:
            self._wav_file.close()
            self._wav_file = None

        self._pyaudio.terminate()
        self._pyaudio = None

        duration = self._frames_written / self._actual_sample_rate
        device_name = "WASAPI Loopback"

        return RecordingResult(
            path=self._output_path,
            duration_seconds=duration,
            sample_rate=self._actual_sample_rate,
            channels=self._actual_channels,
            device_name=device_name,
            source_type="loopback",
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
