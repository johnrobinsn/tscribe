"""Real audio recorder using sounddevice (PortAudio)."""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path

from tscribe.recorder.base import Recorder, RecordingConfig, RecordingResult


class SounddeviceRecorder(Recorder):
    """Records audio from a sounddevice input stream to a WAV file."""

    def __init__(self):
        self._stream = None
        self._wav_file: wave.Wave_write | None = None
        self._config: RecordingConfig | None = None
        self._output_path: Path | None = None
        self._frames_written = 0
        self._start_time: float | None = None
        self._recording = False
        self._level: float = 0.0
        self._lock = threading.Lock()

    def start(self, output_path: Path, config: RecordingConfig) -> None:
        import sounddevice as sd

        if self._recording:
            raise RuntimeError("Already recording")

        # Use the device's native sample rate for compatibility
        dev_info = sd.query_devices(config.device or sd.default.device[0], kind="input")
        sample_rate = int(dev_info["default_samplerate"])

        self._config = config
        self._actual_sample_rate = sample_rate
        self._output_path = output_path
        self._frames_written = 0

        self._wav_file = wave.open(str(output_path), "wb")
        self._wav_file.setnchannels(config.channels)
        self._wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        self._wav_file.setframerate(sample_rate)

        import numpy as np

        def callback(indata, frames, time_info, status):
            with self._lock:
                if self._wav_file is not None:
                    self._wav_file.writeframes(indata.tobytes())
                    self._frames_written += frames
                    peak = float(np.max(np.abs(indata.astype(np.float32)))) / 32768.0
                    self._level = min(1.0, peak * 3)

        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=config.channels,
            dtype=config.dtype,
            device=config.device,
            callback=callback,
        )
        self._stream.start()
        self._start_time = time.monotonic()
        self._recording = True

    def stop(self) -> RecordingResult:
        if not self._recording:
            raise RuntimeError("Not recording")

        self._recording = False
        self._stream.stop()
        self._stream.close()
        self._stream = None

        with self._lock:
            self._wav_file.close()
            self._wav_file = None

        duration = self._frames_written / self._actual_sample_rate
        device_name = str(self._config.device or "default")
        source_type = "loopback" if self._config.loopback else "microphone"

        return RecordingResult(
            path=self._output_path,
            duration_seconds=duration,
            sample_rate=self._actual_sample_rate,
            channels=self._config.channels,
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
