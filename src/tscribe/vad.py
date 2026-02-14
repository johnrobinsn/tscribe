"""Voice Activity Detection using Silero VAD (ONNX Runtime).

Downloads and caches the Silero VAD ONNX model (~2 MB) on first use.
Processes streaming audio frames, detects speech boundaries, and emits
speech chunks via a callback for overlapped transcription.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
_MODEL_FILENAME = "silero_vad.onnx"


def _get_models_dir() -> Path:
    from tscribe.paths import get_data_dir

    models_dir = get_data_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def _ensure_model(model_path: Path | None = None) -> Path:
    """Return path to the Silero VAD ONNX model, downloading if needed."""
    if model_path is not None:
        return model_path
    path = _get_models_dir() / _MODEL_FILENAME
    if path.exists():
        return path
    print("Downloading Silero VAD model... (first time only)")
    urllib.request.urlretrieve(_MODEL_URL, path)
    return path


class VadDetector:
    """Streaming VAD that detects speech boundaries in real-time audio.

    Call :meth:`process_frames` with each chunk of audio from a recorder.
    When a complete speech segment is detected (speech followed by
    sufficient silence), the *chunk_callback* is invoked with the
    accumulated speech audio.

    Parameters
    ----------
    threshold : float
        Speech probability threshold (0-1).
    min_silence_ms : int
        Minimum silence duration (ms) to end a speech segment.
    max_speech_s : float
        Force a chunk boundary after this many seconds of continuous speech.
    overlap_s : float
        Seconds of overlap to keep when forcing a boundary.
    model_path : Path | None
        Override model path (for testing).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_ms: int = 1000,
        max_speech_s: float = 30.0,
        overlap_s: float = 2.0,
        model_path: Path | None = None,
    ):
        self._threshold = threshold
        self._min_silence_ms = min_silence_ms
        self._max_speech_s = max_speech_s
        self._overlap_s = overlap_s
        self._model_path = model_path

        self._session = None  # onnxruntime.InferenceSession
        self._model_version: int = 4
        # v4 state
        self._h: np.ndarray | None = None
        self._c: np.ndarray | None = None
        # v5 state
        self._state: np.ndarray | None = None

        self._target_sr = 16000
        self._window_size = 512  # updated in _load_model based on version

        self._raw_pending = np.array([], dtype=np.float32)  # mono at source rate
        self._source_sr: int | None = None
        self._speech_active = False
        self._speech_start_sample = 0
        self._silence_start_sample: int | None = None
        self._total_samples = 0  # in target_sr units

        self._speech_buffer: list[np.ndarray] = []
        self._speech_regions: list[tuple[float, float]] = []

        self._chunk_callback = None

    # -- public API --

    def set_chunk_callback(self, callback) -> None:
        """Register ``callback(audio_int16, sample_rate, offset_secs)``."""
        self._chunk_callback = callback

    def ensure_ready(self) -> None:
        """Pre-load the VAD model (downloads on first use)."""
        self._load_model()

    def process_frames(
        self, frames: np.ndarray, sample_rate: int, channels: int = 1
    ) -> None:
        """Feed raw audio frames from a recorder callback.

        Resamples to 16 kHz mono internally.
        """
        if frames.dtype == np.int16:
            audio = frames.astype(np.float32) / 32768.0
        else:
            audio = frames.astype(np.float32)

        audio = audio.ravel()

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        self._source_sr = sample_rate
        self._raw_pending = np.concatenate([self._raw_pending, audio])

        if sample_rate == self._target_sr:
            src_per_window = self._window_size
        else:
            src_per_window = round(self._window_size * sample_rate / self._target_sr)

        while len(self._raw_pending) >= src_per_window:
            raw_window = self._raw_pending[:src_per_window]
            self._raw_pending = self._raw_pending[src_per_window:]

            if sample_rate != self._target_sr:
                x_old = np.linspace(0, 1, src_per_window)
                x_new = np.linspace(0, 1, self._window_size)
                window = np.interp(x_new, x_old, raw_window).astype(np.float32)
            else:
                window = raw_window

            prob = self._run_vad(window)
            self._process_vad_result(prob, window)
            self._total_samples += self._window_size

    def flush(self) -> None:
        """Emit any remaining speech buffer (call when recording stops)."""
        if self._speech_active and self._speech_buffer:
            self._emit_chunk()
            self._speech_active = False
            self._speech_buffer = []

    @property
    def speech_regions(self) -> list[tuple[float, float]]:
        return list(self._speech_regions)

    @property
    def threshold(self) -> float:
        return self._threshold

    def reset(self) -> None:
        """Reset all state for a new recording session."""
        self._session = None
        self._h = None
        self._c = None
        self._state = None
        self._raw_pending = np.array([], dtype=np.float32)
        self._source_sr = None
        self._speech_active = False
        self._speech_start_sample = 0
        self._silence_start_sample = None
        self._total_samples = 0
        self._speech_buffer = []
        self._speech_regions = []

    # -- internal --

    def _load_model(self) -> None:
        if self._session is not None:
            return

        import onnxruntime as ort

        model_path = _ensure_model(self._model_path)
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

        input_names = {inp.name for inp in self._session.get_inputs()}
        if "state" in input_names:
            self._model_version = 5
            self._window_size = 256  # v5 requires 256 samples (16 ms) at 16 kHz
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self._model_version = 4
            self._window_size = 512  # v4 uses 512 samples (32 ms) at 16 kHz
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def _run_vad(self, audio_window: np.ndarray) -> float:
        self._load_model()

        inp = audio_window.reshape(1, -1).astype(np.float32)
        sr = np.array(self._target_sr, dtype=np.int64)

        if self._model_version == 5:
            ort_inputs = {"input": inp, "state": self._state, "sr": sr}
            output, state_n = self._session.run(None, ort_inputs)
            self._state = state_n
        else:
            ort_inputs = {"input": inp, "h": self._h, "c": self._c, "sr": sr}
            output, hn, cn = self._session.run(None, ort_inputs)
            self._h = hn
            self._c = cn

        return float(np.squeeze(output))

    def _process_vad_result(self, prob: float, window: np.ndarray) -> None:
        current_sample = self._total_samples

        if prob >= self._threshold:
            self._silence_start_sample = None

            if not self._speech_active:
                self._speech_active = True
                self._speech_start_sample = current_sample
                self._speech_buffer = []

            self._speech_buffer.append(window.copy())

            speech_secs = (
                current_sample + self._window_size - self._speech_start_sample
            ) / self._target_sr
            if speech_secs >= self._max_speech_s:
                self._emit_chunk()
                overlap_windows = int(
                    self._overlap_s * self._target_sr / self._window_size
                )
                if overlap_windows > 0 and len(self._speech_buffer) >= overlap_windows:
                    self._speech_buffer = self._speech_buffer[-overlap_windows:]
                else:
                    self._speech_buffer = []
                self._speech_start_sample = (
                    current_sample
                    + self._window_size
                    - len(self._speech_buffer) * self._window_size
                )
        else:
            if self._speech_active:
                if self._silence_start_sample is None:
                    self._silence_start_sample = current_sample

                self._speech_buffer.append(window.copy())

                silence_ms = (
                    (current_sample + self._window_size - self._silence_start_sample)
                    / self._target_sr
                    * 1000
                )
                if silence_ms >= self._min_silence_ms:
                    self._emit_chunk()
                    self._speech_active = False
                    self._speech_buffer = []
                    self._silence_start_sample = None

    def _emit_chunk(self) -> None:
        if not self._speech_buffer:
            return

        audio = np.concatenate(self._speech_buffer)
        audio_int16 = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)

        offset_secs = self._speech_start_sample / self._target_sr
        end_secs = (self._total_samples + self._window_size) / self._target_sr

        self._speech_regions.append((offset_secs, end_secs))

        if self._chunk_callback is not None:
            self._chunk_callback(audio_int16, self._target_sr, offset_secs)
