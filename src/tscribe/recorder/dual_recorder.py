"""Dual recorder: simultaneous mic + loopback, mixed to single WAV."""

from __future__ import annotations

import shutil
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from tscribe.recorder.base import Recorder, RecordingConfig, RecordingResult


class DualRecorder(Recorder):
    """Records from mic and loopback simultaneously, mixes into one WAV."""

    def __init__(self, mic_recorder: Recorder, loopback_recorder: Recorder):
        self._mic = mic_recorder
        self._loopback = loopback_recorder
        self._output_path: Path | None = None
        self._start_time: float | None = None
        self._recording = False
        self._tmpdir: str | None = None
        self._mic_tmp: Path | None = None
        self._loopback_tmp: Path | None = None

    def start(self, output_path: Path, config: RecordingConfig) -> None:
        if self._recording:
            raise RuntimeError("Already recording")

        self._output_path = output_path
        self._tmpdir = tempfile.mkdtemp(prefix="tscribe_dual_")
        self._mic_tmp = Path(self._tmpdir) / "mic.wav"
        self._loopback_tmp = Path(self._tmpdir) / "loopback.wav"

        mic_config = RecordingConfig(
            sample_rate=config.sample_rate,
            channels=config.channels,
            device=None,
            loopback=False,
        )
        lb_config = RecordingConfig(
            sample_rate=config.sample_rate,
            channels=config.channels,
            device=config.device,
            loopback=True,
        )

        self._loopback.start(self._loopback_tmp, lb_config)
        try:
            self._mic.start(self._mic_tmp, mic_config)
        except Exception:
            try:
                self._loopback.stop()
            except Exception:
                pass
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            raise

        self._start_time = time.monotonic()
        self._recording = True

    def stop(self) -> RecordingResult:
        if not self._recording:
            raise RuntimeError("Not recording")

        self._recording = False

        mic_result = self._mic.stop()
        lb_result = self._loopback.stop()

        duration, sample_rate, channels = _mix_wavs(
            self._mic_tmp, self._loopback_tmp, self._output_path
        )

        shutil.rmtree(self._tmpdir, ignore_errors=True)

        return RecordingResult(
            path=self._output_path,
            duration_seconds=duration,
            sample_rate=sample_rate,
            channels=channels,
            device_name="mic+loopback",
            source_type="both",
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
        return max(self._mic.level, self._loopback.level)


def _read_wav_as_float(path: Path) -> tuple[np.ndarray, int, int]:
    """Read a WAV file as float32 in [-1.0, 1.0]. Returns (samples, rate, channels)."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    if not raw:
        return np.zeros(0, dtype=np.float32), sr, ch
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        samples = samples.reshape(-1, ch)
    return samples, sr, ch


def _stereo_to_mono(audio: np.ndarray) -> np.ndarray:
    """Downmix to mono by averaging channels. Input shape (n, channels)."""
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1).astype(np.float32)


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample audio using numpy linear interpolation."""
    if src_rate == dst_rate:
        return audio
    duration = len(audio) / src_rate
    n_out = int(duration * dst_rate)
    x_old = np.linspace(0, duration, len(audio), endpoint=False)
    x_new = np.linspace(0, duration, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _match_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    """Scale audio so its RMS matches *target_rms*. Skips silence."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-6:
        return audio
    gain = target_rms / rms
    gain = min(gain, 100.0)
    return (audio * gain).astype(np.float32)


def _mix_wavs(
    mic_path: Path, loopback_path: Path, output_path: Path
) -> tuple[float, int, int]:
    """Mix mic and loopback WAVs into a single mono output.

    Returns (duration_seconds, sample_rate, channels).
    """
    mic_audio, mic_rate, mic_ch = _read_wav_as_float(mic_path)
    lb_audio, lb_rate, lb_ch = _read_wav_as_float(loopback_path)

    # Downmix stereo to mono
    if mic_ch > 1:
        mic_audio = _stereo_to_mono(mic_audio)
    if lb_ch > 1:
        lb_audio = _stereo_to_mono(lb_audio)

    # Check for empty streams before resampling/padding
    mic_empty = len(mic_audio) == 0
    lb_empty = len(lb_audio) == 0

    if mic_empty and lb_empty:
        mixed = np.zeros(1, dtype=np.float32)
        output_rate = max(mic_rate, lb_rate)
    elif mic_empty:
        mixed = lb_audio
        output_rate = lb_rate
    elif lb_empty:
        mixed = mic_audio
        output_rate = mic_rate
    else:
        # Resample to the higher rate
        output_rate = max(mic_rate, lb_rate)
        if mic_rate != output_rate:
            mic_audio = _resample_linear(mic_audio, mic_rate, output_rate)
        if lb_rate != output_rate:
            lb_audio = _resample_linear(lb_audio, lb_rate, output_rate)

        # Pad shorter stream
        max_len = max(len(mic_audio), len(lb_audio))
        if len(mic_audio) < max_len:
            mic_audio = np.pad(mic_audio, (0, max_len - len(mic_audio)))
        if len(lb_audio) < max_len:
            lb_audio = np.pad(lb_audio, (0, max_len - len(lb_audio)))

        # Boost mic toward loopback loudness so voice is audible in the mix,
        # but only to 50% of loopback RMS to avoid amplifying mic noise floor.
        lb_rms = float(np.sqrt(np.mean(lb_audio ** 2)))
        if lb_rms > 1e-6:
            mic_audio = _match_rms(mic_audio, lb_rms * 0.5)

        # Asymmetric mix: favour loopback (remote audio) over mic to keep
        # background noise low while preserving mic voice for transcription.
        mixed = np.clip(mic_audio * 0.3 + lb_audio * 0.7, -1.0, 1.0)

    # Normalize to use full dynamic range (peak at 90% to leave headroom)
    peak = float(np.max(np.abs(mixed)))
    if peak > 1e-6:
        mixed = mixed * (0.9 / peak)

    # Write mono int16 WAV
    out_samples = (mixed * 32767).astype(np.int16)
    duration = len(out_samples) / output_rate

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(output_rate)
        wf.writeframes(out_samples.tobytes())

    return duration, output_rate, 1
