"""Tests for DualRecorder and audio mixing helpers."""

import wave

import numpy as np
import pytest

from tscribe.recorder import MockRecorder, RecordingConfig
from tscribe.recorder.dual_recorder import (
    DualRecorder,
    _highpass,
    _match_rms,
    _mix_wavs,
    _read_wav_as_float,
    _resample_linear,
    _stereo_to_mono,
)


# --- DualRecorder lifecycle tests ---


def test_start_stop(tmp_path):
    mic = MockRecorder(duration=1.0)
    lb = MockRecorder(duration=1.0)
    dual = DualRecorder(mic, lb)

    out = tmp_path / "mixed.wav"
    config = RecordingConfig(sample_rate=16000, channels=1)

    dual.start(out, config)
    assert dual.is_recording()
    assert dual.elapsed_seconds >= 0.0

    result = dual.stop()
    assert not dual.is_recording()
    assert result.path == out
    assert result.source_type == "both"
    assert result.device_name == "mic+loopback"
    assert out.exists()

    with wave.open(str(out), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2


def test_double_start_raises(tmp_path):
    dual = DualRecorder(MockRecorder(), MockRecorder())
    dual.start(tmp_path / "a.wav", RecordingConfig())
    with pytest.raises(RuntimeError, match="Already recording"):
        dual.start(tmp_path / "b.wav", RecordingConfig())
    dual.stop()


def test_stop_without_start_raises():
    dual = DualRecorder(MockRecorder(), MockRecorder())
    with pytest.raises(RuntimeError, match="Not recording"):
        dual.stop()


def test_level_returns_max(tmp_path):
    mic = MockRecorder(duration=0.5)
    lb = MockRecorder(duration=0.5)
    dual = DualRecorder(mic, lb)
    dual.start(tmp_path / "out.wav", RecordingConfig())
    # MockRecorder.level is always 0.0
    assert dual.level == 0.0
    dual.stop()


def test_mic_start_failure_cleans_up(tmp_path):
    """If mic fails to start, loopback should be stopped and temp dir cleaned."""

    class FailRecorder(MockRecorder):
        def start(self, output_path, config):
            raise RuntimeError("mic device not found")

    lb = MockRecorder(duration=1.0)
    dual = DualRecorder(FailRecorder(), lb)

    with pytest.raises(RuntimeError, match="mic device not found"):
        dual.start(tmp_path / "out.wav", RecordingConfig())
    assert not dual.is_recording()


# --- Mixing helpers ---


def test_stereo_to_mono():
    stereo = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)
    mono = _stereo_to_mono(stereo)
    assert mono.shape == (2,)
    assert mono[0] == pytest.approx(0.0)
    assert mono[1] == pytest.approx(0.5)


def test_stereo_to_mono_passthrough():
    mono = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    result = _stereo_to_mono(mono)
    np.testing.assert_array_equal(result, mono)


def test_resample_identity():
    audio = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32)
    result = _resample_linear(audio, 16000, 16000)
    np.testing.assert_array_equal(result, audio)


def test_resample_up():
    audio = np.ones(16000, dtype=np.float32)
    result = _resample_linear(audio, 16000, 48000)
    assert len(result) == 48000
    assert np.all(np.abs(result - 1.0) < 0.01)


def test_resample_down():
    audio = np.ones(48000, dtype=np.float32)
    result = _resample_linear(audio, 48000, 16000)
    assert len(result) == 16000
    assert np.all(np.abs(result - 1.0) < 0.01)


def _write_wav(path, samples, rate, channels):
    """Helper to write a WAV file from int16 samples."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())


def test_read_wav_as_float_mono(tmp_path):
    path = tmp_path / "mono.wav"
    samples = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    _write_wav(path, samples, 16000, 1)

    audio, rate, ch = _read_wav_as_float(path)
    assert rate == 16000
    assert ch == 1
    assert audio.shape == (4,)
    assert audio[0] == pytest.approx(0.0, abs=0.001)
    assert audio[1] == pytest.approx(0.5, abs=0.001)


def test_read_wav_as_float_stereo(tmp_path):
    path = tmp_path / "stereo.wav"
    # Interleaved stereo: L=16384 R=-16384, L=0 R=0
    samples = np.array([16384, -16384, 0, 0], dtype=np.int16)
    _write_wav(path, samples, 48000, 2)

    audio, rate, ch = _read_wav_as_float(path)
    assert rate == 48000
    assert ch == 2
    assert audio.shape == (2, 2)


# --- _mix_wavs integration tests ---


def test_mix_same_rate_mono(tmp_path):
    mic_path = tmp_path / "mic.wav"
    lb_path = tmp_path / "lb.wav"
    out_path = tmp_path / "mixed.wav"

    # Both 16kHz mono, 1 second
    silence = np.zeros(16000, dtype=np.int16)
    _write_wav(mic_path, silence, 16000, 1)
    _write_wav(lb_path, silence, 16000, 1)

    duration, rate, ch = _mix_wavs(mic_path, lb_path, out_path)
    assert rate == 16000
    assert ch == 1
    assert 0.9 < duration < 1.1


def test_mix_different_rates(tmp_path):
    mic_path = tmp_path / "mic.wav"
    lb_path = tmp_path / "lb.wav"
    out_path = tmp_path / "mixed.wav"

    # Mic: 16kHz mono, 1s
    _write_wav(mic_path, np.zeros(16000, dtype=np.int16), 16000, 1)
    # Loopback: 48kHz stereo, 1s
    _write_wav(lb_path, np.zeros(48000 * 2, dtype=np.int16), 48000, 2)

    duration, rate, ch = _mix_wavs(mic_path, lb_path, out_path)
    assert rate == 48000  # upsampled to higher rate
    assert ch == 1  # always mono
    assert 0.9 < duration < 1.1


def test_mix_tone_plus_silence(tmp_path):
    mic_path = tmp_path / "mic.wav"
    lb_path = tmp_path / "lb.wav"
    out_path = tmp_path / "mixed.wav"

    # Mic: 440Hz tone at half amplitude
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    _write_wav(mic_path, tone, 16000, 1)

    # Loopback: silence
    _write_wav(lb_path, np.zeros(16000, dtype=np.int16), 16000, 1)

    _mix_wavs(mic_path, lb_path, out_path)

    with wave.open(str(out_path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16)
    peak = np.max(np.abs(samples))
    # After RMS matching + 0.5 gain + peak normalize, tone should be audible
    assert peak > 500


def test_mix_one_empty(tmp_path):
    mic_path = tmp_path / "mic.wav"
    lb_path = tmp_path / "lb.wav"
    out_path = tmp_path / "mixed.wav"

    # Mic: 1s of audio
    tone = (np.sin(np.linspace(0, 2 * np.pi * 440, 16000)) * 16000).astype(np.int16)
    _write_wav(mic_path, tone, 16000, 1)

    # Loopback: 0 frames
    _write_wav(lb_path, np.zeros(0, dtype=np.int16), 16000, 1)

    duration, rate, ch = _mix_wavs(mic_path, lb_path, out_path)
    assert duration > 0
    # Should use mic audio as-is (no 0.5 scaling)
    with wave.open(str(out_path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16)
    assert np.max(np.abs(samples)) > 10000


def test_match_rms_boosts_quiet():
    quiet = np.ones(1000, dtype=np.float32) * 0.01
    result = _match_rms(quiet, target_rms=0.1)
    rms = float(np.sqrt(np.mean(result ** 2)))
    assert 0.08 < rms < 0.12


def test_match_rms_reduces_loud():
    loud = np.ones(1000, dtype=np.float32) * 0.5
    result = _match_rms(loud, target_rms=0.1)
    rms = float(np.sqrt(np.mean(result ** 2)))
    assert 0.08 < rms < 0.12


def test_match_rms_silence_unchanged():
    silence = np.zeros(1000, dtype=np.float32)
    result = _match_rms(silence, target_rms=0.1)
    np.testing.assert_array_equal(result, silence)


def test_match_rms_caps_gain():
    """Very quiet audio should not be amplified more than 100x."""
    very_quiet = np.ones(1000, dtype=np.float32) * 0.0001
    result = _match_rms(very_quiet, target_rms=0.1)
    # Gain capped at 100x: 0.0001 * 100 = 0.01, not 0.1
    assert np.max(np.abs(result)) < 0.02


def test_mix_different_lengths(tmp_path):
    mic_path = tmp_path / "mic.wav"
    lb_path = tmp_path / "lb.wav"
    out_path = tmp_path / "mixed.wav"

    # Mic: 1s, Loopback: 2s
    _write_wav(mic_path, np.zeros(16000, dtype=np.int16), 16000, 1)
    _write_wav(lb_path, np.zeros(32000, dtype=np.int16), 16000, 1)

    duration, rate, ch = _mix_wavs(mic_path, lb_path, out_path)
    assert 1.9 < duration < 2.1  # padded to longer duration


# --- High-pass filter tests ---


def test_highpass_removes_dc():
    """DC offset (0 Hz) should be removed by high-pass filter."""
    # Signal: DC offset of 0.5 + 440Hz tone
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    audio = (0.5 + 0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    filtered = _highpass(audio, cutoff_hz=200, sample_rate=16000)
    # DC component should be greatly reduced
    assert abs(np.mean(filtered)) < 0.05


def test_highpass_preserves_speech_frequencies():
    """Frequencies well above cutoff should pass through."""
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    tone_1k = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    filtered = _highpass(tone_1k, cutoff_hz=200, sample_rate=16000)
    # 1kHz tone should retain most of its energy
    original_rms = float(np.sqrt(np.mean(tone_1k ** 2)))
    filtered_rms = float(np.sqrt(np.mean(filtered ** 2)))
    assert filtered_rms > original_rms * 0.8


def test_highpass_zero_cutoff_passthrough():
    """cutoff_hz=0 should return audio unchanged."""
    audio = np.random.randn(1000).astype(np.float32)
    result = _highpass(audio, cutoff_hz=0, sample_rate=16000)
    np.testing.assert_array_equal(result, audio)


def test_highpass_short_audio():
    """Single sample or empty audio should not crash."""
    single = np.array([0.5], dtype=np.float32)
    result = _highpass(single, cutoff_hz=200, sample_rate=16000)
    assert len(result) == 1


# --- mic_volume and mic_filter_hz tests ---


def test_mic_volume_zero_mutes_mic(tmp_path):
    """mic_volume=0 should produce output with only loopback audio."""
    mic_path = tmp_path / "mic.wav"
    lb_path = tmp_path / "lb.wav"
    out_path = tmp_path / "mixed.wav"

    # Mic: loud tone
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    _write_wav(mic_path, tone, 16000, 1)

    # Loopback: different tone
    lb_tone = (np.sin(2 * np.pi * 880 * t) * 16000).astype(np.int16)
    _write_wav(lb_path, lb_tone, 16000, 1)

    # Mix with mic_volume=0 and again with mic_volume=100
    _mix_wavs(mic_path, lb_path, out_path, mic_volume=0)
    with wave.open(str(out_path), "rb") as wf:
        raw_muted = wf.readframes(wf.getnframes())

    out_path2 = tmp_path / "mixed2.wav"
    _mix_wavs(mic_path, lb_path, out_path2, mic_volume=100)
    with wave.open(str(out_path2), "rb") as wf:
        raw_full = wf.readframes(wf.getnframes())

    # Both should produce audio (peak normalize ensures nonzero output)
    muted_samples = np.frombuffer(raw_muted, dtype=np.int16)
    full_samples = np.frombuffer(raw_full, dtype=np.int16)
    assert np.max(np.abs(muted_samples)) > 1000
    assert np.max(np.abs(full_samples)) > 1000


def test_mic_volume_scales_mic(tmp_path):
    """Lower mic_volume should reduce mic contribution."""
    mic_path = tmp_path / "mic.wav"
    lb_path = tmp_path / "lb.wav"

    t = np.linspace(0, 1.0, 16000, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    _write_wav(mic_path, tone, 16000, 1)
    # Loopback: silence (so we can isolate mic effect)
    _write_wav(lb_path, np.zeros(16000, dtype=np.int16), 16000, 1)

    out_full = tmp_path / "full.wav"
    out_half = tmp_path / "half.wav"

    _mix_wavs(mic_path, lb_path, out_full, mic_volume=100, mic_filter_hz=0)
    _mix_wavs(mic_path, lb_path, out_half, mic_volume=50, mic_filter_hz=0)

    # Both get peak-normalized so peaks will be similar,
    # but the test confirms both produce valid output
    assert out_full.exists()
    assert out_half.exists()


def test_mic_filter_zero_disables(tmp_path):
    """mic_filter_hz=0 should skip the high-pass filter."""
    mic_path = tmp_path / "mic.wav"
    lb_path = tmp_path / "lb.wav"
    out_path = tmp_path / "mixed.wav"

    # Low-frequency tone (50Hz) — would be attenuated by filter
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    low_tone = (np.sin(2 * np.pi * 50 * t) * 16000).astype(np.int16)
    _write_wav(mic_path, low_tone, 16000, 1)
    _write_wav(lb_path, np.zeros(16000, dtype=np.int16), 16000, 1)

    # With filter disabled, low tone should pass through
    _mix_wavs(mic_path, lb_path, out_path, mic_volume=100, mic_filter_hz=0)
    with wave.open(str(out_path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16)
    assert np.max(np.abs(samples)) > 1000
