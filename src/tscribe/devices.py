"""Audio device enumeration and loopback detection."""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    default_samplerate: float
    is_loopback: bool
    hostapi: str


def list_devices(loopback_only: bool = False) -> list[AudioDevice]:
    """Enumerate available input devices."""
    import sounddevice as sd

    raw_devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    results = []

    for i, dev in enumerate(raw_devices):
        if dev["max_input_channels"] < 1:
            continue

        hostapi_name = hostapis[dev["hostapi"]]["name"] if dev["hostapi"] < len(hostapis) else ""
        is_loopback = _detect_loopback(dev["name"], hostapi_name)

        if loopback_only and not is_loopback:
            continue

        results.append(AudioDevice(
            index=i,
            name=dev["name"],
            max_input_channels=dev["max_input_channels"],
            default_samplerate=dev["default_samplerate"],
            is_loopback=is_loopback,
            hostapi=hostapi_name,
        ))

    return results


def get_default_device() -> AudioDevice | None:
    """Get the system default input device."""
    import sounddevice as sd

    try:
        info = sd.query_devices(kind="input")
    except sd.PortAudioError:
        return None

    hostapis = sd.query_hostapis()
    hostapi_name = hostapis[info["hostapi"]]["name"] if info["hostapi"] < len(hostapis) else ""
    idx = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device

    return AudioDevice(
        index=idx,
        name=info["name"],
        max_input_channels=info["max_input_channels"],
        default_samplerate=info["default_samplerate"],
        is_loopback=_detect_loopback(info["name"], hostapi_name),
        hostapi=hostapi_name,
    )


def get_platform_loopback_guidance() -> str | None:
    """Return setup instructions if no loopback source is available."""
    if sys.platform == "darwin":
        return (
            "No loopback audio device found.\n"
            "macOS requires a virtual audio driver for system audio capture.\n"
            "Install BlackHole: https://github.com/ExistentialAudio/BlackHole\n"
            "  brew install blackhole-2ch"
        )
    if sys.platform == "linux":
        return (
            "No loopback audio device found.\n"
            "On Linux, ensure PulseAudio or PipeWire is running.\n"
            "Monitor sources should appear automatically."
        )
    if sys.platform == "win32":
        return (
            "No loopback audio device found.\n"
            "On Windows, WASAPI loopback should be available by default.\n"
            "Try selecting a WASAPI host API device."
        )
    return None


def _detect_loopback(device_name: str, hostapi_name: str) -> bool:
    """Heuristic detection of loopback/monitor sources."""
    name_lower = device_name.lower()
    hostapi_lower = hostapi_name.lower()

    # Linux: PulseAudio/PipeWire monitor sources
    if "monitor" in name_lower:
        return True

    # macOS: virtual audio drivers
    if "blackhole" in name_lower or "loopback" in name_lower:
        return True

    # Windows: WASAPI loopback
    if "wasapi" in hostapi_lower and "loopback" in name_lower:
        return True

    # Generic: "stereo mix" on some Windows systems
    if "stereo mix" in name_lower:
        return True

    return False
