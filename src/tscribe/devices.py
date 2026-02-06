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


def _import_sounddevice():
    """Import sounddevice, raising a clear error on library conflicts."""
    try:
        import sounddevice as sd
        return sd
    except OSError as e:
        if "GLIBCXX" in str(e) or "libstdc++" in str(e):
            raise OSError(
                "PortAudio failed to load due to a libstdc++ conflict (likely Anaconda).\n"
                "Fix: run with the system libstdc++:\n"
                "  LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 tscribe devices\n"
                "Or remove Anaconda from your PATH/environment."
            ) from e
        raise


def list_devices(loopback_only: bool = False) -> list[AudioDevice]:
    """Enumerate available input devices."""
    sd = _import_sounddevice()

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
    sd = _import_sounddevice()

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
            "No loopback audio device found via sounddevice.\n"
            "If PipeWire is running, try: tscribe devices --loopback\n"
            "PipeWire monitor sources are available for loopback recording."
        )
    if sys.platform == "win32":
        return (
            "No loopback audio device found.\n"
            "On Windows, enable 'Stereo Mix' in Sound settings > Recording devices,\n"
            "or install a virtual audio cable (e.g., VB-Cable)."
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
