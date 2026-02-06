"""Tests for device enumeration and loopback detection."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tscribe.devices import (
    AudioDevice,
    _detect_loopback,
    get_platform_loopback_guidance,
    list_devices,
)


MOCK_DEVICES = [
    {
        "name": "Built-in Microphone",
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 44100.0,
        "hostapi": 0,
    },
    {
        "name": "Monitor of Built-in Audio",
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 44100.0,
        "hostapi": 0,
    },
    {
        "name": "Built-in Speakers",
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 44100.0,
        "hostapi": 0,
    },
    {
        "name": "BlackHole 2ch",
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 48000.0,
        "hostapi": 0,
    },
    {
        "name": "Stereo Mix",
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 44100.0,
        "hostapi": 1,
    },
]

MOCK_HOSTAPIS = [
    {"name": "PulseAudio"},
    {"name": "Windows WASAPI"},
]


@pytest.fixture
def mock_sd(monkeypatch):
    """Create a mock sounddevice module and inject it into sys.modules."""
    mock = MagicMock()
    mock.query_devices.return_value = MOCK_DEVICES
    mock.query_hostapis.return_value = MOCK_HOSTAPIS
    monkeypatch.setitem(sys.modules, "sounddevice", mock)
    return mock


def test_list_devices_all(mock_sd):
    devices = list_devices()
    assert len(devices) == 4
    names = [d.name for d in devices]
    assert "Built-in Microphone" in names
    assert "Monitor of Built-in Audio" in names
    assert "BlackHole 2ch" in names
    assert "Stereo Mix" in names
    assert "Built-in Speakers" not in names


def test_list_devices_loopback_only(mock_sd):
    devices = list_devices(loopback_only=True)
    names = [d.name for d in devices]
    assert "Monitor of Built-in Audio" in names
    assert "BlackHole 2ch" in names
    assert "Stereo Mix" in names
    assert "Built-in Microphone" not in names


def test_list_devices_empty(mock_sd):
    mock_sd.query_devices.return_value = []
    devices = list_devices()
    assert devices == []


def test_detect_loopback_monitor():
    assert _detect_loopback("Monitor of Built-in Audio", "PulseAudio") is True


def test_detect_loopback_blackhole():
    assert _detect_loopback("BlackHole 2ch", "CoreAudio") is True


def test_detect_loopback_stereo_mix():
    assert _detect_loopback("Stereo Mix", "Windows WASAPI") is True


def test_detect_loopback_regular_mic():
    assert _detect_loopback("Built-in Microphone", "PulseAudio") is False


def test_detect_loopback_wasapi_loopback():
    assert _detect_loopback("Speakers (Loopback)", "Windows WASAPI") is True


def test_platform_guidance_linux(monkeypatch):
    monkeypatch.setattr("tscribe.devices.sys.platform", "linux")
    guidance = get_platform_loopback_guidance()
    assert guidance is not None
    assert "PipeWire" in guidance


def test_platform_guidance_darwin(monkeypatch):
    monkeypatch.setattr("tscribe.devices.sys.platform", "darwin")
    guidance = get_platform_loopback_guidance()
    assert guidance is not None
    assert "BlackHole" in guidance


def test_platform_guidance_win32(monkeypatch):
    monkeypatch.setattr("tscribe.devices.sys.platform", "win32")
    guidance = get_platform_loopback_guidance()
    assert guidance is not None
    assert "Stereo Mix" in guidance
