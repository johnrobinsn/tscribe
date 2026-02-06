"""Tests for PipeWire device enumeration."""

import json
import subprocess
from unittest.mock import patch

import pytest

from tscribe.pipewire_devices import (
    PipewireNode,
    get_default_sink_name,
    is_pipewire_available,
    list_pipewire_nodes,
    resolve_pipewire_target,
)


# Sample pw-dump output matching real PipeWire structure
SAMPLE_PW_DUMP = [
    {
        "id": 34,
        "type": "PipeWire:Interface:Node/3",
        "info": {
            "props": {
                "object.serial": "57",
                "media.class": "Audio/Sink",
                "node.name": "alsa_output.usb-device.analog-stereo",
                "node.description": "USB Headset Analog Stereo",
                "node.nick": "USB Headset",
            }
        },
    },
    {
        "id": 53,
        "type": "PipeWire:Interface:Node/3",
        "info": {
            "props": {
                "object.serial": "60",
                "media.class": "Audio/Source",
                "node.name": "alsa_input.pci-builtin.analog-stereo",
                "node.description": "Built-in Audio Analog Stereo",
                "node.nick": "Built-in Mic",
            }
        },
    },
    {
        "id": 55,
        "type": "PipeWire:Interface:Node/3",
        "info": {
            "props": {
                "object.serial": "55",
                "media.class": "Audio/Sink",
                "node.name": "alsa_output.pci-builtin.iec958-stereo",
                "node.description": "Built-in Audio Digital Stereo",
                "node.nick": "Built-in Digital",
            }
        },
    },
    # Stream node — should be excluded
    {
        "id": 81,
        "type": "PipeWire:Interface:Node/3",
        "info": {
            "props": {
                "object.serial": "90",
                "media.class": "Stream/Input/Audio",
                "node.name": "Firefox",
            }
        },
    },
    # Device node — should be excluded
    {
        "id": 46,
        "type": "PipeWire:Interface:Device/3",
        "info": {
            "props": {
                "object.serial": "46",
                "media.class": "Audio/Device",
            }
        },
    },
    # Metadata with default sink
    {
        "id": 40,
        "type": "PipeWire:Interface:Metadata/3",
        "metadata": [
            {
                "key": "default.audio.sink",
                "value": {"name": "alsa_output.usb-device.analog-stereo"},
            },
            {
                "key": "default.audio.source",
                "value": {"name": "alsa_input.pci-builtin.analog-stereo"},
            },
        ],
    },
]


def _mock_pw_dump_run(*args, **kwargs):
    """Mock subprocess.run for pw-dump."""
    cmd = args[0] if args else kwargs.get("args", [])
    if cmd and cmd[0] == "pw-dump":
        result = subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=json.dumps(SAMPLE_PW_DUMP),
            stderr="",
        )
        return result
    raise FileNotFoundError(f"No such file: {cmd[0]}")


def _mock_pw_cli_run(*args, **kwargs):
    """Mock subprocess.run for pw-cli info 0."""
    cmd = args[0] if args else kwargs.get("args", [])
    if cmd and cmd[0] == "pw-cli":
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    raise FileNotFoundError(f"No such file: {cmd[0]}")


# ──── is_pipewire_available ────


def test_available_when_pw_cli_succeeds():
    with patch("tscribe.pipewire_devices.subprocess.run", _mock_pw_cli_run):
        assert is_pipewire_available() is True


def test_unavailable_when_pw_cli_missing():
    with patch("tscribe.pipewire_devices.subprocess.run", side_effect=FileNotFoundError):
        assert is_pipewire_available() is False


def test_unavailable_when_pw_cli_fails():
    def mock_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="")

    with patch("tscribe.pipewire_devices.subprocess.run", mock_run):
        assert is_pipewire_available() is False


def test_unavailable_when_pw_cli_times_out():
    with patch(
        "tscribe.pipewire_devices.subprocess.run",
        side_effect=subprocess.TimeoutExpired("pw-cli", 2),
    ):
        assert is_pipewire_available() is False


# ──── list_pipewire_nodes ────


def test_list_all_audio_nodes():
    with patch("tscribe.pipewire_devices.subprocess.run", _mock_pw_dump_run):
        nodes = list_pipewire_nodes()

    assert len(nodes) == 3
    names = {n.name for n in nodes}
    assert "alsa_output.usb-device.analog-stereo" in names
    assert "alsa_input.pci-builtin.analog-stereo" in names
    assert "alsa_output.pci-builtin.iec958-stereo" in names


def test_list_excludes_streams():
    with patch("tscribe.pipewire_devices.subprocess.run", _mock_pw_dump_run):
        nodes = list_pipewire_nodes()

    names = {n.name for n in nodes}
    assert "Firefox" not in names


def test_list_loopback_only():
    with patch("tscribe.pipewire_devices.subprocess.run", _mock_pw_dump_run):
        nodes = list_pipewire_nodes(loopback_only=True)

    assert len(nodes) == 2
    assert all(n.media_class == "Audio/Sink" for n in nodes)
    assert all(n.is_monitor for n in nodes)


def test_list_node_properties():
    with patch("tscribe.pipewire_devices.subprocess.run", _mock_pw_dump_run):
        nodes = list_pipewire_nodes()

    source = next(n for n in nodes if n.media_class == "Audio/Source")
    assert source.serial == 60
    assert source.nick == "Built-in Mic"
    assert source.description == "Built-in Audio Analog Stereo"
    assert source.is_monitor is False


def test_list_empty_when_pw_dump_fails():
    def mock_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="error")

    with patch("tscribe.pipewire_devices.subprocess.run", mock_run):
        nodes = list_pipewire_nodes()

    assert nodes == []


def test_list_empty_when_pw_dump_invalid_json():
    def mock_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="not json", stderr="")

    with patch("tscribe.pipewire_devices.subprocess.run", mock_run):
        nodes = list_pipewire_nodes()

    assert nodes == []


# ──── get_default_sink_name ────


def test_get_default_sink_name():
    with patch("tscribe.pipewire_devices.subprocess.run", _mock_pw_dump_run):
        name = get_default_sink_name()

    assert name == "alsa_output.usb-device.analog-stereo"


def test_get_default_sink_name_not_found():
    # pw-dump returns data without metadata
    def mock_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="[]", stderr=""
        )

    with patch("tscribe.pipewire_devices.subprocess.run", mock_run):
        assert get_default_sink_name() is None


def test_get_default_sink_name_string_value():
    """Handle case where metadata value is a JSON string, not a dict."""
    data = [
        {
            "id": 40,
            "type": "PipeWire:Interface:Metadata/3",
            "metadata": [
                {
                    "key": "default.audio.sink",
                    "value": '{"name": "some_sink"}',
                },
            ],
        },
    ]

    def mock_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=json.dumps(data), stderr=""
        )

    with patch("tscribe.pipewire_devices.subprocess.run", mock_run):
        assert get_default_sink_name() == "some_sink"


# ──── resolve_pipewire_target ────


def test_resolve_target_with_device():
    assert resolve_pipewire_target(device="my_device") == "my_device"


def test_resolve_target_with_numeric_device():
    assert resolve_pipewire_target(device=57) == "57"


def test_resolve_target_loopback():
    with patch("tscribe.pipewire_devices.get_default_sink_name") as mock:
        mock.return_value = "alsa_output.usb-device.analog-stereo"
        target = resolve_pipewire_target(loopback=True)

    assert target == "alsa_output.usb-device.analog-stereo"


def test_resolve_target_loopback_no_default():
    with patch("tscribe.pipewire_devices.get_default_sink_name") as mock:
        mock.return_value = None
        target = resolve_pipewire_target(loopback=True)

    assert target is None


def test_resolve_target_default_mic():
    assert resolve_pipewire_target() is None


def test_resolve_target_device_overrides_loopback():
    target = resolve_pipewire_target(device="specific_device", loopback=True)
    assert target == "specific_device"
