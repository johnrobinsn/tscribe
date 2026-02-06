"""PipeWire device enumeration and target resolution."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


@dataclass
class PipewireNode:
    """A PipeWire audio node (source or sink)."""

    serial: int
    name: str  # node.name (e.g., "alsa_output.usb-...")
    description: str  # node.description (human-readable)
    nick: str  # node.nick (short display name)
    media_class: str  # "Audio/Source" or "Audio/Sink"
    is_monitor: bool  # True if Audio/Sink (targeting it captures monitor)


def is_pipewire_available() -> bool:
    """Check if PipeWire daemon is running and accessible."""
    try:
        result = subprocess.run(
            ["pw-cli", "info", "0"],
            capture_output=True,
            timeout=2,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_pw_dump() -> list[dict]:
    """Run pw-dump and return parsed JSON objects."""
    result = subprocess.run(
        ["pw-dump"],
        capture_output=True,
        text=True,
        timeout=3,
    )
    if result.returncode != 0:
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        # pw-dump may output incomplete JSON if killed early
        text = result.stdout.rstrip().rstrip(",")
        if not text.endswith("]"):
            text += "]"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return []


def list_pipewire_nodes(loopback_only: bool = False) -> list[PipewireNode]:
    """Enumerate PipeWire audio nodes.

    Returns Audio/Source (microphones) and Audio/Sink (speakers/outputs).
    Sinks are included because targeting a sink with pw-record captures
    its monitor output (system audio loopback).

    When loopback_only=True, returns only Audio/Sink nodes.
    """
    data = _run_pw_dump()
    nodes = []

    for obj in data:
        info = obj.get("info", {})
        props = info.get("props", {})
        media_class = props.get("media.class", "")

        if media_class not in ("Audio/Source", "Audio/Sink"):
            continue

        if loopback_only and media_class != "Audio/Sink":
            continue

        serial = int(props.get("object.serial", 0))
        nodes.append(PipewireNode(
            serial=serial,
            name=props.get("node.name", ""),
            description=props.get("node.description", ""),
            nick=props.get("node.nick", ""),
            media_class=media_class,
            is_monitor=media_class == "Audio/Sink",
        ))

    return nodes


def get_default_sink_name() -> str | None:
    """Get the node name of the default audio sink.

    Reads PipeWire metadata to find the default.audio.sink entry.
    Returns the node name, or None if not found.
    """
    data = _run_pw_dump()

    for obj in data:
        metadata = obj.get("metadata", [])
        for entry in metadata:
            if entry.get("key") == "default.audio.sink":
                value = entry.get("value", {})
                if isinstance(value, dict):
                    return value.get("name")
                # value might be a JSON string
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        return parsed.get("name")
                    except (json.JSONDecodeError, AttributeError):
                        pass
    return None


def get_node_audio_info(node_name: str) -> dict | None:
    """Get audio channel count for a PipeWire node.

    Returns dict with 'channels' key, or None if not found.
    """
    data = _run_pw_dump()
    for obj in data:
        info = obj.get("info", {})
        props = info.get("props", {})
        if props.get("node.name") == node_name:
            channels = len(props.get("audio.position", "").split(","))
            if channels < 1:
                channels = 2
            return {"channels": channels}
    return None


def resolve_pipewire_target(device=None, loopback: bool = False) -> str | None:
    """Resolve the --target argument for pw-record.

    Args:
        device: Device name or serial from CLI --device flag.
        loopback: Whether to record system audio (monitor of default sink).

    Returns:
        A target string for pw-record --target, or None for default source.
    """
    if device is not None:
        return str(device)

    if loopback:
        return get_default_sink_name()

    return None
