"""Configuration loading, saving, and management."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import tomli_w

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from tscribe.constants import (
    DEFAULT_CHANNELS,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_FORMATS,
    DEFAULT_SAMPLE_RATE,
    VALID_MODELS,
    VALID_OUTPUT_FORMATS,
)


@dataclass
class RecordingDefaults:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    default_device: str = ""
    auto_transcribe: bool = True
    mic_volume: int = 100
    mic_filter_hz: int = 200


@dataclass
class TranscriptionDefaults:
    model: str = DEFAULT_MODEL
    language: str = DEFAULT_LANGUAGE
    output_formats: list[str] = field(default_factory=lambda: list(DEFAULT_OUTPUT_FORMATS))
    gpu: bool = False


@dataclass
class StorageDefaults:
    data_dir: str = ""


@dataclass
class TscribeConfig:
    recording: RecordingDefaults = field(default_factory=RecordingDefaults)
    transcription: TranscriptionDefaults = field(default_factory=TranscriptionDefaults)
    storage: StorageDefaults = field(default_factory=StorageDefaults)

    @classmethod
    def load(cls, config_path: Path | None = None) -> TscribeConfig:
        """Load config from TOML file, falling back to defaults for missing keys."""
        config = cls()
        if config_path is None or not config_path.exists():
            return config

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        if "recording" in data:
            for k, v in data["recording"].items():
                if hasattr(config.recording, k):
                    setattr(config.recording, k, v)

        if "transcription" in data:
            for k, v in data["transcription"].items():
                if hasattr(config.transcription, k):
                    setattr(config.transcription, k, v)

        if "storage" in data:
            for k, v in data["storage"].items():
                if hasattr(config.storage, k):
                    setattr(config.storage, k, v)

        return config

    def save(self, config_path: Path) -> None:
        """Write current config to TOML file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        data = self._to_dict()
        with open(config_path, "wb") as f:
            tomli_w.dump(data, f)

    def get(self, key: str) -> Any:
        """Get a config value by dotted key (e.g., 'recording.sample_rate')."""
        section, _, name = key.partition(".")
        if not name:
            raise KeyError(f"Invalid key format: {key!r}. Use 'section.key' (e.g., 'recording.sample_rate')")
        obj = getattr(self, section, None)
        if obj is None:
            raise KeyError(f"Unknown config section: {section!r}")
        if not hasattr(obj, name):
            raise KeyError(f"Unknown config key: {key!r}")
        return getattr(obj, name)

    def set(self, key: str, value: Any) -> None:
        """Set a config value by dotted key."""
        section, _, name = key.partition(".")
        if not name:
            raise KeyError(f"Invalid key format: {key!r}. Use 'section.key' (e.g., 'recording.sample_rate')")
        obj = getattr(self, section, None)
        if obj is None:
            raise KeyError(f"Unknown config section: {section!r}")
        if not hasattr(obj, name):
            raise KeyError(f"Unknown config key: {key!r}")

        # Coerce value to match the field type
        current = getattr(obj, name)
        coerced = _coerce_value(value, current, key)
        _validate_value(key, coerced)
        setattr(obj, name, coerced)

    def _to_dict(self) -> dict:
        """Convert config to a nested dict for TOML serialization."""
        result = {}
        for section_field in fields(self):
            section_obj = getattr(self, section_field.name)
            section_dict = {}
            for f in fields(section_obj):
                section_dict[f.name] = getattr(section_obj, f.name)
            result[section_field.name] = section_dict
        return result


def _coerce_value(value: Any, current: Any, key: str) -> Any:
    """Coerce a string value to match the type of the current value."""
    if isinstance(value, str) and not isinstance(current, str):
        if isinstance(current, bool):
            if value.lower() in ("true", "1", "yes"):
                return True
            elif value.lower() in ("false", "0", "no"):
                return False
            raise ValueError(f"Cannot convert {value!r} to bool for key {key!r}")
        if isinstance(current, int):
            return int(value)
        if isinstance(current, list):
            return [s.strip() for s in value.split(",")]
    return value


def _validate_value(key: str, value: Any) -> None:
    """Validate a config value."""
    if key == "transcription.model" and value not in VALID_MODELS:
        raise ValueError(f"Invalid model: {value!r}. Choose from: {', '.join(VALID_MODELS)}")
    if key == "transcription.output_formats":
        for fmt in value:
            if fmt not in VALID_OUTPUT_FORMATS:
                raise ValueError(f"Invalid output format: {fmt!r}. Choose from: {', '.join(VALID_OUTPUT_FORMATS)}")
    if key == "recording.sample_rate" and value <= 0:
        raise ValueError(f"sample_rate must be positive, got {value}")
    if key == "recording.channels" and value <= 0:
        raise ValueError(f"channels must be positive, got {value}")
    if key == "recording.mic_volume" and not (0 <= value <= 100):
        raise ValueError(f"mic_volume must be 0-100, got {value}")
    if key == "recording.mic_filter_hz" and value < 0:
        raise ValueError(f"mic_filter_hz must be >= 0, got {value}")
