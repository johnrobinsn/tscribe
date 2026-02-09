"""Abstract recorder interface and shared data types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RecordingConfig:
    """Configuration for a recording session."""
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    device: Optional[int | str] = None
    loopback: bool = False


@dataclass
class RecordingResult:
    """Result of a completed recording session."""
    path: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    device_name: str
    source_type: str  # "microphone" | "loopback" | "both"


class Recorder(ABC):
    """Abstract interface for audio recording."""

    @abstractmethod
    def start(self, output_path: Path, config: RecordingConfig) -> None:
        """Begin recording to the given path."""
        ...

    @abstractmethod
    def stop(self) -> RecordingResult:
        """Stop recording and finalize the file. Returns result metadata."""
        ...

    @abstractmethod
    def is_recording(self) -> bool:
        """Whether a recording is currently active."""
        ...

    @property
    @abstractmethod
    def elapsed_seconds(self) -> float:
        """Seconds elapsed since recording started."""
        ...

    @property
    @abstractmethod
    def level(self) -> float:
        """Current audio level (0.0 to 1.0)."""
        ...
