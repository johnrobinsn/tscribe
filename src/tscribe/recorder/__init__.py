"""Audio recording module."""

from tscribe.recorder.base import Recorder, RecordingConfig, RecordingResult
from tscribe.recorder.dual_recorder import DualRecorder
from tscribe.recorder.mock_recorder import MockRecorder

__all__ = [
    "Recorder",
    "RecordingConfig",
    "RecordingResult",
    "MockRecorder",
    "DualRecorder",
]
