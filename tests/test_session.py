"""Tests for session management."""

import json
import re
import wave

import numpy as np
import pytest

from tscribe.recorder.base import RecordingResult
from tscribe.session import SessionManager


@pytest.fixture
def session_mgr(tmp_path):
    return SessionManager(tmp_path / "recordings")


def _create_wav(path, duration=1.0, sample_rate=16000):
    """Helper to create a minimal WAV file."""
    samples = np.zeros(int(duration * sample_rate), dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


class TestGenerateSessionStem:
    def test_format(self, session_mgr):
        stem = session_mgr.generate_session_stem()
        assert re.match(r"\d{4}-\d{2}-\d{2}-\d{6}", stem)


class TestMetadata:
    def test_create_metadata(self, session_mgr, tmp_path):
        result = RecordingResult(
            path=tmp_path / "test.wav",
            duration_seconds=5.0,
            sample_rate=16000,
            channels=1,
            device_name="Built-in Microphone",
            source_type="microphone",
        )
        meta = session_mgr.create_metadata(result)
        assert meta["duration_seconds"] == 5.0
        assert meta["sample_rate"] == 16000
        assert meta["device"] == "Built-in Microphone"
        assert meta["transcribed"] is False
        assert meta["original_path"] is None

    def test_create_metadata_with_extras(self, session_mgr, tmp_path):
        result = RecordingResult(
            path=tmp_path / "test.wav",
            duration_seconds=5.0,
            sample_rate=16000,
            channels=1,
            device_name="default",
            source_type="microphone",
        )
        meta = session_mgr.create_metadata(result, transcribed=True, model="base")
        assert meta["transcribed"] is True
        assert meta["model"] == "base"

    def test_write_and_read_metadata(self, session_mgr):
        meta = {"duration_seconds": 10.0, "sample_rate": 16000}
        session_mgr.write_metadata("2025-01-15-143022", meta)
        loaded = session_mgr.read_metadata("2025-01-15-143022")
        assert loaded == meta

    def test_read_missing_metadata(self, session_mgr):
        assert session_mgr.read_metadata("nonexistent") is None

    def test_update_metadata(self, session_mgr):
        meta = {"duration_seconds": 10.0, "transcribed": False}
        session_mgr.write_metadata("test-session", meta)
        session_mgr.update_metadata("test-session", transcribed=True, model="base")
        loaded = session_mgr.read_metadata("test-session")
        assert loaded["transcribed"] is True
        assert loaded["model"] == "base"
        assert loaded["duration_seconds"] == 10.0


class TestListSessions:
    def test_empty_directory(self, session_mgr):
        sessions = session_mgr.list_sessions()
        assert sessions == []

    def test_lists_wav_files(self, session_mgr):
        _create_wav(session_mgr.recordings_dir / "2025-01-15-143022.wav")
        _create_wav(session_mgr.recordings_dir / "2025-01-14-091500.wav")

        sessions = session_mgr.list_sessions()
        assert len(sessions) == 2
        stems = [s.stem for s in sessions]
        # Sorted by date descending
        assert stems[0] == "2025-01-15-143022"
        assert stems[1] == "2025-01-14-091500"

    def test_limit(self, session_mgr):
        for i in range(5):
            _create_wav(session_mgr.recordings_dir / f"2025-01-{10+i:02d}-120000.wav")

        sessions = session_mgr.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_with_transcript(self, session_mgr):
        _create_wav(session_mgr.recordings_dir / "2025-01-15-143022.wav")
        (session_mgr.recordings_dir / "2025-01-15-143022.txt").write_text("hello world")

        sessions = session_mgr.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].transcribed is True
        assert sessions[0].txt_path is not None

    def test_without_transcript(self, session_mgr):
        _create_wav(session_mgr.recordings_dir / "2025-01-15-143022.wav")

        sessions = session_mgr.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].transcribed is False
        assert sessions[0].txt_path is None

    def test_search(self, session_mgr):
        _create_wav(session_mgr.recordings_dir / "2025-01-15-143022.wav")
        (session_mgr.recordings_dir / "2025-01-15-143022.txt").write_text("hello world meeting")

        _create_wav(session_mgr.recordings_dir / "2025-01-14-091500.wav")
        (session_mgr.recordings_dir / "2025-01-14-091500.txt").write_text("grocery list")

        sessions = session_mgr.list_sessions(search="meeting")
        assert len(sessions) == 1
        assert sessions[0].stem == "2025-01-15-143022"

    def test_search_case_insensitive(self, session_mgr):
        _create_wav(session_mgr.recordings_dir / "2025-01-15-143022.wav")
        (session_mgr.recordings_dir / "2025-01-15-143022.txt").write_text("Hello World")

        sessions = session_mgr.list_sessions(search="hello")
        assert len(sessions) == 1

    def test_search_no_match(self, session_mgr):
        _create_wav(session_mgr.recordings_dir / "2025-01-15-143022.wav")
        (session_mgr.recordings_dir / "2025-01-15-143022.txt").write_text("hello world")

        sessions = session_mgr.list_sessions(search="nonexistent")
        assert sessions == []

    def test_with_metadata(self, session_mgr):
        _create_wav(session_mgr.recordings_dir / "2025-01-15-143022.wav")
        session_mgr.write_metadata("2025-01-15-143022", {
            "duration_seconds": 45.5,
            "sample_rate": 16000,
        })

        sessions = session_mgr.list_sessions()
        assert sessions[0].duration_seconds == 45.5
        assert sessions[0].metadata is not None

    def test_sort_by_duration(self, session_mgr):
        _create_wav(session_mgr.recordings_dir / "a.wav")
        session_mgr.write_metadata("a", {"duration_seconds": 10.0})
        _create_wav(session_mgr.recordings_dir / "b.wav")
        session_mgr.write_metadata("b", {"duration_seconds": 30.0})

        sessions = session_mgr.list_sessions(sort_by="duration")
        assert sessions[0].stem == "a"
        assert sessions[1].stem == "b"


class TestGetSession:
    def test_existing(self, session_mgr):
        _create_wav(session_mgr.recordings_dir / "2025-01-15-143022.wav")
        session = session_mgr.get_session("2025-01-15-143022")
        assert session is not None
        assert session.stem == "2025-01-15-143022"

    def test_missing(self, session_mgr):
        assert session_mgr.get_session("nonexistent") is None


class TestImportExternal:
    def test_import_copies_file(self, session_mgr, tmp_path):
        source = tmp_path / "external.wav"
        _create_wav(source)

        stem, dest = session_mgr.import_external(source)
        assert dest.exists()
        assert dest.parent == session_mgr.recordings_dir
        assert re.match(r"\d{4}-\d{2}-\d{2}-\d{6}", stem)
