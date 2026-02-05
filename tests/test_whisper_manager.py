"""Tests for whisper.cpp binary and model management."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tscribe.whisper_manager import WhisperManager


@pytest.fixture
def wm(tmp_path):
    return WhisperManager(tmp_path / "whisper")


class TestFindBinary:
    def test_no_binary(self, wm):
        assert wm.find_binary() is None
        assert wm.is_binary_available() is False

    def test_managed_binary(self, wm):
        wm.binary_path.parent.mkdir(parents=True, exist_ok=True)
        wm.binary_path.write_text("fake binary")
        assert wm.find_binary() == wm.binary_path
        assert wm.is_binary_available() is True

    def test_path_fallback(self, wm, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/whisper-cpp" if name == "whisper-cpp" else None)
        result = wm.find_binary()
        assert result == Path("/usr/bin/whisper-cpp")


class TestModelPath:
    def test_known_model(self, wm):
        path = wm.model_path("base")
        assert path.name == "ggml-base.bin"
        assert path.parent == wm.models_dir

    def test_all_models(self, wm):
        for model in ("tiny", "base", "small", "medium", "large"):
            path = wm.model_path(model)
            assert path.name.startswith("ggml-")
            assert path.name.endswith(".bin")

    def test_model_not_available(self, wm):
        assert wm.is_model_available("base") is False

    def test_model_available(self, wm):
        wm.models_dir.mkdir(parents=True, exist_ok=True)
        wm.model_path("base").write_bytes(b"fake model data")
        assert wm.is_model_available("base") is True


class TestDownloadModel:
    def test_download_unknown_model(self, wm):
        with pytest.raises(ValueError, match="Unknown model"):
            wm.download_model("nonexistent")

    def test_skip_if_exists(self, wm):
        wm.models_dir.mkdir(parents=True, exist_ok=True)
        wm.model_path("base").write_bytes(b"existing")
        result = wm.download_model("base")
        assert result == wm.model_path("base")
