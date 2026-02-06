"""Tests for path resolution."""

import os

from tscribe.paths import (
    ensure_dirs,
    get_config_path,
    get_data_dir,
    get_recordings_dir,
)


def test_data_dir_default():
    data_dir = get_data_dir()
    assert "tscribe" in str(data_dir)


def test_data_dir_config_override(tmp_path):
    override = str(tmp_path / "custom")
    data_dir = get_data_dir(config_override=override)
    assert data_dir == tmp_path / "custom"


def test_data_dir_env_override(tmp_path, monkeypatch):
    env_path = str(tmp_path / "env_dir")
    monkeypatch.setenv("TSCRIBE_DATA_DIR", env_path)
    data_dir = get_data_dir()
    assert data_dir == tmp_path / "env_dir"


def test_config_override_takes_precedence_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TSCRIBE_DATA_DIR", str(tmp_path / "env"))
    data_dir = get_data_dir(config_override=str(tmp_path / "config"))
    assert data_dir == tmp_path / "config"


def test_subdirectory_helpers(tmp_path):
    assert get_recordings_dir(tmp_path) == tmp_path / "recordings"
    assert get_config_path(tmp_path) == tmp_path / "config.toml"


def test_ensure_dirs(tmp_path):
    ensure_dirs(tmp_path)
    assert (tmp_path / "recordings").is_dir()


def test_ensure_dirs_idempotent(tmp_path):
    ensure_dirs(tmp_path)
    ensure_dirs(tmp_path)  # Should not raise
    assert (tmp_path / "recordings").is_dir()
