"""Tests for configuration loading and management."""

import pytest

from tscribe.config import TscribeConfig


def test_defaults():
    config = TscribeConfig()
    assert config.recording.sample_rate == 16000
    assert config.recording.channels == 1
    assert config.recording.auto_transcribe is True
    assert config.transcription.model == "small"
    assert config.transcription.output_formats == ["txt", "json"]
    assert config.transcription.gpu is False
    assert config.storage.data_dir == ""


def test_load_missing_file(tmp_path):
    config = TscribeConfig.load(tmp_path / "nonexistent.toml")
    assert config.recording.sample_rate == 16000


def test_load_none_path():
    config = TscribeConfig.load(None)
    assert config.recording.sample_rate == 16000


def test_save_and_load_roundtrip(tmp_path):
    config = TscribeConfig()
    config.recording.sample_rate = 44100
    config.transcription.model = "small"
    config_path = tmp_path / "config.toml"
    config.save(config_path)

    loaded = TscribeConfig.load(config_path)
    assert loaded.recording.sample_rate == 44100
    assert loaded.transcription.model == "small"
    # Unchanged defaults preserved
    assert loaded.recording.channels == 1
    assert loaded.transcription.gpu is False


def test_get_dotted_key():
    config = TscribeConfig()
    assert config.get("recording.sample_rate") == 16000
    assert config.get("transcription.model") == "small"
    assert config.get("storage.data_dir") == ""


def test_get_invalid_key_format():
    config = TscribeConfig()
    with pytest.raises(KeyError, match="Invalid key format"):
        config.get("sample_rate")


def test_get_unknown_section():
    config = TscribeConfig()
    with pytest.raises(KeyError, match="Unknown config section"):
        config.get("nonexistent.key")


def test_get_unknown_key():
    config = TscribeConfig()
    with pytest.raises(KeyError, match="Unknown config key"):
        config.get("recording.nonexistent")


def test_set_dotted_key():
    config = TscribeConfig()
    config.set("recording.sample_rate", 44100)
    assert config.recording.sample_rate == 44100


def test_set_string_coercion_to_int():
    config = TscribeConfig()
    config.set("recording.sample_rate", "44100")
    assert config.recording.sample_rate == 44100


def test_set_string_coercion_to_bool():
    config = TscribeConfig()
    config.set("recording.auto_transcribe", "false")
    assert config.recording.auto_transcribe is False
    config.set("recording.auto_transcribe", "true")
    assert config.recording.auto_transcribe is True


def test_set_string_coercion_to_list():
    config = TscribeConfig()
    config.set("transcription.output_formats", "txt,srt")
    assert config.transcription.output_formats == ["txt", "srt"]


def test_set_invalid_model():
    config = TscribeConfig()
    with pytest.raises(ValueError, match="Invalid model"):
        config.set("transcription.model", "nonexistent")


def test_set_invalid_output_format():
    config = TscribeConfig()
    with pytest.raises(ValueError, match="Invalid output format"):
        config.set("transcription.output_formats", "txt,invalid")


def test_set_negative_sample_rate():
    config = TscribeConfig()
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        config.set("recording.sample_rate", -1)


def test_set_negative_channels():
    config = TscribeConfig()
    with pytest.raises(ValueError, match="channels must be positive"):
        config.set("recording.channels", 0)


def test_partial_toml_loads_with_defaults(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('[recording]\nsample_rate = 48000\n')

    config = TscribeConfig.load(config_path)
    assert config.recording.sample_rate == 48000
    assert config.recording.channels == 1  # default preserved
    assert config.transcription.model == "small"  # section not in file


def test_unknown_keys_in_toml_ignored(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('[recording]\nsample_rate = 48000\nunknown_key = "ignored"\n')

    config = TscribeConfig.load(config_path)
    assert config.recording.sample_rate == 48000


def test_to_dict():
    config = TscribeConfig()
    d = config._to_dict()
    assert d["recording"]["sample_rate"] == 16000
    assert d["transcription"]["output_formats"] == ["txt", "json"]
    assert "storage" in d
