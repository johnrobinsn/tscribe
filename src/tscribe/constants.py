"""Shared constants and defaults."""

APP_NAME = "tscribe"

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_MODEL = "base"
DEFAULT_LANGUAGE = "auto"
DEFAULT_OUTPUT_FORMATS = ["txt", "json"]

VALID_MODELS = ("tiny", "base", "small", "medium", "large")
VALID_OUTPUT_FORMATS = ("txt", "json", "srt", "vtt")
VALID_SORT_FIELDS = ("date", "duration", "name")
