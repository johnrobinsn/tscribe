"""Cross-platform path resolution for tscribe data directories."""

import os
from pathlib import Path

from platformdirs import user_data_dir

from tscribe.constants import APP_NAME


def get_data_dir(config_override: str = "") -> Path:
    """Resolve the tscribe data directory.

    Priority: config_override > TSCRIBE_DATA_DIR env var > platform default.
    """
    if config_override:
        return Path(config_override)

    env_dir = os.environ.get("TSCRIBE_DATA_DIR")
    if env_dir:
        return Path(env_dir)

    return Path(user_data_dir(APP_NAME))


def get_recordings_dir(data_dir: Path) -> Path:
    return data_dir / "recordings"


def get_config_path(data_dir: Path) -> Path:
    return data_dir / "config.toml"


def ensure_dirs(data_dir: Path) -> None:
    """Create all required subdirectories if they don't exist."""
    get_recordings_dir(data_dir).mkdir(parents=True, exist_ok=True)
