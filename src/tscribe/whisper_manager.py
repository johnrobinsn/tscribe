"""Management of whisper.cpp binary and model downloads."""

from __future__ import annotations

import os
import platform
import shutil
import stat
import sys
import urllib.request
from pathlib import Path

import click

# whisper.cpp release info
WHISPER_CPP_REPO = "ggerganov/whisper.cpp"
WHISPER_CPP_VERSION = "v1.7.4"

# Model download base URL (Hugging Face)
MODEL_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

MODEL_FILES = {
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large": "ggml-large-v3.bin",
}


class WhisperManager:
    def __init__(self, whisper_dir: Path):
        self.whisper_dir = whisper_dir
        self.models_dir = whisper_dir / "models"
        self._binary_name = "main" + (".exe" if sys.platform == "win32" else "")
        self.binary_path = whisper_dir / self._binary_name

    def is_binary_available(self) -> bool:
        """Check if whisper.cpp binary exists (managed or on PATH)."""
        return self.find_binary() is not None

    def find_binary(self) -> Path | None:
        """Find whisper.cpp: check managed location, then PATH."""
        if self.binary_path.exists():
            return self.binary_path

        # Check common names on PATH
        for name in ("whisper-cpp", "whisper", "main"):
            found = shutil.which(name)
            if found:
                return Path(found)

        return None

    def is_model_available(self, model: str) -> bool:
        """Check if a model file exists."""
        return self.model_path(model).exists()

    def model_path(self, model: str) -> Path:
        """Return path for a model file."""
        filename = MODEL_FILES.get(model, f"ggml-{model}.bin")
        return self.models_dir / filename

    def download_binary(self, force: bool = False) -> Path:
        """Download pre-built whisper.cpp binary for current platform."""
        if self.binary_path.exists() and not force:
            return self.binary_path

        self.whisper_dir.mkdir(parents=True, exist_ok=True)

        system = platform.system().lower()
        machine = platform.machine().lower()

        # Map platform to release asset names
        if system == "linux" and machine in ("x86_64", "amd64"):
            asset_suffix = "linux-x86_64"
        elif system == "linux" and machine in ("aarch64", "arm64"):
            asset_suffix = "linux-aarch64"
        elif system == "darwin" and machine in ("arm64", "aarch64"):
            asset_suffix = "darwin-arm64"
        elif system == "darwin" and machine == "x86_64":
            asset_suffix = "darwin-x86_64"
        elif system == "windows" and machine in ("amd64", "x86_64"):
            asset_suffix = "windows-x86_64"
        else:
            raise RuntimeError(
                f"No pre-built binary available for {system}/{machine}. "
                "Please install whisper.cpp manually and ensure it's on your PATH."
            )

        url = (
            f"https://github.com/{WHISPER_CPP_REPO}/releases/download/"
            f"{WHISPER_CPP_VERSION}/whisper-cli-{asset_suffix}.zip"
        )

        click.echo(f"Downloading whisper.cpp from {url}...")
        zip_path = self.whisper_dir / "whisper-cpp.zip"

        try:
            _download_with_progress(url, zip_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download whisper.cpp: {e}")

        # Extract the binary
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Find the main binary in the archive
            for name in zf.namelist():
                basename = Path(name).name
                if basename in ("main", "main.exe", "whisper-cli", "whisper-cli.exe"):
                    with zf.open(name) as src, open(self.binary_path, "wb") as dst:
                        dst.write(src.read())
                    break
            else:
                # Extract all and hope for the best
                zf.extractall(self.whisper_dir)

        zip_path.unlink(missing_ok=True)

        # Make binary executable on Unix
        if sys.platform != "win32":
            self.binary_path.chmod(self.binary_path.stat().st_mode | stat.S_IEXEC)

        click.echo(f"whisper.cpp installed to {self.binary_path}")
        return self.binary_path

    def download_model(self, model: str = "base", force: bool = False) -> Path:
        """Download a ggml model from Hugging Face."""
        model_file = self.model_path(model)
        if model_file.exists() and not force:
            return model_file

        self.models_dir.mkdir(parents=True, exist_ok=True)

        filename = MODEL_FILES.get(model)
        if not filename:
            raise ValueError(f"Unknown model: {model!r}")

        url = f"{MODEL_BASE_URL}/{filename}"
        click.echo(f"Downloading model '{model}' from {url}...")

        try:
            _download_with_progress(url, model_file)
        except Exception as e:
            model_file.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download model: {e}")

        click.echo(f"Model '{model}' saved to {model_file}")
        return model_file

    def setup(self, model: str = "base", force: bool = False) -> None:
        """Full setup: download binary + model."""
        self.download_binary(force=force)
        self.download_model(model=model, force=force)


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a URL to a file with a progress bar."""
    req = urllib.request.Request(url, headers={"User-Agent": "tscribe"})
    with urllib.request.urlopen(req) as response:
        total = int(response.headers.get("Content-Length", 0))
        with open(dest, "wb") as f:
            if total > 0:
                with click.progressbar(length=total, label="Downloading") as bar:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        bar.update(len(chunk))
            else:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
