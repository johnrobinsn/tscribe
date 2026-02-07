# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
uv sync                              # Install dependencies (dev environment)
uv run tscribe <command>             # Run CLI from project dir
uv tool install .                    # Install globally as `tscribe`
```

## Testing

```bash
uv run pytest                        # Run all unit tests (~193 tests)
uv run pytest -m "not slow"          # Skip integration tests (require model download)
uv run pytest tests/test_cli.py      # Run a single test file
uv run pytest -k "test_record"       # Run tests matching pattern
uv run pytest --cov=tscribe --cov-report=term-missing  # With coverage
```

Integration tests are marked `slow` and require downloading a whisper model.

## Architecture

**CLI entry point:** `src/tscribe/cli.py` — Click-based with subcommands (record, play, transcribe, list, search, open, dump, path, devices, config).

**Recorder backends** (`src/tscribe/recorder/`) use a Strategy pattern via `Recorder` ABC in `base.py`:
- `PipewireRecorder` — Linux: `pw-record` subprocess with loopback via `-P '{ stream.capture.sink=true }'`
- `WasapiRecorder` — Windows: PyAudioWPatch in callback mode (blocking reads hang during silence)
- `SounddeviceRecorder` — macOS/mic: PortAudio bindings
- `MockRecorder` — Testing without hardware

Platform routing in `cli.py::_create_recorder()` picks the right backend based on OS and loopback/mic mode.

**Session management** (`session.py`) — `SessionManager` handles file naming (YYYY-MM-DD-HHMMSS), metadata (.meta JSON), REF navigation (HEAD/HEAD~N/stem), listing, and searching.

**Transcription** (`transcriber.py`) — Wraps faster-whisper with lazy model loading. Outputs txt/json/srt/vtt.

**Device enumeration** — `devices.py` (sounddevice) and `pipewire_devices.py` (pw-dump JSON parsing).

## Critical Patterns

### Lazy imports are mandatory for `sounddevice`
PortAudio has a `libstdc++` conflict with Anaconda on some systems. Always import sounddevice **inside functions**, never at module level. Same pattern for `yt-dlp` (lazy-imported only when transcribing URLs).

### Testing hardware-dependent code
- Mock sounddevice: `monkeypatch.setitem(sys.modules, "sounddevice", mock)`
- Mock PyAudioWPatch: `monkeypatch.setitem(sys.modules, "pyaudiowpatch", mock)`
- Use `MockRecorder` for recorder-agnostic tests
- Mock `WhisperModel` by injecting `t._model = mock` — mock info object must have `duration` (float) for progress callbacks
- Simulate Ctrl+C: `signal.raise_signal(signal.SIGINT)` with `threading.Event`
- PipeWire tests mock `subprocess.Popen` and patch `tscribe.pipewire_devices.*` at source

### WASAPI constraints (Windows)
Must use callback mode and device's native sample rate/channels — WASAPI won't resample.

### PipeWire loopback
`pw-record --target <sink>` alone captures mic. Must add `-P '{ stream.capture.sink=true }'` for system audio capture.

## Storage Layout

```
~/.tscribe/
├── config.toml          # User configuration
└── recordings/          # All session files
    ├── YYYY-MM-DD-HHMMSS.wav
    ├── YYYY-MM-DD-HHMMSS.txt
    ├── YYYY-MM-DD-HHMMSS.json
    └── YYYY-MM-DD-HHMMSS.meta
```
