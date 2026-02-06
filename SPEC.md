# tscribe — Specification

## Overview

**tscribe** is a cross-platform CLI tool for recording audio from any source (microphone, system audio, or both), transcribing it to text using open-source speech-to-text models, and managing the resulting recordings and transcripts. It is designed to run on modest hardware (CPU-only) while optionally leveraging GPU acceleration for higher-quality transcription.

### Primary Use Cases

- Capturing both sides of video calls (Teams, Zoom, etc.) for later transcription
- Personal voice memos and audio note-taking
- Transcribing existing audio files (podcasts, lectures, downloaded recordings)
- General-purpose audio recording from any source with searchable transcripts

---

## V1 Scope

V1 focuses on core functionality: **record, transcribe, list/search**.

### In Scope (V1)

- Audio recording from microphone and system audio (loopback)
- Transcription via faster-whisper (CPU by default, GPU optional)
- Transcription of external audio files (not just tscribe recordings)
- Listing and searching past recordings and transcripts
- Auto-transcribe after recording (configurable)
- Manual transcribe command for any audio file

### Deferred (Post-V1)

- **V1.1**: Timestamped notes/annotations during recording
- **V1.2**: Speaker diarization (identifying who said what)
- Real-time streaming transcription during recording

---

## Platform Support

| Platform | Microphone | System Audio (Loopback) | Notes |
|----------|-----------|------------------------|-------|
| Linux    | Yes       | Yes                    | PulseAudio/PipeWire monitor sources |
| Windows  | Yes       | Yes                    | WASAPI loopback |
| macOS    | Yes       | Yes (requires setup)   | Requires BlackHole or similar virtual audio device |

macOS system audio capture requires the user to install a third-party virtual audio driver (e.g., BlackHole). tscribe will detect available loopback sources and guide the user if none are found.

---

## Architecture

### Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.10+ | Rapid development, rich audio ecosystem, cross-platform |
| Package Manager | uv | Fast, modern, handles venvs and lockfiles |
| Audio Capture | sounddevice (PortAudio) | Well-maintained, clean API, cross-platform, numpy integration |
| Transcription | faster-whisper (CTranslate2) | Pre-built wheels on all platforms, fast CPU inference, auto model download |
| CLI Framework | argparse or click | Git-style subcommands |
| Audio Format | WAV (recording) | Uncompressed during recording for zero overhead; whisper.cpp consumes WAV natively |

### System Diagram

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│ Audio Source │────▶│  sounddevice  │────▶│  WAV Writer    │────▶│  .wav file   │
│ (mic/loop)  │     │  (PortAudio)  │     │  (wave module) │     │              │
└─────────────┘     └──────────────┘     └───────────────┘     └──────┬───────┘
                                                                       │
                                                                       ▼
                                                              ┌───────────────┐
                                                              │ faster-whisper │
                                                              │  (Python API)  │
                                                              └───────┬───────┘
                                                                       │
                                                                       ▼
                                                              ┌───────────────┐
                                                              │  Transcript    │
                                                              │  (.txt + .json)│
                                                              └───────────────┘
```

### Component Responsibilities

- **Recorder**: Manages audio device selection, recording lifecycle (start/stop/pause), writes WAV to disk.
- **Transcriber**: Uses faster-whisper Python API for transcription, writes transcript files.
- **Session Manager**: Handles file naming, metadata, listing, and searching across recordings.
- **Device Manager**: Enumerates audio devices, detects loopback sources, provides platform-specific guidance.

---

## CLI Design

Git-style subcommand interface:

```
tscribe <command> [options]
```

### Commands

#### `tscribe record`

Start a recording session.

```
tscribe record [OPTIONS]

Options:
  --device, -d DEVICE     Audio device name or index (default: system default mic)
  --loopback, -l          Record system audio instead of microphone
  --both                  Record both microphone and system audio (mixed or separate tracks)
  --output, -o PATH       Output file path (default: auto-generated in data dir)
  --no-transcribe         Don't auto-transcribe after recording stops
  --sample-rate RATE      Sample rate in Hz (default: 16000, optimal for Whisper)
  --channels N            Number of channels (default: 1, mono)
```

Recording is stopped with Ctrl+C (SIGINT), which triggers a clean shutdown: finalize the WAV file, then optionally start transcription.

#### `tscribe transcribe`

Transcribe an audio file.

```
tscribe transcribe <FILE> [OPTIONS]

Options:
  --model, -m MODEL       Whisper model size (default: base)
                          Choices: tiny, base, small, medium, large
  --language LANG         Language code (default: auto-detect)
  --output, -o PATH       Output file path (default: alongside input file)
  --format FORMAT         Output format: txt, json, srt, vtt, all (default: txt,json)
  --gpu                   Use GPU acceleration if available
```

#### `tscribe list`

List past recordings and their transcription status.

```
tscribe list [OPTIONS]

Options:
  --limit, -n N           Number of entries to show (default: 20)
  --search, -s QUERY      Search transcript text
  --sort FIELD            Sort by: date, duration, name (default: date)
  --no-header             Omit table header
```

Output format:

```
Date                 Duration  Transcribed  File
2025-01-15 14:30:22  00:45:12  Yes          meeting-2025-01-15-143022.wav
2025-01-14 09:15:00  00:02:33  No           memo-2025-01-14-091500.wav
```

#### `tscribe devices`

List available audio input devices and loopback sources.

```
tscribe devices [OPTIONS]

Options:
  --loopback              Show only loopback/monitor sources
```

#### `tscribe config`

View or set configuration.

```
tscribe config [KEY] [VALUE]
tscribe config --list
```

---

## Storage Layout

Flat directory structure under a configurable data directory:

```
~/.local/share/tscribe/          # XDG default on Linux
~/Library/Application Support/tscribe/  # macOS
%LOCALAPPDATA%/tscribe/          # Windows

├── config.toml                   # User configuration
├── recordings/
│   ├── 2025-01-15-143022.wav     # Audio recording
│   ├── 2025-01-15-143022.txt     # Plain text transcript
│   ├── 2025-01-15-143022.json    # Structured transcript (timestamps, segments)
│   ├── 2025-01-15-143022.meta    # Recording metadata (device, duration, sample rate)
│   ├── 2025-01-14-091500.wav
│   └── ...
```

### File Naming Convention

Files are named by timestamp: `YYYY-MM-DD-HHMMSS`. Users can optionally provide a custom name via `--output`.

### Metadata File (.meta)

JSON format:

```json
{
  "created": "2025-01-15T14:30:22Z",
  "duration_seconds": 2712,
  "sample_rate": 16000,
  "channels": 1,
  "device": "Built-in Microphone",
  "source_type": "microphone",
  "transcribed": true,
  "model": "base",
  "original_path": null
}
```

For external files transcribed via `tscribe transcribe`, `original_path` stores the source file path, and the WAV in the recordings directory is either a symlink or copy.

---

## Transcription Details

### faster-whisper Integration

- **Python API**: Uses faster-whisper (CTranslate2 backend) directly as a Python library — no subprocess or binary management needed.
- **Auto-download**: Models are downloaded automatically from Hugging Face on first use of each model size.
- **Cross-platform**: Pre-built wheels available for Linux, macOS, and Windows via pip/uv.

### Default Model

- **Default**: `base` (~74MB, ~10x realtime on modern CPU)
- **Recommended for quality**: `small` (~244MB, ~3-4x realtime on CPU)
- Configurable via `tscribe config model <size>` or `--model` flag

### Output Formats

- **txt**: Plain text transcript, one segment per line
- **json**: Structured output with per-segment timestamps, text, and confidence:

```json
{
  "file": "2025-01-15-143022.wav",
  "model": "base",
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 4.52,
      "text": "Hello, welcome to the meeting.",
      "confidence": 0.92
    }
  ]
}
```

- **srt/vtt**: Standard subtitle formats (available via `--format`)

Default output: both `txt` and `json`.

---

## Configuration

TOML format at `<data_dir>/config.toml`:

```toml
[recording]
sample_rate = 16000
channels = 1
default_device = ""          # empty = system default
auto_transcribe = true

[transcription]
model = "base"
language = "auto"
output_formats = ["txt", "json"]
gpu = false

[storage]
data_dir = ""                # empty = platform default
```

Configuration priority: CLI flags > config file > defaults.

---

## Audio Recording Details

### Sample Rate

Default: 16000 Hz (16 kHz). This is the native input rate for Whisper models. Recording at higher rates is supported but audio will be resampled to 16 kHz before transcription.

### Recording Lifecycle

1. User runs `tscribe record`
2. Device is opened via sounddevice
3. Audio is streamed to a WAV file via Python's `wave` module
4. User presses Ctrl+C to stop
5. WAV file is finalized (header updated with correct size)
6. If auto-transcribe is enabled, transcription starts immediately
7. Metadata file is written

### Signal Handling

Ctrl+C (SIGINT) triggers graceful shutdown:
- Stop the audio stream
- Finalize the WAV file header
- Proceed to transcription if enabled
- A second Ctrl+C forces immediate exit

### System Audio Capture (Loopback)

Platform-specific implementation behind a common interface:

- **Linux**: Use PulseAudio/PipeWire monitor sources. sounddevice + PortAudio can enumerate these as input devices.
- **Windows**: Use WASAPI loopback mode. May require platform-specific sounddevice configuration.
- **macOS**: Requires user to install BlackHole or similar. tscribe detects available virtual audio devices and provides setup guidance if none found.

---

## Dependencies

### Python Dependencies (Minimal)

| Package | Purpose |
|---------|---------|
| sounddevice | Audio capture (PortAudio bindings) |
| numpy | Audio buffer handling (required by sounddevice) |
| click | CLI framework |
| faster-whisper | Speech-to-text transcription (CTranslate2 backend) |

### System Dependencies

| Dependency | Required | Notes |
|-----------|----------|-------|
| PortAudio | Yes | Installed via system package manager or bundled with sounddevice wheels |
| BlackHole (macOS) | For loopback only | User-installed, documented |

### Total Python Dependency Count Target

Aim for **fewer than 10** direct Python dependencies (excluding test/dev dependencies).

---

## Testing Strategy

### Test Pyramid

```
          ┌──────────┐
          │   E2E    │  Full CLI invocations with fixture audio files
         ┌┴──────────┴┐
         │ Integration │  Transcription pipeline with real whisper.cpp + fixtures
        ┌┴────────────┴┐
        │   Unit Tests   │  Individual components with mocked audio layer
        └────────────────┘
```

### Unit Tests

- **Audio layer**: Abstract audio capture behind a `Recorder` interface. In tests, use a `MockRecorder` that produces synthetic audio data (sine waves, silence, or pre-loaded PCM buffers).
- **Session manager**: Test file naming, metadata generation, listing, and search against a temp directory.
- **Config**: Test config file parsing, precedence (CLI > config > defaults), and validation.
- **Device manager**: Mock sounddevice's `query_devices()` to test device enumeration and loopback detection logic.

### Integration Tests

- **Transcription**: Use small (~2-5 second) pre-recorded WAV fixture files with known content. Verify that whisper.cpp produces expected transcript text (fuzzy match, not exact).
- **Record + Transcribe pipeline**: Use `MockRecorder` to produce a WAV file, then run real transcription, verify end-to-end output.
- **CLI**: Invoke tscribe subcommands as subprocesses, verify exit codes and output.

### Test Fixtures

- Ship small WAV files in `tests/fixtures/`:
  - `silence.wav` — 2 seconds of silence
  - `speech-en.wav` — 3-5 seconds of clear English speech with known transcript
  - `speech-noisy.wav` — speech with background noise
- Keep fixture files small (<500KB total) to avoid bloating the repository.

### CI Considerations

- faster-whisper is installed as a Python dependency — no binary setup needed in CI.
- Use the `tiny` model in integration tests for speed; cache the model between CI runs.
- Audio devices are not available in CI — all recording tests use the mock layer.
- Integration tests that download models can be marked with a `@pytest.mark.slow` marker and optionally skipped in fast CI runs.

### Coverage

- Use `pytest-cov` for coverage analysis
- Target: **90%+ line coverage** for core modules (recorder, transcriber, session manager)
- Coverage report integrated into CI — fail the build if coverage drops below threshold
- Coverage gaps should map to identified areas (e.g., platform-specific loopback code that can only run on that OS)

### Running Tests from CLI

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=tscribe --cov-report=term-missing

# Run fast tests only (skip integration)
uv run pytest -m "not slow"

# Run specific test module
uv run pytest tests/test_recorder.py
```

---

## Error Handling

- **No audio device found**: Clear error message listing available devices, suggest `tscribe devices`
- **No loopback source (macOS)**: Guide user to install BlackHole with a link
- **Model not downloaded**: Auto-download the requested model on first use
- **Recording interrupted unexpectedly**: Attempt to salvage the WAV file (update header with bytes written so far)
- **Transcription fails**: Report the error, keep the audio file, allow retry

---

## Future Considerations (Post-V1)

These are documented for planning purposes and are explicitly out of scope for V1.

### V1.1 — Notes & Annotations

- Add `tscribe note "text"` command to attach timestamped notes to the active recording session
- Notes stored in a sidecar `.notes.json` file
- TUI overlay during recording showing elapsed time and notes

### V1.2 — Speaker Diarization

- Integrate pyannote-audio or similar for speaker identification
- Add `--diarize` flag to transcribe command
- Output format extended with speaker labels

### Future Ideas

- Export to markdown with embedded timestamps
- Web-based viewer for browsing recordings and transcripts
- Real-time streaming transcription
- Audio bookmarks / chapter markers
- Integration with calendar for auto-naming meeting recordings
