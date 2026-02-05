# tscribe

Cross-platform CLI tool for recording audio and transcribing it to text using [whisper.cpp](https://github.com/ggerganov/whisper.cpp). Runs on modest hardware — no GPU required.

## Features

- **Record** from microphone or system audio (loopback)
- **Transcribe** recordings or any WAV file via whisper.cpp
- **Auto-transcribe** after recording stops (configurable)
- **Search** past transcripts by keyword
- **Cross-platform**: Linux, macOS, Windows
- Output formats: plain text, JSON (timestamped), SRT, WebVTT

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- PortAudio (usually installed with `sounddevice` wheels)

## Installation

```bash
git clone <repo-url> && cd tscribe
uv sync
```

## Quick Start

```bash
# Download whisper.cpp and the base model (~74MB)
tscribe setup

# Record audio (Ctrl+C to stop, auto-transcribes by default)
tscribe record

# Transcribe an existing file
tscribe transcribe meeting.wav

# List recordings and search transcripts
tscribe list
tscribe list --search "action items"
```

## Commands

| Command | Description |
|---------|-------------|
| `tscribe record` | Record audio from mic or system audio |
| `tscribe transcribe <file>` | Transcribe a WAV file |
| `tscribe list` | List past recordings |
| `tscribe devices` | List audio input devices |
| `tscribe config` | View or set configuration |
| `tscribe setup` | Download whisper.cpp binary and models |

### Record

```bash
tscribe record                        # Default mic, auto-transcribe
tscribe record --loopback             # Record system audio
tscribe record --device 3             # Specific device (see tscribe devices)
tscribe record --no-transcribe        # Skip auto-transcription
tscribe record -o meeting.wav         # Custom output path
```

### Transcribe

```bash
tscribe transcribe audio.wav                    # Default (base model, txt+json)
tscribe transcribe audio.wav --model small      # Higher quality model
tscribe transcribe audio.wav --format all       # Output txt, json, srt, vtt
tscribe transcribe audio.wav --language en      # Force language
```

### List & Search

```bash
tscribe list                          # Show recent recordings
tscribe list --search "meeting"       # Search transcript text
tscribe list --sort duration          # Sort by duration
tscribe list -n 50                    # Show 50 entries
```

### Configuration

```bash
tscribe config                                  # Show all settings
tscribe config recording.sample_rate            # Get a value
tscribe config recording.auto_transcribe false  # Set a value
tscribe config transcription.model small        # Change default model
```

## Whisper Models

| Model | Size | Speed (CPU) | Quality |
|-------|------|-------------|---------|
| tiny | ~39MB | ~30x realtime | Low |
| base | ~74MB | ~10x realtime | Good |
| small | ~244MB | ~3-4x realtime | Better |
| medium | ~769MB | ~1x realtime | High |
| large | ~1.5GB | Slower than realtime | Best |

Default: `base`. Change with `tscribe config transcription.model <size>`.

## System Audio (Loopback)

| Platform | Setup |
|----------|-------|
| Linux | Works automatically via PulseAudio/PipeWire monitor sources |
| Windows | Works via WASAPI loopback |
| macOS | Requires [BlackHole](https://github.com/ExistentialAudio/BlackHole): `brew install blackhole-2ch` |

Run `tscribe devices --loopback` to see available loopback sources.

## Storage

Recordings are stored in your platform's data directory:

- Linux: `~/.local/share/tscribe/recordings/`
- macOS: `~/Library/Application Support/tscribe/recordings/`
- Windows: `%LOCALAPPDATA%/tscribe/recordings/`

Each recording produces:
- `.wav` — audio file
- `.txt` — plain text transcript
- `.json` — timestamped transcript with segments
- `.meta` — recording metadata (device, duration, etc.)

## Development

```bash
# Install with dev dependencies
uv sync

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=tscribe --cov-report=term-missing

# Skip slow integration tests (require whisper.cpp)
uv run pytest -m "not slow"
```

## Architecture

```
src/tscribe/
├── cli.py           # Click CLI with subcommands
├── config.py        # TOML configuration management
├── paths.py         # Cross-platform path resolution
├── devices.py       # Audio device enumeration
├── session.py       # Recording session & file management
├── transcriber.py   # whisper.cpp subprocess integration
├── whisper_manager.py  # Binary & model download management
└── recorder/
    ├── base.py      # Abstract Recorder interface
    ├── sounddevice_recorder.py  # Real audio capture
    └── mock_recorder.py         # Test mock
```

## License

MIT
