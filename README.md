# tscribe
![macOS](https://img.shields.io/badge/-macOS-000000?style=flat-square&logo=apple&logoColor=white)
![Linux](https://img.shields.io/badge/-Linux-FCC624?style=flat-square&logo=linux&logoColor=black)
![Windows](https://img.shields.io/badge/-Windows-0078D6?style=flat-square&logo=windows&logoColor=white)
![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![CLI](https://img.shields.io/badge/-CLI-000000?style=flat-square&logo=gnu-bash&logoColor=white)

Cross-platform CLI tool for recording audio and transcribing it to text using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Runs on modest hardware — no GPU required, but GPU acceleration is supported.

https://private-user-images.githubusercontent.com/1709257/546508788-dcd6ee47-d2bd-429c-be38-3ab8e5f5b243.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzA0NDYwMDAsIm5iZiI6MTc3MDQ0NTcwMCwicGF0aCI6Ii8xNzA5MjU3LzU0NjUwODc4OC1kY2Q2ZWU0Ny1kMmJkLTQyOWMtYmUzOC0zYWI4ZTVmNWIyNDMubXA0P1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI2MDIwNyUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNjAyMDdUMDYyODIwWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9ZWM0OTAzMjdhMDkyNzNmNDA4Y2Y0NDJjOTczNDFhMDFmMjk1ZDNkNmZjNDkyMjEzYzUyZDMxYjQ5OTM1Y2QyNCZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.jjZsx6_5HnEaFd-0CMg7e9vtRPOqNA8OaGR3SToiLns

## Use Cases

- Capture both sides of video calls (Teams, Zoom, Meet) for searchable transcripts
- Voice memos and audio note-taking
- Transcribe podcasts, lectures, or downloaded audio
- Generate subtitles (SRT/VTT) from any audio or video source

## Features

- **Record** system audio (loopback) or microphone with live level meter
- **Transcribe** recordings, WAV files, or URLs (YouTube, etc.) via faster-whisper
- **Auto-transcribe** after recording stops (configurable)
- **Play** recordings with progress bar
- **Open** transcripts in your default editor/viewer
- **Dump** transcripts to stdout for piping
- **Search** past transcripts by keyword
- **Cross-platform**: Linux (PipeWire), macOS, Windows
- Output formats: plain text, JSON (timestamped), SRT, WebVTT

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Linux: PipeWire (for system audio capture)
- macOS: [BlackHole](https://github.com/ExistentialAudio/BlackHole) (for system audio capture). Optional: [switchaudio-osx](https://github.com/deweller/switchaudio-osx) for auto-switching.
- Windows: WASAPI (built-in)

## Installation

```bash
git clone https://github.com/johnrobinsn/tscribe.git && cd tscribe
uv sync
```

Models are downloaded automatically on first transcription (~244MB for the default `small` model).

## Quick Start

```bash
# Record system audio (Ctrl+C to stop, auto-transcribes by default)
tscribe record

# Record from microphone instead
tscribe record --mic

# Transcribe a YouTube video
tscribe transcribe "https://youtu.be/dQw4w9W"

# Play back the last recording
tscribe play

# Open the transcript in your default editor
tscribe open

# Print transcript to stdout
tscribe dump
```

## Commands

| Command | Description |
|---------|-------------|
| `tscribe record` | Record system audio or microphone |
| `tscribe play [REF]` | Play a recording |
| `tscribe open [REF]` | Open transcript in default program |
| `tscribe dump [REF]` | Print transcript to stdout |
| `tscribe path [REF]` | Print file path of a recording artifact |
| `tscribe transcribe [SOURCE]` | Transcribe a file, URL, or recording ref |
| `tscribe list` | List past recordings |
| `tscribe search <query>` | Search transcript text |
| `tscribe devices` | List audio input devices |
| `tscribe config` | View or set configuration |

REF can be `HEAD` (most recent, default), `HEAD~N` (Nth previous), or a session stem (e.g. `2025-01-15-143022`).

### Record

```bash
tscribe record                        # System audio (default on Linux/Windows)
tscribe record --mic                  # Record from microphone
tscribe record --device 56            # Specific device (see tscribe devices)
tscribe record --no-transcribe        # Skip auto-transcription
tscribe record -o meeting.wav         # Custom output path
```

While recording, a live level meter shows audio activity:

```
● REC 00:05  ▁▂▃▅▇█▆▃▁▁▂▅▇█▇▅▃▂▁▁
```

### Play

```bash
tscribe play                          # Play most recent recording
tscribe play HEAD~1                   # Play previous recording
tscribe play 2025-01-15-143022        # Play by session ID
```

### Open & Dump

```bash
tscribe open                          # Open transcript in default editor
tscribe open -f json                  # Open JSON transcript
tscribe open -f wav                   # Open the audio file itself
tscribe dump                          # Print transcript to stdout
tscribe dump -f json                  # Print JSON to stdout
tscribe dump HEAD~1                   # Print previous transcript
tscribe dump | grep "action items"    # Pipe to other tools
```

### Path

```bash
tscribe path                          # Path to most recent transcript
tscribe path -f wav                   # Path to audio file
tscribe path HEAD~1 -f json           # Previous session's JSON
tscribe path -f meta                  # Metadata file
cat $(tscribe path)                   # Use in scripts
```

### Transcribe

```bash
tscribe transcribe                              # Re-transcribe most recent recording
tscribe transcribe --model small                # Re-transcribe with a better model
tscribe transcribe HEAD~2                       # Re-transcribe a previous recording
tscribe transcribe audio.wav                    # Local file (base model, txt+json)
tscribe transcribe "https://youtu.be/dQw4w9W"  # YouTube or other URL
tscribe transcribe audio.wav --format all       # Output txt, json, srt, vtt
tscribe transcribe audio.wav --language en      # Force language
tscribe transcribe audio.wav --gpu              # Use GPU acceleration
```

SOURCE defaults to HEAD (most recent recording). Accepts file paths, URLs, or recording refs (HEAD, HEAD~N, session stems).

URL transcription uses [yt-dlp](https://github.com/yt-dlp/yt-dlp) (installed with tscribe) to download audio, imports it into the recordings directory, and transcribes it. The result appears in `tscribe list` like any other recording.

Use `--format srt` or `--format vtt` to generate subtitle files for adding captions to videos.

A progress bar with ETA is shown during transcription:

```
  ⟳ 02:30/10:00  |████████░░░░░░░░░░░░░░░░░░░░░░|  ETA 01:30
```

### List

```bash
tscribe list                          # Show recent recordings
tscribe list --search "meeting"       # Filter by transcript text
tscribe list --sort duration          # Sort by duration
tscribe list -n 50                    # Show 50 entries
```

Output includes REF, date with day-of-week, duration, transcription status, and source:

```
REF     Date                      Dur Tx  Source
------------------------------------------------------
HEAD    2025-01-15-143022 We  00:05:30  Y  loopback
HEAD~1  2025-01-14-091500 Tu  00:03:15  Y  https://youtu.be/dQw4w9W
HEAD~2  2025-01-13-100000 Mo  00:10:00  N  microphone
```

URLs are clickable hyperlinks in supported terminals.

### Search

```bash
tscribe search "action items"         # Search all transcripts
tscribe search meeting -n 50          # Limit results
tscribe search budget --sort date     # Sort by date (default)
```

Shows matching lines with session context:

```
── 2025-01-15-143022 We (HEAD) ──
discussed the action items for next week

── 2025-01-14-091500 Tu (HEAD~1) ──
review action items from Monday's standup

2 matches found.
```

### Devices

```bash
tscribe devices                       # List all audio devices
tscribe devices --loopback            # Show only loopback/monitor sources
```

### Configuration

```bash
tscribe config                                  # Show all settings
tscribe config recording.sample_rate            # Get a value
tscribe config recording.auto_transcribe false  # Set a value
tscribe config transcription.model small        # Change default model
```

Config is stored at `~/.tscribe/config.toml`. Available keys and defaults:

| Key | Default | Description |
|-----|---------|-------------|
| `recording.sample_rate` | `16000` | Sample rate in Hz |
| `recording.channels` | `1` | Number of channels (mono) |
| `recording.default_device` | `""` | Default audio device (empty = system default) |
| `recording.auto_transcribe` | `true` | Auto-transcribe after recording |
| `transcription.model` | `"small"` | Whisper model size (tiny/base/small/medium/large) |
| `transcription.language` | `"auto"` | Language code or auto-detect |
| `transcription.output_formats` | `["txt","json"]` | Output formats (txt, json, srt, vtt) |
| `transcription.gpu` | `false` | Use GPU acceleration |
| `storage.data_dir` | `""` | Override data directory (empty = `~/.tscribe`) |

## Whisper Models

| Model | Size | Speed (CPU) | Quality |
|-------|------|-------------|---------|
| tiny | ~39MB | ~30x realtime | Low |
| base | ~74MB | ~10x realtime | Good |
| small | ~244MB | ~3-4x realtime | Better |
| medium | ~769MB | ~1x realtime | High |
| large | ~1.5GB | Slower than realtime | Best |

Default: `small` — a good balance of speed and accuracy. If transcriptions have too many errors, try `medium` for better quality at the cost of slower processing. For faster but lower-quality results, try `base` or `tiny`. You can also override per-transcription without changing the default:

```bash
tscribe transcribe --model small
```

Models are downloaded automatically from Hugging Face on first use.

## System Audio (Loopback)

By default, `tscribe record` captures system audio (what you hear through your speakers/headphones). Use `--mic` to record from the microphone instead.

| Platform | How It Works |
|----------|--------------|
| Linux | PipeWire native via `pw-record` with monitor capture |
| Windows | WASAPI loopback via [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch) (auto-installed) |
| macOS | Requires [BlackHole](https://github.com/ExistentialAudio/BlackHole) + Multi-Output Device. See [macOS setup guide](docs/macos-system-audio.md). |

On macOS, tscribe will detect your setup and guide you: if BlackHole is missing it warns and falls back to the microphone, and if [switchaudio-osx](https://github.com/deweller/switchaudio-osx) is installed (`brew install switchaudio-osx`) it can automatically switch to your Multi-Output Device when recording and restore your previous output when done.

Run `tscribe devices --loopback` to see available loopback sources. Use `--device <id>` to record from a specific microphone or input device.

## Storage

Recordings are stored in `~/.tscribe/recordings/` on all platforms. Override with the `TSCRIBE_DATA_DIR` environment variable or `storage.data_dir` config key.

Each recording produces:
- `.wav` — audio file
- `.txt` — plain text transcript
- `.json` — timestamped transcript with segments
- `.meta` — recording metadata (device, duration, etc.)

To delete old recordings, simply remove the corresponding files from `~/.tscribe/recordings/`.

## Development

```bash
# Install with dev dependencies
uv sync

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=tscribe --cov-report=term-missing

# Skip slow integration tests (require model download)
uv run pytest -m "not slow"
```

## Architecture

```
src/tscribe/
├── cli.py                # Click CLI with subcommands
├── config.py             # TOML configuration management
├── paths.py              # Cross-platform path resolution
├── devices.py            # Audio device enumeration (sounddevice)
├── pipewire_devices.py   # PipeWire device enumeration (Linux)
├── session.py            # Recording session & file management
├── transcriber.py        # faster-whisper transcription
└── recorder/
    ├── base.py                  # Abstract Recorder interface
    ├── sounddevice_recorder.py  # sounddevice capture (macOS/Windows mic)
    ├── wasapi_recorder.py       # WASAPI loopback capture (Windows)
    ├── pipewire_recorder.py     # PipeWire capture (Linux)
    └── mock_recorder.py         # Test mock
```

## License

MIT
