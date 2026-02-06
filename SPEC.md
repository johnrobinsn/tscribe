# tscribe — Specification

## Overview

**tscribe** is a cross-platform CLI tool for recording audio from any source (microphone or system audio), transcribing it to text using faster-whisper, and managing the resulting recordings and transcripts. It is designed to run on modest hardware — no GPU required, but GPU acceleration is supported.

### Primary Use Cases

- Capturing both sides of video calls (Teams, Zoom, Meet) for searchable transcripts
- Personal voice memos and audio note-taking
- Transcribing existing audio files (podcasts, lectures, downloaded recordings)
- Transcribing audio from URLs (YouTube, etc.) via yt-dlp
- Generating subtitles (SRT/VTT) from any audio or video source

---

## V1 Scope

V1 focuses on core functionality: **record, transcribe, play, manage, search**.

### In Scope (V1)

- Audio recording from microphone and system audio (loopback)
- Transcription via faster-whisper (CPU by default, GPU optional)
- Transcription of external audio files, URLs, and previous recordings
- Progress bar with ETA during transcription
- Playback with progress bar
- Open transcripts in default editor, dump to stdout
- Listing and searching past recordings and transcripts
- Auto-transcribe after recording (configurable)
- URL transcription via yt-dlp (lazy-loaded)
- REF-based navigation (HEAD, HEAD~N, session stems)

### Deferred (Post-V1)

- **V1.1**: `--both` flag to record microphone and system audio simultaneously
- **V1.1**: Timestamped notes/annotations during recording
- **V1.2**: Speaker diarization (identifying who said what)
- Real-time streaming transcription during recording

---

## Platform Support

| Platform | Microphone | System Audio (Loopback) | Notes |
|----------|-----------|------------------------|-------|
| Linux    | Yes       | Yes                    | PipeWire native via `pw-record` |
| Windows  | Yes       | Yes                    | WASAPI loopback via PyAudioWPatch |
| macOS    | Yes       | Yes (requires setup)   | Requires BlackHole or similar virtual audio device |

macOS system audio capture requires the user to install a third-party virtual audio driver (e.g., BlackHole). tscribe detects available loopback sources and guides the user if none are found.

---

## Architecture

### Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.10+ | Rapid development, rich audio ecosystem, cross-platform |
| Package Manager | uv | Fast, modern, handles venvs and lockfiles |
| Audio Capture (Linux) | PipeWire (`pw-record`) | Native loopback/monitor capture, no PortAudio needed |
| Audio Capture (Windows loopback) | PyAudioWPatch | Native WASAPI loopback support, Windows-only |
| Audio Capture (macOS/Windows mic) | sounddevice (PortAudio) | Well-maintained, clean API, numpy integration |
| Transcription | faster-whisper (CTranslate2) | Pre-built wheels on all platforms, fast CPU inference, auto model download |
| URL Audio | yt-dlp | Supports YouTube and many other sites, lazy-loaded |
| CLI Framework | click | Git-style subcommands, rich option handling |
| Audio Format | WAV | Uncompressed during recording for zero overhead |

### System Diagram

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│ Audio Source │────▶│  pw-record (Linux)│────▶│  .wav file   │
│ (mic/loop)  │     │  sounddevice (other)    │              │
└─────────────┘     └──────────────────┘     └──────┬───────┘
                                                      │
                    ┌──────────────────┐               │
                    │  yt-dlp (URLs)   │───────────────▶│
                    └──────────────────┘               │
                                                      ▼
                                             ┌───────────────┐
                                             │ faster-whisper │
                                             │  (Python API)  │
                                             └───────┬───────┘
                                                      │
                                                      ▼
                                             ┌───────────────┐
                                             │  Transcript    │
                                             │ (.txt .json    │
                                             │  .srt .vtt)    │
                                             └───────────────┘
```

### Component Responsibilities

- **Recorder**: Abstract `Recorder` interface with platform backends (`PipewireRecorder` for Linux, `WasapiRecorder` for Windows loopback, `SounddeviceRecorder` for macOS/Windows mic, `MockRecorder` for tests). Manages recording lifecycle.
- **Transcriber**: Uses faster-whisper Python API for transcription, writes transcript files. Supports `progress_callback` for UI updates.
- **Session Manager**: Handles file naming, metadata, listing, searching, and importing external files.
- **Device Manager**: Enumerates audio devices — PipeWire nodes on Linux (`pw-dump`), sounddevice elsewhere.

---

## CLI Design

Git-style subcommand interface:

```
tscribe <command> [options]
```

### Commands

#### `tscribe record`

Start a recording session. Defaults to system audio (loopback) on Linux and Windows.

```
tscribe record [OPTIONS]

Options:
  --device, -d DEVICE     Audio device name or index
  --loopback/--mic        Record system audio (default) or microphone
  --output, -o PATH       Output file path (default: auto-generated in recordings dir)
  --no-transcribe         Don't auto-transcribe after recording stops
  --sample-rate RATE      Sample rate in Hz (default: 16000)
  --channels N            Number of channels (default: 1, mono)
```

Recording is stopped with Ctrl+C (SIGINT), which triggers a clean shutdown: finalize the WAV file, then optionally start transcription.

Live level meter during recording:

```
● REC 00:05  ▁▂▃▅▇█▆▃▁▁▂▅▇█▇▅▃▂▁▁
```

#### `tscribe transcribe`

Transcribe an audio file, URL, or previous recording.

```
tscribe transcribe [SOURCE] [OPTIONS]

SOURCE defaults to HEAD (most recent recording). Can be:
  - A recording ref: HEAD, HEAD~N, or a session stem
  - A local file path
  - A URL (YouTube, etc.) — uses yt-dlp

Options:
  --model, -m MODEL       Whisper model size (default: base)
                          Choices: tiny, base, small, medium, large
  --language LANG         Language code (default: auto-detect)
  --output, -o PATH       Output file path (default: alongside input file)
  --format FORMAT         Output format: txt, json, srt, vtt, all (default: txt,json)
  --gpu                   Use GPU acceleration if available
```

Progress bar with ETA during transcription:

```
  ⟳ 02:30/10:00  |████████░░░░░░░░░░░░░░░░░░░░░░|  ETA 01:30
```

URL transcription downloads audio via yt-dlp, imports it into the recordings directory, and transcribes it. The result appears in `tscribe list` like any other recording.

#### `tscribe play`

Play a recording with progress bar.

```
tscribe play [REF]

REF defaults to HEAD. Cross-platform player detection: pw-play, aplay, afplay, ffplay.
```

Progress bar during playback:

```
  ▶ 01:30/05:00  |███████████████░░░░░░░░░░░░░░░|
```

#### `tscribe open`

Open a transcript in the OS default program.

```
tscribe open [REF] [-f FORMAT]

FORMAT: txt (default), json, srt, vtt, wav
```

#### `tscribe dump`

Print a transcript to stdout for piping.

```
tscribe dump [REF] [-f FORMAT]
```

#### `tscribe path`

Print the absolute file path of a recording artifact. Useful for scripting.

```
tscribe path [REF] [-f FORMAT]

FORMAT: txt (default), wav, json, srt, vtt, meta
```

#### `tscribe list`

List past recordings and their transcription status.

```
tscribe list [OPTIONS]

Options:
  --limit, -n N           Number of entries to show (default: 20)
  --search, -s QUERY      Filter by transcript text
  --sort FIELD            Sort by: date, duration, name (default: date)
  --no-header             Omit table header
```

Output includes REF, date with day-of-week, duration, transcription status, and source:

```
REF     Date                      Dur Tx  Source
------------------------------------------------------
HEAD    2025-01-15-143022 We  00:05:30  Y  loopback
HEAD~1  2025-01-14-091500 Tu  00:03:15  Y  https://youtu.be/dQw4w9W
HEAD~2  2025-01-13-100000 Mo  00:10:00  N  microphone
```

URLs are clickable hyperlinks (OSC 8) in supported terminals.

#### `tscribe search`

Search transcript text across all recordings.

```
tscribe search <QUERY> [OPTIONS]

Options:
  --limit, -n N           Number of results (default: 20)
  --sort FIELD            Sort by: date, duration (default: date)
```

Output shows matching lines with session context:

```
── 2025-01-15-143022 We (HEAD) ──
discussed the action items for next week

── 2025-01-14-091500 Tu (HEAD~1) ──
review action items from Monday's standup

2 matches found.
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
```

---

## REF Navigation

Commands that accept a REF argument support:

- `HEAD` — most recent recording (default)
- `HEAD~N` — Nth previous recording
- Session stem — e.g., `2025-01-15-143022`

REFs are computed from date-sorted order, so they remain correct regardless of display sort options.

---

## Storage Layout

All platforms use `~/.tscribe/`:

```
~/.tscribe/
├── config.toml                   # User configuration
├── recordings/
│   ├── 2025-01-15-143022.wav     # Audio recording
│   ├── 2025-01-15-143022.txt     # Plain text transcript
│   ├── 2025-01-15-143022.json    # Structured transcript (timestamps, segments)
│   ├── 2025-01-15-143022.meta    # Recording metadata
│   ├── 2025-01-14-091500.wav
│   └── ...
```

Override with `TSCRIBE_DATA_DIR` environment variable or `storage.data_dir` config key.

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
  "source_url": null,
  "transcribed": true,
  "model": "base",
  "original_path": null
}
```

`source_type` is one of: `loopback`, `microphone`, `url`, `file`. For URL sources, `source_url` stores the original URL.

---

## Transcription Details

### faster-whisper Integration

- **Python API**: Uses faster-whisper (CTranslate2 backend) directly as a Python library — no subprocess or binary management needed.
- **Auto-download**: Models are downloaded automatically from Hugging Face on first use of each model size.
- **Cross-platform**: Pre-built wheels available for Linux, macOS, and Windows via pip/uv.
- **Progress callback**: `Transcriber.transcribe()` accepts an optional `progress_callback(segment_end, audio_duration)` called after each segment.

### Default Model

- **Default**: `base` (~74MB, ~10x realtime on modern CPU)
- **Recommended for quality**: `small` (~244MB, ~3-4x realtime on CPU)
- Configurable via `tscribe config transcription.model <size>` or `--model` flag

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

- **srt/vtt**: Standard subtitle formats for adding captions to videos

Default output: both `txt` and `json`.

---

## Configuration

TOML format at `~/.tscribe/config.toml`:

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
data_dir = ""                # empty = ~/.tscribe
```

Configuration priority: CLI flags > config file > defaults.

---

## Audio Recording Details

### Sample Rate

Default: 16000 Hz (16 kHz). This is the native input rate for Whisper models. Recording at higher rates is supported but audio will be resampled to 16 kHz before transcription. PipeWire loopback defaults to 48 kHz to match the system sink.

### Recording Lifecycle

1. User runs `tscribe record`
2. Recorder backend opens audio stream (PipeWire on Linux, sounddevice elsewhere)
3. Audio is written to a WAV file
4. Live level meter displays audio activity
5. User presses Ctrl+C to stop
6. WAV file is finalized
7. If auto-transcribe is enabled, transcription starts immediately with progress bar
8. Metadata file is written

### Signal Handling

Ctrl+C (SIGINT) triggers graceful shutdown:
- Stop the audio stream
- Finalize the WAV file header
- Proceed to transcription if enabled
- A second Ctrl+C forces immediate exit

### System Audio Capture (Loopback)

Platform-specific implementation behind a common `Recorder` interface:

- **Linux**: PipeWire native via `pw-record` with `-P '{ stream.capture.sink=true }'` for monitor capture. Level metering polls the growing WAV file.
- **Windows**: WASAPI loopback via PyAudioWPatch (auto-installed on Windows). Captures from any output device natively, no virtual audio driver needed. Falls back to sounddevice for microphone recording.
- **macOS**: Requires user to install BlackHole or similar. tscribe detects available virtual audio devices.

---

## Dependencies

### Python Dependencies

| Package | Purpose |
|---------|---------|
| click | CLI framework |
| sounddevice | Audio capture on macOS/Windows mic (PortAudio bindings) |
| PyAudioWPatch | WASAPI loopback on Windows (auto-installed, Windows-only) |
| numpy | Audio buffer handling (required by sounddevice) |
| faster-whisper | Speech-to-text transcription (CTranslate2 backend) |
| yt-dlp | URL audio download (lazy-loaded, only when transcribing URLs) |
| tomli-w | TOML config writing |
| tomli | TOML config reading (Python < 3.11) |

### System Dependencies

| Dependency | Required | Notes |
|-----------|----------|-------|
| PipeWire (Linux) | For recording | `pw-record` for capture, `pw-play` for playback |
| PortAudio (macOS/Windows) | For recording | Bundled with sounddevice wheels |
| FFmpeg | For URL transcription | Required by yt-dlp for audio extraction |
| BlackHole (macOS) | For loopback only | User-installed |

---

## Testing Strategy

### Test Pyramid

```
          ┌──────────┐
          │   E2E    │  Full CLI invocations with fixture audio files
         ┌┴──────────┴┐
         │ Integration │  Transcription pipeline with real faster-whisper
        ┌┴────────────┴┐
        │   Unit Tests   │  Individual components with mocked audio layer
        └────────────────┘
```

### Unit Tests

- **Audio layer**: Abstract audio capture behind a `Recorder` interface. In tests, use a `MockRecorder` that produces synthetic audio data.
- **Session manager**: Test file naming, metadata generation, listing, searching, and importing against a temp directory.
- **Config**: Test config file parsing, precedence (CLI > config > defaults), and validation.
- **Device manager**: Mock sounddevice's `query_devices()` and PipeWire's `pw-dump` output to test device enumeration.
- **Transcriber**: Mock `WhisperModel` with injected `_model` attribute. Mock info object must set `duration` (float).
- **CLI**: Use Click's `CliRunner` with mocked backends.

### Integration Tests

- **Transcription**: Use small pre-recorded WAV fixture files with known content. Verify that faster-whisper produces expected transcript text (fuzzy match).
- Integration tests that download models are marked with `@pytest.mark.slow` and optionally skipped.

### CI Considerations

- faster-whisper is installed as a Python dependency — no binary setup needed in CI.
- Use the `tiny` model in integration tests for speed; cache the model between CI runs.
- Audio devices are not available in CI — all recording tests use the mock layer.
- sounddevice imports are lazy to avoid PortAudio load failures in CI environments.

---

## Error Handling

- **No audio device found**: Clear error message listing available devices, suggest `tscribe devices`
- **No loopback source (macOS)**: Guide user to install BlackHole with a link
- **Model not downloaded**: Auto-download the requested model on first use
- **Recording interrupted unexpectedly**: Attempt to salvage the WAV file (update header with bytes written so far)
- **Transcription fails**: Report the error, keep the audio file, allow retry
- **yt-dlp not installed**: Helpful error message with install instructions (should not happen with `uv sync`)

---

## Future Considerations (Post-V1)

These are documented for planning purposes and are explicitly out of scope for V1.

### V1.1 — Dual Recording & Notes

- `--both` flag to record microphone and system audio simultaneously (mixed or separate tracks)
- `tscribe note "text"` command to attach timestamped notes to the active recording session
- Notes stored in a sidecar `.notes.json` file

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
