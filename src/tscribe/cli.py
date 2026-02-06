"""CLI entry point for tscribe."""

import collections
import signal
import sys
import threading
import time

import click

from tscribe import __version__


def _default_loopback() -> bool:
    """Whether to default to loopback (system audio) recording."""
    if sys.platform == "linux":
        from tscribe.pipewire_devices import is_pipewire_available

        return is_pipewire_available()
    if sys.platform == "win32":
        return True
    if sys.platform == "darwin":
        try:
            from tscribe.devices import list_devices

            return any(d.is_loopback for d in list_devices(loopback_only=True))
        except Exception:
            return False
    return False


def _create_recorder(config):
    """Create the appropriate recorder based on platform and availability.

    On Linux, prefers PipeWire when available (especially for loopback).
    Falls back to sounddevice otherwise.
    """
    if sys.platform == "linux":
        from tscribe.pipewire_devices import is_pipewire_available

        if is_pipewire_available():
            from tscribe.recorder.pipewire_recorder import PipewireRecorder

            return PipewireRecorder()

    from tscribe.recorder.sounddevice_recorder import SounddeviceRecorder

    return SounddeviceRecorder()


def _auto_transcribe(cfg, wav_path, stem, session_mgr):
    """Run transcription after recording. Handles errors gracefully."""
    from tscribe.transcriber import Transcriber

    model_name = cfg.transcription.model

    click.echo(f"Transcribing with model '{model_name}'...")
    try:
        transcriber = Transcriber(model_name=model_name)
        result = transcriber.transcribe(
            wav_path,
            language=cfg.transcription.language,
            output_formats=cfg.transcription.output_formats,
        )
        session_mgr.update_metadata(stem, transcribed=True, model=model_name)
        click.echo(f"Transcription complete: {len(result.segments)} segments.")
    except Exception as e:
        click.echo(f"Transcription failed: {e}")


@click.group()
@click.version_option(version=__version__, prog_name="tscribe")
def main():
    """Record audio and transcribe with whisper."""


@main.command()
@click.option("--device", "-d", default=None, help="Audio device name or index.")
@click.option("--loopback/--mic", default=None,
              help="Record system audio (default) or microphone.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path.")
@click.option("--no-transcribe", is_flag=True, help="Don't auto-transcribe after recording.")
@click.option("--sample-rate", type=int, default=None, help="Sample rate in Hz.")
@click.option("--channels", type=int, default=None, help="Number of channels.")
def record(device, loopback, output, no_transcribe, sample_rate, channels):
    """Record audio from system audio (default) or microphone."""
    from pathlib import Path

    from tscribe.config import TscribeConfig
    from tscribe.paths import ensure_dirs, get_config_path, get_data_dir, get_recordings_dir
    from tscribe.recorder import RecordingConfig
    from tscribe.session import SessionManager

    cfg = TscribeConfig.load(get_config_path(get_data_dir()))
    data_dir = get_data_dir(cfg.storage.data_dir)
    ensure_dirs(data_dir)

    session_mgr = SessionManager(get_recordings_dir(data_dir))
    stem = session_mgr.generate_session_stem()

    if output:
        wav_path = Path(output)
    else:
        wav_path = session_mgr.recordings_dir / f"{stem}.wav"

    if loopback is None:
        loopback = False if device else _default_loopback()

    rec_config = RecordingConfig(
        sample_rate=sample_rate or cfg.recording.sample_rate,
        channels=channels or cfg.recording.channels,
        device=int(device) if device and device.isdigit() else device,
        loopback=loopback,
    )

    recorder = _create_recorder(cfg)
    stop_event = threading.Event()

    interrupt_count = 0
    original_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(sig, frame):
        nonlocal interrupt_count
        interrupt_count += 1
        if interrupt_count >= 2:
            click.echo("\nForced exit.")
            sys.exit(1)
        click.echo("\nStopping recording...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        recorder.start(wav_path, rec_config)
        source = "system audio" if loopback else "microphone"
        click.echo(f"Recording {source} to {wav_path} (Ctrl+C to stop)...")

        blocks = " ▁▂▃▄▅▆▇█"
        history = collections.deque([0.0] * 20, maxlen=20)

        while not stop_event.is_set():
            elapsed = recorder.elapsed_seconds
            minutes, secs = divmod(int(elapsed), 60)
            lvl = min(recorder.level ** 0.4, 1.0)
            history.append(lvl)
            meter = "".join(blocks[int(v * 8)] for v in history)
            led = click.style("●", fg="red", blink=True)
            click.echo(f"\r  {led} REC {minutes:02d}:{secs:02d}  {meter}", nl=False)
            stop_event.wait(0.25)

        result = recorder.stop()
        click.echo(f"\nRecording saved: {result.path} ({result.duration_seconds:.1f}s)")

        meta = session_mgr.create_metadata(result)
        session_mgr.write_metadata(stem, meta)

        if not no_transcribe and cfg.recording.auto_transcribe:
            _auto_transcribe(cfg, wav_path, stem, session_mgr)

    finally:
        signal.signal(signal.SIGINT, original_handler)


def _download_audio(url, output_dir):
    """Download audio from a URL using yt-dlp. Returns path to WAV file."""
    from pathlib import Path

    try:
        import yt_dlp
    except ImportError:
        raise click.ClickException(
            "yt-dlp is required for URL transcription. "
            "Install it with: uv pip install yt-dlp"
        )

    output_template = str(Path(output_dir) / "%(title)s.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    click.echo(f"Downloading audio from URL...")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    title = info.get("title", "audio")
    # yt-dlp replaces special chars in filenames; use the prepared filename
    wav_path = Path(ydl.prepare_filename(info)).with_suffix(".wav")
    if not wav_path.exists():
        raise click.ClickException(f"Download failed: expected {wav_path.name}")

    click.echo(f"Downloaded: {wav_path.name}")
    return wav_path


@main.command()
@click.argument("source")
@click.option("--model", "-m", default=None, help="Whisper model size.")
@click.option("--language", default=None, help="Language code (default: auto-detect).")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path.")
@click.option("--format", "fmt", default=None, help="Output format: txt,json,srt,vtt,all")
@click.option("--gpu", is_flag=True, help="Use GPU acceleration.")
def transcribe(source, model, language, output, fmt, gpu):
    """Transcribe an audio file or URL.

    SOURCE can be a local file path or a URL (YouTube, etc.).
    URL transcription requires yt-dlp: pip install yt-dlp
    """
    import tempfile
    import wave
    from pathlib import Path

    from tscribe.config import TscribeConfig
    from tscribe.paths import ensure_dirs, get_config_path, get_data_dir, get_recordings_dir
    from tscribe.session import SessionManager
    from tscribe.transcriber import Transcriber

    cfg = TscribeConfig.load(get_config_path(get_data_dir()))
    data_dir = get_data_dir(cfg.storage.data_dir)

    model_name = model or cfg.transcription.model
    lang = language or cfg.transcription.language

    if fmt:
        if fmt == "all":
            output_formats = ["txt", "json", "srt", "vtt"]
        else:
            output_formats = [f.strip() for f in fmt.split(",")]
    else:
        output_formats = cfg.transcription.output_formats

    is_url = "://" in source

    if is_url:
        ensure_dirs(data_dir)
        tmp_dir = tempfile.mkdtemp(prefix="tscribe_")
        downloaded = _download_audio(source, tmp_dir)

        # Import into recordings directory so it appears in tscribe list
        mgr = SessionManager(get_recordings_dir(data_dir))
        stem, audio_path = mgr.import_external(downloaded)

        # Create metadata from the WAV file
        duration = 0.0
        sample_rate = 16000
        channels = 1
        try:
            with wave.open(str(audio_path), "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                duration = wf.getnframes() / sample_rate
        except Exception:
            pass

        meta = {
            "created": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "channels": channels,
            "device": "url",
            "source_type": "url",
            "source_url": source,
            "transcribed": False,
            "model": None,
            "original_path": None,
        }
        mgr.write_metadata(stem, meta)
    else:
        audio_path = Path(source)
        if not audio_path.exists():
            raise click.ClickException(f"File not found: {source}")
        stem = None
        mgr = None

    device = "cuda" if gpu else "cpu"
    transcriber = Transcriber(model_name=model_name, device=device)

    click.echo(f"Transcribing {audio_path.name} with model '{model_name}'...")
    result = transcriber.transcribe(
        audio_path,
        language=lang,
        output_formats=output_formats,
    )

    # Update metadata if this was a URL import
    if mgr and stem:
        mgr.update_metadata(stem, transcribed=True, model=model_name)

    click.echo(f"Transcription complete: {len(result.segments)} segments.")
    for fmt_name in output_formats:
        out_path = audio_path.with_suffix(f".{fmt_name}")
        if out_path.exists():
            click.echo(f"  {out_path}")


def _stem_date_str(stem: str) -> str:
    """Format a session stem with a two-char day-of-week suffix."""
    from datetime import datetime

    try:
        dt = datetime.strptime(stem, "%Y-%m-%d-%H%M%S")
        day = dt.strftime("%a")[:2]
        return f"{stem} {day}"
    except ValueError:
        return stem


@main.command(name="list")
@click.option("--limit", "-n", default=20, help="Number of entries to show.")
@click.option("--search", "-s", default=None, help="Search transcript text.")
@click.option("--sort", "sort_by", default="date", type=click.Choice(["date", "duration", "name"]),
              help="Sort field.")
@click.option("--no-header", is_flag=True, help="Omit table header.")
def list_recordings(limit, search, sort_by, no_header):
    """List past recordings and their transcription status."""
    from tscribe.config import TscribeConfig
    from tscribe.paths import get_config_path, get_data_dir, get_recordings_dir
    from tscribe.session import SessionManager

    cfg = TscribeConfig.load(get_config_path(get_data_dir()))
    data_dir = get_data_dir(cfg.storage.data_dir)
    mgr = SessionManager(get_recordings_dir(data_dir))

    # Build stem→ref map from date-sorted order
    all_by_date = mgr.list_sessions(limit=None, sort_by="date")
    ref_map = {s.stem: (f"HEAD~{i}" if i else "HEAD") for i, s in enumerate(all_by_date)}

    sessions = mgr.list_sessions(limit=limit, sort_by=sort_by, search=search)
    if not sessions:
        click.echo("No recordings found.")
        return

    if not no_header:
        click.echo(f"{'REF':<9} {'Date':<25} {'Duration':>9}  {'Transcribed':>12}  {'File'}")
        click.echo("-" * 88)

    for s in sessions:
        ref = ref_map.get(s.stem, "?")
        date_str = _stem_date_str(s.stem)
        if s.duration_seconds is not None:
            m, sec = divmod(int(s.duration_seconds), 60)
            h, m = divmod(m, 60)
            dur_str = f"{h:02d}:{m:02d}:{sec:02d}"
        else:
            dur_str = "---"
        trans_str = "Yes" if s.transcribed else "No"
        click.echo(f"{ref:<9} {date_str:<25} {dur_str:>9}  {trans_str:>12}  {s.wav_path.name}")


@main.command()
@click.argument("query")
@click.option("--limit", "-n", default=20, help="Max sessions to show.")
@click.option("--sort", "sort_by", default="date",
              type=click.Choice(["date", "duration", "name"]), help="Sort field.")
def search(query, limit, sort_by):
    """Search transcript text for a keyword or phrase.

    \b
    Examples:
      tscribe search "action items"
      tscribe search meeting -n 50
    """
    from tscribe.config import TscribeConfig
    from tscribe.paths import get_config_path, get_data_dir, get_recordings_dir
    from tscribe.session import SessionManager

    cfg = TscribeConfig.load(get_config_path(get_data_dir()))
    data_dir = get_data_dir(cfg.storage.data_dir)
    mgr = SessionManager(get_recordings_dir(data_dir))

    # Build stem→ref map from date-sorted order
    all_by_date = mgr.list_sessions(limit=None, sort_by="date")
    ref_map = {s.stem: (f"HEAD~{i}" if i else "HEAD") for i, s in enumerate(all_by_date)}

    results = mgr.search_transcripts(query, limit=limit, sort_by=sort_by)
    if not results:
        click.echo("No matches found.")
        return

    for session, lines in results:
        ref = ref_map.get(session.stem, "?")
        click.echo(f"── {_stem_date_str(session.stem)} ({ref}) ──")
        for line in lines:
            click.echo(line)
        click.echo()

    count = len(results)
    click.echo(f"{count} match{'es' if count != 1 else ''} found.")


def _open_file(path) -> None:
    """Open a file with the OS default program."""
    import subprocess as sp

    if sys.platform == "darwin":
        sp.Popen(["open", str(path)])
    elif sys.platform == "win32":
        import os as _os
        _os.startfile(str(path))
    else:
        sp.Popen(["xdg-open", str(path)])


def _find_player() -> list[str]:
    """Return command prefix for audio playback, or empty list if none found."""
    import shutil

    if sys.platform == "linux":
        for cmd in ["pw-play", "aplay"]:
            if shutil.which(cmd):
                return [cmd]
    elif sys.platform == "darwin":
        if shutil.which("afplay"):
            return ["afplay"]
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit"]
    return []


def _resolve_session(ref, mgr):
    """Resolve a HEAD/HEAD~N/stem reference to a SessionInfo."""
    if ref in ("HEAD", "last"):
        index = 0
    elif ref.startswith("HEAD~"):
        try:
            index = int(ref[5:])
        except ValueError:
            raise click.ClickException(f"Invalid ref: {ref}")
    else:
        session = mgr.get_session(ref)
        if session is None:
            raise click.ClickException(f"Recording not found: {ref}")
        return session

    sessions = mgr.list_sessions(limit=index + 1, sort_by="date")
    if index >= len(sessions):
        raise click.ClickException(
            f"Recording {ref} not found (only {len(sessions)} recording(s) exist)."
        )
    return sessions[index]


@main.command()
@click.argument("ref", default="HEAD")
def play(ref):
    """Play a recording.

    REF can be HEAD (most recent), HEAD~N (Nth previous), or a session stem.

    \b
    Examples:
      tscribe play            Play most recent recording
      tscribe play HEAD~1     Play previous recording
      tscribe play 2025-01-15-143022   Play by session ID
    """
    import subprocess as sp

    from tscribe.config import TscribeConfig
    from tscribe.paths import get_config_path, get_data_dir, get_recordings_dir
    from tscribe.session import SessionManager

    cfg = TscribeConfig.load(get_config_path(get_data_dir()))
    data_dir = get_data_dir(cfg.storage.data_dir)
    mgr = SessionManager(get_recordings_dir(data_dir))

    session = _resolve_session(ref, mgr)

    player = _find_player()
    if not player:
        raise click.ClickException(
            "No audio player found. Install pw-play, aplay, afplay, or ffplay."
        )

    dur_str = ""
    if session.duration_seconds is not None:
        m, s = divmod(int(session.duration_seconds), 60)
        dur_str = f" ({m}m {s:02d}s)"

    click.echo(f"Playing: {session.wav_path.name}{dur_str}")

    proc = sp.Popen(player + [str(session.wav_path)])
    try:
        start = time.monotonic()
        total = session.duration_seconds or 0.0
        while proc.poll() is None:
            elapsed = time.monotonic() - start
            if total > 0:
                minutes, secs = divmod(int(elapsed), 60)
                t_min, t_sec = divmod(int(total), 60)
                frac = min(elapsed / total, 1.0)
                filled = int(frac * 30)
                bar = "█" * filled + "░" * (30 - filled)
                play_icon = click.style("▶", fg="green")
                click.echo(
                    f"\r  {play_icon} {minutes:02d}:{secs:02d}/{t_min:02d}:{t_sec:02d}  |{bar}|",
                    nl=False,
                )
            time.sleep(0.25)
        click.echo()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        click.echo("\nStopped.")


def _resolve_transcript_path(ref, fmt):
    """Resolve a session ref + optional format to a transcript file path."""
    from tscribe.config import TscribeConfig
    from tscribe.paths import get_config_path, get_data_dir, get_recordings_dir
    from tscribe.session import SessionManager

    cfg = TscribeConfig.load(get_config_path(get_data_dir()))
    data_dir = get_data_dir(cfg.storage.data_dir)
    mgr = SessionManager(get_recordings_dir(data_dir))

    session = _resolve_session(ref, mgr)

    if fmt:
        path = session.wav_path.with_suffix(f".{fmt}")
        if not path.exists():
            raise click.ClickException(f"File not found: {path.name}")
        return path

    for ext in ["txt", "json", "srt", "vtt"]:
        candidate = session.wav_path.with_suffix(f".{ext}")
        if candidate.exists():
            return candidate

    raise click.ClickException(
        f"No transcript found for {session.stem}. "
        f"Run: tscribe transcribe {session.wav_path}"
    )


@main.command(name="open")
@click.argument("ref", default="HEAD")
@click.option("--format", "-f", "fmt", default=None,
              type=click.Choice(["txt", "json", "srt", "vtt", "wav"]),
              help="File format to open (default: first available transcript).")
def open_file(ref, fmt):
    """Open a transcription file with the default program.

    REF can be HEAD (most recent), HEAD~N (Nth previous), or a session stem.

    \b
    Examples:
      tscribe open                 Open most recent transcript
      tscribe open -f json         Open JSON transcript
      tscribe open HEAD~1          Open previous transcript
      tscribe open -f wav          Open the audio file
    """
    path = _resolve_transcript_path(ref, fmt)
    click.echo(f"Opening: {path.name}")
    _open_file(path)


@main.command()
@click.argument("ref", default="HEAD")
@click.option("--format", "-f", "fmt", default=None,
              type=click.Choice(["txt", "json", "srt", "vtt"]),
              help="Output format (default: first available transcript).")
def dump(ref, fmt):
    """Print a transcription to stdout.

    REF can be HEAD (most recent), HEAD~N (Nth previous), or a session stem.

    \b
    Examples:
      tscribe dump                 Print most recent transcript
      tscribe dump -f json         Print JSON transcript
      tscribe dump HEAD~1          Print previous transcript
    """
    path = _resolve_transcript_path(ref, fmt)
    click.echo(path.read_text(), nl=False)


@main.command()
@click.option("--loopback", "-l", is_flag=True, help="Show only loopback/monitor sources.")
def devices(loopback):
    """List available audio input devices."""
    # Try PipeWire first on Linux
    if sys.platform == "linux":
        from tscribe.pipewire_devices import is_pipewire_available, list_pipewire_nodes

        if is_pipewire_available():
            nodes = list_pipewire_nodes(loopback_only=loopback)
            if not nodes:
                if loopback:
                    click.echo("No PipeWire monitor/loopback sources found.")
                else:
                    click.echo("No PipeWire audio nodes found.")
                return

            click.echo(f"{'Serial':<8} {'Name':<40} {'Type':<10} {'Description'}")
            click.echo("-" * 85)
            for n in nodes:
                type_label = "Monitor" if n.is_monitor else "Input"
                display_name = n.nick or n.name
                click.echo(
                    f"{n.serial:<8} {display_name:<40} {type_label:<10} {n.description}"
                )
            return

    # Fall back to sounddevice enumeration
    from tscribe.devices import get_platform_loopback_guidance, list_devices

    devs = list_devices(loopback_only=loopback)
    if not devs:
        if loopback:
            guidance = get_platform_loopback_guidance()
            if guidance:
                click.echo(guidance)
                return
        click.echo("No audio input devices found.")
        return

    click.echo(f"{'Idx':<5} {'Name':<45} {'Ch':>3} {'Rate':>7} {'Loopback':>9}  {'Host API'}")
    click.echo("-" * 90)
    for dev in devs:
        lb = "Yes" if dev.is_loopback else ""
        click.echo(
            f"{dev.index:<5} {dev.name:<45} {dev.max_input_channels:>3} "
            f"{dev.default_samplerate:>7.0f} {lb:>9}  {dev.hostapi}"
        )


@main.command()
@click.argument("key", required=False)
@click.argument("value", required=False)
@click.option("--list", "list_all", is_flag=True, help="Show all configuration values.")
def config(key, value, list_all):
    """View or set configuration."""
    from tscribe.config import TscribeConfig
    from tscribe.paths import get_config_path, get_data_dir

    cfg = TscribeConfig.load(get_config_path(get_data_dir()))
    config_path = get_config_path(get_data_dir(cfg.storage.data_dir))
    cfg = TscribeConfig.load(config_path)

    if list_all or (key is None and value is None):
        for section_name, section_dict in cfg._to_dict().items():
            for k, v in section_dict.items():
                click.echo(f"{section_name}.{k} = {v!r}")
        return

    if value is None:
        try:
            click.echo(cfg.get(key))
        except KeyError as e:
            raise click.ClickException(str(e))
        return

    try:
        cfg.set(key, value)
    except (KeyError, ValueError) as e:
        raise click.ClickException(str(e))
    cfg.save(config_path)
    click.echo(f"Set {key} = {cfg.get(key)!r}")
