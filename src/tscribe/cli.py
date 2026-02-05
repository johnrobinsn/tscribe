"""CLI entry point for tscribe."""

import signal
import sys
import threading

import click

from tscribe import __version__


def _create_recorder(config):
    """Create the appropriate recorder. Can be monkeypatched for testing."""
    from tscribe.recorder.sounddevice_recorder import SounddeviceRecorder
    return SounddeviceRecorder()


def _auto_transcribe(cfg, data_dir, wav_path, stem, session_mgr):
    """Run transcription after recording. Handles missing whisper.cpp gracefully."""
    from pathlib import Path

    from tscribe.paths import get_whisper_dir
    from tscribe.transcriber import Transcriber
    from tscribe.whisper_manager import WhisperManager

    wm = WhisperManager(get_whisper_dir(data_dir))
    model_name = cfg.transcription.model

    if not wm.is_binary_available():
        click.echo("whisper.cpp not found. Run 'tscribe setup' first to enable auto-transcription.")
        return

    if not wm.is_model_available(model_name):
        click.echo(f"Model '{model_name}' not found. Downloading...")
        try:
            wm.download_model(model_name)
        except Exception as e:
            click.echo(f"Failed to download model: {e}")
            return

    binary = wm.find_binary()
    model_path = wm.model_path(model_name)
    transcriber = Transcriber(binary, model_path)

    click.echo(f"Transcribing with model '{model_name}'...")
    try:
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
    """Record audio and transcribe with whisper.cpp."""


@main.command()
@click.option("--device", "-d", default=None, help="Audio device name or index.")
@click.option("--loopback", "-l", is_flag=True, help="Record system audio.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path.")
@click.option("--no-transcribe", is_flag=True, help="Don't auto-transcribe after recording.")
@click.option("--sample-rate", type=int, default=None, help="Sample rate in Hz.")
@click.option("--channels", type=int, default=None, help="Number of channels.")
def record(device, loopback, output, no_transcribe, sample_rate, channels):
    """Record audio from a microphone or system audio."""
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
        click.echo(f"Recording to {wav_path} (Ctrl+C to stop)...")

        while not stop_event.is_set():
            elapsed = recorder.elapsed_seconds
            minutes, secs = divmod(int(elapsed), 60)
            click.echo(f"\r  {minutes:02d}:{secs:02d}", nl=False)
            stop_event.wait(0.5)

        result = recorder.stop()
        click.echo(f"\nRecording saved: {result.path} ({result.duration_seconds:.1f}s)")

        meta = session_mgr.create_metadata(result)
        session_mgr.write_metadata(stem, meta)

        if not no_transcribe and cfg.recording.auto_transcribe:
            _auto_transcribe(cfg, data_dir, wav_path, stem, session_mgr)

    finally:
        signal.signal(signal.SIGINT, original_handler)


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--model", "-m", default=None, help="Whisper model size.")
@click.option("--language", default=None, help="Language code (default: auto-detect).")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path.")
@click.option("--format", "fmt", default=None, help="Output format: txt,json,srt,vtt,all")
@click.option("--gpu", is_flag=True, help="Use GPU acceleration.")
def transcribe(file, model, language, output, fmt, gpu):
    """Transcribe an audio file."""
    from pathlib import Path

    from tscribe.config import TscribeConfig
    from tscribe.paths import ensure_dirs, get_config_path, get_data_dir, get_whisper_dir
    from tscribe.transcriber import Transcriber
    from tscribe.whisper_manager import WhisperManager

    cfg = TscribeConfig.load(get_config_path(get_data_dir()))
    data_dir = get_data_dir(cfg.storage.data_dir)
    ensure_dirs(data_dir)

    model_name = model or cfg.transcription.model
    lang = language or cfg.transcription.language

    if fmt:
        if fmt == "all":
            output_formats = ["txt", "json", "srt", "vtt"]
        else:
            output_formats = [f.strip() for f in fmt.split(",")]
    else:
        output_formats = cfg.transcription.output_formats

    wm = WhisperManager(get_whisper_dir(data_dir))

    if not wm.is_binary_available():
        click.echo("whisper.cpp not found. Running setup...")
        wm.download_binary()

    if not wm.is_model_available(model_name):
        click.echo(f"Model '{model_name}' not found. Downloading...")
        wm.download_model(model_name)

    binary = wm.find_binary()
    model_path = wm.model_path(model_name)

    transcriber = Transcriber(binary, model_path)
    audio_path = Path(file)

    click.echo(f"Transcribing {audio_path.name} with model '{model_name}'...")
    result = transcriber.transcribe(
        audio_path,
        language=lang,
        output_formats=output_formats,
        gpu=gpu,
    )

    click.echo(f"Transcription complete: {len(result.segments)} segments.")
    for fmt_name in output_formats:
        out_path = audio_path.with_suffix(f".{fmt_name}")
        if out_path.exists():
            click.echo(f"  {out_path}")


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

    sessions = mgr.list_sessions(limit=limit, sort_by=sort_by, search=search)
    if not sessions:
        click.echo("No recordings found.")
        return

    if not no_header:
        click.echo(f"{'Date':<22} {'Duration':>9}  {'Transcribed':>12}  {'File'}")
        click.echo("-" * 75)

    for s in sessions:
        date_str = s.stem
        if s.duration_seconds is not None:
            m, sec = divmod(int(s.duration_seconds), 60)
            h, m = divmod(m, 60)
            dur_str = f"{h:02d}:{m:02d}:{sec:02d}"
        else:
            dur_str = "---"
        trans_str = "Yes" if s.transcribed else "No"
        click.echo(f"{date_str:<22} {dur_str:>9}  {trans_str:>12}  {s.wav_path.name}")


@main.command()
@click.option("--loopback", "-l", is_flag=True, help="Show only loopback/monitor sources.")
def devices(loopback):
    """List available audio input devices."""
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


@main.command()
@click.option("--model", "-m", default="base", help="Model to download.")
@click.option("--force", is_flag=True, help="Re-download even if already present.")
def setup(model, force):
    """Download whisper.cpp and models."""
    from tscribe.config import TscribeConfig
    from tscribe.paths import ensure_dirs, get_config_path, get_data_dir, get_whisper_dir
    from tscribe.whisper_manager import WhisperManager

    cfg = TscribeConfig.load(get_config_path(get_data_dir()))
    data_dir = get_data_dir(cfg.storage.data_dir)
    ensure_dirs(data_dir)

    wm = WhisperManager(get_whisper_dir(data_dir))
    wm.setup(model=model, force=force)
    click.echo("Setup complete.")
