# Capturing System Audio on macOS

macOS does not provide a built-in way to record system audio (what you hear from your speakers). Unlike Linux (PipeWire) and Windows (WASAPI loopback), macOS requires a third-party virtual audio driver to route system sound into an application like tscribe.

This guide walks through the setup.

## 1. Install BlackHole

[BlackHole](https://github.com/ExistentialAudio/BlackHole) is a free, open-source virtual audio driver for macOS.

```
brew install blackhole-2ch
```

After installing, **restart your Mac** (or at minimum log out and back in) for the audio driver to load.

You can verify it installed by opening **Audio MIDI Setup** (Spotlight: "Audio MIDI Setup") and looking for "BlackHole 2ch" in the device list.

## 2. Create a Multi-Output Device

BlackHole is a virtual audio cable — it passes audio from an output to an input, but it doesn't tap into your existing speakers. To hear audio *and* record it, you need a Multi-Output Device that sends sound to both your speakers and BlackHole simultaneously.

1. Open **Audio MIDI Setup** (Spotlight: "Audio MIDI Setup")
2. Click the **+** button in the bottom-left corner
3. Select **Create Multi-Output Device**
4. In the right panel, check both:
   - Your normal output (e.g., "MacBook Pro Speakers" or "External Headphones")
   - **BlackHole 2ch**
5. Make sure your speakers/headphones are listed **first** (drag to reorder) — this keeps audio latency consistent with what you hear

You can optionally rename the device (right-click → "Rename") to something like "Speakers + BlackHole".

## 3. Select the Multi-Output Device as System Output

Before recording system audio with tscribe, you need to set the Multi-Output Device as your system output:

1. Go to **System Settings → Sound → Output**
2. Select **Multi-Output Device** (or whatever you named it)

Now all system audio flows to both your speakers and BlackHole. tscribe can read from BlackHole to capture what you hear.

> **Warning**: While the Multi-Output Device is selected, **the system volume keys and menu bar slider stop working**. macOS does not allow volume control on aggregate/multi-output devices. You must adjust volume per-application, or switch back to your normal output device when you're done recording.

**Switch back to your normal output device when you're done recording.** Leaving the Multi-Output Device selected permanently means no hardware volume control.

## 4. Record with tscribe

With the Multi-Output Device selected as output, tscribe will automatically detect BlackHole and record from it:

```
tscribe record
```

When finished (Ctrl+C), switch your output back to your normal speakers/headphones in System Settings → Sound → Output.

## 5. Optional: Automate Output Switching

Manually toggling the output device before and after every recording is tedious. Install `switchaudio-osx` to let tscribe handle it automatically:

```
brew install switchaudio-osx
```

With this installed, `tscribe record` will:
1. Prompt you to switch to the Multi-Output Device
2. Switch automatically if you confirm
3. Restore your previous output device when recording stops (including on Ctrl+C)

This is the recommended setup for regular use.

## Why Is This So Complicated?

macOS has a system-level restriction called [System Integrity Protection (SIP)](https://support.apple.com/en-us/102149) that prevents applications from capturing audio output directly. On other platforms:

- **Linux (PipeWire/PulseAudio)**: The audio server natively exposes monitor sources for any output — no extra software needed.
- **Windows (WASAPI)**: The audio API has a built-in loopback mode that captures output device audio.

macOS has no equivalent. The only supported approach is routing audio through a virtual driver like BlackHole, which acts as a bridge between output and input.

## Troubleshooting

**No levels showing in tscribe**: The Multi-Output Device is probably not set as your system output. Check System Settings → Sound → Output.

**No sound from speakers**: Make sure your speakers are checked in the Multi-Output Device configuration in Audio MIDI Setup, and that they're listed first.

**BlackHole not appearing**: Try restarting your Mac after installation. If using Apple Silicon, ensure you allowed the system extension in System Settings → Privacy & Security.
