# CLI Guide

Complete guide to using the Champi TTS command-line interface.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Synthesis Commands](#synthesis-commands)
- [Text Reading Commands](#text-reading-commands)
- [Voice Management](#voice-management)
- [UI Commands](#ui-commands)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Scripting with CLI](#scripting-with-cli)

---

## Basic Usage

### Version Check

Check Champi TTS version:

```bash
champi-tts --version
```

### Help

Display help for any command:

```bash
champi-tts <command> --help
```

Examples:
```bash
champi-tts synthesize --help
champi-tts read --help
```

---

## Synthesis Commands

### Synthesize Text

Synthesize text to speech and play it:

```bash
champi-tts synthesize "Hello, world!"
```

### Save to File

Synthesize and save to a file:

```bash
champi-tts synthesize "Hello, world!" --output hello.wav
```

### Specify Voice

Use a specific voice:

```bash
champi-tts synthesize "Hello, world!" --voice af_bella
```

### Specify Voice with Speed

Set both voice and speech speed:

```bash
champi-tts synthesize "Hello, world!" --voice af_bella --speed 1.2
```

### No Audio Playback

Synthesize without playing audio:

```bash
champi-tts synthesize "Hello, world!" --no-play --output hello.wav
```

### Stream Output

Stream audio to stdout:

```bash
champi-tts synthesize "Hello, world!" --stdout
```

---

## Text Reading Commands

### Read Text File

Read a text file:

```bash
champi-tts read document.txt
```

### Read Direct Text

Read text directly from command line:

```bash
champi-tts read --text "This is a test"
```

### Interactive Mode

Enable interactive controls (pause/resume):

```bash
champi-tts read document.txt --interactive
```

### Show UI Indicator

Display visual UI indicator:

```bash
champi-tts read document.txt --show-ui
```

### Combined Options

Read with multiple options:

```bash
champi-tts read document.txt --voice af_bella --show-ui --interactive
```

---

## Voice Management

### List Available Voices

List all available voices:

```bash
champi-tts list-voices
```

Output example:
```
Available voices:
- af_bella: Female voice, friendly tone
- am_adam: Male voice, calm tone
- am_echo: Male voice, deep tone
- af_sarah: Female voice, energetic tone
```

### Get Voice Details

Get details about a specific voice (using Python API):

```python
from champi_tts import get_provider

provider = get_provider()
await provider.initialize()

voices = await provider.list_voices()
print(f"Available voices: {voices}")

await provider.shutdown()
```

---

## UI Commands

### Run UI Indicator

Test the visual UI indicator:

```bash
champi-tts test-ui
```

This will cycle through all visual states (Idle, Processing, Speaking, Paused, Error).

### Configure UI Position

Adjust UI window position:

```bash
champi-tts read document.txt --show-ui --ui-x 50 --ui-y 50
```

### Adjust UI Size

Set UI window size:

```bash
champi-tts read document.txt --show-ui --ui-width 300 --ui-height 150
```

---

## Configuration

### Environment Variables

Configure using environment variables:

```bash
export CHAMPI_TTS_DEFAULT_VOICE=af_bella
export CHAMPI_TTS_DEFAULT_SPEED=1.1
export CHAMPI_TTS_USE_GPU=true
champi-tts read document.txt
```

### Configuration File

Create a `.env` file:

```env
# Champi TTS Configuration
CHAMPI_TTS_DEFAULT_VOICE=af_bella
CHAMPI_TTS_DEFAULT_SPEED=1.2
CHAMPI_TTS_USE_GPU=true
CHAMPI_TTS_MODEL_PATH=/path/to/models
```

Load configuration automatically when running commands.

---

## Advanced Usage

### Batch Processing

Process multiple text files:

```bash
for file in *.txt; do
    champi-tts read "$file" --output "${file%.txt}.wav"
done
```

### Pipeline Processing

Chain multiple commands:

```bash
champi-tts read input.txt --show-ui | \
    convert output.wav --speed 1.1
```

### Custom Configuration

Use specific configuration for single command:

```bash
champi-tts read document.txt \
    --config default_voice=af_bella \
    --config use_gpu=true
```

### Timeout Settings

Set operation timeout:

```bash
champi-tts read long_document.txt --timeout 300
```

---

## Scripting with CLI

### Basic Automation

Create a shell script:

```bash
#!/bin/bash
champi-tts synthesize "Good morning" --voice af_bella --output greeting.wav
aplay greeting.wav
```

### Error Handling

Add error checking:

```bash
#!/bin/bash
if champi-tts synthesize "Test" --output test.wav; then
    echo "Synthesis successful"
    aplay test.wav
else
    echo "Synthesis failed"
    exit 1
fi
```

### Python Integration

Use CLI from Python:

```python
import subprocess

def synthesize_cli(text, output_file):
    """Run Champi TTS CLI command from Python"""
    cmd = [
        "champi-tts", "synthesize",
        text,
        "--output", output_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

# Usage
if synthesize_cli("Hello from Python CLI", "output.wav"):
    print("Success!")
```

### Background Processing

Run in background:

```bash
champi-tts read document.txt --show-ui &
```

---

## Common Workflows

### Daily Briefing

Generate daily audio briefing:

```bash
champi-tts read "daily_briefing.txt" --voice af_bella --show-ui
```

### Accessibility Tool

Read documents for visually impaired users:

```bash
champi-tts read document.txt --show-ui --interactive
```

### Content Creation

Convert text articles to audio:

```bash
champi-tts read article.txt --output article_audio.wav --no-play
```

### Learning Aid

Listen to study materials:

```bash
champi-tts read study_notes.txt --voice am_adam --show-ui --interactive
```

---

## Command Reference

### Synthesize

```bash
champi-tts synthesize [TEXT] [OPTIONS]
```

- `TEXT`: Text to synthesize (required if not using --text)
- `--text TEXT`: Text to synthesize
- `--output FILE`: Save to file
- `--voice VOICE`: Voice to use
- `--speed SPEED`: Speech speed (0.5-2.0)
- `--no-play`: Don't play audio
- `--stdout`: Output to stdout
- `--timeout SECONDS`: Operation timeout

### Read

```bash
champi-tts read [FILE] [OPTIONS]
```

- `FILE`: Text file to read (required if not using --text)
- `--text TEXT`: Text to read
- `--voice VOICE`: Voice to use
- `--show-ui`: Show visual indicator
- `--interactive`: Enable interactive controls
- `--timeout SECONDS`: Operation timeout

### List-Voices

```bash
champi-tts list-voices
```

Lists all available voices.

### Test-UI

```bash
champi-tts test-ui
```

Runs UI indicator test.

---

## Troubleshooting

### Audio Not Playing

**Solution**: Ensure audio output is configured:

```bash
aplay test.wav  # Linux
afplay test.wav # macOS
ffplay test.wav # Cross-platform
```

### Voice Not Found

**Solution**: Download voices:

```bash
champi-tts download-voices
```

### Slow Performance

**Solution**: Enable GPU or adjust model path:

```bash
export CHAMPI_TTS_USE_GPU=true
```

### UI Not Displaying

**Solution**: Install UI dependencies:

```bash
pip install "champi-tts[ui]"
```

---

## Tips and Best Practices

1. **Use Appropriate Voices**: Choose voices that match your content tone
2. **Test Before Production**: Always test with your actual text
3. **Handle Errors**: Always check command return codes
4. **Batch Processing**: Process files in batches to avoid timeouts
5. **Monitor Resources**: Check memory usage for long documents
6. **Use Interactive Mode**: Better control for manual reading sessions

---

**Need more help?** Check out the [Python API Guide](python-api.md) for programmatic access or [Text Reading Guide](text-reading.md) for more details.