# Troubleshooting Guide

Common issues and solutions when using champi-tts.

## Table of Contents

- [Installation Issues](#installation-issues)
- [TTS Provider Issues](#tts-provider-issues)
- [Text Reading Issues](#text-reading-issues)
- [UI Issues](#ui-issues)
- [Audio Issues](#audio-issues)
- [Performance Issues](#performance-issues)
- [API Issues](#api-issues)

## Installation Issues

### Module Not Found After Installation

**Problem**: After installing champi-tts, you get "ModuleNotFoundError: No module named 'champi_tts'"

**Solutions**:

```bash
# Ensure you're using the correct Python environment
python --version
which python

# Reinstall the package
pip uninstall champi-tts
pip install champi-tts

# Check installation location
pip show champi-tts
```

### Import Error in Virtual Environment

**Problem**: Module not found even though installation appears successful

**Solutions**:

```bash
# Activate your virtual environment
# For venv
source venv/bin/activate

# For conda
conda activate champi-tts-env

# Reinstall in virtual environment
pip install champi-tts
```

### Dependency Conflicts

**Problem**: Installation fails due to dependency conflicts

**Solutions**:

```bash
# Install using uv for dependency resolution
uv pip install champi-tts

# Or check for conflicting packages
pip check

# Create a new virtual environment
python -m venv venv_new
source venv_new/bin/activate
pip install champi-tts
```

## TTS Provider Issues

### No Voices Available

**Problem**: When initializing the provider, no voices are available

**Solutions**:

```python
from champi_tts import get_provider

# Get provider and check voices
provider = get_provider("kokoro")
await provider.initialize()

if not provider.voices:
    print("No voices available!")
else:
    print(f"Available voices: {provider.voices}")

await provider.shutdown()
```

### Provider Initialization Fails

**Problem**: Provider fails to initialize with an error

**Solutions**:

```python
import asyncio
from champi_tts import get_provider

async def main():
    try:
        provider = get_provider("kokoro")
        await provider.initialize()
        print("Provider initialized successfully")
        await provider.shutdown()
    except Exception as e:
        print(f"Initialization failed: {e}")
        # Check logs for more details

asyncio.run(main())
```

### Audio Not Playing

**Problem**: Audio is synthesized but doesn't play

**Solutions**:

```bash
# Check system audio settings
# On Linux
alsamixer

# On macOS
System Preferences > Sound > Output

# Test with a different voice
champi-tts synthesize "Test" --voice af_bella

# Try different audio output device
```

### Poor Audio Quality

**Problem**: Synthesized audio has poor quality

**Solutions**:

```python
from champi_tts import get_provider

async def main():
    provider = get_provider("kokoro")

    # Try different voices
    test_voices = ["af_bella", "am_adam", "af_sarah", "am_michael"]

    for voice in test_voices:
        provider.config.voice = voice
        await provider.initialize()

        audio = await provider.synthesize("Testing voice quality")

        # Save to compare
        await save_audio(audio, f"test_{voice}.wav", sample_rate=provider.config.sample_rate)

        await provider.shutdown()

asyncio.run(main())
```

## Text Reading Issues

### Cannot Pause/Resume

**Problem**: Pause and resume controls don't work

**Solutions**:

```python
from champi_tts import get_reader

# Ensure UI is enabled
reader = get_reader("kokoro", show_ui=True)

# Check if pause/resume are supported
if hasattr(reader, 'pause') and hasattr(reader, 'resume'):
    await reader.pause()
    await reader.resume()
else:
    print("Pause/resume not supported for this provider")
```

### Long Documents Not Working

**Problem**: Long documents don't read properly

**Solutions**:

```python
from champi_tts import get_reader

# The reader handles long documents automatically
reader = get_reader("kokoro", show_ui=True)
await reader.provider.initialize()

# Just read the file - it handles queue management
await reader.read_file("large_document.txt", voice="af_bella")

await reader.provider.shutdown()
```

### Text Not Read Properly

**Problem**: Text is read with errors or skipping

**Solutions**:

```python
from champi_tts import get_reader

reader = get_reader("kokoro", show_ui=True)
await reader.provider.initialize()

# Ensure text is properly formatted
text = """
First paragraph.

Second paragraph.

Third paragraph.
"""

# Read with proper formatting
await reader.read(text, voice="af_bella")

await reader.provider.shutdown()
```

### File Not Found Error

**Problem**: FileNotFoundError when reading a file

**Solutions**:

```python
import os
from champi_tts import get_reader

# Check if file exists
filename = "document.txt"
if os.path.exists(filename):
    print(f"File {filename} exists")
else:
    print(f"File {filename} not found!")
    # Check in current directory
    print(f"Current directory: {os.getcwd()}")
    print(f"Files: {os.listdir()}")
```

## UI Issues

### UI Not Showing

**Problem**: Visual UI indicator doesn't appear

**Solutions**:

```bash
# Check if UI dependencies are installed
# Linux
sudo apt-get install libglfw3-dev

# macOS
brew install glfw

# On Windows, usually no action needed

# Test UI in test mode
champi-tts test-ui
```

### UI Window Not Closing

**Problem**: UI window won't close properly

**Solutions**:

```bash
# Close the window normally
# If stuck, use Ctrl+C in terminal

# Or kill the process
# Find the process
ps aux | grep champi

# Kill the process
kill <PID>
```

### UI Not Updating

**Problem**: UI indicator doesn't update states

**Solutions**:

```python
from champi_tts.ui import UIIndicator

# Create UI indicator
indicator = UIIndicator(mode="test")

# Start it
await indicator.start()

# Try different states
await indicator.set_state("idle")
await indicator.set_state("processing")
await indicator.set_state("speaking")
await indicator.set_state("paused")
await indicator.set_state("error")

# Stop it
await indicator.stop()
```

## Audio Issues

### Cannot Save Audio File

**Problem**: save_audio function fails

**Solutions**:

```python
from champi_tts.core.audio import save_audio
from champi_tts import get_provider

# Check write permissions
import os
output_file = "output.wav"
if os.path.exists(output_file):
    print(f"File {output_file} already exists")

# Try different output formats
try:
    await save_audio(audio, "test.wav", sample_rate=24000)
    print("WAV saved successfully")
except Exception as e:
    print(f"Error saving WAV: {e}")

try:
    await save_audio(audio, "test.mp3", sample_rate=24000)
    print("MP3 saved successfully")
except Exception as e:
    print(f"Error saving MP3: {e}")
```

### Audio File Not Playable

**Problem**: Saved audio file cannot be played

**Solutions**:

```bash
# Check file with ffprobe (if available)
ffprobe output.wav

# Try playing with different players
# Linux
aplay output.wav

# macOS
afplay output.wav

# Windows
start output.wav
```

### Sample Rate Mismatch

**Problem**: Audio plays at wrong speed or pitch

**Solutions**:

```python
from champi_tts import get_provider

async def main():
    provider = get_provider("kokoro")
    await provider.initialize()

    # Check sample rate
    print(f"Sample rate: {provider.config.sample_rate}")

    # Save with correct sample rate
    await save_audio(audio, "output.wav", sample_rate=provider.config.sample_rate)

    await provider.shutdown()

asyncio.run(main())
```

## Performance Issues

### Slow Synthesis

**Problem**: Text-to-speech is slow

**Solutions**:

```python
from champi_tts import get_provider

async def main():
    provider = get_provider("kokoro")

    # Enable GPU acceleration
    provider.config.use_gpu = True
    await provider.initialize()

    # Compare CPU vs GPU
    print("Testing with GPU...")
    await provider.initialize(use_gpu=True)

    print("Testing without GPU...")
    provider.config.use_gpu = False
    await provider.initialize(use_gpu=False)

    await provider.shutdown()

asyncio.run(main())
```

### High Memory Usage

**Problem**: Application uses too much memory

**Solutions**:

```python
from champi_tts import get_provider
from champi_tts.core.audio import save_audio
import psutil

async def main():
    provider = get_provider("kokoro")
    await provider.initialize()

    # Process smaller chunks
    text = "Your long text here"
    chunk_size = 1000  # characters per chunk

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        audio = await provider.synthesize(chunk)
        await save_audio(audio, f"output_{i}.wav", sample_rate=provider.config.sample_rate)

    await provider.shutdown()

# Monitor memory
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024} MB")
```

### CPU Usage High

**Problem**: High CPU usage during processing

**Solutions**:

```python
from champi_tts import get_provider
import asyncio

async def main():
    provider = get_provider("kokoro")
    await provider.initialize()

    # Process with reduced quality if needed
    provider.config.quality = "low"  # If available
    provider.config.use_gpu = False  # Disable GPU

    await provider.shutdown()

# Run with asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
loop.run_until_complete(main())
loop.close()
```

## API Issues

### Async Functions Not Working

**Problem**: Async/await not working correctly

**Solutions**:

```python
import asyncio

# Ensure you're using asyncio.run() for async main()
async def main():
    from champi_tts import get_provider
    provider = get_provider("kokoro")
    await provider.initialize()
    # ... rest of your code
    await provider.shutdown()

# Correct way
if __name__ == "__main__":
    asyncio.run(main())

# Incorrect way
main()  # This won't work for async functions
```

### Provider Not Found

**Problem**: get_provider() raises error for specific provider

**Solutions**:

```python
from champi_tts import get_provider

# Check available providers
try:
    provider = get_provider("kokoro")
    print("Kokoro provider available")
except Exception as e:
    print(f"Kokoro provider not available: {e}")

# Note: OpenAI and ElevenLabs are coming soon
try:
    provider = get_provider("openai")
except Exception as e:
    print(f"OpenAI provider not yet available: {e}")
```

### Memory Leak

**Problem**: Memory usage keeps growing over time

**Solutions**:

```python
import asyncio
from champi_tts import get_provider
from champion_signals import SignalBus

# Proper cleanup pattern
async def process_with_cleanup():
    provider = get_provider("kokoro")
    signal_bus = SignalBus()

    try:
        await provider.initialize()
        # ... do work
        await signal_bus.initialize()
        # ... more work
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Always cleanup
        if hasattr(signal_bus, 'cleanup'):
            await signal_bus.cleanup()
        if hasattr(provider, 'shutdown'):
            await provider.shutdown()

# Use this pattern
asyncio.run(process_with_cleanup())
```

## Common Error Messages

### ImportError

**Message**: `ModuleNotFoundError: No module named 'champi_tts'`

**Solution**: Install the package: `pip install champi-tts`

### PermissionError

**Message**: `PermissionError: [Errno 13] Permission denied: 'output.wav'`

**Solution**: Check write permissions: `chmod +w output.wav`

### FileNotFoundError

**Message**: `FileNotFoundError: [Errno 2] No such file or directory: 'document.txt'`

**Solution**: Check file exists: `ls document.txt` or `find . -name "*.txt"`

### RuntimeError

**Message**: `RuntimeError: Event loop is closed`

**Solution**: Use asyncio.run() correctly, don't create multiple event loops

### ValueError

**Message**: `ValueError: Invalid voice name: 'invalid_voice'`

**Solution**: List available voices: `champi-tts list-voices`

## Getting Help

If you encounter an issue not covered here:

1. **Check Documentation**: Review [USER_GUIDE.md](USER_GUIDE.md) and [TUTORIALS.md](TUTORIALS.md)
2. **Check GitHub Issues**: Search existing issues at https://github.com/divagnz/champi-tts/issues
3. **Open a New Issue**: Report your issue with:
   - Python version: `python --version`
   - Installation command used
   - Error message or stack trace
   - Minimal reproducible code
   - OS and platform details

## Logging

Enable debug logging to get more information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed logs from champi-tts
```

---

**Version**: 0.2.0
**Last Updated**: 2026-06-11
**License**: MIT