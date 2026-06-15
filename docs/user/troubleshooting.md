# Troubleshooting Guide

Common problems and solutions when using Champi TTS.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Initialization Issues](#initialization-issues)
- [Synthesis Problems](#synthesis-problems)
- [Audio Playback Issues](#audio-playback-issues)
- [UI Issues](#ui-issues)
- [Performance Issues](#performance-issues)
- [Code Examples](#code-examples)

---

## Installation Issues

### ImportError: No module named 'champi_tts'

**Problem**: Cannot import champi_tts module

**Solutions**:

1. **Install the package**:
```bash
pip install champi-tts
```

2. **Use uv if available**:
```bash
uv pip install champi-tts
```

3. **Install in editable mode for development**:
```bash
pip install -e ".[dev]"
```

4. **Check Python version**:
```bash
python --version  # Must be 3.12 or higher
```

5. **Reinstall with --force-reinstall**:
```bash
pip uninstall champi-tts
pip install champi-tts
```

---

### "pip install" fails with SSL errors

**Problem**: SSL certificate verification fails during installation

**Solutions**:

1. **Use trusted host**:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org champi-tts
```

2. **Update pip**:
```bash
pip install --upgrade pip
pip install champi-tts
```

3. **Use system pip if available**:
```bash
python -m pip install --upgrade pip
python -m pip install champi-tts
```

---

### Package installation succeeds but module still not found

**Problem**: pip succeeds but Python cannot find the module

**Solutions**:

1. **Check installation location**:
```bash
python -c "import champi_tts; print(champi_tts.__file__)"
```

2. **Verify Python path**:
```bash
python -c "import sys; print('\n'.join(sys.path))"
```

3. **Install using python -m pip**:
```bash
python -m pip install champi-tts
```

4. **Clear pip cache**:
```bash
pip cache purge
pip install champi-tts
```

---

## Initialization Issues

### InitializationError: Provider failed to initialize

**Problem**: Cannot initialize TTS provider

**Solutions**:

1. **Check Python version**:
```python
import sys
print(f"Python version: {sys.version}")
# Must be 3.12 or higher
```

2. **Verify provider type**:
```python
from champi_tts import get_provider

# Try default provider
provider = get_provider()
await provider.initialize()

# Or specify provider
provider = get_provider("kokoro")
await provider.initialize()
```

3. **Check for model files**:
```bash
# List available voices
champi-tts list-voices
```

4. **Download voices if needed**:
```bash
champi-tts download-voices
```

5. **Check CUDA if using GPU**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

6. **Review error logs**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Re-run initialization to see detailed logs
```

---

### "Voice not found" error

**Problem**: Voice not found during initialization

**Solutions**:

1. **Download voices**:
```bash
champi-tts download-voices
```

2. **Check voice name**:
```python
voices = await provider.list_voices()
print(f"Available voices: {voices}")
```

3. **Use default voice**:
```python
config = KokoroConfig(default_voice="af_bella")
provider = get_provider("kokoro", config=config)
```

---

### CUDA out of memory error

**Problem**: GPU runs out of memory during synthesis

**Solutions**:

1. **Disable GPU**:
```python
config = KokoroConfig(use_gpu=False)
provider = get_provider("kokoro", config=config)
```

2. **Reduce batch size**:
```python
# Use smaller chunks
async for chunk in provider.synthesize_streaming("Text"):
    # Process each chunk
    pass
```

3. **Free GPU memory**:
```python
import torch

# Clear cache
torch.cuda.empty_cache()

# Or disable GPU entirely
config = KokoroConfig(use_gpu=False)
```

---

## Synthesis Problems

### Synthesis returns empty or invalid audio

**Problem**: Synthesized audio is empty or has incorrect format

**Solutions**:

1. **Check input text**:
```python
# Ensure text is not empty
text = "Your text here"
audio = await provider.synthesize(text)
print(f"Audio shape: {audio.shape}")
```

2. **Verify provider initialization**:
```python
await provider.initialize()
audio = await provider.synthesize("Test")
```

3. **Check sample rate**:
```python
print(f"Sample rate: {provider.config.sample_rate}")
```

4. **Handle exceptions**:
```python
try:
    audio = await provider.synthesize("Text")
    assert audio is not None
    assert len(audio) > 0
except Exception as e:
    print(f"Synthesis failed: {e}")
```

---

### Slow synthesis

**Problem**: Synthesis takes too long

**Solutions**:

1. **Enable GPU**:
```python
config = KokoroConfig(use_gpu=True)
provider = get_provider("kokoro", config=config)
```

2. **Use streaming**:
```python
async for chunk in provider.synthesize_streaming("Text"):
    # Process chunks immediately
    pass
```

3. **Reduce text size**:
```python
# Process in chunks
chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
for chunk in chunks:
    audio = await provider.synthesize(chunk)
```

4. **Reuse providers**:
```python
# Create provider once, reuse
provider = get_provider()
await provider.initialize()

for text in texts:
    audio = await provider.synthesize(text)

await provider.shutdown()
```

---

### Synthesis produces garbled audio

**Problem**: Audio sounds distorted or incorrect

**Solutions**:

1. **Check voice settings**:
```python
config = KokoroConfig(
    default_voice="af_bella",
    default_speed=1.0
)
```

2. **Normalize audio**:
```python
from champi_tts.core.audio import normalize_audio
normalized = normalize_audio(audio)
```

3. **Check sample rate**:
```python
from champi_tts.core.audio import save_audio

# Specify correct sample rate
await save_audio(audio, "output.wav", sample_rate=provider.config.sample_rate)
```

4. **Reinitialize provider**:
```python
await provider.shutdown()
await provider.initialize()
```

---

## Audio Playback Issues

### Audio not playing

**Problem**: Audio won't play after synthesis

**Solutions**:

1. **Check audio file**:
```bash
# Linux
aplay test.wav

# macOS
afplay test.wav

# Windows
start test.wav
```

2. **Use AudioPlayer**:
```python
from champi_tts.core.audio import AudioPlayer

player = AudioPlayer(sample_rate=22050)
await player.play(audio, blocking=False)
```

3. **Check system audio**:
```bash
# Linux - check audio devices
aplay -l

# macOS - check audio preferences
```

4. **Try different player**:
```python
import pygame
pygame.mixer.init()
pygame.mixer.music.load("test.wav")
pygame.mixer.music.play()
```

---

### Audio file cannot be played

**Problem**: Saved audio file won't play

**Solutions**:

1. **Verify file was saved**:
```python
import os
print(f"File exists: {os.path.exists('output.wav')}")
```

2. **Check file size**:
```python
import os
print(f"File size: {os.path.getsize('output.wav')} bytes")
```

3. **Check file format**:
```bash
file output.wav
```

4. **Re-save with correct sample rate**:
```python
await save_audio(audio, "output.wav", sample_rate=provider.config.sample_rate)
```

---

## UI Issues

### UI not displaying

**Problem**: Visual UI indicator doesn't appear

**Solutions**:

1. **Install UI dependencies**:
```bash
pip install "champi-tts[ui]"
```

2. **Check for errors**:
```python
reader = get_reader("kokoro", show_ui=True)
# Check for exceptions
await reader.read_text("Test")
```

3. **Run UI test**:
```bash
champi-tts test-ui
```

4. **Check window manager**:
```bash
# Some window managers may not display UI correctly
```

---

### UI window not visible

**Problem**: UI window appears but is not visible

**Solutions**:

1. **Move window**:
```python
# Try moving window manually
# Some window managers need explicit positioning
```

2. **Check window focus**:
```bash
# Focus the window manually
```

3. **Disable UI and use logging**:
```python
reader = get_reader("kokoro", show_ui=False)

def on_state_changed(sender, **kw):
    print(f"State: {kw.get('new_state')}")

reader.on_state_changed.connect(on_state_changed)
```

---

## Performance Issues

### High memory usage

**Problem**: Application consumes too much memory

**Solutions**:

1. **Use streaming**:
```python
async for chunk in provider.synthesize_streaming("Text"):
    # Process and discard chunks
    pass
```

2. **Process in chunks**:
```python
# Read large file in chunks
chunk_size = 10000
with open("large.txt", 'r') as f:
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break
        audio = await provider.synthesize(chunk)
```

3. **Clear audio data**:
```python
# Explicitly delete when done
audio = await provider.synthesize("Text")
# Process audio
del audio
```

4. **Use generator**:
```python
def audio_generator():
    async for chunk in provider.synthesize_streaming("Text"):
        yield chunk
```

---

### Slow response times

**Problem**: Operations are slow

**Solutions**:

1. **Enable GPU**:
```python
config = KokoroConfig(use_gpu=True)
```

2. **Cache synthesized audio**:
```python
from functools import lru_cache

@lru_cache(maxsize=100)
async def cached_synthesize(text):
    return await provider.synthesize(text)
```

3. **Reduce text size**:
```python
# Summarize or segment text
```

4. **Check CPU/GPU load**:
```python
import psutil

print(f"CPU usage: {psutil.cpu_percent()}")
print(f"Memory: {psutil.virtual_memory().percent}%")
```

---

## Code Examples

### Error handling pattern

```python
async def safe_synthesize(text):
    """Safely synthesize with error handling"""
    try:
        async with get_provider() as provider:
            await provider.initialize()
            audio = await provider.synthesize(text)
            return audio
    except InitializationError as e:
        print(f"Initialization failed: {e}")
        return None
    except SynthesisError as e:
        print(f"Synthesis failed: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    finally:
        # Cleanup if needed
        pass
```

### Retry pattern

```python
import asyncio

async def retry_synthesize(text, max_retries=3):
    """Retry synthesis on failure"""
    for attempt in range(max_retries):
        try:
            async with get_provider() as provider:
                await provider.initialize()
                return await provider.synthesize(text)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(1)  # Wait before retry
```

### Progress tracking

```python
async def synthesize_with_progress(texts):
    """Synthesize with progress tracking"""
    total = len(texts)
    for i, text in enumerate(texts, 1):
        print(f"Progress: {i}/{total}")
        try:
            async with get_provider() as provider:
                await provider.initialize()
                audio = await provider.synthesize(text)
                await save_audio(audio, f"output_{i}.wav")
        except Exception as e:
            print(f"Failed on item {i}: {e}")
```

---

## Getting More Help

If you're still experiencing issues:

1. **Check the [FAQ](faq.md)** for common questions
2. **Review [examples](../examples/)** for working code
3. **Check [API documentation](api.md)** for API details
4. **[Open an issue on GitHub](https://github.com/divagnz/champi-tts/issues)** with details about your problem
5. **[Join GitHub Discussions](https://github.com/divagnz/champi-tts/discussions)** to ask questions

When opening an issue, include:
- Python version
- Operating system
- Error messages
- Minimal reproducible code
- Expected vs actual behavior

---

**Happy troubleshooting!**