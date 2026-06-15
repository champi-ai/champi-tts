# Python API Guide

Comprehensive guide to using the Champi TTS Python API for text-to-speech synthesis and text reading.

## Table of Contents

- [Getting Started](#getting-started)
- [Provider Management](#provider-management)
- [Text Reading Service](#text-reading-service)
- [Audio Features](#audio-features)
- [UI Indicator](#ui-indicator)
- [Event System](#event-system)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)

---

## Getting Started

### Basic Import

```python
from champi_tts import get_provider, get_reader
import asyncio
```

### Async Context Manager

Use async context managers for automatic cleanup:

```python
async with get_provider("kokoro") as provider:
    audio = await provider.synthesize("Hello, world!")
    # Automatic cleanup on exit
```

---

## Provider Management

### Getting a Provider

```python
from champi_tts import get_provider

# Default provider (Kokoro)
provider = get_provider()
await provider.initialize()
```

### Custom Configuration

```python
from champi_tts import get_provider
from champi_tts.providers.kokoro import KokoroConfig

config = KokoroConfig(
    default_voice="af_bella",
    default_speed=1.1,
    use_gpu=True
)

provider = get_provider("kokoro", config=config)
await provider.initialize()
```

### Using Config Parameters

```python
# Using keyword arguments
provider = get_provider(
    "kokoro",
    default_voice="af_bella",
    default_speed=1.1,
    use_gpu=True
)
```

### List Available Voices

```python
voices = await provider.list_voices()
print(f"Available voices: {voices}")
```

### Provider Properties

```python
print(f"Provider initialized: {provider.is_initialized}")
print(f"Currently speaking: {provider.is_speaking}")
```

---

## Text Reading Service

### Basic Text Reading

```python
from champi_tts import get_reader

reader = get_reader("kokoro")
await reader.read_text("Hello, this is a test")
```

### Reading Text Files

```python
reader = get_reader("kokoro")
await reader.read_file("document.txt")
```

### Reading with UI

```python
reader = get_reader("kokoro", show_ui=True)
await reader.read_file("document.txt")
```

### Queue Management

```python
# Add texts to queue
reader.add_to_queue("Text one")
reader.add_to_queue("Text two")
reader.add_to_queue("Text three")

# Read all queued texts
await reader.read_queue()
```

### Queue Control

```python
# Pause reading
await reader.pause()

# Resume reading
await reader.resume()

# Stop reading
await reader.stop()

# Clear queue
reader.clear_queue()
```

### Interrupt Reading

```python
# Stop immediately
await reader.interrupt()
```

---

## Audio Features

### Synthesizing Audio

```python
# Basic synthesis
audio = await provider.synthesize("Hello, world!")

# With voice
audio = await provider.synthesize("Hello", voice="af_bella")

# With speed
audio = await provider.synthesize("Hello", speed=1.2)
```

### Saving Audio

```python
from champi_tts.core.audio import save_audio

await save_audio(audio, "output.wav", sample_rate=22050)
```

### Loading Audio

```python
from champi_tts.core.audio import load_audio

audio = await load_audio("output.wav")
```

### Audio Player

```python
from champi_tts.core.audio import AudioPlayer

player = AudioPlayer(sample_rate=22050)

# Play audio
await player.play(audio, blocking=False)

# Stop playback
player.stop()

# Set volume
player.set_volume(0.8)
```

### Audio Normalization

```python
from champi_tts.core.audio import normalize_audio

normalized = normalize_audio(audio)
```

### Resampling Audio

```python
from champi_tts.core.audio import resample_audio

resampled = resample_audio(audio, from_rate=22050, to_rate=24000)
```

---

## UI Indicator

### Basic UI Usage

```python
from champi_tts import get_reader
from champi_tts.ui import TTSIndicatorUI, TTSState

# Get reader with UI
reader = get_reader("kokoro", show_ui=True)

# Update UI state
reader._ui.update_state(TTSState.SPEAKING, "Reading text...")
reader._ui.update_state(TTSState.PAUSED, "Paused")
reader._ui.update_state(TTSState.IDLE, "")
reader._ui.update_state(TTSState.ERROR, "Error occurred")
```

### Running UI Standalone

```python
from champi_tts.ui import run_standalone

# Run test UI
run_standalone()
```

---

## Event System

### Subscribing to Events

```python
# Reader events
reader.on_reading_started.connect(lambda sender, **kw: print(f"Started: {kw.get('text', '')}"))
reader.on_reading_completed.connect(lambda sender, **kw: print("Completed"))
reader.on_reading_paused.connect(lambda sender, **kw: print("Paused"))
reader.on_reading_resumed.connect(lambda sender, **kw: print("Resumed"))
reader.on_reading_stopped.connect(lambda sender, **kw: print("Stopped"))
reader.on_error.connect(lambda sender, **kw: print(f"Error: {kw.get('error', '')}"))
reader.on_state_changed.connect(lambda sender, **kw:
    print(f"State changed: {kw.get('old_state')} -> {kw.get('new_state')}"))
```

### Disabling Events

```python
# Disable all events
reader.on_reading_started.disconnect()

# Disable specific event
reader.on_reading_completed.disconnect()
```

### Custom Event Handlers

```python
def on_reading_started(sender, **kw):
    text = kw.get('text', '')
    print(f"Started reading: {text[:50]}...")
    # Save to log
    log_reading(text)

def on_error(sender, **kw):
    error = kw.get('error', 'Unknown error')
    print(f"Error: {error}")
    # Send notification
    send_error_notification(error)

# Connect handlers
reader.on_reading_started.connect(on_reading_started)
reader.on_error.connect(on_error)
```

---

## Advanced Usage

### Streaming Synthesis

```python
async def stream_synthesis(text):
    """Generate audio in streaming chunks"""
    async for chunk in provider.synthesize_streaming(text):
        # Process each chunk
        await process_audio_chunk(chunk)

# Usage
await stream_synthesis("Long text for streaming synthesis")
```

### Batch Processing

```python
texts = ["Text one", "Text two", "Text three"]
audio_list = []

for text in texts:
    audio = await provider.synthesize(text)
    audio_list.append(audio)

# Save all audio files
for i, audio in enumerate(audio_list):
    await save_audio(audio, f"output_{i}.wav")
```

### Error Handling

```python
try:
    audio = await provider.synthesize("Test text")
    await save_audio(audio, "test.wav")
except InitializationError as e:
    print(f"Provider initialization failed: {e}")
except SynthesisError as e:
    print(f"Synthesis failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    await provider.shutdown()
```

### Multiple Providers

```python
from champi_tts import get_provider

# Create multiple providers
provider_kokoro = get_provider("kokoro")
provider_openai = get_provider("openai")

await provider_kokoro.initialize()
await provider_openai.initialize()

# Use each provider
audio1 = await provider_kokoro.synthesize("Test")

# Switch providers
audio2 = await provider_openai.synthesize("Test")
```

### Custom Text Preprocessing

```python
import re

def preprocess_text(text):
    """Custom text preprocessing"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters
    text = re.sub(r'[^\w\s.,!?;]', '', text)
    return text.strip()

# Use preprocessing
processed_text = preprocess_text("Text with... special!!! characters??")
await provider.synthesize(processed_text)
```

---

## Best Practices

### Resource Management

```python
# Always use context managers
async with get_provider() as provider:
    audio = await provider.synthesize("Test")
    # Automatic cleanup

# Or explicit cleanup
provider = get_provider()
await provider.initialize()
try:
    # Use provider
    audio = await provider.synthesize("Test")
finally:
    await provider.shutdown()
```

### Error Handling

```python
# Always handle initialization errors
try:
    async with get_provider() as provider:
        await provider.initialize()
        # Use provider
except InitializationError:
    # Fallback to different provider
    async with get_provider("openai") as provider:
        # Use fallback
```

### Performance Optimization

```python
# Use GPU for faster synthesis (if available)
config = KokoroConfig(use_gpu=True)
provider = get_provider("kokoro", config=config)

# Reuse providers when possible
# Avoid creating new providers repeatedly

# For batch processing, consider queue management
```

### Code Organization

```python
# Keep your code organized with classes
class TextToSpeechService:
    def __init__(self, provider_type="kokoro"):
        self.provider = get_provider(provider_type)

    async def initialize(self):
        await self.provider.initialize()

    async def synthesize(self, text, voice=None):
        return await self.provider.synthesize(text, voice)

    async def shutdown(self):
        await self.provider.shutdown()

# Usage
async def main():
    service = TextToSpeechService()
    await service.initialize()
    audio = await service.synthesize("Hello, world!")
    await service.shutdown()
```

### Testing

```python
# Use fixtures for testing
import pytest

@pytest.mark.asyncio
async def test_synthesis():
    async with get_provider() as provider:
        audio = await provider.synthesize("Test")
        assert audio.shape is not None
        assert len(audio) > 0
```

---

## Common Patterns

### Quick Synthesis

```python
async def quick_synthesize(text, output_file):
    async with get_provider() as provider:
        await provider.initialize()
        audio = await provider.synthesize(text)
        await save_audio(audio, output_file)
```

### Text Reader with Logging

```python
async def text_reader_with_logging(file_path, voice="af_bella"):
    reader = get_reader("kokoro", show_ui=False)

    def log_started(sender, **kw):
        print(f"[LOG] Started reading: {kw.get('text', '')[:50]}...")

    reader.on_reading_started.connect(log_started)
    reader.on_reading_completed.connect(lambda s, **kw: print("[LOG] Completed"))
    reader.on_error.connect(lambda s, **kw: print(f"[LOG] Error: {kw.get('error')}"))

    await reader.read_file(file_path, voice=voice)
    await reader.stop()
```

### Audio Export Pipeline

```python
async def export_text_to_audio(text, output_path, voice="af_bella"):
    """Complete pipeline for text to audio export"""
    async with get_provider() as provider:
        # Initialize
        await provider.initialize()

        # Synthesize
        audio = await provider.synthesize(text, voice=voice)

        # Normalize
        from champi_tts.core.audio import normalize_audio
        audio = normalize_audio(audio)

        # Save
        await save_audio(audio, output_path, sample_rate=provider.config.sample_rate)

        # Cleanup
        await provider.shutdown()

    return output_path
```

---

**Need more help?** Check out the [API Reference](api.md) for complete API documentation or [Examples](../examples) for code samples.