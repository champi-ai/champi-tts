# Migration Guide: kokoro_svc → Champi TTS

Complete migration guide for moving from `mcp_champi.kokoro_svc` to `champi_tts`.

---

## Overview

Champi TTS is a modular, multi-provider text-to-speech library that provides:

- Multi-provider TTS support (Kokoro, with OpenAI/ElevenLabs support planned)
- Text reading service with pause/resume/stop controls
- Visual UI indicator using ImGui/GLFW
- CLI interface with rich terminal output
- Event-driven architecture for state tracking

This guide will help you migrate from the older `kokoro_svc` implementation.

---

## Quick Start

### Installation

```bash
pip install champi-tts
```

### Basic Usage

**Old Way (kokoro_svc):**

```python
from kokoro_svc import KokoroService

service = KokoroService()
await service.initialize()
audio = await service.synthesize("Hello, world!")
```

**New Way (Champi TTS):**

```python
from champi_tts import get_provider

provider = get_provider()
await provider.initialize()
audio = await provider.synthesize("Hello, world!")
```

---

## Installation Guide

### Requirements

- Python 3.12+
- GPU (optional, for faster synthesis)

### Install Command

```bash
# Basic installation
pip install champi-tts

# With development dependencies
pip install "champi-tts[dev]"

# With UI support
pip install "champi-tts[ui]"
```

### Configuration

Create a `.env` file for configuration:

```env
# Kokoro TTS settings
CHAMPI_KOKORO_DEFAULT_VOICE=af_bella
CHAMPI_KOKORO_DEFAULT_SPEED=1.1
CHAMPI_KOKORO_USE_GPU=true

# Model paths
CHAMPI_KOKORO_MODEL_PATH=/path/to/models
CHAMPI_KOKORO_CACHE_PATH=/path/to/cache
```

---

## API Migration

### Provider Instantiation

**Before (kokoro_svc):**

```python
from kokoro_svc import KokoroService, KokoroConfig

config = KokoroConfig(
    default_voice="af_bella",
    default_speed=1.1,
    model_path="./models/kokoro"
)

service = KokoroService(config)
```

**After (Champi TTS):**

```python
from champi_tts import get_provider
from champi_tts.providers.kokoro import KokoroConfig

config = KokoroConfig(
    default_voice="af_bella",
    default_speed=1.1,
    use_gpu=True
)

provider = get_provider("kokoro", config=config)
```

### Synthesis

**Before:**

```python
async def synthesize(service, text, voice=None, speed=1.0):
    audio = await service.synthesize(text=text, voice=voice, speed=speed)
    return audio
```

**After:**

```python
async def synthesize(provider, text, voice=None, speed=1.0):
    audio = await provider.synthesize(text=text, voice=voice, speed=speed)
    return audio
```

**Key Differences:**
- Method signature is identical
- Provider object replaces service object
- Same return type (numpy array)

### Voice Listing

**Before:**

```python
voices = await service.list_voices()
```

**After:**

```python
voices = await provider.list_voices()
```

### Reading Text Files

**Before (if implemented in kokoro_svc):**

```python
from kokoro_svc import KokoroService

service = KokoroService()
await service.initialize()

# Read and play text
await service.read_text("Hello world")
```

**After (Champi TTS):**

```python
from champi_tts import get_reader

reader = get_reader("kokoro")
await reader.read_text("Hello world")

# With file reading
await reader.read_file("document.txt")

# With pause/resume
await reader.pause()
await reader.resume()

# Stop reading
await reader.stop()
```

### Event Handling

**Before:**

```python
@service.on_started
def on_started(text):
    print(f"Started: {text}")

@service.on_completed  
def on_completed():
    print("Completed")
```

**After:**

```python
from champi_tts import get_reader

reader = get_reader("kokoro")

# Subscribe to events
reader.on_reading_started.connect(lambda **kw: print(f"Started: {kw.get('text')}"))
reader.on_reading_completed.connect(lambda **kw: print("Completed"))
reader.on_error.connect(lambda **kw: print(f"Error: {kw.get('error')}"))
```

---

## Breaking Changes

### 1. Class Structure

**Old:** `KokoroService` inherited directly from base class

**New:** Uses adapter pattern for type compliance

```python
# Old direct inheritance
class KokoroService(BaseService):
    pass

# New adapter pattern
class KokoroProviderAdapter(BaseTTSProvider):
    def __init__(self, config):
        self._kokoro = OriginalKokoroProvider(config)
```

**Impact:** None for users - the adapter makes KokoroProvider conform to the base interface.

### 2. Async Context Manager

**New:** Added async context manager support

```python
# New way (recommended)
async with get_provider("kokoro") as provider:
    audio = await provider.synthesize("Hello")
# Automatic cleanup on exit
```

### 3. Reader Service

**New:** Full reader service with queue management

```python
# New features
reader.add_to_queue("Text 1")
reader.add_to_queue("Text 2")
await reader.read_queue()  # Read all queued texts
reader.clear_queue()
```

---

## Configuration Migration

### Environment Variables

**Old (.env for kokoro_svc):**

```env
KOKORO_DEFAULT_VOICE=af_bella
KOKORO_DEFAULT_SPEED=1.1
KOKORO_USE_GPU=true
KOKORO_MODEL_PATH=/models
KOKORO_CACHE_PATH=/cache
```

**New (.env for Champi TTS):**

```env
# Same keys work, prefixed with CHAMPI_
CHAMPI_KOKORO_DEFAULT_VOICE=af_bella
CHAMPI_KOKORO_DEFAULT_SPEED=1.1
CHAMPI_KOKORO_USE_GPU=true
CHAMPI_KOKORO_MODEL_PATH=/models
CHAMPI_KOKORO_CACHE_PATH=/cache
```

### Config Class

**Old:**

```python
class KokoroConfig(BaseConfig):
    default_voice: str = "af_bella"
    default_speed: float = 1.0
    model_path: str = "./models/kokoro"
```

**New:**

```python
class KokoroConfig(BaseTTSConfig):
    default_voice: str = "af_bella"
    default_speed: float = 1.0
    use_gpu: bool = False
```

---

## Audio Processing

### Playback

**Old:**

```python
from kokoro_svc import AudioPlayer

player = AudioPlayer(sample_rate=22050)
await player.play(audio)
```

**New:**

```python
from champi_tts.core.audio import AudioPlayer

player = AudioPlayer(sample_rate=22050)
await player.play(audio, blocking=False)
```

### Saving Audio

**New:**

```python
from champi_tts.core.audio import save_audio

await save_audio(audio, "output.wav", sample_rate=22050)
```

---

## Voice Files

### Setup

Voice files are included in the package. No additional download needed for default voices.

### Custom Voices

To add custom voices:

```python
from champi_tts.providers.kokoro import VoiceManager

vm = VoiceManager(model_path="./models")
vm.download_voice("custom_voice")
```

---

## Code Examples

### Complete Migration Example

**Before (kokoro_svc):**

```python
from kokoro_svc import KokoroService, KokoroConfig

async def main():
    config = KokoroConfig(
        default_voice="af_bella",
        default_speed=1.1
    )
    service = KokoroService(config)
    await service.initialize()
    
    try:
        # Synthesize
        audio = await service.synthesize("Hello, world!")
        await service.play_audio(audio)
        
        # Read file
        await service.read_file("document.txt")
        
        # Listen for events
        for event in service.on_events():
            if event.type == "completed":
                print("Done")
    finally:
        await service.shutdown()
```

**After (Champi TTS):**

```python
from champi_tts import get_provider, get_reader
from champi_tts.core.audio import save_audio
from champi_tts.ui import TTSIndicatorUI, TTSState

async def main():
    # Get provider
    provider = get_provider()
    
    async with provider:
        try:
            # Synthesize
            audio = await provider.synthesize("Hello, world!")
            await save_audio(audio, "hello.wav")
            
            # Create reader with UI
            reader = get_reader("kokoro", show_ui=True)
            
            # Subscribe to events
            def on_started(sender, **kw):
                print(f"Started: {kw.get('text', '')[:50]}. ..")
            
            def on_completed(sender, **kw):
                print("Reading completed")
            
            def on_error(sender, **kw):
                print(f"Error: {kw.get('error', '')}")
            
            reader.on_reading_started.connect(on_started)
            reader.on_reading_completed.connect(on_completed)
            reader.on_error.connect(on_error)
            
            # Read text
            await reader.read_text("This is a test of Champi TTS")
            
            # Queue management
            reader.add_to_queue("Second paragraph")
            await reader.read_queue()
            
        except Exception as e:
            print(f"Error: {e}")
            if reader._ui:
                reader._ui.update_state(TTSState.ERROR, str(e))
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'champi_tts'"

**Solution:**

```bash
pip install -e ".[dev]"
```

### Issue: "Voice not found: af_bella"

**Solution:**

Ensure model files are downloaded:

```bash
champi-tts download-voices
```

### Issue: "CUDA out of memory"

**Solution:**

Disable GPU or use smaller batch size:

```python
config = KokoroConfig(use_gpu=False)
```

### Issue: UI not showing

**Solution:**

UI dependencies are optional:

```bash
pip install "champi-tts[ui]"
```

Or disable UI:

```python
reader = get_reader("kokoro", show_ui=False)
```

---

## Performance Considerations

### Lazy Loading

Heavy dependencies are loaded on first use:

```python
# torch and kokoro loaded on first synthesis() call
# Not on import
```

### GPU Acceleration

Enable GPU for faster synthesis:

```python
config = KokoroConfig(use_gpu=True)
```

### Caching

Synthesized audio can be cached:

```python
config = KokoroConfig(
    cache_path="./cache",
    cache_ttl=3600  # Cache for 1 hour
)
```

---

## Testing Your Migration

### Run Tests

```bash
pytest tests/ -v
```

### Run Examples

```bash
# Basic synthesis
champi-tts synthesize "Hello, world!"

# Read file
champi-tts read document.txt
```

### Run UI Test

```bash
champi-tts test-ui
```

---

## Resources

- **API Reference:** `docs/api.md`
- **Architecture:** `ARCHITECTURE.md`
- **Examples:** `examples/` directory
- **GitHub:** https://github.com/champi-ai/champi-tts

---

## Support

If you encounter issues during migration:

1. Check the examples in `examples/`
2. Review the API reference in `docs/api.md`
3. Open an issue on GitHub
4. Check existing issues for similar problems

---

**Migration Complete!** 🎉

Your code should now use Champi TTS. Enjoy the new features and improvements!
