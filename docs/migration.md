# Migration Guide: mcp_champi.kokoro_svc to champi_tts

Complete migration guide for moving from `mcp_champi.kokoro_svc` to `champi_tts`.

---

## Overview

`champi_tts` is a modular, multi-provider text-to-speech library that replaces the
tightly-coupled `mcp_champi.kokoro_svc` with a clean provider/adapter architecture.

Key improvements:

- **Multi-provider support**: Kokoro today, OpenAI and ElevenLabs coming soon
- **Unified interface**: `BaseTTSProvider` contract across all backends
- **Full reader service**: pause, resume, stop, queue management, and interruption
- **Async context manager**: automatic lifecycle management
- **Streaming synthesis**: chunk-by-chunk audio generation
- **Visual UI indicator**: optional GLFW/ImGui overlay for reading status
- **CLI tool**: `champi-tts` command for synthesis and file reading
- **Lazy loading**: heavy dependencies (torch, kokoro) loaded on first use

---

## Quick Start

### Installation

```bash
pip install champi-tts
```

### Minimal migration

**Before (`mcp_champi.kokoro_svc`):**

```python
from mcp_champi.kokoro_svc import KokoroService

service = KokoroService()
await service.initialize()
audio = await service.synthesize("Hello, world!")
```

**After (`champi_tts`):**

```python
from champi_tts import get_provider

provider = get_provider()
await provider.initialize()
audio = await provider.synthesize("Hello, world!")
await provider.shutdown()
```

Or with automatic lifecycle management:

```python
from champi_tts import get_provider

async with get_provider() as provider:
    audio = await provider.synthesize("Hello, world!")
```

---

## Installation Guide

### Requirements

- Python 3.12+
- GPU (optional, for faster synthesis)

### Install options

```bash
# Basic installation
pip install champi-tts

# With development dependencies
pip install "champi-tts[dev]"

# With UI support (GLFW/ImGui)
pip install "champi-tts[ui]"

# Development install from source
git clone https://github.com/champi-ai/champi-tts.git
cd champi-tts
uv sync --extra dev
```

---

## API Migration

### Provider instantiation

**Before:**

```python
from mcp_champi.kokoro_svc import KokoroService, KokoroConfig

config = KokoroConfig(
    default_voice="af_bella",
    default_speed=1.1,
    model_path="./models/kokoro"
)
service = KokoroService(config)
await service.initialize()
```

**After:**

```python
from champi_tts import get_provider
from champi_tts.providers.kokoro import KokoroConfig

config = KokoroConfig(
    default_voice="am_adam",
    default_speed=1.1,
    use_gpu=True,
    model_dir="./models/kokoro"   # was model_path
)
provider = get_provider("kokoro", config=config)
await provider.initialize()
```

Config fields can also be passed directly as kwargs:

```python
provider = get_provider("kokoro", default_voice="am_adam", use_gpu=True)
```

### Synthesis

**Before:**

```python
audio = await service.synthesize(text="Hello, world!", voice="af_bella", speed=1.0)
```

**After:**

```python
audio = await provider.synthesize(text="Hello, world!", voice="am_adam", speed=1.0)
```

The method signature is identical. The return type is a `numpy.ndarray` in both cases.

### Streaming synthesis (new)

`mcp_champi.kokoro_svc` did not support streaming. `champi_tts` adds
`synthesize_streaming` which yields audio chunks as they are generated:

```python
async for chunk in provider.synthesize_streaming("A long piece of text..."):
    # process or play each chunk immediately
    pass
```

### Listing voices

**Before:**

```python
voices = await service.list_voices()
```

**After:**

```python
voices = await provider.list_voices()
```

The return type is `list[str]`. Voice names follow the `<language_gender>_<name>` pattern:

| Prefix | Language / Gender |
|--------|-------------------|
| `af_`  | American English, Female |
| `am_`  | American English, Male |
| `bf_`  | British English, Female |
| `bm_`  | British English, Male |
| `ef_`  | Spanish, Female |
| `em_`  | Spanish, Male |
| `ff_`  | French, Female |

Default voice changed from `af_bella` to `am_adam`.

### Interruption

**Before:**

```python
await service.cancel()
```

**After:**

```python
await provider.interrupt()
```

### Shutdown

**Before:**

```python
await service.shutdown()
```

**After:**

```python
await provider.shutdown()
# or use the async context manager for automatic shutdown
```

---

## Reader Service Migration

`mcp_champi.kokoro_svc` provided basic `read_text` and `read_file` methods on
`KokoroService`. `champi_tts` separates this concern into `TextReaderService` with
full queue management and event signals.

### Reading text

**Before:**

```python
from mcp_champi.kokoro_svc import KokoroService

service = KokoroService()
await service.initialize()
await service.read_text("Hello world")
```

**After:**

```python
from champi_tts import get_reader

reader = get_reader("kokoro")
await reader.provider.initialize()
await reader.read_text("Hello world")
await reader.provider.shutdown()
```

Or with automatic lifecycle via the async context manager:

```python
from champi_tts import get_reader

async with get_reader("kokoro") as reader:
    await reader.read_text("Hello world")
```

### Reading files

**Before:**

```python
await service.read_file("document.txt")
```

**After:**

```python
await reader.read_file("document.txt", voice="am_adam")
```

Files are split on double newlines and read paragraph by paragraph, with a 0.5 s
pause between paragraphs.

### Playback control

**Before:**

```python
await service.pause()
await service.resume()
await service.stop()
```

**After:**

```python
await reader.pause()
await reader.resume()
await reader.stop()
```

### Queue management (new)

`mcp_champi.kokoro_svc` had no queue. `champi_tts` adds a text queue:

```python
reader.add_to_queue("First paragraph")
reader.add_to_queue("Second paragraph")
await reader.read_queue(voice="am_adam")

# Clear the queue at any time
reader.clear_queue()
```

### Reader state

```python
from champi_tts import ReaderState

if reader.state == ReaderState.READING:
    print("Currently reading")
elif reader.state == ReaderState.PAUSED:
    print("Paused")
```

Available states: `IDLE`, `READING`, `PAUSED`, `STOPPED`.

---

## Event Handling Migration

`mcp_champi.kokoro_svc` used decorator-based callbacks. `champi_tts` uses
[blinker](https://blinker.readthedocs.io/) signals.

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

def on_started(sender, **kw):
    print(f"Started: {kw.get('text', '')}")

def on_completed(sender, **kw):
    print("Completed")

def on_error(sender, **kw):
    print(f"Error: {kw.get('error', '')}")

reader.on_reading_started.connect(on_started)
reader.on_reading_completed.connect(on_completed)
reader.on_error.connect(on_error)
```

All available signals on `TextReaderService`:

| Signal | Kwargs |
|--------|--------|
| `on_reading_started` | `text` |
| `on_reading_paused` | — |
| `on_reading_resumed` | — |
| `on_reading_stopped` | — |
| `on_reading_completed` | `text` |
| `on_state_changed` | `old_state`, `new_state` |
| `on_error` | `error` |

---

## Configuration Migration

### Environment variables

**Before (`mcp_champi.kokoro_svc`):**

```env
KOKORO_DEFAULT_VOICE=af_bella
KOKORO_DEFAULT_SPEED=1.1
KOKORO_USE_GPU=true
KOKORO_MODEL_PATH=/models
KOKORO_CACHE_PATH=/cache
```

**After (`champi_tts`):**

```env
KOKORO_DEFAULT_VOICE=am_adam
KOKORO_DEFAULT_SPEED=1.1
KOKORO_USE_GPU=true
KOKORO_FORCE_CPU=false
KOKORO_DEFAULT_LANGUAGE=a
KOKORO_MODEL_DIR=/models
KOKORO_VOICE_DIR=/voices
KOKORO_CACHE_DIR=/cache
KOKORO_NORMALIZE_TEXT=true
KOKORO_WARMUP_ON_INIT=true
KOKORO_AUTO_DOWNLOAD=true
```

Load from environment explicitly:

```python
from champi_tts.providers.kokoro import KokoroConfig

config = KokoroConfig.from_env()
provider = get_provider("kokoro", config=config)
```

### Config class changes

| Old field (`KokoroConfig`) | New field (`KokoroConfig`) | Notes |
|---------------------------|---------------------------|-------|
| `model_path` | `model_dir` | Directory, not file path |
| `cache_path` | `cache_dir` | Directory, not file path |
| `voice_path` | `voice_dir` | Directory for `.pt` files |
| — | `default_language` | Language code, default `"a"` (US English) |
| — | `force_cpu` | Explicitly disable GPU even if available |
| — | `advanced_normalization` | Extended text normalization pipeline |
| — | `memory_management` | Enable memory management optimisations |
| — | `max_text_length` | Maximum characters per synthesis call |

Language codes for `default_language`:

| Code | Language |
|------|----------|
| `a` | American English (default) |
| `b` | British English |
| `e` | Spanish |
| `f` | French |
| `h` | Hindi |
| `i` | Italian |
| `j` | Japanese |
| `p` | Brazilian Portuguese |
| `z` | Mandarin Chinese |

### Configuration presets (new)

```python
from champi_tts.providers.kokoro import KokoroConfig

# High performance (GPU, warmup, advanced normalization)
config = KokoroConfig.performance()

# High quality (GPU, longer chunks, retry on failure)
config = KokoroConfig.quality()

# CPU only (no GPU, no warmup, smaller chunks)
config = KokoroConfig.cpu_only()

# Minimal resources (CPU, no normalization, no auto-download)
config = KokoroConfig.minimal()
```

---

## Audio Processing Migration

### Saving audio

**Before:**

```python
from mcp_champi.kokoro_svc import AudioPlayer

player = AudioPlayer(sample_rate=22050)
await player.play(audio)
```

**After:**

```python
from champi_tts.core.audio import AudioPlayer

player = AudioPlayer(sample_rate=24000)  # Kokoro outputs at 24 kHz
await player.play(audio, blocking=True)
```

The default sample rate changed from 22050 Hz to 24000 Hz.

### Saving audio to file

**Before:**

There was no built-in save function.

**After:**

```python
from champi_tts.core.audio import save_audio

await save_audio(audio, "output.wav", sample_rate=provider.config.sample_rate)
```

Supported formats: `wav`, `mp3`, `opus`, `flac`, `pcm`.

---

## Complete Migration Example

**Before (`mcp_champi.kokoro_svc`):**

```python
import asyncio
from mcp_champi.kokoro_svc import KokoroService, KokoroConfig

async def main():
    config = KokoroConfig(
        default_voice="af_bella",
        default_speed=1.1
    )
    service = KokoroService(config)
    await service.initialize()

    try:
        audio = await service.synthesize("Hello, world!")
        await service.play_audio(audio)
        await service.read_file("document.txt")

        for event in service.on_events():
            if event.type == "completed":
                print("Done")
    finally:
        await service.shutdown()

asyncio.run(main())
```

**After (`champi_tts`):**

```python
import asyncio
from champi_tts import get_provider, get_reader
from champi_tts.core.audio import save_audio

async def main():
    # Synthesize and save
    async with get_provider() as provider:
        audio = await provider.synthesize("Hello, world!")
        await save_audio(audio, "hello.wav", sample_rate=provider.config.sample_rate)

    # Read a file with event callbacks
    async with get_reader("kokoro") as reader:
        def on_started(sender, **kw):
            print(f"Started: {kw.get('text', '')[:50]}...")

        def on_completed(sender, **kw):
            print("Reading completed")

        def on_error(sender, **kw):
            print(f"Error: {kw.get('error', '')}")

        reader.on_reading_started.connect(on_started)
        reader.on_reading_completed.connect(on_completed)
        reader.on_error.connect(on_error)

        await reader.read_file("document.txt", voice="am_adam")

asyncio.run(main())
```

---

## CLI Migration

`mcp_champi.kokoro_svc` had no CLI. `champi_tts` ships the `champi-tts` command:

```bash
# Synthesize and play
champi-tts synthesize "Hello, world!" --voice am_adam

# Synthesize and save to file (no playback)
champi-tts synthesize "Hello, world!" --output hello.wav --no-play

# Read a text file
champi-tts read document.txt --voice am_adam

# Read with visual UI indicator
champi-tts read document.txt --show-ui

# Read inline text
champi-tts read --text "This is a test" --voice am_adam

# Interactive mode (SPACE to pause/resume, S to stop, Q to quit)
champi-tts read document.txt --interactive --show-ui

# List available voices
champi-tts list-voices

# Show version
champi-tts version

# Test the UI indicator
champi-tts test-ui
```

---

## Breaking Changes

### 1. Import path

```python
# Before
from mcp_champi.kokoro_svc import KokoroService

# After
from champi_tts import get_provider
```

### 2. Class name

`KokoroService` is replaced by `KokoroProvider` (accessed via `get_provider()`).
Direct instantiation is possible but the factory function is recommended:

```python
# Factory (recommended)
from champi_tts import get_provider
provider = get_provider("kokoro")

# Direct (not recommended for user code)
from champi_tts.providers.kokoro import KokoroProvider
from champi_tts.providers.kokoro.adapter import KokoroProviderAdapter
```

### 3. Config field names

`model_path` and `cache_path` are renamed to `model_dir` and `cache_dir` to
reflect that they are directory paths, not file paths.

### 4. Default voice

The default voice changed from `af_bella` to `am_adam`. Set `default_voice`
explicitly if you rely on a specific voice:

```python
config = KokoroConfig(default_voice="af_bella")
```

### 5. Default sample rate

Sample rate changed from 22050 Hz to 24000 Hz. Update any downstream audio
processing that assumed the old rate.

### 6. Event system

Decorator-based callbacks (`@service.on_started`) are replaced by blinker
signals (`.connect(handler)`). Signal handlers receive `sender` as their first
positional argument plus keyword arguments:

```python
def handler(sender, **kwargs):
    ...

reader.on_reading_started.connect(handler)
```

### 7. Explicit provider initialization

`get_reader()` does not auto-initialize the provider. Call
`await reader.provider.initialize()` before reading, or use the async context
manager which handles this automatically:

```python
async with get_reader("kokoro") as reader:
    await reader.read_text("Hello")
```

---

## Model and Voice Setup

Kokoro model files are downloaded automatically on first use when
`auto_download_model=True` (the default). To download manually:

```python
from champi_tts.providers.kokoro import ModelDownloader

model_path, config_path = ModelDownloader.download_model(output_dir="./models/kokoro")
```

Voice files (`.pt` tensors) live in the voice directory. Set it explicitly or
let the provider use its default:

```python
from champi_tts.providers.kokoro import KokoroConfig, VoiceManager

config = KokoroConfig(voice_dir="./voices")
VoiceManager.setup_voice_directory("./voices")

# List voices in a directory
voices = VoiceManager.list_voices("./voices")
```

---

## Troubleshooting

### ModuleNotFoundError: No module named 'champi_tts'

Install the package:

```bash
pip install champi-tts
# or for development
pip install -e ".[dev]"
```

### Voice not found

List the voices that are available to your provider:

```bash
champi-tts list-voices
```

Or programmatically:

```python
voices = await provider.list_voices()
print(voices)
```

If the voice directory is empty, download model files first:

```python
from champi_tts.providers.kokoro import ModelDownloader
ModelDownloader.download_model(output_dir="./models/kokoro")
```

### CUDA out of memory

Disable GPU acceleration:

```python
from champi_tts.providers.kokoro import KokoroConfig
config = KokoroConfig(use_gpu=False, force_cpu=True)
provider = get_provider("kokoro", config=config)
```

Or use the `cpu_only` preset:

```python
from champi_tts.providers.kokoro import KokoroConfigPresets
config = KokoroConfigPresets.cpu_only()
```

### UI not showing

Install optional UI dependencies:

```bash
pip install "champi-tts[ui]"
```

Or disable UI entirely:

```python
reader = get_reader("kokoro", show_ui=False)
```

### AttributeError on KokoroConfig fields

Check that you are using the new field names:

| Old field | New field |
|-----------|-----------|
| `model_path` | `model_dir` |
| `cache_path` | `cache_dir` |
| `voice_path` | `voice_dir` |

---

## Resources

- **API Reference**: `docs/api.md`
- **User Guide**: `docs/USER_GUIDE.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`
- **Examples**: `examples/` directory
- **GitHub**: https://github.com/champi-ai/champi-tts
