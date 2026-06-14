# Champi TTS API Reference

Complete API documentation with examples for all public classes and methods.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Factory Functions](#factory-functions)
- [Base Classes](#base-classes)
- [Kokoro Provider](#kokoro-provider)
- [Reader Service](#reader-service)
- [UI Components](#ui-components)
- [Audio Utilities](#audio-utilities)
- [Exceptions](#exceptions)
- [Event System](#event-system)
- [Examples](#examples)

---

## Quick Start

```python
import asyncio
from champi_tts import get_provider, get_reader

async def main():
    # Synthesize audio
    async with get_provider("kokoro") as provider:
        audio = await provider.synthesize("Hello, world!")

    # Read text aloud
    async with get_reader("kokoro", show_ui=True) as reader:
        await reader.read_text("Hello from Champi TTS")

asyncio.run(main())
```

---

## Factory Functions

All factory functions are importable directly from `champi_tts`.

```python
from champi_tts import get_provider, get_reader, get_default_provider, list_providers
```

### `get_provider()`

Create a TTS provider instance. Uses lazy loading to reduce startup time.

```python
def get_provider(
    provider_type: Literal["kokoro"] = "kokoro",
    config: BaseTTSConfig | None = None,
    **config_kwargs,
) -> BaseTTSProvider
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider_type` | `Literal["kokoro"]` | `"kokoro"` | Provider to create |
| `config` | `BaseTTSConfig \| None` | `None` | Pre-built config object (optional) |
| `**config_kwargs` | `Any` | — | Config field values when `config` is not provided |

**Returns** `BaseTTSProvider` — uninitialized provider. Call `await provider.initialize()` or use it as an async context manager before synthesizing.

**Raises** `ValueError` if `provider_type` is not recognized.

**Examples**

```python
from champi_tts import get_provider

# Default provider — must initialize before use
provider = get_provider()
await provider.initialize()
audio = await provider.synthesize("Hello, world!")
await provider.shutdown()

# Async context manager — initialize/shutdown handled automatically
async with get_provider("kokoro") as provider:
    audio = await provider.synthesize("Hello, world!")

# Custom config object
from champi_tts.providers.kokoro import KokoroConfig

config = KokoroConfig(default_voice="af_bella", default_speed=1.1, use_gpu=True)
provider = get_provider("kokoro", config=config)

# Inline config kwargs
provider = get_provider("kokoro", default_voice="af_bella", use_gpu=False)
```

---

### `get_reader()`

Create a `TextReaderService` backed by a provider of the specified type.

```python
def get_reader(
    provider_type: Literal["kokoro"] = "kokoro",
    show_ui: bool = False,
    config: BaseTTSConfig | None = None,
    **config_kwargs,
) -> TextReaderService
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider_type` | `Literal["kokoro"]` | `"kokoro"` | Underlying provider type |
| `show_ui` | `bool` | `False` | Show the ImGui TTS status indicator |
| `config` | `BaseTTSConfig \| None` | `None` | Pre-built config object (optional) |
| `**config_kwargs` | `Any` | — | Config field values passed to the provider |

**Returns** `TextReaderService` — must be initialized (call `await reader.initialize()` or use as an async context manager) before reading.

**Examples**

```python
from champi_tts import get_reader

# Basic reader, no UI
async with get_reader("kokoro") as reader:
    await reader.read_file("document.txt")

# Reader with visual status indicator
async with get_reader("kokoro", show_ui=True) as reader:
    await reader.read_text("This is spoken aloud with UI feedback")

# Custom voice and speed
async with get_reader("kokoro", default_voice="am_adam", default_speed=0.9) as reader:
    await reader.read_text("Slower voice")
```

---

### `get_default_provider()`

Return a `KokoroProviderAdapter` with default configuration.

```python
def get_default_provider() -> BaseTTSProvider
```

**Returns** `BaseTTSProvider` — equivalent to `get_provider("kokoro")`.

```python
from champi_tts import get_default_provider

provider = get_default_provider()
await provider.initialize()
```

---

### `list_providers()`

Return the names of all available providers.

```python
def list_providers() -> list[str]
```

**Returns** `list[str]` — currently `["kokoro"]`.

```python
from champi_tts import list_providers

providers = list_providers()
print(providers)  # ['kokoro']
```

---

## Base Classes

The abstract base classes define the interface that all providers must implement. Import them for type hints or to build a custom provider.

```python
from champi_tts import BaseTTSConfig, BaseTTSProvider, BaseSynthesizer
```

---

### `BaseTTSConfig`

Abstract dataclass that holds common TTS settings. Subclass it to add provider-specific fields.

```python
@dataclass
class BaseTTSConfig(ABC):
    sample_rate: int = 24000
    audio_format: str = "wav"
    default_voice: str = ""
    default_speed: float = 1.0
    normalize_text: bool = True
    enable_streaming: bool = True
    streaming_chunk_size: int = 200
    warmup_on_init: bool = True
    auto_download_model: bool = True
```

**Fields**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sample_rate` | `int` | `24000` | Audio sample rate in Hz |
| `audio_format` | `str` | `"wav"` | Default output audio format |
| `default_voice` | `str` | `""` | Voice name used when no voice is specified |
| `default_speed` | `float` | `1.0` | Speech speed multiplier (`0.1` – `3.0`) |
| `normalize_text` | `bool` | `True` | Pre-process text before synthesis |
| `enable_streaming` | `bool` | `True` | Allow streaming synthesis |
| `streaming_chunk_size` | `int` | `200` | Characters per streaming chunk |
| `warmup_on_init` | `bool` | `True` | Run warm-up synthesis on initialization |
| `auto_download_model` | `bool` | `True` | Download model if not found locally |

**Abstract methods** that concrete subclasses must implement:

| Method | Signature | Description |
|--------|-----------|-------------|
| `from_env` | `classmethod -> BaseTTSConfig` | Build config from environment variables |
| `validate` | `() -> bool` | Validate all configuration values |

**Instance methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `to_dict` | `() -> dict[str, Any]` | Serialize config to a plain dictionary |

---

### `BaseTTSProvider`

Abstract base class for TTS providers. Implements the async context manager protocol so providers can be used with `async with`.

**Constructor**

```python
def __init__(self, config: BaseTTSConfig) -> None
```

**Abstract methods**

#### `async initialize()`

Load the model and prepare the provider for synthesis. Must be called before `synthesize()` unless using the provider as an async context manager.

```python
async def initialize(self) -> None
```

```python
provider = get_provider()
await provider.initialize()
```

#### `async shutdown()`

Release all resources held by the provider.

```python
async def shutdown(self) -> None
```

```python
await provider.shutdown()
```

#### `async synthesize()`

Synthesize text and return the complete audio array.

```python
async def synthesize(
    self,
    text: str,
    voice: str | None = None,
    speed: float | None = None,
    **kwargs: Any,
) -> np.ndarray
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text to convert to speech |
| `voice` | `str \| None` | Voice name; uses `config.default_voice` when `None` |
| `speed` | `float \| None` | Speed multiplier; uses `config.default_speed` when `None` |
| `**kwargs` | `Any` | Provider-specific parameters |

**Returns** `np.ndarray` — floating-point audio array at `config.sample_rate`.

```python
audio = await provider.synthesize("Hello, world!")
audio = await provider.synthesize("Faster", speed=1.5)
audio = await provider.synthesize("Different voice", voice="af_bella")
```

#### `async synthesize_streaming()`

Synthesize text and yield audio in chunks as they become available.

```python
async def synthesize_streaming(
    self,
    text: str,
    voice: str | None = None,
    speed: float | None = None,
    **kwargs: Any,
) -> AsyncGenerator[np.ndarray, None]
```

**Yields** `np.ndarray` — successive audio chunks.

```python
async for chunk in provider.synthesize_streaming("A long piece of text..."):
    process_chunk(chunk)
```

#### `async list_voices()`

Return all voice names available to the provider.

```python
async def list_voices(self) -> list[str]
```

**Returns** `list[str]` — e.g. `["af_bella", "am_adam", "bf_emma"]`.

```python
voices = await provider.list_voices()
print(voices)
```

#### `async interrupt()`

Immediately stop any synthesis or playback in progress.

```python
async def interrupt(self) -> None
```

```python
await provider.interrupt()
```

**Properties**

| Property | Type | Description |
|----------|------|-------------|
| `is_initialized` | `bool` | `True` after `initialize()` has succeeded |
| `is_speaking` | `bool` | `True` while audio is being played |
| `config` | `BaseTTSConfig` | The configuration object passed at construction |

**Async context manager**

```python
async with get_provider("kokoro") as provider:
    audio = await provider.synthesize("Hello!")
# shutdown() called automatically
```

---

### `BaseSynthesizer`

Low-level abstract interface for the synthesis engine. Providers delegate to a `BaseSynthesizer` internally. Implement this to plug in a new synthesis backend.

**Abstract methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `load_model` | `async () -> None` | Load model weights into memory |
| `unload_model` | `async () -> None` | Release model weights |
| `synthesize_audio` | `async (text, voice_data, speed, **kwargs) -> np.ndarray` | Synthesize audio from text |
| `synthesize_streaming` | `async (text, voice_data, speed, **kwargs) -> AsyncGenerator[np.ndarray, None]` | Stream audio chunks |
| `preprocess_text` | `async (text) -> str` | Normalize text before synthesis |

**Properties**

| Property | Type | Description |
|----------|------|-------------|
| `is_loaded` | `bool` | `True` when the model is in memory |

---

## Kokoro Provider

The Kokoro provider is the built-in local neural TTS backend. It is a singleton — only one instance exists per process.

```python
from champi_tts.providers.kokoro import (
    KokoroConfig,
    KokoroConfigPresets,
    KokoroProvider,
    VoiceLanguage,
)
```

---

### `KokoroConfig`

Dataclass with all Kokoro-specific configuration.

```python
@dataclass
class KokoroConfig:
    # Device
    use_gpu: bool = True
    force_cpu: bool = False

    # Model
    default_language: str = "a"   # "a" = en-us, "b" = en-gb
    default_speed: float = 1.0
    default_voice: str = "am_adam"

    # Text processing
    normalize_text: bool = True
    advanced_normalization: bool = True

    # Streaming
    streaming_chunk_size: int = 200
    enable_streaming: bool = True

    # Audio
    sample_rate: int = 24000
    audio_format: str = "wav"
    tts_audio_format: str = "pcm"  # Internal format for TTS output

    # Service
    available_voices: list = ["am_adam"]
    available_models: list = ["tts-1"]
    auto_start: bool = False

    # Initialization
    warmup_on_init: bool = True
    auto_download_model: bool = True

    # Directories (auto-set when None)
    model_dir: str | None = None
    voice_dir: str | None = None
    cache_dir: str | None = None

    # Advanced
    memory_management: bool = True
    max_text_length: int = 100000
    retry_on_failure: bool = True
```

**Field reference**

| Field | Default | Description |
|-------|---------|-------------|
| `use_gpu` | `True` | Use CUDA/MPS when available |
| `force_cpu` | `False` | Disable GPU even when available |
| `default_language` | `"a"` | Language code (`"a"` = en-us, `"b"` = en-gb, etc.) |
| `default_speed` | `1.0` | Speech speed (`0.1` – `3.0`) |
| `default_voice` | `"am_adam"` | Voice name used when none is specified |
| `normalize_text` | `True` | Basic text normalization |
| `advanced_normalization` | `True` | Extended normalization pipeline |
| `streaming_chunk_size` | `200` | Characters per streaming chunk (min 50) |
| `enable_streaming` | `True` | Allow streaming synthesis |
| `sample_rate` | `24000` | Output audio sample rate in Hz |
| `audio_format` | `"wav"` | Saved file format |
| `tts_audio_format` | `"pcm"` | Internal synthesis format (`pcm`, `wav`, `mp3`, `opus`, `flac`) |
| `warmup_on_init` | `True` | Run one synthesis pass after loading to prime the model |
| `auto_download_model` | `True` | Download model weights if not present locally |
| `model_dir` | `None` | Override model directory path |
| `voice_dir` | `None` | Override voice directory path |
| `cache_dir` | `None` | Override disk cache directory path |
| `memory_management` | `True` | Enable memory optimization |
| `max_text_length` | `100000` | Hard limit on input text length in characters |
| `retry_on_failure` | `True` | Retry synthesis once on transient errors |

**Class methods**

| Method | Description |
|--------|-------------|
| `from_env() -> KokoroConfig` | Build from environment variables (see below) |
| `from_dict(d) -> KokoroConfig` | Build from a dictionary (unknown keys are ignored) |
| `from_file(path) -> KokoroConfig` | Load from a JSON file |

**Environment variables read by `from_env()`**

| Variable | Field |
|----------|-------|
| `KOKORO_USE_GPU` | `use_gpu` |
| `KOKORO_FORCE_CPU` | `force_cpu` |
| `KOKORO_DEFAULT_LANGUAGE` | `default_language` |
| `KOKORO_DEFAULT_SPEED` | `default_speed` |
| `KOKORO_DEFAULT_VOICE` | `default_voice` |
| `KOKORO_MODEL_DIR` | `model_dir` |
| `KOKORO_VOICE_DIR` | `voice_dir` |
| `KOKORO_CACHE_DIR` | `cache_dir` |
| `KOKORO_NORMALIZE_TEXT` | `normalize_text` |
| `KOKORO_WARMUP_ON_INIT` | `warmup_on_init` |
| `KOKORO_AUTO_DOWNLOAD` | `auto_download_model` |
| `CHAMPI_TTS_VOICES` | `available_voices` (comma-separated) |
| `CHAMPI_TTS_MODELS` | `available_models` (comma-separated) |
| `CHAMPI_TTS_AUDIO_FORMAT` | `tts_audio_format` |
| `CHAMPI_AUTO_START_KOKORO` | `auto_start` |

**Instance methods**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `dict[str, Any]` | Serialize to dictionary |
| `get_device()` | `str` | Returns `"cuda"`, `"mps"`, or `"cpu"` |
| `validate_tts_audio_format()` | `str` | Returns a supported format, falling back to `"pcm"` |
| `supported_audio_formats` | `list[str]` | Property: `["mp3", "opus", "flac", "wav", "pcm"]` |

```python
from champi_tts.providers.kokoro import KokoroConfig

# From environment
config = KokoroConfig.from_env()

# From file
config = KokoroConfig.from_file("/etc/champi/kokoro.json")

# Manual
config = KokoroConfig(
    default_voice="af_bella",
    default_speed=1.2,
    use_gpu=True,
    streaming_chunk_size=150,
)

print(config.get_device())     # "cuda" / "mps" / "cpu"
print(config.to_dict())
```

---

### `KokoroConfigPresets`

Static factory class for common configuration profiles.

```python
from champi_tts.providers.kokoro import KokoroConfig

class KokoroConfigPresets:
    @staticmethod def performance() -> KokoroConfig   # GPU, chunk_size=150, warmup=True
    @staticmethod def quality()     -> KokoroConfig   # GPU, chunk_size=300, advanced normalization
    @staticmethod def cpu_only()    -> KokoroConfig   # CPU only, chunk_size=100, no warmup
    @staticmethod def minimal()     -> KokoroConfig   # CPU, no normalization, no warmup, no auto-download
```

```python
from champi_tts.providers.kokoro.config import KokoroConfigPresets

config = KokoroConfigPresets.performance()
provider = get_provider("kokoro", config=config)

config = KokoroConfigPresets.cpu_only()
provider = get_provider("kokoro", config=config)
```

---

### `KokoroProvider`

Singleton TTS provider with direct access to Kokoro TTS internals. Use `get_provider("kokoro")` (which returns a `KokoroProviderAdapter` implementing `BaseTTSProvider`) for typical usage. Access `KokoroProvider` directly only when you need Kokoro-specific methods such as `text_to_speech()`, `read_pdf()`, or `get_signals()`.

`KokoroProvider` is a singleton — `KokoroProvider(config)` always returns the same instance.

**Constructor / singleton access**

```python
from champi_tts.providers.kokoro import KokoroProvider, KokoroConfig

config = KokoroConfig(default_voice="af_bella")
provider = KokoroProvider(config=config)

# Async-safe singleton access
provider = await KokoroProvider.get_instance(config=config)
```

**Key methods**

#### `async initialize(download_model=True)`

Load the Kokoro model. Attempts memory cache → disk cache → fresh download.

```python
await provider.initialize()
await provider.initialize(download_model=False)  # Skip download, load from local only
```

#### `async synthesize(text, voice="default", **kwargs)`

Generate audio from text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | — | Text to synthesize |
| `voice` | `str` | `"default"` | Voice name without `.pt` extension |
| `lang_code` | `str` (kwarg) | `config.default_language` | Language code |
| `speed` | `float` (kwarg) | `config.default_speed` | Speed multiplier |
| `normalize` | `bool` (kwarg) | `config.normalize_text` | Normalize text |

**Returns** `np.ndarray`

```python
audio = await provider.synthesize("Hello!", voice="af_bella", speed=1.1)
```

#### `async synthesize_streaming(text, voice="default", **kwargs)`

Stream audio chunks.

| kwarg | Default | Description |
|-------|---------|-------------|
| `lang_code` | `config.default_language` | Language code |
| `speed` | `config.default_speed` | Speed |
| `chunk_size` | `config.streaming_chunk_size` | Characters per chunk |
| `use_phoneme` | `False` | Treat `text` as pre-computed phonemes |

```python
async for chunk in provider.synthesize_streaming("A long text...", voice="am_adam"):
    play(chunk)
```

#### `async synthesize_from_phonemes(phonemes, voice="default", **kwargs)`

Synthesize from a Kokoro phoneme string directly.

```python
audio = await provider.synthesize_from_phonemes("hɛloʊ", voice="af_bella")
```

#### `async text_to_speech(message, voice=None, speed=1.0, lang_code="a", play_audio=True, use_phoneme=False, tts_instructions=None)`

High-level convenience method: synthesize and optionally play audio.

**Returns** `tuple[bool, dict[str, Any]]` — `(success, metrics)` where `metrics` contains `generation`, `playback`, `audio_duration`, `ttfa`, `total`, `voice`, `model`, `text_length`, and `real_time_factor`.

```python
success, metrics = await provider.text_to_speech(
    "Hello, world!",
    voice="af_bella",
    speed=1.0,
    play_audio=True,
)
print(metrics["generation"])      # seconds to generate
print(metrics["audio_duration"])  # seconds of audio produced
```

#### `list_voices()`

Return voice names found in `config.voice_dir`.

```python
voices = provider.list_voices()
print(voices)  # ["af_bella", "am_adam", ...]
```

#### `async get_info()`

Return a dictionary with provider status and configuration.

```python
info = await provider.get_info()
# {
#   "name": "Kokoro",
#   "version": "1.0",
#   "initialized": True,
#   "device": "cuda",
#   "model_loaded": True,
#   "available_voices": [...],
#   "voice_count": 10,
#   "supported_languages": ["a", "b", "e", ...],
#   "config": {...}
# }
```

#### `async unload()`

Release model weights and reset to uninitialized state.

```python
await provider.unload()
```

#### `async reload()`

Unload then reload without re-downloading model files.

```python
await provider.reload()
```

#### `get_signals()`

Return the blinker signals emitted by this provider (see [Event System](#event-system)).

```python
signals = provider.get_signals()
signals["lifecycle"].connect(my_handler)
signals["processing"].connect(my_handler)
```

#### PDF, HTML and URL methods

`KokoroProvider` has built-in document reading capabilities.

| Method | Description |
|--------|-------------|
| `async read_pdf(pdf_path, page_range=None) -> str` | Extract text from PDF (uses PyMuPDF or PyPDF2) |
| `async synthesize_pdf(pdf_path, voice=None, ...) -> tuple[bool, dict]` | Convert PDF to speech |
| `async read_html(html_source, clean_text=True) -> str` | Extract text from HTML string, file, or URL |
| `async synthesize_html(html_source, voice=None, ...) -> tuple[bool, dict]` | Convert HTML to speech |
| `async read_url(url, clean_text=True) -> str` | Extract text from a web URL |
| `async synthesize_url(url, voice=None, ...) -> tuple[bool, dict]` | Convert web page to speech |

```python
# Read and speak a PDF
success, metrics = await provider.synthesize_pdf(
    "document.pdf",
    voice="af_bella",
    page_range=(1, 5),  # Only pages 1–5
    speed=1.0,
    play_audio=True,
)

# Read and speak a web page
success, metrics = await provider.synthesize_url(
    "https://example.com/article",
    voice="am_adam",
    use_phoneme=True,   # Better quality for long content
    use_streaming=True,
)
```

---

### `VoiceLanguage`

Enum mapping voice name prefixes to Kokoro language codes.

```python
from champi_tts.providers.kokoro import VoiceLanguage
```

| Member | Code | Language |
|--------|------|----------|
| `AMERICAN_ENGLISH` | `"a"` | American English |
| `BRITISH_ENGLISH` | `"b"` | British English |
| `SPANISH` | `"e"` | Spanish |
| `FRENCH` | `"f"` | French |
| `HINDI` | `"h"` | Hindi |
| `ITALIAN` | `"i"` | Italian |
| `JAPANESE` | `"j"` | Japanese (requires `pip install misaki[ja]`) |
| `PORTUGUESE_BRAZIL` | `"p"` | Brazilian Portuguese |
| `MANDARIN_CHINESE` | `"z"` | Mandarin Chinese (requires `pip install misaki[zh]`) |

**Voice prefix mapping** — voice names are prefixed with two letters indicating gender and language: `af_` = American Female, `am_` = American Male, `bf_` = British Female, `bm_` = British Male, `ef_` = Spanish Female, etc.

```python
lang_code = VoiceLanguage.from_voice_prefix("af_bella")  # "a"
all_codes  = VoiceLanguage.get_all_codes()               # ["a", "b", "e", ...]
```

---

## Reader Service

```python
from champi_tts import TextReaderService, ReaderState
```

---

### `ReaderState`

Enum describing the current state of a `TextReaderService`.

| Member | Value | Description |
|--------|-------|-------------|
| `IDLE` | `"idle"` | Waiting for text |
| `READING` | `"reading"` | Synthesizing or playing |
| `PAUSED` | `"paused"` | Playback paused |
| `STOPPED` | `"stopped"` | Explicitly stopped |

---

### `TextReaderService`

High-level service that synthesizes and plays text, supports pause/resume/stop, a text queue, and optional UI feedback.

**Constructor**

```python
def __init__(self, provider: BaseTTSProvider, show_ui: bool = False) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `BaseTTSProvider` | — | Uninitialized or initialized provider |
| `show_ui` | `bool` | `False` | Show `TTSIndicatorUI` window |

Prefer using `get_reader()` rather than instantiating directly.

**Properties**

| Property | Type | Description |
|----------|------|-------------|
| `state` | `ReaderState` | Current reader state |

**Lifecycle methods**

#### `async initialize()`

Initialize the underlying provider if it has not been initialized yet.

```python
reader = get_reader("kokoro")
await reader.initialize()
```

#### `async cleanup()`

Stop reading, close the UI, and shut down the provider.

```python
await reader.cleanup()
```

**Async context manager**

```python
async with get_reader("kokoro") as reader:
    await reader.read_text("Hello!")
# cleanup() called automatically
```

**Reading methods**

#### `async read_text(text, voice=None)`

Synthesize and play a single string. Blocks until playback is complete (or the reader is stopped).

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text to read |
| `voice` | `str \| None` | Voice override; uses provider default when `None` |

```python
await reader.read_text("Good morning!")
await reader.read_text("Bonjour", voice="ff_camille")
```

#### `async read_file(file_path, voice=None)`

Read a plain-text file paragraph by paragraph (paragraphs are separated by blank lines). A 0.5-second pause is inserted between paragraphs.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `str \| Path` | Path to a UTF-8 text file |
| `voice` | `str \| None` | Voice override |

**Raises** `FileNotFoundError` if the file does not exist.

```python
await reader.read_file("chapter1.txt")
await reader.read_file(Path("/tmp/notes.txt"), voice="am_adam")
```

#### `async read_queue(voice=None)`

Process all items in the text queue in order.

```python
reader.add_to_queue("First paragraph.")
reader.add_to_queue("Second paragraph.")
await reader.read_queue()
```

**Queue management**

#### `add_to_queue(text)`

Append a string to the reading queue.

```python
reader.add_to_queue("Line one.")
reader.add_to_queue("Line two.")
```

#### `clear_queue()`

Remove all pending items from the queue.

```python
reader.clear_queue()
```

**Playback control**

#### `async pause()`

Pause playback. Only effective when `state == ReaderState.READING`. The current audio chunk stops immediately.

```python
await reader.pause()
# reader.state == ReaderState.PAUSED
```

#### `async resume()`

Resume playback. Only effective when `state == ReaderState.PAUSED`.

```python
await reader.resume()
# reader.state == ReaderState.READING
```

#### `async stop()`

Stop playback, clear the queue, and set state to `STOPPED`.

```python
await reader.stop()
# reader.state == ReaderState.STOPPED
```

#### `async interrupt()`

Immediately interrupt the underlying provider and stop audio output without changing queue state.

```python
await reader.interrupt()
```

**Signals**

`TextReaderService` exposes [blinker](https://blinker.readthedocs.io/) signals. Connect a callable that accepts `(sender, **kwargs)`.

| Signal | `kwargs` | Fired when |
|--------|----------|------------|
| `on_reading_started` | `text: str` | A new `read_text()` call begins |
| `on_reading_completed` | `text: str` | A `read_text()` call finishes successfully |
| `on_reading_paused` | — | `pause()` takes effect |
| `on_reading_resumed` | — | `resume()` takes effect |
| `on_reading_stopped` | — | `stop()` is called |
| `on_state_changed` | `old_state: str, new_state: str` | Any state transition |
| `on_error` | `error: str` | An exception is caught during reading |

```python
def on_started(sender, **kwargs):
    print(f"Started: {kwargs['text'][:60]}")

def on_done(sender, **kwargs):
    print("Done reading")

def on_error(sender, **kwargs):
    print(f"Error: {kwargs['error']}")

reader.on_reading_started.connect(on_started)
reader.on_reading_completed.connect(on_done)
reader.on_error.connect(on_error)
```

---

## UI Components

```python
from champi_tts import TTSIndicatorUI, TTSState
```

Requires the `[ui]` extra: `pip install champi-tts[ui]`.

---

### `TTSState`

Enum representing the visual states of the TTS indicator window.

| Member | Value | Color |
|--------|-------|-------|
| `IDLE` | `"idle"` | Gray `(0.5, 0.5, 0.5)` |
| `PROCESSING` | `"processing"` | Yellow `(1.0, 0.8, 0.0)` |
| `SPEAKING` | `"speaking"` | Green `(0.0, 0.8, 0.0)` — pulses |
| `PAUSED` | `"paused"` | Blue `(0.0, 0.5, 1.0)` |
| `ERROR` | `"error"` | Red `(1.0, 0.0, 0.0)` |

---

### `TTSIndicatorUI`

A small floating ImGui window (powered by `imgui-bundle`) that shows the current TTS state as a colored indicator circle. When `show_ui=True` is passed to `get_reader()`, the reader creates and drives this automatically.

**Constructor**

```python
def __init__(self, window_x: int = 50, window_y: int = 50) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_x` | `int` | `50` | Initial X position of the window in pixels |
| `window_y` | `int` | `50` | Initial Y position of the window in pixels |

**Methods**

#### `update_state(state, text="")`

Update the displayed state. Thread-safe to call from a signal handler.

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `TTSState` | New state to display |
| `text` | `str` | Optional text snippet to show below the indicator |

```python
from champi_tts import TTSIndicatorUI, TTSState

ui = TTSIndicatorUI(window_x=100, window_y=100)
ui.update_state(TTSState.SPEAKING, "Hello, world!")
ui.update_state(TTSState.IDLE)
ui.update_state(TTSState.ERROR, "Synthesis failed")
```

#### `run()`

Enter the ImGui event loop. Blocks until the window is closed.

```python
ui = TTSIndicatorUI()
ui.run()
```

#### `gui()`

Render one ImGui frame. Call this from inside your own `immapp` loop if you want to embed the indicator in a larger UI.

**Standalone test mode**

```python
from champi_tts.ui import run_standalone

run_standalone()  # Cycles through all states every 3 seconds
```

---

## Audio Utilities

```python
from champi_tts.core.audio import AudioPlayer, save_audio, load_audio, normalize_audio
from champi_tts.core.audio_effects import AudioProcessor, process_audio
```

---

### `AudioPlayer`

Playback wrapper around `sounddevice`. Used internally by `TextReaderService`.

**Constructor**

```python
def __init__(self, sample_rate: int = 24000) -> None
```

**Properties**

| Property | Type | Description |
|----------|------|-------------|
| `sample_rate` | `int` | Playback sample rate |
| `is_playing` | `bool` | `True` while audio is playing |

**Methods**

#### `async play(audio, blocking=True)`

Play an audio array.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio` | `np.ndarray` | — | Audio data to play |
| `blocking` | `bool` | `True` | Wait for playback to finish before returning |

```python
player = AudioPlayer(sample_rate=24000)
await player.play(audio)            # blocking
await player.play(audio, blocking=False)  # fire and forget
```

#### `async play_streaming(audio_chunks)`

Play a sequence of audio chunks in order.

```python
chunks = [chunk1, chunk2, chunk3]
await player.play_streaming(chunks)
```

#### `stop()`

Stop playback immediately.

```python
player.stop()
```

---

### `save_audio()`

Save a numpy audio array to a file.

```python
async def save_audio(
    audio: np.ndarray,
    output_path: str | Path,
    sample_rate: int = 24000,
    format: str = "wav",
) -> None
```

Creates parent directories automatically.

```python
from champi_tts.core.audio import save_audio

await save_audio(audio, "output.wav", sample_rate=24000)
await save_audio(audio, "output.flac", sample_rate=24000, format="flac")
```

---

### `load_audio()`

Load an audio file into a numpy array.

```python
async def load_audio(
    file_path: str | Path,
    target_sample_rate: int | None = None,
) -> tuple[np.ndarray, int]
```

**Returns** `tuple[np.ndarray, int]` — `(audio_data, sample_rate)`. Resamples to `target_sample_rate` if provided.

```python
from champi_tts.core.audio import load_audio

audio, rate = await load_audio("input.wav")
audio, rate = await load_audio("input.mp3", target_sample_rate=24000)
```

---

### `normalize_audio()`

Normalize audio amplitude to a target dB level.

```python
def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio` | `np.ndarray` | — | Input audio |
| `target_db` | `float` | `-20.0` | Target RMS level in dB |

**Returns** `np.ndarray` — normalized and clipped to `[-1.0, 1.0]`.

```python
from champi_tts.core.audio import normalize_audio

normalized = normalize_audio(audio, target_db=-18.0)
```

---

### `AudioProcessor`

Higher-level effects processor. Construct one per session; it stores the sample rate so you do not have to pass it to every method.

```python
from champi_tts.core.audio_effects import AudioProcessor

processor = AudioProcessor(sample_rate=24000)
```

**Methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `normalize_volume` | `(audio, target_db=-20.0) -> np.ndarray` | Normalize to target dB |
| `apply_gain` | `(audio, gain_db) -> np.ndarray` | Apply fixed gain |
| `normalize_rms` | `(audio, target_rms=0.02) -> np.ndarray` | Normalize to target RMS |
| `add_silence` | `(audio, duration=0.5) -> np.ndarray` | Pad with silence at both ends |
| `add_fade` | `(audio, fade_duration=0.1) -> np.ndarray` | Fade in at start, fade out at end |
| `add_echo` | `(audio, delay=0.5, decay=0.7, spread=0.5) -> np.ndarray` | Add echo effect |
| `apply_lowpass` | `(audio, cutoff_freq=8000, order=2) -> np.ndarray` | Low-pass Butterworth filter |
| `apply_highpass` | `(audio, cutoff_freq=100, order=2) -> np.ndarray` | High-pass Butterworth filter |
| `apply_bass_boost` | `(audio, boost_db=10.0) -> np.ndarray` | Amplify bass frequencies |
| `apply_compression` | `(audio, threshold=-24, ratio=4.0, attack=0.01, release=0.1) -> np.ndarray` | Dynamic compression |
| `apply_reverb` | `(audio, decay=1.5, early_reflections=5) -> np.ndarray` | Convolution reverb |
| `apply_bitcrush` | `(audio, bits=8) -> np.ndarray` | Bit-depth reduction (lo-fi effect) |
| `chain_effects` | `(audio, effects: list[dict]) -> np.ndarray` | Apply a sequence of effects |

**Chaining effects**

```python
audio = processor.chain_effects(audio, [
    {"type": "normalize", "target_db": -18},
    {"type": "fade",      "fade_duration": 0.05},
    {"type": "echo",      "delay": 0.3, "decay": 0.5},
])
```

Supported `type` values: `normalize`, `gain`, `fade`, `silence`, `echo`, `lowpass`, `highpass`, `bass_boost`, `compression`, `reverb`, `bitcrush`.

---

### `process_audio()`

Convenience function that creates an `AudioProcessor` and chains the given effects.

```python
def process_audio(
    audio: np.ndarray,
    sample_rate: int = 24000,
    effects: list[dict] | None = None,
) -> np.ndarray
```

```python
from champi_tts.core.audio_effects import process_audio

processed = process_audio(audio, effects=[
    {"type": "normalize", "target_db": -20},
    {"type": "fade", "fade_duration": 0.1},
])
```

---

## Exceptions

```python
from champi_tts.providers.kokoro import (
    KokoroError,
    KokoroInitializationError,
    KokoroModelError,
    KokoroVoiceError,
    KokoroSynthesisError,
    KokoroAudioError,
    KokoroConfigurationError,
    KokoroTextProcessingError,
    KokoroFileError,
)
```

All Kokoro exceptions inherit from `KokoroError` which inherits from `Exception`.

| Exception | Raised when |
|-----------|-------------|
| `KokoroError` | Base class — never raised directly |
| `KokoroInitializationError` | `initialize()` fails (e.g. model load error) |
| `KokoroModelError` | Model loading or in-memory operation fails |
| `KokoroVoiceError` | A voice file is missing or invalid |
| `KokoroSynthesisError` | Audio synthesis fails |
| `KokoroAudioError` | Audio playback fails |
| `KokoroConfigurationError` | Config values are invalid |
| `KokoroTextProcessingError` | Text normalization or phonemization fails |
| `KokoroFileError` | A required model or config file is missing |

**Error handling examples**

```python
from champi_tts import get_provider
from champi_tts.providers.kokoro import (
    KokoroInitializationError,
    KokoroVoiceError,
    KokoroSynthesisError,
)

provider = get_provider("kokoro")

try:
    await provider.initialize()
except KokoroInitializationError as e:
    print(f"Failed to initialize: {e}")

try:
    audio = await provider.synthesize("Hello", voice="unknown_voice")
except KokoroVoiceError as e:
    print(f"Voice not found: {e}")
except KokoroSynthesisError as e:
    print(f"Synthesis failed: {e}")
```

---

## Event System

Champi TTS uses [blinker](https://blinker.readthedocs.io/) signals throughout. `TextReaderService` signals are described in [Reader Service / Signals](#signals). `KokoroProvider` emits lower-level provider signals through `TTSSignalManager`.

### `TTSSignalManager`

Manages four blinker signals corresponding to event categories:

| Signal attribute | Category | Fired by |
|-----------------|----------|---------|
| `lifecycle` | Provider lifecycle | `initialize()`, `unload()`, `reload()` |
| `model` | Model loading | cache hits, fresh loads |
| `processing` | Synthesis and playback | `synthesize()`, `text_to_speech()` |
| `telemetry` | Metrics | after each `text_to_speech()` call |

Each signal is sent with keyword arguments: `event_type` (category name), `sub_event` (specific action), and `data` (payload dict).

**Connecting to provider signals**

```python
from champi_tts.providers.kokoro import KokoroProvider, KokoroConfig

provider = KokoroProvider(config=KokoroConfig())

signals = provider.get_signals()

def on_lifecycle(sender, **kwargs):
    print(f"Lifecycle: {kwargs['sub_event']} — {kwargs['data']}")

def on_processing(sender, **kwargs):
    if kwargs["sub_event"] == "synthesis_complete":
        audio_size = kwargs["data"].get("audio_size", 0)
        print(f"Synthesis done, {audio_size} samples")

def on_telemetry(sender, **kwargs):
    metrics = kwargs["data"]
    print(f"RTF: {metrics.get('real_time_factor', 0):.2f}")

signals["lifecycle"].connect(on_lifecycle)
signals["processing"].connect(on_processing)
signals["telemetry"].connect(on_telemetry)

await provider.initialize()
```

**`sub_event` values by category**

`lifecycle`: `initialization_start`, `initialization_complete`, `initialization_error`, `unload_start`, `unload_complete`, `reload_start`, `reload_complete`

`model`: `cache_hit`, `loading_start`

`processing`: `synthesis_start`, `synthesis_complete`, `synthesis_error`, `tts_request_start`, `tts_request_complete`, `tts_request_error`, `voice_selected`, `playback_start`, `playback_complete`

`telemetry`: `performance_metrics`

---

## Examples

### Basic synthesis and save to file

```python
import asyncio
from champi_tts import get_provider
from champi_tts.core.audio import save_audio

async def main():
    async with get_provider("kokoro") as provider:
        audio = await provider.synthesize(
            "Welcome to Champi TTS.",
            voice="af_bella",
            speed=1.0,
        )
        await save_audio(audio, "welcome.wav", sample_rate=provider.config.sample_rate)
        print("Saved welcome.wav")

asyncio.run(main())
```

---

### List voices and synthesize with each

```python
import asyncio
from champi_tts import get_provider
from champi_tts.core.audio import save_audio

async def main():
    async with get_provider("kokoro") as provider:
        voices = await provider.list_voices()
        print(f"Available voices: {voices}")

        for voice in voices[:3]:
            audio = await provider.synthesize(f"Hello, I am {voice}.", voice=voice)
            await save_audio(audio, f"{voice}.wav", sample_rate=24000)

asyncio.run(main())
```

---

### Streaming synthesis

```python
import asyncio
import numpy as np
from champi_tts import get_provider

async def main():
    async with get_provider("kokoro") as provider:
        chunks = []
        async for chunk in provider.synthesize_streaming(
            "This is a long sentence that will be streamed in chunks.",
            voice="am_adam",
        ):
            chunks.append(chunk)

        full_audio = np.concatenate(chunks)
        print(f"Total samples: {len(full_audio)}")

asyncio.run(main())
```

---

### Reader with event handling

```python
import asyncio
from champi_tts import get_reader

async def main():
    async with get_reader("kokoro", show_ui=True) as reader:

        def on_started(sender, **kwargs):
            preview = kwargs.get("text", "")[:50]
            print(f"[started] {preview}...")

        def on_done(sender, **kwargs):
            print("[completed]")

        def on_state(sender, **kwargs):
            print(f"[state] {kwargs['old_state']} -> {kwargs['new_state']}")

        def on_error(sender, **kwargs):
            print(f"[error] {kwargs['error']}")

        reader.on_reading_started.connect(on_started)
        reader.on_reading_completed.connect(on_done)
        reader.on_state_changed.connect(on_state)
        reader.on_error.connect(on_error)

        await reader.read_text("Hello, this is the first paragraph.")
        await reader.read_text("And this is the second paragraph.")

asyncio.run(main())
```

---

### Queue management

```python
import asyncio
from champi_tts import get_reader

async def main():
    async with get_reader("kokoro") as reader:
        reader.add_to_queue("Item one.")
        reader.add_to_queue("Item two.")
        reader.add_to_queue("Item three.")

        print(f"Queue ready, starting...")
        await reader.read_queue()
        print("All items read.")

asyncio.run(main())
```

---

### Pause and resume

```python
import asyncio
from champi_tts import get_reader

async def main():
    async with get_reader("kokoro") as reader:
        # Start reading in a background task
        read_task = asyncio.create_task(
            reader.read_file("long_document.txt")
        )

        await asyncio.sleep(5)
        print("Pausing...")
        await reader.pause()

        await asyncio.sleep(3)
        print("Resuming...")
        await reader.resume()

        await read_task

asyncio.run(main())
```

---

### Advanced configuration with presets

```python
import asyncio
from champi_tts import get_provider
from champi_tts.providers.kokoro.config import KokoroConfigPresets

async def main():
    # High-quality GPU synthesis
    config = KokoroConfigPresets.quality()
    config.default_voice = "af_bella"

    async with get_provider("kokoro", config=config) as provider:
        audio = await provider.synthesize("High quality synthesis.")
        print(f"Device: {config.get_device()}")

asyncio.run(main())
```

---

### CPU-only mode from environment

```shell
export KOKORO_FORCE_CPU=true
export KOKORO_DEFAULT_VOICE=am_adam
export KOKORO_DEFAULT_SPEED=0.9
```

```python
import asyncio
from champi_tts import get_provider
from champi_tts.providers.kokoro import KokoroConfig

async def main():
    config = KokoroConfig.from_env()
    async with get_provider("kokoro", config=config) as provider:
        audio = await provider.synthesize("Environment-configured voice.")

asyncio.run(main())
```

---

### Audio post-processing

```python
import asyncio
from champi_tts import get_provider
from champi_tts.core.audio import save_audio, normalize_audio
from champi_tts.core.audio_effects import AudioProcessor

async def main():
    processor = AudioProcessor(sample_rate=24000)

    async with get_provider("kokoro") as provider:
        audio = await provider.synthesize("Testing audio effects.")

    # Normalize then fade
    audio = normalize_audio(audio, target_db=-18.0)
    audio = processor.add_fade(audio, fade_duration=0.05)

    # Chain multiple effects
    audio = processor.chain_effects(audio, [
        {"type": "compression", "threshold": -20, "ratio": 3.0},
        {"type": "highpass",    "cutoff_freq": 80},
    ])

    await save_audio(audio, "processed.wav", sample_rate=24000)

asyncio.run(main())
```

---

### Error handling

```python
import asyncio
from champi_tts import get_provider
from champi_tts.providers.kokoro import (
    KokoroInitializationError,
    KokoroVoiceError,
    KokoroSynthesisError,
    KokoroFileError,
)

async def main():
    provider = get_provider("kokoro")

    try:
        await provider.initialize()
    except KokoroFileError as e:
        print(f"Model files missing: {e}")
        return
    except KokoroInitializationError as e:
        print(f"Initialization failed: {e}")
        return

    try:
        audio = await provider.synthesize("Hello!", voice="af_bella")
    except KokoroVoiceError as e:
        print(f"Voice unavailable: {e}")
    except KokoroSynthesisError as e:
        print(f"Synthesis error: {e}")
    finally:
        await provider.shutdown()

asyncio.run(main())
```

---

### Custom provider implementation

```python
from collections.abc import AsyncGenerator
import numpy as np
from champi_tts.core.base_config import BaseTTSConfig
from champi_tts.core.base_provider import BaseTTSProvider
from dataclasses import dataclass


@dataclass
class MyConfig(BaseTTSConfig):
    my_api_key: str = ""

    @classmethod
    def from_env(cls) -> "MyConfig":
        import os
        return cls(my_api_key=os.environ.get("MY_TTS_API_KEY", ""))

    def validate(self) -> bool:
        return bool(self.my_api_key)


class MyTTSProvider(BaseTTSProvider):
    def __init__(self, config: MyConfig):
        super().__init__(config)

    async def initialize(self) -> None:
        # connect to your TTS service
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    async def synthesize(self, text, voice=None, speed=None, **kwargs) -> np.ndarray:
        # call your TTS service and return audio array
        return np.zeros(24000, dtype=np.float32)

    async def synthesize_streaming(self, text, voice=None, speed=None, **kwargs):
        yield np.zeros(2400, dtype=np.float32)

    async def list_voices(self) -> list[str]:
        return ["default"]

    async def interrupt(self) -> None:
        pass
```
