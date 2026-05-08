# Champi TTS API Reference

Complete API documentation with examples for all public methods.

---

## Table of Contents

- [Factory Functions](#factory-functions)
- [Base Classes](#base-classes)
- [TTS Provider](#tts-provider)
- [Reader Service](#reader-service)
- [UI Components](#ui-components)
- [Audio Utilities](#audio-utilities)
- [Exceptions](#exceptions)

---

## Factory Functions

### `get_provider()`

Factory function to create a TTS provider instance.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider_type` | `Literal["kokoro"]` | Type of provider to use |
| `config` | `BaseTTSConfig | None` | Pre-configured config (optional) |
| `**config_kwargs` | `Any` | Config parameters (if config not provided) |

#### Returns

`BaseTTSProvider`: Initialized TTS provider instance

#### Examples

**Basic usage:**

```python
from champi_tts import get_provider

# Get default provider (Kokoro)
provider = get_provider()
await provider.initialize()

# Synthesize audio
audio = await provider.synthesize("Hello, world!")

# Save to file
from champi_tts.core.audio import save_audio
await save_audio(audio, "hello.wav", sample_rate=provider.config.sample_rate)

await provider.shutdown()
```

**Custom configuration:**

```python
from champi_tts import get_provider
from champi_tts.providers.kokoro import KokoroConfig

config = KokoroConfig(
    default_voice="af_bella",
    default_speed=1.1,
    use_gpu=True,
)

provider = get_provider("kokoro", config=config)
await provider.initialize()
```

**Using kwargs:**

```python
provider = get_provider("kokoro", default_voice="af_bella", use_gpu=True)
```

### `get_reader()`

Factory function to create a text reader service.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider_type` | `Literal["kokoro"]` | Type of provider to use |
| `show_ui` | `bool` | Show visual UI indicator |
| `config` | `BaseTTSConfig | None` | Pre-configured config (optional) |
| `**config_kwargs` | `Any` | Config parameters |

#### Returns

`TextReaderService`: Text reader service instance

#### Examples

**Basic reader:**

```python
from champi_tts import get_reader

reader = get_reader("kokoro")
await reader.read_file("document.txt")
```

**With UI:**

```python
reader = get_reader("kokoro", show_ui=True)
await reader.read_text("This is a test")
await reader.pause()
await reader.resume()
await reader.stop()
```

**With custom config:**

```python
reader = get_reader(
    "kokoro",
    show_ui=True,
    default_voice="af_bella",
    default_speed=1.2
)
```

### `get_default_provider()`

Get the default TTS provider (Kokoro).

```python
from champi_tts import get_default_provider

provider = get_default_provider()
```

### `list_providers()`

Get list of available TTS providers.

```python
from champi_tts import list_providers

providers = list_providers()  # Returns ["kokoro"]
```

---

## Base Classes

### `BaseTTSConfig`

Abstract base class for TTS provider configurations.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `default_voice` | `str` | Default voice to use |
| `default_speed` | `float` | Default speech speed |
| `use_gpu` | `bool` | Use GPU acceleration |

#### Subclasses

- `KokoroConfig` - Kokoro-specific configuration

### `BaseTTSProvider`

Abstract base class for TTS providers.

#### Methods

##### `async initialize()`

Initialize the TTS provider.

**Returns:** `None`

**Example:**

```python
from champi_tts import get_provider

provider = get_provider()
await provider.initialize()
```

##### `async shutdown()`

Shutdown the TTS provider.

**Returns:** `None`

**Example:**

```python
await provider.shutdown()
```

##### `async synthesize(text, voice=None, speed=None, **kwargs)`

Synthesize text to speech.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text to synthesize |
| `voice` | `str | None` | Voice to use (optional) |
| `speed` | `float | None` | Speech speed (optional) |
| `**kwargs` | `Any` | Additional provider-specific parameters |

**Returns:** `np.ndarray` - Audio data as numpy array

**Example:**

```python
audio = await provider.synthesize("Hello, world!")
```

##### `async synthesize_streaming(text, voice=None, speed=None, **kwargs)`

Synthesize with streaming output.

**Yields:** `np.ndarray` - Audio chunks

**Example:**

```python
async for chunk in provider.synthesize_streaming("Long text"):
    # Process each chunk
    pass
```

##### `async list_voices()`

List available voices.

**Returns:** `List[str]` - List of voice names

**Example:**

```python
voices = await provider.list_voices()
print(voices)
# ['af_bella', 'am_adam', ...]
```

##### `async interrupt()`

Interrupt current synthesis/playback.

**Example:**

```python
await provider.interrupt()
```

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_initialized` | `bool` | Check if provider is initialized |
| `is_speaking` | `bool` | Check if currently speaking |

**Async Context Manager:**

```python
async with get_provider("kokoro") as provider:
    audio = await provider.synthesize("Hello!")
    # Automatic cleanup on exit
```

### `BaseSynthesizer`

Abstract base class for synthesizers.

---

## Reader Service

### `TextReaderService`

Text reading service with interruption support.

#### Constructor

```python
class TextReaderService:
    def __init__(self, provider: BaseTTSProvider, show_ui: bool = False)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider` | `BaseTTSProvider` | TTS provider to use |
| `show_ui` | `bool` | Show visual UI indicator |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `state` | `ReaderState` | Current reader state |

#### States

| State | Value | Description |
|-------|-------|-------------|
| `IDLE` | `"idle"` | Waiting for text |
| `READING` | `"reading"` | Processing text |
| `PAUSED` | `"paused"` | Reading paused |
| `STOPPED` | `"stopped"` | Reading stopped |

#### Methods

##### `async read_text(text, voice=None)`

Read a single text string.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text to read |
| `voice` | `str | None` | Voice to use (optional) |

**Example:**

```python
await reader.read_text("Hello, this is a test")
```

##### `async read_file(file_path, voice=None)`

Read text from file with paragraph-by-paragraph processing.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `str | Path` | Path to text file |
| `voice` | `str | None` | Voice to use (optional) |

**Example:**

```python
await reader.read_file("document.txt")
```

##### `read_queue(voice=None)`

Read all texts in the queue.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `voice` | `str | None` | Voice to use (optional) |

**Example:**

```python
reader.add_to_queue("Text 1")
reader.add_to_queue("Text 2")
await reader.read_queue()
```

##### `add_to_queue(text)`

Add text to reading queue.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text to add to queue |

**Example:**

```python
reader.add_to_queue("Text to read")
```

##### `clear_queue()`

Clear reading queue.

**Example:**

```python
reader.clear_queue()
```

##### `async pause()`

Pause reading.

**Example:**

```python
await reader.pause()
```

##### `async resume()`

Resume reading.

**Example:**

```python
await reader.resume()
```

##### `async stop()`

Stop reading.

**Example:**

```python
await reader.stop()
```

##### `async interrupt()`

Interrupt current reading immediately.

**Example:**

```python
await reader.interrupt()
```

##### Async Context Manager

```python
async with get_reader("kokoro") as reader:
    await reader.read_file("document.txt")
    # Automatic cleanup on exit
```

#### Signals

| Signal | Parameters | Description |
|--------|-----------|-------------|
| `on_reading_started` | `text: str` | Called when reading starts |
| `on_reading_completed` | `text: str` | Called when reading completes |
| `on_reading_paused` | `None` | Called when reading pauses |
| `on_reading_resumed` | `None` | Called when reading resumes |
| `on_reading_stopped` | `None` | Called when reading stops |
| `on_state_changed` | `old_state: str, new_state: str` | Called on state change |
| `on_error` | `error: str` | Called when an error occurs |

**Example:**

```python
reader.on_reading_started.connect(lambda **kw: print(f"Started: {kw['text']}"))
reader.on_reading_completed.connect(lambda **kw: print("Completed"))
reader.on_error.connect(lambda **kw: print(f"Error: {kw['error']}"))
```

---

## UI Components

### `TTSState`

Enumeration of TTS visual states.

| State | Value | Description |
|-------|-------|-------------|
| `IDLE` | `"idle"` | Waiting for text |
| `PROCESSING` | `"processing"` | Processing text |
| `SPEAKING` | `"speaking"` | Currently speaking |
| `PAUSED` | `"paused"` | Reading paused |
| `ERROR` | `"error"` | Error occurred |

### `TTSIndicatorUI`

Visual indicator for TTS states.

#### Constructor

```python
class TTSIndicatorUI:
    def __init__(self, window_x: int = 50, window_y: int = 50)
```

#### Methods

##### `update_state(state, text="")`

Update the current TTS state.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `TTSState` | New state to set |
| `text` | `str` | Text to display (optional) |

**Example:**

```python
from champi_tts import TTSIndicatorUI, TTSState

ui = TTSIndicatorUI()
ui.update_state(TTSState.SPEAKING, "Hello, world!")
```

##### `run()`

Run the UI main loop.

**Example:**

```python
ui = TTSIndicatorUI()
ui.run()
```

**Standalone Test Mode:**

```python
from champi_tts.ui import run_standalone

run_standalone()  # Shows test UI cycling through states
```

#### States Colors

| State | Color |
|-------|-------|
| `IDLE` | Gray (0.5, 0.5, 0.5) |
| `PROCESSING` | Yellow (1.0, 0.8, 0.0) |
| `SPEAKING` | Green (0.0, 0.8, 0.0) |
| `PAUSED` | Blue (0.0, 0.5, 1.0) |
| `ERROR` | Red (1.0, 0.0, 0.0) |

---

## Audio Utilities

### `AudioPlayer`

Audio playback utility.

#### Constructor

```python
class AudioPlayer:
    def __init__(self, sample_rate: int = 22050)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `sample_rate` | `int` | Sample rate |
| `volume` | `float` | Playback volume (0.0-1.0) |
| `is_playing` | `bool` | Whether audio is playing |

#### Methods

##### `play(audio, blocking=False)`

Play audio data.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio` | `np.ndarray` | Audio data to play |
| `blocking` | `bool` | Block until playback completes |

##### `stop()`

Stop playback.

##### `set_volume()`

Reset volume to default.

### `save_audio()`

Save audio to file.

```python
from champi_tts.core.audio import save_audio

await save_audio(audio, "output.wav", sample_rate=22050)
```

### `load_audio()`

Load audio from file.

```python
from champi_tts.core.audio import load_audio

audio = await load_audio("output.wav")
```

### `normalize_audio()`

Normalize audio amplitude.

```python
from champi_tts.core.audio import normalize_audio

normalized = normalize_audio(audio)
```

### `resample_audio()`

Resample audio to different sample rate.

```python
from champi_tts.core.audio import resample_audio

resampled = resample_audio(audio, from_rate, to_rate)
```

---

## Exceptions

### `ChampiTTSException`

Base exception for all Champi TTS exceptions.

### `InitializationError`

Raised when provider fails to initialize.

```python
try:
    await provider.initialize()
except InitializationError as e:
    print(f"Failed to initialize: {e}")
```

### `SynthesisError`

Raised when synthesis fails.

```python
try:
    audio = await provider.synthesize(text)
except SynthesisError as e:
    print(f"Synthesis failed: {e}")
```

### `ReaderError`

Raised when reading fails.

```python
try:
    await reader.read_text(text)
except ReaderError as e:
    print(f"Reading failed: {e}")
```

### `FileNotFoundError`

Raised when file not found.

```python
try:
    await reader.read_file("missing.txt")
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

---

## Backwards Compatibility

For backwards compatibility, Kokoro classes are exposed directly:

```python
# Old way (still works)
from champi_tts import KokoroConfig, KokoroProvider

# New way (recommended)
from champi_tts import get_provider
```

---

## Complete Example

```python
from champi_tts import get_provider, get_reader, TTSState
from champi_tts.core.audio import save_audio, load_audio
import asyncio


async def main():
    # Get provider
    provider = get_provider()
    await provider.initialize()

    try:
        # Synthesize
        audio = await provider.synthesize("Hello, this is Champi TTS!")

        # Save to file
        await save_audio(audio, "hello.wav", sample_rate=provider.config.sample_rate)

        # Create reader with UI
        reader = get_reader("kokoro", show_ui=True)

        # Subscribe to events
        def on_started(sender, **kw):
            print(f"Started reading: {kw.get('text', '')[:50]}...")

        def on_completed(sender, **kw):
            print("Reading completed")

        reader.on_reading_started.connect(on_started)
        reader.on_reading_completed.connect(on_completed)
        reader.on_error.connect(lambda s, **kw: print(f"Error: {kw.get('error', '')}"))

        # Read text
        await reader.read_text("This is a test of the Champi TTS reader")

        # Pause/resume
        await reader.pause()
        await reader.resume()

        # Stop
        await reader.stop()

    finally:
        # Cleanup
        await provider.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
```
