# Champi TTS Architecture

## Overview

Champi TTS is a modular, multi-provider text-to-speech library designed to support multiple TTS backends with a unified interface, featuring full text reading capabilities and visual UI indicators.

## Directory Structure

```
src/champi_tts/
├── core/                          # Generic abstractions (provider-agnostic)
│   ├── base_config.py            # Abstract configuration base class
│   ├── base_provider.py          # Abstract TTS provider interface
│   ├── base_synthesizer.py       # Abstract synthesizer interface
│   └── audio.py                  # Generic audio handling (playback, save/load)
│
├── providers/                     # Provider implementations
│   └── kokoro/                   # Kokoro (neural TTS) provider
│       ├── config.py             # Kokoro configuration
│       ├── provider.py           # Kokoro TTS provider
│       ├── inference.py          # Low-level synthesis
│       ├── models.py             # Model management
│       ├── enums.py              # Kokoro-specific enums
│       ├── events.py             # Event system
│       ├── exceptions.py         # Kokoro exceptions
│       ├── text_utils.py         # Text normalization
│       └── cli/                  # Kokoro-specific CLI
│
├── reader/                        # Text reading service
│   ├── service.py                # Reading service with pause/resume/stop
│   └── __init__.py
│
├── ui/                            # Visual UI indicators
│   ├── tts_indicator_ui.py       # GLFW/ImGui visual indicator
│   └── __init__.py
│
├── cli/                           # CLI interface
│   ├── main.py                   # Main CLI commands
│   └── __init__.py
│
├── factory.py                     # Provider factory
└── __init__.py                    # Public API
```

## Design Principles

### 1. **Provider-Agnostic Core**

All generic functionality is extracted into `core/`:
- Audio handling (playback, file I/O)
- Abstract base classes for configs, providers, and synthesizers
- Common audio utilities (normalization, resampling)

### 2. **Provider Implementations**

Each provider (Kokoro, OpenAI, ElevenLabs, etc.) implements:
- `BaseTTSConfig` - Provider-specific configuration
- `BaseTTSProvider` - Main provider interface with synthesis methods
- `BaseSynthesizer` - Low-level synthesis logic (optional)

### 3. **Factory Pattern**

Providers are instantiated via factory functions:

```python
from champi_tts import get_provider, get_reader

# Get default provider
provider = get_provider()

# Get specific provider with custom config
provider = get_provider("kokoro", default_voice="af_bella", use_gpu=True)

# Get reader service
reader = get_reader("kokoro", show_ui=True)
```

### 4. **Text Reading Service**

The `TextReaderService` provides:
- File reading with paragraph-by-paragraph processing
- Text queue management
- Pause/resume/stop controls
- Interruption support
- Event-driven state tracking

Key features:
- **States**: `IDLE`, `READING`, `PAUSED`, `STOPPED`
- **Events**: `on_reading_started`, `on_reading_paused`, `on_reading_resumed`, `on_reading_stopped`, `on_reading_completed`
- **Controls**: `pause()`, `resume()`, `stop()`, `interrupt()`

### 5. **Visual UI Indicator**

The `TTSIndicatorUI` provides real-time visual feedback:
- Built with ImGui/GLFW for cross-platform support
- Visual states:
  - **Idle** (gray) - Waiting for text
  - **Processing** (yellow) - Processing text
  - **Speaking** (green) - Currently speaking (with pulse animation)
  - **Paused** (blue) - Reading paused
  - **Error** (red) - Error occurred
- Standalone testing mode
- Minimal resource usage

## Usage Examples

### Basic Synthesis

```python
from champi_tts import get_provider

provider = get_provider()
await provider.initialize()

audio = await provider.synthesize("Hello, world!")

await provider.shutdown()
```

### Custom Configuration

```python
from champi_tts import get_provider
from champi_tts.providers.kokoro import KokoroConfig

config = KokoroConfig(
    default_voice="af_bella",
    default_speed=1.1,
    use_gpu=True,
    warmup_on_init=True,
)

provider = get_provider("kokoro", config=config)
await provider.initialize()
```

### Text Reading with UI

```python
from champi_tts import get_reader

# Create reader with UI indicator
reader = get_reader("kokoro", show_ui=True, default_voice="af_bella")
await reader.provider.initialize()

# Read file
await reader.read_file("document.txt")

# Control playback
await reader.pause()
await reader.resume()
await reader.stop()

await reader.provider.shutdown()
```

### Event Handling

```python
from champi_tts import get_reader

reader = get_reader("kokoro")

# Subscribe to events
reader.on_reading_started.connect(lambda sender, **kw: print(f"Started: {kw['text']}"))
reader.on_reading_paused.connect(lambda sender, **kw: print("Paused"))
reader.on_reading_completed.connect(lambda sender, **kw: print("Completed"))

await reader.provider.initialize()
await reader.read_text("This will trigger events")
await reader.provider.shutdown()
```

## Adding New Providers

To add a new provider (e.g., OpenAI):

1. Create `providers/openai/` directory
2. Implement:
   - `OpenAIConfig(BaseTTSConfig)`
   - `OpenAITTSProvider(BaseTTSProvider)`
   - `OpenAISynthesizer(BaseSynthesizer)` (optional)
3. Add to `factory.py`
4. Update `list_providers()`

Example structure:
```python
# providers/openai/config.py
from champi_tts.core.base_config import BaseTTSConfig

@dataclass
class OpenAIConfig(BaseTTSConfig):
    api_key: str = ""
    model: str = "tts-1"
    voice: str = "alloy"
```

## Backwards Compatibility

The library maintains backwards compatibility by exposing Kokoro classes directly:

```python
# Old way (still works)
from champi_tts import KokoroConfig, KokoroProvider

# New way (recommended)
from champi_tts import get_provider
```

## Future Extensions

### Phase 2: Additional Providers
- OpenAI TTS integration
- ElevenLabs integration
- Edge TTS support
- Google Cloud TTS

### Phase 3: Advanced Features
- Voice cloning support
- SSML support
- Emotion control
- Multi-speaker synthesis

### Phase 4: Performance
- Streaming synthesis improvements
- Model caching optimizations
- GPU acceleration enhancements
- Batch processing support
