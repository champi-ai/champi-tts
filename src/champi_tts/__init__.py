"""
Champi TTS - Multi-Provider Text-to-Speech Library
====================================================

A modular, extensible TTS library supporting multiple providers:
- Kokoro (local, neural TTS)
- OpenAI (coming soon)
- ElevenLabs (coming soon)

Quick Start:
-----------
```python
from champi_tts import get_provider

# Get default provider (Kokoro)
provider = get_provider()
await provider.initialize()

# Synthesize audio
audio = await provider.synthesize("Hello world")
```

Text Reading:
------------
```python
from champi_tts import get_reader

# Create reader with UI
reader = get_reader("kokoro", show_ui=True)

# Read text file
await reader.read_file("document.txt")

# Pause/resume/stop
await reader.pause()
await reader.resume()
await reader.stop()
```

Provider-Specific Usage:
-----------------------
```python
from champi_tts import get_provider
from champi_tts.providers.kokoro import KokoroConfig

# Custom Kokoro config
config = KokoroConfig(
    default_voice="af_bella",
    default_speed=1.1,
    use_gpu=True
)
provider = get_provider("kokoro", config=config)
```
"""

# Factory functions (primary API)
# Base classes (for type hints and custom providers)
from champi_tts.core.base_config import BaseTTSConfig
from champi_tts.core.base_provider import BaseTTSProvider
from champi_tts.core.base_synthesizer import BaseSynthesizer
from champi_tts.factory import (
    get_default_provider,
    get_provider,
    get_reader,
    list_providers,
)

# Reader service
from champi_tts.reader import ReaderState, TextReaderService

__version__ = "1.3.2"

# Names from optional extras that are loaded lazily to avoid pulling in
# heavy dependencies (torch, kokoro, imgui-bundle) at import time.
_KOKORO_EXPORTS = frozenset(
    {"KokoroConfig", "KokoroInference", "KokoroProvider", "VoiceManager"}
)
_UI_EXPORTS = frozenset({"TTSIndicatorUI", "TTSState"})


def __getattr__(name: str) -> object:
    """Lazily import optional-extra symbols to keep startup fast."""
    if name in _KOKORO_EXPORTS:
        from champi_tts.providers.kokoro import (
            KokoroConfig,
            KokoroInference,
            KokoroProvider,
            VoiceManager,
        )

        _resolved = {
            "KokoroConfig": KokoroConfig,
            "KokoroInference": KokoroInference,
            "KokoroProvider": KokoroProvider,
            "VoiceManager": VoiceManager,
        }
        globals().update(_resolved)
        return _resolved[name]

    if name in _UI_EXPORTS:
        from champi_tts.ui import TTSIndicatorUI, TTSState

        _resolved = {
            "TTSIndicatorUI": TTSIndicatorUI,
            "TTSState": TTSState,
        }
        globals().update(_resolved)
        return _resolved[name]

    raise AttributeError(f"module 'champi_tts' has no attribute {name!r}")


__all__ = [
    "BaseSynthesizer",
    # Base classes
    "BaseTTSConfig",
    "BaseTTSProvider",
    # Backwards compatibility (Kokoro)
    "KokoroConfig",
    "KokoroInference",
    "KokoroProvider",
    "ReaderState",
    # UI components
    "TTSIndicatorUI",
    "TTSState",
    # Reader service
    "TextReaderService",
    "VoiceManager",
    # Version
    "__version__",
    "get_default_provider",
    # Factory functions (recommended API)
    "get_provider",
    "get_reader",
    "list_providers",
]
