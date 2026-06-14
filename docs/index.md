# Champi TTS

Multi-provider text-to-speech library with voice reading features.

## Overview

Champi TTS is a modular, extensible Python library for text-to-speech synthesis
with support for multiple backends, full text reading capabilities with
interruption support, and visual UI indicators.

## Features

- **Multi-Provider TTS** — Kokoro (local, neural), with OpenAI and ElevenLabs coming soon
- **Text Reading Service** — Paragraph-by-paragraph processing with pause, resume, and stop controls
- **Visual UI Indicator** — Real-time status feedback via GLFW/ImGui
- **Audio Features** — Multiple voice options, adjustable speech speed, audio file export (WAV, MP3)

## Installation

```bash
uv pip install champi-tts
```

## Quick Start

```python
from champi_tts import get_provider
import asyncio

async def main():
    provider = get_provider()
    await provider.initialize()
    audio = await provider.synthesize("Hello, world!")
    await provider.shutdown()

asyncio.run(main())
```

See the [Getting Started guide](user/getting-started.md) for full setup instructions.

## Navigation

- [Getting Started](user/getting-started.md) — Installation and first steps
- [Python API Guide](user/python-api.md) — Full Python programming interface
- [CLI Guide](user/cli-guide.md) — Command-line usage
- [API Reference](api.md) — Complete API documentation
- [Contributing](developer/contributing.md) — How to contribute
