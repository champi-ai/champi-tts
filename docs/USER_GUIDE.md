# Champi TTS User Guide

A comprehensive guide to using champi-tts, the multi-provider text-to-speech library with voice reading features.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Features](#features)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [CLI Guide](#cli-guide)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Contributing](#contributing)

## Introduction

Champi TTS is a modular, extensible Python library for text-to-speech synthesis with support for multiple backends, full text reading capabilities with interruption support, and visual UI indicators. It provides a unified interface across different TTS providers and includes powerful features for reading long documents, managing audio queues, and real-time playback control.

### Key Features

- **Multi-Provider Support**: Kokoro (local, neural TTS), OpenAI TTS API, ElevenLabs (coming soon)
- **Text Reading Service**: Read text files with paragraph-by-paragraph processing, queue management, and pause/resume/stop controls
- **Visual UI Indicator**: Real-time visual status indicator using GLFW/ImGui
- **High-Quality Audio**: Multiple voice options, adjustable speech speed, audio file export
- **Interruption Support**: Immediate stopping of audio playback
- **Event-Driven Architecture**: Robust state tracking and event management

### Who Should Use Champi TTS

- Developers building accessibility tools
- Content creators needing text-to-speech functionality
- Applications requiring real-time audio generation
- Projects needing multi-provider TTS capabilities
- Anyone looking for high-quality, customizable text-to-speech

## Installation

### Basic Installation

Install the latest version using pip:

```bash
pip install champi-tts
```

Or using the recommended package manager uv:

```bash
uv pip install champi-tts
```

### Development Installation

For development purposes:

```bash
git clone https://github.com/divagnz/champi-tts.git
cd champi-tts
uv sync --extra dev
```

### Platform-Specific Installation

#### Linux
No special dependencies required. Standard Python installation works.

#### macOS
No special dependencies required. Ensure you have Python 3.12 or higher.

#### Windows
No special dependencies required. Standard Python installation works.

### GPU Support (Kokoro Provider)

For improved performance with Kokoro provider on systems with NVIDIA GPUs:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Troubleshooting Installation

**Issue**: Module not found after installation

```bash
# Ensure you're using the correct Python environment
python --version
pip show champi-tts
```

**Issue**: CUDA-related errors

```bash
# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Simple Synthesis

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

### Text Reading

```python
from champi_tts import get_reader

# Create reader with UI
reader = get_reader("kokoro", show_ui=True)
await reader.provider.initialize()

# Read text file
await reader.read_file("document.txt", voice="af_bella")

await reader.provider.shutdown()
```

### CLI Usage

```bash
# Synthesize and play
champi-tts synthesize "Hello, world!" --voice af_bella

# Read text file with UI
champi-tts read document.txt --voice af_bella --show-ui
```

## Core Concepts

### Provider

A provider is an abstraction for a TTS backend. Champi TTS supports multiple providers with a unified interface.

```python
from champi_tts import get_provider

# Get a specific provider
provider = get_provider("kokoro")
```

### Reader

The reader is a service for reading text with advanced features like queue management, pause/resume controls, and interruption support.

```python
from champi_tts import get_reader

reader = get_reader("kokoro", show_ui=True)
```

### Audio

Audio management handles the synthesis and playback of text-to-speech output.

```python
from champi_tts.core.audio import save_audio, load_audio
```

### UI Indicator

A visual indicator that shows the current state of text reading (Idle, Processing, Speaking, Paused, Error).

```python
# Start UI indicator in test mode
from champi_tts.ui import UIIndicator
indicator = UIIndicator(mode="test")
```

## Features

### Multi-Provider TTS Support

Champi TTS supports multiple TTS providers with a unified interface.

**Current Providers:**
- **Kokoro**: Local, neural TTS engine
- **OpenAI TTS API**: Cloud-based TTS (coming soon)
- **ElevenLabs**: High-quality cloud TTS (coming soon)

**Switching Providers:**

```python
# Get different providers
kokoro_provider = get_provider("kokoro")
# openai_provider = get_provider("openai")
# elevenlabs_provider = get_provider("elevenlabs")
```

### Text Reading Service

The text reading service provides powerful features for reading text with control and flexibility.

**Features:**
- Read text files paragraph-by-paragraph
- Queue management for long documents
- Pause/resume/stop controls
- Interruption support
- Event-driven state tracking

**Usage:**

```python
reader = get_reader("kokoro", show_ui=True)
await reader.read_file("document.txt", voice="af_bella")

# Control playback
await reader.pause()
await reader.resume()
await reader.stop()
```

### Visual UI Indicator

A visual indicator that shows the current state of text reading operations.

**States:**
- **Idle**: Waiting for text
- **Processing**: Processing text for synthesis
- **Speaking**: Currently speaking
- **Paused**: Reading paused
- **Error**: Error occurred

**Usage:**

```python
from champi_tts.ui import UIIndicator

# Test mode
indicator = UIIndicator(mode="test")
await indicator.start()
# Show different states
await indicator.set_state("speaking")
```

### Audio Features

High-quality audio synthesis with various options.

**Features:**
- Multiple voice options
- Adjustable speech speed
- Audio file export (WAV, MP3, etc.)
- Real-time audio playback with interruption
- Audio quality settings

**Usage:**

```python
# Synthesize with custom settings
audio = await provider.synthesize(
    text="Hello, world!",
    voice="af_bella",
    speed=1.0
)

# Save to file
await save_audio(audio, "output.wav", sample_rate=24000)
```

## Configuration

### Provider Configuration

Configure providers based on your needs.

```python
# Kokoro provider configuration
config = {
    "voice": "af_bella",
    "speed": 1.0,
    "sample_rate": 24000,
    "use_gpu": True
}

provider = get_provider("kokoro", config=config)
```

### Reader Configuration

Configure the text reading service.

```python
# Reader with UI and custom settings
reader = get_reader(
    provider="kokoro",
    voice="af_bella",
    show_ui=True,
    speed=1.0
)
```

### Environment Variables

Configure providers using environment variables.

```bash
# Kokoro provider
export KOKORO_VOICE=af_bella
export KOKORO_SPEED=1.0

# OpenAI provider (if supported)
export OPENAI_API_KEY=your_api_key
```

## API Reference

### Basic Usage

#### Getting a Provider

```python
from champi_tts import get_provider

# Get default provider
provider = get_provider()

# Get specific provider
kokoro = get_provider("kokoro")
```

#### Initializing and Shutting Down

```python
await provider.initialize()

# ... do work ...

await provider.shutdown()
```

#### Synthesizing Text

```python
# Basic synthesis
audio = await provider.synthesize("Hello, world!")

# Synthesize with options
audio = await provider.synthesize(
    text="Hello, world!",
    voice="af_bella",
    speed=1.0
)
```

### Text Reading Service

#### Creating a Reader

```python
from champi_tts import get_reader

# Basic reader
reader = get_reader("kokoro")

# Reader with UI
reader = get_reader("kokoro", show_ui=True)

# Reader with custom settings
reader = get_reader(
    "kokoro",
    voice="af_bella",
    show_ui=True,
    speed=1.0
)
```

#### Reading Text

```python
# Read text file
await reader.read_file("document.txt", voice="af_bella")

# Read text directly
await reader.read("This is a test", voice="af_bella")
```

#### Controlling Playback

```python
# Pause reading
await reader.pause()

# Resume reading
await reader.resume()

# Stop reading
await reader.stop()
```

### Audio Management

#### Saving Audio

```python
from champi_tts.core.audio import save_audio

# Save to WAV
await save_audio(audio, "output.wav", sample_rate=24000)

# Save to MP3 (requires ffmpeg)
await save_audio(audio, "output.mp3", sample_rate=24000)
```

#### Loading Audio

```python
from champi_tts.core.audio import load_audio

audio = await load_audio("input.wav", sample_rate=24000)
```

### UI Indicator

#### Creating UI Indicator

```python
from champi_tts.ui import UIIndicator

# Test mode
indicator = UIIndicator(mode="test")

# Production mode with GLFW/ImGui
indicator = UIIndicator(mode="production")
```

#### Controlling UI Indicator

```python
# Start indicator
await indicator.start()

# Set state
await indicator.set_state("speaking")
await indicator.set_state("paused")
await indicator.set_state("error")

# Stop indicator
await indicator.stop()
```

## CLI Guide

### Installation

Ensure champi-tts is installed:

```bash
pip install champi-tts
```

### Available Commands

#### `champi-tts synthesize`

Synthesize text to speech.

```bash
# Basic usage
champi-tts synthesize "Hello, world!"

# With voice
champi-tts synthesize "Hello, world!" --voice af_bella

# Save to file
champi-tts synthesize "Hello, world!" --output output.wav

# No playback
champi-tts synthesize "Hello, world!" --no-play

# With speed setting
champi-tts synthesize "Hello, world!" --speed 1.2
```

#### `champi-tts read`

Read text file with text-to-speech.

```bash
# Read file
champi-tts read document.txt --voice af_bella

# Read with UI
champi-tts read document.txt --voice af_bella --show-ui

# Read text directly
champi-tts read --text "This is a test" --voice af_bella

# Interactive mode
champi-tts read document.txt --interactive --show-ui
```

#### `champi-tts list-voices`

List available voices.

```bash
champi-tts list-voices
```

#### `champi-tts test-ui`

Run UI indicator in standalone test mode.

```bash
champi-tts test-ui
```

### Common CLI Workflows

#### Synthesize and Save

```bash
champi-tts synthesize "Hello, world!" --output greeting.wav --no-play
```

#### Read Document with UI

```bash
champi-tts read presentation.txt --voice af_bella --show-ui
```

#### List Available Voices

```bash
champi-tts list-voices
```

## Troubleshooting

### Installation Issues

**Issue**: Import error after installation

```bash
# Ensure you're using the correct Python environment
python --version
pip install --upgrade champi-tts
```

**Issue**: Module not found

```bash
# Reinstall the package
pip uninstall champi-tts
pip install champi-tts
```

### TTS Issues

**Issue**: No voices available

```bash
# Check if provider is initialized correctly
python -c "from champi_tts import get_provider; p = get_provider(); print(p.voices)"
```

**Issue**: Audio not playing

```bash
# Check system audio settings
# Try a different voice
champi-tts synthesize "Test" --voice af_bella
```

**Issue**: Poor audio quality

```bash
# Try a different voice
champi-tts list-voices
champi-tts synthesize "Test" --voice <different_voice>
```

### Text Reading Issues

**Issue**: Cannot pause/resume

```python
# Ensure UI is enabled
reader = get_reader("kokoro", show_ui=True)
```

**Issue**: Long documents not working

```python
# The reader handles long documents automatically with queue management
await reader.read_file("large_document.txt", voice="af_bella")
```

### UI Issues

**Issue**: UI not showing

```bash
# Install required dependencies
# On Linux: sudo apt-get install libglfw3-dev
# On macOS: brew install glfw
```

**Issue**: UI window not closing

```bash
# Close the window normally
# If stuck, use Ctrl+C in terminal
```

### Performance Issues

**Issue**: Slow synthesis

```python
# Enable GPU for Kokoro provider
config = {"use_gpu": True}
provider = get_provider("kokoro", config=config)
```

**Issue**: High memory usage

```python
# Process documents in smaller chunks
# The reader handles this automatically
```

## FAQ

### What is the difference between providers?

Different providers offer different trade-offs:

- **Kokoro**: Local, no API keys needed, good quality
- **OpenAI TTS**: High quality, requires API key, cloud-based
- **ElevenLabs**: Premium quality, requires API key, cloud-based

### Can I use multiple providers in the same application?

Yes, you can create multiple provider instances and switch between them as needed.

### How do I save synthesized audio?

Use the `save_audio` function or the `--output` flag with CLI.

### What audio formats are supported?

WAV and MP3 formats are supported (MP3 requires ffmpeg).

### Can I customize speech speed?

Yes, both API and CLI support speed customization.

### How do I handle long documents?

The text reading service automatically manages long documents with queue processing.

### What are the system requirements?

- Python 3.12+
- At least 2GB RAM
- For GPU support: NVIDIA GPU with CUDA

### Can I interrupt audio playback?

Yes, the reader supports immediate interruption of audio playback.

### How do I list available voices?

Use the CLI command: `champi-tts list-voices`

### Is there a GUI application?

The UI indicator provides visual feedback, but there is no standalone GUI application yet.

### Can I integrate with my web application?

Yes, champi-tts is a Python library that can be integrated into web applications using frameworks like Flask or Django.

## Contributing

We welcome contributions! Please see the [Contributing Guide](CONTRIBUTING.md) for details.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit using conventional commits (`feat: add new feature`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

### Code of Conduct

Please be respectful and constructive in all interactions.

### Getting Help

For questions, issues, or feature requests, please [open an issue](https://github.com/divagnz/champi-tts/issues).

---

**Version**: 0.2.0
**Last Updated**: 2026-06-11
**License**: MIT