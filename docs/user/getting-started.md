# Getting Started Guide

Welcome to Champi TTS! This guide will help you get up and running quickly with the multi-provider text-to-speech library.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Verification](#verification)
- [Next Steps](#next-steps)

---

## Prerequisites

Before installing Champi TTS, ensure your system meets the following requirements:

- **Python**: 3.12 or higher
- **Package Manager**: pip or uv
- **Optional**: CUDA-capable GPU (for Kokoro provider)

### System Requirements

- **Minimum**: 2GB RAM
- **Recommended**: 4GB+ RAM
- **Storage**: 500MB+ for model files and cache

---

## Installation

### Basic Installation

The easiest way to install Champi TTS is using pip:

```bash
pip install champi-tts
```

### Development Installation

For development purposes, install with all optional dependencies:

```bash
pip install "champi-tts[dev]"
```

This includes:
- Development tools
- Test suite
- Type checking (mypy)
- Code quality tools (ruff, black)

### UI Support

To use the visual UI indicator:

```bash
pip install "champi-tts[ui]"
```

### GPU Support

For Kokoro provider with GPU acceleration:

```bash
pip install "champi-tts[gpu]"
```

Or simply set `use_gpu=True` in your configuration.

### Using uv

If you're using uv as your package manager:

```bash
uv pip install champi-tts
```

---

## Quick Start

Let's get you running in just a few steps!

### Step 1: Create a Python Script

Create a file named `hello.py`:

```python
from champi_tts import get_provider
import asyncio

async def main():
    # Get default provider (Kokoro)
    provider = get_provider()

    # Initialize the provider
    await provider.initialize()

    # Synthesize some text
    audio = await provider.synthesize("Hello, world!")

    # Save to file
    from champi_tts.core.audio import save_audio
    await save_audio(audio, "hello.wav", sample_rate=provider.config.sample_rate)

    print("Audio saved to hello.wav")

    # Clean up
    await provider.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 2: Run Your Script

```bash
python hello.py
```

### Step 3: Verify Installation

Check that Champi TTS is properly installed:

```bash
python -c "import champi_tts; print(champi_tts.__version__)"
```

---

## Verification

### Test Basic Synthesis

Create a simple test script:

```python
from champi_tts import get_provider
import asyncio

async def test_basic():
    provider = get_provider()
    await provider.initialize()

    try:
        audio = await provider.synthesize("This is a test of Champi TTS")
        print(f"Synthesized audio shape: {audio.shape}")
        print("Success! Champi TTS is working correctly.")
    finally:
        await provider.shutdown()

asyncio.run(test_basic())
```

### Test CLI

Try the command-line interface:

```bash
champi-tts --version
```

```bash
champi-tts list-voices
```

```bash
champi-tts synthesize "Hello from CLI!" --output test.wav
```

### Test UI

Run the UI indicator test:

```bash
champi-tts test-ui
```

---

## Next Steps

Now that you've installed Champi TTS, you can:

- [Read Text with the Reader Service](text-reading.md) - Learn how to read text files and documents
- [Use CLI Commands](cli-guide.md) - Explore all command-line options
- [Python API Guide](python-api.md) - Deep dive into Python programming interface
- [Audio Features Guide](audio-features.md) - Control voice, speed, and audio quality
- [UI Indicator Guide](ui-indicator.md) - Use the visual feedback system
- [Check the API Reference](api.md) - Complete API documentation with examples

## Common Issues

### ImportError: No module named 'champi_tts'

**Solution**: Make sure you installed the package correctly:

```bash
pip install champi-tts
# or
uv pip install champi-tts
```

### Voice Not Found Error

**Solution**: Ensure model files are downloaded:

```bash
champi-tts download-voices
```

### CUDA Out of Memory

**Solution**: Disable GPU usage:

```python
from champi_tts.providers.kokoro import KokoroConfig
config = KokoroConfig(use_gpu=False)
```

---

## Getting Help

- **Documentation**: Full documentation available in the `docs/` directory
- **Examples**: Sample code in `examples/` directory
- **Issues**: [Open an issue on GitHub](https://github.com/divagnz/champi-tts/issues)
- **Discussions**: [GitHub Discussions](https://github.com/divagnz/champi-tts/discussions)

---

**Ready to dive deeper?** Check out the [Text Reading Guide](text-reading.md) to start reading text files!