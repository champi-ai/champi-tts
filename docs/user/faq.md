# FAQ - Frequently Asked Questions

Common questions about Champi TTS answered.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Voice Selection](#voice-selection)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [API](#api)

---

## Installation

### What are the system requirements?

- **Python**: 3.12 or higher
- **RAM**: Minimum 2GB (4GB+ recommended)
- **Storage**: 500MB for model files and cache
- **GPU**: Optional (for Kokoro provider acceleration)

### How do I install Champi TTS?

```bash
# Basic installation
pip install champi-tts

# With development dependencies
pip install "champi-tts[dev]"

# With UI support
pip install "champi-tts[ui]"
```

### Can I use conda?

Yes! Use pip within conda:

```bash
conda create -n champi-tts python=3.12
conda activate champi-tts
pip install champi-tts
```

### Installation fails with SSL errors

```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org champi-tts
```

### How do I uninstall?

```bash
pip uninstall champi-tts
```

---

## Usage

### How do I use the CLI?

```bash
# Synthesize text
champi-tts synthesize "Hello, world!"

# Read a file
champi-tts read document.txt

# List voices
champi-tts list-voices

# Test UI
champi-tts test-ui
```

### How do I use Python API?

```python
from champi_tts import get_provider
import asyncio

async def main():
    provider = get_provider()
    await provider.initialize()
    audio = await provider.synthesize("Hello!")
    await provider.shutdown()

asyncio.run(main())
```

### Can I use multiple providers?

Yes! Create multiple providers:

```python
from champi_tts import get_provider

provider1 = get_provider("kokoro")
provider2 = get_provider("openai")

await provider1.initialize()
await provider2.initialize()

# Use each provider
audio1 = await provider1.synthesize("Test")
audio2 = await provider2.synthesize("Test")
```

### How do I pause/resume reading?

Use the reader service with interactive mode:

```python
reader = get_reader("kokoro", show_ui=True)
await reader.read_file("document.txt")

# Pause during reading
await reader.pause()

# Resume later
await reader.resume()
```

### How do I save audio to file?

```python
from champi_tts import get_provider
from champi_tts.core.audio import save_audio

async with get_provider() as provider:
    await provider.initialize()
    audio = await provider.synthesize("Hello!")
    await save_audio(audio, "output.wav")
```

---

## Voice Selection

### How many voices are available?

Kokoro provider includes multiple voices:
- af_bella: Female, friendly
- am_adam: Male, calm
- am_echo: Male, deep
- af_sarah: Female, energetic

### How do I choose the best voice?

- **Friendly tone**: af_bella
- **Calm tone**: am_adam
- **Deep tone**: am_echo
- **Energetic tone**: af_sarah

### Can I add custom voices?

Yes, download custom voices using the CLI:

```bash
champi-tts download-voices
```

### How do I list available voices programmatically?

```python
voices = await provider.list_voices()
print(voices)
```

---

## Performance

### How fast is synthesis?

Typical synthesis time:
- Short text (10 words): ~1-2 seconds
- Medium text (100 words): ~10-20 seconds
- Long text (1000 words): ~2-5 minutes

GPU acceleration can significantly improve performance.

### Why is my synthesis slow?

Possible reasons:
1. **CPU-only mode**: Enable GPU
2. **Large batch size**: Reduce chunk size
3. **Model path issues**: Check model path configuration
4. **Network latency**: If downloading models

### How can I improve performance?

```python
# Enable GPU
config = KokoroConfig(use_gpu=True)

# Use smaller chunks for streaming
async for chunk in provider.synthesize_streaming(text):
    # Process quickly
    pass

# Reuse providers to avoid initialization overhead
```

### What is the maximum audio duration?

There's no hard limit, but:
- Memory usage increases with duration
- Consider processing in chunks
- Monitor system resources

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'champi_tts'"

**Solution**: Install the package:

```bash
pip install champi-tts
```

### "Voice not found: af_bella"

**Solution**: Download voices:

```bash
champi-tts download-voices
```

### "CUDA out of memory"

**Solution**: Disable GPU:

```python
config = KokoroConfig(use_gpu=False)
```

### Audio not playing

**Solution**: Check your audio output:

```bash
# Linux
aplay test.wav

# macOS
afplay test.wav

# Windows
start test.wav
```

### UI not displaying

**Solution**: Install UI dependencies:

```bash
pip install "champi-tts[ui]"
```

### "InitializationError: Provider failed to initialize"

**Solution**: Check:
1. Python version is 3.12+
2. Model files are downloaded
3. CUDA drivers installed (if using GPU)
4. Check logs for specific errors

### Command not found: champi-tts

**Solution**: Make sure champi-tts is installed and in your PATH:

```bash
# Check installation
python -c "import champi_tts; print(champi_tts.__file__)"

# Or reinstall
pip install --force-reinstall champi-tts
```

---

## Development

### How do I run tests?

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_provider.py -v

# With coverage
pytest tests/ --cov=champi_tts --cov-report=html
```

### How do I contribute?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

See [CONTRIBUTING.md](../developer/contributing.md) for details.

### How do I run development server?

```bash
pip install "champi-tts[dev]"
```

### How do I set up debugging?

```python
import logging

logging.basicConfig(level=logging.DEBUG)

async def main():
    provider = get_provider()
    await provider.initialize()
```

### How do I run linting?

```bash
# Lint code
ruff check .

# Format code
black .
```

---

## API

### What async operations are available?

All synthesis and reading operations are async:

```python
await provider.synthesize("Text")
await provider.initialize()
await provider.shutdown()
```

### How do I handle errors?

```python
try:
    async with get_provider() as provider:
        await provider.initialize()
        audio = await provider.synthesize("Text")
except InitializationError:
    print("Initialization failed")
except SynthesisError:
    print("Synthesis failed")
```

### How do I use events?

```python
reader.on_reading_started.connect(lambda s, **kw: print("Started"))
reader.on_reading_completed.connect(lambda s, **kw: print("Completed"))
```

### Can I use the library in web applications?

Yes! Use it in any Python web framework (Flask, Django, FastAPI).

```python
from fastapi import FastAPI
from champi_tts import get_provider

app = FastAPI()

@app.post("/synthesize")
async def synthesize(text: str):
    async with get_provider() as provider:
        await provider.initialize()
        audio = await provider.synthesize(text)
        return {"status": "success", "audio_size": len(audio)}
```

### How do I stream audio?

```python
async for chunk in provider.synthesize_streaming("Text"):
    # Send chunk to client
    pass
```

### What's the difference between reader and provider?

- **Provider**: Handles TTS synthesis
- **Reader**: Manages reading with controls, queues, and state

---

## General

### Is Champi TTS free?

Yes, Champi TTS is open-source and free to use.

### Can I use it commercially?

Yes, see the LICENSE file for details.

### What's the license?

Champi TTS is licensed under the MIT License.

### How do I report a bug?

[Open an issue on GitHub](https://github.com/divagnz/champi-tts/issues)

### Where can I get help?

- Documentation: `docs/` directory
- Examples: `examples/` directory
- GitHub Discussions: [Discussions](https://github.com/divagnz/champi-tts/discussions)
- Issues: [Issues](https://github.com/divagnz/champi-tts/issues)

### How do I stay updated?

- Star the repository on GitHub
- Follow releases
- Join discussions

---

## Getting More Help

If your question isn't answered here:

1. Check the [API Reference](api.md)
2. Review the [Examples](../examples/)
3. Search existing [Issues](https://github.com/divagnz/champi-tts/issues)
4. Ask in [GitHub Discussions](https://github.com/divagnz/champi-tts/discussions)

---

**Still stuck?** [Open an issue](https://github.com/divagnz/champi-tts/issues) with details about your problem!