# Champi TTS

[![CI](https://github.com/champi-ai/champi-tts/workflows/CI/badge.svg)](https://github.com/champi-ai/champi-tts/actions)
[![Coverage](https://raw.githubusercontent.com/champi-ai/champi-tts/main/badges/coverage.svg)](https://github.com/champi-ai/champi-tts/actions)
[![Pre-commit](https://github.com/champi-ai/champi-tts/workflows/Pre-commit/badge.svg)](https://github.com/champi-ai/champi-tts/actions)
[![PyPI version](https://badge.fury.io/py/champi-tts.svg)](https://badge.fury.io/py/champi-tts)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Multi-Provider Text-to-Speech Library with Voice Reading Features**

A modular, extensible Python library for text-to-speech synthesis with support for multiple backends, full text reading capabilities with interruption support, and visual UI indicators.

---

## ✨ Features

### 🎯 **Multi-Provider TTS Support**
- **Kokoro** (Local, neural TTS) - ✅ Implemented
- **OpenAI TTS API** - 🔜 Coming Soon
- **ElevenLabs** - 🔜 Coming Soon
- Unified interface across all providers

### 📖 **Text Reading Service**
- Read text files with paragraph-by-paragraph processing
- Queue management for long documents
- Pause/resume/stop controls
- Interruption support for immediate stopping
- Event-driven architecture for state tracking

### 👁️ **Visual UI Indicator**
- Real-time visual status indicator using GLFW/ImGui
- Visual states:
  - ⚪ **Idle** - Waiting for text
  - 🔄 **Processing** - Processing text
  - 🔊 **Speaking** - Currently speaking
  - ⏸️ **Paused** - Reading paused
  - ❌ **Error** - Error occurred
- Pulsing animations for active states
- Standalone testing mode

### 🎵 **Audio Features**
- High-quality audio synthesis
- Multiple voice options
- Adjustable speech speed
- Audio file export (WAV, MP3, etc.)
- Real-time audio playback with interruption

---

## 📦 Installation

### Basic Installation

```bash
uv pip install champi-tts
```

### Development Installation

```bash
git clone https://github.com/champi-ai/champi-tts.git
cd champi-tts
uv sync --extra dev
```

### Kokoro on Python 3.13

The `kokoro` extra pulls in `misaki[en]` → `spacy-curated-transformers` 0.3.x,
which caps `curated-tokenizers` below 0.1.0. Its only such release (0.0.9) has no
Python 3.13 wheels and its source build fails on 3.13. `curated-tokenizers` 2.x
keeps the same tokenizer API and ships 3.12/3.13 wheels, so this project
overrides it in `pyproject.toml`:

```toml
[tool.uv]
override-dependencies = ["curated-tokenizers>=2.0.1,<3.0.0"]
```

`uv sync --extra kokoro` therefore works on Python 3.12 and 3.13. The override
is only used by uv project commands (`uv sync`, `uv run`, `uv lock`); it is not
part of the published package metadata. When installing the wheel elsewhere
(`uvx`, `uv tool install`, `pip`) on Python 3.13, pass the same override, or use
Python 3.12:

```bash
echo "curated-tokenizers>=2.0.1,<3.0.0" > overrides.txt
uvx --overrides overrides.txt --from "champi-tts[kokoro] @ <wheel-url>" champi-tts
# or
uvx --python 3.12 --from "champi-tts[kokoro] @ <wheel-url>" champi-tts
```

---

## 🚀 Quick Start

### 1. Simple Synthesis

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

### 2. CLI Synthesis

```bash
# Synthesize and play
champi-tts synthesize "Hello, world!" --voice af_bella

# Synthesize and save
champi-tts synthesize "Hello, world!" --output hello.wav --no-play

# List available voices
champi-tts list-voices
```

### 3. Text Reading

```python
from champi_tts import get_reader

# Create reader with UI
reader = get_reader("kokoro", show_ui=True)
await reader.provider.initialize()

# Read text file
await reader.read_file("document.txt", voice="af_bella")

# Control playback
await reader.pause()
await reader.resume()
await reader.stop()

await reader.provider.shutdown()
```

### 4. CLI Text Reading

```bash
# Read text file with UI
champi-tts read document.txt --voice af_bella --show-ui

# Read text directly
champi-tts read --text "This is a test" --voice am_adam

# Interactive mode (with pause/resume controls)
champi-tts read document.txt --interactive --show-ui
```

### 5. Test UI Indicator

```bash
# Run standalone UI test
champi-tts test-ui
```

---

## 🤖 MCP Server

Champi TTS exposes its speech tools over the
[Model Context Protocol (MCP)](https://modelcontextprotocol.io/), so LLM hosts
such as Claude Desktop and Claude Code can speak text aloud or write it to an
audio file.

Run it from a release wheel (replace `X.Y.Z` with the
[latest release](https://github.com/champi-ai/champi-tts/releases/latest)):

```bash
uvx --python 3.12 --from "champi-tts[mcp,kokoro] @ https://github.com/champi-ai/champi-tts/releases/download/vX.Y.Z/champi_tts-X.Y.Z-py3-none-any.whl" champi-tts mcp serve
```

From a source checkout: `uv sync --extra mcp --extra kokoro`, then
`uv run champi-tts mcp serve`.

Add it to `claude_desktop_config.json` (Claude Desktop) or a project
`.mcp.json` (Claude Code):

```json
{
  "mcpServers": {
    "champi-tts": {
      "command": "uvx",
      "args": [
        "--python",
        "3.12",
        "--from",
        "champi-tts[mcp,kokoro] @ https://github.com/champi-ai/champi-tts/releases/download/vX.Y.Z/champi_tts-X.Y.Z-py3-none-any.whl",
        "champi-tts",
        "mcp",
        "serve"
      ]
    }
  }
}
```

Tools: `speak_tool`, `speak_streaming_tool`, `stop_speaking_tool`,
`synthesize_to_file`, `list_voices`, `list_audio_devices_tool`,
`list_providers` and `get_provider_status`. Speaker playback needs PortAudio
(`sudo apt install portaudio19-dev` on Debian/Ubuntu). Run
`champi-tts mcp serve --help` for the SSE transport options.

See [docs/mcp-integration.md](docs/mcp-integration.md) for prerequisites,
environment variables, a smoke test and the full tool reference.

---

## 📚 Documentation

### Architecture
See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

### Changelog
See [CHANGELOG.md](CHANGELOG.md) for version history and changes.

---

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/champi-ai/champi-tts.git
cd champi-tts

# Install with development dependencies
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/champi_tts --cov-report=term --cov-report=html

# Run specific test file
uv run pytest tests/test_provider.py
```

### Code Quality

```bash
# Format code
uv run ruff format src/

# Lint code
uv run ruff check src/

# Type checking
uv run mypy src/

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

---

## 📁 Project Structure

```
champi-tts/
├── src/champi_tts/
│   ├── core/                  # Generic abstractions
│   │   ├── base_config.py
│   │   ├── base_provider.py
│   │   ├── base_synthesizer.py
│   │   └── audio.py
│   ├── providers/             # TTS implementations
│   │   └── kokoro/
│   ├── reader/                # Text reading service
│   ├── ui/                    # Visual UI indicator
│   ├── cli/                   # CLI interface
│   └── factory.py             # Provider factory
├── tests/                     # Test suite
├── .github/workflows/         # CI/CD workflows
└── pyproject.toml             # Project configuration
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes using [Conventional Commits](https://www.conventionalcommits.org/)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Commit Message Format

This project uses [Conventional Commits](https://www.conventionalcommits.org/) for automatic versioning:

```
feat: add new feature
fix: bug fix
docs: documentation changes
chore: maintenance tasks
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with:
- [Kokoro TTS](https://github.com/thewh1teagle/kokoro-onnx) - Neural text-to-speech engine
- [champi-signals](https://github.com/champi-ai/champi-signals) - Event-driven signal management
- [imgui-bundle](https://github.com/pthom/imgui_bundle) - ImGui bindings for Python
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Beautiful terminal output

---

## 📬 Contact

For questions, issues, or feature requests, please [open an issue](https://github.com/champi-ai/champi-tts/issues).

---

**Status**: ✅ **PRODUCTION READY**
