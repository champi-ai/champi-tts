# Champi-TTS Examples

This directory contains example scripts demonstrating various features of champi-tts.

## Examples

### Basic Usage
- [`basic_synthesis.py`](basic_synthesis.py) - Simple text-to-speech synthesis
- [`file_reading.py`](file_reading.py) - Reading text files with pause/resume
- [`streaming.py`](streaming.py) - Streaming synthesis for lower latency

### Advanced Features
- [`event_handling.py`](event_handling.py) - Using event signals
- [`custom_provider.py`](custom_provider.py) - Creating a custom TTS provider
- [`with_ui.py`](with_ui.py) - Using the visual UI indicator
- [`async_context.py`](async_context.py) - Async context managers for resource management

## Running Examples

```bash
# Install with all features
uv pip install -e ".[all]"

# Run an example
python examples/basic_synthesis.py
```

## Requirements

All examples require champi-tts with the appropriate optional dependencies installed.
