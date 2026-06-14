# Text Reading Guide

Learn how to use the Champi TTS text reading service for reading text files and documents.

## Table of Contents

- [Overview](#overview)
- [Basic Text Reading](#basic-text-reading)
- [Reading Text Files](#reading-text-files)
- [Queue Management](#queue-management)
- [Reading Controls](#reading-controls)
- [Text Processing](#text-processing)
- [Audio Quality](#audio-quality)
- [Advanced Usage](#advanced-usage)

---

## Overview

The Text Reading Service allows you to read text documents paragraph-by-paragraph with:

- Automatic paragraph detection
- Queue management for long documents
- Pause/resume/stop controls
- Real-time state tracking
- Visual UI indicator support

---

## Basic Text Reading

### Read Text String

```python
from champi_tts import get_reader

reader = get_reader("kokoro")
await reader.read_text("This is a sample text")
```

### Read with Voice

```python
reader = get_reader("kokoro")
await reader.read_text("Hello, world!", voice="af_bella")
```

### Read with Voice and Speed

```python
reader = get_reader("kokoro")
await reader.read_text("Hello", voice="af_bella", speed=1.2)
```

---

## Reading Text Files

### Basic File Reading

```python
reader = get_reader("kokoro")
await reader.read_file("document.txt")
```

### Reading with UI

```python
reader = get_reader("kokoro", show_ui=True)
await reader.read_file("document.txt")
```

### Reading with Interactive Controls

```python
reader = get_reader("kokoro", show_ui=True)
await reader.read_file("long_document.txt")

# User can pause/resume during reading
await reader.pause()
# ... do something ...
await reader.resume()
```

### Reading with Custom Voice

```python
reader = get_reader("kokoro")
await reader.read_file("document.txt", voice="af_bella")
```

### Reading with Custom Speed

```python
reader = get_reader("kokoro")
await reader.read_file("document.txt", speed=1.1)
```

---

## Queue Management

### Adding to Queue

```python
reader = get_reader("kokoro")

# Add individual texts
reader.add_to_queue("Text one")
reader.add_to_queue("Text two")
reader.add_to_queue("Text three")

# Start reading
await reader.read_queue()
```

### Reading Queue Items Individually

```python
reader = get_reader("kokoro")

for text in ["First text", "Second text", "Third text"]:
    await reader.read_text(text)
```

### Clearing Queue

```python
reader = get_reader("kokoro")

# Add some items
reader.add_to_queue("Text 1")
reader.add_to_queue("Text 2")

# Clear before reading
reader.clear_queue()
```

### Queue with Audio Export

```python
reader = get_reader("kokoro")

# Add texts
texts = [
    "Introduction to Champi TTS",
    "How to use the library",
    "Advanced features guide"
]

for text in texts:
    reader.add_to_queue(text)

# Read and save audio for each
async for text in reader.read_queue():
    audio = await provider.synthesize(text)
    await save_audio(audio, f"{text[:10]}.wav")
```

---

## Reading Controls

### Pause Reading

```python
reader = get_reader("kokoro")
await reader.read_text("This text will be read...")
await reader.pause()

# Do something else...
# ...

# Resume reading
await reader.resume()
```

### Stop Reading

```python
reader = get_reader("kokoro")
await reader.read_text("This will be stopped")

# Stop immediately
await reader.stop()
```

### Interrupt Reading

```python
reader = get_reader("kokoro")
await reader.read_text("Long text...")

# Stop immediately without saving state
await reader.interrupt()
```

### State Management

```python
from champi_tts.reader import ReaderState

reader = get_reader("kokoro")
print(f"Initial state: {reader.state}")

# Reading state changes automatically
await reader.read_text("Hello")

# Check state
print(f"Current state: {reader.state}")
```

### Async Context Manager

```python
async with get_reader("kokoro") as reader:
    await reader.read_file("document.txt")
    # Automatic cleanup on exit
```

---

## Text Processing

### Automatic Paragraph Detection

Champi TTS automatically detects paragraphs based on line breaks and whitespace:

```python
# Each paragraph is read separately
paragraph1 = "First paragraph"
paragraph2 = "Second paragraph"
paragraph3 = "Third paragraph"
```

### Manual Text Segmentation

```python
import re

def split_text_into_paragraphs(text, max_chars=500):
    """Split text into paragraphs of max characters"""
    paragraphs = re.split(r'(\n{2,}|.{50,}\n)', text)
    return [p.strip() for p in paragraphs if p.strip()]

# Usage
text = """Long text here.
Another paragraph here.
Still more text..."""

paragraphs = split_text_into_paragraphs(text)

for para in paragraphs:
    await reader.read_text(para)
```

### Custom Text Sources

```python
from io import StringIO

# Read from StringIO
text_stream = StringIO("Text from string buffer")
await reader.read_text(text_stream.read())

# Read from file-like object
class TextFileReader:
    def __init__(self, file_path):
        self.file = open(file_path, 'r')

    def read(self):
        return self.file.read()

reader = get_reader("kokoro")
await reader.read_text(TextFileReader("document.txt").read())
```

---

## Audio Quality

### Voice Selection

```python
# Choose appropriate voice
reader = get_reader("kokoro")

# Friendly voice for general content
await reader.read_text("Hello, welcome to our service!", voice="af_bella")

# Serious voice for news
await reader.read_text("Breaking news from around the world", voice="am_adam")

# Energetic voice for podcasts
await reader.read_text("Here's what's happening today", voice="af_sarah")
```

### Speed Adjustment

```python
reader = get_reader("kokoro")

# Slow reading for accessibility
await reader.read_text("Welcome", voice="af_bella", speed=0.8)

# Normal speed
await reader.read_text("Welcome", voice="af_bella", speed=1.0)

# Fast reading for time efficiency
await reader.read_text("Welcome", voice="af_bella", speed=1.5)
```

### Quality Settings

```python
# Use GPU for better quality
from champi_tts.providers.kokoro import KokoroConfig

config = KokoroConfig(use_gpu=True)
reader = get_reader("kokoro", config=config)

# High quality synthesis
await reader.read_text("High quality audio", voice="af_bella")
```

---

## Advanced Usage

### Progress Tracking

```python
reader = get_reader("kokoro")

def on_state_changed(sender, **kw):
    old_state = kw.get('old_state')
    new_state = kw.get('new_state')
    print(f"State: {old_state} -> {new_state}")

reader.on_state_changed.connect(on_state_changed)

# Track progress
await reader.read_text("Progress tracking example")
```

### Error Handling

```python
reader = get_reader("kokoro")

def on_error(sender, **kw):
    error = kw.get('error')
    print(f"Error occurred: {error}")
    # Save error log
    log_error(error)

reader.on_error.connect(on_error)

try:
    await reader.read_file("document.txt")
except FileNotFoundError:
    print("File not found!")
```

### Real-time Audio Modification

```python
import numpy as np

reader = get_reader("kokoro")

# Modify audio after synthesis
def modify_audio(audio, sample_rate):
    # Add effects
    audio = audio * 1.1  # Increase volume
    return audio

# Custom reading with audio modification
async def read_with_modification(text):
    audio = await provider.synthesize(text)
    modified = modify_audio(audio, provider.config.sample_rate)
    await save_audio(modified, "modified.wav")
```

### Long Document Processing

```python
reader = get_reader("kokoro")

# Process large documents in chunks
chunk_size = 1000  # characters

async def process_large_document(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        await reader.read_text(chunk)

# Usage
await process_large_document("large_document.txt")
```

### Multiple Reader Instances

```python
# Different readers with different configurations
reader1 = get_reader("kokoro", show_ui=False, default_voice="af_bella")
reader2 = get_reader("kokoro", show_ui=True, default_voice="am_adam")

# Run in parallel
async def read_in_parallel(file_path):
    tasks = [
        reader1.read_file(file_path),
        reader2.read_file(file_path)
    ]
    await asyncio.gather(*tasks)
```

### Audio Export During Reading

```python
reader = get_reader("kokoro")

# Track reading progress and export audio
reading_texts = []

def on_reading_started(sender, **kw):
    text = kw.get('text', '')
    reading_texts.append(text)

reader.on_reading_started.connect(on_reading_started)

# Read and save audio
async def save_reading_audio(text):
    audio = await provider.synthesize(text)
    await save_audio(audio, f"{text[:20]}.wav")

await reader.read_text("Test text")
for text in reading_texts:
    await save_reading_audio(text)
```

---

## Use Cases

### Accessibility Tool

```python
reader = get_reader("kokoro", show_ui=True)

# Read documents for visually impaired users
def read_document_for_user(file_path):
    return reader.read_file(file_path)
```

### Content Creation

```python
reader = get_reader("kokoro")

# Convert articles to audio
async def article_to_audio(article_text, output_path):
    audio = await provider.synthesize(article_text)
    await save_audio(audio, output_path)
```

### Learning Aid

```python
reader = get_reader("kokoro")

# Read study materials
async def read_study_materials(file_path):
    reader = get_reader("kokoro")
    await reader.read_file(file_path)
```

### News Reader

```python
reader = get_reader("kokoro")

# Read news updates
async def read_news():
    news_items = ["News item 1", "News item 2", "News item 3"]
    for item in news_items:
        await reader.read_text(item)
```

---

## Troubleshooting

### Text Not Being Read

**Solution**: Check text is valid and provider is initialized:

```python
if reader.state == ReaderState.IDLE:
    await reader.read_text("Test")
```

### Slow Performance

**Solution**: Enable GPU and optimize text:

```python
config = KokoroConfig(use_gpu=True)
reader = get_reader("kokoro", config=config)
```

### Memory Issues with Large Files

**Solution**: Process in chunks:

```python
# Read in smaller chunks
chunk_size = 10000
with open("large.txt", 'r') as f:
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break
        await reader.read_text(chunk)
```

---

**Need more help?** Check out the [CLI Guide](cli-guide.md) or [API Reference](api.md).