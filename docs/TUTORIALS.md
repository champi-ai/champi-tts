# Champi TTS Tutorial Series

A step-by-step guide to learning champi-tts from beginner to advanced usage.

## Table of Contents

- [Tutorial 1: Getting Started](#tutorial-1-getting-started)
- [Tutorial 2: Working with Kokoro](#tutorial-2-working-with-kokoro)
- [Tutorial 3: Advanced Text Reading](#tutorial-3-advanced-text-reading)
- [Tutorial 4: Multiple Providers](#tutorial-4-multiple-providers)
- [Tutorial 5: Integration Examples](#tutorial-5-integration-examples)

## Tutorial 1: Getting Started

**Duration**: 15-20 minutes
**Prerequisites**: Python 3.12+, champi-tts installed

### Objective
Learn the basics of installing and using champi-tts for text-to-speech synthesis.

### Step 1: Installation

```bash
# Install using pip
pip install champi-tts

# Or using uv (recommended)
uv pip install champi-tts
```

### Step 2: Verify Installation

```python
from champi_tts import get_provider
print("Installation successful!")
```

### Step 3: Your First Synthesis

```python
import asyncio
from champi_tts import get_provider

async def main():
    # Get default provider
    provider = get_provider()
    await provider.initialize()

    # Synthesize some text
    audio = await provider.synthesize("Hello, world!")
    print(f"Synthesized audio: {audio.shape}")

    # Save to file
    from champi_tts.core.audio import save_audio
    await save_audio(audio, "hello.wav", sample_rate=provider.config.sample_rate)

    await provider.shutdown()

asyncio.run(main())
```

**Expected Output**: Creates `hello.wav` file with synthesized speech

### Step 4: Using the CLI

```bash
# Test synthesis with CLI
champi-tts synthesize "Hello, world!" --output hello.wav
```

### Step 5: Reading a Text File

```python
import asyncio
from champi_tts import get_reader

async def main():
    # Create reader
    reader = get_reader("kokoro", show_ui=True)
    await reader.provider.initialize()

    # Read a text file
    await reader.read_file("document.txt", voice="af_bella")

    await reader.provider.shutdown()

asyncio.run(main())
```

### Step 6: Check Available Voices

```bash
# List available voices
champi-tts list-voices
```

**Summary**: You've successfully installed champi-tts, created your first synthesis, and read a text file. You're now ready to explore more advanced features!

---

## Tutorial 2: Working with Kokoro

**Duration**: 20-25 minutes
**Prerequisites**: Completed Tutorial 1

### Objective
Learn how to configure and use the Kokoro TTS provider effectively.

### Step 1: Understanding Kokoro Provider

```python
import asyncio
from champi_tts import get_provider

async def main():
    # Get Kokoro provider
    provider = get_provider("kokoro")
    await provider.initialize()

    print(f"Provider: {provider.__class__.__name__}")
    print(f"Sample Rate: {provider.config.sample_rate}")
    print(f"Available Voices: {len(provider.voices)}")

    await provider.shutdown()

asyncio.run(main())
```

### Step 2: Exploring Voices

```python
import asyncio
from champi_tts import get_provider

async def main():
    provider = get_provider("kokoro")
    await provider.initialize()

    # List all voices
    print("Available voices:")
    for i, voice in enumerate(provider.voices, 1):
        print(f"{i}. {voice}")

    await provider.shutdown()

asyncio.run(main())
```

**Expected Output**: List of available voices with names and IDs

### Step 3: Custom Voice Selection

```python
import asyncio
from champi_tts import get_provider

async def main():
    provider = get_provider("kokoro")

    # Configure specific voice
    config = {
        "voice": "af_bella",
        "speed": 1.0,
        "sample_rate": 24000
    }

    provider.config.update(config)
    await provider.initialize()

    # Synthesize with custom voice
    audio = await provider.synthesize("This is a test using a custom voice.")
    print(f"Voice: {provider.config.voice}")

    await provider.shutdown()

asyncio.run(main())
```

### Step 4: Adjusting Speech Speed

```python
import asyncio
from champi_tts import get_provider

async def main():
    provider = get_provider("kokoro")

    # Test different speeds
    speeds = [0.8, 1.0, 1.2, 1.5]

    for speed in speeds:
        provider.config.speed = speed
        await provider.initialize()

        print(f"\n--- Speed: {speed} ---")
        audio = await provider.synthesize(f"Speaking at {speed}x speed.")

        await save_audio(audio, f"speed_{speed}.wav", sample_rate=provider.config.sample_rate)
        print(f"Saved: speed_{speed}.wav")

        await provider.shutdown()

asyncio.run(main())
```

### Step 5: Saving Different Audio Formats

```python
import asyncio
from champi_tts import get_provider
from champi_tts.core.audio import save_audio

async def main():
    provider = get_provider("kokoro")
    await provider.initialize()

    # Synthesize
    audio = await provider.synthesize("Hello in different formats!")

    # Save as WAV
    await save_audio(audio, "output.wav", sample_rate=provider.config.sample_rate)
    print("Saved: output.wav")

    # Save as MP3 (requires ffmpeg)
    try:
        await save_audio(audio, "output.mp3", sample_rate=provider.config.sample_rate)
        print("Saved: output.mp3")
    except Exception as e:
        print(f"MP3 save failed: {e}")

    await provider.shutdown()

asyncio.run(main())
```

### Step 6: Using the CLI with Kokoro

```bash
# Synthesize with specific voice
champi-tts synthesize "Hello world!" --voice af_bella

# List voices
champi-tts list-voices

# Read with specific voice
champi-tts read document.txt --voice af_bella
```

**Summary**: You've learned how to configure Kokoro, select voices, adjust speech speed, and save audio in different formats. You're ready to handle more complex text reading tasks!

---

## Tutorial 3: Advanced Text Reading

**Duration**: 30-35 minutes
**Prerequisites**: Completed Tutorial 2

### Objective
Master advanced text reading features including queue management, pause/resume controls, and handling long documents.

### Step 1: Basic Text Reading

```python
import asyncio
from champi_tts import get_reader

async def main():
    reader = get_reader("kokoro", show_ui=True)
    await reader.provider.initialize()

    # Read text directly
    text = "This is a test of the text reading service."
    await reader.read(text, voice="af_bella")

    await reader.provider.shutdown()

asyncio.run(main())
```

### Step 2: Reading a Text File

```python
import asyncio
from champi_tts import get_reader

async def main():
    reader = get_reader("kokoro", show_ui=True)
    await reader.provider.initialize()

    # Read a text file
    await reader.read_file("document.txt", voice="af_bella")

    await reader.provider.shutdown()

asyncio.run(main())
```

### Step 3: Managing Reading Queue

```python
import asyncio
from champi_tts import get_reader

async def main():
    reader = get_reader("kokoro", show_ui=True)
    await reader.provider.initialize()

    # Add multiple texts to queue
    texts = [
        "First paragraph of text.",
        "Second paragraph with more details.",
        "Third paragraph concluding the text."
    ]

    for text in texts:
        await reader.read(text, voice="af_bella")

    await reader.provider.shutdown()

asyncio.run(main())
```

### Step 4: Pause and Resume

```python
import asyncio
from champi_tts import get_reader

async def main():
    reader = get_reader("kokoro", show_ui=True)
    await reader.provider.initialize()

    text = "This text will be paused midway."

    # Start reading
    task = asyncio.create_task(reader.read(text, voice="af_bella"))

    # Wait a bit
    await asyncio.sleep(3)

    # Pause
    print("Pausing...")
    await reader.pause()

    # Wait more
    await asyncio.sleep(2)

    # Resume
    print("Resuming...")
    await reader.resume()

    # Wait for completion
    await task

    await reader.provider.shutdown()

asyncio.run(main())
```

### Step 5: Stopping Reading

```python
import asyncio
from champi_tts import get_reader

async def main():
    reader = get_reader("kokoro", show_ui=True)
    await reader.provider.initialize()

    text = "This text will be stopped midway."

    # Start reading
    task = asyncio.create_task(reader.read(text, voice="af_bella"))

    # Wait a bit
    await asyncio.sleep(3)

    # Stop
    print("Stopping...")
    await reader.stop()

    # Wait a bit more
    await asyncio.sleep(1)

    await reader.provider.shutdown()

asyncio.run(main())
```

### Step 6: Handling Long Documents

```python
import asyncio
from champi_tts import get_reader

async def main():
    reader = get_reader("kokoro", show_ui=True)
    await reader.provider.initialize()

    # Read a long document (file handles queue automatically)
    print("Reading long document...")
    await reader.read_file("long_document.txt", voice="af_bella")

    await reader.provider.shutdown()

asyncio.run(main())
```

### Step 7: Using with Custom Text Sources

```python
import asyncio
from champi_tts import get_reader

async def main():
    reader = get_reader("kokoro", show_ui=True)
    await reader.provider.initialize()

    # Read from a string
    text = """
    Introduction
    ------------
    This is the introduction paragraph.
    It provides an overview of the content.

    Main Content
    ------------
    Here we present the main information.
    The reader will process this sequentially.

    Conclusion
    ----------
    This concludes the text reading demonstration.
    """

    await reader.read(text, voice="af_bella")

    await reader.provider.shutdown()

asyncio.run(main())
```

### Step 8: Error Handling

```python
import asyncio
from champi_tts import get_reader

async def main():
    reader = get_reader("kokoro", show_ui=True)
    await reader.provider.initialize()

    try:
        await reader.read_file("nonexistent.txt", voice="af_bella")
    except FileNotFoundError:
        print("File not found - handled gracefully")

    await reader.provider.shutdown()

asyncio.run(main())
```

**Summary**: You've mastered advanced text reading features including queue management, pause/resume/stop controls, error handling, and working with various text sources!

---

## Tutorial 4: Multiple Providers

**Duration**: 20-25 minutes
**Prerequisites**: Completed Tutorial 3

### Objective
Learn how to use multiple TTS providers and switch between them.

### Step 1: Understanding Providers

```python
import asyncio
from champi_tts import get_provider

async def main():
    # Kokoro provider
    kokoro = get_provider("kokoro")
    await kokoro.initialize()
    print(f"Kokoro voices: {len(kokoro.voices)}")
    await kokoro.shutdown()

    # Note: OpenAI and ElevenLabs would be added here when available
    # openai = get_provider("openai")
    # await openai.initialize()
    # await openai.shutdown()

asyncio.run(main())
```

### Step 2: Creating Multiple Providers

```python
import asyncio
from champi_tts import get_provider

async def main():
    # Create multiple providers
    kokoro = get_provider("kokoro")
    openai = get_provider("openai")  # Would be available when implemented
    elevenlabs = get_provider("elevenlabs")  # Would be available when implemented

    # Initialize each
    await kokoro.initialize()
    await openai.initialize()
    await elevenlabs.initialize()

    print("All providers initialized")

    await kokoro.shutdown()
    await openai.shutdown()
    await elevenlabs.shutdown()

asyncio.run(main())
```

### Step 3: Switching Between Providers

```python
import asyncio
from champi_tts import get_provider

async def synthesize_with_provider(provider_name, text):
    """Synthesize text using a specific provider"""
    provider = get_provider(provider_name)
    await provider.initialize()

    audio = await provider.synthesize(text)
    print(f"Used provider: {provider_name}")
    print(f"Voice: {provider.config.voice}")

    await provider.shutdown()
    return audio

async def main():
    text = "Hello, world!"

    # Synthesize with different providers
    await synthesize_with_provider("kokoro", text)

    # Note: Would work with other providers when implemented
    # await synthesize_with_provider("openai", text)
    # await synthesize_with_provider("elevenlabs", text)

asyncio.run(main())
```

### Step 4: Comparing Providers

```python
import asyncio
from champi_tts import get_provider
from champi_tts.core.audio import save_audio
import time

async def test_provider(provider_name, text):
    """Test provider performance"""
    provider = get_provider(provider_name)
    start = time.time()

    await provider.initialize()
    audio = await provider.synthesize(text)
    duration = time.time() - start

    # Save audio
    await save_audio(audio, f"{provider_name}_test.wav", sample_rate=provider.config.sample_rate)

    await provider.shutdown()

    return duration

async def main():
    text = "Testing provider performance. This is a test."

    print("Testing Kokoro provider...")
    kokoro_time = await test_provider("kokoro", text)
    print(f"Kokoro: {kokoro_time:.2f}s")

    # Note: Would test other providers when implemented
    # print("\nTesting OpenAI provider...")
    # openai_time = await test_provider("openai", text)
    # print(f"OpenAI: {openai_time:.2f}s")

    # Compare
    # if kokoro_time < openai_time:
    #     print("Kokoro is faster")
    # else:
    #     print("OpenAI is faster")

asyncio.run(main())
```

### Step 5: Provider-Specific Configurations

```python
import asyncio
from champi_tts import get_provider

async def main():
    # Kokoro specific config
    kokoro = get_provider("kokoro")
    kokoro.config.update({
        "voice": "af_bella",
        "speed": 1.0,
        "sample_rate": 24000,
        "use_gpu": True
    })
    await kokoro.initialize()

    # Note: Other providers would have different configs
    # openai.config.update({"model": "tts-1"})
    # elevenlabs.config.update({"voice_id": "some_voice_id"})

    await kokoro.shutdown()

asyncio.run(main())
```

### Step 6: Choosing Appropriate Provider

```python
import asyncio
from champi_tts import get_provider

async def select_provider(use_case, preference="speed"):
    """Select provider based on use case"""

    # Local, no internet required
    if use_case == "offline" and preference == "speed":
        return get_provider("kokoro")

    # High quality, internet required
    elif use_case == "quality":
        # OpenAI or ElevenLabs would be preferred
        return get_provider("kokoro")  # Fallback

    # Default
    return get_provider("kokoro")

async def main():
    provider = await select_provider("offline", "quality")
    await provider.initialize()
    print(f"Selected provider: {provider.__class__.__name__}")

    await provider.shutdown()

asyncio.run(main())
```

### Step 7: CLI with Different Providers

```bash
# Kokoro (current)
champi-tts synthesize "Hello world!" --voice af_bella

# OpenAI (when available)
# champi-tts synthesize "Hello world!" --provider openai

# ElevenLabs (when available)
# champi-tts synthesize "Hello world!" --provider elevenlabs
```

**Summary**: You now understand how to work with multiple providers, switch between them, compare their performance, and select the right provider for different use cases!

---

## Tutorial 5: Integration Examples

**Duration**: 30-35 minutes
**Prerequisites**: Completed Tutorial 4

### Objective
Learn how to integrate champi-tts into various applications and use cases.

### Step 1: Flask Web Application Integration

```python
from flask import Flask, request, jsonify
import asyncio
from champi_tts import get_provider
from champi_tts.core.audio import save_audio

app = Flask(__name__)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """Synthesize text to speech"""
    data = request.json
    text = data.get('text')
    voice = data.get('voice', 'af_bella')

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        provider = get_provider("kokoro")
        loop.run_until_complete(provider.initialize())

        audio = loop.run_until_complete(provider.synthesize(text))
        loop.run_until_complete(save_audio(audio, "output.wav", sample_rate=provider.config.sample_rate))

        loop.run_until_complete(provider.shutdown())

        return jsonify({
            "status": "success",
            "output_file": "output.wav"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        loop.close()

if __name__ == '__main__':
    app.run(debug=True)
```

### Step 2: Automation Script

```python
#!/usr/bin/env python3
import asyncio
from champi_tts import get_provider
import subprocess

async def process_news():
    """Read news headlines aloud"""
    headlines = [
        "Breaking: New technology announced.",
        "Market update: Stocks rise today.",
        "Weather forecast: Sunny weekend ahead."
    ]

    provider = get_provider("kokoro")
    await provider.initialize()

    for headline in headlines:
        print(f"Reading: {headline}")
        audio = await provider.synthesize(headline)

        # Save and play
        await save_audio(audio, f"news_{headlines.index(headline)}.wav", sample_rate=provider.config.sample_rate)

        # Play audio
        subprocess.run(["aplay", f"news_{headlines.index(headline)}.wav"])

    await provider.shutdown()

if __name__ == "__main__":
    asyncio.run(process_news())
```

### Step 3: Batch Processing

```python
import asyncio
from champi_tts import get_provider
from champi_tts.core.audio import save_audio
import os

async def batch_process(text_files, output_dir):
    """Process multiple text files"""
    provider = get_provider("kokoro")
    await provider.initialize()

    os.makedirs(output_dir, exist_ok=True)

    for filename in text_files:
        print(f"Processing {filename}...")

        with open(filename, 'r') as f:
            text = f.read()

        audio = await provider.synthesize(text)

        output_file = os.path.join(output_dir, f"{os.path.basename(filename)}.wav")
        await save_audio(audio, output_file, sample_rate=provider.config.sample_rate)

        print(f"Saved: {output_file}")

    await provider.shutdown()

async def main():
    text_files = ["document1.txt", "document2.txt", "document3.txt"]
    await batch_process(text_files, "batch_output")

asyncio.run(main())
```

### Step 4: Real-Time TTS Application

```python
import asyncio
from champi_tts import get_reader
from champion_signals import SignalBus

async def real_time_tts():
    """Real-time text-to-speech with user input"""
    reader = get_reader("kokoro", show_ui=True)
    await reader.provider.initialize()

    signal_bus = SignalBus()

    async def on_text_received(text):
        """Handle incoming text"""
        print(f"Received: {text}")
        await reader.read(text, voice="af_bella")

    signal_bus.subscribe("text_received", on_text_received)

    # Simulate receiving text
    await asyncio.sleep(1)
    await signal_bus.publish("text_received", "Hello, welcome to real-time TTS.")

    await reader.provider.shutdown()

asyncio.run(real_time_tts())
```

### Step 5: Accessibility Tool

```python
import asyncio
from champi_tts import get_reader
import pyautogui

async def accessibility_tool():
    """Text-to-speech for accessibility"""
    reader = get_reader("kokoro", show_ui=True)
    await reader.provider.initialize()

    print("Accessibility Tool Active")
    print("Press SPACE to read screen content")

    while True:
        if pyautogui.keyDown('space'):
            # Get screen content
            text = pyautogui.screenshot().text
            if text:
                await reader.read(text, voice="af_bella")

        await asyncio.sleep(0.1)

    await reader.provider.shutdown()

asyncio.run(accessibility_tool())
```

### Step 6: Learning Application

```python
import asyncio
from champi_tts import get_reader

async def learn_vocabulary(word, pronunciation):
    """Learn vocabulary with text-to-speech"""
    reader = get_reader("kokoro", show_ui=True)
    await reader.provider.initialize()

    # Read word
    print(f"Word: {word}")
    await reader.read(word, voice="af_bella")

    # Read pronunciation
    print(f"Pronunciation: {pronunciation}")
    await reader.read(pronunciation, voice="af_bella")

    await reader.provider.shutdown()

async def main():
    vocabulary = [
        {"word": "algorithm", "pronunciation": "al-go-rith-m"},
        {"word": "variable", "pronunciation": "var-i-a-ble"},
        {"word": "function", "pronunciation": "func-tion"}
    ]

    for item in vocabulary:
        await learn_vocabulary(item["word"], item["pronunciation"])
        await asyncio.sleep(2)

asyncio.run(main())
```

### Step 7: Custom Provider Implementation (Advanced)

```python
import asyncio
from champi_tts.core.base_provider import BaseProvider
from champi_tts.core.audio import Audio

class CustomProvider(BaseProvider):
    """Example custom TTS provider"""

    def __init__(self):
        super().__init__()
        self.voices = ["custom_voice_1", "custom_voice_2"]

    async def initialize(self):
        """Initialize the provider"""
        await super().initialize()
        # Setup custom TTS backend

    async def synthesize(self, text: str, voice: str = None) -> Audio:
        """Synthesize text to speech"""
        await super().synthesize(text, voice)

        # Custom synthesis logic
        # This would connect to your TTS backend

        return Audio(audio_data=b"", sample_rate=24000)

    async def shutdown(self):
        """Cleanup resources"""
        # Close connections, free memory
        await super().shutdown()

# Usage
async def main():
    provider = CustomProvider()
    await provider.initialize()
    await provider.shutdown()

asyncio.run(main())
```

**Summary**: You've learned how to integrate champi-tts into various applications including web apps, automation scripts, batch processing, real-time applications, accessibility tools, and learning applications!

---

## Next Steps

Now that you've completed all tutorials:

1. **Explore Examples**: Check out the practical examples in the `examples/` directory
2. **Customize**: Modify examples to suit your needs
3. **Contribute**: Share your use cases and contribute back
4. **Documentation**: Refer to the API reference for detailed documentation

## Getting Help

If you get stuck or have questions:

- Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
- Review [USER_GUIDE.md](USER_GUIDE.md) for detailed information
- Open an issue on GitHub
- Join the community discussions

---

**Happy Learning!** 🚀