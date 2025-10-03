"""Basic text-to-speech synthesis example.

This example demonstrates simple text-to-speech synthesis using champi-tts.
"""

import asyncio

from champi_tts import get_provider
from champi_tts.providers.kokoro import KokoroConfig


async def main():
    """Basic synthesis example."""
    # Create provider configuration
    config = KokoroConfig(
        model_path="kokoro-v0_19.pth",
        voices_path="voices",
        voice="af_sky",
        speed=1.0,
    )

    # Get TTS provider
    provider = get_provider("kokoro", config)

    # Initialize provider
    await provider.initialize()

    # Synthesize text
    text = "Hello! This is a basic text-to-speech synthesis example."
    print(f"Synthesizing: {text}")

    audio = await provider.synthesize(text)

    # Play audio
    import sounddevice as sd

    sd.play(audio, samplerate=config.sample_rate)
    sd.wait()

    print("Synthesis complete!")

    # Cleanup
    await provider.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
