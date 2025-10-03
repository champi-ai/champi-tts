"""Async context manager example.

This example demonstrates using async context managers for automatic resource management.
"""

import asyncio

from champi_tts import get_provider, get_reader
from champi_tts.providers.kokoro import KokoroConfig


async def example_provider_context():
    """Example using provider with async context manager."""
    config = KokoroConfig(
        model_path="kokoro-v0_19.pth",
        voices_path="voices",
        voice="af_sky",
        speed=1.0,
    )

    # Provider automatically initializes on entry and cleans up on exit
    async with get_provider("kokoro", config) as provider:
        print("Provider initialized automatically!")

        # Synthesize text
        audio = await provider.synthesize("Hello from async context manager!")

        # Play audio
        import sounddevice as sd

        sd.play(audio, samplerate=config.sample_rate)
        sd.wait()

    print("Provider cleaned up automatically!")


async def example_reader_context():
    """Example using reader with async context manager."""
    config = KokoroConfig(
        model_path="kokoro-v0_19.pth",
        voices_path="voices",
        voice="af_sky",
        speed=1.0,
    )

    # Reader automatically initializes on entry and cleans up on exit
    async with get_reader("kokoro", config, show_ui=False) as reader:
        print("Reader initialized automatically!")

        # Read text
        await reader.read_text(
            "This example demonstrates automatic resource management with async context managers."
        )

        await asyncio.sleep(3)

    print("Reader cleaned up automatically!")


async def main():
    """Run examples."""
    print("=" * 60)
    print("Example 1: Provider with async context manager")
    print("=" * 60)
    await example_provider_context()

    print("\n" + "=" * 60)
    print("Example 2: Reader with async context manager")
    print("=" * 60)
    await example_reader_context()


if __name__ == "__main__":
    asyncio.run(main())
