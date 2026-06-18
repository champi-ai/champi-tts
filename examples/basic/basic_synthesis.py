#!/usr/bin/env python3
"""
Basic Synthesis Example

This example demonstrates how to synthesize text to speech using the Kokoro provider.
Shows basic usage: initialize provider, synthesize text, and save audio.
"""

import asyncio
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champi_tts import get_provider
from champi_tts.core.audio import save_audio


async def main():
    """Basic text-to-speech synthesis example"""

    print("=" * 50)
    print("Basic Synthesis Example")
    print("=" * 50)

    # Get Kokoro provider (default)
    print("\n1. Getting provider...")
    provider = get_provider("kokoro")

    # Initialize the provider
    print("2. Initializing provider...")
    await provider.initialize()

    # Display available voices
    print(f"3. Available voices: {len(provider.voices)} voices")
    for i, voice in enumerate(provider.voices[:3], 1):
        print(f"   {i}. {voice}")

    # Synthesize text
    text = "Hello, world! This is a basic synthesis example."
    print(f"\n4. Synthesizing: '{text}'")

    audio = await provider.synthesize(text)

    print(f"5. Audio generated: shape={audio.shape}, dtype={audio.dtype}")

    # Save audio to file
    output_file = "basic_output.wav"
    print(f"\n6. Saving to {output_file}...")
    await save_audio(audio, output_file, sample_rate=provider.config.sample_rate)
    print("   Audio saved successfully!")

    # Cleanup
    print("\n7. Shutting down provider...")
    await provider.shutdown()

    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print(f"Output file: {output_file}")
    print("=" * 50)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
