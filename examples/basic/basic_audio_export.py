#!/usr/bin/env python3
"""
Basic Audio Export Example

This example demonstrates how to save synthesized audio in different formats.
Shows WAV and MP3 export options.
"""

import asyncio
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champi_tts import get_provider
from champi_tts.core.audio import save_audio


async def main():
    """Audio export example"""

    print("=" * 50)
    print("Basic Audio Export Example")
    print("=" * 50)

    # Get provider
    print("\n1. Getting provider...")
    provider = get_provider("kokoro")
    await provider.initialize()

    # Synthesize text
    text = "Audio export example. This demonstrates saving in different formats."
    print(f"2. Synthesizing: '{text}'")

    audio = await provider.synthesize(text)
    print(f"3. Audio generated: shape={audio.shape}, dtype={audio.dtype}")

    # Create output directory
    output_dir = "audio_outputs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"4. Creating output directory: {output_dir}")

    # Save as WAV
    wav_file = os.path.join(output_dir, "output.wav")
    print(f"\n5. Saving as WAV: {wav_file}")
    await save_audio(audio, wav_file, sample_rate=provider.config.sample_rate)
    print("   WAV saved successfully!")

    # Save as MP3
    mp3_file = os.path.join(output_dir, "output.mp3")
    print(f"\n6. Saving as MP3: {mp3_file}")
    try:
        await save_audio(audio, mp3_file, sample_rate=provider.config.sample_rate)
        print("   MP3 saved successfully!")
    except Exception as e:
        print(f"   MP3 save failed: {e}")
        print("   Note: MP3 requires ffmpeg to be installed")

    # Check file sizes
    print("\n7. File sizes:")
    if os.path.exists(wav_file):
        wav_size = os.path.getsize(wav_file)
        print(f"   WAV: {wav_size} bytes")
    if os.path.exists(mp3_file):
        mp3_size = os.path.getsize(mp3_file)
        print(f"   MP3: {mp3_size} bytes")

    # Cleanup
    print("\n8. Shutting down provider...")
    await provider.shutdown()

    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print(f"Output directory: {output_dir}/")
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
