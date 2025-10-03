"""Streaming synthesis example.

This example demonstrates streaming text-to-speech for lower latency.
"""

import asyncio

import numpy as np

from champi_tts import get_provider
from champi_tts.providers.kokoro import KokoroConfig


async def main():
    """Streaming synthesis example."""
    # Create provider configuration
    config = KokoroConfig(
        model_path="kokoro-v0_19.pth",
        voices_path="voices",
        voice="af_sky",
        speed=1.0,
    )

    # Get TTS provider
    provider = get_provider("kokoro", config)
    await provider.initialize()

    # Text to synthesize
    text = (
        "This is a streaming synthesis example. "
        "Instead of waiting for the entire text to be synthesized, "
        "we can start playing audio chunks as they become available. "
        "This reduces latency and improves responsiveness."
    )

    print(f"Streaming: {text}\n")

    # Stream synthesis
    import sounddevice as sd

    stream = sd.OutputStream(samplerate=config.sample_rate, channels=1)
    stream.start()

    chunk_count = 0
    async for audio_chunk in provider.synthesize_streaming(text):
        chunk_count += 1
        print(f"Playing chunk {chunk_count} ({len(audio_chunk)} samples)")

        # Ensure audio is float32 in range [-1, 1]
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        if np.max(np.abs(audio_chunk)) > 1.0:
            audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))

        # Play chunk
        stream.write(audio_chunk.reshape(-1, 1))

    stream.stop()
    stream.close()

    print(f"\nStreaming complete! Played {chunk_count} chunks")

    # Cleanup
    await provider.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
