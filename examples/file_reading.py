"""File reading with pause/resume example.

This example demonstrates reading text from a file with pause/resume controls.
"""

import asyncio
from pathlib import Path

from champi_tts import get_reader
from champi_tts.providers.kokoro import KokoroConfig


async def main():
    """File reading with controls example."""
    # Create provider configuration
    config = KokoroConfig(
        model_path="kokoro-v0_19.pth",
        voices_path="voices",
        voice="af_sky",
        speed=1.0,
    )

    # Get text reader
    reader = get_reader("kokoro", config, show_ui=False)

    # Create sample text file
    sample_file = Path("sample.txt")
    sample_file.write_text(
        "This is a sample text file. "
        "The reader will read this text aloud. "
        "You can pause, resume, or stop the reading at any time."
    )

    # Setup event handlers
    @reader.on_reading_started.connect
    def on_start(sender, **kwargs):
        print(f"Started reading: {kwargs.get('text', '')[:50]}...")

    @reader.on_paused.connect
    def on_pause(sender, **kwargs):
        print("Reading paused")

    @reader.on_resumed.connect
    def on_resume(sender, **kwargs):
        print("Reading resumed")

    @reader.on_stopped.connect
    def on_stop(sender, **kwargs):
        print("Reading stopped")

    @reader.on_reading_completed.connect
    def on_complete(sender, **kwargs):
        print("Reading completed!")

    # Start reading
    print(f"Reading file: {sample_file}")
    await reader.read_file(sample_file)

    # Wait a bit then pause
    await asyncio.sleep(2)
    print("\n[Pausing...]")
    reader.pause()

    # Wait then resume
    await asyncio.sleep(2)
    print("[Resuming...]")
    reader.resume()

    # Wait for completion
    await asyncio.sleep(5)

    # Cleanup
    await reader.cleanup()
    sample_file.unlink()


if __name__ == "__main__":
    asyncio.run(main())
