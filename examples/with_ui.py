"""Visual UI indicator example.

This example demonstrates using the visual UI indicator to show TTS state.
"""

import asyncio

from champi_tts import get_reader
from champi_tts.providers.kokoro import KokoroConfig


async def main():
    """UI indicator example."""
    # Create provider configuration
    config = KokoroConfig(
        model_path="kokoro-v0_19.pth",
        voices_path="voices",
        voice="af_sky",
        speed=1.0,
    )

    # Get text reader with UI enabled
    print("Starting reader with UI indicator...")
    print("Watch the UI window to see state changes!\n")

    reader = get_reader("kokoro", config, show_ui=True)

    # Read some text
    texts = [
        "This is the first sentence. Watch the UI indicator show the speaking state.",
        "Now I will pause briefly to demonstrate the paused state.",
        "And finally, this is the last sentence before completion.",
    ]

    for i, text in enumerate(texts, 1):
        print(f"[{i}/{len(texts)}] Reading: {text}")
        await reader.read_text(text)

        # Demonstrate pause on second text
        if i == 2:
            await asyncio.sleep(1)
            print("  → Pausing...")
            reader.pause()
            await asyncio.sleep(2)
            print("  → Resuming...")
            reader.resume()

        await asyncio.sleep(1)

    print("\nAll text read! UI will close in 3 seconds...")
    await asyncio.sleep(3)

    # Cleanup (this will close the UI)
    await reader.cleanup()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
