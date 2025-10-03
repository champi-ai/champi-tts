"""Event handling example.

This example demonstrates using event signals to monitor TTS state.
"""

import asyncio

from champi_tts import get_reader
from champi_tts.providers.kokoro import KokoroConfig


async def main():
    """Event handling example."""
    # Create provider configuration
    config = KokoroConfig(
        model_path="kokoro-v0_19.pth",
        voices_path="voices",
        voice="af_sky",
        speed=1.0,
    )

    # Get text reader
    reader = get_reader("kokoro", config, show_ui=False)

    # Track events
    events = []

    # Connect to all events
    @reader.on_reading_started.connect
    def on_start(sender, **kwargs):
        events.append(("started", kwargs.get("text", "")))
        print(f"✓ Reading started: {kwargs.get('text', '')[:50]}...")

    @reader.on_chunk_synthesized.connect
    def on_chunk(sender, **kwargs):
        events.append(("chunk", len(kwargs.get("audio", []))))
        print(f"  → Chunk synthesized: {len(kwargs.get('audio', []))} samples")

    @reader.on_chunk_played.connect
    def on_played(sender, **kwargs):
        events.append(("played", None))
        print("  ♪ Chunk played")

    @reader.on_paused.connect
    def on_pause(sender, **kwargs):
        events.append(("paused", None))
        print("⏸ Reading paused")

    @reader.on_resumed.connect
    def on_resume(sender, **kwargs):
        events.append(("resumed", None))
        print("▶ Reading resumed")

    @reader.on_stopped.connect
    def on_stop(sender, **kwargs):
        events.append(("stopped", None))
        print("⏹ Reading stopped")

    @reader.on_reading_completed.connect
    def on_complete(sender, **kwargs):
        events.append(("completed", None))
        print("✓ Reading completed!")

    @reader.on_error.connect
    def on_error(sender, **kwargs):
        events.append(("error", kwargs.get("error")))
        print(f"✗ Error: {kwargs.get('error')}")

    # Read text
    text = "This example demonstrates event handling in champi-tts. Watch the events as they occur!"
    print(f"\nReading: {text}\n")

    await reader.read_text(text)

    # Wait for completion
    await asyncio.sleep(3)

    # Print event summary
    print(f"\n{'='*50}")
    print("Event Summary:")
    print(f"{'='*50}")
    print(f"Total events: {len(events)}")
    for event_type, data in events:
        if data:
            print(f"  - {event_type}: {data}")
        else:
            print(f"  - {event_type}")

    # Cleanup
    await reader.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
