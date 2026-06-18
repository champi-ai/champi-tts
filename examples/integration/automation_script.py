#!/usr/bin/env python3
"""
Automation Script Example

This example demonstrates using champi-tts in automation scripts.
Shows reading news headlines and processing them automatically.
"""

import asyncio
import os
import subprocess
import sys

from champion_signals import SignalBus

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champi_tts import get_provider
from champi_tts.core.audio import save_audio


class NewsReader:
    """Automation script for reading news headlines"""

    def __init__(self):
        """Initialize the news reader"""
        self.provider = None
        self.signal_bus = SignalBus()

    async def initialize(self):
        """Initialize the reader"""
        print("Initializing news reader...")
        self.provider = get_provider("kokoro")
        await self.provider.initialize()
        print("News reader initialized")

    async def read_news_headlines(self, headlines, voice="af_bella"):
        """
        Read news headlines with text-to-speech

        Args:
            headlines: List of news headlines
            voice: Voice to use
        """
        print(f"\nReading {len(headlines)} news headlines...")
        print("=" * 50)

        for i, headline in enumerate(headlines, 1):
            print(f"\n{i}. {headline}")

            # Synthesize headline
            audio = await self.provider.synthesize(headline, voice=voice)

            # Save to file
            output_file = f"news_headline_{i:03d}.wav"
            await save_audio(
                audio, output_file, sample_rate=self.provider.config.sample_rate
            )

            print(f"   Saved: {output_file}")

            # Play audio (optional)
            try:
                # Different commands for different platforms
                if sys.platform == "linux":
                    subprocess.run(["aplay", output_file], check=True)
                elif sys.platform == "darwin":
                    subprocess.run(["afplay", output_file], check=True)
                elif sys.platform == "win32":
                    subprocess.run(["start", output_file], shell=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"   Note: Could not play audio: {e}")

            # Wait before next headline
            if i < len(headlines):
                await asyncio.sleep(2)

    async def process_from_file(self, filename, voice="af_bella"):
        """
        Process news from a file

        Args:
            filename: Path to news file
            voice: Voice to use
        """
        print(f"\nReading from file: {filename}")

        if not os.path.exists(filename):
            print(f"Error: File not found: {filename}")
            return

        with open(filename) as f:
            lines = [line.strip() for line in f if line.strip()]

        await self.read_news_headlines(lines, voice)

    async def cleanup(self):
        """Cleanup resources"""
        if self.provider:
            await self.provider.shutdown()
        print("\nNews reader cleanup complete")


async def main():
    """Main automation script"""

    print("=" * 50)
    print("Automation Script Example")
    print("Reading news headlines with TTS")
    print("=" * 50)

    reader = NewsReader()

    try:
        # Initialize
        await reader.initialize()

        # Example 1: Hardcoded headlines
        print("\nExample 1: Reading hardcoded headlines")
        headlines = [
            "Breaking: New technology announced.",
            "Market update: Stocks rise today.",
            "Weather forecast: Sunny weekend ahead.",
            "Sports: Championship finals this Sunday.",
            "Technology: AI innovation continues.",
        ]

        await reader.read_news_headlines(headlines, voice="af_bella")

        # Example 2: From file
        print("\n\nExample 2: Reading from file")
        test_file = "test_news.txt"
        with open(test_file, "w") as f:
            f.write("""Morning Briefing
Stock Market
- S&P 500 up 1.2%
- NASDAQ up 0.8%

Weather Report
- Sunny, 72 degrees
- Clear skies expected

Sports Update
- Championship finals set for Sunday

Technology News
- AI advancements continue
- New device releases expected""")

        await reader.process_from_file(test_file, voice="af_bella")

        # Cleanup
        os.remove(test_file)
        print(f"\nCleaned up test file: {test_file}")

        print("\n" + "=" * 50)
        print("Automation script completed successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await reader.cleanup()


async def main_scheduled():
    """Scheduled news reader (runs on schedule)"""

    print("=" * 50)
    print("Scheduled News Reader")
    print("=" * 50)

    reader = NewsReader()

    try:
        await reader.initialize()

        print("\nStarting scheduled news reader...")
        print("Press Ctrl+C to stop")

        while True:
            headlines = [
                "This is a scheduled update.",
                "Here's your daily news summary.",
                "Remember to stay informed!",
            ]

            await reader.read_news_headlines(headlines, voice="af_bella")

            # Wait 24 hours before next run
            print("\nWaiting 24 hours until next news reading...")
            await asyncio.sleep(86400)

    except KeyboardInterrupt:
        print("\n\nScheduled reader stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        await reader.cleanup()


if __name__ == "__main__":
    try:
        print("\nExample 1: One-time news reading")
        asyncio.run(main())

        print("\n\n" + "=" * 50)
        print("Example 2: Scheduled news reader (24h cycle)")
        print("Note: This runs indefinitely until Ctrl+C is pressed")
        print("=" * 50)
        asyncio.run(main_scheduled())
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
