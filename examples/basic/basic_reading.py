#!/usr/bin/env python3
"""
Basic Text Reading Example

This example demonstrates how to read text files using the text reading service.
Shows basic usage: create reader, read text file, and control playback.
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champi_tts import get_reader


async def main():
    """Basic text reading example"""

    print("=" * 50)
    print("Basic Text Reading Example")
    print("=" * 50)

    # Create reader with UI
    print("\n1. Creating reader with UI...")
    reader = get_reader("kokoro", show_ui=True)

    # Initialize provider
    print("2. Initializing provider...")
    await reader.provider.initialize()

    # Check if file exists
    test_file = "test_document.txt"
    if not os.path.exists(test_file):
        print(f"3. Creating test file: {test_file}")

        # Create a simple test document
        with open(test_file, 'w') as f:
            f.write("""Basic Text Reading Example

This is the first paragraph of the test document.
It demonstrates basic text reading functionality.

The text reading service can handle multiple paragraphs
and maintain proper queue management.

This is the final paragraph.
""")
        print("   Test file created!")
    else:
        print(f"3. Found existing file: {test_file}")

    # Read the text file
    print(f"\n4. Reading {test_file}...")
    await reader.read_file(test_file, voice="af_bella")

    print("\n   Reading completed!")

    # Cleanup
    print("\n5. Shutting down...")
    await reader.provider.shutdown()

    print("\n" + "=" * 50)
    print("Example completed successfully!")
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