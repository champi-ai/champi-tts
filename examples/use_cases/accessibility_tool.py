#!/usr/bin/env python3
"""
Accessibility Tool Example

This example demonstrates using champi-tts as an accessibility tool.
Shows how to read text from applications, screen readers, etc.
"""

import asyncio
import os
import sys

import pyautogui
from champion_signals import SignalBus

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champi_tts import get_reader


class AccessibilityTool:
    """Accessibility tool for reading text content"""

    def __init__(self):
        """Initialize the accessibility tool"""
        self.reader = None
        self.signal_bus = SignalBus()
        self.running = False

    async def initialize(self):
        """Initialize the tool"""
        print("Initializing accessibility tool...")
        self.reader = get_reader("kokoro", show_ui=True)
        await self.reader.provider.initialize()
        print("Accessibility tool initialized")

    async def read_screen_content(self, voice="af_bella"):
        """
        Read screen content (mock implementation)

        Note: This is a simplified example. In a real implementation,
        you would use screen reading APIs or OCR to get actual screen content.
        """
        print("Reading screen content...")

        # Mock screen content (in real app, use actual screen reading)
        mock_text = "This is a simulated screen content. In a real implementation, this would use screen reading APIs."

        await self.reader.read(mock_text, voice=voice)

    async def read_selection(self, text: str, voice="af_bella"):
        """
        Read selected text

        Args:
            text: Selected text to read
            voice: Voice to use
        """
        print(f"Reading selection: {text}")
        await self.reader.read(text, voice=voice)

    async def read_application(self, app_name: str, window_name: str, voice="af_bella"):
        """
        Read from a specific application window

        Args:
            app_name: Application name
            window_name: Window name
            voice: Voice to use
        """
        print(f"Reading from {app_name} - {window_name}")

        # In a real implementation, you would:
        # - Get text from the window
        # - Process and read it

        mock_text = f"Application: {app_name}, Window: {window_name}. Content would be read here."

        await self.reader.read(mock_text, voice=voice)

    async def read_document(self, filename: str, voice="af_bella"):
        """
        Read a document file

        Args:
            filename: Path to document
            voice: Voice to use
        """
        print(f"Reading document: {filename}")

        if not os.path.exists(filename):
            print(f"Error: File not found: {filename}")
            return

        with open(filename) as f:
            text = f.read()

        await self.reader.read(text, voice=voice)

    async def read_ocr_text(self, text: str, voice="af_bella"):
        """
        Read OCR extracted text

        Args:
            text: OCR text to read
            voice: Voice to use
        """
        print(f"Reading OCR text ({len(text)} characters)")
        await self.reader.read(text, voice=voice)

    async def read_notification(self, notification: str, voice="af_bella"):
        """
        Read system notification

        Args:
            notification: Notification text
            voice: Voice to use
        """
        print(f"Reading notification: {notification}")
        await self.reader.read(notification, voice=voice)

    async def read_message(self, sender: str, message: str, voice="af_bella"):
        """
        Read message

        Args:
            sender: Message sender
            message: Message content
            voice: Voice to use
        """
        full_text = f"Message from {sender}: {message}"
        print(f"Reading message: {full_text}")
        await self.reader.read(full_text, voice=voice)

    async def run_ongoing_reading(self, voice="af_bella"):
        """Run ongoing reading mode"""
        print("\nRunning ongoing reading mode...")
        print("Press SPACE to read screen content")
        print("Press Q to quit")

        self.running = True

        while self.running:
            # Check for key presses
            key = pyautogui.press()

            if key == "space":
                await self.read_screen_content(voice)
                pass

            elif key == "q" or key == "escape":
                self.running = False
                break

            await asyncio.sleep(0.1)

    async def cleanup(self):
        """Cleanup resources"""
        if self.reader:
            await self.reader.provider.shutdown()
        print("Accessibility tool cleanup complete")


async def main():
    """Main accessibility tool example"""

    print("=" * 50)
    print("Accessibility Tool Example")
    print("=" * 50)

    tool = AccessibilityTool()

    try:
        await tool.initialize()

        print("\nAvailable features:")
        print("  1. Read screen content")
        print("  2. Read selected text")
        print("  3. Read from application")
        print("  4. Read document")
        print("  5. Read OCR text")
        print("  6. Read notification")
        print("  7. Read message")
        print("  8. Run ongoing reading mode")
        print("\nExample 1: Read selected text")
        await tool.read_selection("This is selected text that will be read aloud.")

        print("\nExample 2: Read notification")
        await tool.read_notification("New email received from support")

        print("\nExample 3: Read document")
        test_file = "test_document.txt"
        with open(test_file, "w") as f:
            f.write("This is a test document for the accessibility tool.")
        await tool.read_document(test_file)
        os.remove(test_file)

        print("\nExample 4: Read message")
        await tool.read_message("Alice", "Hello! How are you?")

        print("\n" + "=" * 50)
        print("Accessibility tool examples completed!")
        print("=" * 50)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await tool.cleanup()


async def main_ongoing():
    """Run ongoing reading mode"""

    print("=" * 50)
    print("Ongoing Reading Mode")
    print("=" * 50)
    print("\nRunning... Press SPACE to read, Q to quit")
    print("Note: In a real implementation, you would handle screen reading APIs")

    tool = AccessibilityTool()

    try:
        await tool.initialize()
        await tool.run_ongoing_reading("af_bella")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        await tool.cleanup()


if __name__ == "__main__":
    try:
        print("\nExample 1: Demo of all features")
        asyncio.run(main())

        print("\n\n" + "=" * 50)
        print("Example 2: Ongoing reading mode (requires keyboard interaction)")
        print("Press Ctrl+C to stop")
        print("=" * 50)
        asyncio.run(main_ongoing())
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
