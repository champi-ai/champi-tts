#!/usr/bin/env python3
"""
Basic CLI Examples

This file demonstrates various CLI commands available in champi-tts.
Run these commands in your terminal.
"""

import subprocess
import sys


def run_command(command, description):
    """Run a command and display its output"""
    print(f"\n{'=' * 60}")
    print(f"Command: {command}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"Success: {description}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(e.stderr)
    except FileNotFoundError:
        print("Error: Command not found. Is champi-tts installed?")


def main():
    """Run basic CLI examples"""

    print("=" * 60)
    print("Basic CLI Examples for Champi TTS")
    print("=" * 60)

    # Check if champi-tts is installed
    try:
        subprocess.run(["champi-tts", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nError: champi-tts is not installed or not in PATH.")
        print("Install it using: pip install champi-tts")
        sys.exit(1)

    # Example 1: List available voices
    run_command("champi-tts list-voices", "List available voices")

    # Example 2: Simple synthesis
    run_command('champi-tts synthesize "Hello, world!"', "Simple synthesis")

    # Example 3: Synthesize with voice
    run_command(
        'champi-tts synthesize "Hello with a voice!" --voice af_bella',
        "Synthesize with specific voice",
    )

    # Example 4: Save output
    run_command(
        'champi-tts synthesize "Saved to file" --output output.wav --no-play',
        "Save synthesis to file",
    )

    # Example 5: Text reading
    run_command(
        'champi-tts read --text "This is a test" --voice af_bella', "Read text directly"
    )

    # Example 6: UI test
    run_command("champi-tts test-ui", "Test UI indicator")

    print("\n" + "=" * 60)
    print("CLI examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
