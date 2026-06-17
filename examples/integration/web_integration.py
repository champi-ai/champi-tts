#!/usr/bin/env python3
"""
Web Application Integration Example

This example demonstrates how to integrate champi-tts into a Flask web application.
Shows API endpoint for text-to-speech synthesis.
"""

import asyncio
import os
import sys

from flask import Flask, jsonify, request

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champi_tts import get_provider
from champi_tts.core.audio import save_audio

# Create Flask app
app = Flask(__name__)

# Store providers for reuse (in production, use proper pooling)
providers = {}


def get_provider_async(provider_name="kokoro"):
    """Get or create provider (async)"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        if provider_name not in providers:
            providers[provider_name] = get_provider(provider_name)
            loop.run_until_complete(providers[provider_name].initialize())

        return providers[provider_name]
    finally:
        loop.close()


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "champi-tts"})


@app.route("/synthesize", methods=["POST"])
def synthesize():
    """
    Synthesize text to speech

    Request body:
    {
        "text": "Text to synthesize",
        "voice": "af_bella",
        "output_format": "wav"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        text = data.get("text")
        voice = data.get("voice", "af_bella")
        _ = data.get("output_format", "wav")

        if not text:
            return jsonify({"error": "Text parameter is required"}), 400

        # Get provider
        provider = get_provider_async()

        # Synthesize
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            audio = loop.run_until_complete(provider.synthesize(text, voice=voice))

            # Save to file
            output_file = "output.wav"
            loop.run_until_complete(
                save_audio(audio, output_file, sample_rate=provider.config.sample_rate)
            )

            # Return response
            response = {
                "status": "success",
                "output_file": output_file,
                "voice_used": voice,
                "text_length": len(text),
            }

            return jsonify(response), 200

        finally:
            loop.close()

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/voices", methods=["GET"])
def list_voices():
    """List available voices"""
    try:
        provider = get_provider_async()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(provider.initialize())
            loop.run_until_complete(provider.shutdown())

            return jsonify({"status": "success", "voices": provider.voices}), 200

        finally:
            loop.close()

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def main():
    """Run the Flask application"""

    print("=" * 50)
    print("Flask Web Application Integration")
    print("=" * 50)

    print("\nStarting Flask server...")
    print("\nAPI Endpoints:")
    print("  GET  /health           - Health check")
    print("  POST /synthesize        - Synthesize text to speech")
    print("  GET  /voices           - List available voices")
    print("\nExample usage:")
    print("  curl -X POST http://localhost:5000/synthesize \\")
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"text": "Hello, world!", "voice": "af_bella"}\'')
    print("\n" + "=" * 50)

    # Run Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()


# For production, consider:
# 1. Using proper async/await in Flask with threading
# 2. Implementing provider pooling
# 3. Adding authentication
# 4. Implementing rate limiting
# 5. Using proper logging
# 6. Adding request validation
# 7. Implementing error handling and retry logic
# 8. Using a WSGI server like Gunicorn or uWSGI
# 9. Implementing file cleanup
# 10. Adding monitoring and metrics
