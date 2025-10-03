"""Custom provider example.

This example demonstrates creating a custom TTS provider.
"""

import asyncio
from dataclasses import dataclass

import numpy as np

from champi_tts.core import BaseTTSConfig, BaseTTSProvider


@dataclass
class CustomConfig(BaseTTSConfig):
    """Configuration for custom TTS provider."""

    voice_pitch: float = 1.0
    echo_effect: bool = False


class CustomTTSProvider(BaseTTSProvider):
    """Custom TTS provider that generates simple beep patterns.

    This is a demonstration provider that converts text to beep patterns
    based on text length and characters.
    """

    def __init__(self, config: CustomConfig):
        super().__init__(config)
        self._config: CustomConfig = config
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the provider."""
        print("Initializing custom TTS provider...")
        self._initialized = True

    async def cleanup(self) -> None:
        """Cleanup resources."""
        print("Cleaning up custom TTS provider...")
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized

    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return False

    async def synthesize(
        self, text: str, voice: str | None = None, speed: float | None = None, **kwargs
    ) -> np.ndarray:
        """Synthesize text to beep pattern audio.

        Args:
            text: Text to synthesize
            voice: Voice to use (ignored in this example)
            speed: Speech speed (ignored in this example)

        Returns:
            Audio array
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        # Generate beep pattern based on text
        duration = len(text) * 0.1  # 0.1s per character
        sample_rate = self._config.sample_rate
        num_samples = int(duration * sample_rate)

        # Create beep frequency based on text hash
        base_freq = 440.0 * self._config.voice_pitch
        freq = base_freq + (hash(text) % 200)

        # Generate sine wave
        t = np.linspace(0, duration, num_samples)
        audio = 0.3 * np.sin(2 * np.pi * freq * t)

        # Add echo effect if enabled
        if self._config.echo_effect:
            echo_delay = int(0.2 * sample_rate)  # 200ms delay
            echo = np.zeros_like(audio)
            echo[echo_delay:] = audio[:-echo_delay] * 0.5
            audio = audio + echo

        return audio.astype(np.float32)

    async def synthesize_streaming(
        self, text: str, voice: str | None = None, speed: float | None = None, **kwargs
    ):
        """Stream synthesis (yields full audio for this simple example)."""
        audio = await self.synthesize(text, voice, speed, **kwargs)
        yield audio

    def list_voices(self) -> list[str]:
        """List available voices."""
        return ["beep_low", "beep_high", "beep_echo"]

    async def interrupt(self) -> None:
        """Interrupt current synthesis."""
        pass


async def main():
    """Custom provider example."""
    # Create custom configuration
    config = CustomConfig(
        sample_rate=24000, voice_pitch=1.2, echo_effect=True, audio_format="wav"
    )

    # Create custom provider
    provider = CustomTTSProvider(config)
    await provider.initialize()

    # List available voices
    print(f"Available voices: {provider.list_voices()}\n")

    # Synthesize text
    text = "Hello from custom TTS provider!"
    print(f"Synthesizing: {text}")

    audio = await provider.synthesize(text)

    # Play audio
    import sounddevice as sd

    sd.play(audio, samplerate=config.sample_rate)
    sd.wait()

    print("Custom synthesis complete!")

    # Cleanup
    await provider.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
