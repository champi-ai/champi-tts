"""
Adapter to make KokoroProvider conform to BaseTTSProvider interface.
"""

from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
from loguru import logger

from champi_tts.core.base_provider import BaseTTSProvider
from champi_tts.providers.kokoro.config import KokoroConfig
from champi_tts.providers.kokoro.provider import (
    KokoroProvider as OriginalKokoroProvider,
)


class KokoroProviderAdapter(BaseTTSProvider):
    """
    Adapter that wraps KokoroProvider to implement BaseTTSProvider interface.
    """

    def __init__(self, config: KokoroConfig):
        super().__init__(config)
        self._kokoro = OriginalKokoroProvider(config)

    async def initialize(self) -> None:
        """Initialize the Kokoro TTS provider."""
        try:
            await self._kokoro.initialize()
            self._initialized = True
            logger.info("Kokoro provider initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro provider: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the Kokoro TTS provider."""
        try:
            await self._kokoro.shutdown()
            self._initialized = False
            logger.info("Kokoro provider shutdown successfully")
        except Exception as e:
            logger.error(f"Error during Kokoro shutdown: {e}")
            raise

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize
            voice: Voice to use (optional, uses default if None)
            speed: Speech speed (optional, uses default if None)
            **kwargs: Additional Kokoro-specific parameters

        Returns:
            Audio data as numpy array
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        self._is_speaking = True
        try:
            # Use Kokoro's synthesis method
            voice = voice or self.config.default_voice
            speed = speed or self.config.default_speed

            audio = await self._kokoro.synthesize(
                text=text, voice=voice, speed=speed, **kwargs
            )

            return audio

        finally:
            self._is_speaking = False

    async def synthesize_streaming(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Synthesize text to speech with streaming output.

        Args:
            text: Text to synthesize
            voice: Voice to use (optional, uses default if None)
            speed: Speech speed (optional, uses default if None)
            **kwargs: Additional Kokoro-specific parameters

        Yields:
            Audio chunks as numpy arrays
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        self._is_speaking = True
        try:
            voice = voice or self.config.default_voice
            speed = speed or self.config.default_speed

            async for chunk in self._kokoro.synthesize_streaming(
                text=text, voice=voice, speed=speed, **kwargs
            ):
                yield chunk

        finally:
            self._is_speaking = False

    async def list_voices(self) -> list[str]:
        """List available voices."""
        return await self._kokoro.list_voices()

    async def interrupt(self) -> None:
        """Interrupt current synthesis/playback."""
        self._is_speaking = False
        # Kokoro provider should implement interrupt logic
        if hasattr(self._kokoro, "interrupt"):
            await self._kokoro.interrupt()
        logger.info("Synthesis interrupted")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
