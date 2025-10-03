"""
Abstract base provider interface for TTS providers.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np

from champi_tts.core.base_config import BaseTTSConfig


class BaseTTSProvider(ABC):
    """Abstract base class for TTS providers."""

    def __init__(self, config: BaseTTSConfig):
        self.config = config
        self._initialized = False
        self._is_speaking = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
        return False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the TTS provider."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the TTS provider."""
        pass

    @abstractmethod
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
            **kwargs: Additional provider-specific parameters

        Returns:
            Audio data as numpy array
        """
        pass

    @abstractmethod
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
            **kwargs: Additional provider-specific parameters

        Yields:
            Audio chunks as numpy arrays
        """
        pass

    @abstractmethod
    async def list_voices(self) -> list[str]:
        """List available voices."""
        pass

    @abstractmethod
    async def interrupt(self) -> None:
        """Interrupt current synthesis/playback."""
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized

    @property
    def is_speaking(self) -> bool:
        """Check if provider is currently speaking."""
        return self._is_speaking
