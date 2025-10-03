"""
Abstract base synthesizer for low-level TTS operations.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np


class BaseSynthesizer(ABC):
    """Abstract base class for low-level TTS synthesis."""

    @abstractmethod
    async def load_model(self) -> None:
        """Load the TTS model."""
        pass

    @abstractmethod
    async def unload_model(self) -> None:
        """Unload the TTS model."""
        pass

    @abstractmethod
    async def synthesize_audio(
        self,
        text: str,
        voice_data: Any,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Synthesize text to audio.

        Args:
            text: Text to synthesize
            voice_data: Voice model/embedding data
            speed: Speech speed multiplier
            **kwargs: Additional parameters

        Returns:
            Audio data as numpy array
        """
        pass

    @abstractmethod
    async def synthesize_streaming(
        self,
        text: str,
        voice_data: Any,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Synthesize text to audio with streaming.

        Args:
            text: Text to synthesize
            voice_data: Voice model/embedding data
            speed: Speech speed multiplier
            **kwargs: Additional parameters

        Yields:
            Audio chunks as numpy arrays
        """
        pass

    @abstractmethod
    async def preprocess_text(self, text: str) -> str:
        """Preprocess text before synthesis."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass
