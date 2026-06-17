"""
Abstract base configuration for TTS providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

_VALID_AUDIO_FORMATS = frozenset({"wav", "mp3", "flac", "pcm", "ogg"})
_MIN_CHUNK_SIZE = 50


@dataclass
class BaseTTSConfig(ABC):
    """Abstract base class for TTS provider configuration."""

    # Common TTS settings
    sample_rate: int = 24000
    audio_format: str = "wav"
    default_voice: str = ""
    default_speed: float = 1.0

    # Text processing
    normalize_text: bool = True

    # Streaming
    enable_streaming: bool = True
    streaming_chunk_size: int = 200

    # Initialization
    warmup_on_init: bool = True
    auto_download_model: bool = True

    @classmethod
    @abstractmethod
    def from_env(cls) -> "BaseTTSConfig":
        """Create configuration from environment variables."""
        pass

    def validate(self) -> bool:
        """Validate common configuration settings.

        Raises:
            ValueError: If any configuration value is invalid.

        Returns:
            True if all settings are valid.
        """
        if self.sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be a positive integer, got {self.sample_rate}"
            )
        if self.default_speed <= 0:
            raise ValueError(
                f"default_speed must be positive, got {self.default_speed}"
            )
        if self.streaming_chunk_size < _MIN_CHUNK_SIZE:
            raise ValueError(
                f"streaming_chunk_size must be >= {_MIN_CHUNK_SIZE}, "
                f"got {self.streaming_chunk_size}"
            )
        if self.audio_format not in _VALID_AUDIO_FORMATS:
            raise ValueError(
                f"audio_format must be one of {sorted(_VALID_AUDIO_FORMATS)}, "
                f"got '{self.audio_format}'"
            )
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
