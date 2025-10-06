"""
Abstract base configuration for TTS providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


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

    @abstractmethod
    def validate(self) -> bool:
        """Validate configuration settings."""
        pass

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
