"""
Pytest configuration and fixtures for champi-tts tests.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

from champi_tts.core.base_config import BaseTTSConfig
from champi_tts.core.base_provider import BaseTTSProvider


class MockTTSConfig(BaseTTSConfig):
    """Mock configuration for testing."""

    @classmethod
    def from_env(cls) -> "MockTTSConfig":
        return cls()

    def validate(self) -> bool:
        return True


class MockTTSProvider(BaseTTSProvider):
    """Mock TTS provider for testing."""

    def __init__(self, config: BaseTTSConfig | None = None):
        super().__init__(config or MockTTSConfig())

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    async def synthesize(
        self, text: str, voice: str | None = None, speed: float | None = None, **kwargs
    ) -> np.ndarray:
        # Return mock audio data
        return np.zeros(1000, dtype=np.float32)

    async def synthesize_streaming(
        self, text: str, voice: str | None = None, speed: float | None = None, **kwargs
    ):
        # Yield mock audio chunks
        for _ in range(3):
            yield np.zeros(100, dtype=np.float32)

    async def list_voices(self) -> list[str]:
        return ["voice1", "voice2", "voice3"]

    async def interrupt(self) -> None:
        self._is_speaking = False

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
        return False


@pytest.fixture
def mock_sd():
    """Prevent actual audio hardware access during tests.

    Patches the module-level ``sd`` reference in ``champi_tts.core.audio``
    so that ``AudioPlayer.play()`` / ``stop()`` never reach PortAudio even
    when the sounddevice library is absent or PortAudio is not installed.
    """
    mock = MagicMock()
    with patch("champi_tts.core.audio.sd", mock):
        yield mock


@pytest.fixture
def mock_config():
    """Fixture for mock configuration."""
    return MockTTSConfig()


@pytest.fixture
def mock_provider(mock_config):
    """Fixture for mock TTS provider."""
    return MockTTSProvider(mock_config)


@pytest_asyncio.fixture
async def initialized_provider(mock_provider):
    """Fixture for initialized mock provider."""
    await mock_provider.initialize()
    yield mock_provider
    await mock_provider.shutdown()
