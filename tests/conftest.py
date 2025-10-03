"""
Pytest configuration and fixtures for champi-tts tests.
"""


import numpy as np
import pytest

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


@pytest.fixture
def mock_config():
    """Fixture for mock configuration."""
    return MockTTSConfig()


@pytest.fixture
def mock_provider(mock_config):
    """Fixture for mock TTS provider."""
    return MockTTSProvider(mock_config)


@pytest.fixture
async def initialized_provider(mock_provider):
    """Fixture for initialized mock provider."""
    await mock_provider.initialize()
    yield mock_provider
    await mock_provider.shutdown()
