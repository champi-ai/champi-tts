"""
Pytest configuration and fixtures for champi-tts tests.
"""

import asyncio
import threading
import time
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


class FakeSoundDevice:
    """sounddevice stand-in whose wait() blocks like real playback."""

    def __init__(self, playback_seconds: float = 5.0) -> None:
        self.playback_seconds = playback_seconds
        self.played = threading.Event()
        self._finished = threading.Event()
        self.play_calls: list[tuple] = []
        self.stop_calls = 0

    def play(self, data: np.ndarray, samplerate: int, **kwargs) -> None:
        """Record the call and start the fake stream."""
        self.play_calls.append((data, samplerate, kwargs))
        self._finished.clear()
        self.played.set()

    def wait(self) -> None:
        """Block the calling thread until playback ends or stop() is called."""
        self._finished.wait(self.playback_seconds)

    def stop(self) -> None:
        """End the fake stream, releasing any wait()."""
        self.stop_calls += 1
        self._finished.set()

    async def wait_started(self, timeout: float = 2.0) -> None:
        """Yield to the event loop until a worker thread has called play()."""
        deadline = time.monotonic() + timeout
        while not self.played.is_set():
            assert time.monotonic() < deadline, "playback never started"
            await asyncio.sleep(0.005)


@pytest.fixture
def blocking_sd():
    """Unpatched FakeSoundDevice, released at teardown so no worker hangs."""
    fake = FakeSoundDevice()
    yield fake
    fake.stop()


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
