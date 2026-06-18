"""
Shared fixtures for integration tests.
"""

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from champi_tts.core.audio import AudioPlayer
from tests.conftest import MockTTSConfig, MockTTSProvider


class SlowMockProvider(MockTTSProvider):
    """Mock provider with a configurable synthesis delay for async timing tests."""

    def __init__(self, delay: float = 0.3):
        super().__init__()
        self._delay = delay

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Synthesize with artificial delay to allow concurrent state assertions."""
        await asyncio.sleep(self._delay)
        return np.zeros(1000, dtype=np.float32)


@pytest.fixture
def mock_provider():
    """Mock provider for integration tests."""
    return MockTTSProvider(MockTTSConfig())


@pytest.fixture
def slow_provider():
    """Mock provider with a 0.3 s synthesis delay for pause/resume timing tests."""
    return SlowMockProvider(delay=0.3)


@pytest.fixture
def initialized_provider(mock_provider):
    """Mock provider with _initialized set to True."""
    mock_provider._initialized = True
    return mock_provider


@pytest.fixture
def audio_player():
    """AudioPlayer at 22050 Hz for audio playback tests."""
    return AudioPlayer(sample_rate=22050)


@pytest.fixture
def float_audio():
    """Float32 440 Hz sine wave at 22050 Hz, 0.5 s duration."""
    sample_rate = 22050
    t = np.linspace(0, 0.5, int(sample_rate * 0.5))
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


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
def test_document(tmp_path):
    """Three-paragraph text file for file-reading tests."""
    doc = tmp_path / "test_document.txt"
    doc.write_text(
        "This is the first paragraph.\n\n"
        "This is the second paragraph.\n\n"
        "This is the third paragraph."
    )
    return doc
