"""
Tests for the text reader service.
"""

import asyncio
import contextlib

import numpy as np
import pytest

from champi_tts import ReaderState, TextReaderService
from tests.conftest import MockTTSConfig, MockTTSProvider


class _SlowMockProvider(MockTTSProvider):
    """Mock provider with a configurable synthesis delay."""

    def __init__(self, delay: float = 0.3):
        super().__init__(MockTTSConfig())
        self._delay = delay

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Synthesize with artificial delay."""
        await asyncio.sleep(self._delay)
        return np.zeros(1000, dtype=np.float32)


@pytest.fixture
def slow_mock_provider():
    """MockTTSProvider with a 0.3 s synthesis delay."""
    return _SlowMockProvider(delay=0.3)


@pytest.mark.asyncio
async def test_reader_initialization(mock_provider):
    """Test reader service initialization."""
    reader = TextReaderService(mock_provider)
    assert reader.state == ReaderState.IDLE
    assert reader.provider == mock_provider


@pytest.mark.asyncio
async def test_reader_read_text(initialized_provider, mock_sd):
    """Test reading a single text."""
    reader = TextReaderService(initialized_provider)

    states = []

    def track_state(sender, **kwargs):
        states.append(kwargs["new_state"])

    reader.on_state_changed.connect(track_state)

    await reader.read_text("Hello, world!")

    assert "reading" in states
    assert reader.state == ReaderState.IDLE


@pytest.mark.asyncio
async def test_reader_pause_resume(slow_mock_provider, mock_sd):
    """Test pause and resume functionality during active synthesis."""
    await slow_mock_provider.initialize()
    reader = TextReaderService(slow_mock_provider)

    read_task = asyncio.create_task(reader.read_text("Long text to read"))

    # Give the task time to enter synthesis (0.3 s delay).
    await asyncio.sleep(0.05)

    assert reader.state == ReaderState.READING

    await reader.pause()
    assert reader.state == ReaderState.PAUSED

    await reader.resume()
    assert reader.state == ReaderState.READING

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(read_task, timeout=1.0)

    await slow_mock_provider.shutdown()


@pytest.mark.asyncio
async def test_reader_stop(slow_mock_provider, mock_sd):
    """Test stop functionality during active synthesis."""
    await slow_mock_provider.initialize()
    reader = TextReaderService(slow_mock_provider)

    read_task = asyncio.create_task(reader.read_text("Text to read"))

    await asyncio.sleep(0.05)

    await reader.stop()
    assert reader.state == ReaderState.STOPPED

    read_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await read_task

    await slow_mock_provider.shutdown()


@pytest.mark.asyncio
async def test_reader_queue(initialized_provider):
    """Test text queue functionality."""
    reader = TextReaderService(initialized_provider)

    reader.add_to_queue("Text 1")
    reader.add_to_queue("Text 2")
    reader.add_to_queue("Text 3")

    assert len(reader._text_queue) == 3

    reader.clear_queue()
    assert len(reader._text_queue) == 0


@pytest.mark.asyncio
async def test_reader_signals(initialized_provider, mock_sd):
    """Test reader signal emissions."""
    reader = TextReaderService(initialized_provider)

    signals_received = []

    # weak=False is required: blinker 1.9 drops inline lambdas without a
    # strong referent immediately after connect() returns.
    reader.on_reading_started.connect(
        lambda sender, **kw: signals_received.append("started"), weak=False
    )
    reader.on_reading_completed.connect(
        lambda sender, **kw: signals_received.append("completed"), weak=False
    )

    await reader.read_text("Test text")

    assert "started" in signals_received
    assert "completed" in signals_received
