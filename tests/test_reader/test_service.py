"""
Tests for the text reader service.
"""


import contextlib

import pytest

from champi_tts.reader import ReaderState, TextReaderService


@pytest.mark.asyncio
async def test_reader_initialization(mock_provider):
    """Test reader service initialization."""
    reader = TextReaderService(mock_provider)
    assert reader.state == ReaderState.IDLE
    assert reader.provider == mock_provider


@pytest.mark.asyncio
async def test_reader_read_text(initialized_provider):
    """Test reading a single text."""
    reader = TextReaderService(initialized_provider)

    # Track state changes
    states = []

    def track_state(sender, **kwargs):
        states.append(kwargs["new_state"])

    reader.on_state_changed.connect(track_state)

    # Read text
    await reader.read_text("Hello, world!")

    # Verify state transitions
    assert "reading" in states
    assert reader.state == ReaderState.IDLE


@pytest.mark.asyncio
async def test_reader_pause_resume(initialized_provider):
    """Test pause and resume functionality."""
    reader = TextReaderService(initialized_provider)

    # Start reading in background
    import asyncio

    read_task = asyncio.create_task(reader.read_text("Long text to read"))

    # Give it time to start
    await asyncio.sleep(0.1)

    # Pause
    await reader.pause()
    assert reader.state == ReaderState.PAUSED

    # Resume
    await reader.resume()
    assert reader.state == ReaderState.READING

    # Wait for completion
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(read_task, timeout=1.0)


@pytest.mark.asyncio
async def test_reader_stop(initialized_provider):
    """Test stop functionality."""
    reader = TextReaderService(initialized_provider)

    # Start reading
    import asyncio

    read_task = asyncio.create_task(reader.read_text("Text to read"))

    # Give it time to start
    await asyncio.sleep(0.1)

    # Stop
    await reader.stop()
    assert reader.state == ReaderState.STOPPED

    # Cancel the task
    read_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await read_task


@pytest.mark.asyncio
async def test_reader_queue(initialized_provider):
    """Test text queue functionality."""
    reader = TextReaderService(initialized_provider)

    # Add to queue
    reader.add_to_queue("Text 1")
    reader.add_to_queue("Text 2")
    reader.add_to_queue("Text 3")

    # Verify queue
    assert len(reader._text_queue) == 3

    # Clear queue
    reader.clear_queue()
    assert len(reader._text_queue) == 0


@pytest.mark.asyncio
async def test_reader_signals(initialized_provider):
    """Test reader signal emissions."""
    reader = TextReaderService(initialized_provider)

    # Track signals
    signals_received = []

    reader.on_reading_started.connect(
        lambda sender, **kw: signals_received.append("started")
    )
    reader.on_reading_completed.connect(
        lambda sender, **kw: signals_received.append("completed")
    )

    # Read text
    await reader.read_text("Test text")

    # Verify signals
    assert "started" in signals_received
    assert "completed" in signals_received
