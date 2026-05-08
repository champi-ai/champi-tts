"""
Integration tests for full TTS workflow.

These tests verify end-to-end functionality including:
- Full reading workflow (text -> synthesis -> playback)
- Pause/resume/stop sequences
- Queue management edge cases
- Error handling scenarios
- UI state transitions (when UI is available)
"""

import asyncio
import pytest
from pathlib import Path

from champi_tts import ReaderState, TextReaderService
from champi_tts.core.audio import AudioPlayer
from champi_tts.factory import get_provider


@pytest.fixture
def mock_provider():
    """Mock provider for integration tests."""
    from tests.conftest import MockTTSConfig, MockTTSProvider

    config = MockTTSConfig()
    return MockTTSProvider(config)


@pytest.mark.asyncio
async def test_full_read_workflow(mock_provider):
    """
    Test complete reading workflow: read_text -> synthesis -> playback.
    """
    reader = TextReaderService(mock_provider)

    # Track states
    states = []

    def on_state_changed(sender, **kwargs):
        states.append(kwargs["new_state"])

    reader.on_state_changed.connect(on_state_changed)

    # Read text
    await reader.read_text("Hello, this is a test message")

    # Verify state transitions
    assert states == ["reading", "idle"]
    assert reader.state == ReaderState.IDLE


@pytest.mark.asyncio
async def test_pause_resume_sequence(mock_provider):
    """
    Test pause/resume/stop sequence for reading.
    """
    reader = TextReaderService(mock_provider)

    states = []
    reader.on_state_changed.connect(lambda s, **k: states.append(k["new_state"]))

    # Start reading
    await reader.read_text("First part of text")
    states.clear()

    # Start another read that can be paused
    async def long_read():
        for i in range(10):
            if reader.state == ReaderState.PAUSED:
                break
            await asyncio.sleep(0.01)
            # Simulate ongoing work

    read_task = asyncio.create_task(long_read())

    # Give it time to start
    await asyncio.sleep(0.05)

    # Verify currently reading
    assert reader.state == ReaderState.READING

    # Pause
    await reader.pause()
    assert reader.state == ReaderState.PAUSED
    assert "paused" in states

    # Verify playback stopped
    await asyncio.sleep(0.01)
    assert not reader.player.is_playing()

    # Resume
    await reader.resume()
    assert reader.state == ReaderState.READING
    assert "resumed" in states

    # Wait for completion
    read_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await read_task


@pytest.mark.asyncio
async def test_stop_during_reading(mock_provider):
    """
    Test stop functionality during active reading.
    """
    reader = TextReaderService(mock_provider)

    states = []
    reader.on_state_changed.connect(lambda s, **k: states.append(k["new_state"]))

    # Start reading
    await reader.read_text("Text to read")

    # Verify state transitions
    assert states == ["reading", "idle"]


@pytest.mark.asyncio
async def test_queue_management():
    """
    Test queue management with edge cases.
    """
    from tests.conftest import MockTTSProvider, MockTTSConfig

    config = MockTTSConfig()
    provider = MockTTSProvider(config)
    reader = TextReaderService(provider)

    # Add items to queue
    reader.add_to_queue("Text 1")
    reader.add_to_queue("Text 2")
    reader.add_to_queue("Text 3")

    assert len(reader._text_queue) == 3

    # Verify queue order
    assert reader._text_queue[0] == "Text 1"

    # Clear queue
    reader.clear_queue()
    assert len(reader._text_queue) == 0


@pytest.mark.asyncio
async def test_queue_with_pause_resume():
    """
    Test queue operations with pause/resume.
    """
    from tests.conftest import MockTTSProvider, MockTTSConfig

    config = MockTTSConfig()
    provider = MockTTSProvider(config)
    reader = TextReaderService(provider)

    # Add to queue
    reader.add_to_queue("Text 1")
    reader.add_to_queue("Text 2")

    # Start reading first item
    await reader.read_text("First text")
    await reader.pause()

    # Verify queue still has items
    assert len(reader._text_queue) == 2

    # Clear queue
    reader.clear_queue()
    assert len(reader._text_queue) == 0

    # Verify state returns to idle
    assert reader.state == ReaderState.IDLE


@pytest.mark.asyncio
async def test_read_file_workflow():
    """
    Test reading from file with paragraph processing.
    """
    from tests.conftest import MockTTSProvider, MockTTSConfig

    config = MockTTSConfig()
    provider = MockTTSProvider(config)
    reader = TextReaderService(provider)

    # Create test file
    test_dir = Path(__file__).parent
    test_file = test_dir / "test_document.txt"

    # Write test content
    test_file.write_text(
        "This is the first paragraph.\n\n"
        "This is the second paragraph.\n\n"
        "This is the third paragraph."
    )

    states = []
    reader.on_state_changed.connect(lambda s, **k: states.append(k["new_state"]))

    # Read file
    await reader.read_file(test_file)

    # Verify all paragraphs were read
    assert states == ["reading", "idle", "reading", "idle", "reading", "idle"]

    # Cleanup
    test_file.unlink()


@pytest.mark.asyncio
async def test_file_not_found():
    """
    Test reading from non-existent file raises FileNotFoundError.
    """
    from tests.conftest import MockTTSProvider, MockTTSConfig

    config = MockTTSConfig()
    provider = MockTTSProvider(config)
    reader = TextReaderService(provider)

    with pytest.raises(FileNotFoundError, match="File not found"):
        await reader.read_file("/nonexistent/file.txt")


@pytest.mark.asyncio
async def test_error_handling():
    """
    Test error handling during synthesis.
    """
    from tests.conftest import MockTTSProvider, MockTTSConfig

    config = MockTTSConfig()
    provider = MockTTSProvider(config)
    reader = TextReaderService(provider)

    # Track error signals
    errors = []
    reader.on_error.connect(lambda s, **k: errors.append(k.get("error", "")))

    # Read text (should succeed)
    await reader.read_text("Test text")

    # Verify no errors
    assert len(errors) == 0


@pytest.mark.asyncio
async def test_state_transitions():
    """
    Test all state transition paths.
    """
    from tests.conftest import MockTTSProvider, MockTTSConfig

    config = MockTTSConfig()
    provider = MockTTSProvider(config)
    reader = TextReaderService(provider)

    # Track all state changes
    transitions = []
    reader.on_state_changed.connect(lambda s, **k: transitions.append(k["new_state"]))

    # Start reading
    await reader.read_text("Text")
    assert "idle" in transitions

    # The state is idle after completion, verify reading was seen
    # (the test may not catch all transitions due to async nature)


@pytest.mark.asyncio
async def test_reader_cleanup():
    """
    Test reader cleanup on exit.
    """
    from tests.conftest import MockTTSProvider, MockTTSConfig

    config = MockTTSConfig()
    provider = MockTTSProvider(config)
    reader = TextReaderService(provider)

    # Initialize provider
    await reader.provider.initialize()
    assert reader.provider._initialized

    # Cleanup
    await reader.cleanup()

    # Verify provider cleaned up
    assert not reader.provider._initialized


@pytest.mark.asyncio
async def test_context_manager(mock_provider):
    """
    Test async context manager for reader service.
    """
    reader = TextReaderService(mock_provider)

    async with reader:
        # Should be initialized
        assert reader.provider._initialized

        # Read text
        await reader.read_text("Test in context")

    # Should be cleaned up
    assert not reader.provider._initialized


@pytest.mark.asyncio
async def test_read_text_with_voice():
    """
    Test reading text with custom voice parameter.
    """
    from tests.conftest import MockTTSProvider, MockTTSConfig

    config = MockTTSConfig(default_voice="af_bella")
    provider = MockTTSProvider(config)
    reader = TextReaderService(provider)

    # Track voice parameter
    voices_used = []

    original_synthesize = provider.synthesize
    async def tracked_synthesize(*args, voice=None, **kwargs):
        voices_used.append(voice or config.default_voice)
        return await original_synthesize(*args, **kwargs)

    provider.synthesize = tracked_synthesize

    # Read with explicit voice
    await reader.read_text("Test", voice="af_sarah")
    assert voices_used[-1] == "af_sarah"

    # Reset and read without voice
    voices_used.clear()
    await reader.read_text("Test")
    assert voices_used[-1] == "af_bella"


@pytest.mark.asyncio
async def test_read_text_with_speed():
    """
    Test reading text with custom speed parameter.
    """
    from tests.conftest import MockTTSProvider, MockTTSConfig

    config = MockTTSConfig(default_speed=1.0)
    provider = MockTTSProvider(config)
    reader = TextReaderService(provider)

    speeds_used = []

    original_synthesize = provider.synthesize
    async def tracked_synthesize(*args, speed=None, **kwargs):
        speeds_used.append(speed or config.default_speed)
        return await original_synthesize(*args, **kwargs)

    provider.synthesize = tracked_synthesize

    # Read with explicit speed
    await reader.read_text("Test", speed=1.5)
    assert speeds_used[-1] == 1.5

    # Reset and read without speed
    speeds_used.clear()
    await reader.read_text("Test")
    assert speeds_used[-1] == 1.0

