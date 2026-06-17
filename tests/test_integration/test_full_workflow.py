"""
Integration tests for the full TTS reading workflow.

Covers:
- Full reading workflow (text -> synthesis -> playback)
- Pause / resume / stop sequences
- Queue management edge cases
- Error handling scenarios
- Context manager and lifecycle
- Voice parameter forwarding
- Long-document performance
"""

import asyncio
import contextlib

import pytest

from champi_tts import ReaderState, TextReaderService

# ---------------------------------------------------------------------------
# Full reading workflow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_read_workflow_state_transitions(mock_provider, mock_sd):
    """Complete read produces READING then IDLE state transitions."""
    reader = TextReaderService(mock_provider)
    states = []
    reader.on_state_changed.connect(lambda s, **k: states.append(k["new_state"]))

    await reader.read_text("Hello, this is a test message")

    assert "reading" in states
    assert "idle" in states
    assert reader.state == ReaderState.IDLE


@pytest.mark.asyncio
async def test_full_read_emits_started_and_completed_signals(mock_provider, mock_sd):
    """Reading emits on_reading_started and on_reading_completed with the text."""
    reader = TextReaderService(mock_provider)
    started_texts: list[str | None] = []
    completed_texts: list[str | None] = []
    reader.on_reading_started.connect(
        lambda s, **k: started_texts.append(k.get("text"))
    )
    reader.on_reading_completed.connect(
        lambda s, **k: completed_texts.append(k.get("text"))
    )

    await reader.read_text("Test message")

    assert started_texts == ["Test message"]
    assert completed_texts == ["Test message"]


@pytest.mark.asyncio
async def test_state_change_signal_includes_old_and_new_values(mock_provider, mock_sd):
    """on_state_changed signals carry both old_state and new_state values."""
    reader = TextReaderService(mock_provider)
    transitions: list[tuple[str, str]] = []
    reader.on_state_changed.connect(
        lambda s, **k: transitions.append((k["old_state"], k["new_state"]))
    )

    await reader.read_text("Test")

    assert ("idle", "reading") in transitions
    assert ("reading", "idle") in transitions


# ---------------------------------------------------------------------------
# Pause sequences
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pause_transitions_reading_to_paused(mock_provider):
    """pause() transitions the reader from READING to PAUSED."""
    reader = TextReaderService(mock_provider)
    reader._set_state(ReaderState.READING)

    await reader.pause()

    assert reader.state == ReaderState.PAUSED


@pytest.mark.asyncio
async def test_pause_is_noop_when_not_reading(mock_provider):
    """pause() is a no-op when the reader is not in READING state."""
    reader = TextReaderService(mock_provider)
    assert reader.state == ReaderState.IDLE

    await reader.pause()

    assert reader.state == ReaderState.IDLE


@pytest.mark.asyncio
async def test_pause_emits_paused_signal(mock_provider):
    """pause() emits the on_reading_paused signal."""
    reader = TextReaderService(mock_provider)
    reader._set_state(ReaderState.READING)
    paused_count: list[int] = []
    reader.on_reading_paused.connect(lambda s, **k: paused_count.append(1))

    await reader.pause()

    assert len(paused_count) == 1


# ---------------------------------------------------------------------------
# Resume sequences
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_transitions_paused_to_reading(mock_provider):
    """resume() transitions the reader from PAUSED back to READING."""
    reader = TextReaderService(mock_provider)
    reader._set_state(ReaderState.READING)
    await reader.pause()
    assert reader.state == ReaderState.PAUSED

    await reader.resume()

    assert reader.state == ReaderState.READING


@pytest.mark.asyncio
async def test_resume_is_noop_when_not_paused(mock_provider):
    """resume() is a no-op when the reader is not in PAUSED state."""
    reader = TextReaderService(mock_provider)
    assert reader.state == ReaderState.IDLE

    await reader.resume()

    assert reader.state == ReaderState.IDLE


@pytest.mark.asyncio
async def test_resume_emits_resumed_signal(mock_provider):
    """resume() emits the on_reading_resumed signal."""
    reader = TextReaderService(mock_provider)
    reader._set_state(ReaderState.READING)
    await reader.pause()
    resumed_count: list[int] = []
    reader.on_reading_resumed.connect(lambda s, **k: resumed_count.append(1))

    await reader.resume()

    assert len(resumed_count) == 1


@pytest.mark.asyncio
async def test_full_pause_resume_cycle_during_synthesis(slow_provider, mock_sd):
    """Pause/resume during active synthesis produces correct state sequence."""
    reader = TextReaderService(slow_provider)

    read_task = asyncio.create_task(reader.read_text("Long text for pausing"))
    # Yield control so the task can start and enter synthesis (0.3 s sleep).
    await asyncio.sleep(0.05)

    assert reader.state == ReaderState.READING

    await reader.pause()
    assert reader.state == ReaderState.PAUSED

    await reader.resume()
    assert reader.state == ReaderState.READING

    await asyncio.wait_for(read_task, timeout=1.0)
    assert reader.state == ReaderState.IDLE


# ---------------------------------------------------------------------------
# Stop sequences
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_transitions_to_stopped(mock_provider):
    """stop() transitions the reader to STOPPED state."""
    reader = TextReaderService(mock_provider)
    reader._set_state(ReaderState.READING)

    await reader.stop()

    assert reader.state == ReaderState.STOPPED


@pytest.mark.asyncio
async def test_stop_clears_text_queue(mock_provider):
    """stop() empties the pending text queue."""
    reader = TextReaderService(mock_provider)
    reader.add_to_queue("Text 1")
    reader.add_to_queue("Text 2")
    reader._set_state(ReaderState.READING)

    await reader.stop()

    assert len(reader._text_queue) == 0


@pytest.mark.asyncio
async def test_stop_emits_stopped_signal(mock_provider):
    """stop() emits the on_reading_stopped signal."""
    reader = TextReaderService(mock_provider)
    reader._set_state(ReaderState.READING)
    stopped_count: list[int] = []
    reader.on_reading_stopped.connect(lambda s, **k: stopped_count.append(1))

    await reader.stop()

    assert len(stopped_count) == 1


@pytest.mark.asyncio
async def test_stop_during_active_synthesis(slow_provider, mock_sd):
    """stop() sets STOPPED state while synthesis is in progress."""
    reader = TextReaderService(slow_provider)

    read_task = asyncio.create_task(reader.read_text("Text to stop"))
    await asyncio.sleep(0.05)

    assert reader.state == ReaderState.READING

    await reader.stop()
    assert reader.state == ReaderState.STOPPED

    with contextlib.suppress(Exception):
        await asyncio.wait_for(read_task, timeout=0.5)


# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------


def test_add_to_queue_appends_in_fifo_order(mock_provider):
    """add_to_queue preserves FIFO ordering."""
    reader = TextReaderService(mock_provider)

    reader.add_to_queue("Text 1")
    reader.add_to_queue("Text 2")
    reader.add_to_queue("Text 3")

    assert len(reader._text_queue) == 3
    assert reader._text_queue[0] == "Text 1"
    assert reader._text_queue[1] == "Text 2"
    assert reader._text_queue[2] == "Text 3"


def test_clear_queue_removes_all_items(mock_provider):
    """clear_queue empties the queue completely."""
    reader = TextReaderService(mock_provider)
    reader.add_to_queue("Text 1")
    reader.add_to_queue("Text 2")

    reader.clear_queue()

    assert len(reader._text_queue) == 0


@pytest.mark.asyncio
async def test_read_queue_processes_all_items_in_order(mock_provider, mock_sd):
    """read_queue reads all queued items in FIFO order and empties the queue."""
    reader = TextReaderService(mock_provider)
    texts_read: list[str | None] = []
    reader.on_reading_started.connect(lambda s, **k: texts_read.append(k.get("text")))

    reader.add_to_queue("First")
    reader.add_to_queue("Second")
    reader.add_to_queue("Third")

    await reader.read_queue()

    assert texts_read == ["First", "Second", "Third"]
    assert len(reader._text_queue) == 0


@pytest.mark.asyncio
async def test_read_queue_on_empty_queue_is_noop(mock_provider):
    """read_queue on an empty queue leaves the reader in IDLE state."""
    reader = TextReaderService(mock_provider)

    await reader.read_queue()

    assert reader.state == ReaderState.IDLE


@pytest.mark.asyncio
async def test_read_queue_halts_when_stop_event_is_set(mock_provider, mock_sd):
    """read_queue skips all items if the stop event is already set."""
    reader = TextReaderService(mock_provider)
    reader.add_to_queue("Text 1")
    reader.add_to_queue("Text 2")

    reader._stop_event.set()
    await reader.read_queue()

    assert len(reader._text_queue) == 2


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_file_splits_paragraphs(mock_provider, mock_sd, tmp_path, mocker):
    """read_file processes each double-newline-separated paragraph separately."""
    mocker.patch("champi_tts.reader.service.asyncio.sleep")
    doc = tmp_path / "test.txt"
    doc.write_text("Paragraph one.\n\nParagraph two.\n\nParagraph three.")
    reader = TextReaderService(mock_provider)
    texts_read: list[str | None] = []
    reader.on_reading_started.connect(lambda s, **k: texts_read.append(k.get("text")))

    await reader.read_file(doc)

    assert texts_read == ["Paragraph one.", "Paragraph two.", "Paragraph three."]


@pytest.mark.asyncio
async def test_read_file_skips_blank_paragraphs(
    mock_provider, mock_sd, tmp_path, mocker
):
    """read_file ignores blank/whitespace-only paragraph separators."""
    mocker.patch("champi_tts.reader.service.asyncio.sleep")
    doc = tmp_path / "sparse.txt"
    doc.write_text("First.\n\n\n\n\nSecond.")
    reader = TextReaderService(mock_provider)
    texts_read: list[str | None] = []
    reader.on_reading_started.connect(lambda s, **k: texts_read.append(k.get("text")))

    await reader.read_file(doc)

    assert texts_read == ["First.", "Second."]


@pytest.mark.asyncio
async def test_read_file_not_found_raises(mock_provider):
    """read_file raises FileNotFoundError for a missing file."""
    reader = TextReaderService(mock_provider)

    with pytest.raises(FileNotFoundError, match="File not found"):
        await reader.read_file("/nonexistent/path/missing.txt")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesis_error_emits_error_signal(mock_provider):
    """A synthesis error emits on_error with the exception message."""
    reader = TextReaderService(mock_provider)
    errors: list[str] = []
    reader.on_error.connect(lambda s, **k: errors.append(k.get("error", "")))

    async def failing_synthesize(text, **kwargs):
        raise RuntimeError("Synthesis failed")

    mock_provider.synthesize = failing_synthesize

    with pytest.raises(RuntimeError, match="Synthesis failed"):
        await reader.read_text("Will fail")

    assert len(errors) == 1
    assert "Synthesis failed" in errors[0]


@pytest.mark.asyncio
async def test_synthesis_error_resets_state_to_idle(mock_provider):
    """Reader returns to IDLE after a synthesis error."""
    reader = TextReaderService(mock_provider)

    async def failing_synthesize(text, **kwargs):
        raise RuntimeError("Synthesis error")

    mock_provider.synthesize = failing_synthesize

    with pytest.raises(RuntimeError):
        await reader.read_text("Will fail")

    assert reader.state == ReaderState.IDLE


@pytest.mark.asyncio
async def test_no_error_signal_on_successful_read(mock_provider, mock_sd):
    """Successful read does not emit the on_error signal."""
    reader = TextReaderService(mock_provider)
    errors: list[str] = []
    reader.on_error.connect(lambda s, **k: errors.append(k.get("error", "")))

    await reader.read_text("Success")

    assert len(errors) == 0


# ---------------------------------------------------------------------------
# Context manager and lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager_initializes_on_entry(mock_provider):
    """Async context manager initializes the provider on entry."""
    reader = TextReaderService(mock_provider)

    async with reader:
        assert reader.provider._initialized


@pytest.mark.asyncio
async def test_context_manager_cleans_up_on_exit(mock_provider):
    """Async context manager shuts down the provider on exit."""
    reader = TextReaderService(mock_provider)

    async with reader:
        pass

    assert not reader.provider._initialized


@pytest.mark.asyncio
async def test_initialize_is_idempotent(mock_provider):
    """Calling initialize twice does not raise and leaves provider initialized."""
    reader = TextReaderService(mock_provider)

    await reader.initialize()
    await reader.initialize()

    assert reader.provider._initialized
    await reader.cleanup()


@pytest.mark.asyncio
async def test_cleanup_shuts_down_provider(mock_provider, mock_sd):
    """cleanup() shuts down the provider."""
    reader = TextReaderService(mock_provider)
    await reader.initialize()

    await reader.cleanup()

    assert not reader.provider._initialized


# ---------------------------------------------------------------------------
# Voice parameter forwarding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_text_forwards_voice_to_synthesize(mock_provider, mock_sd):
    """read_text passes the voice keyword argument to provider.synthesize."""
    reader = TextReaderService(mock_provider)
    voices_used: list[str | None] = []
    original = mock_provider.synthesize

    async def tracked(text, voice=None, **kwargs):
        voices_used.append(voice)
        return await original(text, voice=voice, **kwargs)

    mock_provider.synthesize = tracked

    await reader.read_text("Test", voice="af_bella")

    assert voices_used[-1] == "af_bella"


@pytest.mark.asyncio
async def test_read_text_without_voice_passes_none(mock_provider, mock_sd):
    """read_text without explicit voice passes None to provider.synthesize."""
    reader = TextReaderService(mock_provider)
    voices_used: list[str | None] = []
    original = mock_provider.synthesize

    async def tracked(text, voice=None, **kwargs):
        voices_used.append(voice)
        return await original(text, voice=voice, **kwargs)

    mock_provider.synthesize = tracked

    await reader.read_text("Test")

    assert voices_used[-1] is None


# ---------------------------------------------------------------------------
# Long-document performance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_long_document_reads_all_paragraphs(mock_provider, mock_sd):
    """Reader handles many sequential read_text calls without errors."""
    reader = TextReaderService(mock_provider)
    completed: list[int] = []
    reader.on_reading_completed.connect(lambda s, **k: completed.append(1))

    for i in range(10):
        await reader.read_text(f"Paragraph {i}")

    assert len(completed) == 10
    assert reader.state == ReaderState.IDLE
