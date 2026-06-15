"""
Integration tests for the TTS visual indicator UI.

Covers UI state management, label/color mappings, and the integration between
TextReaderService signals and UI state transitions.
"""

import pytest

from champi_tts import ReaderState, TextReaderService, TTSIndicatorUI, TTSState

# ---------------------------------------------------------------------------
# TTSIndicatorUI — initial state
# ---------------------------------------------------------------------------


@pytest.fixture
def ui():
    """TTSIndicatorUI instance for unit-level UI tests."""
    return TTSIndicatorUI(window_x=100, window_y=100)


def test_ui_starts_in_idle_state(ui):
    """UI indicator starts in IDLE state."""
    assert ui.current_state == TTSState.IDLE


def test_ui_starts_with_empty_last_text(ui):
    """UI indicator starts with an empty last_text string."""
    assert ui.last_text == ""


def test_ui_initial_pulse_time_is_zero(ui):
    """UI indicator initialises pulse_time to 0.0."""
    assert ui.pulse_time == 0.0


# ---------------------------------------------------------------------------
# TTSIndicatorUI — update_state
# ---------------------------------------------------------------------------


def test_ui_update_state_changes_current_state(ui):
    """update_state sets current_state to the new state."""
    ui.update_state(TTSState.SPEAKING, "Hello")
    assert ui.current_state == TTSState.SPEAKING


def test_ui_update_state_stores_provided_text(ui):
    """update_state stores the provided text in last_text."""
    ui.update_state(TTSState.SPEAKING, "Test text")
    assert ui.last_text == "Test text"


def test_ui_update_state_default_text_is_empty(ui):
    """update_state with no text argument leaves last_text as empty string."""
    ui.update_state(TTSState.PROCESSING)
    assert ui.last_text == ""


def test_ui_update_state_accepts_all_states(ui):
    """update_state can be called with every defined TTSState."""
    for state in [
        TTSState.IDLE,
        TTSState.PROCESSING,
        TTSState.SPEAKING,
        TTSState.PAUSED,
        TTSState.ERROR,
    ]:
        ui.update_state(state)
        assert ui.current_state == state


def test_ui_stores_full_text_without_truncation(ui):
    """update_state stores the full text; display-level truncation happens in gui()."""
    long_text = "X" * 200
    ui.update_state(TTSState.SPEAKING, long_text)
    assert len(ui.last_text) == 200


# ---------------------------------------------------------------------------
# TTSIndicatorUI — color and label mappings
# ---------------------------------------------------------------------------


def test_ui_has_color_for_every_state():
    """COLORS contains an entry for every TTSState."""
    ui = TTSIndicatorUI()
    for state in TTSState:
        assert state in ui.COLORS, f"Missing color for {state}"


def test_ui_has_label_for_every_state():
    """LABELS contains an entry for every TTSState."""
    ui = TTSIndicatorUI()
    for state in TTSState:
        assert state in ui.LABELS, f"Missing label for {state}"


def test_ui_idle_label_is_idle(ui):
    """IDLE state label is 'Idle'."""
    assert ui.LABELS[TTSState.IDLE] == "Idle"


def test_ui_speaking_label_is_speaking(ui):
    """SPEAKING state label is 'Speaking'."""
    assert ui.LABELS[TTSState.SPEAKING] == "Speaking"


def test_ui_paused_label_is_paused(ui):
    """PAUSED state label is 'Paused'."""
    assert ui.LABELS[TTSState.PAUSED] == "Paused"


def test_ui_error_label_is_error(ui):
    """ERROR state label is 'Error'."""
    assert ui.LABELS[TTSState.ERROR] == "Error"


def test_ui_processing_label_is_processing(ui):
    """PROCESSING state label is 'Processing'."""
    assert ui.LABELS[TTSState.PROCESSING] == "Processing"


def test_ui_colors_are_four_component_tuples(ui):
    """Every color value is an RGBA four-tuple of floats."""
    for state, color in ui.COLORS.items():
        assert len(color) == 4, f"Color for {state} should be (R, G, B, A)"


# ---------------------------------------------------------------------------
# Reader — show_ui=False
# ---------------------------------------------------------------------------


def test_reader_without_ui_has_no_ui_instance(mock_provider):
    """TextReaderService with show_ui=False sets _ui to None."""
    reader = TextReaderService(mock_provider, show_ui=False)
    assert reader._ui is None


# ---------------------------------------------------------------------------
# Reader — show_ui=True (UI integration)
# ---------------------------------------------------------------------------


def test_reader_with_ui_creates_indicator_instance(mock_provider):
    """TextReaderService with show_ui=True initialises a TTSIndicatorUI."""
    reader = TextReaderService(mock_provider, show_ui=True)
    assert reader._ui is not None
    assert isinstance(reader._ui, TTSIndicatorUI)


def test_reader_ui_starts_in_idle_state(mock_provider):
    """UI is in IDLE state when the reader is first created."""
    reader = TextReaderService(mock_provider, show_ui=True)
    assert reader._ui.current_state == TTSState.IDLE


@pytest.mark.asyncio
async def test_reader_pause_updates_ui_to_paused(mock_provider):
    """Pausing the reader drives the UI to PAUSED state."""
    reader = TextReaderService(mock_provider, show_ui=True)
    reader._set_state(ReaderState.READING)

    await reader.pause()

    assert reader._ui.current_state == TTSState.PAUSED


@pytest.mark.asyncio
async def test_reader_resume_updates_ui_to_speaking(mock_provider):
    """Resuming the reader drives the UI to SPEAKING state."""
    reader = TextReaderService(mock_provider, show_ui=True)
    reader._set_state(ReaderState.READING)
    await reader.pause()

    await reader.resume()

    assert reader._ui.current_state == TTSState.SPEAKING


@pytest.mark.asyncio
async def test_reader_stop_updates_ui_to_idle(mock_provider):
    """Stopping the reader drives the UI to IDLE state."""
    reader = TextReaderService(mock_provider, show_ui=True)
    reader._set_state(ReaderState.READING)

    await reader.stop()

    assert reader._ui.current_state == TTSState.IDLE


@pytest.mark.asyncio
async def test_reader_error_updates_ui_to_error_state(mock_provider):
    """A synthesis error drives the UI to ERROR state."""
    reader = TextReaderService(mock_provider, show_ui=True)

    async def failing_synthesize(text, **kwargs):
        raise RuntimeError("TTS error")

    mock_provider.synthesize = failing_synthesize

    with pytest.raises(RuntimeError):
        await reader.read_text("Will fail")

    assert reader._ui.current_state == TTSState.ERROR


@pytest.mark.asyncio
async def test_reader_completed_read_updates_ui_to_idle(mock_provider, mock_sd):
    """After a successful read completes, the UI returns to IDLE state."""
    reader = TextReaderService(mock_provider, show_ui=True)

    await reader.read_text("Hello UI")

    assert reader._ui.current_state == TTSState.IDLE


@pytest.mark.asyncio
async def test_reader_reading_started_signal_sets_ui_speaking(mock_provider):
    """on_reading_started signal drives the UI to SPEAKING with the reading text."""
    reader = TextReaderService(mock_provider, show_ui=True)

    reader.on_reading_started.send(reader, text="Captured text")

    assert reader._ui.current_state == TTSState.SPEAKING
    assert reader._ui.last_text == "Captured text"
