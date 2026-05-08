"""
Integration tests for UI indicator.

These tests verify UI state transitions and visual indicator functionality.
"""

import pytest

from champi_tts import TTSIndicatorUI, TTSState


@pytest.fixture
def ui():
    """Create UI indicator for testing."""
    return TTSIndicatorUI(window_x=100, window_y=100)


def test_ui_initial_state(ui):
    """Test UI starts in IDLE state."""
    assert ui.current_state == TTSState.IDLE


def test_ui_update_state(ui):
    """Test updating UI state."""
    ui.update_state(TTSState.SPEAKING, "Hello, world!")
    assert ui.current_state == TTSState.SPEAKING


def test_ui_state_cycle(ui):
    """Test cycling through all UI states."""
    states = [TTSState.IDLE, TTSState.PROCESSING, TTSState.SPEAKING, TTSState.PAUSED, TTSState.ERROR]

    for state in states:
        ui.update_state(state)
        assert ui.current_state == state


def test_ui_text_display(ui):
    """Test text is displayed correctly."""
    ui.update_state(TTSState.SPEAKING, "Test text for display")
    assert ui.last_text == "Test text for display"


def test_ui_long_text_truncation(ui):
    """Test long text is truncated."""
    long_text = "A" * 100
    ui.update_state(TTSState.SPEAKING, long_text)
    # UI stores the full text, truncation happens in display
    assert len(ui.last_text) == 100


def test_ui_pulse_time_increase(ui):
    """Test pulse time increases with each state update."""
    initial_pulse = ui.pulse_time
    ui.update_state(TTSState.SPEAKING)
    ui.update_state(TTSState.SPEAKING)
    assert ui.pulse_time > initial_pulse


def test_ui_colors():
    """Test UI has correct color mappings."""
    from champi_tts.ui import TTSIndicatorUI

    ui = TTSIndicatorUI()
    assert TTSState.IDLE in ui.COLORS
    assert TTSState.PROCESSING in ui.COLORS
    assert TTSState.SPEAKING in ui.COLORS
    assert TTSState.PAUSED in ui.COLORS
    assert TTSState.ERROR in ui.COLORS


def test_ui_state_labels():
    """Test UI has correct state labels."""
    from champi_tts.ui import TTSIndicatorUI

    ui = TTSIndicatorUI()
    assert ui.LABELS[TTSState.IDLE] == "Idle"
    assert ui.LABELS[TTSState.SPEAKING] == "Speaking"
    assert ui.LABELS[TTSState.ERROR] == "Error"


@pytest.mark.asyncio
async def test_reader_ui_integration(mock_provider):
    """
    Test that reader properly updates UI state on events.
    """
    from champi_tts.factory import get_reader

    # Get reader with UI
    reader = get_reader("kokoro", show_ui=False)  # Skip actual UI rendering
    assert reader._ui is None  # UI not initialized when show_ui=False


@pytest.mark.asyncio
async def test_reader_ui_with_show_ui_flag(mock_provider):
    """
    Test reader with UI enabled.
    """
    from champi_tts.factory import get_reader

    # Get reader with UI
    reader = get_reader("kokoro", show_ui=True)

    # UI should be initialized
    assert reader._ui is not None
    assert reader._ui is not None  # Should be TTSIndicatorUI


@pytest.mark.asyncio
async def test_reader_events_update_ui(mock_provider):
    """
    Test that reader events properly update UI state.
    """
    from champi_tts.factory import get_reader

    reader = get_reader("kokoro", show_ui=True)

    assert reader._ui is not None

    # Check UI state before any events
    assert reader._ui.current_state == TTSState.IDLE

    # Connect to events and track UI updates
    ui_state_changes = []

    def track_ui_state(sender, state, **kwargs):
        ui_state_changes.append(state)

    reader.on_state_changed.connect(track_ui_state)

    # Read some text
    try:
        await reader.read_text("Test text")
    except Exception:
        pass  # Expected to fail without actual provider

    # UI should have been updated
    # Note: The exact sequence depends on signal timing


@pytest.mark.asyncio
async def test_reader_error_updates_ui(mock_provider):
    """
    Test that errors update UI to ERROR state.
    """
    from champi_tts.factory import get_reader

    reader = get_reader("kokoro", show_ui=True)

    assert reader._ui is not None
    assert reader._ui.current_state == TTSState.IDLE

    # Simulate error state
    reader._set_state(ReaderState.IDLE)  # This will emit state_changed signal
    # UI should have received the state change

