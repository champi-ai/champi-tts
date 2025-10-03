"""
Visual TTS state indicator using ImGui/GLFW.
"""

import sys
import time
from enum import Enum

from imgui_bundle import imgui, immapp
from loguru import logger


class TTSState(Enum):
    """TTS visual states."""

    IDLE = "idle"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    PAUSED = "paused"
    ERROR = "error"


class TTSIndicatorUI:
    """Visual indicator for TTS states."""

    # State colors (RGBA)
    COLORS = {
        TTSState.IDLE: (0.5, 0.5, 0.5, 1.0),  # Gray
        TTSState.PROCESSING: (1.0, 0.8, 0.0, 1.0),  # Yellow
        TTSState.SPEAKING: (0.0, 0.8, 0.0, 1.0),  # Green
        TTSState.PAUSED: (0.0, 0.5, 1.0, 1.0),  # Blue
        TTSState.ERROR: (1.0, 0.0, 0.0, 1.0),  # Red
    }

    # State labels
    LABELS = {
        TTSState.IDLE: "Idle",
        TTSState.PROCESSING: "Processing",
        TTSState.SPEAKING: "Speaking",
        TTSState.PAUSED: "Paused",
        TTSState.ERROR: "Error",
    }

    def __init__(self, window_x: int = 50, window_y: int = 50):
        self.window_x = window_x
        self.window_y = window_y
        self.current_state = TTSState.IDLE
        self.last_text = ""
        self.pulse_time = 0.0

    def update_state(self, state: TTSState, text: str = "") -> None:
        """Update the current TTS state."""
        self.current_state = state
        self.last_text = text
        logger.debug(f"UI state updated: {state.value}")

    def gui(self) -> None:
        """Render the ImGui interface."""
        # Set window position
        imgui.set_next_window_pos(
            imgui.ImVec2(self.window_x, self.window_y),
            imgui.Cond_.first_use_ever,
        )

        # Set window size
        imgui.set_next_window_size(
            imgui.ImVec2(250, 150),
            imgui.Cond_.first_use_ever,
        )

        # Window flags
        flags = (
            imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.always_auto_resize
        )

        # Main window
        imgui.begin("TTS Status", None, flags)

        # Get color and label
        color = self.COLORS[self.current_state]
        label = self.LABELS[self.current_state]

        # Pulsing effect for speaking state
        if self.current_state == TTSState.SPEAKING:
            self.pulse_time += 0.05
            pulse = (1.0 + 0.3 * abs(time.time() % 1.0 - 0.5)) / 1.15
            color_tuple: tuple[float, float, float, float] = (
                color[0] * pulse,
                color[1] * pulse,
                color[2] * pulse,
                color[3],
            )
            color = color_tuple

        # Status indicator circle
        draw_list = imgui.get_window_draw_list()
        center = imgui.get_cursor_screen_pos()
        center.x += 20
        center.y += 20

        # Draw circle
        draw_list.add_circle_filled(
            center,
            15.0,
            imgui.get_color_u32(imgui.ImVec4(*color)),
        )

        # Draw border
        draw_list.add_circle(
            center,
            15.0,
            imgui.get_color_u32(imgui.ImVec4(1.0, 1.0, 1.0, 0.5)),
            thickness=2.0,
        )

        # State label
        imgui.same_line()
        imgui.set_cursor_pos_x(50)
        imgui.text(f"Status: {label}")

        # Show last text (truncated)
        if self.last_text:
            imgui.separator()
            truncated = (
                self.last_text[:50] + "..."
                if len(self.last_text) > 50
                else self.last_text
            )
            imgui.text_wrapped(f"Text: {truncated}")

        imgui.end()

    def run(self) -> None:
        """Run the UI main loop."""
        immapp.run(
            gui_function=self.gui,
            window_title="Champi TTS",
            window_size=(300, 200),
            fps_idle=10,
        )


def run_standalone(window_x: int = 50, window_y: int = 50) -> None:
    """Run the indicator UI in standalone mode for testing."""
    ui = TTSIndicatorUI(window_x=window_x, window_y=window_y)

    # Test state cycling
    states = [
        (TTSState.IDLE, "Waiting for text"),
        (TTSState.PROCESSING, "Processing text..."),
        (TTSState.SPEAKING, "Hello, this is a test of the TTS system"),
        (TTSState.PAUSED, "Reading paused"),
        (TTSState.ERROR, "An error occurred"),
    ]

    state_index = 0

    def test_gui():
        nonlocal state_index

        # Cycle states every 3 seconds
        if int(time.time()) % 3 == 0 and imgui.get_frame_count() % 180 == 0:
            state, text = states[state_index]
            ui.update_state(state, text)
            state_index = (state_index + 1) % len(states)

        ui.gui()

    immapp.run(
        gui_function=test_gui,
        window_title="Champi TTS - Test Mode",
        window_size=(300, 200),
        fps_idle=10,
    )


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Run standalone
    run_standalone()
