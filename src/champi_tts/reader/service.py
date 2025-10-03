"""
Text reading service with interruption support.
"""

import asyncio
from enum import Enum
from pathlib import Path
from typing import Any

from blinker import Signal
from loguru import logger

from champi_tts.core.audio import AudioPlayer
from champi_tts.core.base_provider import BaseTTSProvider


class ReaderState(Enum):
    """Reader service states."""

    IDLE = "idle"
    READING = "reading"
    PAUSED = "paused"
    STOPPED = "stopped"


class TextReaderService:
    """
    Text reading service with interruption and pause/resume support.
    """

    def __init__(self, provider: BaseTTSProvider, show_ui: bool = False):
        self.provider = provider
        self.show_ui = show_ui
        self.player = AudioPlayer(sample_rate=provider.config.sample_rate)
        self._initialized = False

        # State
        self._state = ReaderState.IDLE
        self._current_text: str = ""
        self._text_queue: list[str] = []
        self._pause_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._pause_event.set()  # Start unpaused

        # Signals
        self.on_reading_started = Signal()
        self.on_reading_paused = Signal()
        self.on_reading_resumed = Signal()
        self.on_reading_stopped = Signal()
        self.on_reading_completed = Signal()
        self.on_state_changed = Signal()
        self.on_error = Signal()

        # UI integration
        self._ui = None
        if show_ui:
            self._setup_ui()

    @property
    def state(self) -> ReaderState:
        """Get current reader state."""
        return self._state

    def _set_state(self, new_state: ReaderState) -> None:
        """Set reader state and emit signal."""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            self.on_state_changed.send(
                self,
                old_state=old_state.value,
                new_state=new_state.value,
            )
            logger.debug(f"Reader state: {old_state.value} -> {new_state.value}")

    async def read_text(self, text: str, voice: str | None = None) -> None:
        """
        Read a single text string.

        Args:
            text: Text to read
            voice: Voice to use (optional)
        """
        try:
            self._set_state(ReaderState.READING)
            self._current_text = text
            self._stop_event.clear()
            self.on_reading_started.send(self, text=text)

            # Wait if paused
            await self._pause_event.wait()

            # Check if stopped
            if self._stop_event.is_set():
                self._set_state(ReaderState.STOPPED)
                return

            # Synthesize
            audio = await self.provider.synthesize(text, voice=voice)

            # Check again if stopped during synthesis
            if self._stop_event.is_set():
                self._set_state(ReaderState.STOPPED)
                return

            # Play audio
            await self.player.play(audio, blocking=True)

            self._set_state(ReaderState.IDLE)
            self.on_reading_completed.send(self, text=text)

        except Exception as e:
            logger.error(f"Error reading text: {e}")
            self._set_state(ReaderState.IDLE)
            self.on_error.send(self, error=str(e))
            raise

    async def read_file(
        self,
        file_path: str | Path,
        voice: str | None = None,
    ) -> None:
        """
        Read text from file with paragraph-by-paragraph processing.

        Args:
            file_path: Path to text file
            voice: Voice to use (optional)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for paragraph in paragraphs:
            if self._stop_event.is_set():
                break

            await self.read_text(paragraph, voice=voice)

            # Small pause between paragraphs
            if not self._stop_event.is_set():
                await asyncio.sleep(0.5)

    async def read_queue(self, voice: str | None = None) -> None:
        """
        Read all texts in the queue.

        Args:
            voice: Voice to use (optional)
        """
        while self._text_queue and not self._stop_event.is_set():
            text = self._text_queue.pop(0)
            await self.read_text(text, voice=voice)

    def add_to_queue(self, text: str) -> None:
        """Add text to reading queue."""
        self._text_queue.append(text)
        logger.debug(
            f"Added to queue: {text[:50]}... (queue size: {len(self._text_queue)})"
        )

    def clear_queue(self) -> None:
        """Clear reading queue."""
        self._text_queue.clear()
        logger.debug("Queue cleared")

    async def pause(self) -> None:
        """Pause reading."""
        if self._state == ReaderState.READING:
            self._pause_event.clear()
            self._set_state(ReaderState.PAUSED)
            self.player.stop()
            self.on_reading_paused.send(self)
            logger.info("Reading paused")

    async def resume(self) -> None:
        """Resume reading."""
        if self._state == ReaderState.PAUSED:
            self._pause_event.set()
            self._set_state(ReaderState.READING)
            self.on_reading_resumed.send(self)
            logger.info("Reading resumed")

    async def stop(self) -> None:
        """Stop reading."""
        self._stop_event.set()
        self._pause_event.set()  # Unblock if paused
        self.player.stop()
        self._set_state(ReaderState.STOPPED)
        self.clear_queue()
        self.on_reading_stopped.send(self)
        logger.info("Reading stopped")

    async def interrupt(self) -> None:
        """Interrupt current reading immediately."""
        await self.provider.interrupt()
        self.player.stop()
        logger.info("Reading interrupted")

    def _setup_ui(self) -> None:
        """Setup UI indicator and connect signals."""
        try:
            from champi_tts.ui import (  # type: ignore[attr-defined]
                TTSIndicatorUI,
                TTSState,
            )

            self._ui = TTSIndicatorUI()  # type: ignore[assignment]

            # Connect reader signals to UI state updates
            self.on_reading_started.connect(
                lambda sender, **kw: self._update_ui_state(
                    TTSState.SPEAKING, kw.get("text", "")
                )
            )
            self.on_reading_paused.connect(
                lambda sender, **kw: self._update_ui_state(TTSState.PAUSED)
            )
            self.on_reading_resumed.connect(
                lambda sender, **kw: self._update_ui_state(TTSState.SPEAKING)
            )
            self.on_reading_stopped.connect(
                lambda sender, **kw: self._update_ui_state(TTSState.IDLE)
            )
            self.on_reading_completed.connect(
                lambda sender, **kw: self._update_ui_state(TTSState.IDLE)
            )
            self.on_error.connect(
                lambda sender, **kw: self._update_ui_state(
                    TTSState.ERROR, kw.get("error", "")
                )
            )

            logger.info("UI indicator initialized and connected")

        except ImportError:
            logger.warning("UI dependencies not available, UI will not be shown")
            self._ui = None

    def _update_ui_state(self, state: Any, text: str = "") -> None:  # type: ignore[misc]
        """Update UI state if UI is enabled."""
        if self._ui:
            self._ui.update_state(state, text)

    async def initialize(self) -> None:
        """Initialize the reader service."""
        if not self._initialized:
            await self.provider.initialize()
            self._initialized = True
            logger.info("Reader service initialized")

    async def cleanup(self) -> None:
        """Cleanup reader resources."""
        if self._initialized:
            await self.stop()
            self.player.stop()
            if self._ui:
                self._ui.close()
            await self.provider.shutdown()
            self._initialized = False
            logger.info("Reader service cleaned up")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
        return False
