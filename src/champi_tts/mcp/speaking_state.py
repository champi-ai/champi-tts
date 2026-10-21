"""
Speaking state tracking and MCP progress notifications for champi-tts.

The MCP server keeps one :class:`SpeakingStateTracker` per server instance.
Tools that synthesize or play audio move it through
``idle → synthesizing → speaking → idle`` and, when an MCP ``Context`` is
available, mirror each transition to the client as a progress notification
(``notifications/progress``) and a log message (``notifications/message``).

Every transition is also published in-process on the
:data:`speaking_state_changed` blinker signal so that co-located consumers
(champi-stt echo gating, champi-imgui speaking indicator) can subscribe
without polling ``get_speaking_state``.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import Any

from blinker import Signal

__all__ = [
    "SpeakingPhase",
    "SpeakingStateTracker",
    "speaking_state_changed",
]

speaking_state_changed = Signal("champi-tts.speaking_state_changed")
"""Sent with ``sender=<tracker>`` and ``state=<snapshot dict>`` on every change."""


class SpeakingPhase(StrEnum):
    """Externally visible speech phase."""

    IDLE = "idle"
    SYNTHESIZING = "synthesizing"
    SPEAKING = "speaking"


_PROGRESS_TOTAL = 3.0
_PROGRESS: dict[SpeakingPhase, float] = {
    SpeakingPhase.SYNTHESIZING: 1.0,
    SpeakingPhase.SPEAKING: 2.0,
    SpeakingPhase.IDLE: 3.0,
}


class SpeakingStateTracker:
    """Track what the server is currently saying and notify listeners."""

    def __init__(self) -> None:
        self.phase = SpeakingPhase.IDLE
        self.text: str | None = None
        self.voice: str | None = None
        self.updated_at = time.time()

    def snapshot(self) -> dict[str, Any]:
        """Return the current state as a JSON-serialisable dict."""
        return {
            "state": self.phase.value,
            "text": self.text,
            "voice": self.voice,
            "updated_at": self.updated_at,
        }

    async def set_phase(
        self,
        phase: SpeakingPhase,
        ctx: Any = None,
        *,
        text: str | None = None,
        voice: str | None = None,
    ) -> None:
        """Move to ``phase``, publish the change and notify the MCP client.

        Args:
            phase: New phase.
            ctx: Optional FastMCP ``Context`` of the calling tool.
            text: Text being spoken; kept from the previous phase when ``None``
                and cleared on ``idle``.
            voice: Voice in use; same retention rules as ``text``.
        """
        self.phase = phase
        if phase is SpeakingPhase.IDLE:
            self.text = None
            self.voice = None
        else:
            self.text = text if text is not None else self.text
            self.voice = voice if voice is not None else self.voice
        self.updated_at = time.time()

        state = self.snapshot()
        speaking_state_changed.send(self, state=state)
        if ctx is not None:
            await _notify(ctx, phase)

    @asynccontextmanager
    async def session(
        self, text: str, voice: str | None = None, ctx: Any = None
    ) -> AsyncIterator[SpeakingStateTracker]:
        """Enter ``synthesizing`` for ``text`` and always return to ``idle``.

        Callers advance to ``speaking`` themselves with
        ``await tracker.set_phase(SpeakingPhase.SPEAKING, ctx)`` right before
        playback starts.
        """
        await self.set_phase(SpeakingPhase.SYNTHESIZING, ctx, text=text, voice=voice)
        try:
            yield self
        finally:
            await self.set_phase(SpeakingPhase.IDLE, ctx)


async def _notify(ctx: Any, phase: SpeakingPhase) -> None:
    # Notifications are best-effort: a client without a progress token, or a
    # call made outside a live MCP request, must not fail the tool.
    with contextlib.suppress(Exception):
        await ctx.report_progress(
            _PROGRESS[phase], _PROGRESS_TOTAL, f"champi-tts: {phase.value}"
        )
    with contextlib.suppress(Exception):
        await ctx.info(f"champi-tts: {phase.value}")
