"""
Input guards and speech concurrency policy for the champi-tts MCP tools.

MCP tool arguments come from an LLM, so file destinations are validated
before anything is written and overlapping speech requests are serialised
instead of playing over each other.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

OUTPUT_DIR_ENV = "CHAMPI_TTS_OUTPUT_DIR"

SUPPORTED_AUDIO_FORMATS: dict[str, str] = {
    ".wav": "wav",
    ".flac": "flac",
    ".ogg": "ogg",
}

SPEECH_POLICIES = ("queue", "interrupt")
DEFAULT_SPEECH_POLICY = "queue"


def resolve_output_path(output_path: str, overwrite: bool = False) -> tuple[Path, str]:
    """Validate an output destination and return its resolved path and format.

    When ``CHAMPI_TTS_OUTPUT_DIR`` is set, relative paths are placed under it
    and any path resolving outside it (including via ``..`` or symlinks) is
    rejected.

    Args:
        output_path: Requested destination path.
        overwrite: Allow replacing an existing file.

    Returns:
        Tuple of the resolved absolute path and the ``save_audio`` format name.

    Raises:
        ValueError: Unsupported extension, or the path escapes the output root.
        FileExistsError: The file exists and ``overwrite`` is false.
        IsADirectoryError: The path points at a directory.
    """
    path = Path(output_path).expanduser()
    fmt = SUPPORTED_AUDIO_FORMATS.get(path.suffix.lower())
    if fmt is None:
        supported = ", ".join(SUPPORTED_AUDIO_FORMATS)
        raise ValueError(
            f"unsupported output extension {path.suffix!r}; use one of {supported}"
        )

    root_env = os.environ.get(OUTPUT_DIR_ENV)
    if root_env:
        root = Path(root_env).expanduser().resolve()
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"output path must be inside {OUTPUT_DIR_ENV} ({root})")
    else:
        resolved = path.resolve()

    if resolved.is_dir():
        raise IsADirectoryError(f"output path is a directory: {resolved}")
    if resolved.exists() and not overwrite:
        raise FileExistsError(
            f"{resolved} already exists; pass overwrite=True to replace it"
        )
    return resolved, fmt


class SpeechTurn:
    """A single granted slot to play speech; may be pre-empted by a newer turn."""

    def __init__(self) -> None:
        self._preempted = False

    @property
    def preempted(self) -> bool:
        """True once a later ``interrupt`` request has taken over this turn."""
        return self._preempted

    def preempt(self) -> None:
        """Mark the turn as superseded so it skips any playback not yet started."""
        self._preempted = True


class SpeechGate:
    """Serialises speech playback so concurrent requests never overlap.

    ``queue`` waits for the active speech to finish. ``interrupt`` pre-empts
    the active and already queued turns, calls ``interrupt`` to stop audio in
    flight, then takes the next slot.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._turns: list[SpeechTurn] = []

    @property
    def busy(self) -> bool:
        """True while a turn is active."""
        return self._lock.locked()

    @asynccontextmanager
    async def turn(
        self,
        policy: str = DEFAULT_SPEECH_POLICY,
        interrupt: Callable[[], Awaitable[None]] | None = None,
    ) -> AsyncIterator[SpeechTurn]:
        """Wait for exclusive use of the speaker according to ``policy``.

        Args:
            policy: ``"queue"`` or ``"interrupt"``.
            interrupt: Coroutine function that stops audio currently playing;
                only called under the ``interrupt`` policy when the gate is busy.

        Yields:
            The granted turn; check ``preempted`` before starting playback.

        Raises:
            ValueError: Unknown policy.
        """
        if policy not in SPEECH_POLICIES:
            raise ValueError(
                f"unknown speech policy {policy!r}; use one of {', '.join(SPEECH_POLICIES)}"
            )

        current = SpeechTurn()
        if policy == "interrupt":
            for pending in self._turns:
                pending.preempt()
            if self.busy and interrupt is not None:
                await interrupt()
        self._turns.append(current)
        try:
            async with self._lock:
                yield current
        finally:
            self._turns.remove(current)
