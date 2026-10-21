"""
MCP server for champi-tts.

Exposes champi-tts synthesis capabilities over the Model Context Protocol
so that MCP-aware hosts (Claude Desktop, Cursor, etc.) can call TTS tools.

Supported transports
--------------------
* ``stdio`` (default) — JSON-RPC over stdin/stdout; stdout MUST stay clean.
* ``sse`` — Server-Sent Events over HTTP for remote/mobile clients.

Provider lifecycle
------------------
The provider is initialised lazily on the first synthesis call and shut down
cleanly via the FastMCP lifespan context manager — the same pattern used
by champi-stt. No atexit hacks, no asyncio.run() in a signal handler.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from contextlib import asynccontextmanager
from importlib.metadata import version as _pkg_version
from typing import Any

from loguru import logger as _logger

try:
    from mcp.server.fastmcp import Context
    from mcp.server.fastmcp import FastMCP as _FastMCP

    MCP_AVAILABLE = True
except ImportError:
    Context = Any  # type: ignore[assignment,misc]
    _FastMCP = None  # type: ignore[assignment,misc]
    MCP_AVAILABLE = False

_DEFAULT_PROVIDER = "kokoro"
_DEFAULT_SSE_PORT = 8766


def _require_mcp() -> None:
    """Raise ``ImportError`` with an install hint when the mcp extra is missing."""
    if not MCP_AVAILABLE:
        _v = _pkg_version("champi-tts")
        _whl = f"champi_tts-{_v}-py3-none-any.whl"
        _url = f"https://github.com/champi-ai/champi-tts/releases/download/v{_v}/{_whl}"
        raise ImportError(
            "mcp is required for the MCP server. "
            f'Install with: uvx --from "champi-tts[mcp] @ {_url}" champi-tts mcp serve'
        )


def _effective_provider(provider: str | None) -> str:
    return provider or os.environ.get("CHAMPI_TTS_PROVIDER") or _DEFAULT_PROVIDER


def create_mcp_server() -> Any:
    """Build and return a configured FastMCP server instance.

    All tool functions are registered here as closures over a shared
    mutable state dict, matching the champi-stt factory pattern.
    The lifespan context manager handles provider shutdown on exit.

    Returns:
        Configured FastMCP application ready for ``mcp.run()``.
    """
    import champi_tts
    from champi_tts.core.audio import save_audio
    from champi_tts.mcp.guards import SpeechGate, resolve_output_path
    from champi_tts.mcp.speaking_state import SpeakingStateTracker

    # Shared mutable state — avoids module-level globals. ``speech`` serialises
    # playback for the speaker tools so concurrent calls never talk over each other.
    # ``speaking`` tracks idle/synthesizing/speaking and notifies clients.
    state: dict[str, Any] = {
        "provider": None,
        "lock": None,
        "speech": SpeechGate(),
        "speaking": SpeakingStateTracker(),
    }

    async def _get_provider(provider_name: str) -> Any:
        if state["lock"] is None:
            state["lock"] = asyncio.Lock()
        async with state["lock"]:
            if state["provider"] is None:
                _logger.info(
                    "[champi-tts] Initializing provider '{}'... (first call may be slow)",
                    provider_name,
                )
                p = champi_tts.get_provider(provider_name)  # type: ignore[arg-type]
                await p.initialize()
                state["provider"] = p
        return state["provider"]

    @asynccontextmanager
    async def _lifespan(app: Any) -> Any:
        try:
            yield
        finally:
            if state["provider"] is not None:
                with contextlib.suppress(Exception):
                    await state["provider"].shutdown()
                state["provider"] = None

    mcp = _FastMCP("champi-tts", lifespan=_lifespan)

    # Expose state for test injection (mirrors champi-stt pattern)
    mcp._state = state  # type: ignore[attr-defined]

    @mcp.tool()
    def list_providers() -> list[str]:
        """Return the names of all available TTS providers."""
        return champi_tts.list_providers()

    @mcp.tool()
    async def get_provider_status(provider: str) -> dict[str, Any]:
        """Return health/status information for a named provider.

        Args:
            provider: Provider key (e.g. ``"kokoro"``).
        """
        try:
            p = champi_tts.get_provider(provider)  # type: ignore[arg-type]
            return await p.get_model_info()
        except Exception as exc:
            return {
                "error": True,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "provider": provider,
            }

    @mcp.tool()
    async def synthesize_to_file(
        text: str,
        output_path: str,
        voice: str | None = None,
        speed: float | None = None,
        provider: str | None = None,
        overwrite: bool = False,
        ctx: Context | None = None,
    ) -> str:
        """Synthesize text to an audio file and return the written file path.

        Sends progress notifications while synthesizing; the speaking state
        is ``synthesizing`` for the duration of the call.

        Args:
            text: Text to synthesize.
            output_path: Destination ``.wav``, ``.flac`` or ``.ogg`` path; parent
                directories are created. Relative to and confined to
                ``CHAMPI_TTS_OUTPUT_DIR`` when that env var is set.
            voice: Voice name (``None`` = provider default).
            speed: Speech speed multiplier (``None`` = provider default).
            provider: Provider key. Falls back to ``CHAMPI_TTS_PROVIDER``
                env var, then ``"kokoro"``.
            overwrite: Replace ``output_path`` if it already exists.
        """
        if not text.strip():
            return "error: text is empty"

        try:
            path, fmt = resolve_output_path(output_path, overwrite=overwrite)
            async with state["speaking"].session(text, voice, ctx):
                p = await _get_provider(_effective_provider(provider))
                audio = await p.synthesize(text, voice=voice, speed=speed)
                await save_audio(
                    audio,
                    path,
                    sample_rate=getattr(p.config, "sample_rate", 24000),
                    format=fmt,
                )
            return str(path)
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    @mcp.tool()
    def get_speaking_state() -> dict[str, Any]:
        """Return what the server is saying right now.

        ``state`` is ``"idle"``, ``"synthesizing"`` or ``"speaking"``;
        ``text`` and ``voice`` describe the active utterance (``None`` when
        idle) and ``updated_at`` is the UNIX time of the last transition.
        """
        return state["speaking"].snapshot()

    @mcp.tool()
    async def list_voices(provider: str | None = None) -> list[str]:
        """List the voices available for a provider.

        Does not load the model; voices are read from the provider's voice
        directory. On failure a single ``"error: <Type>: <msg>"`` entry is
        returned.

        Args:
            provider: Provider key. Falls back to ``CHAMPI_TTS_PROVIDER``
                env var, then ``"kokoro"``.
        """
        try:
            p = champi_tts.get_provider(_effective_provider(provider))  # type: ignore[arg-type]
            return await p.list_voices()
        except Exception as exc:
            return [f"error: {type(exc).__name__}: {exc}"]

    from champi_tts.mcp import speaker_tools as _speaker

    @mcp.tool()
    async def speak_tool(
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        provider: str | None = None,
        device_index: int | None = None,
        wait: bool = True,
    ) -> dict[str, Any] | str:
        """Synthesize text and play it on a speaker; return playback metrics.

        Args:
            text: Text to speak.
            voice: Voice name (``None`` = provider default).
            speed: Speech speed multiplier (``None`` = provider default).
            provider: Provider key. Falls back to ``CHAMPI_TTS_PROVIDER``
                env var, then ``"kokoro"``.
            device_index: Output device index. Falls back to
                ``CHAMPI_OUTPUT_DEVICE_INDEX`` env var, then the system default.
            wait: Block until playback finishes (``False`` = return once
                playback has started).
        """
        if not text.strip():
            return "error: text is empty"
        try:
            _speaker._check_sounddevice()
            p = await _get_provider(_effective_provider(provider))
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"
        return await _speaker.speak(text, voice, speed, p, device_index, wait)

    @mcp.tool()
    async def speak_streaming_tool(
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        provider: str | None = None,
        device_index: int | None = None,
    ) -> dict[str, Any] | str:
        """Synthesize text in chunks and play each as it arrives (low latency).

        Args:
            text: Text to speak.
            voice: Voice name (``None`` = provider default).
            speed: Speech speed multiplier (``None`` = provider default).
            provider: Provider key. Falls back to ``CHAMPI_TTS_PROVIDER``
                env var, then ``"kokoro"``.
            device_index: Output device index. Falls back to
                ``CHAMPI_OUTPUT_DEVICE_INDEX`` env var, then the system default.
        """
        if not text.strip():
            return "error: text is empty"
        try:
            _speaker._check_sounddevice()
            p = await _get_provider(_effective_provider(provider))
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"
        return await _speaker.speak_streaming(text, voice, speed, p, device_index)

    @mcp.tool()
    def stop_speaking_tool() -> dict[str, bool]:
        """Stop any active speaker playback."""
        return _speaker.stop_speaking()

    @mcp.tool()
    def list_audio_devices_tool() -> list[dict[str, object]]:
        """List available audio output devices."""
        try:
            return _speaker.list_audio_devices()
        except ImportError as exc:
            return [{"error": str(exc)}]

    return mcp


# Module-level instance created on import (when mcp package is available)
mcp: Any = create_mcp_server() if MCP_AVAILABLE else None


def main(
    transport: str = "stdio",
    host: str = "localhost",
    port: int = _DEFAULT_SSE_PORT,
) -> None:
    """Start the MCP server.

    Args:
        transport: ``"stdio"`` or ``"sse"``.
        host: Bind address (SSE only).
        port: Port (SSE only). Defaults to 8766 so it can run next to
            champi-stt (8765).
    """
    _require_mcp()

    from loguru import logger

    logger.remove()
    logger.add(sys.stderr)

    assert mcp is not None

    if transport == "sse":
        mcp.settings.host = host
        mcp.settings.port = port

    mcp.run(transport=transport)  # type: ignore[arg-type]
