"""Speaker playback helpers for the MCP speaker tools.

Twin of ``champi_stt.mcp.mic_tools``: instead of capturing microphone input
and transcribing it, these helpers synthesize text through a provider and
play the result on an audio output device. All blocking ``sounddevice``
calls run in an executor so the event loop stays free to serve
``stop_speaking()`` while audio is playing.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import threading
import time
from collections.abc import AsyncGenerator
from typing import Any, cast

import numpy as np

from champi_tts.core.base_provider import BaseTTSProvider

_DEFAULT_PROVIDER = "kokoro"
_DEFAULT_SAMPLE_RATE = 24000
_STREAM_BLOCK_FRAMES = 2048
_QUEUE_POLL_SECONDS = 0.05

# Shared playback state: stop_speaking() sets the event, and the lock makes a
# stop that lands before playback starts skip it instead of racing sd.play().
_stop_event = threading.Event()
_device_lock = threading.Lock()
_active_playbacks = 0


def _resolve_device(device_index: int | None) -> int | None:
    """Return the audio output device index to use.

    Resolves in priority order: explicit argument → ``CHAMPI_OUTPUT_DEVICE_INDEX``
    environment variable → ``None`` (sounddevice system default).

    Args:
        device_index: Caller-supplied device index, or ``None`` to use fallbacks.

    Returns:
        Integer device index, or ``None`` to let sounddevice pick the default.
    """
    if device_index is not None:
        return device_index
    env = os.environ.get("CHAMPI_OUTPUT_DEVICE_INDEX")
    return int(env) if env else None


def _check_sounddevice() -> None:
    """Verify that sounddevice and its PortAudio backend are available.

    Raises:
        ImportError: If sounddevice or PortAudio is not installed.
    """
    try:
        import sounddevice  # noqa: F401
    except (ImportError, OSError) as exc:
        raise ImportError(
            "sounddevice is required for speaker playback. "
            "Install PortAudio (e.g. 'sudo apt install portaudio19-dev' on Debian/Ubuntu "
            "or 'brew install portaudio' on macOS), then reinstall champi-tts."
        ) from exc


def list_audio_devices() -> list[dict[str, object]]:
    """Return available audio output devices.

    Returns:
        List of dicts with keys ``index``, ``name``, ``max_output_channels``,
        ``default_samplerate``, and ``is_default`` for each output-capable device.
    """
    _check_sounddevice()
    import sounddevice as sd

    devices = sd.query_devices()
    result = []
    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            result.append(
                {
                    "index": i,
                    "name": dev["name"],
                    "max_output_channels": dev["max_output_channels"],
                    "default_samplerate": dev["default_samplerate"],
                    "is_default": i == sd.default.device[1],
                }
            )
    return result


async def _get_provider(provider: str | BaseTTSProvider | None) -> BaseTTSProvider:
    """Return an initialised provider.

    A provider instance is used as-is so callers (the MCP server) can reuse
    their cached, already-initialised provider. A name — or ``None``, which
    falls back to ``CHAMPI_TTS_PROVIDER`` then ``"kokoro"`` — is resolved via
    the factory and initialised.
    """
    if provider is not None and not isinstance(provider, str):
        return provider

    import champi_tts

    name = provider or os.environ.get("CHAMPI_TTS_PROVIDER") or _DEFAULT_PROVIDER
    p = champi_tts.get_provider(name)  # type: ignore[arg-type]
    await p.initialize()
    return p


def _sample_rate(provider: Any) -> int:
    return int(getattr(provider.config, "sample_rate", _DEFAULT_SAMPLE_RATE))


def _voice(provider: Any, voice: str | None) -> str | None:
    return voice or getattr(provider.config, "default_voice", None)


async def _text_to_audio(
    text: str,
    voice: str | None,
    speed: float | None,
    provider: str | BaseTTSProvider | None,
) -> tuple[np.ndarray, int, str | None]:
    """Synthesize text into a mono float32 numpy array via the provider pipeline.

    Args:
        text: Text to synthesize.
        voice: Voice name, or ``None`` for the provider default.
        speed: Speech speed multiplier, or ``None`` for the provider default.
        provider: Provider instance, provider key, or ``None`` to fall back to
            ``CHAMPI_TTS_PROVIDER`` then ``"kokoro"``.

    Returns:
        Tuple of ``(audio, sample_rate, voice_used)``.
    """
    p = await _get_provider(provider)
    audio = await p.synthesize(text, voice=voice, speed=speed)
    return _as_mono(audio), _sample_rate(p), _voice(p, voice)


def _as_mono(audio: Any) -> np.ndarray:
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _play_and_wait(audio: np.ndarray, sample_rate: int, device: int | None) -> bool:
    """Play ``audio`` and block until it ends; return ``True`` if it was stopped."""
    import sounddevice as sd

    with _device_lock:
        if _stop_event.is_set():
            return True
        sd.play(audio, sample_rate, device=device)
    sd.wait()
    return _stop_event.is_set()


def _play_queue(
    chunks: queue.Queue[np.ndarray | None], sample_rate: int, device: int | None
) -> bool:
    """Write queued chunks to an output stream until ``None`` or a stop request.

    Returns:
        ``True`` if playback was stopped before the queue was drained.
    """
    import sounddevice as sd

    with sd.OutputStream(
        samplerate=sample_rate, channels=1, dtype="float32", device=device
    ) as stream:
        while not _stop_event.is_set():
            try:
                chunk = chunks.get(timeout=_QUEUE_POLL_SECONDS)
            except queue.Empty:
                continue
            if chunk is None:
                break
            frames = chunk.reshape(-1, 1)
            # Small writes keep stop_speaking() responsive mid-chunk.
            for start in range(0, len(frames), _STREAM_BLOCK_FRAMES):
                if _stop_event.is_set():
                    break
                stream.write(frames[start : start + _STREAM_BLOCK_FRAMES])
        if _stop_event.is_set():
            stream.abort()
            return True
        stream.stop()
    return False


async def speak(
    text: str,
    voice: str | None = None,
    speed: float | None = None,
    provider: str | BaseTTSProvider | None = None,
    device_index: int | None = None,
    wait: bool = True,
) -> dict[str, Any] | str:
    """Synthesize text and play it on a speaker.

    Args:
        text: Text to speak.
        voice: Voice name, or ``None`` for the provider default.
        speed: Speech speed multiplier, or ``None`` for the provider default.
        provider: Provider instance, provider key, or ``None`` to fall back to
            ``CHAMPI_TTS_PROVIDER`` then ``"kokoro"``.
        device_index: Output device index. Falls back to
            ``CHAMPI_OUTPUT_DEVICE_INDEX`` env var, then the system default.
        wait: Block until playback finishes. When ``False`` the call returns
            as soon as playback has started.

    Returns:
        Metrics dict (``duration_seconds``, ``generation_time_seconds``,
        ``voice``, ``sample_rate``, ``device_index``, ``interrupted``) on
        success, or ``"error: <ExcType>: <message>"`` on failure.
    """
    global _active_playbacks
    if not text.strip():
        return "error: text is empty"
    try:
        _check_sounddevice()
        import sounddevice as sd

        device = _resolve_device(device_index)
        _stop_event.clear()

        started = time.perf_counter()
        audio, sample_rate, voice_used = await _text_to_audio(
            text, voice, speed, provider
        )
        generation_time = time.perf_counter() - started

        loop = asyncio.get_running_loop()
        _active_playbacks += 1
        try:
            if wait:
                interrupted = await loop.run_in_executor(
                    None, _play_and_wait, audio, sample_rate, device
                )
            else:
                interrupted = _stop_event.is_set()
                if not interrupted:
                    await loop.run_in_executor(
                        None,
                        lambda: sd.play(audio, sample_rate, device=device),
                    )
        finally:
            _active_playbacks -= 1

        return {
            "duration_seconds": len(audio) / sample_rate,
            "generation_time_seconds": generation_time,
            "voice": voice_used,
            "sample_rate": sample_rate,
            "device_index": device,
            "interrupted": interrupted,
        }
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


async def speak_streaming(
    text: str,
    voice: str | None = None,
    speed: float | None = None,
    provider: str | BaseTTSProvider | None = None,
    device_index: int | None = None,
) -> dict[str, Any] | str:
    """Synthesize text in chunks and play each chunk as soon as it arrives.

    Playback of the first chunk starts while later chunks are still being
    synthesized, which lowers time-to-first-audio for long texts.

    Args:
        text: Text to speak.
        voice: Voice name, or ``None`` for the provider default.
        speed: Speech speed multiplier, or ``None`` for the provider default.
        provider: Provider instance, provider key, or ``None`` to fall back to
            ``CHAMPI_TTS_PROVIDER`` then ``"kokoro"``.
        device_index: Output device index. Falls back to
            ``CHAMPI_OUTPUT_DEVICE_INDEX`` env var, then the system default.

    Returns:
        Metrics dict (``duration_seconds``, ``generation_time_seconds``,
        ``time_to_first_audio_seconds``, ``chunks``, ``voice``,
        ``sample_rate``, ``device_index``, ``interrupted``) on success, or
        ``"error: <ExcType>: <message>"`` on failure.
    """
    global _active_playbacks
    if not text.strip():
        return "error: text is empty"
    try:
        _check_sounddevice()

        device = _resolve_device(device_index)
        _stop_event.clear()
        p = await _get_provider(provider)
        sample_rate = _sample_rate(p)

        chunks: queue.Queue[np.ndarray | None] = queue.Queue()
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        first_audio: float | None = None
        total_frames = 0
        chunk_count = 0

        _active_playbacks += 1
        player = loop.run_in_executor(None, _play_queue, chunks, sample_rate, device)
        try:
            # BaseTTSProvider declares this as ``async def`` without a yield, so
            # mypy sees a coroutine; implementations are async generators.
            stream = cast(
                "AsyncGenerator[np.ndarray, None]",
                p.synthesize_streaming(text, voice=voice, speed=speed),
            )
            async with contextlib.aclosing(stream):
                async for raw in stream:
                    if _stop_event.is_set():
                        break
                    chunk = _as_mono(raw)
                    if first_audio is None:
                        first_audio = time.perf_counter() - started
                    chunks.put(chunk)
                    total_frames += len(chunk)
                    chunk_count += 1
            generation_time = time.perf_counter() - started
        finally:
            chunks.put(None)
            try:
                interrupted = await player
            finally:
                _active_playbacks -= 1

        return {
            "duration_seconds": total_frames / sample_rate,
            "generation_time_seconds": generation_time,
            "time_to_first_audio_seconds": first_audio,
            "chunks": chunk_count,
            "voice": _voice(p, voice),
            "sample_rate": sample_rate,
            "device_index": device,
            "interrupted": interrupted,
        }
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def stop_speaking() -> dict[str, bool]:
    """Interrupt any active ``speak`` / ``speak_streaming`` playback.

    Returns:
        ``{"stopped": True}`` if playback was active, ``{"stopped": False}``
        otherwise.
    """
    was_active = _active_playbacks > 0
    _stop_event.set()
    with _device_lock:
        try:
            import sounddevice as sd
        except (ImportError, OSError):
            return {"stopped": was_active}
        sd.stop()
    return {"stopped": was_active}
