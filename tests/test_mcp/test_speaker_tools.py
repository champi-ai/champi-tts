"""Unit tests for the speaker playback helpers in ``champi_tts.mcp.speaker_tools``.

``sounddevice`` is replaced with a fake module so no audio hardware or
PortAudio install is needed.
"""

from __future__ import annotations

import asyncio
import importlib.abc
import os
import queue
import sys
import threading
from collections.abc import AsyncGenerator, Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

import champi_tts.mcp.speaker_tools as spk


class FakeSoundDevice(SimpleNamespace):
    """Minimal stand-in for the ``sounddevice`` module.

    ``wait()`` blocks until ``stop()`` is called when ``block_until_stop`` is
    set, which lets tests interrupt playback deterministically.
    """

    def __init__(self, *, block_until_stop: bool = False) -> None:
        super().__init__()
        self.block_until_stop = block_until_stop
        self.played: list[tuple[np.ndarray, int, Any]] = []
        self.playing = threading.Event()
        self.stopped = threading.Event()
        self.stop_calls = 0
        self.default = SimpleNamespace(device=(0, 1))
        self.streams: list[FakeOutputStream] = []
        self.devices = [
            {"name": "mic", "max_input_channels": 2, "max_output_channels": 0,
             "default_samplerate": 48000.0},
            {"name": "speakers", "max_input_channels": 0, "max_output_channels": 2,
             "default_samplerate": 48000.0},
            {"name": "headset", "max_input_channels": 1, "max_output_channels": 2,
             "default_samplerate": 44100.0},
        ]  # fmt: skip

    def play(self, audio: np.ndarray, samplerate: int, device: Any = None) -> None:
        self.played.append((audio, samplerate, device))
        self.playing.set()

    def wait(self) -> None:
        if self.block_until_stop:
            assert self.stopped.wait(timeout=5)

    def stop(self) -> None:
        self.stop_calls += 1
        self.stopped.set()

    def query_devices(self) -> list[dict[str, Any]]:
        return self.devices

    def OutputStream(self, **kwargs: Any) -> FakeOutputStream:  # noqa: N802
        stream = FakeOutputStream(self, **kwargs)
        self.streams.append(stream)
        return stream


class FakeOutputStream:
    def __init__(self, sd: FakeSoundDevice, **kwargs: Any) -> None:
        self.sd = sd
        self.kwargs = kwargs
        self.frames: list[np.ndarray] = []
        self.aborted = False
        self.stopped = False
        self.closed = False

    def __enter__(self) -> FakeOutputStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.closed = True

    def write(self, data: np.ndarray) -> None:
        self.frames.append(data)
        self.sd.playing.set()

    def abort(self) -> None:
        self.aborted = True

    def stop(self) -> None:
        self.stopped = True


def _make_provider(
    *,
    audio: np.ndarray | None = None,
    chunks: list[np.ndarray] | None = None,
    sample_rate: int = 24000,
    default_voice: str = "af_bella",
    synthesize_raises: Exception | None = None,
) -> MagicMock:
    prov = MagicMock()
    prov.config.sample_rate = sample_rate
    prov.config.default_voice = default_voice
    prov.initialize = AsyncMock()
    prov.synthesize = AsyncMock(
        side_effect=synthesize_raises,
        return_value=audio if audio is not None else np.zeros(2400, np.float32),
    )
    stream_chunks = chunks if chunks is not None else [np.zeros(2400, np.float32)]

    async def _stream(*_: Any, **__: Any) -> AsyncGenerator[np.ndarray, None]:
        for chunk in stream_chunks:
            await asyncio.sleep(0)
            yield chunk

    prov.synthesize_streaming = MagicMock(side_effect=_stream)
    return prov


@pytest.fixture(autouse=True)
def _reset_playback_state() -> Iterator[None]:
    spk._stop_event.clear()
    spk._active_playbacks = 0
    yield
    spk._stop_event.clear()
    spk._active_playbacks = 0


@pytest.fixture()
def fake_sd() -> Iterator[FakeSoundDevice]:
    sd = FakeSoundDevice()
    with patch.dict(sys.modules, {"sounddevice": sd}):
        yield sd


class _PortAudioMissingFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any, target: Any = None) -> None:
        if fullname == "sounddevice":
            raise OSError("PortAudio library not found")
        return None


@pytest.fixture()
def no_portaudio() -> Iterator[None]:
    finder = _PortAudioMissingFinder()
    with patch.dict(sys.modules):
        sys.modules.pop("sounddevice", None)
        sys.meta_path.insert(0, finder)
        try:
            yield
        finally:
            sys.meta_path.remove(finder)


class TestResolveDevice:
    def test_returns_explicit_device_index_when_provided(self) -> None:
        assert spk._resolve_device(2) == 2

    def test_falls_back_to_env_var_when_device_index_is_none(self) -> None:
        with patch.dict(os.environ, {"CHAMPI_OUTPUT_DEVICE_INDEX": "3"}):
            assert spk._resolve_device(None) == 3

    def test_returns_none_when_neither_argument_nor_env_var_set(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "CHAMPI_OUTPUT_DEVICE_INDEX"}
        with patch.dict(os.environ, env, clear=True):
            assert spk._resolve_device(None) is None

    def test_explicit_index_overrides_env_var(self) -> None:
        with patch.dict(os.environ, {"CHAMPI_OUTPUT_DEVICE_INDEX": "5"}):
            assert spk._resolve_device(1) == 1

    def test_ignores_input_device_env_var(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "CHAMPI_OUTPUT_DEVICE_INDEX"}
        env["CHAMPI_INPUT_DEVICE_INDEX"] = "7"
        with patch.dict(os.environ, env, clear=True):
            assert spk._resolve_device(None) is None


class TestCheckSounddevice:
    def test_passes_when_sounddevice_importable(self, fake_sd: FakeSoundDevice) -> None:
        spk._check_sounddevice()

    def test_raises_import_error_with_hint_when_module_missing(self) -> None:
        with (
            patch.dict(sys.modules, {"sounddevice": None}),
            pytest.raises(ImportError, match="portaudio"),
        ):
            spk._check_sounddevice()

    @pytest.mark.usefixtures("no_portaudio")
    def test_raises_import_error_when_portaudio_missing(self) -> None:
        with pytest.raises(ImportError, match="PortAudio"):
            spk._check_sounddevice()


class TestListAudioDevices:
    def test_returns_only_output_devices(self, fake_sd: FakeSoundDevice) -> None:
        devices = spk.list_audio_devices()
        assert [d["index"] for d in devices] == [1, 2]
        assert devices[0] == {
            "index": 1,
            "name": "speakers",
            "max_output_channels": 2,
            "default_samplerate": 48000.0,
            "is_default": True,
        }
        assert devices[1]["is_default"] is False

    def test_raises_import_error_when_unavailable(self) -> None:
        with (
            patch.dict(sys.modules, {"sounddevice": None}),
            pytest.raises(ImportError),
        ):
            spk.list_audio_devices()


class TestTextToAudio:
    @pytest.mark.asyncio
    async def test_uses_provider_instance_without_reinitialising(self) -> None:
        prov = _make_provider(audio=np.zeros((10, 1), np.float64), sample_rate=22050)
        audio, rate, voice = await spk._text_to_audio("hi", None, 1.1, prov)
        prov.initialize.assert_not_awaited()
        prov.synthesize.assert_awaited_once_with("hi", voice=None, speed=1.1)
        assert audio.shape == (10,)
        assert audio.dtype == np.float32
        assert rate == 22050
        assert voice == "af_bella"

    @pytest.mark.asyncio
    async def test_resolves_provider_name_via_env_var(self) -> None:
        prov = _make_provider()
        with (
            patch.dict(os.environ, {"CHAMPI_TTS_PROVIDER": "custom"}),
            patch("champi_tts.get_provider", return_value=prov) as mock_get,
        ):
            _, _, voice = await spk._text_to_audio("hi", "bm_george", None, None)
        mock_get.assert_called_once_with("custom")
        prov.initialize.assert_awaited_once()
        assert voice == "bm_george"

    @pytest.mark.asyncio
    async def test_defaults_to_kokoro(self) -> None:
        prov = _make_provider()
        env = {k: v for k, v in os.environ.items() if k != "CHAMPI_TTS_PROVIDER"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch("champi_tts.get_provider", return_value=prov) as mock_get,
        ):
            await spk._text_to_audio("hi", None, None, None)
        mock_get.assert_called_once_with("kokoro")


class TestSpeak:
    @pytest.mark.asyncio
    async def test_plays_on_resolved_device_and_returns_metrics(
        self, fake_sd: FakeSoundDevice
    ) -> None:
        prov = _make_provider(audio=np.zeros(48000, np.float32), sample_rate=24000)
        with patch.dict(os.environ, {"CHAMPI_OUTPUT_DEVICE_INDEX": "4"}):
            result = await spk.speak("hello", voice="bm_george", provider=prov)

        assert isinstance(result, dict)
        assert result["duration_seconds"] == pytest.approx(2.0)
        assert result["generation_time_seconds"] >= 0
        assert result["voice"] == "bm_george"
        assert result["sample_rate"] == 24000
        assert result["device_index"] == 4
        assert result["interrupted"] is False
        _, rate, device = fake_sd.played[0]
        assert (rate, device) == (24000, 4)

    @pytest.mark.asyncio
    async def test_empty_text_returns_error(self, fake_sd: FakeSoundDevice) -> None:
        prov = _make_provider()
        assert await spk.speak("  ", provider=prov) == "error: text is empty"
        prov.synthesize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_sounddevice_returns_error_value(self) -> None:
        prov = _make_provider()
        with patch.dict(sys.modules, {"sounddevice": None}):
            result = await spk.speak("hello", provider=prov)
        assert isinstance(result, str)
        assert result.startswith("error: ImportError:")
        prov.synthesize.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("no_portaudio")
    async def test_missing_portaudio_returns_error_value(self) -> None:
        result = await spk.speak("hello", provider=_make_provider())
        assert isinstance(result, str)
        assert "PortAudio" in result

    @pytest.mark.asyncio
    async def test_synthesis_failure_returns_error_value(
        self, fake_sd: FakeSoundDevice
    ) -> None:
        prov = _make_provider(synthesize_raises=RuntimeError("boom"))
        assert await spk.speak("hi", provider=prov) == "error: RuntimeError: boom"
        assert fake_sd.played == []

    @pytest.mark.asyncio
    async def test_stop_speaking_interrupts_playback(
        self, fake_sd: FakeSoundDevice
    ) -> None:
        fake_sd.block_until_stop = True
        task = asyncio.create_task(spk.speak("long text", provider=_make_provider()))
        assert await asyncio.to_thread(fake_sd.playing.wait, 5)

        assert spk.stop_speaking() == {"stopped": True}
        result = await asyncio.wait_for(task, timeout=5)

        assert isinstance(result, dict)
        assert result["interrupted"] is True
        assert fake_sd.stop_calls == 1

    @pytest.mark.asyncio
    async def test_wait_false_returns_once_playback_started(
        self, fake_sd: FakeSoundDevice
    ) -> None:
        fake_sd.block_until_stop = True
        result = await asyncio.wait_for(
            spk.speak("hi", provider=_make_provider(), wait=False), timeout=5
        )
        assert isinstance(result, dict)
        assert result["interrupted"] is False
        assert len(fake_sd.played) == 1

    @pytest.mark.asyncio
    async def test_previous_stop_does_not_block_next_speak(
        self, fake_sd: FakeSoundDevice
    ) -> None:
        spk.stop_speaking()
        result = await spk.speak("hi", provider=_make_provider())
        assert isinstance(result, dict)
        assert result["interrupted"] is False
        assert len(fake_sd.played) == 1


class TestSpeakStreaming:
    @pytest.mark.asyncio
    async def test_writes_every_chunk_and_returns_metrics(
        self, fake_sd: FakeSoundDevice
    ) -> None:
        chunks = [np.ones(5000, np.float32), np.ones(1000, np.float32)]
        prov = _make_provider(chunks=chunks, sample_rate=1000)

        result = await spk.speak_streaming("hello", provider=prov, device_index=2)

        assert isinstance(result, dict)
        assert result["chunks"] == 2
        assert result["duration_seconds"] == pytest.approx(6.0)
        assert result["time_to_first_audio_seconds"] is not None
        assert result["voice"] == "af_bella"
        assert result["device_index"] == 2
        assert result["interrupted"] is False
        stream = fake_sd.streams[0]
        assert stream.kwargs == {
            "samplerate": 1000,
            "channels": 1,
            "dtype": "float32",
            "device": 2,
        }
        assert sum(len(f) for f in stream.frames) == 6000
        assert all(f.shape[1] == 1 for f in stream.frames)
        assert stream.stopped is True
        assert stream.aborted is False
        prov.synthesize_streaming.assert_called_once_with(
            "hello", voice=None, speed=None
        )

    @pytest.mark.asyncio
    async def test_stop_speaking_aborts_stream(self, fake_sd: FakeSoundDevice) -> None:
        gate = asyncio.Event()

        async def _endless(*_: Any, **__: Any) -> AsyncGenerator[np.ndarray, None]:
            yield np.ones(100, np.float32)
            await gate.wait()
            yield np.ones(100, np.float32)

        prov = _make_provider()
        prov.synthesize_streaming = MagicMock(side_effect=_endless)
        task = asyncio.create_task(spk.speak_streaming("long", provider=prov))
        assert await asyncio.to_thread(fake_sd.playing.wait, 5)

        assert spk.stop_speaking() == {"stopped": True}
        gate.set()
        result = await asyncio.wait_for(task, timeout=5)

        assert isinstance(result, dict)
        assert result["interrupted"] is True
        assert result["chunks"] == 1
        assert fake_sd.streams[0].aborted is True

    @pytest.mark.asyncio
    async def test_streaming_failure_returns_error_value(
        self, fake_sd: FakeSoundDevice
    ) -> None:
        async def _broken(*_: Any, **__: Any) -> AsyncGenerator[np.ndarray, None]:
            yield np.ones(10, np.float32)
            raise RuntimeError("model crashed")

        prov = _make_provider()
        prov.synthesize_streaming = MagicMock(side_effect=_broken)
        result = await asyncio.wait_for(
            spk.speak_streaming("hi", provider=prov), timeout=5
        )
        assert result == "error: RuntimeError: model crashed"

    @pytest.mark.asyncio
    async def test_missing_sounddevice_returns_error_value(self) -> None:
        with patch.dict(sys.modules, {"sounddevice": None}):
            result = await spk.speak_streaming("hi", provider=_make_provider())
        assert isinstance(result, str)
        assert result.startswith("error: ImportError:")

    @pytest.mark.asyncio
    async def test_empty_text_returns_error(self) -> None:
        assert await spk.speak_streaming("") == "error: text is empty"


class TestStopSpeaking:
    def test_reports_not_stopped_when_idle(self, fake_sd: FakeSoundDevice) -> None:
        assert spk.stop_speaking() == {"stopped": False}
        assert fake_sd.stop_calls == 1

    def test_safe_without_sounddevice(self) -> None:
        with patch.dict(sys.modules, {"sounddevice": None}):
            assert spk.stop_speaking() == {"stopped": False}


class TestPlaybackHelpers:
    """Direct tests for the blocking helpers run in the executor."""

    def test_play_and_wait_skips_playback_when_already_stopped(
        self, fake_sd: FakeSoundDevice
    ) -> None:
        spk._stop_event.set()
        assert spk._play_and_wait(np.zeros(10, np.float32), 24000, None) is True
        assert fake_sd.played == []

    def test_play_queue_keeps_polling_while_queue_is_empty(
        self, fake_sd: FakeSoundDevice
    ) -> None:
        chunks = MagicMock()
        chunks.get.side_effect = [queue.Empty(), np.ones(4, np.float32), None]
        assert spk._play_queue(chunks, 24000, None) is False
        assert chunks.get.call_count == 3
        assert sum(len(f) for f in fake_sd.streams[0].frames) == 4
        assert fake_sd.streams[0].stopped is True

    def test_play_queue_stops_mid_chunk(
        self, fake_sd: FakeSoundDevice, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(spk, "_STREAM_BLOCK_FRAMES", 2)
        chunks: queue.Queue[np.ndarray | None] = queue.Queue()
        chunks.put(np.ones(10, np.float32))
        chunks.put(None)
        real_output_stream = fake_sd.OutputStream

        def _stream_that_stops(**kwargs: Any) -> FakeOutputStream:
            stream = real_output_stream(**kwargs)
            original_write = stream.write

            def _write(data: np.ndarray) -> None:
                original_write(data)
                spk._stop_event.set()

            stream.write = _write  # type: ignore[method-assign]
            return stream

        fake_sd.OutputStream = _stream_that_stops  # type: ignore[method-assign]

        assert spk._play_queue(chunks, 24000, None) is True
        stream = fake_sd.streams[0]
        assert len(stream.frames) == 1
        assert stream.aborted is True
        assert stream.stopped is False


class TestSpeakStopDuringSynthesis:
    @pytest.mark.asyncio
    async def test_wait_false_skips_playback_when_stopped_during_synthesis(
        self, fake_sd: FakeSoundDevice
    ) -> None:
        prov = _make_provider()

        async def _synth(*_: Any, **__: Any) -> np.ndarray:
            spk.stop_speaking()
            return np.zeros(2400, np.float32)

        prov.synthesize = AsyncMock(side_effect=_synth)
        result = await spk.speak("hi", provider=prov, wait=False)

        assert isinstance(result, dict)
        assert result["interrupted"] is True
        assert fake_sd.played == []
