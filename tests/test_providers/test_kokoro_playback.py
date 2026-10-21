"""
Event-loop responsiveness tests for KokoroProvider._play_audio().

Skipped when the kokoro provider's text-processing dependencies are missing.
"""

import asyncio
from unittest.mock import patch

import numpy as np
import pytest

provider_module = pytest.importorskip("champi_tts.providers.kokoro.provider")

from champi_tts.providers.kokoro.config import KokoroConfig  # noqa: E402
from champi_tts.providers.kokoro.exceptions import KokoroAudioError  # noqa: E402
from tests.conftest import FakeSoundDevice  # noqa: E402


@pytest.fixture
def provider():
    """KokoroProvider built without touching the singleton or loading models."""
    instance = object.__new__(provider_module.KokoroProvider)
    instance.config = KokoroConfig(sample_rate=22050)
    return instance


@pytest.fixture
def fake_sd(blocking_sd: FakeSoundDevice):
    """Patch the provider module with a blocking fake sounddevice."""
    with patch.object(provider_module, "sd", blocking_sd):
        yield blocking_sd


@pytest.fixture
def audio() -> np.ndarray:
    """Short float32 buffer as returned by Kokoro."""
    return np.zeros(2205, dtype=np.float32)


class TestKokoroPlayAudio:
    """_play_audio() must not block the event loop."""

    @pytest.mark.asyncio
    async def test_loop_stays_free_and_stop_ends_playback(
        self, provider, fake_sd: FakeSoundDevice, audio: np.ndarray
    ) -> None:
        """Another coroutine can stop the device while playback is awaited."""
        play_task = asyncio.create_task(provider._play_audio(audio))
        await fake_sd.wait_started()
        await asyncio.sleep(0.05)
        assert not play_task.done()

        fake_sd.stop()
        playback_time = await asyncio.wait_for(play_task, timeout=2.0)

        assert playback_time >= 0.05

    @pytest.mark.asyncio
    async def test_uses_configured_sample_rate_and_pulse_first(
        self, provider, fake_sd: FakeSoundDevice, audio: np.ndarray
    ) -> None:
        """Playback uses config.sample_rate and tries the pulse device first."""
        fake_sd.playback_seconds = 0
        await provider._play_audio(audio)

        _, samplerate, kwargs = fake_sd.play_calls[0]
        assert samplerate == 22050
        assert kwargs == {"device": "pulse"}

    @pytest.mark.asyncio
    async def test_falls_back_to_default_device(
        self, provider, fake_sd: FakeSoundDevice, audio: np.ndarray
    ) -> None:
        """A failing pulse device falls back to the default device."""
        fake_sd.playback_seconds = 0
        real_play = fake_sd.play

        def play(data, samplerate, **kwargs):
            if kwargs.get("device") == "pulse":
                raise RuntimeError("no pulse")
            real_play(data, samplerate, **kwargs)

        fake_sd.play = play
        await provider._play_audio(audio)

        assert fake_sd.play_calls[0][1:] == (22050, {})

    @pytest.mark.asyncio
    async def test_both_devices_failing_raises_audio_error(
        self, provider, fake_sd: FakeSoundDevice, audio: np.ndarray
    ) -> None:
        """When every device fails a KokoroAudioError propagates."""

        def play(data, samplerate, **kwargs):
            raise RuntimeError("no device")

        fake_sd.play = play
        with pytest.raises(KokoroAudioError):
            await provider._play_audio(audio)

    @pytest.mark.asyncio
    async def test_cancelling_playback_stops_device(
        self, provider, fake_sd: FakeSoundDevice, audio: np.ndarray
    ) -> None:
        """Cancelling the awaiting task stops the device so the worker exits."""
        play_task = asyncio.create_task(provider._play_audio(audio))
        await fake_sd.wait_started()

        play_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await play_task

        assert fake_sd.stop_calls == 1
