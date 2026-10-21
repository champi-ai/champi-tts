"""
Event-loop responsiveness tests for AudioPlayer playback.

A fake ``sounddevice`` whose ``wait()`` blocks until ``stop()`` proves that
blocking playback no longer freezes the event loop.
"""

import asyncio
import time
from unittest.mock import patch

import numpy as np
import pytest

from champi_tts.core.audio import AudioPlayer
from tests.conftest import FakeSoundDevice


@pytest.fixture
def fake_sd(blocking_sd: FakeSoundDevice):
    """Patch the audio module with a blocking fake sounddevice."""
    with patch("champi_tts.core.audio.sd", blocking_sd):
        yield blocking_sd


@pytest.fixture
def audio() -> np.ndarray:
    """Short silent buffer; the fake decides how long playback lasts."""
    return np.zeros(2400, dtype=np.float32)


class TestAudioPlayerNonBlocking:
    """AudioPlayer.play() must leave the event loop free while audio plays."""

    @pytest.mark.asyncio
    async def test_other_coroutines_run_during_play(
        self, fake_sd: FakeSoundDevice, audio: np.ndarray
    ) -> None:
        """A concurrent coroutine ticks while play() is awaiting."""
        player = AudioPlayer(sample_rate=24000)
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        tick_task = asyncio.create_task(ticker())
        play_task = asyncio.create_task(player.play(audio, blocking=True))
        await fake_sd.wait_started()
        await asyncio.sleep(0.1)

        assert not play_task.done()
        assert player.is_playing is True
        assert ticks >= 3

        player.stop()
        await asyncio.wait_for(play_task, timeout=2.0)
        tick_task.cancel()

    @pytest.mark.asyncio
    async def test_stop_from_another_coroutine_ends_playback_quickly(
        self, fake_sd: FakeSoundDevice, audio: np.ndarray
    ) -> None:
        """stop() called on the same loop makes play() return within ~100 ms."""
        player = AudioPlayer(sample_rate=24000)
        play_task = asyncio.create_task(player.play(audio, blocking=True))
        await fake_sd.wait_started()

        stopped_at = time.perf_counter()
        player.stop()
        await asyncio.wait_for(play_task, timeout=2.0)
        elapsed = time.perf_counter() - stopped_at

        assert elapsed < 0.1
        assert player.is_playing is False
        assert fake_sd.stop_calls == 1

    def test_stop_before_worker_starts_skips_playback(
        self, fake_sd: FakeSoundDevice, audio: np.ndarray
    ) -> None:
        """A stop() that lands before the worker runs prevents playback."""
        player = AudioPlayer(sample_rate=24000)
        player.stop()

        player._play_and_wait(audio)

        assert fake_sd.play_calls == []

    @pytest.mark.asyncio
    async def test_cancelling_play_stops_device(
        self, fake_sd: FakeSoundDevice, audio: np.ndarray
    ) -> None:
        """Cancelling the awaiting task stops the device so the worker exits."""
        player = AudioPlayer(sample_rate=24000)
        play_task = asyncio.create_task(player.play(audio, blocking=True))
        await fake_sd.wait_started()

        play_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await play_task

        assert fake_sd.stop_calls == 1
        assert player.is_playing is False

    @pytest.mark.asyncio
    async def test_play_streaming_stops_between_chunks(
        self, fake_sd: FakeSoundDevice, audio: np.ndarray
    ) -> None:
        """stop() during the first chunk prevents the remaining chunks."""
        player = AudioPlayer(sample_rate=24000)
        stream_task = asyncio.create_task(player.play_streaming([audio, audio, audio]))
        await fake_sd.wait_started()

        player.stop()
        await asyncio.wait_for(stream_task, timeout=2.0)

        assert len(fake_sd.play_calls) == 1
