"""
Tests for core audio utilities: AudioPlayer, save_audio, load_audio, normalize_audio.

sounddevice is mocked throughout to avoid requiring audio hardware.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from champi_tts.core.audio import AudioPlayer, load_audio, normalize_audio, save_audio


@pytest.fixture
def audio_data() -> np.ndarray:
    """One-second 440 Hz sine wave at 24000 Hz."""
    t = np.linspace(0, 1.0, 24000)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def player() -> AudioPlayer:
    """AudioPlayer with a 24 kHz sample rate."""
    return AudioPlayer(sample_rate=24000)


class TestAudioPlayerProperties:
    """Tests for AudioPlayer initial state and properties."""

    def test_initial_is_playing_false(self, player: AudioPlayer) -> None:
        """Player starts in non-playing state."""
        assert player.is_playing is False

    def test_sample_rate_stored(self, player: AudioPlayer) -> None:
        """Constructor stores sample rate."""
        assert player.sample_rate == 24000

    def test_custom_sample_rate(self) -> None:
        """Constructor accepts custom sample rates."""
        p = AudioPlayer(sample_rate=16000)
        assert p.sample_rate == 16000


class TestAudioPlayerPlay:
    """Tests for AudioPlayer.play()."""

    @pytest.mark.asyncio
    async def test_play_blocking_calls_sd_play_and_wait(
        self, player: AudioPlayer, audio_data: np.ndarray
    ) -> None:
        """Blocking play calls sd.play then sd.wait."""
        with patch("champi_tts.core.audio.sd") as mock_sd:
            await player.play(audio_data, blocking=True)
            mock_sd.play.assert_called_once()
            mock_sd.wait.assert_called_once()

    @pytest.mark.asyncio
    async def test_play_non_blocking_skips_sd_wait(
        self, player: AudioPlayer, audio_data: np.ndarray
    ) -> None:
        """Non-blocking play calls sd.play but not sd.wait."""
        with patch("champi_tts.core.audio.sd") as mock_sd:
            await player.play(audio_data, blocking=False)
            mock_sd.play.assert_called_once()
            mock_sd.wait.assert_not_called()

    @pytest.mark.asyncio
    async def test_is_playing_reset_after_play(
        self, player: AudioPlayer, audio_data: np.ndarray
    ) -> None:
        """is_playing returns False after play completes."""
        with patch("champi_tts.core.audio.sd"):
            await player.play(audio_data, blocking=True)
        assert player.is_playing is False

    @pytest.mark.asyncio
    async def test_is_playing_reset_on_exception(
        self, player: AudioPlayer, audio_data: np.ndarray
    ) -> None:
        """is_playing is False even when play raises an exception."""
        with patch("champi_tts.core.audio.sd") as mock_sd:
            mock_sd.play.side_effect = RuntimeError("no audio device")
            with pytest.raises(RuntimeError):
                await player.play(audio_data, blocking=True)
        assert player.is_playing is False

    @pytest.mark.asyncio
    async def test_play_passes_sample_rate_to_sd(
        self, player: AudioPlayer, audio_data: np.ndarray
    ) -> None:
        """play() passes the player's sample rate to sd.play."""
        with patch("champi_tts.core.audio.sd") as mock_sd:
            await player.play(audio_data, blocking=False)
            args, _ = mock_sd.play.call_args
            assert args[1] == 24000


class TestAudioPlayerStop:
    """Tests for AudioPlayer.stop()."""

    def test_stop_clears_is_playing(self, player: AudioPlayer) -> None:
        """stop() sets is_playing to False."""
        player._is_playing = True
        with patch("champi_tts.core.audio.sd"):
            player.stop()
        assert player.is_playing is False

    def test_stop_calls_sd_stop(self, player: AudioPlayer) -> None:
        """stop() calls sd.stop()."""
        with patch("champi_tts.core.audio.sd") as mock_sd:
            player.stop()
            mock_sd.stop.assert_called_once()

    def test_stop_sets_stop_event(self, player: AudioPlayer) -> None:
        """stop() sets the internal stop event."""
        with patch("champi_tts.core.audio.sd"):
            player.stop()
        assert player._stop_event.is_set()


class TestAudioPlayerStreaming:
    """Tests for AudioPlayer.play_streaming()."""

    @pytest.mark.asyncio
    async def test_play_streaming_plays_all_chunks(
        self, player: AudioPlayer, audio_data: np.ndarray
    ) -> None:
        """play_streaming() calls sd.play for each chunk."""
        chunks = [audio_data[:1000], audio_data[1000:2000], audio_data[2000:3000]]
        with patch("champi_tts.core.audio.sd") as mock_sd:
            await player.play_streaming(chunks)
            assert mock_sd.play.call_count == 3

    @pytest.mark.asyncio
    async def test_play_streaming_stops_when_stop_event_set(
        self, player: AudioPlayer, audio_data: np.ndarray
    ) -> None:
        """play_streaming() skips remaining chunks once stop event is set."""
        chunks = [audio_data[:1000], audio_data[1000:2000]]
        player._stop_event.set()
        with patch("champi_tts.core.audio.sd") as mock_sd:
            await player.play_streaming(chunks)
            mock_sd.play.assert_not_called()

    @pytest.mark.asyncio
    async def test_play_streaming_empty_list(self, player: AudioPlayer) -> None:
        """play_streaming() with empty list does not call sd.play."""
        with patch("champi_tts.core.audio.sd") as mock_sd:
            await player.play_streaming([])
            mock_sd.play.assert_not_called()


class TestSaveAudio:
    """Tests for save_audio()."""

    @pytest.mark.asyncio
    async def test_save_creates_file(
        self, tmp_path: Path, audio_data: np.ndarray
    ) -> None:
        """save_audio() creates the output file."""
        out = tmp_path / "out.wav"
        await save_audio(audio_data, out, sample_rate=24000)
        assert out.exists()
        assert out.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_save_creates_parent_directories(
        self, tmp_path: Path, audio_data: np.ndarray
    ) -> None:
        """save_audio() creates intermediate directories."""
        out = tmp_path / "nested" / "dir" / "out.wav"
        await save_audio(audio_data, out, sample_rate=24000)
        assert out.exists()

    @pytest.mark.asyncio
    async def test_save_accepts_string_path(
        self, tmp_path: Path, audio_data: np.ndarray
    ) -> None:
        """save_audio() accepts string paths as well as Path objects."""
        out = tmp_path / "string_path.wav"
        await save_audio(audio_data, str(out), sample_rate=24000)
        assert out.exists()


class TestLoadAudio:
    """Tests for load_audio()."""

    @pytest.mark.asyncio
    async def test_load_returns_tuple(
        self, tmp_path: Path, audio_data: np.ndarray
    ) -> None:
        """load_audio() returns (array, sample_rate) tuple."""
        out = tmp_path / "audio.wav"
        await save_audio(audio_data, out, sample_rate=24000)
        result = await load_audio(out)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_load_returns_correct_sample_rate(
        self, tmp_path: Path, audio_data: np.ndarray
    ) -> None:
        """load_audio() returns the correct sample rate."""
        out = tmp_path / "audio.wav"
        await save_audio(audio_data, out, sample_rate=24000)
        _, sample_rate = await load_audio(out)
        assert sample_rate == 24000

    @pytest.mark.asyncio
    async def test_load_returns_ndarray(
        self, tmp_path: Path, audio_data: np.ndarray
    ) -> None:
        """load_audio() returns a numpy array."""
        out = tmp_path / "audio.wav"
        await save_audio(audio_data, out, sample_rate=24000)
        loaded_audio, _ = await load_audio(out)
        assert isinstance(loaded_audio, np.ndarray)
        assert len(loaded_audio) > 0

    @pytest.mark.asyncio
    async def test_load_resamples_to_target_rate(self, tmp_path: Path) -> None:
        """load_audio() resamples when target_sample_rate differs from file rate."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000)).astype(np.float32)
        out = tmp_path / "audio.wav"
        await save_audio(audio, out, sample_rate=24000)

        loaded_audio, result_sr = await load_audio(out, target_sample_rate=16000)
        assert result_sr == 16000
        assert abs(len(loaded_audio) - 16000) < 200

    @pytest.mark.asyncio
    async def test_load_no_resample_when_rates_match(
        self, tmp_path: Path, audio_data: np.ndarray
    ) -> None:
        """load_audio() does not resample when target rate matches file rate."""
        out = tmp_path / "audio.wav"
        await save_audio(audio_data, out, sample_rate=24000)
        loaded_audio, sr = await load_audio(out, target_sample_rate=24000)
        assert sr == 24000
        assert len(loaded_audio) == len(audio_data)

    @pytest.mark.asyncio
    async def test_load_raises_for_missing_file(self) -> None:
        """load_audio() raises SoundFileError when file does not exist."""
        with pytest.raises(sf.SoundFileError):
            await load_audio("/nonexistent/path/audio.wav")


class TestNormalizeAudio:
    """Tests for normalize_audio()."""

    def test_normalize_clips_to_range(self) -> None:
        """normalize_audio() clips output to [-1.0, 1.0]."""
        audio = np.array([2.0, -3.0, 1.5, 0.5, -0.5])
        result = normalize_audio(audio)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_normalize_silence_returns_zeros(self) -> None:
        """normalize_audio() returns silence unchanged."""
        silence = np.zeros(100)
        result = normalize_audio(silence)
        assert np.all(result == 0.0)

    def test_normalize_adjusts_rms(self) -> None:
        """normalize_audio() raises RMS of quiet audio toward target."""
        quiet = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000)) * 0.001
        result = normalize_audio(quiet, target_db=-20.0)
        assert np.abs(result).max() > np.abs(quiet).max()

    def test_normalize_default_target_db(self) -> None:
        """normalize_audio() uses -20 dB as default target."""
        audio = np.random.randn(1000).astype(np.float32) * 0.01
        result = normalize_audio(audio)
        assert len(result) == len(audio)
