"""
Integration tests for audio handling.

Covers AudioPlayer playback, audio file I/O, and audio processing utilities.
All tests that invoke the audio hardware patch sounddevice via the mock_sd fixture.
"""

import numpy as np
import pytest

from champi_tts.core.audio import load_audio, normalize_audio, save_audio

# ---------------------------------------------------------------------------
# AudioPlayer — initial state
# ---------------------------------------------------------------------------


def test_audio_player_initial_is_not_playing(audio_player):
    """AudioPlayer starts in a non-playing state."""
    assert not audio_player.is_playing


def test_audio_player_initial_sample_rate(audio_player):
    """AudioPlayer retains the sample rate passed to its constructor."""
    assert audio_player.sample_rate == 22050


# ---------------------------------------------------------------------------
# AudioPlayer — play
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_player_play_blocking(audio_player, float_audio, mock_sd):
    """Blocking play completes without error."""
    await audio_player.play(float_audio, blocking=True)
    assert not audio_player.is_playing


@pytest.mark.asyncio
async def test_audio_player_play_nonblocking(audio_player, float_audio, mock_sd):
    """Non-blocking play does not raise."""
    await audio_player.play(float_audio, blocking=False)


@pytest.mark.asyncio
async def test_audio_player_is_not_playing_after_completion(
    audio_player, float_audio, mock_sd
):
    """is_playing is False after play returns."""
    await audio_player.play(float_audio, blocking=True)
    assert not audio_player.is_playing


@pytest.mark.asyncio
async def test_audio_player_play_silence(audio_player, mock_sd):
    """Playing a silent buffer does not raise."""
    silence = np.zeros(22050, dtype=np.float32)
    await audio_player.play(silence, blocking=False)


@pytest.mark.asyncio
async def test_audio_player_play_single_sample(audio_player, mock_sd):
    """Playing a single-sample buffer does not raise."""
    single = np.array([0.5], dtype=np.float32)
    await audio_player.play(single, blocking=False)


@pytest.mark.asyncio
async def test_audio_player_play_large_buffer(audio_player, mock_sd):
    """Playing a large audio buffer completes without error."""
    large = np.zeros(44100 * 5, dtype=np.float32)
    await audio_player.play(large, blocking=False)


# ---------------------------------------------------------------------------
# AudioPlayer — stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_player_stop_sets_not_playing(audio_player, float_audio, mock_sd):
    """stop() sets is_playing to False."""
    await audio_player.play(float_audio, blocking=False)
    audio_player.stop()
    assert not audio_player.is_playing


# ---------------------------------------------------------------------------
# AudioPlayer — streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_player_play_streaming_processes_all_chunks(audio_player, mock_sd):
    """play_streaming processes every chunk without error."""
    chunks = [np.zeros(100, dtype=np.float32) for _ in range(5)]
    await audio_player.play_streaming(chunks)
    assert not audio_player.is_playing


@pytest.mark.asyncio
async def test_audio_player_play_streaming_halts_on_stop_event(audio_player, mock_sd):
    """play_streaming stops early when the stop event is set beforehand."""
    chunks = [np.zeros(100, dtype=np.float32) for _ in range(10)]
    audio_player._stop_event.set()
    await audio_player.play_streaming(chunks)


# ---------------------------------------------------------------------------
# save_audio
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_audio_creates_nonempty_file(float_audio, tmp_path):
    """save_audio writes a non-empty WAV file to the given path."""
    out = tmp_path / "output.wav"

    await save_audio(float_audio, out, sample_rate=22050, format="wav")

    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.asyncio
async def test_save_audio_creates_parent_directories(float_audio, tmp_path):
    """save_audio creates nested parent directories when they do not exist."""
    nested = tmp_path / "a" / "b" / "c" / "audio.wav"

    await save_audio(float_audio, nested, sample_rate=22050)

    assert nested.exists()


# ---------------------------------------------------------------------------
# load_audio
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_audio_returns_tuple(float_audio, tmp_path):
    """load_audio returns a (audio, sample_rate) tuple."""
    path = tmp_path / "test.wav"
    await save_audio(float_audio, path, sample_rate=22050)

    result = await load_audio(path)

    assert isinstance(result, tuple)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_load_audio_correct_sample_rate(float_audio, tmp_path):
    """load_audio returns the sample rate that was used during save."""
    path = tmp_path / "test.wav"
    await save_audio(float_audio, path, sample_rate=22050)

    _, sr = await load_audio(path)

    assert sr == 22050


@pytest.mark.asyncio
async def test_load_audio_correct_shape(float_audio, tmp_path):
    """Loaded audio has the same number of samples as saved audio."""
    path = tmp_path / "test.wav"
    await save_audio(float_audio, path, sample_rate=22050)

    loaded, _ = await load_audio(path)

    assert loaded.shape == float_audio.shape


@pytest.mark.asyncio
async def test_load_audio_roundtrip_values(float_audio, tmp_path):
    """Loaded audio values are numerically close to the saved values."""
    path = tmp_path / "test.wav"
    await save_audio(float_audio, path, sample_rate=22050)

    loaded, _ = await load_audio(path)

    np.testing.assert_allclose(loaded, float_audio, atol=1e-4, rtol=1e-4)


@pytest.mark.asyncio
async def test_load_audio_nonexistent_file_raises(tmp_path):
    """load_audio raises an exception when the file does not exist."""
    missing = tmp_path / "nonexistent.wav"

    with pytest.raises((FileNotFoundError, OSError, Exception)):
        await load_audio(missing)


@pytest.mark.asyncio
async def test_load_audio_with_resampling(float_audio, tmp_path):
    """load_audio resamples the audio when target_sample_rate is provided."""
    pytest.importorskip("scipy")
    path = tmp_path / "test.wav"
    await save_audio(float_audio, path, sample_rate=22050)

    resampled, sr = await load_audio(path, target_sample_rate=11025)

    assert sr == 11025
    # 11025 samples at 22050 Hz -> 5512 samples at 11025 Hz
    expected_len = int(float_audio.shape[0] * 11025 / 22050)
    assert resampled.shape[0] == expected_len


# ---------------------------------------------------------------------------
# normalize_audio
# ---------------------------------------------------------------------------


def test_normalize_audio_output_within_unit_range():
    """normalize_audio clips output to [-1, 1]."""
    loud = np.array([5.0, -5.0, 3.0, -3.0, 0.0], dtype=np.float64)

    result = normalize_audio(loud)

    assert np.all(result >= -1.0)
    assert np.all(result <= 1.0)


def test_normalize_audio_silent_input_is_returned_unchanged():
    """normalize_audio returns a silent array unchanged."""
    silence = np.zeros(100, dtype=np.float64)

    result = normalize_audio(silence)

    np.testing.assert_array_equal(result, silence)


def test_normalize_audio_preserves_array_shape():
    """normalize_audio output has the same shape as the input."""
    audio = np.random.randn(500)

    result = normalize_audio(audio)

    assert result.shape == audio.shape


def test_normalize_audio_custom_target_db():
    """normalize_audio accepts a custom target_db without raising."""
    audio = np.ones(100, dtype=np.float64)

    result = normalize_audio(audio, target_db=-6.0)

    assert np.all(result >= -1.0)
    assert np.all(result <= 1.0)


def test_normalize_audio_nonzero_input_changes_level():
    """normalize_audio rescales non-silent audio to the target dB level."""
    # A constant 0.001-amplitude signal should be amplified to -20 dB RMS.
    audio = np.full(1000, 0.001)
    target_db = -20.0

    result = normalize_audio(audio, target_db=target_db)

    result_rms = np.sqrt(np.mean(result**2))
    expected_rms = 10 ** (target_db / 20)
    np.testing.assert_allclose(result_rms, expected_rms, rtol=1e-3)
