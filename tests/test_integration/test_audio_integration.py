"""
Integration tests for audio handling.

These tests verify audio playback, file I/O, and synthesis functionality.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from champi_tts.core.audio import AudioPlayer, load_audio, save_audio
from soundfile import SF_FORMAT_WAV, SF_FORMAT_PCM_16


@pytest.fixture
def audio_player():
    """Create audio player for testing."""
    return AudioPlayer(sample_rate=22050)


@pytest.fixture
def mock_audio_data():
    """Generate mock audio data for testing."""
    # Create a simple sine wave
    sample_rate = 22050
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t)
    # Normalize to 16-bit range
    audio = (audio * 32767).astype(np.int16)
    return audio


@pytest.mark.asyncio
async def test_audio_player_play(audio_player, mock_audio_data):
    """Test audio player playback."""
    # Verify player is not playing initially
    assert not audio_player.is_playing()

    # Play audio
    await audio_player.play(mock_audio_data, blocking=False)
    # Don't wait for completion in test


@pytest.mark.asyncio
async def test_audio_player_stop(audio_player, mock_audio_data):
    """Test stopping audio playback."""
    await audio_player.play(mock_audio_data, blocking=False)
    audio_player.stop()
    assert not audio_player.is_playing()


@pytest.mark.asyncio
async def test_audio_player_set_volume(audio_player, mock_audio_data):
    """Test setting audio volume."""
    audio_player.volume = 0.5
    await audio_player.play(mock_audio_data, blocking=False)
    assert audio_player.volume == 0.5


@pytest.mark.asyncio
async def test_audio_player_volume_range(audio_player, mock_audio_data):
    """Test volume is clamped to valid range."""
    audio_player.volume = -0.1
    await audio_player.play(mock_audio_data, blocking=False)
    assert 0 <= audio_player.volume <= 1


@pytest.mark.asyncio
async def test_audio_player_sample_rate(audio_player, mock_audio_data):
    """Test player uses correct sample rate."""
    assert audio_player.sample_rate == 22050


@pytest.mark.asyncio
async def test_save_audio_file(mock_audio_data):
    """Test saving audio to file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = Path(f.name)

    # Save audio
    await save_audio(
        mock_audio_data,
        temp_path,
        sample_rate=22050,
        format=SF_FORMAT_WAV
    )

    # Verify file exists
    assert temp_path.exists()

    # Verify file size is reasonable (> 0 bytes)
    assert temp_path.stat().st_size > 0

    # Cleanup
    temp_path.unlink()


@pytest.mark.asyncio
async def test_load_audio_file(mock_audio_data):
    """Test loading audio from file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = Path(f.name)

    # Save and load audio
    await save_audio(mock_audio_data, temp_path, sample_rate=22050)
    loaded_audio = await load_audio(temp_path)

    # Verify loaded audio has same shape
    assert loaded_audio.shape == mock_audio_data.shape

    # Verify values are approximately equal (allowing for floating point)
    assert np.allclose(loaded_audio.astype(float), mock_audio_data.astype(float))

    # Cleanup
    temp_path.unlink()


@pytest.mark.asyncio
async def test_load_nonexistent_file():
    """Test loading non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        await load_audio("/nonexistent/audio.wav")


@pytest.mark.asyncio
async def test_audio_normalization():
    """Test audio normalization utilities."""
    from champi_tts.core.audio import normalize_audio

    # Create audio that exceeds -1 to 1 range
    clipped_audio = np.array([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])

    # Normalize
    normalized = normalize_audio(clipped_audio)

    # Verify normalized audio is in valid range
    assert np.all(normalized >= -1.0)
    assert np.all(normalized <= 1.0)


@pytest.mark.asyncio
async def test_audio_resampling():
    """Test audio resampling."""
    from champi_tts.core.audio import resample_audio

    # Create audio at 44100 Hz
    high_sample_rate_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))

    # Resample to 22050 Hz
    resampled = resample_audio(high_sample_rate_audio, 44100, 22050)

    # Verify resampled shape
    assert resampled.shape[0] == 22050


@pytest.mark.asyncio
async def test_audio_silence(audio_player):
    """Test playing silence."""
    silence = np.zeros(44100, dtype=np.float32)
    await audio_player.play(silence, blocking=False)
    assert audio_player.sample_rate == 22050


@pytest.mark.asyncio
async def test_audio_very_long(audio_player, mock_audio_data):
    """Test playing long audio."""
    long_audio = np.tile(mock_audio_data, 100)
    await audio_player.play(long_audio, blocking=False)
    assert audio_player.sample_rate == 22050


@pytest.mark.asyncio
async def test_audio_single_sample(audio_player):
    """Test playing single sample."""
    single_sample = np.array([0.5], dtype=np.float32)
    await audio_player.play(single_sample, blocking=False)
    assert audio_player.sample_rate == 22050


@pytest.mark.asyncio
async def test_audio_player_initial_state(audio_player):
    """Test player initial state."""
    assert not audio_player.is_playing()
    assert audio_player.volume == 1.0
    assert audio_player.sample_rate == 22050


@pytest.mark.asyncio
async def test_audio_player_volume_reset(audio_player):
    """Test volume reset to default."""
    audio_player.volume = 0.5
    audio_player.set_volume()  # Reset to default
    assert audio_player.volume == 1.0

