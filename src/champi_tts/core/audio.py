"""
Generic audio utilities for TTS playback and file operations.
"""

import asyncio
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger


class AudioPlayer:
    """Audio playback with interruption support."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._stream: sd.OutputStream | None = None
        self._is_playing = False
        self._stop_event = asyncio.Event()

    async def play(self, audio: np.ndarray, blocking: bool = True) -> None:
        """
        Play audio data.

        Args:
            audio: Audio data as numpy array
            blocking: Whether to wait for playback to complete
        """
        try:
            self._is_playing = True
            self._stop_event.clear()

            if blocking:
                sd.play(audio, self.sample_rate)
                sd.wait()
            else:
                sd.play(audio, self.sample_rate)

        except Exception as e:
            logger.error(f"Audio playback error: {e}")
            raise
        finally:
            self._is_playing = False

    async def play_streaming(self, audio_chunks: list[np.ndarray]) -> None:
        """
        Play audio chunks in sequence.

        Args:
            audio_chunks: List of audio chunks
        """
        for chunk in audio_chunks:
            if self._stop_event.is_set():
                break
            await self.play(chunk, blocking=True)

    def stop(self) -> None:
        """Stop current playback."""
        self._stop_event.set()
        sd.stop()
        self._is_playing = False

    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing


async def save_audio(
    audio: np.ndarray,
    output_path: str | Path,
    sample_rate: int = 24000,
    format: str = "wav",
) -> None:
    """
    Save audio to file.

    Args:
        audio: Audio data as numpy array
        output_path: Output file path
        sample_rate: Audio sample rate
        format: Audio format (wav, mp3, etc.)
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(str(output_path), audio, sample_rate, format=format.upper())
        logger.info(f"Audio saved to {output_path}")

    except Exception as e:
        logger.error(f"Error saving audio: {e}")
        raise


async def load_audio(
    file_path: str | Path,
    target_sample_rate: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Load audio from file.

    Args:
        file_path: Audio file path
        target_sample_rate: Resample to this rate if specified

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        audio, sample_rate = sf.read(str(file_path))

        if target_sample_rate and sample_rate != target_sample_rate:
            # Simple resampling (could use librosa for better quality)
            import scipy.signal as signal

            num_samples = int(len(audio) * target_sample_rate / sample_rate)
            audio = signal.resample(audio, num_samples)
            sample_rate = target_sample_rate

        return audio, sample_rate

    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        raise


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target dB level.

    Args:
        audio: Audio data
        target_db: Target dB level

    Returns:
        Normalized audio
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio

    # Calculate target RMS from dB
    target_rms = 10 ** (target_db / 20)

    # Normalize
    normalized = audio * (target_rms / rms)

    # Clip to prevent overflow
    return np.clip(normalized, -1.0, 1.0)
