"""
Audio effects and processing utilities for Champi TTS.

This module provides audio enhancement capabilities:
- Volume normalization
- Audio reverb
- Echo/Delay effects
- Equalization
- Compression
"""

import asyncio
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal as signal
import soundfile as sf
from loguru import logger

from .audio import AudioPlayer, load_audio


class AudioProcessor:
    """Audio processor with various effects."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    def normalize_volume(
        self,
        audio: np.ndarray,
        target_db: float = -20.0,
    ) -> np.ndarray:
        """
        Normalize audio volume to target dB level.

        Args:
            audio: Audio data
            target_db: Target dB level (default -20.0)

        Returns:
            Normalized audio
        """
        rms = np.sqrt(np.mean(audio**2))
        if rms == 0:
            return audio

        target_rms = 10 ** (target_db / 20)
        normalized = audio * (target_rms / rms)
        return np.clip(normalized, -1.0, 1.0)

    def apply_gain(self, audio: np.ndarray, gain_db: float) -> np.ndarray:
        """
        Apply gain to audio.

        Args:
            audio: Audio data
            gain_db: Gain in dB

        Returns:
            Gained audio
        """
        gain_factor = 10 ** (gain_db / 20)
        return np.clip(audio * gain_factor, -1.0, 1.0)

    def normalize_rms(
        self,
        audio: np.ndarray,
        target_rms: float = 0.02,
    ) -> np.ndarray:
        """
        Normalize audio RMS level.

        Args:
            audio: Audio data
            target_rms: Target RMS level

        Returns:
            Normalized audio
        """
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms == 0:
            return audio

        return audio * np.sqrt(target_rms / current_rms)

    def add_silence(
        self,
        audio: np.ndarray,
        duration: float = 0.5,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """
        Add silence before/after audio.

        Args:
            audio: Audio data
            duration: Silence duration in seconds
            sample_rate: Sample rate (uses instance if None)

        Returns:
            Audio with silence
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        silence_samples = int(duration * sample_rate)
        silence = np.zeros(silence_samples, dtype=audio.dtype)

        return np.concatenate([silence, audio, silence])

    def add_fade(
        self,
        audio: np.ndarray,
        fade_duration: float = 0.1,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """
        Add fade in/out to audio.

        Args:
            audio: Audio data
            fade_duration: Fade duration in seconds
            sample_rate: Sample rate (uses instance if None)

        Returns:
            Audio with fade
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        fade_samples = int(fade_duration * sample_rate)

        if len(audio) < 2 * fade_samples:
            return audio

        fade = np.linspace(0, 1, fade_samples)

        # Fade in at start
        start_audio = audio[:fade_samples] * fade
        # Fade out at end
        end_audio = audio[-fade_samples:] * (1 - fade[::-1])
        # Middle remains unchanged
        middle_audio = audio[fade_samples:-fade_samples]

        return np.concatenate([start_audio, middle_audio, end_audio])

    def add_echo(
        self,
        audio: np.ndarray,
        delay: float = 0.5,
        decay: float = 0.7,
        spread: float = 0.5,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """
        Add echo effect to audio.

        Args:
            audio: Audio data
            delay: Delay in seconds
            decay: Decay factor (0-1)
            spread: Spread for stereo (0-1)
            sample_rate: Sample rate (uses instance if None)

        Returns:
            Audio with echo
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        delay_samples = int(delay * sample_rate)

        if delay_samples < 10:
            return audio

        # Apply multiple echoes with decreasing amplitude
        result = audio.copy()
        for i in range(3):
            decayed = decay ** i
            spreaded = int(spread * delay_samples)
            echo = result[:-delay_samples - spreaded]
            result[-spreaded - delay_samples - 1 : -spreaded] += echo * decayed

        return result

    def apply_lowpass(
        self,
        audio: np.ndarray,
        cutoff_freq: float = 8000,
        order: int = 2,
    ) -> np.ndarray:
        """
        Apply lowpass filter to audio.

        Args:
            audio: Audio data
            cutoff_freq: Cutoff frequency in Hz
            order: Filter order

        Returns:
            Filtered audio
        """
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist

        # Use Butterworth filter
        b, a = signal.butter(order, normalized_cutoff, btype="low")
        filtered = signal.filtfilt(b, a, audio)

        return filtered

    def apply_highpass(
        self,
        audio: np.ndarray,
        cutoff_freq: float = 100,
        order: int = 2,
    ) -> np.ndarray:
        """
        Apply highpass filter to audio.

        Args:
            audio: Audio data
            cutoff_freq: Cutoff frequency in Hz
            order: Filter order

        Returns:
            Filtered audio
        """
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist

        b, a = signal.butter(order, normalized_cutoff, btype="high")
        filtered = signal.filtfilt(b, a, audio)

        return filtered

    def apply_bass_boost(
        self,
        audio: np.ndarray,
        boost_db: float = 10.0,
    ) -> np.ndarray:
        """
        Apply bass boost to audio.

        Args:
            audio: Audio data
            boost_db: Boost amount in dB

        Returns:
            Boosted audio
        """
        # Simple bass boost: amplify low frequencies
        half_rate = self.sample_rate // 2
        freq_range = int(100 * 2 * np.pi / self.sample_rate)

        low_freq_mask = np.arange(len(audio)) < freq_range
        high_freq_mask = ~low_freq_mask

        # Calculate boost factor
        boost_factor = 10 ** (boost_db / 40)

        # Apply boost to low frequencies
        low_freq = audio[low_freq_mask] * boost_factor

        return np.concatenate([low_freq, audio[high_freq_mask]])

    def apply_compression(
        self,
        audio: np.ndarray,
        threshold: float = -24,
        ratio: float = 4.0,
        attack: float = 0.01,
        release: float = 0.1,
    ) -> np.ndarray:
        """
        Apply dynamic compression to audio.

        Args:
            audio: Audio data
            threshold: Compression threshold in dB
            ratio: Compression ratio
            attack: Attack time in seconds
            release: Release time in seconds

        Returns:
            Compressed audio
        """
        # Calculate RMS
        window_size = int(0.01 * self.sample_rate)  # 10ms window

        rms = np.sqrt(np.convolve(audio**2, np.ones(window_size), mode="valid") / window_size)
        rms_db = 20 * np.log10(rms + 1e-10)

        # Calculate gain reduction
        gain_reduction = np.maximum(0, (rms_db - threshold) * ratio / (ratio + 1))

        # Apply gain reduction
        compressed = audio / (1 + gain_reduction)

        return np.clip(compressed, -1.0, 1.0)

    def apply_reverb(
        self,
        audio: np.ndarray,
        decay: float = 1.5,
        early_reflections: int = 5,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply reverb effect to audio.

        Args:
            audio: Audio data
            decay: Reverb decay time
            early_reflections: Number of early reflections
            sample_rate: Sample rate (uses instance if None)

        Returns:
            Audio with reverb
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        # Generate impulse response
        ir_length = int(decay * sample_rate)
        impulse = np.zeros(ir_length)

        # Add early reflections
        for i in range(early_reflections):
            delay = (i + 1) * (decay / early_reflections) / 2
            impulse[int(delay * sample_rate)] = 1 / (i + 1)

        # Add density
        impulse[: int(decay * sample_rate / 10)] += np.random.normal(
            0, 0.1, int(decay * sample_rate / 10)
        )

        # Convolve
        reverb = signal.convolve(audio, impulse, mode="full")

        # Normalize
        rms = np.sqrt(np.mean(reverb**2))
        if rms > 0:
            reverb = reverb / rms

        return reverb

    def apply_bitcrush(
        self,
        audio: np.ndarray,
        bits: int = 8,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply bitcrush effect (lo-fi).

        Args:
            audio: Audio data
            bits: Bit depth
            sample_rate: Sample rate (uses instance if None)

        Returns:
            Bitcrushed audio
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        max_val = 2**bits - 1
        quantized = np.clip((audio + 1) * max_val / 2, 0, max_val).astype(np.uint8)

        return quantized.astype(audio.dtype) / max_val

    def chain_effects(
        self,
        audio: np.ndarray,
        effects: list[dict[str, any]],
    ) -> np.ndarray:
        """
        Chain multiple effects together.

        Args:
            audio: Audio data
            effects: List of effect configurations

        Example:
            effects = [
                {"type": "normalize", "target_db": -20},
                {"type": "fade", "duration": 0.1},
                {"type": "volume", "gain_db": -3},
            ]

        Returns:
            Processed audio
        """
        result = audio

        for effect in effects:
            effect_type = effect.get("type")
            kwargs = {k: v for k, v in effect.items() if k != "type"}

            if effect_type == "normalize":
                result = self.normalize_volume(result, **kwargs)
            elif effect_type == "gain":
                result = self.apply_gain(result, **kwargs)
            elif effect_type == "fade":
                result = self.add_fade(result, **kwargs)
            elif effect_type == "silence":
                result = self.add_silence(result, **kwargs)
            elif effect_type == "echo":
                result = self.add_echo(result, **kwargs)
            elif effect_type == "lowpass":
                result = self.apply_lowpass(result, **kwargs)
            elif effect_type == "highpass":
                result = self.apply_highpass(result, **kwargs)
            elif effect_type == "bass_boost":
                result = self.apply_bass_boost(result, **kwargs)
            elif effect_type == "compression":
                result = self.apply_compression(result, **kwargs)
            elif effect_type == "reverb":
                result = self.apply_reverb(result, **kwargs)
            elif effect_type == "bitcrush":
                result = self.apply_bitcrush(result, **kwargs)

        return result

    async def save_processed(
        self,
        audio: np.ndarray,
        output_path: str | Path,
        sample_rate: Optional[int] = None,
        format: str = "wav",
    ) -> None:
        """
        Save processed audio to file.

        Args:
            audio: Audio data
            output_path: Output file path
            sample_rate: Sample rate for saving
            format: Audio format
        """
        await load_audio.__func__(audio, output_path, sample_rate or self.sample_rate, format)


def process_audio(
    audio: np.ndarray,
    sample_rate: int = 24000,
    effects: Optional[list[dict[str, any]]] = None,
) -> np.ndarray:
    """
    Convenience function to process audio with effects.

    Args:
        audio: Audio data
        sample_rate: Sample rate
        effects: List of effect configurations

    Returns:
        Processed audio
    """
    processor = AudioProcessor(sample_rate=sample_rate)

    if effects:
        return processor.chain_effects(audio, effects)

    return audio
