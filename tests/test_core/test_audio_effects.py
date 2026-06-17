"""
Tests for AudioProcessor and process_audio utility.

All methods operate on numpy arrays and do not require audio hardware.
"""

import numpy as np
import pytest

from champi_tts.core.audio_effects import AudioProcessor, process_audio


@pytest.fixture
def processor() -> AudioProcessor:
    """Default AudioProcessor at 24 kHz."""
    return AudioProcessor(sample_rate=24000)


@pytest.fixture
def audio() -> np.ndarray:
    """One-second 440 Hz sine wave (unit amplitude)."""
    t = np.linspace(0, 1.0, 24000)
    return np.sin(2 * np.pi * 440 * t).astype(np.float64)


@pytest.fixture
def quiet_audio() -> np.ndarray:
    """Very quiet sine wave to test normalization."""
    t = np.linspace(0, 1.0, 24000)
    return (np.sin(2 * np.pi * 440 * t) * 0.001).astype(np.float64)


class TestAudioProcessorInit:
    """Tests for AudioProcessor constructor."""

    def test_default_sample_rate(self, processor: AudioProcessor) -> None:
        """Default sample rate is 24000 Hz."""
        assert processor.sample_rate == 24000

    def test_custom_sample_rate(self) -> None:
        """Custom sample rate is stored correctly."""
        p = AudioProcessor(sample_rate=16000)
        assert p.sample_rate == 16000


class TestNormalizeVolume:
    """Tests for AudioProcessor.normalize_volume()."""

    def test_output_clamped_to_valid_range(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """normalize_volume() clips result to [-1, 1]."""
        result = processor.normalize_volume(audio, target_db=-20.0)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_silence_returns_zeros(self, processor: AudioProcessor) -> None:
        """normalize_volume() leaves silence unchanged."""
        silence = np.zeros(100)
        result = processor.normalize_volume(silence)
        assert np.all(result == 0.0)

    def test_quiet_audio_amplified(
        self, processor: AudioProcessor, quiet_audio: np.ndarray
    ) -> None:
        """normalize_volume() raises level of quiet audio."""
        result = processor.normalize_volume(quiet_audio, target_db=-20.0)
        assert np.abs(result).max() > np.abs(quiet_audio).max()


class TestApplyGain:
    """Tests for AudioProcessor.apply_gain()."""

    def test_positive_gain_increases_peak(
        self, processor: AudioProcessor, quiet_audio: np.ndarray
    ) -> None:
        """Positive gain_db increases peak amplitude."""
        result = processor.apply_gain(quiet_audio, gain_db=20.0)
        assert np.abs(result).max() > np.abs(quiet_audio).max()

    def test_gain_clips_to_range(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """apply_gain() clips result to [-1, 1]."""
        result = processor.apply_gain(audio, gain_db=40.0)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_negative_gain_decreases_peak(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """Negative gain_db decreases peak amplitude."""
        original_max = np.abs(audio).max()
        result = processor.apply_gain(audio, gain_db=-20.0)
        assert np.abs(result).max() < original_max


class TestNormalizeRms:
    """Tests for AudioProcessor.normalize_rms()."""

    def test_adjusts_rms_to_target(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """normalize_rms() adjusts the RMS level to target_rms."""
        result = processor.normalize_rms(audio, target_rms=0.1)
        result_rms = np.sqrt(np.mean(result**2))
        assert abs(result_rms - 0.1) < 0.01

    def test_silence_returns_unchanged(self, processor: AudioProcessor) -> None:
        """normalize_rms() returns silence unchanged."""
        silence = np.zeros(100)
        result = processor.normalize_rms(silence, target_rms=0.1)
        assert np.all(result == 0.0)


class TestAddSilence:
    """Tests for AudioProcessor.add_silence()."""

    def test_length_increases_by_two_silence_periods(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """add_silence() prepends and appends silence."""
        original_len = len(audio)
        result = processor.add_silence(audio, duration=0.5)
        expected_silence_samples = int(0.5 * processor.sample_rate)
        assert len(result) == original_len + 2 * expected_silence_samples

    def test_custom_sample_rate_used_for_silence_length(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """add_silence() uses the provided sample_rate for duration calculation."""
        result = processor.add_silence(audio, duration=0.5, sample_rate=16000)
        expected_silence = int(0.5 * 16000)
        assert len(result) == len(audio) + 2 * expected_silence

    def test_silence_dtype_matches_input(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """add_silence() produces silence with the same dtype as input."""
        result = processor.add_silence(audio, duration=0.1)
        assert result.dtype == audio.dtype


class TestAddFade:
    """Tests for AudioProcessor.add_fade()."""

    def test_preserves_length(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """add_fade() does not change the number of samples."""
        result = processor.add_fade(audio, fade_duration=0.1)
        assert len(result) == len(audio)

    def test_returns_very_short_audio_unchanged(
        self, processor: AudioProcessor
    ) -> None:
        """add_fade() returns audio unchanged when it is shorter than two fade periods."""
        short = np.ones(10)
        result = processor.add_fade(short, fade_duration=0.1)
        np.testing.assert_array_equal(result, short)

    def test_fade_start_is_quiet(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """First sample of faded audio is near zero."""
        result = processor.add_fade(audio, fade_duration=0.1)
        assert abs(result[0]) < abs(audio[0])

    def test_custom_sample_rate(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """add_fade() respects an explicit sample_rate argument."""
        result = processor.add_fade(audio, fade_duration=0.1, sample_rate=16000)
        assert len(result) == len(audio)


class TestAddEcho:
    """Tests for AudioProcessor.add_echo()."""

    def test_very_short_delay_returns_unchanged(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """add_echo() returns the original array when delay_samples < 10."""
        result = processor.add_echo(audio, delay=0.0001)
        np.testing.assert_array_equal(result, audio)

    def test_preserves_length(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """add_echo() preserves the length of the audio array."""
        result = processor.add_echo(audio, delay=0.0001)
        assert len(result) == len(audio)

    def test_main_path_with_crafted_audio_length(
        self, processor: AudioProcessor
    ) -> None:
        """add_echo() main path works when N == 2*delay_samples + spreaded + 1.

        For delay=0.01s at 24000 Hz: delay_samples=240, spread=0.5 → spreaded=120.
        Required N = 2*240 + 120 + 1 = 601.
        """
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 601)).astype(np.float64)
        result = processor.add_echo(audio, delay=0.01, spread=0.5)
        assert len(result) == 601


class TestFilters:
    """Tests for lowpass and highpass filters."""

    def test_lowpass_preserves_length(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """apply_lowpass() keeps the number of samples the same."""
        result = processor.apply_lowpass(audio, cutoff_freq=4000)
        assert len(result) == len(audio)

    def test_highpass_preserves_length(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """apply_highpass() keeps the number of samples the same."""
        result = processor.apply_highpass(audio, cutoff_freq=200)
        assert len(result) == len(audio)

    def test_lowpass_attenuates_high_frequency(self, processor: AudioProcessor) -> None:
        """Lowpass filter attenuates energy above cutoff."""
        # Create audio with two components: 440 Hz (pass) and 8000 Hz (stop)
        t = np.linspace(0, 1.0, 24000)
        low_freq = np.sin(2 * np.pi * 440 * t)
        high_freq = np.sin(2 * np.pi * 8000 * t)
        mixed = (low_freq + high_freq).astype(np.float64)

        result = processor.apply_lowpass(mixed, cutoff_freq=1000)

        # High frequency energy should be reduced
        original_rms = np.sqrt(np.mean(mixed**2))
        result_rms = np.sqrt(np.mean(result**2))
        assert result_rms < original_rms

    def test_lowpass_custom_order(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """apply_lowpass() accepts a custom filter order."""
        result = processor.apply_lowpass(audio, cutoff_freq=4000, order=4)
        assert len(result) == len(audio)

    def test_highpass_custom_order(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """apply_highpass() accepts a custom filter order."""
        result = processor.apply_highpass(audio, cutoff_freq=200, order=4)
        assert len(result) == len(audio)


class TestApplyBassBoost:
    """Tests for AudioProcessor.apply_bass_boost()."""

    def test_preserves_length(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """apply_bass_boost() keeps the number of samples the same."""
        result = processor.apply_bass_boost(audio, boost_db=6.0)
        assert len(result) == len(audio)

    def test_returns_ndarray(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """apply_bass_boost() returns a numpy array."""
        result = processor.apply_bass_boost(audio)
        assert isinstance(result, np.ndarray)


class TestApplyReverb:
    """Tests for AudioProcessor.apply_reverb()."""

    def test_returns_ndarray(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """apply_reverb() returns a numpy array."""
        result = processor.apply_reverb(audio, decay=0.1)
        assert isinstance(result, np.ndarray)

    def test_output_longer_than_input(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """apply_reverb() output is longer due to convolution with impulse response."""
        result = processor.apply_reverb(audio, decay=0.1)
        assert len(result) > len(audio)

    def test_custom_sample_rate(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """apply_reverb() accepts an explicit sample_rate."""
        result = processor.apply_reverb(audio, decay=0.1, sample_rate=16000)
        assert isinstance(result, np.ndarray)

    def test_normalizes_output(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """apply_reverb() normalises the output so it is not silent."""
        result = processor.apply_reverb(audio, decay=0.1)
        rms = np.sqrt(np.mean(result**2))
        assert rms > 0.0


class TestApplyCompression:
    """Tests for AudioProcessor.apply_compression().

    apply_compression() uses a convolution window of int(0.01 * sample_rate) samples.
    When sample_rate=100 the window is 1, making input and output shapes equal.
    """

    def test_returns_clamped_array(self) -> None:
        """apply_compression() clips result to [-1, 1]."""
        processor = AudioProcessor(sample_rate=100)
        audio = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 100)).astype(np.float64)
        result = processor.apply_compression(audio)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_preserves_non_zero_values(self) -> None:
        """apply_compression() does not zero out non-silent audio."""
        processor = AudioProcessor(sample_rate=100)
        audio = np.ones(100, dtype=np.float64) * 0.5
        result = processor.apply_compression(audio)
        assert np.any(result != 0.0)

    def test_custom_threshold(self) -> None:
        """apply_compression() accepts custom threshold parameter."""
        processor = AudioProcessor(sample_rate=100)
        audio = np.random.randn(100).astype(np.float64) * 0.5
        result = processor.apply_compression(audio, threshold=-12)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)


class TestApplyBitcrush:
    """Tests for AudioProcessor.apply_bitcrush()."""

    def test_preserves_length(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """apply_bitcrush() keeps the number of samples the same."""
        result = processor.apply_bitcrush(audio * 0.5, bits=8)
        assert len(result) == len(audio)

    def test_returns_ndarray(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """apply_bitcrush() returns a numpy array."""
        result = processor.apply_bitcrush(audio * 0.5, bits=4)
        assert isinstance(result, np.ndarray)

    def test_custom_sample_rate(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """apply_bitcrush() accepts an explicit sample_rate."""
        result = processor.apply_bitcrush(audio * 0.5, bits=8, sample_rate=16000)
        assert len(result) == len(audio)


class TestChainEffects:
    """Tests for AudioProcessor.chain_effects()."""

    def test_empty_effects_returns_unchanged(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """chain_effects() with no effects returns the original array."""
        result = processor.chain_effects(audio, [])
        np.testing.assert_array_equal(result, audio)

    def test_normalize_effect(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """chain_effects() applies normalize effect."""
        result = processor.chain_effects(
            audio, [{"type": "normalize", "target_db": -20.0}]
        )
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_gain_effect(
        self, processor: AudioProcessor, quiet_audio: np.ndarray
    ) -> None:
        """chain_effects() applies gain effect."""
        result = processor.chain_effects(
            quiet_audio, [{"type": "gain", "gain_db": 20.0}]
        )
        assert np.abs(result).max() > np.abs(quiet_audio).max()

    def test_fade_effect(self, processor: AudioProcessor, audio: np.ndarray) -> None:
        """chain_effects() applies fade effect without changing length."""
        result = processor.chain_effects(
            audio, [{"type": "fade", "fade_duration": 0.05}]
        )
        assert len(result) == len(audio)

    def test_lowpass_effect(self, processor: AudioProcessor, audio: np.ndarray) -> None:
        """chain_effects() applies lowpass filter."""
        result = processor.chain_effects(
            audio, [{"type": "lowpass", "cutoff_freq": 4000}]
        )
        assert len(result) == len(audio)

    def test_highpass_effect(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """chain_effects() applies highpass filter."""
        result = processor.chain_effects(
            audio, [{"type": "highpass", "cutoff_freq": 200}]
        )
        assert len(result) == len(audio)

    def test_bass_boost_effect(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """chain_effects() applies bass_boost effect."""
        result = processor.chain_effects(
            audio, [{"type": "bass_boost", "boost_db": 6.0}]
        )
        assert len(result) == len(audio)

    def test_reverb_effect(self, processor: AudioProcessor, audio: np.ndarray) -> None:
        """chain_effects() applies reverb effect."""
        result = processor.chain_effects(audio, [{"type": "reverb", "decay": 0.1}])
        assert len(result) > 0

    def test_bitcrush_effect(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """chain_effects() applies bitcrush effect."""
        result = processor.chain_effects(audio * 0.5, [{"type": "bitcrush", "bits": 8}])
        assert len(result) == len(audio)

    def test_silence_effect_increases_length(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """chain_effects() with silence effect increases audio length."""
        result = processor.chain_effects(audio, [{"type": "silence", "duration": 0.1}])
        assert len(result) > len(audio)

    def test_echo_effect_short_delay_returns_unchanged(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """chain_effects() with very short echo delay returns audio unchanged."""
        result = processor.chain_effects(audio, [{"type": "echo", "delay": 0.0001}])
        np.testing.assert_array_equal(result, audio)

    def test_unknown_effect_type_is_skipped(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """chain_effects() silently skips unknown effect types."""
        result = processor.chain_effects(audio, [{"type": "unknown_effect"}])
        np.testing.assert_array_equal(result, audio)

    def test_multiple_effects_chained(
        self, processor: AudioProcessor, audio: np.ndarray
    ) -> None:
        """chain_effects() applies multiple effects in sequence."""
        effects = [
            {"type": "normalize", "target_db": -20.0},
            {"type": "gain", "gain_db": -3.0},
            {"type": "fade", "fade_duration": 0.05},
        ]
        result = processor.chain_effects(audio, effects)
        assert len(result) == len(audio)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_compression_effect_with_low_sample_rate(self) -> None:
        """chain_effects() applies compression when window_size is 1."""
        processor = AudioProcessor(sample_rate=100)
        audio = np.random.randn(100).astype(np.float64) * 0.5
        result = processor.chain_effects(audio, [{"type": "compression"}])
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_echo_effect_main_path(self) -> None:
        """chain_effects() echo effect covers main path with crafted audio length."""
        processor = AudioProcessor(sample_rate=24000)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 601)).astype(np.float64)
        result = processor.chain_effects(
            audio, [{"type": "echo", "delay": 0.01, "spread": 0.5}]
        )
        assert len(result) == 601


class TestProcessAudio:
    """Tests for the process_audio() convenience function."""

    def test_no_effects_returns_unchanged(self) -> None:
        """process_audio() without effects returns the original audio."""
        audio = np.random.randn(1000).astype(np.float64)
        result = process_audio(audio, sample_rate=24000)
        np.testing.assert_array_equal(result, audio)

    def test_no_effects_none_default(self) -> None:
        """process_audio() with effects=None returns the original audio."""
        audio = np.random.randn(1000).astype(np.float64)
        result = process_audio(audio, sample_rate=24000, effects=None)
        np.testing.assert_array_equal(result, audio)

    def test_with_effects_applies_chain(self) -> None:
        """process_audio() with effects delegates to chain_effects."""
        audio = np.random.randn(24000).astype(np.float64)
        effects = [{"type": "normalize"}]
        result = process_audio(audio, sample_rate=24000, effects=effects)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_custom_sample_rate_passed_to_processor(self) -> None:
        """process_audio() creates an AudioProcessor with the given sample_rate."""
        audio = np.random.randn(1000).astype(np.float64)
        effects = [{"type": "lowpass", "cutoff_freq": 2000}]
        result = process_audio(audio, sample_rate=16000, effects=effects)
        assert len(result) == len(audio)
