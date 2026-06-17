"""
Tests for configuration validation in BaseTTSConfig and KokoroConfig.
"""

import pytest

from champi_tts.core.base_config import BaseTTSConfig
from champi_tts.providers.kokoro.config import (
    SPEED_MAX,
    SPEED_MIN,
    VALID_LANGUAGE_CODES,
    VALID_VOICE_PREFIXES,
    KokoroConfig,
)

# ---------------------------------------------------------------------------
# Minimal concrete subclass used to exercise BaseTTSConfig.validate() directly.
# ---------------------------------------------------------------------------


class _ConcreteConfig(BaseTTSConfig):
    """Concrete subclass that delegates validate() to the base implementation."""

    @classmethod
    def from_env(cls) -> "_ConcreteConfig":
        return cls()

    def validate(self) -> bool:
        return super().validate()


# ---------------------------------------------------------------------------
# BaseTTSConfig.validate() tests
# ---------------------------------------------------------------------------


class TestBaseTTSConfigValidate:
    """Tests for BaseTTSConfig.validate()."""

    def test_valid_defaults_return_true(self):
        """Default values pass validation without error."""
        cfg = _ConcreteConfig()
        assert cfg.validate() is True

    def test_raises_for_zero_sample_rate(self):
        """sample_rate of 0 is rejected."""
        cfg = _ConcreteConfig()
        cfg.sample_rate = 0
        with pytest.raises(ValueError, match="sample_rate"):
            cfg.validate()

    def test_raises_for_negative_sample_rate(self):
        """Negative sample_rate is rejected."""
        cfg = _ConcreteConfig()
        cfg.sample_rate = -8000
        with pytest.raises(ValueError, match="sample_rate"):
            cfg.validate()

    def test_raises_for_zero_default_speed(self):
        """default_speed of 0 is rejected."""
        cfg = _ConcreteConfig()
        cfg.default_speed = 0.0
        with pytest.raises(ValueError, match="default_speed"):
            cfg.validate()

    def test_raises_for_negative_default_speed(self):
        """Negative default_speed is rejected."""
        cfg = _ConcreteConfig()
        cfg.default_speed = -1.0
        with pytest.raises(ValueError, match="default_speed"):
            cfg.validate()

    def test_raises_for_chunk_size_below_minimum(self):
        """streaming_chunk_size below 50 is rejected."""
        cfg = _ConcreteConfig()
        cfg.streaming_chunk_size = 49
        with pytest.raises(ValueError, match="streaming_chunk_size"):
            cfg.validate()

    def test_accepts_chunk_size_at_minimum(self):
        """streaming_chunk_size of exactly 50 is accepted."""
        cfg = _ConcreteConfig()
        cfg.streaming_chunk_size = 50
        assert cfg.validate() is True

    def test_raises_for_unknown_audio_format(self):
        """Unrecognised audio_format is rejected."""
        cfg = _ConcreteConfig()
        cfg.audio_format = "aiff"
        with pytest.raises(ValueError, match="audio_format"):
            cfg.validate()

    @pytest.mark.parametrize("fmt", ["wav", "mp3", "flac", "pcm", "ogg"])
    def test_accepts_all_valid_audio_formats(self, fmt):
        """Every supported audio format passes validation."""
        cfg = _ConcreteConfig()
        cfg.audio_format = fmt
        assert cfg.validate() is True


# ---------------------------------------------------------------------------
# KokoroConfig.validate() tests
# ---------------------------------------------------------------------------


class TestKokoroConfigValidate:
    """Tests for KokoroConfig.validate()."""

    def test_valid_defaults_return_true(self):
        """Default KokoroConfig passes validation."""
        cfg = KokoroConfig()
        assert cfg.validate() is True

    # -- speed -----------------------------------------------------------

    def test_speed_below_minimum_resets_to_default(self):
        """Speed below SPEED_MIN is silently reset to 1.0."""
        cfg = KokoroConfig(default_speed=SPEED_MIN - 0.1)
        assert cfg.default_speed == 1.0

    def test_speed_above_maximum_resets_to_default(self):
        """Speed above SPEED_MAX is silently reset to 1.0."""
        cfg = KokoroConfig(default_speed=SPEED_MAX + 0.1)
        assert cfg.default_speed == 1.0

    def test_accepts_speed_at_minimum_boundary(self):
        """Speed exactly at SPEED_MIN is accepted."""
        cfg = KokoroConfig(default_speed=SPEED_MIN)
        assert cfg.default_speed == SPEED_MIN

    def test_accepts_speed_at_maximum_boundary(self):
        """Speed exactly at SPEED_MAX is accepted."""
        cfg = KokoroConfig(default_speed=SPEED_MAX)
        assert cfg.default_speed == SPEED_MAX

    def test_accepts_speed_within_range(self):
        """Speed clearly within range is accepted."""
        cfg = KokoroConfig(default_speed=1.2)
        assert cfg.default_speed == 1.2

    # -- voice -----------------------------------------------------------

    def test_raises_for_voice_with_invalid_prefix(self):
        """Voice name with unknown prefix raises ValueError."""
        with pytest.raises(ValueError, match="voice"):
            KokoroConfig(default_voice="xx_unknown")

    def test_raises_for_voice_without_underscore(self):
        """Voice name without underscore separator raises ValueError."""
        with pytest.raises(ValueError, match="voice"):
            KokoroConfig(default_voice="afbella")

    def test_error_message_lists_valid_prefixes(self):
        """Error message for invalid voice name includes the valid prefixes."""
        with pytest.raises(ValueError, match="af"):
            KokoroConfig(default_voice="bad_name")

    def test_accepts_empty_voice_name(self):
        """Empty string for default_voice is accepted (auto-select)."""
        cfg = KokoroConfig(default_voice="")
        assert cfg.default_voice == ""

    @pytest.mark.parametrize("prefix", sorted(VALID_VOICE_PREFIXES))
    def test_accepts_all_valid_voice_prefixes(self, prefix):
        """Every valid voice prefix is accepted."""
        cfg = KokoroConfig(default_voice=f"{prefix}_testvoice")
        assert cfg.default_voice.startswith(prefix)

    # -- language --------------------------------------------------------

    def test_invalid_language_code_resets_to_a(self):
        """Unknown language code is silently reset to 'a'."""
        cfg = KokoroConfig(default_language="xyz")
        assert cfg.default_language == "a"

    @pytest.mark.parametrize("code", sorted(VALID_LANGUAGE_CODES))
    def test_accepts_all_valid_language_codes(self, code):
        """Every valid language code is accepted."""
        cfg = KokoroConfig(default_language=code)
        assert cfg.default_language == code

    # -- streaming_chunk_size --------------------------------------------

    def test_chunk_size_below_minimum_resets_to_default(self):
        """streaming_chunk_size below 50 is silently reset to 200."""
        cfg = KokoroConfig(streaming_chunk_size=10)
        assert cfg.streaming_chunk_size == 200

    def test_accepts_chunk_size_at_minimum(self):
        """streaming_chunk_size of exactly 50 is accepted."""
        cfg = KokoroConfig(streaming_chunk_size=50)
        assert cfg.streaming_chunk_size == 50

    # -- max_text_length -------------------------------------------------

    def test_raises_for_zero_max_text_length(self):
        """max_text_length of 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_text_length"):
            KokoroConfig(max_text_length=0)

    def test_raises_for_negative_max_text_length(self):
        """Negative max_text_length raises ValueError."""
        with pytest.raises(ValueError, match="max_text_length"):
            KokoroConfig(max_text_length=-1)

    # -- audio_format ----------------------------------------------------

    def test_raises_for_unsupported_audio_format(self):
        """Unsupported audio_format raises ValueError."""
        with pytest.raises(ValueError, match="audio_format"):
            KokoroConfig(audio_format="aiff")

    @pytest.mark.parametrize("fmt", ["mp3", "opus", "flac", "wav", "pcm"])
    def test_accepts_all_supported_audio_formats(self, fmt):
        """Every Kokoro-supported audio format is accepted."""
        cfg = KokoroConfig(audio_format=fmt)
        assert cfg.audio_format == fmt

    # -- construction integration ----------------------------------------

    def test_construction_resets_invalid_speed(self):
        """Construction resets out-of-range speed to 1.0."""
        cfg = KokoroConfig(default_speed=5.0)
        assert cfg.default_speed == 1.0

    def test_construction_raises_with_invalid_voice(self):
        """Construction fails immediately when voice name is malformed."""
        with pytest.raises(ValueError, match="voice"):
            KokoroConfig(default_voice="invalid")

    def test_construction_resets_invalid_language(self):
        """Construction resets an unknown language code to 'a'."""
        cfg = KokoroConfig(default_language="zz")
        assert cfg.default_language == "a"

    def test_validate_called_at_construction(self):
        """validate() is invoked by __post_init__ and silently resets invalid args."""
        cfg = KokoroConfig(default_speed=0.1)
        assert cfg.default_speed == 1.0

    def test_explicit_validate_call_returns_true(self):
        """Calling validate() on a valid config returns True."""
        cfg = KokoroConfig()
        result = cfg.validate()
        assert result is True
