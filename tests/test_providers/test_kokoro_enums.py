"""
Tests for Kokoro TTS enums.
"""

from champi_tts.providers.kokoro.enums import (
    AudioFormat,
    LifecycleEvents,
    LoggingStrings,
    ModelEvents,
    ProcessingMode,
    TelemetryEvents,
    TextProcessingEvents,
    TTSEventTypes,
    VoiceLanguage,
)


class TestVoiceLanguage:
    """Tests for VoiceLanguage enum."""

    def test_american_english_value(self) -> None:
        """AMERICAN_ENGLISH has value 'a'."""
        assert VoiceLanguage.AMERICAN_ENGLISH.value == "a"

    def test_british_english_value(self) -> None:
        """BRITISH_ENGLISH has value 'b'."""
        assert VoiceLanguage.BRITISH_ENGLISH.value == "b"

    def test_from_voice_prefix_american_female(self) -> None:
        """'af_bella' resolves to American English."""
        result = VoiceLanguage.from_voice_prefix("af_bella")
        assert result == VoiceLanguage.AMERICAN_ENGLISH.value

    def test_from_voice_prefix_american_male(self) -> None:
        """'am_adam' resolves to American English."""
        result = VoiceLanguage.from_voice_prefix("am_adam")
        assert result == VoiceLanguage.AMERICAN_ENGLISH.value

    def test_from_voice_prefix_british_female(self) -> None:
        """'bf_alice' resolves to British English."""
        result = VoiceLanguage.from_voice_prefix("bf_alice")
        assert result == VoiceLanguage.BRITISH_ENGLISH.value

    def test_from_voice_prefix_british_male(self) -> None:
        """'bm_george' resolves to British English."""
        result = VoiceLanguage.from_voice_prefix("bm_george")
        assert result == VoiceLanguage.BRITISH_ENGLISH.value

    def test_from_voice_prefix_spanish(self) -> None:
        """'ef_' prefix resolves to Spanish."""
        result = VoiceLanguage.from_voice_prefix("ef_voice")
        assert result == VoiceLanguage.SPANISH.value

    def test_from_voice_prefix_french(self) -> None:
        """'ff_' prefix resolves to French."""
        result = VoiceLanguage.from_voice_prefix("ff_voice")
        assert result == VoiceLanguage.FRENCH.value

    def test_from_voice_prefix_japanese(self) -> None:
        """'jf_' prefix resolves to Japanese."""
        result = VoiceLanguage.from_voice_prefix("jf_voice")
        assert result == VoiceLanguage.JAPANESE.value

    def test_from_voice_prefix_portuguese(self) -> None:
        """'pf_' prefix resolves to Brazilian Portuguese."""
        result = VoiceLanguage.from_voice_prefix("pf_voice")
        assert result == VoiceLanguage.PORTUGUESE_BRAZIL.value

    def test_from_voice_prefix_mandarin(self) -> None:
        """'zf_' prefix resolves to Mandarin Chinese."""
        result = VoiceLanguage.from_voice_prefix("zf_voice")
        assert result == VoiceLanguage.MANDARIN_CHINESE.value

    def test_from_voice_prefix_hindi(self) -> None:
        """'hf_' prefix resolves to Hindi."""
        result = VoiceLanguage.from_voice_prefix("hf_voice")
        assert result == VoiceLanguage.HINDI.value

    def test_from_voice_prefix_italian(self) -> None:
        """'if_' prefix resolves to Italian."""
        result = VoiceLanguage.from_voice_prefix("if_voice")
        assert result == VoiceLanguage.ITALIAN.value

    def test_from_voice_prefix_male_variants(self) -> None:
        """Male variants ('em_', 'fm_', etc.) map to the same language as female."""
        assert (
            VoiceLanguage.from_voice_prefix("em_voice") == VoiceLanguage.SPANISH.value
        )
        assert VoiceLanguage.from_voice_prefix("fm_voice") == VoiceLanguage.FRENCH.value
        assert (
            VoiceLanguage.from_voice_prefix("jm_voice") == VoiceLanguage.JAPANESE.value
        )

    def test_from_voice_prefix_unknown_defaults_to_american(self) -> None:
        """Unknown prefix defaults to American English."""
        result = VoiceLanguage.from_voice_prefix("xx_voice")
        assert result == VoiceLanguage.AMERICAN_ENGLISH.value

    def test_from_voice_prefix_empty_string(self) -> None:
        """Empty voice name defaults to American English."""
        result = VoiceLanguage.from_voice_prefix("")
        assert result == VoiceLanguage.AMERICAN_ENGLISH.value

    def test_from_voice_prefix_no_underscore(self) -> None:
        """Voice name without underscore uses first character as prefix."""
        result = VoiceLanguage.from_voice_prefix("abc")
        # 'a' maps to AMERICAN_ENGLISH
        assert result == VoiceLanguage.AMERICAN_ENGLISH.value

    def test_get_all_codes_returns_list(self) -> None:
        """get_all_codes() returns a non-empty list."""
        codes = VoiceLanguage.get_all_codes()
        assert isinstance(codes, list)
        assert len(codes) > 0

    def test_get_all_codes_contains_basic_languages(self) -> None:
        """get_all_codes() includes the primary language codes."""
        codes = VoiceLanguage.get_all_codes()
        assert "a" in codes
        assert "b" in codes

    def test_extended_language_codes(self) -> None:
        """Extended language codes are defined correctly."""
        assert VoiceLanguage.ENGLISH_US.value == "en-us"
        assert VoiceLanguage.ENGLISH_GB.value == "en-gb"


class TestAudioFormat:
    """Tests for AudioFormat enum."""

    def test_wav_value(self) -> None:
        """WAV format has value 'wav'."""
        assert AudioFormat.WAV.value == "wav"

    def test_mp3_value(self) -> None:
        """MP3 format has value 'mp3'."""
        assert AudioFormat.MP3.value == "mp3"

    def test_flac_value(self) -> None:
        """FLAC format has value 'flac'."""
        assert AudioFormat.FLAC.value == "flac"

    def test_ogg_value(self) -> None:
        """OGG format has value 'ogg'."""
        assert AudioFormat.OGG.value == "ogg"

    def test_all_formats_unique(self) -> None:
        """All audio format values are unique."""
        values = [fmt.value for fmt in AudioFormat]
        assert len(values) == len(set(values))


class TestProcessingMode:
    """Tests for ProcessingMode enum."""

    def test_full_pipeline_value(self) -> None:
        """FULL_PIPELINE has value 'full_pipeline'."""
        assert ProcessingMode.FULL_PIPELINE.value == "full_pipeline"

    def test_streaming_value(self) -> None:
        """STREAMING has value 'streaming'."""
        assert ProcessingMode.STREAMING.value == "streaming"

    def test_batch_value(self) -> None:
        """BATCH has value 'batch'."""
        assert ProcessingMode.BATCH.value == "batch"

    def test_all_modes_unique(self) -> None:
        """All processing mode values are unique."""
        values = [mode.value for mode in ProcessingMode]
        assert len(values) == len(set(values))


class TestTTSEventTypes:
    """Tests for TTSEventTypes enum."""

    def test_lifecycle_event_value(self) -> None:
        """LIFECYCLE_EVENT has expected value."""
        assert TTSEventTypes.LIFECYCLE_EVENT.value == "lifecycle_event"

    def test_model_event_value(self) -> None:
        """MODEL_EVENT has expected value."""
        assert TTSEventTypes.MODEL_EVENT.value == "model_event"

    def test_all_event_types_unique(self) -> None:
        """All event type values are unique."""
        values = [et.value for et in TTSEventTypes]
        assert len(values) == len(set(values))


class TestLifecycleEvents:
    """Tests for LifecycleEvents enum."""

    def test_ready_value(self) -> None:
        """READY has value 'ready'."""
        assert LifecycleEvents.READY.value == "ready"

    def test_error_value(self) -> None:
        """ERROR has value 'error'."""
        assert LifecycleEvents.ERROR.value == "error"

    def test_speaking_value(self) -> None:
        """SPEAKING has value 'speaking'."""
        assert LifecycleEvents.SPEAKING.value == "speaking"

    def test_all_lifecycle_events_unique(self) -> None:
        """All lifecycle event values are unique."""
        values = [ev.value for ev in LifecycleEvents]
        assert len(values) == len(set(values))


class TestModelEvents:
    """Tests for ModelEvents enum."""

    def test_model_loaded_value(self) -> None:
        """MODEL_LOADED has value 'model_loaded'."""
        assert ModelEvents.MODEL_LOADED.value == "model_loaded"

    def test_warmup_complete_value(self) -> None:
        """WARMUP_COMPLETE has value 'warmup_complete'."""
        assert ModelEvents.WARMUP_COMPLETE.value == "warmup_complete"


class TestTextProcessingEvents:
    """Tests for TextProcessingEvents enum."""

    def test_text_processing_value(self) -> None:
        """TEXT_PROCESSING has value 'text_processing'."""
        assert TextProcessingEvents.TEXT_PROCESSING.value == "text_processing"

    def test_audio_generating_value(self) -> None:
        """AUDIO_GENERATING has value 'audio_generating'."""
        assert TextProcessingEvents.AUDIO_GENERATING.value == "audio_generating"


class TestTelemetryEvents:
    """Tests for TelemetryEvents enum."""

    def test_metrics_update_value(self) -> None:
        """METRICS_UPDATE has value 'metrics_update'."""
        assert TelemetryEvents.METRICS_UPDATE.value == "metrics_update"

    def test_all_telemetry_events_unique(self) -> None:
        """All telemetry event values are unique."""
        values = [ev.value for ev in TelemetryEvents]
        assert len(values) == len(set(values))


class TestLoggingStrings:
    """Tests for LoggingStrings enum."""

    def test_provider_initialized_is_string(self) -> None:
        """PROVIDER_INITIALIZED is a non-empty string."""
        assert len(LoggingStrings.PROVIDER_INITIALIZED.value) > 0

    def test_synthesis_failed_contains_format(self) -> None:
        """SYNTHESIS_FAILED contains a format placeholder."""
        assert "{}" in LoggingStrings.SYNTHESIS_FAILED.value

    def test_all_logging_strings_unique(self) -> None:
        """All logging string values are unique."""
        values = [s.value for s in LoggingStrings]
        assert len(values) == len(set(values))
