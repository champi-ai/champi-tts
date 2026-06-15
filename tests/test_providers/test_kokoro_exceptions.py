"""
Tests for Kokoro TTS custom exception hierarchy.
"""

import pytest

from champi_tts.providers.kokoro.exceptions import (
    KokoroAudioError,
    KokoroConfigurationError,
    KokoroError,
    KokoroFileError,
    KokoroInitializationError,
    KokoroModelError,
    KokoroSynthesisError,
    KokoroTextProcessingError,
    KokoroVoiceError,
)


class TestKokoroErrorHierarchy:
    """All Kokoro exceptions must subclass KokoroError (and therefore Exception)."""

    def test_kokoro_error_is_exception(self) -> None:
        """KokoroError subclasses Exception."""
        assert issubclass(KokoroError, Exception)

    def test_initialization_error_is_kokoro_error(self) -> None:
        """KokoroInitializationError subclasses KokoroError."""
        assert issubclass(KokoroInitializationError, KokoroError)

    def test_model_error_is_kokoro_error(self) -> None:
        """KokoroModelError subclasses KokoroError."""
        assert issubclass(KokoroModelError, KokoroError)

    def test_voice_error_is_kokoro_error(self) -> None:
        """KokoroVoiceError subclasses KokoroError."""
        assert issubclass(KokoroVoiceError, KokoroError)

    def test_synthesis_error_is_kokoro_error(self) -> None:
        """KokoroSynthesisError subclasses KokoroError."""
        assert issubclass(KokoroSynthesisError, KokoroError)

    def test_audio_error_is_kokoro_error(self) -> None:
        """KokoroAudioError subclasses KokoroError."""
        assert issubclass(KokoroAudioError, KokoroError)

    def test_configuration_error_is_kokoro_error(self) -> None:
        """KokoroConfigurationError subclasses KokoroError."""
        assert issubclass(KokoroConfigurationError, KokoroError)

    def test_text_processing_error_is_kokoro_error(self) -> None:
        """KokoroTextProcessingError subclasses KokoroError."""
        assert issubclass(KokoroTextProcessingError, KokoroError)

    def test_file_error_is_kokoro_error(self) -> None:
        """KokoroFileError subclasses KokoroError."""
        assert issubclass(KokoroFileError, KokoroError)


class TestKokoroErrorRaiseable:
    """All Kokoro exceptions can be raised and caught."""

    def test_raise_kokoro_error(self) -> None:
        with pytest.raises(KokoroError):
            raise KokoroError("base error")

    def test_raise_initialization_error(self) -> None:
        with pytest.raises(KokoroInitializationError):
            raise KokoroInitializationError("init failed")

    def test_raise_model_error(self) -> None:
        with pytest.raises(KokoroModelError):
            raise KokoroModelError("model failed")

    def test_raise_voice_error(self) -> None:
        with pytest.raises(KokoroVoiceError):
            raise KokoroVoiceError("voice not found")

    def test_raise_synthesis_error(self) -> None:
        with pytest.raises(KokoroSynthesisError):
            raise KokoroSynthesisError("synthesis failed")

    def test_raise_audio_error(self) -> None:
        with pytest.raises(KokoroAudioError):
            raise KokoroAudioError("audio playback failed")

    def test_raise_configuration_error(self) -> None:
        with pytest.raises(KokoroConfigurationError):
            raise KokoroConfigurationError("bad config")

    def test_raise_text_processing_error(self) -> None:
        with pytest.raises(KokoroTextProcessingError):
            raise KokoroTextProcessingError("text processing failed")

    def test_raise_file_error(self) -> None:
        with pytest.raises(KokoroFileError):
            raise KokoroFileError("file not found")


class TestKokoroErrorCatchAsBase:
    """Subclass exceptions are catchable as KokoroError."""

    def test_initialization_error_caught_as_kokoro_error(self) -> None:
        with pytest.raises(KokoroError):
            raise KokoroInitializationError("init failed")

    def test_synthesis_error_caught_as_kokoro_error(self) -> None:
        with pytest.raises(KokoroError):
            raise KokoroSynthesisError("synthesis failed")

    def test_voice_error_caught_as_exception(self) -> None:
        with pytest.raises(Exception):
            raise KokoroVoiceError("voice error")


class TestKokoroErrorMessages:
    """Exception messages are preserved correctly."""

    def test_error_message_preserved(self) -> None:
        """The exception message is accessible via str()."""
        msg = "something went wrong"
        error = KokoroError(msg)
        assert msg in str(error)

    def test_initialization_error_message(self) -> None:
        msg = "provider initialization failed"
        error = KokoroInitializationError(msg)
        assert msg in str(error)

    def test_empty_message_allowed(self) -> None:
        """Exceptions can be raised without a message."""
        error = KokoroError()
        assert isinstance(error, KokoroError)
