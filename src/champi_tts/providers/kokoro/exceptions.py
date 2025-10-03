"""
Custom exceptions for Kokoro TTS service.
"""


class KokoroError(Exception):
    """Base exception for all Kokoro TTS errors"""

    pass


class KokoroInitializationError(KokoroError):
    """Raised when provider initialization fails"""

    pass


class KokoroModelError(KokoroError):
    """Raised when model loading or operations fail"""

    pass


class KokoroVoiceError(KokoroError):
    """Raised when voice loading or validation fails"""

    pass


class KokoroSynthesisError(KokoroError):
    """Raised when audio synthesis fails"""

    pass


class KokoroAudioError(KokoroError):
    """Raised when audio playback fails"""

    pass


class KokoroConfigurationError(KokoroError):
    """Raised when configuration is invalid"""

    pass


class KokoroTextProcessingError(KokoroError):
    """Raised when text processing fails"""

    pass


class KokoroFileError(KokoroError):
    """Raised when file operations fail"""

    pass
