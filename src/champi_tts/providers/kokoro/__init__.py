"""
Kokoro TTS provider for champi-tts.

This module provides direct integration with Kokoro TTS engine,
extracted from Kokoro-FastAPI for standalone usage.
"""

from champi_tts.providers.kokoro.config import KokoroConfig
from champi_tts.providers.kokoro.enums import (
    LifecycleEvents,
    TelemetryEvents,
    TextProcessingEvents,
    TTSEventTypes,
    VoiceLanguage,
)
from champi_tts.providers.kokoro.events import TTSSignalManager
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
from champi_tts.providers.kokoro.inference import KokoroInference
from champi_tts.providers.kokoro.models import ModelDownloader, VoiceManager
from champi_tts.providers.kokoro.provider import KokoroProvider

__all__ = [
    "KokoroAudioError",
    # Core components
    "KokoroConfig",
    "KokoroConfigurationError",
    # Exceptions
    "KokoroError",
    "KokoroFileError",
    "KokoroInference",
    "KokoroInitializationError",
    "KokoroModelError",
    "KokoroProvider",
    "KokoroSynthesisError",
    "KokoroTextProcessingError",
    "KokoroVoiceError",
    "LifecycleEvents",
    "ModelDownloader",
    "ProcessingEvents",
    "TTSEventTypes",
    # Event system
    "TTSSignalManager",
    "TelemetryEvents",
    # Enums
    "VoiceLanguage",
    # Model management
    "VoiceManager",
]
