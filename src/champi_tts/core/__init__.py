"""
Core TTS abstractions and utilities.
"""

from champi_tts.core.audio import AudioPlayer, load_audio, normalize_audio, save_audio
from champi_tts.core.base_config import BaseTTSConfig
from champi_tts.core.base_provider import BaseTTSProvider
from champi_tts.core.base_synthesizer import BaseSynthesizer

__all__ = [
    "AudioPlayer",
    "BaseSynthesizer",
    "BaseTTSConfig",
    "BaseTTSProvider",
    "load_audio",
    "normalize_audio",
    "save_audio",
]
