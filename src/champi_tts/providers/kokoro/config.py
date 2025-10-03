"""
Configuration management for Kokoro TTS integration.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class KokoroConfig:
    """Configuration for Kokoro TTS provider"""

    # Device and performance settings
    use_gpu: bool = True
    force_cpu: bool = False

    # Model settings
    default_language: str = "a"  # 'a' = en-us, 'b' = en-gb
    default_speed: float = 1.0
    default_voice: str = "am_adam"

    # Text processing
    normalize_text: bool = True
    advanced_normalization: bool = True

    # Streaming settings
    streaming_chunk_size: int = 200
    enable_streaming: bool = True

    # Audio settings
    sample_rate: int = 24000
    audio_format: str = "wav"

    # TTS configuration
    available_voices: list = field(default_factory=lambda: ["am_adam"])
    available_models: list = field(default_factory=lambda: ["tts-1"])
    tts_audio_format: str = "pcm"  # Default format for TTS output

    # Service management
    auto_start: bool = False  # Auto-start Kokoro service

    # Initialization settings
    warmup_on_init: bool = True
    auto_download_model: bool = True

    # Directories (will be auto-set if None)
    model_dir: str | None = None
    voice_dir: str | None = None
    cache_dir: str | None = None

    # Advanced settings
    memory_management: bool = True
    max_text_length: int = 100000
    retry_on_failure: bool = True

    def __post_init__(self):
        """Post-initialization validation"""
        # Validate language code
        valid_languages = ["a", "b", "en", "en-us", "en-gb"]
        if self.default_language not in valid_languages:
            logger.warning(f"Invalid language code: {self.default_language}, using 'a'")
            self.default_language = "a"

        # Validate speed
        if not (0.1 <= self.default_speed <= 3.0):
            logger.warning(f"Invalid speed: {self.default_speed}, using 1.0")
            self.default_speed = 1.0

        # Validate chunk size
        if self.streaming_chunk_size < 50:
            logger.warning(
                f"Chunk size too small: {self.streaming_chunk_size}, using 200"
            )
            self.streaming_chunk_size = 200

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "KokoroConfig":
        """Create config from dictionary"""
        # Filter out unknown keys
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}

        return cls(**filtered_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary"""
        result = {}
        for field_name, _field_def in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            result[field_name] = value
        return result

    @classmethod
    def from_file(cls, config_path: str) -> "KokoroConfig":
        """Load config from JSON file"""
        try:
            with open(config_path) as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {config_path} - {e}")
            return cls()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return cls()

    def get_device(self) -> str:
        """Determine the device to use"""
        import torch

        if self.force_cpu or not self.use_gpu:
            return "cpu"
        try:
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            logger.error(f"Failed to determine device: {self}")
        return "cpu"

    @classmethod
    def from_env(cls) -> "KokoroConfig":
        """Create configuration from environment variables"""
        config = cls()

        # Device settings
        if env_value := os.environ.get("KOKORO_USE_GPU"):
            config.use_gpu = env_value.lower() in ["true", "1", "yes"]
        if env_value := os.environ.get("KOKORO_FORCE_CPU"):
            config.force_cpu = env_value.lower() in ["true", "1", "yes"]

        # Model settings
        if env_value := os.environ.get("KOKORO_DEFAULT_LANGUAGE"):
            config.default_language = env_value
        if env_value := os.environ.get("KOKORO_DEFAULT_SPEED"):
            try:
                config.default_speed = float(env_value)
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid KOKORO_DEFAULT_SPEED={env_value}: {e}")
        if env_value := os.environ.get("KOKORO_DEFAULT_VOICE"):
            config.default_voice = env_value

        # Directory settings
        if env_value := os.environ.get("KOKORO_MODEL_DIR"):
            config.model_dir = env_value
        if env_value := os.environ.get("KOKORO_VOICE_DIR"):
            config.voice_dir = env_value
        if env_value := os.environ.get("KOKORO_CACHE_DIR"):
            config.cache_dir = env_value

        # Text processing
        if env_value := os.environ.get("KOKORO_NORMALIZE_TEXT"):
            config.normalize_text = env_value.lower() in ["true", "1", "yes"]
        if env_value := os.environ.get("KOKORO_WARMUP_ON_INIT"):
            config.warmup_on_init = env_value.lower() in ["true", "1", "yes"]
        if env_value := os.environ.get("KOKORO_AUTO_DOWNLOAD"):
            config.auto_download_model = env_value.lower() in ["true", "1", "yes"]

        # TTS configuration from main config.py
        if env_value := os.environ.get("CHAMPI_TTS_VOICES"):
            try:
                config.available_voices = [
                    v.strip() for v in env_value.split(",") if v.strip()
                ]
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid CHAMPI_TTS_VOICES={env_value}: {e}")
        if env_value := os.environ.get("CHAMPI_TTS_MODELS"):
            try:
                config.available_models = [
                    v.strip() for v in env_value.split(",") if v.strip()
                ]
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid CHAMPI_TTS_MODELS={env_value}: {e}")
        if env_value := os.environ.get("CHAMPI_TTS_AUDIO_FORMAT"):
            config.tts_audio_format = env_value
        if env_value := os.environ.get("CHAMPI_AUTO_START_KOKORO"):
            config.auto_start = env_value.lower() in ["true", "1", "yes"]

        return config

    @property
    def supported_audio_formats(self) -> list:
        """Get list of audio formats supported by Kokoro TTS."""
        return ["mp3", "opus", "flac", "wav", "pcm"]

    def validate_tts_audio_format(self) -> str:
        """Validate and return a supported TTS audio format."""
        if self.tts_audio_format in self.supported_audio_formats:
            return self.tts_audio_format

        # Fallback to pcm if format is not supported
        logger.warning(
            f"TTS audio format '{self.tts_audio_format}' not supported by Kokoro, using 'pcm'"
        )
        return "pcm"


class KokoroConfigPresets:
    """Predefined configuration presets"""

    @staticmethod
    def performance() -> KokoroConfig:
        """High-performance configuration"""
        return KokoroConfig(
            use_gpu=True,
            force_cpu=False,
            memory_management=True,
            streaming_chunk_size=150,
            warmup_on_init=True,
            advanced_normalization=True,
        )

    @staticmethod
    def quality() -> KokoroConfig:
        """High-quality configuration"""
        return KokoroConfig(
            use_gpu=True,
            normalize_text=True,
            advanced_normalization=True,
            streaming_chunk_size=300,
            memory_management=True,
            retry_on_failure=True,
            default_voice="am_adam",
        )

    @staticmethod
    def cpu_only() -> KokoroConfig:
        """CPU-only configuration"""
        return KokoroConfig(
            use_gpu=False,
            force_cpu=True,
            streaming_chunk_size=100,
            warmup_on_init=False,
            memory_management=False,
        )

    @staticmethod
    def minimal() -> KokoroConfig:
        """Minimal resource configuration"""
        return KokoroConfig(
            use_gpu=False,
            force_cpu=True,
            normalize_text=False,
            advanced_normalization=False,
            streaming_chunk_size=100,
            warmup_on_init=False,
            auto_download_model=False,
            memory_management=False,
        )
