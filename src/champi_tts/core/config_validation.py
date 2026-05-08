"""
Configuration validation utilities for Champi TTS.

Provides validation functions for configuration objects.
"""

import os
from pathlib import Path
from typing import Any, Optional

from loguru import logger


def validate_voice_name(
    voice: str,
    available_voices: Optional[list[str]] = None,
    voice_prefix: str = "af_",
) -> tuple[bool, str]:
    """
    Validate that a voice name is valid.

    Args:
        voice: Voice name to validate
        available_voices: List of available voices (optional)
        voice_prefix: Prefix for voice names

    Returns:
        Tuple of (is_valid, message)
    """
    # Check if voice has correct prefix
    if not voice.startswith(voice_prefix):
        return (
            False,
            f"Voice '{voice}' does not have the expected prefix '{voice_prefix}'",
        )

    # Check against available voices if provided
    if available_voices is not None and voice not in available_voices:
        # Try common voice name patterns
        common_voices = [
            f"{voice_prefix}bella",
            f"{voice_prefix}sarah",
            f"{voice_prefix}amy",
            f"{voice_prefix}tara",
            f"{voice_prefix}jenny",
            f"{voice_prefix}kale",
            f"{voice_prefix}charlie",
            f"{voice_prefix}david",
        ]
        if voice in common_voices or voice.startswith("am_"):
            # Common American voices
            pass
        else:
            available_str = ", ".join(available_voices[:5]) + "..." if len(available_voices) > 5 else ", ".join(
                available_voices
            )
            return (
                False,
                f"Voice '{voice}' not found. Available voices: {available_str}",
            )

    return (True, f"Voice '{voice}' is valid")


def validate_speed(
    speed: Optional[float],
    default_speed: float = 1.0,
    min_speed: float = 0.5,
    max_speed: float = 2.0,
) -> tuple[float, str]:
    """
    Validate and clamp speech speed.

    Args:
        speed: Speech speed (None uses default)
        default_speed: Default speed if speed is None
        min_speed: Minimum allowed speed
        max_speed: Maximum allowed speed

    Returns:
        Tuple of (validated_speed, message)
    """
    if speed is None:
        speed = default_speed

    if speed < min_speed:
        logger.warning(
            f"Speed {speed} is below minimum {min_speed}. Clamping to {min_speed}"
        )
        speed = min_speed
    elif speed > max_speed:
        logger.warning(
            f"Speed {speed} is above maximum {max_speed}. Clamping to {max_speed}"
        )
        speed = max_speed

    return (speed, f"Using speed: {speed}")


def validate_model_path(
    model_path: str,
    require_file: bool = False,
) -> tuple[bool, str]:
    """
    Validate model path.

    Args:
        model_path: Model file path
        require_file: Whether the file must exist

    Returns:
        Tuple of (is_valid, message)
    """
    path = Path(model_path)

    if not path.exists() and require_file:
        return (
            False,
            f"Model file does not exist: {model_path}",
        )

    if not path.parent.exists():
        return (
            False,
            f"Model directory does not exist: {path.parent}",
        )

    return (True, f"Model path is valid: {model_path}")


def validate_cache_path(
    cache_path: str,
) -> tuple[bool, str]:
    """
    Validate cache directory path.

    Args:
        cache_path: Cache directory path

    Returns:
        Tuple of (is_valid, message)
    """
    path = Path(cache_path)

    # Try to create the directory if it doesn't exist
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return (
            False,
            f"Cannot create cache directory: {e}",
        )

    # Check write permission
    try:
        test_file = path / ".test"
        test_file.touch()
        test_file.unlink()
    except OSError as e:
        return (
            False,
            f"Cache directory is not writable: {e}",
        )

    return (True, f"Cache directory is valid: {cache_path}")


def validate_sample_rate(
    sample_rate: Optional[int],
    default_rate: int = 22050,
) -> tuple[int, str]:
    """
    Validate audio sample rate.

    Args:
        sample_rate: Sample rate (None uses default)
        default_rate: Default sample rate

    Returns:
        Tuple of (validated_rate, message)
    """
    if sample_rate is None:
        sample_rate = default_rate

    # Common sample rates
    common_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000]

    if sample_rate not in common_rates:
        logger.warning(
            f"Sample rate {sample_rate} is not a common rate. "
            f"Using default: {default_rate}"
        )
        sample_rate = default_rate

    return (sample_rate, f"Using sample rate: {sample_rate} Hz")


def validate_audio_file(
    file_path: str,
) -> tuple[bool, str]:
    """
    Validate audio file path.

    Args:
        file_path: Audio file path

    Returns:
        Tuple of (is_valid, message)
    """
    path = Path(file_path)

    if not path.exists():
        return (
            False,
            f"Audio file does not exist: {file_path}",
        )

    # Check file extension
    valid_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    if path.suffix.lower() not in valid_extensions:
        return (
            False,
            f"Invalid audio file extension: {path.suffix}. "
            f"Valid extensions: {', '.join(valid_extensions)}",
        )

    # Check file size (not empty)
    if path.stat().st_size == 0:
        return (
            False,
            f"Audio file is empty: {file_path}",
        )

    return (True, f"Audio file is valid: {file_path}")


def validate_config(
    config: dict[str, Any],
) -> list[tuple[str, str]]:
    """
    Validate a configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        List of (field, message) tuples for any validation errors
    """
    errors = []

    # Validate voice
    if "default_voice" in config:
        is_valid, message = validate_voice_name(
            config["default_voice"],
            available_voices=None,  # Would need actual voices list
        )
        if not is_valid:
            errors.append(("default_voice", message))

    # Validate speed
    if "default_speed" in config:
        validated, message = validate_speed(config["default_speed"])
        logger.debug(message)

    # Validate model path
    if "model_path" in config:
        is_valid, message = validate_model_path(
            config["model_path"],
            require_file=True,
        )
        if not is_valid:
            errors.append(("model_path", message))

    # Validate cache path
    if "cache_path" in config:
        is_valid, message = validate_cache_path(config["cache_path"])
        if not is_valid:
            errors.append(("cache_path", message))

    # Validate sample rate
    if "sample_rate" in config:
        validated, message = validate_sample_rate(config["sample_rate"])
        logger.debug(message)

    return errors


def validate_all(
    voice: Optional[str] = None,
    speed: Optional[float] = None,
    model_path: Optional[str] = None,
    cache_path: Optional[str] = None,
    sample_rate: Optional[int] = None,
) -> bool:
    """
    Validate all configuration parameters.

    Args:
        voice: Voice name
        speed: Speech speed
        model_path: Model file path
        cache_path: Cache directory path
        sample_rate: Audio sample rate

    Returns:
        True if all validations pass, False otherwise
    """
    errors = validate_config(
        {
            "default_voice": voice,
            "default_speed": speed,
            "model_path": model_path,
            "cache_path": cache_path,
            "sample_rate": sample_rate,
        }
    )

    if errors:
        logger.error(
            f"Configuration validation failed:\n"
            + "\n".join(f"  {field}: {message}" for field, message in errors)
        )
        return False

    return True


# Environment variable helpers
def get_env_voice() -> Optional[str]:
    """Get voice from environment or None."""
    voice = os.getenv("CHAMPI_KOKORO_DEFAULT_VOICE")
    return voice


def get_env_speed() -> Optional[float]:
    """Get speed from environment or None."""
    speed_str = os.getenv("CHAMPI_KOKORO_DEFAULT_SPEED")
    if speed_str:
        try:
            return float(speed_str)
        except ValueError:
            return None
    return None


def get_env_use_gpu() -> bool:
    """Get use_gpu from environment."""
    gpu_str = os.getenv("CHAMPI_KOKORO_USE_GPU", "true")
    return gpu_str.lower() in ("true", "1", "yes")
