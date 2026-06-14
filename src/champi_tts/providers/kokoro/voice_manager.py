"""
Voice cache manager for Kokoro TTS.

Resolves voice file paths within the user-level cache directory and
provides utilities for listing cached voices. Voice files must be placed
manually; this module handles path resolution and cache directory setup only.
"""

from pathlib import Path

VOICES_CACHE_DIR: Path = Path.home() / ".cache" / "champi-tts" / "voices"


def get_voice_path(voice_name: str, cache_dir: Path | None = None) -> Path:
    """Return the expected path for a voice file in the cache directory.

    The directory is created if it does not already exist.  The caller is
    responsible for verifying that the returned path exists before use.

    Args:
        voice_name: Voice identifier without the ``.pt`` extension.
        cache_dir: Override the default cache directory.  Defaults to
            ``~/.cache/champi-tts/voices``.

    Returns:
        Path pointing to ``<cache_dir>/<voice_name>.pt``.
    """
    directory = cache_dir if cache_dir is not None else VOICES_CACHE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{voice_name}.pt"


def list_cached_voices(cache_dir: Path | None = None) -> list[str]:
    """List voice names available in the cache directory.

    Args:
        cache_dir: Override the default cache directory.  Defaults to
            ``~/.cache/champi-tts/voices``.

    Returns:
        Sorted list of voice names (filenames without the ``.pt`` suffix).
        Returns an empty list when the directory does not exist.
    """
    directory = cache_dir if cache_dir is not None else VOICES_CACHE_DIR
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.iterdir() if p.suffix == ".pt")
