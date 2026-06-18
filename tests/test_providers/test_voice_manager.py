"""
Tests for the voice cache manager.
"""

from pathlib import Path

import pytest

from champi_tts.providers.kokoro.voice_manager import (
    VOICES_CACHE_DIR,
    get_voice_path,
    list_cached_voices,
)


class TestGetVoicePath:
    """Tests for get_voice_path()."""

    def test_returns_path_with_pt_extension(self, tmp_path: Path) -> None:
        """get_voice_path() appends .pt to the voice name."""
        result = get_voice_path("af_bella", cache_dir=tmp_path)
        assert result.suffix == ".pt"
        assert result.stem == "af_bella"

    def test_path_is_inside_cache_dir(self, tmp_path: Path) -> None:
        """Returned path is a direct child of the cache directory."""
        result = get_voice_path("am_adam", cache_dir=tmp_path)
        assert result.parent == tmp_path

    def test_creates_cache_directory(self, tmp_path: Path) -> None:
        """get_voice_path() creates the cache directory when it is absent."""
        new_dir = tmp_path / "deep" / "voices"
        assert not new_dir.exists()
        get_voice_path("af_sarah", cache_dir=new_dir)
        assert new_dir.is_dir()

    def test_default_cache_dir_used_when_none(self) -> None:
        """When cache_dir is None the default VOICES_CACHE_DIR is used."""
        result = get_voice_path("af_sky")
        assert result.parent == VOICES_CACHE_DIR

    def test_full_path_is_correct(self, tmp_path: Path) -> None:
        """Returned path equals <cache_dir>/<voice_name>.pt."""
        expected = tmp_path / "bm_george.pt"
        result = get_voice_path("bm_george", cache_dir=tmp_path)
        assert result == expected


class TestListCachedVoices:
    """Tests for list_cached_voices()."""

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """No .pt files means an empty list is returned."""
        assert list_cached_voices(cache_dir=tmp_path) == []

    def test_lists_pt_files_without_extension(self, tmp_path: Path) -> None:
        """Voice names are returned without the .pt suffix."""
        (tmp_path / "af_bella.pt").touch()
        (tmp_path / "am_adam.pt").touch()
        result = list_cached_voices(cache_dir=tmp_path)
        assert "af_bella" in result
        assert "am_adam" in result

    def test_ignores_non_pt_files(self, tmp_path: Path) -> None:
        """Files with extensions other than .pt are not included."""
        (tmp_path / "af_bella.pt").touch()
        (tmp_path / "README.txt").touch()
        (tmp_path / "config.json").touch()
        result = list_cached_voices(cache_dir=tmp_path)
        assert result == ["af_bella"]

    def test_result_is_sorted(self, tmp_path: Path) -> None:
        """Returned list is sorted alphabetically."""
        for name in ["zz_voice.pt", "aa_voice.pt", "mm_voice.pt"]:
            (tmp_path / name).touch()
        result = list_cached_voices(cache_dir=tmp_path)
        assert result == sorted(result)

    def test_nonexistent_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """Missing cache directory returns an empty list rather than raising."""
        missing = tmp_path / "does_not_exist"
        assert list_cached_voices(cache_dir=missing) == []

    def test_default_cache_dir_used_when_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When cache_dir is None, VOICES_CACHE_DIR is consulted."""
        import champi_tts.providers.kokoro.voice_manager as vm

        fake_dir = Path("/nonexistent/fake")
        monkeypatch.setattr(vm, "VOICES_CACHE_DIR", fake_dir)
        assert list_cached_voices() == []
