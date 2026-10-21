"""
Tests for ModelDownloader and the provider's download_model flag.

Network access is never used: download_with_progress is always mocked.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from champi_tts.providers.kokoro.config import KokoroConfig
from champi_tts.providers.kokoro.exceptions import KokoroFileError
from champi_tts.providers.kokoro.models import ModelDownloader
from champi_tts.providers.kokoro.provider import KokoroProvider

MODEL_NAME = "kokoro-v1_0.pth"
CONFIG_NAME = "config.json"
VALID_MODEL = b"original-model-weights"
VALID_CONFIG = json.dumps({"original": True})


def _write_valid_model(model_dir: Path) -> None:
    """Write a valid pre-existing model/config pair into model_dir."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / MODEL_NAME).write_bytes(VALID_MODEL)
    (model_dir / CONFIG_NAME).write_text(VALID_CONFIG)


def _fake_download(url: str, filepath: str) -> None:
    """Stand-in for download_with_progress that writes valid content."""
    if filepath.endswith(f"{CONFIG_NAME}.part"):
        Path(filepath).write_text(json.dumps({"downloaded": True}))
    else:
        Path(filepath).write_bytes(b"downloaded-model-weights")


class TestDownloadModel:
    """Tests for ModelDownloader.download_model()."""

    def test_valid_files_are_not_downloaded(self, tmp_path: Path) -> None:
        """Existing valid files are returned without any download."""
        _write_valid_model(tmp_path)
        with patch.object(ModelDownloader, "download_with_progress") as download:
            model_path, config_path = ModelDownloader.download_model(str(tmp_path))
        download.assert_not_called()
        assert model_path == str(tmp_path / MODEL_NAME)
        assert config_path == str(tmp_path / CONFIG_NAME)

    def test_allow_download_false_uses_valid_files(self, tmp_path: Path) -> None:
        """allow_download=False returns existing valid files without downloading."""
        _write_valid_model(tmp_path)
        with patch.object(ModelDownloader, "download_with_progress") as download:
            ModelDownloader.download_model(str(tmp_path), allow_download=False)
        download.assert_not_called()

    def test_allow_download_false_raises_when_missing(self, tmp_path: Path) -> None:
        """allow_download=False raises KokoroFileError instead of downloading."""
        with (
            patch.object(ModelDownloader, "download_with_progress") as download,
            pytest.raises(KokoroFileError),
        ):
            ModelDownloader.download_model(str(tmp_path), allow_download=False)
        download.assert_not_called()

    def test_downloads_missing_files(self, tmp_path: Path) -> None:
        """Missing files are downloaded and moved into place."""
        with patch.object(
            ModelDownloader, "download_with_progress", side_effect=_fake_download
        ) as download:
            ModelDownloader.download_model(str(tmp_path))
        assert download.call_count == 2
        assert (tmp_path / MODEL_NAME).read_bytes() == b"downloaded-model-weights"
        assert json.loads((tmp_path / CONFIG_NAME).read_text()) == {"downloaded": True}
        assert list(tmp_path.glob("*.part")) == []

    def test_failed_forced_download_keeps_existing_files(self, tmp_path: Path) -> None:
        """A failed forced download leaves pre-existing model files untouched."""
        _write_valid_model(tmp_path)
        with (
            patch.object(
                ModelDownloader,
                "download_with_progress",
                side_effect=OSError("network down"),
            ),
            pytest.raises(OSError),
        ):
            ModelDownloader.download_model(str(tmp_path), force=True)
        assert (tmp_path / MODEL_NAME).read_bytes() == VALID_MODEL
        assert (tmp_path / CONFIG_NAME).read_text() == VALID_CONFIG
        assert list(tmp_path.glob("*.part")) == []

    def test_failed_partial_download_cleans_up_only_new_files(
        self, tmp_path: Path
    ) -> None:
        """A download failing midway removes only the files it created."""
        _write_valid_model(tmp_path)

        def fail_on_config(url: str, filepath: str) -> None:
            if filepath.endswith(f"{CONFIG_NAME}.part"):
                Path(filepath).write_text("{truncated")
                raise OSError("connection reset")
            _fake_download(url, filepath)

        with (
            patch.object(
                ModelDownloader, "download_with_progress", side_effect=fail_on_config
            ),
            pytest.raises(OSError),
        ):
            ModelDownloader.download_model(str(tmp_path), force=True)
        assert (tmp_path / MODEL_NAME).read_bytes() == VALID_MODEL
        assert (tmp_path / CONFIG_NAME).read_text() == VALID_CONFIG
        assert list(tmp_path.glob("*.part")) == []

    def test_invalid_download_keeps_existing_files(self, tmp_path: Path) -> None:
        """Downloaded files failing verification never replace existing ones."""
        _write_valid_model(tmp_path)

        def invalid_download(url: str, filepath: str) -> None:
            Path(filepath).write_text("not json")

        with (
            patch.object(
                ModelDownloader, "download_with_progress", side_effect=invalid_download
            ),
            pytest.raises(RuntimeError),
        ):
            ModelDownloader.download_model(str(tmp_path), force=True)
        assert (tmp_path / MODEL_NAME).read_bytes() == VALID_MODEL
        assert (tmp_path / CONFIG_NAME).read_text() == VALID_CONFIG
        assert list(tmp_path.glob("*.part")) == []


@pytest.fixture
def provider(tmp_path: Path):
    """Fresh KokoroProvider singleton pointed at temporary directories."""
    KokoroProvider._instance = None
    saved_cache = KokoroProvider._model_cache
    KokoroProvider._model_cache = {}
    config = KokoroConfig(
        model_dir=str(tmp_path / "models"),
        voice_dir=str(tmp_path / "voices"),
        cache_dir=str(tmp_path / "cache"),
        warmup_on_init=False,
    )
    inference = MagicMock()
    inference.load_model = AsyncMock()
    with patch(
        "champi_tts.providers.kokoro.provider.KokoroInference",
        return_value=inference,
    ):
        yield KokoroProvider(config)
    KokoroProvider._instance = None
    KokoroProvider._model_cache = saved_cache


class TestProviderDownloadFlag:
    """Tests for KokoroProvider.initialize(download_model=...) and reload()."""

    @pytest.mark.asyncio
    async def test_reload_does_not_download_valid_files(
        self, provider: KokoroProvider, tmp_path: Path
    ) -> None:
        """reload() never triggers a network download when valid files exist."""
        _write_valid_model(tmp_path / "models")
        with patch.object(ModelDownloader, "download_with_progress") as download:
            await provider.initialize()
            await provider.reload()
        download.assert_not_called()
        assert (tmp_path / "models" / MODEL_NAME).read_bytes() == VALID_MODEL
        assert await provider.is_initialized()

    @pytest.mark.asyncio
    async def test_initialize_without_download_raises_when_missing(
        self, provider: KokoroProvider
    ) -> None:
        """download_model=False raises KokoroFileError when files are missing."""
        with (
            patch.object(ModelDownloader, "download_with_progress") as download,
            pytest.raises(KokoroFileError),
        ):
            await provider.initialize(download_model=False)
        download.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_downloads_missing_files(
        self, provider: KokoroProvider, tmp_path: Path
    ) -> None:
        """download_model=True downloads files that are missing."""
        with patch.object(
            ModelDownloader, "download_with_progress", side_effect=_fake_download
        ) as download:
            await provider.initialize(download_model=True)
        assert download.call_count == 2
        assert (tmp_path / "models" / MODEL_NAME).exists()
