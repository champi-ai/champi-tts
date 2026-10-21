"""
Tests for the canonical voice directory and voice downloads.

Network access is never used: urlretrieve is always mocked.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from champi_tts.cli.main import app
from champi_tts.providers.kokoro import models
from champi_tts.providers.kokoro.config import KokoroConfig
from champi_tts.providers.kokoro.exceptions import KokoroVoiceError
from champi_tts.providers.kokoro.models import VoiceManager, default_voice_dir
from champi_tts.providers.kokoro.provider import KokoroProvider


def _fake_urlretrieve(url: str, filepath: str) -> None:
    """Stand-in for urlretrieve that writes a non-empty file."""
    Path(filepath).write_bytes(b"voice-tensor")


class TestDefaultVoiceDir:
    """Tests for default_voice_dir()."""

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KOKORO_VOICE_DIR", "/custom/voices")
        assert default_voice_dir() == "/custom/voices"

    def test_home_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("KOKORO_VOICE_DIR", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert default_voice_dir() == str(tmp_path / ".mcp_champi/kokoro/voices")

    def test_provider_uses_canonical_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("KOKORO_VOICE_DIR", str(tmp_path))
        assert KokoroProvider().config.voice_dir == str(tmp_path)


class TestDownloadVoice:
    """Tests for VoiceManager.download_voice()."""

    def test_downloads_from_hf(self, tmp_path: Path) -> None:
        with patch.object(
            models, "urlretrieve", side_effect=_fake_urlretrieve
        ) as fetch:
            path = VoiceManager.download_voice("am_adam", str(tmp_path))
        assert path == str(tmp_path / "am_adam.pt")
        assert fetch.call_args.args[0] == (
            "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/am_adam.pt"
        )
        assert VoiceManager.list_voices(str(tmp_path)) == ["am_adam"]
        assert list(tmp_path.glob("*.part")) == []

    def test_existing_voice_is_not_downloaded(self, tmp_path: Path) -> None:
        (tmp_path / "am_adam.pt").write_bytes(b"existing")
        with patch.object(models, "urlretrieve") as fetch:
            VoiceManager.download_voice("am_adam.pt", str(tmp_path))
        fetch.assert_not_called()

    def test_force_redownloads(self, tmp_path: Path) -> None:
        (tmp_path / "am_adam.pt").write_bytes(b"existing")
        with patch.object(models, "urlretrieve", side_effect=_fake_urlretrieve):
            VoiceManager.download_voice("am_adam", str(tmp_path), force=True)
        assert (tmp_path / "am_adam.pt").read_bytes() == b"voice-tensor"

    @pytest.mark.parametrize("name", ["../etc/passwd", "AM_ADAM", "adam", ""])
    def test_invalid_name_rejected(self, tmp_path: Path, name: str) -> None:
        with (
            patch.object(models, "urlretrieve") as fetch,
            pytest.raises(KokoroVoiceError),
        ):
            VoiceManager.download_voice(name, str(tmp_path))
        fetch.assert_not_called()

    def test_failed_download_cleans_up(self, tmp_path: Path) -> None:
        def fail(url: str, filepath: str) -> None:
            Path(filepath).write_bytes(b"trunc")
            raise OSError("404")

        with (
            patch.object(models, "urlretrieve", side_effect=fail),
            pytest.raises(KokoroVoiceError, match="am_adam"),
        ):
            VoiceManager.download_voice("am_adam", str(tmp_path))
        assert list(tmp_path.iterdir()) == []


@pytest.fixture
def provider(tmp_path: Path):
    """Provider with mocked model download and inference."""
    saved_cache = KokoroProvider._model_cache
    KokoroProvider._model_cache = {}
    engine = MagicMock()
    engine.load_model = AsyncMock()
    engine.is_loaded = True
    config = KokoroConfig(
        model_dir=str(tmp_path / "models"),
        voice_dir=str(tmp_path / "voices"),
        cache_dir=str(tmp_path / "cache"),
        warmup_on_init=False,
    )
    with (
        patch(
            "champi_tts.providers.kokoro.provider.KokoroInference",
            return_value=engine,
        ),
        patch(
            "champi_tts.providers.kokoro.provider.ModelDownloader.download_model",
            return_value=("model.pth", "config.json"),
        ),
    ):
        yield KokoroProvider(config)
    KokoroProvider._model_cache = saved_cache


class TestProviderFirstRun:
    """Fresh-install behaviour of KokoroProvider.initialize()."""

    @pytest.mark.asyncio
    async def test_initialize_downloads_default_voice(
        self, provider: KokoroProvider
    ) -> None:
        with patch.object(models, "urlretrieve", side_effect=_fake_urlretrieve):
            await provider.initialize()
        assert provider.list_voices() == ["am_adam"]

    @pytest.mark.asyncio
    async def test_no_voice_download_when_disabled(
        self, provider: KokoroProvider
    ) -> None:
        provider.config.auto_download_model = False
        with patch.object(models, "urlretrieve") as fetch:
            await provider.initialize()
        fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_voice_download_does_not_break_init(
        self, provider: KokoroProvider
    ) -> None:
        await provider.initialize()
        assert await provider.is_initialized()
        assert provider.list_voices() == []

    @pytest.mark.asyncio
    async def test_missing_voice_error_names_command(
        self, provider: KokoroProvider
    ) -> None:
        await provider.initialize()
        with pytest.raises(
            KokoroVoiceError, match="champi-tts voices download af_bella"
        ):
            await provider.synthesize("hi", voice="af_bella")


class TestVoicesCli:
    """CLI and provider share one voice directory."""

    def test_cli_lists_provider_voices(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("KOKORO_VOICE_DIR", str(tmp_path))
        (tmp_path / "af_bella.pt").write_bytes(b"x")
        result = CliRunner().invoke(app, ["voices", "list"])
        assert result.exit_code == 0
        assert "af_bella" in result.output
        assert KokoroProvider().list_voices() == ["af_bella"]

    def test_cli_cache_dir_is_canonical(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("KOKORO_VOICE_DIR", str(tmp_path))
        result = CliRunner().invoke(app, ["voices", "cache-dir"])
        assert result.output.strip() == str(tmp_path)

    def test_cli_download(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("KOKORO_VOICE_DIR", str(tmp_path))
        with patch.object(models, "urlretrieve", side_effect=_fake_urlretrieve):
            result = CliRunner().invoke(app, ["voices", "download", "bf_emma"])
        assert result.exit_code == 0
        assert (tmp_path / "bf_emma.pt").exists()

    def test_cli_download_failure_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("KOKORO_VOICE_DIR", str(tmp_path))
        result = CliRunner().invoke(app, ["voices", "download", "bf_emma"])
        assert result.exit_code == 1
