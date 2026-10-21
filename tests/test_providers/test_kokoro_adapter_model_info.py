"""
Tests for KokoroProviderAdapter status and voice listing before/after initialize.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import champi_tts.providers.kokoro.adapter as adapter
from champi_tts.providers.kokoro.config import KokoroConfig
from champi_tts.providers.kokoro.provider import KokoroProvider


@pytest.fixture
def adapter_module():
    """Yield the adapter module with the KokoroProvider singleton reset."""
    KokoroProvider._instance = None
    yield adapter
    KokoroProvider._instance = None


@pytest.fixture
def voice_dir(tmp_path: Path) -> Path:
    """Voice directory containing two voice files and a non-voice file."""
    directory = tmp_path / "voices"
    directory.mkdir()
    (directory / "af_bella.pt").touch()
    (directory / "am_adam.pt").touch()
    (directory / "README.txt").touch()
    return directory


def _make_adapter(adapter_module, voice_dir: Path, tmp_path: Path):
    config = KokoroConfig(
        model_dir=str(tmp_path / "models"),
        voice_dir=str(voice_dir),
        cache_dir=str(tmp_path / "cache"),
    )
    return adapter_module.KokoroProviderAdapter(config)


class TestBeforeInitialize:
    """Behaviour on a cold provider whose model has not been loaded."""

    @pytest.mark.asyncio
    async def test_get_model_info_does_not_raise(
        self, adapter_module, voice_dir: Path, tmp_path: Path
    ) -> None:
        """get_model_info() reports not_initialized instead of raising."""
        adapter = _make_adapter(adapter_module, voice_dir, tmp_path)

        info = await adapter.get_model_info()

        assert info["status"] == "not_initialized"
        assert info["provider"] == "KokoroProviderAdapter"
        assert info["initialized"] is False
        assert info["model_loaded"] is False
        assert info["device"] == "unknown"
        assert info["available_voices"] == ["af_bella", "am_adam"]
        assert info["voice_count"] == 2

    @pytest.mark.asyncio
    async def test_list_voices_returns_voices_on_disk(
        self, adapter_module, voice_dir: Path, tmp_path: Path
    ) -> None:
        """list_voices() reads the voice directory without loading the model."""
        adapter = _make_adapter(adapter_module, voice_dir, tmp_path)

        assert await adapter.list_voices() == ["af_bella", "am_adam"]
        assert adapter._kokoro.inference_engine is None

    @pytest.mark.asyncio
    async def test_default_dirs_resolved_without_config(
        self, adapter_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Unset model_dir/voice_dir fall back to defaults at construction."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        default_voices = tmp_path / ".mcp_champi" / "kokoro" / "voices"
        default_voices.mkdir(parents=True)
        (default_voices / "bf_emma.pt").touch()

        adapter = adapter_module.KokoroProviderAdapter(KokoroConfig())

        assert adapter.config.voice_dir == str(default_voices)
        assert adapter.config.model_dir == str(
            tmp_path / ".mcp_champi" / "kokoro" / "models"
        )
        assert await adapter.list_voices() == ["bf_emma"]
        assert not (tmp_path / ".mcp_champi" / "kokoro" / "models").exists()


class TestAfterInitialize:
    """Behaviour once the provider has been initialized."""

    @pytest.mark.asyncio
    async def test_get_model_info_reports_loaded(
        self, adapter_module, voice_dir: Path, tmp_path: Path
    ) -> None:
        """get_model_info() reflects the loaded inference engine."""
        adapter = _make_adapter(adapter_module, voice_dir, tmp_path)
        kokoro = adapter._kokoro

        async def fake_initialize() -> None:
            kokoro.inference_engine = SimpleNamespace(device="cpu", is_loaded=True)
            kokoro._initialized = True

        kokoro.initialize = AsyncMock(side_effect=fake_initialize)

        await adapter.initialize()
        info = await adapter.get_model_info()

        assert info["status"] == "loaded"
        assert info["initialized"] is True
        assert info["model_loaded"] is True
        assert info["device"] == "cpu"
        assert info["voice_count"] == 2


class TestBaseProviderDefault:
    """Default get_model_info() on BaseTTSProvider."""

    @pytest.mark.asyncio
    async def test_default_reports_status(self, mock_provider) -> None:
        """Providers without an override get a basic status dict."""
        assert await mock_provider.get_model_info() == {
            "status": "not_initialized",
            "provider": "MockTTSProvider",
        }
        await mock_provider.initialize()
        assert (await mock_provider.get_model_info())["status"] == "loaded"
