"""
Tests for KokoroProvider config handling and shared model cache.
"""

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    import champi_tts.providers.kokoro.text_utils  # noqa: F401
except ImportError:
    # Text processing deps (inflect, phonemizer) ship with the kokoro extra.
    _stub = types.ModuleType("champi_tts.providers.kokoro.text_utils")
    _stub.AsyncTextProcessor = object
    sys.modules[_stub.__name__] = _stub

from champi_tts.providers.kokoro import provider as provider_module
from champi_tts.providers.kokoro.config import KokoroConfig
from champi_tts.providers.kokoro.provider import KokoroProvider


@pytest.fixture(autouse=True)
def _clear_model_cache():
    KokoroProvider._model_cache.clear()
    yield
    KokoroProvider._model_cache.clear()


def _config(tmp_path: Path, **overrides) -> KokoroConfig:
    return KokoroConfig(
        model_dir=str(tmp_path / "models"),
        voice_dir=str(tmp_path / "voices"),
        cache_dir=str(tmp_path / "cache"),
        warmup_on_init=False,
        **overrides,
    )


@pytest.fixture
def fake_inference():
    """Patch model download and inference so initialize() loads nothing real."""

    def _make_engine(_config):
        engine = MagicMock()
        engine.load_model = AsyncMock()
        engine.is_loaded = True

        def _unload():
            engine.is_loaded = False

        engine.unload.side_effect = _unload
        return engine

    with (
        patch.object(
            provider_module.ModelDownloader,
            "download_model",
            return_value=("model.pth", "config.json"),
        ),
        patch.object(provider_module.VoiceManager, "setup_voice_directory"),
        patch.object(
            provider_module, "KokoroInference", side_effect=_make_engine
        ) as factory,
    ):
        yield factory


class TestPerConfigInstances:
    """Each provider keeps the config it was constructed with."""

    def test_new_config_is_applied(self) -> None:
        first = KokoroProvider(KokoroConfig(default_voice="af_bella"))
        second = KokoroProvider(KokoroConfig(default_voice="am_adam"))

        assert first is not second
        assert first.config.default_voice == "af_bella"
        assert second.config.default_voice == "am_adam"

    def test_default_config_when_none(self) -> None:
        provider = KokoroProvider()

        assert provider.config == KokoroConfig()

    @pytest.mark.asyncio
    async def test_get_instance_applies_config(self) -> None:
        provider = await KokoroProvider.get_instance(
            KokoroConfig(default_voice="bf_emma")
        )

        assert provider.config.default_voice == "bf_emma"

    def test_from_env_config_is_respected(self, monkeypatch) -> None:
        KokoroProvider(KokoroConfig(default_voice="af_bella"))
        monkeypatch.setenv("KOKORO_DEFAULT_VOICE", "bm_george")

        provider = KokoroProvider(KokoroConfig.from_env())

        assert provider.config.default_voice == "bm_george"


class TestSharedModelCache:
    """Model weights are loaded once per process and shared across providers."""

    @pytest.mark.asyncio
    async def test_model_loaded_once_for_two_providers(
        self, tmp_path: Path, fake_inference: MagicMock
    ) -> None:
        first = KokoroProvider(_config(tmp_path, default_voice="af_bella"))
        second = KokoroProvider(_config(tmp_path, default_voice="am_adam"))

        await first.initialize()
        await second.initialize()

        assert fake_inference.call_count == 1
        assert first.inference_engine is second.inference_engine
        assert first.config.default_voice == "af_bella"
        assert second.config.default_voice == "am_adam"

    @pytest.mark.asyncio
    async def test_unloaded_engine_is_not_reused(
        self, tmp_path: Path, fake_inference: MagicMock
    ) -> None:
        provider = KokoroProvider(_config(tmp_path))
        await provider.initialize()
        await provider.unload()

        assert KokoroProvider._model_cache == {}

        await provider.initialize()

        assert fake_inference.call_count == 2
        assert provider.inference_engine.is_loaded
