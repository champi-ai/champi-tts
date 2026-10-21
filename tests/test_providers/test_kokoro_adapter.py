"""
Tests for KokoroProviderAdapter against a patched KokoroProvider.
"""

import importlib
import importlib.util
import sys
from collections.abc import Iterator
from types import ModuleType
from unittest.mock import create_autospec, patch

import pytest

_OPTIONAL_TEXT_DEPS = ("inflect", "phonemizer")


@pytest.fixture
def kokoro_modules() -> Iterator[tuple[ModuleType, ModuleType]]:
    """Import the Kokoro provider and factory modules without the kokoro extra.

    The text pipeline imports ``inflect`` and ``phonemizer`` at module load;
    they are stubbed only when missing so the real ``KokoroProvider`` class is
    still importable for autospec.
    """
    stubs = {
        name: ModuleType(name)
        for name in _OPTIONAL_TEXT_DEPS
        if importlib.util.find_spec(name) is None
    }
    with patch.dict(sys.modules, stubs):
        provider_module = importlib.import_module(
            "champi_tts.providers.kokoro.provider"
        )
        adapter_module = importlib.import_module("champi_tts.providers.kokoro.adapter")
        yield provider_module, adapter_module


@pytest.fixture
def patched_kokoro(kokoro_modules: tuple[ModuleType, ModuleType]) -> Iterator:
    """Replace the wrapped KokoroProvider with an autospec of the real class."""
    provider_module, adapter_module = kokoro_modules
    spec_instance = create_autospec(provider_module.KokoroProvider, instance=True)
    spec_instance.list_voices.return_value = ["af_bella", "am_adam"]
    with patch.object(
        adapter_module, "OriginalKokoroProvider", return_value=spec_instance
    ):
        yield spec_instance


@pytest.mark.asyncio
async def test_list_voices_returns_list_from_sync_provider(patched_kokoro) -> None:
    """Adapter returns the synchronous provider's voices without awaiting them."""
    from champi_tts.factory import get_provider

    voices = await get_provider("kokoro").list_voices()

    assert voices == ["af_bella", "am_adam"]
    assert isinstance(voices, list)
    assert all(isinstance(voice, str) for voice in voices)
    patched_kokoro.list_voices.assert_called_once_with()


@pytest.mark.asyncio
async def test_list_voices_empty(patched_kokoro) -> None:
    """An empty voice directory yields an empty list."""
    from champi_tts.factory import get_provider

    patched_kokoro.list_voices.return_value = []

    assert await get_provider("kokoro").list_voices() == []


@pytest.mark.asyncio
async def test_synthesize_rejects_text_over_max_length(patched_kokoro) -> None:
    """Oversized text raises ValueError before reaching the model."""
    from champi_tts.providers.kokoro.adapter import KokoroProviderAdapter
    from champi_tts.providers.kokoro.config import KokoroConfig

    adapter = KokoroProviderAdapter(KokoroConfig(max_text_length=5))
    adapter._initialized = True

    with pytest.raises(ValueError, match="max_text_length is 5"):
        await adapter.synthesize("too long")
    with pytest.raises(ValueError, match="max_text_length is 5"):
        async for _ in adapter.synthesize_streaming("too long"):
            pass
    patched_kokoro.synthesize.assert_not_called()
    patched_kokoro.synthesize_streaming.assert_not_called()


@pytest.mark.asyncio
async def test_synthesize_accepts_text_at_max_length(patched_kokoro) -> None:
    """Text exactly at the limit is passed through."""
    from champi_tts.providers.kokoro.adapter import KokoroProviderAdapter
    from champi_tts.providers.kokoro.config import KokoroConfig

    adapter = KokoroProviderAdapter(KokoroConfig(max_text_length=5))
    adapter._initialized = True

    await adapter.synthesize("hello")

    patched_kokoro.synthesize.assert_awaited_once()
