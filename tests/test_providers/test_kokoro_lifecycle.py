"""
Tests for KokoroProvider / KokoroProviderAdapter shutdown and interrupt.

The heavy inference engine and sounddevice are mocked so these run without
the ``kokoro`` extra installed.
"""

import importlib.util
import sys
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# text_utils imports these at module level; stub them when the extra is absent.
for _name in ("inflect", "phonemizer"):
    if importlib.util.find_spec(_name) is None:
        sys.modules.setdefault(_name, MagicMock())

from champi_tts.providers.kokoro import provider as provider_module  # noqa: E402
from champi_tts.providers.kokoro.adapter import KokoroProviderAdapter  # noqa: E402
from champi_tts.providers.kokoro.config import KokoroConfig  # noqa: E402
from champi_tts.providers.kokoro.enums import LifecycleEvents  # noqa: E402
from champi_tts.providers.kokoro.provider import KokoroProvider  # noqa: E402


@pytest.fixture(autouse=True)
def reset_singleton() -> Iterator[None]:
    """Give every test a fresh KokoroProvider singleton."""
    KokoroProvider._instance = None
    KokoroProvider._model_cache = {}
    yield
    KokoroProvider._instance = None
    KokoroProvider._model_cache = {}


@pytest.fixture
def mock_sd() -> Iterator[MagicMock]:
    """Replace the provider module's sounddevice reference."""
    mock = MagicMock()
    with patch.object(provider_module, "sd", mock):
        yield mock


@pytest.fixture
def engine() -> MagicMock:
    """Mocked Kokoro inference engine."""
    mock = MagicMock()
    mock.is_loaded = True
    mock.load_model = AsyncMock()
    return mock


@pytest.fixture
def config(tmp_path: Path) -> KokoroConfig:
    """Config pointing at temporary directories with warmup disabled."""
    return KokoroConfig(
        model_dir=str(tmp_path / "model"),
        voice_dir=str(tmp_path / "voices"),
        cache_dir=str(tmp_path / "cache"),
        warmup_on_init=False,
    )


@pytest.fixture
def adapter(config: KokoroConfig, engine: MagicMock) -> Iterator[KokoroProviderAdapter]:
    """Adapter whose fresh-load path is backed by the mocked engine."""
    with (
        patch.object(provider_module, "KokoroInference", return_value=engine),
        patch.object(
            provider_module.ModelDownloader,
            "download_model",
            return_value=("model.pth", "config.json"),
        ),
        patch.object(provider_module.VoiceManager, "setup_voice_directory"),
    ):
        yield KokoroProviderAdapter(config)


class TestShutdown:
    """Tests for shutdown()."""

    @pytest.mark.asyncio
    async def test_shutdown_after_initialize(
        self, adapter: KokoroProviderAdapter, engine: MagicMock, mock_sd: MagicMock
    ) -> None:
        """shutdown() unloads the engine and resets state after initialize()."""
        await adapter.initialize()
        assert adapter.is_initialized

        await adapter.shutdown()

        engine.unload.assert_called_once()
        mock_sd.stop.assert_called_once()
        assert not adapter.is_initialized
        assert adapter._kokoro.inference_engine is None
        assert adapter._kokoro._initialized is False
        assert adapter._kokoro._tts_status == LifecycleEvents.UNINITIALIZED.value

    @pytest.mark.asyncio
    async def test_shutdown_without_initialize_is_noop(
        self, adapter: KokoroProviderAdapter, mock_sd: MagicMock
    ) -> None:
        """shutdown() on a never-initialized provider does not raise."""
        await adapter.shutdown()

        assert not adapter.is_initialized
        assert adapter._kokoro._initialized is False

    @pytest.mark.asyncio
    async def test_shutdown_without_sounddevice(
        self, adapter: KokoroProviderAdapter
    ) -> None:
        """shutdown() works when PortAudio is unavailable."""
        with patch.object(provider_module, "sd", None):
            await adapter.initialize()
            await adapter.shutdown()

        assert not adapter.is_initialized

    @pytest.mark.asyncio
    async def test_async_context_manager(
        self, adapter: KokoroProviderAdapter, engine: MagicMock, mock_sd: MagicMock
    ) -> None:
        """``async with adapter`` initializes and shuts down cleanly."""
        async with adapter as entered:
            assert entered.is_initialized

        engine.unload.assert_called_once()
        assert not adapter.is_initialized


class TestInterrupt:
    """Tests for interrupt()."""

    @pytest.mark.asyncio
    async def test_interrupt_stops_playback(
        self, adapter: KokoroProviderAdapter, mock_sd: MagicMock
    ) -> None:
        """interrupt() stops sounddevice playback and resets status to READY."""
        await adapter.initialize()
        adapter._kokoro._tts_status = LifecycleEvents.SPEAKING.value
        adapter._is_speaking = True

        await adapter.interrupt()

        mock_sd.stop.assert_called_once()
        assert adapter.is_speaking is False
        assert adapter._kokoro._tts_status == LifecycleEvents.READY.value

    @pytest.mark.asyncio
    async def test_interrupt_before_initialize_keeps_status(
        self, adapter: KokoroProviderAdapter, mock_sd: MagicMock
    ) -> None:
        """interrupt() on an uninitialized provider does not claim READY."""
        await adapter.interrupt()

        mock_sd.stop.assert_called_once()
        assert adapter._kokoro._tts_status == LifecycleEvents.UNINITIALIZED.value

    @pytest.mark.asyncio
    async def test_interrupt_cancels_streaming(
        self,
        adapter: KokoroProviderAdapter,
        engine: MagicMock,
        config: KokoroConfig,
        mock_sd: MagicMock,
    ) -> None:
        """interrupt() during streaming stops further chunks from being yielded."""

        async def fake_stream(**_: object) -> AsyncGenerator[SimpleNamespace, None]:
            for _ in range(5):
                yield SimpleNamespace(audio=np.zeros(10, dtype=np.float32))

        engine.generate_stream = fake_stream
        await adapter.initialize()
        Path(config.voice_dir, "af_bella.pt").touch()

        received = []
        async for chunk in adapter.synthesize_streaming("hello", voice="af_bella"):
            received.append(chunk)
            if len(received) == 2:
                await adapter.interrupt()

        assert len(received) == 2
        mock_sd.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_new_stream_after_interrupt_runs(
        self,
        adapter: KokoroProviderAdapter,
        engine: MagicMock,
        config: KokoroConfig,
        mock_sd: MagicMock,
    ) -> None:
        """A stale interrupt does not cut short the next streaming request."""

        async def fake_stream(**_: object) -> AsyncGenerator[SimpleNamespace, None]:
            for _ in range(3):
                yield SimpleNamespace(audio=np.zeros(10, dtype=np.float32))

        engine.generate_stream = fake_stream
        await adapter.initialize()
        Path(config.voice_dir, "af_bella.pt").touch()

        await adapter.interrupt()
        received = [
            chunk
            async for chunk in adapter.synthesize_streaming("hello", voice="af_bella")
        ]

        assert len(received) == 3
