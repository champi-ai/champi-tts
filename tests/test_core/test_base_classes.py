"""
Tests for BaseTTSConfig and BaseTTSProvider abstract base classes.

Uses the MockTTSConfig and MockTTSProvider from conftest to exercise
the concrete behaviours defined on the abstract bases.
"""

import pytest

from champi_tts.core.base_config import BaseTTSConfig
from champi_tts.core.base_synthesizer import BaseSynthesizer


class TestBaseTTSConfigToDict:
    """Tests for BaseTTSConfig.to_dict()."""

    def test_returns_dict(self, mock_config: BaseTTSConfig) -> None:
        """to_dict() returns a plain dictionary."""
        result = mock_config.to_dict()
        assert isinstance(result, dict)

    def test_contains_sample_rate(self, mock_config: BaseTTSConfig) -> None:
        """to_dict() includes sample_rate from the base class."""
        result = mock_config.to_dict()
        assert "sample_rate" in result

    def test_contains_default_speed(self, mock_config: BaseTTSConfig) -> None:
        """to_dict() includes default_speed from the base class."""
        result = mock_config.to_dict()
        assert "default_speed" in result

    def test_no_private_keys(self, mock_config: BaseTTSConfig) -> None:
        """to_dict() excludes private attributes (those starting with '_')."""
        result = mock_config.to_dict()
        for key in result:
            assert not key.startswith("_")

    def test_default_values(self, mock_config: BaseTTSConfig) -> None:
        """to_dict() reflects the default values from BaseTTSConfig."""
        result = mock_config.to_dict()
        assert result["default_speed"] == 1.0
        assert result["normalize_text"] is True
        assert result["enable_streaming"] is True


class TestBaseTTSProviderProperties:
    """Tests for BaseTTSProvider property accessors."""

    def test_is_initialized_starts_false(self, mock_provider) -> None:
        """New provider has is_initialized == False."""
        assert mock_provider.is_initialized is False

    def test_is_speaking_starts_false(self, mock_provider) -> None:
        """New provider has is_speaking == False."""
        assert mock_provider.is_speaking is False

    @pytest.mark.asyncio
    async def test_is_initialized_true_after_initialize(
        self, mock_provider
    ) -> None:
        """is_initialized becomes True after initialize()."""
        await mock_provider.initialize()
        assert mock_provider.is_initialized is True

    @pytest.mark.asyncio
    async def test_is_initialized_false_after_shutdown(
        self, mock_provider
    ) -> None:
        """is_initialized becomes False after shutdown()."""
        await mock_provider.initialize()
        await mock_provider.shutdown()
        assert mock_provider.is_initialized is False

    def test_config_stored(self, mock_provider, mock_config) -> None:
        """Provider stores the config passed at construction."""
        assert mock_provider.config is mock_config


class TestBaseTTSProviderContextManager:
    """Tests for BaseTTSProvider async context manager."""

    @pytest.mark.asyncio
    async def test_aenter_calls_initialize(self, mock_provider) -> None:
        """Entering the context manager initializes the provider."""
        async with mock_provider:
            assert mock_provider.is_initialized is True

    @pytest.mark.asyncio
    async def test_aexit_calls_shutdown(self, mock_provider) -> None:
        """Exiting the context manager shuts down the provider."""
        async with mock_provider:
            pass
        assert mock_provider.is_initialized is False

    @pytest.mark.asyncio
    async def test_aexit_returns_false(self, mock_provider) -> None:
        """__aexit__ returns False so exceptions propagate."""
        result = await mock_provider.__aexit__(None, None, None)
        assert result is False


class TestBaseTTSProviderSynthesize:
    """Tests for BaseTTSProvider.synthesize() via the mock implementation."""

    @pytest.mark.asyncio
    async def test_synthesize_returns_ndarray(
        self, initialized_provider
    ) -> None:
        """synthesize() returns a numpy array."""
        import numpy as np

        audio = await initialized_provider.synthesize("Hello")
        assert isinstance(audio, np.ndarray)

    @pytest.mark.asyncio
    async def test_synthesize_streaming_yields_chunks(
        self, initialized_provider
    ) -> None:
        """synthesize_streaming() yields numpy arrays."""
        import numpy as np

        chunks = []
        async for chunk in initialized_provider.synthesize_streaming("Hello"):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert all(isinstance(c, np.ndarray) for c in chunks)

    @pytest.mark.asyncio
    async def test_list_voices_returns_list(self, initialized_provider) -> None:
        """list_voices() returns a list of voice name strings."""
        voices = await initialized_provider.list_voices()
        assert isinstance(voices, list)
        assert len(voices) > 0
        assert all(isinstance(v, str) for v in voices)

    @pytest.mark.asyncio
    async def test_interrupt_clears_is_speaking(
        self, initialized_provider
    ) -> None:
        """interrupt() sets _is_speaking to False."""
        initialized_provider._is_speaking = True
        await initialized_provider.interrupt()
        assert initialized_provider.is_speaking is False
