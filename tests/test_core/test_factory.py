"""
Tests for the provider factory.
"""

import pytest

from champi_tts.core.base_provider import BaseTTSProvider
from champi_tts.factory import (
    get_default_provider,
    get_provider,
    get_reader,
    list_providers,
)
from champi_tts.reader import TextReaderService


def test_list_providers():
    """Test listing available providers."""
    providers = list_providers()
    assert isinstance(providers, list)
    assert "kokoro" in providers


def test_get_provider_default():
    """Test getting provider with defaults."""
    provider = get_provider()
    assert isinstance(provider, BaseTTSProvider)


def test_get_provider_with_kwargs():
    """Test getting provider with config kwargs."""
    provider = get_provider("kokoro", default_voice="af_bella", default_speed=1.2)
    assert isinstance(provider, BaseTTSProvider)
    assert provider.config.default_voice == "af_bella"
    assert provider.config.default_speed == 1.2


def test_get_provider_invalid():
    """Test getting invalid provider raises error."""
    with pytest.raises(ValueError, match="Unknown provider type"):
        get_provider("invalid_provider")


def test_get_default_provider():
    """Test getting default provider."""
    provider = get_default_provider()
    assert isinstance(provider, BaseTTSProvider)


def test_get_reader():
    """Test getting reader service."""
    reader = get_reader("kokoro")
    assert isinstance(reader, TextReaderService)
    assert reader.provider is not None


def test_get_reader_with_ui():
    """Test getting reader with UI."""
    reader = get_reader("kokoro", show_ui=True)
    assert isinstance(reader, TextReaderService)
    assert reader.show_ui is True
