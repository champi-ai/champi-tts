"""
Provider factory for creating TTS providers and readers.
"""

from typing import Literal

from champi_tts.core.base_config import BaseTTSConfig
from champi_tts.core.base_provider import BaseTTSProvider
from champi_tts.reader import TextReaderService

# Type for supported providers
ProviderType = Literal["kokoro"]  # Will add more providers later


def get_provider(
    provider_type: ProviderType = "kokoro",
    config: BaseTTSConfig | None = None,
    **config_kwargs,
) -> BaseTTSProvider:
    """
    Factory function to create TTS providers.

    Args:
        provider_type: Type of provider ("kokoro", etc.)
        config: Pre-configured provider config (optional)
        **config_kwargs: Config parameters (if config not provided)

    Returns:
        Initialized TTS provider instance

    Examples:
        # Using default config
        provider = get_provider("kokoro")

        # Using custom config
        from champi_tts.providers.kokoro import KokoroConfig
        config = KokoroConfig(default_voice="af_bella", default_speed=1.1)
        provider = get_provider("kokoro", config=config)

        # Using kwargs
        provider = get_provider("kokoro", default_voice="af_bella", use_gpu=True)
    """
    if provider_type == "kokoro":
        from champi_tts.providers.kokoro import KokoroConfig
        from champi_tts.providers.kokoro.adapter import KokoroProviderAdapter

        kokoro_config: KokoroConfig
        if config is None:
            if config_kwargs:
                # Create config from kwargs
                kokoro_config = KokoroConfig(**config_kwargs)
            else:
                # Use default config
                kokoro_config = KokoroConfig()
        else:
            assert isinstance(
                config, KokoroConfig
            ), "config must be KokoroConfig for kokoro provider"
            kokoro_config = config

        return KokoroProviderAdapter(config=kokoro_config)

    # Future providers:
    # elif provider_type == "openai":
    #     from champi_tts.providers.openai import OpenAIConfig, OpenAITTSProvider
    #     config = config or OpenAIConfig()
    #     return OpenAITTSProvider(config=config)
    #
    # elif provider_type == "elevenlabs":
    #     from champi_tts.providers.elevenlabs import ElevenLabsConfig, ElevenLabsTTSProvider
    #     config = config or ElevenLabsConfig()
    #     return ElevenLabsTTSProvider(config=config)

    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. " f"Supported providers: kokoro"
        )


def list_providers() -> list[str]:
    """
    Get list of available TTS providers.

    Returns:
        List of provider names
    """
    return ["kokoro"]  # Will grow as we add more providers


def get_default_provider() -> BaseTTSProvider:
    """
    Get the default TTS provider (Kokoro).

    Returns:
        Default provider instance
    """
    return get_provider("kokoro")


def get_reader(
    provider_type: ProviderType = "kokoro",
    show_ui: bool = False,
    config: BaseTTSConfig | None = None,
    **config_kwargs,
) -> TextReaderService:
    """
    Factory function to create a text reader service.

    Args:
        provider_type: Type of provider to use
        show_ui: Whether to show visual UI indicator
        config: Pre-configured provider config (optional)
        **config_kwargs: Config parameters (if config not provided)

    Returns:
        Text reader service instance

    Examples:
        # Basic reader
        reader = get_reader("kokoro")
        await reader.read_file("document.txt")

        # With UI
        reader = get_reader("kokoro", show_ui=True)
        await reader.read_text("Hello world")

        # With custom config
        reader = get_reader(
            "kokoro",
            show_ui=True,
            default_voice="af_bella",
            default_speed=1.2
        )
    """
    provider = get_provider(provider_type, config=config, **config_kwargs)
    return TextReaderService(provider=provider, show_ui=show_ui)
