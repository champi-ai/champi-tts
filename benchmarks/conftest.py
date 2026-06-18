"""Shared fixtures for TTS benchmarks."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from champi_tts.core.base_config import BaseTTSConfig
from champi_tts.core.base_provider import BaseTTSProvider


class BenchmarkConfig(BaseTTSConfig):
    """Minimal config for benchmarks — no model or GPU required."""

    @classmethod
    def from_env(cls) -> BenchmarkConfig:
        return cls()

    def validate(self) -> bool:
        return True


class BenchmarkProvider(BaseTTSProvider):
    """Zero-latency mock provider for measuring dispatch overhead."""

    def __init__(self) -> None:
        super().__init__(BenchmarkConfig())

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        **kwargs,
    ) -> np.ndarray:
        return np.zeros(1000, dtype=np.float32)

    async def synthesize_streaming(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        **kwargs,
    ):
        for _ in range(3):
            yield np.zeros(100, dtype=np.float32)

    async def list_voices(self) -> list[str]:
        return ["voice1", "voice2"]

    async def interrupt(self) -> None:
        self._is_speaking = False


@pytest.fixture(scope="session")
def sample_audio_1s() -> np.ndarray:
    """One second of 440 Hz sine at 24 kHz, float32."""
    t = np.linspace(0, 1.0, 24000, endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture(scope="session")
def sample_audio_5s() -> np.ndarray:
    """Five seconds of 440 Hz sine at 24 kHz, float32."""
    t = np.linspace(0, 5.0, 120000, endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def benchmark_provider() -> BenchmarkProvider:
    """Fresh BenchmarkProvider instance per test."""
    return BenchmarkProvider()


@pytest.fixture
def mocked_reader_provider() -> MagicMock:
    """Minimal MagicMock satisfying TextReaderService's constructor requirements."""
    provider = MagicMock()
    provider.config.sample_rate = 24000
    return provider
