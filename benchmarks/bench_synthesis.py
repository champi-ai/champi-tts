"""
Synthesis dispatch benchmarks.

Measures the async-dispatch overhead of synthesize() and streaming
synthesis using a zero-latency mock provider. No models or GPU required;
safe to run in CI.

Run with:
    uv run pytest benchmarks/ --benchmark-only -v
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_synthesize(provider: Any, text: str) -> np.ndarray:
    """Dispatch synthesize() and block until the coroutine completes."""
    return asyncio.get_event_loop().run_until_complete(provider.synthesize(text))


def _run_list_voices(provider: Any) -> list[str]:
    """Dispatch list_voices() and block until the coroutine completes."""
    return asyncio.get_event_loop().run_until_complete(provider.list_voices())


async def _collect_stream(provider: Any, text: str) -> list[np.ndarray]:
    return [chunk async for chunk in provider.synthesize_streaming(text)]


def _run_stream(provider: Any, text: str) -> list[np.ndarray]:
    """Collect all streaming chunks from synthesize_streaming()."""
    return asyncio.get_event_loop().run_until_complete(_collect_stream(provider, text))


# ---------------------------------------------------------------------------
# Latency benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="synthesis")
def test_bench_synthesize_short_text(benchmark, benchmark_provider: Any) -> None:
    """Baseline: synthesize() dispatch latency for a short phrase."""
    result = benchmark(_run_synthesize, benchmark_provider, "Hello world.")
    assert isinstance(result, np.ndarray)


@pytest.mark.benchmark(group="synthesis")
def test_bench_synthesize_medium_text(benchmark, benchmark_provider: Any) -> None:
    """Synthesize dispatch latency for a medium-length sentence."""
    text = "The quick brown fox jumps over the lazy dog. " * 3
    result = benchmark(_run_synthesize, benchmark_provider, text)
    assert isinstance(result, np.ndarray)


@pytest.mark.benchmark(group="synthesis")
def test_bench_synthesize_long_text(benchmark, benchmark_provider: Any) -> None:
    """Synthesize dispatch latency for a long paragraph."""
    text = (
        "Performance testing ensures that our TTS system meets latency requirements. "
        "Every text processing step adds overhead that accumulates at scale. "
    ) * 5
    result = benchmark(_run_synthesize, benchmark_provider, text)
    assert isinstance(result, np.ndarray)


@pytest.mark.benchmark(group="synthesis")
def test_bench_synthesize_list_voices(benchmark, benchmark_provider: Any) -> None:
    """Overhead of list_voices() dispatch."""
    result = benchmark(_run_list_voices, benchmark_provider)
    assert len(result) == 2


@pytest.mark.benchmark(group="streaming")
def test_bench_streaming_chunk_dispatch(benchmark, benchmark_provider: Any) -> None:
    """Overhead of collecting all chunks from synthesize_streaming()."""
    result = benchmark(_run_stream, benchmark_provider, "Hello streaming world.")
    # BenchmarkProvider yields exactly 3 chunks
    assert len(result) == 3


@pytest.mark.benchmark(group="streaming")
def test_bench_audio_concat_from_chunks(benchmark, sample_audio_5s: np.ndarray) -> None:
    """Overhead of assembling streaming chunks into a single array (numpy concat)."""
    chunk_size = 4800
    chunks = [
        sample_audio_5s[i : i + chunk_size]
        for i in range(0, len(sample_audio_5s), chunk_size)
    ]

    def _assemble() -> np.ndarray:
        return np.concatenate(chunks)

    result = benchmark(_assemble)
    assert len(result) == len(sample_audio_5s)
