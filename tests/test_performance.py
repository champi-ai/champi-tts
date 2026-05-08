"""
Performance benchmark tests for Champi TTS.

These tests measure and track performance characteristics
to detect regressions and track improvements.
"""

import asyncio
import time
import timeit
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.asyncio
async def test_synthesize_performance():
    """Benchmark synthesis performance."""
    from champi_tts.factory import get_provider
    from tests.conftest import MockTTSConfig, MockTTSProvider

    # Use mock provider for timing without actual synthesis
    config = MockTTSConfig()
    provider = MockTTSProvider(config)
    provider._initialized = True

    text = "This is a test sentence for performance benchmarking. " * 10

    # Measure initialization time
    init_times = []
    for _ in range(5):
        start = time.perf_counter()
        provider = get_provider()
        await provider.initialize()
        init_times.append(time.perf_counter() - start)
        await provider.shutdown()

    avg_init_time = np.mean(init_times)
    assert avg_init_time < 5.0, f"Initialization took too long: {avg_init_time}s"

    # Measure first synthesis time (includes loading)
    synthesis_times = []
    for _ in range(3):
        start = time.perf_counter()
        audio = await provider.synthesize(text)
        synthesis_times.append(time.perf_counter() - start)
        await provider.shutdown()

    avg_first_synthesis = np.mean(synthesis_times)
    assert avg_first_synthesis < 30.0, f"First synthesis took too long: {avg_first_synthesis}s"

    # Measure subsequent synthesis times (no loading overhead)
    synthesis_times = []
    for _ in range(3):
        start = time.perf_counter()
        audio = await provider.synthesize(text)
        synthesis_times.append(time.perf_counter() - start)
        await provider.shutdown()

    avg_subsequent_synthesis = np.mean(synthesis_times)
    # Allow some variance but should be reasonable
    assert avg_subsequent_synthesis < 10.0, f"Subsequent synthesis took too long: {avg_subsequent_synthesis}s"


@pytest.mark.asyncio
async def test_reader_performance():
    """Benchmark reader performance."""
    from champi_tts.factory import get_reader
    from tests.conftest import MockTTSConfig, MockTTSProvider

    provider = MockTTSProvider(MockTTSConfig())
    reader = get_reader("kokoro", provider=provider)

    # Measure read_text performance
    text = "This is a test text for reading. " * 20
    read_times = []

    for _ in range(5):
        start = time.perf_counter()
        await reader.read_text(text)
        read_times.append(time.perf_counter() - start)

    avg_read_time = np.mean(read_times)
    assert avg_read_time < 5.0, f"Reading took too long: {avg_read_time}s"

    # Measure queue performance
    queue_times = []
    for _ in range(3):
        start = time.perf_counter()
        for i in range(10):
            reader.add_to_queue(f"Text {i}")
            await reader.read_text(f"Queue text {i}")
        queue_times.append(time.perf_counter() - start)

    assert np.mean(queue_times) < 30.0, f"Queue processing took too long: {np.mean(queue_times)}s"


@pytest.mark.asyncio
async def test_audio_player_performance():
    """Benchmark audio player performance."""
    from champi_tts.core.audio import AudioPlayer

    player = AudioPlayer(sample_rate=22050)

    # Create test audio
    audio = np.random.randn(22050)  # 1 second at 22050 Hz

    play_times = []
    for _ in range(10):
        start = time.perf_counter()
        await player.play(audio, blocking=False)
        player.stop()
        play_times.append(time.perf_counter() - start)

    assert np.mean(play_times) < 0.1, f"Audio playback was too slow: {np.mean(play_times)}s"


def test_config_validation_performance():
    """Benchmark configuration validation performance."""
    from champi_tts.core.config_validation import validate_config

    # Benchmark validation time
    config = {
        "default_voice": "af_bella",
        "default_speed": 1.0,
        "model_path": "/models/kokoro",
        "cache_path": "/cache/champi",
        "sample_rate": 22050,
    }

    validation_times = []
    for _ in range(100):
        start = time.perf_counter()
        errors = validate_config(config)
        validation_times.append(time.perf_counter() - start)

    avg_validation_time = np.mean(validation_times)
    assert avg_validation_time < 0.01, f"Validation took too long: {avg_validation_time}s"


def test_lazy_import_performance():
    """Benchmark lazy import performance."""
    from champi_tts.factory import _lazy_import

    import_time = []

    # First import (should take longer)
    start = time.perf_counter()
    module = _lazy_import("numpy")
    first_import_time = time.perf_counter() - start

    # Subsequent imports (should be fast)
    for _ in range(10):
        start = time.perf_counter()
        module = _lazy_import("numpy")
        import_time.append(time.perf_counter() - start)

    assert first_import_time < 5.0, f"First lazy import took too long: {first_import_time}s"
    assert np.mean(import_time) < 0.001, f"Subsequent imports were slow: {np.mean(import_time)}s"


def test_memory_usage():
    """Benchmark memory usage patterns."""
    import gc

    from champi_tts.factory import get_provider
    from tests.conftest import MockTTSConfig, MockTTSProvider

    # Get initial memory
    gc.collect()
    initial_memory = sum(obj for obj in gc.get_objects() if isinstance(obj, np.ndarray))

    # Create provider
    provider = get_provider()
    gc.collect()
    provider_memory = sum(obj for obj in gc.get_objects() if isinstance(obj, np.ndarray))

    # Cleanup
    provider.shutdown()
    gc.collect()
    final_memory = sum(obj for obj in gc.get_objects() if isinstance(obj, np.ndarray))

    # Memory should not increase significantly
    delta = provider_memory - initial_memory
    assert delta < 1000, f"Memory increased by {delta} objects"


@pytest.mark.asyncio
async def test_long_document_performance():
    """Benchmark reading long documents."""

    from champi_tts.factory import get_reader
    from tests.conftest import MockTTSConfig, MockTTSProvider

    provider = MockTTSProvider(MockTTSConfig())
    reader = get_reader("kokoro", provider=provider)

    # Create long text (like a book chapter)
    paragraphs = ["This is paragraph " + str(i) + ". " * 10 for i in range(100)]
    long_text = "\n\n".join(paragraphs)

    read_times = []

    # Read in chunks
    chunk_size = 10
    for i in range(0, len(paragraphs), chunk_size):
        chunk = paragraphs[i : i + chunk_size]
        chunk_text = "\n\n".join(chunk)

        start = time.perf_counter()
        await reader.read_text(chunk_text)
        read_times.append(time.perf_counter() - start)

    avg_chunk_time = np.mean(read_times)
    assert avg_chunk_time < 2.0, f"Chunk reading took too long: {avg_chunk_time}s"


if __name__ == "__main__":
    # Run benchmarks
    asyncio.run(test_synthesize_performance())
    asyncio.run(test_reader_performance())
    asyncio.run(test_audio_player_performance())
    test_config_validation_performance()
    test_lazy_import_performance()
    test_memory_usage()
    asyncio.run(test_long_document_performance())
    print("All benchmarks passed!")
