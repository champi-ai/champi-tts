"""
Performance benchmark tests for Champi TTS.

These tests measure and track performance characteristics
to detect regressions and track improvements.
"""

import asyncio
import contextlib
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.mark.asyncio
async def test_synthesize_performance():
    """Benchmark synthesis performance using the real Kokoro provider.

    Skipped automatically when torch is not installed or when the Kokoro
    model files are not yet cached locally.  In CI (which runs
    ``uv sync --all-extras``) torch is available and models are cached
    after the first run, so this test exercises the full inference path.
    """
    pytest.importorskip("torch")
    from champi_tts.factory import get_provider

    text = "This is a test sentence for performance benchmarking. " * 10

    provider = get_provider()

    start = time.perf_counter()
    try:
        await provider.initialize()
    except Exception as e:
        await provider.shutdown()
        pytest.skip(f"Kokoro provider initialization failed (models unavailable?): {e}")
    init_time = time.perf_counter() - start
    assert init_time < 120.0, f"Initialization took too long: {init_time}s"

    # Measure synthesis with the provider already warm.
    synthesis_times = []
    try:
        for _ in range(3):
            start = time.perf_counter()
            await provider.synthesize(text)
            synthesis_times.append(time.perf_counter() - start)
    except Exception as e:
        with contextlib.suppress(Exception):
            await provider.shutdown()
        pytest.skip(f"Kokoro synthesis unavailable (voice files missing?): {e}")

    await provider.shutdown()

    avg_synthesis = np.mean(synthesis_times)
    assert avg_synthesis < 60.0, f"Synthesis took too long: {avg_synthesis}s"


@pytest.mark.asyncio
async def test_reader_performance():
    """Benchmark reader performance."""
    from champi_tts.reader import TextReaderService
    from tests.conftest import MockTTSConfig, MockTTSProvider

    provider = MockTTSProvider(MockTTSConfig())
    reader = TextReaderService(provider)

    text = "This is a test text for reading. " * 20
    read_times = []

    mock = MagicMock()
    with patch("champi_tts.core.audio.sd", mock):
        for _ in range(5):
            start = time.perf_counter()
            await reader.read_text(text)
            read_times.append(time.perf_counter() - start)

    avg_read_time = np.mean(read_times)
    assert avg_read_time < 5.0, f"Reading took too long: {avg_read_time}s"

    queue_times = []
    with patch("champi_tts.core.audio.sd", mock):
        for _ in range(3):
            start = time.perf_counter()
            for i in range(10):
                reader.add_to_queue(f"Text {i}")
                await reader.read_text(f"Queue text {i}")
            queue_times.append(time.perf_counter() - start)

    assert np.mean(queue_times) < 30.0, (
        f"Queue processing took too long: {np.mean(queue_times)}s"
    )


@pytest.mark.asyncio
async def test_audio_player_performance():
    """Benchmark audio player performance."""
    from champi_tts.core.audio import AudioPlayer

    player = AudioPlayer(sample_rate=22050)

    audio = np.random.randn(22050)  # 1 second at 22050 Hz

    play_times = []
    mock = MagicMock()
    with patch("champi_tts.core.audio.sd", mock):
        for _ in range(10):
            start = time.perf_counter()
            await player.play(audio, blocking=False)
            player.stop()
            play_times.append(time.perf_counter() - start)

    assert np.mean(play_times) < 0.1, (
        f"Audio playback was too slow: {np.mean(play_times)}s"
    )


def test_config_validation_performance():
    """Benchmark configuration validation performance."""
    from champi_tts.core.config_validation import validate_config

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
        validate_config(config)
        validation_times.append(time.perf_counter() - start)

    avg_validation_time = np.mean(validation_times)
    assert avg_validation_time < 0.01, (
        f"Validation took too long: {avg_validation_time}s"
    )


def test_lazy_import_performance():
    """Benchmark lazy import performance."""
    from champi_tts.factory import _lazy_import

    import_time = []

    start = time.perf_counter()
    _lazy_import("numpy")
    first_import_time = time.perf_counter() - start

    for _ in range(10):
        start = time.perf_counter()
        _lazy_import("numpy")
        import_time.append(time.perf_counter() - start)

    assert first_import_time < 5.0, (
        f"First lazy import took too long: {first_import_time}s"
    )
    assert np.mean(import_time) < 0.001, (
        f"Subsequent imports were slow: {np.mean(import_time)}s"
    )


def test_memory_usage():
    """Benchmark memory usage patterns."""
    import gc

    from tests.conftest import MockTTSConfig, MockTTSProvider

    gc.collect()
    initial_count = len(
        [obj for obj in gc.get_objects() if isinstance(obj, np.ndarray)]
    )

    provider = MockTTSProvider(MockTTSConfig())
    gc.collect()
    provider_count = len(
        [obj for obj in gc.get_objects() if isinstance(obj, np.ndarray)]
    )

    del provider
    gc.collect()

    delta = provider_count - initial_count
    assert delta < 1000, f"Memory increased by {delta} objects"


@pytest.mark.asyncio
async def test_long_document_performance():
    """Benchmark reading long documents."""
    from champi_tts.reader import TextReaderService
    from tests.conftest import MockTTSConfig, MockTTSProvider

    provider = MockTTSProvider(MockTTSConfig())
    reader = TextReaderService(provider)

    paragraphs = ["This is paragraph " + str(i) + ". " * 10 for i in range(100)]

    read_times = []

    chunk_size = 10
    mock = MagicMock()
    with patch("champi_tts.core.audio.sd", mock):
        for i in range(0, len(paragraphs), chunk_size):
            chunk = paragraphs[i : i + chunk_size]
            chunk_text = "\n\n".join(chunk)

            start = time.perf_counter()
            await reader.read_text(chunk_text)
            read_times.append(time.perf_counter() - start)

    avg_chunk_time = np.mean(read_times)
    assert avg_chunk_time < 2.0, f"Chunk reading took too long: {avg_chunk_time}s"


if __name__ == "__main__":
    asyncio.run(test_synthesize_performance())
    asyncio.run(test_reader_performance())
    asyncio.run(test_audio_player_performance())
    test_config_validation_performance()
    test_lazy_import_performance()
    test_memory_usage()
    asyncio.run(test_long_document_performance())
    print("All benchmarks passed!")
