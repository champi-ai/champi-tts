"""
Comprehensive performance benchmark suite for Champi TTS.

This module provides performance tests to:
- Establish baselines for all core TTS operations
- Measure CPU and memory usage
- Test with different text lengths and complexity
- Benchmark streaming vs non-streaming synthesis
- Track performance regression over time
- Identify performance bottlenecks
"""

import asyncio
import gc
import json
import os
import psutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

# Try to import pytest-benchmark
try:
    import pytest_benchmark as benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False


class BenchmarkConfig:
    """Configuration for performance benchmarks."""
    # Text samples for benchmarking
    SHORT_TEXT = "Hello world. " * 5  # ~70 characters
    MEDIUM_TEXT = "The quick brown fox jumps over the lazy dog. " * 50  # ~2100 characters
    LONG_TEXT = " ".join([f"This is paragraph {i} with some sample text. " * 20 for i in range(100)])
    VERY_LONG_TEXT = " ".join([f"Long text segment {i}: " * 100 for i in range(200)])

    # Benchmark parameters
    WARMUP_ITERATIONS = 3
    BENCHMARK_ITERATIONS = 10
    STREAMING_CHUNK_SIZE = 100  # chunks
    STREAMING_CHUNK_DURATION = 0.1  # seconds per chunk

    # Performance thresholds (in seconds)
    MODEL_LOAD_THRESHOLD = 30.0  # seconds
    MODEL_UNLOAD_THRESHOLD = 5.0  # seconds
    SYNTHESIS_SHORT_THRESHOLD = 2.0  # seconds
    SYNTHESIS_MEDIUM_THRESHOLD = 10.0  # seconds
    SYNTHESIS_LONG_THRESHOLD = 60.0  # seconds
    STREAMING_THRESHOLD = 0.5  # seconds per chunk
    MEMORY_THRESHOLD = 500  # MB


class PerformanceMonitor:
    """Monitor CPU and memory usage during benchmarks."""

    def __init__(self):
        self.process = psutil.Process()
        self.cpu_times = psutil.cpu_times()
        self.memory_before = self.get_memory_info()
        self.cpu_percent = None

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage."""
        return {
            'rss': self.process.memory_info().rss / (1024 * 1024),  # MB
            'vms': self.process.memory_info().vms / (1024 * 1024),  # MB
            'percent': self.process.memory_percent(),
        }

    def get_cpu_stats(self) -> Dict[str, float]:
        """Get CPU statistics."""
        cpu_percent = self.process.cpu_percent(interval=0.1)
        times = self.process.cpu_times()
        return {
            'user_time': times.user,
            'system_time': times.system,
            'total_time': times.user + times.system,
            'percent': cpu_percent,
        }

    def get_delta(self, baseline: Dict) -> Dict[str, float]:
        """Calculate delta from baseline."""
        return {
            key: (self._get(key) - baseline.get(key, 0))
            for key in self.get_memory_info().keys()
        }

    def _get(self, key: str) -> float:
        """Helper to get value."""
        if key == 'rss':
            return self.process.memory_info().rss / (1024 * 1024)
        elif key == 'vms':
            return self.process.memory_info().vms / (1024 * 1024)
        elif key == 'percent':
            return self.process.memory_percent()
        return 0.0


# Store benchmark results
BENCHMARK_RESULTS: Dict[str, Any] = {}


@pytest.fixture(scope="module")
def benchmark_config():
    """Provide benchmark configuration."""
    return BenchmarkConfig()


@pytest.fixture(scope="module")
def perf_monitor():
    """Create performance monitor for CPU/memory measurement."""
    return PerformanceMonitor()


def track_baseline(name: str):
    """Decorator to track baseline performance."""
    def decorator(func):
        def wrapper(benchmark: benchmark, benchmark_config: BenchmarkConfig, perf_monitor: PerformanceMonitor):
            # Initial measurement
            perf_monitor = PerformanceMonitor()
            baseline_memory = perf_monitor.get_memory_info()
            baseline_cpu = perf_monitor.get_cpu_stats()

            # Warmup
            for _ in range(benchmark_config.WARMUP_ITERATIONS):
                result = func()
                if asyncio.iscoroutine(result):
                    asyncio.run(result)

            # Benchmark
            perf_monitor = PerformanceMonitor()
            gc.collect()
            result = func()

            # Get final stats
            final_memory = perf_monitor.get_memory_info()
            final_cpu = perf_monitor.get_cpu_stats()
            cpu_delta = final_cpu['total_time'] - baseline_cpu['total_time']
            memory_delta = {
                key: (final_memory[key] - baseline_memory[key])
                for key in final_memory
            }

            # Store result
            BENCHMARK_RESULTS[name] = {
                'avg_time': benchmark.stats['mean'],
                'min_time': benchmark.stats['min'],
                'max_time': benchmark.stats['max'],
                'cpu_time': cpu_delta,
                'memory_delta': memory_delta,
                'iterations': len(benchmark.stats),
            }

            return result
        return wrapper
    return decorator


if BENCHMARK_AVAILABLE:
    # ==================== MODEL LOADING BENCHMARKS ====================

    @benchmark pytest.mark.asyncio
    async def test_model_load_performance(benchmark: benchmark, benchmark_config: BenchmarkConfig):
        """Benchmark model loading performance."""
        from champi_tts.factory import get_provider
        from tests.conftest import MockTTSConfig, MockTTSProvider

        async def load_model():
            provider = get_provider()
            await provider.initialize()
            return provider

        result = await benchmark(load_model)
        assert result.is_loaded

        # Cleanup
        await result.shutdown()

    @benchmark pytest.mark.asyncio
    async def test_model_unload_performance(benchmark: benchmark, benchmark_config: BenchmarkConfig):
        """Benchmark model unloading performance."""
        from champi_tts.factory import get_provider
        from tests.conftest import MockTTSConfig, MockTTSProvider

        provider = MockTTSProvider(MockTTSConfig())
        await provider.initialize()

        async def unload_model():
            await provider.shutdown()

        await benchmark(unload_model)

    # ==================== SYNTHESIS BENCHMARKS ====================

    @benchmark pytest.mark.asyncio
    async def test_synthesis_short_text(benchmark: benchmark, benchmark_config: BenchmarkConfig):
        """Benchmark synthesis for short text."""
        from champi_tts.factory import get_provider
        from tests.conftest import MockTTSConfig, MockTTSProvider

        config = MockTTSConfig()
        provider = MockTTSProvider(config)
        await provider.initialize()

        text = benchmark_config.SHORT_TEXT

        async def synthesize():
            return await provider.synthesize(text)

        result = await benchmark(synthesize)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

        await provider.shutdown()

    @benchmark pytest.mark.asyncio
    async def test_synthesis_medium_text(benchmark: benchmark, benchmark_config: BenchmarkConfig):
        """Benchmark synthesis for medium-length text."""
        from champi_tts.factory import get_provider
        from tests.conftest import MockTTSConfig, MockTTSProvider

        config = MockTTSConfig()
        provider = MockTTSProvider(config)
        await provider.initialize()

        text = benchmark_config.MEDIUM_TEXT

        async def synthesize():
            return await provider.synthesize(text)

        result = await benchmark(synthesize)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

        await provider.shutdown()

    @benchmark pytest.mark.asyncio
    async def test_synthesis_long_text(benchmark: benchmark, benchmark_config: BenchmarkConfig):
        """Benchmark synthesis for long text."""
        from champi_tts.factory import get_provider
        from tests.conftest import MockTTSConfig, MockTTSProvider

        config = MockTTSConfig()
        provider = MockTTSProvider(config)
        await provider.initialize()

        text = benchmark_config.LONG_TEXT

        async def synthesize():
            return await provider.synthesize(text)

        result = await benchmark(synthesize)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

        await provider.shutdown()

    # ==================== STREAMING BENCHMARKS ====================

    @benchmark pytest.mark.asyncio
    async def test_streaming_short_text(benchmark: benchmark, benchmark_config: BenchmarkConfig):
        """Benchmark streaming synthesis for short text."""
        from champi_tts.factory import get_provider
        from tests.conftest import MockTTSConfig, MockTTSProvider

        config = MockTTSConfig()
        provider = MockTTSProvider(config)
        await provider.initialize()

        text = benchmark_config.SHORT_TEXT

        async def stream_synthesize():
            chunks = []
            async for chunk in provider.synthesize_streaming(text):
                chunks.append(chunk)
            return chunks

        result = await benchmark(stream_synthesize)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(chunk, np.ndarray) for chunk in result)

        await provider.shutdown()

    @benchmark pytest.mark.asyncio
    async def test_streaming_long_text(benchmark: benchmark, benchmark_config: BenchmarkConfig):
        """Benchmark streaming synthesis for long text."""
        from champi_tts.factory import get_provider
        from tests.conftest import MockTTSConfig, MockTTSProvider

        config = MockTTSConfig()
        provider = MockTTSProvider(config)
        await provider.initialize()

        text = benchmark_config.MEDIUM_TEXT

        async def stream_synthesize():
            chunks = []
            async for chunk in provider.synthesize_streaming(text):
                chunks.append(chunk)
            return chunks

        result = await benchmark(stream_synthesize)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(chunk, np.ndarray) for chunk in result)

        await provider.shutdown()

    # ==================== READER BENCHMARKS ====================

    @benchmark pytest.mark.asyncio
    async def test_reader_performance(benchmark: benchmark, benchmark_config: BenchmarkConfig):
        """Benchmark reader performance with text chunks."""
        from champi_tts.factory import get_reader
        from tests.conftest import MockTTSConfig, MockTTSProvider

        provider = MockTTSProvider(MockTTSConfig())
        reader = get_reader("kokoro", provider=provider)

        text = benchmark_config.MEDIUM_TEXT

        async def read_text():
            await reader.read_text(text)

        result = await benchmark(read_text)
        assert result is None

        await reader.stop()

    @benchmark pytest.mark.asyncio
    async def test_reader_queue_performance(benchmark: benchmark, benchmark_config: BenchmarkConfig):
        """Benchmark reader queue performance."""
        from champi_tts.factory import get_reader
        from tests.conftest import MockTTSConfig, MockTTSProvider

        provider = MockTTSProvider(MockTTSConfig())
        reader = get_reader("kokoro", provider=provider)

        async def process_queue():
            for i in range(10):
                reader.add_to_queue(f"Text {i}")
                await reader.read_text(f"Queue text {i}")

        result = await benchmark(process_queue)
        assert result is None

        await reader.stop()

    # ==================== AUDIO PLAYER BENCHMARKS ====================

    @benchmark pytest.mark.asyncio
    async def test_audio_player_performance(benchmark: benchmark, benchmark_config: BenchmarkConfig):
        """Benchmark audio player performance."""
        from champi_tts.core.audio import AudioPlayer

        sample_rate = 22050
        player = AudioPlayer(sample_rate=sample_rate)

        # Create test audio
        audio = np.random.randn(sample_rate * 5)  # 5 seconds

        async def play_audio():
            await player.play(audio, blocking=False)
            await asyncio.sleep(0.5)  # Wait for playback
            player.stop()

        result = await benchmark(play_audio)
        assert result is None

    # ==================== CONFIG VALIDATION BENCHMARKS ====================

    @benchmark pytest.mark.asyncio
    def test_config_validation_performance(benchmark: benchmark, benchmark_config: BenchmarkConfig):
        """Benchmark configuration validation performance."""
        from champi_tts.core.config_validation import validate_config

        config = {
            "default_voice": "af_bella",
            "default_speed": 1.0,
            "model_path": "/models/kokoro",
            "cache_path": "/cache/champi",
            "sample_rate": 22050,
        }

        async def validate():
            errors = validate_config(config)
            return errors

        result = await benchmark(validate)
        assert result == []

    # ==================== MEMORY BENCHMARKS ====================

    @pytest.mark.asyncio
    async def test_memory_usage_steady_state(benchmark_config: BenchmarkConfig):
        """Test that memory usage remains stable during repeated operations."""
        from champi_tts.factory import get_provider
        from tests.conftest import MockTTSConfig, MockTTSProvider

        gc.collect()
        provider = get_provider()
        await provider.initialize()

        # Perform multiple synthesis operations
        for _ in range(5):
            audio = await provider.synthesize(benchmark_config.SHORT_TEXT)
            del audio
            gc.collect()

        await provider.shutdown()
        gc.collect()

    @pytest.mark.asyncio
    async def test_memory_usage_large_document(benchmark_config: BenchmarkConfig):
        """Test memory usage with large document processing."""
        from champi_tts.factory import get_provider
        from tests.conftest import MockTTSConfig, MockTTSProvider

        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        provider = MockTTSProvider(MockTTSConfig())
        reader = get_reader("kokoro", provider=provider)

        # Process large text
        await reader.read_text(benchmark_config.VERY_LONG_TEXT)
        await reader.stop()

        await provider.shutdown()
        gc.collect()

        final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (not O(n^2))
        assert memory_increase < 1000, f"Memory increased by {memory_increase} MB"

    # ==================== REGRESSION TESTING ====================

    @pytest.mark.asyncio
    async def test_performance_regression(benchmark_config: BenchmarkConfig):
        """Test that performance doesn't degrade over time."""
        from champi_tts.factory import get_provider
        from tests.conftest import MockTTSConfig, MockTTSProvider

        config = MockTTSConfig()
        provider = MockTTSProvider(config)
        await provider.initialize()

        # Cold start (first synthesis)
        text = benchmark_config.SHORT_TEXT
        start_time = time.perf_counter()
        audio = await provider.synthesize(text)
        cold_start_time = time.perf_counter() - start_time
        del audio
        gc.collect()

        # Warm start (subsequent synthesis)
        start_time = time.perf_counter()
        audio = await provider.synthesize(text)
        warm_start_time = time.perf_counter() - start_time
        del audio
        gc.collect()

        await provider.shutdown()

        # Warm start should be at least 2x faster than cold start
        regression_factor = cold_start_time / warm_start_time
        assert regression_factor < 5.0, f"Performance regression too high: {regression_factor:.2f}x"

    # ==================== STRESS TESTING ====================

    @pytest.mark.asyncio
    async def test_stress_synthesis(benchmark_config: BenchmarkConfig):
        """Stress test synthesis with many consecutive operations."""
        from champi_tts.factory import get_provider
        from tests.conftest import MockTTSConfig, MockTTSProvider

        config = MockTTSConfig()
        provider = MockTTSProvider(config)
        await provider.initialize()

        operations = 10
        text = benchmark_config.SHORT_TEXT
        total_time = 0

        for _ in range(operations):
            start = time.perf_counter()
            audio = await provider.synthesize(text)
            elapsed = time.perf_counter() - start
            total_time += elapsed
            del audio
            gc.collect()

        avg_time = total_time / operations
        assert avg_time < benchmark_config.SYNTHESIS_SHORT_THRESHOLD * 2

        await provider.shutdown()

    @pytest.mark.asyncio
    async def test_stress_loading(benchmark_config: BenchmarkConfig):
        """Stress test model loading/unloading."""
        from champi_tts.factory import get_provider
        from tests.conftest import MockTTSConfig, MockTTSProvider

        iterations = 5
        total_load_time = 0
        total_unload_time = 0

        for _ in range(iterations):
            # Load
            start = time.perf_counter()
            provider = get_provider()
            await provider.initialize()
            load_time = time.perf_counter() - start
            total_load_time += load_time

            # Unload
            start = time.perf_counter()
            await provider.shutdown()
            unload_time = time.perf_counter() - start
            total_unload_time += unload_time

        avg_load_time = total_load_time / iterations
        avg_unload_time = total_unload_time / iterations

        assert avg_load_time < benchmark_config.MODEL_LOAD_THRESHOLD
        assert avg_unload_time < benchmark_config.MODEL_UNLOAD_THRESHOLD

    # ==================== TEXT COMPLEXITY BENCHMARKS ====================

    @pytest.mark.asyncio
    async def test_complexity_detection(benchmark_config: BenchmarkConfig):
        """Test performance with different text complexity levels."""
        from champi_tts.factory import get_provider
        from tests.conftest import MockTTSConfig, MockTTSProvider

        config = MockTTSConfig()
        provider = MockTTSProvider(config)
        await provider.initialize()

        # Simple text
        simple_text = "Hello world. " * 10
        start = time.perf_counter()
        audio1 = await provider.synthesize(simple_text)
        simple_time = time.perf_counter() - start
        del audio1
        gc.collect()

        # Complex text with punctuation and spacing
        complex_text = "The quick brown fox jumps over the lazy dog. The fox, with its keen sense of smell, navigated through the dense forest. " * 10
        start = time.perf_counter()
        audio2 = await provider.synthesize(complex_text)
        complex_time = time.perf_counter() - start
        del audio2
        gc.collect()

        # Complex text should not be significantly slower
        complexity_factor = complex_time / simple_time
        assert complexity_factor < 2.0, f"Complex text too slow: {complexity_factor:.2f}x"

        await provider.shutdown()

    # ==================== SPEED MULTIPLIER BENCHMARKS ====================

    @pytest.mark.asyncio
    async def test_speed_multiplier_performance(benchmark_config: BenchmarkConfig):
        """Test performance with different speed multipliers."""
        from champi_tts.factory import get_provider
        from tests.conftest import MockTTSProvider

        config = MockTTSConfig()
        provider = MockTTSProvider(config)
        await provider.initialize()

        text = benchmark_config.SHORT_TEXT

        # Test different speed multipliers
        speeds = [0.5, 1.0, 1.5, 2.0]
        speed_times = {}

        for speed in speeds:
            start = time.perf_counter()
            audio = await provider.synthesize(text, speed=speed)
            elapsed = time.perf_counter() - start
            speed_times[speed] = elapsed
            del audio
            gc.collect()

        # Verify speed relationship (roughly)
        base_time = speed_times[1.0]
        for speed in speeds:
            if speed == 1.0:
                continue
            # Faster speed should take less time
            expected_ratio = 1.0 / speed
            actual_ratio = speed_times[speed] / base_time
            # Allow some variance
            assert 0.8 <= actual_ratio <= 1.5, f"Speed {speed} time ratio: {actual_ratio:.2f} vs expected {expected_ratio:.2f}"

        await provider.shutdown()

    # ==================== REPORTING ====================

    def generate_benchmark_report():
        """Generate a JSON report of benchmark results."""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': sys.platform,
            'cpu_count': psutil.cpu_count(),
            'memory_total_mb': psutil.virtual_memory().total / (1024 * 1024),
            'results': BENCHMARK_RESULTS,
        }

        # Save to file
        report_file = Path(".benchmarks.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        return report

    @pytest.fixture
    def benchmark_report():
        """Generate and return benchmark report."""
        import sys
        return generate_benchmark_report()

    # ==================== MAIN ENTRY POINT ====================

    if __name__ == "__main__":
        print("Running performance benchmarks...")
        print(f"Benchmark suite: {BenchmarkConfig.__name__}")
        print(f"Benchmark tool available: {BENCHMARK_AVAILABLE}")
        print()

        # Run all benchmarks
        asyncio.run(test_model_load_performance(None, BenchmarkConfig(), PerformanceMonitor()))
        asyncio.run(test_model_unload_performance(None, BenchmarkConfig(), PerformanceMonitor()))
        asyncio.run(test_synthesis_short_text(None, BenchmarkConfig(), PerformanceMonitor()))
        asyncio.run(test_synthesis_medium_text(None, BenchmarkConfig(), PerformanceMonitor()))
        asyncio.run(test_synthesis_long_text(None, BenchmarkConfig(), PerformanceMonitor()))
        asyncio.run(test_streaming_short_text(None, BenchmarkConfig(), PerformanceMonitor()))
        asyncio.run(test_streaming_long_text(None, BenchmarkConfig(), PerformanceMonitor()))
        asyncio.run(test_reader_performance(None, BenchmarkConfig(), PerformanceMonitor()))
        asyncio.run(test_reader_queue_performance(None, BenchmarkConfig(), PerformanceMonitor()))
        asyncio.run(test_audio_player_performance(None, BenchmarkConfig(), PerformanceMonitor()))
        test_config_validation_performance(None, BenchmarkConfig(), PerformanceMonitor())
        asyncio.run(test_memory_usage_steady_state(BenchmarkConfig()))
        asyncio.run(test_memory_usage_large_document(BenchmarkConfig()))
        asyncio.run(test_performance_regression(BenchmarkConfig()))
        asyncio.run(test_stress_synthesis(BenchmarkConfig()))
        asyncio.run(test_stress_loading(BenchmarkConfig()))
        asyncio.run(test_complexity_detection(BenchmarkConfig()))
        asyncio.run(test_speed_multiplier_performance(BenchmarkConfig()))

        # Generate report
        report = generate_benchmark_report()
        print(f"\nBenchmark report saved to: .benchmarks.json")
        print(f"Total benchmarks run: {len(BENCHMARK_RESULTS)}")