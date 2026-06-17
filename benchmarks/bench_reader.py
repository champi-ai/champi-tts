"""
Text reader queue and state benchmarks.

Measures the throughput of queue operations and state-machine transitions
in TextReaderService. All benchmarks are synchronous — no audio playback
or event-loop scheduling is involved, so no hardware is required.

Run with:
    uv run pytest benchmarks/ --benchmark-only -v
"""

from __future__ import annotations

from typing import Any

import pytest

from champi_tts.reader import ReaderState, TextReaderService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reader(mocked_reader_provider: Any) -> TextReaderService:
    """Create a TextReaderService backed by a minimal mock provider."""
    return TextReaderService(provider=mocked_reader_provider, show_ui=False)


# ---------------------------------------------------------------------------
# Queue benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="queue")
def test_bench_add_to_queue_single(benchmark, mocked_reader_provider: Any) -> None:
    """Latency of a single add_to_queue call followed by clear_queue."""
    reader = _make_reader(mocked_reader_provider)

    def _add_and_clear() -> None:
        reader.add_to_queue("Hello world.")
        reader.clear_queue()

    benchmark(_add_and_clear)
    assert reader._text_queue == []


@pytest.mark.benchmark(group="queue")
def test_bench_add_to_queue_batch(benchmark, mocked_reader_provider: Any) -> None:
    """Throughput of adding 100 items to the queue then clearing."""
    reader = _make_reader(mocked_reader_provider)
    texts = [f"Sentence number {i}." for i in range(100)]

    def _fill_and_clear() -> int:
        for t in texts:
            reader.add_to_queue(t)
        size = len(reader._text_queue)
        reader.clear_queue()
        return size

    size = benchmark(_fill_and_clear)
    assert size == 100


@pytest.mark.benchmark(group="queue")
def test_bench_clear_queue_prefilled(benchmark, mocked_reader_provider: Any) -> None:
    """Latency of clear_queue() in isolation on an already-filled 50-item queue."""
    reader = _make_reader(mocked_reader_provider)

    def _setup() -> None:
        for i in range(50):
            reader._text_queue.append(f"Item {i}.")

    benchmark.pedantic(reader.clear_queue, setup=_setup, rounds=200)


# ---------------------------------------------------------------------------
# State-machine benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="state")
def test_bench_state_transition_single(benchmark, mocked_reader_provider: Any) -> None:
    """Latency of one _set_state call including blinker signal dispatch."""
    reader = _make_reader(mocked_reader_provider)

    def _toggle() -> None:
        reader._set_state(ReaderState.READING)
        reader._set_state(ReaderState.IDLE)

    benchmark(_toggle)
    assert reader.state == ReaderState.IDLE


@pytest.mark.benchmark(group="state")
def test_bench_state_full_cycle(benchmark, mocked_reader_provider: Any) -> None:
    """Latency of a complete IDLE -> READING -> PAUSED -> STOPPED -> IDLE cycle."""
    reader = _make_reader(mocked_reader_provider)

    def _cycle() -> None:
        reader._set_state(ReaderState.READING)
        reader._set_state(ReaderState.PAUSED)
        reader._set_state(ReaderState.STOPPED)
        reader._set_state(ReaderState.IDLE)

    benchmark(_cycle)
    assert reader.state == ReaderState.IDLE


@pytest.mark.benchmark(group="state")
def test_bench_reader_construction(benchmark, mocked_reader_provider: Any) -> None:
    """Overhead of constructing a TextReaderService (includes signal setup)."""

    def _create() -> TextReaderService:
        return TextReaderService(provider=mocked_reader_provider, show_ui=False)

    reader = benchmark(_create)
    assert reader.state == ReaderState.IDLE
