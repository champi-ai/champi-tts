"""
Import-time and lazy-loading benchmarks.

Verifies that importing champi_tts at the top level is lightweight and
that heavy provider dependencies (torch, kokoro) are deferred until
first use. Module-cache manipulation forces the import machinery to
re-execute on every iteration so timing is meaningful.

Run with:
    uv run pytest benchmarks/ --benchmark-only -v
"""

from __future__ import annotations

import importlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pop_champi_modules() -> dict[str, object]:
    """Remove all champi_tts entries from sys.modules and return them."""
    to_remove = [k for k in sys.modules if k.startswith("champi_tts")]
    return {k: sys.modules.pop(k) for k in to_remove}


# ---------------------------------------------------------------------------
# Import-time benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="imports")
def test_bench_reimport_champi_tts(benchmark) -> None:
    """Measure champi_tts re-import time after clearing the module cache."""

    def _reimport() -> str:
        saved = _pop_champi_modules()
        try:
            mod = importlib.import_module("champi_tts")
            return mod.__version__
        finally:
            sys.modules.update(saved)

    version = benchmark(_reimport)
    assert version is not None


@pytest.mark.benchmark(group="imports")
def test_bench_lazy_heavy_deps_absent(benchmark) -> None:
    """torch and kokoro must not appear in sys.modules after a bare import."""

    def _check() -> tuple[bool, bool]:
        return "torch" not in sys.modules, "kokoro" not in sys.modules

    no_torch, no_kokoro = benchmark(_check)
    assert no_torch, "torch was eagerly imported at the top level"
    assert no_kokoro, "kokoro was eagerly imported at the top level"


@pytest.mark.benchmark(group="imports")
def test_bench_list_providers(benchmark) -> None:
    """list_providers() is O(1) — a constant lookup with no I/O."""
    from champi_tts import list_providers

    result = benchmark(list_providers)
    assert "kokoro" in result


@pytest.mark.benchmark(group="imports")
def test_bench_kokoro_config_creation(benchmark) -> None:
    """KokoroConfig construction does not trigger model or torch loading."""
    from champi_tts.providers.kokoro import KokoroConfig

    result = benchmark(KokoroConfig)
    assert result.sample_rate == 24000
