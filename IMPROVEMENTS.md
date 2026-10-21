# Champi-TTS Improvement Backlog

This file tracks the original review items and where each one stands. New work is
tracked as GitHub issues under milestones; this file only records status and
pointers.

- **Current phase:** [Phase 5: MCP Server](https://github.com/champi-ai/champi-tts/milestone/6) —
  a twin of champi-stt's `src/champi_stt/mcp`: FastMCP server, speaker tools,
  `mcp serve` CLI, tests and docs, plus the provider fixes it depends on.
- **Hygiene:** [Maintenance: Repo & Tooling Hygiene](https://github.com/champi-ai/champi-tts/milestone/5)

Status legend: ✅ done · 🔄 tracked in an open issue · 📌 backlog (no issue yet)

---

## Critical Issues

| # | Item | Status |
|---|------|--------|
| 1 | `BaseT TSConfig` typo in `base_config.py` | ✅ Fixed |
| 2 | Missing test suite | ✅ `tests/test_core`, `test_providers`, `test_reader`, `test_integration` |
| 3 | `show_ui` not wired into `TextReaderService` | ✅ Reader creates and drives the UI indicator |

## High Priority

| # | Item | Status |
|---|------|--------|
| 4 | KokoroProvider adapter for `BaseTTSProvider` | ✅ `providers/kokoro/adapter.py`; lifecycle gaps in 🔄 #83, #84, #85, #98 |
| 5 | Voice file distribution | ✅ Voices no longer shipped in the package; unified directory and first-run download in 🔄 #96 |
| 6 | `.env.example` | ✅ Added |
| 7 | CLI interactive mode placeholder | ✅ Implemented (`--interactive`) |

## Medium Priority

| # | Item | Status |
|---|------|--------|
| 8 | Optional dependency groups | ✅ `kokoro`, `ui` extras; `mcp` extra in 🔄 #88; undeclared imports in 🔄 #100; Python 3.13 kokoro install in 🔄 #87 |
| 9 | Connect reader signals to UI state | ✅ Done in `reader/service.py` |
| 10 | Reader error state handling | ✅ `on_error` emitted and mapped to `TTSState.ERROR` |
| 11 | Async context manager support | ✅ Providers, adapter and reader support `async with` |
| 12 | Progress callbacks / notifications | 🔄 #99 |

## Documentation

| # | Item | Status |
|---|------|--------|
| 13 | API reference | ✅ `docs/api.md` and MkDocs site |
| 14 | `examples/` directory | ✅ Added |
| 15 | Migration guide from `mcp_champi.kokoro_svc` | ✅ `docs/migration.md` |
| — | MCP integration guide | 🔄 #94 |

## Testing

| # | Item | Status |
|---|------|--------|
| 16 | Unit tests | ✅ Added; MCP unit tests in 🔄 #92 |
| 17 | Integration tests | ✅ `tests/test_integration`; MCP stdio contract tests in 🔄 #93 |
| 18 | `MockTTSProvider` | ✅ `tests/conftest.py` |

## Code Quality

| # | Item | Status |
|---|------|--------|
| 19 | Type safety | ✅ `py.typed` shipped · 📌 mypy-clean run and generic factory types |
| 20 | Logging configuration | 📌 No shared `setup_logging()`; loguru sinks are configured ad hoc in the CLI and UI |
| 21 | Configuration validation | ✅ `KokoroConfig.validate()` and `core/config_validation.py` |

## Performance

| # | Item | Status |
|---|------|--------|
| 22 | Lazy loading of heavy dependencies | ✅ `torch`/`kokoro` imported lazily; UI only loaded when `show_ui=True` |
| 23 | Streaming improvements | ✅ `synthesize_streaming()` available · 📌 chunked low-latency playback; non-blocking playback in 🔄 #86 |

## Packaging

| # | Item | Status |
|---|------|--------|
| 24 | Package metadata (`MANIFEST.in`, `py.typed`) | ✅ Added |
| 25 | Version management | ✅ Commitizen bumps `__version__` and `pyproject.toml` · 📌 `importlib.metadata` lookup |

## Security & Best Practices

| # | Item | Status |
|---|------|--------|
| 26 | Input validation | 🔄 #97 (MCP tool input guards and speech concurrency) · 📌 path traversal checks for the file reader |
| 27 | Resource cleanup | 🔄 #83 (`shutdown()`/`interrupt()`), #98 (singleton config) · 📌 synthesis timeouts, GPU memory release |

---

## Longer-Term Ideas

Taken from `ARCHITECTURE.md` ("Future Extensions") and `README.md` ("Coming Soon").
None are scheduled; file an issue under a new milestone when one is picked up.

- 📌 Additional providers: OpenAI TTS, ElevenLabs, Edge TTS, Google Cloud TTS
- 📌 SSML support, emotion control, multi-speaker synthesis, voice cloning
- 📌 Batch processing and model caching optimizations
