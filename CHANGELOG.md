# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


### Added
- Initial release
- Multi-provider architecture foundation
- Kokoro provider
- Text reading service
- Visual UI indicator
- CLI interface
- Documentation

## v1.3.2 (2026-06-17)

### Fix

- **docs**: fix badge org URLs and replace corrupted emoji in README (#70)

## v1.3.1 (2026-06-17)

### Fix

- **deps**: embed champi-signals wheel URL and re-export EventProcessor in events.py (#69)

## v1.3.0 (2026-06-14)

### Feat

- **voices**: add voice cache manager and CLI voice commands (#66)

### Perf

- add performance benchmark suite (#57)

## v1.2.0 (2026-06-14)

### Feat

- **config**: add configuration validation to TTS config classes (#54)

### Fix

- **ci**: add test extra to benchmark workflow to install pytest-benchmark (#68)

### Perf

- implement lazy loading for heavy optional dependencies (#55)

## v1.1.1 (2026-06-14)

### Fix

- resolve all ruff lint errors across codebase (#67)

## v1.1.0 (2026-06-14)

### Perf

- implement Phase 4 improvements

## v0.2.0 (2026-05-07)

### Feat

- add GitHub issue templates for bug reports, feature requests, docs, and performance
