# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multi-provider TTS architecture with abstract base classes
- Kokoro provider implementation (local, neural TTS)
- **Text Reading Service**:
  - Read text files with paragraph-by-paragraph processing
  - Text queue management for long documents
  - Pause/resume/stop controls
  - Interruption support for immediate stopping
  - Event-driven architecture with state tracking
  - States: IDLE, READING, PAUSED, STOPPED
  - Events: reading_started, reading_paused, reading_resumed, reading_stopped, reading_completed
- **Visual UI Indicator**:
  - Real-time visual status indicator using GLFW/ImGui
  - Visual states: Idle (gray), Processing (yellow), Speaking (green pulse), Paused (blue), Error (red)
  - Pulsing animations for active states
  - Standalone testing mode
  - Minimal resource usage
- **CLI Interface**:
  - `champi-tts synthesize` - Synthesize text to speech
  - `champi-tts read` - Read text files with pause/resume/stop
  - `champi-tts list-voices` - List available voices
  - `champi-tts test-ui` - Test visual UI indicator
  - `champi-tts version` - Show version
  - Rich terminal output with progress indicators
  - Interactive mode support (coming soon)
- **Core Audio Features**:
  - High-quality audio synthesis
  - Multiple voice options per provider
  - Adjustable speech speed (0.5-2.0x)
  - Audio file export (WAV, MP3, etc.)
  - Real-time audio playback with interruption
  - Audio normalization utilities
  - Sample rate conversion
- **Factory Pattern**:
  - `get_provider()` - Get TTS provider
  - `get_reader()` - Get text reader service
  - `list_providers()` - List available providers
  - `get_default_provider()` - Get default provider
- Generic audio handling (playback, file I/O, normalization)
- Abstract base classes for extensibility
- Response formatting utilities
- Event-driven architecture using blinker and champi-signals
- MIT License file
- Complete development tooling setup:
  - Pre-commit hooks with security scanning (detect-secrets, bandit)
  - Conventional commits for semantic versioning
  - Code quality tools (black, ruff, mypy)
  - Comprehensive linting configuration in pyproject.toml
- GitHub Actions CI/CD workflows:
  - CI pipeline (lint, security, test, build) with matrix testing
  - Pre-commit CI workflow
  - Automated release workflow with PyPI publishing
- All dependencies from Kokoro TTS (neural synthesis)
- UV package manager configuration with PyTorch CUDA support
- Project URLs (Homepage, Repository, Issues)

### Documentation
- README.md with badges, quick start guide, and development instructions
- ARCHITECTURE.md with detailed architecture documentation
- CHANGELOG.md for version history tracking
- Inline code documentation and docstrings
- .secrets.baseline for secrets detection

### Changed
- Updated author information to Divagnz
- Updated all repository URLs to divagnz/champi-tts
- Updated Ruff configuration (line-length: 88)
- Updated Black configuration to match project standards

## [0.1.0] - 2025-10-03

### Added
- Initial release
- Multi-provider architecture foundation
- Kokoro provider
- Text reading service
- Visual UI indicator
- CLI interface
- Documentation
