# Champi TTS Release Notes

## v0.2.0 - Integration & Documentation Release (2026-05-08)

### 🎉 Highlights

This release adds comprehensive integration tests, complete API documentation, and a migration guide for users moving from kokoro_svc.

### ✨ New Features

- **Integration Test Suite**: Full workflow tests covering pause/resume/stop, queue management, and error handling
- **API Documentation**: Complete reference with examples for all public methods
- **Migration Guide**: Step-by-step guide for migrating from kokoro_svc
- **Issue Templates**: Standardized templates for bugs, features, docs, and performance issues

### 📦 Improvements

- **Test Coverage**: Increased from ~50% to ~70% with integration tests
- **Documentation**: Comprehensive API reference and migration documentation
- **CI/CD**: Enhanced with issue templates and better issue tracking

### 🔧 Technical Details

- Added 4 integration test files covering full workflow scenarios
- Created complete API reference documentation
- Written migration guide with code examples
- Implemented lazy loading optimizations
- Added configuration validation utilities

### 📚 Documentation

- `docs/api.md` - Complete API reference
- `docs/migration.md` - Migration guide
- `docs/PROGRESS_REVIEW.md` - Progress tracking

### 🧪 Testing

- All integration tests passing
- Basic unit tests complete
- Coverage: ~70%

### 📋 Known Issues

- Performance benchmarks pending (Phase 4)
- Audio processing enhancements in progress
- Voice file distribution strategy planned

### 🔜 Next Steps (v0.3.0)

- Additional providers (OpenAI, ElevenLabs)
- Audio processing effects
- Voice cloning support
- SSML support

### 📄 License

MIT License - See LICENSE file

---

## v0.1.0 - Initial Release (2025-10-03)

### ✨ Features

- Multi-provider TTS architecture
- Kokoro provider implementation
- Text reading service with pause/resume/stop
- Visual UI indicator
- CLI interface
- Event-driven architecture

### 📦 Installation

```bash
pip install champi-tts
```

### 📄 License

MIT License
