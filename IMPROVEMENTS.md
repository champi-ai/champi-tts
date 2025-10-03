# Champi-TTS Review & Improvement Plan

## 📊 Overall Assessment

The project is **well-structured** and follows best practices from champi_stt. However, there are several areas for improvement:

---

## 🐛 Critical Issues to Fix

### 1. **Typo in base_config.py** (Line 11)
- **Issue**: `BaseT TSConfig` has a space (should be `BaseTTSConfig`)
- **Impact**: This will cause import errors
- **Fix**: Remove space in class name

### 2. **Missing Test Suite**
- **Issue**: `tests/` directory only has `__init__.py`, no actual tests
- **Impact**: No quality assurance, CI will fail
- **Fix**: Add test files for core modules

### 3. **UI Integration Not Implemented**
- **Issue**: `show_ui` parameter in TextReaderService is not used
- **Impact**: UI indicator won't actually show when requested
- **Fix**: Integrate TTSIndicatorUI with reader service

---

## ⚠️ High Priority Improvements

### 4. **Missing KokoroProvider Adapter**
- **Issue**: KokoroProvider doesn't inherit from BaseTTSProvider
- **Impact**: Type inconsistency, missing abstract method implementations
- **Fix**: Create adapter that wraps KokoroProvider to implement BaseTTSProvider interface

### 5. **Voice File Distribution Strategy**
- **Issue**: Voice files (70+ .pt files) in src/ will bloat package
- **Impact**: Large package size (~200MB+)
- **Options**:
  - Move to `voices/` in project root, download on first use
  - Use separate package or git-lfs
  - Document manual download process

### 6. **Missing Environment Configuration**
- **Issue**: No `.env.example` or environment variable documentation
- **Impact**: Users won't know how to configure
- **Fix**: Add `.env.example` with all configurable options

### 7. **CLI Interactive Mode Placeholder**
- **Issue**: Interactive mode shows "not yet implemented" message
- **Impact**: Incomplete feature advertised in docs
- **Fix**: Either implement or remove from CLI until ready

---

## 🔧 Medium Priority Improvements

### 8. **Dependency Organization**
- **Issue**: All dependencies in main list (no optional groups)
- **Suggestion**: Split into optional groups:
  ```toml
  [project.optional-dependencies]
  kokoro = ["kokoro", "misaki[en,ja,ko,zh]", ...]
  ui = ["imgui-bundle", "pyglm"]
  cli = ["typer>=0.16.0", "rich>=14.0.0"]
  ```

### 9. **Event System Integration**
- **Issue**: UI state updates not connected to reader events
- **Fix**: Connect reader signals to UI state changes:
  ```python
  reader.on_reading_started.connect(lambda **kw: ui.update_state(TTSState.SPEAKING))
  reader.on_reading_paused.connect(lambda **kw: ui.update_state(TTSState.PAUSED))
  ```

### 10. **Error Handling in Reader**
- **Issue**: Exception caught but state not set to ERROR
- **Fix**: Add error state handling:
  ```python
  except Exception as e:
      self._set_state(ReaderState.ERROR)
      self.on_error.send(self, error=str(e))
      raise
  ```

### 11. **Async Context Manager Support**
- **Suggestion**: Add `async with` support for providers:
  ```python
  async with get_provider("kokoro") as provider:
      audio = await provider.synthesize("Hello")
  ```

### 12. **Progress Callbacks**
- **Suggestion**: Add progress reporting for long syntheses:
  ```python
  async def synthesize(self, text, progress_callback=None):
      # Emit progress updates during synthesis
  ```

---

## 📝 Documentation Improvements

### 13. **API Reference**
- Add docstring examples for all public methods
- Add type hints for all parameters
- Document exception types that can be raised

### 14. **Usage Examples**
- Add `examples/` directory with:
  - `basic_synthesis.py`
  - `file_reading.py`
  - `custom_provider.py`
  - `event_handling.py`

### 15. **Migration Guide**
- Add guide for migrating from `mcp_champi.kokoro_svc` to `champi_tts`

---

## 🧪 Testing Strategy

### 16. **Unit Tests** (Priority: High)
```
tests/
├── test_core/
│   ├── test_base_provider.py
│   ├── test_audio.py
│   └── test_factory.py
├── test_providers/
│   └── test_kokoro.py
├── test_reader/
│   └── test_service.py
└── test_ui/
    └── test_indicator.py
```

### 17. **Integration Tests**
- Test full reading workflow
- Test pause/resume/stop sequences
- Test UI state transitions

### 18. **Mock Provider for Testing**
- Create `MockTTSProvider` for testing without kokoro dependency

---

## 🎨 Code Quality Enhancements

### 19. **Type Safety**
- Add `py.typed` marker file for type checking
- Fix all mypy issues
- Add generic types for provider factory

### 20. **Logging Configuration**
- Add `setup_logging()` function with levels
- Use structured logging for better debugging
- Add log file rotation

### 21. **Configuration Validation**
- Implement `validate()` method in KokoroConfig
- Add pydantic for advanced validation (optional)
- Validate voice names against available voices

---

## 🚀 Performance Optimizations

### 22. **Lazy Loading**
- Lazy import heavy dependencies (torch, kokoro)
- Only load UI when `show_ui=True`
- Cache synthesized audio (optional)

### 23. **Streaming Improvements**
- Implement true streaming synthesis
- Add chunk-based playback for lower latency
- Buffer management for smooth playback

---

## 📦 Packaging & Distribution

### 24. **Package Metadata**
- Add `MANIFEST.in` for non-Python files
- Add `py.typed` for type information
- Configure wheel to exclude test files

### 25. **Version Management**
- Use `importlib.metadata` for version
- Add `__version__` to all submodules
- Consider using `setuptools_scm` for git-based versioning

---

## 🔐 Security & Best Practices

### 26. **Input Validation**
- Validate file paths against path traversal
- Sanitize text input before synthesis
- Add rate limiting for API-based providers

### 27. **Resource Cleanup**
- Ensure all providers properly cleanup on shutdown
- Add timeout for synthesis operations
- Handle GPU memory properly

---

## 📋 Priority Implementation Order

### Phase 1: Critical Fixes (This Week)
1. ✅ Fix `BaseTTSConfig` typo
2. ✅ Create KokoroProvider adapter for BaseTTSProvider
3. ✅ Add basic unit tests (core, factory, reader)
4. ✅ Integrate UI with reader service
5. ✅ Add `.env.example`

### Phase 2: Core Improvements (Next Week)
6. ✅ Implement error state handling
7. ✅ Add async context manager support
8. ✅ Reorganize dependencies into optional groups
9. ✅ Add voice file download strategy
10. ✅ Complete or remove interactive CLI mode

### Phase 3: Polish & Documentation (Following Week)
11. ✅ Add examples directory
12. ✅ Complete API documentation
13. ✅ Add integration tests
14. ✅ Optimize lazy loading
15. ✅ Add migration guide

---

## ✨ Recommended Quick Wins

**Start with these 5 fixes for immediate improvement:**

1. **Fix typo** in `base_config.py` (1 min)
2. **Add 3 basic tests** to validate core functionality (30 min)
3. **Create `.env.example`** with all config options (15 min)
4. **Connect UI to reader events** (30 min)
5. **Add simple example** in `examples/basic.py` (15 min)

**Total time**: ~90 minutes for major quality improvement!

---

## 🎯 Success Metrics

- ✅ All CI checks passing
- ✅ Test coverage > 80%
- ✅ No type errors (mypy clean)
- ✅ All linters passing (ruff, black)
- ✅ Documentation complete
- ✅ Example code works
- ✅ Package builds successfully
