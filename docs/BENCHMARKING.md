# Performance Benchmarks

## Overview

This document establishes performance baselines and guidelines for all Text-to-Speech operations in Champi TTS.

## Supported Benchmarks

### 1. Model Loading
- **Benchmark**: `test_model_load_performance`
- **Description**: Measures time to initialize the TTS model
- **Acceptance**: < 30 seconds for first load
- **Acceptance**: < 5 seconds for subsequent loads

### 2. Model Unloading
- **Benchmark**: `test_model_unload_performance`
- **Description**: Measures time to unload the model and release resources
- **Acceptance**: < 5 seconds

### 3. Synthesis Performance

#### Short Text (approx. 70 chars)
- **Benchmark**: `test_synthesis_short_text`
- **Description**: Synthesis for short text phrases
- **Acceptance**: < 2 seconds

#### Medium Text (approx. 2100 chars)
- **Benchmark**: `test_synthesis_medium_text`
- **Description**: Synthesis for paragraph-length text
- **Acceptance**: < 10 seconds

#### Long Text (approx. 2000+ words)
- **Benchmark**: `test_synthesis_long_text`
- **Description**: Synthesis for document-length text
- **Acceptance**: < 60 seconds

### 4. Streaming Performance

#### Short Text Streaming
- **Benchmark**: `test_streaming_short_text`
- **Description**: Streaming synthesis with chunking
- **Acceptance**: Reasonable chunk generation time

#### Long Text Streaming
- **Benchmark**: `test_streaming_long_text`
- **Description**: Streaming synthesis for larger documents
- **Acceptance**: Reasonable chunk generation time

### 5. Reader Performance

#### Text Reading
- **Benchmark**: `test_reader_performance`
- **Description**: Reading text through the reader service
- **Acceptance**: < 5 seconds for medium text

#### Queue Performance
- **Benchmark**: `test_reader_queue_performance`
- **Description**: Processing multiple text items through queue
- **Acceptance**: < 30 seconds for 10 items

### 6. Audio Player

#### Playback Performance
- **Benchmark**: `test_audio_player_performance`
- **Description**: Audio playback operations
- **Acceptance**: < 0.1 seconds per operation

### 7. Configuration Validation

#### Validation Performance
- **Benchmark**: `test_config_validation_performance`
- **Description**: Configuration validation overhead
- **Acceptance**: < 0.01 seconds per validation

### 8. Memory Benchmarks

#### Steady State Memory
- **Benchmark**: `test_memory_usage_steady_state`
- **Description**: Memory stability during repeated operations
- **Acceptance**: Stable memory usage with no leaks

#### Large Document Memory
- **Benchmark**: `test_memory_usage_large_document`
- **Description**: Memory usage with large documents
- **Acceptance**: < 1000 MB increase for very long text

### 9. Regression Testing

#### Performance Regression
- **Benchmark**: `test_performance_regression`
- **Description**: Ensure no performance degradation
- **Acceptance**: Warm start at least 2x faster than cold start

### 10. Stress Testing

#### Consecutive Synthesis
- **Benchmark**: `test_stress_synthesis`
- **Description**: Performance under load with 10+ operations
- **Acceptance**: < 2x threshold for average time

#### Loading/Unloading Cycle
- **Benchmark**: `test_stress_loading`
- **Description**: Performance with 5+ load/unload cycles
- **Acceptance**: Average load < 30s, unload < 5s

### 11. Text Complexity

#### Complexity Detection
- **Benchmark**: `test_complexity_detection`
- **Description**: Performance with simple vs complex text
- **Acceptance**: Complex text not significantly slower

### 12. Speed Multiplier

#### Different Speeds
- **Benchmark**: `test_speed_multiplier_performance`
- **Description**: Performance with speed multipliers (0.5x to 2.0x)
- **Acceptance**: Speed relationship maintained

## Benchmark Tools

### Required Dependencies

```toml
[project.optional-dependencies]
test = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
    "pytest-mock",
    "pytest-benchmark>=5.0.0",
    "coverage>=7.6.0",
    "psutil>=6.1.0",  # For CPU/memory monitoring
]
```

### Running Benchmarks

#### Run All Benchmarks
```bash
pytest tests/test_performance.py -v --benchmark-only
```

#### Run Specific Benchmark
```bash
pytest tests/test_performance.py::test_synthesis_short_text -v --benchmark-only
```

#### Run with Output
```bash
pytest tests/test_performance.py -v --benchmark-only --benchmark-json=benchmarks.json
```

#### Generate Report
```bash
pytest tests/test_performance.py::generate_benchmark_report
```

## Performance Baselines

### Expected Performance (Current v1.0.0)

| Operation | Threshold | Notes |
|-----------|-----------|-------|
| Model Load | < 30s | First load includes model download if needed |
| Model Unload | < 5s | Resource cleanup time |
| Synthesis (Short) | < 2s | ~70 characters |
| Synthesis (Medium) | < 10s | ~2100 characters |
| Synthesis (Long) | < 60s | ~2000+ words |
| Streaming | < 0.5s/chunk | Depends on chunk size |
| Reader | < 5s | Medium text |
| Audio Player | < 0.1s | Playback operations |
| Config Validation | < 0.01s | Per validation |
| Memory Increase | < 1000 MB | Large document processing |

### Hardware Requirements

- **CPU**: Any modern processor (2+ cores recommended)
- **RAM**: Minimum 4GB (recommended 8GB+ for long documents)
- **Disk**: Space for model files (varies by provider)
- **OS**: Linux, macOS, Windows (recommended Linux for best performance)

## Benchmark Results

### Current Baseline Results

Run benchmarks to see current performance:

```bash
python tests/test_performance.py
```

Results are saved to `.benchmarks.json` and include:

- Average, minimum, maximum times
- CPU time statistics
- Memory usage deltas
- Platform information

## Performance Optimization Guidelines

### Common Optimizations

1. **Model Caching**
   - Keep model loaded when possible
   - Unload when not in use to free resources

2. **Batch Processing**
   - Process text in chunks rather than all at once
   - Reduces peak memory usage

3. **Stream Processing**
   - Use streaming for long documents
   - Improves perceived performance

4. **Text Preprocessing**
   - Use appropriate text complexity
   - Avoid extreme text lengths

5. **Resource Management**
   - Clean up audio buffers promptly
   - Use context managers for resources

### Monitoring

Use the PerformanceMonitor class to track:

- CPU usage (user + system time)
- Memory usage (RSS, VMS)
- Per-operation timing
- Resource deltas

## Performance Regression Testing

### How to Detect Regressions

1. Run benchmarks on clean state
2. Save results to baseline JSON
3. Commit results to repo
4. Run benchmarks after changes
5. Compare with baseline

### Automated Regression Detection

```bash
# Run current benchmarks
pytest tests/test_performance.py -v --benchmark-only --benchmark-json=baseline.json

# Compare with saved baseline
python -c "
import json
with open('baseline.json') as f:
    baseline = json.load(f)
with open('current.json') as f:
    current = json.load(f)
# Compare results and alert if > 2x slower
"
```

## Troubleshooting

### Benchmarks Failing

1. Check hardware resources (CPU, RAM)
2. Ensure no other heavy processes running
3. Verify dependencies installed correctly
4. Clean up and run again

### Slow Performance

1. Check if model is downloading
2. Verify model is properly loaded
3. Monitor memory usage
4. Check disk I/O performance

### Memory Leaks

1. Run garbage collection after operations
2. Use context managers for resources
3. Monitor with `psutil.Process().memory_info()`
4. Use memory profilers (memory_profiler)

## Contributing

When adding features:

1. Add performance benchmarks for new operations
2. Update baselines in this document
3. Document any performance expectations
4. Include hardware information in benchmark results

## Future Improvements

- [ ] Parallel synthesis for batch processing
- [ ] GPU acceleration benchmarks
- [ ] Network benchmark for cloud providers
- [ ] Per-voice performance tracking
- [ ] Benchmark UI for visualization
- [ ] Continuous integration benchmarking
- [ ] Performance baseline tracking over time

## Related Documentation

- [README.md](../README.md) - Project overview
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [TESTING.md](../TESTING.md) - Testing documentation

## References

- pytest-benchmark: https://pytest-benchmark.readthedocs.io/
- psutil: https://psutil.readthedocs.io/
- Performance profiling tools