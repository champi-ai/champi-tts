# Audio Features Guide

Complete guide to audio processing, manipulation, and quality control features in Champi TTS.

## Table of Contents

- [Audio Synthesis](#audio-synthesis)
- [Audio Playback](#audio-playback)
- [Audio Saving](#audio-saving)
- [Audio Loading](#audio-loading)
- [Audio Quality](#audio-quality)
- [Audio Modification](#audio-modification)
- [Advanced Audio Features](#advanced-audio-features)

---

## Audio Synthesis

### Basic Synthesis

```python
from champi_tts import get_provider

async with get_provider() as provider:
    await provider.initialize()
    audio = await provider.synthesize("Hello, world!")
    print(f"Audio shape: {audio.shape}")
```

### Synthesize with Voice

```python
audio = await provider.synthesize("Hello", voice="af_bella")
```

### Synthesize with Speed

```python
audio = await provider.synthesize("Hello", speed=1.2)
```

### Synthesize with Multiple Parameters

```python
audio = await provider.synthesize(
    text="Hello",
    voice="af_bella",
    speed=1.1
)
```

---

## Audio Playback

### Using AudioPlayer

```python
from champi_tts.core.audio import AudioPlayer

player = AudioPlayer(sample_rate=22050)

# Play audio
await player.play(audio, blocking=False)

# Play blocking
await player.play(audio, blocking=True)

# Check if playing
print(f"Playing: {player.is_playing}")
```

### Control Playback

```python
# Pause
player.pause()

# Resume
player.resume()

# Stop
player.stop()

# Set volume (0.0 to 1.0)
player.set_volume(0.8)
```

### Streaming Playback

```python
async for chunk in provider.synthesize_streaming("Long text"):
    player.play(chunk, blocking=False)
```

---

## Audio Saving

### Basic Save

```python
from champi_tts.core.audio import save_audio

await save_audio(audio, "output.wav", sample_rate=22050)
```

### Save with Provider Sample Rate

```python
await save_audio(audio, "output.wav", sample_rate=provider.config.sample_rate)
```

### Save Multiple Files

```python
texts = ["Text 1", "Text 2", "Text 3"]
for i, text in enumerate(texts):
    audio = await provider.synthesize(text)
    await save_audio(audio, f"output_{i}.wav")
```

---

## Audio Loading

### Load Audio File

```python
from champi_tts.core.audio import load_audio

audio = await load_audio("output.wav")
print(f"Loaded audio shape: {audio.shape}")
```

### Load with Custom Sample Rate

```python
audio = await load_audio("output.wav", sample_rate=22050)
```

---

## Audio Quality

### Normalization

```python
from champi_tts.core.audio import normalize_audio

# Normalize audio to -1.0 to 1.0 range
normalized = normalize_audio(audio)

# Save normalized audio
await save_audio(normalized, "normalized.wav")
```

### Resampling

```python
from champi_tts.core.audio import resample_audio

# Resample to different sample rate
resampled = resample_audio(
    audio,
    from_rate=22050,
    to_rate=24000
)

await save_audio(resampled, "resampled.wav", sample_rate=24000)
```

### Volume Adjustment

```python
import numpy as np

# Increase volume
louder = audio * 1.2

# Decrease volume
quieter = audio * 0.8

# Save volume-adjusted audio
await save_audio(louder, "louder.wav")
```

---

## Audio Modification

### Apply Effects

```python
import numpy as np

def add_echo(audio, delay_samples=4410, decay=0.3):
    """Add echo effect to audio"""
    n = len(audio)
    echo = np.zeros_like(audio)

    # Create delayed version
    echo[delay_samples:] = audio[:-delay_samples] * decay

    # Mix original and echo
    return audio + echo

# Use effect
audio_with_echo = add_echo(audio)
await save_audio(audio_with_echo, "echo.wav")
```

### Noise Reduction

```python
from scipy.signal import butter, lfilter

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Bandpass filter for noise reduction"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# Apply filter
filtered = butter_bandpass_filter(audio, lowcut=100, highcut=5000, fs=22050)
```

### Fade In/Out

```python
def apply_fade(audio, fade_duration=0.5):
    """Apply fade in and fade out"""
    sample_rate = 22050
    fade_samples = int(fade_duration * sample_rate)

    # Create fade mask
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)

    # Apply fades
    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out

    return audio

# Use fade
audio_with_fade = apply_fade(audio)
```

---

## Advanced Audio Features

### Audio Export Formats

```python
from champi_tts.core.audio import save_audio

# WAV (default)
await save_audio(audio, "output.wav", sample_rate=22050)

# Specify different formats
# WAV: sample_rate parameter
# MP3: needs pydub library
# OGG: needs pydub library
```

### Audio Metadata

```python
from champi_tts.core.audio import save_audio
import wave

# Add metadata using pydub
from pydub import AudioSegment

audio_segment = AudioSegment(
    audio.tobytes(),
    frame_rate=22050,
    sample_width=2,
    channels=1
)

audio_segment.export("output_with_metadata.mp3", format="mp3", metadata={
    "title": "Test Audio",
    "artist": "Champi TTS",
    "comment": "Generated with Champi TTS"
})
```

### Audio Processing Pipeline

```python
async def process_audio_pipeline(text, output_path):
    """Complete audio processing pipeline"""

    # Synthesis
    async with get_provider() as provider:
        await provider.initialize()
        audio = await provider.synthesize(text)

    # Normalize
    from champi_tts.core.audio import normalize_audio
    audio = normalize_audio(audio)

    # Add fade
    audio = apply_fade(audio)

    # Save
    await save_audio(audio, output_path, sample_rate=provider.config.sample_rate)

    return output_path
```

### Audio Comparison

```python
async def compare_voices(text, voices):
    """Compare multiple voices"""

    results = {}

    for voice_name in voices:
        audio = await provider.synthesize(text, voice=voice_name)

        # Calculate metrics
        from champi_tts.core.audio import calculate_metrics
        metrics = calculate_metrics(audio)

        results[voice_name] = metrics

    return results
```

### Batch Audio Processing

```python
async def batch_process(texts, output_dir):
    """Process multiple texts"""

    import os
    os.makedirs(output_dir, exist_ok=True)

    for i, text in enumerate(texts):
        audio = await provider.synthesize(text)

        # Apply effects
        audio = apply_fade(audio)
        audio = normalize_audio(audio)

        # Save
        output_path = f"{output_dir}/output_{i:03d}.wav"
        await save_audio(audio, output_path, sample_rate=provider.config.sample_rate)

        print(f"Processed {i+1}/{len(texts)}")
```

### Audio Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_audio_waveform(audio, sample_rate=22050, title="Audio Waveform"):
    """Plot audio waveform"""

    plt.figure(figsize=(12, 4))
    plt.plot(audio)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

# Use
plot_audio_waveform(audio)
```

### Audio Statistics

```python
from champi_tts.core.audio import calculate_metrics

metrics = calculate_metrics(audio)

print(f"Duration: {metrics['duration']:.2f} seconds")
print(f"Sample rate: {metrics['sample_rate']} Hz")
print(f"Channels: {metrics['channels']}")
print(f"Max amplitude: {metrics['max_amplitude']:.2f}")
print(f"RMS: {metrics['rms']:.4f}")
print(f"Bit depth: {metrics['bit_depth']} bits")
```

### Audio Segmentation

```python
def split_audio(audio, duration=5.0):
    """Split audio into segments"""

    sample_rate = 22050
    segment_length = int(duration * sample_rate)

    segments = []
    for i in range(0, len(audio), segment_length):
        segment = audio[i:i + segment_length]
        if len(segment) > 0:
            segments.append(segment)

    return segments

# Use
segments = split_audio(audio)
```

### Audio Concatenation

```python
import numpy as np

async def concatenate_audios(audio_list, sample_rate):
    """Concatenate multiple audio arrays"""

    concatenated = np.concatenate(audio_list)
    return concatenated

# Use
audio1 = await provider.synthesize("Hello")
audio2 = await provider.synthesize("World")
combined = await concatenate_audios([audio1, audio2], 22050)
```

---

## Quality Control

### Peak Level Check

```python
def check_peak_level(audio, threshold=0.95):
    """Check if audio exceeds peak level"""

    peak = np.max(np.abs(audio))
    return peak > threshold

# Use
if check_peak_level(audio):
    print("Warning: Audio exceeds -1.0 level")
    # Normalize
    audio = normalize_audio(audio)
```

### Silence Detection

```python
def detect_silence(audio, threshold=0.01, min_duration=0.1):
    """Detect silence segments"""

    sample_rate = 22050
    min_samples = int(min_duration * sample_rate)

    silence_regions = []
    is_silence = np.abs(audio) < threshold

    start = None
    for i, silent in enumerate(is_silence):
        if silent and start is None:
            start = i
        elif not silent and start is not None:
            duration = i - start
            if duration >= min_samples:
                silence_regions.append((start, start + duration))
            start = None

    return silence_regions
```

### Audio Envelope

```python
def calculate_envelope(audio, window_size=1000):
    """Calculate audio envelope"""

    from scipy.signal import hilbert

    analytic_signal = hilbert(audio)
    amplitude_envelope = np.abs(analytic_signal)

    # Smooth
    window = np.ones(window_size) / window_size
    envelope = np.convolve(amplitude_envelope, window, mode='same')

    return envelope
```

---

## Best Practices

1. **Always specify sample rate** when saving audio
2. **Normalize audio** for consistent volume
3. **Use streaming** for long audio to save memory
4. **Handle errors** in audio processing
5. **Validate audio** before saving
6. **Use appropriate sample rate** for your use case

---

**Need more help?** Check the [API Reference](api.md) or [Examples](../examples/) for more information.