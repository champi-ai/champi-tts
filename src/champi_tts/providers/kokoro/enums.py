"""
Enums for Kokoro TTS service.
Defines message types and status codes for consistent communication.
"""

from enum import Enum, unique


@unique
class VoiceLanguage(Enum):
    """Supported voice languages"""

    # Primary language codes for Kokoro TTS synthesis
    AMERICAN_ENGLISH = "a"  # 🇺🇸 American English
    BRITISH_ENGLISH = "b"  # 🇬🇧 British English
    SPANISH = "e"  # 🇪🇸 Spanish (es)
    FRENCH = "f"  # 🇫🇷 French (fr-fr)
    HINDI = "h"  # 🇮🇳 Hindi (hi)
    ITALIAN = "i"  # 🇮🇹 Italian (it)
    JAPANESE = "j"  # 🇯🇵 Japanese (requires: pip install misaki[ja])
    PORTUGUESE_BRAZIL = "p"  # 🇧🇷 Brazilian Portuguese (pt-br)
    MANDARIN_CHINESE = "z"  # 🇨🇳 Mandarin Chinese (requires: pip install misaki[zh])

    # Extended language codes (alternative formats)
    ENGLISH_US = "en-us"  # Alternative US English format
    ENGLISH_GB = "en-gb"  # Alternative UK English format

    @classmethod
    def from_voice_prefix(cls, voice_name: str) -> str:
        """Get language code from voice name prefix"""
        if not voice_name:
            return cls.AMERICAN_ENGLISH.value

        prefix = voice_name.split("_")[0] if "_" in voice_name else voice_name[:1]

        # Map voice prefixes to language codes
        prefix_map = {
            "af": cls.AMERICAN_ENGLISH.value,  # American Female
            "am": cls.AMERICAN_ENGLISH.value,  # American Male
            "bf": cls.BRITISH_ENGLISH.value,  # British Female
            "bm": cls.BRITISH_ENGLISH.value,  # British Male
            "ef": cls.SPANISH.value,  # Spanish Female
            "em": cls.SPANISH.value,  # Spanish Male
            "ff": cls.FRENCH.value,  # French Female
            "fm": cls.FRENCH.value,  # French Male
            "hf": cls.HINDI.value,  # Hindi Female
            "hm": cls.HINDI.value,  # Hindi Male
            "if": cls.ITALIAN.value,  # Italian Female
            "im": cls.ITALIAN.value,  # Italian Male
            "jf": cls.JAPANESE.value,  # Japanese Female
            "jm": cls.JAPANESE.value,  # Japanese Male
            "pf": cls.PORTUGUESE_BRAZIL.value,  # Portuguese Female
            "pm": cls.PORTUGUESE_BRAZIL.value,  # Portuguese Male
            "zf": cls.MANDARIN_CHINESE.value,  # Mandarin Female
            "zm": cls.MANDARIN_CHINESE.value,  # Mandarin Male
        }

        return prefix_map.get(prefix.lower(), cls.AMERICAN_ENGLISH.value)

    @classmethod
    def get_all_codes(cls) -> list[str]:
        """Get all supported language codes"""
        return [
            cls.AMERICAN_ENGLISH.value,
            cls.BRITISH_ENGLISH.value,
            cls.SPANISH.value,
            cls.FRENCH.value,
            cls.HINDI.value,
            cls.ITALIAN.value,
            cls.JAPANESE.value,
            cls.PORTUGUESE_BRAZIL.value,
            cls.MANDARIN_CHINESE.value,
        ]


@unique
class AudioFormat(Enum):
    """Supported audio output formats"""

    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"


@unique
class ProcessingMode(Enum):
    """Text processing modes"""

    FULL_PIPELINE = "full_pipeline"
    TEXT_ONLY = "text_only"
    PHONEMES_ONLY = "phonemes_only"
    STREAMING = "streaming"
    BATCH = "batch"


@unique
class TTSEventTypes(Enum):
    """Main event type categories for TTS service"""

    LIFECYCLE_EVENT = "lifecycle_event"
    MODEL_EVENT = "model_event"
    PROCESSING_EVENT = "processing_event"
    TELEMETRY_EVENT = "telemetry_event"


@unique
class LifecycleEvents(Enum):
    """Lifecycle Events for Kokoro TTS service"""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING_TEXT = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"
    SPEAKING = "speaking"
    LISTENING = "listening"
    STANDBY = "standby"
    LLM_WORKING = "llm_working"
    STOPPED = "stopped"


@unique
class ModelEvents(Enum):
    """Model-related events"""

    MODEL_LOADING = "model_loading"
    MODEL_LOADED = "model_loaded"
    MODEL_UNLOADED = "model_unloaded"
    MODEL_ERROR = "model_error"
    WARMUP_START = "warmup_start"
    WARMUP_COMPLETE = "warmup_complete"
    VOICE_LOADING = "voice_loading"
    VOICE_LOADED = "voice_loaded"
    INITIALIZING = "initializing"
    UNLOADING = "unloading"


@unique
class TextProcessingEvents(Enum):
    """Text and audio processing events"""

    TEXT_PROCESSING = "text_processing"
    TEXT_NORMALIZED = "text_normalized"
    PHONEME_PROCESSING = "phoneme_processing"
    PHONEME_GENERATED = "phoneme_generated"
    AUDIO_GENERATING = "audio_generating"


@unique
class TelemetryEvents(Enum):
    """Metrics and telemetry events"""

    METRICS_UPDATE = "metrics_update"
    PERFORMANCE_STATS = "performance_stats"
    USAGE_STATS = "usage_stats"


@unique
class KokoroStrings(Enum):
    """Functional string constants used in Kokoro TTS service"""

    # Cache filename patterns
    VOICE_CACHE_KEY = "voice_{}.pkl"
    KOKORO_INFERENCE_CACHE = "kokoro_inference_{}.pkl"

    # Device names
    PULSE_DEVICE = "pulse"

    # File extensions
    PT_EXTENSION = ".pt"


@unique
class LoggingStrings(Enum):
    """Logging message templates for Kokoro TTS service"""

    # Initialization and loading messages
    LOADING_FROM_MEMORY_CACHE = "🚀 Loading Kokoro from memory cache..."
    LOADING_FROM_DISK_CACHE = "📦 Loading Kokoro from disk cache..."
    LOADED_FROM_CACHE = "✅ Kokoro loaded from cache"
    LOADING_FROM_SCRATCH = "🔄 Loading Kokoro from scratch..."
    PROVIDER_INITIALIZED = "Kokoro provider initialized successfully"
    PROVIDER_UNLOADED = "Kokoro provider unloaded"
    CACHED_MODEL_TO_PATH = "💾 Cached Kokoro model to {}"

    # Voice selection and configuration
    USING_CONFIGURED_VOICE = "Using configured voice: {}"
    AUTO_SELECTED_VOICE = "Auto-selected voice: {}"
    VOICE_NOT_AVAILABLE = "Voice '{}' not available, using first available"

    # Generation and processing
    GENERATING_SPEECH = "🎵 Generating speech with Kokoro (voice: {})"
    TTS_COMPLETED_METRICS = "✓ Kokoro TTS completed - Generation: {:.1f}s, Playback: {:.1f}s, Audio: {:.1f}s"

    # Directory and file operations
    DIRECTORIES_SETUP = "Kokoro directories setup - Models: {}, Voices: {}"
    DIRECTORIES_VALIDATED = "Directories validated - Model: {}, Voice: {}"
    MODEL_FILE_NOT_FOUND = "Model file not found: {}"
    CONFIG_FILE_NOT_FOUND = "Config file not found: {}"
    VOICE_FILE_NOT_FOUND = "Voice file not found: {}"

    # Error messages
    CACHE_LOAD_FAILED = "Cache load failed: {}, loading fresh..."
    FAILED_TO_CACHE_MODEL = "Failed to cache model: {}"
    FAILED_TO_INITIALIZE = "Failed to initialize Kokoro provider: {}"
    SYNTHESIS_FAILED = "Synthesis failed: {}"
    STREAMING_SYNTHESIS_FAILED = "Streaming synthesis failed: {}"
    PHONEME_SYNTHESIS_FAILED = "Phoneme synthesis failed: {}"
    WARMUP_FAILED = "Warmup failed: {}"
    AUDIO_PLAYBACK_FAILED = "Could not play audio: {}"

    # Warning messages
    NO_VOICES_AVAILABLE = "No Kokoro voices available"
    NO_VOICES_FOR_WARMUP = "No voices available for warmup"

    # Error return messages for API responses
    PROVIDER_NOT_INITIALIZED = "Kokoro TTS provider not initialized"
    NO_VALID_VOICES = "No valid Kokoro voices found"
    EMPTY_AUDIO_GENERATED = "Empty audio generated"
