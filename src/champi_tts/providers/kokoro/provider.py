"""
Kokoro TTS provider for mcp_champi integration.
Provides the interface for mcp_champi to use Kokoro TTS.
"""

import asyncio
import hashlib
import os
import pickle  # nosec B403 - Used for caching trusted TTS model objects only
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
from champi_signals import EventProcessor
from loguru import logger

from champi_tts.providers.kokoro.config import KokoroConfig
from champi_tts.providers.kokoro.enums import (
    LifecycleEvents,
    VoiceLanguage,
)
from champi_tts.providers.kokoro.events import TTSSignalManager
from champi_tts.providers.kokoro.exceptions import (
    KokoroAudioError,
    KokoroFileError,
    KokoroInitializationError,
    KokoroSynthesisError,
    KokoroVoiceError,
)
from champi_tts.providers.kokoro.inference import KokoroInference
from champi_tts.providers.kokoro.models import ModelDownloader, VoiceManager


class VoiceCache:
    """Cache for voice embeddings/data"""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self._voice_cache = {}

    def get_voice_key(self, voice_path: str) -> str:
        """Generate cache key for voice file."""
        voice_stat = Path(voice_path).stat()
        return f"{Path(voice_path).stem}_{voice_stat.st_mtime}"

    async def get_cached_voice(self, voice_path: str) -> Any:
        """Get cached voice data."""
        cache_key = self.get_voice_key(voice_path)

        # Memory cache first
        if cache_key in self._voice_cache:
            return self._voice_cache[cache_key]

        # Disk cache
        cache_file = self.cache_dir / f"voice_{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                # nosec B301 - Loading cached voice data from local trusted cache only
                voice_data = pickle.load(f)
            self._voice_cache[cache_key] = voice_data
            return voice_data

        return None

    async def cache_voice(self, voice_path: str, voice_data):
        """Cache voice data."""
        cache_key = self.get_voice_key(voice_path)

        # Memory cache
        self._voice_cache[cache_key] = voice_data

        # Disk cache
        cache_file = self.cache_dir / f"voice_{cache_key}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(voice_data, f)


class KokoroProvider:
    """TTS provider interface for mcp_champi with async singleton"""

    class Meta:
        event_type = "lifecycle"
        signal_manager = TTSSignalManager()

    _instance = None
    _instance_lock = asyncio.Lock()
    _model_cache = {}  # Class-level cache

    def __new__(
        cls,
        config: KokoroConfig | None = None,
        model_dir: str | None = None,
        voice_dir: str | None = None,
    ):
        """Singleton pattern implementation - creates instance only once"""
        if cls._instance is None:
            instance = super().__new__(cls)
            # Initialize instance attributes that should only be set once
            instance._singleton_setup_done = False
            cls._instance = instance
        return cls._instance

    def __init__(
        self,
        config: KokoroConfig | None = None,
        model_dir: str | None = None,
        voice_dir: str | None = None,
    ):
        """Initialize singleton instance - only runs setup once"""
        # Skip if already setup to prevent re-initialization
        if self._singleton_setup_done:
            return

        # Setup instance
        self.config = config or KokoroConfig()
        self.signals_manager = TTSSignalManager()
        self._initialized = False
        self._tts_status: LifecycleEvents = LifecycleEvents.UNINITIALIZED.value
        self.audio_operation_lock = asyncio.Lock()

        # Mark singleton as setup
        self._singleton_setup_done = True

        # Note: Manual initialization required via await provider.initialize()

    @classmethod
    async def get_instance(
        cls,
        config: KokoroConfig | None = None,
        model_dir: str | None = None,
        voice_dir: str | None = None,
    ) -> "KokoroProvider":
        """Get singleton instance with async-safe access"""
        async with cls._instance_lock:
            return cls(config, model_dir, voice_dir)

    async def _get_cache_key(self) -> str:
        """Generate cache key from model/voice directories and timestamps."""
        model_stat = (
            Path(self.config.model_dir).stat() if self.config.model_dir else None
        )
        voice_stat = (
            Path(self.config.voice_dir).stat() if self.config.voice_dir else None
        )

        key_data = (
            f"{self.config.model_dir}:{model_stat.st_mtime if model_stat else 0}:"
        )
        key_data += (
            f"{self.config.voice_dir}:{voice_stat.st_mtime if voice_stat else 0}"
        )

        return hashlib.md5(key_data.encode(), usedforsecurity=False).hexdigest()[:12]

    @EventProcessor.emits_event(data=["_tts_status"])
    async def initialize(self, download_model: bool = True) -> None:
        """Initialize the provider"""
        try:
            self._tts_status = LifecycleEvents.INITIALIZING.value
            # Emit initialization start signal
            self.signals_manager.lifecycle.send(
                self,
                event_type="lifecycle",
                sub_event="initialization_start",
                data={"download_model": download_model},
            )

            # Validate directories first
            self.validate_directories()

            # Setup directories
            if self.config.model_dir is None or self.config.voice_dir is None:
                (
                    self.config.model_dir,
                    self.config.voice_dir,
                ) = self.setup_kokoro_directories(
                    self.config.model_dir, self.config.voice_dir
                )

            # Check cache
            cache_key = await self._get_cache_key()
            cache_file = (
                Path(self.config.cache_dir) / f"kokoro_inference_{cache_key}.pkl"
            )

            if cache_file.exists() and cache_key in self._model_cache:
                logger.info("🚀 Loading Kokoro from memory cache...")
                self.signals_manager.model.send(
                    self,
                    event_type="model",
                    sub_event="cache_hit",
                    data={"cache_type": "memory", "cache_key": cache_key},
                )
                self.inference_engine = self._model_cache[cache_key]
                self._initialized = True
                return

            if cache_file.exists():
                try:
                    logger.info("📦 Loading Kokoro from disk cache...")
                    self.signals_manager.model.send(
                        self,
                        event_type="model",
                        sub_event="cache_hit",
                        data={"cache_type": "disk", "cache_key": cache_key},
                    )
                    with open(cache_file, "rb") as f:
                        # nosec B301 - Loading cached model from local trusted cache only
                        self.inference_engine = pickle.load(f)

                    # Store in memory cache
                    self._model_cache[cache_key] = self.inference_engine
                    self._initialized = True
                    logger.info("✅ Kokoro loaded from cache")
                    return
                except Exception as e:
                    logger.warning(f"Cache load failed: {e}, loading fresh...")

            # Fresh load
            logger.info("🔄 Loading Kokoro from scratch...")
            self.signals_manager.model.send(
                self,
                event_type="model",
                sub_event="loading_start",
                data={"load_type": "fresh", "download_model": download_model},
            )

            # Setup model files
            self._model_path, self._config_path = ModelDownloader.download_model(
                self.config.model_dir, force=not download_model
            )

            # Initialize inference engine
            self.inference_engine = KokoroInference(self.config)
            await self.inference_engine.load_model(self._model_path, self._config_path)

            # Verify voices directory
            VoiceManager.setup_voice_directory(self.config.voice_dir)

            # Cache the loaded model
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(self.inference_engine, f)
                self._model_cache[cache_key] = self.inference_engine
                logger.info(f"💾 Cached Kokoro model to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache model: {e}")

            self._initialized = True
            logger.debug("Kokoro provider initialized successfully")

            # Emit initialization complete signal
            self.signals_manager.lifecycle.send(
                self,
                event_type="lifecycle",
                sub_event="initialization_complete",
                data={"initialized": True},
            )

            # Optional warmup
            if self.config.warmup_on_init:
                await self._warmup()

        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            self.signals_manager.lifecycle.send(
                self,
                event_type="lifecycle",
                sub_event="initialization_error",
                data={"error": str(e)},
            )
            raise KokoroFileError(f"Model files not found: {e}") from e
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro provider: {e}")
            # Emit initialization error signal
            self.signals_manager.lifecycle.send(
                self,
                event_type="lifecycle",
                sub_event="initialization_error",
                data={"error": str(e)},
            )
            raise KokoroInitializationError(
                f"Provider initialization failed: {e}"
            ) from e

    async def _warmup(self) -> None:
        """Warm up the model with a default voice"""
        try:
            voices = self.list_voices()
            if voices:
                voice_path = await self.get_voice_path(voices[0])
                await self.inference_engine.warmup(voice_path)
            else:
                logger.warning("No voices available for warmup")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    async def is_initialized(self) -> bool:
        """Check if provider is initialized"""
        return (
            self._initialized
            and self.inference_engine
            and self.inference_engine.is_loaded
        )

    def list_voices(self) -> list[str]:
        """List available voices"""
        if not self.config.voice_dir:
            return []
        return VoiceManager.list_voices(self.config.voice_dir)

    def get_voice_language(self, voice_name: str) -> str:
        """Get appropriate language code for a voice based on its prefix"""
        return VoiceLanguage.from_voice_prefix(voice_name)

    async def get_voice_path(self, voice_name: str) -> str:
        """Get full path to voice file"""
        if not self.config.voice_dir:
            raise ValueError("Voice directory not set")

        # Handle voice name with or without .pt extension
        if not voice_name.endswith(".pt"):
            voice_name += ".pt"

        voice_path = os.path.join(self.config.voice_dir, voice_name)

        if not os.path.exists(voice_path):
            raise FileNotFoundError(f"Voice file not found: {voice_path}")

        return voice_path

    async def validate_voice(self, voice_name: str) -> bool:
        """Validate that a voice exists and is usable"""
        try:
            voice_path = await self.get_voice_path(voice_name)
            return VoiceManager.validate_voice_file(voice_path)
        except Exception as _err:
            return False

    @EventProcessor.emits_event(data=["_tts_status"])
    async def synthesize(
        self, text: str, voice: str = "default", **kwargs
    ) -> np.ndarray:
        """Main synthesis interface

        Args:
            text: Text to synthesize
            voice: Voice name (without .pt extension)
            **kwargs: Additional parameters (lang_code, speed, normalize)

        Returns:
            Audio array as numpy array
        """
        try:
            # Emit synthesis start signal
            self.signals_manager.processing.send(
                self,
                event_type="processing",
                sub_event="synthesis_start",
                data={"text_length": len(text), "voice": voice},
            )

            # Get parameters
            lang_code = kwargs.get("lang_code", self.config.default_language)
            speed = kwargs.get("speed", self.config.default_speed)
            normalize = kwargs.get("normalize", self.config.normalize_text)

            # Get voice path
            voice_path = await self.get_voice_path(voice)

            # Generate audio
            result = await self.inference_engine.generate_from_text(
                text=text,
                voice_path=voice_path,
                lang_code=lang_code,
                speed=speed,
                normalize=normalize,
            )

            # Emit synthesis success signal
            self.signals_manager.processing.send(
                self,
                event_type="processing",
                sub_event="synthesis_complete",
                data={
                    "text_length": len(text),
                    "voice": voice,
                    "audio_size": len(result.audio),
                },
            )

            return result.audio

        except FileNotFoundError as e:
            logger.error(f"Voice file not found: {e}")
            self.signals_manager.processing.send(
                self,
                event_type="processing",
                sub_event="synthesis_error",
                data={"error": str(e), "text_length": len(text), "voice": voice},
            )
            raise KokoroVoiceError(f"Voice file not found: {e}") from e
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Emit synthesis error signal
            self.signals_manager.processing.send(
                self,
                event_type="processing",
                sub_event="synthesis_error",
                data={"error": str(e), "text_length": len(text), "voice": voice},
            )
            raise KokoroSynthesisError(f"Audio synthesis failed: {e}") from e

    async def synthesize_streaming(
        self, text: str, voice: str = "default", **kwargs
    ) -> AsyncGenerator[np.ndarray, None]:
        """Streaming synthesis interface

        Args:
            text: Text to synthesize or phonemes (depending on use_phoneme flag)
            voice: Voice name
            **kwargs: Additional parameters (lang_code, speed, chunk_size, use_phoneme)

        Yields:
            Audio chunks as numpy arrays
        """
        if not await self.is_initialized():
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        try:
            # Get parameters
            lang_code = kwargs.get("lang_code", self.config.default_language)
            speed = kwargs.get("speed", self.config.default_speed)
            chunk_size = kwargs.get("chunk_size", self.config.streaming_chunk_size)
            use_phoneme = kwargs.get("use_phoneme", False)

            # Get voice path
            voice_path = await self.get_voice_path(voice)

            # Generate streaming audio
            async for chunk in self.inference_engine.generate_stream(
                text=text,
                voice_path=voice_path,
                lang_code=lang_code,
                speed=speed,
                chunk_size=chunk_size,
                use_phoneme=use_phoneme,
            ):
                yield chunk.audio

        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")
            raise

    async def synthesize_from_phonemes(
        self, phonemes: str, voice: str = "default", **kwargs
    ) -> np.ndarray:
        """Synthesize from phonemes directly

        Args:
            phonemes: Phoneme string in Kokoro format
            voice: Voice name
            **kwargs: Additional parameters

        Returns:
            Audio array as numpy array
        """
        if not await self.is_initialized():
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        try:
            # Get parameters
            lang_code = kwargs.get("lang_code", self.config.default_language)
            speed = kwargs.get("speed", self.config.default_speed)

            # Get voice path
            voice_path = await self.get_voice_path(voice)

            # Generate audio from phonemes
            result = await self.inference_engine.generate_from_phonemes(
                phonemes=phonemes,
                voice_path=voice_path,
                lang_code=lang_code,
                speed=speed,
            )

            return result.audio

        except Exception as e:
            logger.error(f"Phoneme synthesis failed: {e}")
            raise

    async def get_info(self) -> dict[str, Any]:
        """Get provider information"""
        voices = self.list_voices()

        return {
            "name": "Kokoro",
            "version": "1.0",
            "initialized": await self.is_initialized(),
            "device": self.inference_engine.device
            if self.inference_engine
            else "unknown",
            "model_loaded": self.inference_engine.is_loaded
            if self.inference_engine
            else False,
            "model_dir": self.config.model_dir,
            "voice_dir": self.config.voice_dir,
            "available_voices": voices,
            "voice_count": len(voices),
            "supported_languages": VoiceLanguage.get_all_codes(),
            "config": {
                "default_language": self.config.default_language,
                "default_speed": self.config.default_speed,
                "normalize_text": self.config.normalize_text,
                "use_gpu": self.config.use_gpu,
                "force_cpu": self.config.force_cpu,
            },
        }

    @EventProcessor.emits_event(data=["_initialized"])
    async def unload(self) -> None:
        """Unload the provider and free resources"""
        # Emit unload start signal
        self.signals_manager.lifecycle.send(
            self,
            event_type="lifecycle",
            sub_event="unload_start",
            data={"initialized": self._initialized},
        )

        if hasattr(self, "inference_engine") and self.inference_engine:
            self.inference_engine.unload()
            self.inference_engine = None

        self._initialized = False
        logger.debug("Kokoro provider unloaded")

        # Emit unload complete signal
        self.signals_manager.lifecycle.send(
            self,
            event_type="lifecycle",
            sub_event="unload_complete",
            data={"initialized": self._initialized},
        )

    @EventProcessor.emits_event(data=["_initialized"])
    async def reload(self) -> None:
        """Reload the provider"""
        # Emit reload start signal
        self.signals_manager.lifecycle.send(
            self,
            event_type="lifecycle",
            sub_event="reload_start",
            data={"initialized": self._initialized},
        )

        await self.unload()
        await self.initialize(download_model=False)  # Don't re-download

        # Emit reload complete signal
        self.signals_manager.lifecycle.send(
            self,
            event_type="lifecycle",
            sub_event="reload_complete",
            data={"initialized": self._initialized},
        )

    # Compatibility methods for mcp_champi integration

    async def get_voice_list(self) -> list[str]:
        """Get list of available voices (compatibility method)"""
        return self.list_voices()

    async def set_voice(self, voice_name: str) -> bool:
        """Set/validate voice (compatibility method)"""
        return await self.validate_voice(voice_name)

    async def speak(self, text: str, voice: str | None = None, **kwargs) -> np.ndarray:
        """Speak text (compatibility method)"""
        if voice is None:
            voices = self.list_voices()
            voice = voices[0] if voices else "default"

        return await self.synthesize(text, voice, **kwargs)

    async def _select_voice(self, voice: str | None) -> str | None:
        """Select and validate voice for synthesis.

        Args:
            voice: Voice name or None for auto-selection

        Returns:
            Valid voice name or None if no voices available
        """
        if not voice:
            # First check for configured default voice
            if self.config.default_voice:
                voice = self.config.default_voice
                logger.debug(f"Using configured voice: {voice}")
            else:
                # Fall back to auto-selection
                available_voices = self.list_voices()
                if available_voices:
                    # Prefer female voices by default (af_ prefix)
                    female_voices = [v for v in available_voices if v.startswith("af_")]
                    voice = female_voices[0] if female_voices else available_voices[0]
                    logger.debug(f"Auto-selected voice: {voice}")
                else:
                    logger.warning("No Kokoro voices available")
                    return None

        # Validate voice exists
        if not await self.validate_voice(voice):
            logger.warning(f"Voice '{voice}' not available, using first available")
            available_voices = self.list_voices()
            voice = available_voices[0] if available_voices else None

        return voice

    async def _generate_audio(
        self,
        text: str,
        voice: str,
        speed: float,
        lang_code: str,
        use_phoneme: bool,
    ) -> tuple[np.ndarray, float]:
        """Generate audio from text or phonemes.

        Args:
            text: Text or phonemes to synthesize
            voice: Voice name
            speed: Speech speed multiplier
            lang_code: Language code
            use_phoneme: If True, treat text as phonemes

        Returns:
            Tuple of (audio_data, generation_time)
        """
        generation_start = time.perf_counter()

        # Choose synthesis method based on use_phoneme flag
        if use_phoneme:
            logger.debug(f"🔤 Using phoneme synthesis for: '{text}'")
            audio_data = await self.synthesize_from_phonemes(
                phonemes=text,
                voice=voice,
                speed=speed,
                lang_code=lang_code,
            )
        else:
            audio_data = await self.synthesize(
                text=text,
                voice=voice,
                speed=speed,
                lang_code=lang_code,
                normalize=True,
            )

        generation_time = time.perf_counter() - generation_start
        return audio_data, generation_time

    async def _play_audio(self, audio_data: np.ndarray) -> float:
        """Play audio using sounddevice.

        Args:
            audio_data: Audio array to play

        Returns:
            Playback time in seconds

        Raises:
            KokoroAudioError: If audio playback fails
        """
        playback_start = time.perf_counter()

        # Ensure audio is in correct format for playback
        if audio_data.dtype != np.int16:
            # Kokoro typically returns float32, convert to int16 for playback
            if audio_data.dtype == np.float32:
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)

        # Play audio using sounddevice with PulseAudio mixing support
        try:
            # Try PulseAudio device first (allows mixing with other apps)
            sd.play(audio_data, samplerate=24000, device="pulse")
            sd.wait()  # Wait for playback to complete
        except Exception as pulse_error:
            logger.error(f"PulseAudio device failed: {pulse_error}, trying default")
            try:
                # Fallback to default device
                sd.play(audio_data, samplerate=24000)
                sd.wait()
            except Exception as default_error:
                logger.error(f"Audio playback failed: {default_error}")
                raise KokoroAudioError(
                    f"Could not play audio: {default_error}"
                ) from default_error

        return time.perf_counter() - playback_start

    @EventProcessor.emits_event(data=["_tts_status"])
    async def text_to_speech(
        self,
        message: str,
        voice: str | None = None,
        speed: float = 1.0,
        lang_code: str = "a",
        tts_instructions: str | None = None,
        play_audio: bool = True,
        use_phoneme: bool = False,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Text to speech using Kokoro TTS with optional audio playback.

        Args:
            message: Text to synthesize
            voice: Voice name to use (optional, will auto-select if None)
            speed: Speech speed multiplier (default 1.0)
            lang_code: Language code ('a' for en-us, 'b' for en-gb)
            tts_instructions: Additional instructions for TTS synthesis (optional)
            play_audio: Whether to play the audio immediately (default True)
            use_phoneme: If True, treat message as phonemes instead of text (default False)

        Returns:
            Tuple of (success: bool, metrics: dict)
        """
        if not await self.is_initialized():
            return False, {"error": "Kokoro TTS provider not initialized"}

        start_time = time.perf_counter()

        try:
            # Emit TTS request start signal
            self.signals_manager.processing.send(
                self,
                event_type="processing",
                sub_event="tts_request_start",
                data={
                    "message_length": len(message),
                    "voice": voice,
                    "speed": speed,
                    "lang_code": lang_code,
                    "use_phoneme": use_phoneme,
                    "play_audio": play_audio,
                },
            )

            # Select voice
            voice = await self._select_voice(voice)
            if not voice:
                return False, {"error": "No Kokoro voices available"}

            logger.debug(f"🎵 Generating speech with Kokoro (voice: {voice})")

            # Emit voice selection complete signal
            self.signals_manager.processing.send(
                self,
                event_type="processing",
                sub_event="voice_selected",
                data={
                    "voice": voice,
                    "selection_type": "auto"
                    if voice != self.config.default_voice
                    else "manual",
                },
            )

            # Add instructions to the message if provided
            synthesis_text = message
            if tts_instructions:
                logger.debug(f"Using TTS instructions: {tts_instructions}")

            # Generate audio
            audio_data, generation_time = await self._generate_audio(
                synthesis_text, voice, speed, lang_code, use_phoneme
            )

            if audio_data is None or len(audio_data) == 0:
                logger.error("Kokoro returned empty audio")
                return False, {"error": "Empty audio generated"}

            # Calculate audio metrics
            audio_duration = len(audio_data) / 24000
            ttfa = generation_time  # For non-streaming, this is the same as generation time

            # Play audio if requested
            playback_time = 0
            if play_audio:
                self._tts_status = LifecycleEvents.SPEAKING.value
                # Emit playback start signal
                self.signals_manager.processing.send(
                    self,
                    event_type="processing",
                    sub_event="playback_start",
                    data={"audio_duration": audio_duration},
                )

                playback_time = await self._play_audio(audio_data)

                # Emit playback complete signal
                self.signals_manager.processing.send(
                    self,
                    event_type="processing",
                    sub_event="playback_complete",
                    data={"playback_time": playback_time},
                )
                self._tts_status = LifecycleEvents.READY.value

            total_time = time.perf_counter() - start_time

            logger.info(
                f"✓ Kokoro TTS completed - Generation: {generation_time:.1f}s, "
                f"Playback: {playback_time:.1f}s, Audio: {audio_duration:.1f}s"
            )

            # Build metrics
            metrics = {
                "ttfa": ttfa,
                "generation": generation_time,
                "playback": playback_time,
                "total": total_time,
                "audio_duration": audio_duration,
                "provider": "kokoro",
                "voice": voice,
                "model": "kokoro-v1.0",
                "text_length": len(message),
                "real_time_factor": generation_time / audio_duration
                if audio_duration > 0
                else 0,
            }

            # Emit telemetry signal for performance metrics
            self.signals_manager.telemetry.send(
                self,
                event_type="telemetry",
                sub_event="performance_metrics",
                data=metrics,
            )

            # Emit TTS request complete signal
            self.signals_manager.processing.send(
                self,
                event_type="processing",
                sub_event="tts_request_complete",
                data=metrics,
            )

            return True, metrics

        except Exception as e:
            logger.error(f"Kokoro TTS failed: {e}")
            metrics = {"error": str(e)}
            # Emit TTS request error signal
            self.signals_manager.processing.send(
                self,
                event_type="processing",
                sub_event="tts_request_error",
                data={"error": str(e), "message_length": len(message)},
            )
            return False, metrics

    def save_config_to_file(self, config_path: str) -> None:
        """Save config to JSON file"""
        import json
        import os

        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)
            logger.debug(f"Config saved to: {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def validate_directories(self) -> None:
        """Validate and create directories if needed"""
        import os

        # Set default directories if not specified
        if self.config.model_dir is None:
            self.config.model_dir = self.get_default_model_dir()

        if self.config.voice_dir is None:
            self.config.voice_dir = self.get_default_voice_dir()

        if self.config.cache_dir is None:
            self.config.cache_dir = self.get_default_cache_dir()

        # Create directories
        try:
            os.makedirs(self.config.model_dir, exist_ok=True)
            os.makedirs(self.config.voice_dir, exist_ok=True)
            os.makedirs(self.config.cache_dir, exist_ok=True)
            logger.debug(
                f"Directories validated - Model: {self.config.model_dir}, Voice: {self.config.voice_dir}, Cache: {self.config.cache_dir}"
            )
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")

    def get_default_model_dir(self) -> str:
        """Get default model directory path"""
        import os
        from pathlib import Path

        home_dir = Path.home()

        # Try to use appropriate directory for the platform
        if os.name == "nt":  # Windows
            app_data = os.environ.get("APPDATA", home_dir)
            model_dir = os.path.join(app_data, "mcp_champi", "kokoro", "models")
        else:  # Unix-like
            model_dir = os.path.join(home_dir, ".mcp_champi", "kokoro", "models")

        return model_dir

    def get_default_voice_dir(self) -> str:
        """Get default voice directory path"""
        import os
        from pathlib import Path

        home_dir = Path.home()

        if os.name == "nt":  # Windows
            app_data = os.environ.get("APPDATA", home_dir)
            voice_dir = os.path.join(app_data, "mcp_champi", "kokoro", "voices")
        else:  # Unix-like
            voice_dir = os.path.join(home_dir, ".mcp_champi", "kokoro", "voices")

        return voice_dir

    def get_default_cache_dir(self) -> str:
        """Get default cache directory path"""
        import os
        from pathlib import Path

        home_dir = Path.home()

        if os.name == "nt":  # Windows
            app_data = os.environ.get("APPDATA", home_dir)
            cache_dir = os.path.join(app_data, "mcp_champi", "kokoro", "cache")
        else:  # Unix-like
            cache_dir = os.path.join(home_dir, ".mcp_champi", "kokoro", "cache")

        return cache_dir

    def setup_kokoro_directories(
        self, model_dir: str | None = None, voice_dir: str | None = None
    ) -> tuple[str, str]:
        """Setup both model and voice directories

        Returns:
            Tuple of (model_dir, voice_dir)
        """
        import os

        if model_dir is None:
            model_dir = self.get_default_model_dir()

        if voice_dir is None:
            voice_dir = self.get_default_voice_dir()

        # Setup directories
        os.makedirs(model_dir, exist_ok=True)
        VoiceManager.setup_voice_directory(voice_dir)

        logger.debug(
            f"Kokoro directories setup - Models: {model_dir}, Voices: {voice_dir}"
        )
        return model_dir, voice_dir

    def get_signals(self) -> dict:
        """Get available signals that can be connected to by external software

        Returns:
            Dict mapping signal names to Signal objects. Each signal emits events with:
            - event_type: The main event category (lifecycle, model, processing, telemetry)
            - sub_event: The specific event within that category
            - data: Event-specific data payload
        """
        return {
            "lifecycle": self.signals_manager.lifecycle,
            "model": self.signals_manager.model,
            "processing": self.signals_manager.processing,
            "telemetry": self.signals_manager.telemetry,
        }

    async def read_pdf(
        self, pdf_path: str, page_range: tuple[int, int] | None = None
    ) -> str:
        """Extract text content from PDF file

        Args:
            pdf_path: Path to PDF file
            page_range: Optional tuple (start_page, end_page) for specific pages (1-indexed)
                       If None, extracts all pages

        Returns:
            Extracted text content from PDF

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            RuntimeError: If PDF extraction fails
        """
        import os

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            # Try PyMuPDF (fitz) first - best for text extraction
            try:
                import fitz  # PyMuPDF

                doc = fitz.open(pdf_path)
                extracted_text = []

                start_page = 0
                end_page = doc.page_count

                if page_range:
                    start_page = max(0, page_range[0] - 1)  # Convert to 0-indexed
                    end_page = min(doc.page_count, page_range[1])

                logger.debug(
                    f"Extracting text from PDF: {pdf_path} (pages {start_page+1}-{end_page})"
                )

                for page_num in range(start_page, end_page):
                    page = doc[page_num]
                    text = page.get_text()
                    if text.strip():  # Only add non-empty pages
                        extracted_text.append(
                            f"--- Page {page_num + 1} ---\n{text.strip()}"
                        )

                doc.close()

                if not extracted_text:
                    logger.warning("No text content found in PDF")
                    return ""

                full_text = "\n\n".join(extracted_text)
                logger.info(
                    f"Successfully extracted {len(full_text)} characters from {len(extracted_text)} pages"
                )
                return full_text

            except ImportError:
                logger.debug("PyMuPDF not available, trying PyPDF2...")

                # Fallback to PyPDF2
                try:
                    from PyPDF2 import PdfReader

                    reader = PdfReader(pdf_path)
                    extracted_text = []

                    start_page = 0
                    end_page = len(reader.pages)

                    if page_range:
                        start_page = max(0, page_range[0] - 1)
                        end_page = min(len(reader.pages), page_range[1])

                    logger.debug(
                        f"Extracting text from PDF with PyPDF2: {pdf_path} (pages {start_page+1}-{end_page})"
                    )

                    for page_num in range(start_page, end_page):
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        if text.strip():
                            extracted_text.append(
                                f"--- Page {page_num + 1} ---\n{text.strip()}"
                            )

                    if not extracted_text:
                        logger.warning("No text content found in PDF")
                        return ""

                    full_text = "\n\n".join(extracted_text)
                    logger.info(
                        f"Successfully extracted {len(full_text)} characters from {len(extracted_text)} pages"
                    )
                    return full_text

                except ImportError as e:
                    raise RuntimeError(
                        "No PDF library available. Install with: "
                        "pip install PyMuPDF (recommended) or pip install PyPDF2"
                    ) from e

        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise RuntimeError(f"PDF text extraction failed: {e}") from e

    async def synthesize_pdf(
        self,
        pdf_path: str,
        voice: str | None = None,
        page_range: tuple[int, int] | None = None,
        lang_code: str = "a",
        speed: float = 1.0,
        play_audio: bool = True,
        use_phoneme: bool = False,
        use_streaming: bool = True,
    ) -> tuple[bool, dict[str, Any]]:
        """Convert PDF content to speech

        Args:
            pdf_path: Path to PDF file
            voice: Voice to use for synthesis
            page_range: Optional tuple (start_page, end_page) for specific pages
            lang_code: Language code for synthesis
            speed: Speech speed multiplier
            play_audio: Whether to play audio immediately
            use_phoneme: If True, convert text to phonemes first for better quality
            use_streaming: If True, use streaming synthesis for long content

        Returns:
            Tuple of (success: bool, metrics: dict)
        """
        try:
            # Extract text from PDF
            pdf_text = await self.read_pdf(pdf_path, page_range)

            if not pdf_text.strip():
                return False, {"error": "No text content found in PDF"}

            # Clean up the text for better TTS
            # Remove excessive whitespace and page markers
            lines = pdf_text.split("\n")
            cleaned_lines = []

            for line in lines:
                line = line.strip()
                if line and not line.startswith("--- Page "):
                    cleaned_lines.append(line)

            cleaned_text = " ".join(cleaned_lines)

            # Remove multiple spaces
            import re

            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

            logger.info(f"Processing PDF text: {len(cleaned_text)} characters")

            # For long content, use chunked phoneme synthesis
            if use_phoneme and use_streaming and len(cleaned_text) > 1000:
                logger.info("Using chunked phoneme synthesis for long PDF content")

                # Split text into manageable chunks first
                text_chunks = await self.inference_engine.text_processor.split_text(
                    cleaned_text, max_length=200
                )
                logger.info(
                    f"Split PDF content into {len(text_chunks)} chunks for phoneme synthesis"
                )

                # Process each chunk with phonemes
                total_duration = 0

                for i, chunk_text in enumerate(text_chunks, 1):
                    if not chunk_text.strip():
                        continue

                    logger.debug(
                        f"Processing PDF chunk {i}/{len(text_chunks)}: {len(chunk_text)} chars"
                    )

                    try:
                        # Convert chunk to phonemes (should stay under 510 limit)
                        chunk_phonemes = await self.inference_engine.phonemize_text(
                            chunk_text, lang_code
                        )
                        logger.debug(
                            f"PDF chunk {i} phonemes: {len(chunk_phonemes)} chars"
                        )

                        # Synthesize chunk with phonemes
                        chunk_success, chunk_metrics = await self.text_to_speech(
                            message=chunk_phonemes,
                            voice=voice,
                            lang_code=lang_code,
                            speed=speed,
                            play_audio=True,  # Don't play individual chunks
                            use_phoneme=True,
                        )

                        if chunk_success:
                            total_duration += chunk_metrics.get("audio_duration", 0)
                            logger.debug(
                                f"PDF chunk {i} completed: {chunk_metrics.get('audio_duration', 0):.2f}s"
                            )
                        else:
                            logger.warning(
                                f"PDF chunk {i} failed: {chunk_metrics.get('error', 'Unknown')}"
                            )

                    except Exception as e:
                        logger.warning(f"Failed to process PDF chunk {i}: {e}")
                        continue

                # Create combined metrics
                success = True
                metrics = {
                    "generation": 0,  # Simplified - would need to track individual generation times
                    "audio_duration": total_duration,
                    "text_length": len(cleaned_text),
                    "chunks_processed": len(text_chunks),
                    "synthesis_mode": "chunked_phoneme_streaming",
                }

                logger.info(
                    f"Completed chunked PDF synthesis: {len(text_chunks)} chunks, {total_duration:.2f}s total audio"
                )

            else:
                # Regular synthesis
                success, metrics = await self.text_to_speech(
                    message=cleaned_text,
                    voice=voice,
                    lang_code=lang_code,
                    speed=speed,
                    play_audio=play_audio,
                )

            if success:
                metrics["pdf_path"] = pdf_path
                metrics["pdf_pages"] = page_range or "all"
                metrics["text_length"] = len(cleaned_text)
                metrics["synthesis_mode"] = (
                    "phoneme_streaming" if use_phoneme and use_streaming else "regular"
                )

            return success, metrics

        except Exception as e:
            logger.error(f"PDF synthesis failed: {e}")
            return False, {"error": f"PDF synthesis failed: {e}"}

    async def read_html(self, html_source: str, clean_text: bool = True) -> str:
        """Extract text content from HTML source

        Args:
            html_source: HTML content string or file path or URL
            clean_text: If True, clean up extracted text for better TTS readability

        Returns:
            Extracted text content from HTML

        Raises:
            RuntimeError: If HTML extraction fails
        """
        import os
        import re

        try:
            # Determine if input is a file path, URL, or raw HTML
            html_content = ""
            source_type = ""

            if html_source.startswith(("http://", "https://")):
                # It's a URL - fetch the content
                source_type = "URL"
                logger.debug(f"Fetching HTML from URL: {html_source}")

                try:
                    import requests

                    response = requests.get(html_source, timeout=10)
                    response.raise_for_status()
                    html_content = response.text
                    logger.info(
                        f"Successfully fetched {len(html_content)} characters from URL"
                    )
                except ImportError as e:
                    raise RuntimeError(
                        "requests library required for URL fetching. Install with: pip install requests"
                    ) from e
                except Exception as e:
                    raise RuntimeError(f"Failed to fetch URL: {e}") from e

            elif os.path.isfile(html_source):
                # It's a file path
                source_type = "file"
                logger.debug(f"Reading HTML from file: {html_source}")

                with open(html_source, encoding="utf-8", errors="ignore") as f:
                    html_content = f.read()
                logger.info(
                    f"Successfully read {len(html_content)} characters from file"
                )

            else:
                # Assume it's raw HTML content
                source_type = "raw HTML"
                html_content = html_source
                logger.debug(
                    f"Processing raw HTML content: {len(html_content)} characters"
                )

            if not html_content.strip():
                logger.warning("No HTML content to process")
                return ""

            # Parse and extract text from HTML
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(html_content, "html.parser")

                # Remove script, style, and code elements
                for script in soup(
                    [
                        "script",
                        "style",
                        "nav",
                        "footer",
                        "header",
                        "aside",
                        "code",
                        "pre",
                    ]
                ):
                    script.decompose()

                # Remove code blocks and inline code
                for code_block in soup.find_all(["code", "pre", "kbd", "samp", "var"]):
                    code_block.decompose()

                # Remove comments
                from bs4 import Comment

                for comment in soup.find_all(
                    string=lambda text: isinstance(text, Comment)
                ):
                    comment.extract()

                # Extract text content
                text_content = soup.get_text()

                logger.info(
                    f"Successfully extracted {len(text_content)} characters from HTML"
                )

            except ImportError:
                logger.debug(
                    "BeautifulSoup not available, trying basic HTML stripping..."
                )

                # Fallback: Basic HTML tag removal
                text_content = re.sub(
                    r"<script[^>]*>.*?</script>",
                    "",
                    html_content,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                text_content = re.sub(
                    r"<style[^>]*>.*?</style>",
                    "",
                    text_content,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                # Remove code blocks and inline code elements
                text_content = re.sub(
                    r"<code[^>]*>.*?</code>",
                    "",
                    text_content,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                text_content = re.sub(
                    r"<pre[^>]*>.*?</pre>",
                    "",
                    text_content,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                text_content = re.sub(
                    r"<kbd[^>]*>.*?</kbd>",
                    "",
                    text_content,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                text_content = re.sub(
                    r"<samp[^>]*>.*?</samp>",
                    "",
                    text_content,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                text_content = re.sub(
                    r"<var[^>]*>.*?</var>",
                    "",
                    text_content,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                text_content = re.sub(r"<[^>]+>", "", text_content)
                text_content = re.sub(
                    r"&[a-zA-Z0-9#]+;", " ", text_content
                )  # Remove HTML entities

                logger.info(
                    f"Extracted {len(text_content)} characters using basic HTML stripping"
                )

            if clean_text:
                # Clean up the extracted text
                lines = text_content.split("\n")
                cleaned_lines = []

                for line in lines:
                    line = line.strip()
                    if line:  # Only add non-empty lines
                        cleaned_lines.append(line)

                # Join lines and normalize whitespace
                cleaned_text = " ".join(cleaned_lines)
                cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

                logger.info(
                    f"Cleaned text: {len(cleaned_text)} characters (from {source_type})"
                )
                return cleaned_text
            else:
                return text_content.strip()

        except Exception as e:
            logger.error(f"Failed to extract text from HTML: {e}")
            raise RuntimeError(f"HTML text extraction failed: {e}") from e

    async def read_url(self, url: str, clean_text: bool = True) -> str:
        """Extract text content from a web page URL

        Args:
            url: Web page URL to fetch and extract text from
            clean_text: If True, clean up extracted text for better TTS readability

        Returns:
            Extracted text content from the web page

        Raises:
            RuntimeError: If URL fetching or text extraction fails
        """
        return await self.read_html(url, clean_text)

    async def synthesize_html(
        self,
        html_source: str,
        voice: str | None = None,
        clean_text: bool = True,
        lang_code: str = "a",
        speed: float = 1.0,
        play_audio: bool = True,
        use_phoneme: bool = False,
        use_streaming: bool = True,
    ) -> tuple[bool, dict[str, Any]]:
        """Convert HTML content to speech

        Args:
            html_source: HTML content string, file path, or URL
            voice: Voice to use for synthesis
            clean_text: If True, clean up extracted text for better TTS
            lang_code: Language code for synthesis
            speed: Speech speed multiplier
            play_audio: Whether to play audio immediately
            use_phoneme: If True, convert text to phonemes first for better quality
            use_streaming: If True, use streaming synthesis for long content

        Returns:
            Tuple of (success: bool, metrics: dict)
        """
        try:
            # Extract text from HTML
            html_text = await self.read_html(html_source, clean_text)

            if not html_text.strip():
                return False, {"error": "No text content found in HTML"}

            logger.info(f"Processing HTML text: {len(html_text)} characters")

            # For long content, use chunked phoneme synthesis
            if use_phoneme and use_streaming and len(html_text) > 1000:
                logger.info("Using chunked phoneme synthesis for long HTML content")

                # Split text into manageable chunks first
                text_chunks = await self.inference_engine.text_processor.split_text(
                    html_text, max_length=200
                )
                logger.info(
                    f"Split content into {len(text_chunks)} chunks for phoneme synthesis"
                )

                # Process each chunk with phonemes
                total_duration = 0

                for i, chunk_text in enumerate(text_chunks, 1):
                    if not chunk_text.strip():
                        continue

                    logger.debug(
                        f"Processing chunk {i}/{len(text_chunks)}: {len(chunk_text)} chars"
                    )

                    try:
                        # Convert chunk to phonemes (should stay under 510 limit)
                        chunk_phonemes = await self.inference_engine.phonemize_text(
                            chunk_text, lang_code
                        )
                        logger.debug(f"Chunk {i} phonemes: {len(chunk_phonemes)} chars")

                        # Synthesize chunk with phonemes
                        chunk_success, chunk_metrics = await self.text_to_speech(
                            message=chunk_phonemes,
                            voice=voice,
                            lang_code=lang_code,
                            speed=speed,
                            play_audio=play_audio,  # Play chunks if requested
                            use_phoneme=True,
                        )

                        if chunk_success:
                            # For now, we can't easily concatenate audio, so we'll track metrics
                            total_duration += chunk_metrics.get("audio_duration", 0)
                            logger.debug(
                                f"Chunk {i} completed: {chunk_metrics.get('audio_duration', 0):.2f}s"
                            )
                        else:
                            logger.warning(
                                f"Chunk {i} failed: {chunk_metrics.get('error', 'Unknown')}"
                            )

                    except Exception as e:
                        logger.warning(f"Failed to process chunk {i}: {e}")
                        continue

                # Create combined metrics
                success = True
                metrics = {
                    "generation": sum(
                        chunk_metrics.get("generation", 0) for chunk_metrics in [{}]
                    ),  # Simplified
                    "audio_duration": total_duration,
                    "text_length": len(html_text),
                    "chunks_processed": len(text_chunks),
                    "synthesis_mode": "chunked_phoneme_streaming",
                }

                logger.info(
                    f"Completed chunked synthesis: {len(text_chunks)} chunks, {total_duration:.2f}s total audio"
                )

            else:
                # Regular synthesis
                success, metrics = await self.text_to_speech(
                    message=html_text,
                    voice=voice,
                    lang_code=lang_code,
                    speed=speed,
                    play_audio=play_audio,
                )

            if success:
                metrics["html_source"] = (
                    html_source[:100] + "..." if len(html_source) > 100 else html_source
                )
                metrics["text_length"] = len(html_text)
                metrics["source_type"] = "HTML"
                metrics["synthesis_mode"] = (
                    "phoneme_streaming" if use_phoneme and use_streaming else "regular"
                )

            return success, metrics

        except Exception as e:
            logger.error(f"HTML synthesis failed: {e}")
            return False, {"error": f"HTML synthesis failed: {e}"}

    async def synthesize_url(
        self,
        url: str,
        voice: str | None = None,
        clean_text: bool = True,
        lang_code: str = "a",
        speed: float = 1.0,
        play_audio: bool = True,
        use_phoneme: bool = False,
        use_streaming: bool = True,
    ) -> tuple[bool, dict[str, Any]]:
        """Convert web page content to speech

        Args:
            url: Web page URL to fetch and synthesize
            voice: Voice to use for synthesis
            clean_text: If True, clean up extracted text for better TTS
            lang_code: Language code for synthesis
            speed: Speech speed multiplier
            play_audio: Whether to play audio immediately
            use_phoneme: If True, convert text to phonemes first for better quality
            use_streaming: If True, use streaming synthesis for long content

        Returns:
            Tuple of (success: bool, metrics: dict)
        """
        try:
            success, metrics = await self.synthesize_html(
                url,
                voice,
                clean_text,
                lang_code,
                speed,
                play_audio,
                use_phoneme,
                use_streaming,
            )

            if success:
                metrics["url"] = url
                metrics["source_type"] = "URL"

            return success, metrics

        except Exception as e:
            logger.error(f"URL synthesis failed: {e}")
            return False, {"error": f"URL synthesis failed: {e}"}


if __name__ == "__main__":
    import asyncio

    from champi_tts.providers.kokoro.config import KokoroConfig

    async def test_html_synthesis():
        """Test HTML/Web page synthesis functionality"""
        print("🌐 Testing Kokoro HTML/Web Page Synthesis")

        # Load environment from .env.kokoro if it exists
        env_file = Path(__file__).parent.parent.parent / ".env.kokoro"
        if env_file.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(env_file)
                print(f"📁 Loaded config from {env_file}")
            except ImportError:
                print("⚠️  python-dotenv not available, using system environment")

        # Create config
        config = KokoroConfig()
        config = config.from_env()  # Load from environment variables

        # Create provider
        provider = KokoroProvider(config)

        try:
            # Initialize
            print("📦 Initializing provider...")
            await provider.initialize()

            # Get info
            info = await provider.get_info()
            print(f"✅ Provider initialized: {info['name']} v{info['version']}")
            print(f"🎵 Available voices: {info['voice_count']} voices")

            # List voices
            voices = provider.list_voices()
            if not voices:
                print("⚠️  No voices available for testing")
                return

            # Get English voice for test
            english_voices = [
                v for v in voices if v.startswith(("af_", "am_", "bf_", "bm_"))
            ]
            test_voice = english_voices[0] if english_voices else voices[0]
            print(f"🎵 Using voice: {test_voice}")

            # Test web page reading and synthesis
            print("\n🌐 Testing Web Page Reading & Synthesis")
            print(f"{'='*60}")

            test_url = "https://blog.jetbrains.com/pycharm/2025/07/faster-python-unlocking-the-python-global-interpreter-lock/"
            print(f"📄 Testing URL: {test_url}")

            try:
                # First, just read the content to show what we extracted
                print("\n📖 Extracting text content...")
                web_text = await provider.read_url(test_url, clean_text=True)

                # Show preview of extracted content
                preview_length = 300
                preview_text = (
                    web_text[:preview_length] + "..."
                    if len(web_text) > preview_length
                    else web_text
                )
                print(f"   📝 Extracted {len(web_text)} characters from web page")
                print(f"   📄 Preview: '{preview_text}'")

                if len(web_text) > 100:  # Only synthesize if we got meaningful content
                    # Synthesize first part of the article (limit to reasonable length for demo)
                    synthesis_text = (
                        web_text[:800] + "..." if len(web_text) > 800 else web_text
                    )
                    print(
                        f"\n🎵 Synthesizing first {len(synthesis_text)} characters with voice: {test_voice}"
                    )

                    success, metrics = await provider.synthesize_url(
                        test_url,
                        voice=test_voice,
                        lang_code="a",
                        play_audio=True,
                        clean_text=True,
                        use_phoneme=True,
                        use_streaming=True,
                    )

                    if success:
                        gen_time = metrics.get("generation", 0)
                        audio_dur = metrics.get("audio_duration", 0)
                        text_len = metrics.get("text_length", 0)
                        synthesis_mode = metrics.get("synthesis_mode", "regular")
                        print("   ✅ Web synthesis successful!")
                        print(
                            f"   📊 Stats: {text_len} chars → {gen_time:.2f}s gen → {audio_dur:.2f}s audio"
                        )
                        print(f"   🎭 Mode: {synthesis_mode}")
                        print(f"   🌐 Source: {metrics.get('source_type', 'unknown')}")
                    else:
                        print(
                            f"   ❌ Web synthesis failed: {metrics.get('error', 'Unknown error')}"
                        )
                else:
                    print("   ⚠️  Not enough content extracted for synthesis")

            except Exception as e:
                print(f"   ❌ Web page test failed: {e}")
                # Don't let web page test failure stop the entire test

            print(f"\n{'-'*60}")

        except Exception as e:
            print(f"❌ Provider test failed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Cleanup
            await provider.unload()
            print("🧹 Provider unloaded")

    # Run the test
    asyncio.run(test_html_synthesis())
