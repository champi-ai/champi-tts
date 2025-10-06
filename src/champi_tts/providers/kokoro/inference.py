"""
Core Kokoro TTS inference engine.
Extracted from Kokoro-FastAPI for direct integration.
"""

import asyncio
import os
import tempfile
import time
from collections.abc import AsyncGenerator

import numpy as np
import torch
from loguru import logger

try:
    from kokoro import KModel, KPipeline
except ImportError:
    logger.error("Kokoro package not found. Install with: pip install kokoro-tts")
    raise

import contextlib

from champi_tts.providers.kokoro.config import KokoroConfig
from champi_tts.providers.kokoro.enums import (
    TTSEventTypes,
)
from champi_tts.providers.kokoro.events import EventProcessor, TTSSignalManager
from champi_tts.providers.kokoro.text_utils import AsyncTextProcessor


class AudioChunk:
    """Audio chunk with optional timestamps"""

    def __init__(
        self,
        audio: np.ndarray,
        word_timestamps: list | None = None,
        output: bytes | None = None,
    ):
        self.audio = audio
        self.word_timestamps = word_timestamps or []
        self.output = output

    @staticmethod
    def combine(chunks: list["AudioChunk"]) -> "AudioChunk":
        """Combine multiple audio chunks"""
        if not chunks:
            return AudioChunk(np.array([], dtype=np.int16))

        combined_audio = chunks[0].audio
        combined_timestamps = chunks[0].word_timestamps.copy()

        for chunk in chunks[1:]:
            combined_audio = np.concatenate([combined_audio, chunk.audio])
            combined_timestamps.extend(chunk.word_timestamps)

        return AudioChunk(combined_audio, combined_timestamps)


class KokoroInference:
    """Complete Kokoro TTS inference engine"""

    class Meta:
        event_type = TTSEventTypes.MODEL_EVENT.value

    def __init__(self, config: KokoroConfig | None = None):
        self.config = config or KokoroConfig()
        self.device = self._determine_device()
        self.model: KModel | None = None
        self.pipelines: dict[str, KPipeline] = {}
        self.text_processor = AsyncTextProcessor()
        self._voice_tensor = None
        self._voice_tensor_tmp = None
        self._voice_path = None
        self._initialized = False
        self.signals = (
            TTSSignalManager()
        )  # Global signal manager for all inference events

    def _determine_device(self) -> str:
        """Determine best available device"""
        if self.config.force_cpu:
            return "cpu"
        elif torch.cuda.is_available() and self.config.use_gpu:
            return "cuda"
        elif torch.backends.mps.is_available() and self.config.use_gpu:
            return "mps"
        else:
            return "cpu"

    @EventProcessor.emits_event(data=["device", "model"])
    async def load_model(self, model_path: str, config_path: str) -> None:
        """Load Kokoro model from files"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")

            logger.debug(f"Loading Kokoro model on {self.device}")
            logger.debug(f"Model: {model_path}")
            logger.debug(f"Config: {config_path}")

            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, lambda: KModel(config=config_path, model=model_path).eval()
            )

            # Move to device
            if self.device == "cuda":
                self.model = self.model.cuda()
            elif self.device == "mps":
                self.model = self.model.to(torch.device("mps"))
            else:
                self.model = self.model.cpu()

            self._initialized = True
            logger.debug("Kokoro model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Kokoro model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def _get_pipeline(self, lang_code: str) -> KPipeline:
        """Get or create pipeline for language"""
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if lang_code not in self.pipelines:
            logger.debug(f"Creating pipeline for language: {lang_code}")
            self.pipelines[lang_code] = KPipeline(
                lang_code=lang_code, model=self.model, device=self.device
            )

        return self.pipelines[lang_code]

    def _load_voice_tensor(self, voice_path: str) -> torch.Tensor:
        """Load and prepare voice tensor"""
        if not os.path.exists(voice_path):
            raise FileNotFoundError(f"Voice file not found: {voice_path}")

        try:
            # Load voice tensor from trusted voice files
            # nosec B614 - Voice files are user-provided trusted model files
            voice_tensor = torch.load(
                voice_path, map_location="cpu", weights_only=False
            )

            # Move to target device
            if self.device == "cuda":
                voice_tensor = voice_tensor.cuda()
            elif self.device == "mps":
                voice_tensor = voice_tensor.to(torch.device("mps"))
            self._voice_tensor = voice_tensor
            return voice_tensor
        except Exception as e:
            raise RuntimeError(f"Failed to load voice tensor: {e}") from e

    def _save_voice_tensor_temp(
        self, voice_tensor: torch.Tensor, voice_name: str
    ) -> str:
        """Save voice tensor to temporary file"""
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"kokoro_voice_{voice_name}.pt")

        # Save tensor with CPU mapping for portability
        torch.save(voice_tensor.cuda(), temp_path)
        return temp_path

    async def phonemize_text(self, text: str, lang_code: str = "a") -> str:
        """Convert text to phonemes"""
        return await self.text_processor.phonemize(text, lang_code)

    async def generate_from_text(
        self,
        text: str,
        voice_path: str,
        lang_code: str = "a",
        speed: float = 1.0,
        normalize: bool = True,
    ) -> AudioChunk:
        """Generate audio from text"""
        if not self._initialized:
            raise RuntimeError("Model not loaded")

        try:
            # Process text
            if normalize:
                processed_text = await self.text_processor.normalize(text)
            else:
                processed_text = text.strip()

            if not processed_text:
                return AudioChunk(np.array([], dtype=np.int16))

            # Load voice tensor
            self._voice_tensor = self._load_voice_tensor(voice_path)
            self._voice_tensor_tmp = self._save_voice_tensor_temp(
                self._voice_tensor, os.path.basename(voice_path)
            )

            # Get pipeline and generate
            pipeline = self._get_pipeline(lang_code)

            logger.info(
                f"Generating audio for: '{text[:50]}{'...' if len(text) > 50 else ''}'"
            )

            audio_chunks = []

            # Add initial silence to prevent eating first words
            delay_samples = int(0.5 * 24000)  # 500ms delay at 24kHz
            silence = np.zeros(delay_samples, dtype=np.float32)
            audio_chunks.append(silence)

            # Run generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def _generate():
                chunks = []
                for result in pipeline(
                    processed_text,
                    voice=self._voice_tensor_tmp,
                    speed=speed,
                    model=self.model,
                ):
                    if result.audio is not None:
                        chunks.append(result.audio.numpy())
                return chunks

            generated_chunks = await loop.run_in_executor(None, _generate)
            audio_chunks.extend(generated_chunks)

            # Clean up temp file
            with contextlib.suppress(Exception):
                os.unlink(self._voice_tensor_tmp)

            if audio_chunks:
                combined_audio = np.concatenate(audio_chunks)
                return AudioChunk(combined_audio)
            else:
                raise RuntimeError("No audio generated")

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise

    async def generate_from_phonemes(
        self, phonemes: str, voice_path: str, lang_code: str = "a", speed: float = 1.0
    ) -> AudioChunk:
        """Generate audio from phonemes"""
        if not self._initialized:
            raise RuntimeError("Model not loaded")

        try:
            if not phonemes.strip():
                return AudioChunk(np.array([], dtype=np.int16))

            # Load voice tensor
            voice_tensor = self._load_voice_tensor(voice_path)
            temp_voice_path = self._save_voice_tensor_temp(
                voice_tensor, os.path.basename(voice_path)
            )

            # Get pipeline and generate
            pipeline = self._get_pipeline(lang_code)

            logger.debug(
                f"Generating from phonemes: '{phonemes[:50]}{'...' if len(phonemes) > 50 else ''}'"
            )

            # Run generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def _generate():
                for result in pipeline.generate_from_tokens(
                    tokens=phonemes,
                    voice=temp_voice_path,
                    speed=speed,
                    model=self.model,
                ):
                    if result.audio is not None:
                        return result.audio.numpy()
                return None

            audio_data = await loop.run_in_executor(None, _generate)

            # Clean up temp file
            with contextlib.suppress(Exception):
                os.unlink(temp_voice_path)

            if audio_data is not None:
                return AudioChunk(audio_data)
            else:
                raise RuntimeError("No audio generated from phonemes")

        except Exception as e:
            logger.error(f"Phoneme generation failed: {e}")
            raise

    async def generate_stream(
        self,
        text: str,
        voice_path: str,
        lang_code: str = "a",
        speed: float = 1.0,
        chunk_size: int = 200,
        use_phoneme: bool = False,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate audio in streaming chunks

        Args:
            text: Input text or phonemes (depending on use_phoneme flag)
            voice_path: Path to voice file
            lang_code: Language code
            speed: Speech speed multiplier
            chunk_size: Maximum characters per chunk
            use_phoneme: If True, treat input as phonemes instead of text
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded")

        if use_phoneme:
            # For phonemes, split differently - preserve phoneme boundaries
            # Split on whitespace to maintain phoneme integrity
            phoneme_chunks = []
            words = text.split()
            current_chunk = []
            current_length = 0

            for word in words:
                word_length = len(word) + 1  # +1 for space
                if current_length + word_length > chunk_size and current_chunk:
                    phoneme_chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = word_length
                else:
                    current_chunk.append(word)
                    current_length += word_length

            if current_chunk:
                phoneme_chunks.append(" ".join(current_chunk))

            # Generate audio from phoneme chunks
            for chunk_phonemes in phoneme_chunks:
                if chunk_phonemes.strip():
                    try:
                        chunk_audio = await self.generate_from_phonemes(
                            chunk_phonemes, voice_path, lang_code, speed
                        )
                        yield chunk_audio
                    except Exception as e:
                        logger.warning(f"Failed to generate phoneme chunk: {e}")
                        continue
        else:
            # Regular text processing - split text into manageable chunks
            text_chunks = await self.text_processor.split_text(text, chunk_size)

            for chunk_text in text_chunks:
                if chunk_text.strip():
                    try:
                        chunk_audio = await self.generate_from_text(
                            chunk_text, voice_path, lang_code, speed
                        )
                        yield chunk_audio
                    except Exception as e:
                        logger.warning(f"Failed to generate text chunk: {e}")
                        continue

    async def synthesize(self, text: str, voice_path: str, **kwargs) -> np.ndarray:
        """Main synthesis interface for mcp_champi compatibility"""
        lang_code = kwargs.get("lang_code", "a")
        speed = kwargs.get("speed", 1.0)
        normalize = kwargs.get("normalize", True)

        result = await self.generate_from_text(
            text, voice_path, lang_code, speed, normalize
        )
        return result.audio

    def list_voices(self, voice_dir: str) -> list[str]:
        """List available voice files"""
        if not os.path.exists(voice_dir):
            return []

        voices = []
        for file in os.listdir(voice_dir):
            if file.endswith(".pt"):
                voices.append(os.path.splitext(file)[0])

        return sorted(voices)

    def unload(self) -> None:
        """Unload model and free resources"""
        if self.model is not None:
            del self.model
            self.model = None

        for pipeline in self.pipelines.values():
            del pipeline
        self.pipelines.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._initialized = False
        logger.debug("Kokoro model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._initialized and self.model is not None

    async def warmup(self, voice_path: str, lang_code: str = "a") -> None:
        """Warm up the model with a short generation"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        warmup_text = "Warmup text for model initialization."
        try:
            start_time = time.time()
            await self.generate_from_text(warmup_text, voice_path, lang_code)
            warmup_time = (time.time() - start_time) * 1000
            logger.debug(f"Model warmup completed in {warmup_time:.0f}ms")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
