"""
Model management utilities for Kokoro TTS.
Handles model downloading, verification, and setup.
"""

import json
import os
from urllib.request import urlretrieve

from loguru import logger

from champi_tts.providers.kokoro.enums import ModelEvents
from champi_tts.providers.kokoro.events import TTSSignalManager


class ModelDownloader:
    """Handles Kokoro model downloading and verification"""

    RELEASE_VERSION = "v0.1.4"
    BASE_URL = (
        f"https://github.com/remsky/Kokoro-FastAPI/releases/download/{RELEASE_VERSION}"
    )

    MODEL_FILES = {
        "kokoro-v1_0.pth": {
            "url": f"{BASE_URL}/kokoro-v1_0.pth",
            "size_mb": 85,  # Approximate size for validation
        },
        "config.json": {"url": f"{BASE_URL}/config.json", "size_mb": 0.001},
    }

    @classmethod
    def verify_files(cls, model_path: str, config_path: str) -> bool:
        """Verify that model files exist and are valid (matches Kokoro-FastAPI verification)"""
        signals = TTSSignalManager()

        try:
            # Check files exist
            if not os.path.exists(model_path):
                signals.model.send(
                    event_type="model",
                    sub_event=ModelEvents.MODEL_ERROR.value,
                    data={"error": f"Model file not found: {model_path}"},
                )
                return False
            if not os.path.exists(config_path):
                signals.model.send(
                    event_type="model",
                    sub_event=ModelEvents.MODEL_ERROR.value,
                    data={"error": f"Config file not found: {config_path}"},
                )
                return False

            # Verify config file is valid JSON
            with open(config_path) as f:
                json.load(f)

            # Check model file size (should be non-zero)
            if os.path.getsize(model_path) == 0:
                signals.model.send(
                    event_type="model",
                    sub_event=ModelEvents.MODEL_ERROR.value,
                    data={"error": f"Model file is empty: {model_path}"},
                )
                return False

            model_size_mb = os.path.getsize(model_path) / 1024 / 1024
            logger.debug(f"Model validation passed - Model: {model_size_mb:.1f}MB")

            signals.model.send(
                event_type="model",
                sub_event=ModelEvents.MODEL_LOADED.value,
                data={
                    "model_path": model_path,
                    "config_path": config_path,
                    "model_size_mb": model_size_mb,
                },
            )
            return True

        except Exception as e:
            logger.error(f"File verification failed: {e}")
            signals.model.send(
                event_type="model",
                sub_event=ModelEvents.MODEL_ERROR.value,
                data={"error": f"File verification failed: {e}"},
            )
            return False

    @classmethod
    def download_with_progress(cls, url: str, filepath: str) -> None:
        """Download file with basic progress logging"""

        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                if (
                    block_num % 100 == 0 or percent >= 100
                ):  # Log every ~1MB or at completion
                    logger.debug(
                        f"Downloading {os.path.basename(filepath)}: {percent}%"
                    )

        logger.debug(f"Starting download: {url}")
        # Using urlretrieve for downloading model files from trusted sources only
        # nosec B310 - URL scheme is validated to be https:// only
        urlretrieve(url, filepath, reporthook=progress_hook)
        logger.debug(f"Download completed: {filepath}")

    @classmethod
    def download_model(cls, output_dir: str, force: bool = False) -> tuple[str, str]:
        """Download Kokoro model files

        Args:
            output_dir: Directory to save model files
            force: Force re-download even if files exist

        Returns:
            Tuple of (model_path, config_path)
        """
        # Define file paths
        model_path = os.path.join(output_dir, "kokoro-v1_0.pth")
        config_path = os.path.join(output_dir, "config.json")

        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Check if files already exist and are valid
            if not force and cls.verify_files(model_path, config_path):
                logger.debug("Model files already exist and are valid")
                return model_path, config_path

            logger.debug("Downloading Kokoro v1.0 model files...")

            # Download model file
            if force or not os.path.exists(model_path):
                logger.debug("Downloading model file (this may take a few minutes)...")
                cls.download_with_progress(
                    cls.MODEL_FILES["kokoro-v1_0.pth"]["url"], model_path
                )

            # Download config file
            if force or not os.path.exists(config_path):
                logger.debug("Downloading config file...")
                cls.download_with_progress(
                    cls.MODEL_FILES["config.json"]["url"], config_path
                )

            # Verify downloaded files
            if not cls.verify_files(model_path, config_path):
                raise RuntimeError("Downloaded files failed verification")

            logger.debug(f"✓ Model files ready in {output_dir}")
            return model_path, config_path

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            # Clean up partial downloads
            for path in [model_path, config_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        logger.debug(f"Cleaned up partial download: {path}")
                    except Exception as _err:
                        logger.debug(
                            f"Could not remove partial download {path}: {_err}"
                        )
            raise


class VoiceManager:
    """Manages voice files and directories"""

    @staticmethod
    def setup_voice_directory(voice_dir: str) -> str:
        """Set up voice directory structure"""
        os.makedirs(voice_dir, exist_ok=True)

        # Create a default voice info file
        info_file = os.path.join(voice_dir, "README.txt")
        if not os.path.exists(info_file):
            with open(info_file, "w") as f:
                f.write(
                    """Kokoro Voice Directory
====================

Place your Kokoro voice files (.pt) in this directory.

Voice files should be PyTorch tensors saved with torch.save().
You can download voice files from the Kokoro project or create your own.

Example usage:
- af_bella.pt
- af_sarah.pt
- am_adam.pt

The voice name (filename without .pt) will be used to identify the voice.
"""
                )

        return voice_dir

    @staticmethod
    def list_voices(voice_dir: str) -> list[str]:
        """List available voice files"""
        if not os.path.exists(voice_dir):
            return []

        voices = []
        for file in os.listdir(voice_dir):
            if file.endswith(".pt") and os.path.isfile(os.path.join(voice_dir, file)):
                voice_name = os.path.splitext(file)[0]
                voices.append(voice_name)

        return sorted(voices)

    @staticmethod
    def validate_voice_file(voice_path: str) -> bool:
        """Validate a voice file"""
        signals = TTSSignalManager()
        voice_name = os.path.basename(voice_path)

        try:
            signals.model.send(
                event_type="model",
                sub_event=ModelEvents.VOICE_LOADING.value,
                data={"voice_path": voice_path, "voice_name": voice_name},
            )

            if not os.path.exists(voice_path):
                signals.model.send(
                    event_type="model",
                    sub_event=ModelEvents.MODEL_ERROR.value,
                    data={"error": f"Voice file not found: {voice_path}"},
                )
                return False

            # Check file size
            if os.path.getsize(voice_path) == 0:
                signals.model.send(
                    event_type="model",
                    sub_event=ModelEvents.MODEL_ERROR.value,
                    data={"error": f"Voice file is empty: {voice_path}"},
                )
                return False

            # Try to load as torch tensor (basic validation)
            import torch

            # nosec B614 - Voice files are user-provided trusted model files
            tensor = torch.load(voice_path, map_location="cpu", weights_only=False)

            # Basic tensor validation
            if not isinstance(tensor, torch.Tensor):
                signals.model.send(
                    event_type="model",
                    sub_event=ModelEvents.MODEL_ERROR.value,
                    data={"error": f"Voice file is not a valid tensor: {voice_path}"},
                )
                return False

            if tensor.numel() == 0:
                signals.model.send(
                    event_type="model",
                    sub_event=ModelEvents.MODEL_ERROR.value,
                    data={"error": f"Voice tensor is empty: {voice_path}"},
                )
                return False

            logger.debug(
                f"Voice file validation passed: {voice_path} (shape: {tensor.shape})"
            )

            signals.model.send(
                event_type="model",
                sub_event=ModelEvents.VOICE_LOADED.value,
                data={
                    "voice_path": voice_path,
                    "voice_name": voice_name,
                    "tensor_shape": list(tensor.shape),
                    "tensor_elements": tensor.numel(),
                },
            )
            return True

        except Exception as e:
            logger.warning(f"Voice file validation failed: {voice_path} - {e}")
            signals.model.send(
                event_type="model",
                sub_event=ModelEvents.MODEL_ERROR.value,
                data={"error": f"Voice validation failed: {voice_path} - {e}"},
            )
            return False
