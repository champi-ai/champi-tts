"""
Tests for KokoroConfig and KokoroConfigPresets.

Heavy torch/kokoro dependencies are mocked where needed.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from champi_tts.providers.kokoro.config import KokoroConfig, KokoroConfigPresets


class TestKokoroConfigDefaults:
    """Tests for KokoroConfig default values."""

    def test_default_language(self) -> None:
        """Default language is American English 'a'."""
        assert KokoroConfig().default_language == "a"

    def test_default_speed(self) -> None:
        """Default speed is 1.0."""
        assert KokoroConfig().default_speed == 1.0

    def test_default_voice(self) -> None:
        """Default voice is am_adam."""
        assert KokoroConfig().default_voice == "am_adam"

    def test_sample_rate(self) -> None:
        """Default sample rate is 24000."""
        assert KokoroConfig().sample_rate == 24000

    def test_use_gpu_default_true(self) -> None:
        """GPU usage is enabled by default."""
        assert KokoroConfig().use_gpu is True

    def test_enable_streaming_default_true(self) -> None:
        """Streaming is enabled by default."""
        assert KokoroConfig().enable_streaming is True

    def test_normalize_text_default_true(self) -> None:
        """Text normalisation is enabled by default."""
        assert KokoroConfig().normalize_text is True


class TestKokoroConfigPostInit:
    """Tests for KokoroConfig.__post_init__() validation."""

    def test_invalid_language_reset_to_a(self) -> None:
        """Invalid language code is corrected to 'a'."""
        config = KokoroConfig(default_language="xx")
        assert config.default_language == "a"

    def test_valid_language_codes_preserved(self) -> None:
        """Supported language codes are not modified."""
        for lang in ["a", "b", "en", "en-us", "en-gb"]:
            config = KokoroConfig(default_language=lang)
            assert config.default_language == lang

    def test_speed_too_high_reset(self) -> None:
        """Speed above 3.0 is reset to 1.0."""
        config = KokoroConfig(default_speed=5.0)
        assert config.default_speed == 1.0

    def test_speed_too_low_reset(self) -> None:
        """Speed below 0.1 is reset to 1.0."""
        config = KokoroConfig(default_speed=0.05)
        assert config.default_speed == 1.0

    def test_valid_speed_preserved(self) -> None:
        """Speed within [0.1, 3.0] is preserved."""
        config = KokoroConfig(default_speed=1.5)
        assert config.default_speed == 1.5

    def test_small_chunk_size_reset(self) -> None:
        """Chunk size below 50 is reset to 200."""
        config = KokoroConfig(streaming_chunk_size=10)
        assert config.streaming_chunk_size == 200

    def test_valid_chunk_size_preserved(self) -> None:
        """Chunk size of 50 or more is preserved."""
        config = KokoroConfig(streaming_chunk_size=100)
        assert config.streaming_chunk_size == 100


class TestKokoroConfigFromDict:
    """Tests for KokoroConfig.from_dict()."""

    def test_creates_from_valid_dict(self) -> None:
        """from_dict() creates a config from a valid dictionary."""
        config = KokoroConfig.from_dict(
            {"default_voice": "af_bella", "default_speed": 1.2}
        )
        assert config.default_voice == "af_bella"
        assert config.default_speed == 1.2

    def test_ignores_unknown_keys(self) -> None:
        """from_dict() ignores keys that are not config fields."""
        config = KokoroConfig.from_dict(
            {"unknown_key": "ignored", "default_speed": 1.0}
        )
        assert config.default_speed == 1.0

    def test_empty_dict_uses_defaults(self) -> None:
        """from_dict() with empty dict returns a config with defaults."""
        config = KokoroConfig.from_dict({})
        assert config.default_speed == 1.0
        assert config.default_language == "a"


class TestKokoroConfigToDict:
    """Tests for KokoroConfig.to_dict()."""

    def test_returns_dict(self) -> None:
        """to_dict() returns a dictionary."""
        result = KokoroConfig().to_dict()
        assert isinstance(result, dict)

    def test_contains_key_fields(self) -> None:
        """to_dict() includes all major config fields."""
        result = KokoroConfig().to_dict()
        for key in ["default_voice", "default_speed", "sample_rate", "use_gpu"]:
            assert key in result

    def test_values_match_instance(self) -> None:
        """to_dict() values match the instance attributes."""
        config = KokoroConfig(default_voice="af_bella", default_speed=1.3)
        result = config.to_dict()
        assert result["default_voice"] == "af_bella"
        assert result["default_speed"] == 1.3


class TestKokoroConfigFromFile:
    """Tests for KokoroConfig.from_file()."""

    def test_loads_valid_json_file(self, tmp_path: Path) -> None:
        """from_file() reads settings from a JSON config file."""
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps({"default_voice": "af_bella", "default_speed": 1.2})
        )
        config = KokoroConfig.from_file(str(config_file))
        assert config.default_voice == "af_bella"
        assert config.default_speed == 1.2

    def test_missing_file_returns_defaults(self) -> None:
        """from_file() returns default config when file does not exist."""
        config = KokoroConfig.from_file("/nonexistent/config.json")
        assert config.default_speed == 1.0

    def test_invalid_json_returns_defaults(self, tmp_path: Path) -> None:
        """from_file() returns default config for malformed JSON."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json}")
        config = KokoroConfig.from_file(str(bad_file))
        assert config.default_speed == 1.0


class TestKokoroConfigFromEnv:
    """Tests for KokoroConfig.from_env()."""

    def test_from_env_returns_config(self) -> None:
        """from_env() returns a KokoroConfig instance."""
        config = KokoroConfig.from_env()
        assert isinstance(config, KokoroConfig)

    def test_voice_read_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """KOKORO_DEFAULT_VOICE env var sets default_voice."""
        monkeypatch.setenv("KOKORO_DEFAULT_VOICE", "af_bella")
        config = KokoroConfig.from_env()
        assert config.default_voice == "af_bella"

    def test_speed_read_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """KOKORO_DEFAULT_SPEED env var sets default_speed."""
        monkeypatch.setenv("KOKORO_DEFAULT_SPEED", "1.5")
        config = KokoroConfig.from_env()
        assert config.default_speed == 1.5

    def test_invalid_speed_env_uses_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid KOKORO_DEFAULT_SPEED env var falls back to default."""
        monkeypatch.setenv("KOKORO_DEFAULT_SPEED", "not_a_number")
        config = KokoroConfig.from_env()
        assert config.default_speed == 1.0

    def test_use_gpu_false_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """KOKORO_USE_GPU=false disables GPU."""
        monkeypatch.setenv("KOKORO_USE_GPU", "false")
        config = KokoroConfig.from_env()
        assert config.use_gpu is False

    def test_force_cpu_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """KOKORO_FORCE_CPU=true enables force_cpu."""
        monkeypatch.setenv("KOKORO_FORCE_CPU", "true")
        config = KokoroConfig.from_env()
        assert config.force_cpu is True

    def test_language_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """KOKORO_DEFAULT_LANGUAGE env var sets default_language."""
        monkeypatch.setenv("KOKORO_DEFAULT_LANGUAGE", "b")
        config = KokoroConfig.from_env()
        assert config.default_language == "b"

    def test_model_dir_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """KOKORO_MODEL_DIR env var sets model_dir."""
        monkeypatch.setenv("KOKORO_MODEL_DIR", "/models/kokoro")
        config = KokoroConfig.from_env()
        assert config.model_dir == "/models/kokoro"

    def test_normalize_text_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """KOKORO_NORMALIZE_TEXT=false disables text normalisation."""
        monkeypatch.setenv("KOKORO_NORMALIZE_TEXT", "false")
        config = KokoroConfig.from_env()
        assert config.normalize_text is False

    def test_warmup_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """KOKORO_WARMUP_ON_INIT=false disables warmup."""
        monkeypatch.setenv("KOKORO_WARMUP_ON_INIT", "false")
        config = KokoroConfig.from_env()
        assert config.warmup_on_init is False

    def test_auto_download_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """KOKORO_AUTO_DOWNLOAD=false disables auto-download."""
        monkeypatch.setenv("KOKORO_AUTO_DOWNLOAD", "false")
        config = KokoroConfig.from_env()
        assert config.auto_download_model is False

    def test_available_voices_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CHAMPI_TTS_VOICES env var parses comma-separated voice list."""
        monkeypatch.setenv("CHAMPI_TTS_VOICES", "af_bella,af_sarah,am_adam")
        config = KokoroConfig.from_env()
        assert config.available_voices == ["af_bella", "af_sarah", "am_adam"]

    def test_auto_start_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CHAMPI_AUTO_START_KOKORO=true enables auto_start."""
        monkeypatch.setenv("CHAMPI_AUTO_START_KOKORO", "yes")
        config = KokoroConfig.from_env()
        assert config.auto_start is True

    def test_tts_audio_format_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CHAMPI_TTS_AUDIO_FORMAT env var sets tts_audio_format."""
        monkeypatch.setenv("CHAMPI_TTS_AUDIO_FORMAT", "flac")
        config = KokoroConfig.from_env()
        assert config.tts_audio_format == "flac"


class TestKokoroConfigProperties:
    """Tests for KokoroConfig property methods."""

    def test_supported_audio_formats(self) -> None:
        """supported_audio_formats returns expected set of formats."""
        config = KokoroConfig()
        formats = config.supported_audio_formats
        assert "wav" in formats
        assert "mp3" in formats
        assert "flac" in formats

    def test_validate_tts_audio_format_valid(self) -> None:
        """validate_tts_audio_format() returns the format when it is supported."""
        config = KokoroConfig(tts_audio_format="wav")
        assert config.validate_tts_audio_format() == "wav"

    def test_validate_tts_audio_format_invalid_falls_back(self) -> None:
        """validate_tts_audio_format() returns 'pcm' for unsupported formats."""
        config = KokoroConfig(tts_audio_format="xyz")
        assert config.validate_tts_audio_format() == "pcm"

    def test_get_device_force_cpu(self) -> None:
        """get_device() returns 'cpu' immediately when force_cpu is True."""
        config = KokoroConfig(force_cpu=True)
        # force_cpu=True causes early return before any torch.cuda check
        assert config.get_device() == "cpu"

    def test_get_device_use_gpu_false(self) -> None:
        """get_device() returns 'cpu' immediately when use_gpu is False."""
        config = KokoroConfig(use_gpu=False, force_cpu=False)
        assert config.get_device() == "cpu"

    def test_get_device_cuda_available(self) -> None:
        """get_device() returns 'cuda' when CUDA reports available."""
        import torch

        config = KokoroConfig(use_gpu=True, force_cpu=False)
        with patch.object(torch.cuda, "is_available", return_value=True):
            device = config.get_device()
        assert device == "cuda"

    def test_get_device_cpu_fallback_no_cuda(self) -> None:
        """get_device() returns 'cpu' on systems without CUDA or MPS."""
        import torch

        config = KokoroConfig(use_gpu=True, force_cpu=False)
        with patch.object(torch.cuda, "is_available", return_value=False):
            device = config.get_device()
        # MPS is not available on Linux, so final fallback is "cpu"
        assert device in ("cpu", "mps")


class TestKokoroConfigPresets:
    """Tests for KokoroConfigPresets static factory methods."""

    def test_performance_preset(self) -> None:
        """performance() returns a KokoroConfig with GPU enabled."""
        config = KokoroConfigPresets.performance()
        assert isinstance(config, KokoroConfig)
        assert config.use_gpu is True
        assert config.warmup_on_init is True

    def test_quality_preset(self) -> None:
        """quality() returns a KokoroConfig with advanced normalisation enabled."""
        config = KokoroConfigPresets.quality()
        assert isinstance(config, KokoroConfig)
        assert config.advanced_normalization is True
        assert config.normalize_text is True

    def test_cpu_only_preset(self) -> None:
        """cpu_only() returns a KokoroConfig with GPU disabled."""
        config = KokoroConfigPresets.cpu_only()
        assert isinstance(config, KokoroConfig)
        assert config.use_gpu is False
        assert config.force_cpu is True

    def test_minimal_preset(self) -> None:
        """minimal() returns a KokoroConfig with minimal resource usage."""
        config = KokoroConfigPresets.minimal()
        assert isinstance(config, KokoroConfig)
        assert config.use_gpu is False
        assert config.warmup_on_init is False
        assert config.auto_download_model is False
