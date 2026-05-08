"""
Shared fixtures for integration tests.
"""

import pytest

from tests.conftest import MockTTSConfig, MockTTSProvider


@pytest.fixture
def mock_provider():
    """Mock provider for integration tests."""
    config = MockTTSConfig()
    return MockTTSProvider(config)


@pytest.fixture
def initialized_provider(mock_provider):
    """Initialized mock provider."""
    # Mock initialization is a no-op
    mock_provider._initialized = True
    return mock_provider


@pytest.fixture
def temp_audio_file(tmp_path):
    """Create temporary audio file for testing."""
    import numpy as np
    from champi_tts.core.audio import save_audio

    # Create test audio
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))

    # Save to temp file
    audio_file = tmp_path / "test_audio.wav"
    save_audio(audio, audio_file, sample_rate=22050)

    return audio_file


@pytest.fixture
def test_document(tmp_path):
    """Create test document for reading tests."""
    doc_file = tmp_path / "test_document.txt"
    doc_file.write_text(
        "This is the first paragraph.\n\n"
        "This is the second paragraph.\n\n"
        "This is the third paragraph."
    )
    return doc_file
