"""Signals system for Kokoro TTS service using champi-signals."""
from blinker import Signal
from champi_signals import BaseSignalManager, TTSEventTypes


class TTSSignalManager(BaseSignalManager):
    """TTS Signal Manager using champi-signals library"""

    def __init__(self):
        super().__init__()
        # Setup signals using TTS event types
        self.setup_custom_signals(
            {
                "lifecycle": TTSEventTypes,
                "model": TTSEventTypes,
                "processing": TTSEventTypes,
                "telemetry": TTSEventTypes,
            }
        )

    def get_signal_by_signal_type(self, signal_type: str) -> Signal:
        """Return signal object for given signal_type"""
        return self.signals[signal_type]
