#!/usr/bin/env python3
"""
Audio Preprocessing Pipeline Example

This example demonstrates an audio processing pipeline with TTS.
Shows preprocessing, synthesis, and post-processing steps.
"""

import asyncio
import sys
import os
import numpy as np
from champion_signals import SignalBus

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from champi_tts import get_provider
from champi_tts.core.audio import save_audio, load_audio


class AudioPipeline:
    """Audio processing pipeline for TTS applications"""

    def __init__(self):
        """Initialize the audio pipeline"""
        self.provider = None
        self.signal_bus = SignalBus()

    async def initialize(self):
        """Initialize the pipeline components"""
        print("Initializing audio pipeline...")

        self.provider = get_provider("kokoro")
        await self.provider.initialize()

        # Setup signal handlers
        self.signal_bus.subscribe("text_preprocessed", self.on_text_preprocessed)
        self.signal_bus.subscribe("audio_synthesized", self.on_audio_synthesized)

        print("Pipeline initialized successfully")

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for TTS

        Args:
            text: Raw text input

        Returns:
            Preprocessed text
        """
        print("Preprocessing text...")

        # Remove extra whitespace
        text = " ".join(text.split())

        # Add pauses between sentences
        sentences = text.split(".")
        if len(sentences) > 1:
            text = ". ".join(sentences) + "."

        # Capitalize first letter
        text = text[0].upper() + text[1:]

        print(f"   Text preprocessed: {len(text)} characters")
        return text

    async def synthesize_text(self, text: str, voice: str = "af_bella") -> bytes:
        """
        Synthesize text to speech

        Args:
            text: Text to synthesize
            voice: Voice to use

        Returns:
            Audio data
        """
        print(f"Synthesizing text...")

        audio = await self.provider.synthesize(text, voice=voice)

        print(f"   Audio generated: {len(audio)} samples")

        # Notify synthesis complete
        await self.signal_bus.publish("audio_synthesized", audio.shape)

        return audio

    async def postprocess_audio(self, audio_data: bytes, sample_rate: int = 24000):
        """
        Postprocess audio data

        Args:
            audio_data: Raw audio data
            sample_rate: Sample rate

        Returns:
            Postprocessed audio data
        """
        print("Postprocessing audio...")

        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.float32)

        # Normalize audio (if needed)
        if np.abs(audio_array).max() > 1.0:
            audio_array = audio_array / np.abs(audio_array).max()
            print("   Audio normalized")

        # Apply volume adjustment (example: 0.9 for slightly quieter)
        audio_array = audio_array * 0.9
        print("   Audio volume adjusted")

        return audio_array.tobytes()

    async def process_full_pipeline(self, text: str, voice: str = "af_bella"):
        """
        Process text through the full pipeline

        Args:
            text: Input text
            voice: Voice to use
        """
        print("=" * 50)
        print("Processing text through pipeline")
        print("=" * 50)

        # Step 1: Text preprocessing
        await self.signal_bus.publish("text_preprocessed", text)
        processed_text = self.preprocess_text(text)

        # Step 2: Synthesis
        raw_audio = await self.synthesize_text(processed_text, voice)

        # Step 3: Audio postprocessing
        final_audio = await self.postprocess_audio(raw_audio, sample_rate=self.provider.config.sample_rate)

        print("\nPipeline processing complete!")
        return final_audio

    def on_text_preprocessed(self, text):
        """Signal handler for text preprocessing"""
        print(f"   Signal: Text preprocessed - {len(text)} chars")

    def on_audio_synthesized(self, audio_shape):
        """Signal handler for audio synthesis"""
        print(f"   Signal: Audio synthesized - shape {audio_shape}")

    async def save_audio(self, audio_data: bytes, filename: str, sample_rate: int = 24000):
        """Save audio data to file"""
        await save_audio(audio_data, filename, sample_rate)
        print(f"   Audio saved to: {filename}")

    async def cleanup(self):
        """Cleanup pipeline resources"""
        if self.provider:
            await self.provider.shutdown()
        if self.signal_bus:
            await self.signal_bus.cleanup()
        print("Pipeline cleanup complete")


async def main():
    """Main example"""

    print("=" * 50)
    print("Audio Preprocessing Pipeline Example")
    print("=" * 50)

    pipeline = AudioPipeline()

    try:
        # Initialize pipeline
        await pipeline.initialize()

        # Create test text
        test_text = """
        This is a sample text for the audio pipeline example.
        It demonstrates preprocessing, synthesis, and postprocessing.
        The pipeline processes text through multiple stages.
        """

        # Process through pipeline
        output_audio = await pipeline.process_full_pipeline(test_text, voice="af_bella")

        # Save output
        output_file = "pipeline_output.wav"
        await pipeline.save_audio(output_audio, output_file, sample_rate=pipeline.provider.config.sample_rate)

        print("\n" + "=" * 50)
        print("Example completed successfully!")
        print(f"Output file: {output_file}")
        print("=" * 50)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Always cleanup
        await pipeline.cleanup()


async def main_with_batch():
    """Example with batch processing"""

    print("=" * 50)
    print("Batch Processing Pipeline Example")
    print("=" * 50)

    pipeline = AudioPipeline()

    try:
        await pipeline.initialize()

        # Batch of texts
        texts = [
            "This is the first text for batch processing.",
            "The second text demonstrates batch capabilities.",
            "Batch processing allows handling multiple texts efficiently."
        ]

        # Process all texts
        for i, text in enumerate(texts, 1):
            print(f"\nProcessing batch item {i}/{len(texts)}...")

            output_audio = await pipeline.process_full_pipeline(text)
            output_file = f"batch_output_{i}.wav"

            await pipeline.save_audio(output_audio, output_file, sample_rate=pipeline.provider.config.sample_rate)

        print("\n" + "=" * 50)
        print("Batch processing completed!")
        print("=" * 50)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await pipeline.cleanup()


if __name__ == "__main__":
    try:
        print("\nExample 1: Single text processing")
        asyncio.run(main())

        print("\n\n" + "=" * 50)
        print("Example 2: Batch processing")
        print("=" * 50)
        asyncio.run(main_with_batch())
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)