#!/usr/bin/env python3
"""
Test the fixed TTS implementation
"""

import sys
import os


def test_tts_generation():
    """Test TTS generation with 'Hello world' to see if we get clean output"""
    from cyllama.tts import TTSGenerator

    # Use the existing model files
    ttc_model = "models/tts.gguf"
    cts_model = "models/WavTokenizer-Large-75-F16.gguf"

    if not os.path.exists(ttc_model):
        print(f"Model file {ttc_model} not found. Please ensure you have the TTS models.")
        return False

    # Create TTS generator with the fixed implementation
    tts = TTSGenerator(
        ttc_model_path=ttc_model,
        cts_model_path=cts_model,
        n_ctx=8192,  # Larger context to accommodate the prompt
        n_batch=8192,  # Much larger batch size to handle the long prompt
        ngl=99,
        n_predict=1000,  # Limited tokens for testing
        use_guide_tokens=True
    )

    # Test with simple text
    test_text = "Hello world"
    output_file = "/tmp/_output.wav"

    print(f"Testing TTS generation with: '{test_text}'")
    success = tts.generate(test_text, output_file)
    assert success
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file)
