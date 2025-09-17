#!/usr/bin/env python3
"""
Test the TTS text processing and tokenization logic
"""

import sys
sys.path.insert(0, 'src')

def test_text_processing():
    """Test the text processing functions"""
    try:
        from cyllama.tts import process_text, prepare_guide_tokens

        # Test text processing
        test_cases = [
            ("Hello world", "0.2"),
            ("Hello world 123", "0.2"),
            ("Test with numbers 42", "0.3"),
        ]

        print("Testing text processing:")
        for text, version in test_cases:
            processed = process_text(text, version)
            print(f"  '{text}' (v{version}) -> '{processed}'")

        return True

    except Exception as e:
        print(f"Text processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cython_optimizations():
    """Test that Cython optimizations are working"""
    try:
        from cyllama.tts import USE_CYTHON_OPTIMIZATIONS
        print(f"Cython optimizations enabled: {USE_CYTHON_OPTIMIZATIONS}")

        if USE_CYTHON_OPTIMIZATIONS:
            from cyllama.tts import fill_hann_window, save_wav16

            # Test Hann window
            window = fill_hann_window(10, True)
            print(f"Hann window test: {len(window)} samples, first value: {window[0]:.6f}")

            # Test WAV saving with dummy data
            import tempfile
            import os
            test_data = [0.1, 0.2, -0.1, -0.2] * 100
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                success = save_wav16(tmp.name, test_data, 24000)
                file_size = os.path.getsize(tmp.name)
                print(f"WAV save test: {'SUCCESS' if success else 'FAILED'} ({file_size} bytes)")
                os.unlink(tmp.name)

        return True

    except Exception as e:
        print(f"Cython optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== TTS Logic Tests ===")

    success1 = test_text_processing()
    success2 = test_cython_optimizations()

    if success1 and success2:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)