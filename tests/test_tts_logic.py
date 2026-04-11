"""Tests for TTS text processing and Cython optimizations."""

import os
import tempfile


def test_text_processing():
    """Text preprocessing should produce non-empty strings for all supported versions."""
    from cyllama.llama.tts import process_text

    test_cases = [
        ("Hello world", "0.2"),
        ("Hello world 123", "0.2"),
        ("Test with numbers 42", "0.3"),
    ]

    for text, version in test_cases:
        processed = process_text(text, version)
        assert isinstance(processed, str), f"process_text({text!r}, {version!r}) returned non-string"
        assert processed, f"process_text({text!r}, {version!r}) returned empty string"


def test_cython_optimizations():
    """Hann window and WAV saving helpers should produce valid output."""
    from cyllama.llama.tts import save_wav16
    from cyllama.llama import llama_cpp as cy

    # Hann window: known values at the endpoints are 0.0.
    window = cy.fill_hann_window(10, True)
    assert len(window) == 10
    assert abs(window[0]) < 1e-6, f"Hann window[0] should be 0, got {window[0]}"

    # WAV save: file should exist, be non-empty, and start with the RIFF header.
    test_data = [0.1, 0.2, -0.1, -0.2] * 100
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        success = save_wav16(tmp_name, test_data, 24000)
        assert success, "save_wav16 returned False"
        assert os.path.getsize(tmp_name) > 0, "save_wav16 produced empty file"
        with open(tmp_name, "rb") as f:
            header = f.read(4)
        assert header == b"RIFF", f"WAV file has bad header: {header!r}"
    finally:
        os.unlink(tmp_name)
