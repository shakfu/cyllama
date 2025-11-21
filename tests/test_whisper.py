#!/usr/bin/env python3
"""
Tests for cyllama.whisper module.

This module tests the Whisper wrapper functionality including model loading,
parameter configuration, and basic transcription operations.
Tests replicate what `make test-whisper` does but via the Python wrapper.
"""

import os
import numpy as np
import wave
import struct
import tempfile
from pathlib import Path
import pytest

# Import the whisper module
from cyllama.whisper import whisper_cpp as wh


@pytest.fixture
def whisper_model_path():
    """Fixture for whisper model path."""
    model_path = Path("models/ggml-base.en.bin")
    if not model_path.exists():
        pytest.skip(f"Whisper model not found at {model_path}")
    return str(model_path)


@pytest.fixture
def sample_audio_path():
    """Fixture for sample audio path."""
    audio_path = Path("samples/jfk.wav")
    if not audio_path.exists():
        pytest.skip(f"Sample audio not found at {audio_path}")
    return str(audio_path)


def load_wav_file(filepath):
    """Load a WAV file and return samples as float32 array."""
    with wave.open(filepath, 'rb') as wav_file:
        frames = wav_file.readframes(-1)
        sound_info = wav_file.getparams()

        # Convert to float32
        if sound_info.sampwidth == 1:
            fmt = f'{len(frames)}B'
            samples = struct.unpack(fmt, frames)
            samples = [(s - 128) / 128.0 for s in samples]
        elif sound_info.sampwidth == 2:
            fmt = f'{len(frames) // 2}h'
            samples = struct.unpack(fmt, frames)
            samples = [s / 32768.0 for s in samples]
        else:
            raise ValueError(f"Unsupported sample width: {sound_info.sampwidth}")

        return np.array(samples, dtype=np.float32), sound_info.framerate


def test_whisper_constants():
    """Test whisper constants are accessible."""
    assert wh.WHISPER.SAMPLE_RATE == 16000
    assert wh.WHISPER.N_FFT > 0
    assert wh.WHISPER.HOP_LENGTH > 0
    assert wh.WHISPER.CHUNK_SIZE > 0


def test_whisper_sampling_strategy():
    """Test whisper sampling strategy constants."""
    assert hasattr(wh.WhisperSamplingStrategy, 'GREEDY')
    assert hasattr(wh.WhisperSamplingStrategy, 'BEAM_SEARCH')


def test_whisper_aheads_preset():
    """Test whisper attention heads presets."""
    assert hasattr(wh.WhisperAheadsPreset, 'NONE')
    assert hasattr(wh.WhisperAheadsPreset, 'BASE_EN')
    assert hasattr(wh.WhisperAheadsPreset, 'BASE')


def test_whisper_version():
    """Test getting whisper version."""
    version = wh.version()
    assert isinstance(version, str)
    assert len(version) > 0


def test_whisper_system_info():
    """Test getting system info."""
    info = wh.print_system_info()
    assert isinstance(info, str)
    assert len(info) > 0
    assert 'WHISPER' in info


def test_whisper_context_params():
    """Test whisper context parameters."""
    params = wh.WhisperContextParams()

    # Test default values and setters
    assert isinstance(params.use_gpu, bool)
    assert isinstance(params.flash_attn, bool)
    assert isinstance(params.gpu_device, int)
    assert isinstance(params.dtw_token_timestamps, bool)

    # Test setting values
    params.use_gpu = True
    assert params.use_gpu is True

    params.gpu_device = 1
    assert params.gpu_device == 1


def test_whisper_vad_params():
    """Test whisper VAD parameters."""
    params = wh.WhisperVadParams()

    # Test threshold parameter
    assert isinstance(params.threshold, float)

    params.threshold = 0.5
    assert abs(params.threshold - 0.5) < 1e-6


def test_whisper_full_params():
    """Test whisper full processing parameters."""
    params = wh.WhisperFullParams()

    # Test basic properties
    assert isinstance(params.strategy, int)
    assert isinstance(params.n_threads, int)
    assert isinstance(params.n_max_text_ctx, int)
    assert isinstance(params.offset_ms, int)
    assert isinstance(params.duration_ms, int)
    assert isinstance(params.translate, bool)
    assert isinstance(params.no_context, bool)
    assert isinstance(params.no_timestamps, bool)
    assert isinstance(params.single_segment, bool)
    assert isinstance(params.print_special, bool)
    assert isinstance(params.print_progress, bool)
    assert isinstance(params.print_realtime, bool)
    assert isinstance(params.print_timestamps, bool)
    assert isinstance(params.token_timestamps, bool)
    assert isinstance(params.temperature, float)

    # Test setters
    params.n_threads = 2
    assert params.n_threads == 2

    params.translate = True
    assert params.translate is True

    params.temperature = 0.8
    assert abs(params.temperature - 0.8) < 1e-6

    # Test language setting
    params.language = "en"
    assert params.language == "en"

    params.language = None
    assert params.language is None


def test_whisper_context_initialization(whisper_model_path):
    """Test whisper context initialization."""
    ctx = wh.WhisperContext(whisper_model_path)

    # Test that context was created successfully
    assert ctx is not None

    # Test version method
    version = ctx.version()
    assert isinstance(version, str)
    assert len(version) > 0

    # Test system info
    info = ctx.system_info()
    assert isinstance(info, str)
    assert len(info) > 0


def test_whisper_context_model_properties(whisper_model_path):
    """Test whisper context model properties."""
    ctx = wh.WhisperContext(whisper_model_path)

    # Test vocabulary size
    n_vocab = ctx.n_vocab()
    assert n_vocab > 0

    # Test context sizes
    n_text_ctx = ctx.n_text_ctx()
    assert n_text_ctx > 0

    n_audio_ctx = ctx.n_audio_ctx()
    assert n_audio_ctx > 0

    # Test multilingual capability
    is_multilingual = ctx.is_multilingual()
    assert isinstance(is_multilingual, bool)

    # Test model architecture details
    assert ctx.model_n_vocab() > 0
    assert ctx.model_n_audio_ctx() > 0
    assert ctx.model_n_audio_state() > 0
    assert ctx.model_n_audio_head() > 0
    assert ctx.model_n_audio_layer() > 0
    assert ctx.model_n_text_ctx() > 0
    assert ctx.model_n_text_state() > 0
    assert ctx.model_n_text_head() > 0
    assert ctx.model_n_text_layer() > 0
    assert ctx.model_n_mels() > 0

    # Test model type
    model_type = ctx.model_type()
    assert model_type >= 0

    model_type_str = ctx.model_type_readable()
    assert isinstance(model_type_str, str)
    assert len(model_type_str) > 0


def test_whisper_tokens(whisper_model_path):
    """Test whisper token operations."""
    ctx = wh.WhisperContext(whisper_model_path)

    # Test special tokens
    eot_token = ctx.token_eot()
    assert eot_token >= 0

    sot_token = ctx.token_sot()
    assert sot_token >= 0

    # Test token to string conversion
    token_str = ctx.token_to_str(eot_token)
    assert isinstance(token_str, str)

    # Test tokenization
    text = "hello world"
    tokens = ctx.tokenize(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0

    # Test token count
    token_count = ctx.token_count(text)
    assert token_count > 0
    assert token_count == len(tokens)


def test_whisper_language_operations(whisper_model_path):
    """Test whisper language operations."""
    ctx = wh.WhisperContext(whisper_model_path)

    # Test language operations
    lang_max_id = ctx.lang_max_id()
    assert lang_max_id > 0

    # Test English language
    en_id = ctx.lang_id("en")
    assert en_id >= 0

    lang_str = ctx.lang_str(en_id)
    assert lang_str == "en"

    lang_str_full = ctx.lang_str_full(en_id)
    assert isinstance(lang_str_full, str)
    assert len(lang_str_full) > 0


def test_whisper_module_functions():
    """Test module-level whisper functions."""
    # Test version
    version = wh.version()
    assert isinstance(version, str)
    assert len(version) > 0

    # Test system info
    info = wh.print_system_info()
    assert isinstance(info, str)
    assert len(info) > 0

    # Test language functions
    lang_max_id = wh.lang_max_id()
    assert lang_max_id > 0

    en_id = wh.lang_id("en")
    assert en_id >= 0

    lang_str = wh.lang_str(en_id)
    assert lang_str == "en"

    lang_str_full = wh.lang_str_full(en_id)
    assert isinstance(lang_str_full, str)
    assert len(lang_str_full) > 0


def test_whisper_context_invalid_model():
    """Test whisper context with invalid model path."""
    with pytest.raises(RuntimeError):
        wh.WhisperContext("/nonexistent/model.bin")


def test_whisper_audio_loading_basic(sample_audio_path):
    """Test basic audio file loading."""
    samples, sample_rate = load_wav_file(sample_audio_path)

    # Verify audio properties
    assert isinstance(samples, np.ndarray)
    assert samples.dtype == np.float32
    assert len(samples) > 0
    assert sample_rate > 0

    # Audio should be around 11 seconds (JFK sample)
    duration = len(samples) / sample_rate
    assert 10 < duration < 12  # JFK sample is about 11 seconds


def test_whisper_context_with_params(whisper_model_path):
    """Test whisper context creation with custom parameters."""
    params = wh.WhisperContextParams()
    params.use_gpu = True
    params.gpu_device = 0

    ctx = wh.WhisperContext(whisper_model_path, params)
    assert ctx is not None

    # Verify context works
    version = ctx.version()
    assert isinstance(version, str)


def test_whisper_full_params_comprehensive():
    """Test comprehensive whisper full parameters."""
    params = wh.WhisperFullParams()

    # Test all parameter types and ranges
    params.strategy = wh.WhisperSamplingStrategy.GREEDY
    assert params.strategy == wh.WhisperSamplingStrategy.GREEDY

    params.n_threads = 4
    assert params.n_threads == 4

    params.offset_ms = 1000
    assert params.offset_ms == 1000

    params.duration_ms = 5000
    assert params.duration_ms == 5000

    params.translate = True
    assert params.translate is True

    params.no_context = True
    assert params.no_context is True

    params.single_segment = True
    assert params.single_segment is True

    params.print_timestamps = True
    assert params.print_timestamps is True

    params.token_timestamps = True
    assert params.token_timestamps is True


def test_whisper_timing_functions(whisper_model_path):
    """Test whisper timing functions."""
    ctx = wh.WhisperContext(whisper_model_path)

    # These should not raise exceptions
    ctx.reset_timings()
    ctx.print_timings()


# Integration test that mimics the actual whisper CLI usage
# @pytest.mark.integration
def test_whisper_basic_transcription_setup(whisper_model_path, sample_audio_path):
    """Test basic whisper transcription setup (without actual transcription due to missing full() method)."""
    # Load audio
    samples, sample_rate = load_wav_file(sample_audio_path)

    # Verify sample rate matches Whisper's expectation
    if sample_rate != wh.WHISPER.SAMPLE_RATE:
        # In a real implementation, we'd resample here
        print(f"Warning: Sample rate {sample_rate} != {wh.WHISPER.SAMPLE_RATE}")

    # Create context
    ctx = wh.WhisperContext(whisper_model_path)

    # Create parameters
    params = wh.WhisperFullParams()
    params.language = "en"
    params.print_timestamps = True
    params.print_progress = True
    params.n_threads = 4

    # Verify context is ready
    assert ctx.n_vocab() > 50000  # Base model should have ~51k vocab
    assert ctx.model_type_readable() in ["base", "base.en"]

    # Test that we can get language info
    en_id = ctx.lang_id("en")
    assert en_id >= 0

    print(f"Model loaded: {ctx.model_type_readable()}")
    print(f"Vocabulary size: {ctx.n_vocab()}")
    print(f"Audio length: {len(samples) / sample_rate:.1f}s")
    print(f"Sample rate: {sample_rate}Hz")

    # Note: Actual transcription would require implementing the full() method
    # or finding an alternative approach in the whisper wrapper


def test_whisper_token_data():
    """Test whisper token data structure."""
    token_data = wh.WhisperTokenData()

    # Test that properties are accessible (they return default values)
    assert hasattr(token_data, 'id')
    assert hasattr(token_data, 'tid')
    assert hasattr(token_data, 'p')
    assert hasattr(token_data, 'plog')
    assert hasattr(token_data, 'pt')
    assert hasattr(token_data, 'ptsum')
    assert hasattr(token_data, 't0')
    assert hasattr(token_data, 't1')
    assert hasattr(token_data, 't_dtw')
    assert hasattr(token_data, 'vlen')


def test_whisper_state(whisper_model_path):
    """Test whisper state management."""
    ctx = wh.WhisperContext(whisper_model_path)

    # Create state
    state = wh.WhisperState(ctx)
    assert state is not None

    # State should be properly initialized
    # (No direct methods to test on state, but it should not raise errors)


# Test edge cases and error conditions
def test_whisper_tokenize_edge_cases(whisper_model_path):
    """Test whisper tokenization edge cases."""
    ctx = wh.WhisperContext(whisper_model_path)

    # Test empty string
    tokens = ctx.tokenize("")
    assert isinstance(tokens, list)

    # Test very long string (should handle gracefully)
    long_text = "hello " * 1000
    try:
        tokens = ctx.tokenize(long_text, max_tokens=2000)
        assert isinstance(tokens, list)
    except RuntimeError:
        # Expected if text is too long for max_tokens
        pass

    # Test unicode text
    unicode_text = "Hello 世界"
    tokens = ctx.tokenize(unicode_text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_whisper_invalid_language_id(whisper_model_path):
    """Test whisper with invalid language IDs."""
    ctx = wh.WhisperContext(whisper_model_path)

    # Test invalid language ID
    result = ctx.lang_str(9999)
    assert result is None

    result = ctx.lang_str_full(9999)
    assert result is None


def test_whisper_audio_processing_requirements():
    """Test that we understand whisper audio requirements."""
    # Test that we know the expected format
    assert wh.WHISPER.SAMPLE_RATE == 16000

    # Create a simple sine wave for testing
    duration = 1.0  # 1 second
    sample_rate = wh.WHISPER.SAMPLE_RATE
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    frequency = 440  # A note
    samples = np.sin(frequency * 2 * np.pi * t).astype(np.float32)

    # Test that we can create properly formatted audio
    assert samples.dtype == np.float32
    assert len(samples) == sample_rate
    assert -1.0 <= samples.min() <= samples.max() <= 1.0


def test_whisper_language_setter_fixed():
    """Test that the language setter bugs have been fixed."""
    params = wh.WhisperFullParams()

    # Test 1: Setting string values works correctly
    params.language = "en"
    assert params.language == "en"

    # Test 2: Setting None value works correctly
    params.language = None
    assert params.language is None

    # Test 3: Setting different string values
    params.language = "es"
    assert params.language == "es"

    params.language = "fr"
    assert params.language == "fr"

    # Test 4: Back to None
    params.language = None
    assert params.language is None

    # The bugs were fixed in whisper_cpp.pyx:
    # 1. Added _language_bytes member to keep bytes object alive
    # 2. Fixed None handling by checking value before encoding
    # 3. Use proper const char* casting from bytes object