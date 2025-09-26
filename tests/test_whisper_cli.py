#!/usr/bin/env python3
"""
Tests for cyllama.whisper.cli module.

This module tests the Whisper CLI functionality including argument parsing,
audio file processing, transcription, and various output formats.
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Import the CLI module
from cyllama.whisper import cli


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


@pytest.fixture
def temp_dir():
    """Fixture for temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestWhisperParams:
    """Test WhisperParams class."""

    def test_default_params(self):
        """Test default parameter values."""
        params = cli.WhisperParams()

        # Test basic parameters
        assert params.n_threads >= 1
        assert params.n_processors == 1
        assert params.offset_t_ms == 0
        assert params.offset_n == 0
        assert params.duration_ms == 0
        assert params.progress_step == 5
        assert params.max_context == -1
        assert params.max_len == 0
        assert params.best_of == 5
        assert params.beam_size == 5
        assert params.audio_ctx == 0

        # Test thresholds
        assert params.word_thold == 0.01
        assert params.entropy_thold == 2.40
        assert params.logprob_thold == -1.00
        assert params.no_speech_thold == 0.6
        assert params.grammar_penalty == 100.0
        assert params.temperature == 0.0
        assert params.temperature_inc == 0.2

        # Test boolean flags
        assert params.debug_mode is False
        assert params.translate is False
        assert params.detect_language is False
        assert params.diarize is False
        assert params.tinydiarize is False
        assert params.split_on_word is False
        assert params.no_fallback is False
        assert params.output_txt is False
        assert params.output_vtt is False
        assert params.output_srt is False
        assert params.output_wts is False
        assert params.output_csv is False
        assert params.output_jsn is False
        assert params.output_jsn_full is False
        assert params.output_lrc is False
        assert params.no_prints is False
        assert params.print_special is False
        assert params.print_colors is False
        assert params.print_confidence is False
        assert params.print_progress is False
        assert params.no_timestamps is False
        assert params.log_score is False
        assert params.use_gpu is True
        assert params.flash_attn is False
        assert params.suppress_nst is False

        # Test string parameters
        assert params.language == "en"
        assert params.prompt == ""
        assert params.model == "models/ggml-base.en.bin"
        assert params.grammar == ""
        assert params.grammar_rule == ""
        assert params.tdrz_speaker_turn == " [SPEAKER_TURN]"
        assert params.suppress_regex == ""
        assert params.openvino_encode_device == "CPU"
        assert params.dtw == ""

        # Test file lists
        assert isinstance(params.fname_inp, list)
        assert isinstance(params.fname_out, list)
        assert len(params.fname_inp) == 0
        assert len(params.fname_out) == 0

        # Test VAD parameters
        assert params.vad is False
        assert params.vad_model == ""
        assert params.vad_threshold == 0.5
        assert params.vad_min_speech_duration_ms == 250
        assert params.vad_min_silence_duration_ms == 100
        assert params.vad_max_speech_duration_s == float('inf')
        assert params.vad_speech_pad_ms == 30
        assert params.vad_samples_overlap == 0.1


class TestAudioProcessing:
    """Test audio file processing functions."""

    def test_load_wav_file(self, sample_audio_path):
        """Test WAV file loading."""
        samples, sample_rate = cli.load_wav_file(sample_audio_path)

        assert samples is not None
        assert len(samples) > 0
        assert sample_rate > 0
        assert samples.dtype.name == 'float32'

        # Check that samples are normalized to [-1, 1]
        assert samples.min() >= -1.0
        assert samples.max() <= 1.0

    def test_resample_audio(self):
        """Test audio resampling."""
        import numpy as np

        # Create test audio at 44100 Hz
        orig_sr = 44100
        target_sr = 16000
        duration = 1.0  # 1 second
        samples = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(orig_sr * duration))).astype(np.float32)

        resampled = cli.resample_audio(samples, orig_sr, target_sr)

        # Check that the length is approximately correct
        expected_length = int(len(samples) * target_sr / orig_sr)
        assert abs(len(resampled) - expected_length) <= 1
        assert resampled.dtype.name == 'float32'

    def test_resample_audio_same_rate(self):
        """Test resampling with same sample rate."""
        import numpy as np

        samples = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        resampled = cli.resample_audio(samples, 16000, 16000)

        assert len(resampled) == len(samples)
        assert np.allclose(resampled, samples)


class TestOutputFormatting:
    """Test output formatting functions."""

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        # Test basic formatting
        assert cli.format_timestamp(0) == "00:00,000"
        assert cli.format_timestamp(100) == "00:01,000"  # 100 * 10ms = 1000ms = 1s
        assert cli.format_timestamp(6000) == "01:00,000"  # 6000 * 10ms = 60s = 1min
        assert cli.format_timestamp(360000) == "01:00:00,000"  # 360000 * 10ms = 1hr

        # Test with hours always included
        assert cli.format_timestamp(0, always_include_hours=True) == "00:00:00,000"
        assert cli.format_timestamp(100, always_include_hours=True) == "00:00:01,000"

        # Test with different decimal marker
        assert cli.format_timestamp(100, decimal_marker='.') == "00:01.000"

    def test_escape_string_json(self):
        """Test JSON string escaping."""
        assert cli.escape_string_json('hello') == 'hello'
        assert cli.escape_string_json('hello "world"') == 'hello \\"world\\"'
        assert cli.escape_string_json('hello\\world') == 'hello\\\\world'
        assert cli.escape_string_json('hello\nworld') == 'hello\\nworld'
        assert cli.escape_string_json('hello\tworld') == 'hello\\tworld'

    def test_escape_string_csv(self):
        """Test CSV string escaping."""
        assert cli.escape_string_csv('hello') == 'hello'
        assert cli.escape_string_csv('hello "world"') == 'hello ""world""'


class TestArgumentParsing:
    """Test command-line argument parsing."""

    def test_parse_basic_arguments(self):
        """Test parsing basic arguments."""
        with patch('sys.argv', ['cli.py', '-f', 'test.wav', '-m', 'test.bin']):
            params = cli.parse_arguments()

            assert params.fname_inp == ['test.wav']
            assert params.model == 'test.bin'

    def test_parse_output_format_arguments(self):
        """Test parsing output format arguments."""
        with patch('sys.argv', ['cli.py', '-f', 'test.wav', '--output-srt', '--output-json']):
            params = cli.parse_arguments()

            assert params.output_srt is True
            assert params.output_jsn is True
            assert params.output_vtt is False

    def test_parse_processing_arguments(self):
        """Test parsing processing arguments."""
        with patch('sys.argv', ['cli.py', '-f', 'test.wav', '-t', '4', '--translate', '--language', 'es']):
            params = cli.parse_arguments()

            assert params.n_threads == 4
            assert params.translate is True
            assert params.language == 'es'

    def test_parse_threshold_arguments(self):
        """Test parsing threshold arguments."""
        with patch('sys.argv', ['cli.py', '-f', 'test.wav', '--temperature', '0.8', '--word-thold', '0.02']):
            params = cli.parse_arguments()

            assert params.temperature == 0.8
            assert params.word_thold == 0.02


class TestCLIIntegration:
    """Test CLI integration and end-to-end functionality."""

    def test_cli_help(self):
        """Test CLI help output."""
        project_root = Path(__file__).parent.parent
        cli_path = project_root / 'src' / 'cyllama' / 'whisper' / 'cli.py'

        result = subprocess.run([
            sys.executable, str(cli_path), '--help'
        ], capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': str(project_root / 'src')})

        assert result.returncode == 0
        assert 'Whisper CLI - Speech-to-text transcription' in result.stdout
        assert '--file' in result.stdout
        assert '--model' in result.stdout
        assert '--output-srt' in result.stdout

    def test_cli_missing_input_file(self):
        """Test CLI with missing input file."""
        project_root = Path(__file__).parent.parent
        cli_path = project_root / 'src' / 'cyllama' / 'whisper' / 'cli.py'

        result = subprocess.run([
            sys.executable, str(cli_path)
        ], capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': str(project_root / 'src')})

        assert result.returncode == 1
        assert 'No input files specified' in result.stderr

    def test_cli_nonexistent_model(self):
        """Test CLI with non-existent model file."""
        project_root = Path(__file__).parent.parent
        cli_path = project_root / 'src' / 'cyllama' / 'whisper' / 'cli.py'

        result = subprocess.run([
            sys.executable, str(cli_path),
            '-f', 'nonexistent.wav', '-m', 'nonexistent.bin'
        ], capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': str(project_root / 'src')})

        assert result.returncode == 1
        assert 'Model file not found' in result.stderr

    def test_cli_nonexistent_audio_file(self, whisper_model_path):
        """Test CLI with non-existent audio file."""
        project_root = Path(__file__).parent.parent
        cli_path = project_root / 'src' / 'cyllama' / 'whisper' / 'cli.py'
        model_path = project_root / whisper_model_path

        result = subprocess.run([
            sys.executable, str(cli_path),
            '-f', 'nonexistent.wav', '-m', str(model_path), '--no-prints'
        ], capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': str(project_root / 'src')})

        assert result.returncode == 0  # Should continue processing other files
        assert 'Input file not found' in result.stderr

    @pytest.mark.slow
    def test_cli_basic_transcription(self, sample_audio_path, whisper_model_path, temp_dir):
        """Test basic CLI transcription functionality."""
        # Change to temp directory to avoid cluttering
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)

            # Use absolute paths from the project root
            project_root = Path(__file__).parent.parent
            audio_path = project_root / sample_audio_path
            model_path = project_root / whisper_model_path
            cli_path = project_root / 'src' / 'cyllama' / 'whisper' / 'cli.py'

            result = subprocess.run([
                sys.executable, str(cli_path),
                '-f', str(audio_path),
                '-m', str(model_path),
                '--no-prints'
            ], capture_output=True, text=True, timeout=120,
            env={**os.environ, 'PYTHONPATH': str(project_root / 'src')})

            assert result.returncode == 0
            # Check that output contains transcribed text
            assert len(result.stdout.strip()) > 0
            # JFK audio should contain this text
            output_lower = result.stdout.lower()
            assert any(word in output_lower for word in ['americans', 'country', 'ask'])

        finally:
            os.chdir(old_cwd)

    @pytest.mark.slow
    def test_cli_srt_output(self, sample_audio_path, whisper_model_path, temp_dir):
        """Test CLI SRT output generation."""
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)

            # Use absolute paths from the project root
            project_root = Path(__file__).parent.parent
            audio_path = project_root / sample_audio_path
            model_path = project_root / whisper_model_path
            cli_path = project_root / 'src' / 'cyllama' / 'whisper' / 'cli.py'

            result = subprocess.run([
                sys.executable, str(cli_path),
                '-f', str(audio_path),
                '-m', str(model_path),
                '--output-srt', '--no-prints'
            ], capture_output=True, text=True, timeout=120,
            env={**os.environ, 'PYTHONPATH': str(project_root / 'src')})

            assert result.returncode == 0

            # Check that SRT file was created
            srt_file = Path(temp_dir) / 'jfk.srt'
            assert srt_file.exists()

            # Check SRT format
            content = srt_file.read_text()
            assert '1\n' in content  # Subtitle number
            assert '-->' in content  # Timestamp separator
            assert len(content.strip()) > 0

        finally:
            os.chdir(old_cwd)

    @pytest.mark.slow
    def test_cli_json_output(self, sample_audio_path, whisper_model_path, temp_dir):
        """Test CLI JSON output generation."""
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)

            # Use absolute paths from the project root
            project_root = Path(__file__).parent.parent
            audio_path = project_root / sample_audio_path
            model_path = project_root / whisper_model_path
            cli_path = project_root / 'src' / 'cyllama' / 'whisper' / 'cli.py'

            result = subprocess.run([
                sys.executable, str(cli_path),
                '-f', str(audio_path),
                '-m', str(model_path),
                '--output-json', '--no-prints'
            ], capture_output=True, text=True, timeout=120,
            env={**os.environ, 'PYTHONPATH': str(project_root / 'src')})

            assert result.returncode == 0

            # Check that JSON file was created
            json_file = Path(temp_dir) / 'jfk.json'
            assert json_file.exists()

            # Check JSON format
            content = json_file.read_text()
            data = json.loads(content)

            assert 'text' in data
            assert 'segments' in data
            assert isinstance(data['segments'], list)
            assert len(data['segments']) > 0

            # Check first segment structure
            segment = data['segments'][0]
            assert 'id' in segment
            assert 'start' in segment
            assert 'end' in segment
            assert 'text' in segment
            assert isinstance(segment['start'], (int, float))
            assert isinstance(segment['end'], (int, float))
            assert len(segment['text'].strip()) > 0

        finally:
            os.chdir(old_cwd)

    @pytest.mark.slow
    def test_cli_multiple_formats(self, sample_audio_path, whisper_model_path, temp_dir):
        """Test CLI with multiple output formats."""
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)

            # Use absolute paths from the project root
            project_root = Path(__file__).parent.parent
            audio_path = project_root / sample_audio_path
            model_path = project_root / whisper_model_path
            cli_path = project_root / 'src' / 'cyllama' / 'whisper' / 'cli.py'

            result = subprocess.run([
                sys.executable, str(cli_path),
                '-f', str(audio_path),
                '-m', str(model_path),
                '--output-srt', '--output-vtt', '--output-csv', '--no-prints'
            ], capture_output=True, text=True, timeout=120,
            env={**os.environ, 'PYTHONPATH': str(project_root / 'src')})

            assert result.returncode == 0

            # Check that all files were created
            files = ['jfk.srt', 'jfk.vtt', 'jfk.csv']
            for filename in files:
                filepath = Path(temp_dir) / filename
                assert filepath.exists(), f"{filename} was not created"
                assert filepath.stat().st_size > 0, f"{filename} is empty"

            # Verify VTT format
            vtt_content = (Path(temp_dir) / 'jfk.vtt').read_text()
            assert vtt_content.startswith('WEBVTT')

            # Verify CSV format
            csv_content = (Path(temp_dir) / 'jfk.csv').read_text()
            assert 'start,end,text' in csv_content

        finally:
            os.chdir(old_cwd)


class TestWhisperParamsConversion:
    """Test conversion from CLI params to Whisper params."""

    def test_create_whisper_params_from_args(self):
        """Test conversion to WhisperFullParams."""
        params = cli.WhisperParams()
        params.n_threads = 4
        params.offset_t_ms = 1000
        params.duration_ms = 5000
        params.translate = True
        params.no_timestamps = True
        params.print_special = True
        params.print_progress = True
        params.temperature = 0.8
        params.language = "es"

        whisper_params = cli.create_whisper_params_from_args(params)

        assert whisper_params.n_threads == 4
        assert whisper_params.offset_ms == 1000
        assert whisper_params.duration_ms == 5000
        assert whisper_params.translate is True
        assert whisper_params.no_timestamps is True
        assert whisper_params.print_special is True
        assert whisper_params.print_progress is True
        assert abs(whisper_params.temperature - 0.8) < 1e-6
        assert whisper_params.language == "es"


@pytest.mark.slow
class TestRegressionCases:
    """Test regression cases and edge conditions."""

    def test_empty_audio_handling(self, whisper_model_path, temp_dir):
        """Test handling of very short or empty audio."""
        # This test would require creating a minimal WAV file
        # Skip for now as it requires more complex setup
        pytest.skip("Requires creation of minimal test audio files")

    def test_different_sample_rates(self):
        """Test handling of different sample rates."""
        # This test would require creating audio files with different sample rates
        # Skip for now as it requires more complex setup
        pytest.skip("Requires creation of test audio files with different sample rates")

    def test_unicode_handling_in_output(self):
        """Test proper Unicode handling in output formats."""
        # Test with mock data containing Unicode characters
        test_text = "Hello 世界 français español"
        escaped_json = cli.escape_string_json(test_text)
        escaped_csv = cli.escape_string_csv(test_text)

        # Should not raise encoding errors
        assert isinstance(escaped_json, str)
        assert isinstance(escaped_csv, str)


if __name__ == '__main__':
    pytest.main([__file__])