#!/usr/bin/env python3
"""
Tests for cyllama.cli module.

This module tests the CLI functionality including argument parsing,
file operations, prompt processing, and basic CLI operations.
"""

import argparse
import os
import signal
import sys
import tempfile
from pathlib import Path
import pytest
from pytest_mock import MockerFixture

# Import the CLI module
from cyllama.llama.cli import LlamaCLI


@pytest.fixture
def cli():
    """Fixture for LlamaCLI instance."""
    return LlamaCLI()


@pytest.fixture
def test_model_path():
    """Fixture for test model path."""
    return "/fake/path/model.gguf"


def test_cli_initialization():
    """Test CLI initialization."""
    cli = LlamaCLI()
    
    # Check initial state
    assert cli.model is None
    assert cli.ctx is None
    assert cli.sampler is None
    assert cli.vocab is None
    assert cli.is_interacting is False
    assert cli.need_insert_eot is False
    assert cli.t_main_start == 0
    assert cli.n_decode == 0
    
def test_file_exists(cli: LlamaCLI):
    """Test file existence checking."""
    # Test with existing file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"test content")
        tmp_path = tmp.name
    
    try:
        assert cli._file_exists(tmp_path) is True
        assert cli._file_exists("/nonexistent/path/file.txt") is False
    finally:
        os.unlink(tmp_path)
    
def test_file_is_empty(cli: LlamaCLI):
    """Test file empty checking."""
    # Test with empty file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        empty_path = tmp.name
    
    # Test with non-empty file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"content")
        non_empty_path = tmp.name
    
    try:
        assert cli._file_is_empty(empty_path) is True
        assert cli._file_is_empty(non_empty_path) is False
        assert cli._file_is_empty("/nonexistent/path/file.txt") is True
    finally:
        os.unlink(empty_path)
        os.unlink(non_empty_path)


def test_print_usage(cli: LlamaCLI, mocker: MockerFixture):
    """Test usage printing."""
    mock_print = mocker.patch('builtins.print')
    cli._print_usage("test_prog")
    mock_print.assert_called()
    # Check that usage information is printed
    calls = mock_print.call_args_list
    usage_text = " ".join(str(call) for call in calls)
    assert "text generation" in usage_text
    assert "chat" in usage_text
    
def test_parse_args_basic(cli: LlamaCLI, test_model_path, mocker: MockerFixture):
    """Test basic argument parsing."""
    # Test with minimal required arguments
    test_args = ["-m", test_model_path]
    
    mocker.patch('sys.argv', ['test_cli'] + test_args)
    args = cli._parse_args()
    
    assert args.model == test_model_path
    assert args.ctx_size == 4096  # default
    assert args.batch_size == 2048  # default
    assert args.threads == 4  # default
    assert args.temp == 0.8  # default
    assert args.n_predict == -1  # default
    
def test_parse_args_with_options(cli: LlamaCLI, test_model_path, mocker: MockerFixture):
    """Test argument parsing with various options."""
    test_args = [
        "-m", test_model_path,
        "-c", "2048",
        "-b", "1024",
        "-t", "8",
        "--temp", "0.5",
        "-n", "100",
        "--top-k", "20",
        "--top-p", "0.9",
        "--prompt", "Hello world",
        "--interactive",
        "--no-mmap",
        "--mlock"
    ]
    
    mocker.patch('sys.argv', ['test_cli'] + test_args)
    args = cli._parse_args()
    
    assert args.model == test_model_path
    assert args.ctx_size == 2048
    assert args.batch_size == 1024
    assert args.threads == 8
    assert args.temp == 0.5
    assert args.n_predict == 100
    assert args.top_k == 20
    assert args.top_p == 0.9
    assert args.prompt == "Hello world"
    assert args.interactive is True
    assert args.no_mmap is True
    assert args.mlock is True
    
def test_parse_args_file_prompt(cli: LlamaCLI, test_model_path, mocker: MockerFixture):
    """Test argument parsing with file prompt."""
    # Create a temporary file with prompt content
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
        tmp.write("This is a test prompt from file.")
        prompt_file = tmp.name
    
    test_args = ["-m", test_model_path, "-f", prompt_file]
    
    try:
        mocker.patch('sys.argv', ['test_cli'] + test_args)
        args = cli._parse_args()
        
        assert args.model == test_model_path
        assert args.file == prompt_file
    finally:
        os.unlink(prompt_file)
    
def test_load_prompt_from_args(cli: LlamaCLI):
    """Test loading prompt from command line arguments."""
    test_prompt = "This is a test prompt"
    args = argparse.Namespace(prompt=test_prompt, file="", escape=False)
    
    result = cli._load_prompt(args)
    assert result == test_prompt


def test_load_prompt_from_file(cli: LlamaCLI):
    """Test loading prompt from file."""
    test_content = "This is a test prompt from file\nwith multiple lines."
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
        tmp.write(test_content)
        prompt_file = tmp.name
    
    args = argparse.Namespace(prompt="", file=prompt_file, escape=False)
    
    try:
        result = cli._load_prompt(args)
        assert result == test_content
    finally:
        os.unlink(prompt_file)
    
def test_load_prompt_file_not_exists(cli: LlamaCLI, mocker: MockerFixture):
    """Test loading prompt from non-existent file."""
    args = argparse.Namespace(prompt="", file="/nonexistent/file.txt", escape=False)
    
    mocker.patch('builtins.print')
    mock_exit = mocker.patch('sys.exit', side_effect=SystemExit)
    
    with pytest.raises(SystemExit):
        cli._load_prompt(args)
    mock_exit.assert_called_with(1)


def test_load_prompt_with_escape_sequences(cli: LlamaCLI):
    """Test loading prompt with escape sequence processing."""
    test_prompt = "Line1\\nLine2\\tTabbed\\'Quoted\\\"Double\\"
    expected = "Line1\nLine2\tTabbed'Quoted\"Double\\"
    
    args = argparse.Namespace(prompt=test_prompt, file="", escape=True)
    
    result = cli._load_prompt(args)
    assert result == expected
    
def test_sigint_handler_not_interacting(cli: LlamaCLI, mocker: MockerFixture):
    """Test SIGINT handler when not in interactive mode."""
    cli.is_interacting = False
    cli.interactive = True
    
    mock_exit = mocker.patch('sys.exit')
    cli._sigint_handler(signal.SIGINT, None)
    
    assert cli.is_interacting is True
    assert cli.need_insert_eot is True
    mock_exit.assert_not_called()


def test_sigint_handler_interacting(cli: LlamaCLI, mocker: MockerFixture):
    """Test SIGINT handler when already interacting."""
    cli.is_interacting = True
    
    mock_print = mocker.patch('builtins.print')
    mock_perf = mocker.patch.object(cli, '_print_performance')
    mock_exit = mocker.patch('sys.exit')
    
    cli._sigint_handler(signal.SIGINT, None)
    
    mock_perf.assert_called_once()
    # Check that both print calls were made
    print_calls = mock_print.call_args_list
    assert len(print_calls) == 2
    assert print_calls[0][0][0] == "\n"
    assert print_calls[1][0][0] == "Interrupted by user"
    mock_exit.assert_called_with(130)
    
def test_print_performance_no_timing(cli: LlamaCLI, mocker: MockerFixture):
    """Test performance printing when no timing data available."""
    cli.t_main_start = 0
    
    mock_print = mocker.patch('builtins.print')
    cli._print_performance()
    # Should not print timing info when t_main_start is 0
    mock_print.assert_not_called()


def test_print_performance_with_timing(cli: LlamaCLI, mocker: MockerFixture):
    """Test performance printing with timing data."""
    cli.t_main_start = 1000000  # 1 second in microseconds
    cli.n_decode = 50
    
    # Mock the cyllama module
    mock_cy = mocker.patch('cyllama.llama.cli.cy')
    mock_print = mocker.patch('builtins.print')
    
    mock_cy.ggml_time_us.return_value = 2000000  # 2 seconds total
    mock_sampler = mocker.Mock()
    mock_ctx = mocker.Mock()
    cli.sampler = mock_sampler
    cli.ctx = mock_ctx
    
    cli._print_performance()
    
    # Check that performance data was printed
    mock_print.assert_called()
    mock_sampler.print_perf_data.assert_called_once()
    mock_ctx.print_perf_data.assert_called_once()
    
def test_tokenize_prompt_empty(cli: LlamaCLI, mocker: MockerFixture):
    """Test tokenizing empty prompt."""
    # Mock the vocab
    mock_vocab = mocker.Mock()
    mock_vocab.tokenize.return_value = []
    cli.vocab = mock_vocab
    
    result = cli._tokenize_prompt("")
    assert result == []
    # The method should return early for empty prompt, so tokenize might not be called
    # Let's check if it was called or if the early return worked
    if mock_vocab.tokenize.called:
        mock_vocab.tokenize.assert_called_with("", add_special=True, parse_special=True)


def test_tokenize_prompt_non_empty(cli: LlamaCLI, mocker: MockerFixture):
    """Test tokenizing non-empty prompt."""
    # Mock the vocab
    mock_vocab = mocker.Mock()
    mock_vocab.tokenize.return_value = [1, 2, 3, 4, 5]
    cli.vocab = mock_vocab
    
    result = cli._tokenize_prompt("Hello world")
    assert result == [1, 2, 3, 4, 5]
    mock_vocab.tokenize.assert_called_with("Hello world", add_special=True, parse_special=True)


# Integration tests for LlamaCLI that require mocking cyllama components
    
def test_load_model_basic(cli: LlamaCLI, test_model_path, mocker: MockerFixture):
    """Test basic model loading with mocked cyllama."""
    # Setup mocks
    mock_cy = mocker.patch('cyllama.llama.cli.cy')
    mocker.patch('builtins.print')
    
    mock_model = mocker.Mock()
    mock_vocab = mocker.Mock()
    mock_ctx = mocker.Mock()
    mock_sampler = mocker.Mock()
    
    mock_cy.LlamaModel.return_value = mock_model
    mock_cy.LlamaContext.return_value = mock_ctx
    mock_cy.LlamaSampler.return_value = mock_sampler
    mock_model.get_vocab.return_value = mock_vocab
    mock_sampler.get_seed.return_value = 42
    mock_ctx.n_ctx = 4096
    
    # Create test args
    args = argparse.Namespace(
        model=test_model_path,
        numa=False,
        n_gpu_layers=-1,
        no_mmap=False,
        mlock=False,
        ctx_size=4096,
        batch_size=2048,
        ubatch=512,
        threads=4,
        threads_batch=4,
        rope_freq_base=0.0,
        rope_freq_scale=0.0,
        yarn_ext_factor=-1.0,
        yarn_attn_factor=1.0,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
        yarn_orig_ctx=0,
        no_perf=False,
        n_predict=-1,
        keep=0
    )
    
    cli._load_model(args)
    
    # Verify model was loaded
    assert cli.model == mock_model
    assert cli.vocab == mock_vocab
    assert cli.ctx == mock_ctx
    assert cli.sampler == mock_sampler
    
    # Verify cyllama calls
    mock_cy.llama_backend_init.assert_called_once()
    mock_cy.ggml_backend_load_all.assert_called_once()
    mock_cy.LlamaModel.assert_called_once()
    mock_cy.LlamaContext.assert_called_once()
    mock_cy.LlamaSampler.assert_called_once()
    
def test_load_model_failure(cli: LlamaCLI, test_model_path, mocker: MockerFixture):
    """Test model loading failure."""
    # Setup mock to return None (failure)
    mock_cy = mocker.patch('cyllama.llama.cli.cy')
    mocker.patch('builtins.print')
    mock_exit = mocker.patch('sys.exit', side_effect=SystemExit)
    
    mock_cy.LlamaModel.return_value = None
    
    args = argparse.Namespace(
        model=test_model_path,
        numa=False,
        n_gpu_layers=-1,
        no_mmap=False,
        mlock=False,
        ctx_size=4096,
        batch_size=2048,
        ubatch=512,
        threads=4,
        threads_batch=4,
        rope_freq_base=0.0,
        rope_freq_scale=0.0,
        yarn_ext_factor=-1.0,
        yarn_attn_factor=1.0,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
        yarn_orig_ctx=0,
        no_perf=False,
        n_predict=-1,
        keep=0
    )
    
    with pytest.raises(SystemExit):
        cli._load_model(args)
    mock_exit.assert_called_with(1)
    
def test_generate_text_basic(cli, mocker: MockerFixture):
    """Test basic text generation with mocked components."""
    # Setup mocks
    mock_cy = mocker.patch('cyllama.llama.cli.cy')
    mocker.patch('builtins.print')
    
    mock_vocab = mocker.Mock()
    mock_ctx = mocker.Mock()
    mock_sampler = mocker.Mock()
    
    mock_vocab.get_add_bos.return_value = True
    mock_vocab.token_bos.return_value = 1
    mock_vocab.is_eog.return_value = False
    mock_vocab.token_to_piece.return_value = "test"
    mock_vocab.detokenize.return_value = "Hello world"
    
    mock_ctx.n_ctx = 4096
    mock_ctx.decode.return_value = None
    
    mock_sampler.sample.return_value = 2
    mock_sampler.print_perf_data.return_value = None
    
    mock_cy.ggml_time_us.return_value = 1000000
    mock_cy.llama_batch_get_one.return_value = mocker.Mock()
    
    # Set up CLI state
    cli.vocab = mock_vocab
    cli.ctx = mock_ctx
    cli.sampler = mock_sampler
    
    # Create test args
    args = argparse.Namespace(
        n_predict=5,
        batch_size=4,
        verbose_prompt=False,
        display_prompt=False,
        no_display_prompt=False
    )
    
    prompt_tokens = [1, 2, 3]
    
    result = cli._generate_text(args, prompt_tokens)
    
    # Verify generation occurred
    assert isinstance(result, str)
    mock_ctx.decode.assert_called()
    mock_sampler.sample.assert_called()
    
def test_generate_text_empty_prompt(cli, mocker: MockerFixture):
    """Test text generation with empty prompt."""
    # Setup mocks
    mocker.patch('cyllama.llama.cli.cy')
    mocker.patch('builtins.print')
    mock_exit = mocker.patch('sys.exit', side_effect=SystemExit)
    
    mock_vocab = mocker.Mock()
    mock_vocab.get_add_bos.return_value = False  # This should trigger the exit
    
    cli.vocab = mock_vocab
    
    args = argparse.Namespace(
        n_predict=5,
        batch_size=4,
        verbose_prompt=False,
        display_prompt=False,
        no_display_prompt=False
    )
    
    with pytest.raises(SystemExit):
        cli._generate_text(args, [])
    mock_exit.assert_called_with(-1)


def test_generate_text_prompt_too_long(cli, mocker: MockerFixture):
    """Test text generation with prompt too long."""
    # Setup mocks
    mocker.patch('cyllama.llama.cli.cy')
    mocker.patch('builtins.print')
    mock_exit = mocker.patch('sys.exit', side_effect=SystemExit)
    
    mock_ctx = mocker.Mock()
    mock_ctx.n_ctx = 10  # Small context
    
    cli.ctx = mock_ctx
    
    args = argparse.Namespace(
        n_predict=5,
        batch_size=4,
        verbose_prompt=False,
        display_prompt=False,
        no_display_prompt=False
    )
    
    # Create prompt that's too long (more than ctx - 4)
    long_prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # 11 tokens > 10-4=6
    
    with pytest.raises(SystemExit):
        cli._generate_text(args, long_prompt)
    mock_exit.assert_called_with(1)


# Test error handling scenarios

def test_run_embedding_mode(cli, mocker: MockerFixture):
    """Test running in embedding mode."""
    mock_parse = mocker.patch.object(cli, '_parse_args')
    mock_print = mocker.patch('builtins.print')
    
    mock_parse.return_value = argparse.Namespace(embedding=True)
    
    result = cli.run()
    
    assert result == 0
    mock_print.assert_called()


def test_run_context_size_validation(cli, mocker: MockerFixture):
    """Test context size validation."""
    mock_parse = mocker.patch.object(cli, '_parse_args')
    mock_load = mocker.patch.object(cli, '_load_model')
    mock_prompt = mocker.patch.object(cli, '_load_prompt')
    mock_tokenize = mocker.patch.object(cli, '_tokenize_prompt')
    mock_generate = mocker.patch.object(cli, '_generate_text')
    mock_print = mocker.patch('builtins.print')
    
    # Test with context size too small
    mock_parse.return_value = argparse.Namespace(
        embedding=False,
        ctx_size=4,  # Too small
        interactive=False,
        interactive_first=False
    )
    mock_prompt.return_value = "test"
    mock_tokenize.return_value = [1, 2, 3]
    
    cli.run()
    
    # Should print warning and adjust context size
    mock_print.assert_called()
    # Context size should be adjusted to 8
    assert mock_parse.return_value.ctx_size == 8


# Platform-specific test skipping
import platform

PLATFORM = platform.system()
ARCH = platform.machine()

# Skip certain tests on specific platforms if needed
@pytest.mark.skipif(PLATFORM == "Darwin" and ARCH == "x86_64", 
                   reason="Skip on intel macs")
def test_platform_specific_functionality(cli):
    """Test functionality that may be platform-specific."""
    # This is a placeholder for platform-specific tests
    # that might need to be skipped on certain platforms
    pass
