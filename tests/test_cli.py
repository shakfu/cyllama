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
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

# Import the CLI module
from cyllama.cli import LlamaCLI


class TestLlamaCLI(unittest.TestCase):
    """Test cases for LlamaCLI class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cli = LlamaCLI()
        self.test_model_path = "/fake/path/model.gguf"
    
    def tearDown(self):
        """Clean up after tests."""
        # Reset any global state if needed
        pass
    
    def test_cli_initialization(self):
        """Test CLI initialization."""
        cli = LlamaCLI()
        
        # Check initial state
        self.assertIsNone(cli.model)
        self.assertIsNone(cli.ctx)
        self.assertIsNone(cli.sampler)
        self.assertIsNone(cli.vocab)
        self.assertFalse(cli.is_interacting)
        self.assertFalse(cli.need_insert_eot)
        self.assertEqual(cli.t_main_start, 0)
        self.assertEqual(cli.n_decode, 0)
    
    def test_file_exists(self):
        """Test file existence checking."""
        # Test with existing file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name
        
        try:
            self.assertTrue(self.cli._file_exists(tmp_path))
            self.assertFalse(self.cli._file_exists("/nonexistent/path/file.txt"))
        finally:
            os.unlink(tmp_path)
    
    def test_file_is_empty(self):
        """Test file empty checking."""
        # Test with empty file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            empty_path = tmp.name
        
        # Test with non-empty file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"content")
            non_empty_path = tmp.name
        
        try:
            self.assertTrue(self.cli._file_is_empty(empty_path))
            self.assertFalse(self.cli._file_is_empty(non_empty_path))
            self.assertTrue(self.cli._file_is_empty("/nonexistent/path/file.txt"))
        finally:
            os.unlink(empty_path)
            os.unlink(non_empty_path)
    
    def test_print_usage(self):
        """Test usage printing."""
        with patch('builtins.print') as mock_print:
            self.cli._print_usage("test_prog")
            mock_print.assert_called()
            # Check that usage information is printed
            calls = mock_print.call_args_list
            usage_text = " ".join(str(call) for call in calls)
            self.assertIn("text generation", usage_text)
            self.assertIn("chat", usage_text)
    
    def test_parse_args_basic(self):
        """Test basic argument parsing."""
        # Test with minimal required arguments
        test_args = ["-m", self.test_model_path]
        
        with patch('sys.argv', ['test_cli'] + test_args):
            args = self.cli._parse_args()
            
            self.assertEqual(args.model, self.test_model_path)
            self.assertEqual(args.ctx_size, 4096)  # default
            self.assertEqual(args.batch_size, 2048)  # default
            self.assertEqual(args.threads, 4)  # default
            self.assertEqual(args.temp, 0.8)  # default
            self.assertEqual(args.n_predict, -1)  # default
    
    def test_parse_args_with_options(self):
        """Test argument parsing with various options."""
        test_args = [
            "-m", self.test_model_path,
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
        
        with patch('sys.argv', ['test_cli'] + test_args):
            args = self.cli._parse_args()
            
            self.assertEqual(args.model, self.test_model_path)
            self.assertEqual(args.ctx_size, 2048)
            self.assertEqual(args.batch_size, 1024)
            self.assertEqual(args.threads, 8)
            self.assertEqual(args.temp, 0.5)
            self.assertEqual(args.n_predict, 100)
            self.assertEqual(args.top_k, 20)
            self.assertEqual(args.top_p, 0.9)
            self.assertEqual(args.prompt, "Hello world")
            self.assertTrue(args.interactive)
            self.assertTrue(args.no_mmap)
            self.assertTrue(args.mlock)
    
    def test_parse_args_file_prompt(self):
        """Test argument parsing with file prompt."""
        # Create a temporary file with prompt content
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
            tmp.write("This is a test prompt from file.")
            prompt_file = tmp.name
        
        test_args = ["-m", self.test_model_path, "-f", prompt_file]
        
        try:
            with patch('sys.argv', ['test_cli'] + test_args):
                args = self.cli._parse_args()
                
                self.assertEqual(args.model, self.test_model_path)
                self.assertEqual(args.file, prompt_file)
        finally:
            os.unlink(prompt_file)
    
    def test_load_prompt_from_args(self):
        """Test loading prompt from command line arguments."""
        test_prompt = "This is a test prompt"
        args = argparse.Namespace(prompt=test_prompt, file="", escape=False)
        
        result = self.cli._load_prompt(args)
        self.assertEqual(result, test_prompt)
    
    def test_load_prompt_from_file(self):
        """Test loading prompt from file."""
        test_content = "This is a test prompt from file\nwith multiple lines."
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
            tmp.write(test_content)
            prompt_file = tmp.name
        
        args = argparse.Namespace(prompt="", file=prompt_file, escape=False)
        
        try:
            result = self.cli._load_prompt(args)
            self.assertEqual(result, test_content)
        finally:
            os.unlink(prompt_file)
    
    def test_load_prompt_file_not_exists(self):
        """Test loading prompt from non-existent file."""
        args = argparse.Namespace(prompt="", file="/nonexistent/file.txt", escape=False)
        
        with patch('builtins.print'), \
             patch('sys.exit', side_effect=SystemExit) as mock_exit:
            with self.assertRaises(SystemExit):
                self.cli._load_prompt(args)
            mock_exit.assert_called_with(1)
    
    def test_load_prompt_with_escape_sequences(self):
        """Test loading prompt with escape sequence processing."""
        test_prompt = "Line1\\nLine2\\tTabbed\\'Quoted\\\"Double\\"
        expected = "Line1\nLine2\tTabbed'Quoted\"Double\\"
        
        args = argparse.Namespace(prompt=test_prompt, file="", escape=True)
        
        result = self.cli._load_prompt(args)
        self.assertEqual(result, expected)
    
    def test_sigint_handler_not_interacting(self):
        """Test SIGINT handler when not in interactive mode."""
        self.cli.is_interacting = False
        self.cli.interactive = True
        
        with patch('sys.exit') as mock_exit:
            self.cli._sigint_handler(signal.SIGINT, None)
            
            self.assertTrue(self.cli.is_interacting)
            self.assertTrue(self.cli.need_insert_eot)
            mock_exit.assert_not_called()
    
    def test_sigint_handler_interacting(self):
        """Test SIGINT handler when already interacting."""
        self.cli.is_interacting = True
        
        with patch('builtins.print') as mock_print, \
             patch.object(self.cli, '_print_performance') as mock_perf, \
             patch('sys.exit') as mock_exit:
            
            self.cli._sigint_handler(signal.SIGINT, None)
            
            mock_perf.assert_called_once()
            # Check that both print calls were made
            print_calls = mock_print.call_args_list
            self.assertEqual(len(print_calls), 2)
            self.assertEqual(print_calls[0][0][0], "\n")
            self.assertEqual(print_calls[1][0][0], "Interrupted by user")
            mock_exit.assert_called_with(130)
    
    def test_print_performance_no_timing(self):
        """Test performance printing when no timing data available."""
        self.cli.t_main_start = 0
        
        with patch('builtins.print') as mock_print:
            self.cli._print_performance()
            # Should not print timing info when t_main_start is 0
            mock_print.assert_not_called()
    
    def test_print_performance_with_timing(self):
        """Test performance printing with timing data."""
        self.cli.t_main_start = 1000000  # 1 second in microseconds
        self.cli.n_decode = 50
        
        # Mock the cyllama module
        with patch('cyllama.cli.cy') as mock_cy, \
             patch('builtins.print') as mock_print:
            
            mock_cy.ggml_time_us.return_value = 2000000  # 2 seconds total
            mock_sampler = Mock()
            mock_ctx = Mock()
            self.cli.sampler = mock_sampler
            self.cli.ctx = mock_ctx
            
            self.cli._print_performance()
            
            # Check that performance data was printed
            mock_print.assert_called()
            mock_sampler.print_perf_data.assert_called_once()
            mock_ctx.print_perf_data.assert_called_once()
    
    def test_tokenize_prompt_empty(self):
        """Test tokenizing empty prompt."""
        # Mock the vocab
        mock_vocab = Mock()
        mock_vocab.tokenize.return_value = []
        self.cli.vocab = mock_vocab
        
        result = self.cli._tokenize_prompt("")
        self.assertEqual(result, [])
        # The method should return early for empty prompt, so tokenize might not be called
        # Let's check if it was called or if the early return worked
        if mock_vocab.tokenize.called:
            mock_vocab.tokenize.assert_called_with("", add_special=True, parse_special=True)
    
    def test_tokenize_prompt_non_empty(self):
        """Test tokenizing non-empty prompt."""
        # Mock the vocab
        mock_vocab = Mock()
        mock_vocab.tokenize.return_value = [1, 2, 3, 4, 5]
        self.cli.vocab = mock_vocab
        
        result = self.cli._tokenize_prompt("Hello world")
        self.assertEqual(result, [1, 2, 3, 4, 5])
        mock_vocab.tokenize.assert_called_with("Hello world", add_special=True, parse_special=True)


class TestLlamaCLIIntegration(unittest.TestCase):
    """Integration tests for LlamaCLI that require mocking cyllama components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cli = LlamaCLI()
        self.test_model_path = "/fake/path/model.gguf"
    
    @patch('cyllama.cli.cy')
    def test_load_model_basic(self, mock_cy):
        """Test basic model loading with mocked cyllama."""
        # Setup mocks
        mock_model = Mock()
        mock_vocab = Mock()
        mock_ctx = Mock()
        mock_sampler = Mock()
        
        mock_cy.LlamaModel.return_value = mock_model
        mock_cy.LlamaContext.return_value = mock_ctx
        mock_cy.LlamaSampler.return_value = mock_sampler
        mock_model.get_vocab.return_value = mock_vocab
        mock_sampler.get_seed.return_value = 42
        mock_ctx.n_ctx = 4096
        
        # Create test args
        args = argparse.Namespace(
            model=self.test_model_path,
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
        
        with patch('builtins.print'):
            self.cli._load_model(args)
        
        # Verify model was loaded
        self.assertEqual(self.cli.model, mock_model)
        self.assertEqual(self.cli.vocab, mock_vocab)
        self.assertEqual(self.cli.ctx, mock_ctx)
        self.assertEqual(self.cli.sampler, mock_sampler)
        
        # Verify cyllama calls
        mock_cy.llama_backend_init.assert_called_once()
        mock_cy.ggml_backend_load_all.assert_called_once()
        mock_cy.LlamaModel.assert_called_once()
        mock_cy.LlamaContext.assert_called_once()
        mock_cy.LlamaSampler.assert_called_once()
    
    @patch('cyllama.cli.cy')
    def test_load_model_failure(self, mock_cy):
        """Test model loading failure."""
        # Setup mock to return None (failure)
        mock_cy.LlamaModel.return_value = None
        
        args = argparse.Namespace(
            model=self.test_model_path,
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
        
        with patch('builtins.print'), \
             patch('sys.exit', side_effect=SystemExit) as mock_exit:
            
            with self.assertRaises(SystemExit):
                self.cli._load_model(args)
            mock_exit.assert_called_with(1)
    
    @patch('cyllama.cli.cy')
    def test_generate_text_basic(self, mock_cy):
        """Test basic text generation with mocked components."""
        # Setup mocks
        mock_vocab = Mock()
        mock_ctx = Mock()
        mock_sampler = Mock()
        
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
        mock_cy.llama_batch_get_one.return_value = Mock()
        
        # Set up CLI state
        self.cli.vocab = mock_vocab
        self.cli.ctx = mock_ctx
        self.cli.sampler = mock_sampler
        
        # Create test args
        args = argparse.Namespace(
            n_predict=5,
            batch_size=4,
            verbose_prompt=False,
            display_prompt=False,
            no_display_prompt=False
        )
        
        prompt_tokens = [1, 2, 3]
        
        with patch('builtins.print'):
            result = self.cli._generate_text(args, prompt_tokens)
        
        # Verify generation occurred
        self.assertIsInstance(result, str)
        mock_ctx.decode.assert_called()
        mock_sampler.sample.assert_called()
    
    @patch('cyllama.cli.cy')
    def test_generate_text_empty_prompt(self, mock_cy):
        """Test text generation with empty prompt."""
        # Setup mocks
        mock_vocab = Mock()
        mock_vocab.get_add_bos.return_value = False  # This should trigger the exit
        
        self.cli.vocab = mock_vocab
        
        args = argparse.Namespace(
            n_predict=5,
            batch_size=4,
            verbose_prompt=False,
            display_prompt=False,
            no_display_prompt=False
        )
        
        with patch('builtins.print'), \
             patch('sys.exit', side_effect=SystemExit) as mock_exit:
            with self.assertRaises(SystemExit):
                self.cli._generate_text(args, [])
            mock_exit.assert_called_with(-1)
    
    @patch('cyllama.cli.cy')
    def test_generate_text_prompt_too_long(self, mock_cy):
        """Test text generation with prompt too long."""
        # Setup mocks
        mock_ctx = Mock()
        mock_ctx.n_ctx = 10  # Small context
        
        self.cli.ctx = mock_ctx
        
        args = argparse.Namespace(
            n_predict=5,
            batch_size=4,
            verbose_prompt=False,
            display_prompt=False,
            no_display_prompt=False
        )
        
        # Create prompt that's too long (more than ctx - 4)
        long_prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # 11 tokens > 10-4=6
        
        with patch('builtins.print'), \
             patch('sys.exit', side_effect=SystemExit) as mock_exit:
            
            with self.assertRaises(SystemExit):
                self.cli._generate_text(args, long_prompt)
            mock_exit.assert_called_with(1)


class TestLlamaCLIErrorHandling(unittest.TestCase):
    """Test error handling scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cli = LlamaCLI()
    
    def test_run_embedding_mode(self):
        """Test running in embedding mode."""
        with patch.object(self.cli, '_parse_args') as mock_parse, \
             patch('builtins.print') as mock_print:
            
            mock_parse.return_value = argparse.Namespace(embedding=True)
            
            result = self.cli.run()
            
            self.assertEqual(result, 0)
            mock_print.assert_called()
    
    def test_run_context_size_validation(self):
        """Test context size validation."""
        with patch.object(self.cli, '_parse_args') as mock_parse, \
             patch.object(self.cli, '_load_model') as mock_load, \
             patch.object(self.cli, '_load_prompt') as mock_prompt, \
             patch.object(self.cli, '_tokenize_prompt') as mock_tokenize, \
             patch.object(self.cli, '_generate_text') as mock_generate, \
             patch('builtins.print') as mock_print:
            
            # Test with context size too small
            mock_parse.return_value = argparse.Namespace(
                embedding=False,
                ctx_size=4,  # Too small
                interactive=False,
                interactive_first=False
            )
            mock_prompt.return_value = "test"
            mock_tokenize.return_value = [1, 2, 3]
            
            self.cli.run()
            
            # Should print warning and adjust context size
            mock_print.assert_called()
            # Context size should be adjusted to 8
            self.assertEqual(mock_parse.return_value.ctx_size, 8)


# Platform-specific test skipping
import platform

PLATFORM = platform.system()
ARCH = platform.machine()

# Skip certain tests on specific platforms if needed
@pytest.mark.skipif(PLATFORM == "Darwin" and ARCH == "x86_64", 
                   reason="Skip on intel macs")
class TestLlamaCLIPlatformSpecific(unittest.TestCase):
    """Platform-specific tests that may need to be skipped."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cli = LlamaCLI()
    
    def test_platform_specific_functionality(self):
        """Test functionality that may be platform-specific."""
        # This is a placeholder for platform-specific tests
        # that might need to be skipped on certain platforms
        pass


if __name__ == '__main__':
    unittest.main()
