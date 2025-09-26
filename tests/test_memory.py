"""Tests for memory estimation functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from cyllama.memory import (
    MemoryEstimate,
    estimate_gpu_layers,
    estimate_memory_usage,
    graph_size,
    projector_memory_requirements,
    dump_metadata_json,
)


class TestMemoryEstimate:
    """Test MemoryEstimate dataclass."""

    def test_memory_estimate_creation(self):
        """Test creating a MemoryEstimate object."""
        estimate = MemoryEstimate(
            layers=32,
            graph_size=1024 * 1024 * 100,
            vram=1024 * 1024 * 1024 * 8,
            vram_kv=1024 * 1024 * 500,
            total_size=1024 * 1024 * 1024 * 4,
            tensor_split=[16, 16]
        )

        assert estimate.layers == 32
        assert estimate.graph_size == 1024 * 1024 * 100
        assert estimate.vram == 1024 * 1024 * 1024 * 8
        assert estimate.vram_kv == 1024 * 1024 * 500
        assert estimate.total_size == 1024 * 1024 * 1024 * 4
        assert estimate.tensor_split == [16, 16]

    def test_memory_estimate_without_tensor_split(self):
        """Test MemoryEstimate without tensor split (single GPU)."""
        estimate = MemoryEstimate(
            layers=20,
            graph_size=1024 * 1024 * 50,
            vram=1024 * 1024 * 1024 * 4,
            vram_kv=1024 * 1024 * 250,
            total_size=1024 * 1024 * 1024 * 2
        )

        assert estimate.tensor_split is None


class TestGraphSize:
    """Test graph memory size calculations."""

    def test_llama_architecture(self):
        """Test graph size calculation for LLaMA architecture."""
        size = graph_size(
            architecture='llama',
            n_layers=32,
            n_embd=4096,
            n_ff=11008,
            n_head=32,
            n_head_kv=32,
            n_vocab=32000,
            n_ctx=2048,
            n_batch=1
        )

        assert isinstance(size, int)
        assert size > 0

    def test_gemma_architecture(self):
        """Test graph size calculation for Gemma architecture."""
        size = graph_size(
            architecture='gemma',
            n_layers=28,
            n_embd=3072,
            n_ff=8192,
            n_head=16,
            n_head_kv=16,
            n_vocab=256000,
            n_ctx=8192,
            n_batch=1
        )

        assert isinstance(size, int)
        assert size > 0

    def test_unknown_architecture(self):
        """Test graph size calculation for unknown architecture (uses default)."""
        size = graph_size(
            architecture='unknown',
            n_layers=16,
            n_embd=2048,
            n_ff=5504,
            n_head=16,
            n_head_kv=16,
            n_vocab=16000,
            n_ctx=1024,
            n_batch=1
        )

        assert isinstance(size, int)
        assert size > 0

    def test_flash_attention_modifier(self):
        """Test that flash attention reduces memory usage."""
        size_normal = graph_size(
            architecture='llama',
            n_layers=32,
            n_embd=4096,
            n_ff=11008,
            n_head=32,
            n_head_kv=32,
            n_vocab=32000,
            n_ctx=2048,
            n_batch=1,
            flash_attn=False
        )

        size_flash = graph_size(
            architecture='llama',
            n_layers=32,
            n_embd=4096,
            n_ff=11008,
            n_head=32,
            n_head_kv=32,
            n_vocab=32000,
            n_ctx=2048,
            n_batch=1,
            flash_attn=True
        )

        assert size_flash < size_normal


class TestProjectorMemory:
    """Test projector memory calculations."""

    def test_clip_model(self):
        """Test projector memory for CLIP model."""
        metadata = {'general.architecture': 'clip'}
        mem = projector_memory_requirements(metadata)
        assert mem > 0

    def test_non_multimodal_model(self):
        """Test projector memory for non-multimodal model."""
        metadata = {'general.architecture': 'llama'}
        mem = projector_memory_requirements(metadata)
        assert mem == 0

    def test_empty_metadata(self):
        """Test projector memory with empty metadata."""
        metadata = {}
        mem = projector_memory_requirements(metadata)
        assert mem == 0


class TestDumpMetadata:
    """Test metadata extraction."""

    @patch('cyllama.llama.llama_cpp.LlamaModel')
    @patch('cyllama.llama.llama_cpp.LlamaModelParams')
    def test_dump_metadata_success(self, mock_params, mock_model):
        """Test successful metadata extraction."""
        # Mock the model and vocab
        mock_vocab = MagicMock()
        mock_vocab.n_vocab = 32000

        mock_model_instance = MagicMock()
        mock_model_instance.get_vocab.return_value = mock_vocab
        mock_model.return_value = mock_model_instance

        metadata = dump_metadata_json('dummy_path.gguf')

        assert 'general.architecture' in metadata
        assert 'llama.context_length' in metadata
        assert 'llama.embedding_length' in metadata
        assert 'llama.block_count' in metadata
        assert 'tokenizer.ggml.tokens' in metadata

    def test_dump_metadata_fallback(self):
        """Test metadata fallback when model loading fails."""
        with patch('cyllama.llama.llama_cpp.LlamaModel', side_effect=Exception("Model loading failed")):
            metadata = dump_metadata_json('nonexistent.gguf')

            # Should return fallback metadata
            assert metadata['general.architecture'] == 'llama'
            assert metadata['llama.context_length'] == 2048
            assert metadata['llama.embedding_length'] == 4096
            assert len(metadata['tokenizer.ggml.tokens']) == 32000


class TestEstimateGpuLayers:
    """Test GPU layer estimation."""

    @patch('cyllama.memory.dump_metadata_json')
    def test_single_gpu_estimation(self, mock_metadata):
        """Test estimation for single GPU setup."""
        mock_metadata.return_value = {
            'general.architecture': 'llama',
            'llama.context_length': 2048,
            'llama.embedding_length': 4096,
            'llama.block_count': 32,
            'llama.feed_forward_length': 11008,
            'llama.attention.head_count': 32,
            'llama.attention.head_count_kv': 32,
            'general.file_type': 2,  # Q4_0
            'tokenizer.ggml.tokens': [f"token_{i}" for i in range(32000)],
        }

        estimate = estimate_gpu_layers(
            model_path='dummy.gguf',
            gpu_memory_mb=8192,  # 8GB
            ctx_size=2048,
            batch_size=1
        )

        assert isinstance(estimate, MemoryEstimate)
        assert estimate.layers >= 0
        assert estimate.layers <= 32
        assert estimate.tensor_split is None
        assert estimate.graph_size > 0
        assert estimate.total_size > 0

    @patch('cyllama.memory.dump_metadata_json')
    def test_multi_gpu_estimation(self, mock_metadata):
        """Test estimation for multi-GPU setup."""
        mock_metadata.return_value = {
            'general.architecture': 'llama',
            'llama.context_length': 2048,
            'llama.embedding_length': 4096,
            'llama.block_count': 32,
            'llama.feed_forward_length': 11008,
            'llama.attention.head_count': 32,
            'llama.attention.head_count_kv': 32,
            'general.file_type': 2,  # Q4_0
            'tokenizer.ggml.tokens': [f"token_{i}" for i in range(32000)],
        }

        estimate = estimate_gpu_layers(
            model_path='dummy.gguf',
            gpu_memory_mb=[4096, 4096],  # 2x 4GB GPUs
            ctx_size=2048,
            batch_size=1
        )

        assert isinstance(estimate, MemoryEstimate)
        assert estimate.layers >= 0
        assert estimate.layers <= 32
        assert estimate.tensor_split is not None
        assert len(estimate.tensor_split) == 2
        assert sum(estimate.tensor_split) == estimate.layers

    @patch('cyllama.memory.dump_metadata_json')
    def test_insufficient_memory(self, mock_metadata):
        """Test estimation with insufficient GPU memory."""
        mock_metadata.return_value = {
            'general.architecture': 'llama',
            'llama.context_length': 2048,
            'llama.embedding_length': 4096,
            'llama.block_count': 32,
            'llama.feed_forward_length': 11008,
            'llama.attention.head_count': 32,
            'llama.attention.head_count_kv': 32,
            'general.file_type': 0,  # F32 (largest)
            'tokenizer.ggml.tokens': [f"token_{i}" for i in range(32000)],
        }

        estimate = estimate_gpu_layers(
            model_path='dummy.gguf',
            gpu_memory_mb=512,  # Very small memory
            ctx_size=2048,
            batch_size=1
        )

        assert isinstance(estimate, MemoryEstimate)
        assert estimate.layers >= 0  # Should handle gracefully


class TestEstimateMemoryUsage:
    """Test memory usage estimation."""

    @patch('cyllama.memory.dump_metadata_json')
    def test_memory_usage_estimation(self, mock_metadata):
        """Test general memory usage estimation."""
        mock_metadata.return_value = {
            'general.architecture': 'llama',
            'llama.context_length': 2048,
            'llama.embedding_length': 4096,
            'llama.block_count': 32,
            'llama.feed_forward_length': 11008,
            'llama.attention.head_count': 32,
            'llama.attention.head_count_kv': 32,
            'general.file_type': 1,  # F16
            'tokenizer.ggml.tokens': [f"token_{i}" for i in range(32000)],
        }

        result = estimate_memory_usage(
            model_path='dummy.gguf',
            ctx_size=2048,
            batch_size=1
        )

        assert 'model_size_mb' in result
        assert 'kv_cache_mb' in result
        assert 'graph_mb' in result
        assert 'parameters' in result

        # Check model size estimates
        assert 'f32' in result['model_size_mb']
        assert 'f16' in result['model_size_mb']
        assert 'q4_0' in result['model_size_mb']
        assert 'q8_0' in result['model_size_mb']

        # Check KV cache estimates
        assert 'f16' in result['kv_cache_mb']
        assert 'f32' in result['kv_cache_mb']

        # Check parameters
        assert 'n_embd' in result['parameters']
        assert 'n_layer' in result['parameters']
        assert 'total_params' in result['parameters']

        # Verify sizes make sense
        assert result['model_size_mb']['f32'] > result['model_size_mb']['f16']
        assert result['model_size_mb']['f16'] > result['model_size_mb']['q4_0']
        assert result['kv_cache_mb']['f32'] > result['kv_cache_mb']['f16']


# @pytest.mark.integration
class TestMemoryIntegration:
    """Integration tests that require a real model file."""

    def test_real_model_estimation(self, model_path):
        """Test memory estimation with a real model file."""
        if not Path(model_path).exists():
            pytest.skip(f"Model file not found: {model_path}")

        # Test basic memory usage estimation
        result = estimate_memory_usage(model_path, verbose=False)
        assert 'model_size_mb' in result
        assert 'kv_cache_mb' in result
        assert result['parameters']['total_params'] > 0

        # Test GPU layer estimation
        estimate = estimate_gpu_layers(
            model_path=model_path,
            gpu_memory_mb=4096,  # 4GB
            ctx_size=512,
            verbose=False
        )
        assert isinstance(estimate, MemoryEstimate)
        assert estimate.layers >= 0