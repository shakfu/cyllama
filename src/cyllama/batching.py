"""
Batching Utilities for Efficient Inference

This module provides utilities for batching multiple requests together
for efficient parallel processing.

Example:
    >>> from cyllama.batching import BatchGenerator
    >>>
    >>> batch_gen = BatchGenerator("models/llama.gguf", batch_size=4)
    >>>
    >>> prompts = [
    >>>     "What is 2+2?",
    >>>     "What is 3+3?",
    >>>     "What is 4+4?",
    >>>     "What is 5+5?"
    >>> ]
    >>>
    >>> results = batch_gen.generate_batch(prompts)
    >>> for prompt, response in zip(prompts, results):
    >>>     print(f"{prompt} -> {response}")
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)

from .llama.llama_cpp import (
    LlamaModel,
    LlamaContext,
    LlamaModelParams,
    LlamaContextParams,
    LlamaSampler,
    LlamaSamplerChainParams,
    LlamaBatch,
    ggml_backend_load_all,
    disable_logging,
    get_pooled_batch,
    return_batch_to_pool,
)
from .api import GenerationConfig


@dataclass
class BatchRequest:
    """Single request in a batch."""
    id: int
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7


@dataclass
class BatchResponse:
    """Response for a single request in a batch."""
    id: int
    prompt: str
    response: str
    tokens_generated: int
    time_taken: float


class BatchGenerator:
    """
    Batch generator for efficient parallel inference.

    This class processes multiple prompts in parallel using llama.cpp's
    batching capabilities for improved throughput.

    Example:
        >>> batch_gen = BatchGenerator("models/llama.gguf", batch_size=8)
        >>> responses = batch_gen.generate_batch([
        >>>     "What is Python?",
        >>>     "What is Rust?",
        >>>     "What is Go?"
        >>> ])
    """

    def __init__(
        self,
        model_path: str,
        batch_size: int = 512,
        n_ctx: int = 2048,
        n_gpu_layers: int = 99,
        n_seq_max: int = 8,
        verbose: bool = False,
        use_pooling: bool = False
    ):
        """
        Initialize batch generator.

        Args:
            model_path: Path to GGUF model file
            batch_size: Maximum batch size for processing
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU
            n_seq_max: Maximum number of parallel sequences (default: 8)
            verbose: Print detailed information
            use_pooling: Enable batch memory pooling for reduced allocation overhead.
                This can improve performance in high-throughput scenarios by reusing
                batch memory instead of allocating/deallocating for each generation.
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.n_ctx = n_ctx
        self.n_seq_max = n_seq_max
        self.verbose = verbose
        self.use_pooling = use_pooling

        # Disable llama.cpp logging unless verbose mode is enabled
        if not verbose:
            disable_logging()

        # Load backends
        ggml_backend_load_all()

        # Initialize model
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = n_gpu_layers

        if self.verbose:
            print(f"Loading model: {model_path}")

        self.model = LlamaModel(model_path, model_params)
        self.vocab = self.model.get_vocab()

        # Initialize context
        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = batch_size
        ctx_params.n_seq_max = n_seq_max  # Support parallel sequences

        self.ctx = LlamaContext(self.model, ctx_params)

        if self.verbose:
            print(f"Model loaded with context size {n_ctx}, batch size {batch_size}")

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: List of input prompts
            config: Generation configuration (uses defaults if None)

        Returns:
            List of generated responses (same order as inputs)

        Example:
            >>> prompts = ["Hello", "Hi", "Hey"]
            >>> responses = batch_gen.generate_batch(prompts)
        """
        if not prompts:
            return []

        if len(prompts) > self.n_seq_max:
            raise ValueError(
                f"Too many prompts ({len(prompts)}) for configured n_seq_max ({self.n_seq_max}). "
                f"Either reduce the number of prompts or increase n_seq_max when creating BatchGenerator."
            )

        config = config or GenerationConfig()

        # Tokenize all prompts
        tokenized_prompts = []
        for prompt in prompts:
            tokens = self.vocab.tokenize(
                prompt,
                add_special=config.add_bos,
                parse_special=config.parse_special
            )
            tokenized_prompts.append(tokens)

        if self.verbose:
            print(f"Processing {len(prompts)} prompts")
            for i, tokens in enumerate(tokenized_prompts):
                print(f"  Prompt {i}: {len(tokens)} tokens")

        # Create sampler
        sampler_params = LlamaSamplerChainParams()
        sampler = LlamaSampler(sampler_params)

        if config.temperature == 0.0:
            sampler.add_greedy()
        else:
            sampler.add_min_p(config.min_p, 1)
            sampler.add_top_k(config.top_k)
            sampler.add_top_p(config.top_p, 1)
            sampler.add_temp(config.temperature)
            sampler.add_dist(config.seed if config.seed != -1 else int(time.time()))

        # Process prompts in batch (use pooling if enabled)
        if self.use_pooling:
            batch = get_pooled_batch(n_tokens=self.batch_size, embd=0, n_seq_max=self.n_seq_max)
        else:
            batch = LlamaBatch(n_tokens=self.batch_size, embd=0, n_seq_max=self.n_seq_max)
        responses = [""] * len(prompts)
        active_sequences = set(range(len(prompts)))
        seq_positions = {i: 0 for i in range(len(prompts))}

        # Add all prompt tokens to batch, tracking batch index for each sequence's logits
        seq_logits_idx = {}
        batch_idx = 0
        for seq_id, tokens in enumerate(tokenized_prompts):
            for i, token in enumerate(tokens):
                is_last = (i == len(tokens) - 1)
                batch.add(token, i, [seq_id], is_last)  # Use add() with positional args
                if is_last:
                    # Remember the batch index where this sequence's logits will be
                    seq_logits_idx[seq_id] = batch_idx
                batch_idx += 1
            seq_positions[seq_id] = len(tokens)

        # Decode initial batch
        self.ctx.decode(batch)

        # Generate tokens for each sequence
        for _ in range(config.max_tokens):
            if not active_sequences:
                break

            batch.clear()

            # Sample next token for each active sequence using previous logits
            batch_idx = 0
            for seq_id in list(active_sequences):
                # Sample token using the batch index from last decode
                logits_idx = seq_logits_idx[seq_id]
                new_token = sampler.sample(self.ctx, logits_idx)

                # Check for end of generation
                if self.vocab.is_eog(new_token):
                    active_sequences.remove(seq_id)
                    continue

                # Decode token
                try:
                    piece = self.vocab.token_to_piece(new_token, special=True)
                    responses[seq_id] += piece
                except UnicodeDecodeError:
                    logger.warning("Failed to decode token %d in sequence %d: UnicodeDecodeError", new_token, seq_id)

                # Add to batch for next iteration and remember new logits index
                batch.add(new_token, seq_positions[seq_id], [seq_id], True)
                seq_logits_idx[seq_id] = batch_idx
                batch_idx += 1
                seq_positions[seq_id] += 1

            # Decode batch if not empty
            if batch.n_tokens > 0:
                self.ctx.decode(batch)

        # Return batch to pool if pooling is enabled
        if self.use_pooling:
            return_batch_to_pool(batch)

        if self.verbose:
            print(f"Generated {len(prompts)} responses")
            for i, response in enumerate(responses):
                print(f"  Response {i}: {len(response)} characters")

        return responses

    def generate_batch_detailed(
        self,
        requests: List[BatchRequest],
        config: Optional[GenerationConfig] = None
    ) -> List[BatchResponse]:
        """
        Generate responses with detailed statistics.

        Args:
            requests: List of BatchRequest objects
            config: Base generation configuration

        Returns:
            List of BatchResponse objects with statistics
        """
        start_time = time.time()

        prompts = [req.prompt for req in requests]
        responses = self.generate_batch(prompts, config)

        end_time = time.time()
        total_time = end_time - start_time

        # Create detailed responses
        results = []
        for req, response in zip(requests, responses):
            # Approximate token count
            response_tokens = self.vocab.tokenize(response, add_special=False, parse_special=False)

            result = BatchResponse(
                id=req.id,
                prompt=req.prompt,
                response=response,
                tokens_generated=len(response_tokens),
                time_taken=total_time / len(requests)  # Approximate per-request time
            )
            results.append(result)

        return results


def batch_generate(
    prompts: List[str],
    model_path: str,
    batch_size: int = 512,
    n_seq_max: int = 8,
    config: Optional[GenerationConfig] = None,
    **kwargs
) -> List[str]:
    """
    Convenience function for batch generation.

    Args:
        prompts: List of input prompts
        model_path: Path to GGUF model file
        batch_size: Maximum batch size
        n_seq_max: Maximum number of parallel sequences (default: 8)
        config: Generation configuration
        **kwargs: Additional config parameters

    Returns:
        List of generated responses

    Example:
        >>> prompts = ["Hello", "Hi", "Hey"]
        >>> responses = batch_generate(prompts, "models/llama.gguf")
    """
    # Merge config with kwargs
    if config is None:
        config = GenerationConfig(**kwargs)
    else:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    generator = BatchGenerator(model_path, batch_size=batch_size, n_seq_max=n_seq_max)
    return generator.generate_batch(prompts, config)
