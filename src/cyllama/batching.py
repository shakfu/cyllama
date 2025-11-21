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
import time

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
        verbose: bool = False
    ):
        """
        Initialize batch generator.

        Args:
            model_path: Path to GGUF model file
            batch_size: Maximum batch size for processing
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU
            verbose: Print detailed information
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.n_ctx = n_ctx
        self.verbose = verbose

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

        # Process prompts in batch
        batch = LlamaBatch(self.batch_size)
        responses = [""] * len(prompts)
        active_sequences = set(range(len(prompts)))
        seq_positions = {i: 0 for i in range(len(prompts))}

        # Add all prompt tokens to batch
        for seq_id, tokens in enumerate(tokenized_prompts):
            for i, token in enumerate(tokens):
                is_last = (i == len(tokens) - 1)
                batch.add_sequence(
                    token=token,
                    pos=i,
                    seq_ids=[seq_id],
                    logits=is_last  # Only compute logits for last token
                )
            seq_positions[seq_id] = len(tokens)

        # Decode initial batch
        self.ctx.decode(batch)

        # Generate tokens for each sequence
        for _ in range(config.max_tokens):
            if not active_sequences:
                break

            batch.clear()

            # Sample next token for each active sequence
            for seq_id in list(active_sequences):
                # Sample token
                new_token = sampler.sample(self.ctx, seq_id)

                # Check for end of generation
                if self.vocab.is_eog(new_token):
                    active_sequences.remove(seq_id)
                    continue

                # Decode token
                try:
                    piece = self.vocab.token_to_piece(new_token, special=True)
                    responses[seq_id] += piece
                except UnicodeDecodeError:
                    pass

                # Add to batch for next iteration
                batch.add_sequence(
                    token=new_token,
                    pos=seq_positions[seq_id],
                    seq_ids=[seq_id],
                    logits=True
                )
                seq_positions[seq_id] += 1

            # Decode batch if not empty
            if batch.n_tokens > 0:
                self.ctx.decode(batch)

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
    config: Optional[GenerationConfig] = None,
    **kwargs
) -> List[str]:
    """
    Convenience function for batch generation.

    Args:
        prompts: List of input prompts
        model_path: Path to GGUF model file
        batch_size: Maximum batch size
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

    generator = BatchGenerator(model_path, batch_size=batch_size)
    return generator.generate_batch(prompts, config)
