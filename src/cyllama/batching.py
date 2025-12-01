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
from .api import GenerationConfig, Response, GenerationStats


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

    Supports context manager protocol for automatic resource cleanup:
        >>> with BatchGenerator("models/llama.gguf", batch_size=8) as batch_gen:
        >>>     responses = batch_gen.generate_batch([
        >>>         "What is Python?",
        >>>         "What is Rust?",
        >>>         "What is Go?"
        >>>     ])

    Or explicit cleanup:
        >>> batch_gen = BatchGenerator("models/llama.gguf", batch_size=8)
        >>> try:
        >>>     responses = batch_gen.generate_batch(["What is Python?"])
        >>> finally:
        >>>     batch_gen.close()
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
        self._closed = False

        if self.verbose:
            print(f"Model loaded with context size {n_ctx}, batch size {batch_size}")

    def close(self) -> None:
        """
        Release resources held by this BatchGenerator.

        This method releases the model and context resources. After calling close(),
        the BatchGenerator cannot be used for further generation.

        It's safe to call close() multiple times.
        """
        if self._closed:
            return

        self._closed = True

        # Release context first (depends on model)
        if hasattr(self, 'ctx') and self.ctx is not None:
            self.ctx = None

        # Release model
        if hasattr(self, 'model') and self.model is not None:
            self.model = None

        # Clear vocab reference
        if hasattr(self, 'vocab'):
            self.vocab = None

        if self.verbose:
            print("BatchGenerator resources released")

    def __del__(self) -> None:
        """Destructor to ensure resources are cleaned up."""
        try:
            self.close()
        except Exception:
            # Suppress exceptions during garbage collection
            pass

    def __enter__(self) -> "BatchGenerator":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures cleanup."""
        self.close()

    @property
    def is_closed(self) -> bool:
        """Check if the BatchGenerator has been closed."""
        return self._closed

    def _check_closed(self) -> None:
        """Raise an error if the generator has been closed."""
        if self._closed:
            raise RuntimeError(
                "BatchGenerator has been closed and cannot be used for generation. "
                "Create a new BatchGenerator instance to continue."
            )

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None
    ) -> List[Response]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: List of input prompts
            config: Generation configuration (uses defaults if None)

        Returns:
            List of Response objects (same order as inputs).
            Each Response can be used as a string due to __str__.

        Example:
            >>> prompts = ["Hello", "Hi", "Hey"]
            >>> responses = batch_gen.generate_batch(prompts)
            >>> for r in responses:
            >>>     print(r)  # Works like a string
            >>>     print(r.stats)  # Access statistics

        Raises:
            RuntimeError: If the BatchGenerator has been closed
            ValueError: If too many prompts for configured n_seq_max
            TypeError: If prompts is not a list
        """
        self._check_closed()

        if prompts is None:
            raise TypeError("prompts cannot be None, expected a list of strings")

        if not isinstance(prompts, list):
            raise TypeError(
                f"prompts must be a list of strings, got {type(prompts).__name__}. "
                f"For a single prompt, use: generate_batch([prompt])"
            )

        if not prompts:
            return []

        # Validate all prompts are strings
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                raise TypeError(
                    f"All prompts must be strings, but prompt at index {i} is {type(prompt).__name__}. "
                    f"Value: {repr(prompt)[:50]}{'...' if len(repr(prompt)) > 50 else ''}"
                )

        if len(prompts) > self.n_seq_max:
            raise ValueError(
                f"Too many prompts ({len(prompts)}) for configured n_seq_max ({self.n_seq_max}). "
                f"Either reduce the number of prompts, increase n_seq_max when creating BatchGenerator, "
                f"or process prompts in batches of {self.n_seq_max} or fewer."
            )

        config = config or GenerationConfig()
        start_time = time.time()

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

        end_time = time.time()
        total_time = end_time - start_time

        # Wrap responses in Response objects with stats
        result_responses = []
        for i, (text, prompt_tokens) in enumerate(zip(responses, tokenized_prompts)):
            # Approximate token count for response
            response_tokens = self.vocab.tokenize(text, add_special=False, parse_special=False)
            n_generated = len(response_tokens)

            stats = GenerationStats(
                prompt_tokens=len(prompt_tokens),
                generated_tokens=n_generated,
                total_time=total_time / len(prompts),  # Approximate per-prompt time
                tokens_per_second=n_generated / (total_time / len(prompts)) if total_time > 0 else 0.0
            )

            result_responses.append(Response(
                text=text,
                stats=stats,
                finish_reason="stop",
                model=self.model_path
            ))

        if self.verbose:
            print(f"Generated {len(prompts)} responses")
            for i, response in enumerate(result_responses):
                print(f"  Response {i}: {len(response.text)} characters")

        return result_responses

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

        Raises:
            RuntimeError: If the BatchGenerator has been closed
            TypeError: If requests is not a list of BatchRequest objects
            ValueError: If requests list is empty
        """
        self._check_closed()

        if requests is None:
            raise TypeError("requests cannot be None, expected a list of BatchRequest objects")

        if not isinstance(requests, list):
            raise TypeError(
                f"requests must be a list of BatchRequest objects, got {type(requests).__name__}"
            )

        if not requests:
            raise ValueError(
                "requests list cannot be empty. Provide at least one BatchRequest."
            )

        # Validate all requests are BatchRequest objects
        for i, req in enumerate(requests):
            if not isinstance(req, BatchRequest):
                raise TypeError(
                    f"All requests must be BatchRequest objects, but request at index {i} "
                    f"is {type(req).__name__}"
                )

        prompts = [req.prompt for req in requests]
        responses = self.generate_batch(prompts, config)

        # Create detailed responses from Response objects
        results = []
        for req, response in zip(requests, responses):
            result = BatchResponse(
                id=req.id,
                prompt=req.prompt,
                response=response.text,
                tokens_generated=response.stats.generated_tokens if response.stats else 0,
                time_taken=response.stats.total_time if response.stats else 0.0
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
) -> List[Response]:
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
        List of Response objects. Each Response can be used as a string.

    Example:
        >>> prompts = ["Hello", "Hi", "Hey"]
        >>> responses = batch_generate(prompts, "models/llama.gguf")
        >>> for r in responses:
        >>>     print(r)  # Works like a string
        >>>     print(r.stats)  # Access statistics
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
