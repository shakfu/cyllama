"""
High-Level API for cyllama

This module provides the primary user-facing API for cyllama, including
both synchronous and asynchronous interfaces.

Example:
    >>> from cyllama import complete, LLM
    >>>
    >>> # Simple completion
    >>> response = complete("What is 2+2?", model_path="models/llama.gguf")
    >>> print(response)
    >>>
    >>> # Streaming completion
    >>> for chunk in complete("Tell me a story", model_path="models/llama.gguf", stream=True):
    >>>     print(chunk, end="", flush=True)
    >>>
    >>> # Using the LLM class
    >>> llm = LLM("models/llama.gguf")
    >>> response = llm("What is Python?")

Async Example:
    >>> import asyncio
    >>> from cyllama import complete_async, AsyncLLM
    >>>
    >>> async def main():
    >>>     # Simple async completion
    >>>     response = await complete_async("What is 2+2?", model_path="model.gguf")
    >>>     print(response)
    >>>
    >>>     # Using AsyncLLM class
    >>>     async with AsyncLLM("model.gguf") as llm:
    >>>         response = await llm("What is Python?")
    >>>         print(response)
    >>>
    >>>         # Async streaming
    >>>         async for chunk in llm.stream("Tell me a story"):
    >>>             print(chunk, end="", flush=True)
    >>>
    >>> asyncio.run(main())
"""

import asyncio
from typing import (
    AsyncIterator,
    Iterator,
    Optional,
    Dict,
    Any,
    List,
    Callable,
    Union,
    Tuple,
)
from dataclasses import dataclass, field
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
    llama_batch_get_one,
    ggml_backend_load_all,
    disable_logging,
)


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.

    Attributes:
        max_tokens: Maximum number of tokens to generate (default: 512)
        temperature: Sampling temperature, 0.0 = greedy (default: 0.8)
        top_k: Top-k sampling parameter (default: 40)
        top_p: Top-p (nucleus) sampling parameter (default: 0.95)
        min_p: Minimum probability threshold (default: 0.05)
        repeat_penalty: Penalty for repeating tokens (default: 1.1)
        n_gpu_layers: Number of layers to offload to GPU (default: 99)
        n_ctx: Context window size, None = auto (default: None)
        n_batch: Batch size for processing (default: 512)
        seed: Random seed for reproducibility, -1 = random (default: -1)
        stop_sequences: List of strings that stop generation (default: [])
        add_bos: Add beginning-of-sequence token (default: True)
        parse_special: Parse special tokens in prompt (default: True)

    Raises:
        ValueError: If any parameter is outside its valid range.
    """
    max_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    n_gpu_layers: int = 99
    n_ctx: Optional[int] = None
    n_batch: int = 512
    seed: int = -1
    stop_sequences: List[str] = field(default_factory=list)
    add_bos: bool = True
    parse_special: bool = True

    def __post_init__(self):
        """Validate parameters after initialization."""
        errors = []

        if self.max_tokens < 0:
            errors.append(f"max_tokens must be >= 0, got {self.max_tokens}")

        if self.temperature < 0.0:
            errors.append(f"temperature must be >= 0.0, got {self.temperature}")

        if self.top_k < 0:
            errors.append(f"top_k must be >= 0, got {self.top_k}")

        if not 0.0 <= self.top_p <= 1.0:
            errors.append(f"top_p must be between 0.0 and 1.0, got {self.top_p}")

        if not 0.0 <= self.min_p <= 1.0:
            errors.append(f"min_p must be between 0.0 and 1.0, got {self.min_p}")

        if self.repeat_penalty < 0.0:
            errors.append(f"repeat_penalty must be >= 0.0, got {self.repeat_penalty}")

        if self.n_gpu_layers < 0:
            errors.append(f"n_gpu_layers must be >= 0, got {self.n_gpu_layers}")

        if self.n_ctx is not None and self.n_ctx < 1:
            errors.append(f"n_ctx must be >= 1 or None, got {self.n_ctx}")

        if self.n_batch < 1:
            errors.append(f"n_batch must be >= 1, got {self.n_batch}")

        if self.seed < -1:
            errors.append(f"seed must be >= -1, got {self.seed}")

        if errors:
            raise ValueError("Invalid GenerationConfig: " + "; ".join(errors))


@dataclass
class GenerationStats:
    """Statistics from a generation run."""
    prompt_tokens: int
    generated_tokens: int
    total_time: float
    tokens_per_second: float
    prompt_time: float = 0.0
    generation_time: float = 0.0


class LLM:
    """
    High-level LLM interface with model caching and convenient API.

    This class manages model lifecycle and provides simple methods for
    text generation with streaming support. It supports context reuse
    for improved performance when the context size doesn't change.

    Resource Management:
        The LLM class manages GPU memory and contexts. For proper cleanup:
        - Use as a context manager: `with LLM(...) as llm:`
        - Call `llm.close()` explicitly when done
        - Or let Python's garbage collector handle it via `__del__`

    Example:
        >>> # Simple usage with direct parameters
        >>> with LLM("models/llama.gguf", temperature=0.9, max_tokens=100) as llm:
        >>>     response = llm("What is Python?")
        >>>     print(response)
        >>>
        >>> # Streaming output
        >>> with LLM("models/llama.gguf") as llm:
        >>>     for chunk in llm("Tell me a joke", stream=True):
        >>>         print(chunk, end="")
        >>>
        >>> # With explicit GenerationConfig (for reuse or complex configs)
        >>> config = GenerationConfig(temperature=0.9, max_tokens=100)
        >>> with LLM("models/llama.gguf", config=config) as llm:
        >>>     response = llm("Hello!")
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[GenerationConfig] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize generator with a model.

        Args:
            model_path: Path to GGUF model file
            config: Generation configuration (uses defaults if None)
            verbose: Print detailed information during generation
            **kwargs: Generation parameters (temperature, max_tokens, etc.)
                      These override values in config if both are provided.

        Example:
            >>> # Direct parameters (recommended for simple cases)
            >>> llm = LLM("model.gguf", temperature=0.9, max_tokens=100)
            >>>
            >>> # Explicit config
            >>> config = GenerationConfig(temperature=0.9)
            >>> llm = LLM("model.gguf", config=config)
            >>>
            >>> # Config with overrides
            >>> llm = LLM("model.gguf", config=config, temperature=0.5)
        """
        self.model_path = model_path
        self.verbose = verbose
        self._closed = False

        # Build config: start with provided config or defaults, then apply kwargs
        if config is None:
            if kwargs:
                self.config = GenerationConfig(**kwargs)
            else:
                self.config = GenerationConfig()
        else:
            if kwargs:
                # Create a copy of config with kwargs overrides
                config_dict = {
                    'max_tokens': config.max_tokens,
                    'temperature': config.temperature,
                    'top_k': config.top_k,
                    'top_p': config.top_p,
                    'min_p': config.min_p,
                    'repeat_penalty': config.repeat_penalty,
                    'n_gpu_layers': config.n_gpu_layers,
                    'n_ctx': config.n_ctx,
                    'n_batch': config.n_batch,
                    'seed': config.seed,
                    'stop_sequences': config.stop_sequences.copy(),
                    'add_bos': config.add_bos,
                    'parse_special': config.parse_special,
                }
                config_dict.update(kwargs)
                self.config = GenerationConfig(**config_dict)
            else:
                self.config = config

        # Disable llama.cpp logging unless verbose mode is enabled
        if not verbose:
            disable_logging()

        # Load backends
        ggml_backend_load_all()

        # Initialize model
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = self.config.n_gpu_layers

        if self.verbose:
            print(f"Loading model: {model_path}")

        self.model = LlamaModel(model_path, model_params)
        self.vocab = self.model.get_vocab()

        if self.verbose:
            print(f"Model loaded: {self.model.n_params} parameters")
            print(f"Vocabulary size: {self.vocab.n_vocab}")

        # Context will be created on-demand and cached when possible
        self._ctx = None
        self._ctx_size = 0  # Track current context size for reuse decisions
        self._sampler = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False

    def __del__(self):
        """Destructor - cleanup resources if not already done."""
        if not getattr(self, '_closed', True):
            self.close()

    def close(self):
        """
        Explicitly release resources (context, sampler).

        This method frees the context and sampler to release GPU memory.
        The model remains loaded for potential reuse. Call this when you're
        done with generation or want to free memory.

        After calling close(), the LLM instance can still be used - new
        contexts will be created as needed.
        """
        if getattr(self, '_closed', True):
            return

        if getattr(self, 'verbose', False):
            print("Closing LLM resources")

        # Release context and sampler (use getattr for safety in __del__)
        if getattr(self, '_ctx', None) is not None:
            self._ctx = None
            self._ctx_size = 0

        if getattr(self, '_sampler', None) is not None:
            self._sampler = None

        self._closed = True

    def reset_context(self):
        """
        Force recreation of context on next generation.

        This clears the KV cache and ensures a fresh context is created.
        Useful when you want to start a completely new conversation without
        any prior context.
        """
        if self._ctx is not None:
            if self.verbose:
                print("Resetting context")
            self._ctx = None
            self._ctx_size = 0

    def _ensure_context(self, prompt_length: int, config: GenerationConfig):
        """
        Create or recreate context if needed.

        Context is reused when possible to avoid allocation overhead.
        A new context is created when:
        - No context exists
        - The required size exceeds current context size
        - The instance was closed (will reopen)

        The KV cache is cleared via llama_kv_cache_clear() when reusing
        a context to ensure clean state for new generations.
        """
        # Reopen if closed
        if self._closed:
            self._closed = False

        # Calculate required context size
        if config.n_ctx is None:
            required_ctx = prompt_length + config.max_tokens
        else:
            required_ctx = config.n_ctx

        # Check if we can reuse existing context
        if self._ctx is not None and self._ctx_size >= required_ctx:
            # Reuse existing context - just clear the KV cache
            if self.verbose:
                print(f"Reusing context (size {self._ctx_size}, need {required_ctx})")
            self._ctx.kv_cache_clear()
            return

        # Need to create new context (either none exists or too small)
        if self.verbose:
            if self._ctx is not None:
                print(f"Recreating context: {self._ctx_size} -> {required_ctx} tokens")
            else:
                print(f"Creating context: {required_ctx} tokens")

        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = required_ctx
        ctx_params.n_batch = config.n_batch
        ctx_params.no_perf = not self.verbose

        # Note: Seed is set in sampler, not context
        self._ctx = LlamaContext(self.model, ctx_params)
        self._ctx_size = required_ctx

    def _ensure_sampler(self, config: GenerationConfig):
        """Create or recreate sampler if needed."""
        # Always create fresh sampler to respect new config
        sampler_params = LlamaSamplerChainParams()
        sampler_params.no_perf = not self.verbose

        self._sampler = LlamaSampler(sampler_params)

        # Add sampling methods based on config
        if config.temperature == 0.0:
            # Greedy sampling
            self._sampler.add_greedy()
        else:
            # Probabilistic sampling
            self._sampler.add_min_p(config.min_p, 1)
            self._sampler.add_top_k(config.top_k)
            self._sampler.add_top_p(config.top_p, 1)
            self._sampler.add_temp(config.temperature)

            # Distribution sampler
            if config.seed != -1:
                self._sampler.add_dist(config.seed)
            else:
                self._sampler.add_dist(int(time.time()))

    def __call__(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None
    ) -> Union[str, Iterator[str]]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            config: Generation configuration (uses instance config if None)
            stream: If True, return iterator of text chunks
            on_token: Optional callback called for each generated token

        Returns:
            Generated text (str) or iterator of text chunks if stream=True
        """
        if stream:
            return self._generate_stream(prompt, config, on_token)
        else:
            return self._generate(prompt, config, on_token)

    def _generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        on_token: Optional[Callable[[str], None]] = None
    ) -> str:
        """Non-streaming generation."""
        chunks = list(self._generate_stream(prompt, config, on_token))
        return "".join(chunks)

    def _generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        on_token: Optional[Callable[[str], None]] = None
    ) -> Iterator[str]:
        """
        Internal streaming generation implementation.

        Yields:
            Text chunks as they are generated
        """
        # Use provided config or fall back to instance config
        config = config or self.config

        # Tokenize prompt
        prompt_tokens = self.vocab.tokenize(
            prompt,
            add_special=config.add_bos,
            parse_special=config.parse_special
        )
        n_prompt = len(prompt_tokens)

        if self.verbose:
            print(f"Prompt tokens: {n_prompt}")

        # Ensure context and sampler are ready
        # Always recreate sampler to ensure fresh state
        self._ensure_context(n_prompt, config)
        self._ensure_sampler(config)

        # Process prompt in batches to avoid exceeding n_batch limit
        n_batch = config.n_batch
        for i in range(0, n_prompt, n_batch):
            batch_tokens = prompt_tokens[i:i + n_batch]
            batch = llama_batch_get_one(batch_tokens, i)  # Pass position offset
            self._ctx.decode(batch)

        # Generate tokens
        n_pos = n_prompt
        n_generated = 0

        # Stop sequence handling: buffer recent output to detect sequences spanning tokens
        # We only need to buffer enough to detect the longest stop sequence
        stop_buffer = ""
        max_stop_len = max(len(s) for s in config.stop_sequences) if config.stop_sequences else 0

        for _ in range(config.max_tokens):
            # Sample next token
            new_token_id = self._sampler.sample(self._ctx, -1)

            # Check for end of generation
            if self.vocab.is_eog(new_token_id):
                break

            # Decode token to text
            try:
                piece = self.vocab.token_to_piece(new_token_id, special=True)
            except UnicodeDecodeError:
                logger.warning("Failed to decode token %d: UnicodeDecodeError", new_token_id)
                piece = ""

            # Handle stop sequences
            if config.stop_sequences:
                # Add piece to buffer and check for stop sequences
                stop_buffer += piece

                # Find earliest stop sequence in buffer
                stop_pos, stop_len = self._find_stop_sequence(stop_buffer, config.stop_sequences)

                if stop_pos is not None:
                    # Stop sequence found - yield text before it and stop
                    text_before_stop = stop_buffer[:stop_pos]
                    if text_before_stop:
                        if on_token:
                            on_token(text_before_stop)
                        yield text_before_stop
                    # Clear buffer to prevent flush at end
                    stop_buffer = ""
                    break

                # No stop found yet - yield text that can't be part of a stop sequence
                # Keep (max_stop_len - 1) characters to detect sequences spanning tokens
                # Example: if max_stop_len=2 and buffer="abc", safe to yield "ab", keep "c"
                chars_to_keep = max_stop_len - 1
                safe_len = len(stop_buffer) - chars_to_keep
                if safe_len > 0:
                    safe_text = stop_buffer[:safe_len]
                    stop_buffer = stop_buffer[safe_len:]
                    if on_token:
                        on_token(safe_text)
                    yield safe_text
            else:
                # No stop sequences - yield immediately
                if on_token:
                    on_token(piece)
                yield piece

            # Prepare next batch
            batch = llama_batch_get_one([new_token_id], n_pos)
            self._ctx.decode(batch)

            n_pos += 1
            n_generated += 1

        # Flush remaining buffer (no stop sequence found)
        if config.stop_sequences and stop_buffer:
            if on_token:
                on_token(stop_buffer)
            yield stop_buffer

        if self.verbose:
            print(f"\nGenerated {n_generated} tokens")
            self._sampler.print_perf_data()
            self._ctx.print_perf_data()

    def _find_stop_sequence(
        self,
        text: str,
        stop_sequences: List[str]
    ) -> Tuple[Optional[int], int]:
        """
        Find the earliest stop sequence in text.

        Args:
            text: Text to search
            stop_sequences: List of stop sequences to look for

        Returns:
            Tuple of (position, length) where position is the start index of the
            earliest stop sequence found, or (None, 0) if none found.
        """
        earliest_pos = None
        earliest_len = 0

        for stop in stop_sequences:
            pos = text.find(stop)
            if pos != -1:
                # Found a stop sequence - keep track of earliest one
                if earliest_pos is None or pos < earliest_pos:
                    earliest_pos = pos
                    earliest_len = len(stop)

        return earliest_pos, earliest_len

    def generate_with_stats(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Tuple[str, GenerationStats]:
        """
        Generate text and return detailed statistics.

        Args:
            prompt: Input text prompt
            config: Generation configuration

        Returns:
            Tuple of (generated_text, statistics)
        """
        config = config or self.config

        # Tokenize for stats
        prompt_tokens = self.vocab.tokenize(
            prompt,
            add_special=config.add_bos,
            parse_special=config.parse_special
        )

        start_time = time.time()

        # Generate
        response = self._generate(prompt, config)

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate stats (approximate token count)
        response_tokens = self.vocab.tokenize(response, add_special=False, parse_special=False)
        n_generated = len(response_tokens)

        stats = GenerationStats(
            prompt_tokens=len(prompt_tokens),
            generated_tokens=n_generated,
            total_time=total_time,
            tokens_per_second=n_generated / total_time if total_time > 0 else 0.0
        )

        return response, stats


# Convenience functions

def complete(
    prompt: str,
    model_path: str,
    config: Optional[GenerationConfig] = None,
    stream: bool = False,
    verbose: bool = False,
    **kwargs
) -> Union[str, Iterator[str]]:
    """
    Convenience function for one-off text completion.

    For repeated completions, use the LLM class for better performance.

    Args:
        prompt: Input text prompt
        model_path: Path to GGUF model file
        config: Generation configuration
        stream: If True, return iterator of text chunks
        verbose: Enable detailed logging from llama.cpp
        **kwargs: Additional config parameters (override config values)

    Returns:
        Generated text or iterator if stream=True

    Example:
        >>> response = complete("Hello", model_path="models/llama.gguf")
        >>> print(response)
        >>>
        >>> # With custom parameters
        >>> response = complete(
        >>>     "Tell me a joke",
        >>>     model_path="models/llama.gguf",
        >>>     temperature=0.9,
        >>>     max_tokens=100
        >>> )
    """
    # Merge config with kwargs
    if config is None:
        config = GenerationConfig(**kwargs)
    else:
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    llm = LLM(model_path, config=config, verbose=verbose)
    return llm(prompt, stream=stream)


def chat(
    messages: List[Dict[str, str]],
    model_path: str,
    config: Optional[GenerationConfig] = None,
    stream: bool = False,
    verbose: bool = False,
    **kwargs
) -> Union[str, Iterator[str]]:
    """
    Convenience function for chat-style generation.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model_path: Path to GGUF model file
        config: Generation configuration
        stream: If True, return iterator of text chunks
        verbose: Enable detailed logging from llama.cpp
        **kwargs: Additional config parameters

    Returns:
        Generated response text or iterator if stream=True

    Example:
        >>> messages = [
        >>>     {"role": "system", "content": "You are a helpful assistant."},
        >>>     {"role": "user", "content": "What is Python?"}
        >>> ]
        >>> response = chat(messages, model_path="models/llama.gguf")
    """
    # Format messages into a prompt (simple implementation)
    # More sophisticated implementations would use model-specific chat templates
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt_parts.append(f"{role.capitalize()}: {content}")

    prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"

    return complete(prompt, model_path, config, stream, verbose=verbose, **kwargs)


def simple(model_path: str, prompt: str, ngl: int = 99, n_predict: int = 32, n_ctx: Optional[int] = None, verbose: bool = False) -> bool:
    """
    Simple, educational example showing raw llama.cpp usage.

    This function demonstrates how to use llama.cpp primitives directly
    without the abstractions provided by LLM or complete().

    Args:
        model_path: Path to GGUF model file
        prompt: Input text prompt
        ngl: Number of GPU layers (default: 99)
        n_predict: Number of tokens to generate (default: 32)
        n_ctx: Context size (default: auto-calculated)
        verbose: Enable llama.cpp logging (default: False)

    Returns:
        True if successful
    """
    from .llama import llama_cpp as cy

    # load dynamic backends

    if not verbose:
        cy.disable_logging()

    cy.ggml_backend_load_all()

    # initialize the model

    model_params = cy.LlamaModelParams()
    model_params.n_gpu_layers = ngl

    model = cy.LlamaModel(model_path, model_params)
    vocab = model.get_vocab()

    # tokenize the prompt
    print(f"vocab.n_vocab = {vocab.n_vocab}")

    # find the number of tokens in the prompt
    prompt_tokens = vocab.tokenize(prompt, add_special=True, parse_special=True)
    n_prompt = len(prompt_tokens)
    print(f"n_prompt: {n_prompt}")

    # initialize the context

    ctx_params = cy.LlamaContextParams()
    # n_ctx is the context size
    if n_ctx is not None:
        ctx_params.n_ctx = n_ctx
    else:
        ctx_params.n_ctx = n_prompt + n_predict - 1
    # n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt
    # enable performance counters
    ctx_params.no_perf = False

    ctx = cy.LlamaContext(model, ctx_params)

    # initialize the sampler

    sparams = cy.LlamaSamplerChainParams()
    sparams.no_perf = False

    smplr = cy.LlamaSampler(sparams)
    smplr.add_greedy()

    # print the prompt token-by-token
    print()
    prompt=""
    for i in prompt_tokens:
        try:
            prompt += vocab.token_to_piece(i, lstrip=0, special=False)
        except UnicodeDecodeError:
            continue
    print(prompt)

    # prepare a batch for the prompt
    batch = cy.llama_batch_get_one(prompt_tokens)

    # main loop
    t_main_start: int = cy.ggml_time_us()
    n_decode = 0

    n_pos = n_prompt
    response = ""
    for i in range(n_predict):

        ctx.decode(batch)

        # sample the next token
        new_token_id = smplr.sample(ctx, -1)

        # is it an end of generation?
        if vocab.is_eog(new_token_id):
            break

        piece: str = vocab.token_to_piece(new_token_id, special=True)
        response += piece

        # prepare the next batch with the sampled token
        batch = cy.llama_batch_get_one([new_token_id], n_pos)
        n_pos += 1

        n_decode += 1

    print()
    print(f"response: {response}")
    print()

    t_main_end: int = cy.ggml_time_us()

    print("decoded %d tokens in %.2f s, speed: %.2f t/s" %
            (n_decode, (t_main_end - t_main_start) / 1000000.0, n_decode / ((t_main_end - t_main_start) / 1000000.0)))
    print()

    smplr.print_perf_data()
    ctx.print_perf_data()

    return True


# =============================================================================
# Async API
# =============================================================================


class AsyncLLM:
    """
    Async wrapper around the LLM class for non-blocking text generation.

    This class provides an async interface to the synchronous LLM operations.
    Inference runs in a thread pool to avoid blocking the event loop, making
    it suitable for use in async web frameworks like FastAPI, aiohttp, etc.

    Note: The underlying model is still synchronous - this wrapper just moves
    the blocking operations off the main event loop. For true parallelism with
    multiple requests, use multiple AsyncLLM instances or batch processing.

    Resource Management:
        Use as an async context manager for proper cleanup:
        - `async with AsyncLLM(...) as llm:`

    Example:
        >>> async def main():
        >>>     # Simple usage with direct parameters
        >>>     async with AsyncLLM("model.gguf", temperature=0.9) as llm:
        >>>         response = await llm("What is Python?")
        >>>         print(response)
        >>>
        >>>         # Async streaming
        >>>         async for chunk in llm.stream("Tell me a joke"):
        >>>             print(chunk, end="", flush=True)
        >>>
        >>> asyncio.run(main())
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[GenerationConfig] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize async generator with a model.

        Args:
            model_path: Path to GGUF model file
            config: Generation configuration (uses defaults if None)
            verbose: Print detailed information during generation
            **kwargs: Generation parameters (temperature, max_tokens, etc.)
                      These override values in config if both are provided.

        Example:
            >>> # Direct parameters
            >>> llm = AsyncLLM("model.gguf", temperature=0.9, max_tokens=100)
            >>>
            >>> # With config
            >>> config = GenerationConfig(temperature=0.9)
            >>> llm = AsyncLLM("model.gguf", config=config)
        """
        self._llm = LLM(model_path, config=config, verbose=verbose, **kwargs)
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensures cleanup."""
        await self.close()
        return False

    async def close(self):
        """
        Explicitly release resources.

        Runs cleanup in a thread to avoid blocking if cleanup is slow.
        """
        await asyncio.to_thread(self._llm.close)

    async def reset_context(self):
        """Force recreation of context on next generation."""
        await asyncio.to_thread(self._llm.reset_context)

    @property
    def config(self) -> GenerationConfig:
        """Get the current generation config."""
        return self._llm.config

    @property
    def model_path(self) -> str:
        """Get the model path."""
        return self._llm.model_path

    async def __call__(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt asynchronously.

        Args:
            prompt: Input text prompt
            config: Generation configuration (uses instance config if None)
            **kwargs: Override config parameters for this call

        Returns:
            Generated text string

        Example:
            >>> response = await llm("What is the meaning of life?")
            >>> response = await llm("Explain quantum physics", max_tokens=200)
        """
        # Build config with overrides if kwargs provided
        if kwargs:
            effective_config = self._build_config(config, kwargs)
        else:
            effective_config = config

        async with self._lock:
            return await asyncio.to_thread(
                self._llm._generate,
                prompt,
                effective_config
            )

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt asynchronously.

        Alias for __call__ for explicit method name preference.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            **kwargs: Override config parameters

        Returns:
            Generated text string
        """
        return await self(prompt, config, **kwargs)

    async def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream generated text chunks asynchronously.

        Yields text chunks as they are generated. Each chunk is yielded
        as soon as it's available from the underlying model.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            **kwargs: Override config parameters

        Yields:
            Text chunks as they are generated

        Example:
            >>> async for chunk in llm.stream("Tell me a story"):
            >>>     print(chunk, end="", flush=True)
        """
        # Build config with overrides if kwargs provided
        if kwargs:
            effective_config = self._build_config(config, kwargs)
        else:
            effective_config = config

        # Use a queue to bridge sync generator to async iterator
        queue: asyncio.Queue[Union[str, None, Exception]] = asyncio.Queue()

        async def producer():
            """Run sync generator in thread and put items in queue."""
            try:
                def generate_sync():
                    for chunk in self._llm._generate_stream(prompt, effective_config):
                        # Schedule putting item in queue from the thread
                        asyncio.run_coroutine_threadsafe(
                            queue.put(chunk),
                            loop
                        )
                    # Signal completion
                    asyncio.run_coroutine_threadsafe(
                        queue.put(None),
                        loop
                    )

                await asyncio.to_thread(generate_sync)
            except Exception as e:
                await queue.put(e)

        loop = asyncio.get_event_loop()

        # Start producer task
        async with self._lock:
            producer_task = asyncio.create_task(producer())

            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                # Ensure producer completes
                await producer_task

    async def generate_with_stats(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> tuple:
        """
        Generate text and return detailed statistics.

        Args:
            prompt: Input text prompt
            config: Generation configuration

        Returns:
            Tuple of (generated_text, GenerationStats)
        """
        async with self._lock:
            return await asyncio.to_thread(
                self._llm.generate_with_stats,
                prompt,
                config
            )

    def _build_config(
        self,
        base_config: Optional[GenerationConfig],
        overrides: Dict[str, Any]
    ) -> GenerationConfig:
        """Build a config with overrides applied."""
        config = base_config or self._llm.config
        config_dict = {
            'max_tokens': config.max_tokens,
            'temperature': config.temperature,
            'top_k': config.top_k,
            'top_p': config.top_p,
            'min_p': config.min_p,
            'repeat_penalty': config.repeat_penalty,
            'n_gpu_layers': config.n_gpu_layers,
            'n_ctx': config.n_ctx,
            'n_batch': config.n_batch,
            'seed': config.seed,
            'stop_sequences': config.stop_sequences.copy(),
            'add_bos': config.add_bos,
            'parse_special': config.parse_special,
        }
        config_dict.update(overrides)
        return GenerationConfig(**config_dict)


async def complete_async(
    prompt: str,
    model_path: str,
    config: Optional[GenerationConfig] = None,
    verbose: bool = False,
    **kwargs
) -> str:
    """
    Async convenience function for one-off text completion.

    For repeated completions, use the AsyncLLM class for better performance
    (avoids reloading the model each time).

    Args:
        prompt: Input text prompt
        model_path: Path to GGUF model file
        config: Generation configuration
        verbose: Enable detailed logging
        **kwargs: Additional config parameters (temperature, max_tokens, etc.)

    Returns:
        Generated text string

    Example:
        >>> response = await complete_async(
        >>>     "What is Python?",
        >>>     model_path="model.gguf",
        >>>     temperature=0.7
        >>> )
    """
    return await asyncio.to_thread(
        complete,
        prompt,
        model_path,
        config,
        False,  # stream=False
        verbose,
        **kwargs
    )


async def chat_async(
    messages: List[Dict[str, str]],
    model_path: str,
    config: Optional[GenerationConfig] = None,
    verbose: bool = False,
    **kwargs
) -> str:
    """
    Async convenience function for chat-style generation.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model_path: Path to GGUF model file
        config: Generation configuration
        verbose: Enable detailed logging
        **kwargs: Additional config parameters

    Returns:
        Generated response text

    Example:
        >>> messages = [
        >>>     {"role": "system", "content": "You are a helpful assistant."},
        >>>     {"role": "user", "content": "What is Python?"}
        >>> ]
        >>> response = await chat_async(messages, model_path="model.gguf")
    """
    return await asyncio.to_thread(
        chat,
        messages,
        model_path,
        config,
        False,  # stream=False
        verbose,
        **kwargs
    )


async def stream_complete_async(
    prompt: str,
    model_path: str,
    config: Optional[GenerationConfig] = None,
    verbose: bool = False,
    **kwargs
) -> AsyncIterator[str]:
    """
    Async streaming completion for one-off use.

    For repeated completions, use AsyncLLM.stream() for better performance.

    Args:
        prompt: Input text prompt
        model_path: Path to GGUF model file
        config: Generation configuration
        verbose: Enable detailed logging
        **kwargs: Additional config parameters

    Yields:
        Text chunks as they are generated

    Example:
        >>> async for chunk in stream_complete_async("Tell me a story", "model.gguf"):
        >>>     print(chunk, end="", flush=True)
    """
    async with AsyncLLM(model_path, config=config, verbose=verbose, **kwargs) as llm:
        async for chunk in llm.stream(prompt):
            yield chunk
