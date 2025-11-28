"""
High-Level API for cyllama

This module provides the primary user-facing API for cyllama.
It abstracts away the complexity of batches, sampling, and context management.

Example:
    >>> from cyllama import complete
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
"""

from typing import Iterator, Optional, Dict, Any, List, Callable, Union
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

        if self.max_tokens < 1:
            errors.append(f"max_tokens must be >= 1, got {self.max_tokens}")

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
    text generation with streaming support.

    Example:
        >>> llm = LLM("models/llama.gguf")
        >>> response = llm("What is Python?")
        >>> print(response)
        >>>
        >>> # With custom configuration
        >>> config = GenerationConfig(temperature=0.9, max_tokens=100)
        >>> for chunk in llm("Tell me a joke", config=config, stream=True):
        >>>     print(chunk, end="")
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[GenerationConfig] = None,
        verbose: bool = False
    ):
        """
        Initialize generator with a model.

        Args:
            model_path: Path to GGUF model file
            config: Generation configuration (uses defaults if None)
            verbose: Print detailed information during generation
        """
        self.model_path = model_path
        self.config = config or GenerationConfig()
        self.verbose = verbose

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

        # Context will be created on-demand for each generation
        self._ctx = None
        self._sampler = None

    def _ensure_context(self, prompt_length: int, config: GenerationConfig):
        """Create or recreate context if needed."""
        # Calculate required context size
        if config.n_ctx is None:
            n_ctx = prompt_length + config.max_tokens
        else:
            n_ctx = config.n_ctx

        # Always recreate context to clear KV cache
        # This is simpler than trying to manage cache state
        if self.verbose and self._ctx is not None:
            print(f"Recreating context: {n_ctx} tokens")

        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = config.n_batch
        ctx_params.no_perf = not self.verbose

        # Note: Seed is set in sampler, not context
        self._ctx = LlamaContext(self.model, ctx_params)

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

        # Buffer for stop sequence checking (accumulates recent output)
        stop_buffer = ""
        max_stop_len = max((len(s) for s in config.stop_sequences), default=0) if config.stop_sequences else 0

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

            # Check stop sequences against accumulated buffer
            if config.stop_sequences:
                stop_buffer += piece
                # Keep buffer size manageable
                if len(stop_buffer) > max_stop_len * 2:
                    stop_buffer = stop_buffer[-max_stop_len * 2:]

                # Check if any stop sequence appears in the buffer
                stop_found = False
                for stop in config.stop_sequences:
                    if stop in stop_buffer:
                        # Trim output to exclude stop sequence
                        stop_idx = stop_buffer.find(stop)
                        # Calculate how much of current piece to yield (if any)
                        piece_start_in_buffer = len(stop_buffer) - len(piece)
                        if stop_idx > piece_start_in_buffer:
                            # Part of current piece should be yielded
                            partial_len = stop_idx - piece_start_in_buffer
                            if partial_len > 0:
                                partial_piece = piece[:partial_len]
                                if on_token:
                                    on_token(partial_piece)
                                yield partial_piece
                        # else: stop sequence was already in buffer before this piece,
                        # or starts at beginning of piece - don't yield anything
                        stop_found = True
                        break

                if stop_found:
                    break

            # Yield or callback (only if no stop sequence was found)
            if on_token:
                on_token(piece)

            yield piece

            # Prepare next batch
            batch = llama_batch_get_one([new_token_id], n_pos)
            self._ctx.decode(batch)

            n_pos += 1
            n_generated += 1

        if self.verbose:
            print(f"\nGenerated {n_generated} tokens")
            self._sampler.print_perf_data()
            self._ctx.print_perf_data()

    def generate_with_stats(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> tuple[str, GenerationStats]:
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


def simple(model_path: str, prompt: str, ngl: int = 99, n_predict: int = 32, n_ctx: int = None, verbose=False):
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
