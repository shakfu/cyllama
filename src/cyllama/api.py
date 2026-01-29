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
    LlamaChatMessage,
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
        main_gpu: Primary GPU device index for inference (default: 0)
        split_mode: How to split model across GPUs (default: 1 = LAYER)
            0 = NONE: Use single GPU only (main_gpu)
            1 = LAYER: Split layers and KV cache across GPUs
            2 = ROW: Split with tensor parallelism (if supported)
        tensor_split: Proportion of work per GPU (default: None = auto)
            List of floats, one per GPU. Values are normalized by llama.cpp.
            Example: [1, 2] assigns 1/3 to GPU 0 and 2/3 to GPU 1.
        n_ctx: Context window size, None = auto (default: None)
        n_batch: Batch size for processing (default: 512)
        seed: Random seed for reproducibility, -1 = random (default: -1)
        stop_sequences: List of strings that stop generation (default: [])
        add_bos: Add beginning-of-sequence token (default: True)
        parse_special: Parse special tokens in prompt (default: True)

    Raises:
        ValueError: If any parameter is outside its valid range.

    Example:
        >>> # Use GPU 1 as primary device
        >>> config = GenerationConfig(main_gpu=1)
        >>>
        >>> # Multi-GPU with layer splitting
        >>> config = GenerationConfig(split_mode=1, n_gpu_layers=99)
        >>>
        >>> # Multi-GPU with tensor parallelism (row splitting)
        >>> config = GenerationConfig(split_mode=2, n_gpu_layers=99)
        >>>
        >>> # Custom tensor split: 30% GPU 0, 70% GPU 1
        >>> config = GenerationConfig(tensor_split=[0.3, 0.7])
    """
    max_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    n_gpu_layers: int = 99
    main_gpu: int = 0
    split_mode: int = 1
    tensor_split: Optional[List[float]] = None
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

        if self.main_gpu < 0:
            errors.append(f"main_gpu must be >= 0, got {self.main_gpu}")

        if self.split_mode not in (0, 1, 2):
            errors.append(f"split_mode must be 0, 1, or 2, got {self.split_mode}")

        if self.tensor_split is not None:
            if not isinstance(self.tensor_split, list):
                errors.append(f"tensor_split must be a list or None, got {type(self.tensor_split)}")
            elif any(not isinstance(v, (int, float)) or v < 0 for v in self.tensor_split):
                errors.append("tensor_split values must be non-negative numbers")

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


@dataclass
class Response:
    """
    Response from text generation.

    This class wraps generated text with optional metadata and provides
    convenient conversion methods. It implements __str__ for backward
    compatibility, so it can be used anywhere a string is expected.

    Attributes:
        text: The generated text content
        stats: Optional generation statistics (tokens, timing, etc.)
        finish_reason: Why generation stopped ("stop", "length", "error")
        model: Model identifier/path used for generation

    Example:
        >>> response = complete("Hello", model_path="model.gguf")
        >>> print(response)  # Works like a string
        >>> print(response.text)  # Explicit text access
        >>> print(response.to_json())  # JSON output
        >>> data = response.to_dict()  # Dictionary for serialization
    """
    text: str
    stats: Optional[GenerationStats] = None
    finish_reason: str = "stop"
    model: str = ""

    def __str__(self) -> str:
        """Return the text content. Enables backward-compatible string usage."""
        return self.text

    def __repr__(self) -> str:
        """Return a detailed representation."""
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Response(text={text_preview!r}, finish_reason={self.finish_reason!r})"

    def __eq__(self, other) -> bool:
        """Compare with strings or other Response objects."""
        if isinstance(other, str):
            return self.text == other
        if isinstance(other, Response):
            return self.text == other.text
        return NotImplemented

    def __hash__(self) -> int:
        """Hash based on text content."""
        return hash(self.text)

    def __len__(self) -> int:
        """Return length of text content."""
        return len(self.text)

    def __iter__(self):
        """Iterate over characters in text."""
        return iter(self.text)

    def __contains__(self, item) -> bool:
        """Check if substring is in text."""
        return item in self.text

    def __add__(self, other) -> str:
        """Concatenate with strings."""
        if isinstance(other, str):
            return self.text + other
        if isinstance(other, Response):
            return self.text + other.text
        return NotImplemented

    def __radd__(self, other) -> str:
        """Support string + Response."""
        if isinstance(other, str):
            return other + self.text
        return NotImplemented

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert response to dictionary.

        Returns:
            Dictionary containing all response data.

        Example:
            >>> response = complete("Hello", model_path="model.gguf")
            >>> data = response.to_dict()
            >>> print(data["text"])
        """
        result: Dict[str, Any] = {
            "text": self.text,
            "finish_reason": self.finish_reason,
            "model": self.model,
        }
        if self.stats is not None:
            result["stats"] = {
                "prompt_tokens": self.stats.prompt_tokens,
                "generated_tokens": self.stats.generated_tokens,
                "total_time": self.stats.total_time,
                "tokens_per_second": self.stats.tokens_per_second,
                "prompt_time": self.stats.prompt_time,
                "generation_time": self.stats.generation_time,
            }
        return result

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Convert response to JSON string.

        Args:
            indent: JSON indentation level (None for compact)

        Returns:
            JSON string representation.

        Example:
            >>> response = complete("Hello", model_path="model.gguf")
            >>> print(response.to_json(indent=2))
        """
        import json
        return json.dumps(self.to_dict(), indent=indent)


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
                    'main_gpu': config.main_gpu,
                    'split_mode': config.split_mode,
                    'tensor_split': config.tensor_split.copy() if config.tensor_split else None,
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
        model_params.main_gpu = self.config.main_gpu
        model_params.split_mode = self.config.split_mode
        if self.config.tensor_split is not None:
            model_params.tensor_split = self.config.tensor_split

        if self.verbose:
            print(f"Loading model: {model_path}")
            gpu_info = (f"GPU config: n_gpu_layers={self.config.n_gpu_layers}, "
                        f"main_gpu={self.config.main_gpu}, split_mode={self.config.split_mode}")
            if self.config.tensor_split:
                gpu_info += f", tensor_split={self.config.tensor_split}"
            print(gpu_info)

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
    ) -> Union[Response, Iterator[str]]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            config: Generation configuration (uses instance config if None)
            stream: If True, return iterator of text chunks
            on_token: Optional callback called for each generated token

        Returns:
            Response object (if stream=False) or iterator of text chunks (if stream=True).
            The Response object can be used as a string due to __str__ implementation.
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
    ) -> Response:
        """Non-streaming generation returning Response object."""
        config = config or self.config
        start_time = time.time()

        # Tokenize for stats
        prompt_tokens = self.vocab.tokenize(
            prompt,
            add_special=config.add_bos,
            parse_special=config.parse_special
        )
        n_prompt = len(prompt_tokens)

        # Generate text
        chunks = list(self._generate_stream(prompt, config, on_token))
        text = "".join(chunks)

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate stats
        response_tokens = self.vocab.tokenize(text, add_special=False, parse_special=False)
        n_generated = len(response_tokens)

        stats = GenerationStats(
            prompt_tokens=n_prompt,
            generated_tokens=n_generated,
            total_time=total_time,
            tokens_per_second=n_generated / total_time if total_time > 0 else 0.0
        )

        return Response(
            text=text,
            stats=stats,
            finish_reason="stop",
            model=self.model_path
        )

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
    ) -> Response:
        """
        Generate text and return Response with detailed statistics.

        This method is now equivalent to __call__ since Response always
        includes stats. Kept for backward compatibility.

        Args:
            prompt: Input text prompt
            config: Generation configuration

        Returns:
            Response object with text and statistics
        """
        return self._generate(prompt, config)

    def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
        template: Optional[str] = None,
    ) -> Union[Response, Iterator[str]]:
        """
        Generate a response from chat messages using the model's chat template.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            config: Generation configuration (uses instance config if None)
            stream: If True, return iterator of text chunks
            template: Custom chat template name (e.g., "llama3", "chatml").
                      If None, uses the model's default template.

        Returns:
            Response object (if stream=False) or iterator of text chunks (if stream=True)

        Example:
            >>> messages = [
            >>>     {"role": "system", "content": "You are helpful."},
            >>>     {"role": "user", "content": "Hello!"}
            >>> ]
            >>> response = llm.chat(messages)
            >>> print(response)  # Works like a string
            >>> print(response.stats.tokens_per_second)  # Access stats
        """
        prompt = self._apply_template(messages, template)
        return self(prompt, config=config, stream=stream)

    def _apply_template(
        self,
        messages: List[Dict[str, str]],
        template: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply chat template to messages using the loaded model."""
        # Get template - use provided or model's default
        if template:
            tmpl = self.model.get_default_chat_template_by_name(template)
            if not tmpl:
                tmpl = template
        else:
            tmpl = self.model.get_default_chat_template()

        if tmpl:
            chat_messages = [
                LlamaChatMessage(role=msg.get("role", "user"), content=msg.get("content", ""))
                for msg in messages
            ]
            return self.model.chat_apply_template(tmpl, chat_messages, add_generation_prompt)
        else:
            return _format_messages_simple(messages)

    def get_chat_template(self, template_name: Optional[str] = None) -> str:
        """
        Get the chat template string from the loaded model.

        Args:
            template_name: Optional specific template name to retrieve

        Returns:
            Template string, or empty string if not found
        """
        if template_name:
            return self.model.get_default_chat_template_by_name(template_name)
        return self.model.get_default_chat_template()


# Convenience functions

def complete(
    prompt: str,
    model_path: str,
    config: Optional[GenerationConfig] = None,
    stream: bool = False,
    verbose: bool = False,
    **kwargs
) -> Union[Response, Iterator[str]]:
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
        Response object (if stream=False) or iterator of text chunks (if stream=True).
        The Response can be used as a string: print(response), str(response), etc.

    Example:
        >>> response = complete("Hello", model_path="models/llama.gguf")
        >>> print(response)  # Works like a string
        >>> print(response.text)  # Explicit text access
        >>> print(response.stats.tokens_per_second)  # Access stats
        >>> print(response.to_json())  # JSON output
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
    template: Optional[str] = None,
    **kwargs
) -> Union[Response, Iterator[str]]:
    """
    Convenience function for chat-style generation.

    Uses the model's built-in chat template if available, otherwise falls back
    to a simple format. You can also specify a custom template.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model_path: Path to GGUF model file
        config: Generation configuration
        stream: If True, return iterator of text chunks
        verbose: Enable detailed logging from llama.cpp
        template: Custom chat template name (e.g., "llama3", "chatml", "mistral").
                  If None, uses the model's default template.
                  See llama.cpp wiki for supported templates.
        **kwargs: Additional config parameters

    Returns:
        Response object (if stream=False) or iterator of text chunks (if stream=True)

    Example:
        >>> messages = [
        >>>     {"role": "system", "content": "You are a helpful assistant."},
        >>>     {"role": "user", "content": "What is Python?"}
        >>> ]
        >>> response = chat(messages, model_path="models/llama.gguf")
        >>> print(response)  # Works like a string
        >>> print(response.stats)  # Access statistics
        >>>
        >>> # With explicit template
        >>> response = chat(messages, model_path="models/llama.gguf", template="chatml")
    """
    prompt = apply_chat_template(messages, model_path, template, verbose=verbose)
    return complete(prompt, model_path, config, stream, verbose=verbose, **kwargs)


def apply_chat_template(
    messages: List[Dict[str, str]],
    model_path: str,
    template: Optional[str] = None,
    add_generation_prompt: bool = True,
    verbose: bool = False,
) -> str:
    """
    Apply a chat template to format messages into a prompt string.

    Uses the model's built-in chat template from its GGUF metadata. If no template
    is found, falls back to a simple User/Assistant format.

    Supported templates (built into llama.cpp):
        - llama2, llama3
        - chatml (used by many models including Qwen, Yi, etc.)
        - mistral, mistral-v1, mistral-v3, mistral-v3-tekken, mistral-v7
        - phi3, phi4
        - falcon3
        - deepseek, deepseek2, deepseek3
        - command-r
        - vicuna, vicuna-orca
        - zephyr
        - gemma, gemma2
        - orion
        - openchat
        - monarch
        - exaone3
        - granite
        - gigachat
        - megrez

    See: https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
                  Supported roles: 'system', 'user', 'assistant'
        model_path: Path to GGUF model file
        template: Optional template name to use instead of model's default.
                  If None, uses the model's built-in template.
        add_generation_prompt: If True, adds the assistant prompt prefix
        verbose: Enable detailed logging

    Returns:
        Formatted prompt string ready for generation

    Example:
        >>> messages = [
        >>>     {"role": "system", "content": "You are helpful."},
        >>>     {"role": "user", "content": "Hello!"}
        >>> ]
        >>> prompt = apply_chat_template(messages, "model.gguf")
        >>> print(prompt)
        <|im_start|>system
        You are helpful.<|im_end|>
        <|im_start|>user
        Hello!<|im_end|>
        <|im_start|>assistant
    """
    if not verbose:
        disable_logging()

    ggml_backend_load_all()

    # Load model to get template
    model_params = LlamaModelParams()
    model_params.n_gpu_layers = 0  # Don't load to GPU, just need metadata
    model = LlamaModel(model_path, model_params)

    # Get template - use provided or model's default
    if template:
        tmpl = model.get_default_chat_template_by_name(template)
        if not tmpl:
            # Try as-is (some templates are the actual template string)
            tmpl = template
    else:
        tmpl = model.get_default_chat_template()

    if tmpl:
        # Convert messages to LlamaChatMessage objects
        chat_messages = [
            LlamaChatMessage(role=msg.get("role", "user"), content=msg.get("content", ""))
            for msg in messages
        ]
        prompt = model.chat_apply_template(tmpl, chat_messages, add_generation_prompt)
    else:
        # Fallback to simple format if no template available
        logger.debug("No chat template found, using fallback format")
        prompt = _format_messages_simple(messages)

    # Model cleanup handled by garbage collection
    del model
    return prompt


def _format_messages_simple(messages: List[Dict[str, str]]) -> str:
    """Simple fallback format when no chat template is available."""
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
        else:
            prompt_parts.append(f"{role.capitalize()}: {content}")

    return "\n\n".join(prompt_parts) + "\n\nAssistant:"


def get_chat_template(model_path: str, template_name: Optional[str] = None) -> str:
    """
    Get the chat template string from a model.

    Args:
        model_path: Path to GGUF model file
        template_name: Optional specific template name to retrieve

    Returns:
        Template string, or empty string if not found

    Example:
        >>> template = get_chat_template("models/llama.gguf")
        >>> print(template)  # Shows the Jinja-style template
    """
    disable_logging()
    ggml_backend_load_all()

    model_params = LlamaModelParams()
    model_params.n_gpu_layers = 0
    model = LlamaModel(model_path, model_params)

    if template_name:
        result = model.get_default_chat_template_by_name(template_name)
    else:
        result = model.get_default_chat_template()

    # Model cleanup handled by garbage collection
    del model
    return result


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
    ) -> Response:
        """
        Generate text from a prompt asynchronously.

        Args:
            prompt: Input text prompt
            config: Generation configuration (uses instance config if None)
            **kwargs: Override config parameters for this call

        Returns:
            Response object with text and statistics

        Example:
            >>> response = await llm("What is the meaning of life?")
            >>> print(response)  # Works like a string
            >>> print(response.stats.tokens_per_second)  # Access stats
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
    ) -> Response:
        """
        Generate text from a prompt asynchronously.

        Alias for __call__ for explicit method name preference.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            **kwargs: Override config parameters

        Returns:
            Response object with text and statistics
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
    ) -> Response:
        """
        Generate text and return Response with detailed statistics.

        This method is now equivalent to __call__ since Response always
        includes stats. Kept for backward compatibility.

        Args:
            prompt: Input text prompt
            config: Generation configuration

        Returns:
            Response object with text and statistics
        """
        async with self._lock:
            return await asyncio.to_thread(
                self._llm.generate_with_stats,
                prompt,
                config
            )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
        template: Optional[str] = None,
    ) -> Response:
        """
        Generate a response from chat messages using the model's chat template.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            config: Generation configuration (uses instance config if None)
            template: Custom chat template name (e.g., "llama3", "chatml").
                      If None, uses the model's default template.

        Returns:
            Response object with text and statistics

        Example:
            >>> messages = [
            >>>     {"role": "system", "content": "You are helpful."},
            >>>     {"role": "user", "content": "Hello!"}
            >>> ]
            >>> response = await llm.chat(messages)
            >>> print(response)  # Works like a string
            >>> print(response.stats)  # Access statistics
        """
        async with self._lock:
            return await asyncio.to_thread(
                self._llm.chat,
                messages,
                config,
                False,  # stream=False
                template,
            )

    def get_chat_template(self, template_name: Optional[str] = None) -> str:
        """
        Get the chat template string from the loaded model.

        Args:
            template_name: Optional specific template name to retrieve

        Returns:
            Template string, or empty string if not found
        """
        return self._llm.get_chat_template(template_name)

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
) -> Response:
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
        Response object with text and statistics

    Example:
        >>> response = await complete_async(
        >>>     "What is Python?",
        >>>     model_path="model.gguf",
        >>>     temperature=0.7
        >>> )
        >>> print(response)  # Works like a string
        >>> print(response.stats)  # Access statistics
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
) -> Response:
    """
    Async convenience function for chat-style generation.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model_path: Path to GGUF model file
        config: Generation configuration
        verbose: Enable detailed logging
        **kwargs: Additional config parameters

    Returns:
        Response object with text and statistics

    Example:
        >>> messages = [
        >>>     {"role": "system", "content": "You are a helpful assistant."},
        >>>     {"role": "user", "content": "What is Python?"}
        >>> ]
        >>> response = await chat_async(messages, model_path="model.gguf")
        >>> print(response)  # Works like a string
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
