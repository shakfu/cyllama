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
import hashlib
import signal
import threading
from collections import OrderedDict
from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Iterator,
    NamedTuple,
    Optional,
    Dict,
    Any,
    List,
    Callable,
    Union,
    Tuple,
    cast,
)

if TYPE_CHECKING:
    from .agents.mcp import McpClient, McpResource, McpTool
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)

from .defaults import (
    LLAMA_DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_MIN_P,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_PENALTY_LAST_N,
    DEFAULT_PENALTY_FREQ,
    DEFAULT_PENALTY_PRESENT,
    DEFAULT_MIROSTAT,
    DEFAULT_MIROSTAT_TAU,
    DEFAULT_MIROSTAT_ETA,
    DEFAULT_MAX_TOKENS,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_N_BATCH,
    DEFAULT_MAIN_GPU,
    DEFAULT_SPLIT_MODE,
)

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

# Alias the vendored jinja2 TemplateError so the chat-template fallback
# inside _apply_template can catch it without paying the import cost on
# every call. The vendored jinja2 lives at cyllama._vendor.jinja2 and is
# pure Python, so this import is cheap and always succeeds -- but we
# guard with try/except anyway so a corrupted vendor directory doesn't
# break api.py module loading.
try:
    from cyllama._vendor.jinja2.exceptions import TemplateError as _JinjaTemplateError
except ImportError:  # pragma: no cover - vendor directory should always exist
    _JinjaTemplateError = type("_JinjaTemplateError", (Exception,), {})


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.

    Attributes:
        max_tokens: Maximum number of tokens to generate (see defaults.py)
        temperature: Sampling temperature, 0.0 = greedy (see defaults.py)
        top_k: Top-k sampling parameter (see defaults.py)
        top_p: Top-p (nucleus) sampling parameter (see defaults.py)
        min_p: Minimum probability threshold (see defaults.py)
        repeat_penalty: Penalty for repeating tokens, 1.0 = disabled (see defaults.py)
        penalty_last_n: Number of recent tokens considered for penalties,
            0 = disabled, -1 = full context (see defaults.py)
        frequency_penalty: Penalize tokens by frequency in recent window,
            0.0 = disabled (see defaults.py)
        presence_penalty: Penalize tokens already present in recent window,
            0.0 = disabled (see defaults.py)
        mirostat: Mirostat sampling mode (0 = off, 1 = v1, 2 = v2). When
            enabled, replaces top_k / top_p / min_p / temp / dist with the
            mirostat sampler (see defaults.py).
        mirostat_tau: Mirostat target entropy (see defaults.py)
        mirostat_eta: Mirostat learning rate (see defaults.py)
        n_gpu_layers: Number of layers to offload to GPU (see defaults.py)
        main_gpu: Primary GPU device index for inference (see defaults.py)
        split_mode: How to split model across GPUs (see defaults.py)
            0 = NONE: Use single GPU only (main_gpu)
            1 = LAYER: Split layers and KV cache across GPUs
            2 = ROW: Split with tensor parallelism (if supported)
        tensor_split: Proportion of work per GPU (default: None = auto)
            List of floats, one per GPU. Values are normalized by llama.cpp.
            Example: [1, 2] assigns 1/3 to GPU 0 and 2/3 to GPU 1.
        n_ctx: Context window size, None = auto (default: None)
        n_batch: Batch size for processing (see defaults.py)
        seed: Random seed for reproducibility (see defaults.py)
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

    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    top_p: float = DEFAULT_TOP_P
    min_p: float = DEFAULT_MIN_P
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY
    penalty_last_n: int = DEFAULT_PENALTY_LAST_N
    frequency_penalty: float = DEFAULT_PENALTY_FREQ
    presence_penalty: float = DEFAULT_PENALTY_PRESENT
    mirostat: int = DEFAULT_MIROSTAT
    mirostat_tau: float = DEFAULT_MIROSTAT_TAU
    mirostat_eta: float = DEFAULT_MIROSTAT_ETA
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS
    main_gpu: int = DEFAULT_MAIN_GPU
    split_mode: int = DEFAULT_SPLIT_MODE
    tensor_split: Optional[List[float]] = None
    n_ctx: Optional[int] = None
    n_batch: int = DEFAULT_N_BATCH
    seed: int = LLAMA_DEFAULT_SEED
    stop_sequences: List[str] = field(default_factory=list)
    add_bos: bool = True
    parse_special: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a dictionary, copying mutable values."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "repeat_penalty": self.repeat_penalty,
            "penalty_last_n": self.penalty_last_n,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "mirostat": self.mirostat,
            "mirostat_tau": self.mirostat_tau,
            "mirostat_eta": self.mirostat_eta,
            "n_gpu_layers": self.n_gpu_layers,
            "main_gpu": self.main_gpu,
            "split_mode": self.split_mode,
            "tensor_split": self.tensor_split.copy() if self.tensor_split else None,
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "seed": self.seed,
            "stop_sequences": self.stop_sequences.copy(),
            "add_bos": self.add_bos,
            "parse_special": self.parse_special,
        }

    def __post_init__(self) -> None:
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

        if self.penalty_last_n < -1:
            errors.append(f"penalty_last_n must be >= -1 (-1 = full context, 0 = disabled), got {self.penalty_last_n}")

        if self.frequency_penalty < 0.0:
            errors.append(f"frequency_penalty must be >= 0.0, got {self.frequency_penalty}")

        if self.presence_penalty < 0.0:
            errors.append(f"presence_penalty must be >= 0.0, got {self.presence_penalty}")

        if self.mirostat not in (0, 1, 2):
            errors.append(f"mirostat must be 0, 1, or 2, got {self.mirostat}")

        if self.mirostat_tau <= 0.0:
            errors.append(f"mirostat_tau must be > 0.0, got {self.mirostat_tau}")

        if self.mirostat_eta <= 0.0:
            errors.append(f"mirostat_eta must be > 0.0, got {self.mirostat_eta}")

        if self.n_gpu_layers < -1:
            errors.append(f"n_gpu_layers must be >= -1 (-1 = offload all), got {self.n_gpu_layers}")

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

    def __eq__(self, other: object) -> bool:
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

    def __iter__(self) -> Iterator[str]:
        """Iterate over characters in text."""
        return iter(self.text)

    def __contains__(self, item: str) -> bool:
        """Check if substring is in text."""
        return item in self.text

    def __add__(self, other: object) -> str:
        """Concatenate with strings."""
        if isinstance(other, str):
            return self.text + other
        if isinstance(other, Response):
            return self.text + other.text
        return NotImplemented

    def __radd__(self, other: object) -> str:
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


class ResponseCacheInfo(NamedTuple):
    """Cache statistics for LLM response caching."""

    hits: int
    misses: int
    maxsize: int
    currsize: int
    ttl: Optional[float]  # TTL in seconds, None if no expiration


class _ResponseLRUCache:
    """
    LRU cache for Response objects with optional TTL support.

    Stores (Response, timestamp) tuples and checks expiration on get().
    Expired entries are treated as cache misses.
    """

    def __init__(self, maxsize: int, ttl: Optional[float] = None):
        """
        Initialize the cache.

        Args:
            maxsize: Maximum number of entries
            ttl: Time-to-live in seconds, None for no expiration
        """
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.

        Returns None if key not found or entry expired.
        Expired entries are removed from cache.
        """
        if key not in self._cache:
            self._misses += 1
            return None

        value, timestamp = self._cache[key]

        # Check TTL expiration
        if self._ttl is not None:
            if time.time() - timestamp > self._ttl:
                # Entry expired - remove and return miss
                del self._cache[key]
                self._misses += 1
                return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        """
        Store a value in the cache.

        If cache is full, evicts the least recently used entry.
        """
        if key in self._cache:
            # Update existing entry
            self._cache[key] = (value, time.time())
            self._cache.move_to_end(key)
        else:
            # Check capacity
            if len(self._cache) >= self._maxsize:
                # Evict oldest (first item)
                self._cache.popitem(last=False)
            self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all cache entries and reset statistics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def info(self) -> ResponseCacheInfo:
        """Return cache statistics."""
        return ResponseCacheInfo(
            hits=self._hits,
            misses=self._misses,
            maxsize=self._maxsize,
            currsize=len(self._cache),
            ttl=self._ttl,
        )


class _SigintHandle:
    """Restorer returned by ``LLM.install_sigint_handler()``.

    Acts as a context manager (``__exit__`` restores the prior handler) and
    as an imperative handle (call ``.restore()`` directly). Idempotent --
    a second restore is a no-op.

    Standard last-installed-first-restored caveat applies: if multiple
    handlers are stacked, restore them in reverse order.
    """

    __slots__ = ("_previous", "_restored")

    def __init__(self, previous: object) -> None:
        self._previous = previous
        self._restored = False

    def restore(self) -> None:
        if self._restored:
            return
        try:
            signal.signal(signal.SIGINT, self._previous)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            # Off-main-thread or unrestorable handler; best-effort.
            pass
        self._restored = True

    def __enter__(self) -> "_SigintHandle":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.restore()


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
        cache_size: int = 0,
        cache_ttl: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize generator with a model.

        Args:
            model_path: Path to GGUF model file
            config: Generation configuration (uses defaults if None)
            verbose: Print detailed information during generation
            cache_size: Maximum number of responses to cache (0 = disabled)
            cache_ttl: Cache time-to-live in seconds (None = no expiration)
            **kwargs: Per-field overrides on top of ``config``.
                Merge semantics are a **shallow per-field override**:
                kwargs replace matching fields on a copy of ``config``,
                fields not named in kwargs keep their value from
                ``config``. The original ``config`` instance is not
                mutated. If ``config`` is ``None`` and kwargs are
                provided, kwargs initialize a fresh ``GenerationConfig``
                directly (so kwargs alone work as a config shorthand).
                Each kwarg name must be a valid ``GenerationConfig``
                field; unknown names raise ``TypeError`` from the
                dataclass constructor.

        Example:
            >>> # Direct parameters (recommended for simple cases) --
            >>> # kwargs alone construct a GenerationConfig.
            >>> llm = LLM("model.gguf", temperature=0.9, max_tokens=100)
            >>>
            >>> # Explicit config, no overrides.
            >>> config = GenerationConfig(temperature=0.9, top_p=0.95)
            >>> llm = LLM("model.gguf", config=config)
            >>>
            >>> # Partial override: temperature gets replaced, top_p
            >>> # is preserved from config (shallow per-field merge).
            >>> llm = LLM("model.gguf", config=config, temperature=0.5)
            >>> # llm.config.temperature == 0.5 and llm.config.top_p == 0.95
            >>>
            >>> # With response caching
            >>> llm = LLM("model.gguf", cache_size=100, cache_ttl=3600)
        """
        self.model_path = model_path
        self.verbose = verbose
        self._closed = False
        self._last_stream_stats: Optional[GenerationStats] = None

        # Concurrent-use guard for the underlying llama.cpp context.
        # llama_context is not thread-safe and we release the GIL during
        # native generation calls, so two Python threads sharing one LLM
        # can race inside C++ code and corrupt KV cache, sampler, or
        # batch state. We protect every native-touching public method
        # with a non-blocking acquire of this lock: a second concurrent
        # caller fails fast with a clear RuntimeError instead of silently
        # corrupting state.
        #
        # This is intentionally a contention check, NOT a thread-id
        # check: legitimate sequential ownership transfer between threads
        # (e.g. asyncio.to_thread, ThreadPoolExecutor.submit) must keep
        # working, since there is no concurrent access in those patterns.
        # close()/__del__ are intentionally NOT guarded because the
        # garbage collector may run them on any thread.
        self._busy_lock = threading.Lock()

        # Cancellation signal for in-flight generations. ``cancel()`` sets
        # this; ``_generate_stream`` polls it between tokens and breaks out.
        # The same signal is mirrored to the underlying LlamaContext as a
        # C-level bint so ggml's abort callback can short-circuit a
        # long-running ``llama_decode`` (e.g. during prompt prefill on a
        # large context) without waiting for the next token boundary.
        self._cancel_event = threading.Event()

        # Initialize response cache
        self._cache: Optional[_ResponseLRUCache] = None
        if cache_size > 0:
            self._cache = _ResponseLRUCache(cache_size, cache_ttl)

        from .utils.validation import validate_gguf_file

        # Surface clear, typed errors (FileNotFoundError, IsADirectoryError,
        # PermissionError, ValueError) for the common bad-input cases before
        # llama.cpp gets a chance to fail with a NULL pointer or segfault
        # inside its GGUF parser. validate_gguf_file checks magic, version,
        # and tensor/kv counts.
        validate_gguf_file(model_path, kind="GGUF model")

        # Build config: start with provided config or defaults, then apply kwargs
        if config is None:
            if kwargs:
                self.config = GenerationConfig(**kwargs)
            else:
                self.config = GenerationConfig()
        else:
            if kwargs:
                # Create a copy of config with kwargs overrides
                config_dict = config.to_dict()
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
            gpu_info = (
                f"GPU config: n_gpu_layers={self.config.n_gpu_layers}, "
                f"main_gpu={self.config.main_gpu}, split_mode={self.config.split_mode}"
            )
            if self.config.tensor_split:
                gpu_info += f", tensor_split={self.config.tensor_split}"
            print(gpu_info)

        self.model = LlamaModel(model_path, model_params)
        self.vocab = self.model.get_vocab()

        if self.verbose:
            print(f"Model loaded: {self.model.n_params} parameters")
            print(f"Vocabulary size: {self.vocab.n_vocab}")

        # Context will be created on-demand and cached when possible
        self._ctx: Optional[LlamaContext] = None
        self._ctx_size: int = 0  # Track current context size for reuse decisions
        self._sampler: Optional[LlamaSampler] = None

        # MCP client is created lazily on the first add_mcp_server() call so
        # callers who never use MCP pay no import or connection cost.
        self._mcp_client: Optional["McpClient"] = None

    def __enter__(self) -> "LLM":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Context manager exit - ensures cleanup."""
        self.close()

    def __del__(self) -> None:
        """Destructor - cleanup resources if not already done."""
        if not getattr(self, "_closed", True):
            try:
                self.close()
            except Exception:
                # `__del__` may run during interpreter shutdown when even
                # `logger` may have been torn down -- guard the report
                # itself so cleanup never crashes the process. Failures
                # in close() are still surfaced when shutdown is normal.
                try:
                    import traceback

                    traceback.print_exc()
                except Exception:
                    pass

    def close(self) -> None:
        """
        Explicitly release resources (context, sampler).

        This method frees the context and sampler to release GPU memory.
        The model remains loaded for potential reuse. Call this when you're
        done with generation or want to free memory.

        After calling close(), the LLM instance can still be used - new
        contexts will be created as needed.
        """
        if getattr(self, "_closed", True):
            return

        if getattr(self, "verbose", False):
            print("Closing LLM resources")

        # Release context and sampler (use getattr for safety in __del__)
        if getattr(self, "_ctx", None) is not None:
            self._ctx = None
            self._ctx_size = 0

        if getattr(self, "_sampler", None) is not None:
            self._sampler = None

        # Disconnect any MCP servers the caller attached. Best-effort:
        # transports already log their own errors and we don't want a flaky
        # remote server to block local resource cleanup.
        client = getattr(self, "_mcp_client", None)
        if client is not None:
            try:
                client.disconnect_all()
            except Exception:
                # Catch broadly because cleanup must not raise out of
                # close(); but log so a remote-server bug or transport
                # failure doesn't disappear silently.
                logger.warning("MCP disconnect_all failed during close()", exc_info=True)
            self._mcp_client = None

        self._closed = True

    def cancel(self) -> None:
        """
        Request cancellation of an in-flight generation.

        Safe to call from any thread. Has effect at two layers:

        - Between tokens: ``_generate_stream`` polls a ``threading.Event``
          each iteration and exits cleanly when set. This stops generation
          within ~1 token of latency (typically a few milliseconds).
        - Mid-decode: the underlying llama_context has a nogil
          ggml_abort_callback installed that reads a C-level flag. Setting
          it causes ``llama_decode`` to abort the current batch -- this
          matters when prefilling a long prompt, where one ``decode`` call
          can take seconds before the next token-boundary check fires.

        The flag is automatically cleared at the start of the next
        generation, so it is safe to call ``cancel()`` even when no
        generation is currently running -- the request will not "carry
        over" to the next call.

        No-op if no context exists yet (cancellation only matters during
        generation, and generation always creates a context first).
        """
        self._cancel_event.set()
        ctx = getattr(self, "_ctx", None)
        if ctx is not None:
            ctx.cancel = True

    @property
    def cancel_requested(self) -> bool:
        """Whether ``cancel()`` has been called and not yet cleared."""
        return self._cancel_event.is_set()

    def install_sigint_handler(self) -> _SigintHandle:
        """
        Install a SIGINT (Ctrl-C) handler that calls ``self.cancel()``.

        Useful for CLI scripts: Ctrl-C will interrupt generation cleanly
        — including during a long ``llama_decode`` (prompt prefill) where
        the default ``KeyboardInterrupt`` mechanism would otherwise be
        delayed until the C call returns. The generator exits via the
        normal cancellation path and returns whatever was produced so
        far; no exception propagates from the handler itself.

        Must be called from the main Python thread (``signal.signal``
        restriction). The previous SIGINT handler is saved and can be
        restored by calling ``.restore()`` on the returned object, or by
        using it as a context manager.

        Note: this library normally avoids touching signal handlers --
        installing one is opt-in for callers who explicitly want this
        behavior. Multiple installations stack; restore them in reverse
        order (last-installed-first-restored).

        Example (context manager)::

            with llm.install_sigint_handler():
                for chunk in llm("Long prompt", stream=True):
                    print(chunk, end="", flush=True)

        Example (imperative)::

            handle = llm.install_sigint_handler()
            try:
                response = llm("Prompt")
            finally:
                handle.restore()

        Returns:
            A ``_SigintHandle`` that restores the previous handler on
            ``__exit__`` or ``.restore()``.
        """
        previous = signal.signal(signal.SIGINT, lambda *_: self.cancel())
        return _SigintHandle(previous)

    def _try_acquire_busy(self) -> None:
        """Acquire the busy-lock or raise on contention.

        Non-blocking: if another thread is currently inside a guarded
        method, we raise immediately rather than serialize behind it.
        Serializing would hide the bug; raising lets the caller see that
        their concurrent-use pattern is unsafe and fix it.
        """
        if not self._busy_lock.acquire(blocking=False):
            raise RuntimeError(
                "LLM is currently being used by another thread. llama.cpp "
                "contexts are not thread-safe — create one LLM per thread "
                "instead of sharing a single instance across threads."
            )

    def _stream_with_busy_release(self, gen: Iterator[str]) -> Iterator[str]:
        """Wrap a streaming generator so the busy lock is released when
        the stream is exhausted, closed, or garbage collected.

        Generator ``finally`` blocks run on every termination path
        (StopIteration, exception, .close(), gc), so the lock is
        guaranteed to be released even if the caller drops the iterator
        without consuming it.
        """
        try:
            yield from gen
        finally:
            try:
                self._busy_lock.release()
            except RuntimeError:
                # Lock already released — defensive, should not happen.
                pass

    def reset_context(self) -> None:
        """
        Force recreation of context on next generation.

        This clears the KV cache and ensures a fresh context is created.
        Useful when you want to start a completely new conversation without
        any prior context.
        """
        self._try_acquire_busy()
        try:
            if self._ctx is not None:
                if self.verbose:
                    print("Resetting context")
                self._ctx = None
                self._ctx_size = 0
        finally:
            self._busy_lock.release()

    @property
    def cache_enabled(self) -> bool:
        """Return True if response caching is enabled."""
        return self._cache is not None

    def cache_info(self) -> Optional[ResponseCacheInfo]:
        """
        Return cache statistics.

        Returns:
            ResponseCacheInfo with hits, misses, maxsize, currsize, ttl.
            Returns None if caching is disabled.
        """
        if self._cache is None:
            return None
        return self._cache.info()

    def cache_clear(self) -> None:
        """
        Clear all cached responses and reset cache statistics.

        Does nothing if caching is disabled.
        """
        if self._cache is not None:
            self._cache.clear()

    def _make_cache_key(self, prompt: str, config: GenerationConfig) -> Optional[str]:
        """
        Generate a cache key from prompt and config.

        Returns None if caching should be skipped (e.g., random seed).

        The key includes parameters that affect output:
        - prompt, temperature, top_k, top_p, min_p
        - repeat_penalty, max_tokens, stop_sequences (sorted)
        - seed, add_bos, parse_special

        Infrastructure parameters are excluded:
        - n_gpu_layers, main_gpu, split_mode, tensor_split, n_ctx, n_batch
        """
        # Skip caching for random seed
        if config.seed == LLAMA_DEFAULT_SEED:
            return None

        # Build key from output-affecting parameters
        key_parts = [
            prompt,
            str(config.temperature),
            str(config.top_k),
            str(config.top_p),
            str(config.min_p),
            str(config.repeat_penalty),
            str(config.penalty_last_n),
            str(config.frequency_penalty),
            str(config.presence_penalty),
            str(config.mirostat),
            str(config.mirostat_tau),
            str(config.mirostat_eta),
            str(config.max_tokens),
            str(sorted(config.stop_sequences)),
            str(config.seed),
            str(config.add_bos),
            str(config.parse_special),
        ]
        key_str = "\0".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _ensure_context(self, prompt_length: int, config: GenerationConfig) -> LlamaContext:
        """
        Create or recreate context if needed.

        Context is reused when possible to avoid allocation overhead.
        A new context is created when:
        - No context exists
        - The required size exceeds current context size
        - The instance was closed (will reopen)

        The KV cache is cleared via llama_kv_cache_clear() when reusing
        a context to ensure clean state for new generations.

        Returns the live context so callers don't have to re-narrow
        ``self._ctx`` from ``Optional`` after this method runs.
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
            return self._ctx

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
        ctx = LlamaContext(self.model, ctx_params)
        # Wire mid-decode cancellation: a nogil ggml_abort_callback that
        # reads a C bint owned by the context. ``cancel()`` flips that
        # bint and the next ggml op poll aborts the in-progress batch.
        ctx.install_cancel_callback()
        self._ctx = ctx
        self._ctx_size = required_ctx
        return ctx

    def _ensure_sampler(self, config: GenerationConfig) -> LlamaSampler:
        """Create or recreate sampler if needed.

        Returns the live sampler so callers don't have to re-narrow
        ``self._sampler`` from ``Optional`` after this method runs.
        """
        # Always create fresh sampler to respect new config
        sampler_params = LlamaSamplerChainParams()
        sampler_params.no_perf = not self.verbose

        sampler = LlamaSampler(sampler_params)

        # Penalties chain link is added first so it shapes logits before
        # truncation samplers (top_k/top_p/min_p) and the final selector.
        # Skipped entirely when all knobs are at their disabled values to
        # avoid an unnecessary chain node.
        if (
            config.repeat_penalty != 1.0 or config.frequency_penalty != 0.0 or config.presence_penalty != 0.0
        ) and config.penalty_last_n != 0:
            sampler.add_penalties(
                config.penalty_last_n,
                config.repeat_penalty,
                config.frequency_penalty,
                config.presence_penalty,
            )

        # Add sampling methods based on config
        if config.temperature == 0.0:
            # Greedy sampling
            sampler.add_greedy()
        elif config.mirostat == 1:
            # Mirostat v1 owns temperature + selection; truncation
            # samplers are skipped per upstream guidance.
            sampler.add_temp(config.temperature)
            sampler.add_mirostat(
                self.vocab.n_vocab,
                config.seed,
                config.mirostat_tau,
                config.mirostat_eta,
                100,  # m: rolling-window size, llama.cpp default
            )
        elif config.mirostat == 2:
            sampler.add_temp(config.temperature)
            sampler.add_mirostat_v2(
                config.seed,
                config.mirostat_tau,
                config.mirostat_eta,
            )
        else:
            # Probabilistic sampling
            sampler.add_min_p(config.min_p, 1)
            sampler.add_top_k(config.top_k)
            sampler.add_top_p(config.top_p, 1)
            sampler.add_temp(config.temperature)

            # Distribution sampler
            if config.seed != LLAMA_DEFAULT_SEED:
                sampler.add_dist(config.seed)
            else:
                sampler.add_dist(LLAMA_DEFAULT_SEED)

        self._sampler = sampler
        return sampler

    def __call__(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
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
        self._try_acquire_busy()
        if stream:
            # The wrapper releases the lock when the generator is
            # exhausted, closed, or garbage collected. Returning the
            # wrapper unconditionally (no try/finally here) hands lock
            # ownership to the generator.
            return self._stream_with_busy_release(self._generate_stream(prompt, config, on_token))
        try:
            return self._generate(prompt, config, on_token)
        finally:
            self._busy_lock.release()

    def _generate(
        self, prompt: str, config: Optional[GenerationConfig] = None, on_token: Optional[Callable[[str], None]] = None
    ) -> Response:
        """Non-streaming generation returning Response object."""
        config = config or self.config

        # Check cache first (only if no on_token callback)
        cache_key: Optional[str] = None
        cache = self._cache
        if cache is not None and on_token is None:
            cache_key = self._make_cache_key(prompt, config)
            if cache_key is not None:
                cached = cache.get(cache_key)
                if cached is not None:
                    if self.verbose:
                        print("Cache hit")
                    return cast(Response, cached)

        start_time = time.time()

        # Tokenize for stats
        prompt_tokens = self.vocab.tokenize(prompt, add_special=config.add_bos, parse_special=config.parse_special)
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
            tokens_per_second=n_generated / total_time if total_time > 0 else 0.0,
        )

        response = Response(text=text, stats=stats, finish_reason="stop", model=self.model_path)

        # Store in cache if enabled
        if cache_key is not None and cache is not None:
            cache.put(cache_key, response)

        return response

    def _generate_stream(
        self, prompt: str, config: Optional[GenerationConfig] = None, on_token: Optional[Callable[[str], None]] = None
    ) -> Iterator[str]:
        """
        Internal streaming generation implementation.

        Yields:
            Text chunks as they are generated
        """
        # Use provided config or fall back to instance config
        config = config or self.config

        # Reset cancellation state at the start of each generation. A stale
        # ``cancel()`` from before this call does not carry over.
        self._cancel_event.clear()

        start_time = time.time()

        # Tokenize prompt
        prompt_tokens = self.vocab.tokenize(prompt, add_special=config.add_bos, parse_special=config.parse_special)
        n_prompt = len(prompt_tokens)

        if self.verbose:
            print(f"Prompt tokens: {n_prompt}")

        # Ensure context and sampler are ready
        # Always recreate sampler to ensure fresh state
        ctx = self._ensure_context(n_prompt, config)
        sampler = self._ensure_sampler(config)
        # Mirror the Python-side cancel flag onto the C-level flag for the
        # active context. Cleared here; ``cancel()`` sets both layers.
        ctx.cancel = False

        # Process prompt in batches to avoid exceeding n_batch limit. If
        # the caller has already issued ``cancel()`` (or does so during a
        # long prefill), the abort callback aborts the in-progress decode
        # and we bail before generation begins.
        n_batch = config.n_batch
        for i in range(0, n_prompt, n_batch):
            if self._cancel_event.is_set():
                return
            batch_tokens = prompt_tokens[i : i + n_batch]
            batch = llama_batch_get_one(batch_tokens, i)  # Pass position offset
            try:
                ctx.decode(batch)
            except InterruptedError:
                # Mid-decode abort_callback fired (cancel() was called
                # during prefill). Bail cleanly rather than propagating.
                return

        # Generate tokens
        n_pos = n_prompt
        n_generated = 0

        # Stop sequence handling: buffer recent output to detect sequences spanning tokens
        # We only need to buffer enough to detect the longest stop sequence
        stop_buffer = ""
        max_stop_len = max(len(s) for s in config.stop_sequences) if config.stop_sequences else 0

        for _ in range(config.max_tokens):
            # Cooperative cancellation check between tokens. The mid-decode
            # ggml abort callback handles cancellation during long
            # ``llama_decode`` calls; this handles the steady-state
            # token-by-token loop with sub-millisecond latency.
            if self._cancel_event.is_set():
                break

            # Sample next token
            new_token_id = sampler.sample(ctx, -1)

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
            try:
                ctx.decode(batch)
            except InterruptedError:
                # Mid-decode cancellation; the next-token batch did not
                # complete. Stop the stream cleanly with whatever we
                # already yielded.
                break

            n_pos += 1
            n_generated += 1

        # Flush remaining buffer (no stop sequence found)
        if config.stop_sequences and stop_buffer:
            if on_token:
                on_token(stop_buffer)
            yield stop_buffer

        # Store streaming stats so callers can retrieve them after exhaustion
        total_time = time.time() - start_time
        self._last_stream_stats = GenerationStats(
            prompt_tokens=n_prompt,
            generated_tokens=n_generated,
            total_time=total_time,
            tokens_per_second=n_generated / total_time if total_time > 0 else 0.0,
        )

        if self.verbose:
            print(f"\nGenerated {n_generated} tokens")
            sampler.print_perf_data()
            ctx.print_perf_data()

    def _find_stop_sequence(self, text: str, stop_sequences: List[str]) -> Tuple[Optional[int], int]:
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

    def generate_with_stats(self, prompt: str, config: Optional[GenerationConfig] = None) -> Response:
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
        self._try_acquire_busy()
        try:
            return self._generate(prompt, config)
        finally:
            self._busy_lock.release()

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
        # No busy-lock acquire here: chat() is a thin wrapper that
        # delegates to __call__, which holds the lock. Acquiring twice
        # would deadlock since the lock is non-reentrant.
        prompt = self._apply_template(messages, template)
        return self(prompt, config=config, stream=stream)

    def _apply_template(
        self,
        messages: List[Dict[str, str]],
        template: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply chat template to messages using the loaded model.

        Two paths, tried in order:

        1. **Vendored jinja2 (full Jinja support).** When the caller has
           not requested a specific template, evaluate the model's
           embedded chat template via the vendored ``jinja2`` interpreter
           under ``cyllama._vendor``. This handles any GGUF whose
           embedded template uses Jinja syntax that llama.cpp's basic C
           API doesn't recognise (Gemma 4 is the canonical example -- the
           substring heuristic in ``llm_chat_detect_template`` fails to
           match its template, so the legacy path returns -1, but the
           Jinja interpreter evaluates it cleanly).

        2. **Legacy substring-heuristic path.** Falls back to the
           original ``llama_chat_apply_template`` C API on
           ``TemplateError`` (template raised, e.g. Gemma's
           ``raise_exception('System role not supported')``) or any
           other failure inside the Jinja path. The fallback also fires
           when the caller explicitly requests a named template via the
           ``template`` parameter, since named-template lookup is a
           feature of the legacy path.

        The pipeline-level three-tier fallback in
        ``cyllama.rag.pipeline.RAGPipeline._chat_with_fallback`` (system
        + user -> merged user -> raw completion) sits outside this
        method as the outermost safety net, so any RuntimeError that
        propagates here is caught one layer up and degraded gracefully.
        """
        # Try the vendored jinja2 path when no specific template was
        # requested. The Jinja interpreter handles full Jinja semantics,
        # so it works on every embedded template the legacy substring
        # heuristic doesn't recognise.
        if template is None:
            try:
                return self._apply_jinja_template(messages, add_generation_prompt)
            except _JinjaTemplateError:
                # Template raised inside Jinja (e.g. Gemma's
                # raise_exception). Fall through to the legacy path,
                # which has its own per-template-name handling. The
                # pipeline-level fallback above us catches anything
                # that fails on both paths. This is the *expected*
                # fallback trigger so we don't log it.
                pass
            except Exception:
                # Any other failure inside the vendored jinja2 path
                # (TemplateSyntaxError on a malformed template, missing
                # bos_token retrieval, etc.) -- fall back to the legacy
                # path. We catch broadly because we'd rather degrade to
                # the older code path than crash the caller. Log at
                # debug so users can opt in to the diagnostic without
                # noise, but the silent-fall-through behaviour is gone:
                # an upstream wrapper bug that surfaces here can be
                # found by enabling debug logging on this module.
                logger.debug(
                    "Vendored jinja2 chat-template path failed; falling back to legacy template handling",
                    exc_info=True,
                )

        # Get template - use provided or model's default. The provided
        # `template` argument can be (a) a Jinja template string, (b) the
        # GGUF metadata key under which a named template is stored, or
        # (c) one of llama.cpp's built-in aliases (chatml, llama3, gemma,
        # mistral, ...). Cases (a) and (c) are both handled directly by
        # llama_chat_apply_template, so we resolve case (b) first via the
        # metadata lookup and otherwise hand the string off as-is. We do
        # NOT pre-emptively warn about unrecognised names: the metadata
        # lookup and the built-in alias table are two different sources
        # of truth, and asking only the metadata lookup would
        # false-positive on every legitimate built-in alias.
        if template:
            tmpl = self.model.get_default_chat_template_by_name(template) or template
        else:
            tmpl = self.model.get_default_chat_template()

        if tmpl:
            chat_messages = []
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    raise TypeError(f"Message at index {i} must be a dict, got {type(msg).__name__}")
                role = msg.get("role")
                if not role or not isinstance(role, str):
                    raise ValueError(f"Message at index {i} missing or invalid 'role': {msg!r}")
                content = msg.get("content")
                if content is None:
                    raise ValueError(f"Message at index {i} missing 'content' key")
                chat_messages.append(LlamaChatMessage(role=role, content=str(content)))
            return cast(str, self.model.chat_apply_template(tmpl, chat_messages, add_generation_prompt))
        else:
            return _format_messages_simple(messages)

    def _apply_jinja_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Render the model's embedded chat template via vendored jinja2.

        Mirrors HuggingFace ``transformers.PreTrainedTokenizerBase.apply_chat_template``:
        get the embedded template string from GGUF metadata, evaluate it
        in an ``ImmutableSandboxedEnvironment`` with the standard context
        (``messages``, ``bos_token``, ``eos_token``, ``add_generation_prompt``)
        and the standard custom globals (``raise_exception``,
        ``strftime_now``) plus the ``tojson`` filter. The same approach
        is used by millions of HuggingFace users, so any GGUF whose chat
        template was generated from a HuggingFace tokenizer config is
        compatible by construction.

        Raises:
            _JinjaTemplateError: If the template explicitly raised
                (e.g. Gemma's ``raise_exception('System role not
                supported')``) or if the model has no embedded template.
            ImportError: Should not happen because jinja2 is vendored,
                but caught defensively in the caller.
        """
        import json
        from datetime import datetime

        from cyllama._vendor.jinja2 import ext as _jinja2_ext
        from cyllama._vendor.jinja2.exceptions import TemplateError
        from cyllama._vendor.jinja2.sandbox import ImmutableSandboxedEnvironment

        template_str = self.model.get_default_chat_template()
        if not template_str:
            raise TemplateError("Model has no embedded chat template")

        # Resolve bos/eos token strings via the existing public C API.
        # The model exposes token IDs through its vocab; we convert them
        # to text via token_to_piece. Tokens may legitimately be missing
        # (some embedding-only models have no BOS/EOS), in which case we
        # pass empty strings -- a template that references {{ bos_token }}
        # will just render an empty string in that position.
        vocab = self.model.get_vocab()
        bos_id = vocab.token_bos()
        eos_id = vocab.token_eos()
        bos_token = vocab.token_to_piece(bos_id, special=True) if bos_id >= 0 else ""
        eos_token = vocab.token_to_piece(eos_id, special=True) if eos_id >= 0 else ""

        def raise_exception(message: str) -> str:
            raise TemplateError(message)

        def tojson_filter(
            value: Any,
            ensure_ascii: bool = False,
            indent: Optional[int] = None,
            separators: Optional[Tuple[str, str]] = None,
            sort_keys: bool = False,
        ) -> str:
            return json.dumps(
                value,
                ensure_ascii=ensure_ascii,
                indent=indent,
                separators=separators,
                sort_keys=sort_keys,
            )

        def strftime_now(fmt: str) -> str:
            return datetime.now().strftime(fmt)

        env = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[_jinja2_ext.loopcontrols],
        )
        env.filters["tojson"] = tojson_filter
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = strftime_now

        # Validate and normalise messages: same shape checks the legacy
        # path performs, but raised inside the new code path so callers
        # see consistent error types regardless of which branch ran.
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise TypeError(f"Message at index {i} must be a dict, got {type(msg).__name__}")
            if not msg.get("role") or not isinstance(msg.get("role"), str):
                raise ValueError(f"Message at index {i} missing or invalid 'role': {msg!r}")
            if msg.get("content") is None:
                raise ValueError(f"Message at index {i} missing 'content' key")

        compiled = env.from_string(template_str)
        return cast(
            str,
            compiled.render(
                messages=messages,
                bos_token=bos_token,
                eos_token=eos_token,
                add_generation_prompt=add_generation_prompt,
            ),
        )

    def get_chat_template(self, template_name: Optional[str] = None) -> str:
        """
        Get the chat template string from the loaded model.

        Args:
            template_name: Optional specific template name to retrieve

        Returns:
            Template string, or empty string if not found
        """
        if template_name:
            return cast(str, self.model.get_default_chat_template_by_name(template_name))
        return cast(str, self.model.get_default_chat_template())

    # ------------------------------------------------------------------
    # MCP client surface
    #
    # Lifts the transports in ``cyllama.agents.mcp`` onto the high-level
    # LLM API so non-agent callers can attach MCP servers without going
    # through the agent framework. The McpClient is created lazily on the
    # first add_mcp_server() call and torn down in close().
    # ------------------------------------------------------------------

    def _get_or_create_mcp_client(self) -> "McpClient":
        """Lazily construct the McpClient on first use."""
        if self._mcp_client is None:
            from .agents.mcp import McpClient

            self._mcp_client = McpClient()
        return self._mcp_client

    def add_mcp_server(
        self,
        name: str,
        *,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        transport: Optional[Any] = None,
        request_timeout: Optional[float] = None,
        shutdown_timeout: Optional[float] = None,
    ) -> None:
        """Attach an MCP server and connect immediately.

        Transport is inferred from which kwargs are set if ``transport`` is
        omitted: ``command`` -> stdio, ``url`` -> http. Pass ``transport``
        explicitly to disambiguate (e.g. when both happen to be set).

        Args:
            name: Logical name for the server. Tools are exposed as
                ``"<name>/<tool>"`` to disambiguate identically named tools
                from different servers.
            command/args/env/cwd: stdio transport options.
            url/headers: http transport options.
            transport: Optional ``McpTransportType`` override.
            request_timeout/shutdown_timeout: optional per-server overrides.
        """
        from .agents.mcp import (
            McpServerConfig,
            McpTransportType,
            DEFAULT_REQUEST_TIMEOUT,
            DEFAULT_SHUTDOWN_TIMEOUT,
        )

        if transport is None:
            if command is not None:
                transport = McpTransportType.STDIO
            elif url is not None:
                transport = McpTransportType.HTTP
            else:
                raise ValueError(
                    "add_mcp_server requires either 'command' (stdio) or 'url' (http), or an explicit 'transport'."
                )

        config = McpServerConfig(
            name=name,
            transport=transport,
            command=command,
            args=args,
            env=env,
            cwd=cwd,
            url=url,
            headers=headers,
            request_timeout=request_timeout if request_timeout is not None else DEFAULT_REQUEST_TIMEOUT,
            shutdown_timeout=shutdown_timeout if shutdown_timeout is not None else DEFAULT_SHUTDOWN_TIMEOUT,
        )

        client = self._get_or_create_mcp_client()
        client.add_server(config)
        # Connect this single server now rather than deferring to a global
        # connect_all(): callers expect add_mcp_server() to fail-fast on a
        # bad config or unreachable endpoint.
        client._connect_server(config)
        client._connected = True

    def remove_mcp_server(self, name: str) -> None:
        """Disconnect and forget an MCP server by name."""
        if self._mcp_client is None:
            return
        client = self._mcp_client
        conn = client._connections.pop(name, None)
        if conn is not None:
            try:
                conn.disconnect()
            except Exception:
                # Best-effort: drop the registration even if the
                # remote disconnect fails (transport already gone,
                # remote process died, etc.) -- the alternative is
                # leaving a stale entry the user cannot remove. Log
                # so a deeper transport bug doesn't disappear.
                logger.warning(
                    "Failed to disconnect MCP server %r during remove_mcp_server; "
                    "removing the local registration anyway",
                    name,
                    exc_info=True,
                )
        client._servers = [s for s in client._servers if s.name != name]
        # Drop tools and resources owned by this server.
        client._tools = {k: v for k, v in client._tools.items() if v.server_name != name}
        client._resources = {k: v for k, v in client._resources.items() if v.server_name != name}
        if not client._connections:
            client._connected = False

    def list_mcp_tools(self) -> List["McpTool"]:
        """Return all discovered MCP tools as ``McpTool`` instances."""
        if self._mcp_client is None:
            return []
        return list(self._mcp_client.get_tools())

    def list_mcp_resources(self) -> List["McpResource"]:
        """Return all discovered MCP resources as ``McpResource`` instances."""
        if self._mcp_client is None:
            return []
        return list(self._mcp_client.get_resources())

    def call_mcp_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Invoke an MCP tool by full ``"server/tool"`` name.

        Useful when the caller wants explicit control over the tool loop
        rather than handing dispatch to ``chat_with_tools``.
        """
        if self._mcp_client is None:
            raise RuntimeError("No MCP servers attached. Call add_mcp_server() first.")
        return self._mcp_client.call_tool(name, arguments)

    def read_mcp_resource(self, uri: str) -> str:
        """Read the contents of an MCP resource by URI."""
        if self._mcp_client is None:
            raise RuntimeError("No MCP servers attached. Call add_mcp_server() first.")
        return self._mcp_client.read_resource(uri)

    def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        *,
        tools: Optional[List[Any]] = None,
        use_mcp: bool = True,
        max_iterations: int = 8,
        verbose: bool = False,
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> str:
        """Run a tool-calling loop over chat messages.

        Built on top of ``ReActAgent``: the agent prompt is text-based, so
        this works on any GGUF without requiring a model trained for
        OpenAI-style structured tool calls. MCP tools attached via
        ``add_mcp_server()`` are merged with caller-supplied ``tools``
        unless ``use_mcp=False``.

        The last user message is treated as the agent task; any system
        message becomes the agent's system prompt unless ``system_prompt``
        is given explicitly. Multi-turn conversation history beyond a
        single user turn is not yet plumbed through -- the ReAct loop
        operates one task at a time.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            tools: Additional cyllama ``Tool`` instances to expose alongside
                MCP tools.
            use_mcp: When True (default), include all MCP tools from
                attached servers.
            max_iterations: Maximum thought/action cycles.
            verbose: Print agent reasoning to stdout.
            system_prompt: Override the system prompt. If None and the
                messages contain a leading system message, that content is
                used.
            generation_config: Override generation config for the loop.

        Returns:
            The agent's final answer string.
        """
        from .agents.react import ReActAgent

        # Resolve task and system prompt from the messages list.
        task: Optional[str] = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                task = str(msg.get("content", ""))
                break
        if task is None:
            raise ValueError("chat_with_tools requires at least one user message")

        if system_prompt is None:
            for msg in messages:
                if msg.get("role") == "system":
                    system_prompt = str(msg.get("content", ""))
                    break

        merged_tools: List[Any] = list(tools) if tools else []
        if use_mcp and self._mcp_client is not None:
            merged_tools.extend(self._mcp_client.get_tools_for_agent())

        agent = ReActAgent(
            llm=self,
            tools=merged_tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
            generation_config=generation_config,
        )
        result = agent.run(task)
        # AgentResult.answer is always a str; the hasattr guard is
        # defensive against future changes to the agent return type.
        answer = getattr(result, "answer", None)
        return str(answer) if answer is not None else str(result)


# Convenience functions


def complete(
    prompt: str,
    model_path: str,
    config: Optional[GenerationConfig] = None,
    stream: bool = False,
    verbose: bool = False,
    **kwargs: Any,
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

    with LLM(model_path, config=config, verbose=verbose) as llm:
        return llm(prompt, stream=stream)


def chat(
    messages: List[Dict[str, str]],
    model_path: str,
    config: Optional[GenerationConfig] = None,
    stream: bool = False,
    verbose: bool = False,
    template: Optional[str] = None,
    **kwargs: Any,
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

    # Load model to get template (n_gpu_layers=0: metadata only, no GPU alloc)
    model_params = LlamaModelParams()
    model_params.n_gpu_layers = 0
    model = LlamaModel(model_path, model_params)

    # Get template - use provided or model's default
    if template:
        tmpl = model.get_default_chat_template_by_name(template)
        if not tmpl:
            # Try as-is (some templates are the actual template string)
            tmpl = template
    else:
        tmpl = model.get_default_chat_template()

    # Tier 1: try vendored jinja2 when no specific template was requested.
    # This handles models whose embedded template uses Jinja syntax that
    # llama.cpp's C substring heuristic doesn't recognise (e.g. Gemma 4).
    if template is None and tmpl:
        try:
            prompt = _apply_jinja_template_standalone(model, tmpl, messages, add_generation_prompt)
            return prompt
        except _JinjaTemplateError:
            # Expected fallback trigger -- silent.
            pass
        except Exception:
            # Anything non-jinja (NameError/AttributeError from a wrapper
            # bug, KeyError on missing model metadata, etc.) -- fall
            # through to Tier 2 rather than crash, but log at debug so
            # the diagnostic is reachable for upstream bug reports.
            logger.debug(
                "Standalone jinja2 chat-template path failed; falling back to C API tier",
                exc_info=True,
            )

    # Tier 2: C API (substring heuristic)
    if tmpl:
        chat_messages = [
            LlamaChatMessage(role=msg.get("role", "user"), content=msg.get("content", "")) for msg in messages
        ]
        prompt = model.chat_apply_template(tmpl, chat_messages, add_generation_prompt)
    else:
        # Fallback to simple format if no template available
        logger.debug("No chat template found, using fallback format")
        prompt = _format_messages_simple(messages)

    return prompt


def _apply_jinja_template_standalone(
    model: "LlamaModel",
    template_str: str,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """Render a chat template via vendored jinja2 for the standalone apply_chat_template path."""
    import json
    from datetime import datetime

    from cyllama._vendor.jinja2 import ext as _jinja2_ext
    from cyllama._vendor.jinja2.exceptions import TemplateError
    from cyllama._vendor.jinja2.sandbox import ImmutableSandboxedEnvironment

    vocab = model.get_vocab()
    bos_id = vocab.token_bos()
    eos_id = vocab.token_eos()
    bos_token = vocab.token_to_piece(bos_id, special=True) if bos_id >= 0 else ""
    eos_token = vocab.token_to_piece(eos_id, special=True) if eos_id >= 0 else ""

    def raise_exception(message: str) -> str:
        raise TemplateError(message)

    def tojson_filter(
        value: Any,
        ensure_ascii: bool = False,
        indent: Optional[int] = None,
        separators: Optional[Tuple[str, str]] = None,
        sort_keys: bool = False,
    ) -> str:
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(fmt: str) -> str:
        return datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[_jinja2_ext.loopcontrols],
    )
    env.filters["tojson"] = tojson_filter
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now

    compiled = env.from_string(template_str)
    return cast(
        str,
        compiled.render(
            messages=messages,
            bos_token=bos_token,
            eos_token=eos_token,
            add_generation_prompt=add_generation_prompt,
        ),
    )


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

    return cast(str, result)


def simple(
    model_path: str,
    prompt: str,
    ngl: int = DEFAULT_N_GPU_LAYERS,
    n_predict: int = 32,
    n_ctx: Optional[int] = None,
    verbose: bool = False,
) -> bool:
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
    prompt = ""
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

    print(
        "decoded %d tokens in %.2f s, speed: %.2f t/s"
        % (n_decode, (t_main_end - t_main_start) / 1000000.0, n_decode / ((t_main_end - t_main_start) / 1000000.0))
    )
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
        **kwargs: Any,
    ) -> None:
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

    async def __aenter__(self) -> "AsyncLLM":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Async context manager exit - ensures cleanup."""
        await self.close()

    async def close(self) -> None:
        """
        Explicitly release resources.

        Runs cleanup in a thread to avoid blocking if cleanup is slow.
        """
        await asyncio.to_thread(self._llm.close)

    async def reset_context(self) -> None:
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

    async def __call__(self, prompt: str, config: Optional[GenerationConfig] = None, **kwargs: Any) -> Response:
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
        effective_config: Optional[GenerationConfig]
        if kwargs:
            effective_config = self._build_config(config, kwargs)
        else:
            effective_config = config

        async with self._lock:
            return cast(
                Response,
                await asyncio.to_thread(self._llm._generate, prompt, effective_config),
            )

    async def generate(self, prompt: str, config: Optional[GenerationConfig] = None, **kwargs: Any) -> Response:
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
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream generated text chunks asynchronously.

        Yields text chunks as they are generated. Each chunk is yielded
        as soon as it's available from the underlying model.

        Args:
            prompt: Input text prompt
            config: Generation configuration
            timeout: Maximum seconds to wait for each chunk (None = no limit)
            **kwargs: Override config parameters

        Yields:
            Text chunks as they are generated

        Example:
            >>> async for chunk in llm.stream("Tell me a story"):
            >>>     print(chunk, end="", flush=True)
        """
        # Build config with overrides if kwargs provided
        effective_config: Optional[GenerationConfig]
        if kwargs:
            effective_config = self._build_config(config, kwargs)
        else:
            effective_config = config

        # Use a queue to bridge sync generator to async iterator
        queue: asyncio.Queue[Union[str, None, Exception]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        async def producer() -> None:
            """Run sync generator in thread and put items in queue."""
            try:

                def generate_sync() -> None:
                    for chunk in self._llm._generate_stream(prompt, effective_config):
                        # Schedule putting item in queue from the thread
                        asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                    # Signal completion
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

                await asyncio.to_thread(generate_sync)
            except Exception as e:
                # Route any exception (from generation, tokenization,
                # the user's on_token callback, etc.) through the queue
                # so the consumer's `raise item` re-raises it on the
                # awaiting task. Narrowing this catch would orphan
                # exceptions and hang the consumer on `queue.get()`.
                await queue.put(e)

        # Start producer task
        async with self._lock:
            producer_task = asyncio.create_task(producer())

            try:
                while True:
                    if timeout is not None:
                        item = await asyncio.wait_for(queue.get(), timeout=timeout)
                    else:
                        item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                # Ensure producer completes; if it hasn't started yet or is
                # still running, cancel it and suppress CancelledError.
                if not producer_task.done():
                    producer_task.cancel()
                    try:
                        await producer_task
                    except asyncio.CancelledError:
                        pass
                else:
                    # Retrieve (and discard) the result so any exception
                    # doesn't become an unhandled task exception.
                    # Anything raised by the producer was already
                    # routed via `queue.put(e)` above and re-raised on
                    # the consumer side, so swallowing here is correct.
                    try:
                        producer_task.result()
                    except Exception:
                        pass

    async def generate_with_stats(self, prompt: str, config: Optional[GenerationConfig] = None) -> Response:
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
            return cast(
                Response,
                await asyncio.to_thread(self._llm.generate_with_stats, prompt, config),
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
            return cast(
                Response,
                await asyncio.to_thread(
                    self._llm.chat,
                    messages,
                    config,
                    False,  # stream=False
                    template,
                ),
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

    def _build_config(self, base_config: Optional[GenerationConfig], overrides: Dict[str, Any]) -> GenerationConfig:
        """Build a config with overrides applied."""
        config = base_config or self._llm.config
        config_dict = config.to_dict()
        config_dict.update(overrides)
        return GenerationConfig(**config_dict)


async def complete_async(
    prompt: str, model_path: str, config: Optional[GenerationConfig] = None, verbose: bool = False, **kwargs: Any
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
    return cast(
        Response,
        await asyncio.to_thread(
            complete,
            prompt,
            model_path,
            config,
            False,  # stream=False
            verbose,
            **kwargs,
        ),
    )


async def chat_async(
    messages: List[Dict[str, str]],
    model_path: str,
    config: Optional[GenerationConfig] = None,
    verbose: bool = False,
    **kwargs: Any,
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
    return cast(
        Response,
        await asyncio.to_thread(
            chat,
            messages,
            model_path,
            config,
            False,  # stream=False
            verbose,
            **kwargs,
        ),
    )


async def stream_complete_async(
    prompt: str, model_path: str, config: Optional[GenerationConfig] = None, verbose: bool = False, **kwargs: Any
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
