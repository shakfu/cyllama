"""Embedder class for generating text embeddings using llama.cpp."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator, NamedTuple

from ..llama.llama_cpp import (
    LlamaBatch,
    LlamaContext,
    LlamaContextParams,
    LlamaModel,
    LlamaModelParams,
    LlamaVocab,
)
from .types import EmbeddingResult


class CacheInfo(NamedTuple):
    """Cache statistics for Embedder."""

    hits: int
    misses: int
    maxsize: int
    currsize: int


class _LRUCache:
    """Simple LRU cache implementation for embeddings.

    Uses OrderedDict to maintain insertion order and evict oldest entries.
    Thread-safe for single-threaded use (typical for embedding workloads).
    """

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> tuple[float, ...] | None:
        """Get item from cache, moving it to end (most recently used)."""
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: tuple[float, ...]) -> None:
        """Add item to cache, evicting oldest if at capacity."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def clear(self) -> None:
        """Clear the cache and reset statistics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def info(self) -> CacheInfo:
        """Return cache statistics."""
        return CacheInfo(
            hits=self._hits,
            misses=self._misses,
            maxsize=self.maxsize,
            currsize=len(self._cache),
        )


class PoolingType(IntEnum):
    """Pooling types for embedding generation."""

    NONE = 0
    MEAN = 1
    CLS = 2
    LAST = 3


class Embedder:
    """Generate embeddings using llama.cpp embedding models.

    The Embedder class wraps a llama.cpp model to generate vector embeddings
    from text. It supports various pooling strategies and optional L2 normalization.

    Example:
        >>> embedder = Embedder("models/bge-small-en-v1.5-q8_0.gguf")
        >>> embedding = embedder.embed("Hello, world!")
        >>> print(len(embedding))  # e.g., 384 for BGE-small
        384

        >>> # Batch embedding
        >>> texts = ["First text", "Second text", "Third text"]
        >>> embeddings = embedder.embed_batch(texts)
        >>> print(len(embeddings))
        3
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        n_batch: int = 512,
        n_gpu_layers: int = -1,
        pooling: str = "mean",
        normalize: bool = True,
        verbose: bool = False,
        cache_size: int = 0,
    ):
        """Initialize embedder with an embedding model.

        Args:
            model_path: Path to GGUF embedding model (BGE, Snowflake, etc.)
            n_ctx: Context size (should match model's training)
            n_batch: Batch size for processing
            n_gpu_layers: GPU layers (-1 = all, 0 = CPU only)
            pooling: Pooling strategy: "mean", "cls", "last", or "none"
            normalize: Whether to L2-normalize output vectors
            verbose: Whether to print model loading info
            cache_size: Max number of embeddings to cache (0 = disabled)
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self._normalize = normalize
        self._verbose = verbose

        # Parse pooling type
        pooling_map = {
            "none": PoolingType.NONE,
            "mean": PoolingType.MEAN,
            "cls": PoolingType.CLS,
            "last": PoolingType.LAST,
        }
        if pooling.lower() not in pooling_map:
            raise ValueError(
                f"Invalid pooling type: {pooling}. "
                f"Must be one of: {list(pooling_map.keys())}"
            )
        self._pooling_type = pooling_map[pooling.lower()]

        # Load model
        model_params = LlamaModelParams()
        model_params.n_gpu_layers = n_gpu_layers
        model_params.use_mmap = True
        self._model = LlamaModel(model_path, model_params)

        # Create context with embedding settings
        # Note: We always use pooling_type=NONE (0) and do pooling manually
        # because llama.cpp's internal pooling doesn't always work correctly
        # with generative models being used for embeddings
        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = n_batch
        ctx_params.pooling_type = 0  # NONE - we'll pool manually
        self._ctx = LlamaContext(self._model, ctx_params)
        # Enable embedding mode on the context
        self._ctx.set_embeddings_mode(True)

        # Cache embedding dimension and vocab
        self._n_embd = self._model.n_embd
        self._vocab = self._model.get_vocab()

        # Initialize embedding cache if requested
        self._cache: _LRUCache | None = None
        if cache_size > 0:
            self._cache = _LRUCache(cache_size)

    @property
    def dimension(self) -> int:
        """Return embedding dimension (n_embd)."""
        return self._n_embd

    @property
    def pooling(self) -> str:
        """Return current pooling strategy name."""
        pooling_names = {
            PoolingType.NONE: "none",
            PoolingType.MEAN: "mean",
            PoolingType.CLS: "cls",
            PoolingType.LAST: "last",
        }
        return pooling_names[self._pooling_type]

    @property
    def normalize(self) -> bool:
        """Whether L2 normalization is enabled."""
        return self._normalize

    @property
    def cache_enabled(self) -> bool:
        """Whether embedding cache is enabled."""
        return self._cache is not None

    def cache_info(self) -> CacheInfo | None:
        """Return cache statistics, or None if caching is disabled.

        Returns:
            CacheInfo with hits, misses, maxsize, currsize, or None

        Example:
            >>> embedder = Embedder("model.gguf", cache_size=1000)
            >>> embedder.embed("hello")
            >>> embedder.embed("hello")  # Cache hit
            >>> info = embedder.cache_info()
            >>> print(f"Hits: {info.hits}, Misses: {info.misses}")
            Hits: 1, Misses: 1
        """
        if self._cache is None:
            return None
        return self._cache.info()

    def cache_clear(self) -> None:
        """Clear the embedding cache and reset statistics.

        Does nothing if caching is disabled.
        """
        if self._cache is not None:
            self._cache.clear()

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        If caching is enabled (cache_size > 0), repeated calls with the same
        text will return cached results without recomputation.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as a list of floats
        """
        # Check cache first if enabled
        if self._cache is not None:
            cached = self._cache.get(text)
            if cached is not None:
                return list(cached)

        # Compute embedding
        result = self._embed_single(text)

        # Store in cache if enabled
        if self._cache is not None:
            self._cache.put(text, tuple(result.embedding))

        return result.embedding

    def embed_with_info(self, text: str) -> EmbeddingResult:
        """Embed a single text string and return detailed result.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding, text, and token count
        """
        return self._embed_single(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return [self.embed(text) for text in texts]

    def embed_documents(
        self,
        documents: list[str],
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Embed documents with optional progress tracking.

        Args:
            documents: List of document texts
            show_progress: Whether to show progress (prints to stdout)

        Returns:
            List of embedding vectors
        """
        embeddings = []
        total = len(documents)
        for i, doc in enumerate(documents):
            embeddings.append(self.embed(doc))
            if show_progress and (i + 1) % 10 == 0:
                print(f"Embedded {i + 1}/{total} documents")
        return embeddings

    def embed_iter(self, texts: list[str]) -> Iterator[list[float]]:
        """Embed texts and yield embeddings one at a time.

        Useful for processing large collections without storing all
        embeddings in memory at once.

        Args:
            texts: List of texts to embed

        Yields:
            Embedding vectors one at a time
        """
        for text in texts:
            yield self.embed(text)

    def _embed_single(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding vector
        """
        # Tokenize
        tokens = self._vocab.tokenize(text, add_special=True, parse_special=False)
        n_tokens = len(tokens)

        # Truncate if needed
        if n_tokens > self.n_ctx:
            tokens = tokens[: self.n_ctx]
            n_tokens = self.n_ctx

        # Clear context for new embedding
        self._ctx.kv_cache_clear()

        # Create batch - mark all tokens for output to enable pooling
        batch = LlamaBatch(n_tokens=n_tokens, embd=0, n_seq_max=1)
        for i, token in enumerate(tokens):
            # For embedding models, we need to mark all tokens for output
            # so that pooling can be computed correctly
            batch.add(token, i, [0], True)

        # Decode batch to compute embeddings
        self._ctx.decode(batch)

        # Get embeddings - we always use manual pooling for reliability
        raw_embeddings = self._ctx.get_embeddings()

        # Apply pooling strategy
        if self._pooling_type == PoolingType.MEAN:
            embedding = self._mean_pool_manual(raw_embeddings, n_tokens)
        elif self._pooling_type == PoolingType.CLS:
            # CLS pooling - use first token's embedding
            embedding = list(raw_embeddings[: self._n_embd])
        elif self._pooling_type == PoolingType.LAST:
            # Last token pooling
            offset = (n_tokens - 1) * self._n_embd
            embedding = list(raw_embeddings[offset : offset + self._n_embd])
        else:
            # NONE - still use mean as a reasonable default
            embedding = self._mean_pool_manual(raw_embeddings, n_tokens)

        # Normalize if requested
        if self._normalize:
            embedding = self._l2_normalize(embedding)

        return EmbeddingResult(
            embedding=embedding,
            text=text,
            token_count=n_tokens,
        )

    def _mean_pool_manual(
        self, raw_embeddings: list[float], n_tokens: int
    ) -> list[float]:
        """Compute mean pooling over token embeddings.

        Args:
            raw_embeddings: Flat list of embeddings (n_tokens * n_embd)
            n_tokens: Number of tokens

        Returns:
            Mean-pooled embedding vector
        """
        n_embd = self._n_embd
        if len(raw_embeddings) != n_tokens * n_embd:
            # Handle case where not all embeddings are present
            n_tokens = len(raw_embeddings) // n_embd

        result = [0.0] * n_embd
        for t in range(n_tokens):
            offset = t * n_embd
            for i in range(n_embd):
                result[i] += raw_embeddings[offset + i]

        # Divide by number of tokens
        for i in range(n_embd):
            result[i] /= n_tokens

        return result

    def _l2_normalize(self, embedding: list[float]) -> list[float]:
        """L2 normalize an embedding vector.

        Args:
            embedding: Input vector

        Returns:
            Normalized vector with unit L2 norm
        """
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            return [x / norm for x in embedding]
        return embedding

    def close(self) -> None:
        """Release resources."""
        # Context and model will be cleaned up by their destructors
        pass

    def __enter__(self) -> "Embedder":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        parts = [
            f"Embedder(model_path={self.model_path!r}",
            f"dimension={self.dimension}",
            f"pooling={self.pooling!r}",
            f"normalize={self.normalize}",
        ]
        if self._cache is not None:
            info = self._cache.info()
            parts.append(f"cache_size={info.maxsize}")
        return ", ".join(parts) + ")"
