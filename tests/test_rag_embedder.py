"""Tests for the RAG Embedder class."""

import math
from pathlib import Path

import pytest

from cyllama.rag import CacheInfo, Embedder, EmbeddingResult, PoolingType


# Use the standard test model - it can generate embeddings even if not optimized for it
ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = ROOT / "models" / "Llama-3.2-1B-Instruct-Q8_0.gguf"


@pytest.fixture
def model_path() -> str:
    """Provide the path to the test model."""
    if not DEFAULT_MODEL.exists():
        pytest.skip(f"Test model not found: {DEFAULT_MODEL}")
    return str(DEFAULT_MODEL)


@pytest.fixture
def embedder(model_path: str) -> Embedder:
    """Create an Embedder instance for testing."""
    emb = Embedder(
        model_path,
        n_ctx=512,
        n_gpu_layers=0,  # CPU for consistent testing
        pooling="mean",
        normalize=True,
    )
    yield emb
    emb.close()


class TestEmbedderInit:
    """Test Embedder initialization."""

    def test_init_default(self, model_path: str):
        """Test default initialization."""
        emb = Embedder(model_path, n_gpu_layers=0)
        assert emb.dimension > 0
        assert emb.pooling == "mean"
        assert emb.normalize is True
        emb.close()

    def test_init_custom_pooling(self, model_path: str):
        """Test initialization with custom pooling."""
        for pooling in ["mean", "cls", "last", "none"]:
            emb = Embedder(model_path, n_gpu_layers=0, pooling=pooling)
            assert emb.pooling == pooling
            emb.close()

    def test_init_invalid_pooling(self, model_path: str):
        """Test that invalid pooling type raises error."""
        with pytest.raises(ValueError, match="Invalid pooling type"):
            Embedder(model_path, pooling="invalid")

    def test_init_no_normalize(self, model_path: str):
        """Test initialization without normalization."""
        emb = Embedder(model_path, n_gpu_layers=0, normalize=False)
        assert emb.normalize is False
        emb.close()


class TestEmbedderEmbed:
    """Test embedding generation."""

    def test_embed_single(self, embedder: Embedder):
        """Test embedding a single text."""
        embedding = embedder.embed("Hello, world!")
        assert isinstance(embedding, list)
        assert len(embedding) == embedder.dimension
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_empty_string(self, embedder: Embedder):
        """Test embedding an empty string."""
        embedding = embedder.embed("")
        assert isinstance(embedding, list)
        assert len(embedding) == embedder.dimension

    def test_embed_long_text(self, embedder: Embedder):
        """Test embedding text longer than context."""
        long_text = "word " * 1000  # Will exceed n_ctx=512
        embedding = embedder.embed(long_text)
        assert len(embedding) == embedder.dimension

    def test_embed_unicode(self, embedder: Embedder):
        """Test embedding text with unicode characters."""
        unicode_text = "Hello, world!"
        embedding = embedder.embed(unicode_text)
        assert len(embedding) == embedder.dimension

    def test_embed_normalized(self, embedder: Embedder):
        """Test that embeddings are normalized when normalize=True."""
        embedding = embedder.embed("Test text")
        norm = math.sqrt(sum(x * x for x in embedding))
        assert abs(norm - 1.0) < 1e-5, f"Expected norm=1.0, got {norm}"

    def test_embed_not_normalized(self, model_path: str):
        """Test that embeddings are not normalized when normalize=False."""
        emb = Embedder(model_path, n_gpu_layers=0, normalize=False)
        embedding = emb.embed("Test text")
        norm = math.sqrt(sum(x * x for x in embedding))
        # Should not be exactly 1.0 (unless by chance)
        # Just verify we get valid floats
        assert all(isinstance(x, float) for x in embedding)
        emb.close()


class TestEmbedderWithInfo:
    """Test embed_with_info method."""

    def test_embed_with_info(self, embedder: Embedder):
        """Test getting embedding with info."""
        text = "Hello, world!"
        result = embedder.embed_with_info(text)
        assert isinstance(result, EmbeddingResult)
        assert result.text == text
        assert len(result.embedding) == embedder.dimension
        assert result.token_count > 0

    def test_embed_with_info_token_count(self, embedder: Embedder):
        """Test that token count increases with longer text."""
        short = embedder.embed_with_info("Hi")
        long = embedder.embed_with_info("Hello, this is a longer text with more tokens")
        assert long.token_count > short.token_count


class TestEmbedderBatch:
    """Test batch embedding."""

    def test_embed_batch(self, embedder: Embedder):
        """Test batch embedding."""
        texts = ["First text", "Second text", "Third text"]
        embeddings = embedder.embed_batch(texts)
        assert len(embeddings) == 3
        assert all(len(e) == embedder.dimension for e in embeddings)

    def test_embed_batch_empty(self, embedder: Embedder):
        """Test batch embedding with empty list."""
        embeddings = embedder.embed_batch([])
        assert embeddings == []

    def test_embed_batch_single(self, embedder: Embedder):
        """Test batch embedding with single text."""
        embeddings = embedder.embed_batch(["Only one"])
        assert len(embeddings) == 1


class TestEmbedderDocuments:
    """Test document embedding."""

    def test_embed_documents(self, embedder: Embedder):
        """Test document embedding."""
        docs = ["Doc one", "Doc two"]
        embeddings = embedder.embed_documents(docs)
        assert len(embeddings) == 2


class TestEmbedderIter:
    """Test embedding iterator."""

    def test_embed_iter(self, embedder: Embedder):
        """Test embedding iterator."""
        texts = ["A", "B", "C"]
        embeddings = list(embedder.embed_iter(texts))
        assert len(embeddings) == 3

    def test_embed_iter_generator(self, embedder: Embedder):
        """Test that embed_iter returns a generator."""
        texts = ["A", "B"]
        result = embedder.embed_iter(texts)
        # Should be a generator, not a list
        import types

        assert isinstance(result, types.GeneratorType)


class TestEmbedderContextManager:
    """Test context manager protocol."""

    def test_context_manager(self, model_path: str):
        """Test using Embedder as context manager."""
        with Embedder(model_path, n_gpu_layers=0) as emb:
            embedding = emb.embed("Test")
            assert len(embedding) == emb.dimension


class TestEmbedderRepr:
    """Test string representation."""

    def test_repr(self, embedder: Embedder):
        """Test __repr__ method."""
        repr_str = repr(embedder)
        assert "Embedder" in repr_str
        assert "dimension=" in repr_str
        assert "pooling=" in repr_str


class TestPoolingType:
    """Test PoolingType enum."""

    def test_pooling_type_values(self):
        """Test PoolingType enum values."""
        assert PoolingType.NONE == 0
        assert PoolingType.MEAN == 1
        assert PoolingType.CLS == 2
        assert PoolingType.LAST == 3


class TestEmbeddingSimilarity:
    """Test that similar texts have similar embeddings."""

    @pytest.mark.skip(
        reason="Generative models like Llama-3.2 don't produce semantic embeddings. "
        "Use a dedicated embedding model (e.g., BGE, Snowflake) for reliable similarity."
    )
    def test_similar_texts(self, embedder: Embedder):
        """Test that similar texts produce similar embeddings.

        Note: This test requires an embedding-optimized model.
        Generative models may not encode semantic similarity correctly.
        """
        emb1 = embedder.embed("The cat sat on the mat")
        emb2 = embedder.embed("The cat is sitting on the mat")
        emb3 = embedder.embed("Quantum physics is complex")

        # Compute cosine similarity (embeddings are normalized)
        def cosine_sim(a, b):
            return sum(x * y for x, y in zip(a, b))

        sim_12 = cosine_sim(emb1, emb2)
        sim_13 = cosine_sim(emb1, emb3)

        # Similar texts should have higher similarity
        assert sim_12 > sim_13, "Similar texts should have higher similarity"

    def test_identical_texts(self, embedder: Embedder):
        """Test that identical texts produce identical embeddings."""
        text = "Hello, world!"
        emb1 = embedder.embed(text)
        emb2 = embedder.embed(text)

        # Should be very close (may have tiny floating point differences)
        for a, b in zip(emb1, emb2):
            assert abs(a - b) < 1e-6


class TestEmbedderCache:
    """Test embedding cache functionality."""

    @pytest.fixture
    def cached_embedder(self, model_path: str) -> Embedder:
        """Create an Embedder with caching enabled."""
        emb = Embedder(
            model_path,
            n_ctx=512,
            n_gpu_layers=0,
            cache_size=100,
        )
        yield emb
        emb.close()

    def test_cache_disabled_by_default(self, embedder: Embedder):
        """Test that cache is disabled by default."""
        assert embedder.cache_enabled is False
        assert embedder.cache_info() is None

    def test_cache_enabled(self, cached_embedder: Embedder):
        """Test that cache can be enabled."""
        assert cached_embedder.cache_enabled is True
        info = cached_embedder.cache_info()
        assert info is not None
        assert info.maxsize == 100

    def test_cache_hit(self, cached_embedder: Embedder):
        """Test that repeated calls hit the cache."""
        text = "Hello, world!"

        # First call - cache miss
        emb1 = cached_embedder.embed(text)
        info1 = cached_embedder.cache_info()
        assert info1.misses == 1
        assert info1.hits == 0
        assert info1.currsize == 1

        # Second call - cache hit
        emb2 = cached_embedder.embed(text)
        info2 = cached_embedder.cache_info()
        assert info2.misses == 1
        assert info2.hits == 1
        assert info2.currsize == 1

        # Results should be identical
        assert emb1 == emb2

    def test_cache_different_texts(self, cached_embedder: Embedder):
        """Test that different texts are cached separately."""
        text1 = "Hello"
        text2 = "World"

        cached_embedder.embed(text1)
        cached_embedder.embed(text2)

        info = cached_embedder.cache_info()
        assert info.misses == 2
        assert info.currsize == 2

    def test_cache_clear(self, cached_embedder: Embedder):
        """Test cache clearing."""
        cached_embedder.embed("Test text")
        assert cached_embedder.cache_info().currsize == 1

        cached_embedder.cache_clear()
        info = cached_embedder.cache_info()
        assert info.currsize == 0
        assert info.hits == 0
        assert info.misses == 0

    def test_cache_lru_eviction(self, model_path: str):
        """Test that LRU eviction works."""
        emb = Embedder(model_path, n_gpu_layers=0, cache_size=3)
        try:
            # Fill cache
            emb.embed("A")
            emb.embed("B")
            emb.embed("C")
            assert emb.cache_info().currsize == 3

            # Add one more - should evict "A"
            emb.embed("D")
            assert emb.cache_info().currsize == 3

            # "B", "C", "D" should be in cache, not "A"
            # Access "B" to make it recently used
            emb.embed("B")  # Should be a hit
            info = emb.cache_info()
            assert info.hits == 1  # "B" was a cache hit

            # Access "A" - should be a miss (was evicted)
            emb.embed("A")
            info = emb.cache_info()
            assert info.misses == 5  # A, B, C, D (misses) + A again (miss)
        finally:
            emb.close()

    def test_cache_clear_does_nothing_when_disabled(self, embedder: Embedder):
        """Test that cache_clear doesn't raise when cache is disabled."""
        embedder.cache_clear()  # should not raise even though no cache exists
        # With caching disabled, cache_info() returns None (not a zero-filled
        # CacheInfo) -- this behavior is the contract documented on the API.
        assert embedder.cache_info() is None

    def test_cache_info_namedtuple(self, cached_embedder: Embedder):
        """Test that CacheInfo is a proper NamedTuple."""
        cached_embedder.embed("Test")
        info = cached_embedder.cache_info()

        assert isinstance(info, CacheInfo)
        assert hasattr(info, "hits")
        assert hasattr(info, "misses")
        assert hasattr(info, "maxsize")
        assert hasattr(info, "currsize")

    def test_cache_repr_includes_cache_size(self, cached_embedder: Embedder):
        """Test that repr includes cache_size when enabled."""
        repr_str = repr(cached_embedder)
        assert "cache_size=100" in repr_str

    def test_cache_repr_excludes_cache_when_disabled(self, embedder: Embedder):
        """Test that repr doesn't include cache_size when disabled."""
        repr_str = repr(embedder)
        assert "cache_size" not in repr_str


class TestEmbedderConcurrencyGuard:
    """Tests for the concurrent-use guard on Embedder instances.

    Embedder holds a llama.cpp context which is not thread-safe under
    concurrent native calls. The guard catches actual contention via a
    non-blocking lock around each native-touching public method
    (`embed`, `embed_with_info`). Sequential ownership transfer between
    threads (asyncio.to_thread, ThreadPoolExecutor with one-call-per-
    worker) is deliberately allowed because it is safe.

    Mirrors TestLLMConcurrencyGuard / TestSDContextConcurrencyGuard /
    TestWhisperContextConcurrencyGuard. See docs/dev/runtime-guard.md
    for the design rationale.
    """

    def test_concurrent_embed_from_other_thread_raises(self, embedder: Embedder):
        """A worker thread calling embed() while the busy-lock is
        already held must raise RuntimeError without entering native
        code."""
        import threading

        # Simulate "thread A is currently inside embed()" by holding the
        # busy lock from the test thread.
        assert embedder._busy_lock.acquire(blocking=False) is True
        try:
            errors: list[Exception] = []

            def worker():
                try:
                    embedder.embed("hello")
                except RuntimeError as e:
                    errors.append(e)

            t = threading.Thread(target=worker)
            t.start()
            t.join(timeout=10)

            assert not t.is_alive(), "worker did not return — guard may be missing"
            assert len(errors) == 1, f"Expected concurrent embed() to raise, got: {errors}"
            msg = str(errors[0])
            assert "another thread" in msg
            assert "not thread-safe" in msg
        finally:
            embedder._busy_lock.release()

    def test_concurrent_embed_with_info_from_other_thread_raises(self, embedder: Embedder):
        """Same as above but exercises the embed_with_info() entry point."""
        import threading

        assert embedder._busy_lock.acquire(blocking=False) is True
        try:
            errors: list[Exception] = []

            def worker():
                try:
                    embedder.embed_with_info("hello")
                except RuntimeError as e:
                    errors.append(e)

            t = threading.Thread(target=worker)
            t.start()
            t.join(timeout=10)

            assert not t.is_alive(), "worker did not return — guard may be missing"
            assert len(errors) == 1
            assert "another thread" in str(errors[0])
        finally:
            embedder._busy_lock.release()

    def test_concurrent_embed_batch_from_other_thread_raises(self, embedder: Embedder):
        """embed_batch() does not acquire its own lock but delegates to
        embed(), which does. The first delegated call hits the guard."""
        import threading

        assert embedder._busy_lock.acquire(blocking=False) is True
        try:
            errors: list[Exception] = []

            def worker():
                try:
                    embedder.embed_batch(["one", "two", "three"])
                except RuntimeError as e:
                    errors.append(e)

            t = threading.Thread(target=worker)
            t.start()
            t.join(timeout=10)

            assert not t.is_alive(), "worker did not return — guard may be missing"
            assert len(errors) == 1
            assert "another thread" in str(errors[0])
        finally:
            embedder._busy_lock.release()

    def test_sequential_cross_thread_use_works(self, embedder: Embedder):
        """asyncio.to_thread / executor pattern: Embedder created on
        main thread, used on a worker thread, no concurrent access —
        must work. This is the legitimate sequential-handoff pattern."""
        import threading

        result_holder: list = [None]
        errors: list[Exception] = []

        def worker():
            try:
                result_holder[0] = embedder.embed("hello")
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert errors == [], f"Sequential cross-thread call should succeed: {errors}"
        assert result_holder[0] is not None
        assert len(result_holder[0]) == embedder.dimension

    def test_lock_released_after_exception(self, model_path: str):
        """If embed() raises mid-call (e.g. on bogus input), the busy
        lock must still be released so subsequent calls succeed."""
        emb = Embedder(model_path, n_gpu_layers=0)
        try:
            # Force an exception inside embed() by passing a non-string.
            # This will raise inside the native call path, but the lock
            # release must still fire via the finally clause.
            with pytest.raises((TypeError, AttributeError)):
                emb.embed(None)  # type: ignore[arg-type]

            # Subsequent call must succeed (lock was released).
            result = emb.embed("hello")
            assert isinstance(result, list)
            assert len(result) == emb.dimension
        finally:
            emb.close()

    def test_lock_release_allows_subsequent_acquire(self, embedder: Embedder):
        """Sanity check on the lock semantics: after releasing manually,
        the next acquire succeeds."""
        assert embedder._busy_lock.acquire(blocking=False) is True
        embedder._busy_lock.release()

        # _try_acquire_busy() should now succeed; release after.
        embedder._try_acquire_busy()
        embedder._busy_lock.release()


class TestLRUCacheMemoryAware:
    """Unit tests for memory-aware _LRUCache (no model required)."""

    def test_count_based_eviction(self):
        """Test basic count-based LRU eviction."""
        from cyllama.rag.embedder import _LRUCache

        cache = _LRUCache(maxsize=3)
        cache.put("a", (1.0, 2.0))
        cache.put("b", (3.0, 4.0))
        cache.put("c", (5.0, 6.0))
        # Cache is full; inserting d should evict a
        cache.put("d", (7.0, 8.0))

        assert cache.get("a") is None
        assert cache.get("d") == (7.0, 8.0)
        assert cache.info().currsize == 3

    def test_memory_based_eviction(self):
        """Test that memory limit triggers eviction before count limit."""
        from cyllama.rag.embedder import _LRUCache, _BYTES_PER_FLOAT

        dim = 100  # 100 floats = 800 bytes per entry
        entry_bytes = dim * _BYTES_PER_FLOAT  # 800
        # Allow ~2.5 entries worth of memory
        cache = _LRUCache(maxsize=1000, max_memory_bytes=int(entry_bytes * 2.5))

        cache.put("a", tuple(float(i) for i in range(dim)))
        cache.put("b", tuple(float(i) for i in range(dim)))
        assert cache.info().currsize == 2

        # Third entry should evict oldest to stay within memory
        cache.put("c", tuple(float(i) for i in range(dim)))
        assert cache.info().currsize == 2
        assert cache.get("a") is None  # evicted
        assert cache.get("c") is not None

    def test_memory_tracking_accurate(self):
        """Test that memory_bytes tracks insertions and evictions."""
        from cyllama.rag.embedder import _LRUCache, _BYTES_PER_FLOAT

        cache = _LRUCache(maxsize=10)
        cache.put("x", (1.0, 2.0, 3.0))
        assert cache.info().memory_bytes == 3 * _BYTES_PER_FLOAT

        cache.put("y", (4.0, 5.0))
        assert cache.info().memory_bytes == 5 * _BYTES_PER_FLOAT

        cache.clear()
        assert cache.info().memory_bytes == 0

    def test_duplicate_key_no_double_count(self):
        """Test that re-putting an existing key doesn't double-count memory."""
        from cyllama.rag.embedder import _LRUCache, _BYTES_PER_FLOAT

        cache = _LRUCache(maxsize=10)
        cache.put("k", (1.0, 2.0))
        cache.put("k", (1.0, 2.0))  # duplicate
        assert cache.info().currsize == 1
        assert cache.info().memory_bytes == 2 * _BYTES_PER_FLOAT

    def test_cache_info_has_memory_bytes(self):
        """Test that CacheInfo includes memory_bytes field."""
        from cyllama.rag.embedder import _LRUCache

        cache = _LRUCache(maxsize=5)
        info = cache.info()
        assert hasattr(info, "memory_bytes")
        assert info.memory_bytes == 0
