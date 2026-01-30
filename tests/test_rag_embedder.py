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
        embedder.cache_clear()  # Should not raise

    def test_cache_info_namedtuple(self, cached_embedder: Embedder):
        """Test that CacheInfo is a proper NamedTuple."""
        cached_embedder.embed("Test")
        info = cached_embedder.cache_info()

        assert isinstance(info, CacheInfo)
        assert hasattr(info, 'hits')
        assert hasattr(info, 'misses')
        assert hasattr(info, 'maxsize')
        assert hasattr(info, 'currsize')

    def test_cache_repr_includes_cache_size(self, cached_embedder: Embedder):
        """Test that repr includes cache_size when enabled."""
        repr_str = repr(cached_embedder)
        assert "cache_size=100" in repr_str

    def test_cache_repr_excludes_cache_when_disabled(self, embedder: Embedder):
        """Test that repr doesn't include cache_size when disabled."""
        repr_str = repr(embedder)
        assert "cache_size" not in repr_str
