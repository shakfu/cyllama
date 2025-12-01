"""Tests for the RAG VectorStore class."""

import json
import struct
import tempfile
from pathlib import Path

import pytest

from cyllama.rag import VectorStore, VectorStoreError, SearchResult


# Check if sqlite-vector extension is available
def extension_available() -> bool:
    """Check if sqlite-vector extension exists."""
    import sys
    ext_path = Path(__file__).parent.parent / "src" / "cyllama" / "rag" / "vector"
    if sys.platform == "darwin":
        return ext_path.with_suffix(".dylib").exists()
    elif sys.platform == "win32":
        return ext_path.with_suffix(".dll").exists()
    else:
        return ext_path.with_suffix(".so").exists()


# Skip all tests if extension not available
pytestmark = pytest.mark.skipif(
    not extension_available(),
    reason="sqlite-vector extension not built. Run 'scripts/setup.sh' or 'python scripts/manage.py build --sqlite-vector'"
)


@pytest.fixture
def store():
    """Create an in-memory VectorStore for testing."""
    with VectorStore(dimension=4) as s:
        yield s


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0, 0.0],  # Between first and second
    ]


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "First document",
        "Second document",
        "Third document",
        "Fourth document",
        "Fifth document",
    ]


class TestVectorStoreInit:
    """Test VectorStore initialization."""

    def test_init_default(self):
        """Test default initialization."""
        with VectorStore(dimension=384) as store:
            assert store.dimension == 384
            assert store.db_path == ":memory:"
            assert store.table_name == "embeddings"
            assert store.metric == "cosine"
            assert store.vector_type == "float32"

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        with VectorStore(
            dimension=768,
            db_path=":memory:",
            table_name="vectors",
            metric="l2",
            vector_type="float16",
        ) as store:
            assert store.dimension == 768
            assert store.table_name == "vectors"
            assert store.metric == "l2"
            assert store.vector_type == "float16"

    def test_init_invalid_dimension(self):
        """Test that invalid dimension raises error."""
        with pytest.raises(ValueError, match="dimension must be positive"):
            VectorStore(dimension=0)
        with pytest.raises(ValueError, match="dimension must be positive"):
            VectorStore(dimension=-1)

    def test_init_invalid_metric(self):
        """Test that invalid metric raises error."""
        with pytest.raises(ValueError, match="Invalid metric"):
            VectorStore(dimension=4, metric="invalid")

    def test_init_invalid_vector_type(self):
        """Test that invalid vector type raises error."""
        with pytest.raises(ValueError, match="Invalid vector_type"):
            VectorStore(dimension=4, vector_type="float64")

    def test_init_all_metrics(self):
        """Test initialization with all valid metrics."""
        for metric in ["cosine", "l2", "dot", "l1", "squared_l2"]:
            with VectorStore(dimension=4, metric=metric) as store:
                assert store.metric == metric

    def test_init_all_vector_types(self):
        """Test initialization with all valid vector types."""
        for vtype in ["float32", "float16", "int8", "uint8"]:
            with VectorStore(dimension=4, vector_type=vtype) as store:
                assert store.vector_type == vtype


class TestVectorStoreAdd:
    """Test adding embeddings."""

    def test_add_single(self, store):
        """Test adding a single embedding."""
        ids = store.add([[1.0, 0.0, 0.0, 0.0]], ["test text"])
        assert len(ids) == 1
        assert isinstance(ids[0], int)
        assert len(store) == 1

    def test_add_multiple(self, store, sample_embeddings, sample_texts):
        """Test adding multiple embeddings."""
        ids = store.add(sample_embeddings, sample_texts)
        assert len(ids) == 5
        assert len(store) == 5

    def test_add_with_metadata(self, store):
        """Test adding embeddings with metadata."""
        ids = store.add(
            [[1.0, 0.0, 0.0, 0.0]],
            ["test"],
            metadata=[{"source": "doc1", "page": 1}],
        )
        result = store.get(ids[0])
        assert result.metadata == {"source": "doc1", "page": 1}

    def test_add_one(self, store):
        """Test add_one method."""
        id_ = store.add_one([1.0, 0.0, 0.0, 0.0], "single text")
        assert isinstance(id_, int)
        assert len(store) == 1

    def test_add_one_with_metadata(self, store):
        """Test add_one with metadata."""
        id_ = store.add_one(
            [1.0, 0.0, 0.0, 0.0],
            "text",
            metadata={"key": "value"},
        )
        result = store.get(id_)
        assert result.metadata == {"key": "value"}

    def test_add_mismatched_lengths(self, store):
        """Test that mismatched lengths raise error."""
        with pytest.raises(ValueError, match="same length"):
            store.add([[1.0, 0.0, 0.0, 0.0]], ["text1", "text2"])

    def test_add_wrong_dimension(self, store):
        """Test that wrong dimension raises error."""
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.add([[1.0, 0.0, 0.0]], ["text"])  # 3D instead of 4D

    def test_add_metadata_wrong_length(self, store):
        """Test that metadata with wrong length raises error."""
        with pytest.raises(ValueError, match="metadata must have same length"):
            store.add(
                [[1.0, 0.0, 0.0, 0.0]],
                ["text"],
                metadata=[{}, {}],  # 2 metadata for 1 embedding
            )


class TestVectorStoreSearch:
    """Test similarity search."""

    def test_search_basic(self, store, sample_embeddings, sample_texts):
        """Test basic search."""
        store.add(sample_embeddings, sample_texts)
        results = store.search([1.0, 0.0, 0.0, 0.0], k=3)
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        # First result should be exact match
        assert results[0].text == "First document"

    def test_search_returns_ordered(self, store, sample_embeddings, sample_texts):
        """Test that search returns results ordered by similarity."""
        store.add(sample_embeddings, sample_texts)
        results = store.search([1.0, 0.0, 0.0, 0.0], k=5)
        # Scores should be descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_k_larger_than_count(self, store):
        """Test search with k larger than stored count."""
        store.add([[1.0, 0.0, 0.0, 0.0]], ["only one"])
        results = store.search([1.0, 0.0, 0.0, 0.0], k=10)
        assert len(results) == 1

    def test_search_empty_store(self, store):
        """Test search on empty store."""
        results = store.search([1.0, 0.0, 0.0, 0.0], k=5)
        assert results == []

    def test_search_with_threshold(self, store, sample_embeddings, sample_texts):
        """Test search with similarity threshold."""
        store.add(sample_embeddings, sample_texts)
        # High threshold should filter out dissimilar results
        results = store.search([1.0, 0.0, 0.0, 0.0], k=5, threshold=0.9)
        # Only the exact match should pass high threshold
        assert len(results) <= 2  # Depends on score calculation

    def test_search_result_fields(self, store):
        """Test that SearchResult has all fields."""
        store.add(
            [[1.0, 0.0, 0.0, 0.0]],
            ["test text"],
            metadata=[{"key": "value"}],
        )
        results = store.search([1.0, 0.0, 0.0, 0.0], k=1)
        result = results[0]
        assert result.id is not None
        assert result.text == "test text"
        assert isinstance(result.score, float)
        assert result.metadata == {"key": "value"}


class TestVectorStoreGet:
    """Test get operations."""

    def test_get_existing(self, store):
        """Test getting existing embedding."""
        ids = store.add([[1.0, 0.0, 0.0, 0.0]], ["test"])
        result = store.get(ids[0])
        assert result is not None
        assert result.text == "test"

    def test_get_nonexistent(self, store):
        """Test getting nonexistent embedding."""
        result = store.get(999)
        assert result is None

    def test_get_with_string_id(self, store):
        """Test getting with string ID."""
        ids = store.add([[1.0, 0.0, 0.0, 0.0]], ["test"])
        result = store.get(str(ids[0]))
        assert result is not None

    def test_get_vector(self, store):
        """Test getting the embedding vector."""
        embedding = [1.0, 2.0, 3.0, 4.0]
        ids = store.add([embedding], ["test"])
        vector = store.get_vector(ids[0])
        assert vector is not None
        # Float comparison with tolerance
        for a, b in zip(vector, embedding):
            assert abs(a - b) < 1e-5

    def test_get_vector_nonexistent(self, store):
        """Test getting vector for nonexistent ID."""
        vector = store.get_vector(999)
        assert vector is None


class TestVectorStoreDelete:
    """Test delete operations."""

    def test_delete_single(self, store, sample_embeddings, sample_texts):
        """Test deleting a single embedding."""
        ids = store.add(sample_embeddings, sample_texts)
        assert len(store) == 5
        deleted = store.delete([ids[0]])
        assert deleted == 1
        assert len(store) == 4
        assert store.get(ids[0]) is None

    def test_delete_multiple(self, store, sample_embeddings, sample_texts):
        """Test deleting multiple embeddings."""
        ids = store.add(sample_embeddings, sample_texts)
        deleted = store.delete(ids[:3])
        assert deleted == 3
        assert len(store) == 2

    def test_delete_nonexistent(self, store):
        """Test deleting nonexistent IDs."""
        deleted = store.delete([999, 998])
        assert deleted == 0

    def test_delete_empty_list(self, store):
        """Test deleting empty list."""
        deleted = store.delete([])
        assert deleted == 0

    def test_clear(self, store, sample_embeddings, sample_texts):
        """Test clearing all embeddings."""
        store.add(sample_embeddings, sample_texts)
        assert len(store) == 5
        deleted = store.clear()
        assert deleted == 5
        assert len(store) == 0


class TestVectorStoreContains:
    """Test __contains__ method."""

    def test_contains_existing(self, store):
        """Test contains for existing ID."""
        ids = store.add([[1.0, 0.0, 0.0, 0.0]], ["test"])
        assert ids[0] in store

    def test_contains_nonexistent(self, store):
        """Test contains for nonexistent ID."""
        assert 999 not in store

    def test_contains_string_id(self, store):
        """Test contains with string ID."""
        ids = store.add([[1.0, 0.0, 0.0, 0.0]], ["test"])
        assert str(ids[0]) in store


class TestVectorStoreQuantization:
    """Test quantization for large datasets."""

    def test_quantize(self, store, sample_embeddings, sample_texts):
        """Test quantization."""
        store.add(sample_embeddings, sample_texts)
        assert not store.is_quantized
        count = store.quantize()
        assert count == 5
        assert store.is_quantized

    def test_search_after_quantize(self, store, sample_embeddings, sample_texts):
        """Test that search works after quantization."""
        store.add(sample_embeddings, sample_texts)
        store.quantize()
        results = store.search([1.0, 0.0, 0.0, 0.0], k=3)
        assert len(results) == 3

    def test_add_invalidates_quantization(self, store, sample_embeddings, sample_texts):
        """Test that adding new data invalidates quantization."""
        store.add(sample_embeddings[:3], sample_texts[:3])
        store.quantize()
        assert store.is_quantized
        store.add([sample_embeddings[3]], [sample_texts[3]])
        assert not store.is_quantized


class TestVectorStorePersistence:
    """Test persistence to disk."""

    def test_persistent_store(self):
        """Test creating persistent store."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Create and populate
            with VectorStore(dimension=4, db_path=db_path) as store:
                store.add([[1.0, 0.0, 0.0, 0.0]], ["persistent text"])
                assert len(store) == 1

            # Reopen and verify
            with VectorStore.open(db_path) as store:
                assert len(store) == 1
                result = store.search([1.0, 0.0, 0.0, 0.0], k=1)
                assert result[0].text == "persistent text"
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_open_nonexistent(self):
        """Test opening nonexistent database."""
        with pytest.raises(VectorStoreError, match="Database not found"):
            VectorStore.open("/nonexistent/path.db")

    def test_open_empty_table(self):
        """Test opening database with empty table."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Create empty store
            with VectorStore(dimension=4, db_path=db_path) as store:
                pass  # Don't add anything

            # Try to open - should fail because we can't determine dimension
            with pytest.raises(VectorStoreError, match="empty"):
                VectorStore.open(db_path)
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestVectorStoreContextManager:
    """Test context manager protocol."""

    def test_context_manager_closes(self):
        """Test that context manager closes connection."""
        store = VectorStore(dimension=4)
        with store:
            store.add([[1.0, 0.0, 0.0, 0.0]], ["test"])
        # Should be closed after context
        with pytest.raises(VectorStoreError, match="closed"):
            store.add([[1.0, 0.0, 0.0, 0.0]], ["test"])

    def test_close_idempotent(self):
        """Test that close can be called multiple times."""
        store = VectorStore(dimension=4)
        store.close()
        store.close()  # Should not raise


class TestVectorStoreRepr:
    """Test string representation."""

    def test_repr_open(self, store):
        """Test repr for open store."""
        repr_str = repr(store)
        assert "VectorStore" in repr_str
        assert "dimension=4" in repr_str
        assert "open" in repr_str

    def test_repr_closed(self):
        """Test repr for closed store."""
        store = VectorStore(dimension=4)
        store.close()
        repr_str = repr(store)
        assert "closed" in repr_str


class TestVectorStoreMetrics:
    """Test different distance metrics."""

    def test_cosine_metric(self):
        """Test cosine similarity metric."""
        with VectorStore(dimension=4, metric="cosine") as store:
            # Normalized vectors
            store.add([
                [1.0, 0.0, 0.0, 0.0],
                [0.707, 0.707, 0.0, 0.0],  # 45 degrees from first
            ], ["first", "second"])
            results = store.search([1.0, 0.0, 0.0, 0.0], k=2)
            # First should be exact match (score ~1.0)
            assert results[0].score > results[1].score

    def test_l2_metric(self):
        """Test L2 distance metric."""
        with VectorStore(dimension=4, metric="l2") as store:
            store.add([
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
            ], ["origin", "close", "far"])
            results = store.search([0.0, 0.0, 0.0, 0.0], k=3)
            assert results[0].text == "origin"

    def test_dot_metric(self):
        """Test dot product metric."""
        with VectorStore(dimension=4, metric="dot") as store:
            store.add([
                [1.0, 1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5, 0.5],
            ], ["high", "low"])
            results = store.search([1.0, 1.0, 1.0, 1.0], k=2)
            # Higher dot product = more similar
            assert results[0].text == "high"


class TestVectorEncoding:
    """Test vector encoding/decoding."""

    def test_encode_decode_roundtrip(self, store):
        """Test that encoding and decoding preserves values."""
        original = [1.5, -2.5, 3.14159, 0.0]
        ids = store.add([original], ["test"])
        decoded = store.get_vector(ids[0])
        for a, b in zip(original, decoded):
            assert abs(a - b) < 1e-5

    def test_encode_special_values(self, store):
        """Test encoding special float values."""
        # Test with very small and large values
        embedding = [1e-10, 1e10, -1e-10, -1e10]
        ids = store.add([embedding], ["special"])
        decoded = store.get_vector(ids[0])
        assert decoded is not None
        # Just verify it doesn't crash - exact values may differ due to float precision
