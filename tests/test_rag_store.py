"""Tests for the RAG VectorStore class."""

import sys
import tempfile
from pathlib import Path

import pytest

from cyllama.rag import SqliteVectorStore, VectorStoreError, SearchResult


# Check if sqlite-vector extension is available
def extension_available() -> bool:
    """Check if sqlite-vector extension exists and can be loaded."""
    import sqlite3
    import sys

    if not hasattr(sqlite3.Connection, "enable_load_extension"):
        return False
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
    reason="sqlite-vector extension not built. Run 'scripts/setup.sh' or 'python scripts/manage.py build --sqlite-vector'",
)


@pytest.fixture
def store():
    """Create an in-memory VectorStore for testing."""
    with SqliteVectorStore(dimension=4) as s:
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
        with SqliteVectorStore(dimension=384) as store:
            assert store.dimension == 384
            assert store.db_path == ":memory:"
            assert store.table_name == "embeddings"
            assert store.metric == "cosine"
            assert store.vector_type == "float32"

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        with SqliteVectorStore(
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
            SqliteVectorStore(dimension=0)
        with pytest.raises(ValueError, match="dimension must be positive"):
            SqliteVectorStore(dimension=-1)

    def test_init_invalid_metric(self):
        """Test that invalid metric raises error."""
        with pytest.raises(ValueError, match="Invalid metric"):
            SqliteVectorStore(dimension=4, metric="invalid")

    def test_init_invalid_vector_type(self):
        """Test that invalid vector type raises error."""
        with pytest.raises(ValueError, match="Invalid vector_type"):
            SqliteVectorStore(dimension=4, vector_type="float64")

    def test_init_all_metrics(self):
        """Test initialization with all valid metrics."""
        for metric in ["cosine", "l2", "dot", "l1", "squared_l2"]:
            with SqliteVectorStore(dimension=4, metric=metric) as store:
                assert store.metric == metric

    def test_init_all_vector_types(self):
        """Test initialization with all valid vector types."""
        for vtype in ["float32", "float16", "int8", "uint8"]:
            with SqliteVectorStore(dimension=4, vector_type=vtype) as store:
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
            with SqliteVectorStore(dimension=4, db_path=db_path) as store:
                store.add([[1.0, 0.0, 0.0, 0.0]], ["persistent text"])
                assert len(store) == 1

            # Reopen and verify
            with SqliteVectorStore.open(db_path) as store:
                assert len(store) == 1
                result = store.search([1.0, 0.0, 0.0, 0.0], k=1)
                assert result[0].text == "persistent text"
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_open_nonexistent(self):
        """Test opening nonexistent database."""
        with pytest.raises(VectorStoreError, match="Database not found"):
            SqliteVectorStore.open("/nonexistent/path.db")

    def test_open_empty_table(self):
        """Test opening database with empty table succeeds via stored metadata."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Create empty store
            with SqliteVectorStore(dimension=4, db_path=db_path) as store:
                pass  # Don't add anything

            # Should succeed — dimension/metric/vector_type are in metadata table
            with SqliteVectorStore.open(db_path) as store:
                assert store.dimension == 4
                assert store.metric == "cosine"
                assert store.vector_type == "float32"
                assert len(store) == 0
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestVectorStoreMetadataCompatibility:
    """Test that reopening a populated DB with incompatible config raises
    a clear error rather than silently corrupting the index by mixing
    two configurations.

    The metadata schema records: dimension, metric, vector_type (hard
    mismatches), embedding_model_basename, embedding_model_size_bytes,
    chunk_size, chunk_overlap (soft mismatches -- only fire when the
    caller passes the field). All checks happen inside `_init_table`
    on every constructor call against a populated DB.
    """

    def _make_db(self, tmp_path, **kwargs):
        """Create a populated DB at a temp path with the given config,
        close it, and return the path. Used as the 'before' state for
        every reopen test in this class."""
        db_path = str(tmp_path / "store.db")
        defaults = dict(dimension=4, db_path=db_path)
        defaults.update(kwargs)
        with SqliteVectorStore(**defaults) as store:
            store.add([[1.0, 0.0, 0.0, 0.0]], ["sample"])
        return db_path

    # ---- hard mismatches: always fire on any reopen ----

    def test_reopen_with_different_dimension_raises(self, tmp_path):
        db_path = self._make_db(tmp_path)
        with pytest.raises(VectorStoreError, match="dimension"):
            SqliteVectorStore(dimension=8, db_path=db_path)

    def test_reopen_with_different_metric_raises(self, tmp_path):
        db_path = self._make_db(tmp_path, metric="cosine")
        with pytest.raises(VectorStoreError, match="metric"):
            SqliteVectorStore(dimension=4, db_path=db_path, metric="l2")

    def test_reopen_with_different_vector_type_raises(self, tmp_path):
        db_path = self._make_db(tmp_path, vector_type="float32")
        with pytest.raises(VectorStoreError, match="vector_type"):
            SqliteVectorStore(dimension=4, db_path=db_path, vector_type="int8")

    # ---- soft mismatches: only fire when the caller provides the field ----

    def test_reopen_with_different_embedding_model_basename_raises(self, tmp_path):
        db_path = self._make_db(tmp_path, embedding_model_path="/fake/embedder-a.gguf")
        with pytest.raises(VectorStoreError, match="embedding model"):
            SqliteVectorStore(
                dimension=4,
                db_path=db_path,
                embedding_model_path="/fake/embedder-b.gguf",
            )

    def test_reopen_with_different_embedding_model_size_raises(self, tmp_path, monkeypatch):
        """Same basename but different file size: catches the
        re-quantization case (q4 -> q8 of the same model)."""
        # Create the original DB with a real on-disk fake embedder
        # so getsize() returns a real value to record
        fake_embedder = tmp_path / "embedder.gguf"
        fake_embedder.write_bytes(b"x" * 1000)
        db_path = self._make_db(tmp_path, embedding_model_path=str(fake_embedder))

        # Now grow the file to a different size and try to reopen
        fake_embedder.write_bytes(b"x" * 2000)
        with pytest.raises(VectorStoreError, match="bytes"):
            SqliteVectorStore(dimension=4, db_path=db_path, embedding_model_path=str(fake_embedder))

    def test_reopen_with_different_chunk_size_raises(self, tmp_path):
        db_path = self._make_db(tmp_path, chunk_size=512)
        with pytest.raises(VectorStoreError, match="chunk_size"):
            SqliteVectorStore(dimension=4, db_path=db_path, chunk_size=1024)

    def test_reopen_with_different_chunk_overlap_raises(self, tmp_path):
        db_path = self._make_db(tmp_path, chunk_overlap=50)
        with pytest.raises(VectorStoreError, match="chunk_overlap"):
            SqliteVectorStore(dimension=4, db_path=db_path, chunk_overlap=100)

    # ---- soft mismatches OPT-OUT: caller doesn't pass the field ----

    def test_reopen_without_optional_fields_skips_soft_checks(self, tmp_path):
        """A caller that doesn't pass embedding_model_path/chunk_size/
        chunk_overlap explicitly opts out of those compatibility checks.
        Useful for read-only consumers (e.g. the `cyllama rag --db PATH`
        query-only case where the user doesn't want to re-supply the
        original config)."""
        db_path = self._make_db(
            tmp_path,
            embedding_model_path="/fake/embedder.gguf",
            chunk_size=512,
            chunk_overlap=50,
        )
        # Reopen with no soft fields -- should succeed
        with SqliteVectorStore(dimension=4, db_path=db_path) as store:
            assert len(store) == 1

    # ---- happy path: matching config reopens cleanly ----

    def test_reopen_with_matching_config_succeeds(self, tmp_path):
        db_path = self._make_db(
            tmp_path,
            dimension=4,
            metric="cosine",
            vector_type="float32",
            embedding_model_path="/fake/embedder.gguf",
            chunk_size=512,
            chunk_overlap=50,
        )
        with SqliteVectorStore(
            dimension=4,
            db_path=db_path,
            metric="cosine",
            vector_type="float32",
            embedding_model_path="/fake/embedder.gguf",
            chunk_size=512,
            chunk_overlap=50,
        ) as store:
            assert len(store) == 1

    def test_metadata_persists_across_multiple_reopens(self, tmp_path):
        """Metadata written on first init must survive multiple
        reopen cycles. Pin this so a later refactor of `_init_table`
        can't accidentally start overwriting metadata on every open."""
        db_path = str(tmp_path / "store.db")

        # First init: writes metadata
        with SqliteVectorStore(
            dimension=4,
            db_path=db_path,
            embedding_model_path="/fake/e.gguf",
            chunk_size=512,
            chunk_overlap=50,
        ) as store:
            store.add([[1.0, 0.0, 0.0, 0.0]], ["one"])

        # Reopen 1: matches, adds another row
        with SqliteVectorStore(
            dimension=4,
            db_path=db_path,
            embedding_model_path="/fake/e.gguf",
            chunk_size=512,
            chunk_overlap=50,
        ) as store:
            assert len(store) == 1
            store.add([[0.0, 1.0, 0.0, 0.0]], ["two"])

        # Reopen 2: still matches (the metadata wasn't corrupted by
        # the second open)
        with SqliteVectorStore(
            dimension=4,
            db_path=db_path,
            embedding_model_path="/fake/e.gguf",
            chunk_size=512,
            chunk_overlap=50,
        ) as store:
            assert len(store) == 2

    def test_created_at_is_stable_across_reopens(self, tmp_path):
        """`created_at` should be set on the first init and not
        overwritten on subsequent reopens. Lets users see when the
        index was first built, regardless of when it was last
        accessed."""
        db_path = str(tmp_path / "store.db")

        with SqliteVectorStore(dimension=4, db_path=db_path) as store:
            cursor = store.conn.execute(f"SELECT value FROM {store.table_name}_meta WHERE key='created_at'")
            row = cursor.fetchone()
            assert row is not None, "created_at should be written on first init"
            first_created = row[0]

        with SqliteVectorStore(dimension=4, db_path=db_path) as store:
            cursor = store.conn.execute(f"SELECT value FROM {store.table_name}_meta WHERE key='created_at'")
            second_created = cursor.fetchone()[0]

        assert first_created == second_created, (
            f"created_at must not change on reopen (was {first_created!r}, became {second_created!r})"
        )


class TestVectorStoreSourceDedup:
    """Test the source-deduplication table created in `_init_table`
    and the helper methods used by the high-level RAG layer to decide
    whether to skip a previously-indexed source.

    The dedup contract:

    * Every call to ``add(..., source_hash=H, source_label=L)`` inserts
      a row into ``{table_name}_sources`` in the same commit as the
      chunk inserts (atomicity guarantee).
    * ``is_source_indexed(H)`` returns True iff a row with that hash
      already exists.
    * ``get_source_by_label(L)`` lets the caller detect "same label,
      different content" -- a file whose basename matches but whose
      content differs.
    * ``list_sources()`` returns all recorded sources for diagnostics.
    * The legacy ``add()`` call (no ``source_hash`` kwarg) still works
      and does NOT touch the sources table -- backward compatible.
    """

    def test_sources_table_created_on_init(self, tmp_path):
        """The sources table must exist after `__init__` even on a
        store that has never had a source recorded. Otherwise the
        first `is_source_indexed` query would raise."""
        db_path = str(tmp_path / "store.db")
        with SqliteVectorStore(dimension=4, db_path=db_path) as store:
            # Should not raise
            assert store.is_source_indexed("nonexistent") is False
            assert store.list_sources() == []

    def test_add_without_source_hash_does_not_record(self):
        """Backward compat: add() without source_hash should leave the
        sources table empty. Pinned so a future refactor can't
        accidentally start recording every add() as an anonymous
        source."""
        with SqliteVectorStore(dimension=3) as store:
            store.add([[1.0, 0.0, 0.0]], ["chunk"])
            assert store.list_sources() == []

    def test_add_with_source_hash_records_one_row(self):
        with SqliteVectorStore(dimension=3) as store:
            store.add(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                ["chunk a", "chunk b"],
                source_hash="abc123",
                source_label="foo.txt",
            )
            sources = store.list_sources()
            assert len(sources) == 1
            assert sources[0]["content_hash"] == "abc123"
            assert sources[0]["source_label"] == "foo.txt"
            assert sources[0]["chunk_count"] == 2
            assert "indexed_at" in sources[0]

    def test_is_source_indexed_returns_true_after_add(self):
        with SqliteVectorStore(dimension=3) as store:
            assert not store.is_source_indexed("hash1")
            store.add(
                [[1.0, 0.0, 0.0]],
                ["x"],
                source_hash="hash1",
                source_label="x.txt",
            )
            assert store.is_source_indexed("hash1")
            # Different hash still returns False
            assert not store.is_source_indexed("hash2")

    def test_get_source_by_label_returns_row_or_none(self):
        with SqliteVectorStore(dimension=3) as store:
            assert store.get_source_by_label("missing") is None
            store.add(
                [[1.0, 0.0, 0.0]],
                ["x"],
                source_hash="abc",
                source_label="present.txt",
            )
            row = store.get_source_by_label("present.txt")
            assert row is not None
            assert row["content_hash"] == "abc"
            assert row["source_label"] == "present.txt"
            assert row["chunk_count"] == 1

    def test_add_requires_label_when_hash_provided(self):
        with SqliteVectorStore(dimension=3) as store:
            with pytest.raises(ValueError, match="source_label"):
                store.add(
                    [[1.0, 0.0, 0.0]],
                    ["x"],
                    source_hash="abc",  # no source_label
                )

    def test_source_record_persists_across_reopen(self, tmp_path):
        """Critical: the dedup table must round-trip through close+
        reopen so file-backed stores actually deduplicate across
        process invocations."""
        db_path = str(tmp_path / "store.db")

        with SqliteVectorStore(dimension=3, db_path=db_path) as store:
            store.add(
                [[1.0, 0.0, 0.0]],
                ["chunk"],
                source_hash="persistent_hash",
                source_label="persistent.txt",
            )

        # Reopen and verify the source is still there
        with SqliteVectorStore(dimension=3, db_path=db_path) as store:
            assert store.is_source_indexed("persistent_hash")
            sources = store.list_sources()
            assert len(sources) == 1

    def test_atomic_insert_chunks_and_source_in_one_commit(self):
        """The chunks and the source row must be visible in the same
        transaction. We test this indirectly: if the chunk inserts
        succeed but the source insert fails (e.g. duplicate primary
        key), the chunks should be rolled back so the dedup table is
        never out of sync with the data table.
        """
        with SqliteVectorStore(dimension=3) as store:
            # First add succeeds
            store.add(
                [[1.0, 0.0, 0.0]],
                ["original"],
                source_hash="dup_hash",
                source_label="a.txt",
            )
            assert len(store) == 1

            # Second add with the SAME source_hash would violate the
            # PRIMARY KEY constraint on the sources table, which
            # should roll back the whole transaction (no new chunks).
            with pytest.raises(Exception):  # sqlite3.IntegrityError
                store.add(
                    [[0.0, 1.0, 0.0]],
                    ["new"],
                    source_hash="dup_hash",
                    source_label="a.txt",
                )

            # Verify atomicity: count is still 1, not 2.
            assert len(store) == 1

    def test_multiple_sources_recorded_independently(self):
        with SqliteVectorStore(dimension=3) as store:
            store.add([[1.0, 0.0, 0.0]], ["a"], source_hash="h1", source_label="a.txt")
            store.add([[0.0, 1.0, 0.0]], ["b"], source_hash="h2", source_label="b.txt")
            store.add([[0.0, 0.0, 1.0]], ["c"], source_hash="h3", source_label="c.txt")

            assert len(store.list_sources()) == 3
            assert store.is_source_indexed("h1")
            assert store.is_source_indexed("h2")
            assert store.is_source_indexed("h3")


class TestVectorStoreContextManager:
    """Test context manager protocol."""

    def test_context_manager_closes(self):
        """Test that context manager closes connection."""
        store = SqliteVectorStore(dimension=4)
        with store:
            store.add([[1.0, 0.0, 0.0, 0.0]], ["test"])
        # Should be closed after context
        with pytest.raises(VectorStoreError, match="closed"):
            store.add([[1.0, 0.0, 0.0, 0.0]], ["test"])

    def test_close_idempotent(self):
        """Test that close can be called multiple times."""
        store = SqliteVectorStore(dimension=4)
        store.close()
        store.close()  # Should not raise on second call
        # After close, any data operation must raise with a "closed" marker.
        with pytest.raises(VectorStoreError, match="closed"):
            store.add([[1.0, 0.0, 0.0, 0.0]], ["test"])


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
        store = SqliteVectorStore(dimension=4)
        store.close()
        repr_str = repr(store)
        assert "closed" in repr_str


class TestVectorStoreMetrics:
    """Test different distance metrics."""

    def test_cosine_metric(self):
        """Test cosine similarity metric."""
        with SqliteVectorStore(dimension=4, metric="cosine") as store:
            # Normalized vectors
            store.add(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.707, 0.707, 0.0, 0.0],  # 45 degrees from first
                ],
                ["first", "second"],
            )
            results = store.search([1.0, 0.0, 0.0, 0.0], k=2)
            # First should be exact match (score ~1.0)
            assert results[0].score > results[1].score

    def test_l2_metric(self):
        """Test L2 distance metric."""
        with SqliteVectorStore(dimension=4, metric="l2") as store:
            store.add(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0],
                ],
                ["origin", "close", "far"],
            )
            results = store.search([0.0, 0.0, 0.0, 0.0], k=3)
            assert results[0].text == "origin"

    def test_dot_metric(self):
        """Test dot product metric."""
        with SqliteVectorStore(dimension=4, metric="dot") as store:
            store.add(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [0.5, 0.5, 0.5, 0.5],
                ],
                ["high", "low"],
            )
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


class TestConcurrentAccess:
    """Test VectorStore concurrent access from multiple threads.

    SQLite does not allow sharing a connection across threads by default.
    These tests use a file-based database with separate VectorStore instances
    per thread (the correct usage pattern for concurrent access).
    """

    def test_shared_instance_rejects_cross_thread_use(self):
        """Test that a single VectorStore instance raises on cross-thread use."""
        import sqlite3
        import threading

        with SqliteVectorStore(dimension=4) as store:
            store.add([[1.0, 0.0, 0.0, 0.0]], ["seed"])
            error_holder: list[Exception] = []

            def reader():
                try:
                    store.search([1.0, 0.0, 0.0, 0.0], k=1)
                except sqlite3.ProgrammingError as e:
                    error_holder.append(e)

            t = threading.Thread(target=reader)
            t.start()
            t.join()

            assert len(error_holder) == 1
            assert "thread" in str(error_holder[0]).lower()

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Windows mandatory file locking causes SQLite 'database is locked' under heavy concurrent writes",
    )
    def test_concurrent_writes_separate_instances(self):
        """Test concurrent writes via separate store instances on same file."""
        import threading

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Create the store and seed schema
            with SqliteVectorStore(dimension=4, db_path=db_path) as store:
                store.add([[0.0, 0.0, 0.0, 0.0]], ["seed"])

            errors: list[Exception] = []

            def writer(thread_idx: int):
                try:
                    with SqliteVectorStore.open(db_path) as s:
                        for i in range(10):
                            emb = [float(thread_idx), float(i), 0.0, 0.0]
                            s.add([emb], [f"t{thread_idx}-{i}"])
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert errors == [], f"Concurrent writes failed: {errors}"

            with SqliteVectorStore.open(db_path) as s:
                # 1 seed + 4 threads * 10 docs = 41
                assert len(s) == 41
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_concurrent_reads_separate_instances(self):
        """Test concurrent reads via separate store instances on same file."""
        import threading

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            with SqliteVectorStore(dimension=4, db_path=db_path) as store:
                embeddings = [[float(i), 0.0, 0.0, 0.0] for i in range(20)]
                texts = [f"doc-{i}" for i in range(20)]
                store.add(embeddings, texts)

            errors: list[Exception] = []

            def reader(query: list[float]):
                try:
                    with SqliteVectorStore.open(db_path) as s:
                        for _ in range(20):
                            results = s.search(query, k=5)
                            assert len(results) <= 5
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=reader, args=([float(i), 0.0, 0.0, 0.0],)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert errors == [], f"Concurrent reads failed: {errors}"
        finally:
            Path(db_path).unlink(missing_ok=True)
