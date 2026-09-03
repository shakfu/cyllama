"""Tests for the sqlite-vec adapter (:class:`SqliteVecStore`).

Skipped when the ``sqlite-vec`` package isn't installed so the suite
stays green on minimal test environments. Uses SQLite's ``:memory:``
mode unless a test needs on-disk persistence.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sqlite_vec", reason="sqlite-vec not installed")

from cyllama.rag import SearchResult, VectorStoreError  # noqa: E402
from cyllama.rag.stores import SqliteVecStore  # noqa: E402
from cyllama.rag.types import VectorStoreProtocol  # noqa: E402


@pytest.fixture
def store():
    s = SqliteVecStore(dimension=4)
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def sample_embeddings():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0, 0.0],
    ]


@pytest.fixture
def sample_texts():
    return [f"Document {i}" for i in range(5)]


class TestInit:
    def test_defaults(self):
        with SqliteVecStore(dimension=8) as s:
            assert s.dimension == 8
            assert s.table_name == "embeddings"
            assert s.metric == "cosine"
            assert s.vector_type == "float32"
            assert len(s) == 0

    def test_invalid_dimension(self):
        with pytest.raises(ValueError, match="dimension must be positive"):
            SqliteVecStore(dimension=0)

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="Invalid metric"):
            SqliteVecStore(dimension=4, metric="bogus")

    def test_dot_metric_unsupported(self):
        # vec0 offers cosine / L2 / L1 only; the default sqlite-vector
        # backend's "dot" has no equivalent.
        with pytest.raises(ValueError, match="Invalid metric"):
            SqliteVecStore(dimension=4, metric="dot")

    def test_invalid_vector_type(self):
        with pytest.raises(ValueError, match="Invalid vector_type"):
            SqliteVecStore(dimension=4, vector_type="float64")

    def test_uint8_unsupported(self):
        with pytest.raises(ValueError, match="Invalid vector_type"):
            SqliteVecStore(dimension=4, vector_type="uint8")

    def test_invalid_table_name(self):
        with pytest.raises(ValueError, match="Invalid table name"):
            SqliteVecStore(dimension=4, table_name="drop table; --")

    def test_all_metrics(self):
        for metric in ["cosine", "l2", "l1", "squared_l2"]:
            with SqliteVecStore(dimension=4, metric=metric) as s:
                assert s.metric == metric

    def test_all_vector_types(self):
        for vtype in ["float32", "int8"]:
            with SqliteVecStore(dimension=4, vector_type=vtype) as s:
                assert s.vector_type == vtype

    def test_missing_extension_path(self, tmp_path):
        with pytest.raises(VectorStoreError, match="Failed to load sqlite-vec extension"):
            SqliteVecStore(dimension=4, extension_path=tmp_path / "nope")


class TestAddSearch:
    def test_add_returns_sequential_ids(self, store, sample_embeddings, sample_texts):
        ids = store.add(sample_embeddings, sample_texts)
        assert ids == [1, 2, 3, 4, 5]
        assert len(store) == 5

    def test_add_appends_ids(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings[:2], sample_texts[:2])
        ids2 = store.add(sample_embeddings[2:], sample_texts[2:])
        assert ids2 == [3, 4, 5]
        assert len(store) == 5

    def test_add_one(self, store):
        id_ = store.add_one([1.0, 0.0, 0.0, 0.0], "solo", {"k": "v"})
        assert store.get(id_).metadata == {"k": "v"}

    def test_add_length_mismatch(self, store):
        with pytest.raises(ValueError, match="same length"):
            store.add([[1.0, 0.0, 0.0, 0.0]], ["a", "b"])

    def test_add_dimension_mismatch(self, store):
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.add([[1.0, 0.0]], ["a"])

    def test_add_metadata_length_mismatch(self, store):
        with pytest.raises(ValueError, match="same length"):
            store.add([[1.0, 0.0, 0.0, 0.0]], ["a"], [{"x": 1}, {"y": 2}])

    def test_add_unserializable_metadata(self, store):
        with pytest.raises(ValueError, match="not JSON-serializable"):
            store.add([[1.0, 0.0, 0.0, 0.0]], ["a"], [{"bad": object()}])

    def test_add_source_hash_requires_label(self, store, sample_embeddings, sample_texts):
        with pytest.raises(ValueError, match="source_hash requires source_label"):
            store.add(sample_embeddings, sample_texts, source_hash="abc")

    def test_search_returns_top_k(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts)
        results = store.search([1.0, 0.0, 0.0, 0.0], k=3)
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].text == "Document 0"

    def test_search_orders_by_score(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts)
        results = store.search([1.0, 0.0, 0.0, 0.0], k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_k_larger_than_count(self, store):
        store.add([[1.0, 0.0, 0.0, 0.0]], ["only"])
        assert len(store.search([1.0, 0.0, 0.0, 0.0], k=50)) == 1

    def test_search_empty_store(self, store):
        assert store.search([1.0, 0.0, 0.0, 0.0], k=5) == []

    def test_search_with_threshold(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts)
        results = store.search([1.0, 0.0, 0.0, 0.0], k=5, threshold=0.9)
        assert all(r.score >= 0.9 for r in results)

    def test_search_metadata_roundtrip(self, store):
        store.add(
            [[1.0, 0.0, 0.0, 0.0]],
            ["doc"],
            [{"source": "a.txt", "page": 3, "tags": ["x", "y"]}],
        )
        result = store.search([1.0, 0.0, 0.0, 0.0], k=1)[0]
        assert result.metadata == {"source": "a.txt", "page": 3, "tags": ["x", "y"]}

    def test_search_dimension_mismatch(self, store):
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.search([1.0, 0.0], k=1)

    def test_search_zero_norm_vector_ranks_last(self, store):
        # vec0 reports NULL cosine distance against a zero-norm vector;
        # the adapter must rank it last rather than crash.
        store.add([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], ["zero", "one"])
        results = store.search([1.0, 0.0, 0.0, 0.0], k=2)
        assert [r.text for r in results] == ["one", "zero"]


class TestMetrics:
    def test_l2_nearest_is_closest(self):
        with SqliteVecStore(dimension=4, metric="l2") as s:
            s.add([[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]], ["origin", "far"])
            assert s.search([0.1, 0.0, 0.0, 0.0], k=2)[0].text == "origin"

    def test_squared_l2_matches_l2_ordering(self, sample_embeddings, sample_texts):
        query = [0.9, 0.1, 0.0, 0.0]
        with SqliteVecStore(dimension=4, metric="l2") as a:
            a.add(sample_embeddings, sample_texts)
            l2_order = [r.text for r in a.search(query, k=5)]
        with SqliteVecStore(dimension=4, metric="squared_l2") as b:
            b.add(sample_embeddings, sample_texts)
            sq_order = [r.text for r in b.search(query, k=5)]
        assert l2_order == sq_order


class TestVectorTypes:
    def test_float32_roundtrip(self):
        with SqliteVecStore(dimension=4, vector_type="float32") as s:
            id_ = s.add_one([1.0, 0.5, 0.25, 0.125], "doc")
            assert s.get_vector(id_) == pytest.approx([1.0, 0.5, 0.25, 0.125])

    def test_float16_unsupported(self):
        # vec0 parses float16 columns but stores them as float32 as of
        # sqlite-vec 0.1.9, so the adapter doesn't offer it.
        with pytest.raises(ValueError, match="Invalid vector_type"):
            SqliteVecStore(dimension=4, vector_type="float16")

    def test_int8_roundtrip(self):
        with SqliteVecStore(dimension=4, vector_type="int8") as s:
            id_ = s.add_one([1.0, -2.0, 3.0, 4.0], "doc")
            assert s.get_vector(id_) == [1.0, -2.0, 3.0, 4.0]

    def test_int8_search(self):
        with SqliteVecStore(dimension=4, vector_type="int8") as s:
            s.add([[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]], ["x", "y"])
            assert s.search([10.0, 0.0, 0.0, 0.0], k=1)[0].text == "x"


class TestGetDelete:
    def test_get_existing(self, store):
        id_ = store.add_one([1.0, 0.0, 0.0, 0.0], "doc", {"k": "v"})
        result = store.get(id_)
        assert result.text == "doc"
        assert result.metadata == {"k": "v"}

    def test_get_missing(self, store):
        assert store.get(999) is None

    def test_get_vector_missing(self, store):
        assert store.get_vector(999) is None

    def test_delete(self, store, sample_embeddings, sample_texts):
        ids = store.add(sample_embeddings, sample_texts)
        assert store.delete([ids[0], ids[1]]) == 2
        assert len(store) == 3
        assert ids[0] not in store

    def test_delete_removes_from_vec_table(self, store, sample_embeddings, sample_texts):
        ids = store.add(sample_embeddings, sample_texts)
        store.delete([ids[0]])
        # The deleted row must not come back from a KNN scan either.
        assert all(r.id != str(ids[0]) for r in store.search([1.0, 0.0, 0.0, 0.0], k=5))

    def test_delete_empty_list(self, store):
        assert store.delete([]) == 0

    def test_contains(self, store):
        id_ = store.add_one([1.0, 0.0, 0.0, 0.0], "doc")
        assert id_ in store
        assert 999 not in store


class TestSourceDedup:
    def test_is_source_indexed_false_before_add(self, store):
        assert store.is_source_indexed("abc123") is False

    def test_is_source_indexed_true_after_add(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts, source_hash="abc123", source_label="doc.txt")
        assert store.is_source_indexed("abc123") is True

    def test_get_source_by_label(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts, source_hash="abc123", source_label="doc.txt")
        source = store.get_source_by_label("doc.txt")
        assert source["content_hash"] == "abc123"
        assert source["source_label"] == "doc.txt"
        assert source["chunk_count"] == 5
        assert source["indexed_at"]

    def test_get_source_by_label_missing(self, store):
        assert store.get_source_by_label("nope.txt") is None

    def test_list_sources(self, store):
        store.add([[1.0, 0.0, 0.0, 0.0]], ["a"], source_hash="h1", source_label="a.txt")
        store.add([[0.0, 1.0, 0.0, 0.0]], ["b"], source_hash="h2", source_label="b.txt")
        labels = [s["source_label"] for s in store.list_sources()]
        assert sorted(labels) == ["a.txt", "b.txt"]

    def test_duplicate_source_rolls_back_chunks(self, store):
        store.add([[1.0, 0.0, 0.0, 0.0]], ["a"], source_hash="h1", source_label="a.txt")
        import sqlite3

        with pytest.raises(sqlite3.IntegrityError):
            store.add([[0.0, 1.0, 0.0, 0.0]], ["b"], source_hash="h1", source_label="a.txt")
        # The chunk insert must roll back with the failed source insert.
        assert len(store) == 1


class TestPersistence:
    def test_roundtrip_on_disk(self, tmp_path):
        db_path = str(tmp_path / "vectors.db")
        with SqliteVecStore(dimension=4, db_path=db_path) as s:
            s.add([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], ["a", "b"])
        with SqliteVecStore(dimension=4, db_path=db_path) as s:
            assert len(s) == 2
            assert s.search([1.0, 0.0, 0.0, 0.0], k=1)[0].text == "a"

    def test_dedup_survives_reopen(self, tmp_path):
        db_path = str(tmp_path / "vectors.db")
        with SqliteVecStore(dimension=4, db_path=db_path) as s:
            s.add([[1.0, 0.0, 0.0, 0.0]], ["a"], source_hash="h1", source_label="a.txt")
        with SqliteVecStore(dimension=4, db_path=db_path) as s:
            assert s.is_source_indexed("h1") is True

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"dimension": 8}, "dimension"),
            ({"dimension": 4, "metric": "l2"}, "metric"),
            ({"dimension": 4, "vector_type": "int8"}, "vector_type"),
        ],
    )
    def test_incompatible_reopen_raises(self, tmp_path, kwargs, match):
        db_path = str(tmp_path / "vectors.db")
        with SqliteVecStore(dimension=4, db_path=db_path) as s:
            s.add([[1.0, 0.0, 0.0, 0.0]], ["a"])
        with pytest.raises(VectorStoreError, match=match):
            SqliteVecStore(db_path=db_path, **kwargs)


class TestLifecycle:
    def test_clear(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts, source_hash="h1", source_label="a.txt")
        assert store.clear() == 5
        assert len(store) == 0
        assert store.search([1.0, 0.0, 0.0, 0.0], k=5) == []
        assert store.is_source_indexed("h1") is False

    def test_close_idempotent(self):
        s = SqliteVecStore(dimension=4)
        s.close()
        s.close()

    def test_use_after_close_raises(self):
        s = SqliteVecStore(dimension=4)
        s.close()
        with pytest.raises(VectorStoreError, match="closed"):
            len(s)

    def test_context_manager(self, sample_embeddings, sample_texts):
        with SqliteVecStore(dimension=4) as s:
            s.add(sample_embeddings, sample_texts)
            assert len(s) == 5
        with pytest.raises(VectorStoreError, match="closed"):
            len(s)

    def test_repr(self, store):
        assert "SqliteVecStore(" in repr(store)
        assert "dimension=4" in repr(store)


class TestProtocolConformance:
    def test_is_instance_of_protocol(self, store):
        assert isinstance(store, VectorStoreProtocol)

    def test_implements_every_protocol_method(self, store):
        for name in ("search", "add", "is_source_indexed", "get_source_by_label", "clear", "close"):
            assert callable(getattr(store, name))
        assert len(store) == 0


class TestFtsCompatibility:
    """The plain base table must still accept triggers.

    This is the reason the adapter keeps chunks in a regular table with
    a ``vec0`` sidecar rather than putting them inside the virtual
    table: SQLite forbids triggers on virtual tables, which would break
    the FTS5 sync that :class:`~cyllama.rag.advanced.HybridStore` uses.
    """

    def test_fts5_trigger_on_base_table(self, store):
        store.conn.execute(
            "CREATE VIRTUAL TABLE embeddings_fts USING fts5(text, content='embeddings', content_rowid='id')"
        )
        store.conn.execute(
            "CREATE TRIGGER embeddings_ai AFTER INSERT ON embeddings BEGIN "
            "INSERT INTO embeddings_fts(rowid, text) VALUES (new.id, new.text); END"
        )
        store.add([[1.0, 0.0, 0.0, 0.0]], ["hello world"])
        rows = store.conn.execute("SELECT rowid FROM embeddings_fts WHERE embeddings_fts MATCH 'world'").fetchall()
        assert len(rows) == 1
