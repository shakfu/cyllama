"""Tests for the pgvector adapter (:class:`PgVectorStore`).

pgvector is a PostgreSQL extension with no in-process mode, so unlike
the other store adapters these tests need a reachable server. Point
``CYLLAMA_POSTGRES_DSN`` at one and they run; otherwise everything past
:class:`TestArgumentValidation` skips.

Two easy ways to get one:

* ``pip install pgserver`` (Python <= 3.12) -- ships a PostgreSQL binary
  *and* pgvector, needs no root and no daemon::

      import pgserver
      db = pgserver.get_server("/tmp/pgdata")
      os.environ["CYLLAMA_POSTGRES_DSN"] = db.get_uri()

* ``docker run -p 5432:5432 -e POSTGRES_PASSWORD=pw pgvector/pgvector:pg17``

The store-adapters workflow uses the container. Each test class works in
its own table so a shared database stays usable.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("psycopg", reason="psycopg not installed")
pytest.importorskip("pgvector", reason="pgvector not installed")

from cyllama.rag import SearchResult, VectorStoreError  # noqa: E402
from cyllama.rag.stores import PgVectorStore  # noqa: E402
from cyllama.rag.stores.postgres import _parse_version  # noqa: E402
from cyllama.rag.types import VectorStoreProtocol  # noqa: E402

_DSN = os.environ.get("CYLLAMA_POSTGRES_DSN")

requires_server = pytest.mark.skipif(not _DSN, reason="CYLLAMA_POSTGRES_DSN not set")


def _table() -> str:
    """A table name unique to one test, so tests can share a database."""
    return f"t_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def store():
    """A store on its own table, dropped afterwards."""
    name = _table()
    s = PgVectorStore(dimension=4, dsn=_DSN, table_name=name)
    try:
        yield s
    finally:
        _drop(s, name)


def _drop(s: PgVectorStore, name: str) -> None:
    try:
        for suffix in ("", "_meta", "_sources"):
            s.conn.execute(f'DROP TABLE IF EXISTS "{name}{suffix}" CASCADE')
        s.conn.commit()
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


class TestArgumentValidation:
    """Argument checks that fire before any connection is attempted."""

    def test_invalid_dimension(self):
        with pytest.raises(ValueError, match="dimension must be positive"):
            PgVectorStore(dimension=0, dsn="postgresql://unused")

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="Invalid metric"):
            PgVectorStore(dimension=4, dsn="postgresql://unused", metric="bogus")

    def test_invalid_table_name(self):
        with pytest.raises(ValueError, match="Invalid table name"):
            PgVectorStore(dimension=4, dsn="postgresql://unused", table_name="drop table; --")

    def test_requires_dsn_or_conn(self):
        with pytest.raises(ValueError, match="exactly one of"):
            PgVectorStore(dimension=4)

    def test_rejects_both_dsn_and_conn(self):
        with pytest.raises(ValueError, match="exactly one of"):
            PgVectorStore(dimension=4, dsn="postgresql://unused", conn=object())

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("0.8.1", (0, 8, 1)), ("0.6.2", (0, 6, 2)), ("0.7.0-rc1", (0, 7, 0)), ("1.0", (1, 0))],
    )
    def test_parse_version(self, text, expected):
        assert _parse_version(text) == expected


@requires_server
class TestInit:
    def test_defaults(self):
        name = _table()
        s = PgVectorStore(dimension=8, dsn=_DSN, table_name=name)
        try:
            assert s.dimension == 8
            assert s.metric == "cosine"
            assert s.table_name == name
            assert len(s) == 0
            assert s.pgvector_version != "unknown"
        finally:
            _drop(s, name)

    def test_caller_owned_connection_not_closed(self):
        import psycopg

        conn = psycopg.connect(_DSN)
        name = _table()
        s = PgVectorStore(dimension=4, conn=conn, table_name=name)
        s.add([[1.0, 0.0, 0.0, 0.0]], ["a"])
        s.close()
        # The caller still owns the connection, so it must stay usable.
        assert conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0] == 1
        conn.execute(f'DROP TABLE "{name}", "{name}_meta", "{name}_sources" CASCADE')
        conn.commit()
        conn.close()


@requires_server
class TestAddSearch:
    def test_add_returns_sequential_ids(self, store, sample_embeddings, sample_texts):
        assert store.add(sample_embeddings, sample_texts) == [1, 2, 3, 4, 5]
        assert len(store) == 5

    def test_add_appends_ids(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings[:2], sample_texts[:2])
        assert store.add(sample_embeddings[2:], sample_texts[2:]) == [3, 4, 5]

    def test_add_empty(self, store):
        assert store.add([], []) == []

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

    def test_add_source_hash_requires_label(self, store):
        with pytest.raises(ValueError, match="source_hash requires source_label"):
            store.add([[1.0, 0.0, 0.0, 0.0]], ["a"], source_hash="abc")

    def test_search_returns_top_k(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts)
        results = store.search([1.0, 0.0, 0.0, 0.0], k=3)
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].text == "Document 0"

    def test_search_orders_by_score(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts)
        scores = [r.score for r in store.search([1.0, 0.0, 0.0, 0.0], k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_search_k_larger_than_count(self, store):
        store.add([[1.0, 0.0, 0.0, 0.0]], ["only"])
        assert len(store.search([1.0, 0.0, 0.0, 0.0], k=50)) == 1

    def test_search_empty_store(self, store):
        assert store.search([1.0, 0.0, 0.0, 0.0], k=5) == []

    def test_search_with_threshold(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts)
        assert all(r.score >= 0.9 for r in store.search([1.0, 0.0, 0.0, 0.0], k=5, threshold=0.9))

    def test_search_dimension_mismatch(self, store):
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.search([1.0, 0.0], k=1)

    def test_nested_metadata_roundtrip(self, store):
        # JSONB is native, so nested metadata needs no encoding tricks.
        nested = {"span": {"start": 1, "end": 9}, "tags": ["x", "y"], "ok": True}
        store.add([[1.0, 0.0, 0.0, 0.0]], ["doc"], [nested])
        assert store.search([1.0, 0.0, 0.0, 0.0], k=1)[0].metadata == nested

    def test_empty_metadata(self, store):
        store.add([[1.0, 0.0, 0.0, 0.0]], ["doc"], [{}])
        assert store.search([1.0, 0.0, 0.0, 0.0], k=1)[0].metadata == {}


@requires_server
class TestMetrics:
    def _order(self, metric, embeddings, texts, query):
        name = _table()
        s = PgVectorStore(dimension=4, dsn=_DSN, table_name=name, metric=metric)
        try:
            s.add(embeddings, texts)
            return [r.text for r in s.search(query, k=len(texts))]
        finally:
            _drop(s, name)

    def test_l2_nearest_is_closest(self, sample_embeddings, sample_texts):
        order = self._order("l2", [[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]], ["origin", "far"], [0.1, 0, 0, 0])
        assert order[0] == "origin"

    def test_squared_l2_matches_l2_ordering(self, sample_embeddings, sample_texts):
        query = [0.9, 0.1, 0.0, 0.0]
        assert self._order("l2", sample_embeddings, sample_texts, query) == self._order(
            "squared_l2", sample_embeddings, sample_texts, query
        )

    def test_dot_prefers_larger_magnitude(self):
        # The metric sqlite-vec cannot offer at all.
        order = self._order("dot", [[1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]], ["high", "low"], [1.0, 1.0, 1.0, 1.0])
        assert order[0] == "high"

    def test_dot_score_is_the_inner_product(self):
        name = _table()
        s = PgVectorStore(dimension=4, dsn=_DSN, table_name=name, metric="dot")
        try:
            s.add([[1.0, 2.0, 3.0, 4.0]], ["v"])
            # <#> returns the negative inner product; the adapter negates
            # it back, so the score is the plain dot product.
            assert s.search([1.0, 1.0, 1.0, 1.0], k=1)[0].score == pytest.approx(10.0)
        finally:
            _drop(s, name)

    def test_l1_requires_pgvector_070(self):
        """``<+>`` landed in pgvector 0.7.0.

        On an older server the store must refuse at construction with an
        actionable message rather than dying at query time with
        ``operator does not exist: vector <+> vector``.
        """
        name = _table()
        probe = PgVectorStore(dimension=4, dsn=_DSN, table_name=name)
        version = _parse_version(probe.pgvector_version)
        _drop(probe, name)

        name = _table()
        if version and version < (0, 7, 0):
            with pytest.raises(VectorStoreError, match="pgvector 0.7.0"):
                PgVectorStore(dimension=4, dsn=_DSN, table_name=name, metric="l1")
        else:
            s = PgVectorStore(dimension=4, dsn=_DSN, table_name=name, metric="l1")
            try:
                s.add([[0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 0.0, 0.0]], ["near", "far"])
                assert s.search([0.1, 0.0, 0.0, 0.0], k=2)[0].text == "near"
            finally:
                _drop(s, name)


@requires_server
class TestGetDelete:
    def test_get_existing(self, store):
        id_ = store.add_one([1.0, 0.0, 0.0, 0.0], "doc", {"k": "v"})
        result = store.get(id_)
        assert result.text == "doc"
        assert result.metadata == {"k": "v"}

    def test_get_missing(self, store):
        assert store.get(999999) is None

    def test_get_vector(self, store):
        id_ = store.add_one([1.0, 0.5, 0.25, 0.125], "doc")
        assert store.get_vector(id_) == pytest.approx([1.0, 0.5, 0.25, 0.125])

    def test_get_vector_missing(self, store):
        assert store.get_vector(999999) is None

    def test_delete(self, store, sample_embeddings, sample_texts):
        ids = store.add(sample_embeddings, sample_texts)
        assert store.delete([ids[0], ids[1]]) == 2
        assert len(store) == 3
        assert ids[0] not in store

    def test_delete_removes_from_search(self, store, sample_embeddings, sample_texts):
        ids = store.add(sample_embeddings, sample_texts)
        store.delete([ids[0]])
        assert all(r.id != str(ids[0]) for r in store.search([1.0, 0.0, 0.0, 0.0], k=5))

    def test_delete_empty_list(self, store):
        assert store.delete([]) == 0

    def test_contains(self, store):
        id_ = store.add_one([1.0, 0.0, 0.0, 0.0], "doc")
        assert id_ in store
        assert 999999 not in store


@requires_server
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
        assert sorted(s["source_label"] for s in store.list_sources()) == ["a.txt", "b.txt"]

    def test_duplicate_source_rolls_back_chunks(self, store):
        store.add([[1.0, 0.0, 0.0, 0.0]], ["a"], source_hash="h1", source_label="a.txt")
        import psycopg

        with pytest.raises(psycopg.errors.UniqueViolation):
            store.add([[0.0, 1.0, 0.0, 0.0]], ["b"], source_hash="h1", source_label="a.txt")
        # The chunk insert must roll back with the failed source insert.
        assert len(store) == 1


@requires_server
class TestPersistence:
    def test_reopen_sees_existing_rows(self):
        name = _table()
        s = PgVectorStore(dimension=4, dsn=_DSN, table_name=name)
        s.add([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], ["a", "b"])
        s.close()

        s2 = PgVectorStore(dimension=4, dsn=_DSN, table_name=name)
        try:
            assert len(s2) == 2
            assert s2.search([1.0, 0.0, 0.0, 0.0], k=1)[0].text == "a"
        finally:
            _drop(s2, name)

    def test_dedup_survives_reopen(self):
        name = _table()
        s = PgVectorStore(dimension=4, dsn=_DSN, table_name=name)
        s.add([[1.0, 0.0, 0.0, 0.0]], ["a"], source_hash="h1", source_label="a.txt")
        s.close()
        s2 = PgVectorStore(dimension=4, dsn=_DSN, table_name=name)
        try:
            assert s2.is_source_indexed("h1") is True
        finally:
            _drop(s2, name)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"dimension": 8}, "dimension"), ({"dimension": 4, "metric": "l2"}, "metric")],
    )
    def test_incompatible_reopen_raises(self, kwargs, match):
        name = _table()
        s = PgVectorStore(dimension=4, dsn=_DSN, table_name=name)
        s.add([[1.0, 0.0, 0.0, 0.0]], ["a"])
        s.close()
        s2 = None
        try:
            with pytest.raises(VectorStoreError, match=match):
                s2 = PgVectorStore(dsn=_DSN, table_name=name, **kwargs)
        finally:
            cleanup = s2 or PgVectorStore(dimension=4, dsn=_DSN, table_name=name)
            _drop(cleanup, name)


@requires_server
class TestIndexing:
    def test_create_and_drop_hnsw(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts)
        store.create_index("hnsw", m=16, ef_construction=64)
        # The index must serve the same results the scan did.
        assert store.search([1.0, 0.0, 0.0, 0.0], k=1)[0].text == "Document 0"
        store.drop_index("hnsw")
        assert store.search([1.0, 0.0, 0.0, 0.0], k=1)[0].text == "Document 0"

    def test_create_ivfflat(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts)
        store.create_index("ivfflat", lists=1)
        assert store.search([1.0, 0.0, 0.0, 0.0], k=1)[0].text == "Document 0"
        store.drop_index("ivfflat")

    def test_create_index_is_idempotent(self, store):
        store.add([[1.0, 0.0, 0.0, 0.0]], ["a"])
        store.create_index("hnsw")
        store.create_index("hnsw")
        store.drop_index("hnsw")

    def test_invalid_method(self, store):
        with pytest.raises(ValueError, match="Invalid index method"):
            store.create_index("bogus")


@requires_server
class TestLifecycle:
    def test_clear(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts, source_hash="h1", source_label="a.txt")
        assert store.clear() == 5
        assert len(store) == 0
        assert store.search([1.0, 0.0, 0.0, 0.0], k=5) == []
        assert store.is_source_indexed("h1") is False

    def test_close_idempotent(self):
        name = _table()
        s = PgVectorStore(dimension=4, dsn=_DSN, table_name=name)
        _drop(s, name)
        s.close()

    def test_use_after_close_raises(self):
        name = _table()
        s = PgVectorStore(dimension=4, dsn=_DSN, table_name=name)
        _drop(s, name)
        with pytest.raises(VectorStoreError, match="closed"):
            len(s)

    def test_context_manager(self):
        name = _table()
        with PgVectorStore(dimension=4, dsn=_DSN, table_name=name) as s:
            s.add([[1.0, 0.0, 0.0, 0.0]], ["a"])
            assert len(s) == 1
        with pytest.raises(VectorStoreError, match="closed"):
            len(s)
        cleanup = PgVectorStore(dimension=4, dsn=_DSN, table_name=name)
        _drop(cleanup, name)

    def test_repr(self, store):
        assert "PgVectorStore(" in repr(store)
        assert "dimension=4" in repr(store)


@requires_server
class TestProtocolConformance:
    def test_is_instance_of_protocol(self, store):
        assert isinstance(store, VectorStoreProtocol)

    def test_implements_every_protocol_method(self, store):
        for name in ("search", "add", "is_source_indexed", "get_source_by_label", "clear", "close"):
            assert callable(getattr(store, name))
        assert len(store) == 0
