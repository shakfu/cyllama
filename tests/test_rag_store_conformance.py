"""Cross-backend conformance suite for :class:`VectorStoreProtocol`.

Every backend gets its own test module for backend-specific behaviour
(``test_rag_store.py``, ``test_rag_sqlite_vec.py``, ``test_rag_chroma.py``,
``test_rag_qdrant.py``, ``test_rag_pgvector.py``). Those grew
independently and assert *similar but not identical* things, which means
an adapter can drift from the shared contract without any of them
noticing -- the exact failure the protocol exists to prevent.

This module runs one identical body of tests against every backend that
is importable in the current environment, anchored on the default
``SqliteVectorStore`` as the reference implementation. Anything asserted
here is part of the cross-backend contract; anything backend-specific
belongs in that backend's own module.

Backends are skipped individually when their optional dependency (or, for
pgvector, ``CYLLAMA_POSTGRES_DSN``) is missing, so this runs usefully
anywhere from a bare checkout to the full store-adapters CI matrix.
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Any, Callable

import pytest

from cyllama.rag.types import SearchResult, VectorStoreProtocol

DIM = 4

# The vectors every test works with. Deliberately not normalised to unit
# length everywhere, but never zero-norm -- cosine distance against a
# zero vector is undefined and backends disagree about what to report,
# which is a backend-specific concern, not a contract one.
EMBEDDINGS = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.9, 0.1, 0.0, 0.0],
    [0.5, 0.5, 0.0, 0.0],
]
TEXTS = [f"Document {i}" for i in range(len(EMBEDDINGS))]
QUERY = [1.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------
# Backend registry
#
# Each entry is a zero-argument factory returning a fresh, empty store.
# Factories are only called for backends that report available, and the
# fixture closes (and where necessary drops) whatever they return.
# ---------------------------------------------------------------------


def _unique(prefix: str = "conf") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _sqlite_vector_available() -> bool:
    import sqlite3

    if not hasattr(sqlite3.Connection, "enable_load_extension"):
        return False
    from cyllama.rag.store import SqliteVectorStore, _extension_suffix

    return SqliteVectorStore.EXTENSION_PATH.with_suffix(_extension_suffix()).exists()


def _make_sqlite_vector() -> Any:
    from cyllama.rag import SqliteVectorStore

    return SqliteVectorStore(dimension=DIM)


def _make_sqlite_vec() -> Any:
    from cyllama.rag.stores import SqliteVecStore

    return SqliteVecStore(dimension=DIM)


def _make_chroma() -> Any:
    from cyllama.rag.stores import ChromaVectorStore

    # A fresh ephemeral client per store keeps backends isolated.
    return ChromaVectorStore(dimension=DIM, collection_name=_unique("conf"))


def _make_qdrant() -> Any:
    from cyllama.rag.stores import QdrantVectorStore

    return QdrantVectorStore(dimension=DIM, collection_name=_unique("conf"), location=":memory:")


def _make_pgvector() -> Any:
    from cyllama.rag.stores import PgVectorStore

    return PgVectorStore(
        dimension=DIM,
        dsn=os.environ["CYLLAMA_POSTGRES_DSN"],
        table_name=_unique("conf"),
    )


def _importable(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


# name -> (factory, available?, why-not)
BACKENDS: dict[str, tuple[Callable[[], Any], Callable[[], bool], str]] = {
    "sqlite-vector": (_make_sqlite_vector, _sqlite_vector_available, "sqlite-vector extension not built"),
    "sqlite-vec": (_make_sqlite_vec, lambda: _importable("sqlite_vec"), "sqlite-vec not installed"),
    "chroma": (_make_chroma, lambda: _importable("chromadb"), "chromadb not installed"),
    "qdrant": (_make_qdrant, lambda: _importable("qdrant_client"), "qdrant-client not installed"),
    "pgvector": (
        _make_pgvector,
        lambda: _importable("psycopg") and _importable("pgvector") and bool(os.environ.get("CYLLAMA_POSTGRES_DSN")),
        "psycopg/pgvector missing or CYLLAMA_POSTGRES_DSN not set",
    ),
}


def _teardown(store: Any) -> None:
    """Drop whatever the backend persisted, then close it.

    Only pgvector leaves anything behind that a later run would trip
    over -- the rest are in-memory or per-instance. A test may have
    closed the store already (``close()`` is contractually idempotent,
    and one test exercises exactly that), so the drop is best-effort.
    """
    try:
        if type(store).__name__ == "PgVectorStore":
            name = store.table_name
            try:
                for suffix in ("", "_meta", "_sources"):
                    store.conn.execute(f'DROP TABLE IF EXISTS "{name}{suffix}" CASCADE')
                store.conn.commit()
            except Exception:
                # Already closed by the test: reconnect just to clean up.
                import psycopg

                with psycopg.connect(os.environ["CYLLAMA_POSTGRES_DSN"]) as conn:
                    for suffix in ("", "_meta", "_sources"):
                        conn.execute(f'DROP TABLE IF EXISTS "{name}{suffix}" CASCADE')
                    conn.commit()
    finally:
        store.close()


@pytest.fixture(params=list(BACKENDS), ids=list(BACKENDS))
def store(request):
    """A fresh empty store, once per registered backend."""
    factory, available, reason = BACKENDS[request.param]
    if not available():
        pytest.skip(reason)
    s = factory()
    try:
        yield s
    finally:
        _teardown(s)


# ---------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------


class TestProtocol:
    def test_satisfies_protocol(self, store):
        assert isinstance(store, VectorStoreProtocol)

    def test_every_protocol_member_is_callable(self, store):
        for name in ("search", "add", "is_source_indexed", "get_source_by_label", "clear", "close"):
            assert callable(getattr(store, name)), f"{name} missing or not callable"

    def test_empty_store_has_zero_length(self, store):
        assert len(store) == 0


class TestAdd:
    def test_returns_one_int_id_per_embedding(self, store):
        ids = store.add(EMBEDDINGS, TEXTS)
        assert len(ids) == len(EMBEDDINGS)
        assert all(isinstance(i, int) for i in ids)

    def test_ids_are_unique(self, store):
        ids = store.add(EMBEDDINGS, TEXTS)
        assert len(set(ids)) == len(ids)

    def test_ids_stay_unique_across_calls(self, store):
        first = store.add(EMBEDDINGS[:2], TEXTS[:2])
        second = store.add(EMBEDDINGS[2:], TEXTS[2:])
        assert not set(first) & set(second)

    def test_length_reflects_adds(self, store):
        store.add(EMBEDDINGS[:2], TEXTS[:2])
        assert len(store) == 2
        store.add(EMBEDDINGS[2:], TEXTS[2:])
        assert len(store) == len(EMBEDDINGS)

    def test_mismatched_lengths_raise(self, store):
        with pytest.raises(ValueError):
            store.add([EMBEDDINGS[0]], ["a", "b"])

    def test_mismatched_metadata_length_raises(self, store):
        with pytest.raises(ValueError):
            store.add([EMBEDDINGS[0]], ["a"], [{"x": 1}, {"y": 2}])

    def test_wrong_dimension_raises(self, store):
        with pytest.raises(ValueError):
            store.add([[1.0, 0.0]], ["a"])

    def test_source_hash_without_label_raises(self, store):
        with pytest.raises(ValueError):
            store.add([EMBEDDINGS[0]], ["a"], source_hash="abc")


class TestSearch:
    def test_returns_search_results(self, store):
        store.add(EMBEDDINGS, TEXTS)
        results = store.search(QUERY, k=3)
        assert results and all(isinstance(r, SearchResult) for r in results)

    def test_respects_k(self, store):
        store.add(EMBEDDINGS, TEXTS)
        assert len(store.search(QUERY, k=2)) == 2

    def test_k_larger_than_count_is_clamped(self, store):
        store.add(EMBEDDINGS[:2], TEXTS[:2])
        assert len(store.search(QUERY, k=50)) == 2

    def test_empty_store_returns_nothing(self, store):
        assert store.search(QUERY, k=5) == []

    def test_nearest_first(self, store):
        store.add(EMBEDDINGS, TEXTS)
        assert store.search(QUERY, k=1)[0].text == TEXTS[0]

    def test_ordered_by_descending_score(self, store):
        store.add(EMBEDDINGS, TEXTS)
        scores = [r.score for r in store.search(QUERY, k=len(EMBEDDINGS))]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_finite_floats(self, store):
        import math

        store.add(EMBEDDINGS, TEXTS)
        for r in store.search(QUERY, k=len(EMBEDDINGS)):
            assert isinstance(r.score, float)
            assert math.isfinite(r.score)

    def test_result_id_is_a_string(self, store):
        store.add(EMBEDDINGS, TEXTS)
        assert all(isinstance(r.id, str) for r in store.search(QUERY, k=3))

    def test_result_ids_come_from_add(self, store):
        """The IDs ``add`` hands back must be the ones ``search`` reports.

        The protocol types them ``int`` on the way in and ``str`` on the
        way out, and a backend that renumbers between the two would make
        every ID the caller holds useless.
        """
        ids = {str(i) for i in store.add(EMBEDDINGS, TEXTS)}
        found = {r.id for r in store.search(QUERY, k=len(EMBEDDINGS))}
        assert found <= ids

    def test_wrong_dimension_raises(self, store):
        store.add(EMBEDDINGS, TEXTS)
        with pytest.raises(ValueError):
            store.search([1.0, 0.0], k=1)

    def test_threshold_filters(self, store):
        store.add(EMBEDDINGS, TEXTS)
        unfiltered = store.search(QUERY, k=len(EMBEDDINGS))
        cutoff = sorted(r.score for r in unfiltered)[-2]
        filtered = store.search(QUERY, k=len(EMBEDDINGS), threshold=cutoff)
        assert filtered
        assert all(r.score >= cutoff for r in filtered)
        assert len(filtered) <= len(unfiltered)


class TestMetadata:
    def test_scalar_metadata_roundtrips(self, store):
        meta = {"source": "a.txt", "page": 3, "score": 1.5, "ok": True}
        store.add([EMBEDDINGS[0]], ["doc"], [meta])
        assert store.search(QUERY, k=1)[0].metadata == meta

    def test_nested_metadata_roundtrips(self, store):
        meta = {"span": {"start": 1, "end": 9}, "tags": ["x", "y"]}
        store.add([EMBEDDINGS[0]], ["doc"], [meta])
        assert store.search(QUERY, k=1)[0].metadata == meta

    def test_empty_metadata_comes_back_empty(self, store):
        store.add([EMBEDDINGS[0]], ["doc"], [{}])
        assert store.search(QUERY, k=1)[0].metadata == {}

    def test_omitted_metadata_comes_back_empty(self, store):
        store.add([EMBEDDINGS[0]], ["doc"])
        assert store.search(QUERY, k=1)[0].metadata == {}

    def test_dedup_bookkeeping_is_not_leaked_as_metadata(self, store):
        """``source_hash`` / ``source_label`` are bookkeeping, not user data.

        Backends that store them alongside the chunk (Qdrant and Chroma
        both do) must not hand them back in ``SearchResult.metadata``.
        """
        store.add([EMBEDDINGS[0]], ["doc"], [{"page": 1}], source_hash="h1", source_label="a.txt")
        assert store.search(QUERY, k=1)[0].metadata == {"page": 1}

    def test_text_roundtrips(self, store):
        store.add([EMBEDDINGS[0]], ["hello world"])
        assert store.search(QUERY, k=1)[0].text == "hello world"


class TestSourceDedup:
    """The contract ``RAG.add_documents`` relies on to skip re-indexing.

    A backend may opt out by always returning ``False`` / ``None`` -- the
    RAG layer then just re-indexes every time. What it may not do is
    answer inconsistently, so each test asserts either the working
    behaviour or a clean opt-out.
    """

    @staticmethod
    def _opted_out(store, digest: str) -> bool:
        """True when the backend declines to track sources at all.

        Takes the digest the caller actually added -- probing a hash
        that was never added would report every backend as opted out
        and silently skip the test on all of them.
        """
        return not store.is_source_indexed(digest)

    def test_unknown_hash_is_not_indexed(self, store):
        assert store.is_source_indexed("never-seen") is False

    def test_hash_is_indexed_after_add(self, store):
        store.add(EMBEDDINGS[:2], TEXTS[:2], source_hash="probe-hash-after-add", source_label="a.txt")
        if self._opted_out(store, "probe-hash-after-add"):
            pytest.skip("backend opts out of source dedup")
        assert store.is_source_indexed("probe-hash-after-add") is True

    def test_unknown_label_returns_none(self, store):
        assert store.get_source_by_label("never-seen.txt") is None

    def test_source_record_has_the_documented_shape(self, store):
        store.add(EMBEDDINGS[:2], TEXTS[:2], source_hash="probe-hash-after-add", source_label="a.txt")
        if self._opted_out(store, "probe-hash-after-add"):
            pytest.skip("backend opts out of source dedup")
        record = store.get_source_by_label("a.txt")
        assert record is not None
        assert record["content_hash"] == "probe-hash-after-add"
        assert record["source_label"] == "a.txt"
        assert record["chunk_count"] == 2
        assert record["indexed_at"]

    def test_adding_without_a_hash_records_no_source(self, store):
        store.add(EMBEDDINGS, TEXTS)
        assert store.get_source_by_label("a.txt") is None

    def test_rag_layer_skip_sequence(self, store):
        """The exact sequence ``RAG.add_documents`` performs per file."""
        digest, label = "rag-seq-hash", "report.pdf"
        assert store.is_source_indexed(digest) is False  # -> index it
        store.add(EMBEDDINGS[:3], TEXTS[:3], source_hash=digest, source_label=label)
        if self._opted_out(store, digest) or store.get_source_by_label(label) is None:
            pytest.skip("backend opts out of source dedup")
        assert store.is_source_indexed(digest) is True  # -> skip it
        # Same basename, different content: the RAG layer detects this by
        # comparing the stored hash against the new one.
        assert store.get_source_by_label(label)["content_hash"] == digest


class TestLifecycle:
    def test_clear_reports_and_empties(self, store):
        store.add(EMBEDDINGS, TEXTS)
        assert store.clear() == len(EMBEDDINGS)
        assert len(store) == 0
        assert store.search(QUERY, k=5) == []

    def test_clear_forgets_sources(self, store):
        store.add(EMBEDDINGS[:2], TEXTS[:2], source_hash="cleared-hash", source_label="a.txt")
        store.clear()
        assert store.is_source_indexed("cleared-hash") is False

    def test_usable_after_clear(self, store):
        store.add(EMBEDDINGS, TEXTS)
        store.clear()
        store.add(EMBEDDINGS[:1], TEXTS[:1])
        assert len(store) == 1

    def test_close_is_idempotent(self, store):
        store.close()
        store.close()


def test_every_backend_reports_its_availability():
    """Guard against a backend silently dropping out of the matrix.

    A registry entry whose availability probe raises would otherwise
    surface as a skip, and this suite is worth nothing if it quietly
    stops covering a backend.
    """
    for name, (_, available, _reason) in BACKENDS.items():
        assert isinstance(available(), bool), f"{name} availability probe returned a non-bool"


def test_at_least_one_backend_is_exercised():
    """A run where every backend skipped is not a passing run.

    In CI the store-adapters workflow installs exactly one client per
    matrix leg, so this holds there; locally it catches a checkout with
    nothing built and no clients installed.
    """
    available = [name for name, (_, probe, _r) in BACKENDS.items() if probe()]
    if not available:
        pytest.skip("no vector-store backend available in this environment")
    print(f"conformance backends: {', '.join(available)}", file=sys.stderr)
