"""Tests for the Chroma adapter (:class:`ChromaVectorStore`).

Skipped when ``chromadb`` isn't installed so the suite stays green on
minimal test environments. Everything here uses Chroma's in-process
ephemeral client -- no server required -- except :class:`TestRealServer`,
which is opt-in via ``CYLLAMA_CHROMA_HOST``.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("chromadb", reason="chromadb not installed")

from cyllama.rag import SearchResult  # noqa: E402
from cyllama.rag.stores import ChromaVectorStore  # noqa: E402
from cyllama.rag.types import VectorStoreProtocol  # noqa: E402


@pytest.fixture
def store():
    s = ChromaVectorStore(dimension=4)
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
        with ChromaVectorStore(dimension=8) as s:
            assert s.dimension == 8
            assert s.collection_name == "embeddings"
            assert s.metric == "cosine"
            assert len(s) == 0

    def test_invalid_dimension(self):
        with pytest.raises(ValueError, match="dimension must be positive"):
            ChromaVectorStore(dimension=0)

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="Invalid metric"):
            ChromaVectorStore(dimension=4, metric="bogus")

    def test_all_metrics(self):
        for metric in ["cosine", "l2", "dot"]:
            with ChromaVectorStore(dimension=4, metric=metric, collection_name=f"m-{metric}") as s:
                assert s.metric == metric

    def test_invalid_collection_name(self):
        # Chroma requires 3-512 chars of [a-zA-Z0-9._-]; catch it here
        # rather than letting it surface from the Rust bindings.
        with pytest.raises(ValueError, match="Invalid collection_name"):
            ChromaVectorStore(dimension=4, collection_name="ab")

    def test_conflicting_transport(self, tmp_path):
        with pytest.raises(ValueError, match="only one of"):
            ChromaVectorStore(dimension=4, path=str(tmp_path), host="localhost")

    def test_port_without_host(self):
        with pytest.raises(ValueError, match="port requires host"):
            ChromaVectorStore(dimension=4, port=8000)


class TestAddSearch:
    def test_add_returns_sequential_ids(self, store, sample_embeddings, sample_texts):
        ids = store.add(sample_embeddings, sample_texts)
        assert ids == [0, 1, 2, 3, 4]
        assert len(store) == 5

    def test_add_appends_ids(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings[:2], sample_texts[:2])
        ids2 = store.add(sample_embeddings[2:], sample_texts[2:])
        assert ids2 == [2, 3, 4]
        assert len(store) == 5

    def test_add_empty(self, store):
        assert store.add([], []) == []

    def test_add_length_mismatch(self, store):
        with pytest.raises(ValueError, match="same length"):
            store.add([[1.0, 0.0, 0.0, 0.0]], ["a", "b"])

    def test_add_dimension_mismatch(self, store):
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.add([[1.0, 0.0]], ["a"])

    def test_add_metadata_length_mismatch(self, store):
        with pytest.raises(ValueError, match="same length"):
            store.add([[1.0, 0.0, 0.0, 0.0]], ["a"], [{"x": 1}, {"y": 2}])

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
        scores = [r.score for r in store.search([1.0, 0.0, 0.0, 0.0], k=5)]
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

    def test_search_dimension_mismatch(self, store):
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.search([1.0, 0.0], k=1)

    def test_search_scalar_metadata_roundtrip(self, store):
        store.add([[1.0, 0.0, 0.0, 0.0]], ["doc"], [{"source": "a.txt", "page": 3, "ok": True}])
        assert store.search([1.0, 0.0, 0.0, 0.0], k=1)[0].metadata == {
            "source": "a.txt",
            "page": 3,
            "ok": True,
        }

    def test_search_nested_metadata_roundtrip(self, store):
        # Chroma only stores scalars; the adapter JSON-encodes anything
        # else so arbitrary metadata still round-trips.
        nested = {"span": {"start": 1, "end": 9}, "tags": ["x", "y"]}
        store.add([[1.0, 0.0, 0.0, 0.0]], ["doc"], [nested])
        assert store.search([1.0, 0.0, 0.0, 0.0], k=1)[0].metadata == nested

    def test_search_empty_metadata(self, store):
        # Chroma rejects an empty metadata dict outright; the adapter
        # must send a null entry instead.
        store.add([[1.0, 0.0, 0.0, 0.0]], ["doc"], [{}])
        assert store.search([1.0, 0.0, 0.0, 0.0], k=1)[0].metadata == {}

    def test_search_hides_reserved_keys(self, store):
        store.add([[1.0, 0.0, 0.0, 0.0]], ["doc"], source_hash="h1", source_label="a.txt")
        assert store.search([1.0, 0.0, 0.0, 0.0], k=1)[0].metadata == {}


class TestMetrics:
    def test_l2_nearest_is_closest(self):
        with ChromaVectorStore(dimension=4, metric="l2", collection_name="l2-test") as s:
            s.add([[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]], ["origin", "far"])
            assert s.search([0.1, 0.0, 0.0, 0.0], k=2)[0].text == "origin"

    def test_dot_prefers_larger_magnitude(self):
        with ChromaVectorStore(dimension=4, metric="dot", collection_name="dot-test") as s:
            s.add([[1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]], ["high", "low"])
            assert s.search([1.0, 1.0, 1.0, 1.0], k=2)[0].text == "high"


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

    def test_dedup_alongside_user_metadata(self, store):
        store.add(
            [[1.0, 0.0, 0.0, 0.0]],
            ["doc"],
            [{"page": 1}],
            source_hash="h1",
            source_label="a.txt",
        )
        assert store.is_source_indexed("h1") is True
        assert store.search([1.0, 0.0, 0.0, 0.0], k=1)[0].metadata == {"page": 1}


class TestPersistence:
    def test_roundtrip_on_disk(self, tmp_path):
        path = str(tmp_path / "chroma")
        with ChromaVectorStore(dimension=4, path=path) as s:
            s.add([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], ["a", "b"])
        with ChromaVectorStore(dimension=4, path=path) as s:
            assert len(s) == 2
            assert s.search([1.0, 0.0, 0.0, 0.0], k=1)[0].text == "a"

    def test_ids_continue_after_reopen(self, tmp_path):
        path = str(tmp_path / "chroma")
        with ChromaVectorStore(dimension=4, path=path) as s:
            s.add([[1.0, 0.0, 0.0, 0.0]], ["a"])
        with ChromaVectorStore(dimension=4, path=path) as s:
            assert s.add([[0.0, 1.0, 0.0, 0.0]], ["b"]) == [1]


class TestLifecycle:
    def test_clear(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts, source_hash="h1", source_label="a.txt")
        assert store.clear() == 5
        assert len(store) == 0
        assert store.search([1.0, 0.0, 0.0, 0.0], k=5) == []
        assert store.is_source_indexed("h1") is False

    def test_ids_restart_after_clear(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts)
        store.clear()
        assert store.add([[1.0, 0.0, 0.0, 0.0]], ["fresh"]) == [0]

    def test_close_idempotent(self):
        s = ChromaVectorStore(dimension=4)
        s.close()
        s.close()

    def test_use_after_close_raises(self):
        s = ChromaVectorStore(dimension=4)
        s.close()
        with pytest.raises(RuntimeError, match="closed"):
            len(s)

    def test_context_manager(self, sample_embeddings, sample_texts):
        with ChromaVectorStore(dimension=4) as s:
            s.add(sample_embeddings, sample_texts)
            assert len(s) == 5
        with pytest.raises(RuntimeError, match="closed"):
            len(s)

    def test_caller_owned_client_not_closed(self):
        import chromadb

        client = chromadb.EphemeralClient()
        s = ChromaVectorStore(dimension=4, client=client, collection_name="owned-client")
        s.add([[1.0, 0.0, 0.0, 0.0]], ["a"])
        s.close()
        # The caller still owns the client, so it must remain usable.
        assert client.get_collection("owned-client").count() == 1

    def test_repr(self, store):
        assert "ChromaVectorStore(" in repr(store)
        assert "dimension=4" in repr(store)


class TestProtocolConformance:
    def test_is_instance_of_protocol(self, store):
        assert isinstance(store, VectorStoreProtocol)

    def test_implements_every_protocol_method(self, store):
        for name in ("search", "add", "is_source_indexed", "get_source_by_label", "clear", "close"):
            assert callable(getattr(store, name))
        assert len(store) == 0


_CHROMA_HOST = os.environ.get("CYLLAMA_CHROMA_HOST")
_CHROMA_PORT = os.environ.get("CYLLAMA_CHROMA_PORT")


@pytest.mark.integration
@pytest.mark.skipif(not _CHROMA_HOST, reason="CYLLAMA_CHROMA_HOST not set")
class TestRealServer:
    """Exercise the HttpClient transport against a live Chroma server.

    Opt in by pointing ``CYLLAMA_CHROMA_HOST`` (and optionally
    ``CYLLAMA_CHROMA_PORT``) at one; the store-adapters workflow runs
    these against a ``chromadb/chroma`` service container.
    """

    @pytest.fixture
    def server_store(self):
        collection = f"cyllama-test-{uuid.uuid4().hex[:12]}"
        kwargs = {"host": _CHROMA_HOST}
        if _CHROMA_PORT:
            kwargs["port"] = int(_CHROMA_PORT)
        s = ChromaVectorStore(dimension=4, collection_name=collection, **kwargs)
        try:
            yield s
        finally:
            try:
                s.client.delete_collection(name=collection)
            finally:
                s.close()

    def test_add_search_roundtrip(self, server_store):
        ids = server_store.add(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            ["first", "second"],
            metadata=[{"tag": "a"}, {"tag": "b"}],
        )
        assert ids == [0, 1]
        assert len(server_store) == 2
        hits = server_store.search([1.0, 0.05, 0.0, 0.0], k=1)
        assert len(hits) == 1
        assert hits[0].text == "first"
        assert hits[0].metadata == {"tag": "a"}

    def test_nested_metadata_roundtrip_on_real_server(self, server_store):
        nested = {"span": {"start": 1, "end": 9}, "tags": ["x", "y"]}
        server_store.add([[1.0, 0.0, 0.0, 0.0]], ["doc"], [nested])
        assert server_store.search([1.0, 0.0, 0.0, 0.0], k=1)[0].metadata == nested

    def test_source_dedup_roundtrip(self, server_store):
        server_store.add(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            ["a", "b"],
            source_hash="real-hash",
            source_label="real.txt",
        )
        assert server_store.is_source_indexed("real-hash") is True
        assert server_store.is_source_indexed("missing") is False
        record = server_store.get_source_by_label("real.txt")
        assert record is not None
        assert record["content_hash"] == "real-hash"
        assert record["chunk_count"] == 2

    def test_clear_on_real_server(self, server_store):
        server_store.add([[1.0, 0.0, 0.0, 0.0]], ["x"])
        assert len(server_store) == 1
        assert server_store.clear() == 1
        assert len(server_store) == 0
