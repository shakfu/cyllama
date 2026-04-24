"""Tests for the Qdrant adapter (:class:`QdrantVectorStore`).

Skipped when ``qdrant-client`` isn't installed so the suite stays green
on minimal test environments. Uses Qdrant's in-process ``:memory:``
mode -- no server required.
"""

from __future__ import annotations

import pytest

qdrant_client = pytest.importorskip(
    "qdrant_client", reason="qdrant-client not installed (pip install cyllama[qdrant])"
)

from cyllama.rag import SearchResult  # noqa: E402
from cyllama.rag.stores import QdrantVectorStore  # noqa: E402


@pytest.fixture
def store():
    s = QdrantVectorStore(dimension=4, location=":memory:")
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
        s = QdrantVectorStore(dimension=8)
        try:
            assert s.dimension == 8
            assert s.collection_name == "embeddings"
            assert s.metric == "cosine"
            assert len(s) == 0
        finally:
            s.close()

    def test_invalid_dimension(self):
        with pytest.raises(ValueError, match="dimension must be positive"):
            QdrantVectorStore(dimension=0)

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="Invalid metric"):
            QdrantVectorStore(dimension=4, metric="bogus")

    def test_conflicting_transport(self):
        with pytest.raises(ValueError, match="only one of"):
            QdrantVectorStore(dimension=4, location=":memory:", path="/tmp/x")


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

    def test_add_length_mismatch(self, store):
        with pytest.raises(ValueError, match="same length"):
            store.add([[1.0, 0.0, 0.0, 0.0]], ["a", "b"])

    def test_add_dimension_mismatch(self, store):
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.add([[1.0, 2.0]], ["short"])

    def test_add_source_hash_requires_label(self, store, sample_embeddings, sample_texts):
        with pytest.raises(ValueError, match="source_label"):
            store.add(
                sample_embeddings[:1],
                sample_texts[:1],
                source_hash="abc123",
            )

    def test_search_returns_top_k(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts)
        results = store.search([1.0, 0.0, 0.0, 0.0], k=3)
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        # Nearest match should be the first vector
        assert results[0].text == "Document 0"

    def test_search_metadata_roundtrip(self, store):
        store.add(
            [[1.0, 0.0, 0.0, 0.0]],
            ["hello"],
            metadata=[{"source": "doc1", "page": 3}],
        )
        results = store.search([1.0, 0.0, 0.0, 0.0], k=1)
        assert results[0].metadata == {"source": "doc1", "page": 3}
        # Reserved payload keys should not leak into user metadata
        assert "text" not in results[0].metadata

    def test_search_dimension_mismatch(self, store):
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.search([1.0, 0.0], k=1)


class TestSourceDedup:
    def test_is_source_indexed_false_before_add(self, store):
        assert store.is_source_indexed("abc") is False

    def test_is_source_indexed_true_after_add(self, store, sample_embeddings, sample_texts):
        store.add(
            sample_embeddings[:2],
            sample_texts[:2],
            source_hash="hash-1",
            source_label="doc1.txt",
        )
        assert store.is_source_indexed("hash-1") is True
        assert store.is_source_indexed("hash-other") is False

    def test_get_source_by_label(self, store, sample_embeddings, sample_texts):
        store.add(
            sample_embeddings[:3],
            sample_texts[:3],
            source_hash="hash-2",
            source_label="doc2.txt",
        )
        record = store.get_source_by_label("doc2.txt")
        assert record is not None
        assert record["content_hash"] == "hash-2"
        assert record["source_label"] == "doc2.txt"
        assert record["chunk_count"] == 3
        assert record["indexed_at"] is not None

    def test_get_source_by_label_missing(self, store):
        assert store.get_source_by_label("not-there.txt") is None


class TestLifecycle:
    def test_clear(self, store, sample_embeddings, sample_texts):
        store.add(sample_embeddings, sample_texts)
        assert len(store) == 5
        removed = store.clear()
        assert removed == 5
        assert len(store) == 0
        # After clear, IDs restart at 0
        ids = store.add(sample_embeddings[:1], sample_texts[:1])
        assert ids == [0]

    def test_close_idempotent(self):
        s = QdrantVectorStore(dimension=4)
        s.close()
        s.close()  # second call is a no-op

    def test_use_after_close_raises(self):
        s = QdrantVectorStore(dimension=4)
        s.close()
        with pytest.raises(RuntimeError, match="closed"):
            len(s)

    def test_context_manager(self, sample_embeddings, sample_texts):
        with QdrantVectorStore(dimension=4) as s:
            s.add(sample_embeddings[:1], sample_texts[:1])
            assert len(s) == 1
        # After __exit__ the store is closed
        with pytest.raises(RuntimeError):
            len(s)


class TestProtocolConformance:
    def test_is_instance_of_protocol(self, store):
        from cyllama.rag import VectorStoreProtocol

        assert isinstance(store, VectorStoreProtocol)


# ---------------------------------------------------------------------------
# Real-server integration tests.
#
# Enabled by setting CYLLAMA_QDRANT_URL to a reachable Qdrant HTTP endpoint
# (e.g. "http://127.0.0.1:6533"). Skipped otherwise so CI without a server
# stays green. Uses a uniquely named collection per test run and tears it
# down at the end so repeated runs don't leak state.
# ---------------------------------------------------------------------------

import os
import uuid


_QDRANT_URL = os.environ.get("CYLLAMA_QDRANT_URL")


@pytest.mark.integration
@pytest.mark.skipif(not _QDRANT_URL, reason="CYLLAMA_QDRANT_URL not set")
class TestRealServer:
    @pytest.fixture
    def server_store(self):
        collection = f"cyllama_test_{uuid.uuid4().hex[:12]}"
        s = QdrantVectorStore(
            dimension=4,
            collection_name=collection,
            url=_QDRANT_URL,
        )
        try:
            yield s
        finally:
            try:
                s.client.delete_collection(collection_name=collection)
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
