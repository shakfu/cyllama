"""Tests for `cyllama.agents.memory.SemanticMemory`.

Uses a stub RAG that mimics the relevant slice of the real ``RAG`` API
(``add_texts`` + ``search`` returning ``SearchResult``-shaped hits) so
the tests are pure-Python and don't need a real embedder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from cyllama.agents import SemanticMemory, MemoryRecord


# ---------------------------------------------------------------------------
# Stub RAG matching the subset of cyllama.rag.RAG that SemanticMemory uses.
# ---------------------------------------------------------------------------


@dataclass
class _StubHit:
    """Matches the duck-type of ``cyllama.rag.types.SearchResult``."""

    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class _StubRAG:
    """In-memory store keyed by insertion order; search returns all
    entries with score=1.0 (we don't care about ranking quality in
    unit tests -- just metadata-routing)."""

    def __init__(self) -> None:
        self.records: List[_StubHit] = []
        self.add_calls: List[tuple] = []
        self.search_calls: List[tuple] = []

    def add_texts(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        split: bool = True,
    ) -> List[int]:
        self.add_calls.append((list(texts), metadata, split))
        ids: List[int] = []
        for i, text in enumerate(texts):
            meta = (metadata or [{}] * len(texts))[i]
            self.records.append(_StubHit(text=text, score=1.0, metadata=dict(meta)))
            ids.append(len(self.records) - 1)
        return ids

    def search(
        self,
        query: str,
        k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[_StubHit]:
        # Trivial "search": return all records, ranked by simple substring
        # heuristic so ordering is stable.
        self.search_calls.append((query, k, threshold))
        scored: List[tuple[float, _StubHit]] = []
        ql = query.lower()
        for r in self.records:
            tl = r.text.lower()
            # Cheap relevance score: 1.0 if substring, 0.5 if word overlap,
            # 0.1 otherwise.
            if ql in tl:
                score = 1.0
            elif any(w in tl for w in ql.split()):
                score = 0.5
            else:
                score = 0.1
            scored.append((score, _StubHit(text=r.text, score=score, metadata=r.metadata)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [h for _, h in scored[:k]]


# ---------------------------------------------------------------------------
# remember / retrieve basics
# ---------------------------------------------------------------------------


def test_remember_inserts_into_rag_with_namespace_metadata():
    rag = _StubRAG()
    mem = SemanticMemory(rag)
    mem.remember("the sky is blue", namespace="facts")
    assert len(rag.add_calls) == 1
    texts, metadata, split = rag.add_calls[0]
    assert texts == ["the sky is blue"]
    assert metadata == [{"_memory_namespace": "facts"}]
    # split=False by default (memory fragments are usually short)
    assert split is False


def test_remember_returns_inserted_ids():
    rag = _StubRAG()
    mem = SemanticMemory(rag)
    ids = mem.remember("hello", namespace="ns")
    assert ids == [0]


def test_remember_uses_default_namespace_when_unspecified():
    rag = _StubRAG()
    mem = SemanticMemory(rag, default_namespace="myns")
    mem.remember("x")
    assert rag.records[0].metadata["_memory_namespace"] == "myns"


def test_remember_preserves_user_metadata():
    rag = _StubRAG()
    mem = SemanticMemory(rag)
    mem.remember("blah", namespace="ns", metadata={"author": "alice", "tag": "test"})
    stored = rag.records[0].metadata
    assert stored["author"] == "alice"
    assert stored["tag"] == "test"
    assert stored["_memory_namespace"] == "ns"


def test_remember_namespace_overrides_user_supplied_namespace_metadata():
    """If the user puts the namespace field in metadata, the explicit
    namespace kwarg still wins -- avoids namespace-spoofing."""
    rag = _StubRAG()
    mem = SemanticMemory(rag)
    mem.remember(
        "x",
        namespace="actual",
        metadata={"_memory_namespace": "spoofed"},
    )
    assert rag.records[0].metadata["_memory_namespace"] == "actual"


# ---------------------------------------------------------------------------
# retrieve: namespace filtering
# ---------------------------------------------------------------------------


def test_retrieve_returns_only_records_in_namespace():
    rag = _StubRAG()
    mem = SemanticMemory(rag)
    mem.remember("alpha", namespace="A")
    mem.remember("beta", namespace="B")
    mem.remember("gamma", namespace="A")

    hits = mem.retrieve("alpha beta gamma", namespace="A")
    texts = [h.text for h in hits]
    assert "alpha" in texts
    assert "gamma" in texts
    assert "beta" not in texts


def test_retrieve_returns_MemoryRecord_objects():
    rag = _StubRAG()
    mem = SemanticMemory(rag)
    mem.remember("hello", namespace="ns", metadata={"src": "test"})
    hits = mem.retrieve("hello", namespace="ns")
    assert all(isinstance(h, MemoryRecord) for h in hits)
    assert hits[0].text == "hello"
    assert hits[0].namespace == "ns"
    # Surface metadata strips out the internal namespace field.
    assert "_memory_namespace" not in hits[0].metadata
    assert hits[0].metadata["src"] == "test"


def test_retrieve_respects_top_k():
    rag = _StubRAG()
    mem = SemanticMemory(rag)
    for i in range(10):
        mem.remember(f"fact {i}", namespace="ns")
    hits = mem.retrieve("fact", namespace="ns", top_k=3)
    assert len(hits) == 3


def test_retrieve_default_namespace():
    rag = _StubRAG()
    mem = SemanticMemory(rag, default_namespace="defns")
    mem.remember("foo")  # no namespace -> defns
    mem.remember("bar", namespace="other")
    hits = mem.retrieve("foo bar")
    texts = [h.text for h in hits]
    assert "foo" in texts
    assert "bar" not in texts


def test_retrieve_overfetches_to_give_filter_room():
    """When namespace filtering removes most hits, the underlying search
    is called with an over-fetched k to give the filter a chance to find
    enough results."""
    rag = _StubRAG()
    mem = SemanticMemory(rag)
    mem.remember("a", namespace="rare")
    mem.remember("b", namespace="rare")
    for i in range(50):
        mem.remember(f"noise {i}", namespace="common")
    # Ask for top_k=2 from "rare" -- the underlying search should be
    # over-fetched so it has a chance to surface the 2 rare hits.
    hits = mem.retrieve("a b noise", namespace="rare", top_k=2)
    assert len(rag.search_calls) == 1
    _, requested_k, _ = rag.search_calls[0]
    assert requested_k > 2
    assert len(hits) == 2


def test_retrieve_empty_when_namespace_missing():
    rag = _StubRAG()
    mem = SemanticMemory(rag)
    mem.remember("only fact", namespace="A")
    hits = mem.retrieve("only fact", namespace="nonexistent")
    assert hits == []


def test_retrieve_passes_threshold_through():
    rag = _StubRAG()
    mem = SemanticMemory(rag)
    mem.remember("x", namespace="ns")
    mem.retrieve("x", namespace="ns", top_k=5, threshold=0.5)
    _, _, threshold = rag.search_calls[0]
    assert threshold == 0.5


# ---------------------------------------------------------------------------
# Configuration: custom namespace_field
# ---------------------------------------------------------------------------


def test_custom_namespace_field_used_for_storage_and_filtering():
    rag = _StubRAG()
    mem = SemanticMemory(rag, namespace_field="bucket")
    mem.remember("x", namespace="A")
    mem.remember("y", namespace="B")
    # Stored under "bucket", not the default field.
    assert rag.records[0].metadata == {"bucket": "A"}
    assert rag.records[1].metadata == {"bucket": "B"}
    # Filtering uses the same key.
    hits = mem.retrieve("x y", namespace="A")
    assert [h.text for h in hits] == ["x"]


# ---------------------------------------------------------------------------
# forget(): documented as not yet implemented.
# ---------------------------------------------------------------------------


def test_forget_raises_not_implemented():
    """forget() is a documented stub pending RAG-side filtered delete."""
    mem = SemanticMemory(_StubRAG())
    with pytest.raises(NotImplementedError, match="filtered deletion"):
        mem.forget(namespace="any")
