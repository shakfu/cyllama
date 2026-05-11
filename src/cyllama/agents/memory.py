"""Semantic (long-term, RAG-backed) memory for agents.

Bridges the :mod:`cyllama.rag` subsystem to the agent layer as a long-term
memory primitive. Where :mod:`cyllama.agents.session` handles short-term
(within-run) and episodic (across-runs) memory keyed by session ids,
:class:`SemanticMemory` handles cross-session content-addressed recall --
"remember that the user prefers concise answers" -- via vector retrieval
over arbitrary text fragments.

The design choice is deliberately minimal: SemanticMemory is a thin
namespace-aware facade over a RAG instance. It does not introduce its
own vector store, embedder, or storage backend. Whatever the user has
already configured in their :class:`cyllama.rag.RAG` instance is what
gets used.

Example::

    from cyllama.rag import RAG
    from cyllama.agents import SemanticMemory, rag_as_tool, ReActAgent

    rag = RAG.from_documents([])  # empty store
    memory = SemanticMemory(rag)

    # Write into a per-user namespace.
    memory.remember(
        "The user is allergic to peanuts.",
        namespace="user:alice",
    )

    # Recall later.
    hits = memory.retrieve("dietary restrictions", namespace="user:alice")

The namespace is stored as metadata on the underlying RAG records;
retrieval filters by the same field. Records under one namespace are
invisible to retrieve() calls under another, but they share the same
embedding store and so the same vector budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


__all__ = ["MemoryRecord", "SemanticMemory"]


@dataclass
class MemoryRecord:
    """A single retrieved memory fragment.

    Mirrors :class:`cyllama.rag.types.SearchResult` but is what
    :class:`SemanticMemory.retrieve` returns directly, so callers don't
    have to import RAG internals to consume memory.
    """

    text: str
    score: float
    namespace: str
    metadata: Dict[str, Any]


class SemanticMemory:
    """Namespace-aware semantic memory backed by a RAG instance.

    Args:
        rag: A :class:`cyllama.rag.RAG` instance (or any object exposing
            ``add_texts(texts, metadata, split)`` and
            ``search(query, k)`` returning ``SearchResult``-shaped
            objects).
        namespace_field: Metadata key under which the namespace is
            stored on each record. Default ``"_memory_namespace"`` --
            chosen with a leading underscore so it's unlikely to collide
            with user-supplied metadata fields.
        default_namespace: Namespace used when ``remember`` /
            ``retrieve`` is called without one. Default ``"default"``.
    """

    def __init__(
        self,
        rag: Any,
        namespace_field: str = "_memory_namespace",
        default_namespace: str = "default",
    ) -> None:
        self.rag = rag
        self.namespace_field = namespace_field
        self.default_namespace = default_namespace

    def remember(
        self,
        text: str,
        *,
        namespace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        split: bool = False,
    ) -> List[int]:
        """Store a text fragment under the given namespace.

        Args:
            text: The content to remember.
            namespace: Logical bucket the memory belongs to (per-user,
                per-conversation, per-topic, etc.). Defaults to
                ``self.default_namespace``.
            metadata: Optional additional metadata stored alongside.
                The namespace key is reserved -- if present in the dict
                it will be overwritten.
            split: Whether to split long texts into chunks. Defaults
                to False (memory fragments are usually short).

        Returns:
            List of inserted chunk ids from the underlying RAG store
            (may be empty if the content was deduplicated).
        """
        ns = namespace or self.default_namespace
        meta = dict(metadata or {})
        meta[self.namespace_field] = ns
        result = self.rag.add_texts([text], metadata=[meta], split=split)
        # RAG.add_texts returns an IndexResult (subclass of list); cast
        # to plain list for callers who don't want the RAG-specific type.
        return list(result)

    def retrieve(
        self,
        query: str,
        *,
        namespace: Optional[str] = None,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[MemoryRecord]:
        """Retrieve memories matching ``query`` from the given namespace.

        Args:
            query: The query text. Embedded and matched against the
                underlying vector store.
            namespace: Filter results to this namespace only. Defaults
                to ``self.default_namespace``.
            top_k: Maximum number of results to return *after*
                namespace filtering. The underlying search is over-fetched
                (by 4x) to give the filter room to work; if you store
                heavily in one namespace and rarely in another, raise
                this cap.
            threshold: Optional minimum similarity score (passed through
                to the underlying ``RAG.search``).

        Returns:
            List of :class:`MemoryRecord`, ordered by score descending,
            length at most ``top_k``.
        """
        ns = namespace or self.default_namespace
        # Over-fetch to leave room for namespace filtering. Multiplier of
        # 4 is a heuristic -- if a single namespace dominates the store
        # it's wasted work, but if namespaces are roughly equal it gives
        # the filter a fair pool to draw from.
        oversample = max(top_k * 4, top_k + 8)
        hits = self.rag.search(query, k=oversample, threshold=threshold)

        filtered: List[MemoryRecord] = []
        for hit in hits:
            hit_ns = (hit.metadata or {}).get(self.namespace_field)
            if hit_ns != ns:
                continue
            # Strip the namespace key from the surfaced metadata so
            # callers don't see our bookkeeping.
            clean_meta = {k: v for k, v in (hit.metadata or {}).items() if k != self.namespace_field}
            filtered.append(
                MemoryRecord(
                    text=hit.text,
                    score=hit.score,
                    namespace=ns,
                    metadata=clean_meta,
                )
            )
            if len(filtered) >= top_k:
                break
        return filtered

    def forget(self, *, namespace: Optional[str] = None) -> None:
        """Clear all memories under ``namespace``.

        Currently a stub: the underlying :class:`RAG` exposes
        :meth:`~cyllama.rag.RAG.clear` (clears everything, not by
        namespace) and per-id deletion via the store layer, but no
        metadata-filtered delete. Pending a RAG-side API for filtered
        deletion; until then, callers who need per-namespace forgetting
        should manage separate :class:`RAG` instances.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError(
            "SemanticMemory.forget() is not yet implemented. The underlying "
            "RAG store doesn't expose metadata-filtered deletion; either "
            "use rag.clear() to wipe everything or manage separate RAG "
            "instances per namespace."
        )
