"""Qdrant adapter for :class:`~cyllama.rag.types.VectorStoreProtocol`.

Reference implementation of an alternative vector-store backend. Ships
behind the optional ``qdrant`` dep group::

    pip install cyllama[qdrant]

Example:
    >>> from cyllama.rag import RAG
    >>> from cyllama.rag.stores import QdrantVectorStore
    >>> store = QdrantVectorStore(dimension=384, location=":memory:")
    >>> rag = RAG(embedding_model=..., generation_model=..., store=store)

Transport selection (pass exactly one):

* ``location=":memory:"`` -- ephemeral in-process store (default).
* ``path="./qdrant_data"`` -- local on-disk store (no server).
* ``url="http://localhost:6333"`` -- remote Qdrant server.
* ``client=<QdrantClient>`` -- fully configured client the caller owns.

Source deduplication is implemented via payload fields on each chunk
point: ``content_hash``, ``source_label``, ``indexed_at``.
:meth:`is_source_indexed` is a count query on ``content_hash``;
:meth:`get_source_by_label` is a scroll+count on ``source_label``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..types import SearchResult, VectorStoreProtocol


def _require_qdrant() -> tuple[Any, Any]:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels
    except ImportError as e:  # pragma: no cover - exercised only when dep missing
        raise ImportError(
            "qdrant-client is required for QdrantVectorStore. Install with: pip install cyllama[qdrant]"
        ) from e
    return QdrantClient, qmodels


_METRIC_TO_DISTANCE = {
    "cosine": "COSINE",
    "dot": "DOT",
    "l2": "EUCLID",
    "l1": "MANHATTAN",
}

# Payload keys reserved by the adapter. User-supplied metadata that
# collides with these would be overwritten; we surface the reserved
# set so test and docs can reference one list.
_RESERVED_PAYLOAD_KEYS = frozenset({"text", "content_hash", "source_label", "indexed_at"})


class QdrantVectorStore(VectorStoreProtocol):
    """Qdrant-backed :class:`VectorStoreProtocol` implementation.

    IDs are assigned as monotonically increasing integers starting at
    ``len(self)`` on construction. Deletes outside of :meth:`clear`
    are not supported -- the protocol doesn't require a ``delete``
    method and mixing deletes with a counter-based scheme would skip
    IDs. Callers that need arbitrary deletion should drop to the
    underlying ``self.client`` and accept that ID assignment may then
    collide.
    """

    VALID_METRICS = frozenset(_METRIC_TO_DISTANCE.keys())

    def __init__(
        self,
        dimension: int,
        collection_name: str = "embeddings",
        metric: str = "cosine",
        *,
        location: str | None = None,
        path: str | None = None,
        url: str | None = None,
        client: Any = None,
        **client_kwargs: Any,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        metric_lower = metric.lower()
        if metric_lower not in self.VALID_METRICS:
            raise ValueError(f"Invalid metric: {metric!r}. Must be one of: {sorted(self.VALID_METRICS)}")

        QdrantClient, qmodels = _require_qdrant()
        self._qmodels = qmodels

        self.dimension = dimension
        self.collection_name = collection_name
        self.metric = metric_lower
        self._closed = False

        provided = [x for x in (location, path, url, client) if x is not None]
        if len(provided) > 1:
            raise ValueError("Pass only one of: location, path, url, client")

        if client is not None:
            self.client = client
            self._owns_client = False
        else:
            if url is not None:
                self.client = QdrantClient(url=url, **client_kwargs)
            elif path is not None:
                self.client = QdrantClient(path=path, **client_kwargs)
            else:
                # Default to :memory: when nothing is specified so tests
                # and quick-start usage don't require a running server.
                self.client = QdrantClient(location=location or ":memory:", **client_kwargs)
            self._owns_client = True

        self._ensure_collection()
        # Seed the ID counter from the current collection size. This is
        # correct for append-only workloads (fresh store, or reopen
        # without deletes). After clear() we reset to 0.
        self._next_id = self._count_exact()

    def _ensure_collection(self) -> None:
        qmodels = self._qmodels
        distance = getattr(qmodels.Distance, _METRIC_TO_DISTANCE[self.metric])
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in existing:
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(size=self.dimension, distance=distance),
        )

    def _count_exact(self) -> int:
        return int(self.client.count(collection_name=self.collection_name, exact=True).count)

    def _check_closed(self) -> None:
        if self._closed:
            raise RuntimeError("QdrantVectorStore is closed")

    def add(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
        source_hash: str | None = None,
        source_label: str | None = None,
    ) -> list[int]:
        self._check_closed()
        if len(embeddings) != len(texts):
            raise ValueError(f"embeddings and texts must have same length: {len(embeddings)} vs {len(texts)}")
        if source_hash is not None and source_label is None:
            raise ValueError("source_hash requires source_label")
        if metadata is None:
            metadata = [{} for _ in embeddings]
        elif len(metadata) != len(embeddings):
            raise ValueError(f"metadata must have same length as embeddings: {len(metadata)} vs {len(embeddings)}")
        for emb in embeddings:
            if len(emb) != self.dimension:
                raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(emb)}")

        qmodels = self._qmodels
        indexed_at = datetime.now(timezone.utc).isoformat()
        ids = list(range(self._next_id, self._next_id + len(embeddings)))

        points = []
        for id_, emb, text, meta in zip(ids, embeddings, texts, metadata):
            payload: dict[str, Any] = dict(meta) if meta else {}
            payload["text"] = text
            if source_hash is not None:
                payload["content_hash"] = source_hash
                payload["source_label"] = source_label
                payload["indexed_at"] = indexed_at
            points.append(qmodels.PointStruct(id=id_, vector=list(emb), payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        self._next_id += len(embeddings)
        return ids

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        threshold: float | None = None,
    ) -> list[SearchResult]:
        self._check_closed()
        if len(query_embedding) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(query_embedding)}")

        # query_points() is the modern Qdrant API; .search() was
        # deprecated and removed. Returns a QueryResponse whose
        # `.points` list mirrors the legacy ScoredPoint shape.
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=list(query_embedding),
            limit=k,
            score_threshold=threshold,
            with_payload=True,
        )
        hits = response.points

        results: list[SearchResult] = []
        for h in hits:
            payload = dict(h.payload or {})
            text = payload.pop("text", "")
            for reserved in ("content_hash", "source_label", "indexed_at"):
                payload.pop(reserved, None)
            results.append(
                SearchResult(
                    id=str(h.id),
                    text=text,
                    score=float(h.score),
                    metadata=payload,
                )
            )
        return results

    def is_source_indexed(self, content_hash: str) -> bool:
        self._check_closed()
        qmodels = self._qmodels
        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="content_hash",
                    match=qmodels.MatchValue(value=content_hash),
                )
            ]
        )
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=flt,
            exact=True,
        )
        return int(result.count) > 0

    def get_source_by_label(self, source_label: str) -> dict[str, Any] | None:
        self._check_closed()
        qmodels = self._qmodels
        label_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="source_label",
                    match=qmodels.MatchValue(value=source_label),
                )
            ]
        )
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=label_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None
        payload = points[0].payload or {}
        content_hash = payload.get("content_hash")
        indexed_at = payload.get("indexed_at")

        # Count chunks that belong to this source. Prefer content_hash
        # (identifies the exact source) over source_label (may collide
        # across reindexed content, though that case raises one layer
        # up in RAG).
        if content_hash is not None:
            hash_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="content_hash",
                        match=qmodels.MatchValue(value=content_hash),
                    )
                ]
            )
            chunk_count = int(
                self.client.count(
                    collection_name=self.collection_name,
                    count_filter=hash_filter,
                    exact=True,
                ).count
            )
        else:
            chunk_count = int(
                self.client.count(
                    collection_name=self.collection_name,
                    count_filter=label_filter,
                    exact=True,
                ).count
            )

        return {
            "content_hash": content_hash,
            "source_label": source_label,
            "chunk_count": chunk_count,
            "indexed_at": indexed_at,
        }

    def clear(self) -> int:
        self._check_closed()
        count = self._count_exact()
        self.client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection()
        self._next_id = 0
        return count

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_client:
            try:
                self.client.close()
            except Exception:
                # Remote HTTP clients have a no-op close; local/in-memory
                # clients may raise on double-close. Either way the
                # store is considered closed from the caller's view.
                pass
        self._closed = True

    def __len__(self) -> int:
        self._check_closed()
        return self._count_exact()

    def __enter__(self) -> "QdrantVectorStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self._closed else f"open, {self._count_exact()} vectors"
        return (
            f"QdrantVectorStore(dimension={self.dimension}, "
            f"collection_name={self.collection_name!r}, "
            f"metric={self.metric!r}, status={status})"
        )
