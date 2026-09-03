"""Chroma adapter for :class:`~cyllama.rag.types.VectorStoreProtocol`.

Ships behind an optional dependency::

    pip install chromadb

Example:
    >>> from cyllama.rag import RAG
    >>> from cyllama.rag.stores import ChromaVectorStore
    >>> store = ChromaVectorStore(dimension=384)          # ephemeral
    >>> rag = RAG(embedding_model=..., generation_model=..., store=store)

Transport selection (pass at most one):

* nothing            -- ephemeral in-process client (default).
* ``path="./chroma"`` -- local on-disk persistent client.
* ``host="localhost"`` (with optional ``port``) -- remote Chroma server.
* ``client=<ClientAPI>`` -- fully configured client the caller owns.

cyllama supplies the vectors itself, so the collection is created
without an embedding function; Chroma never calls out to embed anything.

Source deduplication is implemented via per-record metadata fields
(``content_hash``, ``source_label``, ``indexed_at``), mirroring the
Qdrant adapter. :meth:`is_source_indexed` and
:meth:`get_source_by_label` are ``where``-filtered ``get`` calls.

Metadata handling
-----------------

Chroma only accepts scalar metadata values (``str``, ``int``, ``float``,
``bool``, ``None`` and lists). Nested values are JSON-encoded on the way
in and decoded on the way out, so arbitrary JSON-serializable metadata
round-trips through :meth:`add` / :meth:`search` unchanged.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from ..types import SearchResult, VectorStoreProtocol

# Chroma's HNSW space names. "ip" is an inner-product space whose
# reported distance is `1 - dot`, which is what makes the dot metric
# recoverable below.
_METRIC_TO_SPACE = {
    "cosine": "cosine",
    "l2": "l2",
    "dot": "ip",
}

# Metadata keys the adapter writes itself. User metadata that collides
# with one of these would be overwritten, so the set is named here for
# tests and docs to reference.
_RESERVED_METADATA_KEYS = frozenset({"content_hash", "source_label", "indexed_at"})

# Marker prefix for values that had to be JSON-encoded to satisfy
# Chroma's scalar-only metadata rule.
_JSON_PREFIX = "\x00json:"

# Chroma rejects collection names outside this shape with a validation
# error from deep inside the Rust bindings; check up front so the caller
# gets a clear message instead.
_VALID_COLLECTION_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")


def _require_chromadb() -> Any:
    try:
        import chromadb
    except ImportError as e:  # pragma: no cover - exercised only when dep missing
        raise ImportError("chromadb is required for ChromaVectorStore. Install with: pip install chromadb") from e
    return chromadb


def _encode_metadata(meta: dict[str, Any]) -> dict[str, Any] | None:
    """Flatten a metadata dict into Chroma's scalar-only value space.

    Returns None for an empty dict -- Chroma rejects ``{}`` outright but
    accepts a null entry.
    """
    if not meta:
        return None
    encoded: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            encoded[key] = value
        else:
            encoded[key] = _JSON_PREFIX + json.dumps(value)
    return encoded


def _decode_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
    """Inverse of :func:`_encode_metadata`, minus the reserved keys."""
    if not meta:
        return {}
    decoded: dict[str, Any] = {}
    for key, value in meta.items():
        if key in _RESERVED_METADATA_KEYS:
            continue
        if isinstance(value, str) and value.startswith(_JSON_PREFIX):
            decoded[key] = json.loads(value[len(_JSON_PREFIX) :])
        else:
            decoded[key] = value
    return decoded


class ChromaVectorStore(VectorStoreProtocol):
    """Chroma-backed :class:`VectorStoreProtocol` implementation.

    IDs are assigned as monotonically increasing integers seeded from
    the collection's size on construction, matching
    :class:`~cyllama.rag.stores.qdrant.QdrantVectorStore`. Chroma stores
    them as strings; :meth:`add` returns them as ints and
    :class:`~cyllama.rag.types.SearchResult` reports them as strings, as
    the protocol requires.

    Args:
        dimension: Embedding dimension. Chroma infers dimensionality
            from the first insert; this is used to validate inputs
            before they reach the client.
        collection_name: Chroma collection to use, created on demand.
            Must be 3-512 characters of ``[a-zA-Z0-9._-]``, starting and
            ending alphanumeric.
        metric: One of ``cosine``, ``l2``, ``dot``.
        path: Directory for a local persistent client.
        host: Hostname for a remote Chroma server.
        port: Port for a remote Chroma server. Defaults to Chroma's own
            default when only ``host`` is given.
        client: A pre-built Chroma client the caller owns and closes.
        **client_kwargs: Forwarded to the client constructor.
    """

    VALID_METRICS = frozenset(_METRIC_TO_SPACE.keys())

    def __init__(
        self,
        dimension: int,
        collection_name: str = "embeddings",
        metric: str = "cosine",
        *,
        path: str | None = None,
        host: str | None = None,
        port: int | None = None,
        client: Any = None,
        **client_kwargs: Any,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")

        metric_lower = metric.lower()
        if metric_lower not in self.VALID_METRICS:
            raise ValueError(f"Invalid metric: {metric!r}. Must be one of: {sorted(self.VALID_METRICS)}")

        if not _VALID_COLLECTION_NAME.match(collection_name):
            raise ValueError(
                f"Invalid collection_name: {collection_name!r}. Chroma requires 3-512 "
                "characters from [a-zA-Z0-9._-], starting and ending alphanumeric."
            )

        chromadb = _require_chromadb()

        self.dimension = dimension
        self.collection_name = collection_name
        self.metric = metric_lower
        self._closed = False

        provided = [x for x in (path, host, client) if x is not None]
        if len(provided) > 1:
            raise ValueError("Pass only one of: path, host, client")
        if port is not None and host is None:
            raise ValueError("port requires host")

        if client is not None:
            self.client = client
            self._owns_client = False
        else:
            if host is not None:
                if port is not None:
                    client_kwargs["port"] = port
                self.client = chromadb.HttpClient(host=host, **client_kwargs)
            elif path is not None:
                self.client = chromadb.PersistentClient(path=path, **client_kwargs)
            else:
                self.client = chromadb.EphemeralClient(**client_kwargs)
            self._owns_client = True

        self._ensure_collection()
        # Seed the ID counter from the current collection size. Correct
        # for append-only use (fresh store, or reopen without deletes);
        # clear() resets it to 0.
        self._next_id = self.collection.count()

    def _ensure_collection(self) -> None:
        # embedding_function=None keeps Chroma from trying to embed
        # anything itself -- cyllama always supplies vectors.
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": _METRIC_TO_SPACE[self.metric]},
            embedding_function=None,
        )

    def _check_closed(self) -> None:
        if self._closed:
            raise RuntimeError("ChromaVectorStore is closed")

    def _score(self, distance: float) -> float:
        """Convert a Chroma distance into a higher-is-better score.

        Chroma reports ``1 - cosine_similarity`` for the cosine space,
        ``1 - dot`` for the inner-product space, and *squared* euclidean
        distance for the l2 space.
        """
        if self.metric == "cosine":
            return 1.0 - distance
        if self.metric == "dot":
            return 1.0 - distance
        return -distance

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------

    def add(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
        source_hash: str | None = None,
        source_label: str | None = None,
    ) -> list[int]:
        """Insert chunks; return their assigned IDs."""
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

        if not embeddings:
            return []

        indexed_at = datetime.now(timezone.utc).isoformat()
        ids = list(range(self._next_id, self._next_id + len(embeddings)))

        metadatas: list[dict[str, Any] | None] = []
        for meta in metadata:
            encoded = _encode_metadata(meta) or {}
            if source_hash is not None:
                encoded["content_hash"] = source_hash
                encoded["source_label"] = source_label
                encoded["indexed_at"] = indexed_at
            metadatas.append(encoded or None)

        self.collection.add(
            ids=[str(i) for i in ids],
            embeddings=[list(e) for e in embeddings],
            documents=list(texts),
            metadatas=metadatas,
        )
        self._next_id += len(embeddings)
        return ids

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        threshold: float | None = None,
    ) -> list[SearchResult]:
        """Return the top-``k`` matches, best first."""
        self._check_closed()
        if len(query_embedding) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(query_embedding)}")

        # Chroma raises on n_results < 1 and clamps anything larger than
        # the collection, so only the lower bound needs guarding.
        if k < 1 or self.collection.count() == 0:
            return []

        response = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # query() returns one list per query embedding; we only ever
        # send one.
        ids = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]

        results: list[SearchResult] = []
        for id_, document, meta, distance in zip(ids, documents, metadatas, distances):
            score = self._score(float(distance))
            if threshold is not None and score < threshold:
                continue
            results.append(
                SearchResult(
                    id=str(id_),
                    text=document or "",
                    score=score,
                    metadata=_decode_metadata(meta),
                )
            )
        return results

    def is_source_indexed(self, content_hash: str) -> bool:
        """Return True if a source with this content hash was added."""
        self._check_closed()
        found = self.collection.get(where={"content_hash": content_hash}, limit=1, include=[])
        return bool(found["ids"])

    def get_source_by_label(self, source_label: str) -> dict[str, Any] | None:
        """Look up an indexed source by its human-readable label."""
        self._check_closed()
        found = self.collection.get(
            where={"source_label": source_label},
            limit=1,
            include=["metadatas"],
        )
        if not found["ids"]:
            return None

        meta = (found["metadatas"] or [{}])[0] or {}
        content_hash = meta.get("content_hash")
        indexed_at = meta.get("indexed_at")

        # Count the chunks belonging to this source. Prefer
        # content_hash, which identifies the exact source; source_label
        # can collide across re-indexed content (a case the RAG layer
        # rejects one level up).
        where = {"content_hash": content_hash} if content_hash is not None else {"source_label": source_label}
        chunk_count = len(self.collection.get(where=where, include=[])["ids"])

        return {
            "content_hash": content_hash,
            "source_label": source_label,
            "chunk_count": chunk_count,
            "indexed_at": indexed_at,
        }

    def clear(self) -> int:
        """Drop and recreate the collection; return the count removed."""
        self._check_closed()
        count = int(self.collection.count())
        self.client.delete_collection(name=self.collection_name)
        self._ensure_collection()
        self._next_id = 0
        return count

    def close(self) -> None:
        """Release the client if this store created it. Idempotent."""
        if self._closed:
            return
        if self._owns_client:
            # Ephemeral and HTTP clients have nothing to release;
            # persistent ones may not expose close() on every version.
            closer = getattr(self.client, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
        self._closed = True

    def __len__(self) -> int:
        self._check_closed()
        return int(self.collection.count())

    def __enter__(self) -> "ChromaVectorStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self._closed else f"open, {self.collection.count()} vectors"
        return (
            f"ChromaVectorStore(dimension={self.dimension}, "
            f"collection_name={self.collection_name!r}, "
            f"metric={self.metric!r}, status={status})"
        )
