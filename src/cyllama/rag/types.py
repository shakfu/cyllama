"""Shared vocabulary for cyllama RAG backends.

Holds the dataclasses (`SearchResult`, `EmbeddingResult`, `Document`,
`Chunk`) and the structural contracts (`EmbedderProtocol`,
`VectorStoreProtocol`) that backends and consumers share. Lives in
its own module so concrete implementations (`embedder.py`,
`store.py`) and consumers (`rag.py`, `pipeline.py`) can import the
shared vocabulary without depending on any specific implementation.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class SearchResult:
    """Result from vector similarity search."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""

    embedding: list[float]
    text: str
    token_count: int


@dataclass
class Document:
    """A document with text content and metadata."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass
class Chunk:
    """A chunk of text from a document."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_id: str | None = None
    chunk_index: int = 0


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Structural contract for backends usable as the RAG embedder.

    The default :class:`~cyllama.rag.embedder.Embedder` (llama.cpp GGUF
    embedding models) satisfies this protocol; alternative backends
    only need to implement these members to be drop-in replacements
    via ``RAG(embedder=...)`` or ``RAGPipeline(embedder=...)``.

    The contract is intentionally narrow -- it covers only what
    :class:`~cyllama.rag.pipeline.RAGPipeline` and
    :class:`~cyllama.rag.RAG` actually call on the embedder. Backend-
    specific extensions (caching introspection, ``embed_with_info``
    with token counts, async APIs) remain on the concrete classes
    and aren't part of the cross-backend interface.
    """

    @property
    def dimension(self) -> int:
        """Embedding dimensionality. Must match the vector store's dimension."""
        ...

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts; return one vector per input."""
        ...

    def close(self) -> None:
        """Release any resources (model handles, network sessions)."""
        ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Structural contract for backends usable as the RAG store.

    The default :class:`~cyllama.rag.store.SqliteVectorStore`
    (sqlite-vector) satisfies this protocol; alternative backends only
    need to implement these methods to be drop-in replacements via
    ``RAG(store=...)`` or ``RAGPipeline(store=...)``.

    The contract is intentionally narrow -- it covers only what
    :class:`~cyllama.rag.pipeline.RAGPipeline` and
    :class:`~cyllama.rag.RAG` actually call on the store. Backend-
    specific features (sqlite-vector quantization, FTS5 hybrid search
    via :class:`~cyllama.rag.advanced.HybridStore`, raw SQL access
    through ``store.conn``) remain on the concrete classes and aren't
    part of the cross-backend interface.

    Source-dedup methods (``is_source_indexed`` and
    ``get_source_by_label``) are required because :class:`RAG` uses
    them to avoid re-indexing unchanged files. Backends without a
    natural place to track per-source content hashes can return
    ``False`` / ``None`` to opt out of dedup -- the RAG layer treats
    that as "always re-index" and behaves correctly, just less
    efficiently on repeated ``add_documents`` calls.
    """

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        threshold: float | None = None,
    ) -> list[SearchResult]:
        """Return the top-``k`` matches above ``threshold`` similarity."""
        ...

    def add(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
        source_hash: str | None = None,
        source_label: str | None = None,
    ) -> list[int]:
        """Insert chunks; return their assigned IDs.

        ``source_hash`` and ``source_label`` are recorded together with
        the chunks for dedup purposes; backends that don't support
        dedup may ignore them.
        """
        ...

    def is_source_indexed(self, content_hash: str) -> bool:
        """Return True if ``add(source_hash=content_hash, ...)`` was
        previously called. Backends without dedup support may return
        False unconditionally (the RAG layer will then re-index)."""
        ...

    def get_source_by_label(self, source_label: str) -> dict[str, Any] | None:
        """Look up an indexed source by its human-readable label.

        Used to detect "same filename, different content" collisions.
        Should return a dict with at least ``content_hash`` and
        ``source_label`` keys, or None if no source with this label is
        indexed. Backends without dedup support may return None.
        """
        ...

    def clear(self) -> int:
        """Remove all data from the store; return the number of items removed."""
        ...

    def close(self) -> None:
        """Release any resources (connections, file handles)."""
        ...

    def __len__(self) -> int:
        """Return the number of stored embeddings."""
        ...
