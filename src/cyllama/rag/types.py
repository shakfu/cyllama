"""Shared vocabulary for cyllama RAG backends.

Holds the dataclasses (`SearchResult`, `EmbeddingResult`, `Document`,
`Chunk`) and the structural contracts (`EmbedderProtocol`) that
backends and consumers share. Lives in its own module so concrete
implementations (`embedder.py`, `store.py`) and consumers (`rag.py`,
`pipeline.py`) can import the shared vocabulary without depending on
any specific implementation.

The :class:`~cyllama.rag.store.VectorStoreProtocol` still lives in
``store.py`` next to :class:`~cyllama.rag.store.SqliteVectorStore` for
now; moving it here would be the obvious follow-up.
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
