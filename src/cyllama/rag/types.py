"""Data types for cyllama RAG module."""

from dataclasses import dataclass, field
from typing import Any


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
