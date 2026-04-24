"""Alternative vector-store backend adapters for :class:`VectorStoreProtocol`.

Each adapter is lazy-imported so its optional dependency (e.g.
``qdrant-client``) is only required when the adapter is actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .qdrant import QdrantVectorStore

__all__ = ["QdrantVectorStore"]


def __getattr__(name: str) -> Any:
    if name == "QdrantVectorStore":
        from .qdrant import QdrantVectorStore

        return QdrantVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
