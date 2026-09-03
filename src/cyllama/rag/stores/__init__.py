"""Alternative vector-store backend adapters for :class:`VectorStoreProtocol`.

Each adapter is lazy-imported so its optional dependency (e.g.
``qdrant-client``, ``sqlite-vec``, ``chromadb``) is only required when
the adapter is actually used.

======================  ==========================  =========================
Adapter                 Install                     Backend
======================  ==========================  =========================
``QdrantVectorStore``   ``pip install qdrant-client``  Qdrant
``SqliteVecStore``      ``pip install sqlite-vec``     sqlite-vec (MIT/Apache-2.0)
``ChromaVectorStore``   ``pip install chromadb``       Chroma
======================  ==========================  =========================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chroma import ChromaVectorStore
    from .qdrant import QdrantVectorStore
    from .sqlite_vec import SqliteVecStore

__all__ = ["ChromaVectorStore", "QdrantVectorStore", "SqliteVecStore"]


def __getattr__(name: str) -> Any:
    if name == "QdrantVectorStore":
        from .qdrant import QdrantVectorStore

        return QdrantVectorStore
    if name == "SqliteVecStore":
        from .sqlite_vec import SqliteVecStore

        return SqliteVecStore
    if name == "ChromaVectorStore":
        from .chroma import ChromaVectorStore

        return ChromaVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
