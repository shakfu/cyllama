"""High-level RAG interface for cyllama."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from .embedder import Embedder
from .loaders import load_document
from .pipeline import RAGConfig, RAGPipeline, RAGResponse
from .splitter import TextSplitter
from .store import VectorStore
from .types import Document

if TYPE_CHECKING:
    from ..api import LLM


class RAG:
    """High-level RAG interface with sensible defaults.

    Provides a simple interface for building RAG applications by combining
    embedding, vector storage, text splitting, and generation into a
    single easy-to-use class.

    Example:
        >>> from cyllama.rag import RAG
        >>>
        >>> # Initialize with models
        >>> rag = RAG(
        ...     embedding_model="models/bge-small.gguf",
        ...     generation_model="models/llama.gguf"
        ... )
        >>>
        >>> # Add documents
        >>> rag.add_texts([
        ...     "Python is a high-level programming language.",
        ...     "Machine learning uses algorithms to learn from data."
        ... ])
        >>>
        >>> # Or add from files
        >>> rag.add_documents(["docs/guide.md", "docs/api.txt"])
        >>>
        >>> # Query the knowledge base
        >>> response = rag.query("What is Python?")
        >>> print(response.text)
        >>> print(response.sources)
        >>>
        >>> # Stream response
        >>> for chunk in rag.stream("Explain machine learning"):
        ...     print(chunk, end="")
    """

    def __init__(
        self,
        embedding_model: str,
        generation_model: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        db_path: str = ":memory:",
        config: RAGConfig | None = None,
        **kwargs,
    ):
        """Initialize RAG with models.

        Creates Embedder, VectorStore, TextSplitter, and RAGPipeline
        with sensible defaults.

        Args:
            embedding_model: Path to embedding model (GGUF file)
            generation_model: Path to generation model (GGUF file)
            chunk_size: Target chunk size for text splitting
            chunk_overlap: Overlap between chunks
            db_path: Path for vector store (":memory:" for in-memory)
            config: RAG configuration (uses defaults if None)
            **kwargs: Additional arguments passed to LLM
        """
        # Import LLM here to avoid circular imports
        from ..api import LLM

        self.embedding_model = embedding_model
        self.generation_model = generation_model

        # Initialize components
        self.embedder = Embedder(embedding_model)
        self.store = VectorStore(
            dimension=self.embedder.dimension,
            db_path=db_path,
        )
        self.splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.generator = LLM(generation_model, **kwargs)

        # Create pipeline
        self.config = config or RAGConfig()
        self.pipeline = RAGPipeline(
            embedder=self.embedder,
            store=self.store,
            generator=self.generator,
            config=self.config,
        )

        self._closed = False

    def add_texts(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
        split: bool = True,
    ) -> list[int]:
        """Add text strings to the knowledge base.

        Args:
            texts: List of text strings to add
            metadata: Optional metadata for each text
            split: Whether to split texts into chunks (default: True)

        Returns:
            List of IDs for the added items
        """
        self._check_closed()

        if metadata is not None and len(metadata) != len(texts):
            raise ValueError(
                f"metadata length ({len(metadata)}) must match texts length ({len(texts)})"
            )

        all_chunks = []
        all_metadata = []

        for i, text in enumerate(texts):
            if split:
                chunks = self.splitter.split(text)
            else:
                chunks = [text]

            text_meta = metadata[i] if metadata else {}
            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_meta = text_meta.copy()
                chunk_meta["chunk_index"] = j
                all_metadata.append(chunk_meta)

        # Generate embeddings and add to store
        embeddings = self.embedder.embed_batch(all_chunks)
        return self.store.add(embeddings, all_chunks, all_metadata)

    def add_documents(
        self,
        paths: list[str | Path],
        split: bool = True,
        **loader_kwargs,
    ) -> list[int]:
        """Load and add documents from files.

        Args:
            paths: List of file paths to load
            split: Whether to split documents into chunks
            **loader_kwargs: Additional arguments passed to loaders

        Returns:
            List of IDs for the added items
        """
        self._check_closed()

        all_ids = []
        for path in paths:
            path = Path(path)
            docs = load_document(path, **loader_kwargs)

            for doc in docs:
                ids = self.add_texts(
                    [doc.text],
                    metadata=[{"source": str(path), **doc.metadata}],
                    split=split,
                )
                all_ids.extend(ids)

        return all_ids

    def add_document(
        self,
        document: Document,
        split: bool = True,
    ) -> list[int]:
        """Add a single Document object.

        Args:
            document: Document to add
            split: Whether to split into chunks

        Returns:
            List of IDs for the added items
        """
        return self.add_texts(
            [document.text],
            metadata=[document.metadata],
            split=split,
        )

    def query(
        self,
        question: str,
        config: RAGConfig | None = None,
    ) -> RAGResponse:
        """Query the knowledge base.

        Args:
            question: The question to answer
            config: Optional config override for this query

        Returns:
            RAGResponse with generated text and sources
        """
        self._check_closed()
        return self.pipeline.query(question, config=config)

    def stream(
        self,
        question: str,
        config: RAGConfig | None = None,
    ) -> Iterator[str]:
        """Stream response tokens for a question.

        Args:
            question: The question to answer
            config: Optional config override

        Yields:
            Response tokens as strings
        """
        self._check_closed()
        yield from self.pipeline.stream(question, config=config)

    def retrieve(
        self,
        question: str,
        config: RAGConfig | None = None,
    ) -> list:
        """Retrieve relevant documents without generation.

        Args:
            question: The question to retrieve documents for
            config: Optional config override

        Returns:
            List of relevant SearchResults
        """
        self._check_closed()
        return self.pipeline.retrieve(question, config=config)

    def search(
        self,
        query: str,
        k: int = 5,
        threshold: float | None = None,
    ) -> list:
        """Direct vector search without RAG formatting.

        Args:
            query: Query text to search for
            k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of SearchResults
        """
        self._check_closed()
        embedding = self.embedder.embed(query)
        return self.store.search(embedding, k=k, threshold=threshold)

    @property
    def count(self) -> int:
        """Return number of documents in the store."""
        return len(self.store)

    def clear(self) -> int:
        """Clear all documents from the store.

        Returns:
            Number of documents removed
        """
        self._check_closed()
        return self.store.clear()

    def _check_closed(self) -> None:
        """Raise error if RAG is closed."""
        if self._closed:
            raise RuntimeError("RAG instance is closed")

    def close(self) -> None:
        """Close all resources."""
        if not self._closed:
            self.store.close()
            self.generator.close()
            self.embedder.close()
            self._closed = True

    def __enter__(self) -> "RAG":
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self._closed else f"open, {self.count} docs"
        return (
            f"RAG(embedding_model={self.embedding_model!r}, "
            f"generation_model={self.generation_model!r}, "
            f"status={status})"
        )
