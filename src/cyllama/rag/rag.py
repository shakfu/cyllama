"""High-level RAG interface for cyllama."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from .types import SearchResult

from .embedder import Embedder
from .loaders import load_document
from .pipeline import RAGConfig, RAGPipeline, RAGResponse
from .splitter import TextSplitter
from .store import SqliteVectorStore, VectorStoreProtocol
from .types import Document, EmbedderProtocol

if TYPE_CHECKING:
    pass


class IndexResult(list[int]):
    """Result of an :meth:`RAG.add_documents` or :meth:`RAG.add_texts` call.

    Subclasses ``list[int]`` so callers using the legacy contract
    (``n_added += len(rag.add_documents(...))``, ``for chunk_id in
    rag.add_documents(...)``, ``rag.add_documents(...)[0]``) keep
    working with no code changes -- the list payload is the IDs of the
    chunks that were *newly inserted* by this call.

    The :attr:`skipped_labels` attribute exposes the dedup information
    for callers that want it: each entry is the human-readable label
    (file basename for ``add_documents``, hash-prefixed text label for
    ``add_texts``) of a source that was already in the store and was
    therefore not re-indexed.

    A source is considered already-indexed when its content hash
    (md5 of the file bytes for documents, md5 of the text bytes for
    raw texts) matches a row in the store's ``{table_name}_sources``
    deduplication table. This means re-running ``cyllama rag -f
    corpus.txt --db rag.db`` after a successful first run is a no-op
    on the indexing side -- the corpus is silently skipped and the
    user goes straight to query mode.
    """

    skipped_labels: list[str]

    def __init__(self, added_ids: list[int], skipped_labels: list[str]) -> None:
        super().__init__(added_ids)
        self.skipped_labels = skipped_labels


def _content_hash(data: bytes) -> str:
    """Return the md5 hex digest of ``data``.

    md5 is the right choice here because we're using it as a content
    fingerprint for deduplication, not as a security primitive. md5 is
    fast, has a small fixed-size hex output (~32 chars, fits comfortably
    in the SQLite primary key), and the chance of a collision across
    a user's corpus is effectively zero.
    """
    return hashlib.md5(data).hexdigest()


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
        store: VectorStoreProtocol | None = None,
        embedder: EmbedderProtocol | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RAG with models.

        Creates Embedder, VectorStore, TextSplitter, and RAGPipeline
        with sensible defaults.

        Args:
            embedding_model: Path to embedding model (GGUF file).
                Ignored when ``embedder`` is provided.
            generation_model: Path to generation model (GGUF file)
            chunk_size: Target chunk size for text splitting
            chunk_overlap: Overlap between chunks
            db_path: Path for vector store (":memory:" for in-memory).
                Ignored when ``store`` is provided -- the supplied store
                owns its own persistence.
            config: RAG configuration (uses defaults if None)
            store: Optional pre-built vector store conforming to
                :class:`VectorStoreProtocol`. When supplied, ``db_path``
                is ignored and the caller is responsible for the store's
                dimension matching the embedder. Use this to plug in
                alternative backends (Qdrant, Chroma, LanceDB, pgvector,
                ...) in place of the default sqlite-vector store. When
                None, a default :class:`VectorStore` is constructed
                using ``db_path``, the embedder's dimension, and the
                chunking config (recorded in the on-disk metadata table
                so reopens detect config-mismatch errors).
            embedder: Optional pre-built embedder conforming to
                :class:`EmbedderProtocol`. When supplied,
                ``embedding_model`` is ignored. Use this to plug in
                alternative embedding backends (OpenAI, sentence-
                transformers, fastembed, Voyage, Jina, ...) in place of
                the default llama.cpp ``Embedder``. When None, a default
                :class:`Embedder` is constructed from ``embedding_model``.
            **kwargs: Additional arguments passed to LLM
        """
        # Import LLM here to avoid circular imports
        from ..api import LLM

        self.embedding_model = embedding_model
        self.generation_model = generation_model

        # Initialize components. We pass the embedding-model path and
        # the chunking config down to VectorStore so they get recorded
        # in the on-disk metadata table. On reopen, VectorStore verifies
        # the caller's config matches what's already in the DB and
        # raises a friendly error on mismatch (different embedder,
        # different chunk size, etc.) instead of silently producing
        # garbage by mixing two configurations.
        if embedder is not None:
            self.embedder: EmbedderProtocol = embedder
        else:
            self.embedder = Embedder(embedding_model)
        if store is not None:
            self.store: VectorStoreProtocol = store
        else:
            self.store = SqliteVectorStore(
                dimension=self.embedder.dimension,
                db_path=db_path,
                embedding_model_path=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
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
    ) -> IndexResult:
        """Add text strings to the knowledge base.

        Each text is hashed (md5 of its UTF-8 bytes) and checked
        against the store's deduplication table. Texts whose hash
        already exists in the store are silently skipped -- their
        labels are returned in the :attr:`IndexResult.skipped_labels`
        list. New texts are split, embedded, and added under their
        content hash so future calls with the same text are
        deduplicated.

        If a text's content hash differs from a previously-indexed
        text with the same source label, ``ValueError`` is raised --
        this catches the "I changed the content but kept the label"
        case where appending would silently produce two versions in
        the index. The caller should rebuild the index in that case.

        Args:
            texts: List of text strings to add
            metadata: Optional metadata for each text
            split: Whether to split texts into chunks (default: True)

        Returns:
            IndexResult (subclass of ``list[int]``) containing the IDs
            of the newly inserted chunks. ``.skipped_labels`` lists the
            labels of texts that were already indexed.
        """
        self._check_closed()

        if metadata is not None and len(metadata) != len(texts):
            raise ValueError(f"metadata length ({len(metadata)}) must match texts length ({len(texts)})")

        added_ids: list[int] = []
        skipped_labels: list[str] = []

        for i, text in enumerate(texts):
            text_hash = _content_hash(text.encode("utf-8"))
            # Use the first 8 hex chars of the hash as a synthetic
            # label. add_texts callers don't have a meaningful name
            # for the text (unlike add_documents which uses the file
            # basename), so the hash prefix is the most stable
            # identifier we can produce.
            text_label = f"text:{text_hash[:8]}"

            if self.store.is_source_indexed(text_hash):
                skipped_labels.append(text_label)
                continue

            # Same label, different content -> caller is appending a
            # changed version of an existing source. We refuse rather
            # than silently producing two versions in the index.
            existing = self.store.get_source_by_label(text_label)
            if existing is not None and existing["content_hash"] != text_hash:
                raise ValueError(
                    f"Source label {text_label!r} is already in the store with "
                    f"a different content hash ({existing['content_hash']!r} vs "
                    f"{text_hash!r}). This shouldn't happen for add_texts since "
                    f"the label is derived from the content hash itself; if "
                    f"you're seeing this, the dedup table is corrupted."
                )

            # New text -- split, embed, add with the source hash so
            # future calls dedupe.
            text_meta = metadata[i] if metadata else {}
            chunks = self.splitter.split(text) if split else [text]
            chunk_metadata = []
            for j, _ in enumerate(chunks):
                m = text_meta.copy()
                m["chunk_index"] = j
                chunk_metadata.append(m)

            embeddings = self.embedder.embed_batch(chunks)
            ids = self.store.add(
                embeddings,
                chunks,
                chunk_metadata,
                source_hash=text_hash,
                source_label=text_label,
            )
            added_ids.extend(ids)

        return IndexResult(added_ids, skipped_labels)

    def add_documents(
        self,
        paths: list[str | Path],
        split: bool = True,
        **loader_kwargs: Any,
    ) -> IndexResult:
        """Load and add documents from files.

        Each file is hashed (md5 of its raw bytes) and checked against
        the store's deduplication table. Files whose hash already
        exists in the store are silently skipped -- their basenames
        are returned in the :attr:`IndexResult.skipped_labels` list.
        Re-running ``add_documents`` with the same files is therefore
        a no-op on the indexing side, which is what users expect when
        they re-launch a CLI session against an existing ``--db PATH``.

        If a file's content hash differs from a previously-indexed
        source with the same basename, ``ValueError`` is raised --
        this catches "I edited the file but kept the name" where
        appending would silently produce two versions in the index.
        The caller should rename the file or use ``--rebuild`` in
        that case.

        Args:
            paths: List of file paths to load
            split: Whether to split documents into chunks
            **loader_kwargs: Additional arguments passed to loaders

        Returns:
            IndexResult (subclass of ``list[int]``) containing the IDs
            of the newly inserted chunks across all newly-indexed
            files. ``.skipped_labels`` lists the basenames of files
            that were already indexed.
        """
        self._check_closed()

        added_ids: list[int] = []
        skipped_labels: list[str] = []

        for path in paths:
            path = Path(path)
            label = path.name

            # Hash the raw bytes of the file. This is what catches
            # "user edited the file" -- a single byte difference
            # produces a different md5. We hash the file content,
            # not the loaded Document text, because the loader may
            # be non-deterministic in subtle ways (whitespace
            # normalisation, encoding fixups) and we want the dedup
            # decision to be based on what the user actually has on
            # disk.
            try:
                file_bytes = path.read_bytes()
            except OSError as e:
                raise ValueError(f"Could not read {path} for hashing: {e}") from e

            file_hash = _content_hash(file_bytes)

            if self.store.is_source_indexed(file_hash):
                skipped_labels.append(label)
                continue

            # Same basename, different content -> file was edited.
            # Refuse rather than producing two versions of the same
            # logical source in the index.
            existing = self.store.get_source_by_label(label)
            if existing is not None and existing["content_hash"] != file_hash:
                raise ValueError(
                    f"File {label!r} is already in the store with a different "
                    f"content hash. The file content has changed since it was "
                    f"indexed. To append the new version, either rename the "
                    f"file (treat it as a new source) or pass --rebuild to "
                    f"recreate the index from scratch."
                )

            # New file. Load it (the loader may produce one or more
            # Document objects per file -- we collect all of them
            # under the same source hash so the dedup table sees the
            # file as a single unit).
            docs = load_document(path, **loader_kwargs)

            all_chunks: list[str] = []
            all_metadata: list[dict[str, Any]] = []
            for doc in docs:
                doc_meta = {"source": str(path), **doc.metadata}
                chunks = self.splitter.split(doc.text) if split else [doc.text]
                for j, chunk in enumerate(chunks):
                    m = doc_meta.copy()
                    m["chunk_index"] = j
                    all_chunks.append(chunk)
                    all_metadata.append(m)

            if not all_chunks:
                # Empty file produced no chunks. Still record it in
                # the dedup table so re-runs skip it instead of
                # repeatedly trying to load and produce nothing.
                # Use add() with empty lists -- the source row will
                # be inserted with chunk_count=0.
                self.store.add(
                    [],
                    [],
                    None,
                    source_hash=file_hash,
                    source_label=label,
                )
                continue

            embeddings = self.embedder.embed_batch(all_chunks)
            ids = self.store.add(
                embeddings,
                all_chunks,
                all_metadata,
                source_hash=file_hash,
                source_label=label,
            )
            added_ids.extend(ids)

        return IndexResult(added_ids, skipped_labels)

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
    ) -> list["SearchResult"]:
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
    ) -> list["SearchResult"]:
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

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self._closed else f"open, {self.count} docs"
        return (
            f"RAG(embedding_model={self.embedding_model!r}, "
            f"generation_model={self.generation_model!r}, "
            f"status={status})"
        )
