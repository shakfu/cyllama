"""Advanced RAG features for cyllama.

This module provides:
- AsyncRAG: Async wrapper for non-blocking RAG operations
- RAGTool: Agent tool for RAG-based knowledge retrieval
- Reranker: Cross-encoder reranking for improved retrieval quality
- HybridStore: Combined FTS5 + vector search for hybrid retrieval
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterator

from .pipeline import RAGConfig, RAGResponse
from .types import SearchResult

if TYPE_CHECKING:
    from ..agents.tools import Tool
    from .embedder import Embedder
    from .rag import RAG
    from .store import VectorStore


class AsyncRAG:
    """Async wrapper around RAG for non-blocking operations.

    Provides an async interface to RAG operations by running blocking
    operations in a thread pool. Suitable for async web frameworks
    like FastAPI, aiohttp, etc.

    Note: The underlying RAG is synchronous - this wrapper moves
    blocking operations off the main event loop.

    Example:
        >>> async def main():
        ...     async with AsyncRAG(
        ...         embedding_model="models/bge-small.gguf",
        ...         generation_model="models/llama.gguf"
        ...     ) as rag:
        ...         await rag.add_texts(["Python is a programming language."])
        ...         response = await rag.query("What is Python?")
        ...         print(response.text)
        ...
        ...         # Async streaming
        ...         async for chunk in rag.stream("Explain more"):
        ...             print(chunk, end="")
        >>>
        >>> asyncio.run(main())
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
        """Initialize AsyncRAG with models.

        Args:
            embedding_model: Path to embedding model (GGUF file)
            generation_model: Path to generation model (GGUF file)
            chunk_size: Target chunk size for text splitting
            chunk_overlap: Overlap between chunks
            db_path: Path for vector store (":memory:" for in-memory)
            config: RAG configuration (uses defaults if None)
            **kwargs: Additional arguments passed to LLM
        """
        # Import here to avoid circular imports
        from .rag import RAG

        self._rag = RAG(
            embedding_model=embedding_model,
            generation_model=generation_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            db_path=db_path,
            config=config,
            **kwargs,
        )
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "AsyncRAG":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - ensures cleanup."""
        await self.close()

    async def close(self) -> None:
        """Close all resources."""
        await asyncio.to_thread(self._rag.close)

    async def add_texts(
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
        async with self._lock:
            return await asyncio.to_thread(
                self._rag.add_texts, texts, metadata, split
            )

    async def add_documents(
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
        async with self._lock:
            return await asyncio.to_thread(
                self._rag.add_documents, paths, split, **loader_kwargs
            )

    async def query(
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
        async with self._lock:
            return await asyncio.to_thread(
                self._rag.query, question, config
            )

    async def stream(
        self,
        question: str,
        config: RAGConfig | None = None,
    ) -> AsyncIterator[str]:
        """Stream response tokens for a question.

        Args:
            question: The question to answer
            config: Optional config override

        Yields:
            Response tokens as strings
        """
        queue: asyncio.Queue[str | None | Exception] = asyncio.Queue()

        async def producer():
            """Run sync generator in thread and put items in queue."""
            try:
                def generate_sync():
                    for chunk in self._rag.stream(question, config):
                        asyncio.run_coroutine_threadsafe(
                            queue.put(chunk),
                            loop
                        )
                    asyncio.run_coroutine_threadsafe(
                        queue.put(None),
                        loop
                    )

                await asyncio.to_thread(generate_sync)
            except Exception as e:
                await queue.put(e)

        loop = asyncio.get_event_loop()

        async with self._lock:
            producer_task = asyncio.create_task(producer())

            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                await producer_task

    async def retrieve(
        self,
        question: str,
        config: RAGConfig | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant documents without generation.

        Args:
            question: The question to retrieve documents for
            config: Optional config override

        Returns:
            List of relevant SearchResults
        """
        async with self._lock:
            return await asyncio.to_thread(
                self._rag.retrieve, question, config
            )

    async def search(
        self,
        query: str,
        k: int = 5,
        threshold: float | None = None,
    ) -> list[SearchResult]:
        """Direct vector search without RAG formatting.

        Args:
            query: Query text to search for
            k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of SearchResults
        """
        async with self._lock:
            return await asyncio.to_thread(
                self._rag.search, query, k, threshold
            )

    @property
    def count(self) -> int:
        """Return number of documents in the store."""
        return self._rag.count

    async def clear(self) -> int:
        """Clear all documents from the store.

        Returns:
            Number of documents removed
        """
        async with self._lock:
            return await asyncio.to_thread(self._rag.clear)

    def __repr__(self) -> str:
        return f"Async{self._rag!r}"


def create_rag_tool(
    rag: "RAG",
    name: str = "search_knowledge",
    description: str | None = None,
    top_k: int = 5,
    include_scores: bool = False,
) -> "Tool":
    """Create an agent tool for RAG-based knowledge retrieval.

    This function creates a Tool that can be used with cyllama agents
    (ReActAgent, ConstrainedAgent, etc.) to search a knowledge base.

    Args:
        rag: RAG instance to use for retrieval
        name: Tool name (default: "search_knowledge")
        description: Tool description (default: auto-generated)
        top_k: Number of results to retrieve
        include_scores: Whether to include similarity scores in output

    Returns:
        Tool instance for agent use

    Example:
        >>> from cyllama.rag import RAG, create_rag_tool
        >>> from cyllama.agents import ReActAgent
        >>>
        >>> rag = RAG(
        ...     embedding_model="models/bge-small.gguf",
        ...     generation_model="models/llama.gguf"
        ... )
        >>> rag.add_texts(["Python is a programming language."])
        >>>
        >>> tool = create_rag_tool(rag)
        >>> agent = ReActAgent(llm=llm, tools=[tool])
        >>> agent.run("What do you know about Python?")
    """
    from ..agents.tools import Tool

    if description is None:
        description = (
            "Search the knowledge base for relevant information. "
            "Use this tool when you need to find facts, context, or "
            "information about specific topics."
        )

    def search_knowledge(query: str) -> str:
        """Search the knowledge base for relevant information.

        Args:
            query: The search query to find relevant documents

        Returns:
            Formatted search results from the knowledge base
        """
        config = RAGConfig(top_k=top_k)
        results = rag.retrieve(query, config=config)

        if not results:
            return "No relevant information found in the knowledge base."

        formatted = []
        for i, result in enumerate(results, 1):
            if include_scores:
                formatted.append(f"[{i}] (score: {result.score:.3f}) {result.text}")
            else:
                formatted.append(f"[{i}] {result.text}")

        return "\n\n".join(formatted)

    # Build tool parameters schema
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant documents",
            }
        },
        "required": ["query"],
    }

    return Tool(
        name=name,
        description=description,
        func=search_knowledge,
        parameters=parameters,
    )


@dataclass
class Reranker:
    """Cross-encoder reranker for improved retrieval quality.

    Reranking uses a cross-encoder model to score query-document pairs
    more accurately than bi-encoder similarity. This is slower but
    produces better ranking, especially for top results.

    The reranker takes initial retrieval results and reorders them
    based on relevance scores from the cross-encoder.

    Example:
        >>> from cyllama.rag import VectorStore, Embedder, Reranker
        >>>
        >>> # Initial retrieval
        >>> embedder = Embedder("models/bge-small.gguf")
        >>> store = VectorStore(dimension=embedder.dimension)
        >>> results = store.search(embedder.embed("query"), k=20)
        >>>
        >>> # Rerank top results
        >>> reranker = Reranker("models/bge-reranker.gguf")
        >>> reranked = reranker.rerank("query", results, top_k=5)
    """

    model_path: str
    n_ctx: int = 512
    n_gpu_layers: int = -1
    _model: Any = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize the reranker model."""
        # Lazy load to avoid import errors if not needed
        self._model = None

    def _ensure_model(self):
        """Load model on first use."""
        if self._model is None:
            from ..llama import LlamaModel, LlamaContext, LlamaModelParams, LlamaContextParams

            params = LlamaModelParams()
            params.n_gpu_layers = self.n_gpu_layers

            self._model = LlamaModel(self.model_path, params)
            ctx_params = LlamaContextParams()
            ctx_params.n_ctx = self.n_ctx
            self._ctx = LlamaContext(self._model, ctx_params)

    def score(self, query: str, document: str) -> float:
        """Score a single query-document pair.

        Args:
            query: Query text
            document: Document text

        Returns:
            Relevance score (higher is more relevant)
        """
        self._ensure_model()

        # Format as query-document pair for cross-encoder
        # Most cross-encoders use this format
        text = f"query: {query}\ndocument: {document}"

        # Tokenize
        tokens = self._model.tokenize(text.encode(), add_bos=True, special=True)

        # Truncate if needed
        if len(tokens) > self.n_ctx:
            tokens = tokens[: self.n_ctx]

        # Get embeddings (cross-encoders output a score)
        from ..llama import LlamaBatch, common_batch_add

        batch = LlamaBatch(n_tokens=len(tokens), embd=0, n_seq_max=1)
        for i, token in enumerate(tokens):
            common_batch_add(batch, token, i, [0], i == len(tokens) - 1)

        self._ctx.decode(batch)

        # Get the score from the last token's logits
        # This assumes a standard cross-encoder architecture
        logits = self._ctx.get_logits()
        if logits is not None and len(logits) > 0:
            # Return the first logit as score (common for binary classifiers)
            return float(logits[0])
        return 0.0

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Rerank search results using cross-encoder scores.

        Args:
            query: Original query
            results: Initial search results to rerank
            top_k: Number of top results to return (None = all)

        Returns:
            Reranked list of SearchResults with updated scores
        """
        if not results:
            return []

        # Score all documents
        scored = []
        for result in results:
            score = self.score(query, result.text)
            scored.append((score, result))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Apply top_k limit
        if top_k is not None:
            scored = scored[:top_k]

        # Return results with updated scores
        return [
            SearchResult(
                id=result.id,
                text=result.text,
                score=score,
                metadata=result.metadata,
            )
            for score, result in scored
        ]

    def close(self) -> None:
        """Release model resources."""
        if self._model is not None:
            self._ctx = None
            self._model = None

    def __enter__(self) -> "Reranker":
        return self

    def __exit__(self, *args) -> None:
        self.close()


class HybridStore:
    """Hybrid search combining FTS5 full-text search with vector similarity.

    This store uses SQLite's FTS5 for keyword-based search and sqlite-vector
    for semantic search, then combines results using reciprocal rank fusion
    or weighted scoring.

    Hybrid search is useful when:
    - Exact keyword matches are important (names, codes, IDs)
    - You want to combine semantic understanding with lexical matching
    - Pure vector search misses important keyword matches

    Example:
        >>> from cyllama.rag import HybridStore, Embedder
        >>>
        >>> embedder = Embedder("models/bge-small.gguf")
        >>> store = HybridStore(dimension=embedder.dimension)
        >>>
        >>> # Add documents
        >>> embeddings = embedder.embed_batch(["Python guide", "Java tutorial"])
        >>> store.add(embeddings, ["Python programming guide", "Java tutorial"])
        >>>
        >>> # Hybrid search
        >>> query_emb = embedder.embed("programming")
        >>> results = store.search(query_emb, query_text="Python", k=5)
    """

    def __init__(
        self,
        dimension: int,
        db_path: str = ":memory:",
        table_name: str = "embeddings",
        metric: str = "cosine",
        vector_type: str = "float32",
        alpha: float = 0.5,
    ):
        """Initialize hybrid store.

        Args:
            dimension: Vector dimension
            db_path: SQLite database path
            table_name: Base table name
            metric: Vector distance metric
            vector_type: Vector storage type
            alpha: Weight for vector search (1-alpha for FTS)
        """
        from .store import VectorStore

        self.dimension = dimension
        self.db_path = db_path
        self.table_name = table_name
        self.alpha = alpha
        self._closed = False

        # Create vector store
        self._vector_store = VectorStore(
            dimension=dimension,
            db_path=db_path,
            table_name=table_name,
            metric=metric,
            vector_type=vector_type,
        )

        # Create FTS5 table
        self._create_fts_table()

    def _create_fts_table(self) -> None:
        """Create FTS5 virtual table for full-text search."""
        fts_table = f"{self.table_name}_fts"

        # Create FTS5 table if it doesn't exist
        self._vector_store.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table}
            USING fts5(
                text,
                content='{self.table_name}',
                content_rowid='id'
            )
        """)

        # Create triggers to keep FTS in sync
        self._vector_store.conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {self.table_name}_ai
            AFTER INSERT ON {self.table_name}
            BEGIN
                INSERT INTO {fts_table}(rowid, text) VALUES (new.id, new.text);
            END
        """)

        self._vector_store.conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {self.table_name}_ad
            AFTER DELETE ON {self.table_name}
            BEGIN
                INSERT INTO {fts_table}({fts_table}, rowid, text)
                VALUES ('delete', old.id, old.text);
            END
        """)

        self._vector_store.conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {self.table_name}_au
            AFTER UPDATE ON {self.table_name}
            BEGIN
                INSERT INTO {fts_table}({fts_table}, rowid, text)
                VALUES ('delete', old.id, old.text);
                INSERT INTO {fts_table}(rowid, text) VALUES (new.id, new.text);
            END
        """)

        self._vector_store.conn.commit()

    def add(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Add embeddings with texts (FTS is auto-populated via trigger).

        Args:
            embeddings: Vector embeddings
            texts: Associated text content
            metadata: Optional metadata for each embedding

        Returns:
            List of IDs for added items
        """
        self._check_closed()
        return self._vector_store.add(embeddings, texts, metadata)

    def search(
        self,
        query_embedding: list[float],
        query_text: str | None = None,
        k: int = 5,
        threshold: float | None = None,
        alpha: float | None = None,
    ) -> list[SearchResult]:
        """Hybrid search combining vector and full-text search.

        Args:
            query_embedding: Query vector for semantic search
            query_text: Query text for FTS search (if None, vector-only)
            k: Number of results to return
            threshold: Minimum similarity threshold
            alpha: Weight for vector search (overrides instance alpha)

        Returns:
            List of SearchResults with combined scores
        """
        self._check_closed()
        alpha = alpha if alpha is not None else self.alpha

        # Get vector search results
        vector_results = self._vector_store.search(
            query_embedding, k=k * 2, threshold=threshold
        )

        # If no query text, return vector results only
        if not query_text:
            return vector_results[:k]

        # Get FTS results
        fts_results = self._fts_search(query_text, k=k * 2)

        # Combine using reciprocal rank fusion
        return self._reciprocal_rank_fusion(
            vector_results, fts_results, k=k, alpha=alpha
        )

    def _fts_search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Search using FTS5.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of SearchResults from FTS
        """
        import json

        fts_table = f"{self.table_name}_fts"

        # Escape the query for FTS5: wrap each word in double quotes to handle
        # special characters like hyphens (e.g., "PEP-8" becomes '"PEP-8"')
        # This treats each word as a literal phrase
        words = query.split()
        escaped_query = " ".join(f'"{word}"' for word in words)

        # Use FTS5 MATCH with BM25 scoring
        cursor = self._vector_store.conn.execute(
            f"""
            SELECT e.id, e.text, e.metadata, bm25({fts_table}) as score
            FROM {fts_table}
            JOIN {self.table_name} e ON {fts_table}.rowid = e.id
            WHERE {fts_table} MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (escaped_query, k),
        )

        results = []
        for row in cursor:
            id_, text, meta_json, score = row
            # BM25 returns negative scores (lower is better)
            # Convert to positive similarity score
            results.append(
                SearchResult(
                    id=str(id_),
                    text=text,
                    score=-score,  # Negate so higher is better
                    metadata=json.loads(meta_json) if meta_json else {},
                )
            )
        return results

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[SearchResult],
        fts_results: list[SearchResult],
        k: int,
        alpha: float,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """Combine results using reciprocal rank fusion.

        Args:
            vector_results: Results from vector search
            fts_results: Results from FTS search
            k: Number of final results
            alpha: Weight for vector scores
            rrf_k: RRF constant (default 60)

        Returns:
            Combined and reranked results
        """
        scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}

        # Score vector results
        for rank, result in enumerate(vector_results):
            rrf_score = alpha / (rrf_k + rank + 1)
            scores[result.id] = scores.get(result.id, 0) + rrf_score
            result_map[result.id] = result

        # Score FTS results
        for rank, result in enumerate(fts_results):
            rrf_score = (1 - alpha) / (rrf_k + rank + 1)
            scores[result.id] = scores.get(result.id, 0) + rrf_score
            if result.id not in result_map:
                result_map[result.id] = result

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Return top k with updated scores
        return [
            SearchResult(
                id=id_,
                text=result_map[id_].text,
                score=scores[id_],
                metadata=result_map[id_].metadata,
            )
            for id_ in sorted_ids[:k]
        ]

    def delete(self, ids: list[str]) -> int:
        """Delete embeddings by ID.

        Args:
            ids: List of IDs to delete

        Returns:
            Number of items deleted
        """
        self._check_closed()
        return self._vector_store.delete(ids)

    def clear(self) -> int:
        """Clear all data from the store.

        Returns:
            Number of items cleared
        """
        self._check_closed()
        count = self._vector_store.clear()

        # Clear FTS table
        fts_table = f"{self.table_name}_fts"
        self._vector_store.conn.execute(f"DELETE FROM {fts_table}")
        self._vector_store.conn.commit()

        return count

    def _check_closed(self) -> None:
        """Raise error if store is closed."""
        if self._closed:
            raise RuntimeError("HybridStore is closed")

    def close(self) -> None:
        """Close the store and release resources."""
        if not self._closed:
            self._vector_store.close()
            self._closed = True

    def __len__(self) -> int:
        return len(self._vector_store)

    def __enter__(self) -> "HybridStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self._closed else f"open, {len(self)} docs"
        return f"HybridStore(dimension={self.dimension}, alpha={self.alpha}, {status})"


# Convenience function for creating async RAG tool
async def async_search_knowledge(
    rag: AsyncRAG,
    query: str,
    top_k: int = 5,
) -> str:
    """Async helper for searching knowledge base.

    Args:
        rag: AsyncRAG instance
        query: Search query
        top_k: Number of results

    Returns:
        Formatted search results
    """
    config = RAGConfig(top_k=top_k)
    results = await rag.retrieve(query, config=config)

    if not results:
        return "No relevant information found."

    return "\n\n".join(f"[{i}] {r.text}" for i, r in enumerate(results, 1))
