"""RAG Pipeline for combining retrieval and generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

from .types import SearchResult

if TYPE_CHECKING:
    from ..api import GenerationStats, LLM
    from .embedder import Embedder
    from .store import VectorStore


# Default prompt template for RAG queries
DEFAULT_PROMPT_TEMPLATE = """Use the following context to answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline.

    Attributes:
        top_k: Number of documents to retrieve (default: 5)
        similarity_threshold: Minimum similarity score for retrieval (default: None)
        max_tokens: Maximum tokens to generate (default: 512)
        temperature: Generation temperature (default: 0.7)
        prompt_template: Template for formatting the RAG prompt
        context_separator: String to join retrieved documents (default: "\\n\\n")
        include_metadata: Whether to include metadata in context (default: False)
    """

    # Retrieval settings
    top_k: int = 5
    similarity_threshold: float | None = None

    # Generation settings
    max_tokens: int = 512
    temperature: float = 0.7

    # Prompt template
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE

    # Context formatting
    context_separator: str = "\n\n"
    include_metadata: bool = False

    def __post_init__(self):
        """Validate configuration values."""
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.similarity_threshold is not None and not 0 <= self.similarity_threshold <= 1:
            raise ValueError(
                f"similarity_threshold must be between 0 and 1, got {self.similarity_threshold}"
            )
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")


@dataclass
class RAGResponse:
    """Response from a RAG query.

    Attributes:
        text: Generated response text
        sources: Retrieved documents used as context
        stats: Optional generation statistics
        query: Original query string
    """

    text: str
    sources: list[SearchResult]
    stats: Any | None = None  # GenerationStats when available
    query: str = ""

    def __str__(self) -> str:
        """Return the response text."""
        return self.text

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "text": self.text,
            "query": self.query,
            "sources": [
                {
                    "id": s.id,
                    "text": s.text,
                    "score": s.score,
                    "metadata": s.metadata,
                }
                for s in self.sources
            ],
        }
        if self.stats is not None:
            result["stats"] = {
                "prompt_tokens": self.stats.prompt_tokens,
                "generated_tokens": self.stats.generated_tokens,
                "total_time": self.stats.total_time,
                "tokens_per_second": self.stats.tokens_per_second,
            }
        return result


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation.

    The RAGPipeline orchestrates the retrieval-augmented generation process:
    1. Embed the user's question
    2. Retrieve relevant documents from the vector store
    3. Format a prompt with the retrieved context
    4. Generate a response using the LLM

    Example:
        >>> from cyllama import LLM
        >>> from cyllama.rag import Embedder, VectorStore, RAGPipeline
        >>>
        >>> embedder = Embedder("models/bge-small.gguf")
        >>> store = VectorStore(dimension=embedder.dimension)
        >>> llm = LLM("models/llama.gguf")
        >>>
        >>> # Add some documents
        >>> docs = ["Python is a programming language.", "The sky is blue."]
        >>> embeddings = embedder.embed_batch(docs)
        >>> store.add(embeddings, docs)
        >>>
        >>> # Query
        >>> pipeline = RAGPipeline(embedder, store, llm)
        >>> response = pipeline.query("What is Python?")
        >>> print(response.text)
    """

    def __init__(
        self,
        embedder: "Embedder",
        store: "VectorStore",
        generator: "LLM",
        config: RAGConfig | None = None,
    ):
        """Initialize RAG pipeline.

        Args:
            embedder: Embedder for converting queries to vectors
            store: VectorStore for similarity search
            generator: LLM for generating responses
            config: RAG configuration (uses defaults if None)
        """
        self.embedder = embedder
        self.store = store
        self.generator = generator
        self.config = config or RAGConfig()

    def query(
        self,
        question: str,
        config: RAGConfig | None = None,
    ) -> RAGResponse:
        """Answer a question using RAG.

        Steps:
        1. Embed the question
        2. Retrieve relevant documents
        3. Format prompt with context
        4. Generate response

        Args:
            question: The question to answer
            config: Optional config override for this query

        Returns:
            RAGResponse with generated text and sources
        """
        cfg = config or self.config

        # 1. Embed the question
        query_embedding = self.embedder.embed(question)

        # 2. Retrieve relevant documents
        sources = self.store.search(
            query_embedding,
            k=cfg.top_k,
            threshold=cfg.similarity_threshold,
        )

        # 3. Format the prompt
        prompt = self._format_prompt(question, sources, cfg)

        # 4. Generate response
        # Import here to avoid circular imports
        from ..api import GenerationConfig

        gen_config = GenerationConfig(
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
        response = self.generator(prompt, config=gen_config)

        return RAGResponse(
            text=str(response),
            sources=sources,
            stats=getattr(response, "stats", None),
            query=question,
        )

    def stream(
        self,
        question: str,
        config: RAGConfig | None = None,
    ) -> Iterator[str]:
        """Stream response tokens for a question.

        Yields tokens as they are generated, useful for real-time display.

        Args:
            question: The question to answer
            config: Optional config override for this query

        Yields:
            Response tokens as strings
        """
        cfg = config or self.config

        # 1. Embed the question
        query_embedding = self.embedder.embed(question)

        # 2. Retrieve relevant documents
        sources = self.store.search(
            query_embedding,
            k=cfg.top_k,
            threshold=cfg.similarity_threshold,
        )

        # 3. Format the prompt
        prompt = self._format_prompt(question, sources, cfg)

        # 4. Stream response
        # Import here to avoid circular imports
        from ..api import GenerationConfig

        gen_config = GenerationConfig(
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
        yield from self.generator(prompt, config=gen_config, stream=True)

    def retrieve(
        self,
        question: str,
        config: RAGConfig | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant documents without generation.

        Useful for debugging or when you only need retrieval.

        Args:
            question: The question to retrieve documents for
            config: Optional config override

        Returns:
            List of relevant SearchResults
        """
        cfg = config or self.config
        query_embedding = self.embedder.embed(question)
        return self.store.search(
            query_embedding,
            k=cfg.top_k,
            threshold=cfg.similarity_threshold,
        )

    def _format_prompt(
        self,
        question: str,
        sources: list[SearchResult],
        config: RAGConfig,
    ) -> str:
        """Format the RAG prompt with retrieved context.

        Args:
            question: User's question
            sources: Retrieved documents
            config: Configuration for formatting

        Returns:
            Formatted prompt string
        """
        # Build context from sources
        context_parts = []
        for source in sources:
            if config.include_metadata and source.metadata:
                meta_str = ", ".join(f"{k}: {v}" for k, v in source.metadata.items())
                context_parts.append(f"[{meta_str}]\n{source.text}")
            else:
                context_parts.append(source.text)

        context = config.context_separator.join(context_parts)

        # Format the template
        return config.prompt_template.format(
            context=context,
            question=question,
        )

    def __repr__(self) -> str:
        return (
            f"RAGPipeline(embedder={self.embedder!r}, "
            f"store={self.store!r}, config={self.config!r})"
        )
