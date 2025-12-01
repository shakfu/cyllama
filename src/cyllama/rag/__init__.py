"""RAG (Retrieval-Augmented Generation) support for cyllama.

This module provides tools for building RAG pipelines using llama.cpp
for embeddings and generation, with sqlite-vector for vector storage.

Components:
    - Embedder: Generate text embeddings using llama.cpp embedding models
    - VectorStore: SQLite-based vector store using sqlite-vector
    - TextSplitter: Split documents into chunks for embedding
    - Document Loaders: Load documents from various file formats
    - RAGPipeline: Orchestrate retrieval and generation
    - RAG: High-level RAG interface with sensible defaults
    - AsyncRAG: Async wrapper for non-blocking RAG operations
    - HybridStore: Combined FTS5 + vector search for hybrid retrieval
    - Reranker: Cross-encoder reranking for improved quality
    - create_rag_tool: Create agent tools from RAG instances

Example:
    >>> from cyllama.rag import RAG, RAGConfig
    >>>
    >>> # High-level interface (recommended)
    >>> rag = RAG(
    ...     embedding_model="models/bge-small.gguf",
    ...     generation_model="models/llama.gguf"
    ... )
    >>> rag.add_texts(["Python is a programming language."])
    >>> response = rag.query("What is Python?")
    >>> print(response.text)
    >>>
    >>> # Async interface
    >>> from cyllama.rag import AsyncRAG
    >>> async with AsyncRAG(...) as rag:
    ...     await rag.add_texts(["Data"])
    ...     response = await rag.query("Question?")
    >>>
    >>> # Agent integration
    >>> from cyllama.rag import create_rag_tool
    >>> tool = create_rag_tool(rag)
    >>> # Use tool with ReActAgent, ConstrainedAgent, etc.
"""

from .advanced import (
    AsyncRAG,
    HybridStore,
    Reranker,
    async_search_knowledge,
    create_rag_tool,
)
from .embedder import Embedder, PoolingType
from .loaders import (
    BaseLoader,
    DirectoryLoader,
    JSONLoader,
    JSONLLoader,
    LoaderError,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
    load_directory,
    load_document,
)
from .pipeline import DEFAULT_PROMPT_TEMPLATE, RAGConfig, RAGPipeline, RAGResponse
from .rag import RAG
from .splitter import MarkdownSplitter, TextSplitter, TokenTextSplitter
from .store import VectorStore, VectorStoreError
from .types import Chunk, Document, EmbeddingResult, SearchResult

__all__ = [
    # High-level RAG Interface
    "RAG",
    "AsyncRAG",
    # RAG Pipeline
    "RAGPipeline",
    "RAGConfig",
    "RAGResponse",
    "DEFAULT_PROMPT_TEMPLATE",
    # Advanced Features
    "HybridStore",
    "Reranker",
    "create_rag_tool",
    "async_search_knowledge",
    # Embedder
    "Embedder",
    "PoolingType",
    # VectorStore
    "VectorStore",
    "VectorStoreError",
    # Text Splitters
    "TextSplitter",
    "TokenTextSplitter",
    "MarkdownSplitter",
    # Document Loaders
    "BaseLoader",
    "TextLoader",
    "MarkdownLoader",
    "JSONLoader",
    "JSONLLoader",
    "DirectoryLoader",
    "PDFLoader",
    "LoaderError",
    "load_document",
    "load_directory",
    # Types
    "Chunk",
    "Document",
    "EmbeddingResult",
    "SearchResult",
]
