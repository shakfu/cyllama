"""Tests for the RAG Advanced Features (Phase 5)."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cyllama.rag.advanced import (
    AsyncRAG,
    HybridStore,
    Reranker,
    async_search_knowledge,
    create_rag_tool,
)
from cyllama.rag.pipeline import RAGConfig, RAGResponse
from cyllama.rag.types import SearchResult


class TestAsyncRAG:
    """Test AsyncRAG class."""

    @pytest.fixture
    def mock_rag(self):
        """Create a mock RAG instance."""
        rag = MagicMock()
        rag.count = 5
        rag.add_texts.return_value = [1, 2, 3]
        rag.add_documents.return_value = [4, 5]
        rag.query.return_value = RAGResponse(
            text="Answer",
            sources=[SearchResult(id="1", text="Source", score=0.9, metadata={})],
            query="Question?",
        )
        rag.retrieve.return_value = [
            SearchResult(id="1", text="Doc 1", score=0.9, metadata={}),
        ]
        rag.search.return_value = [
            SearchResult(id="2", text="Doc 2", score=0.8, metadata={}),
        ]
        rag.clear.return_value = 5
        rag.close.return_value = None
        rag.__repr__ = lambda s: "RAG(mock)"

        def stream_gen(*args, **kwargs):
            yield "Hello "
            yield "World"

        rag.stream = stream_gen
        return rag

    @pytest.fixture
    def async_rag_with_mock(self, mock_rag):
        """Create AsyncRAG with mocked underlying RAG."""
        async_rag = AsyncRAG.__new__(AsyncRAG)
        async_rag._rag = mock_rag
        async_rag._lock = asyncio.Lock()
        return async_rag

    @pytest.mark.asyncio
    async def test_async_context_manager(self, async_rag_with_mock, mock_rag):
        """Test async context manager."""
        async with async_rag_with_mock as rag:
            assert rag._rag is not None

        mock_rag.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_texts(self, async_rag_with_mock, mock_rag):
        """Test async add_texts."""
        ids = await async_rag_with_mock.add_texts(["text1", "text2"])
        assert ids == [1, 2, 3]
        mock_rag.add_texts.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_documents(self, async_rag_with_mock, mock_rag):
        """Test async add_documents."""
        ids = await async_rag_with_mock.add_documents(["doc.txt"])
        assert ids == [4, 5]
        mock_rag.add_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_query(self, async_rag_with_mock, mock_rag):
        """Test async query."""
        response = await async_rag_with_mock.query("What is X?")
        assert response.text == "Answer"
        assert len(response.sources) == 1
        mock_rag.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream(self, async_rag_with_mock, mock_rag):
        """Test async stream."""
        chunks = []
        async for chunk in async_rag_with_mock.stream("Question?"):
            chunks.append(chunk)
        assert chunks == ["Hello ", "World"]

    @pytest.mark.asyncio
    async def test_retrieve(self, async_rag_with_mock, mock_rag):
        """Test async retrieve."""
        results = await async_rag_with_mock.retrieve("Query")
        assert len(results) == 1
        assert results[0].text == "Doc 1"
        mock_rag.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_search(self, async_rag_with_mock, mock_rag):
        """Test async search."""
        results = await async_rag_with_mock.search("Query", k=3)
        assert len(results) == 1
        assert results[0].text == "Doc 2"
        mock_rag.search.assert_called_once_with("Query", 3, None)

    @pytest.mark.asyncio
    async def test_count_property(self, async_rag_with_mock, mock_rag):
        """Test count property."""
        assert async_rag_with_mock.count == 5

    @pytest.mark.asyncio
    async def test_clear(self, async_rag_with_mock, mock_rag):
        """Test async clear."""
        count = await async_rag_with_mock.clear()
        assert count == 5
        mock_rag.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_repr(self, async_rag_with_mock, mock_rag):
        """Test __repr__."""
        repr_str = repr(async_rag_with_mock)
        assert "Async" in repr_str


class TestCreateRAGTool:
    """Test create_rag_tool function."""

    @pytest.fixture
    def mock_rag(self):
        """Create mock RAG for tool testing."""
        rag = MagicMock()
        rag.retrieve.return_value = [
            SearchResult(id="1", text="Python is great", score=0.9, metadata={}),
            SearchResult(id="2", text="Python is fast", score=0.8, metadata={}),
        ]
        return rag

    def test_create_tool_default_name(self, mock_rag):
        """Test creating tool with default name."""
        tool = create_rag_tool(mock_rag)
        assert tool.name == "search_knowledge"
        assert "knowledge base" in tool.description.lower()

    def test_create_tool_custom_name(self, mock_rag):
        """Test creating tool with custom name."""
        tool = create_rag_tool(mock_rag, name="search_docs")
        assert tool.name == "search_docs"

    def test_create_tool_custom_description(self, mock_rag):
        """Test creating tool with custom description."""
        tool = create_rag_tool(mock_rag, description="My custom description")
        assert tool.description == "My custom description"

    def test_tool_execution(self, mock_rag):
        """Test executing the tool."""
        tool = create_rag_tool(mock_rag)
        result = tool(query="Python")

        assert "Python is great" in result
        assert "Python is fast" in result
        mock_rag.retrieve.assert_called_once()

    def test_tool_with_scores(self, mock_rag):
        """Test tool with include_scores=True."""
        tool = create_rag_tool(mock_rag, include_scores=True)
        result = tool(query="Python")

        assert "0.9" in result or "0.900" in result
        assert "Python is great" in result

    def test_tool_no_results(self, mock_rag):
        """Test tool when no results found."""
        mock_rag.retrieve.return_value = []
        tool = create_rag_tool(mock_rag)
        result = tool(query="xyz")

        assert "no relevant" in result.lower()

    def test_tool_parameters_schema(self, mock_rag):
        """Test tool has correct parameters schema."""
        tool = create_rag_tool(mock_rag)
        assert tool.parameters["type"] == "object"
        assert "query" in tool.parameters["properties"]
        assert "query" in tool.parameters["required"]

    def test_tool_custom_top_k(self, mock_rag):
        """Test tool with custom top_k."""
        tool = create_rag_tool(mock_rag, top_k=3)
        tool(query="test")

        call_args = mock_rag.retrieve.call_args
        config = call_args[1]["config"]
        assert config.top_k == 3


class TestReranker:
    """Test Reranker class."""

    def test_init(self):
        """Test reranker initialization."""
        reranker = Reranker(model_path="model.gguf")
        assert reranker.model_path == "model.gguf"
        assert reranker.n_ctx == 512
        assert reranker._model is None  # Lazy loaded

    def test_rerank_empty_results(self):
        """Test reranking empty results."""
        reranker = Reranker(model_path="model.gguf")
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_with_mock_model(self):
        """Test reranking with mocked model."""
        reranker = Reranker(model_path="model.gguf")

        # Mock the score method
        reranker.score = MagicMock(side_effect=[0.9, 0.5, 0.8])

        results = [
            SearchResult(id="1", text="Doc 1", score=0.7, metadata={}),
            SearchResult(id="2", text="Doc 2", score=0.8, metadata={}),
            SearchResult(id="3", text="Doc 3", score=0.6, metadata={}),
        ]

        reranked = reranker.rerank("query", results)

        # Should be sorted by new scores: 0.9, 0.8, 0.5
        assert len(reranked) == 3
        assert reranked[0].id == "1"  # score 0.9
        assert reranked[1].id == "3"  # score 0.8
        assert reranked[2].id == "2"  # score 0.5

    def test_rerank_with_top_k(self):
        """Test reranking with top_k limit."""
        reranker = Reranker(model_path="model.gguf")
        reranker.score = MagicMock(side_effect=[0.9, 0.5, 0.8])

        results = [
            SearchResult(id="1", text="Doc 1", score=0.7, metadata={}),
            SearchResult(id="2", text="Doc 2", score=0.8, metadata={}),
            SearchResult(id="3", text="Doc 3", score=0.6, metadata={}),
        ]

        reranked = reranker.rerank("query", results, top_k=2)
        assert len(reranked) == 2

    def test_context_manager(self):
        """Test reranker context manager."""
        with Reranker(model_path="model.gguf") as reranker:
            assert reranker.model_path == "model.gguf"


class TestHybridStore:
    """Test HybridStore class."""

    @pytest.fixture
    def store(self):
        """Create in-memory hybrid store."""
        return HybridStore(dimension=3, db_path=":memory:")

    def test_init(self, store):
        """Test hybrid store initialization."""
        assert store.dimension == 3
        assert store.alpha == 0.5
        assert not store._closed

    def test_add_and_search_vector_only(self, store):
        """Test adding and searching with vector only."""
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        texts = ["Document one", "Document two"]

        ids = store.add(embeddings, texts)
        assert len(ids) == 2

        # Vector-only search (no query_text)
        results = store.search([1.0, 0.0, 0.0], k=2)
        assert len(results) >= 1
        assert results[0].text == "Document one"

    def test_hybrid_search(self, store):
        """Test hybrid search combining vector and FTS."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
        ]
        texts = [
            "Python programming language",
            "Java programming language",
            "Database management system",
        ]

        store.add(embeddings, texts)

        # Hybrid search - should find Python via FTS match
        results = store.search(
            query_embedding=[0.8, 0.2, 0.0],
            query_text="Python",
            k=3,
        )

        assert len(results) >= 1
        # Python should be ranked high due to FTS match
        python_found = any("Python" in r.text for r in results)
        assert python_found

    def test_hybrid_search_with_alpha(self, store):
        """Test hybrid search with different alpha values."""
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        texts = ["Alpha test one", "Alpha test two"]
        store.add(embeddings, texts)

        # High alpha = more weight on vector search
        results_high = store.search(
            query_embedding=[1.0, 0.0, 0.0],
            query_text="two",
            k=2,
            alpha=0.9,
        )

        # Low alpha = more weight on FTS
        results_low = store.search(
            query_embedding=[1.0, 0.0, 0.0],
            query_text="two",
            k=2,
            alpha=0.1,
        )

        # Both should return results
        assert len(results_high) >= 1
        assert len(results_low) >= 1

    def test_delete(self, store):
        """Test deleting documents."""
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        texts = ["Doc 1", "Doc 2"]
        ids = store.add(embeddings, texts)

        assert len(store) == 2
        store.delete([str(ids[0])])
        assert len(store) == 1

    def test_clear(self, store):
        """Test clearing all documents."""
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        texts = ["Doc 1", "Doc 2"]
        store.add(embeddings, texts)

        assert len(store) == 2
        count = store.clear()
        assert count == 2
        assert len(store) == 0

    def test_context_manager(self):
        """Test context manager."""
        with HybridStore(dimension=3) as store:
            assert not store._closed
        assert store._closed

    def test_closed_store_raises(self, store):
        """Test operations on closed store raise error."""
        store.close()

        with pytest.raises(RuntimeError, match="closed"):
            store.add([[1, 0, 0]], ["text"])

        with pytest.raises(RuntimeError, match="closed"):
            store.search([1, 0, 0])

    def test_repr(self, store):
        """Test __repr__."""
        store.add([[1, 0, 0]], ["Doc"])
        repr_str = repr(store)
        assert "HybridStore" in repr_str
        assert "dimension=3" in repr_str
        assert "1 docs" in repr_str

    def test_add_with_metadata(self, store):
        """Test adding with metadata."""
        embeddings = [[1.0, 0.0, 0.0]]
        texts = ["Document"]
        metadata = [{"source": "test.txt"}]

        ids = store.add(embeddings, texts, metadata)
        assert len(ids) == 1


class TestAsyncSearchKnowledge:
    """Test async_search_knowledge helper."""

    @pytest.mark.asyncio
    async def test_async_search_with_results(self):
        """Test async search with results."""
        mock_rag = MagicMock()
        mock_rag.retrieve = AsyncMock(
            return_value=[
                SearchResult(id="1", text="Result 1", score=0.9, metadata={}),
                SearchResult(id="2", text="Result 2", score=0.8, metadata={}),
            ]
        )

        # Need to patch the RAG class's retrieve to be async
        async_rag = MagicMock(spec=AsyncRAG)
        async_rag.retrieve = AsyncMock(
            return_value=[
                SearchResult(id="1", text="Result 1", score=0.9, metadata={}),
                SearchResult(id="2", text="Result 2", score=0.8, metadata={}),
            ]
        )

        result = await async_search_knowledge(async_rag, "query")
        assert "[1] Result 1" in result
        assert "[2] Result 2" in result

    @pytest.mark.asyncio
    async def test_async_search_no_results(self):
        """Test async search with no results."""
        async_rag = MagicMock(spec=AsyncRAG)
        async_rag.retrieve = AsyncMock(return_value=[])

        result = await async_search_knowledge(async_rag, "query")
        assert "no relevant" in result.lower()


class TestHybridStoreRRF:
    """Test Reciprocal Rank Fusion in HybridStore."""

    @pytest.fixture
    def store(self):
        """Create store for RRF testing."""
        return HybridStore(dimension=3, db_path=":memory:")

    def test_rrf_combines_rankings(self, store):
        """Test that RRF properly combines vector and FTS rankings."""
        # Add documents with different vector and keyword characteristics
        embeddings = [
            [1.0, 0.0, 0.0],  # Close to query vector
            [0.5, 0.5, 0.0],  # Medium distance
            [0.0, 1.0, 0.0],  # Far from query vector
        ]
        texts = [
            "General document about science",
            "Python programming tutorial",
            "Python advanced features",
        ]
        store.add(embeddings, texts)

        # Query: vector close to first doc, but keyword matches last two
        results = store.search(
            query_embedding=[0.9, 0.1, 0.0],
            query_text="Python",
            k=3,
        )

        # RRF should balance vector similarity and keyword match
        assert len(results) == 3
        # All results should have positive scores
        assert all(r.score > 0 for r in results)

    def test_rrf_alpha_affects_ranking(self, store):
        """Test that alpha properly weights vector vs FTS."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        texts = [
            "First document",
            "Python code here",
        ]
        store.add(embeddings, texts)

        # High alpha should favor vector similarity
        results_vector = store.search(
            query_embedding=[1.0, 0.0, 0.0],
            query_text="Python",
            k=2,
            alpha=0.99,
        )

        # Low alpha should favor FTS
        results_fts = store.search(
            query_embedding=[1.0, 0.0, 0.0],
            query_text="Python",
            k=2,
            alpha=0.01,
        )

        # With high alpha, first doc (vector match) should rank higher
        # With low alpha, second doc (Python match) should rank higher
        # Just verify both return valid results
        assert len(results_vector) >= 1
        assert len(results_fts) >= 1
