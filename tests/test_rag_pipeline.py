"""Tests for the RAG Pipeline classes."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cyllama.rag.pipeline import (
    DEFAULT_PROMPT_TEMPLATE,
    RAGConfig,
    RAGPipeline,
    RAGResponse,
)
from cyllama.rag.types import SearchResult


class TestRAGConfig:
    """Test RAGConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RAGConfig()
        assert config.top_k == 5
        assert config.similarity_threshold is None
        assert config.max_tokens == 512
        assert config.temperature == 0.7
        assert config.prompt_template == DEFAULT_PROMPT_TEMPLATE
        assert config.context_separator == "\n\n"
        assert config.include_metadata is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RAGConfig(
            top_k=10,
            similarity_threshold=0.8,
            max_tokens=256,
            temperature=0.5,
            context_separator="---",
            include_metadata=True,
        )
        assert config.top_k == 10
        assert config.similarity_threshold == 0.8
        assert config.max_tokens == 256
        assert config.temperature == 0.5
        assert config.context_separator == "---"
        assert config.include_metadata is True

    def test_custom_prompt_template(self):
        """Test custom prompt template."""
        template = "Context: {context}\n\nQ: {question}\n\nA:"
        config = RAGConfig(prompt_template=template)
        assert config.prompt_template == template

    def test_invalid_top_k(self):
        """Test that invalid top_k raises error."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            RAGConfig(top_k=0)

    def test_invalid_similarity_threshold(self):
        """Test that invalid similarity_threshold raises error."""
        with pytest.raises(ValueError, match="similarity_threshold must be between"):
            RAGConfig(similarity_threshold=1.5)
        with pytest.raises(ValueError, match="similarity_threshold must be between"):
            RAGConfig(similarity_threshold=-0.1)

    def test_invalid_max_tokens(self):
        """Test that invalid max_tokens raises error."""
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            RAGConfig(max_tokens=0)

    def test_invalid_temperature(self):
        """Test that invalid temperature raises error."""
        with pytest.raises(ValueError, match="temperature must be >= 0"):
            RAGConfig(temperature=-0.1)


class TestRAGResponse:
    """Test RAGResponse class."""

    def test_basic_response(self):
        """Test basic response creation."""
        response = RAGResponse(
            text="The answer is 42.",
            sources=[],
        )
        assert response.text == "The answer is 42."
        assert response.sources == []
        assert response.stats is None
        assert response.query == ""

    def test_response_with_sources(self):
        """Test response with sources."""
        sources = [
            SearchResult(id="1", text="Doc 1", score=0.9, metadata={}),
            SearchResult(id="2", text="Doc 2", score=0.8, metadata={}),
        ]
        response = RAGResponse(
            text="Answer",
            sources=sources,
            query="What is life?",
        )
        assert len(response.sources) == 2
        assert response.query == "What is life?"

    def test_response_str(self):
        """Test __str__ returns text."""
        response = RAGResponse(text="Hello world", sources=[])
        assert str(response) == "Hello world"

    def test_response_to_dict(self):
        """Test to_dict conversion."""
        sources = [
            SearchResult(id="1", text="Doc 1", score=0.9, metadata={"key": "val"}),
        ]
        response = RAGResponse(
            text="Answer text",
            sources=sources,
            query="Question?",
        )
        d = response.to_dict()
        assert d["text"] == "Answer text"
        assert d["query"] == "Question?"
        assert len(d["sources"]) == 1
        assert d["sources"][0]["id"] == "1"
        assert d["sources"][0]["score"] == 0.9
        assert d["sources"][0]["metadata"] == {"key": "val"}

    def test_response_to_dict_with_stats(self):
        """Test to_dict includes stats when present."""
        mock_stats = MagicMock()
        mock_stats.prompt_tokens = 10
        mock_stats.generated_tokens = 20
        mock_stats.total_time = 1.5
        mock_stats.tokens_per_second = 15.0

        response = RAGResponse(
            text="Answer",
            sources=[],
            stats=mock_stats,
        )
        d = response.to_dict()
        assert "stats" in d
        assert d["stats"]["prompt_tokens"] == 10
        assert d["stats"]["generated_tokens"] == 20


class TestRAGPipeline:
    """Test RAGPipeline class."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = MagicMock()
        embedder.embed.return_value = [0.1, 0.2, 0.3]
        embedder.dimension = 3
        return embedder

    @pytest.fixture
    def mock_store(self):
        """Create mock vector store."""
        store = MagicMock()
        store.search.return_value = [
            SearchResult(id="1", text="Context document 1", score=0.9, metadata={}),
            SearchResult(id="2", text="Context document 2", score=0.8, metadata={}),
        ]
        return store

    @pytest.fixture
    def mock_generator(self):
        """Create mock LLM generator."""
        generator = MagicMock()
        mock_response = MagicMock()
        mock_response.__str__ = MagicMock(return_value="Generated answer")
        mock_response.stats = None
        generator.return_value = mock_response
        return generator

    @pytest.fixture
    def pipeline(self, mock_embedder, mock_store, mock_generator):
        """Create RAGPipeline with mocks."""
        return RAGPipeline(
            embedder=mock_embedder,
            store=mock_store,
            generator=mock_generator,
        )

    def test_init(self, mock_embedder, mock_store, mock_generator):
        """Test pipeline initialization."""
        pipeline = RAGPipeline(
            embedder=mock_embedder,
            store=mock_store,
            generator=mock_generator,
        )
        assert pipeline.embedder is mock_embedder
        assert pipeline.store is mock_store
        assert pipeline.generator is mock_generator
        assert pipeline.config is not None

    def test_init_with_config(self, mock_embedder, mock_store, mock_generator):
        """Test initialization with custom config."""
        config = RAGConfig(top_k=10, temperature=0.5)
        pipeline = RAGPipeline(
            embedder=mock_embedder,
            store=mock_store,
            generator=mock_generator,
            config=config,
        )
        assert pipeline.config.top_k == 10
        assert pipeline.config.temperature == 0.5

    def test_query(self, pipeline, mock_embedder, mock_store, mock_generator):
        """Test query method."""
        response = pipeline.query("What is the meaning of life?")

        # Verify embedder was called
        mock_embedder.embed.assert_called_once_with("What is the meaning of life?")

        # Verify store search was called
        mock_store.search.assert_called_once()

        # Verify generator was called
        mock_generator.assert_called_once()

        # Verify response
        assert isinstance(response, RAGResponse)
        assert response.text == "Generated answer"
        assert len(response.sources) == 2
        assert response.query == "What is the meaning of life?"

    def test_query_with_config_override(self, pipeline, mock_store, mock_generator):
        """Test query with config override."""
        override_config = RAGConfig(top_k=3, temperature=0.2)
        pipeline.query("Question?", config=override_config)

        # Verify store was called with overridden top_k
        call_args = mock_store.search.call_args
        assert call_args.kwargs.get("k") == 3

        # Verify generator was called with overridden temperature via config object
        call_args = mock_generator.call_args
        gen_config = call_args.kwargs.get("config")
        assert gen_config is not None
        assert gen_config.temperature == 0.2

    def test_retrieve(self, pipeline, mock_embedder, mock_store):
        """Test retrieve method (without generation)."""
        sources = pipeline.retrieve("Question?")

        mock_embedder.embed.assert_called_once_with("Question?")
        mock_store.search.assert_called_once()
        assert len(sources) == 2

    def test_format_prompt_basic(self, pipeline):
        """Test basic prompt formatting."""
        sources = [
            SearchResult(id="1", text="Doc 1", score=0.9, metadata={}),
            SearchResult(id="2", text="Doc 2", score=0.8, metadata={}),
        ]
        config = RAGConfig()
        prompt = pipeline._format_prompt("What is X?", sources, config)

        assert "Doc 1" in prompt
        assert "Doc 2" in prompt
        assert "What is X?" in prompt

    def test_format_prompt_with_metadata(self, pipeline):
        """Test prompt formatting with metadata included."""
        sources = [
            SearchResult(id="1", text="Doc 1", score=0.9, metadata={"source": "file.txt"}),
        ]
        config = RAGConfig(include_metadata=True)
        prompt = pipeline._format_prompt("Question?", sources, config)

        assert "source: file.txt" in prompt
        assert "Doc 1" in prompt

    def test_format_prompt_custom_separator(self, pipeline):
        """Test prompt formatting with custom separator."""
        sources = [
            SearchResult(id="1", text="Doc 1", score=0.9, metadata={}),
            SearchResult(id="2", text="Doc 2", score=0.8, metadata={}),
        ]
        config = RAGConfig(context_separator="---")
        prompt = pipeline._format_prompt("Q?", sources, config)

        assert "Doc 1---Doc 2" in prompt or "Doc 1\n---\nDoc 2" in prompt or "---" in prompt

    def test_format_prompt_custom_template(self, pipeline):
        """Test prompt formatting with custom template."""
        sources = [
            SearchResult(id="1", text="Context here", score=0.9, metadata={}),
        ]
        config = RAGConfig(
            prompt_template="CONTEXT: {context}\nQUESTION: {question}\nANSWER:"
        )
        prompt = pipeline._format_prompt("What?", sources, config)

        assert "CONTEXT: Context here" in prompt
        assert "QUESTION: What?" in prompt
        assert "ANSWER:" in prompt

    def test_repr(self, pipeline):
        """Test __repr__ method."""
        repr_str = repr(pipeline)
        assert "RAGPipeline" in repr_str


class TestRAGPipelineStream:
    """Test RAGPipeline streaming."""

    def test_stream(self):
        """Test stream method."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

        mock_store = MagicMock()
        mock_store.search.return_value = [
            SearchResult(id="1", text="Context", score=0.9, metadata={}),
        ]

        # Mock generator that returns an iterator when stream=True
        def mock_generate(*args, **kwargs):
            if kwargs.get("stream"):
                return iter(["Token1 ", "Token2 ", "Token3"])
            return MagicMock(__str__=lambda: "Full response")

        mock_generator = MagicMock(side_effect=mock_generate)

        pipeline = RAGPipeline(
            embedder=mock_embedder,
            store=mock_store,
            generator=mock_generator,
        )

        tokens = list(pipeline.stream("Question?"))
        assert tokens == ["Token1 ", "Token2 ", "Token3"]


class TestDefaultPromptTemplate:
    """Test the default prompt template."""

    def test_template_has_placeholders(self):
        """Test that template has required placeholders."""
        assert "{context}" in DEFAULT_PROMPT_TEMPLATE
        assert "{question}" in DEFAULT_PROMPT_TEMPLATE

    def test_template_formatting(self):
        """Test that template can be formatted."""
        formatted = DEFAULT_PROMPT_TEMPLATE.format(
            context="Some context here",
            question="What is this?",
        )
        assert "Some context here" in formatted
        assert "What is this?" in formatted
