"""Tests for the RAG Pipeline classes."""

from unittest.mock import MagicMock

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
        config = RAGConfig(prompt_template="CONTEXT: {context}\nQUESTION: {question}\nANSWER:")
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


# ---------------------------------------------------------------------------
# Repetition detection and chat-template path
# ---------------------------------------------------------------------------


def _make_pipeline(generator):
    """Build a RAGPipeline backed by the given mock generator."""
    embedder = MagicMock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]

    store = MagicMock()
    store.search.return_value = [
        SearchResult(id="1", text="Context document", score=0.9, metadata={}),
    ]

    return RAGPipeline(embedder=embedder, store=store, generator=generator)


def _streaming_generator(chunks):
    """Build a mock generator whose __call__ returns ``chunks`` on
    ``stream=True`` and a string-stub Response otherwise."""

    def _call(*args, **kwargs):
        if kwargs.get("stream"):
            return iter(chunks)
        resp = MagicMock()
        resp.__str__ = MagicMock(return_value="".join(chunks))
        resp.stats = None
        return resp

    return MagicMock(side_effect=_call)


class TestRAGConfigRepetitionValidation:
    """Validation rules for the new RAGConfig fields."""

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValueError, match="repetition_threshold must be >= 0"):
            RAGConfig(repetition_threshold=-1)

    def test_invalid_ngram_when_enabled(self):
        with pytest.raises(ValueError, match="repetition_ngram must be >= 2"):
            RAGConfig(repetition_threshold=3, repetition_ngram=1)

    def test_window_smaller_than_ngram_when_enabled(self):
        with pytest.raises(ValueError, match="repetition_window .* must be >="):
            RAGConfig(repetition_threshold=3, repetition_ngram=5, repetition_window=3)

    def test_disabled_skips_geometry_checks(self):
        # Bogus geometry is fine when the feature is disabled
        cfg = RAGConfig(repetition_threshold=0, repetition_ngram=1, repetition_window=1)
        assert cfg.repetition_threshold == 0


class TestRAGPipelineRepetition:
    """Streaming-level loop guard."""

    def test_default_off_uses_legacy_path(self):
        """With detection off, query() should call the generator
        non-streaming and preserve the existing behaviour."""
        gen = MagicMock()
        resp = MagicMock()
        resp.__str__ = MagicMock(return_value="answer")
        resp.stats = None
        gen.return_value = resp

        pipeline = _make_pipeline(gen)
        result = pipeline.query("Q?")

        assert result.text == "answer"
        # Generator was called once, non-streaming (no stream= kwarg)
        call = gen.call_args
        assert call is not None
        assert call.kwargs.get("stream") is None or call.kwargs.get("stream") is False

    def test_query_routes_through_streaming_when_enabled(self):
        """With detection on, query() must consume the streaming
        iterator so the detector can act on chunks."""
        chunks = ["The ", "answer ", "is ", "42."]
        gen = _streaming_generator(chunks)
        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(repetition_threshold=3)

        result = pipeline.query("Q?", config=cfg)

        assert result.text == "The answer is 42."
        call = gen.call_args
        assert call.kwargs.get("stream") is True

    def test_stream_stops_when_loop_detected(self):
        """Pipeline.stream() must terminate as soon as the detector
        flags the trailing n-gram as repetitive."""
        # Five repeats of the same 4-word answer, one word per chunk.
        repeated = ["the ", "answer ", "is ", "forty-two. "] * 5
        gen = _streaming_generator(repeated)
        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(repetition_threshold=3, repetition_ngram=4, repetition_window=80)

        out = list(pipeline.stream("Q?", config=cfg))

        # The detector trips on the third full repeat, so we must have
        # yielded fewer chunks than the full 20-chunk stream.
        assert len(out) < len(repeated), f"expected the loop guard to cut the stream short, got all {len(out)} chunks"
        # And we must have yielded at least the first full answer plus
        # enough of the second to trigger the detector.
        assert "".join(out).count("the") >= 2

    def test_query_stops_when_loop_detected(self):
        """query() routed through streaming should also stop early."""
        repeated = ["the ", "answer ", "is ", "forty-two. "] * 5
        gen = _streaming_generator(repeated)
        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(repetition_threshold=3, repetition_ngram=4)

        result = pipeline.query("Q?", config=cfg)

        # Text is shorter than the full 5x repeat
        full = "".join(repeated)
        assert len(result.text) < len(full)
        # And stats is None on this code path (we don't have a Response
        # object to harvest them from when streaming).
        assert result.stats is None

    def test_triggering_chunk_is_yielded_not_dropped(self):
        """The chunk that *completes* the repeating n-gram is yielded to
        the caller before the generator exits. This pins the
        yield-then-feed ordering in `_generate_chunks`: a tiny dangling
        loop-fragment at the end of the output is the deliberate UX
        trade-off (it acts as a visible "the guard fired" tell, so users
        can distinguish a loop-cut from a max_tokens-cut).

        With ngram=2, threshold=3, and a perfectly periodic ``x y z``
        stream, the trailing 2-gram ``x y`` first hits 3 occurrences when
        the 8th chunk (the third ``y``) is fed. That chunk must be in
        the output, and the generator must stop immediately after.
        """
        looping = ["x ", "y ", "z "] * 10  # 30 chunks
        gen = _streaming_generator(looping)
        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(repetition_threshold=3, repetition_ngram=2)

        out = list(pipeline.stream("Q?", config=cfg))

        # 8 chunks yielded: 7 leading up to the trigger plus the
        # triggering chunk itself ("y ").
        assert len(out) == 8, f"expected 8 chunks (trigger chunk yielded), got {len(out)}: {out!r}"
        text = "".join(out)
        assert text == "x y z x y z x y "
        # The text ends on the loop-completing ``y`` -- the visible
        # signature of yield-then-feed.
        assert text.rstrip().endswith("y")


class TestRAGPipelineChatTemplate:
    """Chat-template generation path."""

    def test_use_chat_template_calls_generator_chat(self):
        """When use_chat_template is True, the pipeline must route
        through generator.chat() with system+user messages instead of
        generator.__call__ with a raw prompt."""
        gen = MagicMock()
        gen.chat.return_value = iter(["chat ", "response"])

        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(use_chat_template=True)
        result = pipeline.query("What is X?", config=cfg)

        # generator.chat was called, not generator.__call__
        gen.chat.assert_called_once()
        # The plain generator was never invoked for generation
        gen.assert_not_called()

        # Inspect the messages: system + user, with question + context
        chat_call = gen.chat.call_args
        messages = chat_call.args[0] if chat_call.args else chat_call.kwargs.get("messages")
        assert messages is not None
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user"]
        assert "What is X?" in messages[1]["content"]
        assert "Context document" in messages[1]["content"]

        # Streaming kwarg must be set so we can iterate
        assert chat_call.kwargs.get("stream") is True

        # The chunks were concatenated into the response text
        assert result.text == "chat response"

    def test_custom_system_prompt(self):
        gen = MagicMock()
        gen.chat.return_value = iter(["ok"])

        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(
            use_chat_template=True,
            system_prompt="You are a brevity bot. Answer in five words.",
        )
        pipeline.query("Q?", config=cfg)

        chat_call = gen.chat.call_args
        messages = chat_call.args[0] if chat_call.args else chat_call.kwargs.get("messages")
        assert messages[0]["content"] == "You are a brevity bot. Answer in five words."

    def test_chat_template_combined_with_repetition(self):
        """Both features together: chat-template path is iterated and
        cut short by the detector."""
        gen = MagicMock()
        looping = ["x ", "y ", "z "] * 10
        gen.chat.return_value = iter(looping)

        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(
            use_chat_template=True,
            repetition_threshold=3,
            repetition_ngram=2,
            strip_think_blocks=False,  # isolate the detector under test
        )
        out = list(pipeline.stream("Q?", config=cfg))
        assert len(out) < len(looping)


class TestRAGPipelineThinkStripping:
    """`<think>...</think>` stripping wiring through the pipeline."""

    def test_strip_off_by_default_in_library_config(self):
        """`RAGConfig()` defaults `strip_think_blocks=False` for the
        same backwards-compat reason as the other streaming-path
        features: enabling it forces routing through the streaming
        code path, which loses GenerationStats. Library callers must
        opt in. The CLI flips this for end users."""
        cfg = RAGConfig()
        assert cfg.strip_think_blocks is False

    def test_strip_think_blocks_when_opted_in(self):
        """A model that emits a Qwen3-style think block must have it
        removed before chunks reach the caller when the user opts in."""
        gen = MagicMock()
        chunks = [
            "<think>",
            "Okay, let me reason about this. ",
            "The user wants the answer.",
            "</think>",
            "The answer is 42.",
        ]
        gen.chat.return_value = iter(chunks)

        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(use_chat_template=True, strip_think_blocks=True)
        out = list(pipeline.stream("Q?", config=cfg))
        text = "".join(out)

        assert "Okay" not in text
        assert "reason about" not in text
        assert text == "The answer is 42."

    def test_strip_think_blocks_passthrough_when_disabled(self):
        """With strip_think_blocks=False, reasoning content is preserved
        verbatim -- e.g. for debugging or transcript capture."""
        gen = MagicMock()
        chunks = ["<think>", "reasoning", "</think>", "answer"]
        gen.chat.return_value = iter(chunks)

        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(use_chat_template=True, strip_think_blocks=False)
        text = "".join(pipeline.stream("Q?", config=cfg))

        assert text == "<think>reasoning</think>answer"

    def test_strip_alone_routes_through_streaming(self):
        """Setting strip_think_blocks=True must route the pipeline
        through `_generate_chunks` even when neither repetition nor
        chat-template is enabled. Otherwise the strip would silently
        no-op on the legacy non-streaming path."""
        gen = MagicMock()
        gen.return_value = iter(["<think>x</think>final"])

        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(
            use_chat_template=False,
            repetition_threshold=0,
            strip_think_blocks=True,
        )
        result = pipeline.query("Q?", config=cfg)

        # Strip applied -> only "final" reaches the caller.
        assert result.text == "final"
        # And stats is None because we took the streaming path (the
        # GenerationStats trade-off documented on the routing gate).
        assert result.stats is None

    def test_strip_then_detect_filter_order(self):
        """Filter order pin: the stripper runs *before* the repetition
        detector. If the order were reversed, the detector's window
        would be polluted with reasoning content the user never sees,
        and could false-positive on legitimate phrase repetition between
        the think block and the answer.

        We construct a stream where the *think block* contains a
        phrase that, if it leaked into the detector's window, would
        cause the detector to fire on the answer's first repetition of
        that phrase. With strip-then-detect, the detector never sees
        the think content, so it does not fire and the full answer is
        emitted.
        """
        gen = MagicMock()
        chunks = [
            "<think>",
            "the answer is yes ",  # this phrase is inside the think block
            "the answer is yes ",  # repeat inside the think block
            "</think>",
            "the answer is yes.",  # appears once in the user-visible answer
        ]
        gen.chat.return_value = iter(chunks)

        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(
            use_chat_template=True,
            strip_think_blocks=True,
            repetition_threshold=2,
            repetition_ngram=4,
        )
        text = "".join(pipeline.stream("Q?", config=cfg))

        # The user sees only the post-strip content, and the detector
        # has not fired (the phrase appears only once after stripping).
        assert text == "the answer is yes."


class TestRAGPipelineSystemRoleFallback:
    """Feature-detection fallback for chat templates that reject the
    `system` role (Gemma 2/3, some Mistral variants, etc.).

    The test surface here is small but load-bearing: without this
    fallback, `cyllama rag -m gemma-*.gguf` crashes immediately on the
    first query with `RuntimeError: Failed to apply chat template`. The
    fallback merges the system content into the first user message and
    retries, so the same downstream code path works for both
    template families.
    """

    def test_canonical_path_used_when_system_role_supported(self):
        """When the underlying chat() call succeeds with [system, user],
        no fallback is invoked and the canonical messages reach the
        generator unchanged."""
        gen = MagicMock()
        gen.chat.return_value = iter(["ok"])

        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(use_chat_template=True)
        list(pipeline.stream("Q?", config=cfg))

        gen.chat.assert_called_once()
        messages = gen.chat.call_args.args[0]
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user"]
        assert pipeline._system_role_supported is True

    def test_falls_back_to_merged_on_template_runtime_error(self):
        """When the first chat() call raises a chat-template error, the
        fallback merges system into user and retries. The retry must
        succeed and the generator must end up with a single user
        message containing the system content as a prefix."""
        gen = MagicMock()
        # First call: simulate Gemma rejecting the system role.
        # Second call: succeeds with the merged messages.
        gen.chat.side_effect = [
            RuntimeError("Failed to apply chat template"),
            iter(["gemma response"]),
        ]

        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(
            use_chat_template=True,
            system_prompt="You are a helpful assistant.",
        )
        out = "".join(pipeline.stream("Q?", config=cfg))

        assert out == "gemma response"
        assert gen.chat.call_count == 2
        # Inspect the retry call's messages
        retry_messages = gen.chat.call_args_list[1].args[0]
        roles = [m["role"] for m in retry_messages]
        assert roles == ["user"], f"expected only a user role after merge, got {roles}"
        merged_user = retry_messages[0]["content"]
        assert "You are a helpful assistant." in merged_user
        assert "Q?" in merged_user
        assert pipeline._system_role_supported is False

    def test_fallback_decision_is_cached_across_queries(self):
        """After the first query has discovered the chat template
        rejects system role, subsequent queries must skip the failed
        attempt and call chat() exactly once with merged messages."""
        gen = MagicMock()
        gen.chat.side_effect = [
            RuntimeError("Failed to apply chat template"),
            iter(["first response"]),
            iter(["second response"]),
            iter(["third response"]),
        ]

        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(use_chat_template=True)
        list(pipeline.stream("Q1?", config=cfg))
        list(pipeline.stream("Q2?", config=cfg))
        list(pipeline.stream("Q3?", config=cfg))

        # 1 failed call + 1 successful retry on Q1, then 1 successful
        # call each on Q2 and Q3 -> 4 total chat() invocations.
        assert gen.chat.call_count == 4
        # And every call after the first must be the merged shape.
        for call in gen.chat.call_args_list[1:]:
            messages = call.args[0]
            assert [m["role"] for m in messages] == ["user"]

    def test_unrelated_runtime_errors_are_not_swallowed(self):
        """A RuntimeError that doesn't look like a chat-template
        rejection must propagate, not be silently retried with merged
        messages. Otherwise we'd mask genuine bugs (OOM, model crash,
        etc.) under a misleading 'fell back to user-only' code path."""
        gen = MagicMock()
        gen.chat.side_effect = RuntimeError("CUDA out of memory")

        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(use_chat_template=True)

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            list(pipeline.stream("Q?", config=cfg))
        # No fallback was attempted.
        assert gen.chat.call_count == 1
        assert pipeline._system_role_supported is None

    def test_merge_helper_preserves_message_list_immutability(self):
        """The fallback must not mutate the canonical messages list
        held by the caller; it returns a new list."""
        original = [
            {"role": "system", "content": "system content"},
            {"role": "user", "content": "user content"},
        ]
        merged = RAGPipeline._merge_system_into_user(original)

        # Caller's list is unchanged
        assert original[0]["role"] == "system"
        assert original[0]["content"] == "system content"
        assert len(original) == 2
        # Merged is the expected shape
        assert len(merged) == 1
        assert merged[0]["role"] == "user"
        assert "system content" in merged[0]["content"]
        assert "user content" in merged[0]["content"]

    def test_merge_helper_handles_no_system_message(self):
        """If there's no system message in the input, the merge is a
        no-op (returns user message unchanged)."""
        original = [{"role": "user", "content": "just a user message"}]
        merged = RAGPipeline._merge_system_into_user(original)
        assert len(merged) == 1
        assert merged[0]["role"] == "user"
        assert merged[0]["content"] == "just a user message"


class TestRAGPipelineRawCompletionFallback:
    """Third-tier fallback: when both chat-template shapes fail with a
    template error, the pipeline degrades to the raw-completion path.

    This handles the case where llama.cpp's basic `llama_chat_apply_template`
    C API can't apply the model's embedded Jinja chat template at all
    (e.g. some Gemma 4 GGUFs whose embedded Jinja doesn't match any of
    the substring heuristics in `llm_chat_detect_template`). The merge
    fallback won't help because the failure isn't about the system role
    -- it's about the template being unevaluable.
    """

    def test_falls_back_to_completion_when_both_chat_shapes_fail(self):
        """When canonical AND merged chat calls both raise template
        errors, the pipeline must call the generator via the raw
        __call__ path with a formatted completion prompt, and the
        result must reach the caller."""
        gen = MagicMock()
        # Both chat attempts raise; the raw completion call succeeds.
        gen.chat.side_effect = [
            RuntimeError("Failed to apply chat template"),
            RuntimeError("Failed to apply chat template"),
        ]
        gen.return_value = iter(["completion ", "answer"])

        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(use_chat_template=True)

        with pytest.warns(RuntimeWarning, match="raw-completion"):
            out = "".join(pipeline.stream("Q?", config=cfg))

        assert out == "completion answer"
        # Two failed chat attempts, then exactly one successful
        # raw-completion call.
        assert gen.chat.call_count == 2
        assert gen.call_count == 1
        # And the cache flag is now sticky.
        assert pipeline._chat_template_unusable is True

    def test_completion_fallback_decision_is_cached(self):
        """After the pipeline has discovered chat template is unusable,
        subsequent queries skip both chat attempts entirely and call
        the generator's raw __call__ directly."""
        gen = MagicMock()
        gen.chat.side_effect = [
            RuntimeError("Failed to apply chat template"),
            RuntimeError("Failed to apply chat template"),
        ]
        gen.return_value = iter(["first"])

        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(use_chat_template=True)

        # First query: 2 chat failures + 1 completion success.
        with pytest.warns(RuntimeWarning):
            list(pipeline.stream("Q1?", config=cfg))

        # Reset and prep generator for two more queries -- chat must
        # not be called at all from here on.
        gen.chat.reset_mock()
        gen.reset_mock()
        gen.return_value = iter(["second"])
        list(pipeline.stream("Q2?", config=cfg))
        gen.return_value = iter(["third"])
        list(pipeline.stream("Q3?", config=cfg))

        gen.chat.assert_not_called()
        assert gen.call_count == 2

    def test_completion_fallback_uses_completion_stop_sequences(self):
        """The fallback must rebuild gen_config with the
        Question:/Answer: stop sequences that the raw-completion path
        needs -- otherwise the model can run past the answer into a
        hallucinated next turn. The original gen_config (built for the
        chat path) had no stop sequences."""
        gen = MagicMock()
        gen.chat.side_effect = [
            RuntimeError("Failed to apply chat template"),
            RuntimeError("Failed to apply chat template"),
        ]
        gen.return_value = iter(["ok"])

        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(use_chat_template=True)

        with pytest.warns(RuntimeWarning):
            list(pipeline.stream("Q?", config=cfg))

        # Inspect the gen_config passed to the raw-completion call.
        completion_call_kwargs = gen.call_args.kwargs
        completion_config = completion_call_kwargs.get("config")
        assert completion_config is not None
        assert "Question:" in completion_config.stop_sequences
        assert "\nContext:" in completion_config.stop_sequences
        assert "\nAnswer:" in completion_config.stop_sequences

    def test_unrelated_error_does_not_trigger_completion_fallback(self):
        """A RuntimeError that doesn't mention 'template' must
        propagate, not trigger the completion fallback. Otherwise we'd
        mask genuine bugs (CUDA OOM, model crash) under a misleading
        'falling back' code path."""
        gen = MagicMock()
        # First call: simulate a non-template RuntimeError.
        gen.chat.side_effect = RuntimeError("CUDA out of memory")

        pipeline = _make_pipeline(gen)
        cfg = RAGConfig(use_chat_template=True)

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            list(pipeline.stream("Q?", config=cfg))
        # No completion fallback was attempted.
        assert gen.call_count == 0
        assert pipeline._chat_template_unusable is False
