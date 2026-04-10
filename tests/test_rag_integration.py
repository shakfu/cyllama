"""End-to-end RAG integration tests using real models and a real corpus.

These complement the mock-based unit tests in:
  - tests/test_rag_pipeline.py        (pipeline routing / config wiring)
  - tests/test_rag_repetition.py      (detector internals)

Where the unit tests verify that the right code paths are exercised, the
integration tests in this file verify that those code paths actually work
end-to-end against real components: a real GGUF embedder, a real GGUF
generator, and a real document indexed from the on-disk corpus.

They're marked ``slow`` because each test loads a ~1.3 GB LLM, and skip
cleanly when either model file is missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL = ROOT / "models" / "bge-small-en-v1.5-q8_0.gguf"
GENERATION_MODEL = ROOT / "models" / "Llama-3.2-1B-Instruct-Q8_0.gguf"
QWEN3_MODEL = ROOT / "models" / "Qwen3-4B-Q8_0.gguf"
CORPUS_PATH = Path(__file__).resolve().parent / "media" / "corpus.txt"


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not EMBEDDING_MODEL.exists(),
        reason=f"Embedding model not found at {EMBEDDING_MODEL}",
    ),
    pytest.mark.skipif(
        not GENERATION_MODEL.exists(),
        reason=f"Generation model not found at {GENERATION_MODEL}",
    ),
    pytest.mark.skipif(
        not CORPUS_PATH.exists(),
        reason=f"Test corpus not found at {CORPUS_PATH}",
    ),
]


# ---------------------------------------------------------------------------
# Shared fixture: a real RAG instance indexed against tests/media/corpus.txt
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_rag():
    """Build a real RAG with bge-small + Llama-3.2-1B and index the
    Hemingway corpus once for the whole module."""
    from cyllama.rag import RAG

    rag = RAG(
        embedding_model=str(EMBEDDING_MODEL),
        generation_model=str(GENERATION_MODEL),
        chunk_size=400,
        chunk_overlap=40,
        n_gpu_layers=0,  # CPU only -- avoid GPU contention with other tests
    )
    # Index the corpus once. The file is short story-length, ~30 KB,
    # so this finishes in well under a second on CPU.
    rag.add_documents([CORPUS_PATH])
    assert rag.count > 0, "expected the corpus to produce at least one chunk"
    yield rag
    rag.close()


# ---------------------------------------------------------------------------
# Legacy path: no detection, no chat template
# ---------------------------------------------------------------------------


class TestRAGIntegrationLegacyPath:
    """The default RAGConfig (detection off, raw completion) should work
    end-to-end against real models and the corpus."""

    def test_query_returns_text_and_sources(self, real_rag):
        from cyllama.rag import RAGConfig

        cfg = RAGConfig(top_k=3, max_tokens=64, temperature=0.0)
        response = real_rag.query("Where is the story set?", config=cfg)

        assert response.text, "expected non-empty answer"
        assert len(response.sources) >= 1
        # Sources should come from the indexed corpus
        for src in response.sources:
            assert src.text, "source chunks should be non-empty"
        # Stats should be present on the legacy path (it preserves the
        # Response object's GenerationStats)
        assert response.stats is not None
        assert response.stats.prompt_tokens > 0


# ---------------------------------------------------------------------------
# Chat-template path
# ---------------------------------------------------------------------------


class TestRAGIntegrationChatTemplate:
    """Routing through generator.chat() with system+user messages must
    produce a coherent answer against real models."""

    def test_query_with_chat_template(self, real_rag):
        from cyllama.rag import RAGConfig

        cfg = RAGConfig(
            top_k=3,
            max_tokens=64,
            temperature=0.0,
            use_chat_template=True,
        )
        response = real_rag.query("Who is the protagonist?", config=cfg)

        assert response.text
        assert len(response.sources) >= 1
        # Streaming path doesn't carry GenerationStats through
        assert response.stats is None

    def test_stream_with_chat_template(self, real_rag):
        from cyllama.rag import RAGConfig

        cfg = RAGConfig(
            top_k=3,
            max_tokens=48,
            temperature=0.0,
            use_chat_template=True,
        )
        chunks = list(real_rag.stream("What is the setting?", config=cfg))
        assert chunks, "stream should yield at least one chunk"
        assert "".join(chunks).strip(), "concatenated text should be non-empty"


# ---------------------------------------------------------------------------
# Repetition detector against real generation
# ---------------------------------------------------------------------------


class TestRAGIntegrationRepetition:
    """The detector must (a) not false-positive on a clean real answer
    and (b) bound the output length when wired into a streaming run."""

    def test_detector_does_not_false_positive_on_clean_answer(self, real_rag):
        """A normal short answer to a clear question should not trip
        the default repetition thresholds."""
        from cyllama.rag import RAGConfig

        cfg = RAGConfig(
            top_k=3,
            max_tokens=64,
            temperature=0.0,
            repetition_threshold=3,
            repetition_ngram=5,
            repetition_window=80,
        )
        response = real_rag.query("Where is the story set?", config=cfg)
        assert response.text
        # On a 64-token cap with no loop, the answer should be a
        # reasonable length (more than just a couple of words).
        assert len(response.text.split()) >= 4

    def test_detection_does_not_break_streaming(self, real_rag):
        """Enabling the detector on the streaming path must still let
        a clean answer flow through to completion."""
        from cyllama.rag import RAGConfig

        cfg = RAGConfig(
            top_k=3,
            max_tokens=64,
            temperature=0.0,
            repetition_threshold=3,
            repetition_ngram=5,
        )
        chunks = list(real_rag.stream("What disease is mentioned in the story?", config=cfg))
        assert chunks
        text = "".join(chunks)
        assert text.strip()
        # Bounded by max_tokens; we just want to verify it terminates
        # cleanly and doesn't dump empty chunks.
        assert len(text) < 4000  # generous upper bound on a 64-token answer


# ---------------------------------------------------------------------------
# Qwen3-4B regression test for the original bug
# ---------------------------------------------------------------------------
#
# These tests pin down the actual failure mode the TODO described:
# Qwen3-4B asked a RAG question via raw completion produces a correct
# answer and then loops, repeating an entire ~50-word paragraph until
# max_tokens runs out. The legacy path (no detector, no chat template)
# is allowed to bloat; the two fix paths must produce a substantially
# shorter response.
#
# Greedy decoding (temperature=0) is used so the bug is reproducible
# across runs without a fixed seed plumbed through.


@pytest.fixture(scope="module")
def qwen3_rag():
    """Build a real RAG with bge-small + Qwen3-4B (the model that
    motivated the original bug report)."""
    if not QWEN3_MODEL.exists():
        pytest.skip(f"Qwen3 model not found at {QWEN3_MODEL}")

    from cyllama.rag import RAG

    rag = RAG(
        embedding_model=str(EMBEDDING_MODEL),
        generation_model=str(QWEN3_MODEL),
        chunk_size=400,
        chunk_overlap=40,
        n_gpu_layers=0,
    )
    rag.add_documents([CORPUS_PATH])
    yield rag
    rag.close()


# A query that empirically triggers the Qwen3-4B paragraph-loop failure
# mode against the indexed Hemingway corpus.
_QWEN3_LOOP_QUERY = "Where is the story set?"
_QWEN3_LEGACY_MAX_TOKENS = 512


class TestQwen3RAGLoopRegression:
    """Pin the original Qwen3-4B paragraph-loop bug and the two fixes."""

    def test_legacy_path_reproduces_loop(self, qwen3_rag):
        """Without any fix, Qwen3-4B should generate a bloated response
        that consumes most of the max_tokens budget."""
        from cyllama.rag import RAGConfig

        cfg = RAGConfig(
            top_k=3,
            max_tokens=_QWEN3_LEGACY_MAX_TOKENS,
            temperature=0.0,
        )
        response = qwen3_rag.query(_QWEN3_LOOP_QUERY, config=cfg)

        word_count = len(response.text.split())
        # The legacy path empirically returns ~400 words for this query
        # against this corpus. Assert at least 250 to make the test
        # robust to small model/inference variation while still proving
        # the loop is happening (a clean answer would be < 60 words).
        assert word_count >= 250, (
            f"expected legacy path to bloat past 250 words (loop reproduction), got {word_count}: {response.text!r}"
        )

    def test_repetition_detector_stops_loop(self, qwen3_rag):
        """With the detector enabled, the same query should produce a
        substantially shorter response."""
        from cyllama.rag import RAGConfig

        cfg = RAGConfig(
            top_k=3,
            max_tokens=_QWEN3_LEGACY_MAX_TOKENS,
            temperature=0.0,
            repetition_threshold=3,
            repetition_ngram=5,
            repetition_window=300,
        )
        response = qwen3_rag.query(_QWEN3_LOOP_QUERY, config=cfg)

        word_count = len(response.text.split())
        # Detector empirically cuts the response to ~115 words (-71%).
        # Allow up to 200 words to absorb variation while still proving
        # the loop was caught well before the 250-word legacy floor.
        assert word_count < 200, (
            f"expected the detector to stop generation under 200 words, got {word_count}: {response.text!r}"
        )

    def test_chat_template_avoids_loop(self, qwen3_rag):
        """With the chat-template path, Qwen3 enters its native
        thinking mode and produces a coherent answer without looping."""
        from cyllama.rag import RAGConfig

        cfg = RAGConfig(
            top_k=3,
            max_tokens=_QWEN3_LEGACY_MAX_TOKENS,
            temperature=0.0,
            use_chat_template=True,
        )
        response = qwen3_rag.query(_QWEN3_LOOP_QUERY, config=cfg)

        word_count = len(response.text.split())
        # Chat template empirically yields ~200 words including the
        # <think> reasoning block. Cap at 250 words to confirm no loop.
        assert word_count < 250, (
            f"expected chat-template path to avoid the loop entirely, got {word_count} words: {response.text!r}"
        )
        # The response must contain a real answer about the story setting.
        text_lower = response.text.lower()
        assert "paris" in text_lower or "africa" in text_lower, (
            f"expected the chat-template answer to mention Paris or Africa, got: {response.text!r}"
        )
