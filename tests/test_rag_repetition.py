"""Tests for the n-gram repetition detector used by RAGPipeline."""

from __future__ import annotations

import pytest

from cyllama.rag.repetition import NGramRepetitionDetector


class TestNGramRepetitionDetectorValidation:
    """Constructor input validation."""

    def test_ngram_too_small(self):
        with pytest.raises(ValueError, match="ngram must be >= 2"):
            NGramRepetitionDetector(window=10, ngram=1, threshold=2)

    def test_threshold_too_small(self):
        with pytest.raises(ValueError, match="threshold must be >= 2"):
            NGramRepetitionDetector(window=10, ngram=2, threshold=1)

    def test_window_smaller_than_ngram(self):
        with pytest.raises(ValueError, match="window .* must be >= ngram"):
            NGramRepetitionDetector(window=2, ngram=5, threshold=2)


class TestNGramRepetitionDetectorBasic:
    """Core feed() behaviour on small inputs."""

    def test_clean_text_does_not_trigger(self):
        det = NGramRepetitionDetector(window=80, ngram=5, threshold=3)
        # 30 unique words, well under any repetition window
        text = " ".join(f"word{i}" for i in range(30))
        assert det.feed(text) is False
        assert det.triggered is False

    def test_simple_loop_triggers(self):
        det = NGramRepetitionDetector(window=80, ngram=4, threshold=3)
        # Same 4-word phrase three times → trigger on the third repeat
        first = det.feed("the answer is forty two.")
        assert first is False
        second = det.feed(" the answer is forty two.")
        assert second is False
        third = det.feed(" the answer is forty two.")
        assert third is True
        assert det.triggered is True

    def test_punctuation_and_case_are_normalised(self):
        det = NGramRepetitionDetector(window=80, ngram=3, threshold=3)
        # Same n-gram with different punctuation and case
        det.feed("Hello there friend.")
        det.feed(" hello, there friend?")
        triggered = det.feed(" HELLO there friend!")
        assert triggered is True

    def test_distinct_ngrams_do_not_trigger(self):
        det = NGramRepetitionDetector(window=80, ngram=3, threshold=3)
        # Three sentences sharing common words but not a 3-gram
        det.feed("The cat sat on the mat.")
        det.feed(" A dog runs in the park.")
        triggered = det.feed(" Birds fly across the sky.")
        assert triggered is False

    def test_empty_chunk_is_noop(self):
        det = NGramRepetitionDetector(window=80, ngram=3, threshold=2)
        assert det.feed("") is False
        assert det.feed("   \n\t   ") is False

    def test_chunks_split_words(self):
        """Word boundaries can land mid-chunk; the detector should still
        treat consecutive chunks as a single stream."""
        det = NGramRepetitionDetector(window=80, ngram=3, threshold=3)
        # Stream "alpha beta gamma" three times in arbitrary chunk sizes
        for _ in range(3):
            det.feed("alpha ")
            det.feed("beta ")
            last = det.feed("gamma ")
        assert last is True


class TestNGramRepetitionDetectorWindow:
    """Rolling-window edge cases."""

    def test_window_evicts_old_repeats(self):
        """An n-gram that repeated earlier but has fallen out of the
        window should not count toward the threshold."""
        det = NGramRepetitionDetector(window=10, ngram=2, threshold=3)
        # Two early repeats inside the window
        det.feed("foo bar")
        det.feed(" foo bar")
        # Push enough unique words to evict the early repeats
        det.feed(" " + " ".join(f"x{i}" for i in range(20)))
        # Now a fresh "foo bar" — count should be 1, not 3
        triggered = det.feed(" foo bar")
        assert triggered is False

    def test_reset_clears_state(self):
        det = NGramRepetitionDetector(window=80, ngram=2, threshold=2)
        det.feed("hi there hi there")
        assert det.triggered is True
        det.reset()
        assert det.triggered is False
        # After reset, two fresh repeats should not be remembered
        assert det.feed("alpha beta") is False

    def test_triggered_stays_sticky(self):
        det = NGramRepetitionDetector(window=80, ngram=2, threshold=2)
        det.feed("loop loop loop")
        assert det.triggered is True
        # Even a clean chunk fed afterward returns True (sticky)
        assert det.feed(" something completely different") is True


class TestRealisticLoopScenarios:
    """Realistic loop patterns that motivated the detector."""

    def test_qwen_style_paraphrase_loop(self):
        """Mimics the Qwen3 RAG loop: same answer repeated verbatim
        a few times."""
        det = NGramRepetitionDetector(window=80, ngram=5, threshold=3)
        answer = "The capital of France is Paris."
        # Stream three repeats word by word
        triggered = False
        for _ in range(3):
            for word in answer.split():
                if det.feed(word + " "):
                    triggered = True
                    break
            if triggered:
                break
        assert triggered is True

    def test_long_distinct_answer_passes(self):
        """A genuine long answer with no internal loop should not trip
        the default thresholds."""
        det = NGramRepetitionDetector(window=80, ngram=5, threshold=3)
        answer = (
            "Python is a high-level programming language created by "
            "Guido van Rossum and first released in 1991. It emphasises "
            "code readability with significant indentation and supports "
            "multiple paradigms including procedural, object-oriented, "
            "and functional programming. The language is dynamically "
            "typed and garbage-collected, with a comprehensive standard "
            "library that ships with the interpreter."
        )
        triggered = False
        for word in answer.split():
            if det.feed(word + " "):
                triggered = True
                break
        assert triggered is False
