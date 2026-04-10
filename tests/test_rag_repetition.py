"""Tests for the streaming output filters used by RAGPipeline."""

from __future__ import annotations

import pytest

from cyllama.rag.repetition import NGramRepetitionDetector, ThinkBlockStripper


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


# ---------------------------------------------------------------------------
# ThinkBlockStripper
# ---------------------------------------------------------------------------


def _drive(stripper: ThinkBlockStripper, chunks: list[str]) -> str:
    """Feed chunks one at a time, flush, return concatenated cleaned output."""
    out: list[str] = []
    for chunk in chunks:
        out.extend(stripper.feed(chunk))
    out.extend(stripper.flush())
    return "".join(out)


class TestThinkBlockStripperBasic:
    """Core block-removal behaviour on whole-string inputs."""

    def test_no_tags_passes_through(self):
        s = ThinkBlockStripper()
        assert _drive(s, ["hello world"]) == "hello world"

    def test_strips_a_single_block(self):
        s = ThinkBlockStripper()
        text = "before<think>reasoning here</think>after"
        assert _drive(s, [text]) == "beforeafter"

    def test_strips_multiple_blocks(self):
        s = ThinkBlockStripper()
        text = "<think>one</think>middle<think>two</think>end"
        assert _drive(s, [text]) == "middleend"

    def test_block_at_start_only(self):
        s = ThinkBlockStripper()
        text = "<think>thinking...</think>The answer is 42."
        assert _drive(s, [text]) == "The answer is 42."

    def test_block_at_end_only(self):
        s = ThinkBlockStripper()
        text = "Answer: 42.<think>checking...</think>"
        assert _drive(s, [text]) == "Answer: 42."

    def test_unclosed_block_discards_remainder(self):
        """If the model never closes the think block, everything after
        ``<think>`` is discarded -- judged correct because the model is
        broken and we cannot prove the residual is anything but
        unfinished reasoning."""
        s = ThinkBlockStripper()
        text = "before<think>incomplete reasoning..."
        assert _drive(s, [text]) == "before"

    def test_empty_block(self):
        s = ThinkBlockStripper()
        assert _drive(s, ["a<think></think>b"]) == "ab"


class TestThinkBlockStripperStreaming:
    """Chunk-boundary handling -- the whole reason this isn't a regex pass."""

    def test_open_tag_split_across_chunks(self):
        """``<think>`` arrives as ``<thi`` then ``nk>``. The stripper
        must not emit ``<thi`` as visible output."""
        s = ThinkBlockStripper()
        out = _drive(s, ["before<thi", "nk>reasoning</think>after"])
        assert out == "beforeafter"

    def test_close_tag_split_across_chunks(self):
        """``</think>`` arrives as ``</thi`` then ``nk>``. The stripper
        must keep discarding until the close tag completes."""
        s = ThinkBlockStripper()
        out = _drive(s, ["pre<think>thinking</thi", "nk>post"])
        assert out == "prepost"

    def test_one_character_per_chunk(self):
        """Worst-case: every character is its own chunk. Output must
        still be correct."""
        s = ThinkBlockStripper()
        text = "ab<think>xyz</think>cd"
        chunks = list(text)
        assert _drive(s, chunks) == "abcd"

    def test_partial_open_tag_at_eof_is_emitted(self):
        """A trailing ``<thi`` with no chunks left is real content (it
        cannot become a tag), so flush must emit it."""
        s = ThinkBlockStripper()
        out = _drive(s, ["hello <thi"])
        assert out == "hello <thi"

    def test_block_spans_many_chunks(self):
        s = ThinkBlockStripper()
        chunks = [
            "answer prefix ",
            "<think>",
            "step one ",
            "step two ",
            "step three",
            "</think>",
            " answer suffix",
        ]
        # The trailing space on "answer prefix " survives because it
        # appears mid-stream (lstrip is not active there). The leading
        # space on " answer suffix" is consumed by the post-close-tag
        # lstrip pass -- see TestThinkBlockStripperLstrip for the
        # full lstrip semantics.
        assert _drive(s, chunks) == "answer prefix answer suffix"


class TestThinkBlockStripperRealistic:
    """Shapes that match real Qwen3 / DeepSeek output."""

    def test_qwen3_summarize_shape(self):
        """The exact failure shape from `scripts/case/rag-chat1.sh`:
        Qwen3 emits its <think> block, then the actual answer. With
        the post-close-tag lstrip in place, the user sees the answer
        with no leading blank line."""
        s = ThinkBlockStripper()
        chunks = [
            "<think>\n",
            "Okay, so I need to summarize this story based on the provided context. ",
            "Let me read through the context again to make sure I understand the key points.\n",
            "</think>\n\n",
            "The story is about a man who is dying in a tent.",
        ]
        out = _drive(s, chunks)
        assert "Okay" not in out
        assert "summarize" not in out
        # Tighter assertion than .strip() == ...: the output must be the
        # answer with NO leading whitespace at all (the `\n\n` after
        # `</think>` was the user-visible bug we're fixing here).
        assert out == "The story is about a man who is dying in a tent."

    def test_no_think_block_means_no_op(self):
        """Models that don't emit <think> at all should see the
        stripper as a transparent passthrough -- modulo any leading
        whitespace at the very start of the stream, which is also
        stripped (see TestThinkBlockStripperLstrip)."""
        s = ThinkBlockStripper()
        chunks = ["The answer ", "is 42. ", "It comes from the question of ", "life, the universe, ", "and everything."]
        out = _drive(s, chunks)
        assert out == "The answer is 42. It comes from the question of life, the universe, and everything."


class TestThinkBlockStripperLstrip:
    """Leading-whitespace stripping after close tags and at stream start.

    This is the behaviour that fixes the user-visible double-newline at
    the start of every Qwen3 RAG answer: `</think>` is followed by
    `\\n\\n` and we don't want those to survive into the user-facing
    output.
    """

    def test_strips_whitespace_immediately_after_close_tag(self):
        """`\\n\\n` directly after `</think>` is consumed."""
        s = ThinkBlockStripper()
        out = _drive(s, ["<think>x</think>\n\nAnswer."])
        assert out == "Answer."

    def test_strips_whitespace_split_across_chunks_after_close_tag(self):
        """The whitespace tail can arrive over several chunks; all of
        it must be consumed before the first non-whitespace char."""
        s = ThinkBlockStripper()
        out = _drive(s, ["<think>x</think>", "\n", "  ", "\n", "Answer."])
        assert out == "Answer."

    def test_lstrip_only_consumes_leading_whitespace(self):
        """Whitespace AFTER the first non-whitespace character of the
        post-block segment must be preserved -- only LEADING whitespace
        is consumed, not all whitespace."""
        s = ThinkBlockStripper()
        out = _drive(s, ["<think>x</think>\n\nAnswer one. Answer two."])
        assert out == "Answer one. Answer two."

    def test_lstrip_re_arms_after_each_close_tag(self):
        """Multiple think blocks: each post-close-tag transition
        re-arms the lstrip independently."""
        s = ThinkBlockStripper()
        chunks = [
            "<think>r1</think>\n\n",
            "First answer.",
            "\n\n<think>r2</think>\n\n",
            "Second answer.",
        ]
        out = _drive(s, chunks)
        # The `\n\n` between the two answers (before the second think
        # block) is mid-stream content, so it survives. The `\n\n` after
        # the SECOND `</think>` is consumed.
        assert out == "First answer.\n\nSecond answer."

    def test_strips_leading_whitespace_at_start_of_stream(self):
        """Leading whitespace before any think block is also consumed,
        because the stripper starts in pending-lstrip state."""
        s = ThinkBlockStripper()
        out = _drive(s, ["\n\n  Hello, world."])
        assert out == "Hello, world."

    def test_pure_whitespace_chunks_at_start_keep_waiting(self):
        """If the first chunks contain only whitespace, the stripper
        consumes them silently and waits for the first non-whitespace
        char without emitting anything."""
        s = ThinkBlockStripper()
        out = _drive(s, ["   ", "\n", "  ", "Hi."])
        assert out == "Hi."

    def test_lstrip_does_not_affect_mid_content_whitespace(self):
        """Once the first non-whitespace char has been seen in a
        post-block segment, subsequent leading whitespace on later
        chunks must pass through unchanged (the lstrip flag is one-shot
        per inside->outside transition)."""
        s = ThinkBlockStripper()
        out = _drive(s, ["<think>x</think>\nAnswer.", "\n\n  Continuation."])
        assert out == "Answer.\n\n  Continuation."

    def test_lstrip_on_unclosed_block_at_eof(self):
        """If the stream is pure whitespace (no real content), flush
        emits nothing rather than the whitespace."""
        s = ThinkBlockStripper()
        out = _drive(s, ["\n\n   "])
        assert out == ""
