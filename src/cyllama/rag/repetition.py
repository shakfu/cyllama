"""Streaming output filters for RAG generation.

This module contains the per-chunk filters that the RAG pipeline routes
generated tokens through:

* :class:`NGramRepetitionDetector` -- catches lexical loops (paragraph-
  level paraphrase loops are the canonical Qwen3-4B failure mode).
* :class:`ThinkBlockStripper` -- removes ``<think>...</think>`` reasoning
  blocks emitted by models that ship with chain-of-thought enabled by
  default (Qwen3, DeepSeek-R1, etc.). The blocks otherwise consume the
  entire token budget on small ``max_tokens`` values, leaving no budget
  for the actual answer.

Both filters are designed for use inside a streaming generator: each
chunk goes in, the filter decides what (if anything) to emit, and any
relevant state survives across chunks (rolling window for the detector,
inside-block buffer for the stripper). Chunk boundaries falling inside
sub-word tokens or in the middle of an XML-style tag are handled.

The N-gram detector operates on word-normalised text -- lowercase,
``\\w+`` tokens only -- so trailing punctuation, capitalisation, and
whitespace differences do not defeat it. It maintains a rolling window
of the most recent words and, after each chunk, checks whether the
trailing n-gram has occurred at least ``threshold`` times within the
window. If so, :meth:`NGramRepetitionDetector.feed` returns True and
the caller should stop generation.

Defaults (window=300, ngram=5, threshold=3) are tuned against the
Qwen3-4B paragraph-loop failure mode: greedy decoding can repeat an
entire ~50-word paragraph multiple times, so the window must be large
enough to hold three full paragraph repeats. A smaller window (~80
words) catches short phrase loops but misses paragraph loops; a larger
window catches both at negligible CPU cost. Tighten the parameters if
you still see loops slip through, loosen them if you see false
positives on formulaic answers (lists, tabular output, etc.).
"""

from __future__ import annotations

import re
from collections import deque
from typing import Deque, Iterator

# Word tokeniser: ``\w+`` matches Unicode word characters (letters, digits,
# underscore) so we treat punctuation/whitespace as separators. Lowercasing
# happens before the regex runs.
_WORD_RE = re.compile(r"\w+", re.UNICODE)


class NGramRepetitionDetector:
    """Stream-friendly word-level n-gram repetition detector.

    Args:
        window: Number of recent words to keep for repetition checks.
            Must be >= ``ngram``. Larger windows catch slower loops but
            cost more per check.
        ngram: Length of the n-gram to detect (in words). Must be >= 2.
            Shorter n-grams are more sensitive but more prone to false
            positives on formulaic content. ``5`` is a reasonable default.
        threshold: Number of times the trailing n-gram must appear within
            the window before :meth:`feed` returns True. Must be >= 2.
            ``3`` catches obvious loops without flagging accidental
            repeats.

    Example:
        >>> det = NGramRepetitionDetector(window=20, ngram=3, threshold=3)
        >>> det.feed("the answer is 42")
        False
        >>> det.feed(" the answer is 42")
        False
        >>> det.feed(" the answer is 42")
        True
    """

    def __init__(
        self,
        window: int = 300,
        ngram: int = 5,
        threshold: int = 3,
    ) -> None:
        if ngram < 2:
            raise ValueError(f"ngram must be >= 2, got {ngram}")
        if threshold < 2:
            raise ValueError(f"threshold must be >= 2, got {threshold}")
        if window < ngram:
            raise ValueError(f"window ({window}) must be >= ngram ({ngram})")
        self.window = window
        self.ngram = ngram
        self.threshold = threshold
        self._words: Deque[str] = deque(maxlen=window)
        self._triggered = False

    @property
    def triggered(self) -> bool:
        """Whether :meth:`feed` has already returned True for this stream."""
        return self._triggered

    def reset(self) -> None:
        """Clear the rolling window. Call between independent streams."""
        self._words.clear()
        self._triggered = False

    def feed(self, chunk: str) -> bool:
        """Append a chunk of generated text to the rolling window.

        Returns:
            True if the trailing n-gram has now appeared at least
            ``threshold`` times within the window. Once True, the
            detector stays "triggered" -- subsequent calls return True
            even if the caller decides to keep going (which would be a
            bug, but the contract is unambiguous).
        """
        if self._triggered:
            return True
        if not chunk:
            return False

        new_words = _WORD_RE.findall(chunk.lower())
        if not new_words:
            return False
        self._words.extend(new_words)

        if len(self._words) < self.ngram:
            return False

        # Snapshot the window so the index math is straightforward.
        words = list(self._words)
        suffix = tuple(words[-self.ngram :])

        # Count occurrences of the suffix n-gram in the window. Stop as
        # soon as we hit the threshold to avoid scanning the rest.
        count = 0
        for i in range(len(words) - self.ngram + 1):
            if tuple(words[i : i + self.ngram]) == suffix:
                count += 1
                if count >= self.threshold:
                    self._triggered = True
                    return True
        return False


# Open and close tags for the chain-of-thought block convention used by
# Qwen3, DeepSeek-R1, and other reasoning-tuned models. Lowercase only --
# none of the supported models emit uppercase variants.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class ThinkBlockStripper:
    """Stream-safe ``<think>...</think>`` block remover.

    Reasoning-tuned models (Qwen3, DeepSeek-R1, etc.) emit a chain-of-
    thought block before their actual answer when invoked through their
    native chat template. Inside a small ``max_tokens`` budget the block
    routinely consumes the entire budget, leaving no room for the
    answer. This stripper consumes those blocks before they reach the
    user.

    The stripper is *not* a regex pass over a complete string -- it must
    work on a token stream, which means:

    * The opening tag ``<think>`` may be split across two chunks (e.g.
      ``"<thi"`` then ``"nk>"``). The stripper buffers a small tail
      between chunks so a partial tag is never emitted as plain text.
    * The closing tag ``</think>`` is handled the same way.
    * Multiple ``<think>`` blocks per stream are supported (state
      machine flips back and forth).
    * Unclosed blocks (model never emits ``</think>``) discard the rest
      of the stream -- judged correct because the model is broken.
    * Nested blocks are *not* supported: a single ``</think>`` always
      closes the current scope. None of the supported models emit
      nested blocks in practice.

    Usage::

        stripper = ThinkBlockStripper()
        for chunk in token_iter:
            for cleaned in stripper.feed(chunk):
                yield cleaned
        for cleaned in stripper.flush():
            yield cleaned
    """

    def __init__(self) -> None:
        # Buffer holds chars we haven't yet decided what to do with.
        # While outside a think block this is content waiting to be
        # emitted (held back only because the trailing edge might be a
        # partial open tag). While inside, it's content waiting to be
        # discarded (held back only because the trailing edge might be a
        # partial close tag).
        self._buf = ""
        self._inside = False
        # When True, swallow leading whitespace from the next emitted
        # output until the first non-whitespace character. Set at
        # construction (so leading whitespace at the start of the stream
        # is consumed) and re-set on every inside->outside transition
        # (so the whitespace Qwen3 wraps around its <think> block --
        # typically `\n` after `</think>` -- doesn't survive into the
        # user-facing answer as a leading blank line).
        self._pending_lstrip = True

    def _clean(self, text: str) -> str:
        """Apply pending lstrip and return the (possibly empty) result.

        Used by every code path that's about to yield ``text``. Once a
        non-whitespace character has been seen, the lstrip flag is
        cleared and subsequent calls return ``text`` unchanged until the
        next inside->outside transition arms it again.
        """
        if not text or not self._pending_lstrip:
            return text
        text = text.lstrip()
        if text:
            self._pending_lstrip = False
        return text

    def feed(self, chunk: str) -> Iterator[str]:
        """Append a chunk and yield zero or more strings of cleaned output.

        The yielded strings, when concatenated, give the input stream
        with all complete ``<think>...</think>`` blocks removed and
        leading whitespace stripped from the start of each post-block
        segment. Any residual content held back to handle a possible
        partial tag at the boundary is yielded by a later :meth:`feed`
        or :meth:`flush`.
        """
        if not chunk:
            return
        self._buf += chunk

        while True:
            if not self._inside:
                # Outside a think block: look for the next opening tag.
                idx = self._buf.find(_THINK_OPEN)
                if idx == -1:
                    # No complete open tag in the buffer. Emit
                    # everything except a possible partial tag at the
                    # tail (the last len(_THINK_OPEN)-1 chars are kept
                    # back so we don't accidentally emit "<thi" as
                    # visible output that should have been part of an
                    # eventual "<think>").
                    safe = max(0, len(self._buf) - (len(_THINK_OPEN) - 1))
                    if safe > 0:
                        text = self._clean(self._buf[:safe])
                        self._buf = self._buf[safe:]
                        if text:
                            yield text
                    return
                # Found a full open tag. Emit everything before it,
                # discard the tag itself, and switch to inside.
                if idx > 0:
                    text = self._clean(self._buf[:idx])
                    if text:
                        yield text
                self._buf = self._buf[idx + len(_THINK_OPEN) :]
                self._inside = True
            else:
                # Inside a think block: look for the next closing tag.
                idx = self._buf.find(_THINK_CLOSE)
                if idx == -1:
                    # No complete close tag yet. Discard everything
                    # except the tail that might be a partial close tag.
                    safe = max(0, len(self._buf) - (len(_THINK_CLOSE) - 1))
                    self._buf = self._buf[safe:]
                    return
                # Found the close tag. Discard up through it, switch
                # back to outside, and re-arm the lstrip so the
                # whitespace immediately following the close tag is
                # swallowed (Qwen3 emits `\n` after `</think>`).
                self._buf = self._buf[idx + len(_THINK_CLOSE) :]
                self._inside = False
                self._pending_lstrip = True

    def flush(self) -> Iterator[str]:
        """Emit any held-back content at end of stream.

        Call this after the token iterator is exhausted. If the stream
        ended outside a think block, any buffered tail (which couldn't
        be a partial open tag because there are no more chunks coming)
        is emitted -- after applying any pending lstrip, in case the
        entire tail is leading whitespace. If it ended inside an
        unclosed block, the buffered content is dropped because we
        cannot prove it isn't reasoning the model never finished.
        """
        if not self._inside and self._buf:
            text = self._clean(self._buf)
            self._buf = ""
            if text:
                yield text
