"""N-gram repetition detection for streaming generation.

The high-level RAG pipeline routes generated tokens through this detector
so that chat-tuned models which loop on a paraphrased answer (Qwen3-4B is
the canonical offender) can be cut off early instead of running until
``max_tokens``.

The detector operates on word-normalised text -- lowercase, ``\\w+`` tokens
only -- so trailing punctuation, capitalisation, and whitespace
differences do not defeat it. It maintains a rolling window of the most
recent words and, after each chunk, checks whether the trailing n-gram
has occurred at least ``threshold`` times within the window. If so,
:meth:`feed` returns True and the caller should stop generation.

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
from typing import Deque

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
