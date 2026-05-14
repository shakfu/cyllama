"""``word_count`` -- multi-statistic string analyzer. Demonstrates
docstring-driven schema generation for a single ``text`` parameter and a
multi-field return.
"""

from typing import Dict

from .core import tool


@tool
def word_count(text: str) -> Dict[str, int]:
    """Count words, characters, and lines in a piece of text.

    Args:
        text: The text to analyse. Any string is accepted; an empty
            string returns zero counts.

    Returns:
        A dict with ``characters`` (length in code points), ``words``
        (whitespace-separated tokens), and ``lines`` (number of newline
        terminators, plus one if the final line is unterminated).
    """
    chars = len(text)
    words = len(text.split())
    if not text:
        lines = 0
    else:
        lines = text.count("\n") + (0 if text.endswith("\n") else 1)
    return {"characters": chars, "words": words, "lines": lines}
