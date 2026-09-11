"""Incremental detokenization for streamed generation.

Byte-level BPE vocabularies can split one UTF-8 character across several
tokens. Decoding each token on its own turns every fragment into U+FFFD, so
generation loops push tokens through ``TokenDecoder``, which holds the bytes
of an incomplete character until the next token completes it.
"""

from __future__ import annotations

import codecs

from .llama_cpp import LlamaVocab


class TokenDecoder:
    """Decodes a stream of token ids to text, one token at a time."""

    def __init__(self, vocab: LlamaVocab) -> None:
        self._vocab = vocab
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def decode(self, token: int) -> str:
        """Return the text completed by ``token``; may be empty."""
        return self._decoder.decode(self._vocab.token_to_bytes(token, special=True))

    def flush(self) -> str:
        """Return held bytes of an unfinished character as U+FFFD, and reset."""
        return self._decoder.decode(b"", final=True)

    def reset(self) -> None:
        """Discard held bytes."""
        self._decoder.reset()
