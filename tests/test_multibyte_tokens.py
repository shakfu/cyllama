"""Multi-byte UTF-8 characters that span several tokens.

Byte-level BPE vocabularies split some characters across tokens:
Llama-3.2 encodes U+1F4D0 (a 4-byte emoji) as three tokens, none of
which is valid UTF-8 on its own. Decoding each token separately turns
every fragment into U+FFFD.
"""

import gc

from cyllama import GenerationConfig
from cyllama.llama.token_decoder import TokenDecoder

EMOJI = "\U0001f4d0"
REPLACEMENT = "\ufffd"


class _ScriptedSampler:
    """Returns a fixed token sequence in place of sampling."""

    def __init__(self, token_ids):
        self._ids = iter(token_ids)

    def sample(self, ctx, idx):
        return next(self._ids)


def _script(llm, text):
    """Token ids that spell ``text``, then end-of-generation."""
    ids = llm.vocab.tokenize(text, add_special=False, parse_special=False)
    llm._ensure_sampler = lambda config: _ScriptedSampler(ids + [llm.vocab.token_eos()])
    return ids


class _ByteVocab:
    """Vocab whose token ``i`` is ``pieces[i]``."""

    def __init__(self, pieces):
        self._pieces = pieces

    def token_to_bytes(self, token, lstrip=0, special=False):
        return self._pieces[token]


# U+1F4D0 split into three fragments, then a plain ASCII token.
_SPLIT = [b"\xf0\x9f", b"\x93", b"\x90", b" ok"]


def test_token_decoder_holds_fragments_until_complete():
    decoder = TokenDecoder(_ByteVocab(_SPLIT))
    assert [decoder.decode(t) for t in range(4)] == ["", "", EMOJI, " ok"]
    assert decoder.flush() == ""


def test_token_decoder_flush_emits_replacement_for_unfinished_character():
    decoder = TokenDecoder(_ByteVocab(_SPLIT))
    assert decoder.decode(0) + decoder.decode(1) == ""
    assert decoder.flush() == REPLACEMENT
    # flush() resets, so the next token decodes cleanly.
    assert decoder.decode(3) == " ok"


def test_token_decoder_reset_discards_held_bytes():
    decoder = TokenDecoder(_ByteVocab(_SPLIT))
    decoder.decode(0)
    decoder.reset()
    assert decoder.decode(3) == " ok"
    assert decoder.flush() == ""


def test_emoji_spans_several_tokens(llm):
    # Precondition for the tests below: the fragments are not valid UTF-8.
    ids = llm.vocab.tokenize(EMOJI, add_special=False, parse_special=False)
    assert len(ids) > 1
    assert all(REPLACEMENT in llm.vocab.token_to_piece(t) for t in ids)


def test_token_to_bytes_returns_the_raw_fragments(llm):
    ids = llm.vocab.tokenize(EMOJI, add_special=False, parse_special=False)
    raw = b"".join(llm.vocab.token_to_bytes(t) for t in ids)
    assert raw == EMOJI.encode("utf-8")


def test_stream_reassembles_a_split_character(llm):
    text = f" {EMOJI} ok"
    _script(llm, text)
    chunks = list(llm("x", config=GenerationConfig(max_tokens=16), stream=True))
    assert "".join(chunks) == text
    assert "" not in chunks
    assert not any(REPLACEMENT in c for c in chunks)


def test_non_stream_reassembles_a_split_character(llm):
    text = f" {EMOJI} ok"
    _script(llm, text)
    assert llm("x", config=GenerationConfig(max_tokens=16)).text == text


def test_stream_with_stop_sequence_reassembles_a_split_character(llm):
    _script(llm, f" {EMOJI} ok STOP tail")
    config = GenerationConfig(max_tokens=16, stop_sequences=["STOP"])
    assert "".join(llm("x", config=config, stream=True)) == f" {EMOJI} ok "


def test_character_cut_off_by_max_tokens_flushes_as_replacement(llm):
    ids = _script(llm, f" {EMOJI}")
    cut = b"".join(llm.vocab.token_to_bytes(t) for t in ids[:-1])
    config = GenerationConfig(max_tokens=len(ids) - 1)
    text = "".join(llm("x", config=config, stream=True))
    assert text == cut.decode("utf-8", errors="replace")
    assert text.endswith(REPLACEMENT)


def test_grammar_constrained_output_reassembles_a_split_character(model_path):
    from cyllama.agents.constrained import GrammarConstrainedLLM

    chunks = []
    with GrammarConstrainedLLM(model_path) as llm:
        out = llm.generate_with_grammar(
            "Say it.",
            f'root ::= "{EMOJI} ok"',
            config=GenerationConfig(max_tokens=16, temperature=0.0),
            on_token=chunks.append,
        )
    del llm
    gc.collect()
    assert out == f"{EMOJI} ok"
    assert "".join(chunks) == out
    assert not any(REPLACEMENT in c for c in chunks)
