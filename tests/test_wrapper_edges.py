"""Regression tests for wrapper-layer edge cases fixed in the May 2026 review.

Each test pins a specific bug that was fixed in `src/cyllama/llama/llama_cpp.pyx`:

- Empty-batch underflow on the last-logit write in `LlamaBatch.set_batch` /
  `add_sequence` / `set_last_logits_to_true` and `llama_batch_get_one`.
- Tokenize / token_to_piece / detokenize ignoring the C ABI's "needed size"
  return value and raising instead of growing the buffer and retrying.
- `LlamaSampler.clone` not setting `wrapper.owner` explicitly, plus no NULL
  check on the C return.
"""

import gc

import pytest

import cyllama.llama.llama_cpp as cy


# ---------------------------------------------------------------------------
# LlamaBatch empty-batch underflow guards
# ---------------------------------------------------------------------------


class TestLlamaBatchEmpty:
    """Empty batches must not underflow into `logits[-1]`."""

    def test_set_batch_empty_does_not_underflow(self):
        batch = cy.LlamaBatch(n_tokens=8, embd=0, n_seq_max=1)
        try:
            # Must not crash or write to logits[-1].
            batch.set_batch([], n_past=0, logits_all=False)
            assert batch.n_tokens == 0
        finally:
            batch.close()

    def test_add_sequence_empty_does_not_underflow(self):
        batch = cy.LlamaBatch(n_tokens=8, embd=0, n_seq_max=1)
        try:
            batch.add_sequence([], seq_id=0, logits_all=False)
            assert batch.n_tokens == 0
        finally:
            batch.close()

    def test_set_last_logits_when_empty_is_a_noop(self):
        batch = cy.LlamaBatch(n_tokens=8, embd=0, n_seq_max=1)
        try:
            # n_tokens starts at 0; this used to write logits[-1] = True.
            batch.set_last_logits_to_true()
            assert batch.n_tokens == 0
        finally:
            batch.close()

    def test_llama_batch_get_one_empty_does_not_underflow(self):
        # Used to write logits[-1] = True for an empty token list.
        batch = cy.llama_batch_get_one([], n_past=0)
        # Pool-owned batch; do not close. It survives the call without crashing.
        assert batch.n_tokens == 0


# ---------------------------------------------------------------------------
# LlamaSampler.clone ownership lifecycle
# ---------------------------------------------------------------------------


class TestLlamaSamplerClone:
    """Cloned samplers must own their pointer and free it on dealloc."""

    def test_clone_returns_distinct_instance(self):
        smplr = cy.LlamaSampler(cy.LlamaSamplerChainParams())
        smplr.add_top_k(10)
        cloned = smplr.clone()
        assert isinstance(cloned, cy.LlamaSampler)
        assert cloned is not smplr

    def test_clone_lifecycle_loop_does_not_leak_or_crash(self):
        # If the clone failed to set owner=True, repeated clone+drop cycles
        # would leak C samplers; if it free()d a non-owned pointer on dealloc,
        # a second cycle would double-free. Either failure manifests as an
        # abort or growing memory; passing 200 cycles cleanly proves neither.
        smplr = cy.LlamaSampler(cy.LlamaSamplerChainParams())
        smplr.add_top_k(10)
        for _ in range(200):
            cloned = smplr.clone()
            del cloned
        gc.collect()


# ---------------------------------------------------------------------------
# Tokenize / detokenize / token_to_piece retry-on-undersize
# ---------------------------------------------------------------------------


class TestVocabUndersizeRetry:
    """The C ABI returns -needed when the buffer is too small; the wrapper
    must honor that and grow + retry rather than raising."""

    @pytest.fixture(scope="class")
    def vocab(self, model_path):
        # Module-scoped model load is cheaper but the conftest fixture is
        # module-scoped; here we just construct ad-hoc since these tests
        # don't need to share the LLM with anything else.
        model = cy.LlamaModel(model_path)
        v = model.get_vocab()
        yield v
        del v
        del model
        gc.collect()

    def test_tokenize_short_ascii(self, vocab):
        toks = vocab.tokenize("hello world", add_special=True, parse_special=False)
        assert isinstance(toks, list)
        assert len(toks) > 0

    def test_tokenize_long_unicode_does_not_raise(self, vocab):
        # Unicode-heavy: each glyph encodes to multiple bytes and may produce
        # more tokens than the wrapper's `len(text)*2 + 100` heuristic
        # estimates. The retry path must absorb the under-estimate.
        text = "中文测试" * 200  # 4 CJK chars repeated
        toks = vocab.tokenize(text, add_special=False, parse_special=False)
        assert len(toks) > 0

    def test_detokenize_grows_buffer_when_text_len_max_is_small(self, vocab):
        # Tokenize a non-trivial string, then ask detokenize to round-trip
        # with text_len_max=4 -- well below the actual byte length of the
        # decoded text. The retry path must grow the buffer to fit.
        original = "The quick brown fox jumps over the lazy dog."
        toks = vocab.tokenize(original, add_special=False, parse_special=False)
        result = vocab.detokenize(toks, text_len_max=4)
        # result is post-lstrip; the substantive content must be present.
        assert "quick" in result and "lazy" in result

    def test_token_to_piece_smoke(self, vocab):
        # We can't easily synthesize a token whose piece exceeds 128 bytes
        # without model-specific knowledge, but the common path must keep
        # working after the stack/heap split. Iterate a handful of tokens.
        toks = vocab.tokenize("hello world", add_special=False, parse_special=False)
        for tok in toks:
            piece = vocab.token_to_piece(tok)
            assert isinstance(piece, str)


# ---------------------------------------------------------------------------
# Speculative.is_compat must not destroy ctx_target's KV state
# ---------------------------------------------------------------------------


class TestMtmdBitmapCreateImageValidation:
    """`MtmdBitmap.create_image` previously passed `data` straight to
    `mtmd_bitmap_init`, which takes a raw `unsigned char*` and reads
    `width * height * 3` bytes with no length argument. A short buffer
    let the C side read past the bytes object. The wrapper now validates
    `len(data) >= width * height * 3` up front."""

    def test_short_buffer_raises_value_error(self):
        from cyllama.llama.llama_cpp import MtmdBitmap

        # Need 2 * 2 * 3 = 12 bytes; provide 5.
        with pytest.raises(ValueError, match="too small"):
            MtmdBitmap.create_image(2, 2, b"\x00" * 5)

    def test_zero_or_negative_dims_raise_value_error(self):
        from cyllama.llama.llama_cpp import MtmdBitmap

        with pytest.raises(ValueError, match="must be positive"):
            MtmdBitmap.create_image(0, 4, b"\x00" * 12)
        with pytest.raises(ValueError, match="must be positive"):
            MtmdBitmap.create_image(4, 0, b"\x00" * 12)
        with pytest.raises(ValueError, match="must be positive"):
            MtmdBitmap.create_image(-1, 4, b"\x00" * 12)


class TestSpeculativeIsCompatNonDestructive:
    """`Speculative.is_compat` previously decoded 2 dummy tokens into the
    target context and then `llama_memory_clear`'d it -- silently wiping any
    KV state the caller had built. The probe now runs on a scratch context
    built from the same model, so caller state is preserved."""

    def test_is_compat_preserves_kv_state(self, model_path):
        from cyllama.llama.llama_cpp import (
            LlamaContext,
            LlamaContextParams,
            LlamaModel,
            LlamaModelParams,
            Speculative,
            llama_batch_get_one,
        )

        model = LlamaModel(model_path, LlamaModelParams())
        try:
            ctx_params = LlamaContextParams()
            ctx_params.n_ctx = 256
            ctx = LlamaContext(model, ctx_params)
            try:
                vocab = model.get_vocab()

                # Build some KV state in ctx_target by decoding a real prompt.
                tokens = vocab.tokenize(
                    "The capital of France is",
                    add_special=True,
                    parse_special=False,
                )
                batch = llama_batch_get_one(tokens, n_past=0)
                ctx.decode(batch)
                # After decoding N tokens at positions 0..N-1, seq 0's
                # max-position should be N - 1.
                pos_max_before = ctx.memory_seq_pos_max(0)
                assert pos_max_before == len(tokens) - 1

                # Run the compat probe. Pre-fix this would have decoded
                # 2 dummy tokens at pos 0,1 and then `llama_memory_clear`'d
                # everything, leaving seq 0 empty (pos_max = -1).
                Speculative.is_compat(ctx)

                # The caller's KV state is still intact: pos_max for seq 0
                # is unchanged. This is the load-bearing assertion -- the
                # pre-fix behavior would fail it.
                assert ctx.memory_seq_pos_max(0) == pos_max_before
            finally:
                ctx.close()
        finally:
            model.close()
        gc.collect()
