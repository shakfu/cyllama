"""irfft length handling, and encode return codes.

`irfft` once took its length from the input: a WavTokenizer row is 1282
floats, so it ran a 1282-point transform that read 1284 floats, two past
the end of the buffer. Whatever lay there entered every frame, and once
it was about 1e18 (tests/test_tts_vocoder.py failed only on a busy heap).
"""

import math
import random

import pytest

import cyllama.llama.llama_cpp as cy

from conftest import MODELS_DIR

N_FFT = 1280
ROW = 1282  # WavTokenizer n_embd_out: 641 complex bins


def reference_irfft(inp, n):
    """The transform irfft implements, in double precision."""
    bins = n // 2 + 1
    re, im = inp[0 : 2 * bins : 2], inp[1 : 2 * bins : 2]
    return [
        sum(re[m] * math.cos(2 * math.pi * k * m / n) - im[m] * math.sin(2 * math.pi * k * m / n) for m in range(bins))
        / bins
        for k in range(n)
    ]


def test_returns_n_samples():
    assert len(cy.irfft([0.0] * ROW, N_FFT)) == N_FFT


def test_zero_input_gives_exactly_zero():
    """Before the fix, stray bytes past the input made this nonzero on some calls."""
    for _ in range(2000):
        junk = [float(i) * 1e15 for i in range(3000)]  # leave non-zero floats on the heap
        assert not any(cy.irfft([0.0] * ROW, N_FFT))
        del junk


def test_matches_reference():
    rng = random.Random(7)
    n = 64
    inp = [rng.uniform(-1, 1) for _ in range(2 * (n // 2 + 1))]
    out = cy.irfft(inp, n)
    assert out == pytest.approx(reference_irfft(inp, n), abs=1e-4)


def test_ignores_floats_past_the_bins():
    rng = random.Random(3)
    row = [rng.uniform(-1, 1) for _ in range(ROW)]
    assert cy.irfft(row + [1e30, -1e30], N_FFT) == cy.irfft(row, N_FFT)


@pytest.mark.parametrize(("size", "n"), [(ROW - 1, N_FFT), (ROW, 0), (ROW, -2)])
def test_rejects_short_input_or_bad_length(size, n):
    with pytest.raises(ValueError):
        cy.irfft([0.0] * size, n)


WAVTOKENIZER = MODELS_DIR / "WavTokenizer-Large-75-Q5_1.gguf"


@pytest.mark.skipif(not WAVTOKENIZER.exists(), reason=f"{WAVTOKENIZER.name} not downloaded")
def test_aborted_encode_raises():
    """llama_encode returns 2 when aborted; the output buffer is then stale."""
    cy.llama_backend_init()
    model = cy.LlamaModel(str(WAVTOKENIZER), verbose=False)
    params = cy.LlamaContextParams()
    params.n_ctx = params.n_batch = params.n_ubatch = 512
    ctx = cy.LlamaContext(model, params, verbose=False)
    ctx.set_embeddings_mode(True)
    ctx.install_cancel_callback()
    ctx.cancel = True
    batch = cy.LlamaBatch(n_tokens=8, embd=0, n_seq_max=1)
    batch.add_sequence(list(range(8)), 0, True)
    with pytest.raises(InterruptedError):
        ctx.encode(batch)
