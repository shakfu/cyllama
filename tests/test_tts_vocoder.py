"""WavTokenizer vocoder stage of TTSGenerator: audio codes to waveform.

Decodes the default speaker's recorded codes, so it needs only the
WavTokenizer model, not the OuteTTS text-to-codes model. WavTokenizer's
output rows are n_embd_out (1282) floats wide, not n_embd (512); reading
512 made embd_to_audio raise, and codes_to_audio returned no audio.
"""

import math
import re

import pytest

import cyllama.llama.llama_cpp as cy
from cyllama.llama.tts import TTSGenerator

from conftest import MODELS_DIR

np = pytest.importorskip("numpy")

MODELS = {q: MODELS_DIR / f"WavTokenizer-Large-75-{q}.gguf" for q in ("F16", "Q5_1")}
HOP = 320
SAMPLE_RATE = 24000


def _speaker_codes():
    g = TTSGenerator.__new__(TTSGenerator)
    g.tts_version = "0.2"
    g.setup_default_speaker()
    blocks = re.findall(r"<\|code_start\|>(.*?)<\|code_end\|>", g.audio_data)
    return [int(c) for blk in blocks for c in re.findall(r"<\|(\d+)\|>", blk)]


def _vocode(path, codes):
    cy.llama_backend_init()
    model = cy.LlamaModel(str(path), verbose=False)
    params = cy.LlamaContextParams()
    params.n_ctx = params.n_batch = params.n_ubatch = 8192
    ctx = cy.LlamaContext(model, params, verbose=False)
    ctx.set_embeddings_mode(True)
    g = TTSGenerator.__new__(TTSGenerator)
    g.model_cts, g.context_cts = model, ctx
    return np.asarray(g.codes_to_audio(codes))


@pytest.fixture(scope="module")
def codes():
    return _speaker_codes()


@pytest.fixture(scope="module")
def audio(codes):
    out = {q: _vocode(p, codes) for q, p in MODELS.items() if p.exists()}
    if not out:
        pytest.skip("no WavTokenizer model downloaded")
    return out


@pytest.mark.parametrize("quant", MODELS)
def test_decodes_speech(audio, codes, quant):
    if quant not in audio:
        pytest.skip(f"{MODELS[quant].name} not downloaded")
    a = audio[quant]
    model = cy.LlamaModel(str(MODELS[quant]), verbose=False)
    assert model.n_embd_out != model.n_embd  # the case the width fix is for

    assert len(a) == len(codes) * HOP
    assert np.all(np.isfinite(a))
    assert 0.05 < np.max(np.abs(a)) < 1.5
    assert math.sqrt(np.mean(a**2)) > 0.01

    # speech: nearly all energy below 4 kHz
    power = np.abs(np.fft.rfft(a)) ** 2
    freqs = np.fft.rfftfreq(len(a), 1 / SAMPLE_RATE)
    assert power[freqs < 4000].sum() / power.sum() > 0.9


def test_quantizations_agree(audio):
    if len(audio) < 2:
        pytest.skip("needs both WavTokenizer models")
    assert np.corrcoef(audio["F16"], audio["Q5_1"])[0, 1] > 0.95
