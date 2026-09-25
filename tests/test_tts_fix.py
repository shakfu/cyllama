"""End-to-end TTS: OuteTTS 0.2/0.3 text-to-codes, WavTokenizer vocoder, and
whisper to check the words that come out.

The check used to be only that a non-empty WAV was written, which an
incompatible model passes by writing noise.
"""

import re
import wave

import pytest

from conftest import MODELS_DIR

np = pytest.importorskip("numpy")

VOCODER = MODELS_DIR / "WavTokenizer-Large-75-F16.gguf"
WHISPER = MODELS_DIR / "ggml-base.en.bin"
TEXT = "Hello world, this is a test of text to speech."


def _ttc_model():
    for pattern in ("tts.gguf", "OuteTTS-0.3*.gguf", "OuteTTS-0.2*.gguf"):
        found = sorted(MODELS_DIR.glob(pattern))
        if found:
            return found[0]
    return None


def _words(text):
    return re.findall(r"[a-z]+", text.lower())


def _transcribe(wav_path):
    from cyllama.whisper import whisper_cpp as wh

    with wave.open(str(wav_path)) as w:
        rate = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    t16 = np.arange(0, len(x) / rate, 1 / 16000)
    x16 = np.interp(t16, np.arange(len(x)) / rate, x).astype(np.float32)

    wh.disable_logging()
    ctx = wh.WhisperContext(str(WHISPER), wh.WhisperContextParams())
    params = wh.WhisperFullParams()
    params.language = "en"
    params.print_progress = False
    params.print_realtime = False
    params.print_timestamps = False
    ctx.full(x16, params)
    return " ".join(ctx.full_get_segment_text(i) for i in range(ctx.full_n_segments()))


def test_tts_generation(tmp_path):
    from cyllama.llama.tts import TTSGenerator

    ttc = _ttc_model()
    if ttc is None or not VOCODER.exists():
        pytest.skip("needs an OuteTTS 0.2/0.3 model and WavTokenizer-Large-75-F16.gguf")

    tts = TTSGenerator(
        ttc_model_path=str(ttc),
        cts_model_path=str(VOCODER),
        n_ctx=8192,
        n_batch=8192,
        ngl=99,
        n_predict=1000,
        use_guide_tokens=True,
    )
    out = tmp_path / "out.wav"
    assert tts.generate(TEXT, str(out))

    with wave.open(str(out)) as w:
        seconds = w.getnframes() / w.getframerate()
    assert 1.0 < seconds < 10.0

    if not WHISPER.exists():
        pytest.skip("ggml-base.en.bin not downloaded; audio written but words unchecked")
    assert _words(_transcribe(out)) == _words(TEXT)


def test_rejects_outetts_1():
    """OuteTTS 1.0 uses a different codec; its audio codes are not <|i|> tokens."""
    from cyllama.llama.tts import TTSGenerator

    found = sorted(MODELS_DIR.glob("OuteTTS-1.0*.gguf"))
    if not found or not VOCODER.exists():
        pytest.skip("needs an OuteTTS 1.0 model and WavTokenizer-Large-75-F16.gguf")
    with pytest.raises(ValueError, match="OuteTTS 0.2/0.3"):
        TTSGenerator(str(found[0]), str(VOCODER), n_ctx=512, n_batch=512, ngl=0)
