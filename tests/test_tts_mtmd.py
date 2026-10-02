"""TTS through libmtmd audio generation (Qwen3-TTS), checked with whisper."""

import gc
import wave

import pytest

from conftest import MODELS_DIR, ROOT
from test_tts_fix import WHISPER, _transcribe, _words

MODEL = MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"
MMPROJ = MODELS_DIR / "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"
TEXT = "I say, how about some eggs?"

pytestmark = pytest.mark.skipif(
    not (MODEL.exists() and MMPROJ.exists()), reason="needs Qwen3-TTS-12Hz-1.7B-Base model and mmproj"
)


def test_generate_speech(tmp_path):
    from cyllama.llama.tts import MtmdTTSGenerator

    out = tmp_path / "out.wav"
    # the second utterance reuses the context; with a stale KV cache its prompt decode fails
    out2 = tmp_path / "out2.wav"
    with MtmdTTSGenerator(str(MODEL), str(MMPROJ), ngl=99, seed=1234) as gen:
        assert gen.sample_rate == 24000
        wav = gen.generate(TEXT, str(out))
        gen.generate(TEXT, str(out2))
    gc.collect()

    assert out.read_bytes() == wav
    with wave.open(str(out)) as w:
        assert w.getframerate() == 24000
        seconds = w.getnframes() / w.getframerate()
    assert 0.5 < seconds < 10.0

    if not WHISPER.exists():
        pytest.skip("ggml-base.en.bin not downloaded; audio written but words unchecked")
    assert _words(_transcribe(out)) == _words(TEXT)
    assert _words(_transcribe(out2)) == _words(TEXT)


def test_seed_is_reproducible():
    from cyllama.llama.tts import MtmdTTSGenerator

    wavs = []
    for _ in range(2):
        with MtmdTTSGenerator(str(MODEL), str(MMPROJ), ngl=99, seed=1234) as gen:
            wavs.append(gen.generate(TEXT))
        gc.collect()
    assert wavs[0] == wavs[1]


def test_audio_generator_api():
    import cyllama.llama.llama_cpp as cy

    cy.ggml_backend_load_all()
    model_params = cy.LlamaModelParams()
    model_params.n_gpu_layers = 99
    model = cy.LlamaModel(str(MODEL), model_params)
    ctx_params = cy.LlamaContextParams()
    ctx_params.n_ctx = 1024
    ctx = cy.LlamaContext(model, ctx_params)
    mtmd = cy.MtmdContext(str(MMPROJ), model)

    info = mtmd.gen_audio_info
    assert info["type"] == cy.MtmdGenAudioType.QWEN3TTS
    assert info["sample_rate"] == 24000

    gen = cy.MtmdAudioGenerator(ctx, mtmd)
    gen.set_input(TEXT)
    with pytest.raises(RuntimeError, match="step_prompt"):
        gen.step_gen(0)
    while gen.step_prompt(512) > 0:
        pass
    # embeddings mode is off, so there is no hidden state to feed the mmproj
    with pytest.raises(RuntimeError, match="embeddings"):
        gen.step_gen(0)

    gen.close()
    gen.close()
    with pytest.raises(RuntimeError, match="closed"):
        gen.get_output()

    mtmd.close()
    del gen, mtmd, ctx, model
    gc.collect()


def test_speaker_file_changes_voice(tmp_path):
    from cyllama.llama.tts import MtmdTTSGenerator

    speaker = ROOT / "tests" / "samples" / "jfk.wav"
    wavs = []
    for speaker_file in (None, str(speaker)):
        with MtmdTTSGenerator(str(MODEL), str(MMPROJ), ngl=99, seed=1234, speaker_file=speaker_file) as gen:
            wavs.append(gen.generate(TEXT))
        gc.collect()
    # same seed and text, so only the speaker embedding can make them differ
    assert wavs[0] != wavs[1]

    if not WHISPER.exists():
        pytest.skip("ggml-base.en.bin not downloaded; audio written but words unchecked")
    out = tmp_path / "cloned.wav"
    out.write_bytes(wavs[1])
    assert _words(_transcribe(out)) == _words(TEXT)
