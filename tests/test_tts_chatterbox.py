"""TTS through libmtmd with Chatterbox Turbo, a model with a predefined voice, checked with whisper."""

import gc

import pytest

from conftest import MODELS_DIR, ROOT
from test_tts_fix import WHISPER, _transcribe, _words

MODEL = MODELS_DIR / "chatterbox-turbo-Q8_0.gguf"
MMPROJ = MODELS_DIR / "mmproj-chatterbox-turbo-F16.gguf"
TEXT = "I say, how about some eggs?"
WAV_HEADER = 44

pytestmark = pytest.mark.skipif(
    not (MODEL.exists() and MMPROJ.exists()),
    reason="needs ggml-org/chatterbox-turbo-GGUF: chatterbox-turbo-Q8_0.gguf and mmproj-chatterbox-turbo-F16.gguf",
)


def test_voice_api():
    import cyllama.llama.llama_cpp as cy

    cy.ggml_backend_load_all()
    model_params = cy.LlamaModelParams()
    model_params.n_gpu_layers = 99
    model = cy.LlamaModel(str(MODEL), model_params)
    mtmd = cy.MtmdContext(str(MMPROJ), model)

    info = mtmd.gen_audio_info
    assert info["type"] == cy.MtmdGenAudioType.CHATTERBOX
    assert info["n_voices"] == 1
    assert mtmd.gen_audio_voices() == ["default"]
    # names match case-insensitively; the bitmap keeps the stored name
    voice = mtmd.gen_audio_voice("DEFAULT")
    assert voice.is_voice is True and voice.is_audio is False
    assert voice.id == "default"
    with pytest.raises(ValueError, match="available: \\['default'\\]"):
        mtmd.gen_audio_voice("giovanni")

    del voice
    mtmd.close()
    del mtmd, model
    gc.collect()


def test_default_voice_speaks(tmp_path):
    from cyllama.llama.tts import MtmdTTSGenerator

    wavs = []
    for kwargs in ({}, {"voice": "default"}):
        with MtmdTTSGenerator(str(MODEL), str(MMPROJ), ngl=99, seed=1234, **kwargs) as gen:
            assert gen.speaker is not None and gen.speaker.is_voice
            wavs.append(gen.generate(TEXT))
        gc.collect()
    # without the model's suppress tokens, <|endoftext|> is sampled first and the WAV is header-only
    assert len(wavs[0]) > WAV_HEADER
    # no voice given: the first predefined voice is used
    assert wavs[0] == wavs[1]

    if not WHISPER.exists():
        pytest.skip("ggml-base.en.bin not downloaded; audio written but words unchecked")
    out = tmp_path / "default.wav"
    out.write_bytes(wavs[0])
    assert _words(_transcribe(out)) == _words(TEXT)


def test_speaker_file_replaces_voice():
    from cyllama.llama.tts import MtmdTTSGenerator

    speaker = ROOT / "tests" / "samples" / "jfk.wav"
    wavs = []
    for kwargs in ({}, {"speaker_file": str(speaker)}):
        with MtmdTTSGenerator(str(MODEL), str(MMPROJ), ngl=99, seed=1234, **kwargs) as gen:
            wavs.append(gen.generate(TEXT))
            is_voice = gen.speaker.is_voice
        gc.collect()
    assert is_voice is False
    assert len(wavs[1]) > WAV_HEADER
    # same seed and text, so only the reference can make them differ.
    # words are not checked: upstream llama-tts also renders "eggs" as "age" with this reference.
    assert wavs[0] != wavs[1]
