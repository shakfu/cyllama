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
    assert cy.get_mmproj_caps(str(MMPROJ))["gen_audio"] is True
    caps = mtmd.capabilities
    assert caps["audio_gen"] is True and caps["audio_gen_spk_ref"] is True
    assert caps["vision"] is False and caps["audio_stream"] is False

    # the Base variant clones from a reference clip and has no predefined voices
    assert info["n_voices"] == 0
    assert mtmd.gen_audio_voices() == []
    with pytest.raises(ValueError, match="unknown voice"):
        mtmd.gen_audio_voice("female")
    ref = cy.MtmdBitmap.from_file(mtmd, str(ROOT / "tests" / "samples" / "jfk.wav"))
    assert ref.is_audio is True and ref.is_voice is False
    del ref

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


def test_stream_output():
    from cyllama.llama.tts import MtmdTTSGenerator

    # longer than one 72-frame (6 s) code2wav window, so audio arrives before flush()
    text = " ".join([TEXT] * 4)
    with MtmdTTSGenerator(str(MODEL), str(MMPROJ), ngl=99, seed=1234) as gen:
        chunks = list(gen.stream(text))
        assert gen.generator.get_output() == b""  # nothing new after the final chunk
        # generate() still works after a stream on the same context
        assert gen.generate(TEXT)[:4] == b"RIFF"
    gc.collect()

    # flush() adds at most one chunk, so two or more means streaming happened
    assert len(chunks) >= 2 and all(chunks)
    head = chunks[0]
    assert head[:4] == b"RIFF" and head[36:40] == b"data"
    assert head[40:44] == b"\xff\xff\xff\xff"  # unknown-length sentinel
    assert all(not c.startswith(b"RIFF") for c in chunks[1:])

    pcm = b"".join(chunks)[44:]
    assert len(pcm) / 2 / 24000 > 6.0


def test_voice_arguments_are_validated():
    from cyllama.llama.tts import MtmdTTSGenerator

    with pytest.raises(ValueError, match="cannot be used together"):
        MtmdTTSGenerator(str(MODEL), str(MMPROJ), voice="female", speaker_file="x.wav")
    with pytest.raises(ValueError, match="no predefined voices"):
        MtmdTTSGenerator(str(MODEL), str(MMPROJ), ngl=99, voice="female")
    gc.collect()


def test_continuation_api_on_a_tts_only_pipeline():
    import cyllama.llama.llama_cpp as cy

    cy.ggml_backend_load_all()
    model_params = cy.LlamaModelParams()
    model_params.n_gpu_layers = 99
    model = cy.LlamaModel(str(MODEL), model_params)
    ctx_params = cy.LlamaContextParams()
    ctx_params.n_ctx = 1024
    ctx = cy.LlamaContext(model, ctx_params)
    ctx.set_embeddings_mode(True)
    mtmd = cy.MtmdContext(str(MMPROJ), model)
    gen = cy.MtmdAudioGenerator(ctx, mtmd)

    with pytest.raises(ValueError, match="requires n_past"):
        gen.set_input(TEXT, out_mode=cy.MtmdGenAudioOutMode.TEXT)
    with pytest.raises(ValueError, match="n_past"):
        gen.set_input(TEXT, n_past=-1)
    with pytest.raises(ValueError):
        gen.set_input(TEXT, n_past=4, out_mode=99)
    # only LFM3-Audio continues a sequence
    with pytest.raises(RuntimeError, match="only LFM3-Audio"):
        gen.set_input(TEXT, n_past=4)

    gen.set_input(TEXT)
    assert gen.get_logit_bias() == []
    while gen.step_prompt(512) > 0:
        pass
    # a TTS utterance has no text steps
    assert gen.next_is_text() is False

    gen.close()
    mtmd.close()
    del gen, mtmd, ctx, model
    gc.collect()
