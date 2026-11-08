"""LFM3-Audio through libmtmd: TTS from the "tts" chat template, and continuing a conversation, checked with whisper."""

import gc

import pytest

import cyllama.llama.llama_cpp as cy
from conftest import MODELS_DIR
from test_tts_fix import WHISPER, _transcribe, _words

MODEL = MODELS_DIR / "LFM3-Audio-2B-Q4_K_M.gguf"
MMPROJ = MODELS_DIR / "mmproj-LFM3-Audio-2B-Q8_0.gguf"
TEXT = "I say, how about some eggs?"
WAV_HEADER = 44

pytestmark = pytest.mark.skipif(
    not (MODEL.exists() and MMPROJ.exists()),
    reason="needs LiquidAI/LFM3-Audio-2B-GGUF: LFM3-Audio-2B-Q4_K_M.gguf and mmproj-LFM3-Audio-2B-Q8_0.gguf",
)


def _heard(tmp_path, wav, name):
    if not WHISPER.exists():
        pytest.skip("ggml-base.en.bin not downloaded; audio written but words unchecked")
    out = tmp_path / name
    out.write_bytes(wav)
    return _words(_transcribe(out))


def test_tts_renders_the_tts_template(tmp_path):
    from cyllama.llama.tts import MtmdTTSGenerator

    with MtmdTTSGenerator(str(MODEL), str(MMPROJ), ngl=99, seed=1234) as gen:
        assert gen.mtmd.gen_audio_info["type"] == cy.MtmdGenAudioType.LFM3A
        assert gen.mtmd.gen_audio_voices() == ["female", "male"]
        # without the template the pipeline rejects the prompt: it must end in <|audio_start|>
        assert gen.render_prompt(TEXT) == (
            "<|startoftext|><|im_start|>system\nPerform TTS in American English using a female voice.<|im_end|>\n"
            f"<|im_start|>user\n{TEXT}<|im_end|>\n<|im_start|>assistant\n<|audio_start|>"
        )
        wav = gen.generate(TEXT)
    gc.collect()
    assert len(wav) > WAV_HEADER
    assert _heard(tmp_path, wav, "female.wav") == _words(TEXT)


def test_tts_voice_and_language_reach_the_template():
    from cyllama.llama.tts import MtmdTTSGenerator

    with MtmdTTSGenerator(str(MODEL), str(MMPROJ), ngl=99, seed=1234, voice="male", lang="fr") as gen:
        assert "Perform TTS in French using a male voice." in gen.render_prompt("Bonjour")
    gc.collect()


@pytest.fixture(scope="module")
def chat():
    cy.ggml_backend_load_all()
    model_params = cy.LlamaModelParams()
    model_params.n_gpu_layers = 99
    model = cy.LlamaModel(str(MODEL), model_params)
    ctx_params = cy.LlamaContextParams()
    ctx_params.n_ctx = 2048
    ctx = cy.LlamaContext(model, ctx_params)
    ctx.set_embeddings_mode(True)
    mtmd = cy.MtmdContext(str(MMPROJ), model)
    gen = cy.MtmdAudioGenerator(ctx, mtmd)
    yield model, ctx, gen
    gen.close()
    mtmd.close()
    ctx.close()
    model.close()
    del gen, mtmd, ctx, model
    gc.collect()


def _respond(chat, system, question, mode):
    """Decode a system turn, then continue it with the user turn. Returns (steps, text, wav, bias)."""
    model, ctx, gen = chat
    vocab = model.get_vocab()
    ctx.kv_cache_clear()
    head = vocab.tokenize(
        f"<|startoftext|><|im_start|>system\n{system}<|im_end|>\n", add_special=False, parse_special=True
    )
    ctx.decode(cy.llama_batch_get_one(head))
    gen.set_input(
        f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n", n_past=len(head), out_mode=mode, seed=1
    )
    bias = gen.get_logit_bias()
    while gen.step_prompt(512) > 0:
        pass
    sampler = cy.LlamaSampler()
    if bias:
        sampler.add_logit_bias(vocab.n_vocab, bias)
    sampler.add_greedy()
    steps, text = [], []
    for _ in range(400):
        if gen.next_is_text():
            token = sampler.sample(ctx, -1)
            steps.append("T")
            more = gen.step_gen(token)
            if more:
                text.append(vocab.token_to_piece(token, 0, False))
        else:
            steps.append("A")
            more = gen.step_gen(cy.LLAMA_TOKEN_NULL)
        if not more:
            break
    sampler.close()
    return "".join(steps), "".join(text), gen.get_output(), [(vocab.token_to_piece(t, 0, True), b) for t, b in bias]


def test_text_continuation(chat):
    steps, text, wav, bias = _respond(
        chat, "You are a helpful assistant.", "What is two plus two? Answer in one word.", cy.MtmdGenAudioOutMode.TEXT
    )
    # a text-only response cannot switch to audio
    assert bias == [("<|audio_start|>", float("-inf")), ("<|mixed_start|>", float("-inf"))]
    assert set(steps) == {"T"}
    assert text.strip().lower().rstrip(".") == "four"
    assert len(wav) == WAV_HEADER


def test_interleaved_continuation(chat, tmp_path):
    # the model card's system prompt for spoken answers
    steps, text, wav, bias = _respond(
        chat, "Respond with interleaved text and audio.", "What is the capital of France?", cy.MtmdGenAudioOutMode.MIXED
    )
    assert bias == []
    # blocks of 6 text slots and 12 audio frames, as upstream's n_text_block / n_audio_block
    assert steps.startswith("T" * 6 + "A" * 12 + "T" * 6)
    assert "Paris" in text
    assert len(wav) > WAV_HEADER
    heard = _heard(tmp_path, wav, "mixed.wav")
    assert heard[: len(_words(text))][:6] == _words(text)[:6]
