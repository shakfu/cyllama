"""mtmd objects raise, not use freed memory, after an explicit close() of a parent context or model."""

import gc

import pytest

import cyllama.llama.llama_cpp as cy
from conftest import MODELS_DIR, ROOT

PAIRS = [
    ("chatterbox-turbo-Q8_0.gguf", "mmproj-chatterbox-turbo-F16.gguf"),
    ("Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf", "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"),
]
PAIR = next(
    ((MODELS_DIR / m, MODELS_DIR / p) for m, p in PAIRS if (MODELS_DIR / m).exists() and (MODELS_DIR / p).exists()),
    None,
)
SPEAKER = ROOT / "tests" / "samples" / "jfk.wav"

pytestmark = pytest.mark.skipif(PAIR is None, reason="needs a Chatterbox or Qwen3-TTS model + mmproj")


@pytest.fixture
def stack():
    """A fresh model, llama context and mtmd context, so each test can close one of them."""
    cy.ggml_backend_load_all()
    params = cy.LlamaModelParams()
    params.n_gpu_layers = 99
    model = cy.LlamaModel(str(PAIR[0]), params)
    ctx_params = cy.LlamaContextParams()
    ctx_params.n_ctx = 512
    lctx = cy.LlamaContext(model, ctx_params)
    mctx = cy.MtmdContext(str(PAIR[1]), model)
    yield model, lctx, mctx
    mctx.close()
    lctx.close()
    model.close()
    del mctx, lctx, model
    gc.collect()


def test_mtmd_context_after_model_close(stack):
    model, _, mctx = stack
    model.close()
    with pytest.raises(RuntimeError, match="LlamaModel is closed"):
        mctx.marker
    with pytest.raises(RuntimeError, match="LlamaModel is closed"):
        mctx.tokenize(mctx.marker, [], True, True)
    with pytest.raises(RuntimeError, match="LlamaModel is closed"):
        cy.MtmdBitmap.from_file(mctx, str(SPEAKER))


def test_batch_after_mtmd_close(stack):
    _, _, mctx = stack
    batch = mctx.batch()
    mctx.close()
    with pytest.raises(RuntimeError, match="MtmdContext is closed"):
        batch.encode()
    # freeing the batch does not touch the context
    batch.close()
    batch.close()


def test_bitmap_from_closed_or_wrong_context(stack):
    _, _, mctx = stack
    with pytest.raises(TypeError):
        cy.MtmdBitmap.from_file(object(), str(SPEAKER))
    mctx.close()
    with pytest.raises(RuntimeError, match="MtmdContext is closed"):
        cy.MtmdBitmap.from_file(mctx, str(SPEAKER))
    with pytest.raises(RuntimeError, match="MtmdContext is closed"):
        cy.MtmdBitmap.from_buffer(mctx, SPEAKER.read_bytes())


@pytest.mark.parametrize("closed", ["mtmd", "llama", "model"])
def test_audio_generator_after_parent_close(stack, closed):
    model, lctx, mctx = stack
    gen = cy.MtmdAudioGenerator(lctx, mctx)
    {"mtmd": mctx, "llama": lctx, "model": model}[closed].close()
    expected = {"mtmd": "MtmdContext is closed", "llama": "LlamaContext is closed", "model": "LlamaModel is closed"}
    for call in (lambda: gen.set_input("hello"), lambda: gen.step_prompt(8), gen.get_output, gen.reset):
        with pytest.raises(RuntimeError, match=expected[closed]):
            call()
    gen.close()


def test_audio_generator_rejects_closed_llama_context(stack):
    _, lctx, mctx = stack
    lctx.close()
    with pytest.raises(RuntimeError, match="LlamaContext is closed"):
        cy.MtmdAudioGenerator(lctx, mctx)


def test_eval_chunks_rejects_closed_llama_context(stack):
    _, lctx, mctx = stack
    chunks = mctx.tokenize("hello", [], True, True)
    lctx.close()
    with pytest.raises(RuntimeError, match="LlamaContext is closed"):
        mctx.eval_chunks(lctx, chunks)
    with pytest.raises(RuntimeError, match="LlamaContext is closed"):
        mctx.model_can_chat(lctx)
