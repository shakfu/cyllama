import platform
import pytest

import cyllama.llama.llama_cpp as cy

PLATFORM = platform.system()


def test_model_instance(model_path):
    cy.llama_backend_init()
    model = cy.LlamaModel(model_path)
    assert model
    cy.llama_backend_free()


def test_model_load_with_progress_callback(model_path):
    """Test model loading with a progress callback that allows loading to complete."""
    progress_values = []

    def on_progress(progress: float) -> bool:
        progress_values.append(progress)
        return True  # continue loading

    cy.llama_backend_init()
    params = cy.LlamaModelParams()
    params.use_mmap = False
    params.progress_callback = on_progress
    model = cy.LlamaModel(model_path, params)
    assert model
    assert len(progress_values) > 0, "Progress callback should have been called"
    cy.llama_backend_free()


def test_model_load_cancel(model_path):
    """Test that returning False from progress callback aborts model loading."""
    def abort_at_50_percent(progress: float) -> bool:
        return progress < 0.50  # abort after 50%

    cy.llama_backend_init()
    params = cy.LlamaModelParams()
    params.use_mmap = False
    params.progress_callback = abort_at_50_percent

    # Loading should fail because we abort after 50%
    with pytest.raises(ValueError, match="Failed to load model"):
        model = cy.LlamaModel(model_path, params)

    cy.llama_backend_free()

def test_autorelease(model_path):
    # need to wrap in a thread here.
    cy.llama_backend_init()    
    model = cy.LlamaModel(model_path)


    # assert model.vocab_type == cy.LLAMA_VOCAB_TYPE_BPE
    # model params
    assert model.rope_type == 0
    assert model.get_vocab().n_vocab == 128256
    assert model.n_ctx_train == 131072
    assert model.n_embd == 2048
    assert model.n_layer == 16
    assert model.n_head == 32
    assert model.n_head_kv == 8
    assert model.rope_freq_scale_train == 1.0
    assert model.desc == 'llama 1B Q8_0'
    assert model.size == 1313251456
    assert model.n_params == 1235814432
    assert model.has_decoder()
    assert model.decoder_start_token() == -1
    assert not model.has_encoder()
    assert not model.is_recurrent()
    assert model.meta_count() == 30
    ctx = cy.LlamaContext(model)
    assert ctx
    # assert model.get_vocab().n_vocab == len(ctx.get_logits())
    cy.llama_backend_free()
