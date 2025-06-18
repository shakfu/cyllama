import pytest
pytest.skip(allow_module_level=True)

import platform
import cyllama.cyllama as cy

PLATFORM = platform.system()

def progress_callback(progress: float) -> bool:
    return progress > 0.50

def test_model_instance(model_path):
    cy.llama_backend_init()
    model = cy.LlamaModel(model_path)
    cy.llama_backend_free()

def test_model_load_cancel(model_path):
    cy.llama_backend_init()
    params = cy.LlamaModelParams()
    params.use_mmap = False
    params.progress_callback = progress_callback
    model = cy.LlamaModel(model_path, params)
    cy.llama_backend_free()

def test_autorelease(model_path):
    # need to wrap in a thread here.
    cy.llama_backend_init()    
    model = cy.LlamaModel(model_path)
    assert model.vocab_type == cy.LLAMA_VOCAB_TYPE_BPE
    # model params
    assert model.rope_type == cy.LLAMA_ROPE_TYPE_NORM
    assert model.n_vocab == 128256
    assert model.n_ctx_train == 131072
    assert model.n_embd == 2048
    assert model.n_layer == 16
    assert model.n_head == 32
    assert model.rope_freq_scale_train == 1.0
    assert model.desc == "llama 1B Q8_0"
    assert model.size == 1313251456
    assert model.n_params == 1235814432
    assert model.has_decoder() == True
    assert model.has_encoder() == False
    assert model.is_recurrent() == False
    ctx = cy.LlamaContext(model)
    assert model.n_vocab == len(ctx.get_logits())
    cy.llama_backend_free()
