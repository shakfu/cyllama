import pytest
pytest.skip(allow_module_level=True)

import platform

import cyllama.cyllama as cy

PLATFORM = platform.system()

def test_context(model_path):
    # need to wrap in a thread here.
    cy.llama_backend_init()    
    model = cy.LlamaModel(model_path)
    ctx = cy.LlamaContext(model)
    assert ctx.model is model
    assert ctx.n_ctx == 512
    assert ctx.n_batch == 512
    assert ctx.n_ubatch == 512
    assert ctx.n_seq_max == 512
    assert ctx.get_state_size() == 425
    assert ctx.pooling_type == cy.LLAMA_POOLING_TYPE_NONE
    assert ctx.model.vocab_type == cy.LLAMA_VOCAB_TYPE_BPE
    # model params
    assert ctx.model.rope_type == cy.LLAMA_ROPE_TYPE_NORM
    assert ctx.model.n_vocab == 128256
    assert ctx.model.n_ctx_train == 131072
    assert ctx.model.n_embd == 2048
    assert ctx.model.n_layer == 16
    assert ctx.model.n_head == 32
    assert ctx.model.rope_freq_scale_train == 1.0
    assert ctx.model.desc == "llama 1B Q8_0"
    assert ctx.model.size == 1313251456
    assert ctx.model.n_params == 1235814432
    assert ctx.model.has_decoder() == True
    assert ctx.model.has_encoder() == False
    assert ctx.model.is_recurrent() == False
    assert ctx.model.n_vocab == len(ctx.get_logits())
    # context params
    assert ctx.params.rope_scaling_type == cy.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
    cy.llama_backend_free()

def test_context_params():
    params = cy.LlamaContextParams()
    assert params.n_threads == 4
    assert params.n_batch == 2048
    assert params.n_ctx == 512

def test_context_params_set():
    params = cy.LlamaContextParams()
    params.n_threads = 8
    params.n_batch = 1024
    params.n_ctx = 1024
    assert params.n_threads == 8
    assert params.n_batch == 1024
    assert params.n_ctx == 1024
