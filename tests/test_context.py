import platform

import cyllama as cy

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
    assert ctx.n_seq_max == 1
    assert ctx.get_state_size() == 46
    # assert ctx.pooling_type == cy.LLAMA_POOLING_TYPE_NONE
    # context params
    # assert ctx.params.rope_scaling_type == cy.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
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
