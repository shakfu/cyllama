import platform

import cyllama.llama.llama_cpp as cy

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
    # State size for an empty context (no tokens evaluated yet)
    assert ctx.get_state_size() == 37
    # context params
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
