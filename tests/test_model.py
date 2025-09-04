import platform

import cyllama as cy

PLATFORM = platform.system()

def progress_callback(progress: float) -> bool:
    return progress > 0.50

def test_model_instance(model_path):
    cy.llama_backend_init()
    model = cy.LlamaModel(model_path)
    assert model
    cy.llama_backend_free()

def test_model_load_cancel(model_path):
    cy.llama_backend_init()
    params = cy.LlamaModelParams()
    params.use_mmap = False
    params.progress_callback = progress_callback
    model = cy.LlamaModel(model_path, params)
    assert model
    cy.llama_backend_free()

def test_autorelease(model_path):
    # need to wrap in a thread here.
    cy.llama_backend_init()    
    model = cy.LlamaModel(model_path)


    # assert model.vocab_type == cy.LLAMA_VOCAB_TYPE_BPE
    # model params
    assert model.rope_type == cy.GGML_ROPE_TYPE_NEOX
    assert model.get_vocab().n_vocab == 262144
    assert model.n_ctx_train == 32768
    assert model.n_embd == 640
    assert model.n_layer == 18
    assert model.n_head == 4
    assert model.n_head_kv == 1
    assert model.rope_freq_scale_train == 1.0
    assert model.desc == 'gemma3 270M Q5_K - Small'
    assert model.size == 251472384
    assert model.n_params == 268098176
    assert model.has_decoder()
    assert model.decoder_start_token() == -1
    assert not model.has_encoder()
    assert not model.is_recurrent()
    assert model.meta_count() == 41
    assert model.metadata()['general.base_model.0.repo_url'] == 'https://huggingface.co/gg-hf-gm/gemma-3-270m-it'
    ctx = cy.LlamaContext(model)
    assert ctx
    # assert model.get_vocab().n_vocab == len(ctx.get_logits())
    cy.llama_backend_free()
