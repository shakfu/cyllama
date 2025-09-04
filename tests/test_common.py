import pytest
pytest.skip(allow_module_level=True)

import cyllama as cy

def test_common(model_path):

    params = cy.CommonParams()
    cy.common_init()

    params.model = model_path
    params.prompt = "When did the universe begin?"
    params.n_predict = 32
    params.n_ctx = 512
    params.cpuparams.n_threads = 4

    # sparams = params.sparams

    # total length of the sequence including the prompt
    n_predict: int = params.n_predict

    # init LLM
    cy.llama_backend_init()
    cy.llama_numa_init(params.numa)

    # load the model and apply lora adapter, if any
    llama_init = cy.CommonInitResult(params)
    model = llama_init.model;
    ctx = llama_init.context;

    cy.llama_backend_free()

    assert True
