from .llama import llama_cpp as cy


def simple(model_path: str, prompt: str, ngl: int = 99, n_predict: int = 32, n_ctx: int = None, verbose=False):
    
    # load dynamic backends

    if not verbose:
        cy.disable_logging()

    cy.ggml_backend_load_all()

    # initialize the model

    model_params = cy.LlamaModelParams()
    model_params.n_gpu_layers = ngl

    model = cy.LlamaModel(model_path, model_params)
    vocab = model.get_vocab()

    # tokenize the prompt
    print(f"vocab.n_vocab = {vocab.n_vocab}")

    # find the number of tokens in the prompt
    prompt_tokens = vocab.tokenize(prompt, add_special=True, parse_special=True)
    n_prompt = len(prompt_tokens)
    print(f"n_prompt: {n_prompt}")

    # initialize the context

    ctx_params = cy.LlamaContextParams()
    # n_ctx is the context size
    if n_ctx is not None:
        ctx_params.n_ctx = n_ctx
    else:
        ctx_params.n_ctx = n_prompt + n_predict - 1
    # n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt
    # enable performance counters
    ctx_params.no_perf = False

    ctx = cy.LlamaContext(model, ctx_params)

    # initialize the sampler

    sparams = cy.LlamaSamplerChainParams()
    sparams.no_perf = False

    smplr = cy.LlamaSampler(sparams)
    smplr.add_greedy()

    # llama_sampler_chain_add(smpl, llama_sampler_init_greedy())

    # print the prompt token-by-token

    # print the prompt token-by-token
    print()
    prompt=""
    for i in prompt_tokens:
        try:
            prompt += vocab.token_to_piece(i, lstrip=0, special=False)
        except UnicodeDecodeError:
            continue
    print(prompt)


    # prepare a batch for the prompt

    # make a static method of LlamaBatch
    batch = cy.llama_batch_get_one(prompt_tokens)

    # main loop

    t_main_start: int = cy.ggml_time_us()
    n_decode = 0
    
    # llama_token new_token_id
    n_pos = n_prompt
    response = ""
    for i in range(n_predict):

        ctx.decode(batch) # may raise ValueError

        # sample the next token
        new_token_id = smplr.sample(ctx, -1)

        # is it an end of generation?
        if vocab.is_eog(new_token_id):
            break

        piece: str = vocab.token_to_piece(new_token_id, special=True)
        response += piece
        # print(f"piece: %s", piece);

        # prepare the next batch with the sampled token
        batch = cy.llama_batch_get_one([new_token_id], n_pos)
        n_pos += 1

        n_decode += 1

    print()

    print(f"response: {response}")

    print()

    t_main_end: int = cy.ggml_time_us()

    print("decoded %d tokens in %.2f s, speed: %.2f t/s" %
            (n_decode, (t_main_end - t_main_start) / 1000000.0, n_decode / ((t_main_end - t_main_start) / 1000000.0)))
    print()

    smplr.print_perf_data()
    ctx.print_perf_data()

    return True
