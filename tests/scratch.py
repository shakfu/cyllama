import sys
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / 'src'))

import cyllama.llama.llama_cpp as cy

model_path = str(ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q8_0.gguf')

params = cy.CommonParams()
params.model = model_path
params.prompt = "When did the universe begin?"
params.n_predict = 32
params.n_ctx = 512
params.cpuparams.n_threads = 4

# total length of the sequence including the prompt
n_predict: int = params.n_predict

# init LLM
cy.llama_backend_init()
cy.llama_numa_init(params.numa)

# initialize the model

model_params = cy.common_model_params_to_llama(params)

# set local test model
params.model = model_path

model = cy.LlamaModel(path_model=params.model, params=model_params)

# initialize the context
ctx_params = cy.common_context_params_to_llama(params)
ctx = cy.LlamaContext(model=model, params=ctx_params)


# build sampler chain
sparams = cy.LlamaSamplerChainParams()
sparams.no_perf = False

smplr = cy.LlamaSampler(sparams)

smplr.add_greedy()


# tokenize the prompt

tokens_list: list[int] = cy.common_tokenize(ctx, params.prompt, True)

n_ctx: int = ctx.n_ctx

n_kv_req: int = len(tokens_list) + (n_predict - len(tokens_list))

print("n_predict = %d, n_ctx = %d, n_kv_req = %d" % (n_predict, n_ctx, n_kv_req))

if (n_kv_req > n_ctx):
    raise SystemExit(
        "error: n_kv_req > n_ctx, the required KV cache size is not big enough\n"
        "either reduce n_predict or increase n_ctx.")

# print the prompt token-by-token
print()
prompt=""
for i in tokens_list:
    prompt += cy.common_token_to_piece(ctx, i)
print(prompt)

# create a llama_batch with size 512
# we use this object to submit token data for decoding

# create batch
batch = cy.LlamaBatch(n_tokens=512, embd=0, n_seq_max=1)

# evaluate the initial prompt
for i, token in enumerate(tokens_list):
    cy.common_batch_add(batch, token, i, [0], False)

# llama_decode will output logits only for the last token of the prompt
# batch.logits[batch.n_tokens - 1] = True
batch.set_last_logits_to_true()

# logits = batch.get_logits()

ctx.decode(batch)

# main loop

# n_cur: int    = batch.n_tokens
# n_decode: int = 0

# t_main_start: int = cy.ggml_time_us()

# result: str = ""

# while (n_cur <= n_predict):
#     # sample the next token

#     if True:
#         new_token_id = smplr.sample(ctx, batch.n_tokens - 1)

#         # print("new_token_id: ", new_token_id)

#         smplr.accept(new_token_id)

#         # is it an end of generation?
#         if (model.token_is_eog(new_token_id) or n_cur == n_predict):
#             print()
#             break

#         result += cy.common_token_to_piece(ctx, new_token_id)

#         # prepare the next batch
#         cy.common_batch_clear(batch);

#         # push this new token for next evaluation
#         cy.common_batch_add(batch, new_token_id, n_cur, [0], True)

#         n_decode += 1

#     n_cur += 1

#     # evaluate the current batch with the transformer model
#     ctx.decode(batch)


# print(result)

# print()

# t_main_end: int = cy.ggml_time_us()

# print("decoded %d tokens in %.2f s, speed: %.2f t/s" %
#         (n_decode, (t_main_end - t_main_start) / 1000000.0, n_decode / ((t_main_end - t_main_start) / 1000000.0)))
# print()

cy.llama_backend_free()

