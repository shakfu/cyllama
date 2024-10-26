# cyllama - a cython wrapper of llama.cpp


The project provides a cython wrapper for @ggerganov's [llama.cpp](https://github.com/ggerganov/llama.cpp) which is likely the most active open-source compiled LLM inference engine. This project was spun-off into its own repository from an earlier wrapping project, [llamalib](https://github.com/shakfu/llamalib)  which provided a early stage, but functional, wrappers of llama.cpp using using [cython](https://github.com/cython/cython), [pybind11](https://github.com/pybind/pybind11), and [nanobind](https://github.com/wjakob/nanobind). Further development of cyllama, the cython wrapper, will continue in this project.

Development goals are to:

- Stay up-to-date with bleeding-edge `llama.cpp`.

- Produce a minimal, performant, compiled, thin python wrapper around the core `llama-cli` feature-set of `llama.cpp`.

- Integrate and wrap `llava-cli` features.

- Integrate and wrap features from related projects such as [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)

- Learn about the internals of this popular C++/C LLM inference engine along the way.

Given that there is a fairly mature, well-maintained and performant ctypes based wrapper provided by @abetlen's [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) project and that llm inference is gpu-driven rather than cpu-driven, this all may see quite redundant. Nonetheless, we anticipate some benefits to using compiled wrappers:

- Packaging benefits with respect to self-contained statically compiled extension modules.

- There may be some performance improvements in the use of compiled wrappers over the use of ctypes.

- It may be possible to incorporate external optimizations more readily into compiled wrappers, and

- It provides a basis for integration with other code written in a wrapper variant.

- It may be useful in case one wants to de-couple the python frontend and wrapper backends to existing frameworks: for example, it may be useful to just replace the ctypes wrapper in `llama-cpp-python` with one of compiled python wrappers and contribute it back as a PR.

- This is the most efficient way, for me at least, to learn about the underlying technologies.


## Status

Development only on macOS to keep things simple. The following table provide an overview of the current wrapping/dev status:


| status                       | cyllama       |
| :--------------------------- | :-----------: |
| wrapper-type                 | cython 	   |
| wrap llama.h         		   | 1 			   |
| wrap high-level simple-cli   | 1 			   |
| wrap low-level simple-cli    | 1 			   |
| wrap low-level llama-cli     | 0 			   |
  

The initial milestone entailed creating a high-level wrapper of the `simple.cpp` llama.cpp example, followed by a low-level one. The high-level wrapper c++ code is placed in `llamalib.h` single-header library. The final object is to fully wrap the functionality of `llama-cli`.

The following is a relatively low-level example of the cython wrapper at work:

```python
from pathlib import Path

import cyllama as cy


# set path to model
MODEL = str(Path.cwd() / "models" / "Llama-3.2-1B-Instruct-Q8_0.gguf")

# configure params & prompt
params = cy.CommonParams()
params.model = MODEL
params.prompt = "When did the universe begin?"
params.n_predict = 32
params.n_ctx = 2048
params.cpuparams.n_threads = 4

# total length of the sequence including the prompt
n_predict: int = params.n_predict

# init LLM
cy.llama_backend_init()
cy.llama_numa_init(params.numa)

# initialize the model
model_params = cy.common_model_params_to_llama(params)
model = cy.LlamaModel(path_model=params.model, params=model_params)

# initialize the context
ctx_params = cy.common_context_params_to_llama(params)
ctx = cy.LlamaContext(model=model, params=ctx_params)

# build sampler chain
sparams = cy.llama_sampler_chain_default_params()
sparams.no_perf = False
smplr = cy.LlamaSampler(sparams)
smplr.add_greedy()

# tokenize the prompt
tokens_list: list[int] = cy.common_tokenize(ctx, params.prompt, True)
n_ctx: int = ctx.n_ctx()
n_kv_req: int = len(tokens_list) + (n_predict - len(tokens_list))

print("n_predict = %d, n_ctx = %d, n_kv_req = %d" % (n_predict, n_ctx, n_kv_req))

if n_kv_req > n_ctx:
    raise SystemExit(
        "error: n_kv_req > n_ctx, the required KV cache size is not big enough\n"
        "either reduce n_predict or increase n_ctx."
    )

# print the prompt token-by-token
print()
prompt = ""
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
batch.set_last_logits_to_true()

ctx.decode(batch)

# main loop

n_cur: int = batch.n_tokens
n_decode: int = 0

t_main_start: int = cy.ggml_time_us()

result: str = ""

while n_cur <= n_predict:

    # sample the next token
    new_token_id = smplr.sample(ctx, batch.n_tokens - 1)
    smplr.accept(new_token_id)

    # is it an end of generation?
    if model.token_is_eog(new_token_id) or n_cur == n_predict:
        print()
        break

    result += cy.common_token_to_piece(ctx, new_token_id)

    # prepare the next batch
    cy.common_batch_clear(batch)

    # push this new token for next evaluation
    cy.common_batch_add(batch, new_token_id, n_cur, [0], True)

    n_decode += 1
    n_cur += 1

    # evaluate the current batch with the transformer model
    ctx.decode(batch)


print(result)
print()

t_main_end: int = cy.ggml_time_us()

print(
    "decoded %d tokens in %.2f s, speed: %.2f t/s"
    % (
        n_decode,
        (t_main_end - t_main_start) / 1000000.0,
        n_decode / ((t_main_end - t_main_start) / 1000000.0),
    )
)
print()

# cleanup
cy.llama_backend_free()
```


It goes without saying that any help / collaboration / contributions to accelerate the above would be welcome!


## Setup

To build `cyllama`:

1. A recent version of `python3` (testing on python 3.12)


2. Git clone the latest version of `cyllama`:

```sh
git clone https://github.com/shakfu/cyllama.git
cd cyllama
```

3. Install dependencies of `cython`, `setuptools`, `numpy` and `pytest` for testing:

```sh
pip install -r requirements.txt
```

3. Type `make` in the terminal.

This will:

1. Download and build `llama.cpp`
2. Install it into `bin`, `include`, and `lib` in the cloned `cyllama` folder
3. Build `cyllama`


## Testing

As a first step, you should download a smallish llm in the `.gguf` model from [huggingface](https://huggingface.co/models?search=gguf). This [document](https://github.com/shakfu/llamalib/blob/main/docs/model-performance.md) provides some examples of models which have been known to work on a 16GB M1 Macbook air.

A good model to start with is [Llama-3.2-1B-Instruct-Q6_K.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-Q6_K.gguf). After downloading it, place the model in the `llamalib/models` folder and run:

```sh
bin/llama-simple -c 512 -n 512 -m models/Llama-3.2-1B-Instruct-Q6_K.gguf \
	-p "Is mathematics discovered or invented?"
```

Now, you will need `pytest` installed to run tests:

```sh
pytest
```

If all tests pass, feel free to `cd` into the `tests` directory and run some examples directly, for example:


```sh
cd tests && python3 simple.py`
```

## TODO

- [x] wrap llama-simple

- [ ] wrap llama-cli

- [ ] wrap llama-llava-cli

- [ ] wrap [whisper.cpp](https://github.com/ggerganov/whisper.cpp)

- [ ] wrap [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
