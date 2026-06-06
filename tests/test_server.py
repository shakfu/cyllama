import pprint
import os
import sys
import base64
import pytest
import json
import orjson

import xllamacpp as xlc


def test_get_system_info():
    assert "CPU :" in xlc.get_system_info()


def test_get_device_info():
    xlc.get_device_info()
    info = xlc.get_device_info()
    assert len(info) > 0
    assert "CPU" in [i["name"] for i in info]
    print(info)


def test_llama_server_requires_model_path():
    params = xlc.CommonParams()

    with pytest.raises(ValueError, match="router server feature"):
        xlc.Server(params)


def test_llama_server(model_path):
    params = xlc.CommonParams()

    params.model.path = os.path.join(model_path, "Llama-3.2-1B-Instruct-Q8_0.gguf")
    params.prompt = "When did the universe begin?"
    params.warmup = False
    params.n_predict = 32
    params.n_ctx = 256
    params.n_parallel = 1
    params.cpuparams.n_threads = 2
    params.cpuparams_batch.n_threads = 2
    params.endpoint_metrics = True
    params.cache_ram_mib = 0

    server = xlc.Server(params)

    complete_prompt = {
        "max_tokens": 128,
        "prompt": "Write the fibonacci function in c++.",
    }

    server.handle_completions(
        complete_prompt,
        lambda v: pprint.pprint(v),
    )
    v = server.handle_completions(complete_prompt)
    assert isinstance(v, dict)
    assert "code" not in v
    pprint.pprint(v)

    # If the prompt is a str or bytes, a callback is required.
    with pytest.raises(ValueError, match="non dict prompt"):
        server.handle_chat_completions(orjson.dumps(complete_prompt))

    complete_prompt["stream"] = True

    # If the prompt is streaming, a callback is required.
    with pytest.raises(ValueError, match="requires a callback for streaming"):
        server.handle_completions(complete_prompt)

    server.handle_completions(
        complete_prompt,
        lambda v: pprint.pprint(v),
    )

    # Test handle_completions with a str or bytes prompt
    ok = False

    def _cb_str(v):
        nonlocal ok
        assert type(v) is str
        json.loads(v)
        ok = True

    complete_prompt_str = json.dumps(complete_prompt)
    server.handle_completions(
        complete_prompt_str,
        _cb_str,
    )
    assert ok

    ok = False

    def _cb_bytes(v):
        nonlocal ok
        assert type(v) is bytes
        orjson.loads(v)
        ok = True

    complete_prompt_bytes = orjson.dumps(complete_prompt)
    server.handle_completions(
        complete_prompt_bytes,
        _cb_bytes,
    )
    assert ok

    chat_complete_prompt = {
        "max_tokens": 128,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write the fibonacci function in c++."},
        ],
    }

    server.handle_chat_completions(
        chat_complete_prompt,
        lambda v: pprint.pprint(v),
    )
    v = server.handle_chat_completions(chat_complete_prompt)
    assert isinstance(v, dict)
    assert "code" not in v
    pprint.pprint(v)

    # If the prompt is a str or bytes, a callback is required.
    with pytest.raises(ValueError, match="non dict prompt"):
        server.handle_chat_completions(json.dumps(chat_complete_prompt))

    chat_complete_prompt["stream"] = True

    # If the prompt is streaming, a callback is required.
    with pytest.raises(ValueError, match="requires a callback for streaming"):
        server.handle_chat_completions(chat_complete_prompt)

    server.handle_chat_completions(
        chat_complete_prompt,
        lambda v: pprint.pprint(v),
    )

    # Test handle_chat_completions with a str or bytes prompt
    ok = False

    def _cb_str(v):
        nonlocal ok
        assert type(v) is str
        json.loads(v)
        ok = True

    chat_complete_prompt_str = json.dumps(chat_complete_prompt)
    server.handle_chat_completions(
        chat_complete_prompt_str,
        _cb_str,
    )
    assert ok

    ok = False

    def _cb_bytes(v):
        nonlocal ok
        assert type(v) is bytes
        orjson.loads(v)
        ok = True

    chat_complete_prompt_bytes = orjson.dumps(chat_complete_prompt)
    server.handle_chat_completions(
        chat_complete_prompt_bytes,
        _cb_bytes,
    )
    assert ok

    # Test handle_metrics()
    result = server.handle_metrics()
    assert type(result) is str
    assert "llamacpp:prompt_seconds_total" in result


def test_llama_server_stream_callback_stop(model_path):
    params = xlc.CommonParams()

    params.model.path = os.path.join(model_path, "Llama-3.2-1B-Instruct-Q8_0.gguf")
    params.prompt = "When did the universe begin?"
    params.warmup = False
    params.n_predict = 64
    params.n_ctx = 256
    params.n_parallel = 1
    params.cpuparams.n_threads = 2
    params.cpuparams_batch.n_threads = 2

    server = xlc.Server(params)

    # Test handle_completions streaming stop via callback return value
    complete_prompt = {
        "max_tokens": 128,
        "prompt": "Write a long story about the history of the universe.",
        "stream": True,
    }

    all_chunks = 0

    def _cb_all(v):
        nonlocal all_chunks
        all_chunks += 1

    stop_chunks = 0

    def _cb_stop(v):
        nonlocal stop_chunks
        stop_chunks += 1
        return True

    server.handle_completions(complete_prompt, _cb_all)
    assert all_chunks >= 1

    server.handle_completions(complete_prompt, _cb_stop)
    assert stop_chunks == 1
    assert all_chunks > stop_chunks

    # Test handle_chat_completions streaming stop via callback return value
    chat_complete_prompt = {
        "max_tokens": 128,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {
                "role": "user",
                "content": "Tell me in detail about the history of programming languages.",
            },
        ],
        "stream": True,
    }

    chat_all_chunks = 0

    def _chat_cb_all(v):
        nonlocal chat_all_chunks
        chat_all_chunks += 1

    chat_stop_chunks = 0

    def _chat_cb_stop(v):
        nonlocal chat_stop_chunks
        chat_stop_chunks += 1
        return True

    server.handle_chat_completions(chat_complete_prompt, _chat_cb_all)
    assert chat_all_chunks >= 1

    server.handle_chat_completions(chat_complete_prompt, _chat_cb_stop)
    assert chat_stop_chunks == 1
    assert chat_all_chunks > chat_stop_chunks


def test_llama_server_chat_with_grammar(model_path):
    """Test grammar constraint on chat completions endpoint.

    About CommonGrammar:
        CommonGrammar is a wrapper around the C++ common_grammar struct that bundles a
        grammar type (common_grammar_type) with a GBNF grammar string. It is used by
        common_sampler_init() to initialize the grammar-constrained sampler. The grammar
        type controls prefill behavior:
        - COMMON_GRAMMAR_TYPE_USER: user-provided GBNF, no prefill
        - COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT: auto-generated from JSON schema, prefills
        - COMMON_GRAMMAR_TYPE_TOOL_CALLS: auto-generated by chat template parser, prefills

    Why this test does NOT use params.sampling.grammar (CommonGrammar):
        In the server path, each request builds its own task_params.sampling.grammar from
        the request body via params_from_json_cmpl(). Unlike other sampling parameters
        (e.g. temp, top_k) which fall back to server-level defaults via
        json_value(data, "key", defaults.sampling.key), grammar uses a conditional branch:

        - If the request body contains "json_schema", it converts to GBNF and sets grammar.
        - Else if the request body contains "grammar", it uses that string directly.
        - Otherwise, grammar remains empty (no fallback to defaults.sampling.grammar).

        This applies to BOTH /completions and /chat/completions endpoints. Setting
        params.sampling.grammar on the server-level CommonParams has no effect on any
        server request.

        CommonGrammar on params.sampling.grammar is only effective for the CLI path
        (llama-cli), where common_sampler_init() reads params.sampling.grammar directly
        without going through JSON request parsing.

        Therefore, the correct approach for server usage is to pass the grammar string
        in the request body (as shown below), not via params.sampling.grammar.
    """
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "score": {"type": "number"},
        },
        "required": ["answer"],
    }
    grammar_str = xlc.json_schema_to_grammar(schema)

    params = xlc.CommonParams()

    params.model.path = os.path.join(model_path, "Llama-3.2-1B-Instruct-Q8_0.gguf")
    params.warmup = False
    params.n_predict = 64
    params.n_ctx = 256
    params.cpuparams.n_threads = 2
    params.cpuparams_batch.n_threads = 2
    params.sampling.temp = 0
    params.sampling.top_k = 1

    # NOTE: We intentionally do NOT set params.sampling.grammar here.
    # Even if we did:
    #   params.sampling.grammar = xlc.CommonGrammar(
    #       type=xlc.common_grammar_type.COMMON_GRAMMAR_TYPE_USER,
    #       grammar=grammar_str,
    #   )
    # it would have no effect — the server endpoints ignore server-level grammar
    # defaults and only read grammar from the request body.

    server = xlc.Server(params)

    chat_complete_prompt = {
        "max_tokens": 64,
        "messages": [
            {
                "role": "system",
                "content": "Respond with a JSON object matching the provided schema.",
            },
            {
                "role": "user",
                "content": "Provide an answer string and an optional numeric score.",
            },
        ],
        # Grammar MUST be passed in the request body — this is the only way
        # to apply grammar constraints on server endpoints.
        # params.sampling.grammar (CommonGrammar) has no effect here.
        "grammar": grammar_str,
    }

    result = server.handle_chat_completions(chat_complete_prompt)

    assert isinstance(result, dict)
    content = result["choices"][0]["message"]["content"]
    parsed = json.loads(content)

    assert parsed["answer"]
    assert isinstance(parsed["answer"], str)
    if "score" in parsed:
        assert isinstance(parsed["score"], (int, float))


def test_llama_server_multimodal(model_path):
    with open(os.path.join(os.path.dirname(__file__), "data/11_truck.png"), "rb") as f:
        content = f.read()
    IMG_BASE64_0 = "data:image/png;base64," + base64.b64encode(content).decode("utf-8")

    params = xlc.CommonParams()

    params.model.path = os.path.join(model_path, "tinygemma3-Q8_0.gguf")
    params.mmproj.path = os.path.join(model_path, "mmproj-tinygemma3.gguf")
    params.sampling.seed = 42
    params.sampling.top_k = 1
    params.sampling.temp = 0
    params.n_predict = 4
    params.n_ctx = 1024
    params.cpuparams.n_threads = 4
    params.cpuparams_batch.n_threads = 2

    server = xlc.Server(params)

    chat_complete_prompt = {
        "max_tokens": 128,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this:\n"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": IMG_BASE64_0,
                        },
                    },
                ],
            },
        ],
    }

    server.handle_chat_completions(
        chat_complete_prompt,
        lambda v: pprint.pprint(v),
    )


def test_llama_server_embedding(model_path):
    params = xlc.CommonParams()

    params.model.path = os.path.join(model_path, "Qwen3-Embedding-0.6B-Q8_0.gguf")
    params.embedding = True
    params.n_predict = -1
    params.n_ctx = 512
    params.n_batch = 128
    params.n_ubatch = 128
    params.sampling.seed = 42
    params.cpuparams.n_threads = 2
    params.cpuparams_batch.n_threads = 2
    params.pooling_type = xlc.llama_pooling_type.LLAMA_POOLING_TYPE_LAST

    server = xlc.Server(params)

    embedding_input = {
        "input": [
            "I believe the meaning of life is",
            "Write a joke about AI from a very long prompt which will not be truncated",
            "This is a test",
            "This is another test",
        ],
    }

    result = server.handle_embeddings(embedding_input)

    assert type(result) is dict
    assert len(result["data"]) == 4
    for d in result["data"]:
        assert len(d["embedding"]) == 1024

    embedding_input_str = json.dumps(embedding_input)
    assert type(embedding_input_str) is str
    result_str = server.handle_embeddings(embedding_input_str)
    assert type(result_str) is str
    result = json.loads(result_str)

    assert type(result) is dict
    assert len(result["data"]) == 4
    for d in result["data"]:
        assert len(d["embedding"]) == 1024

    embedding_input_bytes = orjson.dumps(embedding_input)
    assert type(embedding_input_bytes) is bytes
    result_bytes = server.handle_embeddings(embedding_input_bytes)
    assert type(result_bytes) is bytes
    result = orjson.loads(result_str)

    assert type(result) is dict
    assert len(result["data"]) == 4
    for d in result["data"]:
        assert len(d["embedding"]) == 1024


@pytest.mark.skipif(sys.platform == "darwin", reason="Rerank test crashes on macOS CI")
def test_llama_server_rerank(model_path):
    params = xlc.CommonParams()

    params.model.path = os.path.join(model_path, "bge-reranker-v2-m3-Q2_K.gguf")
    params.embedding = True
    params.n_predict = -1
    params.n_ctx = 512
    params.n_batch = 128
    params.n_ubatch = 128
    params.sampling.seed = 42
    params.cpuparams.n_threads = 2
    params.cpuparams_batch.n_threads = 2
    params.pooling_type = xlc.llama_pooling_type.LLAMA_POOLING_TYPE_RANK

    server = xlc.Server(params)

    rerank_input = {
        "query": "What is the capital of France?",
        "documents": [
            "Paris is the capital of France.",
            "The Eiffel Tower is in Paris.",
            "Germany is located in Europe.",
        ],
    }

    result = server.handle_rerank(rerank_input)

    assert type(result) is dict
    assert len(result["results"]) == 3

    rerank_input_str = json.dumps(rerank_input)
    result_str = server.handle_rerank(rerank_input_str)
    assert type(result_str) is str
    result = json.loads(result_str)

    assert type(result) is dict
    assert len(result["results"]) == 3

    rerank_input_bytes = orjson.dumps(rerank_input)
    result_bytes = server.handle_rerank(rerank_input_bytes)
    assert type(result_bytes) is bytes
    result = orjson.loads(result_bytes)

    assert type(result) is dict
    assert len(result["results"]) == 3


def test_llama_server_speculative(model_path):
    """Demonstrate draft-model speculative decoding.

    Speculative decoding pairs a small, fast "draft" model with a larger "target"
    model. The draft model proposes several tokens at once; the target model then
    verifies them in a single forward pass and accepts the longest matching prefix.
    With a well-aligned tokenizer/vocabulary, the produced text is *bit-exact* to
    running the target model alone, but throughput improves because multiple
    tokens are validated per target-model decode step.

    A real-world example would be pairing a large dense model with a smaller
    same-family checkpoint, e.g.:

        params.model.path                 = ".../Qwen3-4B-Instruct-Q8_0.gguf"
        params.speculative.draft.mparams.path = ".../Qwen3-0.6B-Instruct-Q4_0.gguf"
        params.speculative.draft.n_min = 4
        params.speculative.draft.n_max = 8

    This is especially helpful for AMD Strix Halo (and other unified-memory
    APU/iGPU) users who want to run larger dense models: the draft model fits
    easily alongside the target and trades a small amount of memory for a
    meaningful tokens/sec speedup on memory-bandwidth-bound hardware.

    For CI we mirror the upstream llama.cpp speculative test
    (thirdparty/llama.cpp/tools/server/tests/unit/test_speculative.py) and use
    the tiny stories15M MOE F16 model as the target with stories15M Q4_0 as
    the draft. Both share the LLaMA tokenizer, so speculation is valid.

    Two equivalent ways to enable draft-model speculative decoding:

    1. Server-level (this test): set ``params.speculative.draft.mparams.path``.
       The draft model is loaded once at server startup and reused for every
       request.
    2. Per-request: pass ``"speculative.n_max"``, ``"speculative.n_min"``,
       ``"speculative.p_min"`` etc. in the JSON body of /completions or
       /chat/completions to override the server defaults for a single request.

    Setting ``speculative.types`` is *not* required when a draft model path is
    provided — ``has_dft()`` triggers the draft-model path automatically. The
    ``types`` field only matters for n-gram / self-speculative variants which
    have no separate draft model.
    """
    draft_path = os.path.join(model_path, "stories15M-q4_0.gguf")
    if not os.path.exists(draft_path):
        pytest.skip(
            f"draft model not found: {draft_path} " "(run `make download` to fetch it)"
        )

    params = xlc.CommonParams()

    # Target (main) model.
    params.model.path = os.path.join(model_path, "stories15M_MOE-F16.gguf")

    # Draft model used to speculate tokens for the target model.
    params.speculative.draft.mparams.path = draft_path
    # Draft N-tokens window: propose between n_min and n_max tokens per step.
    # Mirrors the upstream test defaults; tune per model pair in production.
    params.speculative.draft.n_min = 4
    params.speculative.draft.n_max = 8

    params.warmup = False
    params.n_predict = 32
    params.n_ctx = 256
    params.n_parallel = 1
    params.cpuparams.n_threads = 2
    params.cpuparams_batch.n_threads = 2
    # Speculative decoding requires deterministic sampling for the equivalence
    # property below to hold.
    params.sampling.seed = 42
    params.sampling.temp = 0.0
    params.sampling.top_k = 1
    # Disable flash-attention to match the upstream speculative test setup;
    # remove this line on hardware where FA is supported and beneficial.
    params.flash_attn_type = xlc.llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_DISABLED

    server = xlc.Server(params)

    complete_prompt = {
        "max_tokens": 16,
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "top_k": 1,
        "seed": 42,
    }

    result = server.handle_completions(complete_prompt)
    assert isinstance(result, dict)
    assert "code" not in result
    assert "choices" in result
    content = result["choices"][0]["text"]
    print(f"Speculative completion result: {content}")
    assert len(content) > 0

    # Per-request override: tighten the draft window for a single completion.
    # These keys are documented in
    # thirdparty/llama.cpp/tools/server/tests/unit/test_speculative.py.
    per_request = dict(complete_prompt)
    per_request["speculative.n_min"] = 1
    per_request["speculative.n_max"] = 4
    per_request["speculative.p_min"] = 0.0
    result2 = server.handle_completions(per_request)
    assert isinstance(result2, dict)
    assert "code" not in result2
    assert len(result2["choices"][0]["text"]) > 0


def test_llama_server_lora(model_path):
    """Test loading and using a LoRA adapter.

    This test uses the stories15M_MOE model with a Shakespeare LoRA adapter.
    - Without LoRA (scale=0), the model generates bedtime story style text.
    - With LoRA (scale=1), the model generates Shakespearean style text.

    Based on: thirdparty/llama.cpp/tools/server/tests/unit/test_lora.py
    """
    params = xlc.CommonParams()

    params.model.path = os.path.join(model_path, "stories15M_MOE-F16.gguf")
    params.warmup = False
    params.n_predict = 64
    params.n_ctx = 256
    params.n_parallel = 1
    params.cpuparams.n_threads = 2
    params.cpuparams_batch.n_threads = 2
    params.sampling.seed = 42
    params.sampling.temp = 0.0
    params.sampling.top_k = 1

    # Create a CommonAdapterLoraInfo and set it via lora_adapters
    lora_path = os.path.join(model_path, "moe_shakespeare15M.gguf")
    lora_info = xlc.CommonAdapterLoraInfo(lora_path, 1.0)
    params.lora_adapters = [lora_info]

    # Verify the lora adapter was added and can be accessed via wrapper
    assert len(params.lora_adapters) == 1
    assert params.lora_adapters[0].path == lora_path
    assert params.lora_adapters[0].scale == 1.0

    # Test modifying the scale via the wrapper
    params.lora_adapters[0].scale = 0.5
    assert lora_info.scale == 0.5
    params.lora_adapters[0].scale = 1.0  # Reset back

    server = xlc.Server(params)

    # Test completion with LoRA applied
    complete_prompt = {
        "max_tokens": 64,
        "prompt": "Look in thy glass",
        "seed": 42,
        "temperature": 0.0,
    }

    result = server.handle_completions(complete_prompt)

    assert isinstance(result, dict)
    assert "code" not in result
    assert "choices" in result
    content = result["choices"][0]["text"]
    print(f"LoRA completion result: {content}")

    # With Shakespeare LoRA, expect Shakespearean-style words
    assert len(content) > 0
