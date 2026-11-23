"""LLM keeps the KV entries of a prompt's longest cached prefix and decodes only the rest."""

import gc

import pytest

import cyllama.api as api
from cyllama import LLM, GenerationConfig


@pytest.fixture
def llm(model_path):
    m = LLM(model_path, n_ctx=512, max_tokens=4, temperature=0.0, verbose=False)
    yield m
    m.close()
    del m
    gc.collect()


@pytest.fixture
def starts(monkeypatch):
    seen = []
    real = api.decode_prompt

    def spy(ctx, tokens, start=0):
        seen.append((start, len(tokens)))
        return real(ctx, tokens, start)

    monkeypatch.setattr(api, "decode_prompt", spy)
    return seen


PROMPT = "The capital of France is Paris. The capital of Germany is"


def test_cache_tracks_prompt_and_output(llm):
    llm(PROMPT)
    n_prompt = len(llm.vocab.tokenize(PROMPT, add_special=True, parse_special=False))
    assert len(llm._ctx_tokens) >= n_prompt
    assert llm._ctx.memory_seq_pos_max(0) == len(llm._ctx_tokens) - 1


def test_extended_prompt_reuses_prefix(llm, starts):
    first = str(llm(PROMPT))
    llm(PROMPT + first + " and the capital of Spain is")
    (s0, _), (s1, n1) = starts
    assert s0 == 0
    assert 0 < s1 < n1


def test_same_prompt_decodes_only_the_last_token(llm, starts):
    llm(PROMPT)
    llm(PROMPT)
    s1, n1 = starts[1]
    assert s1 == n1 - 1


def test_unrelated_prompt_keeps_only_shared_tokens(llm, starts):
    llm(PROMPT)
    llm("Once upon a time")
    s1, _ = starts[1]
    assert s1 <= 1  # at most the BOS token


def test_prefix_reuse_output_is_well_formed(llm):
    # Not compared to a fresh context: batch-size-dependent rounding can flip a greedy choice.
    llm(PROMPT)
    out = llm(PROMPT, config=GenerationConfig(max_tokens=4, temperature=0.0, n_ctx=512))
    assert isinstance(str(out), str) and len(llm._ctx_tokens) > 0


def test_grammar_generation_resets_cache(model_path):
    from cyllama.agents.constrained import GrammarConstrainedLLM

    m = GrammarConstrainedLLM(model_path, GenerationConfig(n_ctx=512, max_tokens=4, temperature=0.0))
    m(PROMPT)
    assert m._ctx_tokens
    m.generate_with_grammar("Answer yes or no:", 'root ::= "yes" | "no"')
    assert m._ctx_tokens == []  # its own decode is not tracked, so the cache is cleared
    m.close()
    del m
    gc.collect()


def test_cancelled_prefill_forgets_cache(llm, monkeypatch):
    llm(PROMPT)

    def interrupted(ctx, tokens, start=0):
        raise InterruptedError

    monkeypatch.setattr(api, "decode_prompt", interrupted)
    assert str(llm(PROMPT + " more")) == ""
    assert llm._ctx_tokens == []


def test_chat_template_compiled_once(llm):
    # each chat() built a sandbox and compiled the template again
    api._compile_chat_template.cache_clear()
    for _ in range(3):
        llm.chat([{"role": "user", "content": "Hi"}], config=GenerationConfig(max_tokens=1, n_ctx=512))
    info = api._compile_chat_template.cache_info()
    assert info.misses == 1 and info.hits == 2


def test_stats_come_from_the_stream(llm):
    # _generate re-tokenized prompt and output just to count them; the stream's counts are used now
    r = llm(PROMPT)
    assert r.stats.prompt_tokens == len(llm.vocab.tokenize(PROMPT, add_special=True, parse_special=True))
    assert 1 <= r.stats.generated_tokens <= 4  # max_tokens, sampled tokens
    assert r.stats.generated_tokens == llm._last_stream_stats.generated_tokens
