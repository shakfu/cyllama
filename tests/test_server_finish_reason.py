"""Server finish_reason ("stop" vs "length") and top_p passthrough, shared by both servers."""

import gc
from unittest.mock import Mock

import pytest

import cyllama.llama.server.python as srv
from cyllama.llama.server.python import (
    ChatMessage,
    ChatRequest,
    PythonServer,
    ServerConfig,
    ServerSlot,
    parse_chat_request,
    resolve_finish_reason,
    sse_chunks,
    stop_at,
)


def _drain(gen):
    """Return (pieces, generator return value)."""
    out = []
    while True:
        try:
            out.append(next(gen))
        except StopIteration as done:
            return out, done.value


@pytest.mark.parametrize(
    "pieces, stop, matched",
    [(["ab", "cSTOPd"], ["STOP"], True), (["ab"], ["STOP"], False), (["ab"], None, False), (["STOP"], "STOP", True)],
)
def test_stop_at_reports_match(pieces, stop, matched):
    assert _drain(stop_at(iter(pieces), stop))[1] is matched


def test_stop_string_wins_over_length():
    assert resolve_finish_reason("length", True) == "stop"
    assert resolve_finish_reason("length", False) == "length"
    assert resolve_finish_reason("stop", False) == "stop"


def _last_finish(gen):
    import json

    events = list(sse_chunks("m", gen))
    return json.loads(events[-2][len(b"data: ") :])["choices"][0]["finish_reason"]


def test_sse_final_chunk_uses_generator_return():
    def pieces():
        yield "a"
        return "length"

    assert _last_finish(pieces()) == "length"
    assert _last_finish(p for p in ["a"]) == "stop"  # a generator that returns None


def test_top_p_defaults_to_openai_one():
    # 0.9 was the default but was never applied; honouring it would change requests that omit top_p
    assert parse_chat_request({"messages": [{"role": "user", "content": "x"}]}, ServerConfig("m.gguf")).top_p == 1.0
    assert ChatRequest(messages=[]).top_p == 1.0


def test_non_stream_stop_string_overrides_length():
    server = PythonServer(ServerConfig("m.gguf"))
    server.model = Mock()
    server.model.get_vocab().tokenize.return_value = [1]
    slot = Mock(is_processing=False, finish_reason="length")
    slot.reset.side_effect = lambda: setattr(slot, "is_processing", False)  # frees the slot, as ServerSlot.reset does
    slot.process_and_generate.return_value = "abcSTOPdef"
    server.slots = [slot]
    server._messages_to_prompt = lambda messages: "p"
    request = ChatRequest(messages=[ChatMessage("user", "x")], max_tokens=10, stop=["STOP"])
    choice = server.process_chat_completion(request).choices[0]
    assert (choice.message.content, choice.finish_reason) == ("abc", "stop")
    slot.finish_reason = "length"
    slot.process_and_generate.return_value = "abcdef"
    assert server.process_chat_completion(request).choices[0].finish_reason == "length"


@pytest.fixture(scope="module")
def model(model_path):
    import cyllama.llama.llama_cpp as cy

    m = cy.LlamaModel(model_path, verbose=False)
    yield m
    del m
    gc.collect()


def _slot(model, n_ctx=256):
    return ServerSlot(0, model, ServerConfig(model_path="unused.gguf", n_ctx=n_ctx, n_batch=256, n_parallel=1))


def _request(**kw):
    return ChatRequest(messages=[ChatMessage("user", "x")], temperature=0.0, **kw)


def test_max_tokens_is_length(model):
    slot = _slot(model)
    _, reason = _drain(slot.stream("Count: 1, 2, 3,", max_tokens=3, request=_request()))
    assert reason == slot.finish_reason == "length"
    del slot
    gc.collect()


def test_end_of_generation_is_stop(model, monkeypatch):
    # The model's own end-of-generation token: sample_tokens ends before max_tokens
    real = srv.sample_tokens
    monkeypatch.setattr(srv, "sample_tokens", lambda *a, **k: (t for i, t in enumerate(real(*a, **k)) if i < 1))
    slot = _slot(model)
    _, reason = _drain(slot.stream("Hello", max_tokens=5, request=_request()))
    assert reason == "stop"
    del slot
    gc.collect()


def test_over_long_prompt_is_length(model):
    slot = _slot(model, n_ctx=32)
    pieces, reason = _drain(slot.stream("word " * 64, max_tokens=5, request=_request()))
    assert (pieces, reason) == ([], "length")
    del slot
    gc.collect()


def test_request_top_p_reaches_sampler(model, monkeypatch):
    seen = []
    real = srv.build_sampler
    monkeypatch.setattr(
        srv, "build_sampler", lambda vocab, config, n_ctx: seen.append(config.top_p) or real(vocab, config, n_ctx)
    )
    slot = _slot(model)
    _drain(slot.stream("Hello", max_tokens=1, request=_request(top_p=0.5)))
    assert seen[-1] == 0.5
    del slot
    gc.collect()


_MSG = [{"role": "user", "content": "x"}]


@pytest.mark.parametrize(
    "extra, field",
    [
        ({"tools": [{"type": "function", "function": {"name": "f"}}]}, "tools"),
        ({"functions": [{"name": "f"}]}, "functions"),
        ({"response_format": {"type": "json_object"}}, "response_format"),
        ({"response_format": {"type": "json_schema", "json_schema": {}}}, "response_format"),
        ({"response_format": "json"}, "response_format"),
        ({"n": 2}, "n"),
        ({"logprobs": True}, "logprobs"),
    ],
)
def test_unsupported_feature_is_rejected(extra, field):
    with pytest.raises(srv.RequestError, match=f"^{field} is not supported"):
        parse_chat_request({"messages": _MSG, **extra}, ServerConfig("m.gguf"))


@pytest.mark.parametrize(
    "extra",
    [
        {"tools": []},
        {"tools": None},
        {"functions": []},
        {"response_format": {"type": "text"}},
        {"response_format": None},
        {"n": 1},
        {"logprobs": False},
        {"tool_choice": "auto"},
    ],
)
def test_default_values_pass(extra):
    # Some clients send these on every request; rejecting them would break working uses
    assert parse_chat_request({"messages": _MSG, **extra}, ServerConfig("m.gguf")).messages
