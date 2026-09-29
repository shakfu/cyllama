"""SSE streaming in both servers, slot handling in EmbeddedServer, and the stop_at helper."""

import http.client
import json
import signal
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from cyllama.llama.server.embedded import EmbeddedServer
from cyllama.llama.server.python import PythonServer, ServerConfig, stop_at


@pytest.mark.parametrize(
    "pieces, stop, expected",
    [
        (["Hel", "lo"], None, ["Hel", "lo"]),
        (["Hel", "lo"], [], ["Hel", "lo"]),
        (["Hel", "lo"], [""], ["Hel", "lo"]),  # an empty stop string never matches
        (["ab", "cSTOPd"], ["STOP"], ["ab", "c"]),
        (["abST", "OPd"], ["STOP"], ["ab"]),  # held back, then cut
        (["abST", "x"], ["STOP"], ["ab", "STx"]),  # held back, then released
        (["abS"], ["STOP"], ["ab", "S"]),  # released at the end
        (["xSTOP"], "STOP", ["x"]),  # a bare string is one stop
        (["a", "YbX"], ["X", "Yb"], ["a"]),  # earliest match wins, not list order
        (["STOP"], ["STOP"], []),
    ],
)
def test_stop_at(pieces, stop, expected):
    assert list(stop_at(iter(pieces), stop)) == expected


def test_stop_at_stops_consuming():
    consumed = []

    def pieces():
        for p in ["a", "STOP", "never"]:
            consumed.append(p)
            yield p

    assert "".join(stop_at(pieces(), ["STOP"])) == "a"
    assert consumed == ["a", "STOP"]


class _Fake:
    """No model; generation yields canned pieces and records when it starts and closes."""

    pieces = ["Hel", "lo"]
    delay = 0.0

    def load_model(self):
        self.started = threading.Event()
        self.closed = threading.Event()
        return True

    def _fake_pieces(self):
        self.started.set()
        try:
            for p in self.pieces:
                time.sleep(self.delay)
                yield p
        finally:
            self.closed.set()


class _FakeEmbedded(_Fake, EmbeddedServer):
    def _generate(self, prompt, request):
        return self._fake_pieces()


class _FakePython(_Fake, PythonServer):
    def stream_chat_completion(self, request):
        return self._fake_pieces()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(params=["embedded", "python"])
def fake_server(request):
    saved = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    servers = []
    base = _FakeEmbedded if request.param == "embedded" else _FakePython

    def start(pieces=None, delay=0.0):
        port = _free_port()
        server = type("Fake", (base,), {"pieces": pieces or _Fake.pieces, "delay": delay})(
            ServerConfig(model_path="unused.gguf", port=port)
        )
        server.port = port
        server.kind = request.param
        assert server.start()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.stop()
    signal.signal(signal.SIGINT, saved[0])
    signal.signal(signal.SIGTERM, saved[1])


def _post_stream(port, body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", "/v1/chat/completions", body=json.dumps(body), headers={"Content-Type": "application/json"})
    conn.raw_sock = conn.sock  # getresponse() detaches it from a close-delimited response
    return conn, conn.getresponse()


def _events(raw):
    return [e[len("data: ") :] for e in raw.decode("utf-8").split("\n\n") if e]


def test_stream_events(fake_server):
    server = fake_server()
    body = {"messages": [{"role": "user", "content": "hi"}], "stream": True, "model": "m"}
    conn, resp = _post_stream(server.port, body)
    try:
        assert resp.status == 200
        assert resp.getheader("Content-Type").startswith("text/event-stream")
        if server.kind == "embedded":
            assert resp.getheader("Transfer-Encoding") == "chunked"
        events = _events(resp.read())
    finally:
        conn.close()

    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert {c["object"] for c in chunks} == {"chat.completion.chunk"}
    assert len({c["id"] for c in chunks}) == 1
    assert all(c["model"] == "m" for c in chunks)
    deltas = [c["choices"][0]["delta"] for c in chunks]
    assert deltas[0] == {"role": "assistant"}
    assert "".join(d.get("content", "") for d in deltas) == "Hello"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert server.closed.wait(2)


def test_client_disconnect_closes_generator(fake_server):
    server = fake_server(["x"] * 200, 0.01)
    conn, resp = _post_stream(server.port, {"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert server.started.wait(2)
    conn.raw_sock.shutdown(socket.SHUT_RDWR)
    resp.close()
    conn.close()
    # Generation stops at a failed write, not after all 200 pieces (2 s).
    assert server.closed.wait(1.5)


def test_stop_aborts_open_stream(fake_server):
    """EmbeddedServer.stop() ends an open stream at the next chunk and closes its generator."""
    server = fake_server(["x"] * 200, 0.01)
    if server.kind == "python":
        pytest.skip("PythonServer.stop() waits for the request in progress")
    conn, resp = _post_stream(server.port, {"messages": [{"role": "user", "content": "hi"}], "stream": True})
    try:
        assert server.started.wait(2)
        t0 = time.monotonic()
        server.stop()
        assert time.monotonic() - t0 < 1.0  # not the 2 s the stream would take to finish
        assert server.closed.is_set()
    finally:
        conn.close()


# --- real model --------------------------------------------------------------


@pytest.fixture(params=["embedded", "python"])
def model_server(request, model_path):
    saved = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    port = _free_port()
    cls = EmbeddedServer if request.param == "embedded" else PythonServer
    server = cls(ServerConfig(model_path=model_path, port=port, n_ctx=256, n_parallel=1))
    assert server.start()
    try:
        yield port
    finally:
        server.stop()
        signal.signal(signal.SIGINT, saved[0])
        signal.signal(signal.SIGTERM, saved[1])


def test_model_stream(model_server):
    body = {"messages": [{"role": "user", "content": "Say hello."}], "stream": True, "max_tokens": 8}
    conn, resp = _post_stream(model_server, body)
    try:
        assert resp.status == 200
        events = _events(resp.read())
    finally:
        conn.close()
    assert events[-1] == "[DONE]"
    content = "".join(json.loads(e)["choices"][0]["delta"].get("content", "") for e in events[:-1])
    assert content


def test_concurrent_requests_share_one_slot(model_server):
    """With n_parallel=1, later requests wait for the slot instead of failing."""

    def complete(_):
        conn = http.client.HTTPConnection("127.0.0.1", model_server, timeout=60)
        try:
            body = json.dumps({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4})
            conn.request("POST", "/v1/chat/completions", body=body)
            return conn.getresponse().status
        finally:
            conn.close()

    with ThreadPoolExecutor(3) as pool:
        assert list(pool.map(complete, range(3))) == [200, 200, 200]
