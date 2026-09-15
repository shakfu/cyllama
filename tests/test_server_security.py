"""API-key auth and request-size limits for PythonServer and EmbeddedServer."""

import http.client
import json
import signal
import socket
import threading

import pytest

from cyllama.llama.server.embedded import EmbeddedServer
from cyllama.llama.server.python import PythonServer, ServerConfig, check_request, exposed_without_auth

KEY = "s3cret"


def _config(**kwargs):
    kwargs.setdefault("model_path", "unused.gguf")
    return ServerConfig(**kwargs)


# --- check_request ------------------------------------------------------------


@pytest.mark.parametrize(
    "api_key, path, authorization, length, expected",
    [
        (None, "/v1/models", None, 0, None),
        (KEY, "/v1/models", None, 0, 401),
        (KEY, "/v1/models", "Bearer wrong", 0, 401),
        (KEY, "/v1/models", KEY, 0, 401),  # scheme is required
        (KEY, "/v1/models", f"Bearer {KEY}", 0, None),
        (KEY, "/v1/models", f"bearer {KEY}", 0, None),  # scheme is case-insensitive
        (KEY, "/v1/models", f"Basic {KEY}", 0, 401),
        (KEY, "/v1/models", f"Bearer {KEY} ", 0, 401),
        # Non-ASCII key: clients send UTF-8, both servers decode headers as latin-1.
        ("k\u00e9y", "/v1/models", "Bearer k\u00e9y".encode("utf-8").decode("latin-1"), 0, None),
        ("k\u00e9y", "/v1/models", "Bearer k\u0100y", 0, 401),
        (KEY, "/health", None, 0, None),
        ("", "/v1/models", None, 0, None),  # empty key disables auth
        (None, "/v1/chat/completions", None, 101, 413),
        (None, "/v1/chat/completions", None, 100, None),
        (KEY, "/v1/chat/completions", None, 101, 401),  # auth is checked first
    ],
)
def test_check_request(api_key, path, authorization, length, expected):
    result = check_request(_config(api_key=api_key, max_body_bytes=100), path, authorization, length)
    assert (result and result[0]) == expected


@pytest.mark.parametrize(
    "host, api_key, expected",
    [
        ("127.0.0.1", None, False),
        ("127.0.0.2", None, False),
        ("localhost", None, False),
        ("::1", None, False),
        ("[::1]", None, False),
        ("0.0.0.0", None, True),
        ("::", None, True),
        ("192.168.1.10", None, True),
        ("myhost.lan", None, True),
        ("0.0.0.0", KEY, False),
    ],
)
def test_exposed_without_auth(host, api_key, expected):
    assert exposed_without_auth(_config(host=host, api_key=api_key)) is expected


# --- live servers ---------------------------------------------------------------


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _NoModelPythonServer(PythonServer):
    def load_model(self):
        return True


class _NoModelEmbeddedServer(EmbeddedServer):
    def load_model(self):
        return True


@pytest.fixture(params=["python", "embedded"])
def serve(request):
    """Start a server of each kind with model loading stubbed; return its port."""
    started = []
    saved = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)

    def start(**kwargs):
        port = _free_port()
        config = _config(port=port, **kwargs)
        if request.param == "python":
            server = _NoModelPythonServer(config)
            assert server.start()
            started.append(server.stop)
        else:
            server = _NoModelEmbeddedServer(config)
            assert server.start()
            loop = threading.Thread(target=server.wait_for_shutdown, daemon=True)
            loop.start()

            def stop():
                server._signal_handler(signal.SIGTERM, None)
                loop.join(timeout=5)
                server.stop()

            started.append(stop)
        return port

    yield start
    for stop in started:
        stop()
    signal.signal(signal.SIGINT, saved[0])
    signal.signal(signal.SIGTERM, saved[1])


def _request(port, method, path, headers=None, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read() or b"null")
    finally:
        conn.close()


def test_no_key_allows_requests(serve):
    port = serve()
    assert _request(port, "GET", "/v1/models")[0] == 200


def test_key_required(serve):
    port = serve(api_key=KEY)
    status, body = _request(port, "GET", "/v1/models")
    assert status == 401
    assert "API key" in body["error"]["message"]
    assert _request(port, "GET", "/v1/models", {"Authorization": "Bearer wrong"})[0] == 401
    assert _request(port, "POST", "/v1/chat/completions", body=b"{}")[0] == 401


def test_valid_key_accepted(serve):
    port = serve(api_key=KEY)
    status, body = _request(port, "GET", "/v1/models", {"Authorization": f"Bearer {KEY}"})
    assert status == 200
    assert body["object"] == "list"


def test_health_is_public(serve):
    port = serve(api_key=KEY)
    assert _request(port, "GET", "/health") == (200, {"status": "ok"})


def test_oversized_body_rejected(serve):
    port = serve(api_key=KEY, max_body_bytes=1000)
    auth = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    status, body = _request(port, "POST", "/v1/chat/completions", auth, b"x" * 1001)
    assert status == 413
    assert "1000 bytes" in body["error"]["message"]
    # At the limit the body is parsed: invalid JSON, not 413.
    assert _request(port, "POST", "/v1/chat/completions", auth, b"x" * 1000)[0] == 400


def test_exposure_warning(caplog):
    server = _NoModelPythonServer(_config(host="127.0.0.1", port=_free_port()))
    server.config.host = "0.0.0.0"  # checked before bind; bind loopback to avoid opening the port
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(server, "_create_request_handler", lambda: (_ for _ in ()).throw(RuntimeError("stop")))
        with caplog.at_level("WARNING"):
            assert not server.start()
    assert "without an API key" in caplog.text


def test_embedded_loop_handles_signals_while_idle():
    """wait_for_shutdown must release the GIL and run signal handlers with no traffic."""
    import time

    from cyllama.llama.server import embedded

    saved = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    server = _NoModelEmbeddedServer(_config(port=_free_port()))
    assert server.start()
    # The first timer can only fire if the loop releases the GIL. The second
    # unblocks a regressed loop so the test fails instead of hanging.
    # raise_signal, not os.kill: on Windows os.kill(SIGTERM) calls TerminateProcess.
    kill = threading.Timer(0.3, signal.raise_signal, (signal.SIGTERM,))
    rescue = threading.Timer(5.0, setattr, (embedded, "_shutdown_requested", True))
    try:
        kill.start()
        rescue.start()
        t0 = time.monotonic()
        server.wait_for_shutdown()
        elapsed = time.monotonic() - t0
    finally:
        kill.cancel()
        rescue.cancel()
        server.stop()
        signal.signal(signal.SIGINT, saved[0])
        signal.signal(signal.SIGTERM, saved[1])
    assert server.signal_received == signal.SIGTERM
    assert elapsed < 2.0


def test_large_response_not_truncated(serve):
    # Well past the 4096-byte buffer the Mongoose reply wrapper once used, and
    # past Mongoose's 16 KiB IO chunk.
    alias = "a" * 100_000
    port = serve(model_alias=alias)
    status, body = _request(port, "GET", "/v1/models")
    assert status == 200
    assert body["data"][0]["id"] == alias


def test_unauthenticated_invalid_utf8_body_gets_401(serve):
    port = serve(api_key=KEY)
    assert _request(port, "POST", "/v1/chat/completions", body=b"\xff\xfe")[0] == 401


def test_authenticated_invalid_utf8_body_gets_400(serve):
    port = serve(api_key=KEY)
    auth = {"Authorization": f"Bearer {KEY}"}
    assert _request(port, "POST", "/v1/chat/completions", auth, b"\xff\xfe")[0] == 400


def test_embedded_stop_from_other_thread_waits_for_loop():
    """stop() must not touch the Mongoose manager while another thread polls it."""
    saved = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    port = _free_port()
    server = _NoModelEmbeddedServer(_config(port=port))
    try:
        assert server.start()
        loop = threading.Thread(target=server.wait_for_shutdown, daemon=True)
        loop.start()
        assert _request(port, "GET", "/health")[0] == 200
        server.stop()
        assert not loop.is_alive()
    finally:
        signal.signal(signal.SIGINT, saved[0])
        signal.signal(signal.SIGTERM, saved[1])


def test_embedded_loop_exit_by_exception_does_not_block_stop():
    """A raising signal handler ends the loop; stop() must not then wait for it."""
    import time

    class Interrupt(Exception):
        pass

    def raise_interrupt(signum, frame):
        raise Interrupt

    saved = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    server = _NoModelEmbeddedServer(_config(port=_free_port()))
    try:
        assert server.start()
        signal.signal(signal.SIGTERM, raise_interrupt)
        threading.Timer(0.2, signal.raise_signal, (signal.SIGTERM,)).start()
        with pytest.raises(Interrupt):
            server.wait_for_shutdown()
        t0 = time.monotonic()
        server.stop()
        assert time.monotonic() - t0 < 1.0
    finally:
        signal.signal(signal.SIGINT, saved[0])
        signal.signal(signal.SIGTERM, saved[1])
