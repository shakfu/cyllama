"""EmbeddedServer must bind exactly the configured host, never a wider address."""

import ctypes
import errno
import signal
import socket
import sys

import pytest

from cyllama.llama.server.embedded import EmbeddedServer, _listen_url
from cyllama.llama.server.python import ServerConfig


@pytest.mark.parametrize(
    "host, expected",
    [
        ("127.0.0.1", "http://127.0.0.1:8080"),
        ("localhost", "http://localhost:8080"),
        ("0.0.0.0", "http://0.0.0.0:8080"),
        ("::1", "http://[::1]:8080"),
        ("[::1]", "http://[::1]:8080"),
    ],
)
def test_listen_url(host, expected):
    assert _listen_url(host, 8080) == expected


class _NoModelServer(EmbeddedServer):
    def load_model(self):
        return True


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _can_bind(addr, port):
    with socket.socket() as s:
        try:
            s.bind((addr, port))
        except OSError as e:
            assert e.errno == errno.EADDRINUSE
            return False
        return True


# 127.0.0.2 is a loopback address only on Linux. Binding it succeeds while
# the server holds 127.0.0.1:port, and fails if the server holds 0.0.0.0:port.
@pytest.mark.skipif(sys.platform != "linux", reason="needs 127.0.0.0/8 loopback")
@pytest.mark.parametrize(
    "host, wildcard",
    [("127.0.0.1", False), ("localhost", False), ("0.0.0.0", True)],
)
def test_bind_address(host, wildcard):
    port = _free_port()
    saved = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    server = _NoModelServer(ServerConfig(model_path="unused.gguf", host=host, port=port))
    try:
        assert server.start()
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass
        assert _can_bind("127.0.0.2", port) is not wildcard
    finally:
        server.stop()
        del server  # __dealloc__ frees the Mongoose manager and closes the socket
        signal.signal(signal.SIGINT, saved[0])
        signal.signal(signal.SIGTERM, saved[1])


def _start_stop():
    """Start and stop a server on a free port, restoring signal handlers.

    Mongoose logs through C stdio, which is fully buffered when stdout is not a
    terminal, so flush it for capfd to see the output.
    """
    saved = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    server = _NoModelServer(ServerConfig(model_path="unused.gguf", port=_free_port()))
    try:
        assert server.start()
        server.stop()
    finally:
        del server
        ctypes.CDLL(None).fflush(None)
        signal.signal(signal.SIGINT, saved[0])
        signal.signal(signal.SIGTERM, saved[1])


_needs_libc = pytest.mark.skipif(sys.platform == "win32", reason="flushes C stdio via ctypes.CDLL(None)")


@_needs_libc
def test_mongoose_logs_quiet_by_default(capfd, monkeypatch):
    monkeypatch.delenv("CYLLAMA_MONGOOSE_LOG", raising=False)
    _start_stop()
    assert "mongoose.c" not in capfd.readouterr().out


@_needs_libc
@pytest.mark.parametrize("value, logged", [("debug", True), ("DEBUG", True), ("error", False), ("bogus", False)])
def test_mongoose_log_env(capfd, monkeypatch, value, logged):
    monkeypatch.setenv("CYLLAMA_MONGOOSE_LOG", value)
    try:
        _start_stop()  # start() re-reads the variable; mg_listen logs at debug
        assert ("mg_listen" in capfd.readouterr().out) is logged
    finally:
        monkeypatch.delenv("CYLLAMA_MONGOOSE_LOG")
        _start_stop()  # restore the quiet default for later tests
        capfd.readouterr()
