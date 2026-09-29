"""EmbeddedServer must bind exactly the configured host, never a wider address."""

import errno
import signal
import socket
import sys

import pytest

from cyllama.llama.server.embedded import EmbeddedServer, _bind_address
from cyllama.llama.server.python import ServerConfig


@pytest.mark.parametrize(
    "host, expected",
    [
        ("127.0.0.1", ("127.0.0.1", False)),
        ("localhost", ("localhost", False)),
        ("0.0.0.0", ("0.0.0.0", False)),
        ("::1", ("::1", True)),
        ("[::1]", ("::1", True)),
    ],
)
def test_bind_address(host, expected):
    assert _bind_address(host) == expected


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
def test_bound_address(host, wildcard):
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
        signal.signal(signal.SIGINT, saved[0])
        signal.signal(signal.SIGTERM, saved[1])


@pytest.mark.skipif(not hasattr(socket, "SO_REUSEPORT"), reason="needs SO_REUSEPORT")
def test_port_not_shared():
    """httplib's default SO_REUSEPORT would let another socket bind the listening port."""
    port = _free_port()
    saved = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    server = _NoModelServer(ServerConfig(model_path="unused.gguf", port=port))
    try:
        assert server.start()
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            with pytest.raises(OSError) as exc:
                s.bind(("127.0.0.1", port))
            assert exc.value.errno == errno.EADDRINUSE
    finally:
        server.stop()
        signal.signal(signal.SIGINT, saved[0])
        signal.signal(signal.SIGTERM, saved[1])


def test_bind_failure_returns_false():
    saved = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        server = _NoModelServer(ServerConfig(model_path="unused.gguf", port=busy.getsockname()[1]))
        try:
            assert not server.start()
            server.stop()  # no-op after a failed start
        finally:
            signal.signal(signal.SIGINT, saved[0])
            signal.signal(signal.SIGTERM, saved[1])
