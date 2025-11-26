"""Tests for JSON-RPC transport layer."""

import io
import json
import threading
import time
import pytest

from cyllama.agents.jsonrpc import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcError,
    ErrorCode,
    parse_message,
    serialize_message,
    StdioTransport,
    JsonRpcServer,
    AsyncBridge,
)


class TestJsonRpcMessages:
    """Tests for JSON-RPC message classes."""

    def test_request_to_dict(self):
        req = JsonRpcRequest(method="test", params={"foo": "bar"}, id=1)
        d = req.to_dict()

        assert d["jsonrpc"] == "2.0"
        assert d["method"] == "test"
        assert d["params"] == {"foo": "bar"}
        assert d["id"] == 1

    def test_request_from_dict(self):
        d = {"jsonrpc": "2.0", "method": "test", "params": {"x": 1}, "id": 42}
        req = JsonRpcRequest.from_dict(d)

        assert req.method == "test"
        assert req.params == {"x": 1}
        assert req.id == 42

    def test_notification_has_no_id(self):
        req = JsonRpcRequest(method="notify", params={})
        assert req.is_notification
        assert "id" not in req.to_dict()

    def test_response_to_dict_success(self):
        resp = JsonRpcResponse(id=1, result={"answer": 42})
        d = resp.to_dict()

        assert d["jsonrpc"] == "2.0"
        assert d["id"] == 1
        assert d["result"] == {"answer": 42}
        assert "error" not in d

    def test_response_to_dict_error(self):
        error = JsonRpcError(code=-32600, message="Invalid request")
        resp = JsonRpcResponse(id=1, error=error)
        d = resp.to_dict()

        assert d["jsonrpc"] == "2.0"
        assert d["id"] == 1
        assert d["error"]["code"] == -32600
        assert d["error"]["message"] == "Invalid request"
        assert "result" not in d

    def test_response_from_dict(self):
        d = {"jsonrpc": "2.0", "id": 5, "result": "success"}
        resp = JsonRpcResponse.from_dict(d)

        assert resp.id == 5
        assert resp.result == "success"
        assert not resp.is_error

    def test_response_from_dict_with_error(self):
        d = {
            "jsonrpc": "2.0",
            "id": 5,
            "error": {"code": -32601, "message": "Method not found"}
        }
        resp = JsonRpcResponse.from_dict(d)

        assert resp.id == 5
        assert resp.is_error
        assert resp.error.code == -32601


class TestMessageParsing:
    """Tests for message parsing and serialization."""

    def test_parse_request(self):
        data = '{"jsonrpc": "2.0", "method": "test", "id": 1}'
        msg = parse_message(data)

        assert isinstance(msg, JsonRpcRequest)
        assert msg.method == "test"
        assert msg.id == 1

    def test_parse_response(self):
        data = '{"jsonrpc": "2.0", "result": "ok", "id": 1}'
        msg = parse_message(data)

        assert isinstance(msg, JsonRpcResponse)
        assert msg.result == "ok"

    def test_parse_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_message("not json")

    def test_parse_invalid_version(self):
        with pytest.raises(ValueError, match="Invalid JSON-RPC version"):
            parse_message('{"jsonrpc": "1.0", "method": "test"}')

    def test_serialize_request(self):
        req = JsonRpcRequest(method="hello", id=1)
        data = serialize_message(req)
        parsed = json.loads(data)

        assert parsed["method"] == "hello"
        assert parsed["id"] == 1

    def test_roundtrip(self):
        original = JsonRpcRequest(method="test", params={"a": 1}, id=99)
        serialized = serialize_message(original)
        parsed = parse_message(serialized)

        assert parsed.method == original.method
        assert parsed.params == original.params
        assert parsed.id == original.id


class TestStdioTransport:
    """Tests for stdio transport."""

    def test_read_message(self):
        input_data = '{"jsonrpc": "2.0", "method": "test", "id": 1}\n'
        input_stream = io.StringIO(input_data)
        output_stream = io.StringIO()

        transport = StdioTransport(input_stream, output_stream)
        msg = transport.read_message()

        assert isinstance(msg, JsonRpcRequest)
        assert msg.method == "test"

    def test_write_message(self):
        input_stream = io.StringIO()
        output_stream = io.StringIO()

        transport = StdioTransport(input_stream, output_stream)
        req = JsonRpcRequest(method="hello", id=1)
        transport.write_message(req)

        output_stream.seek(0)
        line = output_stream.readline()
        parsed = json.loads(line)

        assert parsed["method"] == "hello"

    def test_read_messages_generator(self):
        input_data = (
            '{"jsonrpc": "2.0", "method": "a", "id": 1}\n'
            '{"jsonrpc": "2.0", "method": "b", "id": 2}\n'
        )
        input_stream = io.StringIO(input_data)
        output_stream = io.StringIO()

        transport = StdioTransport(input_stream, output_stream)
        messages = list(transport.read_messages())

        assert len(messages) == 2
        assert messages[0].method == "a"
        assert messages[1].method == "b"


class TestJsonRpcServer:
    """Tests for JSON-RPC server."""

    def test_register_handler(self):
        transport = StdioTransport(io.StringIO(), io.StringIO())
        server = JsonRpcServer(transport)

        handler_called = []

        def handler(params):
            handler_called.append(params)
            return {"result": "ok"}

        server.register("test", handler)

        # Simulate receiving a request
        req = JsonRpcRequest(method="test", params={"x": 1}, id=1)
        response = server._handle_message(req)

        assert len(handler_called) == 1
        assert handler_called[0] == {"x": 1}
        assert response.result == {"result": "ok"}

    def test_method_not_found(self):
        transport = StdioTransport(io.StringIO(), io.StringIO())
        server = JsonRpcServer(transport)

        req = JsonRpcRequest(method="unknown", id=1)
        response = server._handle_message(req)

        assert response.is_error
        assert response.error.code == ErrorCode.METHOD_NOT_FOUND

    def test_handler_exception(self):
        transport = StdioTransport(io.StringIO(), io.StringIO())
        server = JsonRpcServer(transport)

        def bad_handler(params):
            raise ValueError("oops")

        server.register("bad", bad_handler)

        req = JsonRpcRequest(method="bad", id=1)
        response = server._handle_message(req)

        assert response.is_error
        assert response.error.code == ErrorCode.INTERNAL_ERROR
        assert "oops" in response.error.message

    def test_notification_no_response(self):
        transport = StdioTransport(io.StringIO(), io.StringIO())
        server = JsonRpcServer(transport)

        handler_called = []

        def handler(params):
            handler_called.append(True)

        server.register("notify", handler)

        # Notification has no id
        notification = JsonRpcRequest(method="notify", params={})
        response = server._handle_message(notification)

        assert handler_called == [True]
        assert response is None  # No response for notifications

    def test_send_notification(self):
        output_stream = io.StringIO()
        transport = StdioTransport(io.StringIO(), output_stream)
        server = JsonRpcServer(transport)

        server.send_notification("update", {"status": "done"})

        output_stream.seek(0)
        line = output_stream.readline()
        parsed = json.loads(line)

        assert parsed["method"] == "update"
        assert parsed["params"] == {"status": "done"}
        assert "id" not in parsed


class TestAsyncBridge:
    """Tests for async bridge."""

    def test_queue_notification(self):
        output_stream = io.StringIO()
        transport = StdioTransport(io.StringIO(), output_stream)
        server = JsonRpcServer(transport)

        bridge = AsyncBridge(server)
        bridge.start()

        try:
            bridge.send_notification("test", {"value": 123})
            time.sleep(0.2)  # Give worker time to process
        finally:
            bridge.stop()

        output_stream.seek(0)
        line = output_stream.readline()
        parsed = json.loads(line)

        assert parsed["method"] == "test"
        assert parsed["params"]["value"] == 123

    def test_sync_notification(self):
        output_stream = io.StringIO()
        transport = StdioTransport(io.StringIO(), output_stream)
        server = JsonRpcServer(transport)

        bridge = AsyncBridge(server)

        # Sync notification doesn't need worker thread
        bridge.send_notification_sync("immediate", {"x": 1})

        output_stream.seek(0)
        line = output_stream.readline()
        parsed = json.loads(line)

        assert parsed["method"] == "immediate"
