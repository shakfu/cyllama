"""
JSON-RPC 2.0 transport layer for ACP and MCP protocols.

Provides both synchronous and asynchronous JSON-RPC messaging over stdio and HTTP.
"""

import json
import sys
import threading
import queue
import logging
from typing import Any, Callable, Dict, Generator, Optional, Union
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger(__name__)


class ErrorCode(IntEnum):
    """JSON-RPC 2.0 error codes."""
    # Standard JSON-RPC errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # ACP-specific errors (-32000 to -32099)
    SESSION_NOT_FOUND = -32001
    PERMISSION_DENIED = -32002
    OPERATION_CANCELLED = -32003
    AUTHENTICATION_REQUIRED = -32004


@dataclass
class JsonRpcError:
    """JSON-RPC error object."""
    code: int
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> dict:
        result = {"code": self.code, "message": self.message}
        if self.data is not None:
            result["data"] = self.data
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "JsonRpcError":
        return cls(
            code=d["code"],
            message=d["message"],
            data=d.get("data")
        )


@dataclass
class JsonRpcRequest:
    """JSON-RPC request message."""
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None  # None for notifications

    def to_dict(self) -> dict:
        result = {"jsonrpc": "2.0", "method": self.method}
        if self.params is not None:
            result["params"] = self.params
        if self.id is not None:
            result["id"] = self.id
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "JsonRpcRequest":
        return cls(
            method=d["method"],
            params=d.get("params"),
            id=d.get("id")
        )

    @property
    def is_notification(self) -> bool:
        return self.id is None


@dataclass
class JsonRpcResponse:
    """JSON-RPC response message."""
    id: Union[str, int, None]
    result: Optional[Any] = None
    error: Optional[JsonRpcError] = None

    def to_dict(self) -> dict:
        result = {"jsonrpc": "2.0", "id": self.id}
        if self.error is not None:
            result["error"] = self.error.to_dict()
        else:
            result["result"] = self.result
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "JsonRpcResponse":
        error = None
        if "error" in d:
            error = JsonRpcError.from_dict(d["error"])
        return cls(
            id=d.get("id"),
            result=d.get("result"),
            error=error
        )

    @property
    def is_error(self) -> bool:
        return self.error is not None


JsonRpcMessage = Union[JsonRpcRequest, JsonRpcResponse]


def parse_message(data: str) -> JsonRpcMessage:
    """Parse a JSON-RPC message from a string."""
    try:
        d = json.loads(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if "jsonrpc" not in d or d["jsonrpc"] != "2.0":
        raise ValueError("Invalid JSON-RPC version")

    if "method" in d:
        return JsonRpcRequest.from_dict(d)
    else:
        return JsonRpcResponse.from_dict(d)


def serialize_message(msg: JsonRpcMessage) -> str:
    """Serialize a JSON-RPC message to a string."""
    return json.dumps(msg.to_dict())


class StdioTransport:
    """
    JSON-RPC transport over stdio.

    Messages are newline-delimited JSON.
    """

    def __init__(
        self,
        input_stream=None,
        output_stream=None,
    ):
        self._input = input_stream or sys.stdin
        self._output = output_stream or sys.stdout
        self._lock = threading.Lock()

    def read_message(self) -> Optional[JsonRpcMessage]:
        """Read a single JSON-RPC message from input."""
        try:
            line = self._input.readline()
            if not line:
                return None  # EOF
            line = line.strip()
            if not line:
                return None
            return parse_message(line)
        except Exception as e:
            logger.error("Error reading message: %s", e)
            return None

    def write_message(self, msg: JsonRpcMessage) -> None:
        """Write a JSON-RPC message to output."""
        with self._lock:
            data = serialize_message(msg)
            self._output.write(data + "\n")
            self._output.flush()

    def read_messages(self) -> Generator[JsonRpcMessage, None, None]:
        """Generator that yields messages until EOF."""
        while True:
            msg = self.read_message()
            if msg is None:
                break
            yield msg


class JsonRpcServer:
    """
    JSON-RPC server that dispatches methods to handlers.

    Supports both synchronous and async-bridged operation.
    """

    def __init__(self, transport: StdioTransport):
        self._transport = transport
        self._handlers: Dict[str, Callable] = {}
        self._pending_requests: Dict[Union[str, int], threading.Event] = {}
        self._pending_responses: Dict[Union[str, int], JsonRpcResponse] = {}
        self._request_id = 0
        self._lock = threading.Lock()
        self._running = False

        # Queue for bridging sync inner agents with async protocol
        self._outgoing_queue: queue.Queue = queue.Queue()

    def register(self, method: str, handler: Callable) -> None:
        """Register a method handler."""
        self._handlers[method] = handler

    def unregister(self, method: str) -> None:
        """Unregister a method handler."""
        self._handlers.pop(method, None)

    def _next_request_id(self) -> int:
        with self._lock:
            self._request_id += 1
            return self._request_id

    def send_notification(self, method: str, params: Optional[dict] = None) -> None:
        """Send a notification (no response expected)."""
        msg = JsonRpcRequest(method=method, params=params, id=None)
        self._transport.write_message(msg)

    def send_request(
        self,
        method: str,
        params: Optional[dict] = None,
        timeout: Optional[float] = None
    ) -> JsonRpcResponse:
        """
        Send a request and wait for response.

        Args:
            method: RPC method name
            params: Method parameters
            timeout: Timeout in seconds (None for no timeout)

        Returns:
            JsonRpcResponse from the other side

        Raises:
            TimeoutError: If timeout exceeded
            RuntimeError: If response contains error
        """
        request_id = self._next_request_id()
        event = threading.Event()

        with self._lock:
            self._pending_requests[request_id] = event

        msg = JsonRpcRequest(method=method, params=params, id=request_id)
        self._transport.write_message(msg)

        # Wait for response
        if not event.wait(timeout=timeout):
            with self._lock:
                self._pending_requests.pop(request_id, None)
            raise TimeoutError(f"Request {method} timed out")

        with self._lock:
            response = self._pending_responses.pop(request_id)
            self._pending_requests.pop(request_id, None)

        return response

    def _handle_message(self, msg: JsonRpcMessage) -> Optional[JsonRpcResponse]:
        """Handle an incoming message."""
        if isinstance(msg, JsonRpcResponse):
            # Response to our request
            with self._lock:
                if msg.id in self._pending_requests:
                    self._pending_responses[msg.id] = msg
                    self._pending_requests[msg.id].set()
            return None

        # It's a request
        request = msg

        if request.method not in self._handlers:
            if request.is_notification:
                logger.warning("Unknown notification: %s", request.method)
                return None
            return JsonRpcResponse(
                id=request.id,
                error=JsonRpcError(
                    code=ErrorCode.METHOD_NOT_FOUND,
                    message=f"Method not found: {request.method}"
                )
            )

        try:
            handler = self._handlers[request.method]
            result = handler(request.params or {})

            if request.is_notification:
                return None

            return JsonRpcResponse(id=request.id, result=result)

        except Exception as e:
            logger.exception("Error handling %s", request.method)
            if request.is_notification:
                return None
            return JsonRpcResponse(
                id=request.id,
                error=JsonRpcError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=str(e)
                )
            )

    def serve(self) -> None:
        """
        Main server loop. Blocks until EOF on input.
        """
        self._running = True
        logger.info("JSON-RPC server starting")

        try:
            for msg in self._transport.read_messages():
                if not self._running:
                    break

                response = self._handle_message(msg)
                if response is not None:
                    self._transport.write_message(response)
        finally:
            self._running = False
            logger.info("JSON-RPC server stopped")

    def stop(self) -> None:
        """Signal the server to stop."""
        self._running = False


class AsyncBridge:
    """
    Bridge between synchronous agent execution and async protocol handling.

    Uses a queue to allow the sync agent to emit events that are sent
    as notifications by the protocol handler.
    """

    def __init__(self, server: JsonRpcServer):
        self._server = server
        self._queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start the async bridge worker."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop the async bridge worker."""
        self._running = False
        self._queue.put(None)  # Sentinel to wake up worker
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)

    def _worker(self) -> None:
        """Worker thread that sends queued notifications."""
        while self._running:
            try:
                item = self._queue.get(timeout=0.1)
                if item is None:
                    continue
                method, params = item
                self._server.send_notification(method, params)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Error in async bridge worker: %s", e)

    def send_notification(self, method: str, params: Optional[dict] = None) -> None:
        """
        Queue a notification to be sent asynchronously.

        This is safe to call from the synchronous agent execution thread.
        """
        self._queue.put((method, params))

    def send_notification_sync(self, method: str, params: Optional[dict] = None) -> None:
        """
        Send a notification immediately (blocking).

        Use this when you need the notification sent before continuing.
        """
        self._server.send_notification(method, params)
