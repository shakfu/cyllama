# Mongoose-based HTTP server for cyllama
# High-performance alternative to Python HTTP server

import json
import logging
import signal
import threading
import time
import sys
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

# Import Mongoose C API
from .mongoose cimport *

# Import existing server logic
from .embedded import ServerConfig, ServerSlot, ChatMessage, ChatRequest, ChatResponse, ChatChoice

# Global shutdown flag for signal handling (following pymongoose pattern)
_shutdown_requested = False


cdef class MongooseConnection:
    """Wrapper for mg_connection pointer."""
    cdef mg_connection *_conn

    def __cinit__(self):
        self._conn = NULL

    @property
    def is_valid(self):
        return self._conn != NULL

    def send_json(self, data: dict, status_code: int = 200):
        """Send JSON response."""
        if not self.is_valid:
            return False

        json_str = json.dumps(data)
        headers = "Content-Type: application/json\r\n"

        cdef bytes json_bytes = json_str.encode('utf-8')
        cdef bytes headers_bytes = headers.encode('utf-8')

        # Extract C pointers from bytes objects before nogil section
        cdef const char* headers_ptr = headers_bytes
        cdef const char* json_ptr = json_bytes

        # Send HTTP reply without GIL for better performance
        self._send_reply_nogil(status_code, headers_ptr, json_ptr)
        return True

    cdef void _send_reply_nogil(self, int status_code, const char* headers_ptr, const char* json_ptr) nogil:
        """Send HTTP reply without holding GIL."""
        cyllama_mg_http_reply(self._conn, status_code, headers_ptr, "%s", json_ptr)

    def send_error(self, status_code: int, message: str):
        """Send error response."""
        error_data = {
            "error": {
                "type": "invalid_request_error",
                "message": message
            }
        }
        self.send_json(error_data, status_code)


cdef class MongooseServer:
    """High-performance Mongoose-based HTTP server for LLM inference."""

    cdef mg_mgr _mgr
    cdef mg_connection *_listener
    cdef object _config
    cdef object _model
    cdef list _slots
    cdef object _logger
    cdef bint _running
    cdef object _server_thread
    cdef int _signal_received

    @property
    def signal_received(self):
        """Get the received signal number (0 if no signal)."""
        return self._signal_received

    @signal_received.setter
    def signal_received(self, value):
        """Set the signal received value."""
        self._signal_received = value

    def __cinit__(self):
        cyllama_mg_mgr_init(&self._mgr)
        self._listener = NULL
        self._running = False
        self._server_thread = None
        self._signal_received = 0

    def __dealloc__(self):
        self.stop()
        cyllama_mg_mgr_free(&self._mgr)

    def __enter__(self):
        """Context manager entry."""
        if self.start():
            return self
        else:
            raise RuntimeError("Failed to start Mongoose server")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._logger.info("Context manager __exit__ called - starting graceful shutdown")
        self.stop()
        self._logger.info("Context manager __exit__ completed")

    def __init__(self, config: ServerConfig):
        self._config = config
        self._model = None
        self._slots = []
        self._logger = logging.getLogger(__name__)

    def load_model(self) -> bool:
        """Load the model and initialize slots."""
        try:
            self._logger.info(f"Loading model: {self._config.model_path}")

            # Import here to avoid circular imports
            from ..llama_cpp import LlamaModel

            # Load model
            self._model = LlamaModel(path_model=self._config.model_path)

            # Create slots using existing ServerSlot logic
            self._slots = []
            for i in range(self._config.n_parallel):
                slot = ServerSlot(i, self._model, self._config)
                self._slots.append(slot)

            self._logger.info(f"Model loaded successfully with {len(self._slots)} slots")
            self._logger.info("About to return True from load_model()")
            return True

        except Exception as e:
            self._logger.error(f"Failed to load model: {e}")
            return False

    def get_available_slot(self) -> Optional[ServerSlot]:
        """Get an available slot for processing."""
        for slot in self._slots:
            if not slot.is_processing:
                return slot
        return None

    def _signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM signals for graceful shutdown."""
        global _shutdown_requested
        self._logger.info(f"Received signal {signum}, requesting graceful shutdown...")
        _shutdown_requested = True
        self._signal_received = signum

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self._logger.debug("Signal handlers registered for SIGINT and SIGTERM")

    def start(self) -> bool:
        """Start the Mongoose server."""
        global _shutdown_requested

        # Reset global shutdown flag
        _shutdown_requested = False

        if not self.load_model():
            return False

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        try:
            # Create listener address - try different format that might work better
            if self._config.host == "127.0.0.1" or self._config.host == "localhost":
                listen_addr = f"http://0.0.0.0:{self._config.port}"
            else:
                listen_addr = f"http://{self._config.host}:{self._config.port}"

            self._logger.info(f"Attempting to bind to: {listen_addr}")
            addr_bytes = listen_addr.encode('utf-8')

            # Start HTTP listener with our event handler
            # Store reference to self to prevent garbage collection
            self._mgr.userdata = <void*>self

            self._logger.info("Calling cyllama_mg_http_listen...")
            self._listener = cyllama_mg_http_listen(&self._mgr, addr_bytes,
                                                  <mg_event_handler_t>_http_event_handler,
                                                  NULL)  # Use NULL, get server from mgr.userdata
            self._logger.info(f"cyllama_mg_http_listen returned: {<unsigned long>self._listener}")

            if self._listener == NULL:
                self._logger.error("Failed to create HTTP listener")
                return False

            self._running = True
            self._logger.info(f"Mongoose server started on {listen_addr}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to start server: {e}")
            return False

    def stop(self):
        """Stop the Mongoose server."""
        self._logger.info("Stop method called")
        if self._running:
            self._logger.info("Stopping Mongoose server...")
            self._running = False

            # Set signal to trigger event loop exit
            if self._signal_received == 0:
                self._signal_received = signal.SIGTERM  # Simulate SIGTERM

            # Close connections
            self._close_all_connections_from_main_thread()

            # Clean up Mongoose resources
            if self._listener:
                self._listener = NULL

            # Clear userdata reference to prevent memory leaks
            self._mgr.userdata = NULL

            self._logger.info("Mongoose server stopped")

    def _close_all_connections(self):
        """Close all Mongoose connections using the documented approach."""
        self._logger.debug("Closing all Mongoose connections...")

        cdef int closed_count = self._close_connections_nogil()

        self._logger.debug(f"Set closing flag on {closed_count} connections")

    cdef int _close_connections_nogil(self) nogil:
        """Close connections without GIL for better performance."""
        cdef mg_connection *conn = self._mgr.conns
        cdef int closed_count = 0

        while conn != NULL:
            # For shutdown, use immediate closure for faster response
            conn.is_closing = 1
            closed_count += 1
            conn = conn.next

        return closed_count


    def wait_for_shutdown(self):
        """Wait for shutdown signal using pymongoose pattern for reliable signal handling."""
        global _shutdown_requested
        self._logger.info("Starting Mongoose event loop...")

        # Follow pymongoose pattern: check flag in Python, poll in C
        # This ensures signal handling works correctly across the GIL boundary
        while not _shutdown_requested:
            # Poll with GIL released for performance
            self._poll_nogil(100)  # 100ms like pymongoose

        self._logger.info(f"Exiting on signal {self._signal_received}")
        # Close connections gracefully
        self._close_all_connections_from_main_thread()

    cdef void _poll_nogil(self, int timeout_ms) nogil:
        """Poll Mongoose manager without GIL for maximum performance."""
        cyllama_mg_mgr_poll(&self._mgr, timeout_ms)

    def _close_all_connections_from_main_thread(self):
        """Close all Mongoose connections from the main thread."""
        self._logger.info("Closing all Mongoose connections from main thread...")

        cdef int closed_count = self._close_connections_nogil()

        self._logger.info(f"Set closing flag on {closed_count} connections")

    def handle_http_request(self, conn: MongooseConnection, method: str, uri: str,
                          headers: dict, body: str):
        """Handle HTTP request using existing logic."""
        try:
            if method == "GET":
                if uri == "/health":
                    conn.send_json({"status": "ok"})
                elif uri == "/v1/models":
                    self._handle_models(conn)
                else:
                    conn.send_error(404, "Not Found")

            elif method == "POST":
                if uri == "/v1/chat/completions":
                    self._handle_chat_completions(conn, body)
                elif uri == "/v1/embeddings":
                    conn.send_error(501, "Embeddings not yet implemented")
                else:
                    conn.send_error(404, "Not Found")
            else:
                conn.send_error(405, "Method Not Allowed")

        except Exception as e:
            self._logger.error(f"Request handling error: {e}")
            conn.send_error(500, "Internal Server Error")

    def _handle_models(self, conn: MongooseConnection):
        """Handle /v1/models endpoint."""
        models_data = {
            "object": "list",
            "data": [
                {
                    "id": self._config.model_alias,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "cyllama"
                }
            ]
        }
        conn.send_json(models_data)

    def _handle_chat_completions(self, conn: MongooseConnection, body: str):
        """Handle /v1/chat/completions endpoint."""
        try:
            if not body.strip():
                conn.send_error(400, "Empty request body")
                return

            data = json.loads(body)

            # Parse request using existing logic
            messages_data = data.get("messages", [])
            messages = [ChatMessage(role=msg["role"], content=msg["content"])
                       for msg in messages_data]

            request = ChatRequest(
                messages=messages,
                model=data.get("model", self._config.model_alias),
                max_tokens=data.get("max_tokens"),
                temperature=data.get("temperature", 0.8),
                top_p=data.get("top_p", 0.9),
                stream=data.get("stream", False),
                stop=data.get("stop")
            )

            # Process using existing slot logic
            response = self._process_chat_completion(request)

            # Convert to dict for JSON serialization
            response_data = {
                "id": response.id,
                "object": response.object,
                "created": response.created,
                "model": response.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content
                        },
                        "finish_reason": choice.finish_reason
                    }
                    for choice in response.choices
                ],
                "usage": response.usage
            }

            conn.send_json(response_data)

        except json.JSONDecodeError:
            conn.send_error(400, "Invalid JSON")
        except Exception as e:
            self._logger.error(f"Chat completion error: {e}")
            conn.send_error(500, str(e))

    def _process_chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Process chat completion using existing slot logic."""
        # Get available slot
        slot = self.get_available_slot()
        if slot is None:
            raise RuntimeError("No available slots")

        try:
            # Use existing ServerSlot.process_and_generate logic
            import uuid

            task_id = str(uuid.uuid4())
            slot.task_id = task_id
            slot.is_processing = True

            # Convert messages to prompt (reuse existing logic)
            prompt = self._messages_to_prompt(request.messages)

            # Generate response
            max_tokens = request.max_tokens or 100
            generated_text = slot.process_and_generate(prompt, max_tokens)

            # Handle stop words
            if request.stop and generated_text:
                for stop_word in request.stop:
                    if stop_word in generated_text:
                        generated_text = generated_text.split(stop_word)[0]
                        break

            # Estimate token counts
            vocab = self._model.get_vocab()
            prompt_tokens = len(vocab.tokenize(prompt, add_special=True, parse_special=True))
            completion_tokens = len(vocab.tokenize(generated_text, add_special=False, parse_special=False))

            # Create response
            choice = ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=generated_text),
                finish_reason="stop"
            )

            response = ChatResponse(
                id=task_id,
                model=request.model,
                choices=[choice],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            )

            return response

        finally:
            # Reset slot
            slot.reset()

    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert OpenAI messages to a prompt string."""
        prompt_parts = []

        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"System: {message.content}")
            elif message.role == "user":
                prompt_parts.append(f"User: {message.content}")
            elif message.role == "assistant":
                prompt_parts.append(f"Assistant: {message.content}")

        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)


# C callback function for HTTP events
cdef void _http_event_handler(mg_connection *c, int ev, void *ev_data, void *fn_data) noexcept:
    """C callback for Mongoose HTTP events."""
    cdef mg_http_message *hm
    cdef MongooseServer server
    cdef MongooseConnection conn_wrapper

    if c == NULL or c.mgr == NULL:
        return

    # Get server from manager userdata instead of fn_data
    cdef mg_mgr *mgr = <mg_mgr*>c.mgr
    if mgr.userdata == NULL:
        return

    server = <MongooseServer>mgr.userdata

    if ev == MG_EV_HTTP_MSG:
        hm = <mg_http_message*>ev_data
        if hm == NULL:
            return

        try:
            # Create connection wrapper
            conn_wrapper = MongooseConnection()
            conn_wrapper._conn = c

            # Extract request details
            method = hm.method.buf[:hm.method.len].decode('utf-8')
            uri = hm.uri.buf[:hm.uri.len].decode('utf-8')
            body = hm.body.buf[:hm.body.len].decode('utf-8') if hm.body.len > 0 else ""

            # Extract headers (simplified)
            headers = {}

            # Handle request
            server.handle_http_request(conn_wrapper, method, uri, headers, body)

        except Exception as e:
            # Log error and send 500 response
            if server._logger:
                server._logger.error(f"Event handler error: {e}")
            cyllama_mg_http_reply(c, 500, b"Content-Type: text/plain\r\n", "%s", b"Internal Server Error")


# Convenience function
def start_mongoose_server(model_path: str, **kwargs) -> MongooseServer:
    """
    Start a Mongoose-based server with simple configuration.

    Args:
        model_path: Path to the model file
        **kwargs: Additional configuration parameters

    Returns:
        Started MongooseServer instance
    """
    config = ServerConfig(model_path=model_path, **kwargs)
    server = MongooseServer(config)

    if not server.start():
        raise RuntimeError("Failed to start Mongoose server")

    return server