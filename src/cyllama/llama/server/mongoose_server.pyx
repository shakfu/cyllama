# Mongoose-based HTTP server for cyllama
# High-performance alternative to Python HTTP server

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

# Import Mongoose C API
from .mongoose cimport *

# Import existing server logic
from .embedded import ServerConfig, ServerSlot, ChatMessage, ChatRequest, ChatResponse, ChatChoice


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

        cyllama_mg_http_reply(self._conn, status_code, headers_bytes, "%s", json_bytes)
        return True

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

    def __cinit__(self):
        cyllama_mg_mgr_init(&self._mgr)
        self._listener = NULL
        self._running = False
        self._server_thread = None

    def __dealloc__(self):
        self.stop()
        cyllama_mg_mgr_free(&self._mgr)

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

    def start(self) -> bool:
        """Start the Mongoose server."""
        if not self.load_model():
            return False

        try:
            # Create listener address
            listen_addr = f"http://{self._config.host}:{self._config.port}"
            addr_bytes = listen_addr.encode('utf-8')

            # Start HTTP listener with our event handler
            # Store reference to self to prevent garbage collection
            self._mgr.userdata = <void*>self
            self._listener = cyllama_mg_http_listen(&self._mgr, addr_bytes,
                                                  <mg_event_handler_t>_http_event_handler,
                                                  NULL)  # Use NULL, get server from mgr.userdata

            if self._listener == NULL:
                self._logger.error("Failed to create HTTP listener")
                return False

            self._running = True

            # Start server loop in background thread
            self._server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self._server_thread.start()

            self._logger.info(f"Mongoose server started on {listen_addr}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to start server: {e}")
            return False

    def stop(self):
        """Stop the Mongoose server."""
        if self._running:
            self._running = False
            if self._server_thread:
                self._server_thread.join(timeout=5.0)
            self._logger.info("Mongoose server stopped")

    def _server_loop(self):
        """Main server event loop."""
        try:
            while self._running:
                cyllama_mg_mgr_poll(&self._mgr, 100)  # 100ms timeout
        except Exception as e:
            if self._running:  # Only log if we didn't shutdown intentionally
                self._logger.error(f"Server loop error: {e}")

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