# cpp-httplib-based HTTP server for cyllama, with SSE streaming for chat completions.

import json
import logging
import queue
import signal
import threading
import time
import uuid
from contextlib import closing
from typing import List

from cpython.exc cimport PyErr_CheckSignals
from cpython.ref cimport PyObject, Py_INCREF, Py_XDECREF

from .http_shim cimport *

from .python import (ServerConfig, ServerSlot, ChatMessage, ChatRequest, ChatResponse, ChatChoice,
                     check_request, exposed_without_auth, model_params, sse_chunks, stop_at)

# Global shutdown flag for signal handling
_shutdown_requested = False


def _bind_address(host: str):
    """Return (host, is_ipv6) for the listener: brackets stripped, family taken from the literal.

    A fixed family keeps "localhost" on 127.0.0.1, as PythonServer binds it.
    """
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host, ":" in host


cdef class HttpResponse:
    """The response to one request. Valid only while its handler runs."""
    cdef cy_http_res *_res
    cdef readonly bint sent

    def __cinit__(self):
        self._res = NULL
        self.sent = False

    def _set(self, int status, bytes content_type, bytes body, bint close_connection=False):
        if self._res == NULL or self.sent:
            return False
        cy_http_res_set(self._res, status, content_type, body, len(body), close_connection)
        self.sent = True
        return True

    def send_json(self, data: dict, status_code: int = 200, close_connection: bool = False):
        """Send JSON response."""
        return self._set(status_code, b"application/json", json.dumps(data).encode("utf-8"), close_connection)

    def send_error(self, status_code: int, message: str, close_connection: bool = False):
        """Send error response."""
        error_data = {"error": {"type": "invalid_request_error", "message": message}}
        return self.send_json(error_data, status_code, close_connection)

    def send_stream(self, chunks, content_type: bytes = b"text/event-stream"):
        """Stream the bytes `chunks` yields, chunked. The server closes `chunks` when the response ends."""
        if self._res == NULL or self.sent:
            return False
        Py_INCREF(chunks)  # released by _stream_release
        cy_http_res_stream(self._res, 200, content_type, <void*>chunks)
        self.sent = True
        return True


cdef class EmbeddedServer:
    """Embedded HTTP server for LLM inference using cpp-httplib."""

    cdef cy_http_server *_srv
    cdef object _config
    cdef object _model
    cdef object _embedder
    cdef object _decision
    cdef list _slots
    cdef object _free_slots  # queue.Queue of idle ServerSlots
    cdef object _logger
    cdef bint _running
    cdef object _listen_thread
    cdef int _signal_received
    cdef bint _stop_requested
    cdef object _loop_exited  # threading.Event, clear while wait_for_shutdown runs
    cdef object _loop_ident

    @property
    def signal_received(self):
        """Get the received signal number (0 if no signal)."""
        return self._signal_received

    @signal_received.setter
    def signal_received(self, value):
        """Set the signal received value."""
        self._signal_received = value

    def __cinit__(self):
        self._srv = NULL
        self._running = False
        self._listen_thread = None
        self._signal_received = 0
        self._stop_requested = False
        self._loop_exited = threading.Event()
        self._loop_exited.set()
        self._loop_ident = 0

    def __dealloc__(self):
        # The listener thread holds a reference, so it has exited by now.
        if self._srv != NULL:
            cy_http_server_free(self._srv)
            self._srv = NULL

    def __enter__(self):
        """Context manager entry."""
        if self.start():
            return self
        else:
            raise RuntimeError("Failed to start embedded server")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    def __init__(self, config: ServerConfig):
        self._config = config
        self._model = None
        self._embedder = None
        self._decision = None
        self._slots = []
        self._logger = logging.getLogger(__name__)

    def load_model(self) -> bool:
        """Load the model and initialize slots."""
        try:
            self._logger.info(f"Loading model: {self._config.model_path}")

            # Import here to avoid circular imports
            from ..llama_cpp import LlamaModel

            self._model = LlamaModel(path_model=self._config.model_path, params=model_params(self._config))

            from ..decision import load_decision_model

            self._decision = load_decision_model(self._model, self._config.n_ctx, self._logger)

            self._slots = []
            for i in range(self._config.n_parallel):
                slot = ServerSlot(i, self._model, self._config)
                self._slots.append(slot)

            self._logger.info(f"Model loaded successfully with {len(self._slots)} slots")

            # Initialize embedder if embedding mode is enabled
            if self._config.embedding:
                from ...rag.embedder import Embedder

                emb_model = self._config.embedding_model_path or self._config.model_path
                self._embedder = Embedder(
                    model_path=emb_model,
                    n_ctx=self._config.embedding_n_ctx,
                    n_batch=self._config.embedding_n_batch,
                    n_gpu_layers=self._config.embedding_n_gpu_layers,
                    pooling=self._config.embedding_pooling,
                    normalize=self._config.embedding_normalize,
                )
                self._logger.info(
                    f"Embedder loaded: dim={self._embedder.dimension}, "
                    f"pooling={self._embedder.pooling}"
                )

            return True

        except Exception as e:
            self._logger.error(f"Failed to load model: {e}")
            return False

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
        """Load the model, bind, and serve on a background thread."""
        global _shutdown_requested

        if self._running:
            return True
        _shutdown_requested = False
        self._stop_requested = False

        if not self.load_model():
            return False

        if exposed_without_auth(self._config):
            self._logger.warning(f"Binding {self._config.host} without an API key: any host that can reach it can use it")

        self._setup_signal_handlers()

        host, ipv6 = _bind_address(self._config.host)
        if self._srv != NULL:
            cy_http_server_free(self._srv)
        self._srv = cy_http_server_new(<void*>self, _precheck, _handle, _stream_next, _stream_release,
                                       self._config.max_body_bytes)
        if cy_http_server_bind(self._srv, host.encode("utf-8"), self._config.port, ipv6) != 0:
            self._logger.error(f"Failed to bind {self._config.host}:{self._config.port}")
            cy_http_server_free(self._srv)
            self._srv = NULL
            return False

        self._free_slots = queue.Queue()
        for slot in self._slots:
            self._free_slots.put(slot)

        self._running = True
        self._listen_thread = threading.Thread(target=self._listen, name="cyllama-http", daemon=True)
        self._listen_thread.start()
        self._logger.info(f"Embedded server started on {self._config.host}:{self._config.port}")
        return True

    def _listen(self):
        cdef cy_http_server *srv = self._srv
        with nogil:
            cy_http_server_listen(srv)

    def stop(self):
        """Stop the embedded server. Waits for in-flight requests; open streams end at their next chunk."""
        # A wait_for_shutdown running on another thread must return before stop() does.
        if not self._loop_exited.is_set() and threading.get_ident() != self._loop_ident:
            self._stop_requested = True
            if not self._loop_exited.wait(timeout=5):
                self._logger.error("wait_for_shutdown did not exit within 5s")
        if not self._running:
            return
        if self._signal_received == 0:
            self._signal_received = signal.SIGTERM
        cdef cy_http_server *srv = self._srv
        with nogil:
            cy_http_server_stop(srv)
        # If the join is interrupted, state is unchanged and stop() can be called again.
        self._listen_thread.join()
        self._running = False
        self._listen_thread = None
        cy_http_server_free(self._srv)
        self._srv = NULL
        self._logger.info("Embedded server stopped")

    def wait_for_shutdown(self):
        """Block until a signal arrives or stop() is called."""
        self._loop_ident = threading.get_ident()
        self._loop_exited.clear()
        try:
            while not _shutdown_requested and not self._stop_requested:
                time.sleep(0.1)
                # Compiled code never runs Python signal handlers on its own, and a signal
                # delivered to another thread does not interrupt the sleep. Raises if a handler raises.
                PyErr_CheckSignals()
        finally:
            self._logger.info(f"Exiting on signal {self._signal_received}")
            self._loop_exited.set()

    def handle_http_request(self, conn: HttpResponse, method: str, uri: str, body: bytes):
        """Route a request that passed check_request."""
        try:
            if method == "GET":
                if uri == "/health":
                    conn.send_json({"status": "ok"})
                elif uri == "/v1/models":
                    self._handle_models(conn)
                else:
                    conn.send_error(404, "Not Found")

            elif method == "POST":
                if uri not in ("/v1/chat/completions", "/v1/embeddings", "/v1/systemone"):
                    conn.send_error(404, "Not Found")
                    return
                try:
                    text = body.decode("utf-8")
                except UnicodeDecodeError:
                    conn.send_error(400, "Invalid JSON")
                    return
                if uri == "/v1/chat/completions":
                    self._handle_chat_completions(conn, text)
                elif uri == "/v1/systemone":
                    self._handle_systemone(conn, text)
                else:
                    self._handle_embeddings(conn, text)
            else:
                conn.send_error(405, "Method Not Allowed")

        except Exception as e:
            self._logger.error(f"Request handling error: {e}")
            conn.send_error(500, "Internal Server Error")

    def _handle_models(self, conn: HttpResponse):
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
        if self._decision is not None:
            # clients detect decision models this way, as with llama-server
            models_data["data"][0]["architecture"] = {"output_modalities": ["decisions"]}
        conn.send_json(models_data)

    def _handle_chat_completions(self, conn: HttpResponse, body: str):
        """Handle /v1/chat/completions endpoint."""
        try:
            if not body.strip():
                conn.send_error(400, "Empty request body")
                return

            data = json.loads(body)

            messages = [ChatMessage(role=msg["role"], content=msg["content"])
                        for msg in data.get("messages", [])]

            request = ChatRequest(
                messages=messages,
                model=data.get("model", self._config.model_alias),
                max_tokens=data.get("max_tokens"),
                temperature=data.get("temperature", 0.8),
                top_p=data.get("top_p", 0.9),
                min_p=data.get("min_p", 0.05),
                stream=data.get("stream", False),
                stop=data.get("stop"),
                seed=data.get("seed"),
            )

            if request.stream:
                prompt = self._messages_to_prompt(request.messages)
                conn.send_stream(sse_chunks(request.model, self._generate(prompt, request)))
                return

            response = self._process_chat_completion(request)
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
            # The message can carry model and filesystem paths; log it
            # server-side and tell the client nothing specific.
            self._logger.error(f"Chat completion error: {e}")
            conn.send_error(500, "Internal Server Error")

    def _handle_systemone(self, conn: HttpResponse, body: str):
        """Handle /v1/systemone (decision models)."""
        from ..decision import systemone_response

        try:
            data = json.loads(body) if body.strip() else None
        except json.JSONDecodeError:
            conn.send_error(400, "Invalid JSON")
            return
        status, payload = systemone_response(self._decision, data, self._config.model_alias)
        conn.send_json(payload, status)

    def _handle_embeddings(self, conn: HttpResponse, body: str):
        """Handle /v1/embeddings endpoint."""
        if not self._config.embedding or self._embedder is None:
            conn.send_error(400, "Embeddings not enabled")
            return

        try:
            if not body.strip():
                conn.send_error(400, "Empty request body")
                return

            data = json.loads(body)
            input_data = data.get("input")
            if input_data is None:
                conn.send_error(400, "Missing 'input' field")
                return

            # Normalize input to list of strings
            if isinstance(input_data, str):
                texts = [input_data]
            elif isinstance(input_data, list):
                texts = [str(t) for t in input_data]
            else:
                conn.send_error(400, "Invalid 'input' field: must be string or list of strings")
                return

            model_name = data.get("model", self._config.model_alias)

            results = []
            total_tokens = 0
            for i, text in enumerate(texts):
                result = self._embedder.embed_with_info(text)
                results.append({
                    "object": "embedding",
                    "embedding": result.embedding,
                    "index": i,
                })
                total_tokens += result.token_count

            response_data = {
                "object": "list",
                "data": results,
                "model": model_name,
                "usage": {
                    "prompt_tokens": total_tokens,
                    "total_tokens": total_tokens,
                },
            }

            conn.send_json(response_data)

        except json.JSONDecodeError:
            conn.send_error(400, "Invalid JSON")
        except Exception as e:
            # The message can carry model and filesystem paths; log it
            # server-side and tell the client nothing specific.
            self._logger.error(f"Embeddings error: {e}")
            conn.send_error(500, "Internal Server Error")

    def _generate(self, prompt: str, request: ChatRequest):
        """Yield generated text on a free slot, waiting for one. Closing it frees the slot."""
        slot = self._free_slots.get()
        try:
            slot.task_id = str(uuid.uuid4())
            slot.is_processing = True
            with closing(slot.stream(prompt, request.max_tokens or 100, request)) as pieces:
                yield from stop_at(pieces, request.stop)
        finally:
            slot.reset()
            self._free_slots.put(slot)

    def _process_chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate a complete, non-streamed chat response."""
        prompt = self._messages_to_prompt(request.messages)
        with closing(self._generate(prompt, request)) as pieces:
            generated_text = "".join(pieces)

        vocab = self._model.get_vocab()
        prompt_tokens = len(vocab.tokenize(prompt, add_special=True, parse_special=True))
        completion_tokens = len(vocab.tokenize(generated_text, add_special=False, parse_special=False))

        return ChatResponse(
            id=str(uuid.uuid4()),
            model=request.model,
            choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=generated_text),
                                finish_reason="stop")],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )

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


# httplib callbacks. They run on httplib worker threads and take the GIL.

cdef str _text(const char *p, size_t n):
    return p[:n].decode("utf-8", "replace") if n else ""


cdef int _precheck(void *userdata, const cy_http_req *req, cy_http_res *res) noexcept with gil:
    """Apply check_request before httplib reads the body."""
    cdef EmbeddedServer server = <EmbeddedServer>userdata
    cdef HttpResponse conn = HttpResponse()
    conn._res = res
    try:
        auth = req.auth[:req.auth_len].decode("latin-1") if req.auth != NULL else None
        rejection = check_request(server._config, _text(req.path, req.path_len), auth, req.content_length)
        if rejection is None:
            return 0
        # Close rather than let httplib drain an unread, possibly oversized, body.
        conn.send_error(*rejection, close_connection=True)
        return 1
    except BaseException as e:
        server._logger.error(f"Request check error: {e}")
        conn.send_error(500, "Internal Server Error", close_connection=True)
        return 1
    finally:
        conn._res = NULL


cdef int _handle(void *userdata, const cy_http_req *req, cy_http_res *res) noexcept with gil:
    cdef EmbeddedServer server = <EmbeddedServer>userdata
    cdef HttpResponse conn = HttpResponse()
    conn._res = res
    try:
        server.handle_http_request(conn, _text(req.method, req.method_len), _text(req.path, req.path_len),
                                   req.body[:req.body_len] if req.body_len else b"")
    except BaseException as e:
        server._logger.error(f"Event handler error: {e}")
        conn.send_error(500, "Internal Server Error")
    finally:
        conn._res = NULL
    return 1


cdef int _stream_next(void *stream, cy_http_sink *sink) noexcept with gil:
    cdef bytes chunk
    cdef const char *data
    cdef size_t n
    cdef int ok
    try:
        chunk = next(<object>stream)
    except StopIteration:
        return 0
    except BaseException as e:
        # The status line is sent; aborting truncates the chunked body so the client sees the failure.
        logging.getLogger(__name__).error(f"Stream error: {e}")
        return -1
    data = chunk
    n = len(chunk)
    with nogil:
        ok = cy_http_sink_write(sink, data, n)
    return 1 if ok else -1


cdef void _stream_release(void *stream) noexcept with gil:
    try:
        (<object>stream).close()  # runs the generator's finally blocks, freeing its slot
    except BaseException as e:
        logging.getLogger(__name__).error(f"Stream close error: {e}")
    Py_XDECREF(<PyObject*>stream)


# Convenience function
def start_embedded_server(model_path: str, **kwargs) -> EmbeddedServer:
    """
    Start an embedded server with simple configuration.

    Args:
        model_path: Path to the model file
        **kwargs: Additional configuration parameters

    Returns:
        Started EmbeddedServer instance
    """
    config = ServerConfig(model_path=model_path, **kwargs)
    server = EmbeddedServer(config)

    if not server.start():
        raise RuntimeError("Failed to start embedded server")

    return server
