#!/usr/bin/env python3
"""
Python-based Llama.cpp Server

High-level Python wrapper that provides llama.cpp server functionality
using a pure Python HTTP server implementation. Uses the existing cyllama bindings
to provide OpenAI-compatible API endpoints through a lightweight Python HTTP server.
"""

import codecs
import hmac
import ipaddress
import json
import time
import threading
import logging
from typing import TYPE_CHECKING, Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from ...rag.embedder import Embedder
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import uuid
from contextlib import closing

from ...utils.platform import resolve_n_threads

# Import our existing cyllama bindings
from ..llama_cpp import (
    LlamaModel,
    LlamaModelParams,
    LlamaContext,
    LlamaSampler,
    ggml_backend_load_all,
    llama_batch_get_one,
)


@dataclass
class ServerConfig:
    """Configuration for the embedded server."""

    # Model configuration
    model_path: str

    # Server configuration
    host: str = "127.0.0.1"
    port: int = 8080

    # Model parameters
    n_ctx: int = 4096
    n_batch: int = 2048
    n_threads: int = -1  # per slot; -1: physical cores / n_parallel
    n_gpu_layers: int = -1  # -1: all layers

    # Server features
    embedding: bool = False
    n_parallel: int = 1  # Number of parallel slots

    # Embedding configuration (used when embedding=True)
    embedding_model_path: Optional[str] = None  # defaults to model_path if None
    embedding_n_ctx: int = 512
    embedding_n_batch: int = 512
    embedding_n_gpu_layers: int = -1
    embedding_pooling: str = "mean"
    embedding_normalize: bool = True

    # OpenAI compatibility
    model_alias: str = "gpt-3.5-turbo"

    # Security
    api_key: Optional[str] = None  # if set, require "Authorization: Bearer <key>" except on /health
    max_body_bytes: int = 2 * 1024 * 1024


def model_params(config: ServerConfig) -> LlamaModelParams:
    """Model load parameters for `config`."""
    params = LlamaModelParams()
    params.n_gpu_layers = config.n_gpu_layers
    return params


def check_request(
    config: ServerConfig, path: str, authorization: Optional[str], content_length: int
) -> Optional[Tuple[int, str]]:
    """Return (status, message) if a request must be rejected before its body is handled, else None.

    `authorization` is the header as both servers decode it, latin-1, so
    re-encoding it recovers the bytes the client sent.
    """
    if config.api_key and path != "/health":
        scheme, _, token = (authorization or "").partition(" ")
        sent = token.encode("latin-1", errors="replace")
        # compare_digest runs even on a wrong scheme; the scheme is case-insensitive (RFC 7235).
        if not (hmac.compare_digest(sent, config.api_key.encode("utf-8")) and scheme.lower() == "bearer"):
            return 401, "Invalid or missing API key"
    if content_length > config.max_body_bytes:
        return 413, f"Request body exceeds {config.max_body_bytes} bytes"
    return None


def exposed_without_auth(config: ServerConfig) -> bool:
    """True if the server binds a non-loopback address with no API key."""
    if config.api_key or config.host == "localhost":
        return False
    try:
        return not ipaddress.ip_address(config.host.strip("[]")).is_loopback
    except ValueError:
        return True  # a hostname may resolve to any interface


def stop_at(pieces: Iterable[str], stop: Union[str, List[str], None]) -> Generator[str, None, None]:
    """Yield `pieces` up to the earliest stop string, holding back text that may begin one."""
    stops = [s for s in ([stop] if isinstance(stop, str) else stop or []) if s]
    if not stops:
        yield from pieces
        return
    buf = ""
    for piece in pieces:
        buf += piece
        hits = [i for i in (buf.find(s) for s in stops) if i >= 0]
        if hits:
            if min(hits):
                yield buf[: min(hits)]
            return
        # Longest suffix of buf that is a proper prefix of some stop string.
        hold = max((k for s in stops for k in range(1, min(len(s), len(buf) + 1)) if buf.endswith(s[:k])), default=0)
        if len(buf) > hold:
            yield buf[: len(buf) - hold]
            buf = buf[len(buf) - hold :]
    if buf:
        yield buf


def sse_chunks(model: str, pieces: Generator[str, None, None]) -> Generator[bytes, None, None]:
    """Encode `pieces` as OpenAI chat.completion.chunk server-sent events, then [DONE]. Closes `pieces`."""
    chunk_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    def event(delta: Dict[str, str], finish_reason: Optional[str] = None) -> bytes:
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n"

    with closing(pieces):
        yield event({"role": "assistant"})
        for piece in pieces:
            yield event({"content": piece})
    yield event({}, "stop")
    yield b"data: [DONE]\n\n"


@dataclass
class ChatMessage:
    """OpenAI-compatible chat message."""

    role: str
    content: str


@dataclass
class ChatRequest:
    """OpenAI-compatible chat completion request."""

    messages: List[ChatMessage]
    model: str = "gpt-3.5-turbo"
    max_tokens: Optional[int] = None
    temperature: float = 0.8
    top_p: float = 0.9
    min_p: float = 0.05
    stream: bool = False
    stop: Optional[List[str]] = None
    seed: Optional[int] = None


@dataclass
class ChatChoice:
    """OpenAI-compatible chat choice."""

    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


@dataclass
class ChatResponse:
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = "gpt-3.5-turbo"
    choices: List[ChatChoice] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)


class ServerSlot:
    """Represents a processing slot for handling requests."""

    def __init__(self, slot_id: int, model: LlamaModel, config: ServerConfig):
        self.id = slot_id
        self.model = model
        self.config = config

        # Create context and sampler for this slot
        from ..llama_cpp import LlamaContextParams

        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = config.n_ctx
        ctx_params.n_batch = config.n_batch
        # Slots decode concurrently; more threads in total than cores makes every slot slower.
        n_threads = resolve_n_threads(config.n_threads, config.n_parallel)
        ctx_params.n_threads = n_threads
        ctx_params.n_threads_batch = n_threads
        self.context = LlamaContext(model, ctx_params, verbose=False)

        # Sampler is rebuilt per-request to support per-request parameters
        self.sampler: Optional[LlamaSampler] = None
        self._build_sampler()

        # Slot state
        self.is_processing = False
        self.task_id: Optional[str] = None

        # Processing state
        self.generated_tokens: List[int] = []
        self.response_text = ""

    def _build_sampler(
        self,
        temperature: float = 0.8,
        min_p: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        """Build (or rebuild) the sampler chain with the given parameters."""
        from ..llama_cpp import LlamaSamplerChainParams

        sampler_params = LlamaSamplerChainParams()
        self.sampler = LlamaSampler(sampler_params)
        self.sampler.add_min_p(min_p, 1)
        self.sampler.add_temp(temperature)
        self.sampler.add_dist(seed if seed is not None else 1337)

    def reset(self) -> None:
        """Reset the slot for a new request."""
        self.is_processing = False
        self.task_id = None
        self.generated_tokens.clear()
        self.response_text = ""
        # Reset the context state. The next prompt decodes from position 0, which
        # llama_decode rejects while the previous request's KV entries remain.
        self.context.n_tokens = 0
        self.context.kv_cache_clear()

    def process_and_generate(self, prompt: str, max_tokens: int = 100, request: Optional["ChatRequest"] = None) -> str:
        """Process prompt and generate response using the slot's context."""
        try:
            response_text = "".join(self.stream(prompt, max_tokens, request))
        except Exception as e:
            logging.error(f"Error in process_and_generate: {e}")
            return ""
        self.response_text = response_text
        return response_text

    def stream(
        self, prompt: str, max_tokens: int = 100, request: Optional["ChatRequest"] = None
    ) -> Generator[str, None, None]:
        """Yield generated text as it is produced. Yields nothing for an empty or over-long prompt."""
        # Rebuild sampler with per-request parameters when provided
        if request is not None:
            self._build_sampler(
                temperature=request.temperature,
                min_p=request.min_p,
                seed=request.seed,
            )

        context = self.context
        vocab = self.model.get_vocab()
        prompt_tokens = vocab.tokenize(prompt, add_special=True, parse_special=True)
        if not prompt_tokens or len(prompt_tokens) >= self.config.n_ctx:
            return

        batch = llama_batch_get_one(prompt_tokens, 0)  # Start from position 0
        n_past = len(prompt_tokens)
        ret = context.decode(batch)
        if ret != 0:
            logging.warning(f"Initial decode returned {ret}")
            return

        # A multi-byte character can span tokens.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        for _ in range(max_tokens):
            if n_past >= context.n_ctx - 1:
                break

            assert self.sampler is not None
            new_token_id = self.sampler.sample(context, -1)
            if vocab.is_eog(new_token_id):
                break

            piece = decoder.decode(vocab.token_to_bytes(new_token_id, 0, True))
            if piece:
                yield piece

            batch = llama_batch_get_one([new_token_id], n_past)
            n_past += 1
            ret = context.decode(batch)
            if ret != 0:
                logging.warning(f"Token decode returned {ret}")
                break

        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail

    def get_generated_text(self) -> str:
        """Get the generated text."""
        return self.response_text


class PythonServer:
    """
    Python-based Llama.cpp server using existing cyllama bindings.

    Provides OpenAI-compatible endpoints without requiring external binaries.
    Uses the existing libllama.a linkage through cyllama Python bindings.
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.model: Optional[LlamaModel] = None
        self.slots: List[ServerSlot] = []
        self.embedder: Optional["Embedder"] = None  # Initialized when config.embedding is True

        # HTTP server
        self.httpd: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.running = False

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def load_model(self) -> bool:
        """Load the model and initialize slots."""
        try:
            self.logger.info(f"Loading model: {self.config.model_path}")

            # Load backends
            ggml_backend_load_all()

            # Load model
            self.model = LlamaModel(path_model=self.config.model_path, params=model_params(self.config))

            # Create slots
            self.slots = []
            for i in range(self.config.n_parallel):
                slot = ServerSlot(i, self.model, self.config)
                self.slots.append(slot)

            self.logger.info(f"Model loaded successfully with {len(self.slots)} slots")

            # Initialize embedder if embedding mode is enabled
            if self.config.embedding:
                from ...rag.embedder import Embedder

                emb_model = self.config.embedding_model_path or self.config.model_path
                self.embedder = Embedder(
                    model_path=emb_model,
                    n_ctx=self.config.embedding_n_ctx,
                    n_batch=self.config.embedding_n_batch,
                    n_gpu_layers=self.config.embedding_n_gpu_layers,
                    pooling=self.config.embedding_pooling,
                    normalize=self.config.embedding_normalize,
                )
                emb = self.embedder
                assert emb is not None
                self.logger.info(f"Embedder loaded: dim={emb.dimension}, pooling={emb.pooling}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

    def get_available_slot(self) -> Optional[ServerSlot]:
        """Get an available slot for processing."""
        for slot in self.slots:
            if not slot.is_processing:
                return slot
        return None

    def process_chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Process a chat completion request."""
        # Get available slot
        slot = self.get_available_slot()
        if slot is None:
            raise RuntimeError("No available slots")

        try:
            # Generate task ID
            task_id = str(uuid.uuid4())
            slot.task_id = task_id
            slot.is_processing = True

            # Convert messages to prompt
            prompt = self._messages_to_prompt(request.messages)

            # Use the new simplified approach
            max_tokens = request.max_tokens or 100
            generated_text = slot.process_and_generate(prompt, max_tokens, request=request)

            generated_text = "".join(stop_at([generated_text], request.stop))

            # Estimate token counts (simplified)
            assert self.model is not None
            vocab = self.model.get_vocab()
            prompt_tokens = len(vocab.tokenize(prompt, add_special=True, parse_special=True))
            completion_tokens = len(vocab.tokenize(generated_text, add_special=False, parse_special=False))

            # Create response
            choice = ChatChoice(
                index=0, message=ChatMessage(role="assistant", content=generated_text), finish_reason="stop"
            )

            response = ChatResponse(
                id=task_id,
                model=request.model,
                choices=[choice],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )

            return response

        finally:
            # Reset slot
            slot.reset()

    def stream_chat_completion(self, request: ChatRequest) -> Generator[str, None, None]:
        """Yield generated text for `request`. Closing the generator frees the slot."""
        slot = self.get_available_slot()
        if slot is None:
            raise RuntimeError("No available slots")
        try:
            slot.task_id = str(uuid.uuid4())
            slot.is_processing = True
            prompt = self._messages_to_prompt(request.messages)
            with closing(slot.stream(prompt, request.max_tokens or 100, request)) as pieces:
                yield from stop_at(pieces, request.stop)
        finally:
            slot.reset()

    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert OpenAI messages to a prompt string."""
        # Simple conversion - can be enhanced with chat templates
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

    def start(self) -> bool:
        """Start the embedded server."""
        if not self.load_model():
            return False

        if exposed_without_auth(self.config):
            self.logger.warning(f"Binding {self.config.host} without an API key: any host that can reach it can use it")

        try:
            # Create HTTP server
            handler = self._create_request_handler()
            self.httpd = HTTPServer((self.config.host, self.config.port), handler)

            # Start server in background thread
            self.running = True
            self.server_thread = threading.Thread(target=self._server_loop)
            self.server_thread.daemon = True
            self.server_thread.start()

            self.logger.info(f"Server started on http://{self.config.host}:{self.config.port}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            return False

    def stop(self) -> None:
        """Stop the embedded server."""
        if self.running:
            self.running = False
            if self.httpd:
                self.httpd.shutdown()
                self.httpd.server_close()
            if self.server_thread:
                self.server_thread.join(timeout=5.0)
            self.logger.info("Server stopped")

    def _server_loop(self) -> None:
        """Main server loop."""
        try:
            assert self.httpd is not None
            self.httpd.serve_forever()
        except Exception as e:
            if self.running:  # Only log if we didn't shutdown intentionally
                self.logger.error(f"Server loop error: {e}")

    def _create_request_handler(self) -> type:
        """Create the HTTP request handler class."""
        server_instance = self

        class RequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                # Suppress default logging
                pass

            def do_GET(self) -> None:
                """Handle GET requests."""
                path = urlparse(self.path).path
                if self._reject(path, 0):
                    return

                if path == "/health":
                    self._send_json_response({"status": "ok"})
                elif path == "/v1/models":
                    self._handle_models()
                else:
                    self._send_error(404, "Not Found")

            def do_POST(self) -> None:
                """Handle POST requests."""
                path = urlparse(self.path).path

                try:
                    content_length = int(self.headers.get("Content-Length", 0))
                except ValueError:
                    self.close_connection = True
                    self._send_error(400, "Invalid Content-Length")
                    return
                if self._reject(path, content_length):
                    return

                try:
                    if content_length > 0:
                        body = self.rfile.read(content_length).decode("utf-8")
                        data = json.loads(body)
                    else:
                        data = {}

                    if path == "/v1/chat/completions":
                        self._handle_chat_completions(data)
                    elif path == "/v1/embeddings":
                        self._handle_embeddings(data)
                    else:
                        self._send_error(404, "Not Found")

                except (json.JSONDecodeError, UnicodeDecodeError):  # JSON must be UTF-8 (RFC 8259)
                    self._send_error(400, "Invalid JSON")
                except Exception as e:
                    server_instance.logger.error(f"Request error: {e}")
                    self._send_error(500, "Internal Server Error")

            def _reject(self, path: str, content_length: int) -> bool:
                """Send an error and return True if the request fails auth or size checks."""
                rejection = check_request(
                    server_instance.config, path, self.headers.get("Authorization"), content_length
                )
                if rejection is None:
                    return False
                self.close_connection = True  # the unread body must not be parsed as the next request
                self._send_error(*rejection)
                return True

            def _send_json_response(self, data: Dict[str, Any], status: int = 200) -> None:
                """Send a JSON response."""
                response = json.dumps(data)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response.encode("utf-8"))

            def _send_error(self, status: int, message: str) -> None:
                """Send an error response."""
                error_data = {"error": {"type": "invalid_request_error", "message": message}}
                self._send_json_response(error_data, status)

            def _handle_models(self) -> None:
                """Handle /v1/models endpoint."""
                models_data = {
                    "object": "list",
                    "data": [
                        {
                            "id": server_instance.config.model_alias,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "cyllama",
                        }
                    ],
                }
                self._send_json_response(models_data)

            def _handle_chat_completions(self, data: Dict[str, Any]) -> None:
                """Handle /v1/chat/completions endpoint."""
                try:
                    # Parse request
                    messages_data = data.get("messages", [])
                    messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in messages_data]

                    request = ChatRequest(
                        messages=messages,
                        model=data.get("model", server_instance.config.model_alias),
                        max_tokens=data.get("max_tokens"),
                        temperature=data.get("temperature", 0.8),
                        top_p=data.get("top_p", 0.9),
                        min_p=data.get("min_p", 0.05),
                        stream=data.get("stream", False),
                        stop=data.get("stop"),
                        seed=data.get("seed"),
                    )

                    if request.stream:
                        self._stream(server_instance.stream_chat_completion(request), request.model)
                        return

                    # Process request
                    response = server_instance.process_chat_completion(request)

                    # Convert to dict for JSON serialization
                    response_data = {
                        "id": response.id,
                        "object": response.object,
                        "created": response.created,
                        "model": response.model,
                        "choices": [
                            {
                                "index": choice.index,
                                "message": {"role": choice.message.role, "content": choice.message.content},
                                "finish_reason": choice.finish_reason,
                            }
                            for choice in response.choices
                        ],
                        "usage": response.usage,
                    }

                    self._send_json_response(response_data)

                except Exception as e:
                    # The message can carry model and filesystem paths; log it
                    # server-side and tell the client nothing specific.
                    server_instance.logger.error(f"Chat completion error: {e}")
                    self._send_error(500, "Internal Server Error")

            def _stream(self, pieces: Generator[str, None, None], model: str) -> None:
                """Send `pieces` as server-sent events; the connection closes at the end."""
                self.close_connection = True  # the body has no length, so its end is the close
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with closing(sse_chunks(model, pieces)) as events:
                    try:
                        for event in events:
                            self.wfile.write(event)
                            self.wfile.flush()
                    except OSError:
                        pass  # client disconnected; closing the events stops generation
                    except Exception as e:
                        # Headers are sent; the truncated stream is the only error signal left.
                        server_instance.logger.error(f"Stream error: {e}")

            def _handle_embeddings(self, data: Dict[str, Any]) -> None:
                """Handle /v1/embeddings endpoint."""
                if not server_instance.config.embedding or server_instance.embedder is None:
                    self._send_error(400, "Embeddings not enabled")
                    return

                try:
                    input_data = data.get("input")
                    if input_data is None:
                        self._send_error(400, "Missing 'input' field")
                        return

                    # Normalize input to list of strings
                    if isinstance(input_data, str):
                        texts = [input_data]
                    elif isinstance(input_data, list):
                        texts = [str(t) for t in input_data]
                    else:
                        self._send_error(400, "Invalid 'input' field: must be string or list of strings")
                        return

                    embedder = server_instance.embedder
                    model_name = data.get("model", server_instance.config.model_alias)

                    # Generate embeddings
                    results = []
                    total_tokens = 0
                    for i, text in enumerate(texts):
                        result = embedder.embed_with_info(text)
                        results.append(
                            {
                                "object": "embedding",
                                "embedding": result.embedding,
                                "index": i,
                            }
                        )
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

                    self._send_json_response(response_data)

                except Exception as e:
                    server_instance.logger.error(f"Embeddings error: {e}")
                    self._send_error(500, "Internal Server Error")

        return RequestHandler

    def __enter__(self) -> "PythonServer":
        """Context manager entry."""
        if self.start():
            return self
        else:
            raise RuntimeError("Failed to start server")

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Context manager exit."""
        self.stop()


# Convenience function
def start_python_server(model_path: str, **kwargs: Any) -> PythonServer:
    """
    Start a Python server with simple configuration.

    Args:
        model_path: Path to the model file
        **kwargs: Additional configuration parameters

    Returns:
        Started PythonServer instance
    """
    config = ServerConfig(model_path=model_path, **kwargs)
    server = PythonServer(config)

    if not server.start():
        raise RuntimeError("Failed to start Python server")

    return server


# if __name__ == "__main__":
#     # Example usage
#     import argparse

#     parser = argparse.ArgumentParser(description="Embedded Llama.cpp Server")
#     parser.add_argument("-m", "--model", required=True, help="Path to model file")
#     parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
#     parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
#     parser.add_argument("--ctx-size", type=int, default=2048, help="Context size")
#     parser.add_argument("--gpu-layers", type=int, default=-1, help="GPU layers")

#     args = parser.parse_args()

#     logging.basicConfig(level=logging.INFO)

#     config = ServerConfig(
#         model_path=args.model,
#         host=args.host,
#         port=args.port,
#         n_ctx=args.ctx_size,
#         n_gpu_layers=args.gpu_layers
#     )

#     with PythonServer(config) as server:
#         print(f"Server running at http://{args.host}:{args.port}")
#         print("Press Ctrl+C to stop...")

#         try:
#             while True:
#                 time.sleep(1)
#         except KeyboardInterrupt:
#             print("\nShutting down server...")
