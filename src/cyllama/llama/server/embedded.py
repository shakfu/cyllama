#!/usr/bin/env python3
"""
Embedded Llama.cpp Server

High-level Python wrapper that embeds llama.cpp server functionality
without requiring external binaries. Uses the existing cyllama bindings
to provide OpenAI-compatible API endpoints through a lightweight Python HTTP server.

This approach avoids the complexity of wrapping cpp-httplib and complex C++ templates
while still providing the full server functionality using the existing libllama.a linkage.
"""

import json
import time
import threading
import logging
from typing import Dict, List, Optional, Any, Iterator
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import uuid

# Import our existing cyllama bindings
from ..llama_cpp import LlamaModel, LlamaContext, LlamaSampler, llama_batch_get_one


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
    n_threads: int = -1
    n_gpu_layers: int = -1

    # Server features
    embedding: bool = False
    n_parallel: int = 1  # Number of parallel slots

    # OpenAI compatibility
    model_alias: str = "gpt-3.5-turbo"


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
    stream: bool = False
    stop: Optional[List[str]] = None


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
        from ..llama_cpp import LlamaContextParams, LlamaSamplerChainParams

        ctx_params = LlamaContextParams()
        ctx_params.n_ctx = config.n_ctx
        ctx_params.n_batch = config.n_batch
        self.context = LlamaContext(model, ctx_params, verbose=False)

        # Create sampler with similar settings to chat.py
        sampler_params = LlamaSamplerChainParams()
        self.sampler = LlamaSampler(sampler_params)
        self.sampler.add_min_p(0.05, 1)
        self.sampler.add_temp(0.8)
        self.sampler.add_dist(1337)

        # Slot state
        self.is_processing = False
        self.task_id: Optional[str] = None

        # Processing state
        self.generated_tokens: List[int] = []
        self.response_text = ""

    def reset(self):
        """Reset the slot for a new request."""
        self.is_processing = False
        self.task_id = None
        self.generated_tokens.clear()
        self.response_text = ""
        # Reset the context state
        self.context.n_tokens = 0

    def process_and_generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Process prompt and generate response using the slot's context"""
        try:
            # Use the existing context for this slot
            context = self.context

            # Tokenize the prompt
            vocab = self.model.get_vocab()
            prompt_tokens = vocab.tokenize(prompt, add_special=True, parse_special=True)

            if not prompt_tokens:
                return ""

            if len(prompt_tokens) >= self.config.n_ctx:
                return ""  # Prompt too long

            # Create batch for the prompt (like chat.py approach)
            batch = llama_batch_get_one(prompt_tokens, 0)  # Start from position 0
            n_past = len(prompt_tokens)

            # Decode the initial batch
            ret = context.decode(batch)
            if ret != 0:
                logging.warning(f"Initial decode returned {ret}")
                return ""

            # Generation loop
            response_text = ""
            generated_count = 0

            for i in range(max_tokens):
                # Check context size
                if n_past >= context.n_ctx - 1:
                    break

                # Sample next token (like chat.py)
                new_token_id = self.sampler.sample(context, -1)

                # Check for EOS
                if vocab.is_eog(new_token_id):
                    break

                # Convert token to text
                token_piece = vocab.token_to_piece(new_token_id, 0, True)
                response_text += token_piece

                # Create batch for single token at correct position
                batch = llama_batch_get_one([new_token_id], n_past)
                n_past += 1

                # Decode the new token
                ret = context.decode(batch)
                if ret != 0:
                    logging.warning(f"Token decode returned {ret}")
                    break

                generated_count += 1

            self.response_text = response_text
            return response_text

        except Exception as e:
            logging.error(f"Error in process_and_generate: {e}")
            return ""

    def get_generated_text(self) -> str:
        """Get the generated text."""
        return self.response_text



class EmbeddedServer:
    """
    Embedded Llama.cpp server using existing cyllama bindings.

    Provides OpenAI-compatible endpoints without requiring external binaries.
    Uses the existing libllama.a linkage through cyllama Python bindings.
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.model: Optional[LlamaModel] = None
        self.slots: List[ServerSlot] = []

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

            # Load model
            self.model = LlamaModel(
                path_model=self.config.model_path
            )

            # Create slots
            self.slots = []
            for i in range(self.config.n_parallel):
                slot = ServerSlot(i, self.model, self.config)
                self.slots.append(slot)

            self.logger.info(f"Model loaded successfully with {len(self.slots)} slots")
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
            generated_text = slot.process_and_generate(prompt, max_tokens)

            # Check stop words
            if request.stop and generated_text:
                for stop_word in request.stop:
                    if stop_word in generated_text:
                        # Truncate at stop word
                        generated_text = generated_text.split(stop_word)[0]
                        break

            # Estimate token counts (simplified)
            vocab = self.model.get_vocab()
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

    def stop(self):
        """Stop the embedded server."""
        if self.running:
            self.running = False
            if self.httpd:
                self.httpd.shutdown()
                self.httpd.server_close()
            if self.server_thread:
                self.server_thread.join(timeout=5.0)
            self.logger.info("Server stopped")

    def _server_loop(self):
        """Main server loop."""
        try:
            self.httpd.serve_forever()
        except Exception as e:
            if self.running:  # Only log if we didn't shutdown intentionally
                self.logger.error(f"Server loop error: {e}")

    def _create_request_handler(self):
        """Create the HTTP request handler class."""
        server_instance = self

        class RequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Suppress default logging
                pass

            def do_GET(self):
                """Handle GET requests."""
                path = urlparse(self.path).path

                if path == "/health":
                    self._send_json_response({"status": "ok"})
                elif path == "/v1/models":
                    self._handle_models()
                else:
                    self._send_error(404, "Not Found")

            def do_POST(self):
                """Handle POST requests."""
                path = urlparse(self.path).path

                try:
                    # Read request body
                    content_length = int(self.headers.get('Content-Length', 0))
                    if content_length > 0:
                        body = self.rfile.read(content_length).decode('utf-8')
                        data = json.loads(body)
                    else:
                        data = {}

                    if path == "/v1/chat/completions":
                        self._handle_chat_completions(data)
                    elif path == "/v1/embeddings":
                        self._handle_embeddings(data)
                    else:
                        self._send_error(404, "Not Found")

                except json.JSONDecodeError:
                    self._send_error(400, "Invalid JSON")
                except Exception as e:
                    server_instance.logger.error(f"Request error: {e}")
                    self._send_error(500, "Internal Server Error")

            def _send_json_response(self, data: Dict[str, Any], status: int = 200):
                """Send a JSON response."""
                response = json.dumps(data)
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response)))
                self.end_headers()
                self.wfile.write(response.encode('utf-8'))

            def _send_error(self, status: int, message: str):
                """Send an error response."""
                error_data = {
                    "error": {
                        "type": "invalid_request_error",
                        "message": message
                    }
                }
                self._send_json_response(error_data, status)

            def _handle_models(self):
                """Handle /v1/models endpoint."""
                models_data = {
                    "object": "list",
                    "data": [
                        {
                            "id": server_instance.config.model_alias,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "cyllama"
                        }
                    ]
                }
                self._send_json_response(models_data)

            def _handle_chat_completions(self, data: Dict[str, Any]):
                """Handle /v1/chat/completions endpoint."""
                try:
                    # Parse request
                    messages_data = data.get("messages", [])
                    messages = [ChatMessage(role=msg["role"], content=msg["content"])
                              for msg in messages_data]

                    request = ChatRequest(
                        messages=messages,
                        model=data.get("model", server_instance.config.model_alias),
                        max_tokens=data.get("max_tokens"),
                        temperature=data.get("temperature", 0.8),
                        top_p=data.get("top_p", 0.9),
                        stream=data.get("stream", False),
                        stop=data.get("stop")
                    )

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

                    self._send_json_response(response_data)

                except Exception as e:
                    server_instance.logger.error(f"Chat completion error: {e}")
                    self._send_error(500, str(e))

            def _handle_embeddings(self, data: Dict[str, Any]):
                """Handle /v1/embeddings endpoint."""
                if not server_instance.config.embedding:
                    self._send_error(400, "Embeddings not enabled")
                    return

                # Placeholder - would need embedding model support
                self._send_error(501, "Embeddings not yet implemented")

        return RequestHandler

    def __enter__(self):
        """Context manager entry."""
        if self.start():
            return self
        else:
            raise RuntimeError("Failed to start server")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


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

#     with EmbeddedServer(config) as server:
#         print(f"Server running at http://{args.host}:{args.port}")
#         print("Press Ctrl+C to stop...")

#         try:
#             while True:
#                 time.sleep(1)
#         except KeyboardInterrupt:
#             print("\nShutting down server...")
