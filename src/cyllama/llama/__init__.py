def __getattr__(name):
    if name == "LlamaCLI":
        from .cli import LlamaCLI
        return LlamaCLI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from .server import ServerConfig, LlamaServer, LlamaServerClient, start_server

# Import embedded server only when needed to avoid circular imports
def get_embedded_server():
    from .server.embedded import (
        EmbeddedLlamaServer,
        ServerConfig as EmbeddedServerConfig,
        ChatMessage,
        ChatRequest,
        ChatResponse,
        start_embedded_server
    )
    return {
        'EmbeddedLlamaServer': EmbeddedLlamaServer,
        'EmbeddedServerConfig': EmbeddedServerConfig,
        'ChatMessage': ChatMessage,
        'ChatRequest': ChatRequest,
        'ChatResponse': ChatResponse,
        'start_embedded_server': start_embedded_server
    }
