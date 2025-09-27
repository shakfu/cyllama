# Import from the embedded server
from .embedded import (
    ServerConfig as EmbeddedServerConfig,
    EmbeddedLlamaServer,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    start_embedded_server
)

# Import from the launcher (external binary wrapper)
from .launcher import (
    ServerConfig as LauncherServerConfig,
    LlamaServer,
    LlamaServerClient,
    start_server
)

# Default to embedded server config for new usage
ServerConfig = EmbeddedServerConfig