# Import from the embedded server
from .embedded import (
    ServerConfig as EmbeddedServerConfig,
    EmbeddedServer,
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

# Import from the Mongoose server (high-performance C server)
try:
    from .mongoose_server import (
        MongooseServer,
        start_mongoose_server
    )
    _MONGOOSE_AVAILABLE = True
except ImportError:
    _MONGOOSE_AVAILABLE = False

# Default to embedded server config for new usage
ServerConfig = EmbeddedServerConfig