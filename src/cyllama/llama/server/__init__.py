# Import from the Python server (pure Python implementation)
from .python import (
    ServerConfig as PythonServerConfig,
    PythonServer,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    start_python_server
)

# Import from the launcher (external binary wrapper)
from .launcher import (
    ServerConfig as LauncherServerConfig,
    LlamaServer,
    LlamaServerClient,
    start_server
)

# Import from the embedded server (high-performance C server using Mongoose)
try:
    from .embedded import (
        EmbeddedServer,
        start_embedded_server
    )
    _EMBEDDED_AVAILABLE = True
except ImportError:
    _EMBEDDED_AVAILABLE = False

# Default to Python server config for new usage
ServerConfig = PythonServerConfig