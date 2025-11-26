# Protocol Support (DRAFT)

This document provides an overview of the experimental Model Context Protocol (MCP) and Agent Client Protocol (ACP) implementation in cyllama.

**CAVEAT**: the implementation is an initial-stage prototype, it has not been tested at all, so **don't** expect it to work.

## Architecture

```text
[Editor/IDE] <--ACP--> [cyllama.agents.ACPAgent]
        (JSON-RPC/stdio)         |
                                 v
                         [Inner Agent]
                         (ReActAgent or
                          ConstrainedAgent)
                                 |
                                 v
                    [cyllama core + MCP client]
                                 |
                                 v
                          [MCP servers]
```

- **ACP** (Agent Client Protocol): Standardizes communication between code editors/IDEs and AI coding agents
- **MCP** (Model Context Protocol): Standardizes how models access external tools and resources

## Module Structure

```text
src/cyllama/agents/
    jsonrpc.py     # JSON-RPC 2.0 transport layer
    mcp.py         # MCP client implementation
    session.py     # Session storage backends
    acp.py         # ACP agent implementation
    cli.py         # Command-line interface
```

## Components

### JSON-RPC Transport (`jsonrpc.py`)

Provides the communication layer for both ACP and MCP protocols.

**Key Classes:**

- `JsonRpcRequest` / `JsonRpcResponse` - Message types
- `JsonRpcError` - Error handling with standard error codes
- `StdioTransport` - Newline-delimited JSON over stdin/stdout
- `JsonRpcServer` - Request dispatching and handler registration
- `AsyncBridge` - Queue-based bridge for sending notifications from sync code

**Example:**

```python
from cyllama.agents import JsonRpcServer, StdioTransport

transport = StdioTransport()
server = JsonRpcServer(transport)

@server.register("my_method")
def handle_my_method(params):
    return {"result": "ok"}

server.serve()  # Blocks, processing requests
```

### MCP Client (`mcp.py`)

Client for connecting to MCP servers and discovering/invoking tools.

**Key Classes:**

- `McpServerConfig` - Server connection configuration
- `McpTransportType` - Transport type enum (STDIO, HTTP)
- `McpClient` - Main client class
- `McpTool` / `McpResource` - Tool and resource representations

**Transports:**

- **Stdio**: Spawns MCP server as subprocess, communicates via stdin/stdout
- **HTTP**: Sends JSON-RPC requests over HTTP POST

**Example:**

```python
from cyllama.agents import McpClient, McpServerConfig, McpTransportType

client = McpClient([
    McpServerConfig(
        name="filesystem",
        transport=McpTransportType.STDIO,
        command="npx",
        args=["-y", "@anthropic/mcp-filesystem", "/path/to/dir"],
    ),
])

with client:
    # Tools are automatically discovered
    tools = client.get_tools_for_agent()

    # Call a tool directly
    result = client.call_tool("filesystem/read_file", {"path": "test.txt"})
```

### Session Storage (`session.py`)

Persistent storage for ACP sessions, including conversation history and permission caching.

**Storage Backends:**

- `MemorySessionStore` - In-memory (default, non-persistent)
- `FileSessionStore` - JSON files in a directory
- `SqliteSessionStore` - SQLite database

**Key Classes:**

- `Session` - Session state (messages, tool calls, permissions)
- `Message` - Conversation message
- `ToolCallRecord` - Record of a tool invocation
- `Permission` - Cached permission decision

**Example:**

```python
from cyllama.agents import create_session_store, Session

# Create a SQLite-backed store
store = create_session_store("sqlite", "~/.cyllama/sessions.db")

# Create and save a session
session = Session(id="sess_123")
session.add_message("user", "Hello")
session.add_permission("shell", "allow_always")
store.save(session)

# Load later
restored = store.load("sess_123")
```

### ACP Agent (`acp.py`)

The main ACP-compliant agent that editors connect to.

**Key Classes:**

- `ACPAgent` - Main agent class
- `ContentBlock` - ACP content representation
- `ToolCallUpdate` - Tool call status updates
- `SessionUpdate` - Session update notifications

**Supported ACP Methods:**

| Method | Direction | Description |
|--------|-----------|-------------|
| `initialize` | Client->Agent | Capability negotiation |
| `authenticate` | Client->Agent | Authentication (currently no-op) |
| `session/new` | Client->Agent | Create new session |
| `session/load` | Client->Agent | Resume existing session |
| `session/prompt` | Client->Agent | Process user prompt |
| `session/update` | Agent->Client | Stream response updates |
| `session/cancel` | Client->Agent | Cancel processing |
| `session/set_mode` | Client->Agent | Change agent mode |
| `session/request_permission` | Agent->Client | Request tool permission |
| `fs/read_text_file` | Agent->Client | Read file via editor |
| `fs/write_text_file` | Agent->Client | Write file via editor |
| `terminal/create` | Agent->Client | Execute command |
| `terminal/output` | Agent->Client | Get command output |

**Example:**

```python
from cyllama.api import LLM
from cyllama.agents import ACPAgent, McpServerConfig, McpTransportType

llm = LLM("path/to/model.gguf")

agent = ACPAgent(
    llm=llm,
    mcp_servers=[
        McpServerConfig(
            name="filesystem",
            transport=McpTransportType.STDIO,
            command="npx",
            args=["-y", "@anthropic/mcp-filesystem"],
        ),
    ],
    session_storage="sqlite",
    session_path="~/.cyllama/sessions.db",
)

# Start serving (blocks)
agent.serve()
```

## Command-Line Interface

The `cli.py` module provides commands for running ACP servers and testing MCP connections.

### Start ACP Server

```bash
# Basic usage
python -m cyllama.agents.cli acp --model path/to/model.gguf

# With MCP servers
python -m cyllama.agents.cli acp --model model.gguf \
    --mcp-stdio "fs:npx:-y:@anthropic/mcp-filesystem:/home/user"

# With persistent sessions
python -m cyllama.agents.cli acp --model model.gguf \
    --session-storage sqlite \
    --session-path ~/.cyllama/sessions.db

# Verbose mode
python -m cyllama.agents.cli acp --model model.gguf -v
```

### Run Single Query

```bash
python -m cyllama.agents.cli run --model model.gguf -p "What is 2+2?"

# With shell tool enabled
python -m cyllama.agents.cli run --model model.gguf \
    --enable-shell -p "List files in current directory"
```

### Test MCP Connection

```bash
# Test stdio MCP server
python -m cyllama.agents.cli mcp-test \
    --stdio "fs:npx:-y:@anthropic/mcp-filesystem:/tmp"

# Test HTTP MCP server
python -m cyllama.agents.cli mcp-test \
    --http "myserver:http://localhost:8080/mcp"

# Call a specific tool
python -m cyllama.agents.cli mcp-test \
    --stdio "fs:npx:-y:@anthropic/mcp-filesystem:/tmp" \
    --call-tool 'read_file:{"path": "test.txt"}'
```

## Integration with Editors

### Zed Editor

Add to your Zed settings:

```json
{
  "agents": {
    "cyllama": {
      "command": "python",
      "args": ["-m", "cyllama.agents.cli", "acp", "--model", "/path/to/model.gguf"]
    }
  }
}
```

### Generic Editor Integration

Editors communicate with cyllama via JSON-RPC over stdio. The typical flow:

1. Editor spawns cyllama as subprocess
2. Editor sends `initialize` request
3. Editor sends `session/new` to create a session
4. Editor sends `session/prompt` with user input
5. Agent streams `session/update` notifications
6. Agent may send `session/request_permission` for tool approval
7. Agent may send `fs/read_text_file` etc. to access editor's workspace

## Permission Handling

When the agent needs to execute a sensitive tool, it requests permission from the editor:

1. Agent sends `session/request_permission` with tool details and options
2. Editor displays permission UI to user
3. User selects an option (allow once, always, deny once, always)
4. Editor returns selection to agent
5. "Always" permissions are cached in the session store

```python
# Permissions are cached per-session
session.add_permission("shell", "allow_always")
perm = session.get_permission("shell")
if perm and perm.kind == "allow_always":
    # Skip permission request
    pass
```

## Async Bridge

The inner agents (ReActAgent, etc.) are synchronous, but ACP benefits from streaming updates. The `AsyncBridge` class provides a queue-based mechanism:

```python
from cyllama.agents import AsyncBridge

bridge = AsyncBridge(server)
bridge.start()

# From sync agent code, queue notifications
bridge.send_notification("session/update", {"content": [...]})

# Worker thread sends them asynchronously
bridge.stop()
```

## Error Handling

JSON-RPC errors use standard error codes:

| Code | Name | Description |
|------|------|-------------|
| -32700 | Parse Error | Invalid JSON |
| -32600 | Invalid Request | Invalid JSON-RPC |
| -32601 | Method Not Found | Unknown method |
| -32602 | Invalid Params | Invalid parameters |
| -32603 | Internal Error | Server error |
| -32001 | Session Not Found | ACP: Session doesn't exist |
| -32002 | Permission Denied | ACP: Permission rejected |
| -32003 | Operation Cancelled | ACP: User cancelled |

## Testing

Tests are located in `tests/`:

```bash
# Run all ACP/MCP tests
pytest tests/test_jsonrpc.py tests/test_session.py tests/test_mcp.py tests/test_acp.py -v

# Run specific test class
pytest tests/test_acp.py::TestACPAgentHandlers -v
```

## Future Enhancements

See `ACP_MCP.md` in the project root for the full development roadmap, including:

- Planning support (`Plan`, `PlanEntry`)
- Multiple agent modes
- Image/audio content support
- SSE transport for MCP
- Session export/import
