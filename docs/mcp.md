# MCP Client

cyllama's `LLM` class can attach to one or more [Model Context Protocol](https://modelcontextprotocol.io) servers and use their tools and resources during a chat-driven tool loop. The integration is a thin layer over `cyllama.agents.mcp` and drives a text-based `ReActAgent`, so it works on any GGUF without requiring a model trained for OpenAI-style structured tool calls.

> **Scope.** Only the *client* direction is exposed on `LLM` today — cyllama consumes external MCP servers. The *server* direction (exposing cyllama inference as MCP tools) is tracked as a design in [`docs/dev/mcp.md`](dev/mcp.md) and is not yet implemented.

## Quick Start

```python
from cyllama import LLM

llm = LLM("models/Llama-3.2-3B-Instruct-Q8_0.gguf")

# stdio-transport server (command + args)
llm.add_mcp_server(
    "fs",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)

# HTTP-transport server (url + optional headers)
llm.add_mcp_server(
    "search",
    url="http://localhost:9000/mcp",
    headers={"Authorization": "Bearer …"},
)

# Drive a tool-calling loop; MCP tools are merged in automatically.
answer = llm.chat_with_tools(
    messages=[{"role": "user", "content": "List the files under /tmp/reports"}],
)
print(answer)

llm.close()  # disconnects all MCP servers best-effort
```

## API

All methods live on `LLM`. `McpTool`, `McpResource`, `McpServerConfig`, and `McpTransportType` are re-exported from `cyllama.agents.mcp`.

### `add_mcp_server(name, *, command=…, args=…, url=…, headers=…, …)`

Attach an MCP server and connect immediately (fail-fast on a bad config or unreachable endpoint).

- `name` — logical name. Tools discovered on this server are exposed as `"<name>/<tool>"` to avoid collisions when multiple servers expose identically named tools.

- **stdio transport**: set `command` (required), and optionally `args`, `env`, `cwd`.

- **HTTP transport**: set `url` (required), and optionally `headers`.

- Transport is inferred from which kwargs are present; pass `transport=McpTransportType.STDIO|HTTP` explicitly to disambiguate.

- `request_timeout` / `shutdown_timeout` override per-server timeouts (defaults defined in `cyllama.agents.mcp`).

### `remove_mcp_server(name)`

Disconnect and forget a server by name. Tools and resources it contributed are removed from the client's catalog. Safe to call if no servers are attached.

### `list_mcp_tools() -> list[McpTool]`

Return every discovered tool across all attached servers. Each `McpTool` carries `name`, `description`, `input_schema`, and `server_name`.

### `list_mcp_resources() -> list[McpResource]`

Return every discovered resource: `uri`, `name`, `description`, `mime_type`, plus the owning `server_name`.

### `call_mcp_tool(name, arguments) -> Any`

Invoke a tool directly by its fully-qualified `"server/tool"` name. Use this when you want explicit control of the loop instead of delegating to `chat_with_tools`.

```python
result = llm.call_mcp_tool("fs/list_directory", {"path": "/tmp"})
```

### `read_mcp_resource(uri) -> str`

Fetch a resource's contents by URI.

### `chat_with_tools(messages, *, tools=None, use_mcp=True, max_iterations=8, verbose=False, system_prompt=None, generation_config=None) -> str`

Run a tool-calling ReAct loop over chat messages. The last user message is the agent task; a leading system message becomes the agent's system prompt unless `system_prompt` is given explicitly.

- `tools` — additional cyllama `Tool` instances merged with the MCP tools.

- `use_mcp=False` — drop MCP tools and use only the caller-supplied `tools`.

- `max_iterations` — cap on thought/action cycles.

- Returns the agent's final answer string.

Multi-turn conversation history beyond a single user turn is not yet plumbed through — the ReAct loop operates one task at a time.

## Direct Tool Invocation Example

```python
from cyllama import LLM

with LLM("models/Llama-3.2-3B-Instruct-Q8_0.gguf") as llm:
    llm.add_mcp_server(
        "fs",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )

    for tool in llm.list_mcp_tools():
        print(f"{tool.name}: {tool.description}")

    listing = llm.call_mcp_tool("fs/list_directory", {"path": "/tmp"})
    print(listing)
```

## Mixing MCP Tools with Local Tools

```python
from cyllama.agents import Tool

def add(a: int, b: int) -> int:
    return a + b

local_tool = Tool.from_function(add)

answer = llm.chat_with_tools(
    messages=[{"role": "user", "content": "What is 2+3, and what files are in /tmp?"}],
    tools=[local_tool],     # merged with MCP tools from attached servers
    use_mcp=True,
    verbose=True,
)
```

Set `use_mcp=False` to run the loop against local `tools` only, even when MCP servers are attached.

## Cleanup

`LLM.close()` disconnects every attached server best-effort. Prefer the context-manager form shown above — it closes the LLM and all MCP transports on exit even if the loop raises.

## See Also

- [Agents Overview](agents_overview.md) — `ReActAgent` and the `Tool` abstraction

- [`docs/dev/mcp.md`](dev/mcp.md) — design draft, including the deferred server direction

- `cyllama.agents.mcp` — `McpClient`, `McpServerConfig`, `McpTransportType`, `McpConnectionProtocol`
