# MCP Integration Proposal

Status: design draft. Tracks the `MCP integration` entry in `TODO.md`.

Cyllama's relationship to the Model Context Protocol (MCP) splits into two
independent directions, both of which are cyllama-layer concerns -- llama.cpp
upstream has no server-side MCP support; only the Svelte webui ships a
TypeScript client (`build/llama.cpp/tools/server/webui/src/lib/utils/mcp.ts`).

1. **Client direction** -- a local LLM consumes external MCP servers' tools
   during a tool-calling loop. Already partially implemented in
   `src/cyllama/agents/mcp.py` (stdio + HTTP transports, wired into the
   agent `Tool` abstraction). Needs to be lifted onto the top-level
   `LLM`/`chat()` API so non-agent callers benefit too.
2. **Server direction** -- cyllama exposes its inference capabilities as MCP
   tools and its model catalog as resources, so MCP clients (Claude Code,
   Claude Desktop, etc.) can drive local GGUF models. New package
   `src/cyllama/mcp/`, served via stdio and via Streamable-HTTP routes
   mounted on the existing `EmbeddedServer`.

Both sides reuse `src/cyllama/agents/jsonrpc.py` for framing and the
high-level API in `src/cyllama/api.py` for execution. No new heavy deps.

## 1. MCP client at the top-level `LLM` / `chat()` API

The transports in `agents/mcp.py` (`McpClient`, `McpServerConfig`,
`McpTransportType`, `get_tools_for_agent()`) stay as-is. New surface in
`cyllama/api.py`:

```python
from cyllama.agents.mcp import (
    McpClient, McpServerConfig, McpTransportType, McpTool, McpResource,
)

class LLM:
    def add_mcp_server(
        self,
        name: str,
        *,
        # stdio transport
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        # http transport
        url: str | None = None,
        headers: dict[str, str] | None = None,
        # inferred from which kwargs are set if omitted
        transport: McpTransportType | None = None,
    ) -> None: ...

    def remove_mcp_server(self, name: str) -> None: ...
    def list_mcp_tools(self) -> list[McpTool]: ...
    def list_mcp_resources(self) -> list[McpResource]: ...

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[Tool] | None = None,
        use_mcp: bool = True,             # auto-include attached servers' tools
        max_tool_iterations: int = 8,
        ...
    ) -> ChatResponse: ...
```

Module-level convenience mirrors:

```python
def chat(
    messages,
    *,
    model_path,
    mcp_servers: list[McpServerConfig] | None = None,
    ...,
): ...
```

### Internals

- `LLM` lazily owns a single `McpClient`.

- `chat()` pulls `client.get_tools_for_agent()`, merges with caller-supplied
  `tools`, runs the existing tool-call loop, and dispatches MCP tool calls
  via `client.call_tool(name, args)`.

- Connect on first use; disconnect on `LLM.close()` / `__exit__`.

### Open questions

- Sync-only (matches current `agents/mcp.py`) vs. add async path. Recommend
  sync first; revisit when an async caller appears.

- Resource handling: surface `mcp://` URIs through an explicit
  `read_resource()` helper rather than auto-injecting into the system
  prompt. Auto-injection is hard to undo and easy to abuse.

## 2. Cyllama as MCP server

New package layout:

```
src/cyllama/mcp/
  __init__.py
  protocol.py      # MCP method dispatch over JSON-RPC (uses agents/jsonrpc.py)
  tools.py         # registry: name -> (input_schema, handler)
  resources.py     # model listing, gguf metadata
  stdio.py         # `python -m cyllama.mcp` stdio transport
  http.py          # Streamable-HTTP route handlers (mounted on EmbeddedServer)
```

### Tool surface

One MCP tool per high-level capability; all are thin wrappers over the
existing `cyllama` API.

| MCP tool         | Backed by                        | Inputs (subset)                                |
|------------------|----------------------------------|------------------------------------------------|
| `complete`       | `cyllama.complete`               | `prompt`, `model`, `max_tokens`, `temperature` |
| `chat`           | `cyllama.chat`                   | `messages`, `model`, `tools`                   |
| `embed`          | `LLM.embed`                      | `input` (str or list), `model`                 |
| `transcribe`     | `whisper` high-level             | `audio_path` or base64, `language`             |
| `generate_image` | `cyllama.sd.text_to_image`       | `prompt`, `width`, `height`, `steps`           |

### Resources

- `models://local` -- list of GGUF files discovered under configured roots.

- `model://<name>` -- JSON metadata (arch, params, ctx, quant) via existing
  model-introspection helpers.

### Server entry points

```python
# stdio transport (Claude Desktop config: command=python, args=["-m", "cyllama.mcp"])
def serve_stdio(options: McpServerOptions) -> None: ...

# Embedded HTTP transport (mounted into the existing mongoose server)
class EmbeddedServer:
    def enable_mcp(
        self,
        *,
        path: str = "/mcp",
        options: McpServerOptions | None = None,
    ) -> None: ...
```

`McpServerOptions` carries:

- Allowed model roots (filesystem allowlist).

- Default model.

- Tool subset to expose (a deployment can expose `embed` only, etc.).

- Auth hook for the HTTP route.

### Wire-level

Implement only the subset MCP requires today:

- `initialize`

- `tools/list`, `tools/call`

- `resources/list`, `resources/read`

- `ping`

- `notifications/initialized`

On stdio, reuse `agents/jsonrpc.py` framing. On HTTP, follow the
Streamable-HTTP spec: single `POST /mcp` for requests, `GET /mcp` upgrading
to SSE for server-initiated messages.

### Open questions

- Streaming `tools/call` partial results via SSE: defer until a real client
  consumes it.

- Concurrency: `EmbeddedServer` is single-threaded; long-running
  `generate_image` calls will block other routes. Either gate behind a
  worker thread or document the limitation in the route handler.

## Merits and counterweights

### Server-direction merits

- **Single endpoint, multiple modalities.** Claude Desktop / Claude Code can
  reach `complete`, `embed`, `transcribe`, `generate_image` through one
  configured server. Upstream llama.cpp ships only an OpenAI-compat HTTP API
  -- nothing for whisper or SD.

- **Local, offline, private.** GGUF inference stays on the host but is
  reachable from a frontier-model agent loop. Useful for cheap bulk
  embedding, transcription of sensitive audio, or fast small-model drafts.

- **Capability-gated surface.** `McpServerOptions` lets a deployment expose
  `embed` only (or `transcribe` only). Embedding-as-a-service over MCP is a
  clean fit.

- **Resource surface fits naturally.** `models://local` + `model://<name>`
  is exactly what MCP resources are for; the introspection helpers exist.

- **Reuses existing infra.** `agents/jsonrpc.py` framing plus mongoose
  routes mean small marginal code.

### Counterweights

- **Llama.cpp's HTTP server already covers `complete`/`chat`/`embed`** via
  OpenAI-compat. The MCP server's incremental value over `llama-server` is
  mostly `transcribe` + `generate_image` + the resource catalog.

- **Asymmetric value.** The client direction (local LLM consuming MCP
  tools) is more clearly useful -- it gives small local models real reach.
  The server direction mostly benefits frontier-model users who want a
  local fallback, a smaller audience.

- **Concurrency mismatch.** `EmbeddedServer` is single-threaded;
  `generate_image` blocks for tens of seconds. Either add worker threads
  (non-trivial) or the server is functionally single-user.

- **Protocol churn.** MCP transport spec moved twice in 2025 (stdio ->
  HTTP+SSE -> Streamable HTTP). Building now means tracking spec changes.

- **Better-served by an off-the-shelf wrapper for stdio.** A ~200-line
  script using the `mcp` Python SDK that imports cyllama would deliver most
  of the stdio value without touching `EmbeddedServer`. The HTTP-mounted
  variant is the part that justifies in-tree code.

## Recommendation and build order

Build the **client direction** now -- clear win, code mostly exists. Defer
the **server direction** until either (a) a concrete user wants
`transcribe` / `generate_image` over MCP, or (b) the resource catalog
unlocks a specific workflow. When the server direction is built, start with
the HTTP transport on `EmbeddedServer` (no off-the-shelf substitute) and
skip stdio unless a concrete client needs it.

1. **Now.** Lift `agents/mcp.py` onto `LLM` / `chat()` (client surface).
   Lowest risk; transports and tests already exist (`tests/test_mcp.py`).
2. **Deferred, when triggered.** Stand up `src/cyllama/mcp/` with the
   Streamable-HTTP transport mounted on `EmbeddedServer.enable_mcp()`.
   Start with `embed` + the resource surface (the parts llama-server can't
   already do well over OpenAI-compat).
3. **Deferred further.** Add `transcribe` and `generate_image` tools, and
   address the single-threaded `EmbeddedServer` concurrency limitation
   before exposing them.
4. **Only on demand.** Add the stdio entrypoint. Until then, document the
   off-the-shelf SDK-wrapper pattern for users who need stdio today.
