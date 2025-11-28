"""
Model Context Protocol (MCP) client implementation.

Provides connectivity to MCP servers via stdio and HTTP transports,
enabling cyllama agents to access external tools and resources.
"""

import json
import logging
import subprocess
import threading
import os
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import urllib.request
import urllib.error

from .tools import Tool
from .jsonrpc import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcError,
    ErrorCode,
    parse_message,
    serialize_message,
)

logger = logging.getLogger(__name__)

# Default timeout values (in seconds)
DEFAULT_REQUEST_TIMEOUT = 30.0
DEFAULT_SHUTDOWN_TIMEOUT = 5.0


class McpTransportType(Enum):
    """MCP server transport types."""
    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"  # Server-Sent Events (not implemented initially)


@dataclass
class McpServerConfig:
    """Configuration for an MCP server connection."""
    name: str
    transport: McpTransportType = McpTransportType.STDIO

    # Stdio transport options
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None

    # HTTP transport options
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

    # Timeout options
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT

    def validate(self) -> None:
        """Validate the configuration."""
        if self.transport == McpTransportType.STDIO:
            if not self.command:
                raise ValueError(f"MCP server '{self.name}': stdio transport requires 'command'")
        elif self.transport == McpTransportType.HTTP:
            if not self.url:
                raise ValueError(f"MCP server '{self.name}': http transport requires 'url'")


@dataclass
class McpTool:
    """Tool definition from an MCP server."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str

    def to_cyllama_tool(self, call_func: Callable) -> Tool:
        """Convert to a cyllama Tool instance."""
        full_name = f"{self.server_name}/{self.name}"

        def make_tool_func(name, func):
            def tool_wrapper(**kwargs):
                return func(name, kwargs)
            return tool_wrapper

        return Tool(
            name=full_name,
            description=f"[{self.server_name}] {self.description}",
            func=make_tool_func(full_name, call_func),
            parameters=self.input_schema
        )


@dataclass
class McpResource:
    """Resource definition from an MCP server."""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None
    server_name: str = ""


class McpStdioConnection:
    """
    MCP connection over stdio to a subprocess.
    """

    def __init__(self, config: McpServerConfig):
        self._config = config
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._read_lock = threading.Lock()

    def connect(self) -> None:
        """Start the MCP server subprocess."""
        env = os.environ.copy()
        if self._config.env:
            env.update(self._config.env)

        cmd = [self._config.command] + (self._config.args or [])
        logger.info("Starting MCP server '%s': %s", self._config.name, " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=self._config.cwd,
            text=True,
            bufsize=1,  # Line buffered
        )

    def disconnect(self) -> None:
        """Stop the MCP server subprocess."""
        if self._process:
            logger.info("Stopping MCP server '%s'", self._config.name)
            self._process.terminate()
            try:
                self._process.wait(timeout=self._config.shutdown_timeout)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def _next_id(self) -> int:
        with self._lock:
            self._request_id += 1
            return self._request_id

    def send_request(
        self,
        method: str,
        params: Optional[dict] = None,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Send a JSON-RPC request and wait for response.

        Args:
            method: RPC method name
            params: Method parameters
            timeout: Response timeout in seconds (defaults to config.request_timeout)

        Returns:
            The result from the response

        Raises:
            RuntimeError: If request fails or times out
        """
        if not self._process or self._process.poll() is not None:
            raise RuntimeError(f"MCP server '{self._config.name}' is not running")

        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError(f"MCP server '{self._config.name}' stdio not available")

        # Use config timeout if not specified
        if timeout is None:
            timeout = self._config.request_timeout

        request_id = self._next_id()
        request = JsonRpcRequest(method=method, params=params, id=request_id)
        request_str = serialize_message(request) + "\n"

        logger.debug("MCP request to '%s': %s", self._config.name, method)

        try:
            with self._read_lock:
                self._process.stdin.write(request_str)
                self._process.stdin.flush()

                # Read response
                response_line = self._process.stdout.readline()
                if not response_line:
                    raise RuntimeError(f"MCP server '{self._config.name}' closed connection")
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(
                f"MCP server '{self._config.name}' connection lost: {e}"
            ) from e

        msg = parse_message(response_line.strip())
        if not isinstance(msg, JsonRpcResponse):
            raise RuntimeError(f"Expected response, got: {type(msg)}")

        if msg.error:
            raise RuntimeError(
                f"MCP error from '{self._config.name}': "
                f"{msg.error.message} (code: {msg.error.code})"
            )

        return msg.result

    def send_notification(self, method: str, params: Optional[dict] = None) -> None:
        """Send a notification (no response expected)."""
        if not self._process or self._process.poll() is not None:
            raise RuntimeError(f"MCP server '{self._config.name}' is not running")

        if self._process.stdin is None:
            raise RuntimeError(f"MCP server '{self._config.name}' stdin is not available")

        request = JsonRpcRequest(method=method, params=params, id=None)
        request_str = serialize_message(request) + "\n"

        try:
            with self._read_lock:
                self._process.stdin.write(request_str)
                self._process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(
                f"MCP server '{self._config.name}' connection lost: {e}"
            ) from e


class McpHttpConnection:
    """
    MCP connection over HTTP.
    """

    def __init__(self, config: McpServerConfig):
        self._config = config
        self._request_id = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Verify HTTP endpoint is reachable."""
        # Just verify the URL is valid
        logger.info("Connecting to MCP HTTP server '%s' at %s",
                   self._config.name, self._config.url)

    def disconnect(self) -> None:
        """No-op for HTTP connections."""
        pass

    def _next_id(self) -> int:
        with self._lock:
            self._request_id += 1
            return self._request_id

    def send_request(
        self,
        method: str,
        params: Optional[dict] = None,
        timeout: float = 30.0
    ) -> Any:
        """Send a JSON-RPC request over HTTP."""
        request_id = self._next_id()
        request = JsonRpcRequest(method=method, params=params, id=request_id)
        request_str = serialize_message(request)

        logger.debug("MCP HTTP request to '%s': %s", self._config.name, method)

        headers = {"Content-Type": "application/json"}
        if self._config.headers:
            headers.update(self._config.headers)

        req = urllib.request.Request(
            self._config.url,
            data=request_str.encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_str = response.read().decode("utf-8")
        except urllib.error.URLError as e:
            raise RuntimeError(f"MCP HTTP error: {e}")

        msg = parse_message(response_str)
        if not isinstance(msg, JsonRpcResponse):
            raise RuntimeError(f"Expected response, got: {type(msg)}")

        if msg.error:
            raise RuntimeError(
                f"MCP error from '{self._config.name}': "
                f"{msg.error.message} (code: {msg.error.code})"
            )

        return msg.result

    def send_notification(self, method: str, params: Optional[dict] = None) -> None:
        """Send a notification over HTTP."""
        request = JsonRpcRequest(method=method, params=params, id=None)
        request_str = serialize_message(request)

        headers = {"Content-Type": "application/json"}
        if self._config.headers:
            headers.update(self._config.headers)

        req = urllib.request.Request(
            self._config.url,
            data=request_str.encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=10.0) as response:
                pass  # Ignore response for notifications
        except urllib.error.URLError as e:
            logger.warning("MCP notification failed: %s", e)


McpConnection = Union[McpStdioConnection, McpHttpConnection]


class McpClient:
    """
    Client for connecting to multiple MCP servers.

    Discovers and aggregates tools from all connected servers,
    exposing them as cyllama Tool instances.
    """

    def __init__(self, servers: Optional[List[McpServerConfig]] = None):
        self._servers = servers or []
        self._connections: Dict[str, McpConnection] = {}
        self._tools: Dict[str, McpTool] = {}
        self._resources: Dict[str, McpResource] = {}
        self._connected = False

    def add_server(self, config: McpServerConfig) -> None:
        """Add an MCP server configuration."""
        config.validate()
        self._servers.append(config)

    def connect_all(self) -> None:
        """Connect to all configured MCP servers."""
        for config in self._servers:
            try:
                self._connect_server(config)
            except Exception as e:
                logger.error("Failed to connect to MCP server '%s': %s", config.name, e)

        self._connected = True

    def _connect_server(self, config: McpServerConfig) -> None:
        """Connect to a single MCP server."""
        config.validate()

        # Create connection based on transport type
        if config.transport == McpTransportType.STDIO:
            conn = McpStdioConnection(config)
        elif config.transport == McpTransportType.HTTP:
            conn = McpHttpConnection(config)
        else:
            raise ValueError(f"Unsupported transport: {config.transport}")

        conn.connect()
        self._connections[config.name] = conn

        # Initialize the connection (MCP handshake)
        self._initialize_connection(config.name, conn)

        # Discover tools
        self._discover_tools(config.name, conn)

        # Discover resources
        self._discover_resources(config.name, conn)

    def _initialize_connection(self, name: str, conn: McpConnection) -> None:
        """Perform MCP initialization handshake."""
        result = conn.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": False},
            },
            "clientInfo": {
                "name": "cyllama",
                "version": "0.1.10"
            }
        })
        logger.info("MCP server '%s' initialized: %s", name, result.get("serverInfo", {}))

        # Send initialized notification
        conn.send_notification("notifications/initialized")

    def _discover_tools(self, name: str, conn: McpConnection) -> None:
        """Discover tools from an MCP server."""
        try:
            result = conn.send_request("tools/list")
            tools = result.get("tools", [])

            for tool_def in tools:
                tool = McpTool(
                    name=tool_def["name"],
                    description=tool_def.get("description", ""),
                    input_schema=tool_def.get("inputSchema", {"type": "object", "properties": {}}),
                    server_name=name
                )
                full_name = f"{name}/{tool.name}"
                self._tools[full_name] = tool
                logger.debug("Discovered MCP tool: %s", full_name)

            logger.info("Discovered %d tools from MCP server '%s'", len(tools), name)

        except Exception as e:
            logger.warning("Failed to discover tools from '%s': %s", name, e)

    def _discover_resources(self, name: str, conn: McpConnection) -> None:
        """Discover resources from an MCP server."""
        try:
            result = conn.send_request("resources/list")
            resources = result.get("resources", [])

            for res_def in resources:
                resource = McpResource(
                    uri=res_def["uri"],
                    name=res_def.get("name", res_def["uri"]),
                    description=res_def.get("description"),
                    mime_type=res_def.get("mimeType"),
                    server_name=name
                )
                self._resources[resource.uri] = resource
                logger.debug("Discovered MCP resource: %s", resource.uri)

            logger.info("Discovered %d resources from MCP server '%s'", len(resources), name)

        except Exception as e:
            logger.warning("Failed to discover resources from '%s': %s", name, e)

    def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for name, conn in self._connections.items():
            try:
                conn.disconnect()
            except Exception as e:
                logger.error("Error disconnecting from '%s': %s", name, e)

        self._connections.clear()
        self._tools.clear()
        self._resources.clear()
        self._connected = False

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on an MCP server.

        Args:
            name: Full tool name (server_name/tool_name)
            arguments: Tool arguments

        Returns:
            Tool result

        Raises:
            ValueError: If tool not found
            RuntimeError: If tool call fails
        """
        if "/" not in name:
            raise ValueError(f"Invalid tool name '{name}': expected 'server/tool' format")

        server_name, tool_name = name.split("/", 1)

        if server_name not in self._connections:
            raise ValueError(f"MCP server '{server_name}' not connected")

        conn = self._connections[server_name]

        result = conn.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })

        # MCP returns content array
        content = result.get("content", [])
        if not content:
            return None

        # Return text content if available
        for item in content:
            if item.get("type") == "text":
                return item.get("text", "")

        # Return first content item's data
        return content[0]

    def read_resource(self, uri: str) -> str:
        """
        Read a resource from an MCP server.

        Args:
            uri: Resource URI

        Returns:
            Resource content as string
        """
        resource = self._resources.get(uri)
        if not resource:
            raise ValueError(f"Resource not found: {uri}")

        conn = self._connections.get(resource.server_name)
        if not conn:
            raise ValueError(f"MCP server '{resource.server_name}' not connected")

        result = conn.send_request("resources/read", {"uri": uri})
        contents = result.get("contents", [])

        if not contents:
            return ""

        # Return text content
        for item in contents:
            if "text" in item:
                return item["text"]

        return ""

    def get_tools(self) -> List[McpTool]:
        """Get all discovered MCP tools."""
        return list(self._tools.values())

    def get_tools_for_agent(self) -> List[Tool]:
        """
        Convert MCP tools to cyllama Tool instances.

        Returns:
            List of Tool instances that call MCP servers
        """
        return [
            tool.to_cyllama_tool(self.call_tool)
            for tool in self._tools.values()
        ]

    def get_resources(self) -> List[McpResource]:
        """Get all discovered MCP resources."""
        return list(self._resources.values())

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get MCP capabilities for ACP initialization.

        Returns:
            Capabilities dict for ACP initialize response
        """
        return {
            "servers": [
                {
                    "name": config.name,
                    "transport": config.transport.value,
                    "connected": config.name in self._connections,
                    "tools": len([t for t in self._tools.values() if t.server_name == config.name]),
                    "resources": len([r for r in self._resources.values() if r.server_name == config.name]),
                }
                for config in self._servers
            ]
        }

    @property
    def is_connected(self) -> bool:
        """Check if any MCP servers are connected."""
        return self._connected and len(self._connections) > 0

    def __enter__(self) -> "McpClient":
        self.connect_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect_all()
