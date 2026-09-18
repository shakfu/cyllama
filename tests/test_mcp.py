"""Tests for MCP client implementation."""

import pytest
from unittest.mock import Mock

from cyllama.agents.mcp import (
    McpServerConfig,
    McpTransportType,
    McpTool,
    McpResource,
    McpClient,
    McpStdioConnection,
    McpHttpConnection,
)
from cyllama.agents.tools import Tool


class TestMcpServerConfig:
    """Tests for MCP server configuration."""

    def test_stdio_config(self):
        config = McpServerConfig(
            name="test",
            transport=McpTransportType.STDIO,
            command="python",
            args=["-m", "mcp_server"],
        )
        assert config.name == "test"
        assert config.transport == McpTransportType.STDIO
        assert config.command == "python"

    def test_http_config(self):
        config = McpServerConfig(
            name="remote",
            transport=McpTransportType.HTTP,
            url="http://localhost:8080/mcp",
        )
        assert config.name == "remote"
        assert config.transport == McpTransportType.HTTP
        assert config.url == "http://localhost:8080/mcp"

    def test_validate_stdio_requires_command(self):
        config = McpServerConfig(
            name="bad",
            transport=McpTransportType.STDIO,
        )
        with pytest.raises(ValueError, match="requires 'command'"):
            config.validate()

    def test_validate_http_requires_url(self):
        config = McpServerConfig(
            name="bad",
            transport=McpTransportType.HTTP,
        )
        with pytest.raises(ValueError, match="requires 'url'"):
            config.validate()

    def test_validate_stdio_success(self):
        config = McpServerConfig(
            name="ok",
            transport=McpTransportType.STDIO,
            command="test",
        )
        # validate() should return None on success (contrast: raises ValueError on bad config).
        assert config.validate() is None

    def test_validate_http_success(self):
        config = McpServerConfig(
            name="ok",
            transport=McpTransportType.HTTP,
            url="http://test",
        )
        assert config.validate() is None


class TestMcpTool:
    """Tests for MCP tool representation."""

    def test_mcp_tool_creation(self):
        tool = McpTool(
            name="search",
            description="Search the web",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            server_name="web",
        )
        assert tool.name == "search"
        assert tool.server_name == "web"

    def test_to_cyllama_tool(self):
        mcp_tool = McpTool(
            name="calc",
            description="Calculate",
            input_schema={"type": "object", "properties": {"x": {"type": "number"}}},
            server_name="math",
        )

        call_func = Mock(return_value="42")
        cyllama_tool = mcp_tool.to_cyllama_tool(call_func)

        assert isinstance(cyllama_tool, Tool)
        assert cyllama_tool.name == "math/calc"
        assert "[math]" in cyllama_tool.description

        # Test the tool can be called
        result = cyllama_tool(x=5)
        call_func.assert_called_once()


class TestMcpResource:
    """Tests for MCP resource representation."""

    def test_resource_creation(self):
        res = McpResource(
            uri="file:///test.txt",
            name="test.txt",
            description="A test file",
            mime_type="text/plain",
            server_name="fs",
        )
        assert res.uri == "file:///test.txt"
        assert res.name == "test.txt"
        assert res.server_name == "fs"


class TestMcpClient:
    """Tests for MCP client."""

    def test_client_creation(self):
        client = McpClient()
        assert not client.is_connected
        assert client.get_tools() == []
        assert client.get_resources() == []

    def test_add_server(self):
        client = McpClient()
        config = McpServerConfig(
            name="test",
            transport=McpTransportType.STDIO,
            command="echo",
        )
        client.add_server(config)
        assert len(client._servers) == 1

    def test_add_server_validates(self):
        client = McpClient()
        config = McpServerConfig(
            name="bad",
            transport=McpTransportType.STDIO,
            # Missing command
        )
        with pytest.raises(ValueError):
            client.add_server(config)

    def test_call_tool_invalid_name(self):
        client = McpClient()
        client._connected = True

        with pytest.raises(ValueError, match="expected 'server/tool' format"):
            client.call_tool("no_slash", {})

    def test_call_tool_server_not_connected(self):
        client = McpClient()
        client._connected = True

        with pytest.raises(ValueError, match="not connected"):
            client.call_tool("unknown/tool", {})

    def test_get_capabilities(self):
        client = McpClient(
            [
                McpServerConfig(
                    name="test1",
                    transport=McpTransportType.STDIO,
                    command="echo",
                ),
            ]
        )

        caps = client.get_capabilities()
        assert "servers" in caps
        assert len(caps["servers"]) == 1
        assert caps["servers"][0]["name"] == "test1"

    def test_get_tools_for_agent(self):
        client = McpClient()
        client._connected = True

        # Manually add a tool
        client._tools["test/search"] = McpTool(
            name="search",
            description="Search",
            input_schema={"type": "object"},
            server_name="test",
        )

        tools = client.get_tools_for_agent()
        assert len(tools) == 1
        assert tools[0].name == "test/search"

    def test_context_manager(self):
        client = McpClient()

        # Mock connect_all and disconnect_all
        client.connect_all = Mock()
        client.disconnect_all = Mock()

        with client as c:
            assert c is client
            client.connect_all.assert_called_once()

        client.disconnect_all.assert_called_once()


class TestMcpStdioConnection:
    """Tests for stdio MCP connection."""

    def test_next_id_increments(self):
        config = McpServerConfig(
            name="test",
            transport=McpTransportType.STDIO,
            command="echo",
        )
        conn = McpStdioConnection(config)

        id1 = conn._next_id()
        id2 = conn._next_id()
        id3 = conn._next_id()

        assert id2 == id1 + 1
        assert id3 == id2 + 1

    def test_send_request_not_running(self):
        config = McpServerConfig(
            name="test",
            transport=McpTransportType.STDIO,
            command="echo",
        )
        conn = McpStdioConnection(config)

        with pytest.raises(RuntimeError, match="not running"):
            conn.send_request("test", {})


class TestMcpHttpConnection:
    """Tests for HTTP MCP connection."""

    def test_connect_is_noop(self):
        config = McpServerConfig(
            name="test",
            transport=McpTransportType.HTTP,
            url="http://localhost:8080",
        )
        conn = McpHttpConnection(config)
        # HTTP connect is a no-op: the stateful connection is per-request.
        # Calling it should be safe and return None; repeated calls should
        # not accumulate state.
        assert conn.connect() is None
        assert conn.connect() is None

    def test_disconnect_is_noop(self):
        config = McpServerConfig(
            name="test",
            transport=McpTransportType.HTTP,
            url="http://localhost:8080",
        )
        conn = McpHttpConnection(config)
        # Disconnect without a prior connect must be idempotent.
        assert conn.disconnect() is None
        assert conn.disconnect() is None

    def test_next_id_increments(self):
        config = McpServerConfig(
            name="test",
            transport=McpTransportType.HTTP,
            url="http://localhost:8080",
        )
        conn = McpHttpConnection(config)

        id1 = conn._next_id()
        id2 = conn._next_id()

        assert id2 == id1 + 1


class TestMcpStdioBlockingIo:
    """A stdio MCP server is a subprocess whose pipes must never be left
    unread. Two independent hangs are covered here:

    * stdout read with a blocking ``readline()`` ignores the configured
      timeout, so a server that never answers blocks the agent thread
      forever.
    * stderr piped but not drained fills its buffer, at which point the
      server blocks writing diagnostics and stops reading stdin -- even
      though stdout is healthy.
    """

    SERVER = """
import sys, time
mode = sys.argv[1]
if mode == "hang":
    for line in sys.stdin:
        time.sleep(3600)
elif mode == "chatty":
    for i in range(20000):
        sys.stderr.write("diagnostic " + "x" * 80 + chr(10))
    sys.stderr.flush()
    for line in sys.stdin:
        sys.stdout.write('{"jsonrpc":"2.0","id":1,"result":{"ok":true}}' + chr(10))
        sys.stdout.flush()
"""

    @pytest.fixture
    def server_script(self, tmp_path):
        script = tmp_path / "fake_mcp_server.py"
        script.write_text(self.SERVER)
        return str(script)

    def _connect(self, server_script, mode, timeout):
        import sys

        config = McpServerConfig(
            name=f"fake-{mode}",
            transport=McpTransportType.STDIO,
            command=sys.executable,
            args=[server_script, mode],
            request_timeout=timeout,
        )
        conn = McpStdioConnection(config)
        conn.connect()
        return conn

    def test_unresponsive_server_times_out(self, server_script):
        """Without a deadline this call never returns."""
        import time as _time

        conn = self._connect(server_script, "hang", timeout=2.0)
        try:
            started = _time.monotonic()
            with pytest.raises(TimeoutError, match="did not respond within"):
                conn.send_request("ping")
            elapsed = _time.monotonic() - started
            assert 1.5 < elapsed < 10.0, f"timeout not honoured: returned after {elapsed:.1f}s"
        finally:
            conn.disconnect()

    def test_stderr_flood_does_not_block_requests(self, server_script):
        """~1.6MB of stderr, far past any pipe buffer, must not deadlock."""
        conn = self._connect(server_script, "chatty", timeout=30.0)
        try:
            assert conn.send_request("ping") == {"ok": True}
        finally:
            conn.disconnect()

    def test_late_response_is_not_returned_to_the_next_request(self, server_script):
        """A timed-out request's response must not satisfy a later one."""
        import sys

        config = McpServerConfig(
            name="fake-mismatch",
            transport=McpTransportType.STDIO,
            command=sys.executable,
            args=["-c", "import sys"],
            request_timeout=5.0,
        )
        conn = McpStdioConnection(config)
        conn.connect()
        try:
            # Queue a stale response carrying an id nobody is waiting for.
            conn._responses.put('{"jsonrpc":"2.0","id":999,"result":"stale"}\n')
            conn._responses.put(None)  # then EOF
            with pytest.raises(RuntimeError, match="closed connection"):
                conn._await_response(request_id=1, timeout=5.0)
        finally:
            conn.disconnect()
