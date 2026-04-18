"""Tests for the MCP client surface on the high-level LLM API.

These tests do not load a GGUF model. They exercise only the attachment /
dispatch surface added in ``cyllama.api.LLM``, mocking the underlying
``McpClient`` so the suite stays fast and runs without network or
subprocess dependencies. End-to-end tests against a real MCP server live
separately.
"""

from unittest.mock import MagicMock, patch

import pytest

from cyllama.agents.mcp import McpResource, McpTool, McpTransportType


def _make_unloaded_llm():
    """Construct an LLM-like stub that exposes only the MCP surface.

    The full ``LLM.__init__`` loads a GGUF model. For the attachment-API
    tests we just need an object with the MCP methods bound, so we
    bypass ``__init__`` and set the few attributes the methods touch.
    """
    from cyllama.api import LLM

    obj = LLM.__new__(LLM)
    obj._mcp_client = None
    obj._closed = False
    obj.verbose = False
    obj._ctx = None
    obj._sampler = None
    return obj


class TestTransportInference:
    def test_command_implies_stdio(self):
        llm = _make_unloaded_llm()
        with patch("cyllama.agents.mcp.McpClient") as MockClient:
            instance = MockClient.return_value
            instance._connections = {}

            llm.add_mcp_server("local", command="echo", args=["hi"])

            cfg = instance.add_server.call_args.args[0]
            assert cfg.transport == McpTransportType.STDIO
            assert cfg.command == "echo"

    def test_url_implies_http(self):
        llm = _make_unloaded_llm()
        with patch("cyllama.agents.mcp.McpClient") as MockClient:
            instance = MockClient.return_value
            instance._connections = {}

            llm.add_mcp_server("remote", url="http://localhost:9000/mcp")

            cfg = instance.add_server.call_args.args[0]
            assert cfg.transport == McpTransportType.HTTP
            assert cfg.url == "http://localhost:9000/mcp"

    def test_neither_command_nor_url_raises(self):
        llm = _make_unloaded_llm()
        with pytest.raises(ValueError, match="command|url|transport"):
            llm.add_mcp_server("bad")


class TestAttachmentLifecycle:
    def test_lazy_client_creation(self):
        llm = _make_unloaded_llm()
        assert llm._mcp_client is None
        # list_* before any attach returns empty without constructing a client.
        assert llm.list_mcp_tools() == []
        assert llm.list_mcp_resources() == []
        assert llm._mcp_client is None

    def test_add_then_remove(self):
        llm = _make_unloaded_llm()
        with patch("cyllama.agents.mcp.McpClient") as MockClient:
            instance = MockClient.return_value
            instance._connections = {}
            instance._servers = []
            instance._tools = {}
            instance._resources = {}

            def fake_add(cfg):
                instance._servers.append(cfg)

            def fake_connect(cfg):
                instance._connections[cfg.name] = MagicMock()

            instance.add_server.side_effect = fake_add
            instance._connect_server.side_effect = fake_connect

            llm.add_mcp_server("s1", command="echo")
            assert "s1" in instance._connections

            llm.remove_mcp_server("s1")
            assert "s1" not in instance._connections
            assert instance._servers == []

    def test_close_disconnects_mcp(self):
        llm = _make_unloaded_llm()
        client = MagicMock()
        llm._mcp_client = client

        llm.close()

        client.disconnect_all.assert_called_once()
        assert llm._mcp_client is None

    def test_close_swallows_disconnect_error(self):
        llm = _make_unloaded_llm()
        client = MagicMock()
        client.disconnect_all.side_effect = RuntimeError("transport broke")
        llm._mcp_client = client
        # Should not propagate -- local resource cleanup must not be
        # blocked by a flaky remote.
        llm.close()
        assert llm._mcp_client is None


class TestDispatch:
    def test_call_mcp_tool_without_attachment_raises(self):
        llm = _make_unloaded_llm()
        with pytest.raises(RuntimeError, match="add_mcp_server"):
            llm.call_mcp_tool("srv/tool", {})

    def test_read_mcp_resource_without_attachment_raises(self):
        llm = _make_unloaded_llm()
        with pytest.raises(RuntimeError, match="add_mcp_server"):
            llm.read_mcp_resource("file:///x")

    def test_call_mcp_tool_delegates(self):
        llm = _make_unloaded_llm()
        client = MagicMock()
        client.call_tool.return_value = "ok"
        llm._mcp_client = client

        result = llm.call_mcp_tool("srv/tool", {"x": 1})

        client.call_tool.assert_called_once_with("srv/tool", {"x": 1})
        assert result == "ok"

    def test_read_mcp_resource_delegates(self):
        llm = _make_unloaded_llm()
        client = MagicMock()
        client.read_resource.return_value = "content"
        llm._mcp_client = client

        assert llm.read_mcp_resource("file:///x") == "content"
        client.read_resource.assert_called_once_with("file:///x")

    def test_list_methods_proxy_to_client(self):
        llm = _make_unloaded_llm()
        client = MagicMock()
        client.get_tools.return_value = [
            McpTool(name="search", description="d", input_schema={}, server_name="srv"),
        ]
        client.get_resources.return_value = [
            McpResource(uri="file:///x", name="x", server_name="srv"),
        ]
        llm._mcp_client = client

        tools = llm.list_mcp_tools()
        resources = llm.list_mcp_resources()

        assert len(tools) == 1 and tools[0].name == "search"
        assert len(resources) == 1 and resources[0].uri == "file:///x"


class TestChatWithToolsRouting:
    """``chat_with_tools`` requires a real LLM (ReAct loop calls into the
    model) so we only verify the message-resolution path here. The actual
    ReAct dispatch is covered by existing react/agent tests.
    """

    def test_requires_user_message(self):
        llm = _make_unloaded_llm()
        with pytest.raises(ValueError, match="user message"):
            llm.chat_with_tools([{"role": "system", "content": "be brief"}])

    def test_merges_mcp_tools(self):
        llm = _make_unloaded_llm()

        # Stub out the MCP client so we can observe the tool merge without
        # actually running ReAct.
        client = MagicMock()
        sentinel_tool = MagicMock(name="mcp_tool")
        client.get_tools_for_agent.return_value = [sentinel_tool]
        llm._mcp_client = client

        with patch("cyllama.agents.react.ReActAgent") as MockAgent:
            MockAgent.return_value.run.return_value = MagicMock(answer="done")
            answer = llm.chat_with_tools(
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ],
                tools=[MagicMock(name="caller_tool")],
            )
            assert answer == "done"

            kwargs = MockAgent.call_args.kwargs
            assert sentinel_tool in kwargs["tools"]
            assert len(kwargs["tools"]) == 2
            assert kwargs["system_prompt"] == "sys"
            MockAgent.return_value.run.assert_called_once_with("hi")

    def test_use_mcp_false_excludes_mcp_tools(self):
        llm = _make_unloaded_llm()
        client = MagicMock()
        client.get_tools_for_agent.return_value = [MagicMock(name="mcp_tool")]
        llm._mcp_client = client

        with patch("cyllama.agents.react.ReActAgent") as MockAgent:
            MockAgent.return_value.run.return_value = MagicMock(answer="ok")
            llm.chat_with_tools(
                [{"role": "user", "content": "hi"}],
                tools=[],
                use_mcp=False,
            )
            kwargs = MockAgent.call_args.kwargs
            assert kwargs["tools"] == []
