"""ACP agent: message shapes follow the ACP schema (protocol version 1).

https://agentclientprotocol.com/protocol/schema
"""

import json
import os
import threading

import pytest
from unittest.mock import Mock

from cyllama.agents.acp import (
    ACP_PROTOCOL_VERSION,
    ACPAgent,
    ContentBlock,
    SessionUpdate,
    StopReason,
    ToolCallStatus,
    ToolCallUpdate,
    _prompt_text,
)
from cyllama.agents.jsonrpc import JsonRpcError, JsonRpcResponse
from cyllama.agents.session import MemorySessionStore, Session
from cyllama.agents.types import AgentEvent, EventType

ALL_CLIENT_CAPS = {"fs": {"readTextFile": True, "writeTextFile": True}, "terminal": True}


class TestContentBlock:
    def test_text_content(self):
        block = ContentBlock.from_text("Hello")
        assert block.type == "text" and block.text == "Hello"

    def test_to_dict_text(self):
        assert ContentBlock.from_text("Hello").to_dict() == {"type": "text", "text": "Hello"}

    def test_to_dict_with_data(self):
        block = ContentBlock(type="image", data="base64data", mime_type="image/png")
        assert block.to_dict() == {"type": "image", "data": "base64data", "mimeType": "image/png"}


class TestUpdateShapes:
    def test_text_update(self):
        u = SessionUpdate.text("s1", "agent_message_chunk", "hi")
        assert u.to_dict() == {
            "sessionId": "s1",
            "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "hi"}},
        }

    def test_tool_call_and_update(self):
        call = ToolCallUpdate("call_1", ToolCallStatus.IN_PROGRESS, title="search", kind="other", raw_input={"q": 1})
        first = SessionUpdate.tool_call("s1", call, first=True).to_dict()["update"]
        assert first == {
            "sessionUpdate": "tool_call",
            "toolCallId": "call_1",
            "status": "in_progress",
            "title": "search",
            "kind": "other",
            "rawInput": {"q": 1},
        }
        done = ToolCallUpdate("call_1", ToolCallStatus.COMPLETED, content=[ContentBlock.from_text("ok")])
        later = SessionUpdate.tool_call("s1", done, first=False).to_dict()["update"]
        assert later["sessionUpdate"] == "tool_call_update"
        assert later["content"] == [{"type": "content", "content": {"type": "text", "text": "ok"}}]

    def test_stop_reasons_match_schema(self):
        assert {r.value for r in StopReason} == {"end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"}


class TestPromptText:
    def test_text_and_resources(self):
        blocks = [
            {"type": "text", "text": "Explain this"},
            {"type": "resource", "resource": {"uri": "file:///a.py", "mimeType": "text/x-python", "text": "x = 1"}},
            {"type": "resource_link", "uri": "file:///b.py", "name": "b.py"},
            {"type": "image", "data": "..."},
            "not a block",
        ]
        text = _prompt_text(blocks)
        assert "Explain this" in text and "x = 1" in text and "file:///b.py" in text


@pytest.fixture
def agent():
    llm = Mock()
    llm.return_value = "Answer: test response"
    return ACPAgent(llm=llm, verbose=False)


class TestHandlers:
    def test_initialize(self, agent):
        result = agent._handle_initialize({"protocolVersion": 1, "clientInfo": {"name": "test", "version": "1.0"}})
        assert result["protocolVersion"] == ACP_PROTOCOL_VERSION == 1
        import cyllama

        assert result["agentInfo"] == {"name": "cyllama", "version": cyllama.__version__}  # was a stale "0.2.0"
        caps = result["agentCapabilities"]
        assert caps["loadSession"] is True
        assert caps["promptCapabilities"] == {"image": False, "audio": False, "embeddedContext": True}
        assert result["agentInfo"]["name"] == "cyllama"
        assert result["authMethods"] == []

    def test_initialize_records_client_capabilities(self, agent):
        assert agent._create_acp_tools() == []  # nothing advertised yet
        agent._handle_initialize({"clientCapabilities": {"fs": {"readTextFile": True}}})
        assert [t.name for t in agent._create_acp_tools()] == ["read_file"]
        agent._handle_initialize({"clientCapabilities": ALL_CLIENT_CAPS})
        assert [t.name for t in agent._create_acp_tools()] == ["read_file", "write_file", "edit_file", "run_command"]
        agent._handle_initialize({"clientCapabilities": {"fs": {"writeTextFile": True}}})
        assert [t.name for t in agent._create_acp_tools()] == ["write_file"]  # edit_file must read first

    def test_authenticate(self, agent):
        assert agent._handle_authenticate({}) == {}

    def test_session_new(self, agent):
        result = agent._handle_session_new({"cwd": "/work", "mcpServers": []})
        assert set(result) == {"sessionId"}
        session = agent._session_store.load(result["sessionId"])
        assert session.metadata["cwd"] == "/work"

    def test_session_new_skips_non_stdio_mcp(self, agent):
        agent._handle_session_new({"cwd": "/w", "mcpServers": [{"type": "http", "name": "h", "url": "http://x"}]})
        assert agent._mcp_client is None

    def test_session_load_replays_history(self, agent):
        session = Session(id="s1")
        session.add_message("user", "hi")
        session.add_message("assistant", "hello")
        agent._session_store.save(session)
        server = Mock()
        agent._server = server
        assert agent._handle_session_load({"sessionId": "s1", "cwd": "/w", "mcpServers": []}) == {}
        sent = [c.args[1]["update"] for c in server.send_notification.call_args_list]
        assert [u["sessionUpdate"] for u in sent] == ["user_message_chunk", "agent_message_chunk"]
        assert sent[1]["content"]["text"] == "hello"

    def test_session_load_not_found(self, agent):
        with pytest.raises(ValueError, match="Session not found"):
            agent._handle_session_load({"sessionId": "nonexistent"})

    def test_session_load_missing_id(self, agent):
        with pytest.raises(ValueError, match="sessionId is required"):
            agent._handle_session_load({})

    def test_session_set_mode(self, agent):
        agent._session_store.save(Session(id="mode_test"))
        assert agent._handle_session_set_mode({"sessionId": "mode_test", "modeId": "default"}) == {}
        assert agent._session_store.load("mode_test").mode_id == "default"

    def test_session_cancel_aborts_generation(self, agent):
        agent._current_session_id = "cancel_test"
        agent._handle_session_cancel({"sessionId": "cancel_test"})
        assert agent._cancelled is True
        agent.llm.cancel.assert_called_once()

    def test_session_cancel_wrong_session(self, agent):
        agent._current_session_id = "active"
        agent._handle_session_cancel({"sessionId": "other"})
        assert agent._cancelled is False

    def test_session_store_types(self):
        assert isinstance(ACPAgent(llm=Mock(), session_storage="memory")._session_store, MemorySessionStore)


class TestPromptTurn:
    def _run(self, agent, events, prompt="next question"):
        tasks = []

        class Inner:
            def stream(self, task):
                tasks.append(task)
                yield from events

        agent._create_inner_agent = lambda session: Inner()
        agent._server = Mock()
        session = Session(id="s1")
        session.add_message("user", "first question")
        session.add_message("assistant", "first answer")
        agent._session_store.save(session)
        result = agent._handle_session_prompt({"sessionId": "s1", "prompt": [{"type": "text", "text": prompt}]})
        updates = [c.args[1]["update"] for c in agent._server.send_notification.call_args_list]
        return result, updates, tasks[0]

    def test_prompt_field_and_history(self, agent):
        result, updates, task = self._run(agent, [AgentEvent(type=EventType.ANSWER, content="done")])
        assert result == {"stopReason": "end_turn"}
        assert updates[-1] == {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "done"}}
        # earlier turns reach the inner agent
        assert "first question" in task and "first answer" in task and task.endswith("next question")

    def test_tool_call_ids_link_call_and_result(self, agent):
        events = [
            AgentEvent(type=EventType.ACTION, content="search(q)", metadata={"tool_name": "search", "tool_args": {}}),
            AgentEvent(type=EventType.OBSERVATION, content="found", metadata={"tool_name": "search"}),
            AgentEvent(type=EventType.ANSWER, content="done"),
        ]
        _, updates, _ = self._run(agent, events)
        call, result = updates[0], updates[1]
        assert call["sessionUpdate"] == "tool_call" and result["sessionUpdate"] == "tool_call_update"
        assert call["toolCallId"] == result["toolCallId"]
        assert result["status"] == "completed"

    def test_max_iterations_maps_to_max_turn_requests(self, agent):
        events = [AgentEvent(type=EventType.ERROR, content="Reached maximum iterations (10)")]
        result, _, _ = self._run(agent, events)
        assert result == {"stopReason": "max_turn_requests"}

    def test_tool_error_does_not_end_turn(self, agent):
        # ReAct recovers from a parse error; ending the turn there showed the error as the reply
        events = [
            AgentEvent(type=EventType.ACTION, content="write_file(", metadata={"tool_name": None, "tool_args": {}}),
            AgentEvent(
                type=EventType.ERROR, content="Missing closing parenthesis", metadata={"error_type": "parse_error"}
            ),
            AgentEvent(type=EventType.OBSERVATION, content="Error: Missing closing parenthesis", metadata={}),
            AgentEvent(type=EventType.ANSWER, content="done"),
        ]
        result, updates, _ = self._run(agent, events)
        assert result == {"stopReason": "end_turn"}
        assert [u["sessionUpdate"] for u in updates] == ["tool_call", "tool_call_update", "agent_message_chunk"]
        assert updates[1]["status"] == "failed"
        assert updates[2]["content"]["text"] == "done"

    def test_inner_react_step_budget(self, agent):
        # 512 tokens cut a whole-file write_file off mid-call
        inner = agent._create_inner_agent(Session(id="s9"))
        assert inner.generation_config.max_tokens == ACPAgent.STEP_MAX_TOKENS == 2048

    def test_missing_prompt_rejected(self, agent):
        agent._session_store.save(Session(id="s2"))
        with pytest.raises(ValueError, match="No text"):
            agent._handle_session_prompt({"sessionId": "s2", "prompt": []})


class TestEventToUpdate:
    @pytest.mark.parametrize(
        "event_type,kind",
        [
            (EventType.THOUGHT, "agent_thought_chunk"),
            (EventType.ANSWER, "agent_message_chunk"),
            (EventType.ERROR, "agent_message_chunk"),
        ],
    )
    def test_text_events(self, agent, event_type, kind):
        update = agent._event_to_update("s1", AgentEvent(type=event_type, content="x"))
        assert update.to_dict()["update"]["sessionUpdate"] == kind

    @pytest.mark.parametrize("content,title", [('write_file({"path": "a', "write_file"), ("", "tool call")])
    def test_unparsed_action_has_a_title(self, agent, content, title):
        # tool_name is None when the action failed to parse; Zed dropped a tool_call with a null title
        event = AgentEvent(type=EventType.ACTION, content=content, metadata={"tool_name": None, "tool_args": {}})
        assert agent._event_to_update("s1", event, "c1").to_dict()["update"]["title"] == title

    def test_failed_observation(self, agent):
        update = agent._event_to_update("s1", AgentEvent(type=EventType.OBSERVATION, content="Error: no"), "c1")
        assert update.to_dict()["update"]["status"] == "failed"


class TestToolPermission:
    """write_file and run_command ask the editor before acting (session/request_permission)."""

    def _agent(self, outcome):
        agent = ACPAgent(llm=Mock())
        agent._handle_initialize({"clientCapabilities": ALL_CLIENT_CAPS})
        server = Mock()
        sent = []
        self.requests = []

        def send_request(method, params=None, timeout=None):
            sent.append(method)
            self.requests.append((method, params))
            if method == "session/request_permission":
                return JsonRpcResponse(id=1, result={"outcome": outcome})
            if method == "terminal/create":
                return JsonRpcResponse(id=2, result={"terminalId": "t"})
            if method == "terminal/output":
                return JsonRpcResponse(id=3, result={"output": "ran", "truncated": False})
            if method == "fs/read_text_file":
                if params["path"].endswith("new.txt"):  # the editor reports a missing file as an error
                    return JsonRpcResponse(id=5, error=JsonRpcError(code=-32002, message="not found"))
                if params["path"].endswith("slow.txt"):
                    raise TimeoutError
                return JsonRpcResponse(id=5, result={"content": "file text"})
            return JsonRpcResponse(id=4, result={})

        server.send_request.side_effect = send_request
        agent._server = server
        agent._session_store.save(Session(id="s1", metadata={"cwd": "/work"}))
        agent._current_session_id = "s1"
        tools = {t.name: t for t in agent._create_acp_tools()}
        return tools, sent

    def _params(self, method):
        return next(p for m, p in self.requests if m == method)

    @pytest.mark.parametrize(
        "name,args",
        [("run_command", {"command": "rm -rf x"}), ("write_file", {"path": "new.txt", "content": "b"})],
    )
    def test_denied_does_not_act(self, name, args):
        tools, sent = self._agent({"outcome": "selected", "optionId": "reject_once"})
        assert "permission denied" in tools[name](**args)
        assert sent[-1] == "session/request_permission"
        assert "fs/write_text_file" not in sent and "terminal/create" not in sent

    @pytest.mark.parametrize(
        "name,args,effect",
        [
            ("run_command", {"command": "ls"}, "terminal/create"),
            ("write_file", {"path": "new.txt", "content": "b"}, "fs/write_text_file"),
        ],
    )
    def test_allowed_acts(self, name, args, effect):
        tools, sent = self._agent({"outcome": "selected", "optionId": "allow_once"})
        tools[name](**args)
        assert sent.index("session/request_permission") < sent.index(effect)

    @pytest.mark.parametrize("path,error", [("a.txt", "exists; use edit_file"), ("slow.txt", "could not check")])
    def test_write_refuses_existing_or_unchecked(self, path, error):
        # Qwen3-4B replaced a 579-line README through write_file; existing files go through edit_file
        tools, sent = self._agent({"outcome": "selected", "optionId": "allow_once"})
        assert error in tools["write_file"](path=path, content="x")
        assert sent == ["fs/read_text_file"]  # no prompt, no write

    def test_write_without_read_capability_is_unchecked(self):
        agent = ACPAgent(llm=Mock())
        agent._handle_initialize({"clientCapabilities": {"fs": {"writeTextFile": True}}})
        agent._server = Mock()
        agent._server.send_request.return_value = JsonRpcResponse(
            id=1, result={"outcome": {"outcome": "selected", "optionId": "allow_once"}}
        )
        agent._session_store.save(Session(id="s1", metadata={"cwd": "/work"}))
        agent._current_session_id = "s1"
        (write_file,) = agent._create_acp_tools()
        write_file(path="a.txt", content="x")
        assert [c.args[0] for c in agent._server.send_request.call_args_list] == [
            "session/request_permission",
            "fs/write_text_file",
        ]

    def test_edit_replaces_only_old_text(self):
        tools, sent = self._agent({"outcome": "selected", "optionId": "allow_once"})
        assert tools["edit_file"](path="a.txt", old_text="text", new_text="body") == "Edited a.txt"
        assert sent == ["fs/read_text_file", "session/request_permission", "fs/write_text_file"]
        assert self._params("fs/write_text_file")["content"] == "file body"
        raw = self._params("session/request_permission")["toolCall"]["rawInput"]
        assert raw == {"path": "a.txt", "old_text": "text", "new_text": "body"}

    def test_edit_can_delete(self):
        tools, _ = self._agent({"outcome": "selected", "optionId": "allow_once"})
        tools["edit_file"](path="a.txt", old_text=" text", new_text="")
        assert self._params("fs/write_text_file")["content"] == "file"

    @pytest.mark.parametrize(
        "old_text,error,sent",
        [
            ("missing", "not found", ["fs/read_text_file"]),
            ("t", "occurs 2 times", ["fs/read_text_file"]),  # "file text"
            ("", "old_text is empty", []),
        ],
    )
    def test_edit_that_cannot_apply_asks_nothing(self, old_text, error, sent):
        # whole-file writes lost every line a small model failed to copy; edits change only old_text
        tools, actual = self._agent({"outcome": "selected", "optionId": "allow_once"})
        assert error in tools["edit_file"](path="a.txt", old_text=old_text, new_text="x")
        assert actual == sent

    def test_edit_denied_does_not_write(self):
        tools, sent = self._agent({"outcome": "selected", "optionId": "reject_once"})
        assert "permission denied" in tools["edit_file"](path="a.txt", old_text="text", new_text="body")
        assert "fs/write_text_file" not in sent

    def test_permission_request_matches_schema(self):
        tools, _ = self._agent({"outcome": "selected", "optionId": "allow_once"})
        tools["run_command"](command="ls")
        params = self._params("session/request_permission")
        assert params["toolCall"]["toolCallId"] and params["toolCall"]["kind"] == "execute"
        assert params["toolCall"]["rawInput"] == {"command": "ls"}
        assert all(set(o) == {"optionId", "name", "kind"} for o in params["options"])

    @pytest.mark.parametrize("outcome", [{"optionId": "allow_once"}, {"outcome": "weird", "optionId": "allow_once"}])
    def test_unrecognised_outcome_denies(self, outcome):
        # ACP: agents SHOULD NOT treat an outcome they do not recognise as approval
        tools, _ = self._agent(outcome)
        assert "permission denied" in tools["run_command"](command="ls")

    def test_cancelled_denies(self):
        tools, sent = self._agent({"outcome": "cancelled"})
        assert "permission denied" in tools["run_command"](command="ls")
        assert "terminal/create" not in sent

    def test_fs_requests_match_schema(self):
        tools, _ = self._agent({"outcome": "selected", "optionId": "allow_once"})
        assert tools["read_file"](path="src/a.py") == "file text"
        assert self._params("fs/read_text_file") == {"sessionId": "s1", "path": os.path.normpath("/work/src/a.py")}
        tools["write_file"](path="/abs/new.txt", content="x")  # write_file only creates files
        assert self._params("fs/write_text_file") == {
            "sessionId": "s1",
            "path": os.path.normpath("/abs/new.txt"),
            "content": "x",
        }

    def test_terminal_requests_match_schema_and_release(self):
        tools, sent = self._agent({"outcome": "selected", "optionId": "allow_once"})
        assert tools["run_command"](command="echo hi") == "ran"
        create = self._params("terminal/create")
        assert create["sessionId"] == "s1" and create["cwd"] == "/work" and "echo hi" in create["args"]
        assert sent[-1] == "terminal/release"
        assert self._params("terminal/release") == {"sessionId": "s1", "terminalId": "t"}


def test_end_to_end_over_stdio():
    """initialize, session/new and session/prompt over a JSON-RPC stream; updates precede the response."""
    r_in, w_in = os.pipe()
    r_out, w_out = os.pipe()
    to_agent = os.fdopen(w_in, "w", buffering=1)
    from_agent = os.fdopen(r_out, "r")
    agent = ACPAgent(llm=Mock(return_value="Thought: easy\nAnswer: 42"))
    t = threading.Thread(
        target=agent.serve, args=(os.fdopen(r_in, "r"), os.fdopen(w_out, "w", buffering=1)), daemon=True
    )
    t.start()

    def call(id_, method, params):
        to_agent.write(json.dumps({"jsonrpc": "2.0", "id": id_, "method": method, "params": params}) + "\n")
        msgs = []
        while True:
            msg = json.loads(from_agent.readline())
            msgs.append(msg)
            if msg.get("id") == id_:
                return msgs

    init = call(1, "initialize", {"protocolVersion": 1, "clientCapabilities": {}})[-1]
    assert init["result"]["protocolVersion"] == 1
    sid = call(2, "session/new", {"cwd": "/tmp", "mcpServers": []})[-1]["result"]["sessionId"]
    to_agent.write("\n")  # a blank line is skipped
    to_agent.write("not json\n")  # a bad line gets an error and the stream goes on
    assert json.loads(from_agent.readline())["error"]["code"] == -32700
    msgs = call(3, "session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": "6*7?"}]})
    assert msgs[-1]["result"] == {"stopReason": "end_turn"}
    assert msgs[-2]["params"]["update"] == {
        "sessionUpdate": "agent_message_chunk",
        "content": {"type": "text", "text": "42"},
    }
    to_agent.close()
    t.join(timeout=5)
    assert not t.is_alive()


class TestInnerAgentFlag:
    @pytest.mark.parametrize("argv,expected", [([], "react"), (["--inner-agent", "constrained"], "constrained")])
    def test_cli_passes_inner_agent(self, argv, expected):
        from unittest.mock import patch

        from cyllama.agents import cli

        with (
            patch("sys.argv", ["cyllama-agents", "acp", "-m", "m.gguf", *argv]),
            patch("cyllama.api.LLM"),
            patch("cyllama.agents.acp.ACPAgent") as acp_cls,
        ):
            assert cli.main() == 0
        assert acp_cls.call_args.kwargs["inner_agent_type"] == expected

    def test_constrained_inner_agent_gets_step_budget(self):
        from cyllama.agents.constrained import ConstrainedAgent

        agent = ACPAgent(llm=Mock(), inner_agent_type="constrained")
        inner = agent._create_inner_agent(Session(id="s1"))
        assert isinstance(inner, ConstrainedAgent)
        assert inner.generation_config.max_tokens == ACPAgent.STEP_MAX_TOKENS
