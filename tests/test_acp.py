"""Tests for ACP agent implementation."""

import io
import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from cyllama.agents.acp import (
    ACPAgent,
    ContentBlock,
    ToolCallUpdate,
    ToolCallStatus,
    SessionUpdate,
    StopReason,
    ACP_PROTOCOL_VERSION,
)
from cyllama.agents.session import Session, MemorySessionStore


class TestContentBlock:
    """Tests for ACP content blocks."""

    def test_text_content(self):
        block = ContentBlock.text("Hello world")
        assert block.type == "text"
        assert block.text == "Hello world"

    def test_to_dict_text(self):
        block = ContentBlock.text("test")
        d = block.to_dict()

        assert d["type"] == "text"
        assert d["text"] == "test"
        assert "data" not in d

    def test_to_dict_with_data(self):
        block = ContentBlock(
            type="image",
            data="base64data",
            mime_type="image/png"
        )
        d = block.to_dict()

        assert d["type"] == "image"
        assert d["data"] == "base64data"
        assert d["mimeType"] == "image/png"


class TestToolCallUpdate:
    """Tests for tool call updates."""

    def test_basic_update(self):
        update = ToolCallUpdate(
            id="tc_1",
            name="search",
            status=ToolCallStatus.IN_PROGRESS,
        )
        assert update.id == "tc_1"
        assert update.name == "search"
        assert update.status == ToolCallStatus.IN_PROGRESS

    def test_to_dict(self):
        update = ToolCallUpdate(
            id="tc_1",
            name="calc",
            status=ToolCallStatus.COMPLETED,
            arguments={"x": 1},
            content=[ContentBlock.text("42")],
        )
        d = update.to_dict()

        assert d["id"] == "tc_1"
        assert d["name"] == "calc"
        assert d["status"] == "completed"
        assert d["arguments"] == {"x": 1}
        assert len(d["content"]) == 1


class TestSessionUpdate:
    """Tests for session updates."""

    def test_basic_update(self):
        update = SessionUpdate(
            session_id="sess_1",
            content=[ContentBlock.text("Hello")],
        )
        assert update.session_id == "sess_1"

    def test_to_dict(self):
        update = SessionUpdate(
            session_id="sess_1",
            content=[ContentBlock.text("Done")],
            stop_reason=StopReason.END_TURN,
        )
        d = update.to_dict()

        assert d["sessionId"] == "sess_1"
        assert len(d["content"]) == 1
        assert d["stopReason"] == "end_turn"

    def test_to_dict_with_tool_calls(self):
        update = SessionUpdate(
            session_id="sess_1",
            tool_calls=[
                ToolCallUpdate(
                    id="tc_1",
                    name="test",
                    status=ToolCallStatus.PENDING,
                )
            ],
        )
        d = update.to_dict()

        assert "toolCalls" in d
        assert len(d["toolCalls"]) == 1


class TestACPAgentHandlers:
    """Tests for ACP agent method handlers."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = Mock()
        llm.return_value = "Answer: test response"
        return llm

    @pytest.fixture
    def agent(self, mock_llm):
        """Create an ACP agent for testing."""
        return ACPAgent(llm=mock_llm, verbose=False)

    def test_handle_initialize(self, agent):
        """Test initialize handler."""
        result = agent._handle_initialize({
            "protocolVersion": "2025-01-01",
            "clientInfo": {"name": "test", "version": "1.0"},
        })

        assert "protocolVersion" in result
        assert "capabilities" in result
        assert "agentInfo" in result
        assert result["agentInfo"]["name"] == "cyllama"

    def test_handle_initialize_capabilities(self, agent):
        """Test that initialize returns correct capabilities."""
        result = agent._handle_initialize({})

        caps = result["capabilities"]
        assert caps["prompts"]["text"] is True
        assert caps["tools"] is True
        assert caps["permissions"] is True

    def test_handle_authenticate(self, agent):
        """Test authenticate handler."""
        result = agent._handle_authenticate({})
        assert result["authenticated"] is True

    def test_handle_session_new(self, agent):
        """Test session/new handler."""
        result = agent._handle_session_new({})

        assert "sessionId" in result
        assert len(result["sessionId"]) > 0

        # Verify session was stored
        session = agent._session_store.load(result["sessionId"])
        assert session is not None

    def test_handle_session_new_with_mode(self, agent):
        """Test session/new with mode."""
        result = agent._handle_session_new({"modeId": "default"})

        session = agent._session_store.load(result["sessionId"])
        assert session.mode_id == "default"

    def test_handle_session_load(self, agent):
        """Test session/load handler."""
        # First create a session
        session = Session(id="load_test")
        agent._session_store.save(session)

        # Then load it
        result = agent._handle_session_load({"sessionId": "load_test"})
        assert result["sessionId"] == "load_test"

    def test_handle_session_load_not_found(self, agent):
        """Test session/load with nonexistent session."""
        with pytest.raises(ValueError, match="Session not found"):
            agent._handle_session_load({"sessionId": "nonexistent"})

    def test_handle_session_load_missing_id(self, agent):
        """Test session/load without sessionId."""
        with pytest.raises(ValueError, match="sessionId is required"):
            agent._handle_session_load({})

    def test_handle_session_set_mode(self, agent):
        """Test session/set_mode handler."""
        # Create session
        session = Session(id="mode_test")
        agent._session_store.save(session)

        result = agent._handle_session_set_mode({
            "sessionId": "mode_test",
            "modeId": "default",
        })

        assert "mode" in result

        # Verify mode was updated
        updated = agent._session_store.load("mode_test")
        assert updated.mode_id == "default"

    def test_handle_session_cancel(self, agent):
        """Test session/cancel notification."""
        agent._current_session_id = "cancel_test"
        agent._cancelled = False

        agent._handle_session_cancel({"sessionId": "cancel_test"})
        assert agent._cancelled is True

    def test_handle_session_cancel_wrong_session(self, agent):
        """Test session/cancel for different session."""
        agent._current_session_id = "active"
        agent._cancelled = False

        agent._handle_session_cancel({"sessionId": "other"})
        assert agent._cancelled is False  # Should not cancel


class TestACPAgentIntegration:
    """Integration tests for ACP agent."""

    def test_session_store_types(self):
        """Test different session storage types."""
        mock_llm = Mock()

        # Memory store (default)
        agent1 = ACPAgent(llm=mock_llm, session_storage="memory")
        assert isinstance(agent1._session_store, MemorySessionStore)

    def test_create_acp_tools(self):
        """Test that ACP tools are created."""
        mock_llm = Mock()
        agent = ACPAgent(llm=mock_llm)

        tools = agent._create_acp_tools()

        tool_names = [t.name for t in tools]
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "run_command" in tool_names


class TestEventToUpdate:
    """Tests for converting AgentEvents to ACP updates."""

    @pytest.fixture
    def agent(self):
        mock_llm = Mock()
        return ACPAgent(llm=mock_llm)

    def test_thought_event(self, agent):
        from cyllama.agents.react import AgentEvent, EventType

        event = AgentEvent(type=EventType.THOUGHT, content="I should search")
        update = agent._event_to_update("sess_1", event, 0)

        assert update is not None
        assert update.session_id == "sess_1"
        assert len(update.content) == 1
        assert "Thinking" in update.content[0].text

    def test_action_event(self, agent):
        from cyllama.agents.react import AgentEvent, EventType

        event = AgentEvent(
            type=EventType.ACTION,
            content="search(query='test')",
            metadata={"tool_name": "search", "tool_args": {"query": "test"}}
        )
        update = agent._event_to_update("sess_1", event, 0)

        assert update is not None
        assert len(update.tool_calls) == 1
        assert update.tool_calls[0].name == "search"
        assert update.tool_calls[0].status == ToolCallStatus.IN_PROGRESS

    def test_observation_event(self, agent):
        from cyllama.agents.react import AgentEvent, EventType

        event = AgentEvent(
            type=EventType.OBSERVATION,
            content="Found 5 results",
            metadata={"tool_name": "search"}
        )
        update = agent._event_to_update("sess_1", event, 1)

        assert update is not None
        assert len(update.tool_calls) == 1
        assert update.tool_calls[0].status == ToolCallStatus.COMPLETED

    def test_answer_event(self, agent):
        from cyllama.agents.react import AgentEvent, EventType

        event = AgentEvent(type=EventType.ANSWER, content="The answer is 42")
        update = agent._event_to_update("sess_1", event, 0)

        assert update is not None
        assert update.stop_reason == StopReason.END_TURN
        assert len(update.content) == 1
        assert update.content[0].text == "The answer is 42"

    def test_error_event(self, agent):
        from cyllama.agents.react import AgentEvent, EventType

        event = AgentEvent(type=EventType.ERROR, content="Something went wrong")
        update = agent._event_to_update("sess_1", event, 0)

        assert update is not None
        assert update.stop_reason == StopReason.ERROR
        assert "Error" in update.content[0].text
