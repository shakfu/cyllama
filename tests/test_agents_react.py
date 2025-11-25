"""
Tests for ReAct agent implementation.
"""

import pytest
from unittest.mock import Mock, MagicMock
from cyllama.agents.react import ReActAgent, EventType, AgentEvent, AgentResult
from cyllama.agents.tools import tool
from cyllama.api import LLM, GenerationConfig


class MockLLM:
    """Mock LLM for testing without actual model."""

    def __init__(self, responses=None):
        """
        Args:
            responses: List of responses to return in sequence
        """
        self.responses = responses or []
        self.call_count = 0
        self.prompts = []

    def __call__(self, prompt, config=None, stream=False, on_token=None):
        """Return next response in sequence."""
        self.prompts.append(prompt)
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return "Answer: No more responses"


def test_react_agent_initialization():
    """Test ReAct agent initialization."""
    llm = MockLLM()

    @tool
    def test_tool():
        return "result"

    agent = ReActAgent(llm=llm, tools=[test_tool])

    assert agent.llm is llm
    assert len(agent.registry) == 1
    assert "test_tool" in agent.registry
    assert agent.max_iterations == 10


def test_react_agent_custom_config():
    """Test agent with custom configuration."""
    llm = MockLLM()
    config = GenerationConfig(temperature=0.5, max_tokens=256)

    agent = ReActAgent(
        llm=llm,
        tools=[],
        max_iterations=5,
        generation_config=config
    )

    assert agent.max_iterations == 5
    assert agent.generation_config.temperature == 0.5
    assert agent.generation_config.max_tokens == 256


def test_extract_thought():
    """Test thought extraction from response."""
    llm = MockLLM()
    agent = ReActAgent(llm=llm, tools=[])

    text = "Thought: I need to search for information\nAction: search(query='test')"
    thought = agent._extract_thought(text)

    assert thought == "I need to search for information"


def test_extract_action():
    """Test action extraction from response."""
    llm = MockLLM()
    agent = ReActAgent(llm=llm, tools=[])

    text = "Thought: I need to search\nAction: search(query='test')"
    action = agent._extract_action(text)

    assert action == "search(query='test')"


def test_extract_answer():
    """Test answer extraction from response."""
    llm = MockLLM()
    agent = ReActAgent(llm=llm, tools=[])

    text = "Thought: I now know the answer\nAnswer: The capital of France is Paris"
    answer = agent._extract_answer(text)

    assert answer == "The capital of France is Paris"


def test_parse_action_with_kwargs():
    """Test parsing action with keyword arguments."""
    llm = MockLLM()
    agent = ReActAgent(llm=llm, tools=[])

    action_str = 'search(query="python programming", max_results="5")'
    tool_name, args = agent._parse_action(action_str)

    assert tool_name == "search"
    assert args["query"] == "python programming"
    assert args["max_results"] == "5"


def test_parse_action_with_single_quotes():
    """Test parsing action with single quotes."""
    llm = MockLLM()
    agent = ReActAgent(llm=llm, tools=[])

    action_str = "search(query='test query')"
    tool_name, args = agent._parse_action(action_str)

    assert tool_name == "search"
    assert args["query"] == "test query"


def test_parse_action_no_args():
    """Test parsing action with no arguments."""
    llm = MockLLM()
    agent = ReActAgent(llm=llm, tools=[])

    action_str = "get_time()"
    tool_name, args = agent._parse_action(action_str)

    assert tool_name == "get_time"
    assert args == {}


def test_parse_action_invalid_format():
    """Test parsing invalid action format raises error."""
    llm = MockLLM()
    agent = ReActAgent(llm=llm, tools=[])

    with pytest.raises(ValueError, match="Invalid action format"):
        agent._parse_action("not a function call")


def test_execute_tool():
    """Test tool execution."""
    llm = MockLLM()

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return int(a) * int(b)

    agent = ReActAgent(llm=llm, tools=[multiply])

    result = agent._execute_tool("multiply", {"a": "6", "b": "7"})
    assert result == "42"


def test_execute_unknown_tool():
    """Test executing unknown tool raises error."""
    llm = MockLLM()
    agent = ReActAgent(llm=llm, tools=[])

    with pytest.raises(ValueError, match="Unknown tool"):
        agent._execute_tool("nonexistent", {})


def test_execute_tool_with_error():
    """Test tool execution error handling."""
    llm = MockLLM()

    @tool
    def failing_tool():
        """A tool that fails"""
        raise RuntimeError("Tool failed")

    agent = ReActAgent(llm=llm, tools=[failing_tool])

    result = agent._execute_tool("failing_tool", {})
    assert "Tool execution error" in result
    assert "Tool failed" in result


def test_agent_stream_simple():
    """Test agent streaming with simple task."""
    responses = [
        "Thought: I need to add two numbers\nAction: add(a='5', b='3')",
        "Thought: I now have the answer\nAnswer: The result is 8"
    ]
    llm = MockLLM(responses)

    @tool
    def add(a: str, b: str) -> int:
        """Add two numbers"""
        return int(a) + int(b)

    agent = ReActAgent(llm=llm, tools=[add])

    events = list(agent.stream("What is 5 + 3?"))

    # Check event types
    event_types = [e.type for e in events]
    assert EventType.THOUGHT in event_types
    assert EventType.ACTION in event_types
    assert EventType.OBSERVATION in event_types
    assert EventType.ANSWER in event_types


def test_agent_run_success():
    """Test successful agent run."""
    responses = [
        "Thought: I need to search\nAction: search(query='test')",
        "Thought: I found the answer\nAnswer: The search was successful"
    ]
    llm = MockLLM(responses)

    @tool
    def search(query: str) -> str:
        """Search for information"""
        return f"Results for {query}"

    agent = ReActAgent(llm=llm, tools=[search])

    result = agent.run("Find information about test")

    assert result.success is True
    assert result.answer == "The search was successful"
    assert result.iterations == 2
    assert result.error is None


def test_agent_run_max_iterations():
    """Test agent hitting max iterations (with loop detection disabled)."""
    # Return responses that never include Answer
    responses = [
        "Thought: Still working\nAction: search(query='test')"
    ] * 15  # More than max_iterations

    llm = MockLLM(responses)

    @tool
    def search(query: str) -> str:
        return "result"

    # Disable loop detection to test max_iterations behavior
    agent = ReActAgent(llm=llm, tools=[search], max_iterations=3, detect_loops=False)

    result = agent.run("A task")

    assert result.success is False
    assert result.error is not None
    assert "maximum iterations" in result.error


def test_agent_loop_detection():
    """Test agent loop detection generates summary from observations."""
    # Return same response repeatedly to trigger loop detection
    responses = [
        "Thought: Still working\nAction: search(query='test')"
    ] * 15

    llm = MockLLM(responses)

    @tool
    def search(query: str) -> str:
        return "result"

    # Enable loop detection (default)
    agent = ReActAgent(llm=llm, tools=[search], max_iterations=10)

    result = agent.run("A task")

    # When loop is detected with observations, a summary answer is generated
    assert result.success is True
    assert "Based on available information" in result.answer
    assert agent.metrics is not None
    assert agent.metrics.loop_detected is True


def test_agent_add_tool():
    """Test adding tool after agent creation."""
    llm = MockLLM()
    agent = ReActAgent(llm=llm, tools=[])

    @tool
    def new_tool():
        return "result"

    assert len(agent.registry) == 0
    agent.add_tool(new_tool)
    assert len(agent.registry) == 1
    assert "new_tool" in agent.registry


def test_agent_list_tools():
    """Test listing available tools."""
    llm = MockLLM()

    @tool
    def tool1():
        pass

    @tool
    def tool2():
        pass

    agent = ReActAgent(llm=llm, tools=[tool1, tool2])

    tools = agent.list_tools()
    assert len(tools) == 2


def test_agent_custom_system_prompt():
    """Test agent with custom system prompt."""
    llm = MockLLM(["Answer: Done"])
    custom_prompt = "Custom instructions here"

    agent = ReActAgent(llm=llm, tools=[], system_prompt=custom_prompt)

    assert agent.system_prompt == custom_prompt


def test_agent_default_system_prompt():
    """Test agent generates default system prompt with tools."""
    llm = MockLLM(["Answer: Done"])

    @tool
    def my_tool(x: str) -> str:
        """Does something"""
        return x

    agent = ReActAgent(llm=llm, tools=[my_tool])

    assert "my_tool" in agent.system_prompt
    assert "Does something" in agent.system_prompt


def test_agent_event_structure():
    """Test AgentEvent structure."""
    event = AgentEvent(
        type=EventType.THOUGHT,
        content="I need to think",
        metadata={"iteration": 1}
    )

    assert event.type == EventType.THOUGHT
    assert event.content == "I need to think"
    assert event.metadata["iteration"] == 1


def test_agent_result_structure():
    """Test AgentResult structure."""
    events = [
        AgentEvent(type=EventType.THOUGHT, content="Thinking"),
        AgentEvent(type=EventType.ANSWER, content="Final answer")
    ]

    result = AgentResult(
        answer="Final answer",
        steps=events,
        iterations=1,
        success=True
    )

    assert result.answer == "Final answer"
    assert len(result.steps) == 2
    assert result.iterations == 1
    assert result.success is True
    assert result.error is None


def test_parse_action_json_format():
    """Test parsing action with JSON object format."""
    llm = MockLLM()
    agent = ReActAgent(llm=llm, tools=[])

    action_str = 'search({"query": "test", "max_results": 5})'
    tool_name, args = agent._parse_action(action_str)

    assert tool_name == "search"
    assert args["query"] == "test"
    assert args["max_results"] == 5


def test_parse_action_positional_arg():
    """Test parsing action with single positional argument."""
    llm = MockLLM()

    @tool
    def search(query: str) -> str:
        return "result"

    agent = ReActAgent(llm=llm, tools=[search])

    # Single value without key
    action_str = 'search("test query")'
    tool_name, args = agent._parse_action(action_str)

    assert tool_name == "search"
    assert args["query"] == "test query"


def test_agent_verbose_mode():
    """Test agent in verbose mode."""
    responses = ["Answer: Done"]
    llm = MockLLM(responses)

    agent = ReActAgent(llm=llm, tools=[], verbose=True)
    assert agent.verbose is True

    # Just verify it doesn't crash
    result = agent.run("Test task")
    assert result.success is True


def test_agent_handles_malformed_tool_call():
    """Test agent handles malformed tool call gracefully."""
    responses = [
        "Thought: Let me try\nAction: broken_format_here",
        "Answer: I give up"
    ]
    llm = MockLLM(responses)

    @tool
    def my_tool():
        return "result"

    agent = ReActAgent(llm=llm, tools=[my_tool])

    events = list(agent.stream("Do something"))

    # Should have error event
    event_types = [e.type for e in events]
    assert EventType.ERROR in event_types


def test_agent_tool_execution_error_continues():
    """Test agent continues after tool execution error."""
    responses = [
        "Thought: Try the tool\nAction: failing_tool()",
        "Thought: That failed, but I know the answer\nAnswer: I figured it out anyway"
    ]
    llm = MockLLM(responses)

    @tool
    def failing_tool():
        """A tool that always fails"""
        raise ValueError("This tool always fails")

    agent = ReActAgent(llm=llm, tools=[failing_tool])

    result = agent.run("Try to do something")

    # Agent should still succeed with final answer
    assert result.success is True
    assert result.answer == "I figured it out anyway"


def test_multiple_actions_before_answer():
    """Test agent performing multiple actions."""
    responses = [
        "Thought: First step\nAction: tool1()",
        "Thought: Second step\nAction: tool2()",
        "Thought: Third step\nAction: tool3()",
        "Thought: Done\nAnswer: All steps complete"
    ]
    llm = MockLLM(responses)

    @tool
    def tool1():
        return "result1"

    @tool
    def tool2():
        return "result2"

    @tool
    def tool3():
        return "result3"

    agent = ReActAgent(llm=llm, tools=[tool1, tool2, tool3])

    result = agent.run("Do multiple things")

    assert result.success is True
    assert result.iterations == 4

    # Count action events
    actions = [e for e in result.steps if e.type == EventType.ACTION]
    assert len(actions) == 3


def test_extract_action_double_prefix():
    """Test extracting action with double Action: prefix."""
    llm = MockLLM()
    agent = ReActAgent(llm=llm, tools=[])

    # Model sometimes outputs "Action: Action: tool(...)"
    text = "Thought: I need to search\nAction: Action: search(query='test')"
    action = agent._extract_action(text)

    # Should strip the duplicate prefix
    assert action == "search(query='test')"


def test_extract_action_triple_prefix():
    """Test extracting action with triple Action: prefix."""
    llm = MockLLM()
    agent = ReActAgent(llm=llm, tools=[])

    text = "Thought: Testing\nAction: Action: Action: my_tool()"
    action = agent._extract_action(text)

    assert action == "my_tool()"
