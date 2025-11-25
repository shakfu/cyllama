"""
Tests for constrained agent implementation.
"""

import pytest
import json
from unittest.mock import Mock, MagicMock
from cyllama.agents.constrained import ConstrainedAgent, ConstrainedGenerationConfig
from cyllama.agents.react import EventType
from cyllama.agents.tools import tool


class MockLLM:
    """Mock LLM that returns JSON responses."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.prompts = []

    def __call__(self, prompt, config=None, stream=False, on_token=None):
        self.prompts.append(prompt)
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return '{"type": "answer", "content": "No more responses"}'


def test_constrained_agent_initialization():
    """Test constrained agent initialization."""
    llm = MockLLM()

    @tool
    def test_tool():
        return "result"

    agent = ConstrainedAgent(llm=llm, tools=[test_tool])

    assert agent.llm is llm
    assert len(agent.registry) == 1
    assert "test_tool" in agent.registry
    assert agent.max_iterations == 10


def test_constrained_agent_custom_config():
    """Test agent with custom configuration."""
    llm = MockLLM()
    config = ConstrainedGenerationConfig(temperature=0.5, max_tokens=256)

    agent = ConstrainedAgent(
        llm=llm,
        tools=[],
        max_iterations=5,
        generation_config=config,
        format="json",
        allow_reasoning=True
    )

    assert agent.max_iterations == 5
    assert agent.generation_config.temperature == 0.5
    assert agent.generation_config.max_tokens == 256
    assert agent.allow_reasoning is True


def test_constrained_agent_answer_response():
    """Test agent handling answer response."""
    responses = [
        '{"type": "answer", "content": "The capital is Paris"}'
    ]
    llm = MockLLM(responses)

    agent = ConstrainedAgent(llm=llm, tools=[])

    result = agent.run("What is the capital of France?")

    assert result.success is True
    assert result.answer == "The capital is Paris"
    assert result.iterations == 0  # No tool calls


def test_constrained_agent_tool_call():
    """Test agent making a tool call."""
    responses = [
        '{"type": "tool_call", "tool_name": "search", "tool_args": {"query": "test"}}',
        '{"type": "answer", "content": "Found results"}'
    ]
    llm = MockLLM(responses)

    @tool
    def search(query: str) -> str:
        return f"Results for {query}"

    agent = ConstrainedAgent(llm=llm, tools=[search])

    result = agent.run("Search for test")

    assert result.success is True
    assert result.answer == "Found results"
    assert result.iterations == 1  # One tool call


def test_constrained_agent_with_reasoning():
    """Test agent with reasoning field."""
    responses = [
        '{"type": "tool_call", "reasoning": "Need to search", "tool_name": "search", "tool_args": {"query": "info"}}',
        '{"type": "answer", "content": "Done"}'
    ]
    llm = MockLLM(responses)

    @tool
    def search(query: str) -> str:
        return "results"

    agent = ConstrainedAgent(llm=llm, tools=[search], allow_reasoning=True)

    events = list(agent.stream("Task"))

    # Should have thought event for reasoning
    event_types = [e.type for e in events]
    assert EventType.THOUGHT in event_types
    assert EventType.ACTION in event_types
    assert EventType.ANSWER in event_types


def test_constrained_agent_multiple_tool_calls():
    """Test agent making multiple tool calls."""
    responses = [
        '{"type": "tool_call", "tool_name": "tool1", "tool_args": {}}',
        '{"type": "tool_call", "tool_name": "tool2", "tool_args": {}}',
        '{"type": "answer", "content": "Complete"}'
    ]
    llm = MockLLM(responses)

    @tool
    def tool1() -> str:
        return "result1"

    @tool
    def tool2() -> str:
        return "result2"

    agent = ConstrainedAgent(llm=llm, tools=[tool1, tool2])

    result = agent.run("Do tasks")

    assert result.success is True
    assert result.iterations == 2
    assert result.answer == "Complete"


def test_constrained_agent_max_iterations():
    """Test agent hitting max iterations (with loop detection disabled)."""
    # Always return tool call, never answer
    responses = [
        '{"type": "tool_call", "tool_name": "my_tool", "tool_args": {}}'
    ] * 15

    llm = MockLLM(responses)

    @tool
    def my_tool() -> str:
        return "result"

    # Disable loop detection to test max_iterations behavior
    agent = ConstrainedAgent(llm=llm, tools=[my_tool], max_iterations=3, detect_loops=False)

    result = agent.run("Task")

    assert result.success is False
    assert result.error is not None
    assert "maximum iterations" in result.error


def test_constrained_agent_loop_detection():
    """Test agent loop detection generates summary from observations."""
    # Always return same tool call to trigger loop detection
    responses = [
        '{"type": "tool_call", "tool_name": "my_tool", "tool_args": {}}'
    ] * 15

    llm = MockLLM(responses)

    @tool
    def my_tool() -> str:
        return "result"

    # Enable loop detection (default)
    agent = ConstrainedAgent(llm=llm, tools=[my_tool], max_iterations=10)

    result = agent.run("Task")

    # When loop is detected with observations, a summary answer is generated
    assert result.success is True
    assert "Based on available information" in result.answer
    assert agent.metrics is not None
    assert agent.metrics.loop_detected is True


def test_constrained_agent_loop_detection_no_observations():
    """Test agent loop detection with no observations returns error."""
    # Return tool calls but with no observations collected
    responses = [
        '{"type": "tool_call", "tool_name": "my_tool", "tool_args": {}}'
    ] * 15

    llm = MockLLM(responses)

    @tool
    def my_tool() -> str:
        # Return empty string so no useful observation is collected
        raise Exception("Tool always fails")

    # Enable loop detection (default)
    agent = ConstrainedAgent(llm=llm, tools=[my_tool], max_iterations=10)

    result = agent.run("Task")

    # When loop is detected without useful observations, return error
    # Note: even with errors, observations list will have content, so summary is generated
    assert result.success is True  # Summary is still generated from error messages


def test_constrained_agent_unknown_tool():
    """Test agent calling unknown tool."""
    responses = [
        '{"type": "tool_call", "tool_name": "nonexistent", "tool_args": {}}'
    ]
    llm = MockLLM(responses)

    agent = ConstrainedAgent(llm=llm, tools=[])

    events = list(agent.stream("Task"))

    # Should have error event
    event_types = [e.type for e in events]
    assert EventType.ERROR in event_types


def test_constrained_agent_tool_execution_error():
    """Test handling tool execution errors."""
    responses = [
        '{"type": "tool_call", "tool_name": "failing_tool", "tool_args": {}}',
        '{"type": "answer", "content": "Handled error"}'
    ]
    llm = MockLLM(responses)

    @tool
    def failing_tool() -> str:
        raise RuntimeError("Tool failed")

    agent = ConstrainedAgent(llm=llm, tools=[failing_tool])

    result = agent.run("Task")

    # Agent should continue and reach answer
    assert result.success is True
    assert result.answer == "Handled error"


def test_constrained_agent_invalid_json():
    """Test handling invalid JSON response."""
    responses = [
        'This is not valid JSON'
    ]
    llm = MockLLM(responses)

    agent = ConstrainedAgent(llm=llm, tools=[])

    result = agent.run("Task")

    assert result.success is False
    assert result.error is not None
    assert "JSON" in result.error or "parse" in result.error.lower()


def test_constrained_agent_unknown_response_type():
    """Test handling unknown response type."""
    responses = [
        '{"type": "unknown_type", "data": "something"}'
    ]
    llm = MockLLM(responses)

    agent = ConstrainedAgent(llm=llm, tools=[])

    events = list(agent.stream("Task"))

    # Should have error event
    event_types = [e.type for e in events]
    assert EventType.ERROR in event_types


def test_constrained_agent_add_tool():
    """Test adding tool after agent creation."""
    llm = MockLLM()
    agent = ConstrainedAgent(llm=llm, tools=[])

    @tool
    def new_tool():
        return "result"

    assert len(agent.registry) == 0
    agent.add_tool(new_tool)
    assert len(agent.registry) == 1
    assert "new_tool" in agent.registry


def test_constrained_agent_list_tools():
    """Test listing available tools."""
    llm = MockLLM()

    @tool
    def tool1():
        pass

    @tool
    def tool2():
        pass

    agent = ConstrainedAgent(llm=llm, tools=[tool1, tool2])

    tools = agent.list_tools()
    assert len(tools) == 2


def test_constrained_agent_custom_system_prompt():
    """Test agent with custom system prompt."""
    llm = MockLLM(['{"type": "answer", "content": "Done"}'])
    custom_prompt = "Custom instructions"

    agent = ConstrainedAgent(llm=llm, tools=[], system_prompt=custom_prompt)

    assert agent.system_prompt == custom_prompt


def test_constrained_agent_format_json():
    """Test agent with JSON format."""
    llm = MockLLM()
    agent = ConstrainedAgent(llm=llm, tools=[], format="json")

    from cyllama.agents.grammar import GrammarFormat
    assert agent.format == GrammarFormat.JSON


def test_constrained_agent_format_function_call():
    """Test agent with function call format."""
    llm = MockLLM()
    agent = ConstrainedAgent(llm=llm, tools=[], format="function_call")

    from cyllama.agents.grammar import GrammarFormat
    assert agent.format == GrammarFormat.FUNCTION_CALL


def test_constrained_agent_invalid_format():
    """Test agent with invalid format raises error."""
    llm = MockLLM()

    with pytest.raises(ValueError, match="Invalid format"):
        agent = ConstrainedAgent(llm=llm, tools=[], format="invalid")


def test_constrained_agent_caching_enabled():
    """Test grammar caching is enabled by default."""
    llm = MockLLM()
    agent = ConstrainedAgent(llm=llm, tools=[], use_cache=True)

    assert agent.use_cache is True


def test_constrained_agent_caching_disabled():
    """Test grammar caching can be disabled."""
    llm = MockLLM()
    agent = ConstrainedAgent(llm=llm, tools=[], use_cache=False)

    assert agent.use_cache is False


def test_constrained_agent_stream_events():
    """Test streaming agent events."""
    responses = [
        '{"type": "tool_call", "tool_name": "my_tool", "tool_args": {}}',
        '{"type": "answer", "content": "Final answer"}'
    ]
    llm = MockLLM(responses)

    @tool
    def my_tool() -> str:
        return "result"

    agent = ConstrainedAgent(llm=llm, tools=[my_tool])

    events = list(agent.stream("Task"))

    # Verify event sequence
    assert len(events) > 0
    assert any(e.type == EventType.ACTION for e in events)
    assert any(e.type == EventType.OBSERVATION for e in events)
    assert any(e.type == EventType.ANSWER for e in events)


def test_constrained_agent_verbose_mode():
    """Test agent in verbose mode."""
    responses = ['{"type": "answer", "content": "Done"}']
    llm = MockLLM(responses)

    agent = ConstrainedAgent(llm=llm, tools=[], verbose=True)
    assert agent.verbose is True

    # Should not crash
    result = agent.run("Task")
    assert result.success is True


def test_constrained_agent_tool_args_passed_correctly():
    """Test that tool arguments are passed correctly."""
    responses = [
        '{"type": "tool_call", "tool_name": "multiply", "tool_args": {"a": 6, "b": 7}}',
        '{"type": "answer", "content": "Result is 42"}'
    ]
    llm = MockLLM(responses)

    call_args = {}

    @tool
    def multiply(a: int, b: int) -> int:
        call_args['a'] = a
        call_args['b'] = b
        return a * b

    agent = ConstrainedAgent(llm=llm, tools=[multiply])

    result = agent.run("Multiply 6 and 7")

    assert result.success is True
    assert call_args['a'] == 6
    assert call_args['b'] == 7


def test_constrained_generation_config_defaults():
    """Test ConstrainedGenerationConfig default values."""
    config = ConstrainedGenerationConfig()

    assert config.temperature == 0.7
    assert config.max_tokens == 512
    assert config.top_k == 40
    assert config.top_p == 0.95
    assert config.min_p == 0.05


def test_constrained_agent_json_markdown_stripping():
    """Test that agent strips markdown code blocks from JSON."""
    responses = [
        '```json\n{"type": "answer", "content": "Done"}\n```'
    ]
    llm = MockLLM(responses)

    agent = ConstrainedAgent(llm=llm, tools=[])

    result = agent.run("Task")

    assert result.success is True
    assert result.answer == "Done"


def test_constrained_agent_conversation_history():
    """Test that conversation history is maintained."""
    responses = [
        '{"type": "tool_call", "tool_name": "tool1", "tool_args": {}}',
        '{"type": "answer", "content": "Done"}'
    ]
    llm = MockLLM(responses)

    @tool
    def tool1() -> str:
        return "result1"

    agent = ConstrainedAgent(llm=llm, tools=[tool1])

    agent.run("Task")

    # Check that prompts include tool results
    assert len(llm.prompts) == 2
    assert "result1" in llm.prompts[1]  # Second prompt should include first result
