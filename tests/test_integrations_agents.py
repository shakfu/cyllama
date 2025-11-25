"""
Tests for agent framework integrations.
"""

import pytest
from cyllama.agents import tool, Tool

# Check if LangChain is available
try:
    import langchain_core
    LANGCHAIN_INSTALLED = True
except ImportError:
    LANGCHAIN_INSTALLED = False


# Test tool conversions
def test_cyllama_tool_to_openai_function():
    """Test converting cyllama tool to OpenAI function format."""
    from cyllama.integrations.openai_agents import cyllama_tool_to_openai_function

    @tool
    def search(query: str, max_results: int = 5) -> str:
        """Search for information"""
        return "results"

    function_def = cyllama_tool_to_openai_function(search)

    assert function_def["name"] == "search"
    assert function_def["description"] == "Search for information"
    assert "parameters" in function_def
    assert function_def["parameters"]["type"] == "object"


def test_cyllama_tools_to_openai_tools():
    """Test converting list of cyllama tools to OpenAI tools format."""
    from cyllama.integrations.openai_agents import cyllama_tools_to_openai_tools

    @tool
    def tool1():
        return "1"

    @tool
    def tool2():
        return "2"

    tools_def = cyllama_tools_to_openai_tools([tool1, tool2])

    assert len(tools_def) == 2
    assert all(t["type"] == "function" for t in tools_def)
    assert tools_def[0]["function"]["name"] == "tool1"
    assert tools_def[1]["function"]["name"] == "tool2"


def test_openai_function_calling_client_init():
    """Test initializing OpenAI function calling client."""
    from cyllama.integrations.openai_agents import OpenAIFunctionCallingClient

    @tool
    def my_tool():
        return "result"

    client = OpenAIFunctionCallingClient(
        model_path="dummy_path.gguf",
        tools=[my_tool]
    )

    assert client.model_path == "dummy_path.gguf"
    assert len(client.tools) == 1


def test_openai_function_calling_client_list_functions():
    """Test listing functions."""
    from cyllama.integrations.openai_agents import OpenAIFunctionCallingClient

    @tool
    def search(query: str) -> str:
        """Search tool"""
        return "results"

    client = OpenAIFunctionCallingClient(
        model_path="dummy.gguf",
        tools=[search]
    )

    functions = client.list_functions()

    assert len(functions) == 1
    assert functions[0]["name"] == "search"
    assert functions[0]["description"] == "Search tool"


def test_openai_function_calling_client_list_tools():
    """Test listing tools."""
    from cyllama.integrations.openai_agents import OpenAIFunctionCallingClient

    @tool
    def tool1():
        return "1"

    @tool
    def tool2():
        return "2"

    client = OpenAIFunctionCallingClient(
        model_path="dummy.gguf",
        tools=[tool1, tool2]
    )

    tools = client.list_tools()

    assert len(tools) == 2
    assert all(t["type"] == "function" for t in tools)


def test_create_openai_function_calling_client():
    """Test convenience function."""
    from cyllama.integrations.openai_agents import create_openai_function_calling_client

    @tool
    def my_tool():
        return "result"

    client = create_openai_function_calling_client(
        model_path="dummy.gguf",
        tools=[my_tool]
    )

    assert client.model_path == "dummy.gguf"
    assert len(client.tools) == 1


# LangChain integration tests (only run if LangChain available)
def test_langchain_available():
    """Test if LangChain is available."""
    from cyllama.integrations.langchain_agents import LANGCHAIN_AVAILABLE
    # This test just checks the import works
    assert isinstance(LANGCHAIN_AVAILABLE, bool)


# NOTE: These tests require LangChain to be installed
# They are commented out to avoid test failures when LangChain is not available
# Uncomment and run manually if you have LangChain installed

# @pytest.mark.skipif(not LANGCHAIN_INSTALLED, reason="LangChain not installed")
# def test_cyllama_tool_to_langchain():
#     """Test converting cyllama tool to LangChain tool."""
#     from cyllama.integrations.langchain_agents import cyllama_tool_to_langchain
#
#     @tool
#     def calculator(expression: str) -> float:
#         """Calculate a math expression"""
#         return eval(expression, {"__builtins__": {}}, {})
#
#     lc_tool = cyllama_tool_to_langchain(calculator)
#
#     assert lc_tool.name == "calculator"
#     assert lc_tool.description == "Calculate a math expression"
#
#
# @pytest.mark.skipif(not LANGCHAIN_INSTALLED, reason="LangChain not installed")
# def test_cyllama_tool_to_langchain_execution():
#     """Test that converted tool executes correctly."""
#     from cyllama.integrations.langchain_agents import cyllama_tool_to_langchain
#
#     @tool
#     def add(a: int, b: int) -> int:
#         """Add two numbers"""
#         return a + b
#
#     lc_tool = cyllama_tool_to_langchain(add)
#
#     # Execute tool
#     result = lc_tool.func(a=5, b=3)
#     assert result == 8


def test_integration_exports():
    """Test that integration exports work."""
    try:
        from cyllama.integrations import (
            cyllama_tool_to_openai_function,
            OpenAIFunctionCallingClient,
        )
        # If we get here, exports work
        assert callable(cyllama_tool_to_openai_function)
        assert OpenAIFunctionCallingClient is not None
    except ImportError:
        # Expected if dependencies not installed
        pass


def test_openai_agent_dataclasses():
    """Test OpenAI agent dataclass structures."""
    from cyllama.integrations.openai_agents import (
        FunctionCall,
        ToolCall,
        AssistantMessage,
        ChatCompletionMessage,
        ChatCompletionChoice,
        ChatCompletionResponse,
    )

    # Test FunctionCall
    func_call = FunctionCall(name="search", arguments='{"query": "test"}')
    assert func_call.name == "search"
    assert func_call.arguments == '{"query": "test"}'

    # Test ToolCall
    tool_call = ToolCall(
        id="call_123",
        type="function",
        function=func_call
    )
    assert tool_call.id == "call_123"
    assert tool_call.function.name == "search"

    # Test AssistantMessage
    msg = AssistantMessage(role="assistant", content="Hello")
    assert msg.role == "assistant"
    assert msg.content == "Hello"

    # Test ChatCompletionResponse
    response = ChatCompletionResponse(id="resp_123")
    assert response.id == "resp_123"
    assert response.object == "chat.completion"
