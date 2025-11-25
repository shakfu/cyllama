"""
Tests for agent grammar generation.
"""

import pytest
import json
from cyllama.agents.grammar import (
    GrammarFormat,
    generate_tool_call_schema,
    generate_tool_call_grammar,
    generate_answer_or_tool_schema,
    generate_answer_or_tool_grammar,
    generate_specific_tool_schema,
    generate_specific_tool_grammar,
    get_cached_tool_grammar,
    clear_grammar_cache,
    GrammarCache,
)
from cyllama.agents.tools import tool


def test_generate_json_tool_schema():
    """Test JSON format tool schema generation."""
    @tool
    def search(query: str, max_results: int = 5) -> str:
        """Search for information"""
        return "results"

    @tool
    def calculate(expression: str) -> float:
        """Calculate a math expression"""
        return 0.0

    schema = generate_tool_call_schema(
        [search, calculate],
        allow_reasoning=True,
        format=GrammarFormat.JSON
    )

    # Check structure
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "tool_name" in schema["properties"]
    assert "tool_args" in schema["properties"]
    assert "reasoning" in schema["properties"]  # allow_reasoning=True

    # Check tool name enum
    tool_names = schema["properties"]["tool_name"]["enum"]
    assert "search" in tool_names
    assert "calculate" in tool_names

    # Check required fields
    assert "tool_name" in schema["required"]
    assert "tool_args" in schema["required"]


def test_generate_json_tool_schema_no_reasoning():
    """Test schema without reasoning field."""
    @tool
    def my_tool():
        return "result"

    schema = generate_tool_call_schema(
        [my_tool],
        allow_reasoning=False,
        format=GrammarFormat.JSON
    )

    assert "reasoning" not in schema["properties"]


def test_generate_json_array_tool_schema():
    """Test JSON array format for multiple tool calls."""
    @tool
    def tool1():
        return "1"

    @tool
    def tool2():
        return "2"

    schema = generate_tool_call_schema(
        [tool1, tool2],
        allow_reasoning=True,
        format=GrammarFormat.JSON_ARRAY
    )

    assert "tool_calls" in schema["properties"]
    assert schema["properties"]["tool_calls"]["type"] == "array"
    assert "minItems" in schema["properties"]["tool_calls"]


def test_generate_function_call_schema():
    """Test OpenAI-style function call schema."""
    @tool
    def my_function(arg: str) -> str:
        return arg

    schema = generate_tool_call_schema(
        [my_function],
        allow_reasoning=False,
        format=GrammarFormat.FUNCTION_CALL
    )

    assert "name" in schema["properties"]
    assert "arguments" in schema["properties"]
    assert schema["properties"]["arguments"]["type"] == "string"


def test_generate_tool_call_grammar():
    """Test grammar generation from tools."""
    @tool
    def search(query: str) -> str:
        return "results"

    grammar = generate_tool_call_grammar(
        [search],
        allow_reasoning=True,
        format=GrammarFormat.JSON
    )

    # Should return non-empty grammar string
    assert isinstance(grammar, str)
    assert len(grammar) > 0


def test_generate_answer_or_tool_schema():
    """Test schema for answer OR tool call."""
    @tool
    def my_tool(x: str) -> str:
        return x

    schema = generate_answer_or_tool_schema([my_tool], allow_reasoning=True)

    assert schema["type"] == "object"
    assert "type" in schema["properties"]
    assert schema["properties"]["type"]["enum"] == ["answer", "tool_call"]
    assert "oneOf" in schema


def test_generate_answer_or_tool_grammar():
    """Test grammar for answer or tool call."""
    @tool
    def calculator(expr: str) -> float:
        return 0.0

    grammar = generate_answer_or_tool_grammar([calculator])

    assert isinstance(grammar, str)
    assert len(grammar) > 0


def test_generate_specific_tool_schema():
    """Test schema for specific tool with argument validation."""
    @tool
    def multiply(a: int, b: int) -> int:
        """
        Multiply two numbers.

        Args:
            a: First number
            b: Second number
        """
        return a * b

    schema = generate_specific_tool_schema(multiply)

    assert schema["properties"]["tool_name"]["const"] == "multiply"
    assert "tool_args" in schema["properties"]

    # Check that tool parameters are included
    tool_args_schema = schema["properties"]["tool_args"]
    assert "properties" in tool_args_schema
    assert "a" in tool_args_schema["properties"]
    assert "b" in tool_args_schema["properties"]


def test_generate_specific_tool_grammar():
    """Test grammar for specific tool."""
    @tool
    def my_tool(param: str) -> str:
        return param

    grammar = generate_specific_tool_grammar(my_tool)

    assert isinstance(grammar, str)
    assert len(grammar) > 0


def test_grammar_cache_basic():
    """Test basic grammar caching."""
    cache = GrammarCache()

    call_count = 0

    def generator():
        nonlocal call_count
        call_count += 1
        return "grammar_string"

    # First call generates
    result1 = cache.get_or_create("key1", generator)
    assert result1 == "grammar_string"
    assert call_count == 1

    # Second call with same key uses cache
    result2 = cache.get_or_create("key1", generator)
    assert result2 == "grammar_string"
    assert call_count == 1  # Not called again

    # Different key generates new
    result3 = cache.get_or_create("key2", generator)
    assert result3 == "grammar_string"
    assert call_count == 2


def test_grammar_cache_clear():
    """Test clearing grammar cache."""
    cache = GrammarCache()

    cache.get_or_create("key", lambda: "value")
    assert len(cache) == 1

    cache.clear()
    assert len(cache) == 0


def test_get_cached_tool_grammar():
    """Test cached tool grammar generation."""
    @tool
    def search(query: str) -> str:
        return "results"

    # Clear cache first
    clear_grammar_cache()

    # First call generates and caches
    grammar1 = get_cached_tool_grammar(
        [search],
        allow_reasoning=True,
        format=GrammarFormat.JSON
    )

    assert isinstance(grammar1, str)
    assert len(grammar1) > 0

    # Second call should return same grammar (cached)
    grammar2 = get_cached_tool_grammar(
        [search],
        allow_reasoning=True,
        format=GrammarFormat.JSON
    )

    assert grammar1 == grammar2


def test_cached_grammar_different_settings():
    """Test that different settings create different cache entries."""
    @tool
    def my_tool():
        return "result"

    clear_grammar_cache()

    # Different reasoning settings
    grammar1 = get_cached_tool_grammar([my_tool], allow_reasoning=True)
    grammar2 = get_cached_tool_grammar([my_tool], allow_reasoning=False)

    # Should be different grammars
    assert grammar1 != grammar2


def test_cached_grammar_different_tools():
    """Test that different tool sets create different grammars."""
    @tool
    def tool1():
        return "1"

    @tool
    def tool2():
        return "2"

    clear_grammar_cache()

    grammar1 = get_cached_tool_grammar([tool1])
    grammar2 = get_cached_tool_grammar([tool1, tool2])

    # Should be different grammars
    assert grammar1 != grammar2


def test_invalid_format_raises_error():
    """Test that invalid format raises ValueError."""
    @tool
    def my_tool():
        return "result"

    # Invalid format enum
    with pytest.raises(ValueError):
        schema = generate_tool_call_schema([my_tool], format="invalid_format")


def test_grammar_with_empty_tools():
    """Test grammar generation with no tools."""
    schema = generate_tool_call_schema(
        [],
        allow_reasoning=True,
        format=GrammarFormat.JSON
    )

    # Should still have valid structure
    assert "tool_name" in schema["properties"]
    assert schema["properties"]["tool_name"]["enum"] == []


def test_multiple_tools_schema():
    """Test schema with multiple diverse tools."""
    @tool
    def search(query: str, limit: int = 10) -> list:
        """Search with limit"""
        return []

    @tool
    def calculate(expr: str) -> float:
        """Calculate expression"""
        return 0.0

    @tool
    def read_file(path: str) -> str:
        """Read file contents"""
        return ""

    schema = generate_tool_call_schema(
        [search, calculate, read_file],
        allow_reasoning=True,
        format=GrammarFormat.JSON
    )

    tool_names = schema["properties"]["tool_name"]["enum"]
    assert len(tool_names) == 3
    assert all(name in tool_names for name in ["search", "calculate", "read_file"])


def test_grammar_format_enum():
    """Test GrammarFormat enum values."""
    assert GrammarFormat.JSON.value == "json"
    assert GrammarFormat.JSON_ARRAY.value == "json_array"
    assert GrammarFormat.FUNCTION_CALL.value == "function_call"


def test_schema_is_valid_json():
    """Test that generated schemas are valid JSON-serializable."""
    @tool
    def my_tool(x: str, y: int = 5) -> str:
        return x

    schema = generate_tool_call_schema([my_tool])

    # Should be JSON-serializable
    json_str = json.dumps(schema)
    assert len(json_str) > 0

    # Should round-trip correctly
    parsed = json.loads(json_str)
    assert parsed == schema
