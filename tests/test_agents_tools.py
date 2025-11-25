"""
Tests for agent tool registry and tool definitions.
"""

import pytest
from cyllama.agents.tools import Tool, tool, ToolRegistry, _python_type_to_json_type


def test_tool_decorator_basic():
    """Test basic tool decoration."""
    @tool
    def simple_func(x: str) -> str:
        """A simple function"""
        return f"Result: {x}"

    assert isinstance(simple_func, Tool)
    assert simple_func.name == "simple_func"
    assert simple_func.description == "A simple function"
    assert callable(simple_func)


def test_tool_decorator_with_custom_name():
    """Test tool decorator with custom name."""
    @tool(name="custom_name", description="Custom description")
    def my_func():
        """Original doc"""
        return "result"

    assert my_func.name == "custom_name"
    assert my_func.description == "Custom description"


def test_tool_execution():
    """Test that tool can be called."""
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    result = add(a=5, b=3)
    assert result == 8


def test_tool_schema_generation():
    """Test automatic schema generation from function signature."""
    @tool
    def search(query: str, max_results: int = 5) -> list:
        """Search for something"""
        return []

    schema = search.parameters

    # Check structure
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "required" in schema

    # Check properties
    props = schema["properties"]
    assert "query" in props
    assert "max_results" in props

    # Check types
    assert props["query"]["type"] == "string"
    assert props["max_results"]["type"] == "integer"

    # Check required
    assert "query" in schema["required"]
    assert "max_results" not in schema["required"]  # has default


def test_tool_with_google_docstring():
    """Test parameter description extraction from docstring."""
    @tool
    def function_with_docs(param1: str, param2: int) -> str:
        """
        A function with documentation.

        Args:
            param1: First parameter description
            param2: Second parameter description

        Returns:
            A result string
        """
        return f"{param1}-{param2}"

    schema = function_with_docs.parameters
    props = schema["properties"]

    assert props["param1"]["description"] == "First parameter description"
    assert props["param2"]["description"] == "Second parameter description"


def test_tool_to_prompt_string():
    """Test prompt string generation."""
    @tool
    def my_tool(arg1: str, arg2: int = 10) -> str:
        """Does something useful"""
        return ""

    prompt_str = my_tool.to_prompt_string()

    assert "my_tool" in prompt_str
    assert "Does something useful" in prompt_str
    assert "arg1" in prompt_str
    assert "arg2" in prompt_str
    assert "string" in prompt_str
    assert "integer" in prompt_str
    assert "(optional)" in prompt_str  # arg2 has default


def test_tool_to_json_schema():
    """Test JSON schema export."""
    @tool
    def search(query: str) -> list:
        """Search the web"""
        return []

    schema = search.to_json_schema()

    assert schema["name"] == "search"
    assert schema["description"] == "Search the web"
    assert "parameters" in schema
    assert schema["parameters"]["type"] == "object"


def test_tool_registry_register():
    """Test registering tools in registry."""
    registry = ToolRegistry()

    @tool
    def tool1():
        return 1

    @tool
    def tool2():
        return 2

    registry.register(tool1)
    registry.register(tool2)

    assert len(registry) == 2
    assert "tool1" in registry
    assert "tool2" in registry


def test_tool_registry_duplicate_registration():
    """Test that duplicate registration raises error."""
    registry = ToolRegistry()

    @tool
    def my_tool():
        return 1

    registry.register(my_tool)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(my_tool)


def test_tool_registry_get():
    """Test retrieving tools from registry."""
    registry = ToolRegistry()

    @tool
    def my_tool():
        return 42

    registry.register(my_tool)

    retrieved = registry.get("my_tool")
    assert retrieved is not None
    assert retrieved.name == "my_tool"
    assert retrieved() == 42


def test_tool_registry_get_nonexistent():
    """Test getting non-existent tool returns None."""
    registry = ToolRegistry()
    result = registry.get("nonexistent")
    assert result is None


def test_tool_registry_list_tools():
    """Test listing all tools."""
    registry = ToolRegistry()

    @tool
    def tool1():
        pass

    @tool
    def tool2():
        pass

    registry.register(tool1)
    registry.register(tool2)

    tools = registry.list_tools()
    assert len(tools) == 2
    assert tool1 in tools
    assert tool2 in tools


def test_tool_registry_to_prompt_string():
    """Test generating prompt string for all tools."""
    registry = ToolRegistry()

    @tool
    def search(query: str) -> list:
        """Search the web"""
        return []

    @tool
    def calculate(expression: str) -> float:
        """Evaluate a math expression"""
        return 0.0

    registry.register(search)
    registry.register(calculate)

    prompt = registry.to_prompt_string()

    assert "search" in prompt
    assert "Search the web" in prompt
    assert "calculate" in prompt
    assert "Evaluate a math expression" in prompt


def test_tool_registry_to_json_schema():
    """Test JSON schema export for all tools."""
    registry = ToolRegistry()

    @tool
    def tool1(x: str):
        pass

    @tool
    def tool2(y: int):
        pass

    registry.register(tool1)
    registry.register(tool2)

    schemas = registry.to_json_schema()

    assert len(schemas) == 2
    assert schemas[0]["name"] == "tool1"
    assert schemas[1]["name"] == "tool2"


def test_tool_registry_iteration():
    """Test iterating over registry."""
    registry = ToolRegistry()

    @tool
    def tool1():
        pass

    @tool
    def tool2():
        pass

    registry.register(tool1)
    registry.register(tool2)

    tools_from_iter = list(registry)
    assert len(tools_from_iter) == 2


def test_python_type_to_json_type():
    """Test type conversion."""
    assert _python_type_to_json_type(str) == "string"
    assert _python_type_to_json_type(int) == "integer"
    assert _python_type_to_json_type(float) == "number"
    assert _python_type_to_json_type(bool) == "boolean"
    assert _python_type_to_json_type(list) == "array"
    assert _python_type_to_json_type(dict) == "object"


def test_tool_complex_types():
    """Test tool with complex type hints."""
    from typing import List, Dict

    @tool
    def complex_func(items: List[str], mapping: Dict[str, int]) -> str:
        """Process complex types"""
        return "done"

    schema = complex_func.parameters
    props = schema["properties"]

    assert props["items"]["type"] == "array"
    assert props["mapping"]["type"] == "object"


def test_tool_no_parameters():
    """Test tool with no parameters."""
    @tool
    def no_params() -> str:
        """A tool with no parameters"""
        return "result"

    schema = no_params.parameters

    # Should still have proper structure
    assert schema["type"] == "object"
    assert len(schema["properties"]) == 0
    assert len(schema["required"]) == 0


def test_tool_without_type_hints():
    """Test tool without type hints defaults to string."""
    @tool
    def no_hints(x, y):
        """No type hints"""
        return x + y

    schema = no_hints.parameters
    props = schema["properties"]

    # Should default to string
    assert props["x"]["type"] == "string"
    assert props["y"]["type"] == "string"


def test_tool_with_return_value():
    """Test tool execution returns correct value."""
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b

    result = multiply(a=6, b=7)
    assert result == 42


def test_empty_registry_prompt_string():
    """Test prompt string for empty registry."""
    registry = ToolRegistry()
    prompt = registry.to_prompt_string()
    assert "No tools available" in prompt


def test_registry_contains():
    """Test __contains__ operator."""
    registry = ToolRegistry()

    @tool
    def my_tool():
        pass

    assert "my_tool" not in registry
    registry.register(my_tool)
    assert "my_tool" in registry
