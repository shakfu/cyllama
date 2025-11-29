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


# =============================================================================
# Type Hint Error Handling Tests
# =============================================================================

class TestTypeHintErrorHandling:
    """Test graceful handling of type hint errors."""

    def test_forward_reference_fallback(self):
        """Test that forward references fall back gracefully."""
        # This would fail with get_type_hints() if ForwardRef isn't defined
        @tool
        def func_with_string_annotation(x: "UndefinedType") -> str:
            """Has forward reference"""
            return str(x)

        # Should still create a tool with string type fallback
        assert isinstance(func_with_string_annotation, Tool)
        schema = func_with_string_annotation.parameters
        # Should have the parameter, possibly as string type
        assert "x" in schema["properties"]

    def test_no_type_hints_works(self):
        """Test that functions without type hints still work."""
        @tool
        def no_hints_func(a, b, c):
            """No type hints at all"""
            return a + b + c

        assert isinstance(no_hints_func, Tool)
        schema = no_hints_func.parameters
        assert len(schema["properties"]) == 3
        # All should default to string
        assert all(p["type"] == "string" for p in schema["properties"].values())

    def test_partial_type_hints(self):
        """Test function with partial type hints."""
        @tool
        def partial_hints(a: int, b, c: str):
            """Some hints missing"""
            return f"{a}{b}{c}"

        schema = partial_hints.parameters
        props = schema["properties"]

        assert props["a"]["type"] == "integer"
        assert props["b"]["type"] == "string"  # default
        assert props["c"]["type"] == "string"


# =============================================================================
# Generic Type Support Tests
# =============================================================================

class TestGenericTypeSupport:
    """Test support for generic types in schemas."""

    def test_list_of_strings(self):
        """Test List[str] type."""
        from typing import List

        @tool
        def func(items: List[str]) -> int:
            """List of strings"""
            return len(items)

        schema = func.parameters
        props = schema["properties"]

        assert props["items"]["type"] == "array"
        assert props["items"]["items"]["type"] == "string"

    def test_list_of_ints(self):
        """Test List[int] type."""
        from typing import List

        @tool
        def func(numbers: List[int]) -> int:
            """List of ints"""
            return sum(numbers)

        schema = func.parameters
        props = schema["properties"]

        assert props["numbers"]["type"] == "array"
        assert props["numbers"]["items"]["type"] == "integer"

    def test_dict_str_int(self):
        """Test Dict[str, int] type."""
        from typing import Dict

        @tool
        def func(mapping: Dict[str, int]) -> int:
            """Dict mapping"""
            return sum(mapping.values())

        schema = func.parameters
        props = schema["properties"]

        assert props["mapping"]["type"] == "object"
        assert props["mapping"]["additionalProperties"]["type"] == "integer"

    def test_optional_type(self):
        """Test Optional[str] type."""
        from typing import Optional

        @tool
        def func(name: Optional[str] = None) -> str:
            """Optional param"""
            return name or "default"

        schema = func.parameters
        props = schema["properties"]

        assert props["name"]["type"] == "string"
        assert props["name"]["nullable"] is True
        assert "name" not in schema["required"]

    def test_union_type(self):
        """Test Union[str, int] type."""
        from typing import Union

        @tool
        def func(value: Union[str, int]) -> str:
            """Union type"""
            return str(value)

        schema = func.parameters
        props = schema["properties"]

        # Should have anyOf
        assert "anyOf" in props["value"]
        types = [s["type"] for s in props["value"]["anyOf"]]
        assert "string" in types
        assert "integer" in types

    def test_nested_generic(self):
        """Test nested generic like List[Dict[str, int]]."""
        from typing import List, Dict

        @tool
        def func(data: List[Dict[str, int]]) -> int:
            """Nested generic"""
            return len(data)

        schema = func.parameters
        props = schema["properties"]

        assert props["data"]["type"] == "array"
        assert props["data"]["items"]["type"] == "object"
        assert props["data"]["items"]["additionalProperties"]["type"] == "integer"

    def test_tuple_type(self):
        """Test Tuple type."""
        from typing import Tuple

        @tool
        def func(point: Tuple[int, int]) -> int:
            """Tuple type"""
            return point[0] + point[1]

        schema = func.parameters
        props = schema["properties"]

        assert props["point"]["type"] == "array"
        assert "prefixItems" in props["point"]
        assert len(props["point"]["prefixItems"]) == 2
        assert props["point"]["minItems"] == 2
        assert props["point"]["maxItems"] == 2

    def test_set_type(self):
        """Test Set type."""
        from typing import Set

        @tool
        def func(unique_items: Set[str]) -> int:
            """Set type"""
            return len(unique_items)

        schema = func.parameters
        props = schema["properties"]

        assert props["unique_items"]["type"] == "array"
        assert props["unique_items"]["uniqueItems"] is True
        assert props["unique_items"]["items"]["type"] == "string"

    def test_literal_type(self):
        """Test Literal type."""
        from typing import Literal

        @tool
        def func(mode: Literal["read", "write", "append"]) -> str:
            """Literal type"""
            return mode

        schema = func.parameters
        props = schema["properties"]

        assert props["mode"]["type"] == "string"
        assert "enum" in props["mode"]
        assert set(props["mode"]["enum"]) == {"read", "write", "append"}

    def test_bytes_type(self):
        """Test bytes type."""
        @tool
        def func(data: bytes) -> int:
            """Bytes type"""
            return len(data)

        schema = func.parameters
        props = schema["properties"]

        assert props["data"]["type"] == "string"
        assert props["data"]["contentEncoding"] == "base64"


# =============================================================================
# Docstring Parsing Tests
# =============================================================================

class TestDocstringParsing:
    """Test docstring parsing for multiple formats."""

    def test_google_style_simple(self):
        """Test simple Google-style docstring."""
        @tool
        def func(query: str, limit: int) -> list:
            """
            Search for items.

            Args:
                query: The search query string
                limit: Maximum number of results

            Returns:
                List of results
            """
            return []

        schema = func.parameters
        props = schema["properties"]

        assert props["query"]["description"] == "The search query string"
        assert props["limit"]["description"] == "Maximum number of results"

    def test_google_style_with_types(self):
        """Test Google-style with type annotations in docstring."""
        @tool
        def func(name: str, count: int) -> str:
            """
            Process items.

            Args:
                name (str): The name to process
                count (int): How many times

            Returns:
                str: Processed result
            """
            return name * count

        schema = func.parameters
        props = schema["properties"]

        assert props["name"]["description"] == "The name to process"
        assert props["count"]["description"] == "How many times"

    def test_google_style_multiline(self):
        """Test Google-style with multi-line descriptions."""
        @tool
        def func(data: str) -> str:
            """
            Process data.

            Args:
                data: The data to process. This is a longer
                    description that spans multiple lines
                    for more detail.

            Returns:
                Processed data
            """
            return data

        schema = func.parameters
        props = schema["properties"]

        desc = props["data"]["description"]
        assert "data to process" in desc
        assert "multiple lines" in desc

    def test_numpy_style(self):
        """Test NumPy-style docstring."""
        @tool
        def func(x: float, y: float) -> float:
            """
            Calculate distance.

            Parameters
            ----------
            x : float
                The x coordinate
            y : float
                The y coordinate

            Returns
            -------
            float
                The distance
            """
            return (x**2 + y**2)**0.5

        schema = func.parameters
        props = schema["properties"]

        assert props["x"]["description"] == "The x coordinate"
        assert props["y"]["description"] == "The y coordinate"

    def test_sphinx_style(self):
        """Test Sphinx/reST-style docstring."""
        @tool
        def func(path: str, mode: str) -> str:
            """
            Open a file.

            :param path: Path to the file
            :param mode: Mode to open in
            :returns: File contents
            """
            return ""

        schema = func.parameters
        props = schema["properties"]

        assert props["path"]["description"] == "Path to the file"
        assert props["mode"]["description"] == "Mode to open in"

    def test_epytext_style(self):
        """Test Epytext-style docstring."""
        @tool
        def func(url: str, timeout: int) -> str:
            """
            Fetch a URL.

            @param url: The URL to fetch
            @param timeout: Request timeout in seconds
            @return: Response content
            """
            return ""

        schema = func.parameters
        props = schema["properties"]

        assert props["url"]["description"] == "The URL to fetch"
        assert props["timeout"]["description"] == "Request timeout in seconds"

    def test_no_docstring(self):
        """Test tool without docstring."""
        @tool
        def func(x: int) -> int:
            return x * 2

        schema = func.parameters
        props = schema["properties"]

        # Should not have description
        assert "description" not in props["x"]

    def test_docstring_no_params_section(self):
        """Test docstring without parameter section."""
        @tool
        def func(x: int) -> int:
            """Just a simple function that doubles."""
            return x * 2

        schema = func.parameters
        props = schema["properties"]

        # Should not have description
        assert "description" not in props["x"]


class TestPromptGeneration:
    """Test prompt string generation with new type info."""

    def test_prompt_includes_nested_types(self):
        """Test that prompt includes nested type information."""
        from typing import List, Dict

        @tool
        def func(items: List[str], mapping: Dict[str, int]) -> str:
            """Process items and mapping"""
            return ""

        prompt = func.to_prompt_string()

        assert "func" in prompt
        assert "items" in prompt
        assert "mapping" in prompt
        assert "array" in prompt
        assert "object" in prompt

    def test_prompt_includes_optional(self):
        """Test that prompt shows optional parameters."""
        from typing import Optional

        @tool
        def func(required: str, optional: Optional[str] = None) -> str:
            """Has optional param"""
            return ""

        prompt = func.to_prompt_string()

        assert "(optional)" in prompt
        # required param should not have (optional)
        assert "required" in prompt
