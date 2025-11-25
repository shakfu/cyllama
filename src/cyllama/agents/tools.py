"""
Tool registry and definition system for cyllama agents.

Provides a simple, type-safe way to register and invoke tools that agents can use.
"""

import inspect
import json
from typing import Any, Callable, Dict, List, Optional, get_type_hints
from dataclasses import dataclass, field


@dataclass
class Tool:
    """
    Represents a tool that an agent can invoke.

    Tools are Python functions with type hints that agents can call to perform
    actions (search web, execute code, read files, etc.).

    Attributes:
        name: Tool identifier (defaults to function name)
        description: Human-readable description of what the tool does
        func: The actual Python function to call
        parameters: JSON schema describing the tool's parameters
    """
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        return self.func(*args, **kwargs)

    def to_prompt_string(self) -> str:
        """
        Generate a prompt-friendly description of this tool.

        Format:
            tool_name: description
            Parameters: {param1: type, param2: type, ...}
            Example: {"param1": "value"}
        """
        params = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])

        param_strs = []
        example_args = {}
        for param_name, param_info in params.items():
            param_type = param_info.get("type", "any")
            param_desc = param_info.get("description", "")
            is_required = param_name in required

            req_marker = "" if is_required else " (optional)"
            desc_part = f" - {param_desc}" if param_desc else ""
            param_strs.append(f"  {param_name}: {param_type}{req_marker}{desc_part}")

            # Generate example value for required params
            if is_required:
                if param_type == "string":
                    example_args[param_name] = "example"
                elif param_type == "integer":
                    example_args[param_name] = 1
                elif param_type == "number":
                    example_args[param_name] = 1.0
                elif param_type == "boolean":
                    example_args[param_name] = True
                else:
                    example_args[param_name] = "value"

        param_block = "\n".join(param_strs) if param_strs else "  (no parameters)"

        # Build example call
        import json
        example_json = json.dumps(example_args)
        example_line = f'Example tool_args: {example_json}'

        return f"{self.name}: {self.description}\nParameters:\n{param_block}\n{example_line}"

    def to_json_schema(self) -> Dict[str, Any]:
        """
        Generate JSON schema representation of this tool.

        Compatible with OpenAI function calling format.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


def _generate_schema_from_function(func: Callable) -> Dict[str, Any]:
    """
    Generate JSON schema from function signature and type hints.

    Args:
        func: Function to analyze

    Returns:
        JSON schema dict with properties and required fields
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        # Skip self/cls
        if param_name in ("self", "cls"):
            continue

        # Get type hint
        param_type = type_hints.get(param_name, Any)
        json_type = _python_type_to_json_type(param_type)

        # Get description from docstring if available
        param_desc = _extract_param_description(func, param_name)

        properties[param_name] = {
            "type": json_type,
        }

        if param_desc:
            properties[param_name]["description"] = param_desc

        # Check if required (no default value)
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


def _python_type_to_json_type(py_type: type) -> str:
    """Convert Python type to JSON schema type."""
    # Handle common types
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    # Get origin for generic types (List, Dict, etc.)
    origin = getattr(py_type, "__origin__", None)
    if origin is not None:
        py_type = origin

    return type_map.get(py_type, "string")


def _extract_param_description(func: Callable, param_name: str) -> Optional[str]:
    """
    Extract parameter description from function docstring.

    Looks for Google-style or NumPy-style docstrings:
        Args:
            param_name: Description here
    """
    docstring = inspect.getdoc(func)
    if not docstring:
        return None

    # Simple parsing - look for "param_name:" after "Args:" section
    lines = docstring.split("\n")
    in_args_section = False

    for line in lines:
        stripped = line.strip()

        # Check for Args section
        if stripped.lower().startswith("args:"):
            in_args_section = True
            continue

        # Exit Args section if we hit another section
        if in_args_section and stripped.endswith(":") and not stripped.startswith(param_name):
            break

        # Look for parameter
        if in_args_section and stripped.startswith(f"{param_name}:"):
            desc = stripped[len(param_name)+1:].strip()
            return desc

    return None


def tool(func: Optional[Callable] = None, *, name: Optional[str] = None, description: Optional[str] = None):
    """
    Decorator to register a function as an agent tool.

    Can be used with or without arguments:
        @tool
        def my_func():
            pass

        @tool(name="custom", description="Custom desc")
        def my_func():
            pass

    Args:
        func: Function to decorate (when used without arguments)
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)

    Returns:
        Tool instance that wraps the function

    Example:
        @tool
        def search_web(query: str, max_results: int = 5) -> List[str]:
            '''Search the web and return top results'''
            return web_search_api(query, max_results)

        # Now search_web is a Tool instance
        results = search_web(query="python agents")
    """
    def decorator(f: Callable) -> Tool:
        tool_name = name or f.__name__
        tool_desc = description or inspect.getdoc(f) or f"Execute {tool_name}"

        # Generate schema from function signature
        schema = _generate_schema_from_function(f)

        # Create Tool instance
        tool_instance = Tool(
            name=tool_name,
            description=tool_desc,
            func=f,
            parameters=schema
        )

        return tool_instance

    # Handle both @tool and @tool(...) syntax
    if func is None:
        # Called with arguments: @tool(name="...")
        return decorator
    else:
        # Called without arguments: @tool
        return decorator(func)


class ToolRegistry:
    """
    Registry for managing available tools.

    Provides methods to register tools, retrieve them by name, and generate
    prompt descriptions for all tools.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool_instance: Tool) -> None:
        """
        Register a tool in the registry.

        Args:
            tool_instance: Tool to register

        Raises:
            ValueError: If tool with same name already registered
        """
        if tool_instance.name in self._tools:
            raise ValueError(f"Tool '{tool_instance.name}' already registered")
        self._tools[tool_instance.name] = tool_instance

    def get(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def to_prompt_string(self) -> str:
        """
        Generate prompt string describing all available tools.

        Format suitable for inclusion in agent system prompts.
        """
        if not self._tools:
            return "No tools available."

        tool_descriptions = [
            tool.to_prompt_string()
            for tool in self._tools.values()
        ]

        return "\n\n".join(tool_descriptions)

    def to_json_schema(self) -> List[Dict[str, Any]]:
        """
        Generate JSON schema array for all tools.

        Compatible with OpenAI function calling format.
        """
        return [tool.to_json_schema() for tool in self._tools.values()]

    def __len__(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools

    def __iter__(self):
        """Iterate over tools."""
        return iter(self._tools.values())
