"""
Tool registry and definition system for cyllama agents.

Provides a simple, type-safe way to register and invoke tools that agents can use.
"""

import inspect
import json
import logging
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, get_type_hints
from dataclasses import dataclass, field

# Module logger
logger = logging.getLogger(__name__)


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

    Error Handling:
        - If get_type_hints() fails (e.g., forward references, invalid annotations),
          falls back to using the raw annotations from __annotations__
        - If a specific type cannot be resolved, defaults to "string" type
        - Logs warnings for any type resolution issues
    """
    sig = inspect.signature(func)

    # Try to get resolved type hints, fall back to raw annotations on error
    type_hints = _safe_get_type_hints(func)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        # Skip self/cls
        if param_name in ("self", "cls"):
            continue

        # Get type hint with fallback
        param_type = type_hints.get(param_name, Any)

        # Convert to JSON schema type with full type info
        type_info = _python_type_to_json_schema(param_type)

        # Get description from docstring if available
        param_desc = _extract_param_description(func, param_name)

        properties[param_name] = type_info.copy()

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


def _safe_get_type_hints(func: Callable) -> Dict[str, Any]:
    """
    Safely get type hints from a function, handling errors gracefully.

    This handles common issues with get_type_hints():
    - Forward references that can't be resolved
    - Invalid type annotations
    - Missing imports for type references
    - String annotations in older Python

    Args:
        func: Function to get type hints from

    Returns:
        Dictionary of parameter names to types, empty dict on failure
    """
    try:
        # Try to get fully resolved type hints
        return get_type_hints(func)
    except NameError as e:
        # Forward reference couldn't be resolved
        logger.warning(
            "Could not resolve type hints for %s: %s. Using raw annotations.",
            func.__name__, e
        )
    except TypeError as e:
        # Invalid type annotation
        logger.warning(
            "Invalid type annotation in %s: %s. Using raw annotations.",
            func.__name__, e
        )
    except Exception as e:
        # Unexpected error
        logger.warning(
            "Unexpected error getting type hints for %s: %s. Using raw annotations.",
            func.__name__, e
        )

    # Fall back to raw annotations
    try:
        annotations = getattr(func, '__annotations__', {})
        # Filter out return annotation
        return {k: v for k, v in annotations.items() if k != 'return'}
    except Exception:
        return {}


def _python_type_to_json_type(py_type: type) -> str:
    """
    Convert Python type to JSON schema type string.

    This is a simplified version for backward compatibility.
    For full schema generation with generic type support, use _python_type_to_json_schema().

    Args:
        py_type: Python type to convert

    Returns:
        JSON schema type string
    """
    schema = _python_type_to_json_schema(py_type)
    return schema.get("type", "string")


def _python_type_to_json_schema(py_type: type) -> Dict[str, Any]:
    """
    Convert Python type to full JSON schema with generic type support.

    Supports:
    - Basic types: str, int, float, bool, list, dict, None
    - Generic types: List[T], Dict[K, V], Optional[T], Union[A, B], Tuple[A, B]
    - Nested generics: List[Dict[str, int]], Optional[List[str]]
    - Literal types: Literal["a", "b", "c"]
    - Any type

    Args:
        py_type: Python type to convert

    Returns:
        JSON schema dictionary with type and optionally items/properties/enum

    Examples:
        >>> _python_type_to_json_schema(str)
        {"type": "string"}

        >>> _python_type_to_json_schema(List[int])
        {"type": "array", "items": {"type": "integer"}}

        >>> _python_type_to_json_schema(Optional[str])
        {"type": "string", "nullable": True}

        >>> _python_type_to_json_schema(Dict[str, int])
        {"type": "object", "additionalProperties": {"type": "integer"}}
    """
    # Handle None type
    if py_type is type(None):
        return {"type": "null"}

    # Handle Any type
    if py_type is Any:
        return {"type": "string"}  # Default to string for Any

    # Basic type mapping
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
        bytes: {"type": "string", "contentEncoding": "base64"},
    }

    # Check for basic types first
    if py_type in type_map:
        return type_map[py_type].copy()

    # Get origin and args for generic types
    origin = getattr(py_type, "__origin__", None)
    args = getattr(py_type, "__args__", ())

    if origin is None:
        # Not a generic type, check if it's a known class
        if isinstance(py_type, type):
            # Check if it's a subclass of known types
            for base_type, schema in type_map.items():
                try:
                    if issubclass(py_type, base_type):
                        return schema.copy()
                except TypeError:
                    pass
        # Unknown type, default to string
        logger.debug("Unknown type %s, defaulting to string", py_type)
        return {"type": "string"}

    # Handle Union types (including Optional)
    if origin is Union:
        # Check if it's Optional (Union with None)
        non_none_args = [a for a in args if a is not type(None)]
        has_none = len(non_none_args) < len(args)

        if len(non_none_args) == 1:
            # Optional[T] -> T with nullable
            schema = _python_type_to_json_schema(non_none_args[0])
            if has_none:
                schema["nullable"] = True
            return schema
        elif len(non_none_args) > 1:
            # Union[A, B, C] -> anyOf
            schemas = [_python_type_to_json_schema(a) for a in non_none_args]
            result = {"anyOf": schemas}
            if has_none:
                result["nullable"] = True
            return result
        else:
            # All None? Shouldn't happen
            return {"type": "null"}

    # Handle List/list types
    if origin is list:
        schema = {"type": "array"}
        if args:
            schema["items"] = _python_type_to_json_schema(args[0])
        return schema

    # Handle Dict/dict types
    if origin is dict:
        schema = {"type": "object"}
        if len(args) >= 2:
            # Dict[K, V] - V becomes additionalProperties
            schema["additionalProperties"] = _python_type_to_json_schema(args[1])
        return schema

    # Handle Tuple types
    if origin is tuple:
        schema = {"type": "array"}
        if args:
            # Check for Tuple[T, ...] (variable length)
            if len(args) == 2 and args[1] is ...:
                schema["items"] = _python_type_to_json_schema(args[0])
            else:
                # Fixed length tuple -> prefixItems
                schema["prefixItems"] = [
                    _python_type_to_json_schema(a) for a in args
                ]
                schema["minItems"] = len(args)
                schema["maxItems"] = len(args)
        return schema

    # Handle Set/set types (as array with uniqueItems)
    if origin is set or origin is frozenset:
        schema = {"type": "array", "uniqueItems": True}
        if args:
            schema["items"] = _python_type_to_json_schema(args[0])
        return schema

    # Handle Literal types
    try:
        from typing import Literal, get_args, get_origin
        if get_origin(py_type) is Literal:
            values = get_args(py_type)
            if values:
                # Determine type from values
                if all(isinstance(v, str) for v in values):
                    return {"type": "string", "enum": list(values)}
                elif all(isinstance(v, int) for v in values):
                    return {"type": "integer", "enum": list(values)}
                elif all(isinstance(v, bool) for v in values):
                    return {"type": "boolean", "enum": list(values)}
                else:
                    return {"enum": list(values)}
    except ImportError:
        pass  # Literal not available in older Python

    # Handle Callable (as string describing the function)
    try:
        from typing import Callable as CallableType
        if origin is CallableType or (hasattr(origin, '__name__') and origin.__name__ == 'Callable'):
            return {"type": "string", "description": "callable"}
    except (ImportError, AttributeError):
        pass

    # Fallback: check basic type map for origin
    if origin in type_map:
        return type_map[origin].copy()

    # Unknown generic, default to string
    logger.debug("Unknown generic type %s with origin %s, defaulting to string", py_type, origin)
    return {"type": "string"}


def _extract_param_description(func: Callable, param_name: str) -> Optional[str]:
    """
    Extract parameter description from function docstring.

    Supports multiple docstring formats:
    - Google-style: Args: param_name: Description
    - NumPy-style: Parameters\\n----------\\nparam_name : type\\n    Description
    - Sphinx-style: :param param_name: Description
    - reStructuredText: :param param_name: Description
    - Epytext: @param param_name: Description

    Multi-line descriptions are concatenated.

    Args:
        func: Function to extract docstring from
        param_name: Parameter name to find description for

    Returns:
        Parameter description or None if not found
    """
    docstring = inspect.getdoc(func)
    if not docstring:
        return None

    # Try each parsing strategy
    desc = _extract_google_style(docstring, param_name)
    if desc:
        return desc

    desc = _extract_numpy_style(docstring, param_name)
    if desc:
        return desc

    desc = _extract_sphinx_style(docstring, param_name)
    if desc:
        return desc

    desc = _extract_epytext_style(docstring, param_name)
    if desc:
        return desc

    return None


def _extract_google_style(docstring: str, param_name: str) -> Optional[str]:
    """
    Extract parameter description from Google-style docstring.

    Format:
        Args:
            param_name: Description here
            param_name (type): Description here
            param_name: Description that
                continues on next line
    """
    lines = docstring.split("\n")
    in_args_section = False
    description_lines = []
    found_param = False
    base_indent = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check for Args/Arguments section
        if stripped.lower() in ("args:", "arguments:"):
            in_args_section = True
            continue

        # Exit Args section if we hit another section header (e.g., "Returns:")
        if in_args_section and stripped and stripped.endswith(":") and ":" not in stripped[:-1]:
            break

        if not in_args_section:
            continue

        # Calculate indentation
        line_indent = len(line) - len(line.lstrip())

        # Check for parameter definition at the parameter level
        # Patterns: "param_name:", "param_name (type):", "param_name(type):"
        param_patterns = [
            f"{param_name}:",
            f"{param_name} (",
            f"{param_name}(",
        ]

        is_param_line = any(stripped.startswith(p) for p in param_patterns)

        # Check if this is a new parameter (any word followed by colon at same indent)
        is_new_param = bool(re.match(r"^\w+(\s*\([^)]*\))?\s*:", stripped))

        if is_param_line:
            found_param = True
            base_indent = line_indent
            # Extract description after the colon
            colon_idx = stripped.find(":")
            if colon_idx != -1:
                desc = stripped[colon_idx + 1:].strip()
                if desc:
                    description_lines.append(desc)
            continue

        if found_param:
            # Check if this is a new parameter definition (same or less indent)
            if is_new_param and base_indent is not None and line_indent <= base_indent:
                # New parameter, stop
                break

            # Continuation line (more indented than parameter name)
            if base_indent is not None and line_indent > base_indent:
                if stripped:
                    description_lines.append(stripped)
            elif stripped:
                # Same indent but not a parameter definition - might be continuation
                # Only if it doesn't look like a new param
                if not is_new_param:
                    description_lines.append(stripped)
                else:
                    break

    return " ".join(description_lines) if description_lines else None


def _extract_numpy_style(docstring: str, param_name: str) -> Optional[str]:
    """
    Extract parameter description from NumPy-style docstring.

    Format:
        Parameters
        ----------
        param_name : type
            Description here
        param_name : type, optional
            Description that continues
            on multiple lines
    """
    lines = docstring.split("\n")
    in_params_section = False
    description_lines = []
    found_param = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check for Parameters section
        if stripped.lower() == "parameters":
            # Look for dashes on next line
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("-"):
                in_params_section = True
            continue

        # Skip dashes line
        if in_params_section and stripped.startswith("-") and stripped.replace("-", "") == "":
            continue

        # Exit section if we hit another header (word followed by dashes)
        if in_params_section and stripped and not stripped.startswith("-"):
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("-"):
                if found_param:
                    break
                in_params_section = False
                continue

        if not in_params_section:
            continue

        # Check for parameter definition: "param_name : type"
        if " : " in stripped or stripped.startswith(f"{param_name}:"):
            parts = stripped.split(":", 1) if ":" in stripped else stripped.split(" : ", 1)
            if parts[0].strip() == param_name:
                found_param = True
                continue

        # Description lines are indented
        if found_param:
            if line.startswith("    ") or line.startswith("\t"):
                description_lines.append(stripped)
            elif stripped and not " : " in stripped:
                # Continue if it's a continuation line
                if not re.match(r"^\w+\s*:", stripped):
                    description_lines.append(stripped)
                else:
                    break
            elif stripped:
                break

    return " ".join(description_lines) if description_lines else None


def _extract_sphinx_style(docstring: str, param_name: str) -> Optional[str]:
    """
    Extract parameter description from Sphinx/reST-style docstring.

    Format:
        :param param_name: Description here
        :param type param_name: Description here
        :parameter param_name: Description here
    """
    lines = docstring.split("\n")
    description_lines = []
    found_param = False

    for line in lines:
        stripped = line.strip()

        # Check for :param param_name: or :parameter param_name:
        patterns = [
            f":param {param_name}:",
            f":parameter {param_name}:",
            f":param \\w+ {param_name}:",  # :param type name:
        ]

        for pattern in patterns:
            match = re.match(pattern.replace("\\w+", r"\w+"), stripped)
            if match or stripped.startswith(pattern.rstrip(":")):
                found_param = True
                # Extract description after the last colon
                colon_idx = stripped.rfind(":")
                if colon_idx != -1:
                    desc = stripped[colon_idx + 1:].strip()
                    if desc:
                        description_lines.append(desc)
                break

        if found_param and not any(stripped.startswith(p.split()[0]) for p in patterns):
            # Continuation line
            if stripped and not stripped.startswith(":"):
                description_lines.append(stripped)
            elif stripped.startswith(":"):
                break

    return " ".join(description_lines) if description_lines else None


def _extract_epytext_style(docstring: str, param_name: str) -> Optional[str]:
    """
    Extract parameter description from Epytext-style docstring.

    Format:
        @param param_name: Description here
        @type param_name: type
    """
    lines = docstring.split("\n")
    description_lines = []
    found_param = False

    for line in lines:
        stripped = line.strip()

        # Check for @param param_name:
        if stripped.startswith(f"@param {param_name}:"):
            found_param = True
            desc = stripped[len(f"@param {param_name}:"):].strip()
            if desc:
                description_lines.append(desc)
            continue

        if found_param:
            # Continuation line (not starting with @)
            if stripped and not stripped.startswith("@"):
                description_lines.append(stripped)
            elif stripped.startswith("@"):
                break

    return " ".join(description_lines) if description_lines else None


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
