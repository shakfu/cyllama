"""
Grammar generation for constrained agent tool calling.

Provides utilities to convert tool schemas into GBNF grammars that enforce
valid tool call syntax, ensuring 100% reliable parsing.
"""

import json
from typing import List, Dict, Any, Optional
from enum import Enum

from ..llama.llama_cpp import json_schema_to_grammar
from .tools import Tool, ToolRegistry


class GrammarFormat(Enum):
    """Supported output formats for constrained generation."""
    JSON = "json"
    JSON_ARRAY = "json_array"
    FUNCTION_CALL = "function_call"


def generate_tool_call_schema(
    tools: List[Tool],
    allow_reasoning: bool = True,
    format: GrammarFormat = GrammarFormat.JSON
) -> Dict[str, Any]:
    """
    Generate JSON schema for tool calling.

    Args:
        tools: List of available tools
        allow_reasoning: Include optional reasoning field
        format: Output format (JSON, JSON_ARRAY, or FUNCTION_CALL)

    Returns:
        JSON schema dict that can be converted to GBNF grammar
    """
    if format == GrammarFormat.JSON:
        return _generate_json_tool_schema(tools, allow_reasoning)
    elif format == GrammarFormat.JSON_ARRAY:
        return _generate_json_array_tool_schema(tools, allow_reasoning)
    elif format == GrammarFormat.FUNCTION_CALL:
        return _generate_function_call_schema(tools, allow_reasoning)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _generate_json_tool_schema(tools: List[Tool], allow_reasoning: bool) -> Dict[str, Any]:
    """
    Generate schema for JSON format tool calls.

    Format:
        {
            "reasoning": "optional reasoning text",
            "tool_name": "search",
            "tool_args": {"query": "...", "max_results": 5}
        }
    """
    # Build enum of tool names
    tool_names = [tool.name for tool in tools]

    schema = {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "enum": tool_names,
                "description": "Name of the tool to call"
            },
            "tool_args": {
                "type": "object",
                "description": "Arguments to pass to the tool"
            }
        },
        "required": ["tool_name", "tool_args"]
    }

    if allow_reasoning:
        schema["properties"]["reasoning"] = {
            "type": "string",
            "description": "Brief reasoning about why this tool is needed"
        }

    return schema


def _generate_json_array_tool_schema(tools: List[Tool], allow_reasoning: bool) -> Dict[str, Any]:
    """
    Generate schema for JSON array format (multiple tool calls).

    Format:
        {
            "reasoning": "optional reasoning",
            "tool_calls": [
                {"tool_name": "search", "tool_args": {...}},
                {"tool_name": "calculate", "tool_args": {...}}
            ]
        }
    """
    tool_names = [tool.name for tool in tools]

    schema = {
        "type": "object",
        "properties": {
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "enum": tool_names
                        },
                        "tool_args": {
                            "type": "object"
                        }
                    },
                    "required": ["tool_name", "tool_args"]
                },
                "minItems": 1
            }
        },
        "required": ["tool_calls"]
    }

    if allow_reasoning:
        schema["properties"]["reasoning"] = {
            "type": "string",
            "description": "Brief reasoning about the tool calls"
        }

    return schema


def _generate_function_call_schema(tools: List[Tool], allow_reasoning: bool) -> Dict[str, Any]:
    """
    Generate schema for OpenAI-style function calling format.

    Format:
        {
            "reasoning": "optional",
            "name": "tool_name",
            "arguments": "{\"arg1\": \"value1\"}"  # JSON string
        }
    """
    tool_names = [tool.name for tool in tools]

    schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": tool_names,
                "description": "Function name"
            },
            "arguments": {
                "type": "string",
                "description": "JSON string of arguments"
            }
        },
        "required": ["name", "arguments"]
    }

    if allow_reasoning:
        schema["properties"]["reasoning"] = {
            "type": "string",
            "description": "Reasoning for function call"
        }

    return schema


def generate_tool_call_grammar(
    tools: List[Tool],
    allow_reasoning: bool = True,
    format: GrammarFormat = GrammarFormat.JSON
) -> str:
    """
    Generate GBNF grammar for tool calling.

    Args:
        tools: List of available tools
        allow_reasoning: Include optional reasoning field
        format: Output format

    Returns:
        GBNF grammar string
    """
    schema = generate_tool_call_schema(tools, allow_reasoning, format)
    return json_schema_to_grammar(schema)


def generate_answer_or_tool_schema(tools: List[Tool], allow_reasoning: bool = True) -> Dict[str, Any]:
    """
    Generate schema that allows either a final answer OR a tool call.

    Format:
        {
            "type": "answer",
            "content": "The final answer"
        }
        OR
        {
            "type": "tool_call",
            "reasoning": "optional",
            "tool_name": "search",
            "tool_args": {...}
        }
    """
    tool_names = [tool.name for tool in tools]

    # Using oneOf to allow either answer or tool_call
    schema = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["answer", "tool_call"],
                "description": "Type of response"
            }
        },
        "required": ["type"],
        "oneOf": [
            {
                "properties": {
                    "type": {"const": "answer"},
                    "content": {
                        "type": "string",
                        "description": "The final answer to the question"
                    }
                },
                "required": ["type", "content"]
            },
            {
                "properties": {
                    "type": {"const": "tool_call"},
                    "tool_name": {
                        "type": "string",
                        "enum": tool_names
                    },
                    "tool_args": {
                        "type": "object"
                    }
                },
                "required": ["type", "tool_name", "tool_args"]
            }
        ]
    }

    return schema


def generate_answer_or_tool_grammar(tools: List[Tool], allow_reasoning: bool = True) -> str:
    """
    Generate GBNF grammar for answer-or-tool-call format.

    This allows the agent to either provide a final answer or make a tool call.

    Args:
        tools: List of available tools
        allow_reasoning: Include optional reasoning field

    Returns:
        GBNF grammar string
    """
    schema = generate_answer_or_tool_schema(tools, allow_reasoning)
    return json_schema_to_grammar(schema)


def generate_specific_tool_schema(tool: Tool) -> Dict[str, Any]:
    """
    Generate schema for calling a specific tool with validated arguments.

    This creates a schema that enforces the exact parameter types and
    requirements for a single tool.

    Args:
        tool: Tool to generate schema for

    Returns:
        JSON schema that validates tool arguments
    """
    schema = {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "const": tool.name
            },
            "tool_args": tool.parameters
        },
        "required": ["tool_name", "tool_args"]
    }

    return schema


def generate_specific_tool_grammar(tool: Tool) -> str:
    """
    Generate GBNF grammar for a specific tool with argument validation.

    Args:
        tool: Tool to generate grammar for

    Returns:
        GBNF grammar string
    """
    schema = generate_specific_tool_schema(tool)
    return json_schema_to_grammar(schema)


class GrammarCache:
    """
    Cache for compiled grammars to avoid recompilation.

    Grammar compilation can be expensive, so we cache results.
    """

    def __init__(self):
        self._cache: Dict[str, str] = {}

    def get_or_create(
        self,
        key: str,
        generator: callable
    ) -> str:
        """
        Get grammar from cache or generate it.

        Args:
            key: Cache key
            generator: Function that generates the grammar

        Returns:
            Grammar string
        """
        if key not in self._cache:
            self._cache[key] = generator()
        return self._cache[key]

    def clear(self):
        """Clear the cache."""
        self._cache.clear()

    def __len__(self) -> int:
        """Number of cached grammars."""
        return len(self._cache)


# Global grammar cache
_grammar_cache = GrammarCache()


def get_cached_tool_grammar(
    tools: List[Tool],
    allow_reasoning: bool = True,
    format: GrammarFormat = GrammarFormat.JSON
) -> str:
    """
    Get tool call grammar from cache or generate it.

    Args:
        tools: List of tools
        allow_reasoning: Include reasoning field
        format: Output format

    Returns:
        Cached or newly generated grammar
    """
    # Create cache key from tool names and settings
    tool_names = sorted([t.name for t in tools])
    key = f"tools:{','.join(tool_names)}:reasoning={allow_reasoning}:format={format.value}"

    return _grammar_cache.get_or_create(
        key,
        lambda: generate_tool_call_grammar(tools, allow_reasoning, format)
    )


def get_cached_answer_or_tool_grammar(
    tools: List[Tool],
    allow_reasoning: bool = True
) -> str:
    """
    Get answer-or-tool grammar from cache or generate it.

    Args:
        tools: List of tools
        allow_reasoning: Include reasoning field

    Returns:
        Cached or newly generated grammar
    """
    # Create cache key from tool names and settings
    tool_names = sorted([t.name for t in tools])
    key = f"answer_or_tool:{','.join(tool_names)}:reasoning={allow_reasoning}"

    return _grammar_cache.get_or_create(
        key,
        lambda: generate_answer_or_tool_grammar(tools, allow_reasoning)
    )


def clear_grammar_cache():
    """Clear the grammar cache."""
    _grammar_cache.clear()
