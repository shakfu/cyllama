"""
Agent implementations for cyllama.

This module provides agent architectures that leverage cyllama's strengths:
- Zero dependencies
- High-performance local inference
- Streaming and constrained generation
- Framework-agnostic design

Available agents:
- ReAct: Reasoning + Acting agent with tool calling
- ConstrainedAgent: Grammar-enforced tool calling for 100% reliability
"""

from .tools import Tool, tool, ToolRegistry
from .react import ReActAgent, EventType, AgentEvent, AgentResult, AgentMetrics
from .constrained import ConstrainedAgent, ConstrainedGenerationConfig
from .grammar import (
    GrammarFormat,
    generate_tool_call_grammar,
    generate_tool_call_schema,
    generate_answer_or_tool_grammar,
    get_cached_tool_grammar,
    get_cached_answer_or_tool_grammar,
    clear_grammar_cache,
)

__all__ = [
    # Tools
    "Tool",
    "tool",
    "ToolRegistry",

    # Agents
    "ReActAgent",
    "ConstrainedAgent",

    # Events and Results
    "EventType",
    "AgentEvent",
    "AgentResult",
    "AgentMetrics",

    # Configuration
    "ConstrainedGenerationConfig",

    # Grammar utilities
    "GrammarFormat",
    "generate_tool_call_grammar",
    "generate_tool_call_schema",
    "generate_answer_or_tool_grammar",
    "get_cached_tool_grammar",
    "get_cached_answer_or_tool_grammar",
    "clear_grammar_cache",
]
