"""
Agent implementations for cyllama.

This module provides agent architectures that leverage cyllama's strengths:
- Zero dependencies
- High-performance local inference
- Streaming and constrained generation
- Framework-agnostic design

Available agents:
- ReActAgent: Reasoning + Acting agent with tool calling
- ConstrainedAgent: Grammar-enforced tool calling for 100% reliability
- ContractAgent: Contract-based agent with C++26-inspired pre/post conditions
- ACPAgent: Agent Client Protocol compliant agent for editor integration
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
from .contract import (
    ContractAgent,
    ContractPolicy,
    ContractViolation,
    ContractTermination,
    ContractContext,
    ContractSpec,
    PreCondition,
    PostCondition,
    IterationState,
    pre,
    post,
    contract_assert,
)

# ACP/MCP support
from .acp import ACPAgent, serve_acp
from .mcp import McpClient, McpServerConfig, McpTransportType, McpTool
from .session import (
    Session,
    SessionStore,
    MemorySessionStore,
    FileSessionStore,
    SqliteSessionStore,
    create_session_store,
)
from .jsonrpc import (
    JsonRpcServer,
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcError,
    StdioTransport,
)

__all__ = [
    # Tools
    "Tool",
    "tool",
    "ToolRegistry",

    # Agents
    "ReActAgent",
    "ConstrainedAgent",
    "ContractAgent",

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

    # Contract types
    "ContractPolicy",
    "ContractViolation",
    "ContractTermination",
    "ContractContext",
    "ContractSpec",
    "PreCondition",
    "PostCondition",
    "IterationState",
    "pre",
    "post",
    "contract_assert",

    # ACP (Agent Client Protocol)
    "ACPAgent",
    "serve_acp",

    # MCP (Model Context Protocol)
    "McpClient",
    "McpServerConfig",
    "McpTransportType",
    "McpTool",

    # Session storage
    "Session",
    "SessionStore",
    "MemorySessionStore",
    "FileSessionStore",
    "SqliteSessionStore",
    "create_session_store",

    # JSON-RPC transport
    "JsonRpcServer",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "JsonRpcError",
    "StdioTransport",
]
