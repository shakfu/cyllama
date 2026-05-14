"""
Agent implementations for cyllama.

This module provides agent architectures that leverage cyllama's strengths:
- Zero dependencies
- High-performance local inference
- Streaming and constrained generation
- Framework-agnostic design

The package exports only the user-facing surface area listed in
``__all__``. Implementation details, transport plumbing, workflow
internals, and infrequently-used subsystems remain accessible at their
submodule paths (e.g. ``cyllama.agents.workflow``,
``cyllama.agents.session``, ``cyllama.agents.grammar``,
``cyllama.agents.acp``, ``cyllama.agents.jsonrpc``,
``cyllama.agents.async_agent``).

Available agents:
- ReActAgent: Reasoning + Acting agent with tool calling
- ConstrainedAgent: Grammar-enforced tool calling for 100% reliability
- ContractAgent: Contract-based agent with C++26-inspired pre/post conditions
"""

from .tools import (
    Ge,
    Gt,
    Le,
    Lt,
    MaxLen,
    MinLen,
    MultipleOf,
    Pattern,
    Tool,
    ToolArgumentError,
    ToolTimeoutError,
    ToolRegistry,
    tool,
)
from .react import ReActAgent
from .types import AgentEvent, AgentMetrics, AgentResult, EventType
from .constrained import ConstrainedAgent
from .contract import (
    ContractAgent,
    ContractPolicy,
    ContractViolation,
    contract_assert,
    post,
    pre,
)
from .mcp import McpClient, McpServerConfig
from .composition import (
    AgentRole,
    ReflectionLoop,
    TieredAgentTeam,
    agent_as_tool,
    mcp_agent_tool,
    plan_and_execute,
    rag_as_tool,
)
from .memory import MemoryRecord, SemanticMemory
from .runner import AGENT_KINDS, stream_agent

__all__ = [
    # Tools
    "Tool",
    "tool",
    "ToolRegistry",
    "ToolArgumentError",
    "ToolTimeoutError",
    # Annotated[] constraint markers
    "Ge",
    "Gt",
    "Le",
    "Lt",
    "MultipleOf",
    "MinLen",
    "MaxLen",
    "Pattern",
    # Agents
    "ReActAgent",
    "ConstrainedAgent",
    "ContractAgent",
    # Events and results
    "EventType",
    "AgentEvent",
    "AgentResult",
    "AgentMetrics",
    # Contract user API
    "ContractPolicy",
    "ContractViolation",
    "pre",
    "post",
    "contract_assert",
    # MCP user API
    "McpClient",
    "McpServerConfig",
    # Composition primitives
    "agent_as_tool",
    "AgentRole",
    "TieredAgentTeam",
    "ReflectionLoop",
    "plan_and_execute",
    "mcp_agent_tool",
    "rag_as_tool",
    # Semantic memory
    "MemoryRecord",
    "SemanticMemory",
    # Runner / dispatcher
    "AGENT_KINDS",
    "stream_agent",
]
