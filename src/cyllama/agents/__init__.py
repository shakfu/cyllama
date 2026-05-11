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
  (**experimental** -- protocol version is hardcoded, no conformance tests
  against a reference ACP client; API may change. See acp.py module
  docstring for details.)
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
    coerce_args,
    tool,
)
from .react import ReActAgent
from .types import AgentEvent, AgentMetrics, AgentProtocol, AgentResult, EventType
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

# Multi-agent composition primitives + pattern helpers
from .composition import (
    agent_as_tool,
    AgentRole,
    TieredAgentTeam,
    ReflectionLoop,
    plan_and_execute,
    mcp_agent_tool,
    rag_as_tool,
)

# Semantic (long-term, RAG-backed) memory
from .memory import MemoryRecord, SemanticMemory

# DAG workflow orchestration (Phases 1-4)
from .workflow import (
    Workflow,
    CompiledWorkflow,
    WorkflowResult,
    DryRunPlan,
    WorkflowDefinitionError,
    WorkflowExecutionError,
    WorkflowRoutingError,
    END,
    agent_node,
    tool_node,
    WorkflowInvariant,
    WorkflowExecutionState,
    reducer,
    workflow_node,
)

# Async agent wrappers
from .async_agent import (
    AsyncReActAgent,
    AsyncConstrainedAgent,
    run_agent_async,
)

__all__ = [
    # Tools
    "Tool",
    "tool",
    "ToolRegistry",
    "ToolArgumentError",
    "ToolTimeoutError",
    "coerce_args",
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
    # Events and Results
    "EventType",
    "AgentEvent",
    "AgentResult",
    "AgentMetrics",
    "AgentProtocol",
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
    # Async agents
    # Multi-agent composition + pattern helpers
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
    # DAG workflow orchestration
    "Workflow",
    "CompiledWorkflow",
    "WorkflowResult",
    "DryRunPlan",
    "WorkflowDefinitionError",
    "WorkflowExecutionError",
    "WorkflowRoutingError",
    "END",
    "agent_node",
    "tool_node",
    "WorkflowInvariant",
    "WorkflowExecutionState",
    "reducer",
    "workflow_node",
    # Async wrappers
    "AsyncReActAgent",
    "AsyncConstrainedAgent",
    "run_agent_async",
]
