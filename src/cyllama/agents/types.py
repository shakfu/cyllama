"""Shared agent types.

Holds the vocabulary every cyllama agent uses (event/result/metrics
dataclasses) and the structural :class:`AgentProtocol` contract that
agent implementations satisfy. Lives in its own module so concrete
agents (`react.py`, `constrained.py`, `contract.py`) and consumers of
agents (`async_agent.py`, `acp.py`) can import the shared types
without depending on any one agent implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Protocol, runtime_checkable


class EventType(Enum):
    """Types of events emitted during agent execution."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    ANSWER = "answer"
    ERROR = "error"
    # Contract-related events
    CONTRACT_CHECK = "contract_check"
    CONTRACT_VIOLATION = "contract_violation"


@dataclass
class AgentEvent:
    """Event emitted during agent execution."""

    type: EventType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMetrics:
    """Performance metrics for agent execution."""

    total_time_ms: float = 0.0
    iterations: int = 0
    tool_calls: int = 0
    tool_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    tokens_generated: int = 0
    loop_detected: bool = False
    error_count: int = 0

    def __str__(self) -> str:
        return (
            f"AgentMetrics(iterations={self.iterations}, "
            f"tool_calls={self.tool_calls}, "
            f"total_time={self.total_time_ms:.1f}ms, "
            f"gen_time={self.generation_time_ms:.1f}ms, "
            f"tool_time={self.tool_time_ms:.1f}ms)"
        )


@dataclass
class AgentResult:
    """Result from agent execution."""

    answer: str
    steps: List[AgentEvent]
    iterations: int
    success: bool
    error: Optional[str] = None
    metrics: Optional[AgentMetrics] = None


@runtime_checkable
class AgentProtocol(Protocol):
    """Structural contract for agents driving an LLM tool-calling loop.

    Satisfied by :class:`~cyllama.agents.react.ReActAgent`,
    :class:`~cyllama.agents.constrained.ConstrainedAgent`, and
    :class:`~cyllama.agents.contract.ContractAgent`. Used by the
    async wrappers (:class:`~cyllama.agents.async_agent.AsyncReActAgent`,
    :class:`~cyllama.agents.async_agent.AsyncConstrainedAgent`) and
    by :class:`~cyllama.agents.acp.ACPAgent`'s inner-agent dispatch
    so the wrappers can hand off to either kind of agent without an
    ``isinstance`` ladder.

    Implementations may add backend-specific surface (loop-detection
    metrics, grammar handling, contract violations) -- the protocol
    only nails down the three members the wrappers actually call.
    """

    @property
    def metrics(self) -> Optional[AgentMetrics]:
        """Metrics from the most recent ``run`` / ``stream`` call, or
        None if the agent hasn't been run yet."""
        ...

    def run(self, task: str) -> AgentResult:
        """Execute the agent on ``task`` and return the final result."""
        ...

    def stream(self, task: str) -> Iterator[AgentEvent]:
        """Execute the agent on ``task``, yielding events as they occur."""
        ...
