"""Multi-agent composition primitives.

This module hosts the small set of building blocks that let one agent
invoke another. The design intent (per ``AGENT_TOOL_REVIEW.md`` proposal
#18 and following) is to *not* introduce a new orchestration engine --
instead, every multi-agent pattern composes out of two primitives:

1. :func:`agent_as_tool` -- wrap any :class:`AgentProtocol` as a
   :class:`Tool` so it can be invoked through the ordinary ReAct
   dispatch path. Zero changes to existing agent loops.
2. :class:`TieredAgentTeam` -- ergonomic shorthand for supervisor /
   worker setups where roles run on different LLMs (planner + workers
   with capability-tiered models).

Patterns like plan-and-execute, critic/refinement, and parallel fan-out
are recipes on top of these primitives, not new framework features.

Event nesting: when an agent is invoked via :func:`agent_as_tool`, its
events are re-emitted in the supervisor's stream with
``AgentEvent.source`` set to the inner agent's name and
``parent_event_id`` linked to the supervisor's ACTION event. Streaming
UIs see a hierarchical trace; legacy consumers that ignore those fields
behave unchanged.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

from .tools import Tool
from .types import AgentEvent, AgentProtocol, EventType


__all__ = [
    "agent_as_tool",
    "AgentRole",
    "TieredAgentTeam",
]


def agent_as_tool(
    agent: AgentProtocol,
    name: str,
    description: str,
    task_param: str = "task",
    forward_events: Optional[Callable[[AgentEvent], None]] = None,
) -> Tool:
    """Wrap an :class:`AgentProtocol` as a :class:`Tool`.

    The returned tool, when called with ``{<task_param>: "..."}``,
    invokes ``agent.run(task)`` and returns the answer. The wrapped tool
    fits naturally into any agent's tool registry: a supervisor can
    dispatch to it through the same ReAct / Constrained pipeline it uses
    for ordinary tools, with coercion, timeouts, contracts, and event
    streaming all working unchanged.

    Args:
        agent: Any object satisfying :class:`AgentProtocol` -- typically
            another :class:`ReActAgent` / :class:`ConstrainedAgent` /
            :class:`ContractAgent`. Loop ergonomics are preserved because
            the wrapping uses ``agent.run`` (not ``agent.stream``).
        name: The tool name as it will appear to the supervisor.
            Conventionally lower_snake_case; the ``[\\w./\\-]+`` regex in
            ``ReActAgent._parse_action`` permits namespaced forms like
            ``team/researcher`` too.
        description: Tool description shown to the supervisor LLM. Should
            describe the sub-agent's purpose so the supervisor can decide
            when to route to it.
        task_param: Name of the keyword argument the supervisor must
            supply with the sub-task. Defaults to ``"task"``; override if
            you want something more semantic (``"question"``, ``"query"``,
            etc.).
        forward_events: Optional callback invoked once per event from the
            sub-agent's stream, *in addition to* receiving the final
            answer. Lets a UI render the sub-agent's reasoning even
            though the parent dispatch path only sees a string return
            value. The callback is passed an :class:`AgentEvent` with
            ``source`` set to the agent's name.

    Returns:
        A :class:`Tool` that, when called, executes the wrapped agent.

    Example::

        from cyllama.agents import ReActAgent
        from cyllama.agents.composition import agent_as_tool

        worker = ReActAgent(llm=fast_llm, tools=[search])
        supervisor_tool = agent_as_tool(
            worker,
            name="research",
            description="Investigate a topic and return findings.",
        )
        supervisor = ReActAgent(llm=big_llm, tools=[supervisor_tool])
        result = supervisor.run("Compare TLS 1.2 and TLS 1.3 handshakes.")
    """

    def _invoke(**kwargs: Any) -> str:
        task = kwargs.get(task_param)
        if not isinstance(task, str):
            raise ValueError(f"agent_as_tool({name!r}): missing required string argument {task_param!r}")
        parent_id = uuid.uuid4().hex

        if forward_events is None:
            # Fast path: no event forwarding, use run() directly.
            result = agent.run(task)
            return result.answer or ""

        # Forwarding path: stream events through the callback, annotating
        # each with source/parent_event_id so consumers can reconstruct
        # the nesting. The final ANSWER event's content becomes the tool's
        # return value (or the empty string if the sub-agent errored out).
        answer = ""
        for ev in agent.stream(task):
            # Annotate. AgentEvent fields are mutable; this preserves all
            # other attributes (type, metadata, content) unchanged.
            ev.source = name
            ev.parent_event_id = parent_id
            forward_events(ev)
            if ev.type == EventType.ANSWER:
                answer = ev.content
        return answer

    # Hand-crafted schema: one required string parameter named task_param.
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {task_param: {"type": "string"}},
        "required": [task_param],
    }
    return Tool(
        name=name,
        description=description,
        func=_invoke,
        parameters=schema,
        # The supervisor is calling a Python wrapper, not user code; we
        # don't want coerce_args to second-guess the typed kwargs we
        # construct here.
        coerce=False,
    )


@dataclass
class AgentRole:
    """A named agent within a multi-agent team.

    Attributes:
        name: Identifier used by the supervisor (and event ``source``).
        agent: The actual agent instance (any :class:`AgentProtocol`).
        description: Prompt-facing description; this is what the
            supervisor's LLM sees when deciding whether to route to this
            role.
        task_param: Name of the parameter passed when invoking. See
            :func:`agent_as_tool`.
    """

    name: str
    agent: AgentProtocol
    description: str
    task_param: str = "task"


class TieredAgentTeam:
    """Supervisor + worker team with optional capability tiering.

    A common multi-agent pattern: a strong "planner" model decides which
    specialised worker to dispatch to, and each worker may run on a
    different (often smaller / cheaper) LLM. This class is a thin
    convenience layer over :func:`agent_as_tool` -- it wraps every worker
    role as a tool and registers them all on the supervisor.

    The supervisor itself is the user's responsibility: pass any
    :class:`AgentProtocol` that can accept additional tools. Typically
    that's a :class:`ReActAgent` (its ``registry.register`` method is the
    extension point), but the team class doesn't depend on the concrete
    type.

    Example::

        from cyllama import LLM
        from cyllama.agents import ReActAgent
        from cyllama.agents.composition import AgentRole, TieredAgentTeam

        researcher = ReActAgent(llm=LLM("models/fast.gguf"), tools=[...])
        coder = ReActAgent(llm=LLM("models/code.gguf"), tools=[...])

        team = TieredAgentTeam(
            supervisor=ReActAgent(llm=LLM("models/strong.gguf"), tools=[]),
            workers=[
                AgentRole("researcher", researcher, "Find facts on the web."),
                AgentRole("coder", coder, "Write or modify code."),
            ],
        )
        result = team.run("Refactor X using technique Y.")

    GPU budgeting (multi-model setups): loading two 7B+ models is rarely
    viable on consumer hardware. Typical configurations are one large +
    N small, or model-swapping (one LLM in VRAM at a time, sequential
    dispatch). See ``cyllama/memory.py`` for sizing utilities.
    """

    def __init__(
        self,
        supervisor: AgentProtocol,
        workers: List[AgentRole],
        forward_events: Optional[Callable[[AgentEvent], None]] = None,
    ) -> None:
        if not workers:
            raise ValueError("TieredAgentTeam requires at least one worker")
        names = [w.name for w in workers]
        if len(set(names)) != len(names):
            raise ValueError(f"TieredAgentTeam: duplicate worker names: {names}")

        self.supervisor = supervisor
        self.workers = workers
        self._forward_events = forward_events

        # The supervisor must expose a tool registry to register against.
        # Both ReActAgent and ConstrainedAgent satisfy this; ContractAgent
        # delegates to its inner agent. If we ever wrap a supervisor that
        # doesn't, this is the line that errors and points the user at
        # the contract.
        registry = getattr(supervisor, "registry", None)
        if registry is None:
            raise TypeError(
                f"TieredAgentTeam supervisor must expose a `registry` attribute (got {type(supervisor).__name__})"
            )

        for role in workers:
            registry.register(
                agent_as_tool(
                    role.agent,
                    name=role.name,
                    description=role.description,
                    task_param=role.task_param,
                    forward_events=forward_events,
                )
            )

    def run(self, task: str) -> Any:
        """Run the supervisor; sub-agents are dispatched through tools."""
        return self.supervisor.run(task)

    def stream(self, task: str) -> Iterator[AgentEvent]:
        """Stream supervisor events; sub-agent events arrive via the
        ``forward_events`` callback configured at construction time."""
        yield from self.supervisor.stream(task)
