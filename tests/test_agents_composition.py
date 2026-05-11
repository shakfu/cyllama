"""Tests for the multi-agent composition primitives in agents/composition.py.

Covers:
- agent_as_tool wraps any AgentProtocol as a callable Tool
- The wrapped tool reports its schema correctly (single required string)
- Sub-agent events forwarded via forward_events callback carry source +
  parent_event_id annotations so streaming UIs can render nesting
- TieredAgentTeam registers workers on the supervisor and rejects bad
  configurations (no workers, duplicate names, supervisor without registry)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List, Optional

import pytest

from cyllama.agents import (
    AgentEvent,
    AgentMetrics,
    AgentResult,
    EventType,
    Tool,
    ToolRegistry,
)
from cyllama.agents.composition import AgentRole, TieredAgentTeam, agent_as_tool


# ---------------------------------------------------------------------------
# Stubs that satisfy AgentProtocol
# ---------------------------------------------------------------------------


@dataclass
class _StubAgent:
    """Minimal AgentProtocol-shaped stub for composition tests.

    Implements ``run`` / ``stream`` with a fixed scripted output so tests
    don't need a real LLM.
    """

    answer: str = "stub answer"
    events: Optional[List[AgentEvent]] = None
    _metrics: Optional[AgentMetrics] = None

    @property
    def metrics(self) -> Optional[AgentMetrics]:
        return self._metrics

    def run(self, task: str) -> AgentResult:
        steps = self.events or []
        return AgentResult(answer=self.answer, steps=steps, iterations=1, success=True)

    def stream(self, task: str) -> Iterator[AgentEvent]:
        for ev in self.events or []:
            yield ev
        yield AgentEvent(type=EventType.ANSWER, content=self.answer)


class _StubSupervisor:
    """Stub supervisor with a real ToolRegistry so TieredAgentTeam can
    register workers against it."""

    def __init__(self) -> None:
        self.registry = ToolRegistry()
        self._metrics = None

    @property
    def metrics(self) -> Optional[AgentMetrics]:
        return self._metrics

    def run(self, task: str) -> AgentResult:
        return AgentResult(answer="supervisor done", steps=[], iterations=0, success=True)

    def stream(self, task: str) -> Iterator[AgentEvent]:
        yield AgentEvent(type=EventType.ANSWER, content="supervisor done")


# ---------------------------------------------------------------------------
# agent_as_tool: basic wrapping
# ---------------------------------------------------------------------------


def test_agent_as_tool_returns_tool_with_correct_schema():
    agent = _StubAgent(answer="hi")
    t = agent_as_tool(agent, name="helper", description="A helper.")
    assert isinstance(t, Tool)
    assert t.name == "helper"
    assert t.description == "A helper."
    # Schema: one required string parameter named `task` by default.
    assert t.parameters == {
        "type": "object",
        "properties": {"task": {"type": "string"}},
        "required": ["task"],
    }


def test_agent_as_tool_invokes_inner_agent_run():
    agent = _StubAgent(answer="42")
    t = agent_as_tool(agent, name="solver", description="Solve.")
    assert t(task="what is the answer?") == "42"


def test_agent_as_tool_respects_custom_task_param():
    agent = _StubAgent(answer="found it")
    t = agent_as_tool(agent, name="searcher", description="Search.", task_param="query")
    assert t.parameters["required"] == ["query"]
    assert t(query="anything") == "found it"


def test_agent_as_tool_missing_task_arg_raises():
    agent = _StubAgent(answer="x")
    t = agent_as_tool(agent, name="x", description="x")
    with pytest.raises(ValueError, match="task"):
        t()  # no kwargs


def test_agent_as_tool_non_string_task_arg_raises():
    agent = _StubAgent(answer="x")
    t = agent_as_tool(agent, name="x", description="x")
    with pytest.raises(ValueError, match="string"):
        t(task=123)  # type: ignore[arg-type]


def test_agent_as_tool_opts_out_of_coercion():
    """The wrapper builds its own typed kwargs; coerce_args would be redundant
    and would interfere if the schema had unusual shapes. coerce=False."""
    agent = _StubAgent(answer="x")
    t = agent_as_tool(agent, name="x", description="x")
    assert t.coerce is False


# ---------------------------------------------------------------------------
# agent_as_tool: event forwarding
# ---------------------------------------------------------------------------


def test_forward_events_callback_receives_substream():
    """When forward_events is supplied, every event from the sub-agent's
    stream is passed through, with source + parent_event_id annotated."""
    sub_events = [
        AgentEvent(type=EventType.THOUGHT, content="thinking"),
        AgentEvent(type=EventType.ACTION, content="act"),
    ]
    agent = _StubAgent(answer="done", events=list(sub_events))
    forwarded: List[AgentEvent] = []
    t = agent_as_tool(agent, name="worker", description="Do work.", forward_events=forwarded.append)

    out = t(task="any")
    assert out == "done"
    # Substream events + the final ANSWER = 3.
    assert len(forwarded) == 3
    # Every forwarded event carries source set to the tool/agent name.
    for ev in forwarded:
        assert ev.source == "worker"
    # parent_event_id is consistent across this run.
    parent_ids = {ev.parent_event_id for ev in forwarded}
    assert len(parent_ids) == 1
    assert next(iter(parent_ids)) is not None


def test_forward_events_different_parent_id_per_call():
    """Two invocations of the same tool produce different parent_event_ids
    so observers can disambiguate concurrent / sequential runs."""
    agent = _StubAgent(answer="x")
    forwarded: List[AgentEvent] = []
    t = agent_as_tool(agent, name="w", description="w", forward_events=forwarded.append)

    t(task="first")
    parent_a = forwarded[-1].parent_event_id
    forwarded.clear()
    t(task="second")
    parent_b = forwarded[-1].parent_event_id

    assert parent_a is not None and parent_b is not None
    assert parent_a != parent_b


def test_no_forward_events_means_no_callback_invocation():
    """Fast path: when forward_events is None, the wrapper calls run() not
    stream(); we verify by giving the stub an `events` list that should
    not surface anywhere."""
    sub_events = [AgentEvent(type=EventType.THOUGHT, content="should not be seen")]
    agent = _StubAgent(answer="done", events=sub_events)
    # No forward_events callback supplied.
    t = agent_as_tool(agent, name="silent", description="x")
    assert t(task="any") == "done"
    # The substream events were not surfaced anywhere observable; the
    # caller just gets the final answer string. Nothing to assert beyond
    # that we got the answer and didn't raise.


# ---------------------------------------------------------------------------
# TieredAgentTeam
# ---------------------------------------------------------------------------


def test_team_registers_workers_on_supervisor():
    sup = _StubSupervisor()
    w1 = _StubAgent(answer="researched")
    w2 = _StubAgent(answer="coded")
    team = TieredAgentTeam(
        supervisor=sup,
        workers=[
            AgentRole("researcher", w1, "Investigate things."),
            AgentRole("coder", w2, "Write code."),
        ],
    )
    # Both workers should now be tools on the supervisor's registry.
    names = {t.name for t in sup.registry.list_tools()}
    assert names == {"researcher", "coder"}


def test_team_workers_are_callable_through_registry():
    sup = _StubSupervisor()
    w = _StubAgent(answer="searched")
    team = TieredAgentTeam(
        supervisor=sup,
        workers=[AgentRole("researcher", w, "Find facts.")],
    )
    tool = sup.registry.get("researcher")
    assert tool is not None
    assert tool(task="anything") == "searched"


def test_team_rejects_empty_workers():
    sup = _StubSupervisor()
    with pytest.raises(ValueError, match="at least one worker"):
        TieredAgentTeam(supervisor=sup, workers=[])


def test_team_rejects_duplicate_worker_names():
    sup = _StubSupervisor()
    w = _StubAgent()
    with pytest.raises(ValueError, match="duplicate"):
        TieredAgentTeam(
            supervisor=sup,
            workers=[
                AgentRole("dup", w, "x"),
                AgentRole("dup", w, "y"),
            ],
        )


def test_team_rejects_supervisor_without_registry():
    class _NoRegistry:
        @property
        def metrics(self) -> Optional[AgentMetrics]:
            return None

        def run(self, task: str) -> AgentResult:
            return AgentResult(answer="", steps=[], iterations=0, success=True)

        def stream(self, task: str) -> Iterator[AgentEvent]:
            yield AgentEvent(type=EventType.ANSWER, content="")

    with pytest.raises(TypeError, match="registry"):
        TieredAgentTeam(
            supervisor=_NoRegistry(),
            workers=[AgentRole("w", _StubAgent(), "x")],
        )


def test_team_run_delegates_to_supervisor():
    sup = _StubSupervisor()
    team = TieredAgentTeam(supervisor=sup, workers=[AgentRole("w", _StubAgent(), "x")])
    result = team.run("anything")
    assert result.answer == "supervisor done"


def test_team_forward_events_propagates_to_each_worker_tool():
    """A team-level forward_events callback should be the callback used
    when each worker's tool is invoked."""
    sup = _StubSupervisor()
    sub_events = [AgentEvent(type=EventType.THOUGHT, content="t")]
    w = _StubAgent(answer="ok", events=list(sub_events))
    captured: List[AgentEvent] = []
    team = TieredAgentTeam(
        supervisor=sup,
        workers=[AgentRole("worker", w, "Do work.")],
        forward_events=captured.append,
    )
    # Directly invoke the worker tool to verify the callback wires up.
    tool = sup.registry.get("worker")
    assert tool is not None
    tool(task="any")
    # 1 thought + 1 answer from stub.stream() => 2 events forwarded.
    assert len(captured) == 2
    assert all(ev.source == "worker" for ev in captured)
