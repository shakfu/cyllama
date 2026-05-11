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


# ===========================================================================
# Gap #1 -- ReflectionLoop (Reflexion pattern)
# ===========================================================================


class _ScriptedAgent:
    """Stub agent that yields a scripted sequence of answers across calls."""

    def __init__(self, answers: List[str]) -> None:
        self._answers = list(answers)
        self._call = 0
        self._metrics: Optional[AgentMetrics] = None

    @property
    def metrics(self) -> Optional[AgentMetrics]:
        return self._metrics

    def _next(self) -> str:
        if self._call >= len(self._answers):
            return self._answers[-1] if self._answers else ""
        a = self._answers[self._call]
        self._call += 1
        return a

    def run(self, task: str) -> AgentResult:
        ans = self._next()
        return AgentResult(answer=ans, steps=[], iterations=1, success=True)

    def stream(self, task: str) -> Iterator[AgentEvent]:
        ans = self._next()
        yield AgentEvent(type=EventType.ANSWER, content=ans)


def test_reflection_loop_returns_on_critic_acceptance():
    """First-pass acceptance -- worker draft becomes the final answer."""
    from cyllama.agents import ReflectionLoop

    worker = _ScriptedAgent(["draft v1"])
    critic = _ScriptedAgent(["ACCEPT"])

    loop = ReflectionLoop(worker, critic, max_attempts=3)
    result = loop.run("any task")
    assert result.answer == "draft v1"
    assert result.success is True


def test_reflection_loop_iterates_until_critic_accepts():
    """Critic rejects v1, accepts v2 -- v2 is returned."""
    from cyllama.agents import ReflectionLoop

    worker = _ScriptedAgent(["draft v1", "draft v2"])
    critic = _ScriptedAgent(["please revise: more detail", "ACCEPT"])

    loop = ReflectionLoop(worker, critic, max_attempts=3)
    result = loop.run("any task")
    assert result.answer == "draft v2"


def test_reflection_loop_returns_last_draft_when_max_attempts_exhausted():
    """All critic responses are revisions -- the final draft is returned."""
    from cyllama.agents import ReflectionLoop

    worker = _ScriptedAgent(["draft 1", "draft 2", "draft 3"])
    critic = _ScriptedAgent(["needs work", "still wrong", "almost there"])

    loop = ReflectionLoop(worker, critic, max_attempts=3)
    result = loop.run("any task")
    assert result.answer == "draft 3"
    # Critic never accepted -> loop_detected stays True (using the existing
    # metrics field as a marker for "no acceptance signal").
    assert loop.metrics is not None
    assert loop.metrics.loop_detected is True


def test_reflection_loop_stream_annotates_source():
    """Streaming surfaces worker + critic events with source labels."""
    from cyllama.agents import ReflectionLoop

    worker = _ScriptedAgent(["draft v1"])
    critic = _ScriptedAgent(["ACCEPT"])
    loop = ReflectionLoop(worker, critic, max_attempts=2)

    events = list(loop.stream("any task"))
    worker_events = [e for e in events if e.source == "worker"]
    critic_events = [e for e in events if e.source == "critic"]
    assert len(worker_events) >= 1
    assert len(critic_events) >= 1
    # Final ANSWER event has no source (it's the loop's own event, not
    # forwarded from worker/critic).
    final = events[-1]
    assert final.type == EventType.ANSWER
    assert final.content == "draft v1"


def test_reflection_loop_custom_acceptance_marker():
    """Custom acceptance string is honoured."""
    from cyllama.agents import ReflectionLoop

    worker = _ScriptedAgent(["draft"])
    critic = _ScriptedAgent(["LOOKS GOOD"])

    loop = ReflectionLoop(worker, critic, max_attempts=3, acceptance_marker="LOOKS GOOD")
    result = loop.run("any task")
    assert result.answer == "draft"


def test_reflection_loop_custom_revision_template():
    """Custom revision template controls the next-iteration task."""
    from cyllama.agents import ReflectionLoop

    captured: List[str] = []

    class _Capture:
        @property
        def metrics(self):
            return None

        def run(self, task: str) -> AgentResult:
            captured.append(task)
            return AgentResult(answer="ok", steps=[], iterations=1, success=True)

        def stream(self, task: str):
            captured.append(task)
            yield AgentEvent(type=EventType.ANSWER, content="ok")

    worker = _Capture()
    critic = _ScriptedAgent(["revise pls", "ACCEPT"])

    def template(orig: str, draft: str, critique: str) -> str:
        return f"REVISE: {orig} | DRAFT: {draft} | CRIT: {critique}"

    loop = ReflectionLoop(
        worker,  # type: ignore[arg-type]
        critic,
        max_attempts=2,
        revision_template=template,
    )
    loop.run("the task")
    # First call uses original task; second uses the template.
    assert captured[0] == "the task"
    assert "REVISE: the task" in captured[1]
    assert "DRAFT: ok" in captured[1]
    assert "CRIT: revise pls" in captured[1]


# ===========================================================================
# Gap #2 -- rag_as_tool
# ===========================================================================


class _StubRAGHit:
    """Mimics rag.types.SearchResult shape."""

    def __init__(self, text: str, score: float, metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.score = score
        self.metadata = metadata or {}


class _StubRAG:
    """Minimal RAG-like stub for composition tests."""

    def __init__(self, hits: List[_StubRAGHit]) -> None:
        self._hits = hits
        self.search_calls: List[tuple] = []

    def search(self, query: str, k: int = 5) -> List[_StubRAGHit]:
        self.search_calls.append((query, k))
        return self._hits[:k]


def test_rag_as_tool_returns_callable_tool():
    from cyllama.agents import rag_as_tool

    rag = _StubRAG([_StubRAGHit("alpha", 0.9), _StubRAGHit("beta", 0.7)])
    t = rag_as_tool(rag, description="Search project docs.")
    assert t.name == "search_kb"
    assert t.description == "Search project docs."
    assert t.parameters["required"] == ["query"]
    # coerce=False because we build typed kwargs ourselves.
    assert t.coerce is False


def test_rag_as_tool_renders_hits_with_scores():
    from cyllama.agents import rag_as_tool

    rag = _StubRAG([_StubRAGHit("alpha", 0.9), _StubRAGHit("beta", 0.7)])
    t = rag_as_tool(rag, top_k=2)
    out = t(query="anything")
    assert "[0.900] alpha" in out
    assert "[0.700] beta" in out


def test_rag_as_tool_deduplicates_repeated_text():
    """Default formatter drops repeated text fragments."""
    from cyllama.agents import rag_as_tool

    rag = _StubRAG(
        [
            _StubRAGHit("same content", 0.9),
            _StubRAGHit("same content", 0.85),
            _StubRAGHit("unique content", 0.7),
        ]
    )
    t = rag_as_tool(rag, top_k=3)
    out = t(query="x")
    # "same content" appears once even though the store has two copies.
    assert out.count("same content") == 1
    assert "unique content" in out


def test_rag_as_tool_empty_result():
    from cyllama.agents import rag_as_tool

    rag = _StubRAG([])
    t = rag_as_tool(rag)
    assert t(query="anything") == "(no results)"


def test_rag_as_tool_passes_top_k_to_search():
    from cyllama.agents import rag_as_tool

    rag = _StubRAG([_StubRAGHit(f"hit-{i}", 0.5) for i in range(20)])
    t = rag_as_tool(rag, top_k=3)
    t(query="anything")
    assert rag.search_calls[-1] == ("anything", 3)


def test_rag_as_tool_missing_query_raises():
    from cyllama.agents import rag_as_tool

    rag = _StubRAG([])
    t = rag_as_tool(rag)
    with pytest.raises(ValueError, match="query"):
        t()


def test_rag_as_tool_custom_formatter():
    from cyllama.agents import rag_as_tool

    rag = _StubRAG([_StubRAGHit("foo", 0.5), _StubRAGHit("bar", 0.4)])
    t = rag_as_tool(rag, formatter=lambda hits: " | ".join(h.text for h in hits))
    assert t(query="x") == "foo | bar"


def test_rag_as_tool_rejects_object_without_method():
    from cyllama.agents import rag_as_tool

    class _NoSearch:
        pass

    t = rag_as_tool(_NoSearch(), description="x")
    with pytest.raises(AttributeError, match="no callable"):
        t(query="anything")


# ===========================================================================
# Gap #4 -- plan_and_execute
# ===========================================================================


def test_plan_and_execute_runs_steps_sequentially():
    from cyllama.agents import plan_and_execute

    planner = _ScriptedAgent(['["step A", "step B", "step C"]'])
    executed: List[str] = []

    class _Executor:
        @property
        def metrics(self):
            return None

        def run(self, task: str) -> AgentResult:
            executed.append(task)
            return AgentResult(answer=f"done({task})", steps=[], iterations=1, success=True)

        def stream(self, task):
            yield AgentEvent(type=EventType.ANSWER, content=f"done({task})")

    results = plan_and_execute(planner, _Executor(), "do the thing")
    assert executed == ["step A", "step B", "step C"]
    assert [r.answer for r in results] == ["done(step A)", "done(step B)", "done(step C)"]


def test_plan_and_execute_parses_steps_key():
    from cyllama.agents import plan_and_execute

    planner = _ScriptedAgent(['{"steps": ["one", "two"]}'])
    seen: List[str] = []

    class _Exec:
        @property
        def metrics(self):
            return None

        def run(self, task: str) -> AgentResult:
            seen.append(task)
            return AgentResult(answer="ok", steps=[], iterations=1, success=True)

        def stream(self, task):
            yield AgentEvent(type=EventType.ANSWER, content="ok")

    plan_and_execute(planner, _Exec(), "x")
    assert seen == ["one", "two"]


def test_plan_and_execute_falls_back_to_newline_split():
    """Non-JSON planner output is split on newlines."""
    from cyllama.agents import plan_and_execute

    planner = _ScriptedAgent(["1. first step\n2. second step\n* third step"])

    seen: List[str] = []

    class _Exec:
        @property
        def metrics(self):
            return None

        def run(self, task: str) -> AgentResult:
            seen.append(task)
            return AgentResult(answer="ok", steps=[], iterations=1, success=True)

        def stream(self, task):
            yield AgentEvent(type=EventType.ANSWER, content="ok")

    plan_and_execute(planner, _Exec(), "x")
    # Numbering and bullet prefixes are stripped.
    assert seen == ["first step", "second step", "third step"]


def test_plan_and_execute_stops_on_error_by_default():
    from cyllama.agents import plan_and_execute

    planner = _ScriptedAgent(['["a", "b", "c"]'])

    class _Exec:
        def __init__(self):
            self.n = 0

        @property
        def metrics(self):
            return None

        def run(self, task: str) -> AgentResult:
            self.n += 1
            if self.n == 2:
                return AgentResult(answer="", steps=[], iterations=1, success=False, error="boom")
            return AgentResult(answer="ok", steps=[], iterations=1, success=True)

        def stream(self, task):
            yield AgentEvent(type=EventType.ANSWER, content="ok")

    e = _Exec()
    results = plan_and_execute(planner, e, "x")
    # First step succeeds, second fails -> stop.
    assert len(results) == 2
    assert results[0].success
    assert not results[1].success


def test_plan_and_execute_continues_on_error_when_disabled():
    from cyllama.agents import plan_and_execute

    planner = _ScriptedAgent(['["a", "b", "c"]'])

    class _Exec:
        def __init__(self):
            self.n = 0

        @property
        def metrics(self):
            return None

        def run(self, task: str) -> AgentResult:
            self.n += 1
            ok = self.n != 2
            return AgentResult(answer="x", steps=[], iterations=1, success=ok)

        def stream(self, task):
            yield AgentEvent(type=EventType.ANSWER, content="x")

    results = plan_and_execute(planner, _Exec(), "x", stop_on_error=False)
    assert len(results) == 3


def test_plan_and_execute_failed_planner_returns_planner_result():
    """If the planner itself fails, the failure is returned as-is."""
    from cyllama.agents import plan_and_execute

    class _BadPlanner:
        @property
        def metrics(self):
            return None

        def run(self, task):
            return AgentResult(answer="", steps=[], iterations=0, success=False, error="planner crashed")

        def stream(self, task):
            yield AgentEvent(type=EventType.ERROR, content="planner crashed")

    class _Exec:
        @property
        def metrics(self):
            return None

        def run(self, task):
            return AgentResult(answer="x", steps=[], iterations=1, success=True)

        def stream(self, task):
            yield AgentEvent(type=EventType.ANSWER, content="x")

    results = plan_and_execute(_BadPlanner(), _Exec(), "x")
    assert len(results) == 1
    assert not results[0].success


def test_plan_and_execute_custom_parser():
    from cyllama.agents import plan_and_execute

    planner = _ScriptedAgent(["step1;step2;step3"])
    seen: List[str] = []

    class _Exec:
        @property
        def metrics(self):
            return None

        def run(self, task):
            seen.append(task)
            return AgentResult(answer="ok", steps=[], iterations=1, success=True)

        def stream(self, task):
            yield AgentEvent(type=EventType.ANSWER, content="ok")

    plan_and_execute(planner, _Exec(), "x", plan_parser=lambda s: s.split(";"))
    assert seen == ["step1", "step2", "step3"]


# ===========================================================================
# Gap #5 -- mcp_agent_tool
# ===========================================================================


class _StubMcpClient:
    """Minimal McpClient-like stub for composition tests."""

    def __init__(self, results: Optional[Dict[str, Any]] = None) -> None:
        self.results = results or {}
        self.calls: List[tuple] = []

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        if name in self.results:
            return self.results[name]
        return f"mcp result for {name}: {arguments}"


def test_mcp_agent_tool_builds_namespaced_tool():
    from cyllama.agents import mcp_agent_tool

    client = _StubMcpClient()
    t = mcp_agent_tool(client, server_name="srv", agent_name="ag", description="d")
    assert t.name == "srv/ag"
    assert t.coerce is False
    assert t.parameters["required"] == ["task"]


def test_mcp_agent_tool_dispatches_via_client():
    from cyllama.agents import mcp_agent_tool

    client = _StubMcpClient(results={"srv/ag": "remote answer"})
    t = mcp_agent_tool(client, "srv", "ag", "d")
    out = t(task="do it")
    assert out == "remote answer"
    assert client.calls == [("srv/ag", {"task": "do it"})]


def test_mcp_agent_tool_missing_task_raises():
    from cyllama.agents import mcp_agent_tool

    t = mcp_agent_tool(_StubMcpClient(), "srv", "ag", "d")
    with pytest.raises(ValueError, match="task"):
        t()


def test_mcp_agent_tool_respects_custom_task_param():
    from cyllama.agents import mcp_agent_tool

    client = _StubMcpClient()
    t = mcp_agent_tool(client, "srv", "ag", "d", task_param="question")
    t(question="why?")
    assert client.calls[0][1] == {"question": "why?"}


def test_mcp_agent_tool_none_result_becomes_empty_string():
    from cyllama.agents import mcp_agent_tool

    client = _StubMcpClient(results={"srv/ag": None})
    t = mcp_agent_tool(client, "srv", "ag", "d")
    assert t(task="x") == ""


def test_mcp_agent_tool_timeout_propagates_to_Tool_field():
    from cyllama.agents import mcp_agent_tool

    t = mcp_agent_tool(_StubMcpClient(), "srv", "ag", "d", timeout=5.0)
    assert t.timeout == 5.0
