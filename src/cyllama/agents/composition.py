"""Multi-agent composition primitives.

This module hosts the building blocks that let one agent invoke another
and the canonical pattern helpers built on top of them. The design intent
is to *not* introduce a new orchestration engine -- multi-agent patterns
compose out of small primitives:

**Base primitives:**

1. :func:`agent_as_tool` -- wrap any :class:`AgentProtocol` as a
   :class:`Tool` so it can be invoked through the ordinary ReAct
   dispatch path. Zero changes to existing agent loops.
2. :class:`TieredAgentTeam` -- ergonomic shorthand for supervisor /
   worker setups where roles run on different LLMs (planner + workers
   with capability-tiered models).

**Pattern helpers (each ~50 LoC over the primitives):**

3. :class:`ReflectionLoop` -- worker emits draft, critic emits
   ACCEPT/revision, loop up to N times. The Reflexion pattern.
4. :func:`plan_and_execute` -- planner emits structured task list,
   executor runs each step sequentially. The Plan-and-Execute pattern.
5. :func:`mcp_agent_tool` -- cross-process analog of :func:`agent_as_tool`
   wrapping a remote agent exposed via MCP.
6. :func:`rag_as_tool` -- bridges the :mod:`cyllama.rag` subsystem to
   the agent layer as a canned knowledge-base tool.

Event nesting: when an agent is invoked via :func:`agent_as_tool`, its
events are re-emitted in the supervisor's stream with
``AgentEvent.source`` set to the inner agent's name and
``parent_event_id`` linked to the supervisor's ACTION event. Streaming
UIs see a hierarchical trace; legacy consumers that ignore those fields
behave unchanged.

See ``docs/agents/patterns.md`` for the full pattern catalog and the
ones the framework intentionally does not support.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

from .tools import Tool
from .types import AgentEvent, AgentMetrics, AgentProtocol, AgentResult, EventType


__all__ = [
    # Base primitives
    "agent_as_tool",
    "AgentRole",
    "TieredAgentTeam",
    # Pattern helpers
    "ReflectionLoop",
    "plan_and_execute",
    "mcp_agent_tool",
    "rag_as_tool",
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


# ---------------------------------------------------------------------------
# Pattern helpers
# ---------------------------------------------------------------------------


class ReflectionLoop:
    """Worker + critic loop -- the Reflexion pattern.

    A worker agent produces a draft answer; a critic agent reviews it and
    either accepts (returns the draft) or returns revision feedback that
    the worker incorporates on the next pass. Loops up to ``max_attempts``
    times, returning the final draft regardless of acceptance.

    The critic is just another :class:`AgentProtocol` -- typically a
    :class:`ReActAgent` with a different system prompt focused on
    critique. The acceptance check is a simple substring match on the
    critic's answer (case-insensitive); pass a custom ``acceptance_marker``
    if you need different framing than ``"ACCEPT"``.

    Example::

        from cyllama import LLM
        from cyllama.agents import ReActAgent, ReflectionLoop

        llm = LLM("model.gguf")
        worker = ReActAgent(
            llm=llm, tools=[...],
            system_prompt="You are a careful coder. ...",
        )
        critic = ReActAgent(
            llm=llm, tools=[],
            system_prompt=(
                "You are a code reviewer. If the draft is correct and "
                "complete, respond with 'ACCEPT'. Otherwise list specific "
                "issues for the next revision."
            ),
        )

        loop = ReflectionLoop(worker, critic, max_attempts=3)
        result = loop.run("Implement quicksort with a 3-way partition.")

    Reference: *Reflexion: Language Agents with Verbal Reinforcement
    Learning* (Shinn et al., 2023).
    """

    def __init__(
        self,
        worker: AgentProtocol,
        critic: AgentProtocol,
        max_attempts: int = 3,
        acceptance_marker: str = "ACCEPT",
        critique_prefix: str = "Critique this draft:",
        revision_template: Optional[Callable[[str, str, str], str]] = None,
    ) -> None:
        """Initialize the loop.

        Args:
            worker: Agent that produces drafts. Receives the task (and
                progressively-enriched revision prompts).
            critic: Agent that reviews drafts. Returns either text
                containing ``acceptance_marker`` to accept, or revision
                feedback to be folded into the next worker call.
            max_attempts: Hard cap on iterations. The final draft is
                returned even if the critic never accepts.
            acceptance_marker: Substring search (case-insensitive) on the
                critic's answer to detect acceptance. Default ``"ACCEPT"``.
            critique_prefix: Prefix added to the worker's draft when
                asking the critic. Override for non-English prompts or
                different framing.
            revision_template: Callable ``(task, draft, critique) -> str``
                producing the next-iteration task. Defaults to a simple
                concatenation that includes the prior draft and the
                critic's feedback.
        """
        self.worker = worker
        self.critic = critic
        self.max_attempts = max_attempts
        self.acceptance_marker = acceptance_marker.upper()
        self.critique_prefix = critique_prefix
        self.revision_template = revision_template or _default_revision_template
        self._metrics: Optional[AgentMetrics] = None

    @property
    def metrics(self) -> Optional[AgentMetrics]:
        """Metrics from the most recent ``run`` / ``stream`` call.

        Aggregates worker iterations + critic iterations; ``loop_detected``
        is True if the critic never produced an acceptance.
        """
        return self._metrics

    def run(self, task: str) -> AgentResult:
        """Run the loop to completion, return the final draft as an
        :class:`AgentResult`."""
        events: List[AgentEvent] = []
        for ev in self.stream(task):
            events.append(ev)
        answer = ""
        success = True
        error: Optional[str] = None
        for ev in reversed(events):
            if ev.type == EventType.ANSWER:
                answer = ev.content
                break
            if ev.type == EventType.ERROR:
                success = False
                error = ev.content
                break
        return AgentResult(
            answer=answer,
            steps=events,
            iterations=self._metrics.iterations if self._metrics else 0,
            success=success,
            error=error,
            metrics=self._metrics,
        )

    def stream(self, task: str) -> Iterator[AgentEvent]:
        """Run the loop, yielding events from each worker + critic pass.

        Worker events carry ``source="worker"``; critic events carry
        ``source="critic"``. A final ANSWER event is emitted with the
        accepted (or last) draft.
        """
        start = time.perf_counter()
        metrics = AgentMetrics()
        current_task = task
        last_draft = ""
        accepted = False

        for attempt in range(self.max_attempts):
            parent_id = uuid.uuid4().hex
            # Worker pass.
            for ev in self.worker.stream(current_task):
                ev.source = "worker"
                ev.parent_event_id = parent_id
                metrics.iterations += 1
                if ev.type == EventType.ACTION:
                    metrics.tool_calls += 1
                elif ev.type == EventType.ERROR:
                    metrics.error_count += 1
                if ev.type == EventType.ANSWER:
                    last_draft = ev.content
                yield ev

            if not last_draft:
                # Worker failed to produce anything; bail.
                yield AgentEvent(
                    type=EventType.ERROR,
                    content="ReflectionLoop: worker produced no answer",
                    metadata={"attempt": attempt + 1},
                )
                break

            # Critic pass.
            critic_task = f"{self.critique_prefix}\n\n{last_draft}"
            critic_parent = uuid.uuid4().hex
            critic_answer = ""
            for ev in self.critic.stream(critic_task):
                ev.source = "critic"
                ev.parent_event_id = critic_parent
                metrics.iterations += 1
                if ev.type == EventType.ACTION:
                    metrics.tool_calls += 1
                elif ev.type == EventType.ERROR:
                    metrics.error_count += 1
                if ev.type == EventType.ANSWER:
                    critic_answer = ev.content
                yield ev

            # Accept-check is a substring match on the critic's verdict.
            if self.acceptance_marker in critic_answer.upper():
                accepted = True
                break

            # Not accepted: prepare revision task for the next pass.
            current_task = self.revision_template(task, last_draft, critic_answer)

        metrics.loop_detected = not accepted
        metrics.total_time_ms = (time.perf_counter() - start) * 1000.0
        self._metrics = metrics

        # Final ANSWER event carries the last draft so callers that only
        # listen for ANSWER (rather than walking the stream) get the value.
        yield AgentEvent(
            type=EventType.ANSWER,
            content=last_draft,
            metadata={"accepted": accepted, "attempts": metrics.iterations and (attempt + 1)},
        )


def _default_revision_template(task: str, draft: str, critique: str) -> str:
    """Default next-iteration prompt: prior task + draft + critic feedback."""
    return f"{task}\n\nYour previous attempt:\n{draft}\n\nCritic feedback (please address):\n{critique}"


def plan_and_execute(
    planner: AgentProtocol,
    executor: AgentProtocol,
    task: str,
    plan_parser: Optional[Callable[[str], List[str]]] = None,
    stop_on_error: bool = True,
) -> List[AgentResult]:
    """Plan-and-Execute pattern.

    A planner agent emits a structured task list; the executor runs each
    step sequentially. Returns the list of per-step :class:`AgentResult`
    in order.

    The planner is typically a :class:`ConstrainedAgent` whose grammar
    forces a JSON list (or a :class:`ReActAgent` whose answer is parsed
    by a custom ``plan_parser``). The executor is any
    :class:`AgentProtocol` -- often a single :class:`ReActAgent` reused
    across all steps.

    Example::

        from cyllama.agents import ConstrainedAgent, ReActAgent, plan_and_execute

        planner = ConstrainedAgent(
            llm=planner_llm,
            tools=[],
            system_prompt="Emit a JSON list of step strings.",
        )
        executor = ReActAgent(llm=worker_llm, tools=[read, write])

        results = plan_and_execute(
            planner=planner,
            executor=executor,
            task="Refactor module X to use pattern Y.",
        )
        for step_result in results:
            print(step_result.answer)

    Args:
        planner: Agent that emits the plan. Its ``answer`` string is fed
            to ``plan_parser`` to produce a list of step strings.
        executor: Agent that runs each step. Called once per step with
            the step string as input.
        task: The top-level task description sent to the planner.
        plan_parser: Callable ``(planner_answer) -> List[step_str]``.
            Defaults to :func:`_default_plan_parser` which tries
            ``json.loads`` first (looking for a list at the top level or
            under a "steps"/"plan"/"tasks" key) and falls back to
            splitting on newlines.
        stop_on_error: If True (default), stop after the first step that
            returns ``success=False``. If False, run all steps regardless.

    Returns:
        List of ``AgentResult`` -- one per executed step. May be shorter
        than the plan if ``stop_on_error`` triggered.

    Reference: *Plan-and-Solve Prompting* (Wang et al., 2023).
    """
    parser = plan_parser or _default_plan_parser

    plan_result = planner.run(task)
    if not plan_result.success:
        return [plan_result]

    try:
        steps = parser(plan_result.answer)
    except Exception as e:
        return [
            AgentResult(
                answer="",
                steps=plan_result.steps,
                iterations=plan_result.iterations,
                success=False,
                error=f"plan_and_execute: failed to parse plan: {e}",
                metrics=plan_result.metrics,
            )
        ]

    results: List[AgentResult] = []
    for step in steps:
        result = executor.run(step)
        results.append(result)
        if stop_on_error and not result.success:
            break
    return results


def _default_plan_parser(text: str) -> List[str]:
    """Heuristic plan parser: try JSON list / {"steps": [...]}, fall
    back to newline-split.

    Recognised JSON shapes:
        ["step 1", "step 2", ...]
        {"steps": [...]}
        {"plan": [...]}
        {"tasks": [...]}
    """
    import json as _json

    text = text.strip()
    # Try JSON.
    try:
        parsed = _json.loads(text)
    except (ValueError, TypeError):
        parsed = None

    if isinstance(parsed, list):
        return [str(s).strip() for s in parsed if str(s).strip()]
    if isinstance(parsed, dict):
        for key in ("steps", "plan", "tasks"):
            if key in parsed and isinstance(parsed[key], list):
                return [str(s).strip() for s in parsed[key] if str(s).strip()]

    # Fallback: newline-separated lines, stripping common bullet/number prefixes.
    import re as _re

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned: List[str] = []
    for line in lines:
        # Strip leading bullets / list numbers.
        line = _re.sub(r"^\s*(?:[-*+]|\d+[\.\)])\s+", "", line)
        if line:
            cleaned.append(line)
    return cleaned


def mcp_agent_tool(
    client: Any,
    server_name: str,
    agent_name: str,
    description: str,
    task_param: str = "task",
    timeout: Optional[float] = None,
) -> Tool:
    """Cross-process analog of :func:`agent_as_tool`.

    Wraps an MCP-exposed agent endpoint as a local :class:`Tool` that
    dispatches via :class:`McpClient.call_tool`. The supervisor doesn't
    know or care whether the sub-agent is local Python, a remote Python
    service, or a totally different runtime -- as long as it speaks MCP.

    Args:
        client: An :class:`McpClient` instance with ``server_name``
            already connected.
        server_name: The MCP server name the agent lives behind.
        agent_name: The tool name exposed by the server.
        description: Tool description shown to the supervisor LLM.
        task_param: Name of the parameter passed to the remote agent.
            Default ``"task"``.
        timeout: Optional per-call timeout (seconds) enforced by the
            ordinary ``Tool.timeout`` mechanism. Network-level timeouts
            on the MCP transport are separate.

    Returns:
        A :class:`Tool` whose ``name`` is ``"{server_name}/{agent_name}"``
        -- the format the action parser supports for namespaced tools.

    Failure modes map onto the existing exception taxonomy:
    :class:`ToolTimeoutError` for the local budget,
    ``RuntimeError`` from MCP for network/remote failures (caught by
    the agent's generic exception handler and surfaced as ERROR).

    Example::

        from cyllama.agents import McpClient, McpServerConfig
        from cyllama.agents import ReActAgent, mcp_agent_tool

        client = McpClient([McpServerConfig(...)])
        client.connect_all()

        remote = mcp_agent_tool(
            client,
            server_name="research",
            agent_name="web_search",
            description="Search the web and return findings.",
        )
        agent = ReActAgent(llm=llm, tools=[remote])
    """
    full_name = f"{server_name}/{agent_name}"

    def _invoke(**kwargs: Any) -> str:
        task = kwargs.get(task_param)
        if not isinstance(task, str):
            raise ValueError(f"mcp_agent_tool({full_name!r}): missing required string argument {task_param!r}")
        result = client.call_tool(full_name, {task_param: task})
        return str(result) if result is not None else ""

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {task_param: {"type": "string"}},
        "required": [task_param],
    }
    return Tool(
        name=full_name,
        description=description,
        func=_invoke,
        parameters=schema,
        coerce=False,
        timeout=timeout,
    )


def rag_as_tool(
    rag: Any,
    name: str = "search_kb",
    description: str = "Search the knowledge base for relevant passages.",
    top_k: int = 5,
    query_param: str = "query",
    method: str = "search",
    formatter: Optional[Callable[[List[Any]], str]] = None,
) -> Tool:
    """Wrap a :class:`cyllama.rag.RAG` instance (or compatible object) as
    a :class:`Tool` so an agent can search the knowledge base.

    Bridges the RAG subsystem to the agent layer without forcing every
    user to hand-roll the same wrapper. The returned tool, when called
    with ``{<query_param>: "..."}``, invokes ``rag.<method>(query, k=top_k)``
    (the ``RAG.search`` API) and returns a formatted string suitable for
    an OBSERVATION event.

    Args:
        rag: A :class:`cyllama.rag.RAG` instance or any object exposing
            a compatible ``search(query, k)`` (or ``retrieve(query)``)
            method that returns a list of objects with ``text``,
            ``score``, and ``metadata`` attributes.
        name: Tool name. Default ``"search_kb"``.
        description: Tool description shown to the supervisor LLM.
        top_k: Number of results to return. Default 5.
        query_param: Name of the query keyword argument. Default ``"query"``.
        method: Which method on ``rag`` to call. Default ``"search"``
            (uses :meth:`cyllama.rag.RAG.search`). Use ``"retrieve"`` for
            the higher-level :meth:`cyllama.rag.RAG.retrieve` that
            applies the RAG pipeline config.
        formatter: Callable ``(results) -> str`` producing the
            observation string. Defaults to :func:`_default_rag_formatter`
            which emits one ``[score] text`` line per hit.

    Returns:
        A :class:`Tool` configured with the right schema and
        ``coerce=False``.

    Example::

        from cyllama.rag import RAG
        from cyllama.agents import ReActAgent, rag_as_tool

        kb = RAG.from_documents([...])  # or load existing
        search = rag_as_tool(kb, description="Search the project docs.")
        agent = ReActAgent(llm=llm, tools=[search])
    """
    fmt = formatter or _default_rag_formatter

    def _invoke(**kwargs: Any) -> str:
        query = kwargs.get(query_param)
        if not isinstance(query, str):
            raise ValueError(f"rag_as_tool({name!r}): missing required string argument {query_param!r}")
        bound = getattr(rag, method, None)
        if not callable(bound):
            raise AttributeError(f"rag_as_tool: rag object has no callable {method!r} method")
        # Try the canonical signature first, fall back to no-k for
        # retrieve()-style methods.
        try:
            results = bound(query, k=top_k)
        except TypeError:
            results = bound(query)
        return fmt(list(results))

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {query_param: {"type": "string"}},
        "required": [query_param],
    }
    return Tool(
        name=name,
        description=description,
        func=_invoke,
        parameters=schema,
        coerce=False,
    )


def _default_rag_formatter(results: List[Any]) -> str:
    """Render search hits as one ``[score] text`` line per result.

    Deduplicates by ``text`` (case-sensitive) so the agent doesn't see
    repeated content when the store contains near-duplicate entries.
    """
    if not results:
        return "(no results)"

    seen: set[str] = set()
    lines: List[str] = []
    for r in results:
        text = getattr(r, "text", str(r))
        if text in seen:
            continue
        seen.add(text)
        score = getattr(r, "score", None)
        if score is not None:
            lines.append(f"[{score:.3f}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines)
