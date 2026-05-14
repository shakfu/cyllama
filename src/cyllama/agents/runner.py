"""Shared agent dispatcher.

Provides a kind-based entry point for streaming agent execution. Both
cyllama's interactive chat (:mod:`cyllama.llama.chat`) and external
orchestrators (e.g. the cyllama-desktop sidecar) call this so they
share one implementation of the plan-and-execute and reflection
loops.

The dispatcher takes an :class:`~cyllama.api.LLM`, the agent ``kind``,
and the user ``task``, and yields :class:`AgentEvent` objects. For
``"react"`` / ``"constrained"`` / ``"contract"`` it forwards
directly to that agent's ``.stream(task)``. For ``"plan"`` and
``"reflect"`` it composes multiple ReAct passes and stamps each
event's ``metadata["source"]`` with the phase (``"planner"``,
``"step-N"``, ``"worker-N"``, ``"critic-N"``, ``"final"``).
"""

from __future__ import annotations

import re
from typing import Any, Iterator, List, Optional

from .types import AgentEvent, EventType
from .tools import Tool
from .react import ReActAgent
from .constrained import ConstrainedAgent
from .contract import ContractAgent

AGENT_KINDS = ("react", "constrained", "contract", "plan", "reflect")

DEFAULT_PLANNER_PROMPT = (
    "You are a planner. Given a task, output a numbered list of concrete "
    "steps, one per line. Do not execute anything; just plan."
)
DEFAULT_CRITIC_PROMPT = (
    "You are a careful reviewer. Read the draft answer the user supplies. "
    "If it is correct and complete, reply with exactly 'ACCEPT' (one word). "
    "Otherwise list the specific issues that need to be fixed in the next "
    "revision -- be concrete and brief."
)
DEFAULT_CRITIQUE_PREFIX = "Critique this draft:"


def _parse_plan(answer: str) -> List[str]:
    """Newline-split a planner answer into step strings, stripping bullets/numbering."""
    out: List[str] = []
    for line in (answer or "").splitlines():
        s = re.sub(r"^\s*(?:[-*+]|\d+[\.\)])\s+", "", line.strip())
        if s:
            out.append(s)
    return out


def _revision_task(task: str, draft: str, critique: str) -> str:
    """Build the next worker task in a reflection loop after a non-accepting critique."""
    return f"{task}\n\nYour previous attempt:\n{draft}\n\nCritic feedback (please address):\n{critique}"


def _tag(ev: AgentEvent, source: str) -> AgentEvent:
    md = dict(ev.metadata or {})
    md["source"] = source
    return AgentEvent(type=ev.type, content=ev.content, metadata=md)


def stream_agent(
    kind: str,
    llm: Any,
    task: str,
    *,
    tools: Optional[List[Tool]] = None,
    system_prompt: Optional[str] = None,
    max_iterations: int = 10,
    # plan-specific
    max_steps: int = 10,
    planner_system_prompt: Optional[str] = None,
    executor_system_prompt: Optional[str] = None,
    stop_on_error: bool = True,
    # reflect-specific
    max_attempts: int = 3,
    worker_system_prompt: Optional[str] = None,
    critic_system_prompt: Optional[str] = None,
    acceptance_marker: str = "ACCEPT",
    critique_prefix: Optional[str] = None,
    **agent_kwargs: Any,
) -> Iterator[AgentEvent]:
    """Stream events from an agent of the requested ``kind``.

    ``kind`` is one of ``AGENT_KINDS``. Unknown kinds raise
    ``ValueError``. For composed kinds (``"plan"``, ``"reflect"``)
    the final aggregated result is emitted as an ``EventType.ANSWER``
    event with ``metadata["source"] == "final"`` plus phase-specific
    metadata (``plan`` / ``attempts`` / ``accepted``).
    """
    tools = tools or []

    # Tool-using kinds are structurally pointless without a tool registry:
    # the model will produce `Action: <something>` for an empty tool set
    # and the run will either parse-error or loop-detect. Fail fast with
    # a clear message so callers know to register tools (or use plan /
    # reflect, which are orchestration patterns over the LLM itself and
    # don't require tools).
    if not tools and kind in {"react", "constrained", "contract"}:
        yield AgentEvent(
            type=EventType.ANSWER,
            content=(
                f"No tools are registered for the '{kind}' agent. "
                "Register tools when constructing the agent, or ask a "
                "direct question without /agent."
            ),
        )
        return

    if kind == "react":
        agent: Any = ReActAgent(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            **agent_kwargs,
        )
        yield from agent.stream(task)
        return

    if kind == "constrained":
        agent = ConstrainedAgent(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            **agent_kwargs,
        )
        yield from agent.stream(task)
        return

    if kind == "contract":
        agent = ContractAgent(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            **agent_kwargs,
        )
        yield from agent.stream(task)
        return

    if kind == "plan":
        planner = ReActAgent(
            llm=llm,
            tools=[],
            system_prompt=planner_system_prompt or DEFAULT_PLANNER_PROMPT,
            max_iterations=max_iterations,
            verbose=False,
        )
        plan_answer: Optional[str] = None
        for ev in planner.stream(task):
            if ev.type == EventType.ANSWER:
                plan_answer = ev.content
            yield _tag(ev, "planner")
        steps = _parse_plan(plan_answer or "")[:max_steps]
        if not steps:
            yield AgentEvent(
                type=EventType.ANSWER,
                content=plan_answer or "",
                metadata={"source": "final", "plan": []},
            )
            return
        step_summaries: List[str] = []
        for idx, step in enumerate(steps, start=1):
            executor = ReActAgent(
                llm=llm,
                tools=tools,
                system_prompt=executor_system_prompt,
                max_iterations=max_iterations,
                verbose=False,
            )
            step_answer: Optional[str] = None
            for ev in executor.stream(step):
                if ev.type == EventType.ANSWER:
                    step_answer = ev.content
                yield _tag(ev, f"step-{idx}")
            step_summaries.append(f"{idx}. {step}\n   -> {step_answer or ''}")
            if step_answer is None and stop_on_error:
                break
        yield AgentEvent(
            type=EventType.ANSWER,
            content="\n".join(step_summaries),
            metadata={"source": "final", "plan": steps},
        )
        return

    if kind == "reflect":
        current = task
        last_draft = ""
        accepted = False
        marker = (acceptance_marker or "ACCEPT").upper()
        cprefix = critique_prefix or DEFAULT_CRITIQUE_PREFIX
        attempts = 0
        for n in range(1, max_attempts + 1):
            attempts = n
            worker = ReActAgent(
                llm=llm,
                tools=tools,
                system_prompt=worker_system_prompt,
                max_iterations=max_iterations,
                verbose=False,
            )
            draft: Optional[str] = None
            for ev in worker.stream(current):
                if ev.type == EventType.ANSWER:
                    draft = ev.content
                yield _tag(ev, f"worker-{n}")
            if draft is None:
                yield AgentEvent(
                    type=EventType.ERROR,
                    content="worker produced no answer",
                    metadata={"source": f"worker-{n}"},
                )
                break
            last_draft = draft

            critic = ReActAgent(
                llm=llm,
                tools=[],
                system_prompt=critic_system_prompt or DEFAULT_CRITIC_PROMPT,
                max_iterations=max_iterations,
                verbose=False,
            )
            critique: Optional[str] = None
            for ev in critic.stream(f"{cprefix}\n\n{draft}"):
                if ev.type == EventType.ANSWER:
                    critique = ev.content
                yield _tag(ev, f"critic-{n}")
            if critique and marker in critique.upper():
                accepted = True
                break
            current = _revision_task(task, draft, critique or "")
        yield AgentEvent(
            type=EventType.ANSWER,
            content=last_draft,
            metadata={"source": "final", "attempts": attempts, "accepted": accepted},
        )
        return

    raise ValueError(f"unknown agent kind: {kind!r} (expected one of {AGENT_KINDS})")
