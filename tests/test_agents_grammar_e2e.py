"""End-to-end verification that ConstrainedAgent's GBNF grammar actually
constrains real-model output to valid tool-call JSON.

The existing `tests/test_agents_constrained.py` suite uses MockLLM, which
short-circuits the grammar pipeline entirely — so a green ConstrainedAgent
test does not prove that grammar enforcement works against a real sampler.
This module fills that gap with one slow integration test against the
standard `Llama-3.2-1B-Instruct-Q8_0.gguf` fixture.

What it asserts
---------------
The grammar-enforcement claim is "every generation step produces JSON
conforming to the answer-or-tool envelope". It is *not* about which
branch of the envelope (`tool_call` vs `answer`) the model picks —
that's a behavioral choice that the small 1B test model makes
inconsistently. The test therefore:

1. Runs the agent and collects all events.
2. Asserts no parse-failure ERROR events were emitted. If the grammar
   weren't enforced, the upstream JSON parser would fail on free text
   and we'd see `Failed to parse JSON response` ERRORs.
3. Asserts at least one envelope-conforming event landed (ACTION or
   ANSWER), so we know generation actually ran.
4. For every ACTION event, validates `metadata` against the tool
   registry — confirming `tool_name` is registered and `tool_args` only
   contains declared parameter names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from cyllama.agents import ConstrainedAgent, Tool, tool
from cyllama.agents.types import EventType


pytestmark = [pytest.mark.integration, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Tools with non-trivial parameter schemas (int + Literal enum + str)
# ---------------------------------------------------------------------------


@tool
def fetch_rows(table: str, limit: int) -> str:
    """Fetch up to `limit` rows from a table by name."""
    return f"[fetched {limit} rows from {table}]"


@tool
def set_log_level(level: Literal["debug", "info", "warn", "error"]) -> str:
    """Set the application log level."""
    return f"log level set to {level}"


TOOLS: list[Tool] = [fetch_rows, set_log_level]


# ---------------------------------------------------------------------------
# Schema-shape predicates
# ---------------------------------------------------------------------------


def _validate_action_event(meta: dict, tools_by_name: dict[str, Tool]) -> None:
    """Assert an ACTION event's metadata tracks the registry + schema."""
    name = meta.get("tool_name")
    args = meta.get("tool_args")
    assert isinstance(name, str), f"tool_name must be str, got {type(name).__name__}: {meta!r}"
    assert isinstance(args, dict), f"tool_args must be dict, got {type(args).__name__}: {meta!r}"
    assert name in tools_by_name, f"unknown tool: {name!r} (registry: {list(tools_by_name)})"
    declared = tools_by_name[name].parameters.get("properties", {})
    for arg_name in args:
        assert arg_name in declared, f"tool '{name}' received undeclared arg {arg_name!r}; declared: {list(declared)}"


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def test_grammar_actually_constrains_real_inference(model_path: str) -> None:
    """Run ConstrainedAgent against a real model and assert grammar holds.

    The 1B model is weak — it will not necessarily solve the task — but the
    grammar should still hold for every ACTION it emits. Schema enforcement
    is independent of model capability; that's the whole point of GBNF.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        pytest.skip(f"test model not present: {model_path} (run `make download`)")

    from cyllama import LLM, GenerationConfig

    # Greedy sampling for determinism. max_tokens kept modest — we only
    # need a handful of ACTION events to exercise the grammar.
    cfg = GenerationConfig(temperature=0.0, max_tokens=192)
    llm = LLM(model_path, config=cfg)
    try:
        agent = ConstrainedAgent(
            llm=llm,
            tools=TOOLS,
            max_iterations=4,
            allow_reasoning=False,
            use_cache=True,
        )
        tools_by_name = {t.name: t for t in TOOLS}

        events = list(agent.stream("Fetch 3 rows from the users table."))

        action_events = [e for e in events if e.type == EventType.ACTION]
        answer_events = [e for e in events if e.type == EventType.ANSWER]
        error_events = [e for e in events if e.type == EventType.ERROR]

        # The load-bearing assertion: no JSON-parse ERROR events. If the
        # grammar weren't enforced, the upstream parser would fail on free
        # text and emit these.
        parse_errors = [
            e for e in error_events if "Failed to parse JSON" in e.content or "JSONDecodeError" in e.content
        ]
        assert not parse_errors, (
            f"agent recorded JSON parse failures despite grammar enforcement: {[e.content[:120] for e in parse_errors]}"
        )

        # Sanity: at least one envelope-shaped event must have landed.
        assert action_events or answer_events, (
            "ConstrainedAgent emitted neither ACTION nor ANSWER — "
            f"generation likely failed. Events: {[(e.type.name, e.content[:60]) for e in events]}"
        )

        # When the model did pick tool_call, validate against the registry.
        for ev in action_events:
            _validate_action_event(ev.metadata, tools_by_name)
    finally:
        llm.close()
