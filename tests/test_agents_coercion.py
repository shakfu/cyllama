"""Tests for ``coerce_args`` — the tool-argument validation/coercion helper.

The agent dispatch path runs ``coerce_args`` before ``tool(**args)`` when
``tool.coerce`` is True (default). This module covers:

- Missing required args
- Unknown args
- Scalar coercion (str -> int / float / bool)
- Enum violations
- Pass-through of correctly-typed values
- The ``coerce=False`` opt-out
"""

from __future__ import annotations

from typing import Annotated, List, Literal

import pytest

from cyllama.agents import tool, ReActAgent, Tool
from cyllama.agents.tools import (
    Ge,
    Gt,
    Le,
    Lt,
    MaxLen,
    MinLen,
    MultipleOf,
    Pattern,
    ToolArgumentError,
    coerce_args,
)


# ---------------------------------------------------------------------------
# Fixture tools
# ---------------------------------------------------------------------------


@tool
def fetch(table: str, limit: int) -> str:
    """Fetch limit rows from table."""
    return f"{table}:{limit}"


@tool
def measure(value: float, unit: Literal["kg", "lb"]) -> str:
    """Measurement helper."""
    return f"{value}{unit}"


@tool
def toggle(enabled: bool) -> str:
    """Toggle a flag."""
    return f"flag={enabled}"


@tool
def collect(items: List[int]) -> int:
    """Sum a list of ints."""
    return sum(items)


@tool(coerce=False)
def loose(**kwargs: object) -> str:
    """Tool that opts out of coercion to accept arbitrary kwargs."""
    return repr(kwargs)


# ---------------------------------------------------------------------------
# Missing / unknown args
# ---------------------------------------------------------------------------


def test_missing_required_arg_raises() -> None:
    with pytest.raises(ToolArgumentError) as exc:
        coerce_args(fetch, {"table": "users"})
    assert "missing required" in exc.value.message
    assert "limit" in exc.value.message
    assert exc.value.tool_name == "fetch"


def test_unknown_arg_raises() -> None:
    with pytest.raises(ToolArgumentError) as exc:
        coerce_args(fetch, {"table": "users", "limit": 3, "extra": "nope"})
    assert "unknown argument" in exc.value.message
    assert "'extra'" in exc.value.message


# ---------------------------------------------------------------------------
# Scalar coercion
# ---------------------------------------------------------------------------


def test_str_to_int_coerced() -> None:
    out = coerce_args(fetch, {"table": "users", "limit": "5"})
    assert out == {"table": "users", "limit": 5}
    assert isinstance(out["limit"], int)


def test_int_passes_through_unchanged() -> None:
    out = coerce_args(fetch, {"table": "users", "limit": 5})
    assert out["limit"] == 5


def test_str_to_float_coerced() -> None:
    out = coerce_args(measure, {"value": "1.5", "unit": "kg"})
    assert out["value"] == pytest.approx(1.5)


def test_int_accepted_for_number_field() -> None:
    out = coerce_args(measure, {"value": 2, "unit": "kg"})
    assert out["value"] == 2


def test_uncoerceable_str_to_int_raises() -> None:
    with pytest.raises(ToolArgumentError) as exc:
        coerce_args(fetch, {"table": "users", "limit": "many"})
    assert "cannot coerce" in exc.value.message


def test_str_to_bool_truthy() -> None:
    for raw in ("true", "True", "1", "yes"):
        assert coerce_args(toggle, {"enabled": raw}) == {"enabled": True}


def test_str_to_bool_falsy() -> None:
    for raw in ("false", "False", "0", "no"):
        assert coerce_args(toggle, {"enabled": raw}) == {"enabled": False}


def test_bool_not_accepted_as_int() -> None:
    # bool is a subclass of int in Python — make sure we reject it.
    with pytest.raises(ToolArgumentError) as exc:
        coerce_args(fetch, {"table": "users", "limit": True})
    assert "expected integer" in exc.value.message


# ---------------------------------------------------------------------------
# Enum (Literal[...]) violations
# ---------------------------------------------------------------------------


def test_enum_violation_raises() -> None:
    with pytest.raises(ToolArgumentError) as exc:
        coerce_args(measure, {"value": 1.0, "unit": "stones"})
    assert "not in allowed values" in exc.value.message
    assert "'stones'" in exc.value.message


def test_enum_match_passes() -> None:
    out = coerce_args(measure, {"value": 1.0, "unit": "lb"})
    assert out["unit"] == "lb"


# ---------------------------------------------------------------------------
# Container types
# ---------------------------------------------------------------------------


def test_array_passes_through() -> None:
    out = coerce_args(collect, {"items": [1, 2, 3]})
    assert out["items"] == [1, 2, 3]


def test_array_rejects_non_list() -> None:
    with pytest.raises(ToolArgumentError) as exc:
        coerce_args(collect, {"items": "1,2,3"})
    assert "expected array" in exc.value.message


# ---------------------------------------------------------------------------
# Opt-out via coerce=False
# ---------------------------------------------------------------------------


def test_coerce_false_skips_helper_at_call_site() -> None:
    # coerce_args itself is not called when tool.coerce is False — that's
    # the agent's responsibility. Verify the flag is set on the tool.
    assert loose.coerce is False
    # The Tool is still callable with arbitrary kwargs.
    assert loose(anything=1, else_=2) == repr({"anything": 1, "else_": 2})


def test_coerce_true_by_default() -> None:
    assert fetch.coerce is True
    assert measure.coerce is True


# ---------------------------------------------------------------------------
# Agent-integration smoke (sync ReActAgent dispatch path)
# ---------------------------------------------------------------------------


class _StubLLM:
    """Minimal stub for ReActAgent.__init__ — we only need _execute_tool_raw."""

    def __init__(self) -> None:
        self.config = None


def test_react_execute_tool_raw_coerces() -> None:
    agent = ReActAgent(llm=_StubLLM(), tools=[fetch])  # type: ignore[arg-type]
    # The LLM would emit string-typed limits in JSON; coercion fixes it.
    result = agent._execute_tool_raw("fetch", {"table": "users", "limit": "7"})
    assert result == "users:7"


def test_react_execute_tool_raw_rejects_unknown_arg() -> None:
    agent = ReActAgent(llm=_StubLLM(), tools=[fetch])  # type: ignore[arg-type]
    with pytest.raises(ToolArgumentError):
        agent._execute_tool_raw("fetch", {"table": "users", "limit": 1, "bogus": 0})


def test_react_execute_tool_raw_opt_out_loose_kwargs() -> None:
    agent = ReActAgent(llm=_StubLLM(), tools=[loose])  # type: ignore[arg-type]
    # coerce=False — unknown kwargs are allowed through.
    result = agent._execute_tool_raw("loose", {"a": 1, "b": "two"})
    assert "a" in result and "b" in result


# ---------------------------------------------------------------------------
# Annotated[] constraint markers — Proposal #16
# ---------------------------------------------------------------------------


@tool
def page_query(
    page: Annotated[int, Ge(1), Le(1000)],
    page_size: Annotated[int, Gt(0), Lt(100), MultipleOf(10)],
) -> str:
    """Paginated query."""
    return f"page={page} size={page_size}"


@tool
def named_lookup(
    name: Annotated[str, MinLen(2), MaxLen(50), Pattern(r"^[A-Za-z][A-Za-z0-9_]*$")],
) -> str:
    """Identifier lookup."""
    return f"looked up {name}"


@tool
def tag_set(tags: Annotated[List[str], MinLen(1), MaxLen(5)]) -> int:
    """Tag aggregation with size bounds."""
    return len(tags)


# --- Schema generation ---


def test_annotated_int_produces_minimum_maximum() -> None:
    spec = page_query.parameters["properties"]["page"]
    assert spec["type"] == "integer"
    assert spec["minimum"] == 1
    assert spec["maximum"] == 1000


def test_annotated_exclusive_and_multipleof() -> None:
    spec = page_query.parameters["properties"]["page_size"]
    assert spec["type"] == "integer"
    assert spec["exclusiveMinimum"] == 0
    assert spec["exclusiveMaximum"] == 100
    assert spec["multipleOf"] == 10


def test_annotated_string_produces_length_and_pattern() -> None:
    spec = named_lookup.parameters["properties"]["name"]
    assert spec["type"] == "string"
    assert spec["minLength"] == 2
    assert spec["maxLength"] == 50
    assert spec["pattern"] == r"^[A-Za-z][A-Za-z0-9_]*$"


def test_annotated_list_produces_minitems_maxitems() -> None:
    spec = tag_set.parameters["properties"]["tags"]
    assert spec["type"] == "array"
    assert spec["minItems"] == 1
    assert spec["maxItems"] == 5


# --- Constraint enforcement (numeric) ---


def test_ge_violation() -> None:
    with pytest.raises(ToolArgumentError) as exc:
        coerce_args(page_query, {"page": 0, "page_size": 10})
    assert "minimum" in exc.value.message


def test_le_violation() -> None:
    with pytest.raises(ToolArgumentError) as exc:
        coerce_args(page_query, {"page": 1001, "page_size": 10})
    assert "maximum" in exc.value.message


def test_gt_exclusive_violation() -> None:
    with pytest.raises(ToolArgumentError) as exc:
        coerce_args(page_query, {"page": 1, "page_size": 0})
    assert "exclusiveMinimum" in exc.value.message


def test_lt_exclusive_violation() -> None:
    with pytest.raises(ToolArgumentError) as exc:
        coerce_args(page_query, {"page": 1, "page_size": 100})
    assert "exclusiveMaximum" in exc.value.message


def test_multipleof_violation() -> None:
    with pytest.raises(ToolArgumentError) as exc:
        coerce_args(page_query, {"page": 1, "page_size": 25})
    assert "multiple of" in exc.value.message


def test_constraints_run_after_coercion() -> None:
    # str "5" coerced to int 5, then bounds check sees 5 — pass.
    out = coerce_args(page_query, {"page": "5", "page_size": "20"})
    assert out == {"page": 5, "page_size": 20}


# --- Constraint enforcement (string) ---


def test_min_length_violation() -> None:
    with pytest.raises(ToolArgumentError) as exc:
        coerce_args(named_lookup, {"name": "a"})
    assert "minLength" in exc.value.message


def test_max_length_violation() -> None:
    with pytest.raises(ToolArgumentError) as exc:
        coerce_args(named_lookup, {"name": "x" * 60})
    assert "maxLength" in exc.value.message


def test_pattern_violation() -> None:
    with pytest.raises(ToolArgumentError) as exc:
        coerce_args(named_lookup, {"name": "1startswithdigit"})
    assert "pattern" in exc.value.message


def test_string_constraints_all_pass() -> None:
    out = coerce_args(named_lookup, {"name": "valid_name"})
    assert out == {"name": "valid_name"}


# --- Constraint enforcement (array) ---


def test_min_items_violation() -> None:
    with pytest.raises(ToolArgumentError) as exc:
        coerce_args(tag_set, {"tags": []})
    assert "minItems" in exc.value.message


def test_max_items_violation() -> None:
    with pytest.raises(ToolArgumentError) as exc:
        coerce_args(tag_set, {"tags": ["a", "b", "c", "d", "e", "f"]})
    assert "maxItems" in exc.value.message


def test_array_constraints_all_pass() -> None:
    out = coerce_args(tag_set, {"tags": ["a", "b"]})
    assert out == {"tags": ["a", "b"]}
