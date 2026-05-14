"""Tests for the Layer-B DAG workflow runtime (Phase 1).

Covers:
- Builder API (add_node / add_edge / add_conditional_edge / set_entry /
  set_exit) and compile-time validation.
- Topological-level execution + fan-in synchronization.
- Conditional routing (with and without edge_map, with the END sentinel).
- State merging across nodes (shallow, last-writer-wins within a level).
- Error capture from user node bodies into WorkflowResult.
- max_steps cycle bound.
- Per-node timeouts.
- Sync vs async node dispatch.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, TypedDict

import pytest

from cyllama.agents.workflow import (
    Workflow,
    CompiledWorkflow,
    WorkflowResult,
    WorkflowDefinitionError,
    WorkflowExecutionError,
    WorkflowRoutingError,
    END,
)


# ---------------------------------------------------------------------------
# Builder API + compile-time validation
# ---------------------------------------------------------------------------


def test_minimal_workflow_runs():
    flow = Workflow()
    flow.add_node("a", lambda s: {"a": 1})
    flow.set_entry("a")
    result = flow.run()
    assert result.success
    assert result.state == {"a": 1}
    assert result.nodes_run == ["a"]


def test_add_node_rejects_empty_name():
    flow = Workflow()
    with pytest.raises(WorkflowDefinitionError, match="non-empty"):
        flow.add_node("", lambda s: {})


def test_add_node_rejects_duplicate_name():
    flow = Workflow()
    flow.add_node("a", lambda s: {})
    with pytest.raises(WorkflowDefinitionError, match="duplicate"):
        flow.add_node("a", lambda s: {})


def test_compile_requires_entry():
    flow = Workflow()
    flow.add_node("a", lambda s: {})
    with pytest.raises(WorkflowDefinitionError, match="no entry"):
        flow.compile()


def test_compile_entry_must_be_registered():
    flow = Workflow()
    flow.add_node("a", lambda s: {})
    flow.set_entry("nonexistent")
    with pytest.raises(WorkflowDefinitionError, match="not registered"):
        flow.compile()


def test_compile_rejects_edge_with_unknown_endpoint():
    flow = Workflow()
    flow.add_node("a", lambda s: {})
    flow.set_entry("a")
    flow.add_edge("a", "ghost")
    with pytest.raises(WorkflowDefinitionError, match="unknown node 'ghost'"):
        flow.compile()


def test_compile_rejects_cycle_in_static_edges():
    flow = Workflow()
    flow.add_node("a", lambda s: {})
    flow.add_node("b", lambda s: {})
    flow.add_node("c", lambda s: {})
    flow.add_edge("a", "b")
    flow.add_edge("b", "c")
    flow.add_edge("c", "a")  # cycle
    flow.set_entry("a")
    with pytest.raises(WorkflowDefinitionError, match="cycle"):
        flow.compile()


def test_compile_rejects_mixed_static_and_conditional_outgoing():
    flow = Workflow()
    flow.add_node("a", lambda s: {})
    flow.add_node("b", lambda s: {})
    flow.add_node("c", lambda s: {})
    flow.add_edge("a", "b")
    flow.add_conditional_edge("a", lambda s: "c", {"c": "c"})
    flow.set_entry("a")
    with pytest.raises(WorkflowDefinitionError, match="both static and conditional"):
        flow.compile()


def test_compile_rejects_conditional_with_unknown_target():
    flow = Workflow()
    flow.add_node("a", lambda s: {})
    flow.set_entry("a")
    flow.add_conditional_edge("a", lambda s: "ghost", {"ghost": "ghost"})
    with pytest.raises(WorkflowDefinitionError, match="unknown node 'ghost'"):
        flow.compile()


def test_compile_rejects_double_conditional_edge_from_same_node():
    flow = Workflow()
    flow.add_node("a", lambda s: {})
    flow.add_node("b", lambda s: {})
    flow.add_conditional_edge("a", lambda s: "b", {"b": "b"})
    with pytest.raises(WorkflowDefinitionError, match="already has a conditional"):
        flow.add_conditional_edge("a", lambda s: "b", {"b": "b"})


def test_compile_idempotent():
    flow = Workflow()
    flow.add_node("a", lambda s: {})
    flow.set_entry("a")
    c1 = flow.compile()
    c2 = flow.compile()
    assert c1 is c2


def test_compile_invalidated_by_mutation():
    flow = Workflow()
    flow.add_node("a", lambda s: {})
    flow.set_entry("a")
    c1 = flow.compile()
    flow.add_node("b", lambda s: {})
    c2 = flow.compile()
    assert c1 is not c2


def test_compile_rejects_exit_unknown_node():
    flow = Workflow()
    flow.add_node("a", lambda s: {})
    flow.set_entry("a")
    flow.set_exit("ghost")
    with pytest.raises(WorkflowDefinitionError, match="exit node 'ghost'"):
        flow.compile()


# ---------------------------------------------------------------------------
# Static-edge execution
# ---------------------------------------------------------------------------


def test_linear_three_node_pipeline():
    flow = Workflow()
    flow.add_node("a", lambda s: {"a_val": 1})
    flow.add_node("b", lambda s: {"b_val": s["a_val"] + 10})
    flow.add_node("c", lambda s: {"c_val": s["b_val"] + 100})
    flow.add_edge("a", "b")
    flow.add_edge("b", "c")
    flow.set_entry("a")
    result = flow.run()
    assert result.success
    assert result.state == {"a_val": 1, "b_val": 11, "c_val": 111}
    assert result.nodes_run == ["a", "b", "c"]


def test_parallel_fan_out_then_join():
    """Two siblings of `a` run; `join` waits for both."""
    flow = Workflow()
    flow.add_node("a", lambda s: {"a_val": 1})
    flow.add_node("b", lambda s: {"b_val": s["a_val"] * 2})
    flow.add_node("c", lambda s: {"c_val": s["a_val"] * 3})
    flow.add_node("join", lambda s: {"joined": s["b_val"] + s["c_val"]})
    flow.add_edge("a", "b")
    flow.add_edge("a", "c")
    flow.add_edge("b", "join")
    flow.add_edge("c", "join")
    flow.set_entry("a")
    result = flow.run()
    assert result.success
    assert result.state["joined"] == 5  # 2 + 3
    # Order of b/c within the level is deterministic (alphabetical).
    assert result.nodes_run == ["a", "b", "c", "join"]


def test_initial_state_passed_through():
    """run(**kwargs) populates initial state."""
    flow = Workflow()
    flow.add_node("greet", lambda s: {"greeting": f"Hello, {s['name']}!"})
    flow.set_entry("greet")
    result = flow.run(name="world")
    assert result.state == {"name": "world", "greeting": "Hello, world!"}


def test_empty_dict_return_is_valid():
    """A node returning {} contributes no state but doesn't fail."""
    flow = Workflow()
    flow.add_node("noop", lambda s: {})
    flow.add_node("write", lambda s: {"x": 1})
    flow.add_edge("noop", "write")
    flow.set_entry("noop")
    result = flow.run()
    assert result.success
    assert result.state == {"x": 1}


def test_none_return_treated_as_empty():
    """A node returning None is treated as {} (no-op)."""
    flow = Workflow()
    flow.add_node("noop", lambda s: None)
    flow.add_node("write", lambda s: {"x": 1})
    flow.add_edge("noop", "write")
    flow.set_entry("noop")
    result = flow.run()
    assert result.success
    assert result.state == {"x": 1}


def test_non_dict_return_fails_run():
    """A node returning a non-dict, non-None value fails the run."""
    flow = Workflow()
    flow.add_node("bad", lambda s: 42)
    flow.set_entry("bad")
    result = flow.run()
    assert not result.success
    assert "expected dict" in result.error


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------


def test_conditional_route_with_edge_map():
    flow = Workflow()
    flow.add_node("search", lambda s: {"hits": s.get("hits", [])})
    flow.add_node("summarize", lambda s: {"answer": f"summary of {len(s['hits'])}"})
    flow.add_node("fallback", lambda s: {"answer": "no results"})
    flow.add_conditional_edge(
        "search",
        lambda s: "has_results" if s["hits"] else "empty",
        {"has_results": "summarize", "empty": "fallback"},
    )
    flow.set_entry("search")
    result_with = flow.run(hits=["doc1", "doc2"])
    assert result_with.success
    assert result_with.state["answer"] == "summary of 2"
    assert "summarize" in result_with.nodes_run
    assert "fallback" not in result_with.nodes_run

    result_without = flow.run(hits=[])
    assert result_without.success
    assert result_without.state["answer"] == "no results"
    assert "fallback" in result_without.nodes_run
    assert "summarize" not in result_without.nodes_run


def test_conditional_route_without_edge_map():
    """When edge_map is None, the router's return is the target name directly."""
    flow = Workflow()
    flow.add_node("router_node", lambda s: {})
    flow.add_node("alpha", lambda s: {"chosen": "alpha"})
    flow.add_node("beta", lambda s: {"chosen": "beta"})
    flow.add_conditional_edge("router_node", lambda s: s.get("pick", "alpha"))
    flow.set_entry("router_node")

    r = flow.run(pick="beta")
    assert r.success
    assert r.state["chosen"] == "beta"
    assert "beta" in r.nodes_run and "alpha" not in r.nodes_run


def test_conditional_route_END_terminates_branch():
    flow = Workflow()
    flow.add_node("start", lambda s: {"done": True})
    flow.add_node("never", lambda s: {"reached": True})
    flow.add_conditional_edge("start", lambda s: END)
    flow.set_entry("start")
    result = flow.run()
    assert result.success
    assert "reached" not in result.state
    assert result.nodes_run == ["start"]


def test_conditional_route_unknown_target_fails():
    flow = Workflow()
    flow.add_node("router", lambda s: {})
    flow.add_node("a", lambda s: {})
    flow.add_conditional_edge("router", lambda s: "ghost", {"x": "a"})
    flow.set_entry("router")
    result = flow.run()
    assert not result.success
    assert "not in edge_map" in result.error


def test_conditional_route_router_exception_captured():
    flow = Workflow()
    flow.add_node("a", lambda s: {})
    flow.add_node("b", lambda s: {})

    def bad_router(s):
        raise RuntimeError("router exploded")

    flow.add_conditional_edge("a", bad_router, {"b": "b"})
    flow.set_entry("a")
    result = flow.run()
    assert not result.success
    assert "router exploded" in result.error


def test_conditional_route_unmapped_value_when_edge_map_set():
    """edge_map present but router returns a key not in it."""
    flow = Workflow()
    flow.add_node("a", lambda s: {})
    flow.add_node("b", lambda s: {})
    flow.add_conditional_edge("a", lambda s: "unmapped", {"b": "b"})
    flow.set_entry("a")
    result = flow.run()
    assert not result.success
    assert "not in edge_map" in result.error


# ---------------------------------------------------------------------------
# Errors and limits
# ---------------------------------------------------------------------------


def test_node_exception_captured():
    flow = Workflow()

    def boom(state):
        raise ValueError("node died")

    flow.add_node("boom", boom)
    flow.set_entry("boom")
    result = flow.run()
    assert not result.success
    assert "node died" in result.error
    assert result.metrics is not None
    assert result.metrics.error_count == 1


def test_node_exception_skips_downstream():
    flow = Workflow()
    flow.add_node("a", lambda s: (_ for _ in ()).throw(ValueError("die")))
    flow.add_node("b", lambda s: {"b_ran": True})
    flow.add_edge("a", "b")
    flow.set_entry("a")
    result = flow.run()
    assert not result.success
    assert "b_ran" not in result.state
    assert "b" not in result.nodes_run


def test_max_steps_cycle_bound():
    """Conditional edges can create cycles; max_steps must terminate."""
    flow = Workflow(max_steps=5)
    flow.add_node("a", lambda s: {})
    flow.add_node("b", lambda s: {})
    flow.add_conditional_edge("a", lambda s: "b", {"b": "b"})
    flow.add_conditional_edge("b", lambda s: "a", {"a": "a"})
    flow.set_entry("a")
    result = flow.run()
    assert not result.success
    assert "max_steps" in result.error


# ---------------------------------------------------------------------------
# Async + timeout
# ---------------------------------------------------------------------------


def test_async_node_awaited():
    flow = Workflow()

    async def async_node(state):
        await asyncio.sleep(0)
        return {"value": 42}

    flow.add_node("a", async_node)
    flow.set_entry("a")
    result = flow.run()
    assert result.state == {"value": 42}


def test_per_node_timeout_fires():
    """A node exceeding its declared timeout fails the run.

    Note: the underlying worker thread isn't killed (Python cannot
    safely kill threads, matching the documented ``Tool.timeout``
    limitation) -- ``asyncio.run`` waits for the executor's pending
    threads at shutdown. So we don't assert on wall-clock elapsed,
    only on the error being captured.
    """
    flow = Workflow()

    def slow(state):
        time.sleep(0.2)
        return {"slow_done": True}

    flow.add_node("slow", slow, timeout=0.05)
    flow.set_entry("slow")
    result = flow.run()
    assert not result.success
    assert "timeout" in result.error.lower()


def test_async_node_timeout_fires_and_returns_promptly():
    """For async nodes, timeout is enforced cleanly via asyncio.wait_for."""
    flow = Workflow()

    async def slow(state):
        await asyncio.sleep(0.3)
        return {"slow_done": True}

    flow.add_node("slow", slow, timeout=0.05)
    flow.set_entry("slow")
    started = time.perf_counter()
    result = flow.run()
    elapsed = time.perf_counter() - started
    assert not result.success
    assert "timeout" in result.error.lower()
    # Async cancellation IS clean -- the coroutine is cancelled, not
    # stranded on a worker thread.
    assert elapsed < 0.25


def test_arun_async_variant():
    """arun() returns the same result via asyncio.run."""
    flow = Workflow()
    flow.add_node("a", lambda s: {"x": 1})
    flow.set_entry("a")
    result = asyncio.run(flow.arun())
    assert result.success
    assert result.state == {"x": 1}


# ---------------------------------------------------------------------------
# State isolation
# ---------------------------------------------------------------------------


def test_node_cannot_mutate_shared_state():
    """Mutating the state dict passed to a node doesn't affect other nodes."""
    flow = Workflow()

    def evil(state):
        state["sneaky"] = "should not appear"
        return {"a": 1}

    flow.add_node("a", evil)
    flow.add_node("b", lambda s: {"sneaky_seen": "sneaky" in s})
    flow.add_edge("a", "b")
    flow.set_entry("a")
    result = flow.run()
    assert result.success
    # 'sneaky' did NOT propagate; only the explicit return value did.
    assert "sneaky" not in result.state
    assert result.state["sneaky_seen"] is False


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def test_inspection_views_return_copies():
    """nodes / static_edges / conditional_edges expose defensive copies."""
    flow = Workflow()
    flow.add_node("a", lambda s: {})
    flow.add_node("b", lambda s: {})
    flow.add_edge("a", "b")

    nodes = flow.nodes
    nodes["c"] = "tampered"  # type: ignore[assignment]
    assert "c" not in flow.nodes

    edges = flow.static_edges
    edges["a"].append("tampered")
    assert "tampered" not in flow.static_edges["a"]


def test_compiled_workflow_exposes_levels():
    """CompiledWorkflow.levels is the topological-level decomposition."""
    flow = Workflow()
    flow.add_node("a", lambda s: {})
    flow.add_node("b", lambda s: {})
    flow.add_node("c", lambda s: {})
    flow.add_node("d", lambda s: {})
    flow.add_edge("a", "b")
    flow.add_edge("a", "c")
    flow.add_edge("b", "d")
    flow.add_edge("c", "d")
    flow.set_entry("a")
    compiled = flow.compile()
    assert compiled.levels == (("a",), ("b", "c"), ("d",))


# ===========================================================================
# Phase 2 -- Layer C decorator sugar
# ===========================================================================


class TestLayerCDecorator:
    """``@flow.node`` desugars to ``add_node`` + edge inference."""

    def test_bare_decorator_registers_node(self):
        flow = Workflow()

        @flow.node
        def search(query: str) -> list:
            return [query.upper()]

        flow.set_entry("search")
        result = flow.run(query="hi")
        assert result.success
        # Return value stored under the node's name.
        assert result.state["search"] == ["HI"]

    def test_decorator_returns_original_function_unchanged(self):
        """A decorated function remains callable as a plain function."""
        flow = Workflow()

        @flow.node
        def double(x: int) -> int:
            return x * 2

        # The original function is still directly callable for unit tests.
        assert double(5) == 10

    def test_decorator_with_name_override(self):
        flow = Workflow()

        @flow.node(name="primary_search")
        def search(query: str) -> list:
            return [query]

        flow.set_entry("primary_search")
        result = flow.run(query="x")
        assert "primary_search" in result.state
        assert "search" not in result.state

    def test_decorator_with_timeout(self):
        import time as _time

        flow = Workflow()

        @flow.node(timeout=0.05)
        def slow() -> str:
            _time.sleep(0.2)
            return "done"

        flow.set_entry("slow")
        result = flow.run()
        assert not result.success
        assert "timeout" in result.error.lower()

    def test_async_function_decorated_as_node(self):
        flow = Workflow()

        @flow.node
        async def fetch(url: str) -> str:
            await asyncio.sleep(0)
            return f"contents of {url}"

        flow.set_entry("fetch")
        result = flow.run(url="https://example.com")
        assert result.state["fetch"] == "contents of https://example.com"


class TestLayerCDependencyInference:
    """Parameter names matching other node names become static edges."""

    def test_param_matching_node_creates_edge(self):
        flow = Workflow()

        @flow.node
        def a() -> int:
            return 7

        @flow.node
        def b(a: int) -> int:
            return a * 2

        flow.set_entry("a")
        flow.set_exit("b")
        compiled = flow.compile()
        # Static edge from a to b inferred from b's param name.
        assert "b" in compiled.static_edges.get("a", ())
        result = flow.run()
        assert result.state["b"] == 14

    def test_decoration_order_does_not_matter(self):
        """Decorating consumer before producer still wires correctly."""
        flow = Workflow()

        @flow.node
        def consumer(producer: str) -> str:  # decorated before producer
            return producer.upper()

        @flow.node
        def producer() -> str:
            return "hello"

        flow.set_entry("producer")
        flow.set_exit("consumer")
        result = flow.run()
        assert result.success
        assert result.state["consumer"] == "HELLO"

    def test_param_not_matching_node_is_workflow_input(self):
        flow = Workflow()

        @flow.node
        def greet(name: str) -> str:
            return f"hello {name}"

        flow.set_entry("greet")
        compiled = flow.compile()
        assert "name" in compiled.layer_c_inputs

    def test_missing_workflow_input_fails_cleanly(self):
        flow = Workflow()

        @flow.node
        def greet(name: str) -> str:
            return f"hello {name}"

        flow.set_entry("greet")
        result = flow.run()  # no `name=` supplied
        assert not result.success
        assert "missing required workflow inputs" in result.error
        assert "name" in result.error

    def test_self_referential_param_treated_as_input(self):
        """A param matching the same node's own name is a workflow input."""
        flow = Workflow()

        @flow.node
        def acc(acc: int) -> int:
            return acc + 1

        flow.set_entry("acc")
        compiled = flow.compile()
        # No self-loop edge.
        assert "acc" not in compiled.static_edges.get("acc", ())
        # `acc` is recorded as an input requirement.
        assert "acc" in compiled.layer_c_inputs
        result = flow.run(acc=10)
        assert result.state["acc"] == 11

    def test_multiple_param_dependencies(self):
        """A node with multiple params depending on multiple producers."""
        flow = Workflow()

        @flow.node
        def a() -> int:
            return 1

        @flow.node
        def b() -> int:
            return 2

        @flow.node
        def c() -> int:
            return 3

        @flow.node
        def sum_all(a: int, b: int, c: int) -> int:
            return a + b + c

        flow.set_entry("a")
        flow.set_exit("sum_all")
        compiled = flow.compile()
        # All three are static predecessors of sum_all.
        assert set(compiled.static_reverse["sum_all"]) == {"a", "b", "c"}


class TestLayerCRoute:
    """``@flow.route`` desugars to ``add_conditional_edge``."""

    def test_route_decorator_dispatches(self):
        flow = Workflow()

        @flow.node
        def start() -> dict:
            return {}

        @flow.node
        def alpha() -> str:
            return "alpha"

        @flow.node
        def beta() -> str:
            return "beta"

        @flow.route(after="start")
        def pick(start: dict) -> str:
            return "alpha"

        flow.set_entry("start")
        result = flow.run()
        assert result.success
        assert "alpha" in result.nodes_run
        assert "beta" not in result.nodes_run

    def test_route_returns_END_terminates(self):
        flow = Workflow()

        @flow.node
        def start() -> int:
            return 0

        @flow.node
        def never() -> str:
            return "should not run"

        @flow.route(after="start")
        def pick(start: int):
            return END

        flow.set_entry("start")
        result = flow.run()
        assert result.success
        assert "never" not in result.nodes_run

    def test_route_param_must_be_known(self):
        """Router params must resolve at runtime."""
        flow = Workflow()

        @flow.node
        def start() -> str:
            return "x"

        @flow.node
        def target() -> str:
            return "y"

        # router takes an unknown param `mystery`.
        @flow.route(after="start")
        def pick(mystery: str) -> str:
            return "target"

        flow.set_entry("start")
        result = flow.run()
        # Should fail with missing-input error.
        assert not result.success
        assert "mystery" in result.error


class TestProgrammaticAddNode:
    """``flow.add_node(fn)`` non-decorator form."""

    def test_add_node_with_callable(self):
        flow = Workflow()

        def search(query: str) -> str:
            return query.upper()

        flow.add_node(search)
        flow.set_entry("search")
        result = flow.run(query="hi")
        assert result.state["search"] == "HI"

    def test_add_node_with_name_override(self):
        flow = Workflow()

        def search(query: str) -> str:
            return query

        flow.add_node(search, name="custom")
        flow.set_entry("custom")
        result = flow.run(query="x")
        assert "custom" in result.state
        assert "search" not in result.state

    def test_add_node_rejects_non_string_non_callable(self):
        flow = Workflow()
        with pytest.raises(TypeError, match="must be a node name"):
            flow.add_node(42)  # type: ignore[arg-type]

    def test_add_node_layer_b_form_missing_fn(self):
        flow = Workflow()
        with pytest.raises(TypeError, match="fn argument is required"):
            flow.add_node("a")  # type: ignore[call-arg]


class TestLayerCSignatureRejection:
    """Signatures with *args / **kwargs are not Layer-C-compatible."""

    def test_starargs_rejected(self):
        flow = Workflow()

        def bad(*args):
            return None

        with pytest.raises(WorkflowDefinitionError, match=r"\*args"):
            flow.add_node(bad)

    def test_kwargs_rejected(self):
        flow = Workflow()

        def bad(**kwargs):
            return None

        with pytest.raises(WorkflowDefinitionError, match=r"\*\*kwargs"):
            flow.add_node(bad)


class TestMixedLayering:
    """Layer-B and Layer-C calls on the same Workflow interoperate."""

    def test_mixed_layers_on_same_workflow(self):
        flow = Workflow()

        @flow.node
        def a() -> int:
            return 5

        # Layer-B: explicit (state) -> dict, multiple state keys.
        def b_layer_b(state: dict) -> dict:
            return {"b_x": state["a"] * 2, "b_y": state["a"] + 1}

        flow.add_node("b", b_layer_b)
        flow.add_edge("a", "b")
        flow.set_entry("a")
        flow.set_exit("b")
        result = flow.run()
        assert result.state["a"] == 5
        assert result.state["b_x"] == 10
        assert result.state["b_y"] == 6

    def test_layer_c_node_duplicate_name_rejected(self):
        flow = Workflow()

        @flow.node
        def a() -> int:
            return 1

        with pytest.raises(WorkflowDefinitionError, match="duplicate"):

            @flow.node(name="a")
            def b() -> int:
                return 2


class TestDerivedStateSchema:
    """The derived schema synthesizes from Layer-C metadata."""

    def test_derived_schema_includes_node_returns(self):
        flow = Workflow()

        @flow.node
        def search(query: str) -> list:
            return []

        @flow.node
        def count(search: list) -> int:
            return len(search)

        schema = flow.derived_state_schema
        assert schema["search"] is list
        assert schema["count"] is int
        # `query` is an input requirement -> Any
        from typing import Any as _Any

        assert schema["query"] is _Any

    def test_derived_schema_respects_explicit_state_schema(self):
        from typing import TypedDict

        class MyState(TypedDict):
            x: int
            y: str

        flow = Workflow(MyState)
        schema = flow.derived_state_schema
        assert schema == {"x": int, "y": str}


# ===========================================================================
# Phase 3 -- streaming, events, helpers, visualization
# ===========================================================================


class TestEventStreaming:
    """``run()`` populates ``events``; ``stream()`` yields the same sequence."""

    def test_run_populates_events_with_workflow_brackets(self):
        from cyllama.agents import EventType as ET

        flow = Workflow()

        @flow.node
        def a():
            return 1

        flow.set_entry("a")
        result = flow.run()
        assert result.success
        kinds = [e.type for e in result.events]
        # First event is WORKFLOW_START, last is WORKFLOW_END.
        assert kinds[0] == ET.WORKFLOW_START
        assert kinds[-1] == ET.WORKFLOW_END
        # NODE_START and NODE_END appear for `a`.
        assert ET.NODE_START in kinds
        assert ET.NODE_END in kinds

    def test_workflow_end_metadata_carries_final_state(self):
        from cyllama.agents import EventType as ET

        flow = Workflow()

        @flow.node
        def a():
            return 42

        flow.set_entry("a")
        result = flow.run()
        end = result.events[-1]
        assert end.type == ET.WORKFLOW_END
        assert end.metadata["success"] is True
        assert end.metadata["state"] == {"a": 42}
        assert end.metadata["nodes_run"] == ["a"]

    def test_node_end_metadata_carries_update_and_elapsed(self):
        from cyllama.agents import EventType as ET

        flow = Workflow()

        @flow.node
        def a():
            return "hello"

        flow.set_entry("a")
        result = flow.run()
        node_end = next(e for e in result.events if e.type == ET.NODE_END and e.metadata["node"] == "a")
        assert node_end.metadata["update"] == {"a": "hello"}
        assert "elapsed_ms" in node_end.metadata
        assert node_end.metadata["elapsed_ms"] >= 0

    def test_stream_yields_same_events_as_run(self):
        flow = Workflow()

        @flow.node
        def a():
            return 1

        @flow.node
        def b(a: int):
            return a + 10

        flow.set_entry("a")
        flow.set_exit("b")

        run_events = [(e.type, e.metadata.get("node")) for e in flow.run().events]
        stream_events = [(e.type, e.metadata.get("node")) for e in flow.stream()]
        assert run_events == stream_events

    def test_stream_works_for_parallel_workflows(self):
        from cyllama.agents import EventType as ET

        flow = Workflow()

        @flow.node
        def start(query: str) -> str:
            return query

        @flow.node
        def a(start: str) -> int:
            return 1

        @flow.node
        def b(start: str) -> int:
            return 2

        @flow.node
        def join(a: int, b: int) -> int:
            return a + b

        flow.set_entry("start")
        flow.set_exit("join")

        events = list(flow.stream(query="hi"))
        node_names_started = [e.metadata["node"] for e in events if e.type == ET.NODE_START]
        assert sorted(node_names_started) == ["a", "b", "join", "start"]

    def test_stream_error_path_emits_workflow_end_with_failure(self):
        from cyllama.agents import EventType as ET

        flow = Workflow()

        def boom(state):
            raise RuntimeError("boom")

        flow.add_node("boom", boom)
        flow.set_entry("boom")

        events = list(flow.stream())
        # Last event is WORKFLOW_END with success=False.
        end = events[-1]
        assert end.type == ET.WORKFLOW_END
        assert end.metadata["success"] is False
        assert "boom" in end.metadata["error"]

    def test_astream_async_iteration(self):
        """astream is the canonical async generator."""
        from cyllama.agents import EventType as ET

        flow = Workflow()

        @flow.node
        async def a():
            return 1

        flow.set_entry("a")

        async def collect():
            return [e async for e in flow.astream()]

        events = asyncio.run(collect())
        assert events[0].type == ET.WORKFLOW_START
        assert events[-1].type == ET.WORKFLOW_END


# ===========================================================================
# agent_node + tool_node
# ===========================================================================


class TestAgentNode:
    """``agent_node`` wraps any AgentProtocol-shaped object as a workflow node."""

    class _StubAgent:
        def __init__(self, answer="stub answer"):
            self._answer = answer
            self._metrics = None
            self.calls = []

        @property
        def metrics(self):
            return self._metrics

        def run(self, task):
            self.calls.append(task)

            # Mimic AgentResult with .answer.
            class _R:
                pass

            r = _R()
            r.answer = f"{self._answer}: {task}"
            return r

        def stream(self, task):
            yield None

    def test_agent_node_dispatches_to_agent(self):
        from cyllama.agents.workflow import agent_node

        agent = self._StubAgent()
        flow = Workflow()
        flow.add_node("research", agent_node(agent, "research", task_param="topic"))
        flow.set_entry("research")

        result = flow.run(topic="Westphalia")
        assert result.success
        assert result.state["research"] == "stub answer: Westphalia"
        assert agent.calls == ["Westphalia"]

    def test_agent_node_default_task_param(self):
        from cyllama.agents.workflow import agent_node

        agent = self._StubAgent()
        flow = Workflow()
        flow.add_node("solve", agent_node(agent, "solve"))
        flow.set_entry("solve")
        result = flow.run(task="2+2")
        assert result.success
        assert agent.calls == ["2+2"]

    def test_agent_node_missing_task_in_state_fails(self):
        from cyllama.agents.workflow import agent_node

        agent = self._StubAgent()
        flow = Workflow()
        flow.add_node("solve", agent_node(agent, "solve", task_param="question"))
        flow.set_entry("solve")
        result = flow.run()  # no `question=`
        assert not result.success
        assert "missing required key" in result.error

    def test_agent_node_non_string_task_fails(self):
        from cyllama.agents.workflow import agent_node

        agent = self._StubAgent()
        flow = Workflow()
        flow.add_node("solve", agent_node(agent, "solve", task_param="x"))
        flow.set_entry("solve")
        result = flow.run(x=123)
        assert not result.success
        assert "must be a str" in result.error


class TestToolNode:
    """``tool_node`` wraps a Tool as a workflow node."""

    def test_tool_node_invokes_tool_with_state_args(self):
        from cyllama.agents import tool
        from cyllama.agents.workflow import tool_node

        @tool
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        flow = Workflow()
        flow.add_node("add", tool_node(add))
        flow.set_entry("add")
        result = flow.run(a=3, b=4)
        assert result.success
        assert result.state["add"] == 7

    def test_tool_node_uses_tool_name_by_default(self):
        from cyllama.agents import tool
        from cyllama.agents.workflow import tool_node

        @tool(name="custom_op")
        def op(x: int) -> int:
            """Op."""
            return x * 2

        flow = Workflow()
        flow.add_node("custom_op", tool_node(op))
        flow.set_entry("custom_op")
        result = flow.run(x=5)
        assert result.state["custom_op"] == 10


# ===========================================================================
# Visualization + dry-run
# ===========================================================================


class TestVisualization:
    def test_mermaid_renders_static_edges(self):
        flow = Workflow()

        @flow.node
        def a():
            return 1

        @flow.node
        def b(a: int):
            return a + 1

        flow.set_entry("a")
        flow.set_exit("b")

        m = flow.to_mermaid()
        assert m.startswith("graph TD")
        assert "a --> b" in m
        # Entry node has the circle shape.
        assert "a((" in m
        # Exit node has the stadium/pill shape.
        assert "b([" in m

    def test_mermaid_renders_conditional_edges(self):
        flow = Workflow()
        flow.add_node("router", lambda s: {})
        flow.add_node("alpha", lambda s: {})
        flow.add_node("beta", lambda s: {})
        flow.add_conditional_edge("router", lambda s: "alpha", {"alpha": "alpha", "beta": "beta"})
        flow.set_entry("router")
        m = flow.to_mermaid()
        # Dashed edge with key label.
        assert "router -.->" in m
        assert '"alpha"' in m

    def test_dot_renders_static_edges(self):
        flow = Workflow()

        @flow.node
        def a():
            return 1

        @flow.node
        def b(a: int):
            return a + 1

        flow.set_entry("a")
        flow.set_exit("b")

        d = flow.to_dot()
        assert "digraph workflow" in d
        assert "a -> b" in d

    def test_dry_run_returns_plan_without_executing(self):
        from cyllama.agents.workflow import DryRunPlan

        executed = []
        flow = Workflow()

        @flow.node
        def a():
            executed.append("a")
            return 1

        @flow.node
        def b(a: int):
            executed.append("b")
            return a + 1

        flow.set_entry("a")
        flow.set_exit("b")
        plan = flow.dry_run()
        assert isinstance(plan, DryRunPlan)
        assert plan.entry == "a"
        assert "b" in plan.exits
        assert plan.levels == (("a",), ("b",))
        # Critical: nothing executed.
        assert executed == []

    def test_dry_run_marks_conditional_nodes(self):
        flow = Workflow()
        flow.add_node("router", lambda s: {})
        flow.add_node("alpha", lambda s: {})
        flow.add_node("beta", lambda s: {})
        flow.add_conditional_edge("router", lambda s: "alpha", {"alpha": "alpha", "beta": "beta"})
        flow.set_entry("router")
        plan = flow.dry_run()
        # Both targets are conditional.
        assert plan.conditional_nodes == frozenset({"alpha", "beta"})

    def test_dry_run_lists_inputs_required(self):
        flow = Workflow()

        @flow.node
        def greet(name: str, greeting: str):
            return f"{greeting}, {name}!"

        flow.set_entry("greet")
        plan = flow.dry_run()
        assert "name" in plan.inputs_required
        assert "greeting" in plan.inputs_required


# ===========================================================================
# Phase 4 -- reducers + invariants + typed state
# ===========================================================================


class TestReducers:
    """Built-in reducers and registration."""

    def test_append_reducer(self):
        from cyllama.agents.workflow import reducer

        assert reducer.append(None, "x") == ["x"]
        assert reducer.append(["a"], "b") == ["a", "b"]
        # Existing list is not mutated.
        base = ["a"]
        reducer.append(base, "b")
        assert base == ["a"]

    def test_extend_reducer(self):
        from cyllama.agents.workflow import reducer

        assert reducer.extend(None, ["a"]) == ["a"]
        assert reducer.extend(["a"], ["b", "c"]) == ["a", "b", "c"]
        assert reducer.extend(None, None) == []

    def test_merge_dict_reducer(self):
        from cyllama.agents.workflow import reducer

        assert reducer.merge_dict(None, {"k": 1}) == {"k": 1}
        # Update wins on key collision.
        assert reducer.merge_dict({"a": 1, "b": 2}, {"b": 99, "c": 3}) == {
            "a": 1,
            "b": 99,
            "c": 3,
        }

    def test_add_reducer(self):
        from cyllama.agents.workflow import reducer

        assert reducer.add(None, 5) == 5
        assert reducer.add(3, 4) == 7
        assert reducer.add(1.5, 2.5) == 4.0

    def test_last_reducer(self):
        from cyllama.agents.workflow import reducer

        assert reducer.last("old", "new") == "new"
        assert reducer.last(None, 42) == 42

    def test_reducer_combines_multi_writer_state(self):
        """Two nodes writing the same key with a reducer aggregate cleanly."""
        from cyllama.agents.workflow import reducer

        flow = Workflow(reducers={"messages": reducer.append})

        def a(state):
            return {"messages": "hello"}

        def b(state):
            return {"messages": "world"}

        flow.add_node("a", a)
        flow.add_node("b", b)
        flow.add_edge("a", "b")
        flow.set_entry("a")
        result = flow.run()
        assert result.success
        assert result.state["messages"] == ["hello", "world"]

    def test_multi_writer_without_reducer_fails(self):
        """Two nodes writing the same key without a reducer fails the run."""
        flow = Workflow()

        def a(state):
            return {"shared": 1}

        def b(state):
            return {"shared": 2}

        flow.add_node("a", a)
        flow.add_node("b", b)
        flow.add_edge("a", "b")
        flow.set_entry("a")
        result = flow.run()
        assert not result.success
        assert "written by both" in result.error
        assert "register a reducer" in result.error

    def test_node_overwriting_own_key_on_revisit_is_allowed(self):
        """Cyclic routing that re-runs the same node is not multi-writer.

        A is the sole writer of ``state["a"]``; the multi-writer check
        (which compares writer names) doesn't fire when A runs twice
        via conditional routing. This pins the false-positive guard.
        """
        flow = Workflow()

        def a(state):
            return {"a": state.get("a", 0) + 1}

        def b(state):
            return {}

        flow.add_node("a", a)
        flow.add_node("b", b)
        flow.add_edge("a", "b")
        flow.add_conditional_edge("b", lambda s: "a", {"a": "a"})
        flow.set_entry("a")
        result = flow.run()
        # The cycle exhausts naturally; the run completes successfully
        # WITHOUT a "written by both" error -- that's the point.
        assert result.success
        assert "a" in result.nodes_run
        # No multi-writer error in the captured events.
        for ev in result.events:
            err = ev.metadata.get("error") if hasattr(ev, "metadata") else None
            if err:
                assert "written by both" not in err

    def test_custom_reducer_callable(self):
        """Any callable matching (existing, update) -> merged works."""

        def max_reducer(existing, update):
            if existing is None:
                return update
            return max(existing, update)

        flow = Workflow(reducers={"score": max_reducer})

        def lo(state):
            return {"score": 50}

        def hi(state):
            return {"score": 90}

        flow.add_node("lo", lo)
        flow.add_node("hi", hi)
        flow.add_edge("lo", "hi")
        flow.set_entry("lo")
        result = flow.run()
        assert result.state["score"] == 90


# ===========================================================================
# WorkflowInvariant + WorkflowExecutionState
# ===========================================================================


class TestWorkflowInvariants:
    def test_no_invariants_default_behavior_unchanged(self):
        """Workflow without invariants behaves exactly as before."""
        flow = Workflow()
        flow.add_node("a", lambda s: {"a": 1})
        flow.set_entry("a")
        result = flow.run()
        assert result.success
        # No CONTRACT_VIOLATION events.
        from cyllama.agents import EventType as ET

        assert all(e.type != ET.CONTRACT_VIOLATION for e in result.events)

    def test_passing_invariant_emits_no_violation(self):
        from cyllama.agents.workflow import WorkflowInvariant

        flow = Workflow(
            invariants=[
                WorkflowInvariant(predicate=lambda s: s.node_count < 10, message="under 10 nodes"),
            ],
        )
        flow.add_node("a", lambda s: {})
        flow.set_entry("a")
        result = flow.run()
        assert result.success

    def test_failing_invariant_under_enforce_terminates(self):
        from cyllama.agents.workflow import WorkflowInvariant
        from cyllama.agents import ContractPolicy, EventType as ET

        flow = Workflow(
            invariants=[
                WorkflowInvariant(predicate=lambda s: s.node_count <= 1, message="only one node"),
            ],
            policy=ContractPolicy.ENFORCE,
        )
        flow.add_node("a", lambda s: {"a": 1})
        flow.add_node("b", lambda s: {"b": 2})
        flow.add_edge("a", "b")
        flow.set_entry("a")
        flow.set_exit("b")
        result = flow.run()
        assert not result.success
        assert "only one node" in result.error
        # Violation event emitted before terminal WORKFLOW_END.
        kinds = [e.type for e in result.events]
        assert ET.CONTRACT_VIOLATION in kinds

    def test_failing_invariant_under_observe_continues(self):
        from cyllama.agents.workflow import WorkflowInvariant
        from cyllama.agents import ContractPolicy, EventType as ET

        flow = Workflow(
            invariants=[
                WorkflowInvariant(predicate=lambda s: s.node_count <= 1, message="only one node"),
            ],
            policy=ContractPolicy.OBSERVE,
        )
        flow.add_node("a", lambda s: {"a": 1})
        flow.add_node("b", lambda s: {"b": 2})
        flow.add_edge("a", "b")
        flow.set_entry("a")
        flow.set_exit("b")
        result = flow.run()
        assert result.success
        # Violation event emitted but workflow completed.
        viol = [e for e in result.events if e.type == ET.CONTRACT_VIOLATION]
        assert len(viol) >= 1

    def test_ignore_policy_skips_invariant_checks(self):
        from cyllama.agents.workflow import WorkflowInvariant
        from cyllama.agents import ContractPolicy

        flow = Workflow(
            invariants=[
                WorkflowInvariant(predicate=lambda s: False, message="always fails"),
            ],
            policy=ContractPolicy.IGNORE,
        )
        flow.add_node("a", lambda s: {})
        flow.set_entry("a")
        result = flow.run()
        assert result.success
        # No CONTRACT_VIOLATION emitted.
        from cyllama.agents import EventType as ET

        assert all(e.type != ET.CONTRACT_VIOLATION for e in result.events)

    def test_per_invariant_policy_override(self):
        """An invariant's own policy beats the workflow's default."""
        from cyllama.agents.workflow import WorkflowInvariant
        from cyllama.agents import ContractPolicy

        flow = Workflow(
            invariants=[
                WorkflowInvariant(
                    predicate=lambda s: False,
                    message="observe-only",
                    policy=ContractPolicy.OBSERVE,
                ),
            ],
            policy=ContractPolicy.ENFORCE,  # workflow default
        )
        flow.add_node("a", lambda s: {})
        flow.set_entry("a")
        result = flow.run()
        # Workflow default is ENFORCE; but this invariant overrides to
        # OBSERVE, so the workflow completes despite the violation.
        assert result.success

    def test_invariant_predicate_exception_treated_as_violation(self):
        from cyllama.agents.workflow import WorkflowInvariant

        def broken(state):
            raise RuntimeError("predicate borked")

        flow = Workflow(invariants=[WorkflowInvariant(predicate=broken, message="x")])
        flow.add_node("a", lambda s: {})
        flow.set_entry("a")
        result = flow.run()
        assert not result.success
        # Captured as a violation (workflow terminates under ENFORCE).
        from cyllama.agents import EventType as ET

        viol = [e for e in result.events if e.type == ET.CONTRACT_VIOLATION]
        assert len(viol) == 1
        # The violation context records the predicate error.
        assert viol[0].metadata["violation"].context.get("predicate_error") is not None

    def test_execution_state_carries_runtime_metrics(self):
        """Invariants see elapsed_ms, node_count, nodes_run, etc."""
        from cyllama.agents.workflow import WorkflowInvariant

        seen_states = []

        def capture(state):
            seen_states.append(
                {
                    "node_count": state.node_count,
                    "elapsed_ms": state.elapsed_ms,
                    "nodes_run": list(state.nodes_run),
                }
            )
            return True

        flow = Workflow(invariants=[WorkflowInvariant(predicate=capture, message="x")])
        flow.add_node("a", lambda s: {})
        flow.add_node("b", lambda s: {})
        flow.add_edge("a", "b")
        flow.set_entry("a")
        result = flow.run()
        assert result.success
        # Invariant fired after each node completed.
        assert len(seen_states) == 2
        assert seen_states[0]["node_count"] == 1
        assert seen_states[0]["nodes_run"] == ["a"]
        assert seen_states[1]["node_count"] == 2
        assert seen_states[1]["nodes_run"] == ["a", "b"]
        assert seen_states[1]["elapsed_ms"] >= seen_states[0]["elapsed_ms"]

    def test_multiple_invariants_independent(self):
        """Each invariant evaluates independently per node-completion."""
        from cyllama.agents.workflow import WorkflowInvariant
        from cyllama.agents import ContractPolicy

        flow = Workflow(
            invariants=[
                WorkflowInvariant(predicate=lambda s: s.node_count < 5, message="limit"),
                WorkflowInvariant(predicate=lambda s: False, message="always fails"),
            ],
            policy=ContractPolicy.OBSERVE,
        )
        flow.add_node("a", lambda s: {})
        flow.add_node("b", lambda s: {})
        flow.add_edge("a", "b")
        flow.set_entry("a")
        result = flow.run()
        # Second invariant fails after every node; under OBSERVE we
        # complete with N violations recorded.
        from cyllama.agents import EventType as ET

        viol = [e for e in result.events if e.type == ET.CONTRACT_VIOLATION]
        # Both nodes triggered the failing invariant.
        assert len(viol) == 2
        assert all(v.metadata["violation"].message == "always fails" for v in viol)


# ===========================================================================
# Typed-state generic (smoke test only -- the generic exists since Phase 1)
# ===========================================================================


def test_workflow_generic_accepts_typeddict():
    """``Workflow[State]`` is parameterizable for static type-checkers.

    Runtime behavior is unaffected by the type parameter; this test
    just confirms the generic shape compiles and runs.
    """
    from typing import TypedDict

    class MyState(TypedDict):
        x: int
        y: str

    flow: Workflow[MyState] = Workflow(MyState)
    flow.add_node("init", lambda s: {"x": 42, "y": "hello"})
    flow.set_entry("init")
    result = flow.run()
    assert result.success
    assert result.state == {"x": 42, "y": "hello"}


# ===========================================================================
# Phase 5: AgentProtocol compliance + sub-workflow composition
# ===========================================================================


class _StubAgent:
    """Minimal AgentProtocol-conformant test double.

    Streams a fixed sequence of THOUGHT / ANSWER events; ``run`` returns
    an AgentResult-shape stub. Used to verify that ``agent_node`` and
    ``agent_as_tool`` integrate with workflows without dragging in a
    real LLM.
    """

    def __init__(self, name: str = "stub", answer: str = "stub answer", emit_thoughts: int = 2) -> None:
        self.name = name
        self._answer = answer
        self._emit_thoughts = emit_thoughts
        self._metrics: Any = None

    @property
    def metrics(self) -> Any:
        return self._metrics

    def stream(self, task: str):
        from cyllama.agents import AgentEvent, EventType

        for i in range(self._emit_thoughts):
            yield AgentEvent(type=EventType.THOUGHT, content=f"{self.name}: thinking {i} on {task!r}")
        yield AgentEvent(type=EventType.ANSWER, content=f"{self._answer} (re: {task!r})")

    def run(self, task: str):
        from cyllama.agents import AgentResult

        events = list(self.stream(task))
        return AgentResult(
            answer=events[-1].content,
            steps=events,
            iterations=len(events),
            success=True,
        )


class TestAgentProtocolAdapter:
    """Phase 5 (post-refactor): AgentProtocol conformance lives on the
    ``flow.as_agent()`` adapter, not on Workflow itself. The native
    workflow API stays kwargs-only.
    """

    def test_workflow_run_is_kwargs_only(self):
        """Positional args to flow.run() are rejected -- it's kwargs-only."""
        flow = Workflow()
        flow.add_node("a", lambda s: {"a": 1})
        flow.set_entry("a")
        with pytest.raises(TypeError):
            flow.run("hello")  # type: ignore[call-arg]

    def test_adapter_satisfies_agent_protocol(self):
        from cyllama.agents.types import AgentProtocol

        flow = Workflow()
        flow.add_node("hello", lambda s: {"hello": "world"})
        flow.set_entry("hello")
        agent = flow.as_agent()
        # Structural check -- AgentProtocol is runtime_checkable.
        assert isinstance(agent, AgentProtocol)

    def test_compiled_workflow_as_agent(self):
        from cyllama.agents.types import AgentProtocol

        flow = Workflow()
        flow.add_node("hello", lambda s: {"hello": "world"})
        flow.set_entry("hello")
        agent = flow.compile().as_agent()
        assert isinstance(agent, AgentProtocol)

    def test_adapter_run_binds_task_to_task_param(self):
        """adapter.run(task) binds the string to state[task_param]."""
        flow = Workflow()
        flow.add_node("echo", lambda s: {"echo": f"got: {s['task']}"})
        flow.set_entry("echo")
        flow.set_exit("echo")
        result = flow.as_agent().run("hello world")
        assert result.success
        assert result.answer == "got: hello world"

    def test_adapter_run_returns_agent_result_shape(self):
        from cyllama.agents import AgentResult

        flow = Workflow()
        flow.add_node("greet", lambda s: {"greet": f"hello {s['task']}"})
        flow.set_entry("greet")
        flow.set_exit("greet")
        result = flow.as_agent().run("world")
        assert isinstance(result, AgentResult)
        assert result.answer == "hello world"
        assert result.iterations >= 1
        assert result.success

    def test_adapter_run_rejects_non_string_task(self):
        flow = Workflow()
        flow.add_node("a", lambda s: {"a": 1})
        flow.set_entry("a")
        with pytest.raises(TypeError, match="task must be a str"):
            flow.as_agent().run(42)  # type: ignore[arg-type]

    def test_adapter_custom_task_param_override(self):
        """as_agent(task_param=...) overrides the workflow default."""
        flow = Workflow()
        flow.add_node("echo", lambda s: {"echo": s["prompt"]})
        flow.set_entry("echo")
        flow.set_exit("echo")
        agent = flow.as_agent(task_param="prompt")
        result = agent.run("hi")
        assert result.answer == "hi"

    def test_adapter_uses_workflow_task_param_config(self):
        """Workflow(task_param=...) flows through to the adapter."""
        flow = Workflow(task_param="prompt")
        flow.add_node("echo", lambda s: {"echo": s["prompt"]})
        flow.set_entry("echo")
        flow.set_exit("echo")
        result = flow.as_agent().run("hello")
        assert result.answer == "hello"

    def test_adapter_metrics_delegates_to_workflow(self):
        flow = Workflow()
        flow.add_node("a", lambda s: {"a": 1})
        flow.set_entry("a")
        flow.set_exit("a")
        agent = flow.as_agent()
        assert agent.metrics is None
        agent.run("x")
        assert agent.metrics is not None
        assert agent.metrics.tool_calls == 1

    def test_adapter_stream_yields_workflow_events(self):
        from cyllama.agents import EventType

        flow = Workflow()
        flow.add_node("a", lambda s: {"a": 1})
        flow.set_entry("a")
        flow.set_exit("a")
        events = list(flow.as_agent().stream("ignored"))
        # WORKFLOW_START ... NODE_START a ... NODE_END a ... ANSWER ... WORKFLOW_END
        assert events[0].type == EventType.WORKFLOW_START
        assert events[-1].type == EventType.WORKFLOW_END
        assert any(e.type == EventType.ANSWER for e in events)

    def test_workflow_metrics_property(self):
        """Workflow.metrics is still useful for direct callers (non-protocol path)."""
        flow = Workflow()
        flow.add_node("a", lambda s: {"a": 1})
        flow.set_entry("a")
        assert flow.metrics is None
        flow.run()
        assert flow.metrics is not None
        assert flow.metrics.tool_calls == 1

    def test_explicit_answer_key_override(self):
        flow = Workflow(answer_key="out")
        flow.add_node("a", lambda s: {"a": "ignored", "out": "the answer"})
        flow.set_entry("a")
        result = flow.run()
        assert result.answer == "the answer"

    @pytest.mark.asyncio
    async def test_adapter_arun(self):
        flow = Workflow()
        flow.add_node("echo", lambda s: {"echo": f"hi {s['task']}"})
        flow.set_entry("echo")
        flow.set_exit("echo")
        result = await flow.as_agent().arun("there")
        assert result.answer == "hi there"


class TestAgentNodeStreaming:
    def test_agent_node_forwards_sub_events(self):
        """Sub-agent events are interleaved into the outer workflow stream."""
        from cyllama.agents import EventType
        from cyllama.agents.workflow import agent_node

        flow = Workflow()
        flow.add_node("sub", agent_node(_StubAgent("inner", "done"), "sub"))
        flow.set_entry("sub")

        events = list(flow.stream(task="hi"))
        # Sub-agent THOUGHT events should appear with source="sub" and
        # parent_event_id pointing to the outer NODE_START's event_id.
        sub_thoughts = [e for e in events if e.type == EventType.THOUGHT]
        assert len(sub_thoughts) == 2
        assert all(e.source == "sub" for e in sub_thoughts)
        # The NODE_START event_id should be the parent for sub events.
        node_starts = [e for e in events if e.type == EventType.NODE_START]
        assert len(node_starts) == 1
        parent_id = node_starts[0].metadata["event_id"]
        assert all(e.parent_event_id == parent_id for e in sub_thoughts)

    def test_sub_events_arrive_before_node_end(self):
        """Real-time streaming: sub-events precede the node's NODE_END
        in the outer stream, not after it. Regression guard for the
        original batched-after-completion behavior.
        """
        from cyllama.agents import EventType
        from cyllama.agents.workflow import agent_node

        flow = Workflow()
        flow.add_node("sub", agent_node(_StubAgent("inner", "done"), "sub"))
        flow.set_entry("sub")

        events = list(flow.stream(task="hi"))
        # Find positions of the "sub" NODE_END and the THOUGHT events.
        types_and_nodes = [(i, e.type, e.metadata.get("node")) for i, e in enumerate(events)]
        sub_node_end_idx = next(i for i, t, n in types_and_nodes if t == EventType.NODE_END and n == "sub")
        thought_indices = [i for i, t, _ in types_and_nodes if t == EventType.THOUGHT]
        assert thought_indices, "expected sub THOUGHT events"
        assert all(i < sub_node_end_idx for i in thought_indices), (
            f"thoughts {thought_indices} must precede NODE_END {sub_node_end_idx}"
        )

    def test_agent_node_no_forward_uses_run_path(self):
        from cyllama.agents import EventType
        from cyllama.agents.workflow import agent_node

        flow = Workflow()
        flow.add_node("sub", agent_node(_StubAgent("inner", "done"), "sub", forward_events=False))
        flow.set_entry("sub")
        events = list(flow.stream(task="hi"))
        # No sub-events forwarded.
        thoughts = [e for e in events if e.type == EventType.THOUGHT]
        assert thoughts == []
        # But the answer still lands in state.
        end = events[-1]
        assert "sub" in end.metadata["state"]
        assert "done" in end.metadata["state"]["sub"]


class TestWorkflowNode:
    def _build_inner(self):
        inner = Workflow()
        inner.add_node("compute", lambda s: {"compute": f"<{s['task']}>"})
        inner.set_entry("compute")
        inner.set_exit("compute")
        return inner

    def test_workflow_node_runs_subworkflow(self):
        from cyllama.agents.workflow import workflow_node

        outer = Workflow()
        outer.add_node("inner", workflow_node(self._build_inner(), "inner"))
        outer.set_entry("inner")
        result = outer.run(task="hello")
        assert result.success
        assert result.state["inner"] == "<hello>"

    def test_workflow_node_forwards_inner_events(self):
        from cyllama.agents import EventType
        from cyllama.agents.workflow import workflow_node

        outer = Workflow()
        outer.add_node("inner", workflow_node(self._build_inner(), "inner"))
        outer.set_entry("inner")
        events = list(outer.stream(task="hello"))

        # Inner workflow's NODE_START/NODE_END for "compute" should be
        # forwarded with source="inner" and parent_event_id linked to
        # the outer NODE_START.
        forwarded = [e for e in events if e.source == "inner"]
        assert forwarded, "expected sub-workflow events to be forwarded"
        inner_node_starts = [
            e for e in forwarded if e.type == EventType.NODE_START and e.metadata.get("node") == "compute"
        ]
        assert len(inner_node_starts) == 1
        outer_node_start = next(
            e for e in events if e.type == EventType.NODE_START and e.metadata.get("node") == "inner"
        )
        assert inner_node_starts[0].parent_event_id == outer_node_start.metadata["event_id"]

    def test_workflow_node_project_state(self):
        """``project_state=True`` exposes the inner final state."""
        from cyllama.agents.workflow import workflow_node

        outer = Workflow()
        outer.add_node("inner", workflow_node(self._build_inner(), "inner", project_state=True))
        outer.set_entry("inner")
        result = outer.run(task="ping")
        assert result.state["inner"] == {"task": "ping", "compute": "<ping>"}

    def test_workflow_node_propagates_inner_failure(self):
        from cyllama.agents.workflow import workflow_node

        inner = Workflow()

        def boom(state):
            raise RuntimeError("inner exploded")

        inner.add_node("kaboom", boom)
        inner.set_entry("kaboom")
        inner.set_exit("kaboom")

        outer = Workflow()
        outer.add_node("inner", workflow_node(inner, "inner"))
        outer.set_entry("inner")
        result = outer.run(task="x")
        assert not result.success
        assert "inner workflow failed" in (result.error or "")

    def test_workflow_node_accepts_already_compiled(self):
        from cyllama.agents.workflow import workflow_node

        compiled = self._build_inner().compile()
        outer = Workflow()
        outer.add_node("inner", workflow_node(compiled, "inner"))
        outer.set_entry("inner")
        result = outer.run(task="ok")
        assert result.state["inner"] == "<ok>"


class TestAgentAsToolIntegration:
    def test_workflow_wraps_as_tool_via_as_agent(self):
        """A Workflow plugs into ``agent_as_tool`` via ``.as_agent()``."""
        from cyllama.agents.composition import agent_as_tool

        flow = Workflow()
        flow.add_node("echo", lambda s: {"echo": f"answer: {s['task']}"})
        flow.set_entry("echo")
        flow.set_exit("echo")

        tool = agent_as_tool(flow.as_agent(), name="echo_flow", description="Echo the task.")
        # Tool returns the workflow's projected answer.
        out = tool(task="hi")
        assert out == "answer: hi"

    def test_compiled_workflow_wraps_as_tool_via_as_agent(self):
        from cyllama.agents.composition import agent_as_tool

        flow = Workflow()
        flow.add_node("echo", lambda s: {"echo": f"x:{s['task']}"})
        flow.set_entry("echo")
        flow.set_exit("echo")
        agent = flow.compile().as_agent()
        tool = agent_as_tool(agent, name="echo", description="echo")
        assert tool(task="abc") == "x:abc"


class TestReflectionLoopIntegration:
    def test_workflow_as_worker_in_reflection_loop(self):
        """A workflow adapter can act as a worker inside ReflectionLoop."""
        from cyllama.agents import ReflectionLoop

        # Worker workflow: emits a draft.
        worker = Workflow()
        worker.add_node("draft", lambda s: {"draft": f"DRAFT[{s['task']}]"})
        worker.set_entry("draft")
        worker.set_exit("draft")

        # Critic always accepts on first pass.
        critic = _StubAgent("critic", answer="ACCEPT", emit_thoughts=0)

        loop = ReflectionLoop(worker.as_agent(), critic, max_attempts=2)
        result = loop.run("topic")
        assert result.success
        assert "DRAFT[topic]" in result.answer

    def test_workflow_as_critic_in_reflection_loop(self):
        """A workflow adapter can also serve as the critic."""
        from cyllama.agents import ReflectionLoop

        worker = _StubAgent("worker", answer="my draft", emit_thoughts=0)

        # Critic workflow: returns ACCEPT verbatim.
        critic = Workflow()
        critic.add_node("review", lambda s: {"review": "ACCEPT"})
        critic.set_entry("review")
        critic.set_exit("review")

        loop = ReflectionLoop(worker, critic.as_agent(), max_attempts=2)
        result = loop.run("anything")
        assert result.success
        assert "my draft" in result.answer
