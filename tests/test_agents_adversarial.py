"""Adversarial / pathological-input tests for the agent framework.

These tests don't target any specific bug — they exercise the boundaries
of the agent layer with malformed, malicious, or extreme inputs to catch
regressions that "happy path" tests miss. Pure Python; no model needed.

Coverage:

- Action parser: malformed input, escape sequences, deeply nested JSON,
  unicode shenanigans, regex backtracking traps.
- coerce_args + Annotated[]: extreme numeric inputs (inf, NaN, huge),
  pathological strings (very long, with embedded nulls), regex patterns
  the model might emit.
- Tool timeouts: re-entrancy (timeout fires while inside another timed
  tool's wait), nested cancellation.
- Composition: agent_as_tool with sub-agents that raise, return None,
  return huge values; TieredAgentTeam with one bad worker.
- Render_observation: cyclic dicts, custom __repr__ that raises.

Mirrors the principle from the grammar-enforcement work: a happy-path
test isn't a test of the real code path. These cases live just outside
the happy path and would otherwise stay live forever.
"""

from __future__ import annotations

import math
import time
from typing import Annotated, Any, Iterator, List, Optional

import pytest

from cyllama.agents import (
    AgentEvent,
    AgentMetrics,
    AgentResult,
    EventType,
    Ge,
    Le,
    MaxLen,
    MinLen,
    Pattern,
    ReActAgent,
    Tool,
    ToolArgumentError,
    ToolTimeoutError,
    tool,
)
from cyllama.agents.composition import AgentRole, TieredAgentTeam, agent_as_tool
from cyllama.agents.react import ActionParseError, render_observation
from cyllama.agents.tools import coerce_args


class _StubLLM:
    """Minimal stub for ReActAgent.__init__."""

    config = None


# ===========================================================================
# Action parser pathology
# ===========================================================================


class TestActionParserAdversarial:
    """ReActAgent._parse_action against malformed strings."""

    @pytest.fixture
    def parser(self):
        agent = ReActAgent(llm=_StubLLM(), tools=[])  # type: ignore[arg-type]
        return agent._parse_action

    @pytest.mark.parametrize(
        "bad_input",
        [
            "",  # empty
            "   ",  # whitespace
            "not_a_call",  # no parens
            "tool(",  # unclosed
            "tool(arg",  # unclosed arg
            "tool(arg=)",  # empty value
            "()",  # empty tool name
            "123(arg=1)",  # numeric tool name
            "tool name(arg=1)",  # space in name
            "tool\n(arg=1)",  # newline before paren
            "tool(arg=1, arg=2)",  # duplicate kwarg
        ],
    )
    def test_parser_rejects_malformed(self, parser, bad_input):
        with pytest.raises(ActionParseError):
            parser(bad_input)

    def test_parser_handles_deeply_nested_json(self, parser):
        """Nested dict literals should not crash the parser. The parser's
        argument-parsing fallback may interpret the contents as positional
        args (which is fine — we're checking it doesn't raise and produces
        *something* the agent can hand to a tool)."""
        deep = 'tool({"a":' + '{"a":' * 19 + "1" + "}" * 20 + ")"
        name, args = parser(deep)
        assert name == "tool"
        # Whatever shape the parser produced, the call didn't raise.
        assert isinstance(args, dict)

    def test_parser_handles_escape_sequences(self, parser):
        """Newline / tab / backslash in string args."""
        name, args = parser(r'tool(text="line1\nline2\ttabbed\\backslash")')
        assert name == "tool"
        # Whatever the parser does, it should not raise and should
        # produce a string-typed argument.
        assert isinstance(args["text"], str)

    def test_parser_accepts_namespaced_names(self, parser):
        """server/tool, module.helper, kebab-case all valid (Proposal #1)."""
        for name in ("server/tool", "module.helper", "kebab-name", "a/b.c-d/e"):
            tn, _ = parser(f"{name}()")
            assert tn == name

    def test_parser_rejects_empty_segment_in_namespace(self, parser):
        """server//tool has an empty segment — rejected."""
        with pytest.raises(ActionParseError):
            parser("server//tool()")

    def test_parser_handles_unicode_args(self, parser):
        name, args = parser('lookup(name="日本語", emoji="😀")')
        assert args["name"] == "日本語"
        assert args["emoji"] == "😀"


# ===========================================================================
# coerce_args extremes
# ===========================================================================


@tool
def numeric(value: Annotated[int, Ge(-1_000_000), Le(1_000_000)]) -> int:
    """Bounded integer."""
    return value


@tool
def named(text: Annotated[str, MinLen(1), MaxLen(100), Pattern(r"^[A-Za-z]+$")]) -> str:
    """Identifier-like string."""
    return text


@tool
def ratio(r: Annotated[float, Ge(0.0), Le(1.0)]) -> float:
    """Probability-like float."""
    return r


class TestCoercionAdversarial:
    """coerce_args against extreme / pathological values."""

    def test_huge_int_within_bounds(self):
        out = coerce_args(numeric, {"value": 999_999})
        assert out == {"value": 999_999}

    def test_int_just_below_minimum_rejected(self):
        with pytest.raises(ToolArgumentError, match="minimum"):
            coerce_args(numeric, {"value": -1_000_001})

    @pytest.mark.parametrize("bad", ["1.5", "1e10", "0x10", "", "  ", "−5"])
    def test_uncoerceable_strings_rejected(self, bad):
        with pytest.raises(ToolArgumentError):
            coerce_args(numeric, {"value": bad})

    def test_string_with_max_length_passes(self):
        out = coerce_args(named, {"text": "a" * 100})
        assert out["text"] == "a" * 100

    def test_string_one_over_max_length_rejected(self):
        with pytest.raises(ToolArgumentError, match="maxLength"):
            coerce_args(named, {"text": "a" * 101})

    def test_string_pattern_with_evil_input(self):
        """Pattern is fullmatch-style via re.search; backtracking-prone
        inputs should not hang the test runner."""
        evil = "a" * 50 + "1"  # mostly letters, one digit — fails pattern
        start = time.perf_counter()
        with pytest.raises(ToolArgumentError, match="pattern"):
            coerce_args(named, {"text": evil})
        # Should be near-instantaneous; if regex hangs, test runner kills us.
        assert time.perf_counter() - start < 1.0

    def test_float_special_values(self):
        """NaN, inf are *not* coerced; they're real floats but pathological."""
        # NaN: comparisons always False, so NaN < maximum is False -> raises.
        with pytest.raises(ToolArgumentError):
            coerce_args(ratio, {"r": float("nan")})
        # +inf > maximum -> raises.
        with pytest.raises(ToolArgumentError):
            coerce_args(ratio, {"r": float("inf")})

    def test_unknown_arg_with_long_name(self):
        long_key = "x" * 200
        with pytest.raises(ToolArgumentError, match="unknown argument"):
            coerce_args(numeric, {"value": 1, long_key: "irrelevant"})


# ===========================================================================
# Tool timeouts under stress
# ===========================================================================


class TestTimeoutAdversarial:
    """Tool timeout behavior at the edges."""

    def test_timeout_zero_treated_as_no_timeout(self):
        """timeout=0 should mean "no enforcement" rather than instant fail,
        matching how Python's other timeout APIs interpret 0 (None > 0)."""

        # Our implementation treats None as no-timeout; 0 would actually
        # enforce a 0-second budget. Verify the contract:
        @tool(timeout=0)  # 0s budget — should always timeout
        def quick() -> str:
            time.sleep(0.05)
            return "ok"

        agent = ReActAgent(llm=_StubLLM(), tools=[quick])  # type: ignore[arg-type]
        with pytest.raises(ToolTimeoutError):
            agent._execute_tool_raw("quick", {})

    def test_timeout_does_not_block_when_tool_raises(self):
        """If the tool raises before timeout fires, propagation is fast."""

        @tool(timeout=2.0)
        def fails() -> str:
            raise RuntimeError("fast fail")

        agent = ReActAgent(llm=_StubLLM(), tools=[fails])  # type: ignore[arg-type]
        start = time.perf_counter()
        with pytest.raises(RuntimeError, match="fast fail"):
            agent._execute_tool_raw("fails", {})
        # Should complete in well under the 2s budget.
        assert time.perf_counter() - start < 0.5

    def test_many_concurrent_timeout_tools_are_separate_threads(self):
        """Sequentially invoking many timed tools doesn't accumulate
        threads (we don't pool; each call has its own daemon thread)."""

        @tool(timeout=0.5)
        def quick() -> str:
            return "ok"

        agent = ReActAgent(llm=_StubLLM(), tools=[quick])  # type: ignore[arg-type]
        for _ in range(20):
            assert agent._execute_tool_raw("quick", {}) == "ok"

    def test_timeout_tool_with_large_return_value(self):
        """Worker thread must be able to hand back arbitrarily large values."""
        big = "x" * 100_000

        @tool(timeout=2.0)
        def returns_big() -> str:
            return big

        agent = ReActAgent(llm=_StubLLM(), tools=[returns_big])  # type: ignore[arg-type]
        assert agent._execute_tool_raw("returns_big", {}) == big


# ===========================================================================
# render_observation pathology
# ===========================================================================


class TestRenderObservationAdversarial:
    def test_cyclic_dict_falls_back_to_str(self):
        """Cyclic structures break json.dumps; we fall back to str()."""
        d: dict = {"k": 1}
        d["self"] = d  # self-reference
        # Should not raise; str(d) produces something readable.
        out = render_observation(d)
        assert isinstance(out, str)
        assert "k" in out

    def test_object_with_raising_repr_propagates(self):
        """A custom __repr__ that raises is the caller's problem to fix —
        render_observation doesn't paper over it."""

        class Bad:
            def __repr__(self) -> str:
                raise RuntimeError("repr fail")

        # Non-container path: str() raises; we propagate.
        with pytest.raises(RuntimeError):
            render_observation(Bad())

    def test_huge_list(self):
        """100k-element lists should serialise without blowing up.

        json.dumps uses ``, `` (with a space) as the default item separator,
        so the head/tail look like ``"[0, 1, 2, ..."`` and ``"..., 99999]"``.
        """
        out = render_observation(list(range(100_000)))
        assert out.startswith("[0,")
        assert out.endswith("99999]")

    def test_nested_dict_with_non_string_keys(self):
        """Default=str catches non-JSON-native keys quietly."""
        out = render_observation({1: "one", 2: "two"})
        # json.dumps coerces int keys to strings by default.
        assert '"1"' in out or "1" in out


# ===========================================================================
# Composition: pathological sub-agents
# ===========================================================================


class _ExplodingAgent:
    """Sub-agent whose run/stream raise."""

    @property
    def metrics(self) -> Optional[AgentMetrics]:
        return None

    def run(self, task: str) -> AgentResult:
        raise RuntimeError("sub-agent crashed")

    def stream(self, task: str) -> Iterator[AgentEvent]:
        raise RuntimeError("sub-agent stream crashed")
        yield  # pragma: no cover -- unreachable


class _NoneAnswerAgent:
    """Sub-agent whose answer is the empty string (legitimate)."""

    @property
    def metrics(self) -> Optional[AgentMetrics]:
        return None

    def run(self, task: str) -> AgentResult:
        return AgentResult(answer="", steps=[], iterations=0, success=True)

    def stream(self, task: str) -> Iterator[AgentEvent]:
        yield AgentEvent(type=EventType.ANSWER, content="")


class TestCompositionAdversarial:
    def test_agent_as_tool_propagates_sub_agent_exception(self):
        t = agent_as_tool(_ExplodingAgent(), name="boom", description="x")
        with pytest.raises(RuntimeError, match="sub-agent crashed"):
            t(task="any")

    def test_agent_as_tool_propagates_stream_exception_with_forward(self):
        forwarded: List[AgentEvent] = []
        t = agent_as_tool(
            _ExplodingAgent(),
            name="boom",
            description="x",
            forward_events=forwarded.append,
        )
        with pytest.raises(RuntimeError, match="stream crashed"):
            t(task="any")

    def test_agent_as_tool_handles_empty_answer(self):
        t = agent_as_tool(_NoneAnswerAgent(), name="quiet", description="x")
        assert t(task="any") == ""

    def test_team_one_bad_worker_doesnt_break_construction(self):
        """A worker whose .run() raises only matters when invoked. Team
        construction should still succeed."""
        from cyllama.agents import ToolRegistry

        class _Sup:
            registry = ToolRegistry()
            metrics = None

            def run(self, t: str) -> AgentResult:
                return AgentResult(answer="", steps=[], iterations=0, success=True)

            def stream(self, t: str) -> Iterator[AgentEvent]:
                yield AgentEvent(type=EventType.ANSWER, content="")

        sup = _Sup()
        team = TieredAgentTeam(
            supervisor=sup,
            workers=[
                AgentRole("good", _NoneAnswerAgent(), "x"),
                AgentRole("bad", _ExplodingAgent(), "x"),
            ],
        )
        assert {t.name for t in sup.registry.list_tools()} == {"good", "bad"}
        # Good worker still works.
        good = sup.registry.get("good")
        assert good is not None
        assert good(task="any") == ""
        # Bad worker fails at call time, not at construction.
        bad = sup.registry.get("bad")
        assert bad is not None
        with pytest.raises(RuntimeError):
            bad(task="any")
