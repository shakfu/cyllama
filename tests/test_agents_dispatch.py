"""One tool-dispatch path (call_tool) for ReAct, constrained and workflow agents."""

import contextvars
import time
from unittest.mock import MagicMock

import pytest

from cyllama.agents.contract import ContractAgent, ContractPolicy, contract_assert, pre
from cyllama.agents.tools import AgentTermination, ToolArgumentError, ToolTimeoutError, call_tool, tool


@tool(timeout=0.2)
def slow() -> str:
    """Sleeps past its timeout."""
    time.sleep(2)
    return "late"


@tool
def add_one(count: int) -> int:
    """Adds one."""
    return count + 1


class TestCallTool:
    def test_coerces(self):
        assert call_tool(add_one, {"count": "5"}) == 6

    def test_coerce_false_skips_coercion(self):
        with pytest.raises(TypeError):
            call_tool(add_one, {"count": "5"}, coerce=False)

    def test_rejects_bad_args(self):
        with pytest.raises(ToolArgumentError):
            call_tool(add_one, {"count": "x"})

    def test_timeout(self):
        with pytest.raises(ToolTimeoutError):
            call_tool(slow, {})

    def test_timed_tool_sees_caller_context(self):
        var = contextvars.ContextVar("var", default="unset")

        @tool(timeout=5)
        def read_var() -> str:
            """Returns the context variable."""
            return var.get()

        var.set("caller")
        assert call_tool(read_var, {}) == "caller"


def test_constrained_agent_honours_timeout():
    from cyllama.agents.constrained import ConstrainedAgent

    agent = ConstrainedAgent(llm=MagicMock(), tools=[slow])
    observation, raw = agent._call_tool("slow", {})
    assert "exceeded" in observation and raw is None


def test_constrained_agent_returns_raw_result_for_postconditions():
    from cyllama.agents.constrained import ConstrainedAgent

    agent = ConstrainedAgent(llm=MagicMock(), tools=[add_one])
    assert agent._call_tool("add_one", {"count": "1"}) == ("2", 2)


def test_constrained_agent_reraises_termination():
    from cyllama.agents.constrained import ConstrainedAgent

    @tool
    def stop() -> str:
        """Stops the run."""
        raise AgentTermination("stop")

    with pytest.raises(AgentTermination):
        ConstrainedAgent(llm=MagicMock(), tools=[stop])._call_tool("stop", {})


def test_workflow_tool_node_honours_timeout():
    from cyllama.agents.workflow import tool_node

    with pytest.raises(ToolTimeoutError):
        tool_node(slow)({})


def _llm(*replies):
    llm = MagicMock()
    llm.side_effect = list(replies)
    return llm


def test_contract_assert_enforce_stops_react_run():
    @tool
    def guarded() -> str:
        """Fails its contract."""
        contract_assert(False, "bad state")
        return "unreachable"

    llm = _llm("Thought: call it\nAction: guarded()\n", "Thought: done\nAnswer: fine")
    result = ContractAgent(llm=llm, tools=[guarded], policy=ContractPolicy.ENFORCE).run("task")
    # it used to become an observation and the run went on to answer
    assert result.answer == ""
    assert "bad state" in (result.error or "")
    assert llm.call_count == 1


def test_precondition_sees_coerced_args():
    @tool
    @pre(lambda args: args["count"] > 0, "count must be positive")
    def positive(count: int) -> int:
        """Needs a positive count."""
        return count

    agent = ContractAgent(llm=MagicMock(), tools=[positive], policy=ContractPolicy.ENFORCE)
    # the model writes "5"; coercion makes it 5 before the tool runs
    assert agent._check_preconditions("positive", {"count": "5"}) is None
    assert agent._check_preconditions("positive", {"count": "-1"}) is not None


class TestCalculatorLimits:
    """The exponent cap alone allowed ((9**999)**999)**999: ~3e9 bits, computed holding the GIL."""

    def _calc(self, expr):
        from cyllama.agents.tools import calculator

        return calculator(expression=expr)

    def test_nested_power_rejected_quickly(self):
        start = time.perf_counter()
        with pytest.raises(ValueError, match="result too large"):
            self._calc("((9**999)**999)**999")
        assert time.perf_counter() - start < 1.0

    def test_large_product_rejected(self):
        with pytest.raises(ValueError, match="result too large"):
            self._calc("(9**999)**30 * (9**999)**30")  # each factor ~95k bits, under the cap alone

    @pytest.mark.parametrize("expr,expected", [("2**10", "1024"), ("(2+3)*4", "20"), ("9**999 % 7", str(9**999 % 7))])
    def test_ordinary_arithmetic(self, expr, expected):
        assert self._calc(expr) == expected

    def test_exponent_cap_still_applies(self):
        with pytest.raises(ValueError, match="exponent"):
            self._calc("2**10000")


class TestModelOutputParsing:
    def _agent(self):
        from cyllama.agents.react import ReActAgent

        return ReActAgent(llm=MagicMock(), tools=[])

    def test_answer_inside_thought_is_not_final(self):
        text = "Thought: The answer: is unclear, I should search.\nAction: search(query='x')"
        agent = self._agent()
        assert agent._extract_answer(text) is None
        assert agent._extract_action(text) == "search(query='x')"

    @pytest.mark.parametrize("text", ["Answer: 42", "Thought: done\nAnswer: 42", "Thought: done\nFinal Answer: 42"])
    def test_answer_line(self, text):
        assert self._agent()._extract_answer(text) == "42"

    @pytest.mark.parametrize(
        "action,expected",
        [
            ("server/tool(count=5)", {"count": 5}),
            ('server/tool(path="a", n=5)', {"path": "a", "n": 5}),  # n was dropped
        ],
    )
    def test_namespaced_tool_args(self, action, expected):
        assert self._agent()._parse_action(action) == ("server/tool", expected)


class TestVerdictAccepts:
    @pytest.mark.parametrize("answer", ["ACCEPT", "I accept this draft.", "Accepted.", "Verdict: ACCEPT"])
    def test_accepts(self, answer):
        from cyllama.agents.composition import verdict_accepts

        assert verdict_accepts(answer)

    @pytest.mark.parametrize("answer", ["This is unacceptable.", "I cannot accept this.", "Do not accept.", "REVISE"])
    def test_rejects(self, answer):
        from cyllama.agents.composition import verdict_accepts

        assert not verdict_accepts(answer)


def test_plan_and_execute_caps_steps():
    from cyllama.agents.composition import plan_and_execute
    from cyllama.agents.types import AgentResult

    planner = MagicMock()
    planner.run.return_value = AgentResult(
        answer="\n".join(f"step {i}" for i in range(500)), steps=[], iterations=1, success=True
    )
    executor = MagicMock()
    executor.run.return_value = AgentResult(answer="ok", steps=[], iterations=1, success=True)
    assert len(plan_and_execute(planner, executor, "task", max_steps=5)) == 5
    assert executor.run.call_count == 5
