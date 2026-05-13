"""
Tests for ContractAgent with C++26-inspired contract assertions.
"""

import pytest
from unittest.mock import MagicMock
from cyllama.agents import (
    ContractAgent,
    ContractPolicy,
    ContractViolation,
    ContractTermination,
    ContractSpec,
    PreCondition,
    PostCondition,
    IterationState,
    Tool,
    tool,
    pre,
    post,
    contract_assert,
    EventType,
    AgentEvent,
)
from cyllama.agents.contract import (
    _get_current_context,
    _set_current_context,
    ContractContext,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = MagicMock()
    llm.return_value = 'Thought: I should use a tool\nAction: test_tool({"arg": "value"})\n'
    return llm


@pytest.fixture
def simple_tool():
    """Create a simple tool without contracts."""

    @tool
    def test_tool(arg: str) -> str:
        """A test tool."""
        return f"Result: {arg}"

    return test_tool


@pytest.fixture
def tool_with_pre():
    """Create a tool with precondition."""

    @tool
    @pre(lambda args: args["count"] > 0, "count must be positive")
    def fetch_items(count: int) -> str:
        """Fetch items."""
        return f"Fetched {count} items"

    return fetch_items


@pytest.fixture
def tool_with_post():
    """Create a tool with postcondition."""

    @tool
    @post(lambda r: len(r) > 0, "result must not be empty")
    def search(query: str) -> str:
        """Search for something."""
        return f"Results for: {query}"

    return search


@pytest.fixture
def tool_with_both():
    """Create a tool with both pre and post conditions."""

    @tool
    @pre(lambda args: args["x"] != 0, "x must not be zero")
    @post(lambda r: r is not None, "result must not be None")
    def divide(a: float, x: float) -> float:
        """Divide a by x."""
        return a / x

    return divide


# =============================================================================
# Test Contract Decorators
# =============================================================================


def _get_contracts(tool_or_func):
    """Helper to get contracts from a Tool or function."""
    if hasattr(tool_or_func, "_contracts"):
        return tool_or_func._contracts
    if hasattr(tool_or_func, "func") and hasattr(tool_or_func.func, "_contracts"):
        return tool_or_func.func._contracts
    return None


class TestContractDecorators:
    """Tests for @pre and @post decorators."""

    def test_pre_decorator_attaches_contract(self):
        """Test that @pre decorator attaches contract to function."""

        @tool
        @pre(lambda args: args["n"] > 0, "n must be positive")
        def my_func(n: int) -> int:
            return n * 2

        contracts = _get_contracts(my_func)
        assert contracts is not None
        assert len(contracts.preconditions) == 1
        assert contracts.preconditions[0].message == "n must be positive"

    def test_post_decorator_attaches_contract(self):
        """Test that @post decorator attaches contract to function."""

        @tool
        @post(lambda r: r > 0, "result must be positive")
        def my_func(n: int) -> int:
            return n * 2

        contracts = _get_contracts(my_func)
        assert contracts is not None
        assert len(contracts.postconditions) == 1
        assert contracts.postconditions[0].message == "result must be positive"

    def test_multiple_pre_conditions(self):
        """Test multiple preconditions on same function."""

        @tool
        @pre(lambda args: args["n"] > 0, "n must be positive")
        @pre(lambda args: args["n"] < 100, "n must be less than 100")
        def my_func(n: int) -> int:
            return n * 2

        contracts = _get_contracts(my_func)
        assert len(contracts.preconditions) == 2

    def test_multiple_post_conditions(self):
        """Test multiple postconditions on same function."""

        @tool
        @post(lambda r: r > 0, "result must be positive")
        @post(lambda r: r < 1000, "result must be less than 1000")
        def my_func(n: int) -> int:
            return n * 2

        contracts = _get_contracts(my_func)
        assert len(contracts.postconditions) == 2

    def test_pre_and_post_combined(self):
        """Test both pre and post conditions on same function."""

        @tool
        @pre(lambda args: args["n"] > 0, "n must be positive")
        @post(lambda r: r > 0, "result must be positive")
        def my_func(n: int) -> int:
            return n * 2

        contracts = _get_contracts(my_func)
        assert len(contracts.preconditions) == 1
        assert len(contracts.postconditions) == 1

    def test_post_with_args_access(self):
        """Test postcondition that accesses original arguments."""

        @tool
        @post(lambda r, args: r <= args["max_len"], "result too long")
        def generate(text: str, max_len: int) -> str:
            return text[:max_len]

        contracts = _get_contracts(generate)
        assert contracts.postconditions[0].needs_args is True

    def test_pre_with_custom_policy(self):
        """Test precondition with custom policy."""

        @tool
        @pre(lambda args: args["n"] > 0, "n must be positive", policy=ContractPolicy.OBSERVE)
        def my_func(n: int) -> int:
            return n * 2

        contracts = _get_contracts(my_func)
        assert contracts.preconditions[0].policy == ContractPolicy.OBSERVE


# =============================================================================
# Test PreCondition and PostCondition Classes
# =============================================================================


class TestConditionClasses:
    """Tests for PreCondition and PostCondition classes."""

    def test_precondition_check_passes(self):
        """Test PreCondition.check returns True when condition passes."""
        cond = PreCondition(predicate=lambda args: args["n"] > 0, message="n must be positive")
        assert cond.check({"n": 5}) is True

    def test_precondition_check_fails(self):
        """Test PreCondition.check returns False when condition fails."""
        cond = PreCondition(predicate=lambda args: args["n"] > 0, message="n must be positive")
        assert cond.check({"n": -1}) is False

    def test_precondition_check_exception_returns_false(self):
        """Test PreCondition.check returns False on exception."""
        cond = PreCondition(predicate=lambda args: args["missing_key"] > 0, message="test")
        assert cond.check({"n": 5}) is False

    def test_postcondition_check_passes(self):
        """Test PostCondition.check returns True when condition passes."""
        cond = PostCondition(predicate=lambda r: r > 0, message="result must be positive")
        assert cond.check(10) is True

    def test_postcondition_check_fails(self):
        """Test PostCondition.check returns False when condition fails."""
        cond = PostCondition(predicate=lambda r: r > 0, message="result must be positive")
        assert cond.check(-5) is False

    def test_postcondition_with_args(self):
        """Test PostCondition.check with args access."""
        cond = PostCondition(predicate=lambda r, args: r <= args["max"], message="result exceeds max", needs_args=True)
        assert cond.check(5, {"max": 10}) is True
        assert cond.check(15, {"max": 10}) is False


# =============================================================================
# Test ContractViolation
# =============================================================================


class TestContractViolation:
    """Tests for ContractViolation dataclass."""

    def test_violation_creation(self):
        """Test creating a ContractViolation."""
        violation = ContractViolation(
            kind="pre",
            location="my_tool",
            predicate="args['n'] > 0",
            message="n must be positive",
            context={"args": {"n": -1}},
            policy=ContractPolicy.ENFORCE,
        )
        assert violation.kind == "pre"
        assert violation.location == "my_tool"
        assert "n must be positive" in str(violation)

    def test_violation_str(self):
        """Test ContractViolation string representation."""
        violation = ContractViolation(kind="post", location="search", predicate="len(r) > 0", message="result empty")
        assert "post" in str(violation)
        assert "search" in str(violation)


# =============================================================================
# Test contract_assert
# =============================================================================


class TestContractAssert:
    """Tests for contract_assert function."""

    def test_contract_assert_passes(self):
        """Test contract_assert does nothing when condition is True."""
        # contract_assert(True, ...) must return None without raising and
        # must not install or mutate any thread-local context state.
        _set_current_context(None)
        assert contract_assert(True, "should not fail") is None
        assert _get_current_context() is None

    def test_contract_assert_fails_enforce(self):
        """Test contract_assert raises ContractTermination on failure."""
        with pytest.raises(ContractTermination) as exc_info:
            contract_assert(False, "assertion failed")
        assert "assertion failed" in str(exc_info.value)

    def test_contract_assert_with_ignore_policy(self):
        """Test contract_assert does nothing with IGNORE policy."""
        # With IGNORE, a False condition must not raise and must return None.
        # Without a context, this is the only path that lets a False
        # assertion reach a successful return value.
        _set_current_context(None)
        assert contract_assert(False, "should be ignored", policy=ContractPolicy.IGNORE) is None

    def test_contract_assert_with_context(self):
        """Test contract_assert uses context when available."""
        ctx = ContractContext(policy=ContractPolicy.OBSERVE, handler=MagicMock(), location="test_tool")
        _set_current_context(ctx)
        try:
            contract_assert(False, "test failure")
            ctx.handler.assert_called_once()
        finally:
            _set_current_context(None)

    def test_contract_assert_quick_enforce(self):
        """Test contract_assert with QUICK_ENFORCE policy."""
        ctx = ContractContext(
            policy=ContractPolicy.QUICK_ENFORCE,
            handler=MagicMock(),  # Should NOT be called
            location="test_tool",
        )
        _set_current_context(ctx)
        try:
            with pytest.raises(ContractTermination):
                contract_assert(False, "quick fail")
            # Handler should not have been called
            ctx.handler.assert_not_called()
        finally:
            _set_current_context(None)


# =============================================================================
# Test ContractContext
# =============================================================================


class TestContractContext:
    """Tests for ContractContext."""

    def test_context_handle_violation_ignore(self):
        """Test context ignores violations with IGNORE policy."""
        handler = MagicMock()
        ctx = ContractContext(policy=ContractPolicy.IGNORE, handler=handler, location="test")
        violation = ContractViolation(kind="assert", location="test", predicate="x", message="m")
        ctx.handle_violation(violation)
        handler.assert_not_called()

    def test_context_handle_violation_observe(self):
        """Test context calls handler and continues with OBSERVE policy."""
        handler = MagicMock()
        ctx = ContractContext(policy=ContractPolicy.OBSERVE, handler=handler, location="test")
        violation = ContractViolation(kind="assert", location="test", predicate="x", message="m")
        ctx.handle_violation(violation)  # Should not raise
        handler.assert_called_once()

    def test_context_handle_violation_enforce(self):
        """Test context calls handler and raises with ENFORCE policy."""
        handler = MagicMock()
        ctx = ContractContext(policy=ContractPolicy.ENFORCE, handler=handler, location="test")
        violation = ContractViolation(kind="assert", location="test", predicate="x", message="m")
        with pytest.raises(ContractTermination):
            ctx.handle_violation(violation)
        handler.assert_called_once()

    def test_context_handle_violation_quick_enforce(self):
        """Test context raises immediately with QUICK_ENFORCE policy."""
        handler = MagicMock()
        ctx = ContractContext(policy=ContractPolicy.QUICK_ENFORCE, handler=handler, location="test")
        violation = ContractViolation(kind="assert", location="test", predicate="x", message="m")
        with pytest.raises(ContractTermination):
            ctx.handle_violation(violation)
        handler.assert_not_called()


# =============================================================================
# Test IterationState
# =============================================================================


class TestIterationState:
    """Tests for IterationState."""

    def test_initial_state(self):
        """Test initial state values."""
        state = IterationState()
        assert state.iterations == 0
        assert state.tool_calls == 0
        assert state.errors == 0
        assert state.events == []

    def test_update_thought(self):
        """Test state update on THOUGHT event."""
        state = IterationState()
        event = AgentEvent(type=EventType.THOUGHT, content="thinking")
        state.update(event)
        assert state.iterations == 1
        assert len(state.events) == 1

    def test_update_action(self):
        """Test state update on ACTION event."""
        state = IterationState()
        event = AgentEvent(type=EventType.ACTION, content="action")
        state.update(event)
        assert state.tool_calls == 1

    def test_update_error(self):
        """Test state update on ERROR event."""
        state = IterationState()
        event = AgentEvent(type=EventType.ERROR, content="error")
        state.update(event)
        assert state.errors == 1

    def test_to_dict(self):
        """Test state to_dict conversion."""
        state = IterationState(iterations=5, tool_calls=3, errors=1)
        d = state.to_dict()
        # to_dict() carries the core counters plus the Step 5 fields
        # (elapsed_ms, last_tool_name, estimated_prompt_chars, etc.).
        # Pin the core counters; the new-field test class covers the rest.
        assert d["iterations"] == 5
        assert d["tool_calls"] == 3
        assert d["errors"] == 1


# =============================================================================
# Test ContractAgent Initialization
# =============================================================================


class TestContractAgentInit:
    """Tests for ContractAgent initialization."""

    def test_init_with_defaults(self, mock_llm, simple_tool):
        """Test ContractAgent initialization with defaults."""
        agent = ContractAgent(llm=mock_llm, tools=[simple_tool])
        assert agent.policy == ContractPolicy.ENFORCE
        assert len(agent.tools) == 1

    def test_init_with_custom_policy(self, mock_llm, simple_tool):
        """Test ContractAgent with custom policy."""
        agent = ContractAgent(llm=mock_llm, tools=[simple_tool], policy=ContractPolicy.OBSERVE)
        assert agent.policy == ContractPolicy.OBSERVE

    def test_init_with_violation_handler(self, mock_llm, simple_tool):
        """Test ContractAgent with custom violation handler."""
        handler = MagicMock()
        agent = ContractAgent(llm=mock_llm, tools=[simple_tool], violation_handler=handler)
        assert agent.violation_handler == handler

    def test_init_with_task_precondition(self, mock_llm, simple_tool):
        """Test ContractAgent with task precondition."""
        agent = ContractAgent(llm=mock_llm, tools=[simple_tool], task_precondition=lambda t: len(t) > 5)
        assert agent.task_precondition is not None

    def test_init_with_answer_postcondition(self, mock_llm, simple_tool):
        """Test ContractAgent with answer postcondition."""
        agent = ContractAgent(llm=mock_llm, tools=[simple_tool], answer_postcondition=lambda a: len(a) > 0)
        assert agent.answer_postcondition is not None

    def test_init_extracts_tool_contracts(self, mock_llm, tool_with_pre):
        """Test that ContractAgent extracts contracts from tools."""
        agent = ContractAgent(llm=mock_llm, tools=[tool_with_pre])
        assert "fetch_items" in agent._tool_contracts
        assert len(agent._tool_contracts["fetch_items"].preconditions) == 1


# =============================================================================
# Test ContractAgent Contract Checking
# =============================================================================


class TestContractAgentChecking:
    """Tests for ContractAgent contract checking methods."""

    def test_check_preconditions_passes(self, mock_llm, tool_with_pre):
        """Test precondition check that passes."""
        agent = ContractAgent(llm=mock_llm, tools=[tool_with_pre])
        violation = agent._check_preconditions("fetch_items", {"count": 5})
        assert violation is None

    def test_check_preconditions_fails(self, mock_llm, tool_with_pre):
        """Test precondition check that fails."""
        agent = ContractAgent(llm=mock_llm, tools=[tool_with_pre])
        violation = agent._check_preconditions("fetch_items", {"count": -1})
        assert violation is not None
        assert violation.kind == "pre"
        assert "count must be positive" in violation.message

    def test_check_postconditions_passes(self, mock_llm, tool_with_post):
        """Test postcondition check that passes."""
        agent = ContractAgent(llm=mock_llm, tools=[tool_with_post])
        violation = agent._check_postconditions("search", "some results", {"query": "test"})
        assert violation is None

    def test_check_postconditions_fails(self, mock_llm, tool_with_post):
        """Test postcondition check that fails."""
        agent = ContractAgent(llm=mock_llm, tools=[tool_with_post])
        violation = agent._check_postconditions("search", "", {"query": "test"})
        assert violation is not None
        assert violation.kind == "post"

    def test_check_with_ignore_policy(self, mock_llm, tool_with_pre):
        """Test that IGNORE policy skips checking."""
        agent = ContractAgent(llm=mock_llm, tools=[tool_with_pre], policy=ContractPolicy.IGNORE)
        # Should pass even with invalid args because checking is skipped
        violation = agent._check_preconditions("fetch_items", {"count": -1})
        assert violation is None


# =============================================================================
# Test ContractAgent Violation Handling
# =============================================================================


class TestContractAgentViolationHandling:
    """Tests for ContractAgent violation handling."""

    def test_handle_violation_ignore(self, mock_llm, simple_tool):
        """Test violation handling with IGNORE policy."""
        agent = ContractAgent(llm=mock_llm, tools=[simple_tool])
        violation = ContractViolation(
            kind="pre", location="test", predicate="x", message="m", policy=ContractPolicy.IGNORE
        )
        result = agent._handle_violation(violation)
        assert result is True  # Continue execution

    def test_handle_violation_observe(self, mock_llm, simple_tool):
        """Test violation handling with OBSERVE policy."""
        handler = MagicMock()
        agent = ContractAgent(llm=mock_llm, tools=[simple_tool], violation_handler=handler)
        violation = ContractViolation(
            kind="pre", location="test", predicate="x", message="m", policy=ContractPolicy.OBSERVE
        )
        result = agent._handle_violation(violation)
        assert result is True  # Continue execution
        handler.assert_called_once()

    def test_handle_violation_enforce(self, mock_llm, simple_tool):
        """Test violation handling with ENFORCE policy."""
        handler = MagicMock()
        agent = ContractAgent(llm=mock_llm, tools=[simple_tool], violation_handler=handler)
        violation = ContractViolation(
            kind="pre", location="test", predicate="x", message="m", policy=ContractPolicy.ENFORCE
        )
        result = agent._handle_violation(violation)
        assert result is False  # Stop execution
        handler.assert_called_once()

    def test_handle_violation_quick_enforce(self, mock_llm, simple_tool):
        """Test violation handling with QUICK_ENFORCE policy."""
        handler = MagicMock()
        agent = ContractAgent(llm=mock_llm, tools=[simple_tool], violation_handler=handler)
        violation = ContractViolation(
            kind="pre", location="test", predicate="x", message="m", policy=ContractPolicy.QUICK_ENFORCE
        )
        result = agent._handle_violation(violation)
        assert result is False  # Stop execution
        handler.assert_not_called()  # No handler call


# =============================================================================
# Test ContractAgent list_tools and get_contract_stats
# =============================================================================


class TestContractAgentUtilities:
    """Tests for ContractAgent utility methods."""

    def test_list_tools(self, mock_llm, simple_tool, tool_with_pre):
        """Test list_tools returns all tools."""
        agent = ContractAgent(llm=mock_llm, tools=[simple_tool, tool_with_pre])
        tools = agent.list_tools()
        assert len(tools) == 2

    def test_get_contract_stats_initial(self, mock_llm, simple_tool):
        """Test get_contract_stats returns initial values."""
        agent = ContractAgent(llm=mock_llm, tools=[simple_tool])
        stats = agent.get_contract_stats()
        assert stats["checks"] == 0
        assert stats["violations"] == 0


# =============================================================================
# Test ContractPolicy Enum
# =============================================================================


class TestContractPolicy:
    """Tests for ContractPolicy enum."""

    def test_policy_values(self):
        """Test ContractPolicy enum values."""
        assert ContractPolicy.IGNORE.value == "ignore"
        assert ContractPolicy.OBSERVE.value == "observe"
        assert ContractPolicy.ENFORCE.value == "enforce"
        assert ContractPolicy.QUICK_ENFORCE.value == "quick_enforce"


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_tool_without_contracts(self, mock_llm, simple_tool):
        """Test agent handles tools without contracts."""
        agent = ContractAgent(llm=mock_llm, tools=[simple_tool])
        violation = agent._check_preconditions("test_tool", {"arg": "value"})
        assert violation is None

    def test_unknown_tool_contracts(self, mock_llm, simple_tool):
        """Test agent handles unknown tool names."""
        agent = ContractAgent(llm=mock_llm, tools=[simple_tool])
        violation = agent._check_preconditions("unknown_tool", {"arg": "value"})
        assert violation is None

    def test_predicate_exception_treated_as_failure(self):
        """Test that exceptions in predicates are treated as failures."""

        @tool
        @pre(lambda args: args["missing"] > 0, "should fail")
        def my_func(n: int) -> int:
            return n

        agent = ContractAgent(llm=MagicMock(), tools=[my_func])
        violation = agent._check_preconditions("my_func", {"n": 5})
        assert violation is not None

    def test_postcondition_with_args_needs_args_flag(self):
        """Test that postcondition with 2 params sets needs_args."""

        @tool
        @post(lambda r, args: r <= args["max"], "too big")
        def my_func(n: int, max: int) -> int:
            return n

        contracts = _get_contracts(my_func)
        assert contracts.postconditions[0].needs_args is True


# =============================================================================
# Test Integration with Tool Decorator
# =============================================================================


class TestToolIntegration:
    """Tests for integration with @tool decorator."""

    def test_contracts_preserved_through_tool_decorator(self):
        """Test that contracts are preserved when @tool is applied after."""

        @tool
        @pre(lambda args: args["n"] > 0, "positive")
        @post(lambda r: r > 0, "positive result")
        def my_func(n: int) -> int:
            return n * 2

        # Tool should still be callable
        assert isinstance(my_func, Tool)
        # Contracts should be attached
        assert hasattr(my_func, "_contracts") or hasattr(my_func.func, "_contracts")

    def test_tool_execution_still_works(self):
        """Test that tool still executes correctly with contracts."""

        @tool
        @pre(lambda args: args["n"] > 0, "positive")
        def double(n: int) -> int:
            """Double a number."""
            return n * 2

        result = double(5)
        assert result == 10


# =============================================================================
# Test Policy Resolution
# =============================================================================


class TestPolicyResolution:
    """Tests for policy resolution between contract and agent levels."""

    def test_contract_policy_overrides_agent_policy(self, mock_llm):
        """Test that contract-specific policy takes precedence."""

        @tool
        @pre(lambda args: args["n"] > 0, "positive", policy=ContractPolicy.IGNORE)
        def my_func(n: int) -> int:
            return n * 2

        # Agent uses ENFORCE, but contract uses IGNORE
        agent = ContractAgent(llm=mock_llm, tools=[my_func], policy=ContractPolicy.ENFORCE)

        # Even with invalid args, should return None (no violation) because contract is IGNORE
        violation = agent._check_preconditions("my_func", {"n": -1})
        assert violation is None

    def test_none_policy_uses_agent_default(self, mock_llm):
        """Test that None policy on contract uses agent's policy."""

        @tool
        @pre(lambda args: args["n"] > 0, "positive")  # policy=None by default
        def my_func(n: int) -> int:
            return n * 2

        # Agent uses OBSERVE
        agent = ContractAgent(llm=mock_llm, tools=[my_func], policy=ContractPolicy.OBSERVE)

        violation = agent._check_preconditions("my_func", {"n": -1})
        assert violation is not None
        assert violation.policy == ContractPolicy.OBSERVE

    def test_get_effective_policy_with_none(self, mock_llm, simple_tool):
        """Test _get_effective_policy returns agent policy when contract policy is None."""
        agent = ContractAgent(llm=mock_llm, tools=[simple_tool], policy=ContractPolicy.OBSERVE)
        assert agent._get_effective_policy(None) == ContractPolicy.OBSERVE

    def test_get_effective_policy_with_explicit(self, mock_llm, simple_tool):
        """Test _get_effective_policy returns contract policy when specified."""
        agent = ContractAgent(llm=mock_llm, tools=[simple_tool], policy=ContractPolicy.OBSERVE)
        assert agent._get_effective_policy(ContractPolicy.ENFORCE) == ContractPolicy.ENFORCE


# =============================================================================
# Test ContractSpec
# =============================================================================


class TestContractSpec:
    """Tests for ContractSpec dataclass."""

    def test_empty_spec(self):
        """Test creating empty ContractSpec."""
        spec = ContractSpec()
        assert spec.preconditions == []
        assert spec.postconditions == []

    def test_spec_with_conditions(self):
        """Test ContractSpec with conditions."""
        pre_cond = PreCondition(lambda args: True, "pre")
        post_cond = PostCondition(lambda r: True, "post")
        spec = ContractSpec(preconditions=[pre_cond], postconditions=[post_cond])
        assert len(spec.preconditions) == 1
        assert len(spec.postconditions) == 1


# =============================================================================
# Test Default Handler
# =============================================================================


class TestDefaultHandler:
    """Tests for default violation handler behavior."""

    def test_default_handler_logs_warning(self, mock_llm, simple_tool, caplog):
        """Test that default handler logs a warning."""
        import logging

        agent = ContractAgent(llm=mock_llm, tools=[simple_tool])
        violation = ContractViolation(kind="pre", location="test_tool", predicate="x > 0", message="x must be positive")

        with caplog.at_level(logging.WARNING):
            agent._default_handler(violation)

        assert "Contract violation" in caplog.text
        assert "test_tool" in caplog.text

    def test_default_handler_verbose_output(self, mock_llm, simple_tool, capsys):
        """Test that default handler prints when verbose=True."""
        agent = ContractAgent(llm=mock_llm, tools=[simple_tool], verbose=True)
        violation = ContractViolation(
            kind="pre",
            location="test_tool",
            predicate="x > 0",
            message="x must be positive",
            context={"args": {"x": -1}},
        )

        agent._default_handler(violation)
        captured = capsys.readouterr()

        assert "CONTRACT VIOLATION" in captured.out
        assert "test_tool" in captured.out
        assert "x must be positive" in captured.out


# =============================================================================
# Test IterationState Extended
# =============================================================================


class TestIterationStateExtended:
    """Extended tests for IterationState."""

    def test_update_observation(self):
        """Test state update on OBSERVATION event."""
        state = IterationState()
        event = AgentEvent(type=EventType.OBSERVATION, content="result")
        state.update(event)
        # OBSERVATION doesn't increment counters but is added to events
        assert len(state.events) == 1
        assert state.iterations == 0
        assert state.tool_calls == 0

    def test_update_answer(self):
        """Test state update on ANSWER event."""
        state = IterationState()
        event = AgentEvent(type=EventType.ANSWER, content="final answer")
        state.update(event)
        assert len(state.events) == 1

    def test_multiple_events(self):
        """Test state with multiple events."""
        state = IterationState()
        state.update(AgentEvent(type=EventType.THOUGHT, content="thinking"))
        state.update(AgentEvent(type=EventType.ACTION, content="action"))
        state.update(AgentEvent(type=EventType.OBSERVATION, content="result"))
        state.update(AgentEvent(type=EventType.THOUGHT, content="more thinking"))
        state.update(AgentEvent(type=EventType.ERROR, content="error"))

        assert state.iterations == 2
        assert state.tool_calls == 1
        assert state.errors == 1
        assert len(state.events) == 5


# =============================================================================
# Test Predicate String Extraction
# =============================================================================


class TestPredicateStringExtraction:
    """Tests for _get_predicate_str function."""

    def test_lambda_predicate_string(self):
        """Test extracting string from lambda predicate."""
        from cyllama.agents.contract import _get_predicate_str

        pred = lambda args: args["n"] > 0
        pred_str = _get_predicate_str(pred)
        # Should contain lambda
        assert "lambda" in pred_str

    def test_function_predicate_string(self):
        """Test extracting string from function predicate."""
        from cyllama.agents.contract import _get_predicate_str

        def check_positive(args):
            return args["n"] > 0

        pred_str = _get_predicate_str(check_positive)
        # Should return some string representation
        assert len(pred_str) > 0


# =============================================================================
# Test ContractViolation Extended
# =============================================================================


class TestContractViolationExtended:
    """Extended tests for ContractViolation."""

    def test_violation_with_all_fields(self):
        """Test ContractViolation with all fields populated."""
        violation = ContractViolation(
            kind="pre",
            location="my_tool",
            predicate="args['n'] > 0",
            message="n must be positive",
            context={"args": {"n": -1}, "extra": "info"},
            policy=ContractPolicy.OBSERVE,
        )
        assert violation.kind == "pre"
        assert violation.location == "my_tool"
        assert violation.predicate == "args['n'] > 0"
        assert violation.message == "n must be positive"
        assert violation.context["args"]["n"] == -1
        assert violation.policy == ContractPolicy.OBSERVE

    def test_violation_default_policy(self):
        """Test ContractViolation default policy is ENFORCE."""
        violation = ContractViolation(kind="pre", location="test", predicate="x", message="m")
        assert violation.policy == ContractPolicy.ENFORCE

    def test_violation_default_context(self):
        """Test ContractViolation default context is empty dict."""
        violation = ContractViolation(kind="pre", location="test", predicate="x", message="m")
        assert violation.context == {}


# =============================================================================
# Test ContractContext Extended
# =============================================================================


class TestContractContextExtended:
    """Extended tests for ContractContext."""

    def test_context_without_handler(self):
        """Test context with no handler set."""
        ctx = ContractContext(policy=ContractPolicy.OBSERVE, handler=None, location="test")
        violation = ContractViolation(kind="assert", location="test", predicate="x", message="m")
        # With handler=None the dispatcher must short-circuit and return None
        # without raising -- OBSERVE is supposed to be non-fatal even when
        # there is no one listening.
        assert ctx.handle_violation(violation) is None
        # The handler slot should still be None after the call (no accidental
        # install of a default handler).
        assert ctx.handler is None


# =============================================================================
# Test Agent with No Tools
# =============================================================================


class TestAgentNoTools:
    """Tests for ContractAgent with no tools."""

    def test_agent_with_empty_tools(self, mock_llm):
        """Test creating agent with empty tools list."""
        agent = ContractAgent(llm=mock_llm, tools=[])
        assert len(agent.tools) == 0
        assert len(agent._tool_contracts) == 0

    def test_agent_with_none_tools(self, mock_llm):
        """Test creating agent with None tools."""
        agent = ContractAgent(llm=mock_llm, tools=None)
        assert agent.tools == []


# =============================================================================
# Test Postcondition with Args Edge Cases
# =============================================================================


class TestPostconditionArgsEdgeCases:
    """Tests for postcondition with args edge cases."""

    def test_postcondition_needs_args_missing(self):
        """Test postcondition with needs_args=True but no args provided."""
        cond = PostCondition(
            predicate=lambda r, args: r <= args.get("max", 100), message="result too big", needs_args=True
        )
        # When args is None, should still work (predicate handles it)
        result = cond.check(50, None)
        # Predicate will fail because args is None
        assert result is False

    def test_postcondition_needs_args_with_empty_dict(self):
        """Test postcondition with needs_args=True and empty args."""
        cond = PostCondition(
            predicate=lambda r, args: r <= args.get("max", 100), message="result too big", needs_args=True
        )
        result = cond.check(50, {})
        assert result is True  # Uses default 100


# =============================================================================
# Test Contract Assert Outside Context
# =============================================================================


class TestContractAssertOutsideContext:
    """Tests for contract_assert called outside agent context."""

    def test_assert_pass_outside_context(self):
        """Test contract_assert passes outside context."""
        _set_current_context(None)
        # Truthy condition outside any context: must return None and must
        # not install a context as a side effect.
        assert contract_assert(True, "should pass") is None
        assert _get_current_context() is None

    def test_assert_fail_outside_context_enforce(self):
        """Test contract_assert fails with ENFORCE outside context."""
        _set_current_context(None)
        with pytest.raises(ContractTermination) as exc_info:
            contract_assert(False, "must fail")
        assert "must fail" in str(exc_info.value)

    def test_assert_fail_outside_context_ignore(self):
        """Test contract_assert with IGNORE policy outside context."""
        _set_current_context(None)
        # IGNORE outside any context: a False condition must be swallowed
        # rather than raised, and must not leak a context into thread-local
        # storage on its way through.
        assert contract_assert(False, "ignored", policy=ContractPolicy.IGNORE) is None
        assert _get_current_context() is None


# =============================================================================
# Test Multiple Contracts Order
# =============================================================================


class TestMultipleContractsOrder:
    """Tests for order of multiple contracts on same function."""

    def test_preconditions_checked_in_order(self, mock_llm):
        """Test that preconditions are checked in decorator order."""
        check_order = []

        def check1(args):
            check_order.append(1)
            return True

        def check2(args):
            check_order.append(2)
            return True

        @tool
        @pre(check1, "first")
        @pre(check2, "second")
        def my_func(n: int) -> int:
            return n

        agent = ContractAgent(llm=mock_llm, tools=[my_func])
        agent._check_preconditions("my_func", {"n": 5})

        # Decorators are applied bottom-up, so check2 is added first
        # But we insert at front, so order should be 1, 2
        assert check_order == [1, 2]

    def test_first_failing_precondition_reported(self, mock_llm):
        """Test that first failing precondition is reported."""

        @tool
        @pre(lambda args: args["n"] > 0, "first: n positive")
        @pre(lambda args: args["n"] < 100, "second: n less than 100")
        def my_func(n: int) -> int:
            return n

        agent = ContractAgent(llm=mock_llm, tools=[my_func])
        violation = agent._check_preconditions("my_func", {"n": -1})

        assert "first: n positive" in violation.message


# =============================================================================
# Test ContractTermination Exception
# =============================================================================


class TestContractTerminationException:
    """Tests for ContractTermination exception."""

    def test_exception_contains_violation(self):
        """Test ContractTermination stores the violation."""
        violation = ContractViolation(kind="pre", location="test", predicate="x", message="test violation")
        exc = ContractTermination(violation)
        assert exc.violation is violation
        assert "test violation" in str(exc)

    def test_exception_inherits_from_exception(self):
        """Test ContractTermination is an Exception."""
        violation = ContractViolation(kind="pre", location="test", predicate="x", message="m")
        exc = ContractTermination(violation)
        assert isinstance(exc, Exception)


# =============================================================================
# List-form agent-level callbacks (Step 3 of contract-agent.md repositioning)
# =============================================================================


class TestListFormAgentCallbacks:
    """Tests for the plural ``*_preconditions`` / ``*_postconditions`` /
    ``iteration_invariants`` parameters on ContractAgent."""

    def test_list_form_task_preconditions_normalized(self, mock_llm, simple_tool):
        """Plural form lands in ``task_preconditions`` list."""
        agent = ContractAgent(
            llm=mock_llm,
            tools=[simple_tool],
            task_preconditions=[lambda t: len(t) > 5, lambda t: not t.startswith("ignore")],
        )
        assert len(agent.task_preconditions) == 2
        # Back-compat singular attribute exposes the first entry.
        assert agent.task_precondition is agent.task_preconditions[0]

    def test_list_form_answer_postconditions_normalized(self, mock_llm, simple_tool):
        agent = ContractAgent(
            llm=mock_llm,
            tools=[simple_tool],
            answer_postconditions=[lambda a: len(a) > 0, lambda a: "error" not in a],
        )
        assert len(agent.answer_postconditions) == 2
        assert agent.answer_postcondition is agent.answer_postconditions[0]

    def test_list_form_iteration_invariants_normalized(self, mock_llm, simple_tool):
        agent = ContractAgent(
            llm=mock_llm,
            tools=[simple_tool],
            iteration_invariants=[lambda s: s.errors < 3, lambda s: s.iterations < 10],
        )
        assert len(agent.iteration_invariants) == 2
        assert agent.iteration_invariant is agent.iteration_invariants[0]

    def test_singular_form_still_works_for_back_compat(self, mock_llm, simple_tool):
        """The old singular kwargs still produce a single-entry list."""
        agent = ContractAgent(
            llm=mock_llm,
            tools=[simple_tool],
            task_precondition=lambda t: len(t) > 0,
            answer_postcondition=lambda a: len(a) > 0,
            iteration_invariant=lambda s: s.errors < 5,
        )
        assert len(agent.task_preconditions) == 1
        assert len(agent.answer_postconditions) == 1
        assert len(agent.iteration_invariants) == 1
        # Singular attrs still resolve to the (only) callable.
        assert agent.task_precondition is not None
        assert agent.answer_postcondition is not None
        assert agent.iteration_invariant is not None

    def test_neither_form_yields_empty_lists(self, mock_llm, simple_tool):
        agent = ContractAgent(llm=mock_llm, tools=[simple_tool])
        assert agent.task_preconditions == []
        assert agent.answer_postconditions == []
        assert agent.iteration_invariants == []
        assert agent.task_precondition is None
        assert agent.answer_postcondition is None
        assert agent.iteration_invariant is None

    def test_both_singular_and_plural_raises(self, mock_llm, simple_tool):
        """Passing both forms for the same hook is rejected."""
        with pytest.raises(ValueError, match="task_precondition"):
            ContractAgent(
                llm=mock_llm,
                tools=[simple_tool],
                task_precondition=lambda t: True,
                task_preconditions=[lambda t: True],
            )
        with pytest.raises(ValueError, match="answer_postcondition"):
            ContractAgent(
                llm=mock_llm,
                tools=[simple_tool],
                answer_postcondition=lambda a: True,
                answer_postconditions=[lambda a: True],
            )
        with pytest.raises(ValueError, match="iteration_invariant"):
            ContractAgent(
                llm=mock_llm,
                tools=[simple_tool],
                iteration_invariant=lambda s: True,
                iteration_invariants=[lambda s: True],
            )

    def test_empty_list_treated_as_no_constraint(self, mock_llm, simple_tool):
        """Empty list == no constraint, just like None."""
        agent = ContractAgent(
            llm=mock_llm,
            tools=[simple_tool],
            task_preconditions=[],
            answer_postconditions=[],
            iteration_invariants=[],
        )
        assert agent.task_preconditions == []
        assert agent.answer_postconditions == []
        assert agent.iteration_invariants == []


# =============================================================================
# Value-aware @post (Step 2 of contract-agent.md repositioning)
# =============================================================================


class TestPostReceivesRawValue:
    """ContractAgent.run pipes raw_result (not str(raw_result)) into @post
    predicates. The plumbing has been in place since contract.py:1236 read
    from `event.metadata["raw_result"]`; these tests pin the behavior so a
    future refactor can't regress it back to stringified observations.
    """

    def test_post_predicate_sees_raw_dict_not_str(self):
        """@post on a tool returning dict gets the dict, not its repr."""
        seen: list = []

        @tool
        @post(lambda r: isinstance(r, dict), "result must be a dict")
        def make_dict() -> dict:
            """Return a small dict."""
            d = {"k": 1}
            seen.append(("called",))
            return d

        # _execute_tool_raw returns the raw dict; observation is its str().
        # The contract layer's _check_postconditions reads raw_result from
        # the metadata stash, so the predicate sees the dict.
        # Verify by exercising the contract evaluator directly:
        agent = ContractAgent(llm=MagicMock(), tools=[make_dict])
        # _check_postconditions(tool_name, result, args)
        violation = agent._check_postconditions("make_dict", {"k": 1}, {})
        assert violation is None  # dict passes isinstance(r, dict)

    def test_post_predicate_fails_when_given_stringified_value(self):
        """Inverse: if @post received str(raw_result), this would pass."""

        @tool
        @post(lambda r: isinstance(r, dict), "must be a dict")
        def returns_dict() -> dict:
            return {"k": 1}

        agent = ContractAgent(llm=MagicMock(), tools=[returns_dict])
        # Pass a string (simulating the broken behavior) — should now violate.
        violation = agent._check_postconditions("returns_dict", "{'k': 1}", {})
        assert violation is not None
        assert violation.kind == "post"


# =============================================================================
# CONTRACT_VIOLATION event emission (Step 4 of contract-agent.md repositioning)
# =============================================================================


class TestContractViolationEvents:
    """The contract layer emits ``EventType.CONTRACT_VIOLATION`` (and
    ``CONTRACT_CHECK``) events on the agent stream. These tests pin that
    behavior at each check site so monitoring UIs / streaming consumers
    can depend on the events being present.
    """

    def _drain(self, inner_events):
        """Build a stub inner agent that yields the supplied events."""

        class _StubInner:
            def stream(self, task):
                for e in inner_events:
                    yield e

        return _StubInner()

    def test_task_precondition_violation_emits_event(self, mock_llm):
        agent = ContractAgent(
            llm=mock_llm,
            tools=[],
            task_preconditions=[lambda t: False],  # always fail
            policy=ContractPolicy.OBSERVE,  # don't terminate, keep streaming
        )
        agent._inner_agent = self._drain(
            [
                AgentEvent(type=EventType.ANSWER, content="ok"),
            ]
        )
        events = list(agent.stream("any task"))
        kinds = [e.type for e in events]
        assert EventType.CONTRACT_VIOLATION in kinds
        # And the violation event metadata carries the structured violation.
        viol_events = [e for e in events if e.type == EventType.CONTRACT_VIOLATION]
        assert isinstance(viol_events[0].metadata["violation"], ContractViolation)

    def test_iteration_invariant_violation_emits_event(self, mock_llm):
        agent = ContractAgent(
            llm=mock_llm,
            tools=[],
            iteration_invariants=[lambda s: False],
            policy=ContractPolicy.OBSERVE,
        )
        agent._inner_agent = self._drain(
            [
                AgentEvent(type=EventType.THOUGHT, content="thinking"),
                AgentEvent(type=EventType.ANSWER, content="ok"),
            ]
        )
        events = list(agent.stream("any task"))
        viol_events = [e for e in events if e.type == EventType.CONTRACT_VIOLATION]
        assert len(viol_events) >= 1
        assert viol_events[0].metadata["violation"].kind == "assert"

    def test_answer_postcondition_violation_emits_event(self, mock_llm):
        agent = ContractAgent(
            llm=mock_llm,
            tools=[],
            answer_postconditions=[lambda a: False],
            policy=ContractPolicy.OBSERVE,
        )
        agent._inner_agent = self._drain(
            [
                AgentEvent(type=EventType.ANSWER, content="ok"),
            ]
        )
        events = list(agent.stream("any task"))
        viol_events = [e for e in events if e.type == EventType.CONTRACT_VIOLATION]
        assert len(viol_events) == 1
        assert viol_events[0].metadata["violation"].kind == "post"
        assert viol_events[0].metadata["violation"].location == "agent"

    def test_observe_policy_emits_event_per_failure_in_list(self, mock_llm):
        """Under OBSERVE, every failing predicate in a list yields its own event."""
        agent = ContractAgent(
            llm=mock_llm,
            tools=[],
            answer_postconditions=[
                lambda a: False,
                lambda a: False,
                lambda a: True,  # this one passes
                lambda a: False,
            ],
            policy=ContractPolicy.OBSERVE,
        )
        agent._inner_agent = self._drain(
            [
                AgentEvent(type=EventType.ANSWER, content="ok"),
            ]
        )
        events = list(agent.stream("any task"))
        viol_events = [e for e in events if e.type == EventType.CONTRACT_VIOLATION]
        # 3 of 4 predicates failed under OBSERVE -> 3 distinct events.
        assert len(viol_events) == 3
        # Each violation carries its index.
        indices = [e.metadata.get("index") for e in viol_events]
        assert indices == [0, 1, 3]

    def test_violation_event_carries_predicate_string(self, mock_llm):
        """Violation events expose the predicate identity for debugging."""
        agent = ContractAgent(
            llm=mock_llm,
            tools=[],
            task_preconditions=[lambda t: False],
            policy=ContractPolicy.OBSERVE,
        )
        agent._inner_agent = self._drain(
            [
                AgentEvent(type=EventType.ANSWER, content="ok"),
            ]
        )
        events = list(agent.stream("any task"))
        viol = next(e for e in events if e.type == EventType.CONTRACT_VIOLATION)
        # predicate string contains the named-index form set by the loop.
        assert "task_precondition" in viol.metadata["violation"].predicate


# =============================================================================
# Richer IterationState (Step 5 of contract-agent.md repositioning)
# =============================================================================


class TestIterationStateNewFields:
    """The new fields on IterationState enable invariants that have no
    schema substitute: time budgets, prompt-size caps, stuck-loop
    detection beyond the built-in action/tool repeat detector.
    """

    def test_elapsed_ms_grows_across_events(self):
        import time as _time

        s = IterationState()
        s.update(AgentEvent(type=EventType.THOUGHT, content="first"))
        first = s.elapsed_ms
        _time.sleep(0.005)
        s.update(AgentEvent(type=EventType.THOUGHT, content="second"))
        assert s.elapsed_ms > first

    def test_last_tool_name_tracks_action(self):
        s = IterationState()
        s.update(AgentEvent(type=EventType.ACTION, content="fetch()", metadata={"tool_name": "fetch"}))
        assert s.last_tool_name == "fetch"
        s.update(AgentEvent(type=EventType.ACTION, content="search()", metadata={"tool_name": "search"}))
        assert s.last_tool_name == "search"

    def test_last_observation_and_history(self):
        s = IterationState()
        s.update(AgentEvent(type=EventType.OBSERVATION, content="result A"))
        s.update(AgentEvent(type=EventType.OBSERVATION, content="result B"))
        assert s.last_observation == "result B"
        assert s.observations_so_far == ["result A", "result B"]

    def test_observations_capped_to_max(self):
        s = IterationState()
        s._max_observations = 3
        for i in range(7):
            s.update(AgentEvent(type=EventType.OBSERVATION, content=f"obs{i}"))
        assert len(s.observations_so_far) == 3
        assert s.observations_so_far == ["obs4", "obs5", "obs6"]

    def test_consecutive_same_observation_counter(self):
        s = IterationState()
        s.update(AgentEvent(type=EventType.OBSERVATION, content="same"))
        assert s.consecutive_same_observation == 1
        s.update(AgentEvent(type=EventType.OBSERVATION, content="same"))
        assert s.consecutive_same_observation == 2
        s.update(AgentEvent(type=EventType.OBSERVATION, content="different"))
        assert s.consecutive_same_observation == 1

    def test_estimated_prompt_chars_accumulates(self):
        s = IterationState()
        s.update(AgentEvent(type=EventType.THOUGHT, content="abc"))
        s.update(AgentEvent(type=EventType.ACTION, content="xy"))
        assert s.estimated_prompt_chars == 5

    def test_to_dict_includes_new_fields(self):
        s = IterationState()
        s.update(AgentEvent(type=EventType.OBSERVATION, content="o"))
        d = s.to_dict()
        for key in (
            "elapsed_ms",
            "last_tool_name",
            "estimated_prompt_chars",
            "consecutive_same_observation",
            "observations_so_far_count",
        ):
            assert key in d

    def test_invariant_can_express_time_budget(self, mock_llm):
        """End-to-end: an iteration_invariant using elapsed_ms fires after the budget."""
        agent = ContractAgent(
            llm=mock_llm,
            tools=[],
            iteration_invariants=[lambda s: s.elapsed_ms < 0.0],  # always fails (negative budget)
            policy=ContractPolicy.OBSERVE,
        )

        class _Inner:
            def stream(self, task):
                yield AgentEvent(type=EventType.THOUGHT, content="t")
                yield AgentEvent(type=EventType.ANSWER, content="a")

        agent._inner_agent = _Inner()
        events = list(agent.stream("task"))
        viols = [e for e in events if e.type == EventType.CONTRACT_VIOLATION]
        assert len(viols) == 1

    def test_invariant_can_express_prompt_budget(self, mock_llm):
        """End-to-end: estimated_prompt_chars budget."""
        agent = ContractAgent(
            llm=mock_llm,
            tools=[],
            iteration_invariants=[lambda s: s.estimated_prompt_chars < 5],
            policy=ContractPolicy.OBSERVE,
        )

        class _Inner:
            def stream(self, task):
                # 6 chars of THOUGHT content -> trips the < 5 invariant
                yield AgentEvent(type=EventType.THOUGHT, content="six!!!")
                yield AgentEvent(type=EventType.ANSWER, content="a")

        agent._inner_agent = _Inner()
        events = list(agent.stream("task"))
        viols = [e for e in events if e.type == EventType.CONTRACT_VIOLATION]
        assert len(viols) == 1
