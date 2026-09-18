"""
Tests for async agent wrappers.

Tests the async/await support for ReActAgent and ConstrainedAgent.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from cyllama.agents.async_agent import (
    AsyncReActAgent,
    AsyncConstrainedAgent,
    run_agent_async,
)
from cyllama.agents import (
    ReActAgent,
    ConstrainedAgent,
    AgentResult,
    AgentEvent,
    EventType,
    AgentMetrics,
    tool,
)


# Sample tools for testing
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


class TestAsyncReActAgent:
    """Tests for AsyncReActAgent."""

    def test_init(self):
        """Test AsyncReActAgent initialization."""
        with patch("cyllama.agents.async_agent.LLM"):
            with patch.object(ReActAgent, "__init__", return_value=None):
                agent = AsyncReActAgent("model.gguf", tools=[calculator])
                assert agent._agent is not None
                assert agent._lock is not None
                assert agent._llm is not None

    def test_init_with_kwargs(self):
        """Test initialization with kwargs."""
        with patch("cyllama.agents.async_agent.LLM"):
            with patch.object(ReActAgent, "__init__", return_value=None) as mock_init:
                agent = AsyncReActAgent("model.gguf", tools=[calculator], max_iterations=5, verbose=True)
                mock_init.assert_called_once()
                call_kwargs = mock_init.call_args[1]
                assert call_kwargs.get("max_iterations") == 5
                assert call_kwargs.get("verbose") is True

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        mock_llm = Mock()
        mock_llm.close = Mock()

        with patch("cyllama.agents.async_agent.LLM", return_value=mock_llm):
            with patch.object(ReActAgent, "__init__", return_value=None):
                async with AsyncReActAgent("model.gguf") as agent:
                    assert agent is not None

                mock_llm.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_run(self):
        """Test async run method."""
        mock_result = AgentResult(answer="The answer is 42", steps=[], iterations=1, success=True)

        with patch("cyllama.agents.async_agent.LLM"):
            with patch.object(ReActAgent, "__init__", return_value=None):
                with patch.object(ReActAgent, "run", return_value=mock_result):
                    agent = AsyncReActAgent("model.gguf")
                    result = await agent.run("What is the answer?")

                    assert result.answer == "The answer is 42"
                    assert result.success is True

    @pytest.mark.asyncio
    async def test_metrics_property(self):
        """Test metrics property access."""
        metrics = AgentMetrics(iterations=3, tool_calls=2)

        with patch("cyllama.agents.async_agent.LLM"):
            with patch.object(ReActAgent, "__init__", return_value=None):
                agent = AsyncReActAgent("model.gguf")
                agent._agent._metrics = metrics

                with patch.object(ReActAgent, "metrics", new_callable=lambda: property(lambda s: metrics)):
                    # Access through the wrapper
                    assert agent._agent._metrics.iterations == 3

    @pytest.mark.asyncio
    async def test_stream(self):
        """Test async streaming."""
        events = [
            AgentEvent(type=EventType.THOUGHT, content="Thinking..."),
            AgentEvent(type=EventType.ACTION, content="search[query]"),
            AgentEvent(type=EventType.OBSERVATION, content="Results"),
            AgentEvent(type=EventType.ANSWER, content="Final answer"),
        ]

        def mock_stream(task):
            for event in events:
                yield event

        with patch("cyllama.agents.async_agent.LLM"):
            with patch.object(ReActAgent, "__init__", return_value=None):
                agent = AsyncReActAgent("model.gguf")
                agent._agent.stream = mock_stream

                received_events = []
                async for event in agent.stream("Test task"):
                    received_events.append(event)

                assert len(received_events) == 4
                assert received_events[0].type == EventType.THOUGHT
                assert received_events[-1].type == EventType.ANSWER


class TestAsyncConstrainedAgent:
    """Tests for AsyncConstrainedAgent."""

    def test_init(self):
        """Test AsyncConstrainedAgent initialization."""
        with patch("cyllama.agents.async_agent.LLM"):
            with patch.object(ConstrainedAgent, "__init__", return_value=None):
                agent = AsyncConstrainedAgent("model.gguf", tools=[calculator])
                assert agent._agent is not None
                assert agent._lock is not None
                assert agent._llm is not None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        mock_llm = Mock()
        mock_llm.close = Mock()

        with patch("cyllama.agents.async_agent.LLM", return_value=mock_llm):
            with patch.object(ConstrainedAgent, "__init__", return_value=None):
                async with AsyncConstrainedAgent("model.gguf") as agent:
                    assert agent is not None

                mock_llm.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_run(self):
        """Test async run method."""
        mock_result = AgentResult(answer="4", steps=[], iterations=1, success=True)

        with patch("cyllama.agents.async_agent.LLM"):
            with patch.object(ConstrainedAgent, "__init__", return_value=None):
                with patch.object(ConstrainedAgent, "run", return_value=mock_result):
                    agent = AsyncConstrainedAgent("model.gguf")
                    result = await agent.run("What is 2+2?")

                    assert result.answer == "4"
                    assert result.success is True

    @pytest.mark.asyncio
    async def test_stream(self):
        """Test async streaming."""
        events = [
            AgentEvent(type=EventType.THOUGHT, content="Calculating..."),
            AgentEvent(type=EventType.ANSWER, content="4"),
        ]

        def mock_stream(task):
            for event in events:
                yield event

        with patch("cyllama.agents.async_agent.LLM"):
            with patch.object(ConstrainedAgent, "__init__", return_value=None):
                agent = AsyncConstrainedAgent("model.gguf")
                agent._agent.stream = mock_stream

                received_events = []
                async for event in agent.stream("Calculate 2+2"):
                    received_events.append(event)

                assert len(received_events) == 2


class TestRunAgentAsync:
    """Tests for run_agent_async helper function."""

    @pytest.mark.asyncio
    async def test_run_react_agent(self):
        """Test running ReActAgent asynchronously."""
        mock_result = AgentResult(answer="Result", steps=[], iterations=1, success=True)

        mock_agent = Mock(spec=ReActAgent)
        mock_agent.run.return_value = mock_result

        result = await run_agent_async(mock_agent, "Test task")

        assert result.answer == "Result"
        mock_agent.run.assert_called_once_with("Test task")

    @pytest.mark.asyncio
    async def test_run_constrained_agent(self):
        """Test running ConstrainedAgent asynchronously."""
        mock_result = AgentResult(answer="Constrained result", steps=[], iterations=1, success=True)

        mock_agent = Mock(spec=ConstrainedAgent)
        mock_agent.run.return_value = mock_result

        result = await run_agent_async(mock_agent, "Test task")

        assert result.answer == "Constrained result"


class TestConcurrency:
    """Tests for concurrent behavior."""

    @pytest.mark.asyncio
    async def test_agent_lock_serializes_calls(self):
        """Test that agent lock serializes concurrent calls."""
        call_order = []

        def slow_run_sync(task):
            """Sync function that will be called by to_thread."""
            import time

            call_order.append(f"start_{task}")
            time.sleep(0.05)  # Use sync sleep
            call_order.append(f"end_{task}")
            return AgentResult(answer=f"Result for {task}", steps=[], iterations=1, success=True)

        with patch("cyllama.agents.async_agent.LLM"):
            with patch.object(ReActAgent, "__init__", return_value=None):
                agent = AsyncReActAgent("model.gguf")
                # Mock the sync agent's run method
                agent._agent.run = slow_run_sync

                # Start concurrent calls
                task1 = asyncio.create_task(agent.run("task1"))
                task2 = asyncio.create_task(agent.run("task2"))

                await asyncio.gather(task1, task2)

                # Due to lock, calls should be serialized
                assert call_order == ["start_task1", "end_task1", "start_task2", "end_task2"]


# Integration tests
class TestAsyncAgentIntegration:
    """Integration tests with real model."""

    @pytest.fixture
    def model_path(self):
        """Get test model path."""
        import os

        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "Llama-3.2-1B-Instruct-Q8_0.gguf")
        if not os.path.exists(path):
            pytest.skip("Test model not available")
        return path

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_real_async_react_agent(self, model_path):
        """Test real async ReActAgent execution."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        async with AsyncReActAgent(model_path, tools=[add], max_iterations=3, max_tokens=100) as agent:
            result = await agent.run("What is 2 plus 3?")
            # Agent should complete (success or not)
            assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_real_async_stream(self, model_path):
        """Test real async streaming."""

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        async with AsyncReActAgent(model_path, tools=[greet], max_iterations=2, max_tokens=50) as agent:
            events = []
            async for event in agent.stream("Greet Alice"):
                events.append(event)

            # Should have at least one event
            assert len(events) > 0


class TestAsyncAgentCloseSerialization:
    """``close()`` frees the native context, so it must take the same lock
    ``run()`` and ``stream()`` hold. Closing underneath a running operation
    lets in-flight work touch freed native resources.
    """

    @pytest.mark.parametrize("agent_cls", [AsyncReActAgent, AsyncConstrainedAgent])
    async def test_close_waits_for_a_running_operation(self, agent_cls):
        import threading

        # run() executes on a worker thread, so these are threading events.
        started = threading.Event()
        release = threading.Event()
        closed_during_run = False

        def slow_run(task):
            started.set()
            release.wait(timeout=10)
            return AgentResult(answer="done", steps=[], iterations=1, success=True)

        with patch("cyllama.agents.async_agent.LLM") as mock_llm_cls:
            agent = agent_cls("model.gguf")
            agent._agent.run = slow_run

            def record_close():
                nonlocal closed_during_run
                closed_during_run = started.is_set() and not release.is_set()

            mock_llm_cls.return_value.close.side_effect = record_close

            run_task = asyncio.create_task(agent.run("task"))
            while not started.is_set():
                await asyncio.sleep(0.01)
            close_task = asyncio.create_task(agent.close())
            await asyncio.sleep(0.05)  # give close a chance to jump the queue
            release.set()
            await run_task
            await close_task

        assert closed_during_run is False, "close() ran while an operation held the lock"

    @pytest.mark.parametrize("agent_cls", [AsyncReActAgent, AsyncConstrainedAgent])
    async def test_close_is_idempotent(self, agent_cls):
        with patch("cyllama.agents.async_agent.LLM") as mock_llm_cls:
            agent = agent_cls("model.gguf")
            await agent.close()
            await agent.close()
            assert mock_llm_cls.return_value.close.call_count == 1

    @pytest.mark.parametrize("agent_cls", [AsyncReActAgent, AsyncConstrainedAgent])
    async def test_run_after_close_raises(self, agent_cls):
        with patch("cyllama.agents.async_agent.LLM"):
            agent = agent_cls("model.gguf")
            await agent.close()
            with pytest.raises(RuntimeError, match="closed"):
                await agent.run("task")

    @pytest.mark.parametrize("agent_cls", [AsyncReActAgent, AsyncConstrainedAgent])
    async def test_stream_after_close_raises(self, agent_cls):
        with patch("cyllama.agents.async_agent.LLM"):
            agent = agent_cls("model.gguf")
            await agent.close()
            with pytest.raises(RuntimeError, match="closed"):
                async for _ in agent.stream("task"):
                    pass
