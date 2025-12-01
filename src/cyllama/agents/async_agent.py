"""
Async wrapper for cyllama agents.

This module provides async/await support for ReActAgent and ConstrainedAgent.
It wraps the synchronous agent execution using asyncio.to_thread() to avoid
blocking the event loop during inference.

Example:
    >>> import asyncio
    >>> from cyllama.agents import AsyncReActAgent, tool
    >>>
    >>> @tool
    >>> def search(query: str) -> str:
    >>>     '''Search for information.'''
    >>>     return f"Results for: {query}"
    >>>
    >>> async def main():
    >>>     async with AsyncReActAgent("model.gguf", tools=[search]) as agent:
    >>>         result = await agent.run("What is Python?")
    >>>         print(result.answer)
    >>>
    >>>         # Async streaming
    >>>         async for event in agent.stream("Tell me about AI"):
    >>>             print(f"{event.type}: {event.content}")
    >>>
    >>> asyncio.run(main())
"""

import asyncio
from typing import (
    AsyncIterator,
    Optional,
    List,
    Union,
    Callable,
)
import logging

from .react import ReActAgent, AgentEvent, AgentResult, AgentMetrics
from .constrained import ConstrainedAgent, ConstrainedGenerationConfig
from .tools import Tool
from ..api import LLM, GenerationConfig

logger = logging.getLogger(__name__)


class AsyncReActAgent:
    """
    Async wrapper for ReActAgent.

    Provides an async interface to the ReActAgent, running the synchronous
    agent execution in a thread pool to avoid blocking the event loop.

    Example:
        >>> async def main():
        >>>     async with AsyncReActAgent("model.gguf", tools=[search]) as agent:
        >>>         result = await agent.run("Find information about Python")
        >>>         print(result.answer)
        >>>
        >>>         # Stream events
        >>>         async for event in agent.stream("Explain machine learning"):
        >>>             print(f"{event.type.value}: {event.content}")
        >>>
        >>> asyncio.run(main())
    """

    def __init__(
        self,
        model_path: str,
        tools: Optional[List[Union[Tool, Callable]]] = None,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = False,
        detect_loops: bool = True,
        max_consecutive_same_action: int = 3,
        max_consecutive_same_tool: int = 5,
        **kwargs
    ):
        """
        Initialize async ReAct agent.

        Args:
            model_path: Path to GGUF model file
            tools: List of Tool instances or @tool decorated functions
            config: Generation configuration
            system_prompt: Custom system prompt (optional)
            max_iterations: Maximum reasoning iterations
            verbose: Print detailed output
            detect_loops: Enable loop detection
            max_consecutive_same_action: Trigger loop detection after N identical actions
            max_consecutive_same_tool: Trigger loop detection after N calls to same tool
            **kwargs: Additional generation config parameters
        """
        # Create LLM instance from model_path
        self._llm = LLM(model_path, config=config, verbose=verbose, **kwargs)
        self._agent = ReActAgent(
            llm=self._llm,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
            generation_config=config,
            detect_loops=detect_loops,
            max_consecutive_same_action=max_consecutive_same_action,
            max_consecutive_same_tool=max_consecutive_same_tool,
        )
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False

    async def close(self):
        """Release resources."""
        if self._llm:
            await asyncio.to_thread(self._llm.close)

    @property
    def metrics(self) -> Optional[AgentMetrics]:
        """Get metrics from the last run."""
        return self._agent.metrics

    async def run(self, task: str) -> AgentResult:
        """
        Run the agent on a task asynchronously.

        Args:
            task: Task description or question

        Returns:
            AgentResult with answer and execution trace

        Example:
            >>> result = await agent.run("What is the capital of France?")
            >>> print(result.answer)
        """
        async with self._lock:
            return await asyncio.to_thread(self._agent.run, task)

    async def stream(self, task: str) -> AsyncIterator[AgentEvent]:
        """
        Stream agent execution events asynchronously.

        Args:
            task: Task description or question

        Yields:
            AgentEvent instances as agent executes

        Example:
            >>> async for event in agent.stream("Explain quantum computing"):
            >>>     if event.type == EventType.THOUGHT:
            >>>         print(f"Thinking: {event.content}")
            >>>     elif event.type == EventType.ACTION:
            >>>         print(f"Acting: {event.content}")
        """
        queue: asyncio.Queue[Union[AgentEvent, None, Exception]] = asyncio.Queue()

        async def producer():
            """Run sync generator in thread and put items in queue."""
            try:
                def generate_sync():
                    for event in self._agent.stream(task):
                        asyncio.run_coroutine_threadsafe(
                            queue.put(event),
                            loop
                        )
                    asyncio.run_coroutine_threadsafe(
                        queue.put(None),
                        loop
                    )

                await asyncio.to_thread(generate_sync)
            except Exception as e:
                await queue.put(e)

        loop = asyncio.get_event_loop()

        async with self._lock:
            producer_task = asyncio.create_task(producer())

            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                await producer_task


class AsyncConstrainedAgent:
    """
    Async wrapper for ConstrainedAgent.

    Provides an async interface to the ConstrainedAgent, which uses
    grammar-constrained generation for 100% reliable tool calling.

    Example:
        >>> async def main():
        >>>     async with AsyncConstrainedAgent("model.gguf", tools=[calc]) as agent:
        >>>         result = await agent.run("What is 2 + 2?")
        >>>         print(result.answer)
        >>>
        >>> asyncio.run(main())
    """

    def __init__(
        self,
        model_path: str,
        tools: Optional[List[Union[Tool, Callable]]] = None,
        config: Optional[GenerationConfig] = None,
        generation_config: Optional[ConstrainedGenerationConfig] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize async constrained agent.

        Args:
            model_path: Path to GGUF model file
            tools: List of Tool instances or @tool decorated functions
            config: LLM generation configuration
            generation_config: Constrained generation configuration
            system_prompt: Custom system prompt (optional)
            max_iterations: Maximum reasoning iterations
            verbose: Print detailed output
            **kwargs: Additional generation config parameters
        """
        # Create LLM instance from model_path
        self._llm = LLM(model_path, config=config, verbose=verbose, **kwargs)
        self._agent = ConstrainedAgent(
            llm=self._llm,
            tools=tools,
            generation_config=generation_config,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False

    async def close(self):
        """Release resources."""
        if self._llm:
            await asyncio.to_thread(self._llm.close)

    @property
    def metrics(self) -> Optional[AgentMetrics]:
        """Get metrics from the last run."""
        return self._agent.metrics

    async def run(self, task: str) -> AgentResult:
        """
        Run the agent on a task asynchronously.

        Args:
            task: Task description or question

        Returns:
            AgentResult with answer and execution trace
        """
        async with self._lock:
            return await asyncio.to_thread(self._agent.run, task)

    async def stream(self, task: str) -> AsyncIterator[AgentEvent]:
        """
        Stream agent execution events asynchronously.

        Args:
            task: Task description or question

        Yields:
            AgentEvent instances as agent executes
        """
        queue: asyncio.Queue[Union[AgentEvent, None, Exception]] = asyncio.Queue()

        async def producer():
            try:
                def generate_sync():
                    for event in self._agent.stream(task):
                        asyncio.run_coroutine_threadsafe(
                            queue.put(event),
                            loop
                        )
                    asyncio.run_coroutine_threadsafe(
                        queue.put(None),
                        loop
                    )

                await asyncio.to_thread(generate_sync)
            except Exception as e:
                await queue.put(e)

        loop = asyncio.get_event_loop()

        async with self._lock:
            producer_task = asyncio.create_task(producer())

            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                await producer_task


async def run_agent_async(
    agent: Union[ReActAgent, ConstrainedAgent],
    task: str
) -> AgentResult:
    """
    Run any synchronous agent asynchronously.

    Convenience function for running an existing agent instance
    without blocking the event loop.

    Args:
        agent: ReActAgent or ConstrainedAgent instance
        task: Task description or question

    Returns:
        AgentResult with answer and execution trace

    Example:
        >>> agent = ReActAgent("model.gguf", tools=[search])
        >>> result = await run_agent_async(agent, "Find Python info")
    """
    return await asyncio.to_thread(agent.run, task)
