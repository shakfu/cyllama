"""
Constrained agent implementation using grammar-enforced tool calling.

Uses GBNF grammars to guarantee valid tool call syntax, eliminating parsing
failures and enabling reliable agent execution even with smaller models.
"""

import json
import logging
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, cast
from dataclasses import dataclass

from ..api import LLM, GenerationConfig

# Module logger
logger = logging.getLogger(__name__)
from ..llama.llama_cpp import (
    LlamaSampler,
    LlamaSamplerChainParams,
    llama_batch_get_one,
)
from ._loop_detection import detect_loop, format_loop_error
from .react import render_observation
from .tools import Tool, ToolArgumentError, ToolRegistry, coerce_args
from .grammar import (
    GrammarFormat,
    generate_answer_or_tool_grammar,
    get_cached_answer_or_tool_grammar,
)
from .types import AgentEvent, AgentProtocol, AgentResult, EventType


class GrammarConstrainedLLM(LLM):
    """
    LLM subclass that supports GBNF grammar constraints for generation.

    This class extends the base LLM to add grammar-constrained generation,
    ensuring outputs conform to a specified GBNF grammar format.
    """

    def __init__(self, model_path: str, config: Optional[GenerationConfig] = None, verbose: bool = False):
        """Initialize grammar-constrained LLM."""
        super().__init__(model_path, config, verbose)

    def generate_with_grammar(
        self,
        prompt: str,
        grammar: str,
        grammar_root: str = "root",
        config: Optional[GenerationConfig] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Generate text with grammar constraint.

        Args:
            prompt: Input text prompt
            grammar: GBNF grammar string
            grammar_root: Root rule name in grammar (default: "root")
            config: Generation configuration (uses instance config if None)
            on_token: Optional callback called for each generated token

        Returns:
            Generated text constrained by the grammar
        """
        config = config or self.config

        # Tokenize prompt
        prompt_tokens = self.vocab.tokenize(prompt, add_special=config.add_bos, parse_special=config.parse_special)
        n_prompt = len(prompt_tokens)

        if self.verbose:
            print(f"Prompt tokens: {n_prompt}")
            print(f"Applying grammar constraint: {len(grammar)} chars")

        # Ensure context is ready
        ctx = self._ensure_context(n_prompt, config)

        # Create sampler with grammar constraint
        self._ensure_sampler_with_grammar(config, grammar, grammar_root)
        sampler = self._sampler
        assert sampler is not None

        # Process prompt
        batch = llama_batch_get_one(prompt_tokens)
        ctx.decode(batch)

        # Generate tokens
        n_pos = n_prompt
        n_generated = 0
        output_tokens = []

        for _ in range(config.max_tokens):
            # Sample next token with grammar constraint
            new_token_id = sampler.sample(ctx, -1)

            # Check for end of generation
            if self.vocab.is_eog(new_token_id):
                break

            # Note: Do NOT call sampler.accept() manually - the sampler chain
            # handles acceptance internally when using grammar constraints

            # Decode token to text
            try:
                piece = self.vocab.token_to_piece(new_token_id, special=True)
            except UnicodeDecodeError:
                logger.warning("Failed to decode token %d: UnicodeDecodeError", new_token_id)
                piece = ""

            output_tokens.append(piece)

            # Call token callback if provided
            if on_token:
                on_token(piece)

            # Prepare next iteration
            batch = llama_batch_get_one([new_token_id], n_pos)
            ctx.decode(batch)
            n_pos += 1
            n_generated += 1

        return "".join(output_tokens)

    def _ensure_sampler_with_grammar(self, config: GenerationConfig, grammar: str, grammar_root: str) -> None:
        """Create sampler with grammar constraint."""
        sampler_params = LlamaSamplerChainParams()
        sampler_params.no_perf = not self.verbose

        self._sampler = LlamaSampler(sampler_params)

        # IMPORTANT: Grammar constraint must come FIRST in the sampler chain.
        # The grammar sampler filters logits to only allow tokens that match
        # the grammar, then subsequent samplers select from valid tokens.
        self._sampler.add_grammar(self.vocab, grammar, grammar_root)

        # Add penalties if configured
        if config.repeat_penalty != 1.0:
            self._sampler.add_penalties(
                penalty_last_n=64, penalty_repeat=config.repeat_penalty, penalty_freq=0.0, penalty_present=0.0
            )

        # Add sampling methods based on config
        if config.temperature == 0.0:
            # Greedy sampling
            self._sampler.add_greedy()
        else:
            # Probabilistic sampling
            self._sampler.add_min_p(config.min_p, 1)
            self._sampler.add_top_k(config.top_k)
            self._sampler.add_top_p(config.top_p, 1)
            self._sampler.add_temp(config.temperature)

            # Distribution sampler for final selection
            if config.seed != -1:
                self._sampler.add_dist(config.seed)
            else:
                self._sampler.add_dist(int(time.time()))


@dataclass
class ConstrainedGenerationConfig:
    """Configuration for constrained generation."""

    temperature: float = 0.7
    max_tokens: int = 512
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05


# AgentMetrics is shared with ReActAgent so callers (notably the
# AsyncConstrainedAgent wrapper and the AgentProtocol) can treat the
# return type of ``metrics`` uniformly across both agents. The two
# classes were byte-identical until this consolidation; the canonical
# definition now lives in ``agents/types.py``.
from .types import AgentMetrics  # noqa: E402,F401  (re-exported for backwards compat)


class ConstrainedAgent(AgentProtocol):
    """
    Agent that uses grammar-constrained generation for reliable tool calling.

    Unlike ReAct which parses freeform text, this agent uses GBNF grammars
    to enforce valid JSON structure, guaranteeing 100% parseable tool calls.

    Example:
        from cyllama import LLM
        from cyllama.agents import ConstrainedAgent, tool

        @tool
        def search(query: str) -> str:
            return "Results for: " + query

        llm = LLM("models/mistral-7b-instruct.gguf")
        agent = ConstrainedAgent(
            llm=llm,
            tools=[search],
            format="json"
        )

        result = agent.run("Search for information about Python")
        print(result.answer)
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that uses tools to solve tasks.

You have access to the following tools:

{tools}

IMPORTANT RULES:
1. You MUST use the EXACT parameter names shown in the tool descriptions
2. Do NOT invent new parameter names
3. When giving the final answer, include the specific information from tool results

Respond with JSON in one of these formats:

Tool call: {{"type": "tool_call", "tool_name": "name", "tool_args": {{...}}}}
Final answer: {{"type": "answer", "content": "..."}}

Use tools when needed, then provide a helpful final answer based on the results."""

    def __init__(
        self,
        llm: LLM,
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = False,
        generation_config: Optional[ConstrainedGenerationConfig] = None,
        format: str = "json",
        allow_reasoning: bool = False,
        use_cache: bool = True,
        detect_loops: bool = True,
        max_consecutive_same_action: int = 2,
        max_consecutive_same_tool: int = 4,
        max_context_chars: int = 6000,
    ):
        """
        Initialize constrained agent.

        Args:
            llm: LLM instance for generation
            tools: List of tools available to the agent
            system_prompt: Custom system prompt (uses default if None)
            max_iterations: Maximum number of tool call cycles
            verbose: Print agent actions to stdout
            generation_config: Custom generation config
            format: Output format ("json", "json_array", "function_call")
            allow_reasoning: Include reasoning field in output
            use_cache: Cache compiled grammars for performance
            detect_loops: Enable loop detection to prevent infinite loops (default: True)
            max_consecutive_same_action: Number of times the exact same action can repeat
                                         before considered a loop (default: 2)
            max_consecutive_same_tool: Number of times the same tool can be called
                                       consecutively (with any args) before loop (default: 4)
            max_context_chars: Maximum characters for the prompt context. Older history
                              is truncated to stay within this limit (default: 16000)
        """
        # Make grammar-constrained generation available on `self.llm`.
        #
        # Previously this path used `GrammarConstrainedLLM.__new__(...)` and
        # hand-copied a fixed list of attributes from the source LLM. That
        # was fragile: `LLM.__init__` sets ~12 internal attributes (`_closed`,
        # `_busy_lock`, `_cancel_event`, `_cache`, `_mcp_client`, `_ctx_size`,
        # ...) which were *not* in the copy list — so the wrapped object
        # crashed on the first call to `_ensure_context`. The grammar test
        # caught this when run against a real LLM.
        #
        # We instead bind the two grammar-only methods onto the LLM object
        # directly. They only touch standard LLM attrs (`vocab`, `config`,
        # `_ensure_context`) plus their own helper, so nothing about the
        # underlying LLM lifecycle changes.
        self.llm: Any
        if isinstance(llm, GrammarConstrainedLLM):
            self.llm = llm
        elif hasattr(llm, "model") and hasattr(llm, "vocab"):
            import types as _types

            llm.generate_with_grammar = _types.MethodType(  # type: ignore[attr-defined]
                GrammarConstrainedLLM.generate_with_grammar, llm
            )
            llm._ensure_sampler_with_grammar = _types.MethodType(  # type: ignore[attr-defined]
                GrammarConstrainedLLM._ensure_sampler_with_grammar, llm
            )
            self.llm = llm
        else:
            # Mocks / test LLMs use whatever `generate_with_grammar` they
            # already define (or none — MockLLM only stubs `__call__`).
            self.llm = llm

        self.registry = ToolRegistry()
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.generation_config = generation_config or ConstrainedGenerationConfig()
        self.use_cache = use_cache
        self.detect_loops = detect_loops
        self.max_consecutive_same_action = max_consecutive_same_action
        self.max_consecutive_same_tool = max_consecutive_same_tool
        self.max_context_chars = max_context_chars

        # Parse format
        try:
            self.format = GrammarFormat(format)
        except ValueError:
            raise ValueError(f"Invalid format: {format}. Use 'json', 'json_array', or 'function_call'")

        self.allow_reasoning = allow_reasoning

        # Register tools
        if tools:
            for tool in tools:
                self.registry.register(tool)

        # Set system prompt
        if system_prompt is None:
            tools_description = self.registry.to_prompt_string()
            self.system_prompt = self.DEFAULT_SYSTEM_PROMPT.format(tools=tools_description)
        else:
            self.system_prompt = system_prompt

    def run(self, task: str) -> AgentResult:
        """
        Run the agent on a task.

        Args:
            task: Task description or question

        Returns:
            AgentResult with answer and execution trace
        """
        events = list(self.stream(task))

        # Extract final answer
        answer = ""
        error = None
        success = False

        for event in reversed(events):
            if event.type == EventType.ANSWER:
                answer = event.content
                success = True
                break
            elif event.type == EventType.ERROR:
                error = event.content
                break

        return AgentResult(
            answer=answer,
            steps=events,
            iterations=len([e for e in events if e.type == EventType.ACTION]),
            success=success,
            error=error,
        )

    def stream(self, task: str) -> Iterator[AgentEvent]:
        """
        Stream agent execution events.

        Args:
            task: Task description or question

        Yields:
            AgentEvent instances as agent executes
        """
        start_time = time.perf_counter()
        self._metrics = AgentMetrics()

        logger.info("Starting agent task: %s", task[:100] + "..." if len(task) > 100 else task)

        # Build initial prompt
        conversation_history = f"{self.system_prompt}\n\nQuestion: {task}\n\n"

        # Track recent actions for loop detection
        recent_actions: List[str] = []
        # Track recent tool names for same-tool detection
        recent_tools: List[str] = []
        # Track observations for potential summary
        observations: List[str] = []
        # Track if last action had an error (for loop detection relaxation)
        last_action_had_error: bool = False

        for iteration in range(self.max_iterations):
            self._metrics.iterations = iteration + 1
            logger.debug("Iteration %d/%d", iteration + 1, self.max_iterations)

            # Ensure prompt is within limits BEFORE sending to LLM
            conversation_history = self._truncate_prompt(conversation_history, task)

            if self.verbose:
                print(f"\n--- Iteration {iteration + 1} ---")

            # Generate with grammar constraint
            try:
                gen_start = time.perf_counter()
                response_json = self._generate_constrained(conversation_history)
                gen_time = (time.perf_counter() - gen_start) * 1000
                self._metrics.generation_time_ms += gen_time

                logger.debug("Generation took %.1fms", gen_time)

                if self.verbose:
                    print(f"Response: {json.dumps(response_json, indent=2)}")

                # Parse response type
                response_type = response_json.get("type")

                if response_type == "answer":
                    # Final answer
                    answer = response_json.get("content", "")
                    self._metrics.total_time_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(
                        "Agent completed successfully in %.1fms with %d iterations",
                        self._metrics.total_time_ms,
                        self._metrics.iterations,
                    )
                    event = AgentEvent(type=EventType.ANSWER, content=answer)
                    yield event
                    return

                elif response_type == "tool_call":
                    # Extract reasoning if present
                    reasoning = response_json.get("reasoning")
                    if reasoning:
                        logger.debug("Reasoning: %s", reasoning)
                        event = AgentEvent(type=EventType.THOUGHT, content=reasoning)
                        yield event

                    # Tool call
                    tool_name = str(response_json.get("tool_name") or "")
                    tool_args = response_json.get("tool_args", {})

                    action_str = f"{tool_name}({json.dumps(tool_args)})"
                    logger.debug("Tool call: %s", action_str)

                    # Loop detection: check if same action or same tool repeated too many times
                    # But allow one retry after an error (model may self-correct based on error feedback)
                    if self.detect_loops:
                        # Only add to loop detection if the previous action didn't result in an error
                        # This gives the model a chance to retry after seeing error feedback
                        if not last_action_had_error:
                            recent_actions.append(action_str)
                            recent_tools.append(tool_name)
                        else:
                            # Reset error flag - we've given one retry chance
                            last_action_had_error = False

                        det = detect_loop(
                            recent_actions,
                            recent_tools,
                            self.max_consecutive_same_action,
                            self.max_consecutive_same_tool,
                        )
                        if det is not None:
                            self._metrics.loop_detected = True
                            self._metrics.total_time_ms = (time.perf_counter() - start_time) * 1000
                            logger.warning(
                                "Loop detected (%s) after %d iterations: %s",
                                det.kind,
                                iteration + 1,
                                det.value,
                            )
                            if observations:
                                summary = self._generate_loop_summary(task, observations)
                                logger.info("Generated summary from %d observations", len(observations))
                                yield AgentEvent(type=EventType.ANSWER, content=summary)
                            else:
                                yield AgentEvent(type=EventType.ERROR, content=format_loop_error(det))
                            return

                    self._metrics.tool_calls += 1
                    event = AgentEvent(
                        type=EventType.ACTION,
                        content=action_str,
                        metadata={"tool_name": tool_name, "tool_args": tool_args},
                    )
                    yield event

                    # Execute tool
                    try:
                        tool_start = time.perf_counter()
                        observation = self._execute_tool(tool_name, tool_args)
                        tool_time = (time.perf_counter() - tool_start) * 1000
                        self._metrics.tool_time_ms += tool_time
                        logger.debug("Tool %s executed in %.1fms", tool_name, tool_time)
                        observations.append(f"{tool_name}: {observation}")
                    except Exception as e:
                        self._metrics.error_count += 1
                        observation = f"Error: {str(e)}"
                        last_action_had_error = True  # Allow retry without loop detection
                        logger.error("Tool %s failed: %s", tool_name, str(e))
                        error_event = AgentEvent(
                            type=EventType.ERROR, content=str(e), metadata={"tool_name": tool_name}
                        )
                        yield error_event

                    # Emit observation
                    obs_event = AgentEvent(
                        type=EventType.OBSERVATION, content=observation, metadata={"tool_name": tool_name}
                    )
                    yield obs_event

                    # Update conversation
                    conversation_history += f"Tool Call: {action_str}\nResult: {observation}\n\n"

                    # Truncate prompt if it exceeds max context
                    conversation_history = self._truncate_prompt(conversation_history, task)

                else:
                    # Unknown response type
                    self._metrics.error_count += 1
                    error_msg = f"Unknown response type: {response_type}"
                    logger.error(error_msg)
                    yield AgentEvent(type=EventType.ERROR, content=error_msg)
                    return

            except json.JSONDecodeError as e:
                # This should not happen with grammar constraints, but handle it
                self._metrics.error_count += 1
                error_msg = f"Failed to parse JSON response: {str(e)}"
                logger.error(error_msg)
                yield AgentEvent(type=EventType.ERROR, content=error_msg)
                return
            except Exception as e:
                self._metrics.error_count += 1
                error_msg = f"Unexpected error: {str(e)}"
                logger.exception("Unexpected error during agent execution")
                yield AgentEvent(type=EventType.ERROR, content=error_msg)
                return

        # Max iterations reached
        self._metrics.total_time_ms = (time.perf_counter() - start_time) * 1000
        logger.warning("Agent reached max iterations (%d) in %.1fms", self.max_iterations, self._metrics.total_time_ms)
        error_event = AgentEvent(
            type=EventType.ERROR,
            content=f"Reached maximum iterations ({self.max_iterations})",
            metadata={"iterations": self.max_iterations},
        )
        yield error_event

    def _truncate_prompt(self, prompt: str, task: str) -> str:
        """
        Truncate prompt to stay within max_context_chars limit.

        Preserves the system prompt and question, removes older conversation
        history from the middle.

        Args:
            prompt: Current full prompt
            task: Original task (to preserve in truncation)

        Returns:
            Truncated prompt if needed, otherwise unchanged
        """
        if len(prompt) <= self.max_context_chars:
            return prompt

        logger.debug("Truncating prompt from %d to %d chars", len(prompt), self.max_context_chars)

        # Find the start of conversation history (after "Question: ...")
        question_marker = f"Question: {task}"
        question_idx = prompt.find(question_marker)
        if question_idx == -1:
            # Fallback: just truncate from the end (not ideal)
            return prompt[: self.max_context_chars]

        # Split into header (system prompt + question) and history
        header_end = question_idx + len(question_marker)
        header = prompt[:header_end]
        history = prompt[header_end:]

        # Calculate how much history we can keep
        available_for_history = self.max_context_chars - len(header) - 100  # 100 char buffer

        if available_for_history <= 0:
            logger.warning("System prompt and question exceed max_context_chars")
            return prompt[: self.max_context_chars]

        # Keep the most recent history (from the end)
        if len(history) > available_for_history:
            # Add a truncation marker
            truncation_notice = "\n\n[...earlier conversation truncated...]\n\n"
            truncated_history = truncation_notice + history[-(available_for_history - len(truncation_notice)) :]
            return header + truncated_history

        return prompt

    def _generate_loop_summary(self, task: str, observations: List[str]) -> str:
        """
        Generate a summary answer from collected observations when loop is detected.

        Args:
            task: Original task
            observations: List of tool observations collected

        Returns:
            Summary string
        """
        # Simple summary: combine observations
        summary_parts = []
        for obs in observations:
            # Extract the key information from each observation
            if ": " in obs:
                tool, result = obs.split(": ", 1)
                # Truncate long results
                if len(result) > 200:
                    result = result[:200] + "..."
                summary_parts.append(result)
            else:
                summary_parts.append(obs)

        if summary_parts:
            return "Based on available information: " + " ".join(summary_parts[:5])
        return "Unable to complete task due to loop detection."

    @property
    def metrics(self) -> Optional[AgentMetrics]:
        """Get metrics from the last run."""
        return getattr(self, "_metrics", None)

    def _generate_constrained(self, prompt: str) -> Dict[str, Any]:
        """
        Generate with grammar constraint and parse JSON.

        Args:
            prompt: Input prompt

        Returns:
            Parsed JSON response
        """
        # Get or generate grammar
        # Use answer_or_tool grammar which allows both final answers and tool calls
        tools_list = self.registry.list_tools()

        if self.use_cache:
            grammar = get_cached_answer_or_tool_grammar(tools_list, allow_reasoning=self.allow_reasoning)
        else:
            grammar = generate_answer_or_tool_grammar(tools_list, allow_reasoning=self.allow_reasoning)

        # Create generation config
        # IMPORTANT: Preserve n_batch and n_ctx from LLM's config to handle large grammars
        # Get batch and ctx from LLM config if available (for testing, LLM might not have config)
        llm_config = getattr(self.llm, "config", None)
        n_batch = llm_config.n_batch if llm_config and llm_config.n_batch else 512
        n_ctx = llm_config.n_ctx if llm_config else None

        config = GenerationConfig(
            temperature=self.generation_config.temperature,
            max_tokens=self.generation_config.max_tokens,
            top_k=self.generation_config.top_k,
            top_p=self.generation_config.top_p,
            min_p=self.generation_config.min_p,
            n_batch=n_batch,  # Preserve batch size from LLM or use default
            n_ctx=n_ctx,  # Preserve context size from LLM (None is OK)
        )

        # Use grammar-constrained generation if available. We test for the
        # method rather than the class because real-LLM callers don't get an
        # actual GrammarConstrainedLLM instance — the methods are bound onto
        # their existing LLM in `__init__`. The previous isinstance check
        # silently routed every real call through the un-constrained mock
        # fallback, making the grammar-enforcement claim untrue in
        # production.
        if callable(getattr(self.llm, "generate_with_grammar", None)):
            response = self.llm.generate_with_grammar(prompt, grammar=grammar, grammar_root="root", config=config)
        else:
            # Fallback for mock LLMs in tests
            result = self.llm(prompt, config=config, stream=False)
            response = str(result)  # Convert Response to string

        # Parse JSON from response
        # Try to extract JSON if wrapped in markdown or other text
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        response = response.strip()

        # Parse JSON
        return cast(Dict[str, Any], json.loads(response))

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Execute a tool with given arguments.

        Args are validated and coerced against the tool's JSON-schema before
        dispatch (when ``tool.coerce`` is True, the default). See
        ``coerce_args`` for the policy.

        Args:
            tool_name: Name of tool to execute
            args: Arguments to pass to tool

        Returns:
            String representation of tool result, or an error string the
            agent loop will surface as an observation.

        Raises:
            ValueError: If tool not found
        """
        tool = self.registry.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        try:
            if tool.coerce:
                args = coerce_args(tool, args)
            result = tool(**args)
            return render_observation(result)
        except ToolArgumentError as e:
            # Surface schema-violation details verbatim so the LLM can fix
            # the call on the next iteration.
            return f"Tool argument error: {e.message}"
        except Exception as e:
            return f"Tool execution error: {str(e)}"

    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the agent's registry.

        Args:
            tool: Tool to add
        """
        self.registry.register(tool)

    def list_tools(self) -> List[Tool]:
        """Get list of available tools."""
        return self.registry.list_tools()
