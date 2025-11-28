"""
Constrained agent implementation using grammar-enforced tool calling.

Uses GBNF grammars to guarantee valid tool call syntax, eliminating parsing
failures and enabling reliable agent execution even with smaller models.
"""

import json
import logging
import time
from typing import List, Optional, Iterator, Dict, Any, Callable
from dataclasses import dataclass, field

from ..api import LLM, GenerationConfig

# Module logger
logger = logging.getLogger(__name__)
from ..llama.llama_cpp import (
    LlamaModel,
    LlamaContext,
    LlamaModelParams,
    LlamaContextParams,
    LlamaSampler,
    LlamaSamplerChainParams,
    llama_batch_get_one,
    ggml_backend_load_all,
    disable_logging,
)
from .tools import Tool, ToolRegistry
from .grammar import (
    GrammarFormat,
    generate_tool_call_grammar,
    generate_answer_or_tool_grammar,
    get_cached_tool_grammar,
    get_cached_answer_or_tool_grammar,
)
from .react import EventType, AgentEvent, AgentResult


class GrammarConstrainedLLM(LLM):
    """
    LLM subclass that supports GBNF grammar constraints for generation.

    This class extends the base LLM to add grammar-constrained generation,
    ensuring outputs conform to a specified GBNF grammar format.
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[GenerationConfig] = None,
        verbose: bool = False
    ):
        """Initialize grammar-constrained LLM."""
        super().__init__(model_path, config, verbose)

    def generate_with_grammar(
        self,
        prompt: str,
        grammar: str,
        grammar_root: str = "root",
        config: Optional[GenerationConfig] = None,
        on_token: Optional[Callable[[str], None]] = None
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
        prompt_tokens = self.vocab.tokenize(
            prompt,
            add_special=config.add_bos,
            parse_special=config.parse_special
        )
        n_prompt = len(prompt_tokens)

        if self.verbose:
            print(f"Prompt tokens: {n_prompt}")
            print(f"Applying grammar constraint: {len(grammar)} chars")

        # Ensure context is ready
        self._ensure_context(n_prompt, config)

        # Create sampler with grammar constraint
        self._ensure_sampler_with_grammar(config, grammar, grammar_root)

        # Process prompt
        batch = llama_batch_get_one(prompt_tokens)
        self._ctx.decode(batch)

        # Generate tokens
        n_pos = n_prompt
        n_generated = 0
        output_tokens = []

        for _ in range(config.max_tokens):
            # Sample next token with grammar constraint
            new_token_id = self._sampler.sample(self._ctx, -1)

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
            self._ctx.decode(batch)
            n_pos += 1
            n_generated += 1

        return "".join(output_tokens)

    def _ensure_sampler_with_grammar(
        self,
        config: GenerationConfig,
        grammar: str,
        grammar_root: str
    ):
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
                penalty_last_n=64,
                penalty_repeat=config.repeat_penalty,
                penalty_freq=0.0,
                penalty_present=0.0
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


@dataclass
class AgentMetrics:
    """Performance metrics for agent execution."""
    total_time_ms: float = 0.0
    iterations: int = 0
    tool_calls: int = 0
    tool_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    tokens_generated: int = 0
    loop_detected: bool = False
    error_count: int = 0

    def __str__(self) -> str:
        return (
            f"AgentMetrics(iterations={self.iterations}, "
            f"tool_calls={self.tool_calls}, "
            f"total_time={self.total_time_ms:.1f}ms, "
            f"gen_time={self.generation_time_ms:.1f}ms, "
            f"tool_time={self.tool_time_ms:.1f}ms)"
        )


class ConstrainedAgent:
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
        # Wrap LLM in GrammarConstrainedLLM if needed
        if isinstance(llm, GrammarConstrainedLLM):
            self.llm = llm
        else:
            # Check if this is a real LLM with model (not a mock)
            if hasattr(llm, 'model') and hasattr(llm, 'vocab'):
                # Create a GrammarConstrainedLLM using the same model
                self.llm = GrammarConstrainedLLM.__new__(GrammarConstrainedLLM)
                # Copy all attributes from the original LLM
                self.llm.model_path = llm.model_path
                self.llm.config = llm.config
                self.llm.verbose = llm.verbose
                self.llm.model = llm.model
                self.llm.vocab = llm.vocab
                self.llm._ctx = llm._ctx
                self.llm._sampler = llm._sampler
            else:
                # For mocks or test LLMs, use as-is
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
            error=error
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
                    logger.info("Agent completed successfully in %.1fms with %d iterations",
                               self._metrics.total_time_ms, self._metrics.iterations)
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
                    tool_name = response_json.get("tool_name")
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

                        # Check for exact same action repeated
                        if len(recent_actions) >= self.max_consecutive_same_action:
                            last_n = recent_actions[-self.max_consecutive_same_action:]
                            if all(a == last_n[0] for a in last_n):
                                self._metrics.loop_detected = True
                                self._metrics.total_time_ms = (time.perf_counter() - start_time) * 1000
                                logger.warning("Loop detected (same action) after %d iterations: %s",
                                             iteration + 1, action_str)

                                # Generate summary from observations if available
                                if observations:
                                    summary = self._generate_loop_summary(task, observations)
                                    logger.info("Generated summary from %d observations", len(observations))
                                    yield AgentEvent(type=EventType.ANSWER, content=summary)
                                else:
                                    error_msg = (
                                        f"Loop detected: same action repeated "
                                        f"{self.max_consecutive_same_action} times: {action_str}"
                                    )
                                    yield AgentEvent(type=EventType.ERROR, content=error_msg)
                                return

                        # Check for same tool called too many times (even with different args)
                        if len(recent_tools) >= self.max_consecutive_same_tool:
                            last_n_tools = recent_tools[-self.max_consecutive_same_tool:]
                            if all(t == last_n_tools[0] for t in last_n_tools):
                                self._metrics.loop_detected = True
                                self._metrics.total_time_ms = (time.perf_counter() - start_time) * 1000
                                logger.warning("Loop detected (same tool) after %d iterations: %s called %d times",
                                             iteration + 1, tool_name, self.max_consecutive_same_tool)

                                # Generate summary from observations if available
                                if observations:
                                    summary = self._generate_loop_summary(task, observations)
                                    logger.info("Generated summary from %d observations", len(observations))
                                    yield AgentEvent(type=EventType.ANSWER, content=summary)
                                else:
                                    error_msg = (
                                        f"Loop detected: tool '{tool_name}' called "
                                        f"{self.max_consecutive_same_tool} times consecutively"
                                    )
                                    yield AgentEvent(type=EventType.ERROR, content=error_msg)
                                return

                    self._metrics.tool_calls += 1
                    event = AgentEvent(
                        type=EventType.ACTION,
                        content=action_str,
                        metadata={"tool_name": tool_name, "tool_args": tool_args}
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
                            type=EventType.ERROR,
                            content=str(e),
                            metadata={"tool_name": tool_name}
                        )
                        yield error_event

                    # Emit observation
                    obs_event = AgentEvent(
                        type=EventType.OBSERVATION,
                        content=observation,
                        metadata={"tool_name": tool_name}
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
        logger.warning("Agent reached max iterations (%d) in %.1fms",
                      self.max_iterations, self._metrics.total_time_ms)
        error_event = AgentEvent(
            type=EventType.ERROR,
            content=f"Reached maximum iterations ({self.max_iterations})",
            metadata={"iterations": self.max_iterations}
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
            return prompt[:self.max_context_chars]

        # Split into header (system prompt + question) and history
        header_end = question_idx + len(question_marker)
        header = prompt[:header_end]
        history = prompt[header_end:]

        # Calculate how much history we can keep
        available_for_history = self.max_context_chars - len(header) - 100  # 100 char buffer

        if available_for_history <= 0:
            logger.warning("System prompt and question exceed max_context_chars")
            return prompt[:self.max_context_chars]

        # Keep the most recent history (from the end)
        if len(history) > available_for_history:
            # Add a truncation marker
            truncation_notice = "\n\n[...earlier conversation truncated...]\n\n"
            truncated_history = truncation_notice + history[-(available_for_history - len(truncation_notice)):]
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
        return getattr(self, '_metrics', None)

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
            grammar = get_cached_answer_or_tool_grammar(
                tools_list,
                allow_reasoning=self.allow_reasoning
            )
        else:
            grammar = generate_answer_or_tool_grammar(
                tools_list,
                allow_reasoning=self.allow_reasoning
            )

        # Create generation config
        # IMPORTANT: Preserve n_batch and n_ctx from LLM's config to handle large grammars
        # Get batch and ctx from LLM config if available (for testing, LLM might not have config)
        llm_config = getattr(self.llm, 'config', None)
        n_batch = llm_config.n_batch if llm_config and llm_config.n_batch else 512
        n_ctx = llm_config.n_ctx if llm_config else None

        config = GenerationConfig(
            temperature=self.generation_config.temperature,
            max_tokens=self.generation_config.max_tokens,
            top_k=self.generation_config.top_k,
            top_p=self.generation_config.top_p,
            min_p=self.generation_config.min_p,
            n_batch=n_batch,  # Preserve batch size from LLM or use default
            n_ctx=n_ctx,      # Preserve context size from LLM (None is OK)
        )

        # Use grammar-constrained generation if available
        if isinstance(self.llm, GrammarConstrainedLLM):
            response = self.llm.generate_with_grammar(
                prompt,
                grammar=grammar,
                grammar_root="root",
                config=config
            )
        else:
            # Fallback for mock LLMs in tests
            response = self.llm(prompt, config=config, stream=False)

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
        return json.loads(response)

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Execute a tool with given arguments.

        Args:
            tool_name: Name of tool to execute
            args: Arguments to pass to tool

        Returns:
            String representation of tool result

        Raises:
            ValueError: If tool not found
        """
        tool = self.registry.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        try:
            result = tool(**args)
            return str(result)
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


class EnhancedConstrainedAgent(ConstrainedAgent):
    """
    Agent with true grammar-constrained generation.

    This will use the LlamaSampler.add_grammar() method to enforce
    grammar constraints during generation, ensuring 100% valid JSON.

    Note:
        This class is not yet implemented. Instantiating it will raise
        NotImplementedError. Use ConstrainedAgent instead.
    """

    def __init__(self, *args, **kwargs):
        """Raise NotImplementedError as this class is not yet implemented."""
        raise NotImplementedError(
            "EnhancedConstrainedAgent is not yet implemented. "
            "Use ConstrainedAgent instead, which provides JSON validation "
            "through post-processing rather than grammar constraints."
        )
