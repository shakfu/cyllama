"""
ReAct (Reasoning + Acting) agent implementation.

Implements the ReAct pattern where the agent alternates between:
1. Thought: Reasoning about what to do next
2. Action: Invoking a tool
3. Observation: Seeing the result
4. (Repeat until task is complete)

Reference: https://arxiv.org/abs/2210.03629
"""

import logging
import re
import time
from typing import List, Optional, Iterator, Callable, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from ..api import LLM, GenerationConfig
from .tools import Tool, ToolRegistry

# Module logger
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events emitted during agent execution."""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    ANSWER = "answer"
    ERROR = "error"
    # Contract-related events
    CONTRACT_CHECK = "contract_check"
    CONTRACT_VIOLATION = "contract_violation"


@dataclass
class AgentEvent:
    """Event emitted during agent execution."""
    type: EventType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


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


@dataclass
class AgentResult:
    """Result from agent execution."""
    answer: str
    steps: List[AgentEvent]
    iterations: int
    success: bool
    error: Optional[str] = None
    metrics: Optional[AgentMetrics] = None


class ReActAgent:
    """
    ReAct agent that uses reasoning and tool calling to solve tasks.

    The agent follows this pattern:
        Thought: [reasoning about what to do]
        Action: tool_name(arg1="value1", arg2="value2")
        Observation: [result from tool]
        ... (repeat)
        Thought: I now know the final answer
        Answer: [final answer]

    Example:
        from cyllama import LLM
        from cyllama.agents import ReActAgent, tool

        @tool
        def search(query: str) -> str:
            return "Search results for: " + query

        agent = ReActAgent(
            llm=LLM("models/mistral-7b-instruct.gguf"),
            tools=[search],
        )

        result = agent.run("What is the capital of France?")
        print(result.answer)
    """

    # Default system prompt template
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that uses tools to solve tasks.

You have access to the following tools:

{tools}

Use the following format:

Thought: think about what to do next
Action: tool_name({{"arg1": "value1", "arg2": "value2"}})
Observation: the result of the action
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Answer: the final answer to the user's question

IMPORTANT RULES:
1. Only call ONE tool at a time - do not nest tool calls
2. For multi-line strings, use escaped newlines: {{"code": "line1\\nline2\\nline3"}}
3. Do NOT use triple quotes (\"\"\" or ''') - they are not valid JSON
4. Wait for the Observation before deciding the next action

Begin!"""

    def __init__(
        self,
        llm: LLM,
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = False,
        generation_config: Optional[GenerationConfig] = None,
        detect_loops: bool = True,
        max_consecutive_same_action: int = 2,
        max_consecutive_same_tool: int = 4,
        max_context_chars: int = 6000,
    ):
        """
        Initialize ReAct agent.

        Args:
            llm: LLM instance for generation
            tools: List of tools available to the agent
            system_prompt: Custom system prompt (uses default if None)
            max_iterations: Maximum number of thought/action cycles
            verbose: Print agent reasoning to stdout
            generation_config: Custom generation config for LLM
            detect_loops: Enable loop detection to prevent infinite loops (default: True)
            max_consecutive_same_action: Number of times the exact same action can repeat
                                         before considered a loop (default: 2)
            max_consecutive_same_tool: Number of times the same tool can be called
                                       consecutively (with any args) before loop (default: 4)
            max_context_chars: Maximum characters for the prompt context. Older history
                              is truncated to stay within this limit (default: 16000)
        """
        self.llm = llm
        self.registry = ToolRegistry()
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.detect_loops = detect_loops
        self.max_consecutive_same_action = max_consecutive_same_action
        self.max_consecutive_same_tool = max_consecutive_same_tool
        self.max_context_chars = max_context_chars
        self.generation_config = generation_config or GenerationConfig(
            temperature=0.7,
            max_tokens=512,
            stop_sequences=[
                "\nObservation:",   # Most common - newline then Observation:
                "\nObservation :",  # With space before colon
                "\nObservation\n",  # Newline, Observation, newline (no colon)
                "Observation:",     # At start or mid-text
                "Observation :",    # With space before colon
            ]
        )

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
            iterations=len([e for e in events if e.type == EventType.THOUGHT]),
            success=success,
            error=error,
            metrics=self.metrics
        )

    @property
    def metrics(self) -> Optional[AgentMetrics]:
        """Get metrics from the last run."""
        return getattr(self, '_metrics', None)

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

        logger.info("Starting ReAct agent task: %s", task[:100] + "..." if len(task) > 100 else task)

        # Build initial prompt
        prompt = f"{self.system_prompt}\n\nQuestion: {task}\n\n"

        # Track recent actions for loop detection
        recent_actions: List[str] = []
        # Track recent tool names for same-tool detection
        recent_tools: List[str] = []
        # Track observations for potential summary on loop detection
        observations: List[str] = []

        for iteration in range(self.max_iterations):
            self._metrics.iterations = iteration + 1
            logger.debug("Iteration %d/%d", iteration + 1, self.max_iterations)

            # Ensure prompt is within limits BEFORE sending to LLM
            old_len = len(prompt)
            prompt = self._truncate_prompt(prompt, task)
            if self.verbose and old_len != len(prompt):
                print(f"[Truncated prompt from {old_len} to {len(prompt)} chars]")
            elif self.verbose:
                print(f"[Prompt size: {len(prompt)} chars]")

            # Generate thought and action
            gen_start = time.perf_counter()
            response = self.llm(prompt, config=self.generation_config, stream=False)
            gen_time = (time.perf_counter() - gen_start) * 1000
            self._metrics.generation_time_ms += gen_time
            logger.debug("Generation took %.1fms", gen_time)

            # Strip any hallucinated observations - the model should NOT generate these
            response = self._strip_hallucinated_observation(response)

            if self.verbose:
                print(f"\n--- Iteration {iteration + 1} ---")
                print(response)

            # Parse response
            thought = self._extract_thought(response)
            action_str = self._extract_action(response)

            # Emit thought event
            if thought:
                logger.debug("Thought: %s", thought[:100] + "..." if len(thought) > 100 else thought)
                event = AgentEvent(type=EventType.THOUGHT, content=thought)
                yield event

            # Check if agent has final answer
            answer = self._extract_answer(response)
            if answer:
                self._metrics.total_time_ms = (time.perf_counter() - start_time) * 1000
                logger.info("Agent completed successfully in %.1fms with %d iterations",
                           self._metrics.total_time_ms, self._metrics.iterations)
                event = AgentEvent(type=EventType.ANSWER, content=answer)
                yield event
                return

            # Execute action if present
            if action_str:
                logger.debug("Action: %s", action_str)

                # Parse tool name early for loop detection
                parse_failed = False
                try:
                    tool_name, tool_args = self._parse_action(action_str)
                except Exception as e:
                    tool_name = None
                    tool_args = {}
                    parse_failed = True
                    logger.debug("Action parse failed: %s", str(e))

                # Loop detection: track both successful and failed actions
                if self.detect_loops:
                    # Always track action strings for exact-match detection
                    recent_actions.append(action_str)

                    # Track tool name if parsed, or a placeholder for failed parses
                    if tool_name:
                        recent_tools.append(tool_name)
                    elif parse_failed:
                        # Track failed parses as a special "tool" to detect parse failure loops
                        recent_tools.append("__PARSE_FAILED__")

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

                            tool_or_error = tool_name if tool_name else "parse failures"
                            logger.warning("Loop detected (same tool) after %d iterations: %s occurred %d times",
                                         iteration + 1, tool_or_error, self.max_consecutive_same_tool)

                            # Generate summary from observations if available
                            if observations:
                                summary = self._generate_loop_summary(task, observations)
                                logger.info("Generated summary from %d observations", len(observations))
                                yield AgentEvent(type=EventType.ANSWER, content=summary)
                            else:
                                error_msg = (
                                    f"Loop detected: {tool_or_error} occurred "
                                    f"{self.max_consecutive_same_tool} times consecutively"
                                )
                                yield AgentEvent(type=EventType.ERROR, content=error_msg)
                            return

                self._metrics.tool_calls += 1

                # Emit action event
                event = AgentEvent(
                    type=EventType.ACTION,
                    content=action_str,
                    metadata={
                        "iteration": iteration + 1,
                        "tool_name": tool_name,
                        "tool_args": tool_args
                    }
                )
                yield event

                # Execute tool call (already parsed above)
                raw_result = None
                try:
                    if tool_name is None:
                        # Re-parse if initial parse failed
                        tool_name, tool_args = self._parse_action(action_str)
                    tool_start = time.perf_counter()
                    raw_result = self._execute_tool_raw(tool_name, tool_args)
                    observation = str(raw_result)
                    tool_time = (time.perf_counter() - tool_start) * 1000
                    self._metrics.tool_time_ms += tool_time
                    logger.debug("Tool %s executed in %.1fms", tool_name, tool_time)
                except Exception as e:
                    self._metrics.error_count += 1
                    observation = f"Error: {str(e)}"
                    logger.error("Tool execution failed: %s", str(e))
                    error_event = AgentEvent(
                        type=EventType.ERROR,
                        content=str(e),
                        metadata={"action": action_str}
                    )
                    yield error_event

                # Emit observation event
                obs_event = AgentEvent(
                    type=EventType.OBSERVATION,
                    content=observation,
                    metadata={
                        "action": action_str,
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "raw_result": raw_result  # Actual return value for contract checking
                    }
                )
                yield obs_event

                # Track observation for potential summary on loop detection
                if observation and not observation.startswith("Error:"):
                    observations.append(f"{tool_name}: {observation}")

                # Truncate observation if too long to prevent context overflow
                max_obs_len = 1000
                if len(observation) > max_obs_len:
                    observation = observation[:max_obs_len] + "... [truncated]"

                # Truncate response if model hallucinated too much
                max_response_len = 500
                if len(response) > max_response_len:
                    response = response[:max_response_len]
                    logger.debug("Truncated long response from %d chars", len(response))

                # Update prompt with observation
                prompt += response + f"\nObservation: {observation}\n"

                # Truncate prompt if it exceeds max context
                prompt = self._truncate_prompt(prompt, task)
            else:
                # No action found, prompt for next step
                prompt += response + "\n"

                # Truncate prompt if it exceeds max context
                prompt = self._truncate_prompt(prompt, task)

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

    def _strip_hallucinated_observation(self, text: str) -> str:
        """
        Remove any self-generated Observation: content from model response.

        The model should not generate observations - those come from actual
        tool execution. This prevents the model from hallucinating results.
        """
        # Find "Observation" patterns and truncate everything after it
        # Order matters - check longer/more specific patterns first
        patterns = [
            "\nObservation:",
            "\nObservation :",
            "\nObservation\n",
            "\nObservation",   # Observation at end after newline
            "Observation:",
            "Observation :",
        ]
        for pattern in patterns:
            if pattern in text:
                text = text[:text.index(pattern)]
                break  # Stop after first match

        # Also strip trailing "Observation" at end of text (no newline before)
        text = text.strip()
        if text.endswith("Observation"):
            text = text[:-len("Observation")].strip()

        return text

    def _extract_thought(self, text: str) -> Optional[str]:
        """Extract thought from agent response."""
        match = re.search(r"Thought:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
        if match:
            # Stop at next keyword
            thought = match.group(1)
            for keyword in ["Action:", "Answer:", "Observation:"]:
                if keyword in thought:
                    thought = thought[:thought.index(keyword)]
            return thought.strip()
        return None

    def _extract_action(self, text: str) -> Optional[str]:
        """Extract action from agent response."""
        match = re.search(r"Action:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if match:
            action = match.group(1).strip()
            # Handle double Action: prefix (model sometimes outputs "Action: Action: tool(...)")
            while action.lower().startswith("action:"):
                action = action[7:].strip()
            return action
        return None

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract final answer from agent response."""
        match = re.search(r"Answer:\s*(.+?)$", text, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # Truncate at common continuation markers that indicate hallucination
            # These patterns suggest the model is generating new questions/content
            stop_patterns = [
                "\n\nQuestion:",
                "\nQuestion:",
                "\n\nThought:",
                "\nThought:",
                "\n\n```",  # Code block start
                "\n```",    # Code block with single newline
                "```python",  # Direct code block
                "```",      # Any code block marker
                "\n\nNote:",
                "\nNote:",
                "\n\nLet's",  # "Let's try another example"
                "\nLet's",
                "\n\nThis is",  # "This is a basic example..."
                "\nThis is a basic",
                "\n\nIn a real",  # "In a real-world scenario..."
                "\n\n#",  # Hashtags
                "\n#",
                "\n\nI will",  # "I will provide the output..."
                "\nI will",
                "\n\nHere is",  # "Here is the code..."
                "\nHere is the code",
                "\n\nclass ",  # Direct class definition
                "\ndef ",     # Direct function definition
            ]
            for pattern in stop_patterns:
                if pattern in answer:
                    answer = answer[:answer.index(pattern)]
            return answer.strip()
        return None

    def _parse_action(self, action_str: str) -> tuple[str, Dict[str, Any]]:
        """
        Parse action string into tool name and arguments.

        Supports formats:
            tool_name(arg1="value1", arg2="value2")
            tool_name(arg1='value1', arg2='value2')
            tool_name({"arg1": "value1", "arg2": "value2"})

        Args:
            action_str: Action string to parse

        Returns:
            Tuple of (tool_name, arguments_dict)

        Raises:
            ValueError: If action string is malformed
        """
        import json

        # Try function call format: tool_name(arg1="val", arg2="val")
        # Use DOTALL to handle multi-line arguments
        match = re.match(r"(\w+)\((.*)\)", action_str.strip(), re.DOTALL)
        if not match:
            raise ValueError(f"Invalid action format: {action_str}")

        tool_name = match.group(1)
        args_str = match.group(2).strip()

        if not args_str:
            return tool_name, {}

        # Parse arguments
        args = {}

        # Try JSON object format first
        if args_str.startswith("{"):
            # Handle triple-quoted strings - these aren't valid JSON
            # Provide helpful error message guiding the model to use \n escapes
            if '"""' in args_str or "'''" in args_str:
                raise ValueError(
                    f"Triple-quoted strings are not valid JSON. "
                    f"Use escaped newlines instead: {{\"code\": \"line1\\nline2\"}}"
                )

            try:
                args = json.loads(args_str)
                # Post-process string values to convert literal \n to actual newlines
                # This handles cases where the model outputs {"code": "line1\nline2"}
                # where the \n is meant to be a newline but JSON didn't interpret it
                args = self._convert_escape_sequences(args)
                return tool_name, args
            except json.JSONDecodeError as e:
                # If JSON parsing fails, provide helpful error
                raise ValueError(
                    f"Invalid JSON in tool arguments. "
                    f"For multi-line strings, use escaped newlines: {{\"code\": \"line1\\nline2\"}}"
                )

        # Parse key=value pairs with proper string handling
        # First, check for triple-quoted strings which aren't valid
        if '"""' in args_str or "'''" in args_str:
            raise ValueError(
                f"Triple-quoted strings are not supported. "
                f"Use escaped newlines instead: code=\"line1\\nline2\""
            )

        # Handle both single and double quotes, including escaped quotes within
        # Pattern matches: key="value" or key='value'
        # This simpler pattern works for basic cases
        pattern = r'(\w+)\s*=\s*["\']([^"\']*)["\']'
        matches = re.findall(pattern, args_str)

        if matches:
            args = {key: value for key, value in matches}
            # Convert escape sequences in parsed values
            args = self._convert_escape_sequences(args)
        else:
            # Try positional argument (single value)
            # Remove quotes if present
            value = args_str.strip().strip('"').strip("'")
            # Get first parameter name from tool
            tool = self.registry.get(tool_name)
            if tool:
                param_names = list(tool.parameters.get("properties", {}).keys())
                if param_names:
                    args = {param_names[0]: value}
                    # Convert escape sequences
                    args = self._convert_escape_sequences(args)

        return tool_name, args

    def _convert_escape_sequences(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert literal escape sequences in string values to actual characters.

        This handles cases where the model outputs strings like "line1\\nline2"
        where the \\n should be interpreted as a newline character.

        Args:
            args: Dictionary of arguments

        Returns:
            Dictionary with escape sequences converted in string values
        """
        result = {}
        for key, value in args.items():
            if isinstance(value, str):
                # Convert common escape sequences that may be literal
                # Only convert if the string contains literal backslash-n patterns
                value = value.replace('\\n', '\n')
                value = value.replace('\\t', '\t')
                value = value.replace('\\r', '\r')
                value = value.replace('\\"', '"')
                value = value.replace("\\'", "'")
            elif isinstance(value, dict):
                value = self._convert_escape_sequences(value)
            elif isinstance(value, list):
                value = [
                    self._convert_escape_sequences({'_': v})['_']
                    if isinstance(v, (str, dict, list)) else v
                    for v in value
                ]
            result[key] = value
        return result

    def _execute_tool_raw(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """
        Execute a tool and return the raw result.

        Args:
            tool_name: Name of tool to execute
            args: Arguments to pass to tool

        Returns:
            Raw tool result (not converted to string)

        Raises:
            ValueError: If tool not found
            Exception: If tool execution fails
        """
        tool = self.registry.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        return tool(**args)

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
        # Check if tool exists first (raises ValueError if not)
        tool = self.registry.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        try:
            result = self._execute_tool_raw(tool_name, args)
            return str(result)
        except Exception as e:
            return f"Tool execution error: {str(e)}"

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
        Generate a summary answer from collected observations when a loop is detected.

        Args:
            task: The original task
            observations: List of tool observations collected during execution

        Returns:
            A summary answer based on the observations
        """
        summary_parts = []
        for obs in observations:
            if ": " in obs:
                tool, result = obs.split(": ", 1)
                # Truncate long results
                if len(result) > 200:
                    result = result[:200] + "..."
                summary_parts.append(result)
            else:
                summary_parts.append(obs)

        if summary_parts:
            # Combine first few observations into a summary
            return "Based on available information: " + " ".join(summary_parts[:5])

        return "Unable to complete task due to loop detection."

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
