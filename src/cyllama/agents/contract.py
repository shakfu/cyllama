"""
Contract-based agent with C++26-inspired contract assertions.

This module provides a Python implementation of contract programming inspired by
C++26's contract assertions proposal (P2900). It enables preconditions, postconditions,
and runtime assertions for tools and agents with configurable violation handling.

Key Differences from C++26
--------------------------
While inspired by C++26, this Python implementation differs in several ways:

1. **Runtime-only checking**: C++26 contracts can be compile-time or runtime. Python
   contracts are always checked at runtime.

2. **Dynamic predicate evaluation**: Predicates are Python callables, not expressions
   evaluated at compile time. This allows more flexibility but less static analysis.

3. **No undefined behavior**: C++26's contract violations can result in UB. Python
   contracts always have well-defined behavior based on the policy.

4. **Agent integration**: This module extends the contract concept to AI agents,
   adding task preconditions, answer postconditions, and iteration invariants.

Contract Policies
-----------------
Four policies control how violations are handled (see ContractPolicy enum):

- **IGNORE**: Skip all checking. Use for production when contracts are verified.
- **OBSERVE**: Check and log violations, but continue execution. Useful for monitoring.
- **ENFORCE** (default): Check, call violation handler, then terminate on violation.
- **QUICK_ENFORCE**: Check and terminate immediately without calling handler.

Basic Usage
-----------
Apply contracts to tools using decorators::

    from cyllama.agents import tool, pre, post, ContractAgent

    @tool
    @pre(lambda args: args['count'] > 0, "count must be positive")
    @post(lambda r: len(r) > 0, "must return results")
    def fetch_items(count: int) -> str:
        '''Fetch items from database.'''
        return f"Fetched {count} items"

    # Create agent with contract checking
    agent = ContractAgent(
        llm=my_llm,
        tools=[fetch_items],
        policy=ContractPolicy.ENFORCE  # default
    )

Runtime Assertions
------------------
Use contract_assert() within tool implementations for invariant checking::

    from cyllama.agents import tool, contract_assert

    @tool
    def process_data(data: str) -> dict:
        '''Process JSON data.'''
        import json
        parsed = json.loads(data)
        contract_assert(isinstance(parsed, dict), "data must be JSON object")
        return parsed

Agent-Level Contracts
---------------------
ContractAgent supports contracts on the overall task and answer::

    agent = ContractAgent(
        llm=my_llm,
        tools=[...],
        task_precondition=lambda task: len(task) >= 10,  # minimum task length
        answer_postcondition=lambda answer: "error" not in answer.lower(),
        iteration_invariant=lambda state: state.errors < 3  # max 2 errors
    )

See Also
--------
- C++26 contract assertions: https://wg21.link/p2900
- ContractPolicy: Enum defining violation handling policies
- ContractAgent: Agent class with contract checking
- pre, post: Decorators for tool contracts
- contract_assert: Runtime assertion function
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Protocol, Union, Generator,
    TypeVar, Generic, Tuple
)
import functools
import inspect
import logging
import time

from .tools import Tool
from .react import AgentEvent, AgentResult, AgentMetrics, EventType, ReActAgent

logger = logging.getLogger(__name__)


# =============================================================================
# Contract Policy (Evaluation Semantics)
# =============================================================================

class ContractPolicy(Enum):
    """
    Contract evaluation policy, inspired by C++26 contract semantics.

    This enum determines how contract violations are handled at runtime.
    The default policy for ContractAgent is ENFORCE, which provides a balance
    between safety (violations are caught) and debuggability (handlers are called).

    Policies
    --------
    IGNORE
        Skip all contract checking entirely. The predicate is never evaluated.
        Use this in production after contracts have been verified during development.
        Provides zero runtime overhead.

    OBSERVE
        Check contracts and call the violation handler if a violation occurs,
        but continue execution regardless. Useful for monitoring and gradual
        contract adoption where you want to log violations without breaking
        existing behavior.

    ENFORCE (default)
        Check contracts, call the violation handler on violation, then terminate
        execution by raising ContractTermination. This is the default and
        recommended policy for development and testing.

    QUICK_ENFORCE
        Check contracts and terminate immediately on violation WITHOUT calling
        the violation handler. Useful when handler overhead is unacceptable or
        when you want the fastest possible failure path.

    Policy Resolution
    -----------------
    Policies can be specified at multiple levels with the following precedence:

    1. Individual contract (highest priority)::

           @pre(lambda args: args['n'] > 0, "positive", policy=ContractPolicy.OBSERVE)

    2. ContractAgent default::

           agent = ContractAgent(llm=llm, tools=[...], policy=ContractPolicy.ENFORCE)

    When a contract's policy is None (the default for @pre/@post), the agent's
    policy is used. If no agent context exists (e.g., calling contract_assert()
    outside an agent), ENFORCE is used as the fallback.

    Examples
    --------
    Setting policy at agent level::

        # All contracts use OBSERVE unless they specify otherwise
        agent = ContractAgent(
            llm=llm,
            tools=[my_tool],
            policy=ContractPolicy.OBSERVE
        )

    Overriding policy for a specific contract::

        @tool
        @pre(lambda args: args['n'] > 0, "positive", policy=ContractPolicy.ENFORCE)
        def my_func(n: int) -> int:
            # This precondition uses ENFORCE even if agent uses OBSERVE
            return n * 2
    """
    IGNORE = "ignore"
    OBSERVE = "observe"
    ENFORCE = "enforce"
    QUICK_ENFORCE = "quick_enforce"


# =============================================================================
# Contract Violation
# =============================================================================

@dataclass
class ContractViolation:
    """
    Represents a contract violation event.

    Attributes:
        kind: Type of contract ("pre", "post", "assert")
        location: Where the violation occurred (tool name or "agent")
        predicate: String representation of the condition that failed
        message: Human-readable description of the violation
        context: Additional context (arguments, result, etc.)
        policy: The policy that was in effect when violation occurred
    """
    kind: str
    location: str
    predicate: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    policy: ContractPolicy = ContractPolicy.ENFORCE

    def __str__(self) -> str:
        return f"ContractViolation({self.kind} at {self.location}): {self.message}"


# =============================================================================
# Violation Handler Protocol
# =============================================================================

class ViolationHandler(Protocol):
    """Protocol for contract violation handlers."""

    def __call__(self, violation: ContractViolation) -> None:
        """
        Handle a contract violation.

        Args:
            violation: The violation that occurred

        May raise ContractTermination to abort execution.
        """
        ...


class ContractTermination(Exception):
    """Exception raised to terminate agent execution due to contract violation."""

    def __init__(self, violation: ContractViolation):
        self.violation = violation
        super().__init__(str(violation))


# =============================================================================
# Contract Definitions
# =============================================================================

@dataclass
class PreCondition:
    """
    A precondition contract that validates tool arguments before execution.

    Preconditions express requirements that must be true before a tool can be
    safely called. If a precondition fails, it indicates the caller provided
    invalid input.

    Attributes
    ----------
    predicate : Callable[[Dict[str, Any]], bool]
        Function that receives a dict of argument names to values and returns
        True if the precondition is satisfied.
    message : str
        Human-readable description of what the precondition requires.
    predicate_str : str
        String representation of the predicate for error reporting.
    policy : Optional[ContractPolicy]
        Policy override for this specific contract. When None (default),
        the ContractAgent's policy is used.

    Examples
    --------
    Using with the @pre decorator::

        @tool
        @pre(lambda args: args['count'] > 0, "count must be positive")
        def fetch(count: int) -> str:
            return f"Fetched {count}"

    Multiple preconditions (all must pass)::

        @tool
        @pre(lambda args: args['start'] >= 0, "start must be non-negative")
        @pre(lambda args: args['end'] > args['start'], "end must be after start")
        def get_range(start: int, end: int) -> list:
            return list(range(start, end))
    """
    predicate: Callable[[Dict[str, Any]], bool]
    message: str
    predicate_str: str = ""
    policy: Optional[ContractPolicy] = None  # None means use default

    def check(self, args: Dict[str, Any]) -> bool:
        """
        Check if precondition holds for given arguments.

        Parameters
        ----------
        args : Dict[str, Any]
            Dictionary mapping argument names to their values.

        Returns
        -------
        bool
            True if precondition is satisfied, False otherwise.
            Returns False if the predicate raises an exception.
        """
        try:
            return bool(self.predicate(args))
        except Exception as e:
            logger.warning("Precondition check raised exception: %s", e)
            return False


@dataclass
class PostCondition:
    """
    A postcondition contract that validates tool results after execution.

    Postconditions express guarantees about what a tool will return. If a
    postcondition fails, it indicates the tool implementation has a bug.

    Postcondition predicates can take either one argument (the result) or
    two arguments (result and original args). The @post decorator automatically
    detects which form is used based on the predicate's signature.

    Attributes
    ----------
    predicate : Callable[..., bool]
        Function that receives the result (and optionally args) and returns
        True if the postcondition is satisfied.
    message : str
        Human-readable description of what the postcondition guarantees.
    predicate_str : str
        String representation of the predicate for error reporting.
    policy : Optional[ContractPolicy]
        Policy override for this specific contract. When None (default),
        the ContractAgent's policy is used.
    needs_args : bool
        True if the predicate requires access to the original arguments.
        Automatically set by the @post decorator based on signature.

    Examples
    --------
    Simple result validation::

        @tool
        @post(lambda r: len(r) > 0, "result must not be empty")
        def search(query: str) -> str:
            return f"Results for: {query}"

    Postcondition with access to original arguments::

        @tool
        @post(lambda r, args: len(r) <= args['max_len'], "result too long")
        def generate(prompt: str, max_len: int) -> str:
            return prompt[:max_len]
    """
    predicate: Callable[..., bool]  # (result) or (result, args)
    message: str
    predicate_str: str = ""
    policy: Optional[ContractPolicy] = None
    needs_args: bool = False  # Whether predicate takes (result, args)

    def check(self, result: Any, args: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if postcondition holds for given result.

        Parameters
        ----------
        result : Any
            The return value from the tool execution.
        args : Optional[Dict[str, Any]]
            Original arguments passed to the tool. Required if needs_args is True.

        Returns
        -------
        bool
            True if postcondition is satisfied, False otherwise.
            Returns False if the predicate raises an exception.
        """
        try:
            if self.needs_args and args is not None:
                return bool(self.predicate(result, args))
            else:
                return bool(self.predicate(result))
        except Exception as e:
            logger.warning("Postcondition check raised exception: %s", e)
            return False


@dataclass
class ContractSpec:
    """Collection of contracts for a tool or agent."""
    preconditions: List[PreCondition] = field(default_factory=list)
    postconditions: List[PostCondition] = field(default_factory=list)


# =============================================================================
# Contract Decorators
# =============================================================================

def _get_predicate_str(predicate: Callable) -> str:
    """Try to get a string representation of a predicate."""
    try:
        source = inspect.getsource(predicate)
        # Try to extract just the lambda body
        if "lambda" in source:
            # Find the lambda and extract its body
            idx = source.find("lambda")
            if idx >= 0:
                # Find the end (comma, closing paren, or newline)
                rest = source[idx:]
                # Simple heuristic: find balanced parens
                depth = 0
                end = len(rest)
                for i, c in enumerate(rest):
                    if c == '(':
                        depth += 1
                    elif c == ')':
                        if depth == 0:
                            end = i
                            break
                        depth -= 1
                    elif c == ',' and depth == 0:
                        end = i
                        break
                return rest[:end].strip()
        return str(predicate)
    except (OSError, TypeError):
        return str(predicate)


def pre(
    predicate: Callable[[Dict[str, Any]], bool],
    message: str = "",
    policy: Optional[ContractPolicy] = None
) -> Callable:
    """
    Decorator to add a precondition to a tool.

    The predicate receives a dictionary of arguments and should return True
    if the precondition is satisfied.

    Args:
        predicate: Function (args: Dict) -> bool
        message: Human-readable description of the precondition
        policy: Optional policy override for this specific contract

    Example:
        @tool
        @pre(lambda args: args['count'] > 0, "count must be positive")
        def fetch_items(count: int) -> str:
            ...
    """
    def decorator(func_or_tool: Union[Callable, Tool]) -> Union[Callable, Tool]:
        # Get or create contract spec
        if isinstance(func_or_tool, Tool):
            target = func_or_tool
            if not hasattr(target, '_contracts'):
                target._contracts = ContractSpec()
        else:
            target = func_or_tool
            if not hasattr(target, '_contracts'):
                target._contracts = ContractSpec()

        predicate_str = _get_predicate_str(predicate)
        condition = PreCondition(
            predicate=predicate,
            message=message or f"Precondition failed: {predicate_str}",
            predicate_str=predicate_str,
            policy=policy
        )
        target._contracts.preconditions.insert(0, condition)  # Insert at front for correct order

        return target

    return decorator


def post(
    predicate: Callable[..., bool],
    message: str = "",
    policy: Optional[ContractPolicy] = None
) -> Callable:
    """
    Decorator to add a postcondition to a tool.

    The predicate receives the result, and optionally the original arguments,
    and should return True if the postcondition is satisfied.

    Args:
        predicate: Function (result) -> bool or (result, args) -> bool
        message: Human-readable description of the postcondition
        policy: Optional policy override for this specific contract

    Example:
        @tool
        @post(lambda r: len(r) > 0, "result must not be empty")
        def search(query: str) -> str:
            ...

        @tool
        @post(lambda r, args: len(r) <= args['max_len'], "result too long")
        def generate(prompt: str, max_len: int) -> str:
            ...
    """
    def decorator(func_or_tool: Union[Callable, Tool]) -> Union[Callable, Tool]:
        if isinstance(func_or_tool, Tool):
            target = func_or_tool
            if not hasattr(target, '_contracts'):
                target._contracts = ContractSpec()
        else:
            target = func_or_tool
            if not hasattr(target, '_contracts'):
                target._contracts = ContractSpec()

        # Check if predicate takes 1 or 2 arguments
        sig = inspect.signature(predicate)
        needs_args = len(sig.parameters) >= 2

        predicate_str = _get_predicate_str(predicate)
        condition = PostCondition(
            predicate=predicate,
            message=message or f"Postcondition failed: {predicate_str}",
            predicate_str=predicate_str,
            policy=policy,
            needs_args=needs_args
        )
        target._contracts.postconditions.insert(0, condition)

        return target

    return decorator


# =============================================================================
# Contract Assert (Runtime Assertion)
# =============================================================================

# Thread-local storage for current contract context
import threading
_contract_context = threading.local()


def _get_current_context() -> Optional['ContractContext']:
    """Get the current contract context if any."""
    return getattr(_contract_context, 'current', None)


def _set_current_context(ctx: Optional['ContractContext']) -> None:
    """Set the current contract context."""
    _contract_context.current = ctx


@dataclass
class ContractContext:
    """Context for contract evaluation during tool execution."""
    policy: ContractPolicy
    handler: Optional[ViolationHandler]
    location: str = "unknown"

    def handle_violation(self, violation: ContractViolation) -> None:
        """Handle a contract violation according to policy."""
        if self.policy == ContractPolicy.IGNORE:
            return

        violation.policy = self.policy

        if self.policy == ContractPolicy.QUICK_ENFORCE:
            raise ContractTermination(violation)

        if self.handler:
            self.handler(violation)

        if self.policy == ContractPolicy.ENFORCE:
            raise ContractTermination(violation)
        # OBSERVE: continue after handler


def contract_assert(
    condition: bool,
    message: str = "Assertion failed",
    policy: Optional[ContractPolicy] = None
) -> None:
    """
    Runtime contract assertion for checking invariants within tool implementations.

    Similar to C++26's contract_assert, this function allows you to verify conditions
    that should always be true at a particular point in your code. Unlike @pre and @post
    which are checked automatically by ContractAgent, contract_assert() must be called
    explicitly in your tool code.

    When called within a tool that's being executed by ContractAgent, the assertion
    participates in the same violation handling system as @pre and @post contracts.
    The agent's policy and violation handler are used automatically.

    When called outside of an agent context (e.g., in unit tests or standalone code),
    the assertion uses ENFORCE as the default policy, raising ContractTermination on
    failure.

    Parameters
    ----------
    condition : bool
        The condition that must be True. If False, a violation is triggered.
    message : str
        Human-readable description of what was expected. Defaults to "Assertion failed".
    policy : Optional[ContractPolicy]
        Policy override for this specific assertion. When None (default), uses the
        current ContractContext's policy if available, otherwise ENFORCE.

    Raises
    ------
    ContractTermination
        If condition is False and the effective policy is ENFORCE or QUICK_ENFORCE.

    Examples
    --------
    Basic invariant checking::

        @tool
        def process_data(data: str) -> dict:
            import json
            parsed = json.loads(data)
            contract_assert(isinstance(parsed, dict), "data must be JSON object")
            return parsed

    Multiple assertions in sequence::

        @tool
        def calculate(values: list) -> float:
            contract_assert(len(values) > 0, "values must not be empty")
            result = sum(values) / len(values)
            contract_assert(result >= 0, "average must be non-negative")
            return result

    Using with explicit policy::

        @tool
        def risky_operation(data: str) -> str:
            # This assertion only logs, doesn't terminate
            contract_assert(
                len(data) < 10000,
                "data unusually large",
                policy=ContractPolicy.OBSERVE
            )
            return process(data)

    Notes
    -----
    Unlike Python's built-in assert statement:

    - contract_assert is never removed by optimization (-O flag)
    - contract_assert participates in the agent's violation handling system
    - contract_assert can be configured to log-and-continue via OBSERVE policy
    - contract_assert provides structured error information via ContractViolation
    """
    if condition:
        return

    ctx = _get_current_context()
    effective_policy = policy or (ctx.policy if ctx else ContractPolicy.ENFORCE)

    if effective_policy == ContractPolicy.IGNORE:
        return

    violation = ContractViolation(
        kind="assert",
        location=ctx.location if ctx else "unknown",
        predicate=message,
        message=message,
        policy=effective_policy
    )

    if ctx:
        ctx.handle_violation(violation)
    else:
        # No context - use default behavior
        if effective_policy in (ContractPolicy.ENFORCE, ContractPolicy.QUICK_ENFORCE):
            raise ContractTermination(violation)
        else:
            logger.warning("Contract assertion failed: %s", message)


# =============================================================================
# Contract Agent
# =============================================================================

class ContractAgent:
    """
    Agent wrapper that adds C++26-inspired contract checking to tool execution.

    ContractAgent wraps an inner agent (ReActAgent or ConstrainedAgent) and
    intercepts tool calls to verify contracts at multiple checkpoints:

    1. **Task precondition** - Validated once before agent execution begins
    2. **Tool preconditions** - Validated before each tool call
    3. **Tool postconditions** - Validated after each tool returns
    4. **Answer postcondition** - Validated once before returning final answer
    5. **Iteration invariant** - Validated at each reasoning iteration

    Contracts can be defined in two ways:

    - **Decorator-based**: Use @pre and @post on tool functions
    - **Agent-level**: Pass callables to task_precondition, answer_postcondition, etc.

    Default Policy Behavior
    -----------------------
    The default policy is ContractPolicy.ENFORCE. This means:

    - All contracts are checked at runtime
    - On violation, the violation_handler is called
    - After the handler, ContractTermination is raised and execution stops
    - The agent's run() method returns with success=False and error set

    When a tool's @pre/@post contract has policy=None (the default), the agent's
    policy is used. This allows you to set a global policy while optionally
    overriding it for specific contracts.

    Examples
    --------
    Basic usage with tool contracts::

        from cyllama.agents import tool, pre, post, ContractAgent, ContractPolicy

        @tool
        @pre(lambda args: args['query'], "query must not be empty")
        @post(lambda r: r is not None, "must return a result")
        def search(query: str) -> str:
            return f"Results for: {query}"

        agent = ContractAgent(
            llm=my_llm,
            tools=[search],
            policy=ContractPolicy.ENFORCE
        )
        result = agent.run("Find information about Python")

    Agent-level contracts::

        agent = ContractAgent(
            llm=my_llm,
            tools=[search],
            task_precondition=lambda task: len(task) >= 10,
            answer_postcondition=lambda ans: len(ans) > 0,
            iteration_invariant=lambda state: state.iterations < 5
        )

    Custom violation handler::

        def my_handler(violation):
            print(f"Contract violated: {violation}")
            # Log to monitoring system, send alert, etc.

        agent = ContractAgent(
            llm=my_llm,
            tools=[search],
            violation_handler=my_handler
        )

    Observe-only mode for gradual adoption::

        agent = ContractAgent(
            llm=my_llm,
            tools=[search],
            policy=ContractPolicy.OBSERVE  # Log violations but don't stop
        )

    See Also
    --------
    ContractPolicy : Enum defining violation handling policies
    pre : Decorator to add preconditions to tools
    post : Decorator to add postconditions to tools
    contract_assert : Runtime assertion function for use in tool code
    """

    def __init__(
        self,
        llm,
        tools: Optional[List[Tool]] = None,
        policy: ContractPolicy = ContractPolicy.ENFORCE,
        violation_handler: Optional[ViolationHandler] = None,
        task_precondition: Optional[Callable[[str], bool]] = None,
        answer_postcondition: Optional[Callable[[str], bool]] = None,
        iteration_invariant: Optional[Callable[['IterationState'], bool]] = None,
        inner_agent: Optional[Union[ReActAgent, 'ConstrainedAgent']] = None,
        agent_type: str = "react",
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = False,
        **agent_kwargs
    ):
        """
        Initialize ContractAgent with contract checking capabilities.

        Parameters
        ----------
        llm : callable
            Language model instance that takes a prompt string and returns a response.
        tools : Optional[List[Tool]]
            List of tools available to the agent. Tools may have @pre/@post contracts
            attached via decorators. Contracts are automatically extracted and checked.
        policy : ContractPolicy
            Default policy for all contracts. Individual contracts can override this.
            **Default: ContractPolicy.ENFORCE** - violations call handler then terminate.
        violation_handler : Optional[ViolationHandler]
            Callback invoked when a contract is violated (except with IGNORE or
            QUICK_ENFORCE policies). Receives a ContractViolation with full context.
            If None, uses a default handler that logs to the 'cyllama.agents.contract'
            logger at WARNING level.
        task_precondition : Optional[Callable[[str], bool]]
            Validates the input task before execution begins. Receives the task string,
            returns True if valid. Checked once at the start of run()/stream().
        answer_postcondition : Optional[Callable[[str], bool]]
            Validates the final answer before returning. Receives the answer string,
            returns True if valid. Checked once after agent produces an answer.
        iteration_invariant : Optional[Callable[[IterationState], bool]]
            Checked at each reasoning iteration. Receives an IterationState with
            counters for iterations, tool_calls, and errors. Return False to
            trigger a violation (e.g., to limit total iterations).
        inner_agent : Optional[Union[ReActAgent, ConstrainedAgent]]
            Pre-configured inner agent. If provided, agent_type is ignored.
        agent_type : str
            Type of inner agent to create: "react" (default) or "constrained".
        system_prompt : Optional[str]
            Custom system prompt for the inner agent.
        max_iterations : int
            Maximum iterations for the inner agent. Default: 10.
        verbose : bool
            Enable verbose output including contract check details. Default: False.
        **agent_kwargs
            Additional keyword arguments passed to the inner agent constructor.

        Notes
        -----
        The policy parameter sets the **default** policy used when a contract's own
        policy is None. You can override the policy for specific contracts::

            @pre(lambda args: ..., "msg", policy=ContractPolicy.OBSERVE)

        The violation_handler is called for OBSERVE and ENFORCE policies, but NOT for:
        - IGNORE: No checking occurs
        - QUICK_ENFORCE: Terminates immediately without handler
        """
        self.llm = llm
        self.tools = tools or []
        self.policy = policy
        self.violation_handler = violation_handler or self._default_handler
        self.task_precondition = task_precondition
        self.answer_postcondition = answer_postcondition
        self.iteration_invariant = iteration_invariant
        self.verbose = verbose

        # Extract contracts from tools
        self._tool_contracts: Dict[str, ContractSpec] = {}
        for tool in self.tools:
            if hasattr(tool, '_contracts'):
                self._tool_contracts[tool.name] = tool._contracts
            elif hasattr(tool.func, '_contracts'):
                self._tool_contracts[tool.name] = tool.func._contracts

        # Create or use inner agent
        if inner_agent is not None:
            self._inner_agent = inner_agent
        else:
            if agent_type == "constrained":
                from .constrained import ConstrainedAgent
                self._inner_agent = ConstrainedAgent(
                    llm=llm,
                    tools=tools,
                    system_prompt=system_prompt,
                    max_iterations=max_iterations,
                    verbose=verbose,
                    **agent_kwargs
                )
            else:
                self._inner_agent = ReActAgent(
                    llm=llm,
                    tools=tools,
                    system_prompt=system_prompt,
                    max_iterations=max_iterations,
                    verbose=verbose,
                    **agent_kwargs
                )

        # Metrics
        self._metrics: Optional[AgentMetrics] = None
        self._contract_checks = 0
        self._contract_violations = 0

    def _default_handler(self, violation: ContractViolation) -> None:
        """Default violation handler - logs the violation."""
        logger.warning(
            "Contract violation [%s] at %s: %s",
            violation.kind, violation.location, violation.message
        )
        if self.verbose:
            print(f"CONTRACT VIOLATION [{violation.kind}] at {violation.location}")
            print(f"  Predicate: {violation.predicate}")
            print(f"  Message: {violation.message}")
            if violation.context:
                print(f"  Context: {violation.context}")

    def _get_effective_policy(
        self,
        contract_policy: Optional[ContractPolicy]
    ) -> ContractPolicy:
        """Get effective policy, using contract-specific or default."""
        return contract_policy if contract_policy is not None else self.policy

    def _check_preconditions(
        self,
        tool_name: str,
        args: Dict[str, Any]
    ) -> Optional[ContractViolation]:
        """Check all preconditions for a tool call."""
        contracts = self._tool_contracts.get(tool_name)
        if not contracts:
            return None

        for pre_cond in contracts.preconditions:
            effective_policy = self._get_effective_policy(pre_cond.policy)
            self._contract_checks += 1

            if effective_policy == ContractPolicy.IGNORE:
                continue

            if not pre_cond.check(args):
                self._contract_violations += 1
                return ContractViolation(
                    kind="pre",
                    location=tool_name,
                    predicate=pre_cond.predicate_str,
                    message=pre_cond.message,
                    context={"args": args},
                    policy=effective_policy
                )

        return None

    def _check_postconditions(
        self,
        tool_name: str,
        result: Any,
        args: Dict[str, Any]
    ) -> Optional[ContractViolation]:
        """Check all postconditions for a tool result."""
        contracts = self._tool_contracts.get(tool_name)
        if not contracts:
            return None

        for post_cond in contracts.postconditions:
            effective_policy = self._get_effective_policy(post_cond.policy)
            self._contract_checks += 1

            if effective_policy == ContractPolicy.IGNORE:
                continue

            if not post_cond.check(result, args):
                self._contract_violations += 1
                return ContractViolation(
                    kind="post",
                    location=tool_name,
                    predicate=post_cond.predicate_str,
                    message=post_cond.message,
                    context={"result": result, "args": args},
                    policy=effective_policy
                )

        return None

    def _handle_violation(self, violation: ContractViolation) -> bool:
        """
        Handle a contract violation.

        Returns:
            True if execution should continue, False if it should stop
        """
        if violation.policy == ContractPolicy.IGNORE:
            return True

        if violation.policy == ContractPolicy.QUICK_ENFORCE:
            return False

        # Call handler for OBSERVE and ENFORCE
        self.violation_handler(violation)

        if violation.policy == ContractPolicy.ENFORCE:
            return False

        # OBSERVE: continue
        return True

    def stream(self, task: str) -> Generator[AgentEvent, None, None]:
        """
        Stream agent execution with contract checking.

        Yields:
            AgentEvent instances including CONTRACT_CHECK and CONTRACT_VIOLATION
        """
        start_time = time.perf_counter()
        self._contract_checks = 0
        self._contract_violations = 0

        # Check task precondition
        if self.task_precondition is not None:
            self._contract_checks += 1
            if self.policy != ContractPolicy.IGNORE:
                yield AgentEvent(
                    type=EventType.CONTRACT_CHECK,
                    content="Checking task precondition",
                    metadata={"kind": "pre", "location": "agent"}
                )

                if not self.task_precondition(task):
                    self._contract_violations += 1
                    violation = ContractViolation(
                        kind="pre",
                        location="agent",
                        predicate="task_precondition",
                        message="Task precondition failed",
                        context={"task": task},
                        policy=self.policy
                    )
                    yield AgentEvent(
                        type=EventType.CONTRACT_VIOLATION,
                        content=str(violation),
                        metadata={"violation": violation}
                    )
                    if not self._handle_violation(violation):
                        yield AgentEvent(
                            type=EventType.ERROR,
                            content=f"Contract terminated: {violation.message}"
                        )
                        return

        # Track iteration state
        iteration_state = IterationState()

        # Stream from inner agent, intercepting tool calls
        answer = None
        for event in self._inner_agent.stream(task):
            iteration_state.update(event)

            # Check iteration invariant
            if (event.type == EventType.THOUGHT and
                self.iteration_invariant is not None and
                self.policy != ContractPolicy.IGNORE):

                self._contract_checks += 1
                if not self.iteration_invariant(iteration_state):
                    self._contract_violations += 1
                    violation = ContractViolation(
                        kind="assert",
                        location="agent",
                        predicate="iteration_invariant",
                        message="Iteration invariant failed",
                        context={"state": iteration_state.to_dict()},
                        policy=self.policy
                    )
                    yield AgentEvent(
                        type=EventType.CONTRACT_VIOLATION,
                        content=str(violation),
                        metadata={"violation": violation}
                    )
                    if not self._handle_violation(violation):
                        yield AgentEvent(
                            type=EventType.ERROR,
                            content=f"Contract terminated: {violation.message}"
                        )
                        return

            # Intercept tool calls to check contracts
            if event.type == EventType.ACTION:
                tool_name = event.metadata.get("tool_name", "")
                tool_args = event.metadata.get("tool_args", {})

                # Check preconditions
                violation = self._check_preconditions(tool_name, tool_args)
                if violation:
                    yield AgentEvent(
                        type=EventType.CONTRACT_VIOLATION,
                        content=str(violation),
                        metadata={"violation": violation}
                    )
                    if not self._handle_violation(violation):
                        yield AgentEvent(
                            type=EventType.ERROR,
                            content=f"Contract terminated: {violation.message}"
                        )
                        return

                # Set up context for contract_assert calls within tool
                ctx = ContractContext(
                    policy=self.policy,
                    handler=self.violation_handler,
                    location=tool_name
                )
                _set_current_context(ctx)

            # After observation, check postconditions
            if event.type == EventType.OBSERVATION:
                _set_current_context(None)  # Clear context

                tool_name = event.metadata.get("tool_name", "")
                tool_args = event.metadata.get("tool_args", {})

                # Use raw_result if available (actual return value), otherwise fall back to content
                result = event.metadata.get("raw_result")
                if result is None:
                    result = event.content

                violation = self._check_postconditions(tool_name, result, tool_args)
                if violation:
                    yield AgentEvent(
                        type=EventType.CONTRACT_VIOLATION,
                        content=str(violation),
                        metadata={"violation": violation}
                    )
                    if not self._handle_violation(violation):
                        yield AgentEvent(
                            type=EventType.ERROR,
                            content=f"Contract terminated: {violation.message}"
                        )
                        return

            # Track answer for postcondition check
            if event.type == EventType.ANSWER:
                answer = event.content

            yield event

        # Check answer postcondition
        if (answer is not None and
            self.answer_postcondition is not None and
            self.policy != ContractPolicy.IGNORE):

            self._contract_checks += 1
            yield AgentEvent(
                type=EventType.CONTRACT_CHECK,
                content="Checking answer postcondition",
                metadata={"kind": "post", "location": "agent"}
            )

            if not self.answer_postcondition(answer):
                self._contract_violations += 1
                violation = ContractViolation(
                    kind="post",
                    location="agent",
                    predicate="answer_postcondition",
                    message="Answer postcondition failed",
                    context={"answer": answer},
                    policy=self.policy
                )
                yield AgentEvent(
                    type=EventType.CONTRACT_VIOLATION,
                    content=str(violation),
                    metadata={"violation": violation}
                )
                if not self._handle_violation(violation):
                    yield AgentEvent(
                        type=EventType.ERROR,
                        content=f"Contract terminated: {violation.message}"
                    )

    def run(self, task: str) -> AgentResult:
        """
        Run agent with contract checking.

        Args:
            task: Task description or question

        Returns:
            AgentResult with execution trace including contract events
        """
        events = []
        answer = None
        error = None

        try:
            for event in self.stream(task):
                events.append(event)
                if event.type == EventType.ANSWER:
                    answer = event.content
                elif event.type == EventType.ERROR:
                    error = event.content

        except ContractTermination as e:
            events.append(AgentEvent(
                type=EventType.CONTRACT_VIOLATION,
                content=str(e.violation),
                metadata={"violation": e.violation}
            ))
            events.append(AgentEvent(
                type=EventType.ERROR,
                content=f"Contract terminated: {e.violation.message}"
            ))
            error = str(e.violation)

        # Count iterations from events
        iterations = sum(1 for e in events if e.type == EventType.THOUGHT)

        return AgentResult(
            answer=answer or "",
            steps=events,
            iterations=iterations,
            success=answer is not None and error is None,
            error=error,
            metrics=self._inner_agent._metrics if hasattr(self._inner_agent, '_metrics') else None
        )

    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return self.tools

    def get_contract_stats(self) -> Dict[str, int]:
        """Get contract checking statistics."""
        return {
            "checks": self._contract_checks,
            "violations": self._contract_violations
        }


# =============================================================================
# Iteration State
# =============================================================================

@dataclass
class IterationState:
    """State tracked during agent iteration for invariant checking."""
    iterations: int = 0
    tool_calls: int = 0
    errors: int = 0
    events: List[AgentEvent] = field(default_factory=list)

    def update(self, event: AgentEvent) -> None:
        """Update state based on event."""
        self.events.append(event)
        if event.type == EventType.THOUGHT:
            self.iterations += 1
        elif event.type == EventType.ACTION:
            self.tool_calls += 1
        elif event.type == EventType.ERROR:
            self.errors += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for context."""
        return {
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "errors": self.errors
        }
