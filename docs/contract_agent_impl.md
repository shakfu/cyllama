# ContractAgent Implementation

Technical documentation for the ContractAgent implementation in cyllama.

## Overview

`ContractAgent` is a contract-based agent whose design is inspired by C++26's contracts. It adds preconditions, postconditions, and runtime assertions to the cyllama agent framework. It wraps an inner agent (`ReActAgent` or `ConstrainedAgent`) and intercepts tool calls to verify contracts.

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                      ContractAgent                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Contract Registry                                    │  │
│  │  _tool_contracts: Dict[str, ContractSpec]             │  │
│  │  - Maps tool names to their contract specifications   │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                 │
│  ┌────────────────────────▼──────────────────────────────┐  │
│  │  Contract Evaluator                                   │  │
│  │  - _check_preconditions(tool_name, args)              │  │
│  │  - _check_postconditions(tool_name, result, args)     │  │
│  │  - _handle_violation(violation)                       │  │
│  └────────────────────────┬──────────────────────────────┘  │
│                           │                                 │
│  ┌────────────────────────▼──────────────────────────────┐  │
│  │  Inner Agent (ReActAgent or ConstrainedAgent)         │  │
│  │  - Handles LLM interaction and tool execution         │  │
│  │  - ContractAgent intercepts events via stream()       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

```text
src/cyllama/agents/
├── __init__.py      # Exports ContractAgent and related types
├── contract.py      # ContractAgent implementation
├── react.py         # ReActAgent (inner agent option)
├── constrained.py   # ConstrainedAgent (inner agent option)
├── tools.py         # Tool decorator and registry
└── grammar.py       # Grammar generation for ConstrainedAgent
```

## Core Components

### ContractPolicy Enum

Defines how contract violations are handled:

```python
class ContractPolicy(Enum):
    IGNORE = "ignore"               # Skip checking entirely
    OBSERVE = "observe"             # Check, log violation, continue
    ENFORCE = "enforce"             # Check, call handler, terminate
    QUICK_ENFORCE = "quick_enforce" # Check, terminate immediately
```

| Policy | Checks | Handler Called | Continues | Terminates |
|--------|--------|----------------|-----------|------------|
| IGNORE | No | No | Yes | No |
| OBSERVE | Yes | Yes (on fail) | Yes | No |
| ENFORCE | Yes | Yes (on fail) | No | Yes |
| QUICK_ENFORCE | Yes | No | No | Yes |

### ContractViolation Dataclass

Represents a contract violation event:

```python
@dataclass
class ContractViolation:
    kind: str                    # "pre", "post", "assert"
    location: str                # Tool name or "agent"
    predicate: str               # String representation of condition
    message: str                 # Human-readable description
    context: Dict[str, Any]      # Arguments, result, etc.
    policy: ContractPolicy       # Policy when violation occurred
```

### PreCondition and PostCondition

```python
@dataclass
class PreCondition:
    predicate: Callable[[Dict[str, Any]], bool]  # (args) -> bool
    message: str
    predicate_str: str = ""
    policy: Optional[ContractPolicy] = None  # Override default

@dataclass
class PostCondition:
    predicate: Callable[..., bool]  # (result) or (result, args) -> bool
    message: str
    predicate_str: str = ""
    policy: Optional[ContractPolicy] = None
    needs_args: bool = False  # True if predicate takes (result, args)
```

### ContractSpec

Container for a tool's contracts:

```python
@dataclass
class ContractSpec:
    preconditions: List[PreCondition]
    postconditions: List[PostCondition]
```

## Contract Decorators

### @pre Decorator

Attaches a precondition to a tool:

```python
@tool
@pre(lambda args: args['count'] > 0, "count must be positive")
@pre(lambda args: args['count'] <= 100, "count must not exceed 100")
def fetch_items(count: int) -> str:
    return f"Fetched {count} items"
```

Implementation details:

- Decorators are applied bottom-up, so `@pre` runs before `@tool`
- Contracts are stored on the function's `_contracts` attribute
- Multiple preconditions are stored in order and checked sequentially
- The `@tool` decorator preserves the `_contracts` attribute

### @post Decorator

Attaches a postcondition to a tool:

```python
@tool
@post(lambda r: len(r) > 0, "result must not be empty")
@post(lambda r, args: len(r) <= args['max_len'], "result too long")
def search(query: str, max_len: int = 100) -> str:
    return f"Results for: {query}"
```

Implementation details:

- Automatically detects if predicate takes 1 or 2 arguments
- Sets `needs_args=True` if predicate signature has 2+ parameters
- When `needs_args=True`, passes both result and original args to predicate

### Policy Override

Individual contracts can override the agent's default policy:

```python
@tool
@pre(lambda args: args['n'] > 0, "positive", policy=ContractPolicy.OBSERVE)
def my_func(n: int) -> int:
    return n * 2
```

## Runtime Assertions

### contract_assert Function

For runtime invariant checking within tool implementations:

```python
def contract_assert(
    condition: bool,
    message: str = "Assertion failed",
    policy: Optional[ContractPolicy] = None
) -> None:
```

Usage:

```python
@tool
def process_data(data: str) -> str:
    parsed = json.loads(data)
    contract_assert(isinstance(parsed, dict), "data must be JSON object")

    result = transform(parsed)
    contract_assert(result is not None, "transform must not return None")

    return str(result)
```

### ContractContext

Thread-local context for `contract_assert` calls:

```python
@dataclass
class ContractContext:
    policy: ContractPolicy
    handler: Optional[ViolationHandler]
    location: str = "unknown"
```

The ContractAgent sets this context before tool execution and clears it after, allowing `contract_assert` calls within tools to participate in the agent's violation handling.

## ContractAgent Class

### Initialization

```python
agent = ContractAgent(
    llm=llm,
    tools=[tool1, tool2],
    policy=ContractPolicy.ENFORCE,      # Default policy
    violation_handler=my_handler,        # Custom handler
    task_precondition=lambda t: len(t) > 10,
    answer_postcondition=lambda a: len(a) > 0,
    iteration_invariant=lambda s: s.iterations < 20,
    agent_type="react",  # or "constrained"
    max_iterations=10,
    verbose=False,
)
```

### Contract Extraction

During initialization, ContractAgent extracts contracts from tools:

```python
self._tool_contracts: Dict[str, ContractSpec] = {}
for tool in self.tools:
    if hasattr(tool, '_contracts'):
        self._tool_contracts[tool.name] = tool._contracts
    elif hasattr(tool.func, '_contracts'):
        self._tool_contracts[tool.name] = tool.func._contracts
```

### Execution Flow

```text
run(task)
  │
  ├─ stream(task) [generator]
  │   │
  │   ├─ CHECK task_precondition(task)
  │   │   └─ On violation: handle according to policy
  │   │
  │   ├─ For each event from inner agent:
  │   │   │
  │   │   ├─ On THOUGHT event:
  │   │   │   └─ CHECK iteration_invariant(state)
  │   │   │
  │   │   ├─ On ACTION event:
  │   │   │   ├─ CHECK tool @pre conditions(args)
  │   │   │   └─ Set ContractContext for contract_assert
  │   │   │
  │   │   ├─ On OBSERVATION event:
  │   │   │   ├─ Clear ContractContext
  │   │   │   ├─ Extract raw_result from metadata (actual typed return value)
  │   │   │   └─ CHECK tool @post conditions(raw_result, args)
  │   │   │
  │   │   └─ On ANSWER event:
  │   │       └─ Store answer for postcondition check
  │   │
  │   └─ CHECK answer_postcondition(answer)
  │
  └─ Return AgentResult with all events
```

### Raw Result Handling

Postconditions need access to the actual typed return value, not the stringified observation. The inner agent stores `raw_result` in OBSERVATION event metadata:

```python
# In ReActAgent._execute_tool_raw()
raw_result = tool(**args)  # Actual return value (e.g., 25.0 as float)
observation = str(raw_result)  # String for LLM prompt (e.g., "25.0")

# OBSERVATION event includes both
obs_event = AgentEvent(
    type=EventType.OBSERVATION,
    content=observation,  # String for display/LLM
    metadata={
        "tool_name": tool_name,
        "tool_args": tool_args,
        "raw_result": raw_result  # Typed value for contracts
    }
)
```

ContractAgent extracts `raw_result` for postcondition checking:

```python
# Use raw_result if available, fall back to content string
result = event.metadata.get("raw_result")
if result is None:
    result = event.content
violation = self._check_postconditions(tool_name, result, tool_args)
```

This ensures postconditions like `@post(lambda r: isinstance(r, float))` receive the actual float, not the string `"25.0"`.

### Contract Checking Methods

```python
def _check_preconditions(
    self,
    tool_name: str,
    args: Dict[str, Any]
) -> Optional[ContractViolation]:
    """Check all preconditions for a tool call."""

def _check_postconditions(
    self,
    tool_name: str,
    result: Any,
    args: Dict[str, Any]
) -> Optional[ContractViolation]:
    """Check all postconditions for a tool result."""

def _handle_violation(
    self,
    violation: ContractViolation
) -> bool:
    """Handle violation. Returns True to continue, False to stop."""
```

### Violation Handling

```python
def _handle_violation(self, violation: ContractViolation) -> bool:
    if violation.policy == ContractPolicy.IGNORE:
        return True  # Continue

    if violation.policy == ContractPolicy.QUICK_ENFORCE:
        return False  # Stop immediately, no handler

    # Call handler for OBSERVE and ENFORCE
    self.violation_handler(violation)

    if violation.policy == ContractPolicy.ENFORCE:
        return False  # Stop after handler

    return True  # OBSERVE: continue
```

## Event Types

ContractAgent adds two new event types:

```python
class EventType(Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    ANSWER = "answer"
    ERROR = "error"
    CONTRACT_CHECK = "contract_check"        # Contract being evaluated
    CONTRACT_VIOLATION = "contract_violation" # Violation detected
```

## IterationState

Tracks agent execution state for invariant checking:

```python
@dataclass
class IterationState:
    iterations: int = 0      # Number of THOUGHT events
    tool_calls: int = 0      # Number of ACTION events
    errors: int = 0          # Number of ERROR events
    events: List[AgentEvent] = field(default_factory=list)

    def update(self, event: AgentEvent) -> None:
        """Update state based on event."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for context."""
```

## Exception Handling

### ContractTermination

Raised when a contract violation should terminate execution:

```python
class ContractTermination(Exception):
    def __init__(self, violation: ContractViolation):
        self.violation = violation
        super().__init__(str(violation))
```

This exception can be raised by:

- `contract_assert()` with ENFORCE or QUICK_ENFORCE policy
- `ContractContext.handle_violation()` with ENFORCE or QUICK_ENFORCE policy

The `ContractAgent.run()` method catches this exception and converts it to an error event.

## Thread Safety

Contract context is stored in thread-local storage:

```python
import threading
_contract_context = threading.local()

def _get_current_context() -> Optional[ContractContext]:
    return getattr(_contract_context, 'current', None)

def _set_current_context(ctx: Optional[ContractContext]) -> None:
    _contract_context.current = ctx
```

This ensures that `contract_assert` calls in concurrent tool executions use the correct context.

## Statistics

ContractAgent tracks contract checking statistics:

```python
def get_contract_stats(self) -> Dict[str, int]:
    return {
        "checks": self._contract_checks,
        "violations": self._contract_violations
    }
```

## Usage Examples

### Basic Usage

```python
from cyllama.agents import ContractAgent, tool, pre, post, ContractPolicy

@tool
@pre(lambda args: args['x'] != 0, "cannot divide by zero")
@post(lambda r: r is not None, "result must not be None")
def divide(a: float, x: float) -> float:
    return a / x

agent = ContractAgent(
    llm=llm,
    tools=[divide],
    policy=ContractPolicy.ENFORCE,
)

result = agent.run("What is 100 divided by 4?")
```

### Custom Violation Handler

```python
def my_handler(violation: ContractViolation) -> None:
    print(f"VIOLATION: {violation.kind} at {violation.location}")
    print(f"  Message: {violation.message}")
    # Log to file, send alert, etc.

agent = ContractAgent(
    llm=llm,
    tools=[...],
    violation_handler=my_handler,
)
```

### Agent-Level Contracts

```python
agent = ContractAgent(
    llm=llm,
    tools=[...],
    task_precondition=lambda task: len(task) >= 10,
    answer_postcondition=lambda ans: "error" not in ans.lower(),
    iteration_invariant=lambda state: state.iterations < 20,
)
```

### Using contract_assert

```python
from cyllama.agents import tool, contract_assert

@tool
def complex_operation(data: str) -> str:
    parsed = json.loads(data)
    contract_assert(isinstance(parsed, dict), "must be JSON object")

    if "required_field" not in parsed:
        contract_assert(False, "missing required_field")

    return process(parsed)
```

## Testing

Tests are located in `tests/test_agents_contract.py` and cover:

- Contract decorator behavior
- `PreCondition` and `PostCondition` classes
- `ContractViolation` creation and string representation
- `contract_assert` function
- `ContractContext` violation handling
- `IterationState` tracking
- `ContractAgent` initialization
- Contract checking methods
- Violation handling with different policies
- Edge cases and error handling

Run tests with:

```bash
pytest tests/test_agents_contract.py -v
```

## Files

| File | Description |
|------|-------------|
| `src/cyllama/agents/contract.py` | Main implementation |
| `src/cyllama/agents/__init__.py` | Public exports |
| `src/cyllama/agents/react.py` | EventType definitions |
| `tests/test_agents_contract.py` | Unit tests |
| `tests/examples/agent_contract_example.py` | Usage examples |
| `CONTRACT_AGENT.md` | Design document |
| `CONTRACT_AGENT_IMPL.md` | This file |

## References

- [C++26 Contract Assertions - cppreference](https://en.cppreference.com/w/cpp/language/contracts.html)
- [P2900R14 - Contracts for C++](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2900r14.pdf)
