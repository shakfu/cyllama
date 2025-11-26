# ContractAgent Design

Contract-based agent inspired by C++26's contract specification (P2900). Now implemented in cyllama.

## Core Concepts Mapping

| C++26 Contract | ContractAgent Equivalent |
|----------------|--------------------------|
| `pre` | Preconditions on tool calls, agent inputs |
| `post` | Postconditions on tool results, agent outputs |
| `contract_assert` | Runtime invariants during execution |
| Evaluation policies (`ignore`, `observe`, `enforce`, `quick_enforce`) | Configurable violation handling modes |
| `handle_contract_violation` | User-defined violation callback |
| Contract termination | Agent abort/terminate behavior |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  ContractAgent                      │
│  ┌───────────────────────────────────────────────┐  │
│  │  Contract Registry                            │  │
│  │  - Preconditions (tool inputs, task inputs)   │  │
│  │  - Postconditions (tool outputs, final answer)│  │
│  │  - Invariants (during execution)              │  │
│  └───────────────────────────────────────────────┘  │
│                        │                            │
│  ┌─────────────────────▼─────────────────────────┐  │
│  │  Contract Evaluator                           │  │
│  │  - Policy: ignore | observe | enforce |       │  │
│  │             quick_enforce                     │  │
│  │  - Violation handler callback                 │  │
│  └─────────────────────┬─────────────────────────┘  │
│                        │                            │
│  ┌─────────────────────▼─────────────────────────┐  │
│  │  Inner Agent (ReActAgent or ConstrainedAgent) │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Composition over Inheritance

`ContractAgent` wraps an inner agent (ReAct or Constrained), adding contract checking as a cross-cutting concern. This maintains separation of concerns.

### 2. Decorator-based Contract Definition

Similar to the `@tool` decorator, provide `@pre`, `@post`, and `contract_assert()` for defining contracts:

```python
from cyllama.agents import ContractAgent, tool, pre, post

@tool
@pre(lambda args: args['count'] > 0, "count must be positive")
@post(lambda result: len(result) > 0, "must return non-empty result")
def fetch_items(count: int) -> str:
    """Fetch items from database."""
    return f"Fetched {count} items"
```

### 3. Evaluation Semantics as Enum

```python
class ContractPolicy(Enum):
    IGNORE = "ignore"           # Skip checking entirely
    OBSERVE = "observe"         # Check, log violation, continue
    ENFORCE = "enforce"         # Check, handle violation, terminate
    QUICK_ENFORCE = "quick_enforce"  # Check, terminate immediately (no handler)
```

### 4. Contract Violation as Event

```python
@dataclass
class ContractViolation:
    kind: str                    # "pre", "post", "assert"
    location: str                # Tool name or "agent"
    predicate: str               # String representation of condition
    message: str                 # Human-readable description
    context: Dict[str, Any]      # Arguments, result, etc.
    policy: ContractPolicy       # How it was evaluated
```

### 5. Violation Handler Protocol

```python
class ViolationHandler(Protocol):
    def __call__(self, violation: ContractViolation) -> None:
        """Handle a contract violation. May raise to abort."""
        ...
```

## API

```python
from cyllama.agents import (
    ContractAgent, ContractPolicy, ContractViolation,
    tool, pre, post, contract_assert
)

# Define tools with contracts
@tool
@pre(lambda args: args['query'], "query must not be empty")
@post(lambda r: "error" not in r.lower(), "result must not contain error")
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
@pre(lambda args: args['x'] != 0, "cannot divide by zero")
def divide(a: float, x: float) -> float:
    """Divide a by x."""
    return a / x

# Custom violation handler
def my_handler(violation: ContractViolation):
    print(f"CONTRACT VIOLATION: {violation.kind} at {violation.location}")
    print(f"  Condition: {violation.predicate}")
    print(f"  Message: {violation.message}")
    # Could log to file, send alert, etc.

# Create agent with contracts
agent = ContractAgent(
    llm=llm,
    tools=[search, divide],
    policy=ContractPolicy.ENFORCE,  # Default for all contracts
    violation_handler=my_handler,

    # Agent-level contracts
    task_precondition=lambda task: len(task) > 10,
    answer_postcondition=lambda answer: len(answer) > 0,
)

# Run with contract checking
result = agent.run("Search for Python tutorials")
```

## Contract Types in Detail

### 1. Tool Preconditions (`@pre`)

Checked before tool execution. Has access to tool arguments. Multiple preconditions evaluated in order.

```python
@tool
@pre(lambda args: args['n'] >= 0, "n must be non-negative")
@pre(lambda args: args['n'] <= 100, "n must not exceed 100")
def fibonacci(n: int) -> int:
    ...
```

### 2. Tool Postconditions (`@post`)

Checked after tool execution. Has access to result (and optionally original args). Can validate output format, ranges, invariants.

```python
@tool
@post(lambda r: isinstance(r, str), "result must be string")
@post(lambda r, args: len(r) <= args['max_len'], "result exceeds max length")
def generate_text(prompt: str, max_len: int = 100) -> str:
    ...
```

### 3. Agent-Level Contracts

- `task_precondition`: Validates input task before execution
- `answer_postcondition`: Validates final answer before returning
- `iteration_invariant`: Checked at each iteration

```python
agent = ContractAgent(
    llm=llm,
    tools=[...],
    task_precondition=lambda task: not contains_pii(task),
    answer_postcondition=lambda ans: is_factual(ans),
    iteration_invariant=lambda state: state.iterations < 20,
)
```

### 4. Runtime Assertions (`contract_assert`)

Can be called within tool implementations. Participates in the same violation handling system.

```python
from cyllama.agents import contract_assert

@tool
def complex_operation(data: str) -> str:
    parsed = json.loads(data)
    contract_assert(isinstance(parsed, dict), "data must be JSON object")

    result = process(parsed)
    contract_assert(result is not None, "processing must not return None")

    return str(result)
```

## Execution Flow with Contracts

```
run(task)
  │
  ├─ CHECK task_precondition(task)
  │   └─ On violation: handle according to policy
  │
  ├─ For each iteration:
  │   │
  │   ├─ CHECK iteration_invariant(state)
  │   │   └─ On violation: handle according to policy
  │   │
  │   ├─ LLM generates action
  │   │
  │   ├─ If tool call:
  │   │   │
  │   │   ├─ CHECK tool @pre conditions(args)
  │   │   │   └─ On violation: handle according to policy
  │   │   │
  │   │   ├─ Execute tool (may call contract_assert internally)
  │   │   │
  │   │   └─ CHECK tool @post conditions(result, args)
  │   │       └─ On violation: handle according to policy
  │   │
  │   └─ Continue
  │
  ├─ Final answer generated
  │
  └─ CHECK answer_postcondition(answer)
      └─ On violation: handle according to policy
```

## New Event Types

```python
class EventType(Enum):
    # ... existing types ...
    CONTRACT_CHECK = "contract_check"      # Contract being evaluated
    CONTRACT_VIOLATION = "contract_violation"  # Violation detected
```

## Semantic Behavior Table

| Semantic | Check | Handler Called | Continues | Terminates |
|----------|-------|----------------|-----------|------------|
| `IGNORE` | No | No | Yes | No |
| `OBSERVE` | Yes | Yes (if fails) | Yes | No |
| `ENFORCE` | Yes | Yes (if fails) | No | Yes (after handler) |
| `QUICK_ENFORCE` | Yes | No | No | Yes (immediately) |

## Design Decisions

The following design questions were resolved during implementation:

### 1. Contract Granularity

Contracts can specify policy per-tool via the `policy` parameter on `@pre`/`@post` decorators, with a global default set on the agent.

### 2. Postcondition Access

Postconditions can access both the result AND original arguments. The decorator auto-detects if the predicate takes 1 or 2 parameters:
- `@post(lambda r: r > 0)` - result only
- `@post(lambda r, args: r <= args['max'])` - result and args

### 3. LLM-Evaluated Contracts

Not implemented. Contracts are evaluated directly via Python predicates. LLM-based semantic validation could be a future extension.

### 4. Contract Inheritance

Tools retain their contracts when used across agents. Agents can override behavior via their `policy` setting (e.g., `IGNORE` to disable checking).

### 5. Abort vs Terminate

Only `terminate` is implemented via `ContractTermination` exception. No separate abort mechanism - `QUICK_ENFORCE` provides immediate termination without handler invocation.

## Implementation Notes

- Postconditions receive the actual typed return value (`raw_result`), not the stringified observation
- `ReActAgent` now includes `tool_name`, `tool_args`, and `raw_result` in event metadata
- Thread-local `ContractContext` enables `contract_assert` to participate in agent's violation handling

See [CONTRACT_AGENT_IMPL.md](CONTRACT_AGENT_IMPL.md) for detailed implementation documentation.

## References

- [Contract assertions (C++26) - cppreference.com](https://en.cppreference.com/w/cpp/language/contracts.html)
- [Contracts for C++ P2900R14](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2900r14.pdf)
- [Contracts for C++ explained in 5 minutes](https://timur.audio/contracts_explained_in_5_mins)
- [What's new in C++26: contracts (part 3)](https://mariusbancila.ro/blog/2025/03/29/whats-new-in-cpp26-contracts-part-3/)
