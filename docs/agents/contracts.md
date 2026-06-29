# Contract recipes

> **TL;DR.** Use `Annotated[]` constraint markers (`Ge`, `Le`, `Pattern`, > etc.) for per-argument type, range, enum, pattern, and length checks. > Use **contracts** (`@pre`, `@post`, agent-level callbacks) for the > things schema cannot express: cross-field rules, state-dependent rules, > behavioral postconditions on return values, and invariants that span > multiple agent iterations.

This document is a cookbook of working contract patterns. Each recipe explains *what* the rule is, *why* schema can't express it, and *where* to attach it. For the design rationale, see [`docs/dev/contract-agent.md`](../dev/contract-agent.md).

---

## Recipe 1 — Cross-field precondition

**Use case.** A relationship *between* arguments. JSON Schema has `dependentRequired` / `if`/`then`/`else`, but they're awkward to write and the LLM can't easily reason about them.

```python
from cyllama.agents import tool, pre

@tool
@pre(lambda args: args["end"] > args["start"], "end must follow start")
def fetch_range(start: int, end: int) -> list[Row]:
    """Fetch rows where row.id is in [start, end)."""
    ...
```

**What schema covers:** that `start` and `end` are both integers (via `Annotated[int, Ge(0)]` etc.). **What the contract covers:** that they relate correctly. **Failure mode:** under ENFORCE the agent aborts before the SQL runs; under OBSERVE a `CONTRACT_VIOLATION` event lands on the stream and the loop continues so the LLM can self-correct.

---

## Recipe 2 — State-dependent precondition

**Use case.** The validity of a call depends on runtime state — an external resource, the clock, an authentication context. Schema has zero runtime visibility.

```python
from cyllama.agents import tool, pre

db = MyDatabase(...)

@tool
@pre(lambda args: db.is_connected(), "database must be open")
@pre(lambda args: now() < args["deadline"], "deadline already past")
def query_users(deadline: float) -> list[User]:
    """Query the user table; bails if the db is closed or the deadline is past."""
    return db.fetch_users()
```

**Closures capture the state.** The predicate is plain Python, so it can read from any object in scope. Multiple `@pre` decorators stack; they all run, in declaration order.

---

## Recipe 3 — Behavioral postcondition on return value

**Use case.** A property of the *output* that schema can't express: sorted, non-empty, idempotent, free of secrets, etc.

```python
from cyllama.agents import tool, post

@tool
@post(lambda r: len(r) > 0, "must return at least one row")
@post(lambda r: r == sorted(r), "must return sorted output")
def fetch_ordered(table: str) -> list[int]:
    """Fetch ordered ids from a table."""
    rows = db.query(f"select id from {table} order by id")
    return [row.id for row in rows]
```

**Predicates receive the raw value**, not its string render — so `len(r)`, `r == sorted(r)`, and `isinstance(r, MyType)` all work as written. (The pre-Step-2 implementation passed `str(raw_result)`, which silently broke this pattern. Pinned with regression tests in `tests/test_agents_contract.py::TestPostReceivesRawValue`.)

---

## Recipe 4 — Cost-cap invariant across iterations

**Use case.** A budget that spans the whole run, not any single tool call. No schema substitute exists; this is the niche ContractAgent was built for.

```python
from cyllama.agents import ContractAgent, ContractPolicy

class CostTracker:
    def __init__(self, budget_usd: float):
        self.budget_usd = budget_usd
        self.spent = 0.0

    def charge(self, usd: float) -> None:
        self.spent += usd

tracker = CostTracker(budget_usd=1.00)

# Tools update the tracker as they run (e.g. paid API calls).
# The invariant fires on every THOUGHT, terminating the run if budget exceeded.
agent = ContractAgent(
    llm=my_llm,
    tools=[paid_search, paid_lookup],
    iteration_invariants=[
        lambda s: tracker.spent < tracker.budget_usd,
    ],
    policy=ContractPolicy.ENFORCE,
)
```

**Why this isn't `max_iterations`.** `max_iterations` caps the count of loops, which is an indirect proxy for cost (cheap tools cost less). A cost cap is the actual constraint; expressing it directly is clearer and lets users mix tools with different cost profiles.

---

## Recipe 5 — Answer-content postcondition

**Use case.** Properties of the *final* answer, not of any individual tool's output. Common in compliance or product-safety contexts.

```python
from cyllama.agents import ContractAgent

required_entities = ["name", "email", "ticket_id"]

agent = ContractAgent(
    llm=my_llm,
    tools=[...],
    answer_postconditions=[
        lambda a: all(e in a.lower() for e in required_entities),
        lambda a: not a.lower().startswith("i cannot"),
        lambda a: len(a) < 4000,  # response-length cap for downstream UI
    ],
)
```

**Composition is via lists.** Each predicate runs independently; under OBSERVE every failing predicate yields its own `CONTRACT_VIOLATION` event, so monitoring can attribute each failure to its specific rule. The singular `answer_postcondition=` form is still accepted for back-compat.

---

## Recipe 6 — No-PII postcondition with a real predicate

**Use case.** Tool output must not contain PII. Combines a tool-level `@post` (for narrow scope) with an answer-level postcondition (for defense in depth on the final answer).

```python
import re
from cyllama.agents import tool, post, ContractAgent

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

def _contains_pii(text: str) -> bool:
    return bool(_EMAIL_RE.search(text) or _SSN_RE.search(text))


@tool
@post(lambda r: not _contains_pii(str(r)), "tool output must not leak PII")
def summarize_records(query: str) -> str:
    """Summarize records matching the query."""
    ...


agent = ContractAgent(
    llm=my_llm,
    tools=[summarize_records],
    answer_postconditions=[
        lambda a: not _contains_pii(a),
    ],
)
```

**Two attachment points, deliberately.** The tool-level `@post` catches the violation as close to the source as possible (so the agent can retry with a different tool / args). The agent-level postcondition is the last-line check on the final answer, in case PII slipped through a tool that doesn't have `@post` or via the LLM's own paraphrasing.

---

## Recipe 7 — Stuck-loop detector beyond the built-in

**Use case.** ReAct's built-in loop detector catches *identical* action repeats and same-tool spam. It does not catch "the agent is making progress but the output isn't changing" — same observation N times in a row. Express this directly using `IterationState`:

```python
agent = ContractAgent(
    llm=my_llm,
    tools=[...],
    iteration_invariants=[
        lambda s: s.consecutive_same_observation < 3,
    ],
    policy=ContractPolicy.ENFORCE,
)
```

**What's available on the state.** See `IterationState` in `cyllama/agents/contract.py`: `iterations`, `tool_calls`, `errors`, `elapsed_ms`, `last_tool_name`, `last_observation`, `observations_so_far` (capped to last 10), `estimated_prompt_chars`, `consecutive_same_observation`. All filled in automatically as events stream through; predicates just read whatever's relevant.

---

## Recipe 8 — Time-budget invariant

**Use case.** Don't let the agent run forever, but express it in terms of wall-clock time rather than iteration count.

```python
agent = ContractAgent(
    llm=my_llm,
    tools=[...],
    iteration_invariants=[
        lambda s: s.elapsed_ms < 30_000,  # 30 second budget
    ],
)
```

**`elapsed_ms` is wall-clock since the first event.** The clock starts the first time `IterationState.update()` runs — typically the first event from the inner agent. Includes time spent in LLM generation, tool dispatch, and contract checks.

---

## Recipe 9 — Combined budgets

**Use case.** Cap multiple budgets simultaneously. Lists make this trivial; the singular form would have forced a single mega-lambda.

```python
agent = ContractAgent(
    llm=my_llm,
    tools=[...],
    iteration_invariants=[
        lambda s: s.errors < 3,
        lambda s: s.tool_calls < 20,
        lambda s: s.elapsed_ms < 30_000,
        lambda s: s.estimated_prompt_chars < 6_000,
        lambda s: s.consecutive_same_observation < 3,
    ],
    answer_postconditions=[
        lambda a: not contains_pii(a),
        lambda a: len(a) < 4000,
    ],
)
```

Under OBSERVE, every failing predicate yields its own event with its list index in `metadata["index"]`, so you can attribute failures back to specific rules in dashboards or logs.

---

## Anti-pattern reference

These are the cases where contracts *look* like the right tool but are actually inferior to schema. They're listed here to make the distinction concrete.

### Don't: per-arg type / range with `@pre`

```python
# Bad — the constraint is invisible to the LLM, invisible across the wire,
# and harder to read.
@tool
@pre(lambda args: args["count"] > 0, "count > 0")
@pre(lambda args: args["count"] <= 100, "count <= 100")
def fetch(count: int) -> list[Row]: ...

# Good — schema reaches the model, survives MCP/OpenAI serialization,
# and is enforced by coerce_args before the tool ever runs.
from typing import Annotated
from cyllama.agents import Ge, Le

@tool
def fetch(count: Annotated[int, Ge(1), Le(100)]) -> list[Row]: ...
```

### Don't: enum check with `@pre`

```python
# Bad
@tool
@pre(lambda args: args["mode"] in {"sync", "async"}, "bad mode")
def run(mode: str) -> ...: ...

# Good
from typing import Literal

@tool
def run(mode: Literal["sync", "async"]) -> ...: ...
```

### Don't: regex/length with `@pre`

```python
# Bad
@tool
@pre(lambda args: re.fullmatch(r"\w+", args["name"]), "bad name")
@pre(lambda args: 2 <= len(args["name"]) <= 50, "bad length")
def lookup(name: str) -> ...: ...

# Good
from typing import Annotated
from cyllama.agents import Pattern, MinLen, MaxLen

@tool
def lookup(name: Annotated[str, MinLen(2), MaxLen(50), Pattern(r"\w+")]) -> ...: ...
```

The rule of thumb: **if the rule is about one argument's type/range/shape and you can write it in `Annotated[]`, do.** Reserve contracts for the things `Annotated[]` cannot reach.

---

## Further reading

- [`docs/dev/contract-agent.md`](../dev/contract-agent.md) — design rationale, repositioning plan, success criteria

- [`src/cyllama/agents/contract.py`](../../src/cyllama/agents/contract.py) — module docstring (the canonical `@pre` / `@post` reference)

- [`src/cyllama/agents/tools.py`](../../src/cyllama/agents/tools.py) — `Annotated[]` constraint markers (`Ge`, `Le`, `Pattern`, ...) and `coerce_args`
