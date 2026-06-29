# ContractAgent: design notes and repositioning plan

> **The thesis:** ContractAgent's value proposition needs to be re-centered. > As currently documented, `@pre` is used for argument validation — which is > now largely subsumed by `coerce_args` and `Annotated[]` constraint markers > (landed in the second-pass agent work). What's left for contracts is the > set of things schema cannot express: cross-field rules, state-dependent > rules, behavioral postconditions, and cross-call invariants. The docs lead > with the redundant use case and bury the unique one, which makes the agent > look useless. This document captures the design of how to fix that.

Maintainer audience. The plan is referenced from `AGENT_TOOL_REVIEW.md` proposal #17 and from the contract.py module docstring.

## 1. The problem

Today's `contract.py` (1196 LoC) supports two attachment points:

1. **Tool-level**: `@pre` / `@post` decorators on tool functions.

2. **Agent-level**: `task_precondition`, `answer_postcondition`, `iteration_invariant` passed to `ContractAgent.__init__`.

The module docstring (lines 33-77) and the canonical examples lead with `@pre`:

```python
@tool
@pre(lambda args: args['count'] > 0, "count must be positive")
def fetch_items(count: int) -> str: ...
```

After the second-pass landing of `coerce_args` + Annotated[] markers, the same constraint is now expressible as:

```python
@tool
def fetch_items(count: Annotated[int, Ge(1)]) -> str: ...
```

The Annotated form is declarative, runs at schema-generation time (so the LLM can see the constraint in the prompt), and survives the wire boundary (MCP, OpenAI). The `@pre` form is imperative, runs at dispatch time only, and is invisible to anything that doesn't speak ContractAgent.

For type, range, enum, pattern, length, required, and multipleOf checks — which is what most documented `@pre` examples do — `@pre` is *strictly inferior* to schema. A user who reads the docs and tries ContractAgent for this purpose correctly concludes it's redundant.

What `@pre` and the agent-level callbacks *uniquely* enable is buried.

## 2. What contracts uniquely enable

Schema is purely structural. It cannot express:

1. **Cross-field rules.** "`end > start`", "`payment_type == 'card'` implies `card_number` is present". JSON Schema has `dependentRequired`, `dependentSchemas`, and `if/then/else`, but they're awkward to write and the LLM can't easily reason about them.

2. **State-dependent rules.** "the database connection is open", "the user is authenticated", "now < deadline", "the budget hasn't been exhausted". Schema is purely structural and has zero runtime visibility.

3. **Behavioral postconditions.** "the returned list is non-empty", "the output is sorted", "the answer doesn't contain PII", "the answer doesn't leak system-prompt content". Schema has a `returns` field conceptually but cyllama doesn't enforce it, and even if it did, postconditions are usually behavioral rather than structural.

4. **Cross-call / cross-iteration invariants.** "errors < 3", "total cost < budget", "elapsed < deadline", "same tool not called more than N times consecutively" (distinct from loop detection — that's built-in, policy-driven; this is user-defined).

5. **Agent-level postconditions.** "the final answer mentions all required entities", "the answer doesn't reveal internal reasoning".

These five categories are the legitimate niche. They have no schema substitute, and they would be hard to express even if schema were richer.

## 3. The repositioning plan

Six steps, scoped so they can land independently. The minimum-viable landing is steps 1 + 3 + 6 — that's a documentation reframe + small API addition + concrete recipes, and it's enough to flip ContractAgent from "looks redundant" to "has a clear niche". Steps 2, 4, 5 add real new capability that the niche needs but can land later as pull develops.

### Step 1 — Reposition `@pre` in the module docs

**Scope:** docstring rewrite in `contract.py` (lines 1-86), no code changes. Lead with the four scenarios `@pre` is uniquely good at; demote argument-validation examples; cross-link to `Annotated[]` markers.

**Concrete edits:**

- Replace the `@pre(lambda args: args['count'] > 0)` example in the Basic Usage block with a cross-field example (`lambda args: args['end'] > args['start']`) and a state-dependent example (`lambda args: db.is_connected()`).

- Add a "When to use schema vs contracts" callout near the top, pointing at `tools.py` for `Annotated[]` markers and reserving `@pre` for the scenarios in §2.

- Update the docstring on the `pre` decorator function itself (lines ~432-450) with the same framing.

**Blast radius:** zero (docs only). Estimated effort: half a day.

### Step 2 — Make `@post` first-class (value-aware postconditions)

**Scope:** rewire `@post` to receive the raw return value, not the stringified observation.

**Today's behavior.** `ReActAgent._execute_tool_raw` returns the raw value (`react.py:402`), then the dispatch loop stringifies it (`react.py:403`: `observation = str(raw_result)`) before any postcondition runs. So `@post(lambda r: len(r) > 0)` is testing `len(str(raw_result))` — a string length, not the actual list length.

**Target behavior.** `@post` receives `raw_result` itself, so:

```python
@tool
@post(lambda r: len(r) > 0, "result must not be empty")
def fetch_rows(table: str, limit: int) -> List[Dict]: ...
```

evaluates `len(r) > 0` on the actual list. Two-arg form `@post(lambda r, args: ...)` already exists and continues to work.

**Concrete edits:**

- ContractAgent's postcondition evaluation site reads `raw_result` from the ACTION event metadata (which already preserves it via `react.py:472` — `"raw_result": raw_result`) instead of the OBSERVATION event's `content`.

- Document that `@post` predicates see the typed return value, not its string render.

- Add a test asserting the predicate receives the actual type.

**Blast radius:** low. Touches contract.py's postcondition check and a small section of ContractAgent's stream loop. Estimated effort: half a day. Pairs naturally with Proposal #9 (typed observation rendering) but doesn't depend on it.

### Step 3 — List-form agent-level callbacks

**Scope:** API addition. The current `ContractAgent.__init__` takes single callables:

```python
ContractAgent(llm=..., tools=[...], iteration_invariant=lambda s: s.errors < 3)
```

The model of "one big lambda" forces users to AND-compose unrelated invariants by hand. Accept lists instead — each entry checked independently, each violation surfaces as its own event:

```python
ContractAgent(
    llm=..., tools=[...],
    iteration_invariants=[
        lambda s: s.errors < 3,
        lambda s: s.elapsed_ms < 30_000,
        lambda s: s.tool_calls < 20,
    ],
    answer_postconditions=[
        lambda a: "system_prompt" not in a,
        lambda a: contains_required_entities(a, ["name", "email"]),
    ],
    task_preconditions=[
        lambda t: len(t) >= 10,
        lambda t: not t.startswith("ignore previous"),
    ],
)
```

**Backwards-compat.** Singular forms (`iteration_invariant=`) still accepted; auto-wrapped into a single-element list. Document the plural forms as preferred and deprecate (not remove) the singular forms in a future minor version.

**Concrete edits:**

- Add `iteration_invariants: Optional[List[Callable]]`, `answer_postconditions: Optional[List[Callable]]`, `task_preconditions: Optional[List[Callable]]` to `ContractAgent.__init__`.

- Normalize: if singular form passed, wrap in list. If both passed, raise `ValueError`.

- The `_check_*` methods loop over the list, calling violation handler per-failure (governed by the policy as today).

- Update the module docstring's Agent-Level Contracts example.

**Blast radius:** low. Strict superset of current API. Estimated effort: half a day.

### Step 4 — Surface contract violations as `AgentEvent`s

**Scope:** make violations observable on the event stream, not just on loggers and exceptions.

**Today's behavior.** A violation under ENFORCE raises `ContractTermination`. A violation under OBSERVE calls the violation handler (typically a logger). Neither emits anything on the agent's event stream. So a monitoring UI that subscribes to `agent.stream(...)` has no way to know contracts ran, let alone whether they passed.

**Target behavior.** Emit `EventType.CONTRACT_VIOLATION` (and optionally `CONTRACT_PASS` under OBSERVE) on every check:

```python
yield AgentEvent(
    type=EventType.CONTRACT_VIOLATION,
    content=violation.message,
    metadata={
        "kind": "pre" | "post" | "iteration_invariant" | "answer_postcondition" | "task_precondition",
        "location": tool_name or "agent",
        "predicate": violation.predicate,   # source-extracted if available
        "policy": violation.policy.value,
        "passed": False,
    },
)
```

**Why this matters.** Three concrete unlocks:

1. **OBSERVE policy becomes useful in production**, not just debugging. Today OBSERVE logs and continues — but logs are out-of-band. With events on the stream, an observer can react in real time.

2. **Monitoring UIs can show contract state** alongside thoughts and actions, making the agent's reasoning auditable.

3. **Test assertions become natural** — `assert any(e.type == CONTRACT_VIOLATION for e in events)` rather than introspecting agent metrics.

**Concrete edits:**

- Add `EventType.CONTRACT_VIOLATION` (and `CONTRACT_PASS`) to `types.py`.

- Each `_check_*` site in contract.py yields the event before applying policy.

- Update the EventType docstring with the new variant.

- AsyncAgent wrappers in `async_agent.py` need no changes — they pass-through unknown event types.

- Existing consumers that ignore unknown event types are unaffected.

- ACP/integrations layer needs an event-type map update; verify no hardcoded list of event types exists in serialization code.

**Blast radius:** medium. New event type ripples through three agent loops (ReAct, Constrained, Contract), async wrappers, and any consumer that exhaustively-matches on `EventType`. Estimated effort: one day.

### Step 5 — Make `IterationState` actually useful

**Scope:** expand the state object that `iteration_invariant` predicates receive.

**Today's `IterationState`** (`contract.py:1175-1197`) tracks counters only: `iteration`, `errors`, `tool_calls`. Predicates can't express "stop if last observation hasn't changed in 3 iterations" or "stop if context size exceeds 4000 tokens" — both of which are exactly the kind of stateful invariant that justifies ContractAgent's existence.

**Target shape:**

```python
@dataclass
class IterationState:
    iteration: int
    errors: int
    tool_calls: int
    # NEW fields:
    elapsed_ms: float
    last_tool_name: Optional[str]
    last_observation: Optional[str]
    observations_so_far: List[str]   # capped to last N (default 10)
    estimated_prompt_chars: int       # cheap proxy for token budget
    consecutive_same_observation: int  # 0 if last two observations differ
```

**Concrete edits:**

- Extend the `IterationState` dataclass in `contract.py`.

- Fill the new fields in `ContractAgent._update_iteration_state` (or equivalent — the method that runs per-iteration in the stream loop).

- Document each new field.

- Cap `observations_so_far` to avoid unbounded memory growth.

**Blast radius:** low. Strict additive change to the dataclass. Estimated effort: half a day. Pairs naturally with Step 3 — list-form invariants plus richer state turn `iteration_invariants` into a real composition surface.

### Step 6 — Recipes doc

**Scope:** new file `docs/agents/contracts.md` (or section in existing agents docs). Six concrete recipes, ~10 lines of code + 2-3 sentences each:

1. **Cross-field precondition** — `@pre(lambda a: a['end'] > a['start'])`

2. **State-dependent precondition** — `@pre(lambda a: db.is_open())` with closure over the db handle

3. **Behavioral postcondition** — `@post(lambda r: r == sorted(r))`

4. **Cost-cap invariant** — `iteration_invariants=[lambda s: s.cost < 1.0]` composed with a cost-tracking handler

5. **Answer-content postcondition** — `answer_postconditions=[lambda a: required_entity in a]`

6. **No-PII postcondition** — `@post(lambda r: not detects_pii(r))`

Each recipe links to the relevant API in `cyllama.agents` and explicitly says "this is not expressible in JSON Schema — that's why contracts exist".

**Blast radius:** zero (new doc file). Estimated effort: half a day.

## 4. Total scope and sequencing

| Step | Title | Blast radius | Effort |
|------|-------|---|---|
| 1 | Reposition `@pre` docs | zero | 0.5d |
| 2 | Value-aware `@post` | low | 0.5d |
| 3 | List-form agent callbacks | low | 0.5d |
| 4 | Violation events | medium | 1d |
| 5 | Richer `IterationState` | low | 0.5d |
| 6 | Recipes doc | zero | 0.5d |

Total: ~2.5 days for the full landing. Natural sequence: 1 → 3 → 6 (minimum) → 2 → 5 → 4 (full).

**Minimum-viable subset (1 + 3 + 6, ~1 day):** docs reframe + list-form callbacks + recipes. Does not add new capability; repositions existing capability so users find the parts that aren't redundant. Lowest-risk way to test whether the repositioning lands. Steps 2, 4, 5 add real new capability and can wait for pull.

## 5. Success criteria

After the full landing, the test is concrete: can a user articulate a constraint that ContractAgent expresses and `Annotated[]` doesn't?

After Step 1 + 6: yes, by reading the docs and recipes.

After Step 2: yes, for any behavioral postcondition.

After Step 3 + 5: yes, for any cross-iteration invariant that needs elapsed time, observation history, or prompt-size visibility.

After Step 4: yes, for any agent that needs to *expose* its contracts to an observer — UIs, monitoring, audit logs.

If after all six steps a reader still concludes ContractAgent is useless, the repositioning has failed and the agent should be retired, not patched. This document is the bet that the repositioning will land.

## 6. Non-goals

- **Not subsuming schema.** `Annotated[]` constraint markers remain the canonical place for type/range/enum/pattern checks. Contracts do not compete with schema; they fill the gaps schema cannot express.

- **Not adding a DSL.** Predicates stay as plain Python callables. No CEL, no JSONata, no "contract expression language". Python's type-and-test ergonomics are sufficient; introducing a DSL would reintroduce the cross-language coordination problem (predicates invisible to JSON Schema, MCP, ACP) that this design accepts as a constraint.

- **Not making contracts serializable across the wire.** They are Python-only and local to the agent process. If a use case demands cross-process invariants, that's a different design problem (and arguably belongs to MCP / ACP capabilities, not to ContractAgent).

- **Not deprecating the singular-form `iteration_invariant=` etc.** Backwards compatibility is preserved; the plural forms are additive.

## 7. Open questions

- **How should violation handlers see the `CONTRACT_VIOLATION` event?** Currently handlers receive a `ContractViolation` dataclass. Should they also see the emitted `AgentEvent`? If yes, the handler signature grows; if no, the event becomes the only source of truth and the handler is just a side-effect site.

- **Per-iteration vs per-event invariant checking.** Today `iteration_invariant` fires once per iteration. Should it fire on *every* event (with type filtering), giving users finer-grained intervention points? Trade-off: latency vs flexibility. Leave at per-iteration for v1; revisit if pull develops.

- **Cost tracking inside `IterationState`.** Step 5 lists `cost` as a recipe (#4) but doesn't list it as a built-in field. Cost is user-defined (depends on the model, the API, the accounting policy) so it should stay user-supplied via a closure, not built into the state object. Document the recipe.

## 8. Relationship to other proposals

- **Proposal #7 (tool-arg coercion)**: landed. Subsumed most existing `@pre` use cases. This document is the consequence.

- **Proposal #16 (Annotated[] markers)**: landed. Same.

- **Proposal #9 (typed observation rendering)**: pending. Step 2 above benefits from this but doesn't require it — `raw_result` is already preserved in event metadata.

- **Proposal #11 (real-inference grammar test)**: landed. Established the precedent for "verify before extend" that this document follows.

- **Proposal #15 (adversarial test suite)**: pending. The contract events from Step 4 should be exercised by the adversarial tests.

This document supersedes the brief Proposal #17 entry in `AGENT_TOOL_REVIEW.md` (~lines 427-470). The review entry will be updated with a one-line pointer here once the work lands.
