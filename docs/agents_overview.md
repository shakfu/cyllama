# Cyllama Agent Framework Overview

Cyllama includes a zero-dependency agent framework for building tool-using LLM agents. The framework provides three agent architectures, each designed for different reliability and control requirements.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Tools](#tools) -- including `Annotated[]` constraints, coercion, timeouts
4. [Agents](#agents)
   - [ReActAgent](#reactagent)

   - [ConstrainedAgent](#constrainedagent)

   - [ContractAgent](#contractagent)
5. [Multi-Agent Composition](#multi-agent-composition) -- `agent_as_tool`, `TieredAgentTeam`
6. [Events and Results](#events-and-results)
7. [Configuration](#configuration)
8. [Best Practices](#best-practices)
9. [Async Agents](#async-agents)
10. [Further Reading](#further-reading) -- recipes + design docs
11. [Experimental: ACPAgent](#experimental-acpagent)

## Quick Start

```python
from cyllama import LLM
from cyllama.agents import ReActAgent, tool

# Define a tool
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

# Create agent
llm = LLM("models/Llama-3.2-1B-Instruct-Q8_0.gguf")
agent = ReActAgent(llm=llm, tools=[calculate])

# Run task
result = agent.run("What is 25 * 4?")
print(result.answer)  # "100"
```

## Architecture

```text
                        ┌──────────────────────────────────┐
                        │           User Task              │
                        └──────────────┬───────────────────┘
                                       │
                        ┌──────────────▼───────────────────┐
                        │         Agent Layer              │
                        │  ┌─────────────────────────────┐ │
                        │  │ ReActAgent | ContractAgent  │ │
                        │  │      ConstrainedAgent       │ │
                        │  └─────────────────────────────┘ │
                        └──────────────┬───────────────────┘
                                       │
              ┌────────────────────────┼──────────────────────┐
              │                        │                      │
    ┌─────────▼─────────┐   ┌──────────▼────────┐   ┌─────────▼─────────┐
    │   Tool Registry   │   │       LLM         │   │  Event Stream     │
    │  - Tool lookup    │   │  - Generation     │   │  - THOUGHT        │
    │  - Schema gen     │   │  - Streaming      │   │  - ACTION         │
    │  - Execution      │   │  - Grammar (opt)  │   │  - OBSERVATION    │
    └───────────────────┘   └───────────────────┘   │  - ANSWER         │
                                                    │  - ERROR          │
                                                    └───────────────────┘
```

## Tools

Tools are Python functions that agents can invoke. The `@tool` decorator automatically extracts type information and generates JSON schemas.

### Defining Tools

```python
from cyllama.agents import tool

@tool
def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web for information.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
    """
    # Implementation
    return f"Results for: {query}"

@tool(name="calc", description="Evaluate math expressions")
def calculate(expression: str) -> float:
    """Safe math evaluation."""
    return eval(expression)
```

### Tool Parameters

Type hints are automatically converted to JSON schema types:

| Python Type | JSON Schema Type |
|------------|------------------|
| `str` | `string` |
| `int` | `integer` |
| `float` | `number` |
| `bool` | `boolean` |
| `list` | `array` |
| `dict` | `object` |

Parameters without default values are marked as required.

### Tool Registry

Tools can be managed via the `ToolRegistry` class:

```python
from cyllama.agents import Tool, ToolRegistry

registry = ToolRegistry()
registry.register(search_web)
registry.register(calculate)

# Get tool by name
tool = registry.get("search_web")

# Generate prompt descriptions
prompt = registry.to_prompt_string()

# Generate JSON schemas (OpenAI format)
schemas = registry.to_json_schema()
```

### Schema Constraints with `Annotated[]`

Type hints alone capture *kind* (int, str, list); for *constraints* — range, length, pattern, enum — annotate the type with stdlib-style markers exported from `cyllama.agents`. The marker values land in the generated JSON Schema and are also enforced by the dispatch-time validator (next subsection).

```python
from typing import Annotated, Literal
from cyllama.agents import tool, Ge, Le, Gt, Lt, MultipleOf, MinLen, MaxLen, Pattern

@tool
def fetch_rows(
    table: Annotated[str, MinLen(1), MaxLen(64), Pattern(r"^[a-z_][a-z0-9_]*$")],
    limit: Annotated[int, Ge(1), Le(1000)],
    chunk_size: Annotated[int, Gt(0), Lt(500), MultipleOf(10)] = 100,
    mode: Literal["preview", "full"] = "preview",
) -> list[dict]:
    """Fetch rows from a table with bounded paging."""
    ...
```

Marker -> JSON Schema keyword mapping:

| Marker | Schema keyword | Applies to |
|---|---|---|
| `Ge(n)` | `minimum` | integer, number |
| `Gt(n)` | `exclusiveMinimum` | integer, number |
| `Le(n)` | `maximum` | integer, number |
| `Lt(n)` | `exclusiveMaximum` | integer, number |
| `MultipleOf(n)` | `multipleOf` | integer, number |
| `MinLen(n)` | `minLength` / `minItems` | string / array |
| `MaxLen(n)` | `maxLength` / `maxItems` | string / array |
| `Pattern(s)` | `pattern` | string |

`Literal["a", "b"]` continues to produce `{"type": "string", "enum": [...]}` via the existing path. Zero runtime dependency -- no `annotated_types` or `pydantic`.

### Tool-Argument Coercion and Validation

LLMs frequently emit string-typed JSON values for numeric fields ("5" instead of 5). The dispatch layer runs `coerce_args(tool, args)` before invoking the tool function when `tool.coerce` is True (the default; opt out via `@tool(coerce=False)` for tools that genuinely want loose typing or `**kwargs`). Coercion is narrow on purpose -- it fixes shape mismatches an LLM is likely to emit, not arbitrary type juggling:

- `"5"` -> `int(5)` for `integer` fields; `"1.5"` -> `float(1.5)` for `number` fields
- `"true"` / `"false"` (any case, plus `"1"`/`"0"`, `"yes"`/`"no"`) -> `bool`
- Missing required args -> `ToolArgumentError`
- Unknown args (not declared in the schema) -> `ToolArgumentError`
- `Literal[...]` enum violations -> `ToolArgumentError`
- `bool` passed for an `int` field -> rejected (Python's int-subclass trap)
- `Annotated[]` bounds violated (out of range, wrong pattern, too long) -> `ToolArgumentError`
- NaN / infinity values for bounded numeric fields -> rejected (NaN comparisons silently pass otherwise)

`ToolArgumentError` is a `ValueError` subclass, so existing agent exception handlers catch it; the precise message gets fed back to the LLM as the observation so it can self-correct on the next iteration.

### Tool Timeouts

Set `Tool.timeout` (in seconds) on tools that may hang -- network calls, recursive operations, anything that could miss a deadline:

```python
@tool(timeout=10.0)
def fetch_url(url: str) -> str:
    """Fetch a URL; abandoned after 10 seconds."""
    ...
```

The agent runs the tool on a daemon thread, joins for the declared budget, and raises `ToolTimeoutError(tool_name, timeout)` if it exceeds. Default is `None` (no enforcement). The worker thread *keeps running* after the timeout -- Python cannot safely kill threads -- but the agent abandons the result and continues. For hard resource limits (memory, file descriptors), use out-of-process tools and let the OS enforce.

### Observation Rendering

The dispatch layer converts a tool's raw return value to the string that lands on the `OBSERVATION` event `content` via `render_observation()`. Dicts and lists become JSON (so the LLM sees something re-parseable, not Python's `repr` form with single quotes and `None` instead of `null`); scalars and objects that can't be JSON-encoded fall back to `str()`. The raw value is preserved on event metadata as `raw_result`, so `@post` contracts see the actual typed return value, not its string render.

## Agents

### ReActAgent

Implements the ReAct (Reasoning + Acting) pattern where the agent alternates between thinking and acting.

**Pattern:**

```text
Thought: [reasoning about what to do]
Action: tool_name({"arg": "value"})
Observation: [result from tool]
... (repeat)
Thought: I now know the answer
Answer: [final answer]
```

**Reference:** [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

**Usage:**

```python
from cyllama import LLM
from cyllama.agents import ReActAgent, tool

@tool
def search(query: str) -> str:
    return f"Results for: {query}"

agent = ReActAgent(
    llm=LLM("model.gguf"),
    tools=[search],
    max_iterations=10,
    verbose=True,
)

result = agent.run("Search for Python tutorials")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `LLM` | required | LLM instance for generation |
| `tools` | `List[Tool]` | `None` | Available tools |
| `system_prompt` | `str` | default | Custom system prompt |
| `max_iterations` | `int` | `10` | Maximum thought/action cycles |
| `verbose` | `bool` | `False` | Print reasoning to stdout |
| `generation_config` | `GenerationConfig` | default | LLM generation settings |
| `detect_loops` | `bool` | `True` | Enable loop detection |
| `max_consecutive_same_action` | `int` | `2` | Same action repeat limit |
| `max_consecutive_same_tool` | `int` | `4` | Same tool repeat limit |
| `max_context_chars` | `int` | `6000` | Context truncation limit |

**Strengths:**

- Natural reasoning trace for debugging

- Works well with most instruction-tuned models

- Flexible action format

**Weaknesses:**

- Parsing can fail on malformed output

- Requires larger models for reliable tool calling

---

### ConstrainedAgent

Uses GBNF grammar constraints to guarantee valid JSON tool calls. Eliminates parsing failures by constraining the LLM's output space.

**Usage:**

```python
from cyllama import LLM
from cyllama.agents import ConstrainedAgent, tool

@tool
def calculate(expression: str) -> str:
    return str(eval(expression))

agent = ConstrainedAgent(
    llm=LLM("model.gguf"),
    tools=[calculate],
    format="json",
    allow_reasoning=True,
)

result = agent.run("What is 100 / 4?")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `LLM` | required | LLM instance for generation |
| `tools` | `List[Tool]` | `None` | Available tools |
| `system_prompt` | `str` | default | Custom system prompt |
| `max_iterations` | `int` | `10` | Maximum tool call cycles |
| `verbose` | `bool` | `False` | Print actions to stdout |
| `generation_config` | `ConstrainedGenerationConfig` | default | Generation settings |
| `format` | `str` | `"json"` | Output format (`json`, `json_array`, `function_call`) |
| `allow_reasoning` | `bool` | `False` | Include reasoning field |
| `use_cache` | `bool` | `True` | Cache compiled grammars |
| `detect_loops` | `bool` | `True` | Enable loop detection |

**Output Format:**

```json
{"type": "tool_call", "tool_name": "calculate", "tool_args": {"expression": "100/4"}}
```

or

```json
{"type": "answer", "content": "The result is 25"}
```

**Strengths:**

- 100% valid JSON output (grammar-enforced)

- Works with smaller models

- Eliminates parsing failures

**Weaknesses:**

- Less natural output format

- Grammar compilation overhead (mitigated by caching)

---

### ContractAgent

Contract-based agent inspired by C++26 contracts (P2900). Adds preconditions, postconditions, and runtime assertions to tool calls -- specifically the *non-schema* checks that `Annotated[]` markers cannot express.

> **When to use `@pre` vs `Annotated[]`:** Reserve `@pre` for cross-field rules (`end > start`), state-dependent rules (`db.is_connected()`), and value-dependency rules. For simple argument constraints (type, range, enum, pattern, length), use `Annotated[int, Ge(1)]` / `Literal["a","b"]` in the type hint -- schema-side constraints reach the model in the prompt and grammar; contracts do not. See [`docs/agents/contracts.md`](agents/contracts.md) for nine worked recipes plus the anti-patterns to avoid, and [`docs/dev/contract-agent.md`](dev/contract-agent.md) for the design rationale.

**Usage:**

```python
from cyllama import LLM
from cyllama.agents import ContractAgent, tool, pre, post, ContractPolicy

@tool
@pre(lambda args: args['end'] > args['start'], "end must follow start")  # cross-field
def fetch_range(start: int, end: int) -> list[int]:
    """Cross-field rule: relationship between two args."""
    ...

@tool
@post(lambda r: r == sorted(r), "must return sorted output")  # behavioural postcondition
def fetch_ordered(table: str) -> list[int]:
    """Behavioural rule: schema cannot express 'sorted'."""
    ...

agent = ContractAgent(
    llm=LLM("model.gguf"),
    tools=[fetch_range, fetch_ordered],
    policy=ContractPolicy.ENFORCE,
    task_preconditions=[lambda task: len(task) >= 10],
    answer_postconditions=[lambda ans: "error" not in ans.lower()],
    iteration_invariants=[
        lambda s: s.errors < 3,
        lambda s: s.elapsed_ms < 30_000,
    ],
)

result = agent.run("Fetch the first 10 sorted records from `users`.")
```

**Contract Decorators:**

```python
# Precondition - cross-field or state-dependent only (use Annotated[] for arg validation)
@pre(lambda args: args['end'] > args['start'], "end must follow start")

# Postcondition - receives raw typed return value
@post(lambda result: len(result) > 0, "must return non-empty result")

# Postcondition with access to original arguments
@post(lambda r, args: len(r) <= args['max_len'], "result too long")
```

**Contract Policies:**

| Policy | Checks | Handler Called | Continues | Terminates |
|--------|--------|----------------|-----------|------------|
| `IGNORE` | No | No | Yes | No |
| `OBSERVE` | Yes | Yes (on fail) | Yes | No |
| `ENFORCE` | Yes | Yes (on fail) | No | Yes |
| `QUICK_ENFORCE` | Yes | No | No | Yes |

**Runtime Assertions:**

```python
from cyllama.agents import contract_assert

@tool
def process_data(data: str) -> str:
    parsed = json.loads(data)
    contract_assert(isinstance(parsed, dict), "data must be JSON object")
    return str(parsed)
```

**Agent-Level Contracts:**

The plural-form kwargs (`task_preconditions=`, `answer_postconditions=`, `iteration_invariants=`) accept lists so you can stack independent invariants without writing one AND-composed mega-lambda. The singular forms are still accepted for back-compat; passing both forms for the same hook raises `ValueError`.

```python
agent = ContractAgent(
    llm=llm,
    tools=[...],
    task_preconditions=[
        lambda task: len(task) >= 10,
        lambda task: not task.startswith("ignore previous"),
    ],
    answer_postconditions=[
        lambda ans: "error" not in ans.lower(),
        lambda ans: len(ans) < 4000,
    ],
    iteration_invariants=[
        lambda s: s.errors < 3,
        lambda s: s.elapsed_ms < 30_000,
        lambda s: s.tool_calls < 20,
        lambda s: s.consecutive_same_observation < 3,
    ],
)
```

Under OBSERVE policy each failing predicate surfaces its own `CONTRACT_VIOLATION` event with `metadata["index"]` pointing back to its list position, so a monitoring UI can attribute failures to specific rules. Under ENFORCE the first failure terminates.

**`IterationState` fields available to `iteration_invariants`:**

| Field | Type | Description |
|---|---|---|
| `iterations` | `int` | THOUGHT events seen so far |
| `tool_calls` | `int` | ACTION events seen so far |
| `errors` | `int` | ERROR events seen so far |
| `elapsed_ms` | `float` | Wall-clock since first event |
| `last_tool_name` | `Optional[str]` | Most recent ACTION's tool name |
| `last_observation` | `Optional[str]` | Most recent OBSERVATION content |
| `observations_so_far` | `List[str]` | Capped to last 10 observations |
| `estimated_prompt_chars` | `int` | Sum of all event content lengths (cheap proxy for context size) |
| `consecutive_same_observation` | `int` | Stuck-loop detection beyond the built-in detector |

The new fields enable invariants with no schema substitute: time budgets, prompt-budget caps, and "the agent keeps producing the same observation" detection.

**Violation Handler:**

```python
def my_handler(violation: ContractViolation) -> None:
    print(f"VIOLATION: {violation.kind} at {violation.location}")
    print(f"  Message: {violation.message}")
    # Log, alert, etc.

agent = ContractAgent(
    llm=llm,
    tools=[...],
    violation_handler=my_handler,
)
```

**Strengths:**

- Runtime verification of tool behavior

- Configurable violation handling

- Agent-level invariants

**Weaknesses:**

- Additional overhead for contract checking

- Requires explicit contract definitions

---

## Multi-Agent Composition

Common multi-agent patterns (supervisor/worker, plan-and-execute, reflection/critic, debate, parallel fan-out) compose from two small primitives in `cyllama.agents.composition` rather than requiring a new orchestration engine.

### `agent_as_tool` -- wrap any agent as a Tool

```python
from cyllama import LLM
from cyllama.agents import ReActAgent, agent_as_tool

worker = ReActAgent(llm=LLM("models/fast.gguf"), tools=[search])

worker_tool = agent_as_tool(
    worker,
    name="research",
    description="Investigate a topic and return findings.",
)

supervisor = ReActAgent(
    llm=LLM("models/strong.gguf"),
    tools=[worker_tool],
)

result = supervisor.run("Compare TLS 1.2 and TLS 1.3 handshakes.")
```

The wrapped tool fits naturally into the supervisor's registry: dispatch, coercion, timeouts, and contracts all work unchanged.

Pass a `forward_events` callback to surface the inner agent's reasoning in a streaming UI:

```python
def on_sub_event(ev):
    print(f"[{ev.source}] {ev.type.name}: {ev.content}")

worker_tool = agent_as_tool(
    worker, name="research", description="...",
    forward_events=on_sub_event,
)
```

Each forwarded event has `source` set to the inner agent's name and a `parent_event_id` linking it to the supervisor's ACTION event that triggered the sub-run.

### `TieredAgentTeam` -- supervisor + named workers

For supervisor/worker setups where each role may run on a different LLM (capability tiering: strong planner + cheap workers), use the ergonomic container:

```python
from cyllama.agents import AgentRole, TieredAgentTeam

team = TieredAgentTeam(
    supervisor=ReActAgent(llm=LLM("models/strong.gguf"), tools=[]),
    workers=[
        AgentRole("researcher", researcher_agent, "Investigate facts."),
        AgentRole("coder", coder_agent, "Write or modify code."),
        AgentRole("summarizer", summarizer_agent, "Condense findings."),
    ],
)

result = team.run("Refactor X using technique Y, then summarize.")
```

The team registers each worker as a tool on the supervisor's registry; rejects empty worker lists, duplicate names, and supervisors without a `registry` attribute. Both `ReActAgent` and `ConstrainedAgent` satisfy the registry requirement.

**GPU-budgeting note:** loading multiple 7B+ models is rarely viable on consumer hardware. Typical configurations are *one large + N small* (a 7B planner with 1B workers) or *model swapping* (one LLM in VRAM at a time, sequential dispatch). See `src/cyllama/memory.py` for sizing utilities.

### Pattern helpers built on the primitives

Beyond `agent_as_tool` + `TieredAgentTeam`, four canned helpers cover the
most-used multi-agent patterns. All live in
`src/cyllama/agents/composition.py` and re-export from
`cyllama.agents`. Each is ~50 LoC over the primitives; see
[`patterns.md`](agents/patterns.md) for the full pattern catalog plus
the ones the framework intentionally doesn't support.

#### `ReflectionLoop` -- worker / critic loop (Reflexion pattern)

```python
from cyllama.agents import ReActAgent, ReflectionLoop

worker = ReActAgent(llm=llm, tools=tools, system_prompt=WORKER_PROMPT)
critic = ReActAgent(
    llm=llm, tools=[],
    system_prompt="Respond with 'ACCEPT' if correct, otherwise list issues.",
)

loop = ReflectionLoop(worker, critic, max_attempts=3)
result = loop.run("Implement quicksort with a 3-way partition.")
```

The critic's answer is checked for `acceptance_marker` (default
`"ACCEPT"`, case-insensitive substring match); on a miss, the worker is
re-invoked with the prior draft and the critic's feedback folded in via
an overridable `revision_template`. Streamed events from each pass carry
`source="worker"` and `source="critic"` annotations.

#### `plan_and_execute` -- Plan-and-Execute pattern

```python
from cyllama.agents import ConstrainedAgent, ReActAgent, plan_and_execute

planner = ConstrainedAgent(llm=planner_llm, tools=[],
                           system_prompt="Emit a JSON list of step strings.")
executor = ReActAgent(llm=worker_llm, tools=[read_file, edit_file])

results = plan_and_execute(planner, executor, "Refactor module X.")
```

Default plan parser handles `[...]` lists, `{"steps"|"plan"|"tasks": [...]}`
dicts, and newline-split with bullet/number-prefix stripping. Pass
`plan_parser=...` for custom formats. `stop_on_error=True` (default)
halts after the first failing step.

#### `mcp_agent_tool` -- cross-process sub-agents

```python
from cyllama.agents import McpClient, McpServerConfig, mcp_agent_tool, ReActAgent

client = McpClient([McpServerConfig(name="research", ...)])
client.connect_all()

remote = mcp_agent_tool(
    client,
    server_name="research",
    agent_name="web_search",
    description="Search the web and return findings.",
)
supervisor = ReActAgent(llm=llm, tools=[remote])
```

Symmetric to `agent_as_tool` but dispatches via `McpClient.call_tool`.
The returned `Tool` is named `"{server_name}/{agent_name}"` (the
namespaced format the action parser supports). Network failures surface
as `RuntimeError` from MCP; the agent's generic exception handler
catches them.

#### `rag_as_tool` -- knowledge-base search as a tool

```python
from cyllama.rag import RAG
from cyllama.agents import ReActAgent, rag_as_tool

kb = RAG(...)  # load or build
search = rag_as_tool(kb, description="Search project docs.", top_k=5)
agent = ReActAgent(llm=llm, tools=[search])
```

Default formatter emits one `[score] text` line per hit, deduplicated by
text. Pass `method="retrieve"` to use the higher-level
`RAG.retrieve` path (applies the RAG pipeline config), or a custom
`formatter` callable for non-default observation shapes.

### Long-term semantic memory

`SemanticMemory` (in `src/cyllama/agents/memory.py`) bridges the
`cyllama.rag` subsystem to the agent layer as a namespace-aware
long-term memory primitive. Use it for cross-session continuity
("remember the user's preferences") that the in-process `Session`
stores can't provide:

```python
from cyllama.rag import RAG
from cyllama.agents import SemanticMemory

rag = RAG(...)
memory = SemanticMemory(rag)

memory.remember("The user prefers concise answers.", namespace="user:alice")
hits = memory.retrieve("response style", namespace="user:alice")
for hit in hits:
    print(f"[{hit.score:.2f}] {hit.text}")
```

Namespaces are stored as metadata on the underlying RAG records and
filtered at retrieval time, so one `RAG` instance can back many logical
memory buckets. Pair with `rag_as_tool` if you also want the agent to
search the memory store directly.

### Remaining recipes (still pure composition)

Some patterns remain user-side compositions rather than canned helpers:

| Pattern | How to compose |
|---|---|
| Debate | Two workers with opposing prompts; judge agent (third role) selects. Two `ReflectionLoop` instances + a chooser. |
| Parallel fan-out | Supervisor splits task; `asyncio.gather` over `AsyncReActAgent` workers; reducer agent or plain Python aggregates. Or build a parallel-DAG `Workflow` and let the runtime fan out. |

Ship the recipe when you write the first real one -- the primitives are
documented; preemptive helpers risk locking in shapes before they've
been exercised against real tasks.

## Workflows

DAG-based orchestration on top of the agent primitives. A `Workflow`
declares typed state, named nodes (Python callables, agents, tools,
or other workflows), and edges (static or conditional). The runtime
topologically sorts the graph, runs independent nodes in parallel,
threads state updates between them, and emits events for every node
boundary.

Workflows expose their own kwargs-only API (`flow.run(**state)` →
`WorkflowResult`). To plug a workflow into the multi-agent layer,
call `flow.as_agent()` — the returned adapter satisfies
`AgentProtocol` (`run(task: str) → AgentResult`) and binds `task`
to the workflow's `state[task_param]`. Workflows nested inside
other workflows forward their events into the outer event stream
with `source` / `parent_event_id` set so streaming UIs can render
nested execution.

Two authoring layers, same runtime:

```python
# Layer C -- decorator sugar; dependencies inferred from parameter names.
from cyllama.agents.workflow import Workflow

flow = Workflow()

@flow.node
def fetch(url: str) -> str:
    return requests.get(url).text

@flow.node
def extract(fetch: str) -> list[str]:
    return re.findall(r"pattern", fetch)

@flow.node
def summarize(extract: list[str]) -> str:
    return llm(f"Summarize: {extract}").text

flow.set_entry("fetch")
flow.set_exit("summarize")

result = flow.run(url="https://example.com")
print(result.answer)  # AgentResult-shape adapter -> state["summarize"]
```

```python
# Layer B -- explicit StateGraph; the canonical runtime form.
from cyllama.agents.workflow import Workflow, END

flow = Workflow()
flow.add_node("classify", classify_fn)
flow.add_node("answer", answer_fn)
flow.add_node("escalate", escalate_fn)
flow.add_conditional_edge(
    "classify",
    lambda s: "answer" if s["confidence"] > 0.7 else "escalate",
    {"answer": "answer", "escalate": "escalate"},
)
flow.set_entry("classify")
result = flow.run(query="What is the capital of France?")
```

Capabilities (Phases 1-5, all landed):

- **Layer B + Layer C** -- explicit StateGraph or decorator sugar; the
  two coexist on the same `Workflow` and interop freely.
- **Parallel execution** -- nodes at the same topological level run
  concurrently via `asyncio.gather`; sync bodies dispatched on
  `asyncio.to_thread`.
- **Conditional routing** -- router callbacks return either a target
  node name or the `END` sentinel; supports `edge_map` for explicit
  enumeration.
- **Streaming + events** -- `flow.stream(...)` and `flow.astream(...)`
  yield `WORKFLOW_START`, `NODE_START`, `NODE_END`, `ANSWER`,
  `WORKFLOW_END`, and `CONTRACT_VIOLATION` events.
- **Contracts** -- `WorkflowInvariant` predicates check the running
  state after each node; same `ContractPolicy` semantics as
  `ContractAgent` (IGNORE / OBSERVE / ENFORCE / QUICK_ENFORCE) with
  per-invariant policy overrides.
- **Reducers** -- multi-writer state keys require an explicit reducer
  registered via `Workflow(reducers={"key": reducer.append})`; the
  built-in namespace exposes `append`, `extend`, `merge_dict`, `add`,
  and `last`. Runtime detects unreduced multi-writer collisions and
  raises `WorkflowExecutionError`.
- **AgentProtocol compliance via adapter** -- `flow.run(**kwargs)`
  is the workflow's native API; `flow.as_agent().run("task")` is the
  `AgentProtocol` shape (binds `task` to `state[task_param]`, returns
  `AgentResult`). The explicit adapter keeps the native and protocol
  shapes from colliding. `WorkflowResult` also exposes `answer` /
  `steps` / `iterations` convenience properties so direct callers
  can read the projected output without going through the adapter.
- **Sub-workflows** -- `workflow_node(inner, name="research")` wraps
  one workflow as a node in another; inner events forwarded with
  source/parent_event_id rewriting.
- **Visualization** -- `flow.to_mermaid()` / `flow.to_dot()` render
  the graph for docs; `flow.dry_run()` returns a `DryRunPlan` showing
  topological levels without executing.
- **Typed state** -- `Workflow[State]` is a PEP 484 generic; pass a
  `TypedDict` to `Workflow(StateSchema)` for static type-checking of
  state access.

For the full design and per-phase landing notes, see
[`agents/workflow.md`](agents/workflow.md). The pattern catalog in
[`agents/patterns.md`](agents/patterns.md) §9 documents the
workflow / state-machine pattern.

## Events and Results

### Event Types

Agents emit events during execution:

```python
from cyllama.agents import EventType

class EventType(Enum):
    THOUGHT = "thought"           # Agent reasoning
    ACTION = "action"             # Tool invocation
    OBSERVATION = "observation"   # Tool result
    ANSWER = "answer"             # Final answer (also emitted by workflows
                                  # just before WORKFLOW_END so AgentProtocol
                                  # consumers see the canonical "done" event)
    ERROR = "error"               # Error occurred
    CONTRACT_CHECK = "contract_check"         # Contract being evaluated
    CONTRACT_VIOLATION = "contract_violation" # Violation detected
    # Workflow-orchestration events (cyllama.agents.workflow):
    WORKFLOW_START = "workflow_start"  # Start of a workflow run
    WORKFLOW_END = "workflow_end"      # End of a workflow run; metadata
                                       # carries final state + metrics
    NODE_START = "node_start"          # Workflow node about to execute
    NODE_END = "node_end"              # Workflow node completed; metadata
                                       # carries the state update + event_id
```

`AgentEvent` carries two optional fields populated by the multi-agent composition layer (default `None` in single-agent code paths, so existing event consumers are unaffected):

- `source: Optional[str]` -- name of the sub-agent that emitted the event when forwarded through `agent_as_tool`'s `forward_events` callback.
- `parent_event_id: Optional[str]` -- links the sub-event back to the supervisor's ACTION event that triggered the sub-run; used by streaming UIs to render nested execution.

### Streaming Events

```python
for event in agent.stream("What is 2+2?"):
    if event.type == EventType.THOUGHT:
        print(f"Thinking: {event.content}")
    elif event.type == EventType.ACTION:
        print(f"Calling: {event.content}")
    elif event.type == EventType.OBSERVATION:
        print(f"Result: {event.content}")
    elif event.type == EventType.ANSWER:
        print(f"Answer: {event.content}")
```

### AgentResult

```python
result = agent.run("What is 2+2?")

print(result.answer)      # Final answer string
print(result.success)     # True if completed without error
print(result.error)       # Error message if failed
print(result.iterations)  # Number of iterations
print(result.steps)       # List of AgentEvent
print(result.metrics)     # AgentMetrics (timing, counts)
```

### AgentMetrics

```python
metrics = result.metrics
print(f"Total time: {metrics.total_time_ms}ms")
print(f"Iterations: {metrics.iterations}")
print(f"Tool calls: {metrics.tool_calls}")
print(f"Generation time: {metrics.generation_time_ms}ms")
print(f"Tool time: {metrics.tool_time_ms}ms")
print(f"Loop detected: {metrics.loop_detected}")
print(f"Errors: {metrics.error_count}")
```

## Configuration

### GenerationConfig (ReActAgent)

```python
from cyllama import GenerationConfig

config = GenerationConfig(
    temperature=0.7,
    max_tokens=512,
    top_k=40,
    top_p=0.95,
    min_p=0.05,
    stop_sequences=["Observation:"],
)

agent = ReActAgent(llm=llm, tools=tools, generation_config=config)
```

### ConstrainedGenerationConfig

```python
from cyllama.agents.constrained import ConstrainedGenerationConfig

config = ConstrainedGenerationConfig(
    temperature=0.7,
    max_tokens=512,
    top_k=40,
    top_p=0.95,
    min_p=0.05,
)

agent = ConstrainedAgent(llm=llm, tools=tools, generation_config=config)
```

## Best Practices

### 1. Choose the Right Agent

| Use Case | Recommended Agent |
|----------|-------------------|
| General-purpose tasks | ReActAgent |
| Smaller models | ConstrainedAgent |
| Critical applications | ContractAgent |
| Debugging/explainability | ReActAgent (verbose) |
| High reliability required | ConstrainedAgent + ContractAgent |

### 2. Tool Design

```python
# Good: Clear description, typed parameters, docstring
@tool
def search_database(query: str, limit: int = 10) -> str:
    """
    Search the database for records matching query.

    Args:
        query: Search term
        limit: Maximum results to return
    """
    return db.search(query, limit)

# Bad: Vague description, no types
@tool
def search(q):
    return db.search(q)
```

### 3. Error Handling

```python
@tool
def risky_operation(data: str) -> str:
    """Perform operation that might fail."""
    try:
        result = process(data)
        return f"Success: {result}"
    except ValueError as e:
        return f"Error: Invalid data - {e}"
    except Exception as e:
        return f"Error: {e}"
```

### 4. Loop Prevention

Configure loop detection to prevent infinite loops:

```python
agent = ReActAgent(
    llm=llm,
    tools=tools,
    detect_loops=True,
    max_consecutive_same_action=2,  # Same exact action
    max_consecutive_same_tool=4,    # Same tool with any args
    max_iterations=10,              # Hard limit
)
```

### 5. Context Management

Prevent context overflow with truncation:

```python
agent = ReActAgent(
    llm=llm,
    tools=tools,
    max_context_chars=6000,  # Truncate older history
)
```

### 6. Contracts for Safety

Use contracts for safety-critical tools:

```python
@tool
@pre(lambda args: 0 <= args['amount'] <= 1000, "amount must be 0-1000")
@pre(lambda args: args['account_id'].isalnum(), "invalid account ID")
@post(lambda r: r.startswith("TX"), "must return transaction ID")
def transfer_funds(account_id: str, amount: float) -> str:
    """Transfer funds to account."""
    return banking_api.transfer(account_id, amount)
```

## Async Agents

For non-blocking agent execution in async applications, use the async agent wrappers.

### AsyncReActAgent

Async wrapper for ReActAgent:

```python
import asyncio
from cyllama.agents import tool
from cyllama.agents.async_agent import AsyncReActAgent

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

async def main():
    async with AsyncReActAgent(
        "models/Llama-3.2-1B-Instruct-Q8_0.gguf",
        tools=[search],
        max_iterations=5
    ) as agent:
        # Async run
        result = await agent.run("Search for Python tutorials")
        print(result.answer)

        # Async streaming
        async for event in agent.stream("Find information about AI"):
            print(f"{event.type.value}: {event.content}")

asyncio.run(main())
```

### AsyncConstrainedAgent

Async wrapper for ConstrainedAgent:

```python
from cyllama.agents import tool
from cyllama.agents.async_agent import AsyncConstrainedAgent

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

async def main():
    async with AsyncConstrainedAgent(
        "models/Llama-3.2-1B-Instruct-Q8_0.gguf",
        tools=[calculate]
    ) as agent:
        result = await agent.run("What is 100 / 4?")
        print(result.answer)

asyncio.run(main())
```

### run_agent_async()

Helper function to run any synchronous agent asynchronously:

```python
from cyllama import LLM
from cyllama.agents import ReActAgent, tool
from cyllama.agents.async_agent import run_agent_async

@tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Create sync agent
llm = LLM("models/Llama-3.2-1B-Instruct-Q8_0.gguf")
agent = ReActAgent(llm=llm, tools=[greet])

# Run asynchronously
async def main():
    result = await run_agent_async(agent, "Greet Alice")
    print(result.answer)

asyncio.run(main())
```

### Async Agent Parameters

Both `AsyncReActAgent` and `AsyncConstrainedAgent` accept the same parameters as their synchronous counterparts:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | required | Path to GGUF model file |
| `tools` | `List[Tool]` | `None` | Available tools |
| `config` | `GenerationConfig` | `None` | LLM configuration |
| `system_prompt` | `str` | default | Custom system prompt |
| `max_iterations` | `int` | `10` | Maximum iterations |
| `verbose` | `bool` | `False` | Print output |

### Thread Safety

Async agents use an internal lock to serialize access, ensuring thread-safe operation. For true parallel execution, create multiple agent instances:

```python
async def parallel_agents():
    async with AsyncReActAgent("model.gguf", tools=tools) as agent1, \
               AsyncReActAgent("model.gguf", tools=tools) as agent2:

        task1 = agent1.run("Task 1")
        task2 = agent2.run("Task 2")

        results = await asyncio.gather(task1, task2)
```

## References

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

- [C++26 Contract Assertions](https://en.cppreference.com/w/cpp/language/contracts.html)

- [Contracts for C++ P2900R14](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2900r14.pdf)

## Further Reading

- [`docs/agents/contracts.md`](agents/contracts.md) -- nine worked recipes for the contract patterns (cross-field, state-dependent, behavioural postconditions, cost caps, answer-content checks, PII defence, stuck-loop detection, time budgets) plus three explicit anti-patterns
- [`docs/dev/contract-agent.md`](dev/contract-agent.md) -- design notes and repositioning rationale for ContractAgent; reads like a maintainer's commentary on why the API looks the way it does

## Experimental: ACPAgent

`ACPAgent` (and `serve_acp`) implement the Agent Client Protocol for editor integration (Zed, Neovim, ...). The module is **experimental**:

- `ACP_PROTOCOL_VERSION` is hardcoded to `"2025-01-01"`; no version negotiation against the client's announced version.
- No conformance test against a reference ACP client.
- API surface may change as the protocol stabilizes and real editor integrations exercise the rough edges.

Build on it for prototypes and editor experiments; do not build a production integration on this surface without expecting churn. See the warning in `src/cyllama/agents/acp.py` module docstring for the full statement.
