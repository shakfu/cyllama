# DAG-based agent workflow orchestration

**Implementation specification for `src/cyllama/agents/workflow.py`.** This document is the design that will be built. Ports byte-identical to `inferna` modulo namespace.

## 1. Thesis

The framework ships **two layered APIs** for the same underlying runtime:

- **Layer B (explicit StateGraph):** a LangGraph-style typed state-graph: explicit nodes, explicit edges, explicit state schema. This is the **canonical runtime model**. Every workflow ultimately compiles to this.

- **Layer C (decorated functions):** a higher-level decorator API that desugars to Layer B. Parameter names bind from state; function return values update state; dependencies between nodes are inferred from matching parameter names. Concise for the common case.

The two layers are interoperable. A single workflow can mix decorator nodes (Layer C) with explicit `add_node` / `add_edge` calls (Layer B) on the same `Workflow` object. The decorator is sugar; the underlying graph is one shape.

## 2. Goals

1. Express any directed acyclic agent workflow with typed state, parallel execution of independent nodes, and conditional routing.

2. Provide a single canonical runtime model (Layer B) so every workflow has the same execution semantics regardless of authoring style.

3. Provide a concise decorator API (Layer C) for the common case where node dependencies are obvious from data flow.

4. Integrate with the existing agent primitives (`AgentProtocol`, `Tool`, `ContractPolicy`, `AgentEvent`, `SemanticMemory`) without re-implementing them.

5. Zero new runtime dependencies. Pure stdlib only (asyncio, typing, dataclasses, contextvars).

6. Synchronous-default API; async variant available; matches the `ReActAgent` / `AsyncReActAgent` pattern.

7. Single-process. Cross-process sub-agents are already handled by `mcp_agent_tool`.

## 3. Three shapes weighed

For completeness, the three design shapes considered:

### Shape A — Prefect-style implicit DAG from data flow

User writes a workflow function; the runtime traces data dependencies between decorated tasks. The DAG is inferred from `task_b(task_a())` patterns in plain Python.

```python
@workflow_task
def search(query): ...

@workflow_task
def summarize(docs): ...

@workflow
def research_flow(query):
    return summarize(search(query))
```

Inference is fragile: conditionals, loops, and assignments in the workflow function need special handling. Hard to introspect the graph before running, hard to visualize, hard to checkpoint (state is implicit in the Python stack). Rejected for cyllama.

### Shape B — Explicit StateGraph

User defines a state schema, declares nodes that receive and update state, declares edges (static or conditional). Compile then run.

```python
class State(TypedDict):
    query: str
    docs: list[str]
    answer: str

flow = Workflow(State)
flow.add_node("search", search_fn)
flow.add_node("summarize", summarize_fn)
flow.add_edge("search", "summarize")
flow.set_entry("search")
flow.set_exit("summarize")

compiled = flow.compile()
result = compiled.run({"query": "..."})
```

Verbose but explicit. Graph is a first-class object. Easy to introspect, visualize, validate, checkpoint. Conditional edges are natural. The canonical model.

### Shape C — Decorated functions over a Workflow

Nodes are decorated functions whose parameters bind from state by name. Dependencies between nodes come from parameter names that match other nodes' names.

```python
flow = Workflow()

@flow.node
def search(query: str) -> list[str]:
    return rag.search(query)

@flow.node
def summarize(search: list[str]) -> str:
    return llm(...).text

result = flow.run(query="...")
```

Concise. Type-checkable: parameters and returns are real types. Dependencies visible from a casual read.

### Choice: B is the runtime, C is sugar

The doc commits to **building both, with C compiling to B**. Every `@flow.node` decorator emits the same `Workflow.add_node` + `Workflow.add_edge` calls a user could write by hand. Every workflow authored in pure Layer B works without the decorator. Mixed authoring on the same `Workflow` object is supported.

The reason for the layering:

- **B alone** is verbose for the 80% case (single-state-key per node, parameter-name = upstream-node-name).

- **C alone** can't express the 20% case (custom state schemas with multiple keys per node, reducer-based merging, hand-written routers with complex edge maps).

- **B + C** lets users start in C and drop to B when the shape doesn't fit. The runtime is one model regardless.

## 4. State model

State is a `TypedDict` (or plain `dict`) threaded through every node. Each node receives the current state and returns a **partial update** that gets shallow-merged into the running state.

### 4.1 Explicit state schema (Layer B)

```python
from typing import TypedDict

class ResearchState(TypedDict):
    query: str
    docs: list[str]
    summary: str
    error: str | None

flow = Workflow(ResearchState)
```

The state class is the workflow's I/O contract:

- Initial state passed to `run()` must satisfy the schema (any required keys; mypy can check this if the user enables type checking).

- Each node returns a `dict` containing a subset of state keys.

- Final state matches the schema.

When no schema is supplied (`Workflow()` without an argument), state is a plain `dict[str, Any]` with no validation.

### 4.2 Implicit state schema (Layer C)

When using only the decorator API, the state schema is **derived**: each node's name becomes a state key whose value type is the node's return type. So:

```python
@flow.node
def search(query: str) -> list[str]: ...
```

is equivalent to:

```python
class _AutoState(TypedDict):
    query: str
    search: list[str]
```

`query` is in the schema because some node consumes it; `search` is in the schema because the node named `search` produces it. The derived schema is available on `flow.derived_state_schema` after compilation.

### 4.3 Reducer semantics

By default, each state key has **one writer** — a node returning a key already written by another node raises `WorkflowDefinitionError` at compile time. This is the safest default and what the decorator API enforces.

For multi-writer keys (e.g., a `messages` list aggregated across nodes), declare a reducer on the state field:

```python
from cyllama.agents.workflow import Workflow, reducer

class State(TypedDict):
    messages: list[str]

flow = Workflow(State, reducers={"messages": reducer.append})
```

Built-in reducers (in `cyllama.agents.workflow.reducer`):

- `reducer.append` — append to list

- `reducer.extend` — extend list with another list

- `reducer.merge_dict` — dict.update style

- `reducer.add` — numeric accumulation

- `reducer.last` — last-writer-wins (explicit override of the default conflict error)

Custom reducers: any `Callable[[Existing, Update], Merged]`.

## 5. Layer B specification

Full canonical API. Everything else compiles to this.

### 5.1 `Workflow[StateT]` class

```python
class Workflow(Generic[StateT]):
    def __init__(
        self,
        state_schema: type[StateT] | None = None,
        *,
        reducers: dict[str, Reducer] | None = None,
    ) -> None: ...

    # Static graph building
    def add_node(
        self,
        name: str,
        fn: Callable[[StateT], dict],
        *,
        timeout: float | None = None,
    ) -> None: ...

    def add_edge(self, from_node: str, to_node: str) -> None: ...

    def add_conditional_edge(
        self,
        from_node: str,
        router: Callable[[StateT], str],
        edge_map: dict[str, str] | None = None,
    ) -> None: ...

    def set_entry(self, name: str) -> None: ...
    def set_exit(self, name: str) -> None: ...  # or set_exits([...]) for multi-exit

    # Decorator API (Layer C; section 6)
    def node(self, fn: Callable | None = None, *, name: str | None = None,
             timeout: float | None = None) -> Callable: ...
    def route(self, after: str) -> Callable: ...

    # Inspection
    @property
    def nodes(self) -> dict[str, Node]: ...
    @property
    def edges(self) -> list[Edge]: ...
    @property
    def conditional_edges(self) -> list[ConditionalEdge]: ...
    @property
    def derived_state_schema(self) -> type[TypedDict]: ...

    # Compile + run
    def compile(self) -> "CompiledWorkflow[StateT]": ...
    def run(self, **initial_state) -> "WorkflowResult[StateT]": ...
    def arun(self, **initial_state) -> Awaitable["WorkflowResult[StateT]"]: ...
    def stream(self, **initial_state) -> Iterator[AgentEvent]: ...

    # Visualization
    def to_mermaid(self) -> str: ...
    def to_dot(self) -> str: ...
    def dry_run(self, **initial_state) -> "DryRunPlan": ...
```

`run()` is a thin sync wrapper over `arun()` (uses `asyncio.run`). Calling `run()` or `stream()` on an uncompiled workflow lazily compiles first; the compile step is idempotent.

### 5.2 Node functions (Layer B)

A Layer-B node has the signature:

```python
def node(state: StateT) -> dict: ...
```

It returns a `dict` whose keys are state keys to update. Returning `{}` is valid (no state change). Returning a key not in the schema is a runtime error (unless `state_schema=None`).

`async def` nodes are supported and awaited; sync nodes run on `asyncio.to_thread` so the event loop isn't blocked.

### 5.3 Edges and routing

Two edge kinds:

**Static edge.** `add_edge("a", "b")` declares "`b` runs after `a`". A node may have multiple incoming static edges (all must complete before it runs) and multiple outgoing static edges (all run after it completes).

**Conditional edge.** `add_conditional_edge(from_node, router, edge_map)` declares "after `from_node` runs, call `router(state)`, take its return value, look up the target node in `edge_map`".

```python
def route(state: State) -> str:
    return "summarize" if state["docs"] else "fallback"

flow.add_conditional_edge("search", route, {
    "summarize": "summarize",
    "fallback": "fallback",
})
```

The `edge_map` argument is optional. If omitted, the router's return value is used directly as the next node's name. Routers may return the sentinel `END` (importable from `cyllama.agents.workflow`) to terminate this branch without going to another node.

A node may have *either* static outgoing edges *or* conditional outgoing edges, not both — keeps the routing model deterministic. Mixing them at compile time raises `WorkflowDefinitionError`.

### 5.4 Entry and exit

`set_entry(name)` declares the workflow's start node. Exactly one entry is required.

`set_exit(name)` declares an exit node. Multiple calls register multiple exits — the workflow terminates whenever any exit node completes (useful for branching workflows where each branch ends at its own exit).

If `set_exit` is never called, all sink nodes (nodes with no outgoing edges) are exits.

### 5.5 Compile

`Workflow.compile()` runs validation and returns a `CompiledWorkflow`:

- All referenced node names exist (in `add_edge`, `add_conditional_edge`, `set_entry`, `set_exit`, and in `edge_map` values).

- Entry node is set.

- Graph (over static edges) is acyclic — topological sort succeeds. Conditional edges *can* create cycles (a router could route back to an earlier node), but the cycle bound is the workflow's configurable `max_steps` (default 100); see §7.5.

- For each state key, at most one writer node (unless a reducer is registered for that key).

- All node return types (declared via type hints when available) match the state schema field types.

Compilation is the only failure mode for definition errors — runtime errors are different (§9).

### 5.6 Run, stream, arun

The workflow's native API is **kwargs-only**:

```python
# Canonical: kwargs map to state.
result = flow.run(query="Treaty of Westphalia")
# WorkflowResult(state=..., events=..., success=True, error=None,
#                metrics=AgentMetrics(...))

# CompiledWorkflow takes a single optional positional dict:
result = compiled.run({"query": "Treaty of Westphalia"})
```

`Workflow.run(**kwargs)` → `WorkflowResult`. Same kwargs-only shape for `arun` (async), `stream` (sync iterator over `AgentEvent`), and `astream` (async iterator). `CompiledWorkflow` takes the dict positionally because compiled callers usually have the state pre-built.

`stream(...)` yields `AgentEvent`s as the workflow executes (§8). The stream emits `EventType.ANSWER` just before `EventType.WORKFLOW_END`, carrying the resolved answer string (the value at `state[answer_key]`, or the last-run node's state entry when `answer_key` is unset). `WORKFLOW_END` carries the full state under `metadata["state"]`.

The `metrics` property on `Workflow` delegates to the cached `CompiledWorkflow`, which stashes the most-recent run's `AgentMetrics` on the (frozen) instance via `object.__setattr__`. Both run paths (`Workflow.run` → `CompiledWorkflow.arun`, or direct `CompiledWorkflow.run`) update the same cache; the sync `stream()` path records metrics on `WORKFLOW_END`.

The `answer_key` (set via the `Workflow(answer_key=...)` constructor kwarg, or implicitly to the sole exit when one is registered) controls what `WorkflowResult.answer` returns and what content lands on the final `ANSWER` event.

#### AgentProtocol compliance via `as_agent()`

A workflow's native shape is "typed state in, typed state out"; the `AgentProtocol` shape (`run(task: str) -> AgentResult`) is a lossy projection of that. Rather than overloading `run` to mean both things, `Workflow` and `CompiledWorkflow` expose `as_agent()` returning a small adapter that satisfies `AgentProtocol`:

```python
flow = Workflow()
# ... build flow ...
flow.set_exit("answer")

agent = flow.as_agent()                 # AgentProtocol-conformant
result = agent.run("hello world")       # binds to state[task_param]
                                        #  -> AgentResult

# Integration points opt in explicitly:
tool = agent_as_tool(flow.as_agent(), name="research", description="...")
loop = ReflectionLoop(flow.as_agent(), critic.as_agent(), max_attempts=3)
team = TieredAgentTeam(
    supervisor=supervisor,
    workers=[AgentRole("researcher", flow.as_agent(), "..."), ...],
)
```

The adapter:

- `run(task: str) -> AgentResult` — binds `task` to `state[task_param]` (default `"task"`; configured via `Workflow(task_param=...)` or `flow.as_agent(task_param=...)`), runs the workflow, projects the final state to an `AgentResult`.

- `arun(task)` — async variant.

- `stream(task)` / `astream(task)` — yield the workflow's `AgentEvent`s.

- `metrics` — delegates to the underlying compiled workflow.

`isinstance(flow.as_agent(), AgentProtocol)` returns `True`; `isinstance(flow, AgentProtocol)` is structurally true (workflows have `run`/`stream`/`metrics` members) but the workflow's `run(**kwargs)` shape is *not* protocol-compatible. The adapter exists precisely so the protocol projection is explicit at the call site.

## 6. Layer C specification

The decorator API is sugar over Layer B. Every Layer-C operation desugars to specific Layer-B calls.

### 6.1 `@flow.node`

```python
@flow.node
def search(query: str) -> list[str]:
    return rag.search(query)
```

**Desugars to:**

```python
def _search_layer_b(state: dict) -> dict:
    coerced = coerce_args_from_state(state, search)  # see §6.4
    result = search(**coerced)
    return {"search": result}

flow.add_node("search", _search_layer_b)
for upstream in inferred_dependencies(search):  # see §6.3
    flow.add_edge(upstream, "search")
```

The function's `__name__` becomes the node name. Override:

```python
@flow.node(name="fast_search")
def search(query: str) -> list[str]: ...
```

The return type annotation becomes the state field type (when a state schema is being derived; §4.2).

### 6.2 `@flow.node(timeout=...)`

```python
@flow.node(timeout=10.0)
def slow_fetch(url: str) -> str: ...
```

Forwards to `Workflow.add_node(..., timeout=10.0)`. Enforced via the same daemon-thread + join pattern used by `Tool.timeout` (`react.py:_execute_tool_raw`); the node raises `ToolTimeoutError` on timeout, which the workflow converts to an error event and skips downstream nodes.

### 6.3 Dependency inference

For each parameter of a decorated node:

1. If the parameter name matches a previously-registered node name: add a static edge from that node to this one.

2. Otherwise: the parameter is a workflow input — it must be supplied via `run(**initial_state)`.

A parameter that is **neither** a registered node nor in `initial_state` at compile time raises `WorkflowDefinitionError`.

Order of decoration matters only when names collide. The same decorator on a function whose parameter shadows a real workflow input is allowed; the input wins (i.e., `initial_state["query"]` is what `search(query=...)` receives).

### 6.4 `@flow.route(after=...)`

```python
@flow.route(after="search")
def route_after_search(search: list[str]) -> str:
    return "summarize" if search else "fallback"
```

**Desugars to:**

```python
def _route_layer_b(state: dict) -> str:
    return route_after_search(search=state["search"])

flow.add_conditional_edge("search", _route_layer_b, edge_map=None)
```

The router function's parameter binding follows the same rules as node parameter binding (§6.3): each parameter pulls from state by name.

The router's return value is treated as the target node name directly (equivalent to `edge_map=None`). To use an explicit edge map, drop to Layer B and call `add_conditional_edge` directly.

### 6.5 `flow.add_node(fn, name=None)` (non-decorator form)

For reusable node functions or programmatic registration:

```python
def search(query: str) -> list[str]: ...

flow.add_node(search)  # name="search", inferred from __name__
flow.add_node(search, name="primary_search")  # explicit name
```

Same desugaring as `@flow.node`.

### 6.6 Mixing layers

Layer B and Layer C calls coexist on the same `Workflow`:

```python
flow = Workflow(State)

@flow.node  # Layer C
def search(query: str) -> list[str]: ...

# Layer B: a node with custom multi-key state output
def merge(state: State) -> dict:
    return {"summary": "...", "metadata": {"sources": len(state["search"])}}

flow.add_node("merge", merge)
flow.add_edge("search", "merge")
flow.set_entry("search")
flow.set_exit("merge")
```

When the decorator can't express the node (multi-key writes, custom state binding, edge maps with non-trivial fanout), drop to Layer B for that specific node.

## 7. Execution model

The runtime that both layers compile to. Sections 7.1-7.5 are the contract every workflow obeys.

### 7.1 Compile

`Workflow.compile()` produces a `CompiledWorkflow` whose internals are:

```python
@dataclass(frozen=True)
class CompiledWorkflow(Generic[StateT]):
    nodes: dict[str, _CompiledNode]
    static_edges: dict[str, list[str]]      # from_node -> [to_node, ...]
    conditional_edges: dict[str, _ConditionalEdge]
    entry: str
    exits: frozenset[str]
    state_schema: type[StateT] | None
    reducers: dict[str, Reducer]
    max_steps: int
    levels: list[list[str]]   # topological levels for static-edge nodes
```

`_CompiledNode` wraps the user function with type-coercion (via `coerce_args`) and timeout enforcement.

`levels` is computed once at compile time via Kahn's algorithm over static edges only. Conditional edges are not in the level structure; they're resolved at runtime.

### 7.2 Run

Execution starts at `entry` and proceeds via a hybrid level-scheduled
+ conditional-routed loop:

```python
state = initial_state
active = {entry}
visited = set()
steps = 0

while active:
    if steps >= max_steps:
        raise WorkflowExecutionError("max_steps exceeded")
    steps += 1

    # Group active nodes by their topological level.
    # Nodes at the same level run concurrently.
    batches = group_by_level(active)
    for batch in batches:
        results = await asyncio.gather(*[run_node(n, state) for n in batch])
        state = merge_all(state, results, reducers)
        visited.update(batch)

    # Compute next active set.
    next_active = set()
    for node in batch:
        # Static successors
        for succ in static_edges.get(node, []):
            if all_predecessors_visited(succ, visited):
                next_active.add(succ)
        # Conditional successor
        if node in conditional_edges:
            target = conditional_edges[node].router(state)
            target = conditional_edges[node].edge_map.get(target, target)
            if target == END or target in exits:
                continue
            next_active.add(target)

    active = next_active

# Workflow terminates when active is empty or we hit an exit.
```

In practice the implementation is more careful about fan-in synchronization (a node with multiple predecessors waits for all of them), but the shape above captures it.

### 7.3 Node execution

A single node's execution:

1. Bind args from state (Layer C: by parameter name; Layer B: pass the whole state).

2. Apply `coerce_args` to bind values to declared types (Annotated constraints enforced).

3. If `async def`, await directly; else, run on `asyncio.to_thread`.

4. If a timeout is set, race against an `asyncio.wait_for` with the declared budget; on timeout raise `ToolTimeoutError`.

5. Emit `EventType.ACTION` at start (with `metadata={"node": name, "inputs": ...}`) and `EventType.OBSERVATION` at end (with `metadata={"node": name, "result": ...}`).

6. Capture the return dict and hand back for state merging.

### 7.4 State merging

After a level completes, all node return dicts are merged into the running state:

- For each (key, value) pair in each return dict:

  - If `key` has a registered reducer, call `reducer(state[key], value)` and update state.

  - Else: state[key] = value (last-write wins within the level; but multiple writers in the same level without a reducer is a compile error per §5.5).

State merges across levels are sequential. State merges within a level are deterministic (alphabetical by node name) when no reducer is registered, but the compile-time single-writer rule means this shouldn't matter.

### 7.5 Termination

Workflow terminates when:

1. The active set is empty (all reachable nodes have run).

2. An active node is in the `exits` set (it runs to completion, then the workflow terminates).

3. A conditional router returns `END` (the path terminates; workflow continues if other paths are still active; if no paths remain, terminate).

4. A node raises (workflow terminates with `success=False`, downstream nodes skipped, error captured on `WorkflowResult`).

5. `max_steps` reached (default 100, configurable via `Workflow(max_steps=...)`). Raises `WorkflowExecutionError`. The cap exists to bound conditional-edge cycles.

## 8. Event model

Every workflow execution emits a stream of `AgentEvent`s. New event type added to `EventType`:

```python
class EventType(Enum):
    ...
    NODE_START = "node_start"
    NODE_END = "node_end"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"
```

(The existing `ACTION` / `OBSERVATION` semantics are tool-centric; workflows are level above that, so dedicated event types are clearer than overloading.)

For each workflow run, the event sequence is:

```
WORKFLOW_START               metadata={"entry": "...", "initial_state": ...}
  NODE_START (entry)         metadata={"node": "...", "inputs": ...}
  ... (sub-agent events, if the node wraps an agent, with source + parent_event_id)
  NODE_END (entry)           metadata={"node": "...", "result": ...}
  NODE_START (next level)
  ...
WORKFLOW_END                 metadata={"final_state": ..., "success": bool}
```

`source` on workflow-level events is `None` (they're emitted by the workflow itself). Sub-agent events forwarded into a workflow run carry their `source` (set by `agent_as_tool` or by the node wrapping the agent) and a `parent_event_id` linking to the enclosing `NODE_START` event.

Workflows nested inside workflows (sub-workflow as a node) preserve the hierarchy: inner `NODE_START` events carry `parent_event_id` linking to the outer `NODE_START`, and `source` carries the outer node's name.

## 9. Error handling

Three categories:

### 9.1 Compile-time errors (`WorkflowDefinitionError`)

Raised by `Workflow.compile()`:

- Unknown node name referenced in an edge / router / entry / exit.

- No entry node set.

- Cycle in static-edge graph.

- Multiple writers for the same state key without a reducer.

- Node return type incompatible with state schema field type.

- Conditional edge target node has only conditional incoming edges (over-specified).

- A node has both static outgoing and conditional outgoing edges (mixed routing; ambiguous).

### 9.2 Runtime errors (captured on `WorkflowResult`)

Raised inside a node and caught by the runtime:

- Any exception from a node body.

- `ToolTimeoutError` from a timed-out node.

- `ToolArgumentError` from `coerce_args` boundary validation.

- Conditional router returns a node name not in `edge_map` (and `edge_map` is non-None) — `WorkflowRoutingError`.

- `max_steps` exceeded — `WorkflowExecutionError`.

The workflow terminates, emits `EventType.ERROR` with the exception, and returns `WorkflowResult(success=False, error=<formatted>, ...)`. Downstream nodes are skipped.

### 9.3 Invariant violations (`ContractPolicy`)

Workflow-level invariants (§10) emit `EventType.CONTRACT_VIOLATION` events. Under `ContractPolicy.ENFORCE` the workflow terminates; under `OBSERVE` it continues.

## 10. Contracts integration

`Workflow(invariants=[...])` accepts the same `WorkflowInvariant` shape used by `ContractAgent.iteration_invariants` — but operating over workflow state instead of `IterationState`:

```python
from cyllama.agents.workflow import Workflow, WorkflowInvariant
from cyllama.agents import ContractPolicy

flow = Workflow(
    State,
    invariants=[
        WorkflowInvariant(
            predicate=lambda s: s.elapsed_ms < 60_000,
            message="workflow time < 60s",
        ),
        WorkflowInvariant(
            predicate=lambda s: s.node_count < 50,
            message="no more than 50 nodes executed",
        ),
        WorkflowInvariant(
            predicate=lambda s: s.estimated_cost_usd < 1.00,
            message="cost cap",
        ),
    ],
    policy=ContractPolicy.ENFORCE,
)
```

The `WorkflowInvariant` predicate receives a `WorkflowExecutionState` view of the running state augmented with framework-tracked counters:

```python
@dataclass
class WorkflowExecutionState:
    state: dict           # the user-facing state
    elapsed_ms: float
    node_count: int       # nodes completed so far
    error_count: int
    estimated_cost_usd: float  # sum of per-node costs if tracked
    nodes_run: list[str]  # in execution order
```

Invariants fire after each node completes. Violations follow `ContractPolicy` semantics, reusing the same `_handle_violation` logic as `ContractAgent`.

## 11. Public API surface

Complete list of names exported from `cyllama.agents.workflow` and re-exported from `cyllama.agents`.

### Classes

| Name | Description |
|---|---|
| `Workflow[StateT]` | Main entry point. Both Layer B and Layer C APIs. |
| `CompiledWorkflow[StateT]` | Result of `Workflow.compile()`. Immutable, runnable. |
| `WorkflowResult[StateT]` | Result of `compiled.run()`. Has `state`, `events`, `success`, `error`, `metrics`. |
| `WorkflowExecutionState` | State view passed to invariants and routers (when they declare it). |
| `WorkflowInvariant` | Same shape as iteration invariants in ContractAgent. |
| `DryRunPlan` | Result of `Workflow.dry_run()`. Shows execution order without running. |

### Exceptions

| Name | When raised |
|---|---|
| `WorkflowDefinitionError` | Compile-time validation failure. |
| `WorkflowExecutionError` | Runtime structural failure (max_steps, etc.). |
| `WorkflowRoutingError` | Router returned an unmapped name. |

### Sentinels and modules

| Name | Purpose |
|---|---|
| `END` | Sentinel router return value; terminates the branch. |
| `cyllama.agents.workflow.reducer` | Built-in reducers: `append`, `extend`, `merge_dict`, `add`, `last`. |

### Free functions

| Name | Purpose |
|---|---|
| `agent_node(agent, name, task_param="task", *, forward_events=True)` | Build a Layer-B node from an `AgentProtocol`. With `forward_events=True` (default), inner agent events are buffered and interleaved into the outer workflow event stream with `source` + `parent_event_id` set. |
| `tool_node(tool, name)` | Build a Layer-B node from a `Tool`. |
| `workflow_node(compiled, name, *, task_param="task", project_state=False)` | Wrap a `CompiledWorkflow` (or `Workflow`, auto-compiled) as a node inside another workflow. Inner events forwarded with source/parent rewriting; inner failures re-raise as `WorkflowExecutionError`. |

All names are added to `cyllama.agents.__init__.py` `__all__`.

## 12. File layout (as landed)

```
src/cyllama/agents/
  workflow.py            # main module (~2100 LoC across all five phases)
                         # includes the `reducer` namespace inline
                         # rather than a separate workflow_reducers.py
  __init__.py            # re-exports

tests/
  test_agents_workflow.py  # all 118 workflow tests in a single file,
                           # organized by phase + feature (TestReducers,
                           # TestWorkflowInvariants, TestAgentProtocolCompliance,
                           # TestAgentNodeStreaming, TestWorkflowNode,
                           # TestAgentAsToolIntegration,
                           # TestReflectionLoopIntegration)

docs/
  agents/workflow.md     # this design doc, kept as the implementation reference
```

The originally-planned `workflow-user.md` and the planned three-file test split did not ship; the single test file stays navigable at 1800 lines, and the patterns doc (`docs/agents/patterns.md`) covers the user-tutorial role.

## 13. Implementation phases

**All five phases have landed.** Each phase shipped as a self-contained landing with its own tests, CHANGELOG entry, and validation pass.

### Phase 1 — Layer B core (Workflow + compile + run) — **landed**

- `Workflow.__init__` with optional state schema.

- `add_node`, `add_edge`, `add_conditional_edge`, `set_entry`, `set_exit`.

- `compile()` — validation + topological levels.

- `CompiledWorkflow.run()` / `arun()` — sequential + level-parallel execution with conditional routing.

- `WorkflowResult` dataclass.

- `WorkflowDefinitionError`, `WorkflowExecutionError`, `WorkflowRoutingError`.

- `END` sentinel.

- Sync nodes (via `asyncio.to_thread`), async nodes (awaited directly).

- ~400 LoC + 25 tests.

### Phase 2 — Layer C sugar (decorators + inference) — **landed**

- `@flow.node` and `@flow.node(name=..., timeout=...)`.

- `@flow.route(after=...)`.

- `flow.add_node(fn)` non-decorator form.

- Dependency inference from parameter names.

- Derived state schema (`flow.derived_state_schema`).

- ~150 LoC + 20 tests.

### Phase 3 — Helpers, streaming, visualization — **landed**

- `agent_node()` and `tool_node()` adapters.

- `stream()` method on `Workflow` / `CompiledWorkflow`.

- New `EventType` variants (`NODE_START`, `NODE_END`, `WORKFLOW_START`, `WORKFLOW_END`).

- `to_mermaid()`, `to_dot()`, `dry_run()`.

- ~150 LoC + 15 tests.

### Phase 4 — Contracts, reducers, typed state — **landed**

- `WorkflowInvariant` + `WorkflowExecutionState`.

- `ContractPolicy` integration; `CONTRACT_VIOLATION` events emitted during workflow runs.

- Per-node `timeout=`.

- Built-in reducers (`append`, `extend`, `merge_dict`, `add`, `last`).

- Custom reducer registration.

- Generic `Workflow[StateT]` typing (PEP 484 compatible).

- ~250 LoC + 25 tests.

### Phase 5 — `AgentProtocol` compliance via `as_agent()`, sub-workflow composition — **landed**

- `Workflow.as_agent()` and `CompiledWorkflow.as_agent()` return a `_WorkflowAgentAdapter` that satisfies `AgentProtocol` (`run(task: str) -> AgentResult`, `arun(task)`, `stream(task)`, `astream(task)`, `metrics` property). Callers opt in at the protocol boundary: `agent_as_tool(flow.as_agent(), ...)`, `ReflectionLoop(worker.as_agent(), critic.as_agent(), ...)`, `TieredAgentTeam(workers=[AgentRole("name", flow.as_agent(), ...)])`.

- The workflow's native API stays **kwargs-only**: `flow.run(**kwargs)` returns `WorkflowResult`. No polymorphic positional overloading. This trades one extra method call at protocol boundaries for an unambiguous native API — `flow.run` always means "kwargs in, WorkflowResult out".

- `_WorkflowAgentAdapter.run(task)` projects the final workflow state to an `AgentResult` by reading `state[answer_key]` (set via `Workflow(answer_key=...)`, or implicitly to the sole exit when one is registered). The adapter accepts a per-call override: `flow.as_agent(task_param="query", answer_key="response")`.

- `WorkflowResult` exposes convenience properties (`answer` / `steps` / `iterations`) so direct callers can read the projected output without going through the adapter; same projection rules.

- Final `EventType.ANSWER` event emitted just before `WORKFLOW_END`, carrying the resolved answer string — so consumers that only watch for `ANSWER` (e.g. `ReflectionLoop.stream`) see the canonical "I'm done" event without walking metadata.

- `metrics` property on `Workflow` delegates to the cached `CompiledWorkflow`, which stores the most-recent run's `AgentMetrics` via `object.__setattr__` on the frozen dataclass. Both run paths (`Workflow.run` → `CompiledWorkflow.arun`, and direct `CompiledWorkflow.run`) update the same cache; sync `stream()` records metrics on `WORKFLOW_END`. The adapter delegates to this property.

- `NODE_START` and `NODE_END` events carry a stable `metadata["event_id"]` (uuid4) so consumers can reconstruct the nesting tree.

- `agent_node(agent, name, task_param="task", *, forward_events=True)` default flipped to stream-based dispatch: events from the inner agent stream live (not batched) into the outer event stream between `NODE_START` and `NODE_END` with `source=name` and `parent_event_id=<NODE_START event_id>`. The runtime sets a per-call `_event_sink` attribute on streaming node bodies, then drains a shared `asyncio.Queue` concurrently with the node tasks (one DONE sentinel per node) so each event yields the instant it lands. Sync bodies hand events across thread boundaries via `loop.call_soon_threadsafe`; async bodies use plain `await queue.put`. Pass `forward_events=False` for the original fast (run-based) path.

- New `workflow_node(compiled, name, *, task_param="task", project_state=False)` free function wraps another `CompiledWorkflow` (or `Workflow`, auto-compiled) as a node in an outer workflow. Inner events forwarded with `source` / `parent_event_id` rewriting (preserving any deeper nesting set by the inner workflow). Inner failures re-raise as `WorkflowExecutionError` on the outer (captured into `WorkflowResult.error`). `project_state=True` exposes the entire final sub-state under `state[name]`; default surfaces only the inner `answer_key` value (AgentResult-shape).

- Internal `_run_one` dispatch extended to detect async-`__call__` callable instances (`asyncio.iscoroutinefunction(node.fn.__call__)`) so `_SubWorkflowNodeBody` and similar stateful node bodies dispatch via the async path instead of being incorrectly run on `asyncio.to_thread`.

- ~250 LoC + 21 tests (across `TestAgentProtocolCompliance`, `TestAgentNodeStreaming`, `TestWorkflowNode`, `TestAgentAsToolIntegration`, `TestReflectionLoopIntegration`).

### Out-of-scope (not in this implementation)

The following are explicitly **out of scope** for the initial implementation. They land as separate proposals when forced by use cases:

- Durable execution across process restarts (would need deterministic re-execution semantics).

- Distributed execution across hosts.

- Automatic checkpointing to a `SessionStore` (in-process state preservation works; the user can call `result.state` themselves).

- Human-approval gate nodes (composable as a regular node that calls `input()` or a UI hook; no special primitive).

- Bounded retry loops on individual nodes (composable by writing a retry wrapper as a node body; no special primitive).

- Dynamic graph extension at runtime (a node deciding to add a new node based on its output).

## 14. Documentation updates

When the implementation lands:

- `docs/agents/workflow.md` (this doc) stays as the maintainer reference.

- `docs/agents/workflow-user.md` is new — five-example user tutorial.

- `docs/agents/patterns.md` §9 flipped from "not supported" to "first-class"; recipe shows a Layer-C workflow that branches on data.

- `docs/agents_overview.md` gains a "Workflows" section that references the user tutorial and lists the Phase 1-5 capability matrix.

- `CHANGELOG.md` `[Unreleased]` section in both repos: one entry per phase landing (5 entries total), each in the existing prose style.

- `TODO.md` pattern-gaps "not on the roadmap" entry for Workflow/State-Machine is removed.

## 15. Worked examples

Five end-to-end examples that drive the design. The implementation must support all five; the user tutorial walks through them.

### 15.1 Linear pipeline (Layer C)

```python
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
assert result.success
print(result.state["summarize"])
```

### 15.2 Parallel fan-out + join (Layer C)

```python
flow = Workflow()

@flow.node
def search_wikipedia(query: str) -> str: ...

@flow.node
def search_local_docs(query: str) -> str: ...

@flow.node
def calculator(query: str) -> str: ...

@flow.node
def synthesize(
    search_wikipedia: str,
    search_local_docs: str,
    calculator: str,
) -> str:
    return llm(combine(...)).text

flow.set_entry("search_wikipedia")  # one of the three; the others run in parallel
flow.set_exit("synthesize")

result = flow.run(query="...")
```

(The three searches run concurrently because none depends on the others; `synthesize` waits for all three.)

### 15.3 Conditional branching (Layer C)

```python
flow = Workflow()

@flow.node
def search(query: str) -> list[str]:
    return rag.search(query)

@flow.node
def summarize(search: list[str]) -> str: ...

@flow.node
def fallback(query: str) -> str:
    return "no results found"

@flow.route(after="search")
def route_after_search(search: list[str]) -> str:
    return "summarize" if search else "fallback"

flow.set_entry("search")
flow.set_exit("summarize")
flow.set_exit("fallback")

result = flow.run(query="...")
```

### 15.4 Explicit StateGraph with reducer (Layer B)

```python
from typing import TypedDict
from cyllama.agents.workflow import Workflow, agent_node, reducer
from cyllama.agents import ReActAgent

class State(TypedDict):
    task: str
    messages: list[str]   # multi-writer
    final: str

flow = Workflow(State, reducers={"messages": reducer.append})

researcher_agent = ReActAgent(...)
coder_agent = ReActAgent(...)

flow.add_node("researcher", agent_node(researcher_agent, name="researcher",
                                       task_param="task"))
flow.add_node("coder", agent_node(coder_agent, name="coder", task_param="task"))

def finalize(state: State) -> dict:
    return {"final": "\n".join(state["messages"])}

flow.add_node("finalize", finalize)
flow.add_edge("researcher", "finalize")
flow.add_edge("coder", "finalize")
flow.set_entry("researcher")  # second entry not supported; use a router from a dummy entry
flow.set_exit("finalize")

result = flow.compile().run({"task": "..."})
```

### 15.5 Workflow as a sub-agent

```python
inner_flow = Workflow()
# ... build inner_flow as in 15.1 ...

# Now use it as a worker in a higher-level multi-agent setup:
from cyllama.agents import TieredAgentTeam, AgentRole, ReActAgent

team = TieredAgentTeam(
    supervisor=ReActAgent(llm=LLM("strong.gguf"), tools=[]),
    workers=[
        AgentRole(
            name="research_workflow",
            agent=inner_flow.as_agent(),  # explicit AgentProtocol adapter
            description="Run the multi-step research pipeline.",
        ),
    ],
)

result = team.run("Compare X and Y in technical depth.")
```

`inner_flow.run(**kwargs)` is the workflow's native API — kwargs map to state. To plug a workflow into the multi-agent layer (`agent_as_tool`, `TieredAgentTeam`, `ReflectionLoop`), call `inner_flow.as_agent()` first; the returned adapter satisfies `AgentProtocol` (`run(task: str) -> AgentResult`) and binds `task` to the inner workflow's `state[task_param]` (default `"task"`, override via `Workflow(task_param="prompt")` or `inner_flow.as_agent(task_param="prompt")`).

For nesting one workflow inside another with full event forwarding, prefer `workflow_node(inner, name="research")` over the `.as_agent()` adapter — it preserves the inner workflow's structured state surface and forwards the inner event stream into the outer workflow's events with `source` / `parent_event_id` set.

## 16. Validation

Each phase's landing must pass:

1. `make test` (all existing + new tests).

2. `make qa` (lint + typecheck strict).

3. Cross-cutting integration: at least one Phase-5 test exercises a workflow that contains a `ReActAgent`, hits a `ToolTimeoutError` path, and trips a `WorkflowInvariant` — verifying that all three subsystems compose correctly.

4. Documentation updates land in the same PR as the code.

5. CHANGELOG entry written in the existing dense prose style.

## 17. Cost summary (estimated vs. actual)

| Phase | Estimated LoC | Estimated Tests | Actual Tests |
|---|---|---|---|
| 1. Layer B core | 400 | 25 | 35 |
| 2. Layer C sugar | 150 | 20 | 24 |
| 3. Helpers + streaming + viz | 150 | 15 | 19 |
| 4. Contracts + reducers + typed state | 250 | 25 | 19 |
| 5. AgentProtocol + sub-workflows | 100 | 15 | 21 |
| **Total** | **~1050** | **~100** | **118** |

Final implementation: a single `workflow.py` (~2100 LoC including the inline `reducer` namespace) and a single test file `test_agents_workflow.py` (118 tests, all passing in <1s). The landed code reuses `ContractPolicy` / `ContractViolation` from `contract.py`, `AgentEvent.source` / `parent_event_id` from `types.py`, the `Tool.timeout` daemon-thread pattern (for sync nodes), and the `AgentProtocol` structural contract — no parallel reimplementations.

## 18. Further reading

- [`patterns.md`](patterns.md) §9 — current "not supported" stance flipped to first-class on Phase 1 landing.

- [`../agents_overview.md`](../agents_overview.md) — the agent layer this workflow composes with.

- [`../dev/contract-agent.md`](../dev/contract-agent.md) — `ContractPolicy` semantics inherited by `WorkflowInvariant`.

- LangGraph's `StateGraph` — closest external precedent for Layer B.
