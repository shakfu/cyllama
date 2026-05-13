# Agent patterns in cyllama

This document maps the canonical agent patterns from the literature onto
their cyllama implementations. Each entry says:

- **Status** — first-class (canned helper or primitive), recipe (you
  compose existing primitives yourself), or not supported.
- **How to do it** — the actual cyllama API call, with pointers to
  source files.
- **Gap** — what's missing or possible as a future refinement, if
  anything.

The five "worth-doing" pattern gaps identified in the original audit
have all landed; the [Gap summary](#gap-summary) records what shipped
and what residual refinements remain. Patterns the framework
intentionally does not support are explicitly listed.

For the broader question "should I use schema or contracts for X?" see
[`contracts.md`](contracts.md) and [`../dev/contract-agent.md`](../dev/contract-agent.md).

---

## 1. ReAct (Reason + Act)

**Status: first-class primitive.**
Reference: *ReAct: Synergizing Reasoning and Acting in Language Models*
(Yao et al., 2022).

```python
from cyllama import LLM
from cyllama.agents import ReActAgent, tool

@tool
def search(query: str) -> str: ...

agent = ReActAgent(llm=LLM("model.gguf"), tools=[search])
result = agent.run("What is the capital of France?")
```

Implementation: `src/cyllama/agents/react.py`. The loop alternates
THOUGHT / ACTION / OBSERVATION events emitted on the stream. Loop
detection (`_loop_detection.py`) catches stuck patterns; coercion
(`tools.coerce_args`) catches malformed arguments before dispatch;
timeouts (`Tool.timeout`) cap individual tool calls.

**Gap**: none. This is the strongest pattern in the framework.

---

## 2. Plan-and-Execute

**Status: first-class helper.**
Reference: *Plan-and-Solve Prompting* (Wang et al., 2023).

A planner agent emits a structured task list; an executor runs each step
sequentially.

```python
from cyllama.agents import ConstrainedAgent, ReActAgent, plan_and_execute

planner = ConstrainedAgent(
    llm=planner_llm,
    tools=[],
    system_prompt="Emit a JSON list of step strings.",
)
executor = ReActAgent(llm=worker_llm, tools=[read_file, edit_file])

results = plan_and_execute(planner, executor, "Refactor module X.")
for step_result in results:
    print(step_result.answer)
```

`plan_and_execute(planner, executor, task, plan_parser=None,
stop_on_error=True)` lives in `composition.py`. The default
`plan_parser` tries `json.loads` first (recognizing top-level lists or
`{"steps"|"plan"|"tasks": [...]}` shapes) and falls back to newline
splitting with bullet/number-prefix stripping. Pass a custom callable
for other plan formats.

**Gap**: none for the basic pattern. Streaming events from each step
through a unified iterator is a possible future extension.

---

## 3. Reflection / Reflexion

**Status: first-class helper.**
Reference: *Reflexion: Language Agents with Verbal Reinforcement
Learning* (Shinn et al., 2023).

A worker emits a draft; a critic agent emits acceptance or revision
feedback; the loop iterates up to N times.

```python
from cyllama.agents import ReActAgent, ReflectionLoop

worker = ReActAgent(llm=llm, tools=[...], system_prompt=WORKER_PROMPT)
critic = ReActAgent(
    llm=llm, tools=[],
    system_prompt=(
        "Review the draft. Respond with 'ACCEPT' if correct, "
        "otherwise list specific issues."
    ),
)

loop = ReflectionLoop(worker, critic, max_attempts=3)
result = loop.run("Implement quicksort with a 3-way partition.")
```

`ReflectionLoop(worker, critic, max_attempts=3,
acceptance_marker="ACCEPT", critique_prefix=..., revision_template=...)`
lives in `composition.py`. The critic's answer is checked for a
substring match (case-insensitive) on `acceptance_marker`; on a miss,
the worker is re-invoked with the task augmented by the draft and the
critic feedback (overridable via `revision_template`). Streams emit
worker events tagged `source="worker"` and critic events tagged
`source="critic"`, with `parent_event_id` linking each pass; the
loop's own final ANSWER event has no source.

**Gap**: none for the basic pattern. Parallel critic ensembles
(multiple critics voting) and reward-model-based acceptance are
possible future extensions.

---

## 4. Tree of Thoughts (ToT)

**Status: not supported.**
Reference: *Tree of Thoughts: Deliberate Problem Solving with Large
Language Models* (Yao et al., 2023).

ToT requires forking the generation state at each thought node and
exploring multiple branches before committing — typically with
KV-cache snapshotting so divergent branches don't recompute the shared
prefix. cyllama's public API does not expose KV-cache snapshot /
restore primitives, and the agent loops are linear (single stream of
events, no branching).

Implementing ToT would require: (a) public `LlamaContext.snapshot()` /
`restore()` (currently absent), and (b) a branching agent loop that
maintains a frontier of candidate states with scoring. Significant new
machinery; non-trivial value for cyllama's typical user.

**Gap**: structurally hard. See [Gap summary](#gap-summary) -- explicitly
**not on the roadmap** unless a use case forces it.

---

## 5. Multi-Agent Systems

**Status: first-class primitives.**

Two building blocks in `src/cyllama/agents/composition.py`:

- **`agent_as_tool(agent, name, description)`** — wraps any
  `AgentProtocol` instance as a `Tool` so a supervisor can dispatch to
  it through the normal ReAct flow.
- **`TieredAgentTeam(supervisor, workers)`** — ergonomic container for
  supervisor + named workers, each potentially on a different LLM
  (capability tiering).

```python
from cyllama.agents import ReActAgent, AgentRole, TieredAgentTeam

team = TieredAgentTeam(
    supervisor=ReActAgent(llm=LLM("strong.gguf"), tools=[]),
    workers=[
        AgentRole("researcher", researcher, "Find facts."),
        AgentRole("coder", coder, "Modify code."),
    ],
)
result = team.run("Refactor X using technique Y.")
```

Sub-agent events surface in the supervisor's stream with `source` and
`parent_event_id` annotations (`AgentEvent` fields) when a
`forward_events` callback is supplied. See [`agents_overview.md`](../agents_overview.md#multi-agent-composition)
for the full reference.

**Cross-process variant**: `mcp_agent_tool(client, server_name,
agent_name, description)` wraps a remote agent exposed through MCP as
a local `Tool`, symmetric to `agent_as_tool` but the dispatch crosses
a process boundary:

```python
from cyllama.agents import McpClient, McpServerConfig, mcp_agent_tool, ReActAgent

client = McpClient([McpServerConfig(...)])
client.connect_all()

remote = mcp_agent_tool(
    client,
    server_name="research",
    agent_name="web_search",
    description="Search the web and return findings.",
)
supervisor = ReActAgent(llm=llm, tools=[remote])
```

The wrapped tool is named `"{server_name}/{agent_name}"` -- the
namespaced format the action parser supports. Network failures surface
as `RuntimeError` from `McpClient.call_tool`; the agent's generic
exception handler catches them. Pass `timeout=...` to apply a local
budget (separate from MCP transport timeouts).

**Gap**: streaming sub-agent events across the wire (the current MCP
shape returns a single value per call). Possible future extension if
the MCP server-streaming RFC stabilizes.

---

## 6. Memory-Augmented Agents

**Status: partial — short-term and episodic memory only.**

cyllama ships three session-storage backends in
`src/cyllama/agents/session.py`:

- `MemorySessionStore` — in-process dict, lost at process end.
- `FileSessionStore` — JSON files on disk; multi-process via OS locks.
- `SqliteSessionStore` — SQLite-backed, supports concurrent readers.

Each stores `Message` records, `ToolCallRecord` records, and
`Permission` records keyed by session id. This covers short-term
(within a run) and episodic (across runs) memory.

**Semantic memory primitive** in `src/cyllama/agents/memory.py`:

```python
from cyllama.rag import RAG
from cyllama.agents import SemanticMemory

rag = RAG(...)
memory = SemanticMemory(rag)

# Write into a per-user namespace.
memory.remember(
    "The user prefers concise answers.",
    namespace="user:alice",
)

# Recall later.
hits = memory.retrieve("response style", namespace="user:alice")
for hit in hits:
    print(hit.text, hit.score)
```

`SemanticMemory(rag, namespace_field="_memory_namespace",
default_namespace="default")` is a thin namespace-aware facade over any
RAG-shaped object (`add_texts` + `search`). Records under different
namespaces share the same embedding store but are filtered apart at
retrieval time -- so one cyllama-managed `RAG` instance can back many
logical memory buckets (per-user, per-conversation, per-topic).

`retrieve()` over-fetches from the underlying search to give the
namespace filter room to find enough matches. `remember()` defaults
`split=False` since memory fragments are typically short. `forget()`
exists but currently raises `NotImplementedError` -- the underlying
RAG store doesn't expose metadata-filtered deletion; the documented
workaround is to use separate `RAG` instances per namespace when you
need per-namespace clearing.

**Gap**: filtered deletion -- pending a RAG-side API. Long-term
"profile" semantics (richer than text fragments -- structured user
preferences, time-decay, summarization) is a possible future
extension but the primitive covers the common cases as-is.

---

## 7. Retrieval-Augmented Agents (RAG agents)

**Status: first-class helper.**

cyllama ships a complete RAG pipeline in `src/cyllama/rag/`. The
`rag_as_tool` helper bridges it to the agent layer:

```python
from cyllama.rag import RAG
from cyllama.agents import ReActAgent, rag_as_tool

kb = RAG(...)  # load or build the knowledge base
search = rag_as_tool(
    kb,
    name="search_docs",
    description="Search the project documentation.",
    top_k=5,
)
agent = ReActAgent(llm=llm, tools=[search])
```

`rag_as_tool(rag, name="search_kb", description=..., top_k=5,
query_param="query", method="search", formatter=None)` lives in
`composition.py`. The default formatter emits one `[score] text` line
per hit, deduplicated by text so the agent doesn't see repeated
content. Pass `method="retrieve"` to use the higher-level
`RAG.retrieve` path (applies the RAG pipeline config); pass a custom
`formatter` callable to control the observation shape.

**Gap**: streaming results to the agent as they arrive (currently
returns a single concatenated observation). Possible future extension.

---

## 8. Autonomous / AutoGPT-style

**Status: not supported, by design.**

The Auto-GPT / BabyAGI pattern is recursive goal decomposition with
no external bound: the agent generates its own subgoals, executes
them, generates more subgoals, etc. The literature consistently
reports goal drift, infinite loops, cost explosion, and unreliability.

cyllama's design leans in the opposite direction:

- `ReActAgent.max_iterations` caps every run (default 10).
- `_loop_detection.py` actively terminates runs that show signs of
  repetition.
- `ContractAgent` lets you express *budget invariants*
  (`elapsed_ms < 30_000`, `tool_calls < 20`, `errors < 3`) that
  enforce hard ceilings.

These are the *opposite* of what an autonomous agent needs. If you
genuinely want unbounded goal-decomposition, cyllama is the wrong
framework — use an explicitly autonomy-oriented one and accept the
trade-offs.

**Gap**: not on the roadmap. Documented here so readers know the
framework's stance.

---

## 9. Workflow / State-Machine Agents

**Status: first-class.**

`cyllama.agents.workflow` ships an explicit DAG / state-graph runtime
with two authoring layers over the same compile + execute model.
Nodes are typed Python callables (or agents, tools, or other
workflows); edges are static or conditional; state is a typed dict
threaded between nodes; independent nodes at the same topological
level run concurrently.

```python
from cyllama.agents.workflow import Workflow

flow = Workflow()

# Layer C -- decorator sugar; dependencies inferred from parameter names.
@flow.node
def classify(query: str) -> dict:
    confidence = score(query)
    return {"confidence": confidence, "topic": guess_topic(query)}

@flow.node
def answer(classify: dict) -> str:
    return llm.complete(f"Topic={classify['topic']}: {classify}").text

@flow.node
def escalate(classify: dict) -> str:
    return queue_human_review(classify)

# Conditional router decides which downstream node runs.
@flow.route(after="classify")
def route(classify: dict) -> str:
    return "answer" if classify["confidence"] > 0.7 else "escalate"

flow.set_entry("classify")
result = flow.run(query="What is the capital of France?")
print(result.answer)  # state["answer"] or state["escalate"]
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
result = flow.run(query="...")
```

A workflow's native API is kwargs-only (`flow.run(**state)`); to
plug it into the multi-agent layer, call `flow.as_agent()` to get
an `AgentProtocol`-conformant adapter. `agent_as_tool(flow.as_agent(),
...)`, `ReflectionLoop(flow.as_agent(), critic.as_agent(), ...)`,
and `TieredAgentTeam(workers=[AgentRole("research", flow.as_agent(),
...)])` all work via the same explicit-adapter pattern. Workflows
nest one inside another via `workflow_node(inner, name="research")`;
inner events forward into the outer event stream with `source` /
`parent_event_id` set so streaming UIs can render the tree.

Other capabilities (all landed in the Phase 1-5 rollout):

- Parallel level execution (`asyncio.gather`); sync nodes dispatched
  on `asyncio.to_thread`.
- Per-node `timeout=` using `asyncio.wait_for`.
- `WorkflowInvariant` reusing `ContractPolicy` semantics for
  workflow-level pre/postconditions.
- Reducer registry (`reducer.append` / `extend` / `merge_dict` /
  `add` / `last`) for explicit multi-writer state keys; runtime
  detects unreduced collisions.
- `flow.stream(...)` / `flow.astream(...)` event streams with
  `WORKFLOW_START` / `NODE_START` / `NODE_END` / `ANSWER` /
  `WORKFLOW_END` / `CONTRACT_VIOLATION`.
- `flow.to_mermaid()` / `flow.to_dot()` for graph rendering;
  `flow.dry_run()` for execution-order inspection without running.
- PEP 484 `Workflow[StateT]` generic for typed state.

**Reference**: `cyllama/agents/workflow.py`; design + per-phase
notes in [`workflow.md`](workflow.md); 118 tests in
`tests/test_agents_workflow.py`.

---

## Summary table

| Pattern | Status | Where |
|---|---|---|
| 1. ReAct | First-class | `ReActAgent` (`react.py`) |
| 2. Plan-and-Execute | First-class helper | `plan_and_execute()` (`composition.py`) |
| 3. Reflection / Reflexion | First-class helper | `ReflectionLoop` (`composition.py`) |
| 4. Tree of Thoughts | Not supported | requires KV-cache snapshot/restore + branching loop |
| 5. Multi-Agent Systems | First-class | `agent_as_tool`, `TieredAgentTeam`, `mcp_agent_tool` (`composition.py`) |
| 6. Memory-Augmented | First-class | session stores + `SemanticMemory` (`memory.py`) |
| 7. RAG Agents | First-class helper | `rag_as_tool()` (`composition.py`) |
| 8. Autonomous / AutoGPT | Intentionally not supported | counter to bounded-loop design stance |
| 9. Workflow / State-Machine | First-class | `Workflow` / `CompiledWorkflow` (`workflow.py`); decorator sugar (`@flow.node`, `@flow.route`) over explicit StateGraph |

## Gap summary

All six tractable pattern gaps from the original audit have **landed**.
The table below records what was added, where it lives, and the
tests that pin each helper.

| Rank | Gap | Status | Where | Tests |
|---|---|---|---|---|
| **1** | `ReflectionLoop` (Reflection/Reflexion) | **Landed** | `composition.py` | `tests/test_agents_composition.py::TestReflection*` |
| **2** | `rag_as_tool` (RAG agents) | **Landed** | `composition.py` | `tests/test_agents_composition.py::test_rag_as_tool_*` |
| **3** | `SemanticMemory` (long-term memory) | **Landed** | `memory.py` | `tests/test_agents_memory.py` |
| **4** | `plan_and_execute` (Plan-and-Execute) | **Landed** | `composition.py` | `tests/test_agents_composition.py::test_plan_and_execute_*` |
| **5** | `mcp_agent_tool` (cross-process) | **Landed** | `composition.py` | `tests/test_agents_composition.py::test_mcp_agent_tool_*` |
| **6** | `Workflow` (DAG / State-Machine, §9) | **Landed** (Phases 1-5) | `workflow.py` | `tests/test_agents_workflow.py` (118 tests) |

Documented gaps that remain (each is a future extension, not a missing
fundamental):

- **Streaming sub-agent events across MCP** -- `mcp_agent_tool` returns
  a single value per call; streaming would require the MCP
  server-streaming RFC to stabilize.
- **Streaming RAG results to the agent** -- `rag_as_tool` returns a
  single concatenated observation today.
- **Filtered deletion in SemanticMemory** -- `forget()` raises
  `NotImplementedError` pending a RAG-side metadata-filtered delete API.
- **Parallel critic ensembles in ReflectionLoop** -- multiple critics
  voting, reward-model-based acceptance.
- **Unified streaming for plan_and_execute steps** -- one iterator
  surfacing events from all steps in sequence.

None of these block the pattern; they're refinements waiting on real
use cases.

### Explicitly **not on the roadmap**

These appear in the table above but won't be addressed without a
forcing use case. Listed here to make the position explicit.

- **Tree of Thoughts (§4)** -- requires public `LlamaContext.snapshot()` /
  `restore()` (currently absent) plus a branching agent loop with a
  scored candidate frontier. Significant new machinery; non-trivial
  value for cyllama's typical user.

- **Autonomous / AutoGPT (§8)** -- structurally opposed to cyllama's
  design stance (bounded loops, loop detection, max_iterations,
  contracts for budget invariants). Unbounded goal-decomposition is
  what the framework actively prevents.

### Positioning consequence

Reading down the "Value" column reveals the pattern: every "high value"
gap closes a **reliability or composition** gap (reflection loops, RAG
bridging, persistent memory, plan-then-execute). Every "not on
roadmap" item belongs to the **autonomy or graph-orchestration** half
of the agent space.

This isn't accidental. cyllama leans into bounded, observable,
schema-constrained behavior (loop detection, `max_iterations`,
`Annotated[]` markers, ContractAgent invariants); it explicitly
declines unbounded autonomy and graph orchestration. Users who want
those should reach for an autonomy-first framework. Users who want
debuggable, schema-grounded, locally-runnable agents are in the right
place.

## Further reading

- [`agents_overview.md`](../agents_overview.md) — the full agent
  framework reference: agent types, tools, events, configuration.
- [`contracts.md`](contracts.md) — nine worked recipes for contract
  patterns and the schema-vs-contract decision.
- [`../dev/contract-agent.md`](../dev/contract-agent.md) — design
  rationale for the ContractAgent reposition.
