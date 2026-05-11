# Agent patterns in cyllama

This document maps the canonical agent patterns from the literature onto
their cyllama implementations. Each entry says:

- **Status** — first-class primitive, recipe (composes existing primitives),
  or not supported.
- **How to do it** — the actual cyllama API call or recipe sketch, with
  pointers to source files.
- **Gap** — what's missing, if anything.

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

**Status: recipe (composes `ConstrainedAgent` + `agent_as_tool`).**
Reference: *Plan-and-Solve Prompting* (Wang et al., 2023).

A planner agent emits a structured task list; a supervisor dispatches
each task to a worker (in-process via `agent_as_tool` or a single
ReActAgent shared across calls). cyllama doesn't ship a canned
`PlanAndExecute` class, but the primitives compose cleanly:

```python
# Sketch (not in cyllama; ~40 LoC to write yourself):
from cyllama.agents import ConstrainedAgent, ReActAgent, agent_as_tool, tool

# Planner emits {"steps": [...]} via grammar-constrained generation.
planner = ConstrainedAgent(llm=planner_llm, tools=[])
plan = planner.run("Refactor module X to use new pattern Y")
# plan.answer parses to a structured list

# Executor runs each step.
executor = ReActAgent(llm=worker_llm, tools=[read_file, edit_file])
results = [executor.run(step) for step in steps]
```

**Gap**: no canned helper. A `plan_and_execute(planner, executor, task,
plan_schema)` function in `composition.py` (~50 LoC) would crystallize
this. See [Gap summary](#gap-summary) -- rank #4.

---

## 3. Reflection / Reflexion

**Status: recipe (composes two `AgentProtocol` instances), no canned
helper.**
Reference: *Reflexion: Language Agents with Verbal Reinforcement
Learning* (Shinn et al., 2023).

A worker emits a draft; a critic agent (different system prompt,
possibly same LLM) emits acceptance or revision; loop up to N times.
Pure composition:

```python
# Sketch (~50 LoC to write yourself):
worker = ReActAgent(llm=llm, tools=tools, system_prompt=WORKER_PROMPT)
critic = ReActAgent(llm=llm, tools=[], system_prompt=CRITIC_PROMPT)

for attempt in range(max_attempts):
    draft = worker.run(task).answer
    verdict = critic.run(f"Critique this draft: {draft}").answer
    if "ACCEPT" in verdict:
        return draft
    task = f"{task}\n\nPrior attempt: {draft}\nCritic said: {verdict}"
```

**Gap**: no canned `ReflectionLoop(worker, critic, max_attempts)`
helper. Common production pattern for accuracy-critical tasks — coding
agents, scientific reasoning, factual lookup. See [Gap summary](#gap-summary) -- rank #1.

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

**Gap**: cross-process / cross-host agents (sub-agent running behind
MCP). The primitives are in-process only. See [Gap summary](#gap-summary) -- rank #5.

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

**Gaps**:

1. **No semantic memory wiring.** Cyllama has a full RAG subsystem
   (`src/cyllama/rag/`) but it isn't integrated into agents as a
   memory layer. Users have to manually wrap a RAG retriever as a
   `@tool`. See [Gap summary](#gap-summary) -- rank #2.
2. **No long-term profile primitive.** Cross-session
   "remember the user's preferences" semantics — common in production
   chatbots — has no canned API. See [Gap summary](#gap-summary) -- rank #3.

---

## 7. Retrieval-Augmented Agents (RAG agents)

**Status: partial — RAG subsystem exists, agent integration is
user-side recipe.**

cyllama ships a complete RAG pipeline in `src/cyllama/rag/`:

- `RAG` and `RAGPipeline` in `rag.py` / `pipeline.py`
- Embedder, splitter, loaders, advanced query helpers
- Multiple vector-store backends in `store.py`

But the bridge to the agent layer is user code:

```python
from cyllama.rag import RAGPipeline
from cyllama.agents import tool, ReActAgent

rag = RAGPipeline(...)

@tool
def search_kb(query: str, top_k: int = 5) -> list[str]:
    """Search the knowledge base."""
    return [hit.content for hit in rag.query(query, top_k=top_k)]

agent = ReActAgent(llm=llm, tools=[search_kb])
```

**Gap**: a `rag_as_tool(rag_pipeline, name, description, top_k)`
helper in `composition.py` (~20 LoC) would crystallize the recipe and
avoid a class of user mistakes (forgetting to deduplicate, wrong
return type, no result-limit). See [Gap summary](#gap-summary) -- rank #2.

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

**Status: not supported.**

Frameworks like LangGraph model agent behavior as an explicit DAG or
state machine — nodes are tools or LLM calls, edges encode transitions,
state is threaded through. cyllama's agent loops are linear (single
THOUGHT/ACTION/OBSERVATION stream); the multi-agent composition is
hierarchical (supervisor dispatching to workers via tools), not
graph-based.

Building DAG orchestration on top of cyllama is possible
(`AsyncReActAgent` + your own orchestrator), but no graph DSL is
shipped.

**Gap**: significant. Adding graph orchestration is a big design
conversation: do we ship a DSL (locks in shape), depend on an
external library (breaks the zero-dependency charter), or stay
linear-only? See gap-list — explicitly **deferred** until a real use
case demands it.

---

## Summary table

| Pattern | Status | Where |
|---|---|---|
| 1. ReAct | First-class | `ReActAgent` (`react.py`) |
| 2. Plan-and-Execute | Recipe (no helper) | composes `ConstrainedAgent` + `agent_as_tool` |
| 3. Reflection / Reflexion | Recipe (no helper) | composes two `AgentProtocol` instances |
| 4. Tree of Thoughts | Not supported | requires KV-cache snapshot/restore + branching loop |
| 5. Multi-Agent Systems | First-class | `agent_as_tool`, `TieredAgentTeam` (`composition.py`) |
| 6. Memory-Augmented | Partial | session stores; no semantic memory or profiles |
| 7. RAG Agents | Partial | RAG subsystem exists; agent bridge is user-side |
| 8. Autonomous / AutoGPT | Intentionally not supported | counter to bounded-loop design stance |
| 9. Workflow / State-Machine | Not supported | linear-loop framework; no DAG orchestration |

## Gap summary

Each gap above maps to a tractable improvement. The list is ordered by
**value-to-cost ratio for cyllama's typical user**, not by literature
completeness. Engineering detail (sketches, file references, triggers)
lives in [`TODO.md`](../../TODO.md) under "Pattern gaps".

| Rank | Gap | Pattern | Value | Cost |
|---|---|---|---|---|
| **1** | `ReflectionLoop(worker, critic, max_attempts)` helper | §3 Reflection | High -- coding assistants, factual review | ~50 LoC |
| **2** | `rag_as_tool(rag, name, description)` helper | §7 RAG | High -- bridges two healthy subsystems | ~20 LoC |
| **3** | `SemanticMemory` primitive (RAG-backed long-term memory) | §6 Memory | High -- table-stakes for chatbots | Medium -- new class + Session integration |
| **4** | `plan_and_execute(planner, executor, task, schema)` helper | §2 Plan-and-Execute | Medium -- the recipe is fine for most | ~50 LoC |
| **5** | `mcp_agent_tool(client, server, agent)` cross-process helper | §5 Multi-Agent | Medium -- niche but a real capability | Medium -- failure-mode mapping |

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

- **Workflow / State-Machine (§9)** -- big design conversation. Three
  options (ship a DSL, depend on an external library, stay linear-only),
  each with costs the project currently isn't paying. Defer until a
  concrete need forces the choice.

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
