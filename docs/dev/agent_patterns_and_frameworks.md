# Agent Patterns and Agentic Frameworks

## Agent Patterns Overview

Agent patterns are reusable reasoning-and-control-loop designs for building AI systems that can plan, use tools, maintain memory, collaborate, and recover from errors.

Different patterns optimize for different tradeoffs:
- reliability
- latency
- cost
- transparency
- autonomy
- controllability

---

# 1. ReAct (Reason + Act)

Originated in the paper *ReAct: Reasoning and Acting in Language Models*.

## Core Idea
The model alternates between:
1. Reasoning ("Thought")
2. Taking an action ("Act")
3. Observing tool output ("Observation")

Loop repeats until completion.

## Typical Flow

```text
Question
  ↓
Thought
  ↓
Action (tool/API/search/code)
  ↓
Observation
  ↓
Thought
  ↓
Final Answer
```

## Strengths
- Simple and powerful
- Transparent reasoning chain
- Great for tool usage
- Easy to debug
- Strong baseline for most agents

## Weaknesses
- Can become verbose
- Reasoning loops may drift
- Tool calls can explode in cost
- Long contexts degrade performance

## Best Use Cases
- Search agents
- Coding assistants
- Research copilots
- Tool-using chatbots

---

# 2. Plan-and-Execute

## Core Idea
Split planning from execution.

One model/planner creates a task plan.
Another executor agent performs steps.

## Architecture

```text
User Goal
   ↓
Planner
   ↓
Task List
   ↓
Executor(s)
   ↓
Results
```

## Strengths
- More structured than ReAct
- Better for long tasks
- Easier orchestration
- Supports retries and checkpoints

## Weaknesses
- Plans can become stale
- Requires coordination layer
- Less adaptive mid-execution

## Best Use Cases
- Workflow automation
- Multi-step enterprise tasks
- Data pipelines
- Long coding tasks

---

# 3. Reflection / Reflexion

Based on *Reflexion: Language Agents with Verbal Reinforcement Learning*.

## Core Idea
The agent critiques itself after acting.

It stores lessons from failures and improves future attempts.

## Flow

```text
Act
 ↓
Evaluate
 ↓
Reflect
 ↓
Retry with improvements
```

## Strengths
- Stronger reliability
- Improves over iterations
- Good for hard reasoning/coding

## Weaknesses
- More latency
- Higher token cost
- Reflection quality varies

## Best Use Cases
- Coding agents
- Scientific reasoning
- Debugging
- High-accuracy workflows

---

# 4. Tree of Thoughts (ToT)

Based on *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*.

## Core Idea
Instead of one reasoning chain, explore multiple reasoning branches.

The agent:
- generates candidate thoughts,
- evaluates them,
- searches through possibilities.

## Strengths
- Better for hard reasoning
- Can outperform linear CoT
- Supports backtracking

## Weaknesses
- Expensive
- Computationally heavy
- Hard to tune

## Best Use Cases
- Math
- Strategic planning
- Puzzle solving
- Complex decision trees

---

# 5. Multi-Agent Systems

## Core Idea
Multiple specialized agents collaborate.

Examples:
- Planner
- Researcher
- Critic
- Coder
- Reviewer

## Strengths
- Specialization improves quality
- Parallelism
- Separation of concerns

## Weaknesses
- Coordination overhead
- Agent disagreement
- Context synchronization issues
- Cost explosion

## Best Use Cases
- Enterprise automation
- Large research tasks
- Software engineering teams

---

# 6. Memory-Augmented Agents

## Core Idea
Agents maintain memory beyond context windows.

Memory types:
- Short-term scratchpad
- Episodic memory
- Semantic memory
- Vector DB retrieval
- Long-term profiles

## Strengths
- Personalization
- Long-running tasks
- Cross-session continuity

## Weaknesses
- Memory retrieval errors
- Context pollution
- Privacy/security concerns

---

# 7. Retrieval-Augmented Agents (RAG Agents)

## Core Idea
Agent retrieves external knowledge before reasoning.

Combines:
- retrieval
- reasoning
- tool usage

## Flow

```text
Question
  ↓
Retrieve docs
  ↓
Reason over docs
  ↓
Answer
```

## Strengths
- Fresh knowledge
- Reduces hallucination
- Enterprise-ready

## Weaknesses
- Retrieval quality bottleneck
- Context ranking challenges

---

# 8. Autonomous / AutoGPT-style Agents

Popularized by Auto-GPT and BabyAGI.

## Core Idea
Agent recursively creates and executes tasks toward a broad goal.

## Flow

```text
Goal
 ↓
Generate tasks
 ↓
Prioritize
 ↓
Execute
 ↓
Create more tasks
```

## Strengths
- High autonomy
- Open-ended exploration
- Long horizon capability

## Weaknesses
- Goal drift
- Infinite loops
- Cost/latency explosion
- Reliability issues

---

# 9. Workflow / State-Machine Agents

## Core Idea
Agent behavior is explicitly modeled as states and transitions.

Often implemented with:
- DAGs
- orchestrators
- BPM systems
- finite state machines

## Strengths
- Deterministic
- Auditable
- Production-safe
- Easy observability

## Weaknesses
- Less flexible
- More engineering effort
- Brittle for ambiguous tasks

---

# High-Level Pattern Comparison

| Pattern | Flexibility | Reliability | Cost | Transparency | Best For |
|---|---|---|---|---|---|
| ReAct | High | Medium | Medium | High | General agents |
| Plan-and-Execute | Medium | High | Medium | High | Structured workflows |
| Reflexion | High | Higher | High | High | Coding/reasoning |
| Tree of Thoughts | Very High | High | Very High | Medium | Hard reasoning |
| Multi-Agent | Very High | Medium | Very High | Medium | Large systems |
| RAG Agents | Medium | High | Medium | High | Knowledge systems |
| Workflow FSM | Low-Medium | Very High | Low | Very High | Production systems |
| Autonomous Agents | Very High | Low | Very High | Low | Experimental autonomy |

---

# Agentic Frameworks Comparison

| Framework | Core Philosophy | Best At | Weaknesses |
|---|---|---|---|
| LangChain | Modular agent components | Fast prototyping | Historically complex APIs |
| LangGraph | Stateful graph orchestration | Production agents | More engineering-heavy |
| LlamaIndex | Retrieval + knowledge agents | RAG systems | Less orchestration depth |
| AutoGen | Conversational multi-agent systems | Agent collaboration | Can become chaotic |
| CrewAI | Role-based teams | Easy multi-agent workflows | Less flexible than graphs |
| Semantic Kernel | Enterprise orchestration | Microsoft ecosystem | More enterprise-oriented |
| OpenAI Agents SDK | Structured agent runtime | Reliable tool agents | Less open-ended autonomy |
| Haystack | NLP pipelines + RAG | Search/retrieval systems | Agent support newer |
| DSPy | Declarative LM programming | Optimization/reliability | Less runtime orchestration |
| PydanticAI | Typed Pythonic agents | Structured outputs | Smaller ecosystem |

---

# Framework-by-Framework Breakdown

## LangChain

### Philosophy
Composable primitives:
- prompts
- tools
- memory
- retrievers
- agents
- chains

### Supported Patterns
- ReAct
- Tool Calling
- RAG
- Reflection (manual)
- Workflow agents

### Strengths
- Huge ecosystem
- Rapid experimentation
- Strong integrations

### Weaknesses
- API churn historically
- Debugging complexity

---

## LangGraph

### Philosophy
Agents are stateful graphs.

Each node:
- reasons
- calls tools
- updates state
- routes conditionally

### Supported Patterns
- ReAct
- Workflow FSM
- Reflection
- Human-in-the-loop
- Multi-agent
- Planner-executor

### Strengths
- Production-grade orchestration
- Durable execution
- Strong observability

### Weaknesses
- More engineering-heavy

---

## LlamaIndex

### Philosophy
Knowledge-centric agents.

### Supported Patterns
- RAG
- Retrieval agents
- ReAct
- Tool agents

### Strengths
- Excellent retrieval abstractions
- Strong document pipelines

### Weaknesses
- Less orchestration sophistication

---

## AutoGen

### Philosophy
Agents communicate primarily through conversation.

### Supported Patterns
- Multi-agent collaboration
- Debate agents
- Reflection
- Planner-executor

### Strengths
- Natural collaborative agents
- Flexible conversations

### Weaknesses
- Coordination complexity
- Token-heavy

---

## CrewAI

### Philosophy
Role-based AI teams.

### Supported Patterns
- Multi-agent teams
- Sequential workflows
- Hierarchical delegation

### Strengths
- Easy onboarding
- Intuitive abstractions

### Weaknesses
- Less low-level control

---

## Semantic Kernel

### Philosophy
Enterprise AI orchestration.

### Supported Patterns
- Planner-executor
- Tool/plugin agents
- Workflow agents
- Memory-augmented agents

### Strengths
- Enterprise integration
- Strong Microsoft ecosystem support

### Weaknesses
- More enterprise-oriented

---

## OpenAI Agents SDK

### Philosophy
Reliable structured agents with:
- tool calling
- tracing
- memory
- handoffs
- guardrails

### Supported Patterns
- ReAct-like tool use
- Handoffs
- Retrieval agents
- Workflow agents

### Strengths
- Reliable tool orchestration
- Strong tracing
- Production-focused

### Weaknesses
- Less experimental flexibility

---

## Haystack

### Philosophy
Search + NLP pipelines.

### Supported Patterns
- RAG
- Retrieval pipelines
- Workflow DAGs

### Strengths
- Strong retrieval stack

### Weaknesses
- Agent ecosystem less mature

---

## DSPy

### Philosophy
Declarative LM programming.

### Supported Patterns
- Reflection
- Self-improving pipelines
- RAG optimization

### Strengths
- Automatic prompt tuning
- Reliability optimization

### Weaknesses
- Not a runtime orchestration framework

---

## PydanticAI

### Philosophy
Typed, structured Python-first agents.

### Supported Patterns
- Tool agents
- Structured reasoning
- Validation loops

### Strengths
- Type safety
- Clean Python ergonomics

### Weaknesses
- Smaller ecosystem

---

# Pattern Coverage Matrix

| Pattern | LC | LG | LI | AG | Crew | SK | OpenAI | Haystack | DSPy | PydanticAI |
|---|---|---|---|---|---|---|---|---|---|---|
| ReAct | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | △ | △ | ✓ |
| RAG | ✓ | ✓ | ✓✓ | △ | △ | ✓ | ✓ | ✓✓ | ✓ | △ |
| Multi-Agent | △ | ✓ | △ | ✓✓ | ✓✓ | △ | △ | △ | △ | △ |
| Reflection | △ | ✓ | △ | ✓ | △ | △ | △ | △ | ✓✓ | ✓ |
| Workflow FSM | △ | ✓✓ | △ | △ | ✓ | ✓ | ✓ | ✓ | △ | ✓ |
| Planner-Executor | ✓ | ✓✓ | △ | ✓ | ✓ | ✓ | ✓ | △ | △ | △ |
| Autonomous Loops | ✓ | ✓ | △ | ✓✓ | ✓ | △ | △ | △ | △ | △ |
| Typed Reliability | △ | △ | △ | △ | △ | ✓ | ✓ | △ | ✓ | ✓✓ |

Legend:
- ✓✓ = exceptional
- ✓ = strong
- △ = partial/manual

---

# Practical Recommendations

| Goal | Recommended Frameworks |
|---|---|
| Fast experimentation | LangChain, CrewAI |
| Production reliability | LangGraph, OpenAI Agents SDK |
| Heavy RAG systems | LlamaIndex, Haystack |
| Multi-agent collaboration | AutoGen, CrewAI, LangGraph |
| Reliability optimization | DSPy, PydanticAI |
| Enterprise orchestration | Semantic Kernel |

---

# Industry Direction (2025–2026)

The ecosystem is converging toward hybrid architectures:

```text
Workflow Graph
    ↓
ReAct Agent Nodes
    ↓
Tools + Retrieval
    ↓
Evaluator/Reflection Nodes
    ↓
Memory Layer
```

The dominant modern stack increasingly combines:
- graph orchestration
- retrieval
- tool use
- evaluators
- memory systems
- human approval checkpoints


