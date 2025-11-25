# Cyllama Agents

Agent implementations for cyllama that leverage its strengths: zero dependencies, high-performance local inference, and streaming generation.

## Overview

The agents module provides agent architectures that can use tools to solve tasks through reasoning and action loops.

## Quick Start

```python
from cyllama import LLM
from cyllama.agents import ReActAgent, tool

# Define a tool
@tool
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression"""
    return eval(expression, {"__builtins__": {}}, {})

# Create an agent
llm = LLM("path/to/model.gguf")
agent = ReActAgent(llm=llm, tools=[calculator])

# Run the agent
result = agent.run("What is 15 times 23?")
print(result.answer)
```

## Available Agents

### ReActAgent

Implements the ReAct (Reasoning + Acting) pattern where the agent alternates between:
1. **Thought**: Reasoning about what to do next
2. **Action**: Invoking a tool
3. **Observation**: Seeing the result
4. *(Repeat until task is complete)*

**Best for**: Flexible reasoning, larger models (13B+), tasks requiring creative problem-solving

```python
from cyllama.agents import ReActAgent

agent = ReActAgent(
    llm=llm,
    tools=[tool1, tool2],
    max_iterations=10,
    verbose=True
)

# Run agent
result = agent.run("Your task here")

# Stream agent execution
for event in agent.stream("Your task here"):
    if event.type == "thought":
        print(f"Thinking: {event.content}")
    elif event.type == "action":
        print(f"Action: {event.content}")
    elif event.type == "observation":
        print(f"Observation: {event.content}")
```

### ConstrainedAgent (NEW - Phase 2)

Uses GBNF grammar constraints to enforce valid JSON tool call syntax, guaranteeing 100% parseable outputs. Eliminates parsing failures and enables reliable agent execution even with smaller models (7B-13B).

**Best for**: Production deployments, smaller models, reliability-critical applications

```python
from cyllama.agents import ConstrainedAgent

agent = ConstrainedAgent(
    llm=llm,
    tools=[tool1, tool2],
    max_iterations=10,
    format="json",           # or "function_call"
    allow_reasoning=False,   # Optional reasoning field
    use_cache=True          # Cache grammars for performance
)

# Run agent - guaranteed valid JSON tool calls
result = agent.run("Your task here")

# Stream events (same as ReActAgent)
for event in agent.stream("Your task here"):
    # Process events...
    pass
```

**Key Differences**:
- **ReActAgent**: Parses freeform text, flexible but may fail with small models
- **ConstrainedAgent**: Grammar-enforced JSON, 100% reliable, works with smaller models

## Creating Tools

Tools are Python functions decorated with `@tool`:

```python
from cyllama.agents import tool

@tool
def search(query: str, max_results: int = 5) -> list:
    """
    Search for information.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of search results
    """
    # Your implementation here
    return search_results

# Tool with custom name and description
@tool(name="web_search", description="Search the web")
def my_search_function(query: str) -> list:
    return []
```

### Tool Features

- **Automatic schema generation**: Tools automatically generate JSON schemas from function signatures
- **Type hints**: Use Python type hints for parameter types
- **Docstring parsing**: Parameter descriptions are extracted from docstrings
- **Flexible arguments**: Tools support required and optional parameters

## Tool Registry

The `ToolRegistry` manages available tools:

```python
from cyllama.agents import ToolRegistry, tool

registry = ToolRegistry()

@tool
def my_tool():
    return "result"

# Register tool
registry.register(my_tool)

# Get tool
tool = registry.get("my_tool")

# List all tools
tools = registry.list_tools()

# Generate prompt description
prompt = registry.to_prompt_string()
```

## Agent Configuration

### ReActAgent Configuration

```python
agent = ReActAgent(
    llm=llm,                          # LLM instance
    tools=[tool1, tool2],             # List of tools
    system_prompt=None,               # Custom system prompt (optional)
    max_iterations=10,                # Maximum reasoning loops
    verbose=False,                    # Print agent thoughts
    generation_config=None,           # Custom GenerationConfig
)
```

### Generation Configuration

```python
from cyllama import GenerationConfig

config = GenerationConfig(
    temperature=0.7,
    max_tokens=512,
    stop_sequences=["Observation:", "\nObservation:"]
)

agent = ReActAgent(llm=llm, tools=[], generation_config=config)
```

## Agent Events

When streaming agent execution, you receive `AgentEvent` objects:

```python
from cyllama.agents import EventType

for event in agent.stream(task):
    if event.type == EventType.THOUGHT:
        # Agent is reasoning
        print(event.content)
    elif event.type == EventType.ACTION:
        # Agent is calling a tool
        print(event.content)
        print(event.metadata)  # Additional info
    elif event.type == EventType.OBSERVATION:
        # Tool result
        print(event.content)
    elif event.type == EventType.ANSWER:
        # Final answer
        print(event.content)
    elif event.type == EventType.ERROR:
        # Error occurred
        print(event.content)
```

## Agent Results

The `run()` method returns an `AgentResult`:

```python
result = agent.run(task)

print(result.answer)        # Final answer string
print(result.success)       # True if completed successfully
print(result.iterations)    # Number of reasoning loops
print(result.error)         # Error message if failed
print(result.steps)         # List of AgentEvents
```

## Best Practices

### 1. Tool Design

- Keep tools focused and single-purpose
- Use descriptive names and docstrings
- Handle errors gracefully
- Return string representations for complex objects

```python
@tool
def read_file(filepath: str) -> str:
    """
    Read contents of a file.

    Args:
        filepath: Path to the file to read

    Returns:
        File contents as string
    """
    try:
        with open(filepath) as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"
```

### 2. Agent Prompting

- Use clear, specific task descriptions
- Provide context when needed
- Set appropriate max_iterations for complex tasks

```python
# Good
result = agent.run("Calculate the factorial of 5 and explain the steps")

# Less effective
result = agent.run("factorial 5")
```

### 3. Model Selection

- Larger models (13B+) generally perform better at following agent patterns
- Instruct-tuned models work best
- Consider quantization levels for speed/quality tradeoff

```python
# Recommended models
llm = LLM("Llama-3.2-3B-Instruct-Q8_0.gguf")  # Good balance
llm = LLM("Mistral-7B-Instruct-v0.2.gguf")     # Strong reasoning
llm = LLM("Llama-3.1-8B-Instruct-Q4_K_M.gguf") # Faster inference
```

### 4. Error Handling

Always handle potential failures:

```python
result = agent.run(task)

if result.success:
    print(f"Success: {result.answer}")
else:
    print(f"Failed after {result.iterations} iterations")
    print(f"Error: {result.error}")

    # Examine steps for debugging
    for event in result.steps:
        print(f"{event.type}: {event.content}")
```

## Grammar-Based Constrained Generation

The ConstrainedAgent uses GBNF (Greibach Normal Form) grammars to enforce valid JSON structure during generation.

### Generating Grammars

```python
from cyllama.agents import (
    tool,
    generate_tool_call_schema,
    generate_tool_call_grammar,
    GrammarFormat
)

@tool
def search(query: str) -> str:
    return "results"

# Generate JSON schema for tools
schema = generate_tool_call_schema(
    tools=[search],
    allow_reasoning=True,
    format=GrammarFormat.JSON
)

# Convert schema to GBNF grammar
grammar = generate_tool_call_grammar(
    tools=[search],
    allow_reasoning=True,
    format=GrammarFormat.JSON
)
```

### Supported Formats

1. **JSON** (default): Simple tool call format
   ```json
   {
     "reasoning": "optional reasoning text",
     "tool_name": "search",
     "tool_args": {"query": "..."}
   }
   ```

2. **FUNCTION_CALL**: OpenAI-compatible format
   ```json
   {
     "reasoning": "optional",
     "name": "search",
     "arguments": "{\"query\": \"...\"}"
   }
   ```

3. **JSON_ARRAY**: Multiple tool calls
   ```json
   {
     "reasoning": "optional",
     "tool_calls": [
       {"tool_name": "search", "tool_args": {...}},
       {"tool_name": "calculate", "tool_args": {...}}
     ]
   }
   ```

### Grammar Caching

Grammars are automatically cached for performance:

```python
from cyllama.agents import get_cached_tool_grammar, clear_grammar_cache

# Automatically cached
grammar = get_cached_tool_grammar(tools=[search])

# Clear cache if needed
clear_grammar_cache()
```

## Examples

- `tests/examples/agent_example.py` - ReAct agent with calculator tools
- `tests/examples/constrained_agent_example.py` - Constrained agent with grammar enforcement

## Architecture

The agent implementation follows cyllama's core principles:

- **Zero dependencies**: No external libraries required
- **Performance-first**: Direct integration with cyllama's LLM class
- **Pythonic simplicity**: Clean, straightforward API
- **Framework-agnostic**: No opinionated abstractions

## Implementation Status

- **Phase 1 (Complete)**: ReAct agent with tool calling
- **Phase 2 (Complete)**: Grammar-constrained agents for reliability
- **Phase 3 (Planned)**: Framework integrations (LangChain, AutoGPT, CrewAI)
- **Phase 4 (Planned)**: Multi-agent orchestration and advanced planning

See `AGENT_ANALYSIS.md` for detailed architecture analysis and roadmap.

## Testing

The agent module includes comprehensive tests:

```bash
# Run specific agent tests
pytest tests/test_agents_tools.py -v        # Tool registry tests
pytest tests/test_agents_react.py -v        # ReAct agent tests
pytest tests/test_agents_grammar.py -v      # Grammar generation tests
pytest tests/test_agents_constrained.py -v  # Constrained agent tests

# Run all tests
make test
```

**Test Coverage**:
- 82 tests for agent functionality (29 tools, 31 ReAct, 22 grammar, 33 constrained)
- All 353 tests pass (278 existing + 75 new agent tests)

## Contributing

When adding new agent types:

1. Follow the existing module structure
2. Add comprehensive tests
3. Update this README
4. Maintain zero-dependency principle
5. Ensure all tests pass with `make test`
