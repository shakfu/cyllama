"""
LangChain Agent Integration

Provides adapters to use cyllama agents with LangChain's agent framework.
"""

from typing import Any, List, Optional, Dict, Sequence
import warnings

from ..agents import ReActAgent, ConstrainedAgent, Tool as CyllaTool, tool as cyllama_tool
from ..api import LLM as CyllamaLLMCore

try:
    from langchain_core.language_models.llms import BaseLLM as LangChainLLM
    from langchain_core.tools import BaseTool as LangChainTool
    from langchain_core.tools import StructuredTool
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.agents import AgentExecutor, create_react_agent
    from langchain_core.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.llms.base import LLM as LangChainLLM
        from langchain.tools import BaseTool as LangChainTool, StructuredTool
        from langchain.callbacks.manager import CallbackManagerForLLMRun
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain.prompts import PromptTemplate
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        LangChainLLM = object
        LangChainTool = object
        StructuredTool = object
        CallbackManagerForLLMRun = None
        AgentExecutor = None
        create_react_agent = None
        PromptTemplate = None


def cyllama_tool_to_langchain(cyllama_tool: CyllaTool) -> Any:
    """
    Convert a cyllama Tool to a LangChain tool.

    Args:
        cyllama_tool: Cyllama Tool instance

    Returns:
        LangChain StructuredTool instance
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is not installed")

    # Extract schema information
    schema = cyllama_tool.parameters
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Build args_schema dict for LangChain
    from pydantic import BaseModel, Field, create_model

    # Create field definitions
    field_definitions = {}
    for param_name, param_info in properties.items():
        param_type = param_info.get("type", "string")
        param_desc = param_info.get("description", "")

        # Map JSON types to Python types
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        python_type = type_map.get(param_type, str)

        # Determine if optional
        is_required = param_name in required

        if is_required:
            field_definitions[param_name] = (python_type, Field(..., description=param_desc))
        else:
            field_definitions[param_name] = (python_type, Field(None, description=param_desc))

    # Create Pydantic model for args_schema
    ArgsSchema = create_model(
        f"{cyllama_tool.name}Args",
        **field_definitions
    )

    # Create LangChain tool
    def tool_func(**kwargs):
        return cyllama_tool(**kwargs)

    return StructuredTool(
        name=cyllama_tool.name,
        description=cyllama_tool.description,
        func=tool_func,
        args_schema=ArgsSchema
    )


def langchain_tool_to_cyllama(langchain_tool: Any) -> CyllaTool:
    """
    Convert a LangChain tool to a cyllama Tool.

    Args:
        langchain_tool: LangChain BaseTool instance

    Returns:
        Cyllama Tool instance
    """
    # Extract tool information
    name = langchain_tool.name
    description = langchain_tool.description

    # Try to get args schema
    schema = {"type": "object", "properties": {}, "required": []}

    if hasattr(langchain_tool, "args_schema") and langchain_tool.args_schema:
        try:
            # Extract from Pydantic model
            from pydantic import BaseModel
            if isinstance(langchain_tool.args_schema, type) and issubclass(langchain_tool.args_schema, BaseModel):
                schema_dict = langchain_tool.args_schema.model_json_schema()
                if "properties" in schema_dict:
                    schema["properties"] = schema_dict["properties"]
                if "required" in schema_dict:
                    schema["required"] = schema_dict["required"]
        except Exception:
            pass

    # Create wrapper function
    def wrapper(**kwargs):
        return langchain_tool.run(kwargs)

    # Create cyllama tool
    tool_instance = CyllaTool(
        name=name,
        description=description,
        func=wrapper,
        parameters=schema
    )

    return tool_instance


def create_langchain_agent_executor(
    llm: CyllamaLLMCore,
    tools: List[CyllaTool],
    agent_type: str = "react",
    verbose: bool = False,
    max_iterations: int = 10,
) -> Any:
    """
    Create a LangChain AgentExecutor using cyllama tools.

    Args:
        llm: Cyllama LLM instance
        tools: List of cyllama tools
        agent_type: Type of agent ("react" only for now)
        verbose: Print agent steps
        max_iterations: Maximum iterations

    Returns:
        LangChain AgentExecutor
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is not installed")

    if agent_type != "react":
        raise ValueError(f"Unsupported agent type: {agent_type}")

    # Import LangChain wrapper
    from .langchain import CyllamaLLM

    # Convert cyllama tools to LangChain tools
    lc_tools = [cyllama_tool_to_langchain(tool) for tool in tools]

    # Create LangChain LLM wrapper
    lc_llm = CyllamaLLM(model_path=llm.model_path)

    # Create ReAct agent
    if create_react_agent is None or PromptTemplate is None or AgentExecutor is None:
        raise ImportError("LangChain agent components not available")

    # Default ReAct prompt
    prompt = PromptTemplate.from_template(
        """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    )

    agent = create_react_agent(lc_llm, lc_tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=lc_tools,
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=True
    )


class CyllamaAgentLangChainAdapter:
    """
    Adapter to use cyllama agents with LangChain.

    This allows you to use cyllama's ReActAgent or ConstrainedAgent
    within LangChain workflows.
    """

    def __init__(
        self,
        agent: Any,  # ReActAgent or ConstrainedAgent
        return_intermediate_steps: bool = True
    ):
        """
        Initialize adapter.

        Args:
            agent: Cyllama agent instance (ReActAgent or ConstrainedAgent)
            return_intermediate_steps: Include intermediate steps in output
        """
        self.agent = agent
        self.return_intermediate_steps = return_intermediate_steps

    def run(self, input: str) -> str:
        """
        Run the agent on an input.

        Args:
            input: Task or question

        Returns:
            Agent's answer
        """
        result = self.agent.run(input)
        return result.answer

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call agent with LangChain-style inputs.

        Args:
            inputs: Dict with "input" key

        Returns:
            Dict with "output" and optionally "intermediate_steps"
        """
        input_text = inputs.get("input", "")
        result = self.agent.run(input_text)

        output = {"output": result.answer}

        if self.return_intermediate_steps:
            # Convert agent events to intermediate steps
            steps = []
            for event in result.steps:
                steps.append((event.type.value, event.content))
            output["intermediate_steps"] = steps

        return output


def create_cyllama_react_agent(
    model_path: str,
    tools: List[CyllaTool],
    **kwargs
) -> ReActAgent:
    """
    Convenience function to create a cyllama ReAct agent.

    Args:
        model_path: Path to GGUF model
        tools: List of tools
        **kwargs: Additional arguments for ReActAgent

    Returns:
        ReActAgent instance
    """
    llm = CyllamaLLMCore(model_path)
    return ReActAgent(llm=llm, tools=tools, **kwargs)


def create_cyllama_constrained_agent(
    model_path: str,
    tools: List[CyllaTool],
    **kwargs
) -> ConstrainedAgent:
    """
    Convenience function to create a cyllama ConstrainedAgent.

    Args:
        model_path: Path to GGUF model
        tools: List of tools
        **kwargs: Additional arguments for ConstrainedAgent

    Returns:
        ConstrainedAgent instance
    """
    llm = CyllamaLLMCore(model_path)
    return ConstrainedAgent(llm=llm, tools=tools, **kwargs)
