"""
OpenAI-Compatible Function Calling Integration

Provides OpenAI-style function calling using cyllama agents.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import json
import time
import uuid

from ..agents import Tool as CyllaTool, ConstrainedAgent
from ..api import LLM as CyllamaLLMCore


@dataclass
class FunctionCall:
    """Function call information."""
    name: str
    arguments: str  # JSON string


@dataclass
class ToolCall:
    """Tool call in OpenAI format."""
    id: str
    type: str = "function"
    function: Optional[FunctionCall] = None


@dataclass
class AssistantMessage:
    """Assistant message with optional tool calls."""
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


@dataclass
class FunctionMessage:
    """Function/tool response message."""
    content: str
    tool_call_id: str
    role: str = "tool"


@dataclass
class ChatCompletionMessage:
    """Message in chat completion response."""
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


@dataclass
class ChatCompletionChoice:
    """Choice in chat completion response."""
    index: int
    message: ChatCompletionMessage
    finish_reason: str


@dataclass
class ChatCompletionResponse:
    """OpenAI-style chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = "cyllama"
    choices: List[ChatCompletionChoice] = field(default_factory=list)


def cyllama_tool_to_openai_function(tool: CyllaTool) -> Dict[str, Any]:
    """
    Convert a cyllama Tool to OpenAI function definition.

    Args:
        tool: Cyllama Tool instance

    Returns:
        OpenAI function definition dict
    """
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters
    }


def cyllama_tools_to_openai_tools(tools: List[CyllaTool]) -> List[Dict[str, Any]]:
    """
    Convert cyllama tools to OpenAI tools format.

    Args:
        tools: List of cyllama tools

    Returns:
        List of OpenAI tool definitions
    """
    return [
        {
            "type": "function",
            "function": cyllama_tool_to_openai_function(tool)
        }
        for tool in tools
    ]


class OpenAIFunctionCallingClient:
    """
    OpenAI-compatible function calling client using cyllama agents.

    Provides an OpenAI-style API for function calling, backed by
    cyllama's ConstrainedAgent for reliable tool execution.
    """

    def __init__(
        self,
        model_path: str,
        tools: Optional[List[CyllaTool]] = None,
        verbose: bool = False
    ):
        """
        Initialize function calling client.

        Args:
            model_path: Path to GGUF model
            tools: List of available tools
            verbose: Print debug information
        """
        self.model_path = model_path
        self.tools = tools or []
        self.verbose = verbose
        self._llm = None
        self._agent = None

    @property
    def llm(self) -> CyllamaLLMCore:
        """Lazy-load LLM."""
        if self._llm is None:
            self._llm = CyllamaLLMCore(self.model_path, verbose=self.verbose)
        return self._llm

    def create_agent(self, tools: Optional[List[CyllaTool]] = None) -> ConstrainedAgent:
        """
        Create or recreate the agent with given tools.

        Args:
            tools: Tools to use (defaults to instance tools)

        Returns:
            ConstrainedAgent instance
        """
        tools = tools or self.tools
        return ConstrainedAgent(
            llm=self.llm,
            tools=tools,
            format="function_call",
            verbose=self.verbose
        )

    def chat_completion_with_functions(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[CyllaTool]] = None,
        tool_choice: str = "auto",
        max_iterations: int = 5
    ) -> ChatCompletionResponse:
        """
        Create chat completion with function calling.

        Args:
            messages: List of message dicts
            tools: Available tools (uses instance tools if None)
            tool_choice: "auto", "required", or "none"
            max_iterations: Maximum tool call iterations

        Returns:
            ChatCompletionResponse with tool calls or final answer
        """
        tools = tools or self.tools

        # Extract user message (last user message)
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            raise ValueError("No user message found")

        # Create agent
        agent = self.create_agent(tools)

        # Run agent
        result = agent.run(user_message)

        # Build response
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        if result.success:
            # Check if there were tool calls
            tool_calls_made = [
                event for event in result.steps
                if event.type.value == "action"
            ]

            if tool_calls_made and tool_choice != "none":
                # Return tool calls
                tool_calls = []
                for i, event in enumerate(tool_calls_made):
                    # Parse tool call from metadata
                    tool_name = event.metadata.get("tool_name", "unknown")
                    tool_args = event.metadata.get("tool_args", {})

                    tool_call = ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=FunctionCall(
                            name=tool_name,
                            arguments=json.dumps(tool_args)
                        )
                    )
                    tool_calls.append(tool_call)

                message = ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=tool_calls
                )

                finish_reason = "tool_calls"
            else:
                # Return final answer
                message = ChatCompletionMessage(
                    role="assistant",
                    content=result.answer
                )
                finish_reason = "stop"

            choice = ChatCompletionChoice(
                index=0,
                message=message,
                finish_reason=finish_reason
            )

            return ChatCompletionResponse(
                id=response_id,
                choices=[choice]
            )
        else:
            # Error case
            message = ChatCompletionMessage(
                role="assistant",
                content=f"Error: {result.error}"
            )
            choice = ChatCompletionChoice(
                index=0,
                message=message,
                finish_reason="error"
            )
            return ChatCompletionResponse(
                id=response_id,
                choices=[choice]
            )

    def list_functions(self) -> List[Dict[str, Any]]:
        """
        List available functions in OpenAI format.

        Returns:
            List of function definitions
        """
        return [cyllama_tool_to_openai_function(tool) for tool in self.tools]

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools in OpenAI format.

        Returns:
            List of tool definitions
        """
        return cyllama_tools_to_openai_tools(self.tools)


def create_openai_function_calling_client(
    model_path: str,
    tools: List[CyllaTool],
    **kwargs
) -> OpenAIFunctionCallingClient:
    """
    Convenience function to create OpenAI function calling client.

    Args:
        model_path: Path to GGUF model
        tools: List of tools
        **kwargs: Additional arguments

    Returns:
        OpenAIFunctionCallingClient instance
    """
    return OpenAIFunctionCallingClient(
        model_path=model_path,
        tools=tools,
        **kwargs
    )
