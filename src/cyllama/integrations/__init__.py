"""
Integration helpers for popular Python frameworks.

This module provides adapters and utilities to integrate cyllama with
popular frameworks like LangChain, OpenAI API, and others.
"""

from .langchain import CyllamaLLM
from .openai_compat import OpenAICompatibleClient as OpenAIClient

# Agent integrations (optional - require agent module)
try:
    from .langchain_agents import (
        cyllama_tool_to_langchain,
        langchain_tool_to_cyllama,
        create_langchain_agent_executor,
        CyllamaAgentLangChainAdapter,
        create_cyllama_react_agent,
        create_cyllama_constrained_agent,
    )
    from .openai_agents import (
        OpenAIFunctionCallingClient,
        create_openai_function_calling_client,
        cyllama_tool_to_openai_function,
        cyllama_tools_to_openai_tools,
    )
    AGENT_INTEGRATIONS_AVAILABLE = True
except ImportError:
    AGENT_INTEGRATIONS_AVAILABLE = False

__all__ = [
    "CyllamaLLM",
    "OpenAIClient",
]

if AGENT_INTEGRATIONS_AVAILABLE:
    __all__.extend([
        # LangChain agents
        "cyllama_tool_to_langchain",
        "langchain_tool_to_cyllama",
        "create_langchain_agent_executor",
        "CyllamaAgentLangChainAdapter",
        "create_cyllama_react_agent",
        "create_cyllama_constrained_agent",
        # OpenAI function calling
        "OpenAIFunctionCallingClient",
        "create_openai_function_calling_client",
        "cyllama_tool_to_openai_function",
        "cyllama_tools_to_openai_tools",
    ])
