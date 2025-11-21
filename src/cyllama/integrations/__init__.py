"""
Integration helpers for popular Python frameworks.

This module provides adapters and utilities to integrate cyllama with
popular frameworks like LangChain, OpenAI API, and others.
"""

from .langchain import CyllamaLLM

__all__ = ["CyllamaLLM"]
