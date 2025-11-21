"""
LangChain Integration

This module provides a LangChain-compatible LLM wrapper for cyllama.

Example:
    >>> from cyllama.integrations import CyllamaLLM
    >>> from langchain.prompts import PromptTemplate
    >>> from langchain.chains import LLMChain
    >>>
    >>> llm = CyllamaLLM(model_path="models/llama.gguf", temperature=0.7)
    >>> prompt = PromptTemplate.from_template("Tell me about {topic}")
    >>> chain = LLMChain(llm=llm, prompt=prompt)
    >>> result = chain.run(topic="Python")
"""

from typing import Any, List, Optional, Iterator, Dict
import warnings

from ..api import LLM as CyllamaLLMCore, GenerationConfig


try:
    from langchain.llms.base import LLM as LangChainLLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create dummy base class and types
    class LangChainLLM:
        pass
    # Dummy type for type hints
    CallbackManagerForLLMRun = None


class CyllamaLLM(LangChainLLM):
    """
    LangChain-compatible LLM wrapper for cyllama.

    This class adapts cyllama's Generator to work seamlessly with LangChain
    chains, agents, and tools.

    Attributes:
        model_path: Path to GGUF model file
        temperature: Sampling temperature (0.0 = greedy)
        max_tokens: Maximum tokens to generate
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        repeat_penalty: Penalty for repeating tokens
        n_gpu_layers: Number of layers to offload to GPU
        verbose: Print detailed generation information
    """

    model_path: str
    temperature: float = 0.7
    max_tokens: int = 512
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    n_gpu_layers: int = 99
    n_ctx: Optional[int] = None
    stop_sequences: List[str] = []
    verbose: bool = False

    # LangChain-specific attributes
    _generator: Optional[CyllamaLLMCore] = None

    def __init__(self, **kwargs):
        """Initialize the LLM."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install it with: pip install langchain"
            )
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM."""
        return "cyllama"

    @property
    def generator(self) -> CyllamaLLMCore:
        """Lazy-load the generator."""
        if self._generator is None:
            config = GenerationConfig(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                min_p=self.min_p,
                repeat_penalty=self.repeat_penalty,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                stop_sequences=self.stop_sequences,
            )
            self._generator = CyllamaLLMCore(
                self.model_path,
                config=config,
                verbose=self.verbose
            )
        return self._generator

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            stop: List of stop sequences (overrides instance stop_sequences)
            run_manager: LangChain callback manager
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        # Update config with kwargs
        config = GenerationConfig(
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            top_k=kwargs.get("top_k", self.top_k),
            top_p=kwargs.get("top_p", self.top_p),
            min_p=kwargs.get("min_p", self.min_p),
            repeat_penalty=kwargs.get("repeat_penalty", self.repeat_penalty),
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            stop_sequences=stop or self.stop_sequences,
        )

        # Generate with callbacks if provided
        if run_manager:
            def on_token(token: str):
                run_manager.on_llm_new_token(token)

            return self.generator(prompt, config=config, on_token=on_token)
        else:
            return self.generator(prompt, config=config)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Stream text generation from a prompt.

        Args:
            prompt: Input text prompt
            stop: List of stop sequences
            run_manager: LangChain callback manager
            **kwargs: Additional generation parameters

        Yields:
            Text chunks as they are generated
        """
        config = GenerationConfig(
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            top_k=kwargs.get("top_k", self.top_k),
            top_p=kwargs.get("top_p", self.top_p),
            min_p=kwargs.get("min_p", self.min_p),
            repeat_penalty=kwargs.get("repeat_penalty", self.repeat_penalty),
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            stop_sequences=stop or self.stop_sequences,
        )

        def on_token(token: str):
            if run_manager:
                run_manager.on_llm_new_token(token)

        for chunk in self.generator(prompt, config=config, stream=True, on_token=on_token):
            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return parameters that identify this LLM."""
        return {
            "model_path": self.model_path,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "n_gpu_layers": self.n_gpu_layers,
        }


# Provide helpful error message if imported without LangChain
if not LANGCHAIN_AVAILABLE:
    def __getattr__(name):
        if name == "CyllamaLLM":
            raise ImportError(
                "LangChain integration requires langchain to be installed. "
                "Install it with: pip install langchain"
            )
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
